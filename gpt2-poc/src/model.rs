use anyhow::{Result, bail};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{self as nn, Embedding, LayerNorm, Linear, Module, RmsNorm, VarBuilder, VarMap};

use crate::config::Config;

#[derive(Clone)]
struct Operator {
    up: Linear,
    down: Linear,
}

#[derive(Debug, Clone)]
struct SparseRoute {
    row_indices: Vec<u32>,
    gate_values: Vec<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct RouterMetrics {
    pub routing_entropy: f32,
    pub max_operator_share: f32,
    pub num_active_operators: usize,
    pub operator_usage: Vec<f32>,
    pub gate_mass: Vec<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct StateMetrics {
    pub state_norm_mean: f32,
    pub state_norm_std: f32,
    pub delta_state_norm: f32,
    pub state_to_prev_ratio: f32,
}

#[derive(Debug, Clone, Default)]
pub struct ModelMetrics {
    pub router: RouterMetrics,
    pub state: StateMetrics,
}

impl Operator {
    fn new(
        vb: VarBuilder<'_>,
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
    ) -> Result<Self> {
        Ok(Self {
            up: nn::linear(input_dim, hidden_dim, vb.pp("up"))?,
            down: nn::linear(hidden_dim, output_dim, vb.pp("down"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = self.up.forward(xs)?.gelu()?;
        Ok(self.down.forward(&h)?)
    }
}

pub struct OperatorStateLM {
    config: Config,
    dtype: DType,
    token_embedding: Embedding,
    router_norm: LayerNorm,
    router_proj: Linear,
    operators: Vec<Operator>,
    reset_proj: Linear,
    forget_proj: Linear,
    write_proj: Linear,
    candidate_proj: Linear,
    candidate_norm: RmsNorm,
    readout_norm: LayerNorm,
    readout_residual_proj: Linear,
    readout_proj: Linear,
}

impl OperatorStateLM {
    pub fn new(vb: VarBuilder<'_>, config: Config, dtype: DType) -> Result<Self> {
        if config.top_k == 0 || config.top_k > config.num_operators {
            bail!(
                "top_k must be in [1, num_operators], got {} with {} operators",
                config.top_k,
                config.num_operators
            );
        }
        let token_embedding =
            nn::embedding(config.vocab_size, config.d_model, vb.pp("token_embedding"))?;
        let router_norm = nn::layer_norm(
            config.input_dim(),
            config.layer_norm_eps,
            vb.pp("router_norm"),
        )?;
        let router_proj = nn::linear(
            config.input_dim(),
            config.num_operators,
            vb.pp("router_proj"),
        )?;

        let mut operators = Vec::with_capacity(config.num_operators);
        for idx in 0..config.num_operators {
            operators.push(Operator::new(
                vb.pp(format!("operator_{idx}")),
                config.input_dim(),
                config.operator_hidden,
                config.d_state,
            )?);
        }

        let reset_proj = nn::linear(
            config.state_update_dim(),
            config.d_state,
            vb.pp("reset_proj"),
        )?;
        let forget_proj = nn::linear(
            config.state_update_dim(),
            config.d_state,
            vb.pp("forget_proj"),
        )?;
        let write_proj = nn::linear(
            config.state_update_dim(),
            config.d_state,
            vb.pp("write_proj"),
        )?;
        let candidate_proj = nn::linear(
            config.candidate_dim(),
            config.d_state,
            vb.pp("candidate_proj"),
        )?;
        let candidate_norm =
            nn::rms_norm(config.d_state, config.layer_norm_eps, vb.pp("candidate_norm"))?;
        let readout_norm =
            nn::layer_norm(config.d_state, config.layer_norm_eps, vb.pp("readout_norm"))?;
        let readout_residual_proj = nn::linear(
            config.d_model,
            config.d_state,
            vb.pp("readout_residual_proj"),
        )?;
        let readout_proj = nn::linear(config.d_state, config.vocab_size, vb.pp("readout_proj"))?;

        Ok(Self {
            config,
            dtype,
            token_embedding,
            router_norm,
            router_proj,
            operators,
            reset_proj,
            forget_proj,
            write_proj,
            candidate_proj,
            candidate_norm,
            readout_norm,
            readout_residual_proj,
            readout_proj,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let (logits, state, _) = self.forward_with_metrics(xs)?;
        Ok((logits, state))
    }

    pub fn forward_with_metrics(&self, xs: &Tensor) -> Result<(Tensor, Tensor, ModelMetrics)> {
        let (batch_size, seq_len) = xs.dims2()?;
        let mut state = Tensor::zeros((batch_size, self.config.d_state), self.dtype, xs.device())?;
        let mut prev_norms = vec![0f32; batch_size];
        let embeddings = self.token_embedding.forward(xs)?;
        let mut logits_steps = Vec::with_capacity(seq_len);
        let mut usage_counts = vec![0usize; self.config.num_operators];
        let mut gate_mass = vec![0f32; self.config.num_operators];
        let mut total_route_entropy = 0f32;
        let mut route_rows = 0usize;
        let mut sum_state_norm = 0f32;
        let mut sum_state_norm_sq = 0f32;
        let mut sum_delta_state_norm = 0f32;
        let mut sum_state_ratio = 0f32;
        let mut state_samples = 0usize;
        let mut ratio_samples = 0usize;

        for t in 0..seq_len {
            let e_t = embeddings.i((.., t, ..))?.contiguous()?;
            let router_input = Tensor::cat(&[&e_t, &state], 1)?;
            let router_hidden = self.router_norm.forward(&router_input)?;
            let router_scores = self.router_proj.forward(&router_hidden)?;
            let routing =
                topk_routes(&router_scores, self.config.top_k, self.config.num_operators)?;
            for (op_idx, route) in routing.iter().enumerate() {
                usage_counts[op_idx] += route.row_indices.len();
                gate_mass[op_idx] += route.gate_values.iter().sum::<f32>();
                total_route_entropy += route
                    .gate_values
                    .iter()
                    .filter(|p| **p > 0.0)
                    .map(|p| -p * p.ln())
                    .sum::<f32>();
                route_rows += route.row_indices.len();
            }

            let mut mixed =
                Tensor::zeros((batch_size, self.config.d_state), self.dtype, xs.device())?;
            for (idx, operator) in self.operators.iter().enumerate() {
                let route = &routing[idx];
                if route.row_indices.is_empty() {
                    continue;
                }
                let row_indices = Tensor::from_vec(
                    route.row_indices.clone(),
                    route.row_indices.len(),
                    xs.device(),
                )?;
                let op_input = router_input.index_select(&row_indices, 0)?;
                let op_out = operator.forward(&op_input)?;
                let gate = Tensor::from_vec(
                    route.gate_values.clone(),
                    (route.gate_values.len(), 1),
                    xs.device(),
                )?
                .to_dtype(self.dtype)?;
                let weighted = op_out.broadcast_mul(&gate)?;
                mixed = mixed.index_add(&row_indices, &weighted, 0)?;
            }

            let gate_input = Tensor::cat(&[&state, &mixed, &e_t], 1)?;
            let r_t = nn::ops::sigmoid(&self.reset_proj.forward(&gate_input)?)?;
            let f_t = nn::ops::sigmoid(&self.forget_proj.forward(&gate_input)?)?;
            let w_t = nn::ops::sigmoid(&self.write_proj.forward(&gate_input)?)?;
            let reset_state = r_t.broadcast_mul(&state)?;
            let h_t = self
                .candidate_proj
                .forward(&Tensor::cat(&[&reset_state, &mixed, &e_t], 1)?)?
                .tanh()?;
            let h_t = self.candidate_norm.forward(&h_t)?;
            let carry = f_t.broadcast_mul(&state)?;
            let write = w_t.broadcast_mul(&h_t)?;
            state = carry.broadcast_add(&write)?;
            let state_norms = state
                .sqr()?
                .sum(1)?
                .sqrt()?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;
            for (current, previous) in state_norms.iter().zip(prev_norms.iter()) {
                sum_state_norm += *current;
                sum_state_norm_sq += current * current;
                state_samples += 1;
                if *previous > 1e-6 {
                    sum_delta_state_norm += (current - previous).abs();
                    sum_state_ratio += current / previous;
                    ratio_samples += 1;
                }
            }
            prev_norms = state_norms;

            let readout_state = state.broadcast_add(&self.readout_residual_proj.forward(&e_t)?)?;
            let logits_t = self
                .readout_proj
                .forward(&self.readout_norm.forward(&readout_state)?)?;
            logits_steps.push(logits_t);
        }

        let logits = Tensor::stack(&logits_steps.iter().collect::<Vec<_>>(), 1)?;
        let total_assignments = (batch_size * seq_len * self.config.top_k) as f32;
        let operator_usage = if total_assignments > 0.0 {
            usage_counts
                .iter()
                .map(|count| *count as f32 / total_assignments)
                .collect::<Vec<_>>()
        } else {
            vec![0.0; self.config.num_operators]
        };
        let total_gate_mass = gate_mass.iter().sum::<f32>().max(1e-9);
        let gate_mass = gate_mass
            .into_iter()
            .map(|mass| mass / total_gate_mass)
            .collect::<Vec<_>>();
        let max_operator_share = operator_usage
            .iter()
            .copied()
            .fold(0.0f32, f32::max);
        let num_active_operators = usage_counts.iter().filter(|count| **count > 0).count();
        let routing_entropy = if route_rows > 0 {
            total_route_entropy / route_rows as f32
        } else {
            0.0
        };

        let state_norm_mean = if state_samples > 0 {
            sum_state_norm / state_samples as f32
        } else {
            0.0
        };
        let state_norm_var = if state_samples > 0 {
            (sum_state_norm_sq / state_samples as f32) - state_norm_mean * state_norm_mean
        } else {
            0.0
        };
        let state_norm_std = state_norm_var.max(0.0).sqrt();
        let delta_state_norm = if ratio_samples > 0 {
            sum_delta_state_norm / ratio_samples as f32
        } else {
            0.0
        };
        let state_to_prev_ratio = if ratio_samples > 0 {
            sum_state_ratio / ratio_samples as f32
        } else {
            0.0
        };

        Ok((
            logits,
            state,
            ModelMetrics {
                router: RouterMetrics {
                routing_entropy,
                max_operator_share,
                num_active_operators,
                operator_usage,
                gate_mass,
            },
                state: StateMetrics {
                    state_norm_mean,
                    state_norm_std,
                    delta_state_norm,
                    state_to_prev_ratio,
                },
            },
        ))
    }
}

pub fn build_model(
    config: Config,
    dtype: DType,
    device: &Device,
) -> Result<(VarMap, OperatorStateLM)> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, device);
    let model = OperatorStateLM::new(vb, config, dtype)?;
    Ok((varmap, model))
}

fn topk_routes(scores: &Tensor, top_k: usize, num_operators: usize) -> Result<Vec<SparseRoute>> {
    let values = scores.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let actual_num_operators = values
        .first()
        .map(|row| row.len())
        .ok_or_else(|| anyhow::anyhow!("router scores cannot be empty"))?;
    if actual_num_operators != num_operators {
        anyhow::bail!(
            "router score width {} does not match configured operators {}",
            actual_num_operators,
            num_operators
        );
    }
    let mut routes = vec![
        SparseRoute {
            row_indices: Vec::new(),
            gate_values: Vec::new(),
        };
        num_operators
    ];

    for (batch_idx, row) in values.iter().enumerate() {
        let mut ranked: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        let active = &ranked[..top_k];
        let max_score = active
            .iter()
            .map(|(_, score)| *score)
            .fold(f32::NEG_INFINITY, f32::max);
        let denom: f32 = active
            .iter()
            .map(|(_, score)| (*score - max_score).exp())
            .sum();
        for (op_idx, score) in active {
            routes[*op_idx].row_indices.push(batch_idx as u32);
            routes[*op_idx]
                .gate_values
                .push((*score - max_score).exp() / denom);
        }
    }
    Ok(routes)
}

pub fn cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let (_, _, vocab_size) = logits.dims3()?;
    let flat_logits = logits.reshape(((), vocab_size))?.to_dtype(DType::F32)?;
    let flat_targets = targets.flatten_all()?;
    Ok(nn::loss::cross_entropy(&flat_logits, &flat_targets)?)
}
