use anyhow::{Result, bail};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{self as nn, Embedding, LayerNorm, Linear, Module, VarBuilder, VarMap};

use crate::config::Config;

#[derive(Clone)]
struct Operator {
    up: Linear,
    down: Linear,
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
    gate_proj: Linear,
    candidate_proj: Linear,
    readout_norm: LayerNorm,
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

        let gate_proj = nn::linear(
            config.state_update_dim(),
            config.d_state,
            vb.pp("gate_proj"),
        )?;
        let candidate_proj = nn::linear(
            config.candidate_dim(),
            config.d_state,
            vb.pp("candidate_proj"),
        )?;
        let readout_norm =
            nn::layer_norm(config.d_state, config.layer_norm_eps, vb.pp("readout_norm"))?;
        let readout_proj = nn::linear(config.d_state, config.vocab_size, vb.pp("readout_proj"))?;

        Ok(Self {
            config,
            dtype,
            token_embedding,
            router_norm,
            router_proj,
            operators,
            gate_proj,
            candidate_proj,
            readout_norm,
            readout_proj,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let (batch_size, seq_len) = xs.dims2()?;
        let mut state = Tensor::zeros((batch_size, self.config.d_state), self.dtype, xs.device())?;
        let embeddings = self.token_embedding.forward(xs)?;
        let mut logits_steps = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let e_t = embeddings.i((.., t, ..))?.contiguous()?;
            let router_input = Tensor::cat(&[&e_t, &state], 1)?;
            let router_hidden = self.router_norm.forward(&router_input)?;
            let router_scores = self.router_proj.forward(&router_hidden)?;
            let routing = topk_softmax_dense(&router_scores, self.config.top_k)?;

            let mut mixed =
                Tensor::zeros((batch_size, self.config.d_state), self.dtype, xs.device())?;
            for (idx, operator) in self.operators.iter().enumerate() {
                let op_out = operator.forward(&router_input)?;
                let gate = routing.i((.., idx))?.unsqueeze(1)?;
                mixed = mixed.broadcast_add(&op_out.broadcast_mul(&gate)?)?;
            }

            let gate_input = Tensor::cat(&[&state, &mixed, &e_t], 1)?;
            let z_t = nn::ops::sigmoid(&self.gate_proj.forward(&gate_input)?)?;
            let h_t = self
                .candidate_proj
                .forward(&Tensor::cat(&[&mixed, &e_t], 1)?)?
                .tanh()?;
            let one = Tensor::ones(z_t.shape(), z_t.dtype(), z_t.device())?;
            let carry = one.broadcast_sub(&z_t)?.broadcast_mul(&state)?;
            let write = z_t.broadcast_mul(&h_t)?;
            state = carry.broadcast_add(&write)?;

            let logits_t = self
                .readout_proj
                .forward(&self.readout_norm.forward(&state)?)?;
            logits_steps.push(logits_t);
        }

        let logits = Tensor::stack(&logits_steps.iter().collect::<Vec<_>>(), 1)?;
        Ok((logits, state))
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

fn topk_softmax_dense(scores: &Tensor, top_k: usize) -> Result<Tensor> {
    let device = scores.device().clone();
    let values = scores.to_vec2::<f32>()?;
    let num_operators = values
        .first()
        .map(|row| row.len())
        .ok_or_else(|| anyhow::anyhow!("router scores cannot be empty"))?;
    let mut dense = vec![0f32; values.len() * num_operators];

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
            dense[batch_idx * num_operators + *op_idx] = (*score - max_score).exp() / denom;
        }
    }

    Ok(
        Tensor::from_vec(dense, (values.len(), num_operators), &device)?
            .to_dtype(scores.dtype())?,
    )
}

pub fn cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let (_, _, vocab_size) = logits.dims3()?;
    let flat_logits = logits.reshape(((), vocab_size))?.to_dtype(DType::F32)?;
    let flat_targets = targets.flatten_all()?;
    Ok(nn::loss::cross_entropy(&flat_logits, &flat_targets)?)
}
