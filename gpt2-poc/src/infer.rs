use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{VarMap, ops};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tokenizers::Tokenizer;

use crate::config::CheckpointMeta;
use crate::model::{OperatorStateLM, build_model};
use crate::utils::resolve_model_dtype;

pub struct InferenceOutput {
    pub prompt_tokens: usize,
    pub generated_tokens: Vec<u32>,
    pub decoded_text: String,
}

#[derive(Debug, Clone, Copy)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub seed: u64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 1,
            seed: 42,
        }
    }
}

pub struct InferenceSession {
    pub meta: CheckpointMeta,
    tokenizer: Tokenizer,
    _varmap: VarMap,
    model: OperatorStateLM,
    device: Device,
}

impl InferenceSession {
    pub fn load(checkpoint: &Path, device: &Device, requested_dtype: Option<&str>) -> Result<Self> {
        let meta = load_checkpoint_meta(checkpoint)?;
        let model_path = resolve_model_path(checkpoint);
        let model_dtype = resolve_model_dtype(
            device,
            normalize_dtype_request(requested_dtype)
                .or_else(|| normalize_dtype_request(Some(&meta.train.model_dtype))),
        )?;

        let tokenizer = load_tokenizer()?;
        let (mut varmap, model) = build_model(meta.model.clone(), model_dtype, device)?;
        varmap
            .load(&model_path)
            .with_context(|| format!("failed to load checkpoint {}", model_path.display()))?;

        Ok(Self {
            meta,
            tokenizer,
            _varmap: varmap,
            model,
            device: device.clone(),
        })
    }

    pub fn generate_greedy(&self, prompt: &str, max_new_tokens: usize) -> Result<InferenceOutput> {
        self.generate(prompt, max_new_tokens, SamplingConfig::default())
    }

    pub fn generate(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        sampling: SamplingConfig,
    ) -> Result<InferenceOutput> {
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|err| anyhow::anyhow!("failed to tokenize prompt: {err}"))?;
        let mut tokens = encoding.get_ids().to_vec();
        if tokens.is_empty() {
            bail!("prompt produced zero tokens");
        }

        let mut rng = StdRng::seed_from_u64(sampling.seed);
        let mut generated = Vec::with_capacity(max_new_tokens);
        for _ in 0..max_new_tokens {
            let input = Tensor::from_vec(tokens.clone(), (1, tokens.len()), &self.device)?;
            let (logits, _) = self.model.forward(&input)?;
            let last_logits = logits
                .i((0, logits.dim(1)? - 1, ..))?
                .to_dtype(DType::F32)?;
            let next_token = sample_next_token(
                &last_logits.to_vec1::<f32>()?,
                sampling.temperature,
                sampling.top_k,
                &mut rng,
            )?;
            tokens.push(next_token);
            generated.push(next_token);
        }

        let decoded_text = self
            .tokenizer
            .decode(&tokens, true)
            .map_err(|err| anyhow::anyhow!("failed to decode generated tokens: {err}"))?;

        Ok(InferenceOutput {
            prompt_tokens: encoding.len(),
            generated_tokens: generated,
            decoded_text,
        })
    }

    pub fn score_continuation(&self, prompt: &str, continuation: &str) -> Result<f64> {
        let prompt_ids = self.encode(prompt)?;
        if prompt_ids.is_empty() {
            bail!("prompt produced zero tokens, cannot score continuation");
        }

        let full_text = format!("{prompt}{continuation}");
        let full_ids = self.encode(&full_text)?;
        if full_ids.len() <= prompt_ids.len() {
            bail!("continuation produced zero new tokens");
        }
        if !full_ids.starts_with(&prompt_ids) {
            bail!("prompt tokenization is not a prefix of prompt+continuation");
        }

        let input_ids = &full_ids[..full_ids.len() - 1];
        let input = Tensor::from_vec(input_ids.to_vec(), (1, input_ids.len()), &self.device)?;
        let (logits, _) = self.model.forward(&input)?;
        let log_probs = ops::log_softmax(&logits.to_dtype(DType::F32)?, 2)?;

        let mut total = 0f64;
        for target_index in prompt_ids.len()..full_ids.len() {
            let logit_pos = target_index - 1;
            let target_id = full_ids[target_index] as usize;
            total += log_probs
                .i((0, logit_pos, target_id))?
                .to_scalar::<f32>()? as f64;
        }
        Ok(total)
    }

    pub fn next_token_logits(&self, prompt: &str) -> Result<Vec<f32>> {
        let prompt_ids = self.encode(prompt)?;
        if prompt_ids.is_empty() {
            bail!("prompt produced zero tokens, cannot get next-token logits");
        }
        let input = Tensor::from_vec(prompt_ids.clone(), (1, prompt_ids.len()), &self.device)?;
        let (logits, _) = self.model.forward(&input)?;
        logits
            .i((0, logits.dim(1)? - 1, ..))?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()
            .map_err(Into::into)
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(|err| anyhow::anyhow!("failed to decode tokens: {err}"))
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|err| anyhow::anyhow!("failed to tokenize text: {err}"))?;
        Ok(encoding.get_ids().to_vec())
    }
}

pub fn run_greedy_inference(
    checkpoint: &Path,
    prompt: &str,
    max_new_tokens: usize,
    device: &Device,
    requested_dtype: Option<&str>,
) -> Result<InferenceOutput> {
    run_inference(
        checkpoint,
        prompt,
        max_new_tokens,
        device,
        requested_dtype,
        SamplingConfig::default(),
    )
}

pub fn run_inference(
    checkpoint: &Path,
    prompt: &str,
    max_new_tokens: usize,
    device: &Device,
    requested_dtype: Option<&str>,
    sampling: SamplingConfig,
) -> Result<InferenceOutput> {
    let session = InferenceSession::load(checkpoint, device, requested_dtype)?;
    session.generate(prompt, max_new_tokens, sampling)
}

pub fn load_checkpoint_meta(checkpoint: &Path) -> Result<CheckpointMeta> {
    let meta_path = if checkpoint.is_dir() {
        checkpoint.join("meta.json")
    } else {
        checkpoint
            .parent()
            .context("checkpoint path has no parent directory")?
            .join("meta.json")
    };
    Ok(serde_json::from_reader(File::open(&meta_path)?)?)
}

pub fn resolve_model_path(checkpoint: &Path) -> PathBuf {
    if checkpoint.is_dir() {
        checkpoint.join("model.safetensors")
    } else {
        checkpoint.to_path_buf()
    }
}

fn normalize_dtype_request(requested: Option<&str>) -> Option<&str> {
    match requested {
        Some("auto") | None => None,
        Some(other) => Some(other),
    }
}

fn load_tokenizer() -> Result<Tokenizer> {
    Tokenizer::from_pretrained("gpt2", None)
        .map_err(|err| anyhow::anyhow!("failed to load GPT-2 tokenizer: {err}"))
}

fn sample_next_token(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    rng: &mut StdRng,
) -> Result<u32> {
    if logits.is_empty() {
        bail!("cannot sample from empty logits");
    }
    if temperature <= 0.0 || top_k <= 1 {
        let (idx, _) = logits
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .context("failed to select argmax token")?;
        return Ok(idx as u32);
    }

    let mut ranked = logits
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<(usize, f32)>>();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    let keep = top_k.min(ranked.len());
    ranked.truncate(keep);

    let max_logit = ranked
        .iter()
        .map(|(_, logit)| *logit / temperature)
        .fold(f32::NEG_INFINITY, f32::max);
    let weights = ranked
        .iter()
        .map(|(_, logit)| ((*logit / temperature) - max_logit).exp())
        .collect::<Vec<_>>();
    let total = weights.iter().sum::<f32>();
    if total <= 0.0 || !total.is_finite() {
        return Ok(ranked[0].0 as u32);
    }

    let mut draw = rng.r#gen::<f32>() * total;
    for ((token_idx, _), weight) in ranked.iter().zip(weights.iter()) {
        draw -= *weight;
        if draw <= 0.0 {
            return Ok(*token_idx as u32);
        }
    }

    Ok(ranked.last().map(|(idx, _)| *idx as u32).unwrap_or(0))
}
