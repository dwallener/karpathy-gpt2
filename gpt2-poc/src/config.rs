use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub d_model: usize,
    pub d_state: usize,
    pub num_operators: usize,
    pub operator_hidden: usize,
    pub top_k: usize,
    pub layer_norm_eps: f64,
}

impl Config {
    pub fn input_dim(&self) -> usize {
        self.d_model + self.d_state
    }

    pub fn state_update_dim(&self) -> usize {
        self.d_state + self.d_state + self.d_model
    }

    pub fn candidate_dim(&self) -> usize {
        self.d_state + self.d_model
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    pub seq_len: usize,
    pub batch_size: usize,
    pub tokenizer_workers: usize,
    pub model_dtype: String,
    pub mini_core_every: Option<usize>,
    pub mini_core_limit: Option<usize>,
    pub diag_every: Option<usize>,
    pub scaling_predictor: Option<bool>,
    pub steps: usize,
    pub lr: f64,
    pub weight_decay: f64,
    pub eval_every: usize,
    pub save_every: usize,
    pub log_every: usize,
    pub val_batches: usize,
    pub grad_clip: f64,
    pub out_dir: PathBuf,
    pub checkpoint: Option<PathBuf>,
    pub max_docs: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMeta {
    pub step: usize,
    pub model: Config,
    pub train: TrainConfig,
}
