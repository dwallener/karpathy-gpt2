mod config;
mod dataset;
mod model;
mod train;
mod utils;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Parser, Subcommand, ValueEnum};

use crate::config::{Config, TrainConfig};
use crate::dataset::TokenDataset;
use crate::train::{eval_main, inspect_batch_main, train_main};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DeviceArg {
    Cpu,
    Cuda,
}

#[derive(Debug, Parser)]
#[command(
    author,
    version,
    about = "Operator/state language-model POC in Rust + Candle"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    Train(TrainArgs),
    Eval(EvalArgs),
    InspectBatch(DataArgs),
}

#[derive(Debug, Clone, Args)]
struct DataArgs {
    #[arg(long)]
    train_tokens: PathBuf,
    #[arg(long)]
    val_tokens: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "cpu")]
    device: DeviceArg,
    #[arg(long, default_value_t = 128)]
    seq_len: usize,
    #[arg(long, default_value_t = 8)]
    batch_size: usize,
    #[arg(long)]
    vocab_size: Option<usize>,
}

#[derive(Debug, Clone, Args)]
struct TrainArgs {
    #[command(flatten)]
    data: DataArgs,
    #[arg(long, default_value_t = 1000)]
    steps: usize,
    #[arg(long, default_value_t = 3e-4)]
    lr: f64,
    #[arg(long, default_value_t = 0.01)]
    weight_decay: f64,
    #[arg(long, default_value_t = 100)]
    eval_every: usize,
    #[arg(long, default_value_t = 500)]
    save_every: usize,
    #[arg(long, default_value_t = 100)]
    log_every: usize,
    #[arg(long, default_value_t = 16)]
    val_batches: usize,
    #[arg(long, default_value_t = 1.0)]
    grad_clip: f64,
    #[arg(long)]
    out_dir: PathBuf,
    #[arg(long)]
    checkpoint: Option<PathBuf>,
    #[arg(long, default_value_t = 768)]
    d_model: usize,
    #[arg(long, default_value_t = 2048)]
    d_state: usize,
    #[arg(long, default_value_t = 16)]
    num_operators: usize,
    #[arg(long, default_value_t = 1024)]
    operator_hidden: usize,
    #[arg(long, default_value_t = 4)]
    top_k: usize,
}

#[derive(Debug, Clone, Args)]
struct EvalArgs {
    #[command(flatten)]
    data: DataArgs,
    #[arg(long)]
    checkpoint: PathBuf,
    #[arg(long, default_value_t = 16)]
    val_batches: usize,
    #[arg(long, default_value_t = 768)]
    d_model: usize,
    #[arg(long, default_value_t = 2048)]
    d_state: usize,
    #[arg(long, default_value_t = 16)]
    num_operators: usize,
    #[arg(long, default_value_t = 1024)]
    operator_hidden: usize,
    #[arg(long, default_value_t = 4)]
    top_k: usize,
}

impl DataArgs {
    fn dataset_paths(&self) -> Result<(TokenDataset, Option<TokenDataset>)> {
        let train = TokenDataset::from_file(&self.train_tokens)?;
        let val = match &self.val_tokens {
            Some(path) => Some(TokenDataset::from_file(path)?),
            None => None,
        };
        Ok((train, val))
    }

    fn resolve_vocab_size(&self, train: &TokenDataset, val: Option<&TokenDataset>) -> usize {
        self.vocab_size
            .unwrap_or_else(|| train.derived_vocab_size(val))
    }
}

impl TrainArgs {
    fn model_config(&self, vocab_size: usize) -> Config {
        Config {
            vocab_size,
            d_model: self.d_model,
            d_state: self.d_state,
            num_operators: self.num_operators,
            operator_hidden: self.operator_hidden,
            top_k: self.top_k,
            layer_norm_eps: 1e-5,
        }
    }

    fn train_config(&self) -> TrainConfig {
        TrainConfig {
            seq_len: self.data.seq_len,
            batch_size: self.data.batch_size,
            steps: self.steps,
            lr: self.lr,
            weight_decay: self.weight_decay,
            eval_every: self.eval_every,
            save_every: self.save_every,
            log_every: self.log_every,
            val_batches: self.val_batches,
            grad_clip: self.grad_clip,
            out_dir: self.out_dir.clone(),
            checkpoint: self.checkpoint.clone(),
        }
    }
}

impl EvalArgs {
    fn model_config(&self, vocab_size: usize) -> Config {
        Config {
            vocab_size,
            d_model: self.d_model,
            d_state: self.d_state,
            num_operators: self.num_operators,
            operator_hidden: self.operator_hidden,
            top_k: self.top_k,
            layer_norm_eps: 1e-5,
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Train(args) => {
            let (train_ds, val_ds) = args.data.dataset_paths()?;
            let vocab_size = args.data.resolve_vocab_size(&train_ds, val_ds.as_ref());
            train_main(
                args.model_config(vocab_size),
                args.train_config(),
                train_ds,
                val_ds,
                &utils::resolve_device(matches!(args.data.device, DeviceArg::Cuda))?,
            )
        }
        Commands::Eval(args) => {
            let (train_ds, val_ds) = args.data.dataset_paths()?;
            let vocab_size = args.data.resolve_vocab_size(&train_ds, val_ds.as_ref());
            eval_main(
                args.model_config(vocab_size),
                &args.checkpoint,
                train_ds,
                val_ds,
                args.data.seq_len,
                args.data.batch_size,
                args.val_batches,
                &utils::resolve_device(matches!(args.data.device, DeviceArg::Cuda))?,
            )
        }
        Commands::InspectBatch(args) => {
            let (train_ds, val_ds) = args.dataset_paths()?;
            inspect_batch_main(
                train_ds,
                val_ds,
                args.seq_len,
                args.batch_size,
                &utils::resolve_device(matches!(args.device, DeviceArg::Cuda))?,
            )
        }
    }
}
