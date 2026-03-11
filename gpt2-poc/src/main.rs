use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Parser, Subcommand, ValueEnum};

use gpt2_poc::config::{Config, TrainConfig};
use gpt2_poc::stream_dataset::{DatasetSplit, StreamDataset, list_shards};
use gpt2_poc::train::{eval_main, inspect_batch_main, train_main};
use gpt2_poc::utils;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DeviceArg {
    Cpu,
    Cuda,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ModelDTypeArg {
    F32,
    Bf16,
    F16,
}

impl ModelDTypeArg {
    fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Bf16 => "bf16",
            Self::F16 => "f16",
        }
    }
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
    shard_dir: PathBuf,
    #[arg(long, value_enum, default_value = "cpu")]
    device: DeviceArg,
    #[arg(long, default_value_t = 128)]
    seq_len: usize,
    #[arg(long, default_value_t = 8)]
    batch_size: usize,
    #[arg(long, default_value_t = default_tokenizer_workers())]
    tokenizer_workers: usize,
    #[arg(long, value_enum)]
    model_dtype: Option<ModelDTypeArg>,
    #[arg(long)]
    max_docs: Option<usize>,
}

#[derive(Debug, Clone, Args)]
struct TrainArgs {
    #[command(flatten)]
    data: DataArgs,
    #[arg(long)]
    mini_core_every: Option<usize>,
    #[arg(long)]
    mini_core_limit: Option<usize>,
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
            tokenizer_workers: self.data.tokenizer_workers,
            model_dtype: self
                .data
                .model_dtype
                .map(ModelDTypeArg::as_str)
                .unwrap_or("auto")
                .to_string(),
            mini_core_every: self.mini_core_every,
            mini_core_limit: self.mini_core_limit,
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
            max_docs: self.data.max_docs,
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
            let device = utils::resolve_device(matches!(args.data.device, DeviceArg::Cuda))?;
            let model_dtype = utils::resolve_model_dtype(
                &device,
                args.data.model_dtype.map(ModelDTypeArg::as_str),
            )?;
            let shards = list_shards(&args.data.shard_dir)?;
            let vocab_size = StreamDataset::tokenizer_vocab_size()?;
            let (train_shards, val_shards) = DatasetSplit::train_val_split(shards)?;
            let train_ds = StreamDataset::new(
                train_shards,
                args.data.seq_len,
                args.data.batch_size,
                true,
                args.data.max_docs,
                args.data.tokenizer_workers,
                true,
            )?;
            let val_ds = StreamDataset::new(
                val_shards,
                args.data.seq_len,
                args.data.batch_size,
                false,
                args.data.max_docs,
                1,
                false,
            )?;
            train_main(
                args.model_config(vocab_size),
                args.train_config(),
                model_dtype,
                train_ds,
                val_ds,
                &device,
            )
        }
        Commands::Eval(args) => {
            let device = utils::resolve_device(matches!(args.data.device, DeviceArg::Cuda))?;
            let model_dtype = utils::resolve_model_dtype(
                &device,
                args.data.model_dtype.map(ModelDTypeArg::as_str),
            )?;
            let shards = list_shards(&args.data.shard_dir)?;
            let vocab_size = StreamDataset::tokenizer_vocab_size()?;
            let (train_shards, val_shards) = DatasetSplit::train_val_split(shards)?;
            let train_ds = StreamDataset::new(
                train_shards,
                args.data.seq_len,
                args.data.batch_size,
                false,
                args.data.max_docs,
                1,
                false,
            )?;
            let val_ds = StreamDataset::new(
                val_shards,
                args.data.seq_len,
                args.data.batch_size,
                false,
                args.data.max_docs,
                1,
                false,
            )?;
            eval_main(
                args.model_config(vocab_size),
                model_dtype,
                &args.checkpoint,
                train_ds,
                val_ds,
                args.data.seq_len,
                args.data.batch_size,
                args.val_batches,
                &device,
            )
        }
        Commands::InspectBatch(args) => {
            let shards = list_shards(&args.shard_dir)?;
            let (train_shards, val_shards) = DatasetSplit::train_val_split(shards)?;
            let mut train_ds = StreamDataset::new(
                train_shards,
                args.seq_len,
                args.batch_size,
                false,
                args.max_docs,
                args.tokenizer_workers,
                true,
            )?;
            let val_ds = StreamDataset::new(
                val_shards,
                args.seq_len,
                args.batch_size,
                false,
                args.max_docs,
                1,
                false,
            )?;
            inspect_batch_main(
                &mut train_ds,
                &val_ds,
                &utils::resolve_device(matches!(args.device, DeviceArg::Cuda))?,
            )
        }
    }
}

fn default_tokenizer_workers() -> usize {
    let cpus = num_cpus::get();
    if cpus <= 1 { 1 } else { cpus / 2 }
}
