use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, ValueEnum};

use gpt2_poc::eval::run_mini_core;
use gpt2_poc::utils::{format_float, resolve_device};

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
#[command(author, version, about = "Mini-CORE benchmark eval for saved checkpoints")]
struct Cli {
    #[arg(long)]
    checkpoint: PathBuf,
    #[arg(long, value_enum, default_value = "cpu")]
    device: DeviceArg,
    #[arg(long, value_enum)]
    model_dtype: Option<ModelDTypeArg>,
    #[arg(long)]
    limit: Option<usize>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let device = resolve_device(matches!(cli.device, DeviceArg::Cuda))?;
    let report = run_mini_core(
        &cli.checkpoint,
        &device,
        cli.model_dtype.map(ModelDTypeArg::as_str),
        cli.limit,
    )?;

    for dataset in &report.datasets {
        println!(
            "{}: {} centered={} examples={} ex/s={}",
            dataset.name,
            format_float(dataset.accuracy),
            format_float(dataset.centered_accuracy),
            dataset.examples,
            format_float(dataset.examples_per_sec),
        );
    }
    println!("miniCORE: {}", format_float(report.mini_core));
    Ok(())
}
