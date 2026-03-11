use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use gpt2_poc::infer::run_greedy_inference;
use gpt2_poc::utils::{resolve_device, resolve_model_dtype};

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
    about = "Greedy inference against a saved operator/state checkpoint"
)]
struct Cli {
    #[arg(long)]
    checkpoint: PathBuf,
    #[arg(long)]
    prompt: String,
    #[arg(long, default_value_t = 32)]
    max_new_tokens: usize,
    #[arg(long, value_enum, default_value = "cpu")]
    device: DeviceArg,
    #[arg(long, value_enum)]
    model_dtype: Option<ModelDTypeArg>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let device = resolve_device(matches!(cli.device, DeviceArg::Cuda))?;
    let _dtype = resolve_model_dtype(&device, cli.model_dtype.map(ModelDTypeArg::as_str))?;

    let output = run_greedy_inference(
        &cli.checkpoint,
        &cli.prompt,
        cli.max_new_tokens,
        &device,
        cli.model_dtype.map(ModelDTypeArg::as_str),
    )?;

    println!("prompt_tokens={}", output.prompt_tokens);
    println!("generated_tokens={:?}", output.generated_tokens);
    println!("{}", output.decoded_text);
    Ok(())
}
