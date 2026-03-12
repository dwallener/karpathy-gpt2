use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use gpt2_poc::infer::{SamplingConfig, run_inference};
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
    about = "Inference against a saved operator/state checkpoint"
)]
struct Cli {
    #[arg(long)]
    checkpoint: PathBuf,
    #[arg(long)]
    prompt: Vec<String>,
    #[arg(long)]
    prompts_file: Option<PathBuf>,
    #[arg(long, default_value_t = 32)]
    max_new_tokens: usize,
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,
    #[arg(long, default_value_t = 1)]
    top_k: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, value_enum, default_value = "cpu")]
    device: DeviceArg,
    #[arg(long, value_enum)]
    model_dtype: Option<ModelDTypeArg>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let device = resolve_device(matches!(cli.device, DeviceArg::Cuda))?;
    let _dtype = resolve_model_dtype(&device, cli.model_dtype.map(ModelDTypeArg::as_str))?;
    let prompts = load_prompts(&cli)?;
    if prompts.is_empty() {
        bail!("provide at least one --prompt or --prompts-file");
    }

    for (idx, prompt) in prompts.iter().enumerate() {
        let output = run_inference(
            &cli.checkpoint,
            prompt,
            cli.max_new_tokens,
            &device,
            cli.model_dtype.map(ModelDTypeArg::as_str),
            SamplingConfig {
                temperature: cli.temperature,
                top_k: cli.top_k,
                seed: cli.seed.saturating_add(idx as u64),
            },
        )?;

        if prompts.len() > 1 {
            println!("=== Prompt {} ===", idx + 1);
            println!("{}", prompt);
        }
        println!("prompt_tokens={}", output.prompt_tokens);
        println!("generated_tokens={:?}", output.generated_tokens);
        println!("{}", output.decoded_text);
        if idx + 1 < prompts.len() {
            println!();
        }
    }
    Ok(())
}

fn load_prompts(cli: &Cli) -> Result<Vec<String>> {
    let mut prompts = cli.prompt.clone();
    if let Some(path) = &cli.prompts_file {
        let contents = fs::read_to_string(path)
            .with_context(|| format!("failed to read prompts file {}", path.display()))?;
        prompts.extend(
            contents
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(ToOwned::to_owned),
        );
    }
    Ok(prompts)
}
