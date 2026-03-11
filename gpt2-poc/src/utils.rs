use std::fs::File;
use std::path::Path;

use anyhow::Result;
use candle_core::{DType, Device};
use serde::Serialize;

pub fn resolve_device(want_cuda: bool) -> Result<Device> {
    if !want_cuda {
        return Ok(Device::Cpu);
    }
    #[cfg(feature = "cuda")]
    {
        return Ok(Device::new_cuda(0)?);
    }
    #[cfg(not(feature = "cuda"))]
    {
        anyhow::bail!("cuda requested but this binary was not built with the `cuda` feature");
    }
}

pub fn resolve_model_dtype(device: &Device, requested: Option<&str>) -> Result<DType> {
    let dtype = match requested {
        Some("f32") => DType::F32,
        Some("bf16") => DType::BF16,
        Some("f16") => DType::F16,
        Some(other) => anyhow::bail!("unsupported model dtype '{other}', expected f32|bf16|f16"),
        None if device.is_cuda() => DType::BF16,
        None => DType::F32,
    };
    Ok(dtype)
}

pub fn write_json_pretty(path: &Path, value: &impl Serialize) -> Result<()> {
    let file = File::create(path)?;
    serde_json::to_writer_pretty(file, value)?;
    Ok(())
}

pub fn format_float(value: f64) -> String {
    format!("{value:.6}")
}
