use std::fs::File;
use std::path::Path;

use anyhow::{Result, bail};
use candle_core::Device;
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
        bail!("cuda requested but this binary was not built with the `cuda` feature");
    }
}

pub fn write_json_pretty(path: &Path, value: &impl Serialize) -> Result<()> {
    let file = File::create(path)?;
    serde_json::to_writer_pretty(file, value)?;
    Ok(())
}

pub fn format_float(value: f64) -> String {
    format!("{value:.6}")
}
