use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use candle_core::{Device, Tensor};
use memmap2::{Mmap, MmapOptions};
use rand::Rng;

pub struct TokenDataset {
    path: PathBuf,
    mmap: Arc<Mmap>,
    len_tokens: usize,
    max_token: u32,
}

pub struct Batch {
    pub xs: Tensor,
    pub ys: Tensor,
    pub starts: Vec<usize>,
}

impl TokenDataset {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)
            .with_context(|| format!("failed to open token file {}", path.display()))?;
        let mmap = unsafe { MmapOptions::new().map(&file) }
            .with_context(|| format!("failed to mmap token file {}", path.display()))?;
        if mmap.len() % 4 != 0 {
            bail!(
                "token file {} has {} bytes, not divisible by 4",
                path.display(),
                mmap.len()
            );
        }
        let len_tokens = mmap.len() / 4;
        if len_tokens < 2 {
            bail!(
                "token file {} must contain at least 2 tokens",
                path.display()
            );
        }

        let mut max_token = 0u32;
        for idx in 0..len_tokens {
            max_token = max_token.max(read_u32_le(&mmap, idx)?);
        }

        Ok(Self {
            path,
            mmap: Arc::new(mmap),
            len_tokens,
            max_token,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn len_tokens(&self) -> usize {
        self.len_tokens
    }

    pub fn max_token(&self) -> u32 {
        self.max_token
    }

    pub fn derived_vocab_size(&self, other: Option<&TokenDataset>) -> usize {
        let other_max = other.map(|ds| ds.max_token()).unwrap_or(0);
        self.max_token.max(other_max) as usize + 1
    }

    pub fn sample_batch<R: Rng + ?Sized>(
        &self,
        batch_size: usize,
        seq_len: usize,
        rng: &mut R,
        device: &Device,
    ) -> Result<Batch> {
        if self.len_tokens <= seq_len + 1 {
            bail!(
                "dataset {} has {} tokens but needs > {} for seq_len {}",
                self.path.display(),
                self.len_tokens,
                seq_len + 1,
                seq_len
            );
        }
        let max_start = self.len_tokens - seq_len - 1;
        let mut starts = Vec::with_capacity(batch_size);
        let mut xs = Vec::with_capacity(batch_size * seq_len);
        let mut ys = Vec::with_capacity(batch_size * seq_len);
        for _ in 0..batch_size {
            let start = rng.random_range(0..=max_start);
            starts.push(start);
            for offset in 0..seq_len {
                xs.push(read_u32_le(&self.mmap, start + offset)?);
                ys.push(read_u32_le(&self.mmap, start + offset + 1)?);
            }
        }
        let xs = Tensor::from_vec(xs, (batch_size, seq_len), device)?;
        let ys = Tensor::from_vec(ys, (batch_size, seq_len), device)?;
        Ok(Batch { xs, ys, starts })
    }

    pub fn inspect_window(&self, start: usize, len: usize) -> Result<Vec<u32>> {
        if start + len > self.len_tokens {
            bail!(
                "requested window [{}..{}) exceeds dataset length {}",
                start,
                start + len,
                self.len_tokens
            );
        }
        let mut out = Vec::with_capacity(len);
        for idx in start..start + len {
            out.push(read_u32_le(&self.mmap, idx)?);
        }
        Ok(out)
    }
}

fn read_u32_le(mmap: &[u8], index: usize) -> Result<u32> {
    let start = index
        .checked_mul(4)
        .ok_or_else(|| anyhow!("u32 index overflow for index {index}"))?;
    let end = start + 4;
    let bytes = mmap
        .get(start..end)
        .ok_or_else(|| anyhow!("failed to read token at u32 index {index}"))?;
    Ok(u32::from_le_bytes(bytes.try_into()?))
}
