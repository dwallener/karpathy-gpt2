use anyhow::Result;
use candle_core::{Device, Tensor};

use crate::stream_dataset::TokenChunk;

#[derive(Debug, Clone, Default)]
pub struct TokenBatchMetrics {
    pub docs_consumed: usize,
    pub current_shard: String,
    pub tokenizer_queue_depth: usize,
    pub parquet_queue_depth: usize,
}

pub struct TokenBatcher {
    seq_len: usize,
    batch_size: usize,
    token_buffer: Vec<u32>,
    x_buf: Vec<u32>,
    y_buf: Vec<u32>,
}

impl TokenBatcher {
    pub fn new(seq_len: usize, batch_size: usize) -> Self {
        let flat_len = seq_len * batch_size;
        Self {
            seq_len,
            batch_size,
            token_buffer: Vec::with_capacity(flat_len * 4),
            x_buf: Vec::with_capacity(flat_len),
            y_buf: Vec::with_capacity(flat_len),
        }
    }

    pub fn push(&mut self, chunk: TokenChunk) -> TokenBatchMetrics {
        self.token_buffer.extend_from_slice(&chunk.tokens);
        TokenBatchMetrics {
            docs_consumed: chunk.docs,
            current_shard: chunk.current_shard,
            tokenizer_queue_depth: chunk.tokenizer_queue_depth,
            parquet_queue_depth: chunk.parquet_queue_depth,
        }
    }

    pub fn try_build_batch(&mut self, device: &Device) -> Result<Option<(Tensor, Tensor)>> {
        if self.token_buffer.len() < self.batch_token_requirement() {
            return Ok(None);
        }

        self.x_buf.clear();
        self.y_buf.clear();
        let mut offset = 0usize;
        for _ in 0..self.batch_size {
            self.x_buf
                .extend_from_slice(&self.token_buffer[offset..offset + self.seq_len]);
            self.y_buf
                .extend_from_slice(&self.token_buffer[offset + 1..offset + self.seq_len + 1]);
            offset += self.seq_len;
        }
        self.token_buffer.drain(..offset);

        let xs = Tensor::from_vec(self.x_buf.clone(), (self.batch_size, self.seq_len), device)?;
        let ys = Tensor::from_vec(self.y_buf.clone(), (self.batch_size, self.seq_len), device)?;
        Ok(Some((xs, ys)))
    }

    fn batch_token_requirement(&self) -> usize {
        self.seq_len * self.batch_size + 1
    }
}
