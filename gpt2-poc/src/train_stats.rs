use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainPoint {
    pub step: u64,
    pub tokens_seen: u64,
    pub elapsed_sec: f32,
    pub train_loss: f32,
    pub val_loss: Option<f32>,
    pub train_bpb: f32,
    pub val_bpb: Option<f32>,
    pub mini_core: Option<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct TrainStats {
    pub points: Vec<TrainPoint>,
}

impl TrainStats {
    pub fn push(&mut self, point: TrainPoint) {
        self.points.push(point);
    }

    pub fn recent(&self, n: usize) -> &[TrainPoint] {
        let start = self.points.len().saturating_sub(n);
        &self.points[start..]
    }

    pub fn last_mut(&mut self) -> Option<&mut TrainPoint> {
        self.points.last_mut()
    }

    pub fn last(&self) -> Option<&TrainPoint> {
        self.points.last()
    }

    pub fn retain_up_to_step(&mut self, max_step: u64) {
        self.points.retain(|point| point.step <= max_step);
    }

    pub fn load_jsonl(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }

        let file = File::open(path)
            .with_context(|| format!("failed to open training stats {}", path.display()))?;
        let reader = BufReader::new(file);
        let mut points = Vec::new();
        for (line_idx, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let point = serde_json::from_str::<TrainPoint>(&line).with_context(|| {
                format!(
                    "failed to parse training stats JSONL at {} line {}",
                    path.display(),
                    line_idx + 1
                )
            })?;
            points.push(point);
        }
        Ok(Self { points })
    }

    pub fn rewrite_jsonl(&self, path: &Path) -> Result<()> {
        let mut file = File::create(path)
            .with_context(|| format!("failed to rewrite training stats {}", path.display()))?;
        for point in &self.points {
            serde_json::to_writer(&mut file, point)?;
            writeln!(file)?;
        }
        Ok(())
    }
}
