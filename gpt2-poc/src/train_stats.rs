use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainPoint {
    pub step: u64,
    pub tokens_seen: u64,
    pub train_loss: f32,
    pub val_loss: Option<f32>,
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
}
