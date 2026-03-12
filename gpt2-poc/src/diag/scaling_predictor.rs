use crate::train_stats::{TrainPoint, TrainStats};

const MIN_TOKENS_SEEN: u64 = 100_000;
const MIN_POINTS: usize = 5;

#[derive(Debug, Clone, Copy)]
pub struct ScalingPrediction {
    pub alpha: f32,
    pub predicted_loss_10m: f32,
    pub predicted_loss_100m: f32,
    pub predicted_loss_1b: f32,
    pub predicted_mini_core: f32,
}

pub fn compute_learning_slope(points: &[TrainPoint]) -> f32 {
    let n = points.len();
    if n < 5 {
        return 0.0;
    }

    let mut sx = 0.0f32;
    let mut sy = 0.0f32;
    let mut sxx = 0.0f32;
    let mut sxy = 0.0f32;

    for point in points {
        let x = (point.tokens_seen.max(1) as f32).log10();
        let y = point.train_loss;
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }

    let n = n as f32;
    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-6 {
        return 0.0;
    }

    (n * sxy - sx * sy) / denom
}

pub fn predict_final_loss(stats: &TrainStats) -> Option<ScalingPrediction> {
    let mut n = 0f32;
    let mut sx = 0f32;
    let mut sy = 0f32;
    let mut sxx = 0f32;
    let mut sxy = 0f32;

    for point in stats.points.iter().rev().take(2000).rev() {
        let Some(val_loss) = point.val_loss else {
            continue;
        };
        if point.tokens_seen < MIN_TOKENS_SEEN || val_loss <= 0.0 {
            continue;
        }
        let x = (point.tokens_seen as f32).log10();
        let y = val_loss.log10();
        n += 1.0;
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }

    if n < MIN_POINTS as f32 {
        return None;
    }

    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-6 {
        return None;
    }

    let m = (n * sxy - sx * sy) / denom;
    let b = (sy - m * sx) / n;
    let alpha = -m;

    let predicted_loss_10m = predict_loss(m, b, 10_000_000.0);
    let predicted_loss_100m = predict_loss(m, b, 100_000_000.0);
    let predicted_loss_1b = predict_loss(m, b, 1_000_000_000.0);

    Some(ScalingPrediction {
        alpha,
        predicted_loss_10m,
        predicted_loss_100m,
        predicted_loss_1b,
        predicted_mini_core: estimate_mini_core(predicted_loss_100m),
    })
}

fn predict_loss(m: f32, b: f32, tokens: f32) -> f32 {
    10f32.powf(m * tokens.log10() + b)
}

fn estimate_mini_core(loss: f32) -> f32 {
    (0.25 * (11.0 - loss)).clamp(0.0, 0.25)
}
