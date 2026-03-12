use crate::train_stats::TrainStats;

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

pub fn predict_final_loss(stats: &TrainStats) -> Option<ScalingPrediction> {
    let mut n = 0f64;
    let mut sum_x = 0f64;
    let mut sum_y = 0f64;
    let mut sum_xx = 0f64;
    let mut sum_xy = 0f64;

    for point in stats.recent(200) {
        let Some(val_loss) = point.val_loss else {
            continue;
        };
        if point.tokens_seen < MIN_TOKENS_SEEN || val_loss <= 0.0 {
            continue;
        }
        let x = (point.tokens_seen as f64).log10();
        let y = (val_loss as f64).log10();
        n += 1.0;
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }

    if n < MIN_POINTS as f64 {
        return None;
    }

    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-9 {
        return None;
    }

    let m = (n * sum_xy - sum_x * sum_y) / denom;
    let b = (sum_y - m * sum_x) / n;
    let alpha = (-m) as f32;

    let predicted_loss_10m = predict_loss(m, b, 10_000_000.0);
    let predicted_loss_100m = predict_loss(m, b, 100_000_000.0);
    let predicted_loss_1b = predict_loss(m, b, 1_000_000_000.0);
    let predicted_mini_core = estimate_mini_core(predicted_loss_100m);

    Some(ScalingPrediction {
        alpha,
        predicted_loss_10m,
        predicted_loss_100m,
        predicted_loss_1b,
        predicted_mini_core,
    })
}

fn predict_loss(m: f64, b: f64, tokens: f64) -> f32 {
    let log_loss = m * tokens.log10() + b;
    10f64.powf(log_loss) as f32
}

fn estimate_mini_core(loss: f32) -> f32 {
    (0.25 * (11.0 - loss)).clamp(0.0, 0.25)
}
