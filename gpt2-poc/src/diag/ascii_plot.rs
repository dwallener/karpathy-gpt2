use crate::diag::scaling_predictor::{ScalingPrediction, compute_learning_slope, predict_final_loss};
use crate::train_stats::{TrainPoint, TrainStats};

const WIDTH: usize = 72;
const HEIGHT: usize = 16;
const MAX_POINTS: usize = 2000;

pub struct DiagnosticsReport {
    pub latest_step: u64,
    pub latest_tokens_seen: u64,
    pub latest_elapsed_sec: f32,
    pub latest_train_loss: f32,
    pub latest_val_loss: Option<f32>,
    pub latest_mini_core: Option<f32>,
    pub learning_slope: f32,
    pub tokens_per_second: f32,
    pub tokens_per_hour: f32,
    pub scaling_prediction: Option<ScalingPrediction>,
    pub plot: String,
}

pub fn build_diagnostics(stats: &TrainStats) -> Option<DiagnosticsReport> {
    let points = recent_points(stats);
    let latest = points.last()?;
    let elapsed_sec = latest.elapsed_sec.max(1e-6);

    Some(DiagnosticsReport {
        latest_step: latest.step,
        latest_tokens_seen: latest.tokens_seen,
        latest_elapsed_sec: latest.elapsed_sec,
        latest_train_loss: latest.train_loss,
        latest_val_loss: latest.val_loss,
        latest_mini_core: latest.mini_core,
        learning_slope: compute_learning_slope(points),
        tokens_per_second: latest.tokens_seen as f32 / elapsed_sec,
        tokens_per_hour: latest.tokens_seen as f32 / elapsed_sec * 3600.0,
        scaling_prediction: predict_final_loss(stats),
        plot: plot_loss_vs_tokens(stats),
    })
}

pub fn plot_loss_vs_tokens(stats: &TrainStats) -> String {
    let points = recent_points(stats);
    if points.is_empty() {
        return "Loss vs Tokens (log scale)\n\n(no data)\n".to_string();
    }

    let train_buckets = bucketize_train(points, WIDTH);
    let val_buckets = bucketize_val(points, WIDTH);

    let mut min_loss = f32::INFINITY;
    let mut max_loss = f32::NEG_INFINITY;
    for value in train_buckets.iter().flatten().chain(val_buckets.iter().flatten()) {
        min_loss = min_loss.min(*value);
        max_loss = max_loss.max(*value);
    }
    if !min_loss.is_finite() || !max_loss.is_finite() {
        return "Loss vs Tokens (log scale)\n\n(no plottable losses)\n".to_string();
    }

    let mut y_min = (min_loss * 10.0).floor() / 10.0;
    let mut y_max = (max_loss * 10.0).ceil() / 10.0;
    if (y_max - y_min).abs() < 1e-6 {
        y_min -= 0.1;
        y_max += 0.1;
    }

    let min_tokens = points.first().map(|p| p.tokens_seen.max(1)).unwrap_or(1);
    let max_tokens = points.last().map(|p| p.tokens_seen.max(1)).unwrap_or(min_tokens);
    let log_min = (min_tokens as f32).log10();
    let log_max = (max_tokens as f32).log10().max(log_min + 1e-6);

    let mut grid = vec![vec![' '; WIDTH]; HEIGHT];
    render_bucketed_series(&mut grid, &train_buckets, y_min, y_max, '.');
    render_bucketed_series(&mut grid, &val_buckets, y_min, y_max, '*');

    let mut lines = Vec::with_capacity(HEIGHT + 4);
    lines.push("Loss vs Tokens (log scale)".to_string());
    lines.push(String::new());
    for row in 0..HEIGHT {
        let frac = row as f32 / HEIGHT.saturating_sub(1).max(1) as f32;
        let label = y_max - frac * (y_max - y_min);
        let row_text = grid[row].iter().collect::<String>();
        lines.push(format!("{:>4.1} | {}", label, row_text));
    }
    lines.push(format!("     +{}", "-".repeat(WIDTH)));
    lines.push(format!(
        "      {}",
        token_axis_labels(log_min, log_max, WIDTH)
    ));
    lines.join("\n")
}

fn recent_points(stats: &TrainStats) -> &[TrainPoint] {
    let len = stats.points.len();
    let start = len.saturating_sub(MAX_POINTS);
    &stats.points[start..]
}

fn bucketize_train(points: &[TrainPoint], width: usize) -> Vec<Option<f32>> {
    bucketize(points, width, |point| Some(point.train_loss))
}

fn bucketize_val(points: &[TrainPoint], width: usize) -> Vec<Option<f32>> {
    bucketize(points, width, |point| point.val_loss)
}

fn bucketize<F>(points: &[TrainPoint], width: usize, mut value_fn: F) -> Vec<Option<f32>>
where
    F: FnMut(&TrainPoint) -> Option<f32>,
{
    if points.is_empty() {
        return Vec::new();
    }

    let mut buckets = Vec::with_capacity(width);
    let bucket_size = points.len().div_ceil(width).max(1);
    for start in (0..points.len()).step_by(bucket_size) {
        let end = (start + bucket_size).min(points.len());
        let mut sum = 0f32;
        let mut count = 0usize;
        for point in &points[start..end] {
            if let Some(value) = value_fn(point) {
                sum += value;
                count += 1;
            }
        }
        buckets.push((count > 0).then_some(sum / count as f32));
    }
    if buckets.len() < width {
        buckets.resize(width, None);
    } else if buckets.len() > width {
        buckets.truncate(width);
    }
    buckets
}

fn render_bucketed_series(
    grid: &mut [Vec<char>],
    series: &[Option<f32>],
    y_min: f32,
    y_max: f32,
    marker: char,
) {
    let y_span = (y_max - y_min).max(1e-6);
    for (col, value) in series.iter().enumerate().take(WIDTH) {
        let Some(value) = value else {
            continue;
        };
        let norm = ((*value - y_min) / y_span).clamp(0.0, 1.0);
        let row = HEIGHT.saturating_sub(1)
            - (norm * (HEIGHT.saturating_sub(1) as f32)).round() as usize;
        grid[row][col] = marker;
    }
}

fn token_axis_labels(log_min: f32, log_max: f32, width: usize) -> String {
    let ticks = 5usize;
    let mut chars = vec![' '; width];
    for idx in 0..ticks {
        let frac = if ticks == 1 {
            0.0
        } else {
            idx as f32 / (ticks - 1) as f32
        };
        let tick = log_min + frac * (log_max - log_min);
        let label = format_token_tick(10f32.powf(tick));
        let col = ((width.saturating_sub(1) as f32) * frac).round() as usize;
        for (offset, ch) in label.chars().enumerate() {
            let pos = col.saturating_add(offset).min(width.saturating_sub(1));
            chars[pos] = ch;
        }
    }
    chars.into_iter().collect::<String>() + " tokens"
}

fn format_token_tick(tokens: f32) -> String {
    let log = tokens.log10();
    let exponent = log.floor() as i32;
    let mantissa = 10f32.powf(log - exponent as f32);
    if (mantissa - 1.0).abs() < 0.15 {
        format!("1e{exponent}")
    } else if (mantissa - 3.0).abs() < 0.35 {
        format!("3e{exponent}")
    } else {
        format!("{:.0}e{}", mantissa, exponent)
    }
}
