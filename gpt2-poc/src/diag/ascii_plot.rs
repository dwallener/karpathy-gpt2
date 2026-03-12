use crate::diag::scaling_predictor::{ScalingPrediction, compute_learning_slope, predict_final_loss};
use crate::train_stats::{RouterMetricPoint, TrainPoint, TrainStats};

const WIDTH: usize = 72;
const HEIGHT: usize = 16;
const WINDOW_TOKENS: u64 = 2_000_000;

pub struct DiagnosticsReport {
    pub latest_step: u64,
    pub latest_tokens_seen: u64,
    pub latest_elapsed_sec: f32,
    pub latest_train_loss: f32,
    pub latest_train_bpb: f32,
    pub latest_val_loss: Option<f32>,
    pub latest_val_bpb: Option<f32>,
    pub latest_mini_core: Option<f32>,
    pub learning_slope: f32,
    pub router: RouterMetricPoint,
    pub tokens_per_second: f32,
    pub tokens_per_hour: f32,
    pub scaling_prediction: Option<ScalingPrediction>,
    pub plot: String,
}

pub fn build_diagnostics(stats: &TrainStats) -> Option<DiagnosticsReport> {
    let points = recent_window_points(stats);
    let latest = points.last()?;
    let elapsed_sec = latest.elapsed_sec.max(1e-6);

    Some(DiagnosticsReport {
        latest_step: latest.step,
        latest_tokens_seen: latest.tokens_seen,
        latest_elapsed_sec: latest.elapsed_sec,
        latest_train_loss: latest.train_loss,
        latest_train_bpb: latest.train_bpb,
        latest_val_loss: latest.val_loss,
        latest_val_bpb: latest.val_bpb,
        latest_mini_core: latest.mini_core,
        learning_slope: compute_learning_slope(points),
        router: latest.router.clone(),
        tokens_per_second: latest.tokens_seen as f32 / elapsed_sec,
        tokens_per_hour: latest.tokens_seen as f32 / elapsed_sec * 3600.0,
        scaling_prediction: predict_final_loss(stats),
        plot: plot_loss_vs_tokens(stats),
    })
}

pub fn plot_loss_vs_tokens(stats: &TrainStats) -> String {
    let points = recent_window_points(stats);
    if points.is_empty() {
        return "Loss / BPB vs Tokens (log scale)\n\n(no data)\n".to_string();
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
        return "Loss / BPB vs Tokens (log scale)\n\n(no plottable losses)\n".to_string();
    }

    let mut y_min = (min_loss * 10.0).floor() / 10.0;
    let mut y_max = (max_loss * 10.0).ceil() / 10.0;
    if (y_max - y_min).abs() < 1e-6 {
        y_min -= 0.1;
        y_max += 0.1;
    }

    let mut grid = vec![vec![' '; WIDTH]; HEIGHT];
    render_bucketed_series(&mut grid, &train_buckets, y_min, y_max, '.');
    render_bucketed_series(&mut grid, &val_buckets, y_min, y_max, '*');

    let mut lines = Vec::with_capacity(HEIGHT + 4);
    lines.push("Loss / BPB vs Tokens (log scale)".to_string());
    lines.push(String::new());
    for row in 0..HEIGHT {
        let frac = row as f32 / HEIGHT.saturating_sub(1).max(1) as f32;
        let label = y_max - frac * (y_max - y_min);
        let row_text = grid[row].iter().collect::<String>();
        lines.push(format!("{:>4.1} | {}", label, row_text));
    }
    lines.push(format!("     +{}", "-".repeat(WIDTH)));
    let min_tokens = points.first().map(|p| p.tokens_seen).unwrap_or(0);
    let max_tokens = points.last().map(|p| p.tokens_seen).unwrap_or(min_tokens);
    lines.push(format!(
        "      {}",
        token_axis_labels(min_tokens, max_tokens, WIDTH)
    ));
    lines.join("\n")
}

fn recent_window_points(stats: &TrainStats) -> &[TrainPoint] {
    if stats.points.is_empty() {
        return &stats.points[..];
    }
    let latest_tokens = stats.points.last().map(|p| p.tokens_seen).unwrap_or(0);
    let window_start = latest_tokens.saturating_sub(WINDOW_TOKENS);
    let start = stats
        .points
        .iter()
        .position(|point| point.tokens_seen >= window_start)
        .unwrap_or(0);
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

fn token_axis_labels(min_tokens: u64, max_tokens: u64, width: usize) -> String {
    let ticks = linear_ticks(min_tokens, max_tokens);
    let mut chars = vec![' '; width];
    let mut occupied = vec![false; width];
    for (tokens, label) in ticks {
        let frac = if max_tokens == min_tokens {
            0.0
        } else {
            (tokens.saturating_sub(min_tokens)) as f32 / (max_tokens - min_tokens).max(1) as f32
        }
        .clamp(0.0, 1.0);
        let col = ((width.saturating_sub(1) as f32) * frac).round() as usize;
        let start = centered_label_start(col, label.len(), width);
        let end = (start + label.len()).min(width);
        if occupied[start..end].iter().any(|used| *used) {
            continue;
        }
        for (offset, ch) in label.chars().enumerate() {
            let pos = start + offset;
            if pos < width {
                chars[pos] = ch;
                occupied[pos] = true;
            }
        }
    }

    chars.into_iter().collect::<String>() + " tokens"
}

fn centered_label_start(col: usize, label_len: usize, width: usize) -> usize {
    col.saturating_sub(label_len / 2)
        .min(width.saturating_sub(label_len))
}

fn linear_ticks(min_tokens: u64, max_tokens: u64) -> Vec<(u64, String)> {
    let span = max_tokens.saturating_sub(min_tokens).max(1);
    let step = choose_linear_tick_step(span);
    let first_tick = (min_tokens / step) * step;
    let mut ticks = Vec::new();
    let mut value = first_tick;
    while value <= max_tokens {
        if value >= min_tokens {
            ticks.push((value, format_compact_tick(value)));
        }
        match value.checked_add(step) {
            Some(next) => value = next,
            None => break,
        }
    }
    if ticks.is_empty() {
        ticks.push((min_tokens, format_compact_tick(min_tokens)));
        if max_tokens != min_tokens {
            ticks.push((max_tokens, format_compact_tick(max_tokens)));
        }
    }
    ticks
}

fn choose_linear_tick_step(span: u64) -> u64 {
    match span {
        0..=1_200_000 => 100_000,
        1_200_001..=2_400_000 => 200_000,
        2_400_001..=6_000_000 => 500_000,
        _ => 1_000_000,
    }
}

fn format_compact_tick(tokens: u64) -> String {
    if tokens >= 1_000_000_000 {
        format!("{:.1}B", tokens as f64 / 1_000_000_000.0)
    } else if tokens >= 1_000_000 {
        format!("{:.1}M", tokens as f64 / 1_000_000.0)
    } else if tokens >= 1_000 {
        format!("{:.0}k", tokens as f64 / 1_000.0)
    } else {
        tokens.to_string()
    }
}
