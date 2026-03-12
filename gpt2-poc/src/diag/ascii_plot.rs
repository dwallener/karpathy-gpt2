use crate::train_stats::{TrainPoint, TrainStats};

const WIDTH: usize = 60;
const HEIGHT: usize = 12;
const RECENT_POINTS: usize = 200;
const SMOOTH_WINDOW: usize = 5;

pub struct DiagnosticsReport {
    pub latest_step: u64,
    pub latest_tokens_seen: u64,
    pub latest_train_loss: f32,
    pub latest_val_loss: Option<f32>,
    pub latest_mini_core: Option<f32>,
    pub learning_slope: Option<f64>,
    pub plot: String,
}

pub fn build_diagnostics(stats: &TrainStats) -> Option<DiagnosticsReport> {
    let recent = stats.recent(RECENT_POINTS);
    let latest = recent.last()?;

    Some(DiagnosticsReport {
        latest_step: latest.step,
        latest_tokens_seen: latest.tokens_seen,
        latest_train_loss: latest.train_loss,
        latest_val_loss: latest.val_loss,
        latest_mini_core: latest.mini_core,
        learning_slope: compute_recent_slope(recent),
        plot: plot_loss_vs_tokens(stats),
    })
}

pub fn plot_loss_vs_tokens(stats: &TrainStats) -> String {
    let recent = stats.recent(RECENT_POINTS);
    if recent.is_empty() {
        return "Loss vs Tokens (log scale)\n\n(no data)\n".to_string();
    }

    let train_series = smooth_train_series(recent);
    let val_series = smooth_val_series(recent);
    let mut all_losses = train_series.iter().map(|(_, y)| *y).collect::<Vec<_>>();
    all_losses.extend(val_series.iter().map(|(_, y)| *y));
    if all_losses.is_empty() {
        return "Loss vs Tokens (log scale)\n\n(no plottable losses)\n".to_string();
    }

    let min_loss = all_losses.iter().copied().fold(f64::INFINITY, f64::min);
    let max_loss = all_losses.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let y_span = (max_loss - min_loss).max(1e-6);

    let x_values = recent
        .iter()
        .map(|point| (point.tokens_seen.max(1) as f64).log10())
        .collect::<Vec<_>>();
    let min_x = x_values.iter().copied().fold(f64::INFINITY, f64::min);
    let max_x = x_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let x_span = (max_x - min_x).max(1e-6);

    let mut grid = vec![vec![' '; WIDTH]; HEIGHT];
    plot_series(&mut grid, &train_series, min_x, x_span, min_loss, y_span, '.');
    plot_series(&mut grid, &val_series, min_x, x_span, min_loss, y_span, '*');

    let mut lines = Vec::with_capacity(HEIGHT + 3);
    lines.push("Loss vs Tokens (log scale)".to_string());
    lines.push(String::new());
    for row in 0..HEIGHT {
        let y = max_loss - (row as f64 / (HEIGHT.saturating_sub(1).max(1) as f64)) * y_span;
        let label = format!("{:>4.1}", y);
        let row_text = grid[row].iter().collect::<String>();
        lines.push(format!("{label} | {row_text}"));
    }
    lines.push(format!("     +{}", "-".repeat(WIDTH)));
    lines.push(format!(
        "      {}   {}   {} tokens",
        token_tick(10f64.powf(min_x)),
        token_tick(10f64.powf((min_x + max_x) / 2.0)),
        token_tick(10f64.powf(max_x)),
    ));
    lines.join("\n")
}

fn plot_series(
    grid: &mut [Vec<char>],
    series: &[(u64, f64)],
    min_x: f64,
    x_span: f64,
    min_loss: f64,
    y_span: f64,
    marker: char,
) {
    for (tokens_seen, loss) in series {
        let x = ((*tokens_seen).max(1) as f64).log10();
        let x_norm = ((x - min_x) / x_span).clamp(0.0, 1.0);
        let y_norm = ((*loss - min_loss) / y_span).clamp(0.0, 1.0);
        let col = (x_norm * (WIDTH.saturating_sub(1) as f64)).round() as usize;
        let row = HEIGHT.saturating_sub(1)
            - (y_norm * (HEIGHT.saturating_sub(1) as f64)).round() as usize;
        grid[row][col] = marker;
    }
}

fn compute_recent_slope(points: &[TrainPoint]) -> Option<f64> {
    let series = smooth_val_series(points);
    if series.len() >= 2 {
        return slope_between(series[series.len() - 2], series[series.len() - 1]);
    }
    let series = smooth_train_series(points);
    if series.len() >= 2 {
        return slope_between(series[series.len() - 2], series[series.len() - 1]);
    }
    None
}

fn slope_between(a: (u64, f64), b: (u64, f64)) -> Option<f64> {
    let x0 = (a.0.max(1) as f64).log10();
    let x1 = (b.0.max(1) as f64).log10();
    let dx = x1 - x0;
    if dx.abs() < 1e-9 {
        None
    } else {
        Some((b.1 - a.1) / dx)
    }
}

fn smooth_train_series(points: &[TrainPoint]) -> Vec<(u64, f64)> {
    let raw = points
        .iter()
        .map(|point| (point.tokens_seen, point.train_loss as f64))
        .collect::<Vec<_>>();
    smooth_series(&raw)
}

fn smooth_val_series(points: &[TrainPoint]) -> Vec<(u64, f64)> {
    let raw = points
        .iter()
        .filter_map(|point| point.val_loss.map(|loss| (point.tokens_seen, loss as f64)))
        .collect::<Vec<_>>();
    smooth_series(&raw)
}

fn smooth_series(raw: &[(u64, f64)]) -> Vec<(u64, f64)> {
    if raw.is_empty() {
        return Vec::new();
    }
    let mut smoothed = Vec::with_capacity(raw.len());
    for idx in 0..raw.len() {
        let start = idx.saturating_sub(SMOOTH_WINDOW - 1);
        let window = &raw[start..=idx];
        let avg = window.iter().map(|(_, loss)| *loss).sum::<f64>() / window.len() as f64;
        smoothed.push((raw[idx].0, avg));
    }
    smoothed
}

fn token_tick(tokens: f64) -> String {
    if tokens >= 1_000_000_000.0 {
        format!("{:.1}e9", tokens / 1_000_000_000.0)
    } else if tokens >= 1_000_000.0 {
        format!("{:.1}e6", tokens / 1_000_000.0)
    } else if tokens >= 1_000.0 {
        format!("{:.0}k", tokens / 1_000.0)
    } else {
        format!("{:.0}", tokens)
    }
}
