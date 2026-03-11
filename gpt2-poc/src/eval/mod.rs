use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};

use crate::infer::InferenceSession;
use crate::utils::format_float;

pub mod arc_easy;
pub mod hellaswag;
pub mod piqa;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalExample {
    pub prompt: String,
    pub choices: Vec<String>,
    pub correct_index: usize,
}

#[derive(Debug, Clone)]
pub struct DatasetScore {
    pub name: &'static str,
    pub accuracy: f64,
    pub centered_accuracy: f64,
    pub examples: usize,
    pub examples_per_sec: f64,
}

#[derive(Debug, Clone)]
pub struct MiniCoreReport {
    pub datasets: Vec<DatasetScore>,
    pub mini_core: f64,
}

pub fn run_mini_core(
    checkpoint: &Path,
    device: &candle_core::Device,
    requested_dtype: Option<&str>,
    limit: Option<usize>,
) -> Result<MiniCoreReport> {
    let session = InferenceSession::load(checkpoint, device, requested_dtype)?;
    let cache_dir = PathBuf::from("data").join("eval");
    fs::create_dir_all(&cache_dir)?;

    let datasets = vec![
        evaluate_dataset(
            &session,
            "HellaSwag",
            0.25,
            hellaswag::load_examples(&cache_dir)?,
            limit,
        )?,
        evaluate_dataset(&session, "PIQA", 0.50, piqa::load_examples(&cache_dir)?, limit)?,
        evaluate_dataset(
            &session,
            "ARC-Easy",
            0.25,
            arc_easy::load_examples(&cache_dir)?,
            limit,
        )?,
    ];

    let mini_core = datasets
        .iter()
        .map(|score| score.centered_accuracy)
        .sum::<f64>()
        / datasets.len() as f64;

    Ok(MiniCoreReport {
        datasets,
        mini_core,
    })
}

fn evaluate_dataset(
    session: &InferenceSession,
    name: &'static str,
    random_baseline: f64,
    mut examples: Vec<EvalExample>,
    limit: Option<usize>,
) -> Result<DatasetScore> {
    if let Some(limit) = limit {
        examples.truncate(limit);
    }
    if examples.is_empty() {
        bail!("{name} produced zero evaluation examples");
    }

    let progress = ProgressBar::new(examples.len() as u64);
    progress.set_style(ProgressStyle::with_template(
        "{msg} [{elapsed_precise}] {wide_bar} {pos}/{len}",
    )?);
    progress.set_message(format!("{name} acc=na"));

    let start = Instant::now();
    let mut correct = 0usize;
    for (idx, example) in examples.iter().enumerate() {
        let prediction = best_choice(session, example)?;
        if prediction == example.correct_index {
            correct += 1;
        }
        let seen = idx + 1;
        let accuracy = correct as f64 / seen as f64;
        progress.set_message(format!("{name} acc={}", format_float(accuracy)));
        progress.inc(1);
    }
    progress.finish_and_clear();

    let accuracy = correct as f64 / examples.len() as f64;
    let centered_accuracy = accuracy - random_baseline;
    let elapsed = start.elapsed().as_secs_f64().max(1e-9);
    Ok(DatasetScore {
        name,
        accuracy,
        centered_accuracy,
        examples: examples.len(),
        examples_per_sec: examples.len() as f64 / elapsed,
    })
}

fn best_choice(session: &InferenceSession, example: &EvalExample) -> Result<usize> {
    let mut best_index = 0usize;
    let mut best_score = f64::NEG_INFINITY;
    for (idx, choice) in example.choices.iter().enumerate() {
        let score = session.score_continuation(&example.prompt, choice)?;
        if score > best_score {
            best_score = score;
            best_index = idx;
        }
    }
    Ok(best_index)
}

pub fn download_to_cache(cache_path: &Path, urls: &[&str]) -> Result<()> {
    if cache_path.exists() {
        return Ok(());
    }
    let parent = cache_path
        .parent()
        .context("cache path must have a parent directory")?;
    fs::create_dir_all(parent)?;

    let client = Client::builder().build()?;
    let mut last_error = None;
    for url in urls {
        match client.get(*url).send() {
            Ok(response) => {
                let response = response.error_for_status()?;
                let mut file = File::create(cache_path)?;
                let body = response.bytes()?;
                file.write_all(&body)?;
                return Ok(());
            }
            Err(err) => last_error = Some(anyhow::anyhow!("{}: {err}", url)),
        }
    }

    Err(last_error.unwrap_or_else(|| {
        anyhow::anyhow!("no download URLs were configured for {}", cache_path.display())
    }))
}

pub fn read_jsonl<T>(path: &Path) -> Result<Vec<T>>
where
    T: for<'de> Deserialize<'de>,
{
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut rows = Vec::new();
    for (line_idx, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let value = serde_json::from_str::<T>(&line).with_context(|| {
            format!("failed to parse JSONL at {} line {}", path.display(), line_idx + 1)
        })?;
        rows.push(value);
    }
    Ok(rows)
}
