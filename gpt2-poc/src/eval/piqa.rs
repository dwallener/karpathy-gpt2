use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::eval::{EvalExample, download_to_cache, read_jsonl};

const DATASET_DIR: &str = "piqa";
const DATA_FILE: &str = "valid.jsonl";
const LABEL_FILE: &str = "valid-labels.lst";
const DATA_URLS: &[&str] = &[
    "https://raw.githubusercontent.com/ybisk/physicaliqa/master/piqa/valid.jsonl",
];
const LABEL_URLS: &[&str] = &[
    "https://raw.githubusercontent.com/ybisk/physicaliqa/master/piqa/valid-labels.lst",
];

#[derive(Debug, Deserialize)]
struct PiqaRow {
    goal: String,
    sol1: String,
    sol2: String,
}

pub fn load_examples(root: &Path) -> Result<Vec<EvalExample>> {
    let dir = root.join(DATASET_DIR);
    let data_path = dir.join(DATA_FILE);
    let label_path = dir.join(LABEL_FILE);
    download_to_cache(&data_path, DATA_URLS)?;
    download_to_cache(&label_path, LABEL_URLS)?;

    let rows = read_jsonl::<PiqaRow>(&data_path)?;
    let labels = std::fs::read_to_string(&label_path)?;
    let label_values = labels
        .lines()
        .enumerate()
        .map(|(idx, line)| {
            line.trim().parse::<usize>().with_context(|| {
                format!("failed to parse PIQA label at {} line {}", label_path.display(), idx + 1)
            })
        })
        .collect::<Result<Vec<_>>>()?;

    if rows.len() != label_values.len() {
        anyhow::bail!(
            "PIQA rows/labels mismatch: {} rows vs {} labels",
            rows.len(),
            label_values.len()
        );
    }

    let mut examples = Vec::with_capacity(rows.len());
    for (row, correct_index) in rows.into_iter().zip(label_values) {
        let prompt = format!(
            "Question: {}\n\nA) {}\nB) {}\n\nAnswer:",
            row.goal, row.sol1, row.sol2
        );
        examples.push(EvalExample {
            prompt,
            choices: vec![" A".into(), " B".into()],
            correct_index,
        });
    }
    Ok(examples)
}
