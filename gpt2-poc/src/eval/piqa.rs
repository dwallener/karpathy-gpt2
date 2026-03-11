use std::path::Path;

use anyhow::{Context, Result};
use polars::prelude::{DataFrame, ParquetReader, SerReader};

use crate::eval::{EvalExample, download_to_cache};

const DATASET_DIR: &str = "piqa";
const DATA_FILE: &str = "validation-00000-of-00001.parquet";
const DATA_URLS: &[&str] = &[
    "https://huggingface.co/datasets/lighteval/piqa/resolve/main/plain_text/validation-00000-of-00001.parquet",
];

pub fn load_examples(root: &Path) -> Result<Vec<EvalExample>> {
    let dir = root.join(DATASET_DIR);
    let data_path = dir.join(DATA_FILE);
    download_to_cache(&data_path, DATA_URLS)?;
    let rows = read_piqa_parquet(&data_path)?;

    let mut examples = Vec::with_capacity(rows.len());
    for (goal, sol1, sol2, correct_index) in rows {
        let prompt = format!(
            "Question: {}\n\nA) {}\nB) {}\n\nAnswer:",
            goal, sol1, sol2
        );
        examples.push(EvalExample {
            prompt,
            choices: vec![" A".into(), " B".into()],
            correct_index,
        });
    }
    Ok(examples)
}

fn read_piqa_parquet(path: &Path) -> Result<Vec<(String, String, String, usize)>> {
    let file = std::fs::File::open(path)?;
    let df = ParquetReader::new(file)
        .finish()
        .with_context(|| format!("failed to read PIQA parquet {}", path.display()))?;
    extract_rows(df, path)
}

fn extract_rows(df: DataFrame, path: &Path) -> Result<Vec<(String, String, String, usize)>> {
    let goals = df
        .column("goal")
        .with_context(|| format!("missing goal column in {}", path.display()))?
        .str()
        .with_context(|| format!("goal column is not utf8 in {}", path.display()))?;
    let sol1 = df
        .column("sol1")
        .with_context(|| format!("missing sol1 column in {}", path.display()))?
        .str()
        .with_context(|| format!("sol1 column is not utf8 in {}", path.display()))?;
    let sol2 = df
        .column("sol2")
        .with_context(|| format!("missing sol2 column in {}", path.display()))?
        .str()
        .with_context(|| format!("sol2 column is not utf8 in {}", path.display()))?;
    let labels = df
        .column("label")
        .with_context(|| format!("missing label column in {}", path.display()))?
        .i64()
        .with_context(|| format!("label column is not i64 in {}", path.display()))?;

    let mut rows = Vec::with_capacity(df.height());
    for idx in 0..df.height() {
        let goal = goals
            .get(idx)
            .with_context(|| format!("null goal at row {} in {}", idx, path.display()))?;
        let sol1_value = sol1
            .get(idx)
            .with_context(|| format!("null sol1 at row {} in {}", idx, path.display()))?;
        let sol2_value = sol2
            .get(idx)
            .with_context(|| format!("null sol2 at row {} in {}", idx, path.display()))?;
        let label = labels
            .get(idx)
            .with_context(|| format!("null label at row {} in {}", idx, path.display()))?;
        rows.push((goal.to_string(), sol1_value.to_string(), sol2_value.to_string(), label as usize));
    }
    Ok(rows)
}
