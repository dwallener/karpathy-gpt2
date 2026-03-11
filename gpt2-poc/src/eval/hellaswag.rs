use std::path::Path;

use anyhow::Result;
use serde::Deserialize;

use crate::eval::{EvalExample, download_to_cache, read_jsonl};

const DATASET_DIR: &str = "hellaswag";
const DATASET_FILE: &str = "validation.jsonl";
const URLS: &[&str] = &[
    "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
];

#[derive(Debug, Deserialize)]
struct HellaSwagRow {
    ctx: String,
    endings: Vec<String>,
    label: LabelValue,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum LabelValue {
    String(String),
    Integer(usize),
}

impl LabelValue {
    fn as_index(&self) -> Result<usize> {
        match self {
            Self::String(value) => Ok(value.parse::<usize>()?),
            Self::Integer(value) => Ok(*value),
        }
    }
}

pub fn load_examples(root: &Path) -> Result<Vec<EvalExample>> {
    let path = root.join(DATASET_DIR).join(DATASET_FILE);
    download_to_cache(&path, URLS)?;
    let rows = read_jsonl::<HellaSwagRow>(&path)?;
    let mut examples = Vec::with_capacity(rows.len());
    for row in rows {
        let correct_index = row.label.as_index()?;
        let prompt = format!(
            "Context: {}\n\nA) {}\nB) {}\nC) {}\nD) {}\n\nAnswer:",
            row.ctx, row.endings[0], row.endings[1], row.endings[2], row.endings[3]
        );
        examples.push(EvalExample {
            prompt,
            choices: vec![" A".into(), " B".into(), " C".into(), " D".into()],
            correct_index,
        });
    }
    Ok(examples)
}
