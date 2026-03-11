use std::fs::File;
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};

use crate::eval::{EvalExample, read_jsonl};

const DATASET_DIR: &str = "arc_easy";
const CACHE_FILE: &str = "validation.jsonl";
const ROWS_URL: &str =
    "https://datasets-server.huggingface.co/rows?dataset=allenai/ai2_arc&config=ARC-Easy&split=validation";
const PAGE_SIZE: usize = 100;

#[derive(Debug, Deserialize)]
struct RowsResponse {
    rows: Vec<RowEnvelope>,
}

#[derive(Debug, Deserialize)]
struct RowEnvelope {
    row: ArcRow,
}

#[derive(Debug, Deserialize, Serialize)]
struct ArcRow {
    question: String,
    choices: ArcChoices,
    #[serde(rename = "answerKey")]
    answer_key: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct ArcChoices {
    label: Vec<String>,
    text: Vec<String>,
}

pub fn load_examples(root: &Path) -> Result<Vec<EvalExample>> {
    let path = root.join(DATASET_DIR).join(CACHE_FILE);
    if !path.exists() {
        download_validation_jsonl(&path)?;
    }

    let rows = read_jsonl::<ArcRow>(&path)?;
    let mut examples = Vec::with_capacity(rows.len());
    for row in rows {
        if row.choices.label.len() < 4 || row.choices.text.len() < 4 {
            continue;
        }

        let mut ordered = vec![String::new(), String::new(), String::new(), String::new()];
        for (label, text) in row.choices.label.into_iter().zip(row.choices.text) {
            let slot = match label.as_str() {
                "A" | "1" => 0,
                "B" | "2" => 1,
                "C" | "3" => 2,
                "D" | "4" => 3,
                _ => continue,
            };
            ordered[slot] = text;
        }
        if ordered.iter().any(|text| text.is_empty()) {
            continue;
        }

        let correct_index = match row.answer_key.as_str() {
            "A" | "1" => 0,
            "B" | "2" => 1,
            "C" | "3" => 2,
            "D" | "4" => 3,
            other => {
                return Err(anyhow::anyhow!(
                    "unsupported ARC-Easy answer key '{}'",
                    other
                ));
            }
        };

        let prompt = format!(
            "Question: {}\n\nA) {}\nB) {}\nC) {}\nD) {}\n\nAnswer:",
            row.question, ordered[0], ordered[1], ordered[2], ordered[3]
        );
        examples.push(EvalExample {
            prompt,
            choices: vec![" A".into(), " B".into(), " C".into(), " D".into()],
            correct_index,
        });
    }

    if examples.is_empty() {
        return Err(anyhow::anyhow!(
            "ARC-Easy loader produced zero valid examples from {}",
            path.display()
        ));
    }
    Ok(examples)
}

fn download_validation_jsonl(path: &Path) -> Result<()> {
    let parent = path
        .parent()
        .context("ARC-Easy cache path must have a parent directory")?;
    std::fs::create_dir_all(parent)?;

    let client = Client::builder().build()?;
    let mut file = File::create(path)?;
    let mut offset = 0usize;
    loop {
        let response = client
            .get(format!("{ROWS_URL}&offset={offset}&length={PAGE_SIZE}"))
            .send()?
            .error_for_status()?;
        let body: RowsResponse = response.json()?;
        if body.rows.is_empty() {
            break;
        }

        for row in body.rows {
            serde_json::to_writer(&mut file, &row.row)?;
            file.write_all(b"\n")?;
        }

        offset += PAGE_SIZE;
    }

    Ok(())
}
