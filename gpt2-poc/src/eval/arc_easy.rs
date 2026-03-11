use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::eval::{EvalExample, download_to_cache, read_jsonl};

const DATASET_DIR: &str = "arc_easy";
const DATASET_FILE: &str = "ARC-Easy-Dev.jsonl";
const URLS: &[&str] = &[
    "https://raw.githubusercontent.com/allenai/ARC-Solvers/master/Arc-Solvers/Data/ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy-Dev.jsonl",
];

#[derive(Debug, Deserialize)]
struct ArcRow {
    question: ArcQuestion,
    #[serde(rename = "answerKey")]
    answer_key: String,
}

#[derive(Debug, Deserialize)]
struct ArcQuestion {
    stem: String,
    choices: Vec<ArcChoice>,
}

#[derive(Debug, Deserialize)]
struct ArcChoice {
    label: String,
    text: String,
}

pub fn load_examples(root: &Path) -> Result<Vec<EvalExample>> {
    let path = root.join(DATASET_DIR).join(DATASET_FILE);
    download_to_cache(&path, URLS)?;
    let rows = read_jsonl::<ArcRow>(&path)?;

    let mut examples = Vec::new();
    for row in rows {
        let mut ordered = vec![String::new(), String::new(), String::new(), String::new()];
        for choice in row.question.choices {
            let slot = match choice.label.as_str() {
                "A" | "1" => 0,
                "B" | "2" => 1,
                "C" | "3" => 2,
                "D" | "4" => 3,
                _ => continue,
            };
            ordered[slot] = choice.text;
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
                    "unsupported ARC-Easy answer key '{}' in {}",
                    other,
                    path.display()
                ));
            }
        };

        let prompt = format!(
            "Question: {}\n\nA) {}\nB) {}\nC) {}\nD) {}\n\nAnswer:",
            row.question.stem, ordered[0], ordered[1], ordered[2], ordered[3]
        );
        examples.push(EvalExample {
            prompt,
            choices: vec![" A".into(), " B".into(), " C".into(), " D".into()],
            correct_index,
        });
    }

    examples
        .first()
        .context("ARC-Easy loader produced zero valid examples")?;
    Ok(examples)
}
