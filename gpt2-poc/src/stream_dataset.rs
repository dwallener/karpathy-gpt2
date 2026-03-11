use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::thread;

use anyhow::{Context, Result, anyhow, bail};
use candle_core::{Device, Tensor};
use crossbeam_channel::{Receiver, Sender, bounded};
use futures::executor::block_on;
use polars::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use tokenizers::Tokenizer;

use crate::token_batcher::{TokenBatchMetrics, TokenBatcher};

const TEXT_COLUMN: &str = "text";
const VAL_SHARDS: usize = 2;
const DOC_QUEUE_CAPACITY: usize = 256;
const TOKEN_QUEUE_CAPACITY: usize = 1024;
const DEFAULT_ROW_GROUP_DOCS: usize = 1024;

#[derive(Debug)]
pub struct StreamBatch {
    pub xs: Tensor,
    pub ys: Tensor,
    pub docs_consumed: usize,
    pub current_shard: String,
    pub tokenizer_queue_depth: usize,
    pub parquet_queue_depth: usize,
}

#[derive(Debug, Clone)]
pub struct TokenChunk {
    pub tokens: Vec<u32>,
    pub docs: usize,
    pub current_shard: String,
    pub tokenizer_queue_depth: usize,
    pub parquet_queue_depth: usize,
}

#[derive(Debug, Clone)]
struct DocumentWork {
    texts: Vec<String>,
    current_shard: String,
    parquet_queue_depth: usize,
}

pub struct DatasetSplit;

impl DatasetSplit {
    pub fn train_val_split(shards: Vec<PathBuf>) -> Result<(Vec<PathBuf>, Vec<PathBuf>)> {
        if shards.len() < 3 {
            bail!(
                "expected at least 3 parquet shards so train/val can be split, found {}",
                shards.len()
            );
        }
        let split_at = shards.len() - VAL_SHARDS;
        Ok((shards[..split_at].to_vec(), shards[split_at..].to_vec()))
    }
}

pub struct ParquetReader {
    shard_paths: Vec<PathBuf>,
    reshuffle: bool,
    max_docs: Option<usize>,
}

pub struct TokenizerWorker {
    worker_id: usize,
    tokenizer: Tokenizer,
}

pub struct StreamDataset {
    shard_paths: Vec<PathBuf>,
    batcher: TokenBatcher,
    mode: StreamMode,
}

enum StreamMode {
    Prefetch(PrefetchState),
    Sequential(SequentialState),
}

struct PrefetchState {
    token_rx: Receiver<TokenChunk>,
}

struct SequentialState {
    shard_paths: Vec<PathBuf>,
    tokenizer: Tokenizer,
    reshuffle: bool,
    max_docs: Option<usize>,
    docs_seen: usize,
    rng: StdRng,
    shard_order: Vec<usize>,
    shard_cursor: usize,
    current_shard: Option<String>,
    current_docs: Vec<String>,
    doc_cursor: usize,
}

impl StreamDataset {
    pub fn new(
        shard_paths: Vec<PathBuf>,
        seq_len: usize,
        batch_size: usize,
        reshuffle: bool,
        max_docs: Option<usize>,
        tokenizer_workers: usize,
        parallel_tokenization: bool,
    ) -> Result<Self> {
        if shard_paths.is_empty() {
            bail!("stream dataset requires at least one parquet shard");
        }
        let batcher = TokenBatcher::new(seq_len, batch_size);
        let mode = if parallel_tokenization {
            StreamMode::Prefetch(PrefetchState::new(
                shard_paths.clone(),
                reshuffle,
                max_docs,
                tokenizer_workers,
            )?)
        } else {
            StreamMode::Sequential(SequentialState::new(
                shard_paths.clone(),
                reshuffle,
                max_docs,
            )?)
        };
        Ok(Self {
            shard_paths,
            batcher,
            mode,
        })
    }

    pub fn tokenizer_vocab_size() -> Result<usize> {
        Ok(load_tokenizer()?.get_vocab_size(false))
    }

    pub fn next_batch(&mut self, device: &Device) -> Result<Option<StreamBatch>> {
        let mut aggregate = TokenBatchMetrics::default();

        loop {
            if let Some((xs, ys)) = self.batcher.try_build_batch(device)? {
                return Ok(Some(StreamBatch {
                    xs,
                    ys,
                    docs_consumed: aggregate.docs_consumed,
                    current_shard: if aggregate.current_shard.is_empty() {
                        self.current_shard_name()
                            .unwrap_or_else(|| "none".to_string())
                    } else {
                        aggregate.current_shard
                    },
                    tokenizer_queue_depth: aggregate.tokenizer_queue_depth,
                    parquet_queue_depth: aggregate.parquet_queue_depth,
                }));
            }

            let next = match &mut self.mode {
                StreamMode::Prefetch(state) => state.recv_token_chunk()?,
                StreamMode::Sequential(state) => state.next_token_chunk()?,
            };

            let Some(chunk) = next else {
                return if aggregate.docs_consumed == 0 {
                    Ok(None)
                } else {
                    self.batcher
                        .try_build_batch(device)?
                        .map(|(xs, ys)| StreamBatch {
                            xs,
                            ys,
                            docs_consumed: aggregate.docs_consumed,
                            current_shard: if aggregate.current_shard.is_empty() {
                                self.current_shard_name()
                                    .unwrap_or_else(|| "none".to_string())
                            } else {
                                aggregate.current_shard
                            },
                            tokenizer_queue_depth: aggregate.tokenizer_queue_depth,
                            parquet_queue_depth: aggregate.parquet_queue_depth,
                        })
                        .ok_or_else(|| anyhow!("dataset ended before enough tokens for a batch"))
                        .map(Some)
                };
            };

            let metrics = self.batcher.push(chunk);
            aggregate.docs_consumed += metrics.docs_consumed;
            aggregate.current_shard = metrics.current_shard;
            aggregate.tokenizer_queue_depth = metrics.tokenizer_queue_depth;
            aggregate.parquet_queue_depth = metrics.parquet_queue_depth;
        }
    }

    pub fn current_shard_name(&self) -> Option<String> {
        match &self.mode {
            StreamMode::Prefetch(_) => None,
            StreamMode::Sequential(state) => state.current_shard.clone(),
        }
    }

    pub fn shard_count(&self) -> usize {
        self.shard_paths.len()
    }
}

impl PrefetchState {
    fn new(
        shard_paths: Vec<PathBuf>,
        reshuffle: bool,
        max_docs: Option<usize>,
        tokenizer_workers: usize,
    ) -> Result<Self> {
        let worker_count = tokenizer_workers.max(1);
        let (doc_tx, doc_rx) = bounded::<DocumentWork>(DOC_QUEUE_CAPACITY);
        let (token_tx, token_rx) = bounded::<TokenChunk>(TOKEN_QUEUE_CAPACITY);

        let producer = ParquetReader {
            shard_paths,
            reshuffle,
            max_docs,
        };
        let producer_doc_tx = doc_tx.clone();
        thread::spawn(move || {
            if let Err(err) = producer.run(producer_doc_tx) {
                eprintln!("parquet producer terminated: {err}");
            }
        });

        for worker_id in 0..worker_count {
            let worker = TokenizerWorker {
                worker_id,
                tokenizer: load_tokenizer()?,
            };
            let worker_doc_rx = doc_rx.clone();
            let worker_token_tx = token_tx.clone();
            let worker_token_rx = token_rx.clone();
            thread::spawn(move || {
                if let Err(err) = worker.run(worker_doc_rx, worker_token_tx, worker_token_rx) {
                    eprintln!("tokenizer worker {worker_id} terminated: {err}");
                }
            });
        }

        drop(doc_tx);
        drop(token_tx);

        Ok(Self { token_rx })
    }

    fn recv_token_chunk(&mut self) -> Result<Option<TokenChunk>> {
        match self.token_rx.recv() {
            Ok(chunk) => Ok(Some(chunk)),
            Err(_) => Ok(None),
        }
    }
}

impl SequentialState {
    fn new(shard_paths: Vec<PathBuf>, reshuffle: bool, max_docs: Option<usize>) -> Result<Self> {
        let tokenizer = load_tokenizer()?;
        let mut shard_order: Vec<usize> = (0..shard_paths.len()).collect();
        let mut rng = StdRng::seed_from_u64(42);
        if reshuffle {
            shard_order.shuffle(&mut rng);
        }
        Ok(Self {
            shard_paths,
            tokenizer,
            reshuffle,
            max_docs,
            docs_seen: 0,
            rng,
            shard_order,
            shard_cursor: 0,
            current_shard: None,
            current_docs: Vec::new(),
            doc_cursor: 0,
        })
    }

    fn next_token_chunk(&mut self) -> Result<Option<TokenChunk>> {
        let Some((text, shard)) = self.next_document()? else {
            return Ok(None);
        };
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|err| anyhow!("failed to tokenize document: {err}"))?;
        Ok(Some(TokenChunk {
            tokens: encoding.get_ids().to_vec(),
            docs: 1,
            current_shard: shard,
            tokenizer_queue_depth: 0,
            parquet_queue_depth: 0,
        }))
    }

    fn next_document(&mut self) -> Result<Option<(String, String)>> {
        loop {
            if self.doc_cursor < self.current_docs.len() {
                let idx = self.doc_cursor;
                self.doc_cursor += 1;
                self.docs_seen += 1;
                let shard = self
                    .current_shard
                    .clone()
                    .unwrap_or_else(|| "none".to_string());
                return Ok(self.current_docs.get(idx).cloned().map(|doc| (doc, shard)));
            }
            if self.max_docs.is_some_and(|limit| self.docs_seen >= limit) {
                return Ok(None);
            }
            self.load_next_chunk()?;
            if self.current_docs.is_empty() {
                continue;
            }
        }
    }

    fn load_next_chunk(&mut self) -> Result<()> {
        if self.shard_cursor >= self.shard_order.len() {
            self.shard_cursor = 0;
            if self.reshuffle {
                self.shard_order.shuffle(&mut self.rng);
            }
        }
        let shard_idx = self.shard_order[self.shard_cursor];
        self.shard_cursor += 1;
        let shard_path = &self.shard_paths[shard_idx];
        self.current_shard = Some(shard_path.display().to_string());
        self.current_docs = read_text_groups_batched(shard_path, self.max_docs, self.docs_seen)?
            .into_iter()
            .flatten()
            .collect();
        self.doc_cursor = 0;
        Ok(())
    }
}

impl ParquetReader {
    fn run(self, doc_tx: Sender<DocumentWork>) -> Result<()> {
        let mut shard_order: Vec<usize> = (0..self.shard_paths.len()).collect();
        let mut rng = StdRng::seed_from_u64(42);
        let mut docs_seen = 0usize;
        if self.reshuffle {
            shard_order.shuffle(&mut rng);
        }

        loop {
            for &shard_idx in &shard_order {
                let path = &self.shard_paths[shard_idx];
                let current_shard = path.display().to_string();
                for texts in read_text_groups_batched(path, self.max_docs, docs_seen)? {
                    if texts.is_empty() {
                        continue;
                    }
                    docs_seen += texts.len();
                    let work = DocumentWork {
                        texts,
                        current_shard: current_shard.clone(),
                        parquet_queue_depth: doc_tx.len(),
                    };
                    if doc_tx.send(work).is_err() {
                        return Ok(());
                    }
                    if self.max_docs.is_some_and(|limit| docs_seen >= limit) {
                        return Ok(());
                    }
                }
            }
            if !self.reshuffle {
                return Ok(());
            }
            shard_order.shuffle(&mut rng);
        }
    }
}

impl TokenizerWorker {
    fn run(
        self,
        doc_rx: Receiver<DocumentWork>,
        token_tx: Sender<TokenChunk>,
        token_rx: Receiver<TokenChunk>,
    ) -> Result<()> {
        let _worker_id = self.worker_id;
        while let Ok(work) = doc_rx.recv() {
            for text in work.texts {
                let encoding = self
                    .tokenizer
                    .encode(text, false)
                    .map_err(|err| anyhow!("failed to tokenize document: {err}"))?;
                let chunk = TokenChunk {
                    tokens: encoding.get_ids().to_vec(),
                    docs: 1,
                    current_shard: work.current_shard.clone(),
                    tokenizer_queue_depth: token_rx.len(),
                    parquet_queue_depth: doc_rx.len().max(work.parquet_queue_depth),
                };
                if token_tx.send(chunk).is_err() {
                    return Ok(());
                }
            }
        }
        Ok(())
    }
}

pub fn list_shards(shard_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut shards = fs::read_dir(shard_dir)
        .with_context(|| format!("failed to read shard dir {}", shard_dir.display()))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "parquet"))
        .collect::<Vec<_>>();
    shards.sort();
    if shards.is_empty() {
        bail!("no parquet shards found in {}", shard_dir.display());
    }
    Ok(shards)
}

fn load_tokenizer() -> Result<Tokenizer> {
    Tokenizer::from_pretrained("gpt2", None)
        .map_err(|err| anyhow!("failed to load GPT-2 tokenizer: {err}"))
}

fn read_text_groups_batched(
    path: &Path,
    max_docs: Option<usize>,
    docs_seen: usize,
) -> Result<Vec<Vec<String>>> {
    let remaining = max_docs.map(|limit| limit.saturating_sub(docs_seen));
    if remaining == Some(0) {
        return Ok(Vec::new());
    }

    let file = File::open(path)
        .with_context(|| format!("failed to open parquet shard {}", path.display()))?;
    let reader =
        polars::prelude::ParquetReader::new(file).with_columns(Some(vec![TEXT_COLUMN.to_string()]));
    let reader = if let Some(limit) = remaining {
        reader.with_n_rows(Some(limit))
    } else {
        reader
    };
    let mut reader = reader.batched(DEFAULT_ROW_GROUP_DOCS)?;
    let mut groups = Vec::new();
    while let Some(dfs) = block_on(reader.next_batches(1))? {
        for df in dfs {
            let texts = extract_texts(df, path)?;
            if !texts.is_empty() {
                groups.push(texts);
            }
        }
    }
    Ok(groups)
}

fn extract_texts(df: DataFrame, path: &Path) -> Result<Vec<String>> {
    let text_col = df
        .column(TEXT_COLUMN)
        .with_context(|| format!("missing '{TEXT_COLUMN}' column in {}", path.display()))?;
    let utf8 = text_col
        .str()
        .with_context(|| format!("column '{TEXT_COLUMN}' is not utf8 in {}", path.display()))?;
    Ok(utf8.into_iter().flatten().map(ToOwned::to_owned).collect())
}
