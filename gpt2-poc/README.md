# Operator/State LM POC

Rust-only next-token training POC for an operator/state/mixer/readout language model built on `candle-core` and `candle-nn`. This repo does not use PyTorch, HuggingFace model wrappers, pretrained checkpoints, or any transformer attention blocks.

## File Layout

- `src/main.rs`: CLI with `train`, `eval`, and `inspect-batch`
- `src/config.rs`: model, train, and checkpoint config structs
- `src/stream_dataset.rs`: Parquet shard streaming, tokenization, and batch generation
- `src/model.rs`: operator/state recurrent LM with top-k routed operators
- `src/train.rs`: AdamW training loop, eval, checkpointing, CSV logging
- `src/utils.rs`: device resolution and JSON helpers

## Streaming Input

Training reads Parquet shards directly from a shard directory:

- train shards: all but the last 2 shards
- val shards: the last 2 shards
- expected column: `text`

Data flow:

```text
parquet -> text -> GPT-2 tokenizer -> rolling token buffer -> x/y next-token batches
```

## Build

CPU build:

```bash
cargo build --release
```

CUDA build:

```bash
cargo build --release --features cuda
```

## Smoke Test

Inspect a sampled batch first:

```bash
cargo run --release -- inspect-batch \
  --shard-dir data/fineweb \
  --seq-len 128 \
  --batch-size 4 \
  --max-docs 100
```

Then run a short CPU training smoke test:

```bash
cargo run --release -- train \
  --shard-dir data/fineweb \
  --device cpu \
  --seq-len 64 \
  --batch-size 2 \
  --tokenizer-workers 2 \
  --steps 20 \
  --lr 3e-4 \
  --weight-decay 0.01 \
  --eval-every 10 \
  --save-every 20 \
  --max-docs 100 \
  --out-dir runs/smoke_cpu
```

Example CUDA command:

```bash
cargo run --release --features cuda -- train \
  --shard-dir data/fineweb \
  --device cuda \
  --seq-len 256 \
  --batch-size 8 \
  --tokenizer-workers 8 \
  --steps 1000 \
  --lr 3e-4 \
  --weight-decay 0.01 \
  --eval-every 100 \
  --save-every 500 \
  --out-dir runs/train_cuda
```

Example CPU command:

```bash
cargo run --release -- train \
  --shard-dir data/fineweb \
  --device cpu \
  --seq-len 256 \
  --batch-size 8 \
  --tokenizer-workers 8 \
  --steps 1000 \
  --lr 3e-4 \
  --weight-decay 0.01 \
  --eval-every 100 \
  --save-every 500 \
  --out-dir runs/train_cpu
```

## Checkpoints And Logs

- checkpoints: `OUT_DIR/checkpoints/step-XXXXXXXX/`
- weights: `model.safetensors`
- metadata: `meta.json`
- training CSV: `OUT_DIR/logs/train.csv`
- CSV fields: `step,train_loss,val_loss,tokens_per_sec,docs_per_sec,tokenizer_queue_depth,parquet_queue_depth,current_shard,elapsed_sec`

## Notes / TODOs

- Top-k routing is currently implemented by copying router scores to host, selecting top-`k`, and rebuilding dense gates. This keeps the implementation explicit and simple but is not optimized.
- Checkpoints currently store model weights and run metadata. Optimizer state resume is not implemented yet.
- Validation uses the last 2 shards in deterministic shard order.
- The GPT-2 tokenizer is loaded via `Tokenizer::from_pretrained("gpt2", None)`, so first use needs access to the tokenizer assets.
- Training uses a bounded prefetch pipeline: parquet producer -> document queue -> tokenizer worker pool -> token queue -> batch builder.

Example log line:

```text
step=100 train_loss=4.812341 val_loss=4.900112 tok/s=1185320.221004 docs/s=624.334019 tokenizer_q=847 parquet_q=173 current_shard=data/fineweb/shard_00003.parquet elapsed_sec=21.606122
```
