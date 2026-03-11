# Operator/State LM POC

Rust-only next-token training POC for an operator/state/mixer/readout language model built on `candle-core` and `candle-nn`. This repo does not use PyTorch, HuggingFace model wrappers, pretrained checkpoints, or any transformer attention blocks.

## File Layout

- `src/main.rs`: CLI with `train`, `eval`, and `inspect-batch`
- `src/config.rs`: model, train, and checkpoint config structs
- `src/dataset.rs`: memory-mapped flat `u32` token loader and batch sampler
- `src/model.rs`: operator/state recurrent LM with top-k routed operators
- `src/train.rs`: AdamW training loop, eval, checkpointing, CSV logging
- `src/utils.rs`: device resolution and JSON helpers

## Token File Format

Input token files are flat binary streams of little-endian `u32` token ids with no header:

1. `token_0`
2. `token_1`
3. `token_2`
4. ...

The dataset loader samples contiguous windows:

- `x = tokens[t..t+seq_len]`
- `y = tokens[t+1..t+seq_len+1]`

`vocab_size` defaults to `max_token_id + 1` across the provided train and validation files, unless overridden with `--vocab-size`.

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
  --train-tokens data/train_tokens.bin \
  --val-tokens data/val_tokens.bin \
  --seq-len 128 \
  --batch-size 4
```

Then run a short CPU training smoke test:

```bash
cargo run --release -- train \
  --train-tokens data/train_tokens.bin \
  --val-tokens data/val_tokens.bin \
  --device cpu \
  --seq-len 64 \
  --batch-size 2 \
  --steps 20 \
  --lr 3e-4 \
  --weight-decay 0.01 \
  --eval-every 10 \
  --save-every 20 \
  --out-dir runs/smoke_cpu
```

Example CUDA command:

```bash
cargo run --release --features cuda -- train \
  --train-tokens data/train_tokens.bin \
  --val-tokens data/val_tokens.bin \
  --device cuda \
  --seq-len 128 \
  --batch-size 8 \
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
  --train-tokens data/train_tokens.bin \
  --val-tokens data/val_tokens.bin \
  --device cpu \
  --seq-len 128 \
  --batch-size 8 \
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

## Notes / TODOs

- Top-k routing is currently implemented by copying router scores to host, selecting top-`k`, and rebuilding dense gates. This keeps the implementation explicit and simple but is not optimized.
- Checkpoints currently store model weights and run metadata. Optimizer state resume is not implemented yet.
- Validation uses random sampled batches rather than a deterministic full sweep.
