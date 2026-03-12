# Operator/State LM POC

Rust-only next-token training POC for an operator/state/mixer/readout language model built on `candle-core` and `candle-nn`. This repo does not use PyTorch, HuggingFace model wrappers, pretrained checkpoints, or any transformer attention blocks.

## File Layout

- `src/main.rs`: CLI with `train`, `eval`, and `inspect-batch`
- `src/bin/infer.rs`: separate inference binary for saved checkpoints
- `src/bin/eval.rs`: mini-CORE benchmark binary for saved checkpoints
- `src/config.rs`: model, train, and checkpoint config structs
- `src/infer.rs`: checkpoint loading, greedy generation, and continuation scoring helpers
- `src/eval/`: mini-CORE dataset loaders and evaluation harness
- `src/stream_dataset.rs`: Parquet shard streaming, tokenization, and batch generation
- `src/train_stats.rs`: in-memory training curve points
- `src/diag/ascii_plot.rs`: ASCII loss-vs-tokens diagnostics
- `src/diag/scaling_predictor.rs`: early scaling-law fit and extrapolation
- `src/diag/token_entropy.rs`: token entropy and rank-distribution diagnostics
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

## Precision

- CPU default: `f32`
- CUDA default: `bf16`
- override with `--model-dtype f32|bf16|f16`

Example:

```bash
cargo run --release --features cuda -- train \
  --shard-dir ../data/fineweb \
  --device cuda \
  --model-dtype bf16 \
  --out-dir runs/train_cuda
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

Example CUDA command that fits, barely, on an 8 GB RTX 4060:

```bash
cargo run --release --features cuda -- train \
  --shard-dir ../data/fineweb \
  --device cuda \
  --model-dtype bf16 \
  --seq-len 128 \
  --batch-size 4 \
  --tokenizer-workers 2 \
  --steps 2 \
  --eval-every 1 \
  --save-every 2 \
  --max-docs 50 \
  --d-model 384 \
  --d-state 1536 \
  --num-operators 8 \
  --operator-hidden 1536 \
  --top-k 4 \
  --out-dir runs/smoke_cuda_small
```

Run periodic mini-CORE from saved checkpoints every `N` steps:

```bash
cargo run --release --features cuda -- train \
  --shard-dir ../data/fineweb \
  --device cuda \
  --model-dtype bf16 \
  --seq-len 128 \
  --batch-size 4 \
  --tokenizer-workers 2 \
  --steps 500 \
  --save-every 100 \
  --mini-core-every 100 \
  --mini-core-limit 200 \
  --diag-every 100 \
  --scaling-predictor true \
  --d-model 384 \
  --d-state 1536 \
  --num-operators 8 \
  --operator-hidden 1536 \
  --top-k 4 \
  --out-dir runs/train_4060
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
- mini-CORE CSV: `OUT_DIR/logs/mini_core.csv`
- training curve JSONL: `OUT_DIR/training_curve.jsonl`
- CSV fields: `step,train_loss,val_loss,train_tokens_per_sec,train_docs_per_sec,batch_wait_ms,step_time_ms,eval_time_ms,tokenizer_queue_depth,parquet_queue_depth,current_shard,elapsed_sec`

## Inference

Build the separate inference binary:

```bash
cargo build --release --bin infer
```

CUDA inference build:

```bash
cargo build --release --features cuda --bin infer
```

Run greedy generation from a saved checkpoint:

```bash
cargo run --release --bin infer -- \
  --checkpoint runs/smoke_cuda_small/checkpoints/step-00000050 \
  --prompt "The meaning of life is" \
  --max-new-tokens 32
```

CUDA greedy inference:

```bash
cargo run --release --features cuda --bin infer -- \
  --checkpoint runs/smoke_cuda_small/checkpoints/step-00000050 \
  --device cuda \
  --model-dtype bf16 \
  --prompt "The meaning of life is" \
  --max-new-tokens 32
```

CUDA sampled inference:

```bash
cargo run --release --features cuda --bin infer -- \
  --checkpoint runs/train_4060_full/checkpoints/step-00003200 \
  --device cuda \
  --model-dtype bf16 \
  --prompt "The meaning of life is" \
  --max-new-tokens 64 \
  --temperature 0.8 \
  --top-k 40 \
  --seed 123
```

Inference defaults to greedy decoding. Sampling activates when `--temperature > 0` and `--top-k > 1`.

Bulk inference with repeated prompts:

```bash
cargo run --release --features cuda --bin infer -- \
  --checkpoint runs/train_4060_full/checkpoints/step-00003200 \
  --device cuda \
  --model-dtype bf16 \
  --max-new-tokens 24 \
  --prompt "The capital of France is" \
  --prompt "The chemical symbol for gold is" \
  --prompt "The opposite of cold is" \
  --prompt "2 + 2 =" \
  --prompt "The largest planet in the solar system is"
```

Bulk inference from a newline-delimited prompts file:

```bash
cargo run --release --features cuda --bin infer -- \
  --checkpoint runs/train_4060_full/checkpoints/step-00003200 \
  --device cuda \
  --model-dtype bf16 \
  --max-new-tokens 24 \
  --prompts-file prompts.txt
```

## Mini-CORE Evaluation

Mini-CORE is a lightweight capability check over:

- HellaSwag
- PIQA
- ARC-Easy

This harness scores answer-letter continuations conditionally on the formatted prompt plus full answer choices, then reports accuracy and centered accuracy relative to random baseline:

- HellaSwag baseline: `0.25`
- PIQA baseline: `0.50`
- ARC-Easy baseline: `0.25`

Standalone checkpoint evaluation:

```bash
cargo run --release --features cuda --bin eval -- \
  --checkpoint runs/train/checkpoints/step-005000 \
  --device cuda \
  --model-dtype bf16
```

Optional subset for fast smoke tests:

```bash
cargo run --release --bin eval -- \
  --checkpoint runs/train/checkpoints/step-000100 \
  --limit 100
```

The evaluator caches downloaded files under:

```text
data/eval/hellaswag/
data/eval/piqa/
data/eval/arc_easy/
```

Example output:

```text
HellaSwag: 0.420000 centered=0.170000 examples=1000 ex/s=52.300000
PIQA: 0.640000 centered=0.140000 examples=1000 ex/s=60.120000
ARC-Easy: 0.550000 centered=0.300000 examples=570 ex/s=48.440000
miniCORE: 0.203333
```

Whenever mini-CORE runs during training, the trainer also prints token-distribution diagnostics from prompt-only next-token predictions over a small evaluation sample:

```text
Token Distribution Diagnostics
------------------------------
entropy=6.420000
top1_prob=0.180000
top10_mass=0.640000
samples=200

Token Prob vs Rank

 1 | ****************************************
 2 | ********************
 3 | *************
 4 | ********
 5 | ******
 6 | ****
```

## Training Diagnostics

The trainer now records per-step curve points, BPB estimates, router-utilization snapshots, and can print a compact ASCII loss plot.

Stored per-step fields:

```json
{"step":100,"tokens_seen":102400,"elapsed_sec":168.99,"train_loss":8.25,"val_loss":null,"train_bpb":11.90,"val_bpb":null,"router":{"routing_entropy":1.22,"max_operator_share":0.31,"num_active_operators":8,"operator_usage":[0.31,0.18,0.14,0.10,0.09,0.08,0.06,0.04],"gate_mass":[0.29,0.19,0.15,0.11,0.09,0.08,0.06,0.03]},"mini_core":0.0}
```

Diagnostics trigger:

- by default whenever mini-CORE runs
- or explicitly with `--diag-every N`
- scaling-law prediction is enabled automatically when mini-CORE is enabled, or override with `--scaling-predictor true|false`

Example output:

```text
==============================
Training Diagnostics
==============================
tokens_seen=102400
steps=100
train_loss=8.252387
train_bpb=11.905804
val_loss=na
val_bpb=na
mini_core=0.000000
learning_slope=-0.680000
router_entropy=1.220000
router_max_share=0.310000
router_active_ops=8
router_usage=0:0.310 1:0.180 2:0.140 3:0.100 4:0.090 5:0.080 6:0.060 7:0.040
router_gate_mass=0:0.290 1:0.190 2:0.150 3:0.110 4:0.090 5:0.080 6:0.060 7:0.030

Loss / BPB vs Tokens (log scale)

11.9 | *                                                           
11.6 |  .                                                          
11.3 |   .                                                         
11.0 |    .                                                        
10.7 |      .                                                      
10.4 |        .                                                    
10.1 |          .                                                  
 9.8 |            .                                                
 9.5 |              .                                              
 9.2 |                .                                            
 8.9 |                   .                                         
 8.6 |                      .                                      
     +------------------------------------------------------------
      1e5   2e5   3e5 tokens
```

When enough validation points exist, diagnostics also print a simple scaling-law extrapolation based on the approximate fit:

```text
log10(loss) = a + b * log10(tokens)
alpha ~= -b
```

Example:

```text
Scaling Law Prediction
----------------------
alpha=0.310000
pred_loss@10M=6.800000
pred_loss@100M=5.300000
pred_loss@1B=4.400000
predicted_mini_core=0.100000
scaling_hint=healthy
```

Interpretation:

- `alpha > 0.5`: very strong scaling
- `0.3 <= alpha <= 0.5`: healthy
- `0.2 <= alpha < 0.3`: weak
- `0.1 <= alpha < 0.2`: architecture weak
- `alpha < 0.1`: training broken

## TODO

- Activation checkpointing to reduce sequence-length activation memory and push `seq-len` higher on 8 GB GPUs.
- Truncated BPTT / detach windows so longer contexts do not require backpropagating through the full recurrent unroll.
- Optional further model-size tuning once the current sparse top-k execution path is characterized.
- Optimizer-state checkpoint save/load for full training resume.
- More efficient inference with recurrent state caching instead of recomputing the full prompt window every generated token.
- Optional periodic mini-CORE on a separate device/process so long training runs do not pause for eval.

## Notes / TODOs

- Top-k routing still selects routes on the host, but operator execution is now truly sparse: only operators touched by the current top-k routing are evaluated for that step.
- The default architecture is too large for an 8 GB RTX 4060 with the current dense operator execution path. Use the smaller CUDA example above as the starting point on that class of GPU.
- Mixed precision is now supported via `--model-dtype`. CUDA defaults to `bf16`, CPU defaults to `f32`.
- Checkpoints currently store model weights and run metadata. Optimizer state resume is not implemented yet.
- Validation uses the last 2 shards in deterministic shard order.
- The GPT-2 tokenizer is loaded via `Tokenizer::from_pretrained("gpt2", None)`, so first use needs access to the tokenizer assets.
- Training uses a bounded prefetch pipeline: parquet producer -> document queue -> tokenizer worker pool -> token queue -> batch builder.
- Mini-CORE evaluation reuses the same checkpoint loader and GPT-2 tokenizer path as inference, and downloads benchmark caches on first use.

Example log line:

```text
step=100 train_loss=4.812341 val_loss=4.900112 train_tok/s=1185320.221004 train_docs/s=624.334019 batch_wait_ms=0.021123 step_time_ms=128.442001 eval_time_ms=na tokenizer_q=847 parquet_q=173 current_shard=data/fineweb/shard_00003.parquet elapsed_sec=21.606122
```
