use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use indicatif::{ProgressBar, ProgressStyle};

use crate::config::{CheckpointMeta, Config, TrainConfig};
use crate::diag::ascii_plot::build_diagnostics;
use crate::eval::run_mini_core;
use crate::model::{build_model, cross_entropy_loss};
use crate::stream_dataset::StreamDataset;
use crate::train_stats::{TrainPoint, TrainStats};
use crate::utils::{format_float, write_json_pretty};

pub fn train_main(
    model_config: Config,
    train_config: TrainConfig,
    model_dtype: DType,
    mut train_ds: StreamDataset,
    mut val_ds: StreamDataset,
    device: &Device,
) -> Result<()> {
    fs::create_dir_all(&train_config.out_dir)?;
    fs::create_dir_all(train_config.out_dir.join("checkpoints"))?;
    fs::create_dir_all(train_config.out_dir.join("logs"))?;

    let (mut varmap, model) = build_model(model_config.clone(), model_dtype, device)?;
    let mut start_step = 0usize;
    if let Some(checkpoint) = &train_config.checkpoint {
        load_checkpoint(checkpoint, &mut varmap)?;
        start_step = load_checkpoint_meta(checkpoint)?.step;
    }

    let params = ParamsAdamW {
        lr: train_config.lr,
        weight_decay: train_config.weight_decay,
        ..ParamsAdamW::default()
    };
    let mut optimizer = AdamW::new(varmap.all_vars(), params)?;

    let csv_path = train_config.out_dir.join("logs").join("train.csv");
    let mini_core_csv_path = train_config.out_dir.join("logs").join("mini_core.csv");
    let training_curve_path = train_config.out_dir.join("training_curve.jsonl");
    init_csv_log(&csv_path)?;
    init_mini_core_csv_log(&mini_core_csv_path)?;
    init_training_curve_log(&training_curve_path)?;

    let progress = ProgressBar::new(train_config.steps as u64);
    progress.set_position(start_step as u64);
    progress.set_style(ProgressStyle::with_template(
        "[{elapsed_precise}] {wide_bar} {pos}/{len} step={msg}",
    )?);

    let start_time = Instant::now();
    let mut running_tokens = 0usize;
    let mut running_docs = 0usize;
    let mut train_compute_time = 0f64;
    let mut stats = TrainStats::default();

    for step in (start_step + 1)..=train_config.steps {
        let batch_wait_start = Instant::now();
        let Some(batch) = train_ds.next_batch(device)? else {
            println!(
                "stream dataset ended early at step={} max_docs={:?}",
                step - 1,
                train_config.max_docs
            );
            break;
        };
        let batch_wait_ms = batch_wait_start.elapsed().as_secs_f64() * 1_000.0;

        let step_start = Instant::now();
        let (logits, _) = model.forward(&batch.xs)?;
        let loss = cross_entropy_loss(&logits, &batch.ys)?;

        let mut grads = loss.backward()?;
        clip_gradients(&varmap, &mut grads, train_config.grad_clip)?;
        optimizer.step(&grads)?;
        let step_time_ms = step_start.elapsed().as_secs_f64() * 1_000.0;
        train_compute_time += step_time_ms / 1_000.0;

        let train_loss = loss.to_dtype(DType::F32)?.to_scalar::<f32>()? as f64;
        let tokens_per_step = train_config.batch_size * train_config.seq_len;
        let tokens_seen = (step * tokens_per_step) as u64;
        running_tokens += tokens_per_step;
        running_docs += batch.docs_consumed;
        let elapsed = start_time.elapsed().as_secs_f64().max(1e-9);
        let train_elapsed = train_compute_time.max(1e-9);
        let train_tokens_per_sec = running_tokens as f64 / train_elapsed;
        let train_docs_per_sec = running_docs as f64 / train_elapsed;

        let mut val_loss = None;
        let mut eval_time_ms = None;
        if step % train_config.eval_every == 0 {
            let eval_start = Instant::now();
            let loss = evaluate_dataset(&model, &mut val_ds, train_config.val_batches, device)?;
            val_loss = Some(loss);
            eval_time_ms = Some(eval_start.elapsed().as_secs_f64() * 1_000.0);
        }

        if step % train_config.log_every == 0 || val_loss.is_some() || step == 1 {
            println!(
                "step={} tokens_seen={} train_loss={} val_loss={} train_tok/s={} train_docs/s={} batch_wait_ms={} step_time_ms={} eval_time_ms={} tokenizer_q={} parquet_q={} current_shard={} elapsed_sec={}",
                step,
                tokens_seen,
                format_float(train_loss),
                val_loss
                    .map(format_float)
                    .unwrap_or_else(|| "na".to_string()),
                format_float(train_tokens_per_sec),
                format_float(train_docs_per_sec),
                format_float(batch_wait_ms),
                format_float(step_time_ms),
                eval_time_ms
                    .map(format_float)
                    .unwrap_or_else(|| "na".to_string()),
                batch.tokenizer_queue_depth,
                batch.parquet_queue_depth,
                batch.current_shard,
                format_float(elapsed),
            );
            append_csv_log(
                &csv_path,
                step,
                train_loss,
                val_loss,
                train_tokens_per_sec,
                train_docs_per_sec,
                batch_wait_ms,
                step_time_ms,
                eval_time_ms,
                batch.tokenizer_queue_depth,
                batch.parquet_queue_depth,
                &batch.current_shard,
                elapsed,
            )?;
        }

        stats.push(TrainPoint {
            step: step as u64,
            tokens_seen,
            train_loss: train_loss as f32,
            val_loss: val_loss.map(|value| value as f32),
            mini_core: None,
        });

        let should_save = step % train_config.save_every == 0
            || step == train_config.steps
            || train_config
                .mini_core_every
                .is_some_and(|every| step % every == 0);
        let mut mini_core_score = None;
        if should_save {
            let checkpoint_dir = train_config
                .out_dir
                .join("checkpoints")
                .join(format!("step-{step:08}"));
            save_checkpoint(
                &checkpoint_dir,
                &varmap,
                &CheckpointMeta {
                    step,
                    model: model_config.clone(),
                    train: train_config.clone(),
                },
            )?;
            if train_config
                .mini_core_every
                .is_some_and(|every| step % every == 0)
            {
                let mini_core = run_mini_core(
                    &checkpoint_dir,
                    device,
                    Some(train_config.model_dtype.as_str()),
                    train_config.mini_core_limit,
                )?;
                for dataset in &mini_core.datasets {
                    println!(
                        "mini_core step={} dataset={} acc={} centered={} ex/s={}",
                        step,
                        dataset.name,
                        format_float(dataset.accuracy),
                        format_float(dataset.centered_accuracy),
                        format_float(dataset.examples_per_sec),
                    );
                    append_mini_core_csv_log(
                        &mini_core_csv_path,
                        step,
                        dataset.name,
                        dataset.accuracy,
                        dataset.centered_accuracy,
                        dataset.examples,
                        dataset.examples_per_sec,
                        &checkpoint_dir,
                    )?;
                }
                println!(
                    "mini_core step={} score={}",
                    step,
                    format_float(mini_core.mini_core)
                );
                mini_core_score = Some(mini_core.mini_core as f32);
            }
        }

        if let Some(point) = stats.last_mut() {
            point.mini_core = mini_core_score;
        }
        append_training_curve_log(&training_curve_path, stats.last().expect("point was pushed"))?;

        let should_diag = train_config
            .diag_every
            .is_some_and(|every| step % every == 0)
            || (train_config.diag_every.is_none()
                && train_config
                    .mini_core_every
                    .is_some_and(|every| step % every == 0));
        let scaling_predictor_enabled = train_config.scaling_predictor.unwrap_or(
            train_config.mini_core_every.is_some(),
        );
        if should_diag {
            if let Some(report) = build_diagnostics(&stats) {
                print_training_diagnostics(&report, scaling_predictor_enabled);
            }
        }

        progress.set_message(step.to_string());
        progress.inc(1);
    }

    progress.finish_and_clear();
    Ok(())
}

pub fn eval_main(
    model_config: Config,
    model_dtype: DType,
    checkpoint: &Path,
    mut train_ds: StreamDataset,
    mut val_ds: StreamDataset,
    _seq_len: usize,
    _batch_size: usize,
    val_batches: usize,
    device: &Device,
) -> Result<()> {
    let (mut varmap, model) = build_model(model_config, model_dtype, device)?;
    load_checkpoint(checkpoint, &mut varmap)?;

    let dataset = if val_ds.shard_count() > 0 {
        &mut val_ds
    } else {
        &mut train_ds
    };
    let loss = evaluate_dataset(&model, dataset, val_batches, device)?;
    println!(
        "eval_shard={} val_loss={} batches={}",
        dataset
            .current_shard_name()
            .unwrap_or_else(|| "not-started".to_string()),
        format_float(loss),
        val_batches
    );
    Ok(())
}

pub fn inspect_batch_main(
    train_ds: &mut StreamDataset,
    val_ds: &StreamDataset,
    device: &Device,
) -> Result<()> {
    let batch = train_ds
        .next_batch(device)?
        .context("inspect-batch could not produce a batch")?;
    println!(
        "train_shards={} val_shards={} x_shape={:?} y_shape={:?}",
        train_ds.shard_count(),
        val_ds.shard_count(),
        batch.xs.dims(),
        batch.ys.dims(),
    );
    println!("docs_consumed={}", batch.docs_consumed);
    println!("tokenizer_queue_depth={}", batch.tokenizer_queue_depth);
    println!("parquet_queue_depth={}", batch.parquet_queue_depth);
    println!("current_shard={}", batch.current_shard);
    println!("sample_x0={:?}", batch.xs.get(0)?.to_vec1::<u32>()?);
    println!("sample_y0={:?}", batch.ys.get(0)?.to_vec1::<u32>()?);
    Ok(())
}

fn evaluate_dataset(
    model: &crate::model::OperatorStateLM,
    dataset: &mut StreamDataset,
    batches: usize,
    device: &Device,
) -> Result<f64> {
    if batches == 0 {
        bail!("val_batches must be >= 1");
    }
    let mut total = 0f64;
    let mut seen = 0usize;
    for _ in 0..batches {
        let Some(batch) = dataset.next_batch(device)? else {
            break;
        };
        let (logits, _) = model.forward(&batch.xs)?;
        let loss = cross_entropy_loss(&logits, &batch.ys)?;
        total += loss.to_dtype(DType::F32)?.to_scalar::<f32>()? as f64;
        seen += 1;
    }
    if seen == 0 {
        bail!("evaluation dataset produced zero batches")
    }
    Ok(total / seen as f64)
}

fn init_csv_log(path: &Path) -> Result<()> {
    if !path.exists() {
        let mut file = File::create(path)?;
        writeln!(
            file,
            "step,train_loss,val_loss,train_tokens_per_sec,train_docs_per_sec,batch_wait_ms,step_time_ms,eval_time_ms,tokenizer_queue_depth,parquet_queue_depth,current_shard,elapsed_sec"
        )?;
        file.flush()?;
    }
    Ok(())
}

fn append_csv_log(
    path: &Path,
    step: usize,
    train_loss: f64,
    val_loss: Option<f64>,
    train_tokens_per_sec: f64,
    train_docs_per_sec: f64,
    batch_wait_ms: f64,
    step_time_ms: f64,
    eval_time_ms: Option<f64>,
    tokenizer_queue_depth: usize,
    parquet_queue_depth: usize,
    current_shard: &str,
    elapsed_sec: f64,
) -> Result<()> {
    let mut file = OpenOptions::new().append(true).open(path)?;
    writeln!(
        file,
        "{step},{train_loss},{},{train_tokens_per_sec},{train_docs_per_sec},{batch_wait_ms},{step_time_ms},{},{tokenizer_queue_depth},{parquet_queue_depth},{current_shard},{elapsed_sec}",
        val_loss.map(|v| v.to_string()).unwrap_or_default(),
        eval_time_ms.map(|v| v.to_string()).unwrap_or_default(),
    )?;
    Ok(())
}

fn save_checkpoint(path: &Path, varmap: &candle_nn::VarMap, meta: &CheckpointMeta) -> Result<()> {
    fs::create_dir_all(path)?;
    varmap
        .save(path.join("model.safetensors"))
        .with_context(|| format!("failed to save weights into {}", path.display()))?;
    write_json_pretty(&path.join("meta.json"), meta)?;
    Ok(())
}

fn init_mini_core_csv_log(path: &Path) -> Result<()> {
    if !path.exists() {
        let mut file = File::create(path)?;
        writeln!(
            file,
            "step,dataset,accuracy,centered_accuracy,examples,examples_per_sec,checkpoint"
        )?;
        file.flush()?;
    }
    Ok(())
}

fn append_mini_core_csv_log(
    path: &Path,
    step: usize,
    dataset: &str,
    accuracy: f64,
    centered_accuracy: f64,
    examples: usize,
    examples_per_sec: f64,
    checkpoint: &Path,
) -> Result<()> {
    let mut file = OpenOptions::new().append(true).open(path)?;
    writeln!(
        file,
        "{step},{dataset},{accuracy},{centered_accuracy},{examples},{examples_per_sec},{}",
        checkpoint.display()
    )?;
    Ok(())
}

fn init_training_curve_log(path: &Path) -> Result<()> {
    if !path.exists() {
        File::create(path)?;
    }
    Ok(())
}

fn append_training_curve_log(path: &Path, point: &TrainPoint) -> Result<()> {
    let mut file = OpenOptions::new().append(true).open(path)?;
    serde_json::to_writer(&mut file, point)?;
    writeln!(file)?;
    Ok(())
}

fn print_training_diagnostics(
    report: &crate::diag::ascii_plot::DiagnosticsReport,
    scaling_predictor_enabled: bool,
) {
    println!("==============================");
    println!("Training Diagnostics");
    println!("==============================");
    println!("tokens_seen={}", report.latest_tokens_seen);
    println!("steps={}", report.latest_step);
    println!("train_loss={}", format_float(report.latest_train_loss as f64));
    println!(
        "val_loss={}",
        report
            .latest_val_loss
            .map(|value| format_float(value as f64))
            .unwrap_or_else(|| "na".to_string())
    );
    println!(
        "mini_core={}",
        report
            .latest_mini_core
            .map(|value| format_float(value as f64))
            .unwrap_or_else(|| "na".to_string())
    );
    println!(
        "learning_slope={}",
        report
            .learning_slope
            .map(format_float)
            .unwrap_or_else(|| "na".to_string())
    );
    println!();
    println!("{}", report.plot);
    if scaling_predictor_enabled {
        if let Some(pred) = report.scaling_prediction {
            println!();
            println!("Scaling Law Prediction");
            println!("----------------------");
            println!("alpha={}", format_float(pred.alpha as f64));
            println!("pred_loss@10M={}", format_float(pred.predicted_loss_10m as f64));
            println!(
                "pred_loss@100M={}",
                format_float(pred.predicted_loss_100m as f64)
            );
            println!("pred_loss@1B={}", format_float(pred.predicted_loss_1b as f64));
            println!(
                "predicted_mini_core={}",
                format_float(pred.predicted_mini_core as f64)
            );
            println!(
                "scaling_hint={}",
                scaling_hint(pred.alpha)
            );
        }
    }
}

fn scaling_hint(alpha: f32) -> &'static str {
    if alpha > 0.5 {
        "very strong scaling"
    } else if alpha >= 0.3 {
        "healthy"
    } else if alpha >= 0.2 {
        "weak"
    } else if alpha >= 0.1 {
        "architecture weak"
    } else {
        "training broken"
    }
}

fn load_checkpoint(path: &Path, varmap: &mut candle_nn::VarMap) -> Result<()> {
    let model_path = if path.is_dir() {
        path.join("model.safetensors")
    } else {
        path.to_path_buf()
    };
    varmap
        .load(&model_path)
        .with_context(|| format!("failed to load checkpoint {}", model_path.display()))
}

fn load_checkpoint_meta(path: &Path) -> Result<CheckpointMeta> {
    let meta_path = if path.is_dir() {
        path.join("meta.json")
    } else {
        let parent = path
            .parent()
            .context("checkpoint path has no parent directory")?;
        parent.join("meta.json")
    };
    Ok(serde_json::from_reader(File::open(&meta_path)?)?)
}

fn clip_gradients(
    varmap: &candle_nn::VarMap,
    grads: &mut candle_core::backprop::GradStore,
    max_norm: f64,
) -> Result<()> {
    if max_norm <= 0.0 {
        return Ok(());
    }
    let vars = varmap.all_vars();
    let mut total_sq = 0f64;
    for var in &vars {
        if let Some(grad) = grads.get(var.as_tensor()) {
            let grad_sq = grad
                .sqr()?
                .sum_all()?
                .to_dtype(DType::F32)?
                .to_scalar::<f32>()? as f64;
            total_sq += grad_sq;
        }
    }
    let total_norm = total_sq.sqrt();
    if total_norm <= max_norm {
        return Ok(());
    }
    let scale = max_norm / (total_norm + 1e-6);
    for var in &vars {
        if let Some(grad) = grads.remove(var.as_tensor()) {
            let clipped = (&grad * scale)?;
            grads.insert(var.as_tensor(), clipped);
        }
    }
    Ok(())
}
