use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use candle_core::Device;
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use indicatif::{ProgressBar, ProgressStyle};
use rand::{SeedableRng, rngs::StdRng};

use crate::config::{CheckpointMeta, Config, TrainConfig};
use crate::dataset::TokenDataset;
use crate::model::{build_model, cross_entropy_loss};
use crate::utils::{format_float, write_json_pretty};

pub fn train_main(
    model_config: Config,
    train_config: TrainConfig,
    train_ds: TokenDataset,
    val_ds: Option<TokenDataset>,
    device: &Device,
) -> Result<()> {
    fs::create_dir_all(&train_config.out_dir)?;
    fs::create_dir_all(train_config.out_dir.join("checkpoints"))?;
    fs::create_dir_all(train_config.out_dir.join("logs"))?;

    let (mut varmap, model) = build_model(model_config.clone(), device)?;
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
    init_csv_log(&csv_path)?;

    let progress = ProgressBar::new(train_config.steps as u64);
    progress.set_position(start_step as u64);
    progress.set_style(ProgressStyle::with_template(
        "[{elapsed_precise}] {wide_bar} {pos}/{len} step={msg}",
    )?);

    let mut rng = StdRng::seed_from_u64(42);
    let start_time = Instant::now();
    let mut running_tokens = 0usize;

    for step in (start_step + 1)..=train_config.steps {
        let batch = train_ds.sample_batch(
            train_config.batch_size,
            train_config.seq_len,
            &mut rng,
            device,
        )?;
        let (logits, _) = model.forward(&batch.xs)?;
        let loss = cross_entropy_loss(&logits, &batch.ys)?;

        let mut grads = loss.backward()?;
        clip_gradients(&varmap, &mut grads, train_config.grad_clip)?;
        optimizer.step(&grads)?;

        let train_loss = loss.to_scalar::<f32>()? as f64;
        running_tokens += train_config.batch_size * train_config.seq_len;
        let elapsed = start_time.elapsed().as_secs_f64().max(1e-9);
        let tokens_per_sec = running_tokens as f64 / elapsed;

        let mut val_loss = None;
        if step % train_config.eval_every == 0 {
            let eval_ds = val_ds.as_ref().unwrap_or(&train_ds);
            let loss = evaluate_dataset(
                &model,
                eval_ds,
                train_config.seq_len,
                train_config.batch_size,
                train_config.val_batches,
                device,
            )?;
            val_loss = Some(loss);
        }

        if step % train_config.log_every == 0 || val_loss.is_some() || step == 1 {
            println!(
                "step={} train_loss={} val_loss={} tok/s={} elapsed_sec={}",
                step,
                format_float(train_loss),
                val_loss
                    .map(format_float)
                    .unwrap_or_else(|| "na".to_string()),
                format_float(tokens_per_sec),
                format_float(elapsed),
            );
            append_csv_log(
                &csv_path,
                step,
                train_loss,
                val_loss,
                tokens_per_sec,
                elapsed,
            )?;
        }

        if step % train_config.save_every == 0 || step == train_config.steps {
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
        }

        progress.set_message(step.to_string());
        progress.inc(1);
    }

    progress.finish_and_clear();
    Ok(())
}

pub fn eval_main(
    model_config: Config,
    checkpoint: &Path,
    train_ds: TokenDataset,
    val_ds: Option<TokenDataset>,
    seq_len: usize,
    batch_size: usize,
    val_batches: usize,
    device: &Device,
) -> Result<()> {
    let (mut varmap, model) = build_model(model_config, device)?;
    load_checkpoint(checkpoint, &mut varmap)?;
    let dataset = val_ds.as_ref().unwrap_or(&train_ds);
    let loss = evaluate_dataset(&model, dataset, seq_len, batch_size, val_batches, device)?;
    println!(
        "eval_dataset={} val_loss={} batches={}",
        dataset.path().display(),
        format_float(loss),
        val_batches
    );
    Ok(())
}

pub fn inspect_batch_main(
    train_ds: TokenDataset,
    val_ds: Option<TokenDataset>,
    seq_len: usize,
    batch_size: usize,
    device: &Device,
) -> Result<()> {
    let mut rng = StdRng::seed_from_u64(7);
    let batch = train_ds.sample_batch(batch_size, seq_len, &mut rng, device)?;
    println!(
        "train_path={} len_tokens={} max_token={} x_shape={:?} y_shape={:?}",
        train_ds.path().display(),
        train_ds.len_tokens(),
        train_ds.max_token(),
        batch.xs.dims(),
        batch.ys.dims(),
    );
    println!("sample_starts={:?}", batch.starts);
    println!(
        "sample_x0={:?}",
        train_ds.inspect_window(batch.starts[0], seq_len)?
    );
    println!(
        "sample_y0={:?}",
        train_ds.inspect_window(batch.starts[0] + 1, seq_len)?
    );
    if let Some(val_ds) = val_ds {
        println!(
            "val_path={} len_tokens={} max_token={}",
            val_ds.path().display(),
            val_ds.len_tokens(),
            val_ds.max_token(),
        );
    }
    Ok(())
}

fn evaluate_dataset(
    model: &crate::model::OperatorStateLM,
    dataset: &TokenDataset,
    seq_len: usize,
    batch_size: usize,
    batches: usize,
    device: &Device,
) -> Result<f64> {
    if batches == 0 {
        bail!("val_batches must be >= 1");
    }
    let mut rng = StdRng::seed_from_u64(1234);
    let mut total = 0f64;
    for _ in 0..batches {
        let batch = dataset.sample_batch(batch_size, seq_len, &mut rng, device)?;
        let (logits, _) = model.forward(&batch.xs)?;
        let loss = cross_entropy_loss(&logits, &batch.ys)?;
        total += loss.to_scalar::<f32>()? as f64;
    }
    Ok(total / batches as f64)
}

fn init_csv_log(path: &Path) -> Result<()> {
    if !path.exists() {
        let mut file = File::create(path)?;
        writeln!(file, "step,train_loss,val_loss,tokens_per_sec,elapsed_sec")?;
    }
    Ok(())
}

fn append_csv_log(
    path: &Path,
    step: usize,
    train_loss: f64,
    val_loss: Option<f64>,
    tokens_per_sec: f64,
    elapsed_sec: f64,
) -> Result<()> {
    let mut file = OpenOptions::new().append(true).open(path)?;
    writeln!(
        file,
        "{step},{train_loss},{},{tokens_per_sec},{elapsed_sec}",
        val_loss.map(|v| v.to_string()).unwrap_or_default(),
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
            let grad_sq = grad.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
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
