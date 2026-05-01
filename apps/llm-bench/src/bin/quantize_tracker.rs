//! End-to-end multi-input quantization demo + accuracy validator.
//!
//! Targets the kind of edge-CV model that breaks RKNN's calibration step
//! — the bundled Siamese tracker at
//! `/home/null/YSCV/private/private/model.onnx` takes TWO inputs
//! simultaneously (template 1×3×128×128 and search 1×3×256×256).
//!
//! Pipeline:
//!   1. Load fp32 model + run optimizer (Conv weights → KHWC, fusions).
//!   2. Run `n_calib` synthetic samples through `CalibrationCollector`
//!      with both inputs fed each step (the multi-input case).
//!   3. Call `rewrite_to_qdq` on the model.
//!   4. (Optionally) `save_onnx_model_to_file` + reload to verify the
//!      QDQ model round-trips through the protobuf exporter.
//!   5. Run a separate `n_eval` samples through *both* fp32 and the
//!      QDQ model, report L∞ / RMSE / max-abs delta on each output
//!      head. <2% L∞ delta is the "shipping" threshold.
//!
//! Usage:
//!
//! ```sh
//! cargo run --release --no-default-features --bin quantize_tracker \
//!   -p yscv-llm-bench -- \
//!   --model /home/null/YSCV/private/private/model.onnx \
//!   --shape input.1:1x3x128x128,input.249:1x3x256x256 \
//!   --output tracker_int8.onnx \
//!   --calib-samples 16 --eval-samples 8
//! ```
//!
//! `--shape` is the same `name:DxDxD,name:DxDxD` form as `inspect` and
//! `calib_accuracy --model`. Without `--output` the QDQ save+reload step
//! is skipped — useful when iterating on the calibration count alone.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use serde::Deserialize;
use yscv_onnx::quantize::{
    CalibrationCollector, fold_constant_qdq_weights_for_yscv_fast, prune_unused_initializers,
    rewrite_to_qdq, rewrite_to_qlinear,
};
use yscv_onnx::{
    OnnxRunner, export_onnx_model_to_file, load_onnx_model_from_file, onnx_model_to_export_graph,
    optimize_onnx_graph, quant_runtime_stats, reset_quant_runtime_stats,
    strip_qdq_within_fusion_chains,
};
use yscv_tensor::Tensor;

#[derive(Debug)]
struct Args {
    model: String,
    shape_spec: String,
    output: Option<String>,
    format: QuantFormat,
    calibration_jsonl: Option<String>,
    eval_jsonl: Option<String>,
    calib_samples: Option<usize>,
    eval_samples: Option<usize>,
    seed: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QuantFormat {
    Qdq,
    QLinear,
}

fn parse_args() -> Result<Args, String> {
    let mut model: Option<String> = None;
    let mut shape_spec: Option<String> = None;
    let mut output: Option<String> = None;
    let mut format = QuantFormat::Qdq;
    let mut calibration_jsonl: Option<String> = None;
    let mut eval_jsonl: Option<String> = None;
    let mut calib_samples: Option<usize> = None;
    let mut eval_samples: Option<usize> = None;
    let mut seed: u64 = 0xCA11;
    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--model" => model = iter.next(),
            "--shape" => shape_spec = iter.next(),
            "--output" => output = iter.next(),
            "--calibration-jsonl" => calibration_jsonl = iter.next(),
            "--eval-jsonl" => eval_jsonl = iter.next(),
            "--format" => {
                let value = iter.next().ok_or("missing --format value")?;
                format = match value.as_str() {
                    "qdq" => QuantFormat::Qdq,
                    "qlinear" => QuantFormat::QLinear,
                    _ => {
                        return Err(format!(
                            "--format must be `qdq` or `qlinear`, got `{value}`"
                        ));
                    }
                };
            }
            "--calib-samples" => {
                calib_samples = Some(
                    iter.next()
                        .ok_or("missing --calib-samples value")?
                        .parse()
                        .map_err(|e| format!("--calib-samples: {e}"))?,
                );
            }
            "--eval-samples" => {
                eval_samples = Some(
                    iter.next()
                        .ok_or("missing --eval-samples value")?
                        .parse()
                        .map_err(|e| format!("--eval-samples: {e}"))?,
                );
            }
            "--seed" => {
                seed = iter
                    .next()
                    .ok_or("missing --seed value")?
                    .parse()
                    .map_err(|e| format!("--seed: {e}"))?;
            }
            "-h" | "--help" => {
                eprintln!(
                    "usage: quantize_tracker --model PATH --shape NAME:DxDxD[,NAME:DxDxD]* \
                     [--output OUT.onnx] [--format qdq|qlinear] [--calib-samples N] \
                     [--eval-samples N] [--calibration-jsonl JSONL|NAME=JSONL,...] \
                     [--eval-jsonl JSONL|NAME=JSONL,...] [--seed N]"
                );
                return Err("help".into());
            }
            other => return Err(format!("unknown arg: {other}")),
        }
    }
    Ok(Args {
        model: model.ok_or("--model is required")?,
        shape_spec: shape_spec.ok_or("--shape is required")?,
        output,
        format,
        calibration_jsonl,
        eval_jsonl,
        calib_samples,
        eval_samples,
        seed,
    })
}

#[derive(Debug, Clone, Deserialize)]
struct SampleTensor {
    shape: Vec<usize>,
    values: Vec<f32>,
}

fn tensor_from_sample(name: &str, sample: SampleTensor, line: usize) -> Result<Tensor, String> {
    let expected: usize = sample.shape.iter().product();
    if expected != sample.values.len() {
        return Err(format!(
            "line {line}: tensor `{name}` shape {:?} expects {expected} values, got {}",
            sample.shape,
            sample.values.len()
        ));
    }
    Tensor::from_vec(sample.shape, sample.values)
        .map_err(|e| format!("line {line}: tensor `{name}`: {e}"))
}

fn read_single_tensor_stream(name: &str, path: &PathBuf) -> Result<Vec<Tensor>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut tensors = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        let line_no = idx + 1;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let sample: SampleTensor = match serde_json::from_str(line) {
            Ok(sample) => sample,
            Err(_) => {
                let wrapped: HashMap<String, SampleTensor> =
                    serde_json::from_str(line).map_err(|e| {
                        format!("{}:{line_no}: invalid tensor JSONL: {e}", path.display())
                    })?;
                wrapped
                    .into_iter()
                    .find(|(k, _)| k == name)
                    .map(|(_, v)| v)
                    .ok_or_else(|| {
                        format!(
                            "{}:{line_no}: wrapped sample does not contain `{name}`",
                            path.display()
                        )
                    })?
            }
        };
        tensors.push(tensor_from_sample(name, sample, line_no)?);
    }
    Ok(tensors)
}

fn read_jsonl_samples(
    spec: &str,
    inputs: &[(String, Vec<usize>)],
) -> Result<Vec<Vec<Tensor>>, String> {
    if spec.contains('=') {
        let mut streams = Vec::new();
        for pair in spec.split(',') {
            let (name, path) = pair
                .split_once('=')
                .ok_or_else(|| format!("paired JSONL entry `{pair}` must be NAME=PATH"))?;
            if name.is_empty() || path.is_empty() {
                return Err(format!("paired JSONL entry `{pair}` must be NAME=PATH"));
            }
            streams.push((
                name.to_string(),
                read_single_tensor_stream(name, &PathBuf::from(path))?,
            ));
        }
        let expected_len = streams.first().map(|(_, s)| s.len()).unwrap_or(0);
        for (name, tensors) in &streams {
            if tensors.len() != expected_len {
                return Err(format!(
                    "paired JSONL stream `{name}` has {} sample(s), expected {expected_len}",
                    tensors.len()
                ));
            }
        }
        let mut by_name: HashMap<String, Vec<Tensor>> = streams.into_iter().collect();
        let mut samples = Vec::with_capacity(expected_len);
        for idx in 0..expected_len {
            let mut row = Vec::with_capacity(inputs.len());
            for (name, _) in inputs {
                let tensors = by_name
                    .get_mut(name)
                    .ok_or_else(|| format!("paired JSONL spec missing input `{name}`"))?;
                row.push(tensors[idx].clone());
            }
            samples.push(row);
        }
        validate_sample_shapes(&samples, inputs)?;
        return Ok(samples);
    }

    let path = PathBuf::from(spec);
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut samples = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        let line_no = idx + 1;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parsed: HashMap<String, SampleTensor> = serde_json::from_str(line)
            .map_err(|e| format!("{}:{line_no}: invalid sample JSONL: {e}", path.display()))?;
        let mut row = Vec::with_capacity(inputs.len());
        for (name, _) in inputs {
            let sample = parsed
                .remove(name)
                .ok_or_else(|| format!("{}:{line_no}: missing input `{name}`", path.display()))?;
            row.push(tensor_from_sample(name, sample, line_no)?);
        }
        samples.push(row);
    }
    validate_sample_shapes(&samples, inputs)?;
    Ok(samples)
}

fn validate_sample_shapes(
    samples: &[Vec<Tensor>],
    inputs: &[(String, Vec<usize>)],
) -> Result<(), String> {
    for (sample_idx, row) in samples.iter().enumerate() {
        for ((name, expected), tensor) in inputs.iter().zip(row) {
            if tensor.shape() != expected.as_slice() {
                return Err(format!(
                    "sample {} input `{name}` shape {:?}, expected {:?}",
                    sample_idx + 1,
                    tensor.shape(),
                    expected
                ));
            }
        }
    }
    Ok(())
}

fn parse_shapes(spec: &str) -> Result<Vec<(String, Vec<usize>)>, String> {
    spec.split(',')
        .map(|pair| {
            let mut p = pair.split(':');
            let name = p.next().ok_or("missing name in --shape")?.to_string();
            let dims_str = p.next().ok_or("missing dims in --shape")?;
            let dims: Result<Vec<usize>, _> = dims_str.split('x').map(|d| d.parse()).collect();
            let dims = dims.map_err(|e| format!("--shape: {e}"))?;
            Ok((name, dims))
        })
        .collect()
}

/// Reproducible LCG so callers can re-run exactly the same calibration
/// + eval inputs (so the L∞/RMSE numbers are comparable across runs).
fn lcg_f32(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (((*state >> 33) as i64 % 2001) - 1000) as f32 * 0.001
}

fn build_input_tensors(inputs: &[(String, Vec<usize>)], seed: u64) -> Result<Vec<Tensor>, String> {
    let mut s = seed;
    inputs
        .iter()
        .map(|(_, shape)| {
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n).map(|_| lcg_f32(&mut s)).collect();
            Tensor::from_vec(shape.clone(), data).map_err(|e| format!("build input: {e}"))
        })
        .collect()
}

#[derive(Default, Clone, Copy, Debug)]
struct OutputDelta {
    linf: f32,
    rmse: f32,
    rel_linf: f32,
    rel_rmse: f32,
    finite: bool,
}

fn delta(fp32: &Tensor, qdq: &Tensor) -> OutputDelta {
    let a = fp32.data();
    let b = qdq.data();
    let n = a.len().min(b.len()) as f32;
    let mut linf = 0.0_f32;
    let mut sse = 0.0_f64;
    let mut fp32_sse = 0.0_f64;
    let mut max_abs_fp32 = 1e-6_f32;
    let mut finite = true;
    for (&x, &y) in a.iter().zip(b.iter()) {
        finite &= x.is_finite() && y.is_finite();
        let d = (x - y).abs();
        if d > linf {
            linf = d;
        }
        sse += (d as f64) * (d as f64);
        fp32_sse += (x as f64) * (x as f64);
        let ax = x.abs();
        if ax > max_abs_fp32 {
            max_abs_fp32 = ax;
        }
    }
    let rmse = (sse / n as f64).sqrt() as f32;
    let fp32_rmse = (fp32_sse / n as f64).sqrt() as f32;
    OutputDelta {
        linf,
        rmse,
        rel_linf: linf / max_abs_fp32,
        rel_rmse: rmse / fp32_rmse.max(1e-6),
        finite,
    }
}

fn run(args: Args) -> Result<(), String> {
    let inputs = parse_shapes(&args.shape_spec)?;

    eprintln!("loading {}…", args.model);
    let mut model_fp32 =
        load_onnx_model_from_file(Path::new(&args.model)).map_err(|e| format!("load fp32: {e}"))?;
    optimize_onnx_graph(&mut model_fp32);
    let runner_fp32 = OnnxRunner::new(&model_fp32).map_err(|e| format!("runner fp32: {e}"))?;

    let calibration_samples = if let Some(spec) = args.calibration_jsonl.as_ref() {
        let mut samples = read_jsonl_samples(spec, &inputs)?;
        if samples.is_empty() {
            return Err(format!("--calibration-jsonl `{spec}` produced 0 samples"));
        }
        if let Some(limit) = args.calib_samples {
            samples.truncate(limit.min(samples.len()));
        }
        eprintln!(
            "calibration: loaded {} representative JSONL sample(s) from {spec}",
            samples.len()
        );
        Some(samples)
    } else {
        None
    };
    let calib_count = calibration_samples
        .as_ref()
        .map(|s| s.len())
        .unwrap_or_else(|| args.calib_samples.unwrap_or(16));
    eprintln!(
        "calibrating: {} sample(s) through {} input(s){}…",
        calib_count,
        inputs.len(),
        if calibration_samples.is_some() {
            " (representative JSONL)"
        } else {
            " (synthetic smoke)"
        }
    );
    let coll = CalibrationCollector::new();
    coll.enable_histograms(true);
    {
        let _scope = coll.scope();
        for sample in 0..calib_count {
            let tensors = if let Some(samples) = calibration_samples.as_ref() {
                samples[sample].clone()
            } else {
                build_input_tensors(&inputs, args.seed.wrapping_add(sample as u64))?
            };
            let feed: Vec<(&str, &Tensor)> = inputs
                .iter()
                .zip(&tensors)
                .map(|((n, _), t)| (n.as_str(), t))
                .collect();
            runner_fp32
                .run(&feed)
                .map_err(|e| format!("calib sample {sample}: {e}"))?;
        }
    }
    let stats = coll.snapshot();
    eprintln!("calibration: {} per-tensor stats collected", stats.len());

    eprintln!(
        "rewriting to {}…",
        match args.format {
            QuantFormat::Qdq => "QDQ",
            QuantFormat::QLinear => "QLinear",
        }
    );
    let mut model_qdq =
        load_onnx_model_from_file(Path::new(&args.model)).map_err(|e| format!("load q: {e}"))?;
    optimize_onnx_graph(&mut model_qdq);
    match args.format {
        QuantFormat::Qdq => {
            rewrite_to_qdq(&mut model_qdq, &stats).map_err(|e| format!("rewrite_to_qdq: {e}"))?
        }
        QuantFormat::QLinear => rewrite_to_qlinear(&mut model_qdq, &stats)
            .map_err(|e| format!("rewrite_to_qlinear: {e}"))?,
    }
    if args.format == QuantFormat::Qdq && std::env::var("YSCV_QUANT_FAST").as_deref() != Ok("0") {
        let folded = fold_constant_qdq_weights_for_yscv_fast(&mut model_qdq)
            .map_err(|e| format!("fold_constant_qdq_weights_for_yscv_fast: {e}"))?;
        let removed = strip_qdq_within_fusion_chains(&mut model_qdq);
        let pruned = prune_unused_initializers(&mut model_qdq);
        eprintln!(
            "yscv-fast QDQ: folded {folded} constant weight DQ node(s), stripped {removed} inner Q+DQ pair(s) inside Conv-like chains, pruned {pruned} unused initializer(s)"
        );
    } else {
        let pruned = prune_unused_initializers(&mut model_qdq);
        if pruned != 0 {
            eprintln!("quant cleanup: pruned {pruned} unused initializer(s)");
        }
    }

    if let Some(output_path) = args.output.as_ref() {
        eprintln!(
            "saving {} model to {output_path}…",
            match args.format {
                QuantFormat::Qdq => "QDQ",
                QuantFormat::QLinear => "QLinear",
            }
        );
        let mut graph = onnx_model_to_export_graph(&model_qdq);
        for value in &mut graph.inputs {
            if let Some((_, shape)) = inputs.iter().find(|(name, _)| name == &value.name) {
                value.shape = shape.iter().map(|&d| d as i64).collect();
            }
        }
        export_onnx_model_to_file(
            &graph,
            "yscv",
            &model_qdq.graph_name,
            Path::new(output_path),
        )
        .map_err(|e| format!("save: {e}"))?;
        eprintln!("reloading quantized model to verify protobuf round-trip…");
        let mut reloaded = load_onnx_model_from_file(Path::new(output_path))
            .map_err(|e| format!("reload: {e}"))?;
        if args.format == QuantFormat::Qdq {
            optimize_onnx_graph(&mut reloaded);
        }
        model_qdq = reloaded;
    }

    if args.format == QuantFormat::QLinear {
        eprintln!(
            "QLinear export reload succeeded; skipping yscv accuracy eval because this mode targets standard ORT/checker NCHW interoperability. Use --format qdq for yscv fp32-vs-quantized deltas."
        );
        return Ok(());
    }

    let runner_qdq = OnnxRunner::new(&model_qdq).map_err(|e| format!("runner qdq: {e}"))?;

    let eval_samples = if let Some(spec) = args.eval_jsonl.as_ref() {
        let mut samples = read_jsonl_samples(spec, &inputs)?;
        if samples.is_empty() {
            return Err(format!("--eval-jsonl `{spec}` produced 0 samples"));
        }
        if let Some(limit) = args.eval_samples {
            samples.truncate(limit.min(samples.len()));
        }
        eprintln!(
            "eval: loaded {} representative JSONL sample(s) from {spec}",
            samples.len()
        );
        Some(samples)
    } else {
        None
    };
    let eval_count = eval_samples
        .as_ref()
        .map(|s| s.len())
        .unwrap_or_else(|| args.eval_samples.unwrap_or(8));
    eprintln!(
        "evaluating: {} sample(s) on fp32 vs quantized{}…",
        eval_count,
        if eval_samples.is_some() {
            " (representative JSONL)"
        } else {
            " (synthetic smoke)"
        }
    );
    let outputs = model_fp32.outputs.clone();
    let mut per_output_linf: Vec<f32> = vec![0.0; outputs.len()];
    let mut per_output_rmse_sq: Vec<f64> = vec![0.0; outputs.len()];
    let mut per_output_rel_linf: Vec<f32> = vec![0.0; outputs.len()];
    let mut per_output_rel_rmse_sq: Vec<f64> = vec![0.0; outputs.len()];
    let mut per_output_finite: Vec<bool> = vec![true; outputs.len()];

    reset_quant_runtime_stats();
    for sample in 0..eval_count {
        let tensors = if let Some(samples) = eval_samples.as_ref() {
            samples[sample].clone()
        } else {
            build_input_tensors(&inputs, args.seed.wrapping_add(0xE100 + sample as u64))?
        };
        let feed: Vec<(&str, &Tensor)> = inputs
            .iter()
            .zip(&tensors)
            .map(|((n, _), t)| (n.as_str(), t))
            .collect();
        let out_fp32 = runner_fp32
            .run(&feed)
            .map_err(|e| format!("fp32 eval {sample}: {e}"))?;
        let out_qdq = runner_qdq
            .run(&feed)
            .map_err(|e| format!("qdq eval {sample}: {e}"))?;

        for (i, name) in outputs.iter().enumerate() {
            let Some(a) = out_fp32.get(name) else {
                continue;
            };
            let Some(b) = out_qdq.get(name) else {
                continue;
            };
            let d = delta(a, b);
            if d.linf > per_output_linf[i] {
                per_output_linf[i] = d.linf;
            }
            if d.rel_linf > per_output_rel_linf[i] {
                per_output_rel_linf[i] = d.rel_linf;
            }
            per_output_rmse_sq[i] += (d.rmse * d.rmse) as f64;
            per_output_rel_rmse_sq[i] += (d.rel_rmse * d.rel_rmse) as f64;
            per_output_finite[i] &= d.finite;
        }
    }
    let qstats = quant_runtime_stats();

    println!(
        "\n=== fp32 vs quantized delta on {} eval samples",
        eval_count
    );
    println!(
        "quant runtime: qdq_boundaries={} qlinear_conv_fast={} qlinear_conv_fallback={} qlinear_matmul_fast={} qlinear_matmul_fallback={} quant_i8_stores={} quant_i8_materializations={}",
        qstats.qdq_boundaries,
        qstats.qlinear_conv_fast,
        qstats.qlinear_conv_fallback,
        qstats.qlinear_matmul_fast,
        qstats.qlinear_matmul_fallback,
        qstats.quant_i8_stores,
        qstats.quant_i8_materializations
    );
    println!("  output                    L∞       rel-L∞       RMSE    rel-RMSE   finite");
    for (i, name) in outputs.iter().enumerate() {
        let denom = eval_count.max(1) as f64;
        let rmse = (per_output_rmse_sq[i] / denom).sqrt() as f32;
        let rel_rmse = (per_output_rel_rmse_sq[i] / denom).sqrt() as f32;
        println!(
            "  {:<22}  {:>9.4e}   {:>8.2}%   {:>9.4e}   {:>7.2}%   {}",
            name,
            per_output_linf[i],
            per_output_rel_linf[i] * 100.0,
            rmse,
            rel_rmse * 100.0,
            if per_output_finite[i] { "yes" } else { "NO" }
        );
    }
    let representative_gate = args.eval_jsonl.is_some();
    let gate_failed = per_output_rel_linf.iter().any(|&r| r > 0.10)
        || per_output_rel_rmse_sq
            .iter()
            .any(|&r| (r / eval_count.max(1) as f64).sqrt() > 0.02)
        || per_output_finite.iter().any(|&ok| !ok);
    if gate_failed {
        let msg = "accuracy gate failed: require rel-RMSE <= 2%, rel-Linf <= 10%, finite outputs";
        if representative_gate {
            return Err(msg.to_string());
        }
        eprintln!("WARNING: {msg} (synthetic random is smoke-only, not a ship gate).");
    } else {
        println!("\n→ accuracy gate OK: rel-RMSE <= 2%, rel-L∞ <= 10%, finite outputs.");
    }
    Ok(())
}

fn main() -> ExitCode {
    match parse_args().and_then(run) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) if e == "help" => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("quantize_tracker: {e}");
            ExitCode::FAILURE
        }
    }
}
