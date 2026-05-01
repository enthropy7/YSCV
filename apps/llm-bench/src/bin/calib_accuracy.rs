//! PTQ calibration accuracy harness — compares MinMax vs Percentile
//! vs MSE-optimal scale derivation on synthetic activation distributions.
//!
//! Pushes 100k samples from each distribution through `CalibrationCollector`,
//! derives `(scale, zero_point)` three ways for symmetric int8 (`[-128, 127]`),
//! then quantizes/dequantizes the same population and reports:
//!
//! - **RMSE** — root-mean-squared error fp32 vs round-tripped fp32
//! - **MAE**  — mean absolute error
//! - **MaxE** — max absolute error
//! - **SNR**  — signal-to-noise ratio in dB (higher is better)
//!
//! Distributions cover the cases where each method is supposed to win:
//!   1. uniform[-1,1]                        → MinMax close to optimal
//!   2. gaussian σ=1                          → percentile/MSE win modestly
//!   3. gaussian σ=1 + 0.1% outlier at ±50    → percentile/MSE win big
//!   4. heavy-tail (Cauchy-ish)               → percentile/MSE win biggest
//!   5. bimodal (two narrow peaks at ±0.3)    → MinMax over-allocates range
//!
//! Run from repo root:
//!
//! ```sh
//! cargo run --release --bin calib_accuracy -p yscv-llm-bench
//! ```

use yscv_onnx::quantize::{
    CalibrationCollector,
    derive::{QuantTarget, derive_mse_optimal, derive_percentile, derive_symmetric},
};
use yscv_onnx::{OnnxRunner, load_onnx_model_from_file, optimize_onnx_graph};
use yscv_tensor::Tensor;

const N_SAMPLES: usize = 100_000;
const QMIN: f32 = -128.0;
const QMAX: f32 = 127.0;

fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    // U(0, 1)
    ((*state >> 40) as f32) / ((1u64 << 24) as f32)
}

/// Box-Muller standard normal.
fn randn(state: &mut u64) -> f32 {
    let u1 = (lcg(state)).max(1e-7);
    let u2 = lcg(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

fn dist_uniform(seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..N_SAMPLES).map(|_| 2.0 * lcg(&mut s) - 1.0).collect()
}

fn dist_gaussian(seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..N_SAMPLES).map(|_| randn(&mut s)).collect()
}

fn dist_gaussian_with_outliers(seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..N_SAMPLES)
        .map(|i| {
            // ~0.1% extreme outliers at +/- 50.
            if i % 1000 == 0 {
                if i % 2000 == 0 { 50.0 } else { -50.0 }
            } else {
                randn(&mut s)
            }
        })
        .collect()
}

fn dist_heavy_tail(seed: u64) -> Vec<f32> {
    let mut s = seed;
    // Cauchy-like via tan(pi*(u-0.5)) — very heavy tails.
    (0..N_SAMPLES)
        .map(|_| (std::f32::consts::PI * (lcg(&mut s) - 0.5)).tan())
        .collect()
}

fn dist_bimodal(seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..N_SAMPLES)
        .map(|_| {
            let pick = lcg(&mut s);
            let centre = if pick < 0.5 { -0.3 } else { 0.3 };
            centre + 0.05 * randn(&mut s)
        })
        .collect()
}

#[derive(Default, Clone, Copy)]
struct Errors {
    rmse: f64,
    mae: f64,
    max: f64,
    snr_db: f64,
    scale: f32,
}

fn round_trip_errors(values: &[f32], scale: f32) -> Errors {
    if scale <= 0.0 || !scale.is_finite() {
        return Errors {
            scale,
            ..Default::default()
        };
    }
    let inv = 1.0 / scale;
    let mut sse = 0.0_f64;
    let mut sae = 0.0_f64;
    let mut max_e = 0.0_f64;
    let mut signal = 0.0_f64;
    for &v in values {
        let q = (v * inv).round().clamp(QMIN, QMAX);
        let dq = q * scale;
        let e = (v - dq) as f64;
        sse += e * e;
        sae += e.abs();
        max_e = max_e.max(e.abs());
        signal += (v as f64) * (v as f64);
    }
    let n = values.len() as f64;
    let rmse = (sse / n).sqrt();
    let mae = sae / n;
    let snr_db = if sse > 0.0 {
        10.0 * (signal / sse).log10()
    } else {
        f64::INFINITY
    };
    Errors {
        rmse,
        mae,
        max: max_e,
        snr_db,
        scale,
    }
}

fn evaluate(label: &str, values: &[f32]) {
    let coll = CalibrationCollector::new();
    coll.enable_histograms(true);
    // Push in chunks so the histogram reservoir actually exercises its
    // replacement branch (capacity 16384, samples 100000).
    for chunk in values.chunks(8192) {
        coll.record("activations", chunk);
    }

    let minmax = coll.snapshot()["activations"];
    let hist = coll.histograms()["activations"].clone();

    let p_minmax = derive_symmetric(minmax, QuantTarget::Int8);
    let p_pct_001 = derive_percentile(&hist, QuantTarget::Int8, 0.001);
    let p_pct_01 = derive_percentile(&hist, QuantTarget::Int8, 0.01);
    let p_mse = derive_mse_optimal(&hist, QuantTarget::Int8, 100);

    let e_mm = round_trip_errors(values, p_minmax.scale);
    let e_p1 = round_trip_errors(values, p_pct_001.scale);
    let e_p2 = round_trip_errors(values, p_pct_01.scale);
    let e_ms = round_trip_errors(values, p_mse.scale);

    println!("\n=== {label}");
    println!(
        "  observed: count={} min={:.4} max={:.4} abs_max={:.4}",
        minmax.count,
        minmax.min,
        minmax.max,
        minmax.abs_max()
    );
    println!(
        "  {:<24} scale={:>10.5}  RMSE={:>9.4e}  MAE={:>9.4e}  MaxE={:>9.4e}  SNR={:>6.2} dB",
        "MinMax", e_mm.scale, e_mm.rmse, e_mm.mae, e_mm.max, e_mm.snr_db
    );
    println!(
        "  {:<24} scale={:>10.5}  RMSE={:>9.4e}  MAE={:>9.4e}  MaxE={:>9.4e}  SNR={:>6.2} dB",
        "Percentile (0.1%)", e_p1.scale, e_p1.rmse, e_p1.mae, e_p1.max, e_p1.snr_db
    );
    println!(
        "  {:<24} scale={:>10.5}  RMSE={:>9.4e}  MAE={:>9.4e}  MaxE={:>9.4e}  SNR={:>6.2} dB",
        "Percentile (1%)", e_p2.scale, e_p2.rmse, e_p2.mae, e_p2.max, e_p2.snr_db
    );
    println!(
        "  {:<24} scale={:>10.5}  RMSE={:>9.4e}  MAE={:>9.4e}  MaxE={:>9.4e}  SNR={:>6.2} dB",
        "MSE-optimal (grid 100)", e_ms.scale, e_ms.rmse, e_ms.mae, e_ms.max, e_ms.snr_db
    );

    let best = [
        ("MinMax", e_mm.snr_db),
        ("Percentile(0.1%)", e_p1.snr_db),
        ("Percentile(1%)", e_p2.snr_db),
        ("MSE-optimal", e_ms.snr_db),
    ]
    .into_iter()
    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    .unwrap();
    println!("  → best by SNR: {} ({:.2} dB)", best.0, best.1);
}

/// Per-tensor activation collected from a real ONNX inference run.
/// Each entry is `(activation_name, sample_values)` — `evaluate`
/// then plays the synthetic-distribution path against real data.
fn collect_real_activations(
    model_path: &str,
    inputs: &[(&str, Vec<usize>)],
    n_runs: usize,
) -> Result<Vec<(String, Vec<f32>)>, String> {
    let mut model = load_onnx_model_from_file(model_path).map_err(|e| format!("load: {e}"))?;
    // The CPU runner expects the optimizer to have run first — it
    // pre-permutes Conv weights OIHW → KHWC and registers fusions the
    // exec paths rely on. Skipping this step makes Conv kernels see
    // raw OIHW shapes and triggers "bias shape mismatch" on the first
    // conv layer.
    optimize_onnx_graph(&mut model);
    let runner = OnnxRunner::new(&model).map_err(|e| format!("runner: {e}"))?;
    let coll = CalibrationCollector::new();
    coll.enable_histograms(true);
    // Probe one run *without* calibration first — failures here are
    // model/runner bugs, not anything to do with the calibration hook.
    {
        let n: usize = inputs[0].1.iter().product();
        let mut s = 0xCA11_u64;
        let probe_data: Vec<f32> = (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (((s >> 33) as i64 % 2001) - 1000) as f32 * 0.001
            })
            .collect();
        let probe_t = Tensor::from_vec(inputs[0].1.clone(), probe_data).unwrap();
        let mut tensors2: Vec<Tensor> = vec![probe_t];
        for (_, shape) in &inputs[1..] {
            let n2: usize = shape.iter().product();
            tensors2.push(Tensor::from_vec(shape.clone(), vec![0.0_f32; n2]).unwrap());
        }
        let feed: Vec<(&str, &Tensor)> = inputs
            .iter()
            .zip(&tensors2)
            .map(|((n, _), t)| (*n, t))
            .collect();
        if let Err(e) = runner.run(&feed) {
            return Err(format!("probe run failed (no calibration scope): {e:?}"));
        }
    }
    {
        let _scope = coll.scope();
        for run in 0..n_runs {
            // Slightly different random init per run so we get distribution
            // statistics rather than re-quantising the same activation.
            let seed = 0xCA11 ^ (run as u64);
            let mut tensors: Vec<Tensor> = Vec::with_capacity(inputs.len());
            for (_, shape) in inputs {
                let n: usize = shape.iter().product();
                let mut s = seed.wrapping_add(n as u64);
                let data: Vec<f32> = (0..n)
                    .map(|_| {
                        s = s
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        (((s >> 33) as i64 % 2001) - 1000) as f32 * 0.001
                    })
                    .collect();
                tensors.push(
                    Tensor::from_vec(shape.clone(), data)
                        .map_err(|e| format!("build input {run}: {e}"))?,
                );
            }
            let feed: Vec<(&str, &Tensor)> = inputs
                .iter()
                .zip(&tensors)
                .map(|((n, _), t)| (*n, t))
                .collect();
            runner.run(&feed).map_err(|e| format!("run {run}: {e}"))?;
        }
    }
    let snap = coll.histograms();
    Ok(snap
        .into_iter()
        .map(|(name, h)| (name, h.samples))
        .collect())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    println!(
        "PTQ calibration accuracy — int8 symmetric round-trip\n\
         {} samples per distribution; SNR higher = lower quantization error.",
        N_SAMPLES
    );

    if let Some(model_path) = args
        .iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1))
    {
        let model_path = model_path.as_str();
        println!("\nReal-activation mode: {model_path}");
        // Shape parse: --shape NAME:DxDxD,NAME:DxDxD (repeatable inline).
        // Default falls back to the bundled tracker shape pair.
        let shapes_arg = args
            .iter()
            .position(|a| a == "--shape")
            .and_then(|i| args.get(i + 1))
            .map(|s| s.as_str());
        let inputs: Vec<(String, Vec<usize>)> = match shapes_arg {
            Some(spec) => spec
                .split(',')
                .map(|pair| {
                    let mut p = pair.split(':');
                    let name = p.next().expect("shape name").to_string();
                    let shape: Vec<usize> = p
                        .next()
                        .expect("shape dims")
                        .split('x')
                        .map(|d| d.parse().expect("dim"))
                        .collect();
                    (name, shape)
                })
                .collect(),
            None => vec![
                // bundled tracker: template 128×128, search 256×256.
                ("input.1".to_string(), vec![1, 3, 128, 128]),
                ("input.249".to_string(), vec![1, 3, 256, 256]),
            ],
        };
        let inputs_borrow: Vec<(&str, Vec<usize>)> = inputs
            .iter()
            .map(|(n, s)| (n.as_str(), s.clone()))
            .collect();
        match collect_real_activations(model_path, &inputs_borrow, 8) {
            Ok(real) => {
                println!(
                    "\ncollected {} per-tensor histograms over 8 runs",
                    real.len()
                );
                let mut sorted = real;
                sorted.sort_by_key(|x| std::cmp::Reverse(x.1.len()));
                for (name, samples) in sorted.iter().take(5) {
                    evaluate(&format!("real:{name}"), samples);
                }
            }
            Err(e) => {
                eprintln!(
                    "\n!! real-activation run failed: {e}\n   \
                     Model not currently runnable through this build of yscv-onnx. \
                     CalibrationCollector / scope plumbing IS exercised by the unit \
                     tests; falling through to synthetic distributions below."
                );
            }
        }
    }

    evaluate("uniform[-1, 1]", &dist_uniform(0xA1));
    evaluate("gaussian (σ=1)", &dist_gaussian(0xA2));
    evaluate(
        "gaussian + 0.1% outliers at ±50",
        &dist_gaussian_with_outliers(0xA3),
    );
    evaluate("heavy-tail (Cauchy-like)", &dist_heavy_tail(0xA4));
    evaluate("bimodal (peaks at ±0.3, σ=0.05)", &dist_bimodal(0xA5));
}
