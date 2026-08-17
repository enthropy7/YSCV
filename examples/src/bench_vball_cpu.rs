//! Benchmark: VballNetGrid CPU inference timing.
//!
//! Usage:
//!   cargo run --release --example bench_vball_cpu -- /path/to/model.onnx [iterations]

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::time::Instant;

use rustc_hash::FxHashMap;
use yscv_onnx::{load_onnx_model_from_file, profile_onnx_model_cpu, run_onnx_model};
use yscv_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("Usage: bench_vball_cpu <model.onnx> [iterations]");
        std::process::exit(1);
    }
    let model_path = &args[0];
    let iterations: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10);

    println!("Loading model: {model_path}");
    let model = load_onnx_model_from_file(model_path)?;
    let input_name = model.inputs.first().cloned().unwrap();
    let output_name = model.outputs.first().cloned().unwrap();
    println!("Input: {input_name}, Output: {output_name}");

    // VballNetGridV1b: input [1, 9, 432, 768]
    let input = Tensor::zeros(vec![1, 9, 432, 768])?;

    // Profile
    if std::env::var("PROFILE").is_ok() {
        println!("Profiling...");
        let mut pinputs = FxHashMap::default();
        pinputs.insert(input_name.clone(), input.clone());
        let _ = profile_onnx_model_cpu(&model, pinputs);
    }

    // Warmup
    println!("Warmup...");
    let mut inputs = FxHashMap::default();
    inputs.insert(input_name.clone(), input.clone());
    let _ = run_onnx_model(&model, inputs)?;

    // Benchmark
    println!("Running {iterations} iterations...");
    let mut times = Vec::with_capacity(iterations);
    for i in 0..iterations {
        let mut inputs = FxHashMap::default();
        inputs.insert(input_name.clone(), input.clone());
        let t0 = Instant::now();
        let _ = run_onnx_model(&model, inputs)?;
        let elapsed = t0.elapsed();
        times.push(elapsed.as_secs_f64() * 1000.0);
        println!("  iter {}: {:.1} ms", i + 1, times[i]);
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let median = times[times.len() / 2];
    let min = times[0];
    let max = times[times.len() - 1];

    println!("\n--- VballNetGrid CPU Inference ---");
    println!("Iterations: {iterations}");
    println!("Mean:   {mean:.1} ms");
    println!("Median: {median:.1} ms");
    println!("Min:    {min:.1} ms");
    println!("Max:    {max:.1} ms");
    println!("FPS:    {:.1}", 1000.0 / mean);

    Ok(())
}
