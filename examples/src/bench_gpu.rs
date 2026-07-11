//! Quick GPU inference benchmark for YOLO models.
//!
//! Usage:
//!   cargo run --release --example bench_gpu --features gpu -- <model.onnx>

use rustc_hash::FxHashMap;
use yscv_kernels::GpuBackend;
use yscv_onnx::{
    GpuWeightCache, load_onnx_model_from_file, plan_gpu_execution, profile_onnx_model_gpu,
    run_onnx_model_gpu_cached, run_onnx_model_gpu_with,
};
use yscv_tensor::Tensor;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: bench_gpu <model.onnx> [--profile]");
        std::process::exit(1);
    }

    let do_profile = args.iter().any(|a| a == "--profile");
    let model_path = &args[1];
    eprintln!("Loading model: {model_path}");
    let model = load_onnx_model_from_file(model_path).expect("Failed to load ONNX model");
    eprintln!("  Nodes: {}", model.nodes.len());

    let gpu = GpuBackend::new().expect("GPU init failed");
    let mut wc = GpuWeightCache::new();

    let input_data = vec![0.5f32; 3 * 640 * 640];
    let input_tensor = Tensor::from_vec(vec![1, 3, 640, 640], input_data).unwrap();

    // Precompute execution plan once
    let exec_plan = plan_gpu_execution(&model);

    // Print fusion stats
    {
        let mut counts: FxHashMap<String, usize> = FxHashMap::default();
        for a in &exec_plan.actions {
            let key = match a {
                yscv_onnx::GpuExecAction::ConvSiLU { .. } => "ConvSiLU",
                yscv_onnx::GpuExecAction::SiLU { .. } => "SiLU",
                yscv_onnx::GpuExecAction::ConvBnRelu { .. } => "ConvBnRelu",
                yscv_onnx::GpuExecAction::OpRelu { .. } => "OpRelu",
                yscv_onnx::GpuExecAction::MatMulAdd { .. } => "MatMulAdd",
                yscv_onnx::GpuExecAction::Normal => "Normal",
                yscv_onnx::GpuExecAction::Skip => "Skip",
            };
            *counts.entry(key.to_string()).or_default() += 1;
        }
        eprintln!("Fusion plan:");
        let mut sorted: Vec<_> = counts.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));
        for (k, v) in &sorted {
            eprintln!("  {}: {}", k, v);
        }
    }

    // Warm-up
    eprintln!("Warm-up...");
    for _ in 0..3 {
        let mut inputs = FxHashMap::default();
        inputs.insert("images".to_string(), input_tensor.clone());
        let _ = run_onnx_model_gpu_cached(&gpu, &model, inputs, &mut wc, Some(&exec_plan))
            .expect("fail");
    }

    // Benchmark: cached path
    let n_runs = 10;
    eprintln!("\nBenchmark (cached):");
    let mut times = Vec::new();
    for i in 0..n_runs {
        let mut inputs = FxHashMap::default();
        inputs.insert("images".to_string(), input_tensor.clone());
        let t0 = std::time::Instant::now();
        let _ = run_onnx_model_gpu_cached(&gpu, &model, inputs, &mut wc, Some(&exec_plan))
            .expect("fail");
        let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
        times.push(elapsed);
        eprintln!("  Run {}: {:.1}ms", i + 1, elapsed);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eprintln!(
        "  Min: {:.1}ms  Median: {:.1}ms  Avg: {:.1}ms",
        times[0],
        times[n_runs / 2],
        times.iter().sum::<f64>() / n_runs as f64
    );

    // Benchmark: uncached (new gc each time) for comparison
    eprintln!("\nBenchmark (uncached, reusing GpuBackend):");
    let mut times2 = Vec::new();
    for i in 0..5 {
        let mut inputs = FxHashMap::default();
        inputs.insert("images".to_string(), input_tensor.clone());
        let t0 = std::time::Instant::now();
        let _ = run_onnx_model_gpu_with(&gpu, &model, inputs).expect("fail");
        let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
        times2.push(elapsed);
        eprintln!("  Run {}: {:.1}ms", i + 1, elapsed);
    }
    times2.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eprintln!("  Min: {:.1}ms  Median: {:.1}ms", times2[0], times2[2]);

    eprintln!("\nPool hits: {}", gpu.pool_cache_hits());

    if do_profile {
        eprintln!("\nProfiling (sync per op)...");
        let mut inputs = FxHashMap::default();
        inputs.insert("images".to_string(), input_tensor.clone());
        let _ = profile_onnx_model_gpu(&model, inputs).expect("profile failed");
    }
}
