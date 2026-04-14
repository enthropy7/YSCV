//! Benchmark: MPSGraph whole-model GPU inference.
//!
//! Usage:
//!   cargo run --release --example bench_mpsgraph --features metal-backend -- /path/to/model.onnx [iterations]
//!
//! Also supports YOLO models from slowwork/ when no arg is given.

use std::collections::HashMap;
use std::time::Instant;

use yscv_onnx::{
    compile_metal_plan, compile_mpsgraph_plan, load_onnx_model_from_file, run_metal_plan,
    run_mpsgraph_plan, run_onnx_model,
};
use yscv_tensor::Tensor;

fn bench_model(
    model_path: &str,
    input_shape: Vec<usize>,
    iterations: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{} {} {}", "=".repeat(20), model_path, "=".repeat(20));

    let model = load_onnx_model_from_file(model_path)?;
    let input_name = model.inputs.first().cloned().unwrap();
    let output_name = model.outputs.first().cloned().unwrap();
    println!("Input: {input_name} {:?}", input_shape);
    println!("Nodes: {}", model.nodes.len());

    let n: usize = input_shape.iter().product();
    let input_data = vec![0.5f32; n];
    let input_tensor = Tensor::from_vec(input_shape.clone(), input_data.clone())?;

    // ── CPU baseline (few runs) ──
    println!("\n--- CPU ---");
    {
        let mut inputs = HashMap::new();
        inputs.insert(input_name.clone(), input_tensor.clone());
        let _ = run_onnx_model(&model, inputs)?;
    }
    let n_cpu = 5;
    let mut cpu_times = Vec::new();
    for _ in 0..n_cpu {
        let mut inputs = HashMap::new();
        inputs.insert(input_name.clone(), input_tensor.clone());
        let t0 = Instant::now();
        let _ = run_onnx_model(&model, inputs)?;
        cpu_times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    cpu_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let cpu_min = cpu_times[0];
    println!("CPU min: {cpu_min:.1} ms ({n_cpu} runs)");

    // ── Metal (per-op dispatch) ──
    println!("\n--- Metal (per-op) ---");
    let metal_plan = compile_metal_plan(&model, &input_name, &input_tensor)?;
    println!("Compiled: {} ops", metal_plan.ops_count());
    metal_plan.dump_op_stats();

    for _ in 0..10 {
        let _ = run_metal_plan(&metal_plan, &input_data);
    }
    let mut metal_times = Vec::new();
    for _ in 0..iterations {
        let t0 = Instant::now();
        let _ = run_metal_plan(&metal_plan, &input_data);
        metal_times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    metal_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if metal_times.is_empty() {
        println!("Metal per-op: (no iterations)");
        return Ok(());
    }
    let metal_min = metal_times[0];
    let metal_median = metal_times[metal_times.len() / 2];
    println!("Metal per-op: min={metal_min:.1} ms, median={metal_median:.1} ms");

    // ── MPSGraph (whole-model) ──
    println!("\n--- MPSGraph (whole-model) ---");
    let compile_start = Instant::now();
    let mpsg_plan = match compile_mpsgraph_plan(&model, &[(input_name.as_str(), &input_tensor)]) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("MPSGraph compile FAILED: {e}");
            return Ok(());
        }
    };
    let compile_ms = compile_start.elapsed().as_secs_f64() * 1000.0;
    println!("Compiled in {compile_ms:.1} ms");

    let mpsg_inputs: [(&str, &[f32]); 1] = [(input_name.as_str(), input_data.as_slice())];

    // warmup
    for _ in 0..10 {
        let _ = run_mpsgraph_plan(&mpsg_plan, &mpsg_inputs);
    }

    let mut mpsg_times = Vec::new();
    for i in 0..iterations {
        let t0 = Instant::now();
        let result = run_mpsgraph_plan(&mpsg_plan, &mpsg_inputs);
        let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
        mpsg_times.push(elapsed);
        if i == 0 {
            match &result {
                Ok(out) => {
                    for (name, t) in out {
                        println!("  Output '{name}': {:?}", t.shape());
                    }
                }
                Err(e) => {
                    eprintln!("MPSGraph run FAILED: {e}");
                    return Ok(());
                }
            }
        }
    }
    mpsg_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mpsg_min = mpsg_times[0];
    let mpsg_median = mpsg_times[mpsg_times.len() / 2];
    println!("MPSGraph: min={mpsg_min:.1} ms, median={mpsg_median:.1} ms");

    // ── Accuracy check: MPSGraph vs CPU ──
    println!("\n--- Accuracy (MPSGraph vs CPU) ---");
    let mpsg_out = run_mpsgraph_plan(&mpsg_plan, &mpsg_inputs)?;
    let mut cpu_inputs = HashMap::new();
    cpu_inputs.insert(input_name.clone(), input_tensor.clone());
    let cpu_out = run_onnx_model(&model, cpu_inputs)?;

    if let (Some(cpu_t), Some(mpsg_t)) = (cpu_out.get(&output_name), mpsg_out.get(&output_name)) {
        let cpu_d = cpu_t.data();
        let mpsg_d = mpsg_t.data();
        if cpu_d.len() == mpsg_d.len() {
            let max_diff = cpu_d
                .iter()
                .zip(mpsg_d.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let mean_diff = cpu_d
                .iter()
                .zip(mpsg_d.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
                / cpu_d.len() as f32;
            println!("Max diff:  {max_diff:.6}");
            println!("Mean diff: {mean_diff:.6}");
            if max_diff < 0.1 {
                println!("PASS");
            } else {
                println!("WARNING: large divergence");
            }
        } else {
            println!(
                "Shape mismatch: CPU {:?} vs MPSGraph {:?}",
                cpu_t.shape(),
                mpsg_t.shape()
            );
        }
    }

    // ── Summary ──
    println!("\n--- Summary ---");
    println!("CPU:          {cpu_min:.1} ms",);
    println!(
        "Metal per-op: {metal_min:.1} ms  ({:.1}x vs CPU)",
        cpu_min / metal_min
    );
    println!(
        "MPSGraph:     {mpsg_min:.1} ms  ({:.1}x vs CPU)",
        cpu_min / mpsg_min
    );
    println!("MPSGraph vs Metal: {:.1}x", metal_min / mpsg_min);

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if !args.is_empty() {
        let model_path = &args[0];
        let iterations: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(50);
        // Auto-detect input shape from model
        let model = load_onnx_model_from_file(model_path)?;
        let _input_name = model.inputs.first().cloned().unwrap();
        let shape = model
            .initializers
            .values()
            .next()
            .map(|_| {
                // Try to get shape from first Conv weight to guess input
                // Default to common shapes
                vec![1, 3, 640, 640]
            })
            .unwrap_or(vec![1, 3, 640, 640]);
        drop(model);

        // Let user override via model name detection
        let shape = if model_path.contains("vball") {
            vec![1, 9, 432, 768]
        } else {
            shape
        };

        bench_model(model_path, shape, iterations)?;
    } else {
        // Run all known models
        let models: Vec<(&str, Vec<usize>)> = vec![
            ("/tmp/vball_model.onnx", vec![1, 9, 432, 768]),
            ("examples/src/slowwork/yolov8n.onnx", vec![1, 3, 640, 640]),
            ("examples/src/slowwork/yolo11n.onnx", vec![1, 3, 640, 640]),
        ];
        for (path, shape) in &models {
            if std::path::Path::new(path).exists() {
                if let Err(e) = bench_model(path, shape.clone(), 30) {
                    eprintln!("  Error: {e}");
                }
            } else {
                println!("\nSkipping {path} (not found)");
            }
        }
    }

    Ok(())
}
