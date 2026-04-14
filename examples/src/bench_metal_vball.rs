//! Benchmark: Metal-native GPU inference for VballNetGrid.
//!
//! Usage:
//!   cargo run --release --example bench_metal_vball --features metal-backend -- /path/to/model.onnx

use std::collections::HashMap;
use std::time::Instant;

use yscv_onnx::{compile_metal_plan, load_onnx_model_from_file, run_metal_plan, run_onnx_model};
use yscv_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("Usage: bench_metal_vball <model.onnx> [iterations]");
        std::process::exit(1);
    }
    let model_path = &args[0];
    let iterations: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(50);

    println!("Loading model: {model_path}");
    let model = load_onnx_model_from_file(model_path)?;
    let input_name = model.inputs.first().cloned().unwrap();
    let output_name = model.outputs.first().cloned().unwrap();
    println!("Input: {input_name}, Output: {output_name}");
    println!("Nodes: {}", model.nodes.len());

    // VballNetGridV1b: input [1, 9, 432, 768]
    let input_data = vec![0.0f32; 9 * 432 * 768];
    let input_tensor = Tensor::from_vec(vec![1, 9, 432, 768], input_data.clone())?;

    // ── CPU baseline ──
    println!("\n--- CPU Inference ---");
    {
        let mut inputs = HashMap::new();
        inputs.insert(input_name.clone(), input_tensor.clone());
        let _ = run_onnx_model(&model, inputs)?; // warmup
    }
    let n_cpu = 10;
    let mut cpu_times = Vec::new();
    for _ in 0..n_cpu {
        let mut inputs = HashMap::new();
        inputs.insert(input_name.clone(), input_tensor.clone());
        let t0 = Instant::now();
        let _ = run_onnx_model(&model, inputs)?;
        cpu_times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    cpu_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let cpu_median = cpu_times[cpu_times.len() / 2];
    println!("CPU median: {cpu_median:.1} ms ({n_cpu} runs)");

    // ── Metal compile ──
    println!("\n--- Metal GPU Inference ---");
    let compile_start = Instant::now();
    let metal_plan = compile_metal_plan(&model, &input_name, &input_tensor)?;
    let compile_ms = compile_start.elapsed().as_secs_f64() * 1000.0;
    println!(
        "Compiled in {compile_ms:.1} ms ({} ops)",
        metal_plan.ops_count()
    );
    metal_plan.dump_op_stats();

    // ── Metal warmup ──
    for _ in 0..10 {
        let _ = run_metal_plan(&metal_plan, &input_data);
    }

    // ── Metal benchmark ──
    let mut metal_times = Vec::new();
    for i in 0..iterations {
        let t0 = Instant::now();
        let result = run_metal_plan(&metal_plan, &input_data);
        let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
        metal_times.push(elapsed);
        if i == 0 {
            match &result {
                Ok(out) => {
                    for (name, t) in out {
                        println!("Metal output '{name}': {:?}", t.shape());
                    }
                }
                Err(e) => {
                    eprintln!("Metal run FAILED: {e}");
                    std::process::exit(1);
                }
            }
        }
    }

    metal_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let metal_median = metal_times[metal_times.len() / 2];
    let metal_min = metal_times[0];
    let metal_max = metal_times[metal_times.len() - 1];
    let metal_mean = metal_times.iter().sum::<f64>() / metal_times.len() as f64;

    println!("\nMetal ({iterations} runs):");
    println!("  Mean:   {metal_mean:.1} ms");
    println!("  Median: {metal_median:.1} ms");
    println!("  Min:    {metal_min:.1} ms");
    println!("  Max:    {metal_max:.1} ms");
    println!("  FPS:    {:.1}", 1000.0 / metal_median);

    // ── Verify vs CPU ──
    println!("\n--- Accuracy Check ---");
    let metal_out = run_metal_plan(&metal_plan, &input_data)?;
    let mut cpu_inputs = HashMap::new();
    cpu_inputs.insert(input_name.clone(), input_tensor.clone());
    let cpu_out = run_onnx_model(&model, cpu_inputs)?;

    if let (Some(cpu_t), Some(mtl_t)) = (cpu_out.get(&output_name), metal_out.get(&output_name)) {
        let cpu_d = cpu_t.data();
        let mtl_d = mtl_t.data();
        if cpu_d.len() == mtl_d.len() {
            let max_diff = cpu_d
                .iter()
                .zip(mtl_d.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let mean_diff = cpu_d
                .iter()
                .zip(mtl_d.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
                / cpu_d.len() as f32;
            println!("Max diff:  {max_diff:.6}");
            println!("Mean diff: {mean_diff:.6}");
            if max_diff < 0.05 {
                println!("PASS: Metal output matches CPU within f16 tolerance");
            } else {
                println!("WARNING: Large divergence between Metal and CPU");
            }
        } else {
            println!(
                "Shape mismatch: CPU {:?} vs Metal {:?}",
                cpu_t.shape(),
                mtl_t.shape()
            );
        }
    }

    println!("\n--- Summary ---");
    println!(
        "CPU:   {cpu_median:.1} ms  ({:.1} FPS)",
        1000.0 / cpu_median
    );
    println!(
        "Metal: {metal_median:.1} ms  ({:.1} FPS)",
        1000.0 / metal_median
    );
    println!("Speedup: {:.1}x", cpu_median / metal_median);

    Ok(())
}
