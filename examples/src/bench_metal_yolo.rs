//! Benchmark: Metal-native ONNX inference for YOLO models.
//! Compares Metal vs CPU (and GPU/wgpu if enabled).

#[cfg(not(all(target_os = "macos", feature = "metal-backend")))]
fn main() {
    eprintln!("bench_metal_yolo requires macOS with the metal-backend feature");
}

#[cfg(all(target_os = "macos", feature = "metal-backend"))]
use yscv_onnx::load_onnx_model_from_file;
#[cfg(all(target_os = "macos", feature = "metal-backend"))]
use yscv_tensor::Tensor;

#[cfg(all(target_os = "macos", feature = "metal-backend"))]
fn main() {
    let models = [
        "examples/src/slowwork/yolov8n.onnx",
        "examples/src/slowwork/yolo11n.onnx",
    ];

    let input_data = vec![0.5f32; 3 * 640 * 640];
    let input_tensor = Tensor::from_vec(vec![1, 3, 640, 640], input_data.clone()).unwrap();

    for model_path in &models {
        println!("\n{} {} {}", "=".repeat(20), model_path, "=".repeat(20));

        let model = match load_onnx_model_from_file(model_path) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("  Failed to load: {e}");
                continue;
            }
        };
        println!("  Nodes: {}", model.nodes.len());

        // ── CPU baseline ──
        let cpu_out = {
            let mut inp = std::collections::HashMap::new();
            inp.insert("images".to_string(), input_tensor.clone());
            yscv_onnx::run_onnx_model(&model, inp).expect("cpu fail")
        };
        for (name, t) in &cpu_out {
            println!("  CPU output '{}': {:?}", name, t.shape());
        }

        // (CPU per-layer check skipped — no easy intermediate access)

        // ── Metal compile ──
        println!("\n  [Metal] Compiling...");
        let compile_start = std::time::Instant::now();
        let metal_plan = yscv_onnx::compile_metal_plan(&model, "images", &input_tensor)
            .expect("Metal compile failed");
        let compile_ms = compile_start.elapsed().as_secs_f64() * 1000.0;
        println!(
            "  Compiled in {:.1}ms  ({} ops)",
            compile_ms,
            metal_plan.ops_count()
        );
        metal_plan.dump_op_stats();

        // ── Metal warmup (10 runs to stabilize GPU clocks) ──
        for _ in 0..10 {
            let _ = yscv_onnx::run_metal_plan(&metal_plan, &input_data);
        }

        // ── Metal benchmark ──
        let n_runs = 50;
        let mut times = Vec::new();
        for i in 0..n_runs {
            let start = std::time::Instant::now();
            let result = yscv_onnx::run_metal_plan(&metal_plan, &input_data);
            let elapsed = start.elapsed();
            times.push(elapsed);
            if i == 0 {
                match &result {
                    Ok(out) => {
                        for (name, t) in out {
                            println!("  Metal output '{}': {:?}", name, t.shape());
                        }
                    }
                    Err(e) => {
                        println!("  Metal run FAILED: {}", e);
                        break;
                    }
                }
            }
        }

        if !times.is_empty() {
            let avg_ms =
                times.iter().map(|t| t.as_secs_f64()).sum::<f64>() / times.len() as f64 * 1000.0;
            let min_ms = times
                .iter()
                .map(|t| t.as_secs_f64())
                .fold(f64::INFINITY, f64::min)
                * 1000.0;
            let max_ms = times.iter().map(|t| t.as_secs_f64()).fold(0.0f64, f64::max) * 1000.0;
            println!(
                "\n  Metal ({} runs): avg={:.1}ms  min={:.1}ms  max={:.1}ms",
                n_runs, avg_ms, min_ms, max_ms
            );
        }

        // ── Verify vs CPU ──
        let metal_out =
            yscv_onnx::run_metal_plan(&metal_plan, &input_data).expect("metal verify fail");

        // Show first few values for debugging
        for (name, cpu_t) in &cpu_out {
            let cd = cpu_t.data();
            print!("  CPU  first 8: ");
            for i in 0..8.min(cd.len()) {
                print!("{:.4} ", cd[i]);
            }
            println!();
            if let Some(mt) = metal_out.get(name) {
                let md = mt.data();
                print!("  Metal first 8: ");
                for i in 0..8.min(md.len()) {
                    print!("{:.4} ", md[i]);
                }
                println!();
            }
        }

        for (name, cpu_t) in &cpu_out {
            if let Some(mtl_t) = metal_out.get(name) {
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
                    // Find position of max diff
                    let (max_idx, _) = cpu_d
                        .iter()
                        .zip(mtl_d.iter())
                        .enumerate()
                        .map(|(i, (a, b))| (i, (a - b).abs()))
                        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap_or((0, 0.0));
                    let shape = cpu_t.shape();
                    // Decode flat index to multi-dim
                    let mut pos = vec![0usize; shape.len()];
                    let mut rem = max_idx;
                    for d in (0..shape.len()).rev() {
                        pos[d] = rem % shape[d];
                        rem /= shape[d];
                    }
                    println!(
                        "  CPU vs Metal '{}': max_diff={:.6} mean_diff={:.8} at {:?} cpu={:.4} metal={:.4}",
                        name, max_diff, mean_diff, pos, cpu_d[max_idx], mtl_d[max_idx]
                    );
                    // Check for inf/NaN
                    let mtl_inf = mtl_d.iter().filter(|x| x.is_infinite()).count();
                    let mtl_nan = mtl_d.iter().filter(|x| x.is_nan()).count();
                    if mtl_inf + mtl_nan > 0 {
                        println!(
                            "  WARNING: metal has {} inf, {} nan values",
                            mtl_inf, mtl_nan
                        );
                    }
                    // Per-row error analysis for [1, 84, 8400] shaped output
                    if shape.len() == 3 && shape[0] == 1 {
                        let rows = shape[1];
                        let cols = shape[2];
                        println!("  Per-row (dim1) error analysis:");
                        for r in 0..rows.min(10) {
                            let start = r * cols;
                            let end = start + cols;
                            let row_cpu = &cpu_d[start..end];
                            let row_mtl = &mtl_d[start..end];
                            let row_max_diff = row_cpu
                                .iter()
                                .zip(row_mtl.iter())
                                .map(|(a, b)| (a - b).abs())
                                .fold(0.0f32, f32::max);
                            let row_mean_diff = row_cpu
                                .iter()
                                .zip(row_mtl.iter())
                                .map(|(a, b)| (a - b).abs())
                                .sum::<f32>()
                                / cols as f32;
                            println!(
                                "    row {:2}: max_diff={:.4} mean_diff={:.6} cpu[0..3]=[{:.2},{:.2},{:.2}] mtl[0..3]=[{:.2},{:.2},{:.2}]",
                                r,
                                row_max_diff,
                                row_mean_diff,
                                row_cpu[0],
                                row_cpu[1],
                                row_cpu[2],
                                row_mtl[0],
                                row_mtl[1],
                                row_mtl[2]
                            );
                        }
                    }
                } else {
                    println!(
                        "  CPU vs Metal '{}': SIZE MISMATCH cpu={} metal={}",
                        name,
                        cpu_d.len(),
                        mtl_d.len()
                    );
                }
            } else {
                println!("  Metal missing output '{}'", name);
            }
        }
    }
}
