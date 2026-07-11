//! Quick benchmark: measure YOLO ONNX inference time (CPU vs GPU).
//!
//! Set BENCH_COOLDOWN=<secs> to insert a cooldown pause between benchmarks
//! (default: 20s). This prevents CPU thermal throttling from skewing results.

use rustc_hash::FxHashMap;
use yscv_onnx::load_onnx_model_from_file;
use yscv_tensor::Tensor;

fn cooldown(label: &str) {
    let secs: u64 = std::env::var("BENCH_COOLDOWN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    if secs > 0 {
        println!("\n  --- cooldown {secs}s before {label} ---");
        std::thread::sleep(std::time::Duration::from_secs(secs));
    }
}

fn main() {
    let models = [
        "examples/src/slowwork/yolov8n.onnx",
        "examples/src/slowwork/yolo11n.onnx",
    ];

    let input_data = vec![0.5f32; 3 * 640 * 640];
    let input_tensor = Tensor::from_vec(vec![1, 3, 640, 640], input_data).unwrap();

    for (model_idx, model_path) in models.iter().enumerate() {
        if model_idx > 0 {
            cooldown("next model");
        }
        println!("\n{} {} {}", "=".repeat(20), model_path, "=".repeat(20));

        let model = match load_onnx_model_from_file(model_path) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("  Failed to load: {e}");
                continue;
            }
        };
        println!("  Nodes: {}", model.nodes.len());

        // ── CPU benchmark ───────────────────────────────────────────
        println!("\n  [CPU]");
        #[allow(unused_variables)]
        let cpu_out = {
            let mut inp = FxHashMap::default();
            inp.insert("images".to_string(), input_tensor.clone());
            yscv_onnx::run_onnx_model(&model, inp).expect("cpu fail")
        };
        bench_run(&model, &input_tensor, |m, inp| {
            yscv_onnx::run_onnx_model(m, inp)
        });

        {
            println!("\n  [CPU Profile]");
            let mut inp = FxHashMap::default();
            inp.insert("images".to_string(), input_tensor.clone());
            let _ = yscv_onnx::profile_onnx_model_cpu(&model, inp);
        }

        // ── GPU profile ──────────────────────────────────────────
        #[cfg(feature = "gpu")]
        {
            cooldown("GPU profile");
            let mut inp = FxHashMap::default();
            inp.insert("images".to_string(), input_tensor.clone());
            let _ = yscv_onnx::profile_onnx_model_gpu(&model, inp);
        }

        // ── GPU benchmark ──────────────────────────────────────────
        #[cfg(feature = "gpu")]
        {
            cooldown("GPU benchmark");
            println!("\n  [GPU]");
            let gpu = yscv_kernels::GpuBackend::new().expect("GPU init");
            println!("    Adapter: {}", gpu.adapter_name());
            let mut wc = yscv_onnx::GpuWeightCache::new();
            let plan = yscv_onnx::plan_gpu_execution(&model);

            // Warmup (populates weight cache)
            let mut inputs = FxHashMap::default();
            inputs.insert("images".to_string(), input_tensor.clone());
            let _ =
                yscv_onnx::run_onnx_model_gpu_cached(&gpu, &model, inputs, &mut wc, Some(&plan));
            let dispatches = gpu.take_dispatch_count();
            println!("    Dispatches: {}", dispatches);

            let n = 5;
            let mut times = Vec::new();
            for i in 0..n {
                let mut inputs = FxHashMap::default();
                inputs.insert("images".to_string(), input_tensor.clone());
                let start = std::time::Instant::now();
                let result = yscv_onnx::run_onnx_model_gpu_cached(
                    &gpu,
                    &model,
                    inputs,
                    &mut wc,
                    Some(&plan),
                );
                let elapsed = start.elapsed();
                times.push(elapsed);
                match &result {
                    Ok(out) => {
                        let shape: Vec<_> = out.values().map(|t| t.shape().to_vec()).collect();
                        println!(
                            "    Run {}: {:.1}ms {:?}",
                            i + 1,
                            elapsed.as_secs_f64() * 1000.0,
                            shape
                        );
                    }
                    Err(e) => {
                        println!("    Run {}: FAILED - {}", i + 1, e);
                        break;
                    }
                }
            }
            if !times.is_empty() {
                let avg_ms = times.iter().map(|t| t.as_secs_f64()).sum::<f64>()
                    / times.len() as f64
                    * 1000.0;
                let min_ms = times
                    .iter()
                    .map(|t| t.as_secs_f64())
                    .fold(f64::INFINITY, f64::min)
                    * 1000.0;
                println!("    Avg: {:.1}ms  Min: {:.1}ms", avg_ms, min_ms);
            }

            // ── Compiled plan benchmark ──
            {
                cooldown("GPU Compiled");
                println!("\n  [GPU Compiled]");
                let compiled = yscv_onnx::compile_gpu_plan(
                    &gpu,
                    &model,
                    &plan,
                    &mut wc,
                    "images",
                    &input_tensor,
                )
                .expect("compile failed");
                println!("    Recorded ops: {}", compiled.ops_count());

                let input_data = input_tensor.data();
                let _ = yscv_onnx::run_compiled_gpu(&gpu, &compiled, input_data);

                let n = 10;
                let mut times = Vec::new();
                for i in 0..n {
                    let start = std::time::Instant::now();
                    let result = yscv_onnx::run_compiled_gpu(&gpu, &compiled, input_data);
                    let elapsed = start.elapsed();
                    times.push(elapsed);
                    match &result {
                        Ok(out) => {
                            let shape: Vec<_> = out.values().map(|t| t.shape().to_vec()).collect();
                            println!(
                                "    Run {}: {:.1}ms {:?}",
                                i + 1,
                                elapsed.as_secs_f64() * 1000.0,
                                shape
                            );
                        }
                        Err(e) => {
                            println!("    Run {}: FAILED - {}", i + 1, e);
                            break;
                        }
                    }
                }
                if !times.is_empty() {
                    let avg_ms = times.iter().map(|t| t.as_secs_f64()).sum::<f64>()
                        / times.len() as f64
                        * 1000.0;
                    let min_ms = times
                        .iter()
                        .map(|t| t.as_secs_f64())
                        .fold(f64::INFINITY, f64::min)
                        * 1000.0;
                    println!("    Avg: {:.1}ms  Min: {:.1}ms", avg_ms, min_ms);
                }

                // Verify compiled vs CPU
                let compiled_out = yscv_onnx::run_compiled_gpu(&gpu, &compiled, input_data)
                    .expect("compiled verify fail");
                for (name, cpu_t) in &cpu_out {
                    if let Some(comp_t) = compiled_out.get(name) {
                        let cpu_d = cpu_t.data();
                        let comp_d = comp_t.data();
                        if cpu_d.len() == comp_d.len() {
                            let max_diff = cpu_d
                                .iter()
                                .zip(comp_d.iter())
                                .map(|(a, b)| (a - b).abs())
                                .fold(0.0f32, f32::max);
                            let mean_diff = cpu_d
                                .iter()
                                .zip(comp_d.iter())
                                .map(|(a, b)| (a - b).abs())
                                .sum::<f32>()
                                / cpu_d.len() as f32;
                            println!(
                                "    CPU vs Compiled {}: max_diff={:.6} mean_diff={:.6}",
                                name, max_diff, mean_diff
                            );
                        }
                    }
                }
            }

            // ── Fused single-pass benchmark (Metal-only) ──
            {
                cooldown("GPU Fused");
                println!("\n  [GPU Fused Single-Pass]");
                let compiled = yscv_onnx::compile_gpu_plan(
                    &gpu,
                    &model,
                    &plan,
                    &mut wc,
                    "images",
                    &input_tensor,
                )
                .expect("compile failed");

                let input_data = input_tensor.data();
                let _ = yscv_onnx::run_compiled_gpu_fused(&gpu, &compiled, input_data);

                let n = 10;
                let mut times = Vec::new();
                for i in 0..n {
                    let start = std::time::Instant::now();
                    let result = yscv_onnx::run_compiled_gpu_fused(&gpu, &compiled, input_data);
                    let elapsed = start.elapsed();
                    times.push(elapsed);
                    match &result {
                        Ok(out) => {
                            let shape: Vec<_> = out.values().map(|t| t.shape().to_vec()).collect();
                            println!(
                                "    Run {}: {:.1}ms {:?}",
                                i + 1,
                                elapsed.as_secs_f64() * 1000.0,
                                shape
                            );
                        }
                        Err(e) => {
                            println!("    Run {}: FAILED - {}", i + 1, e);
                            break;
                        }
                    }
                }
                if !times.is_empty() {
                    let avg_ms = times.iter().map(|t| t.as_secs_f64()).sum::<f64>()
                        / times.len() as f64
                        * 1000.0;
                    let min_ms = times
                        .iter()
                        .map(|t| t.as_secs_f64())
                        .fold(f64::INFINITY, f64::min)
                        * 1000.0;
                    println!("    Avg: {:.1}ms  Min: {:.1}ms", avg_ms, min_ms);
                }

                // Timing breakdown
                println!("    --- Timing breakdown (3 runs) ---");
                for i in 0..3 {
                    match yscv_onnx::run_compiled_gpu_fused_timed(&gpu, &compiled, input_data) {
                        Ok((_out, upload, encode, gpu_t, download)) => {
                            println!(
                                "    Run {}: upload={:.2}ms encode={:.2}ms gpu={:.2}ms download={:.2}ms total={:.2}ms",
                                i + 1,
                                upload,
                                encode,
                                gpu_t,
                                download,
                                upload + encode + gpu_t + download
                            );
                        }
                        Err(e) => println!("    Timed run {}: FAILED - {}", i + 1, e),
                    }
                }

                // Verify fused vs CPU
                let fused_out = yscv_onnx::run_compiled_gpu_fused(&gpu, &compiled, input_data)
                    .expect("fused verify fail");
                for (name, cpu_t) in &cpu_out {
                    if let Some(fused_t) = fused_out.get(name) {
                        let cpu_d = cpu_t.data();
                        let fused_d = fused_t.data();
                        if cpu_d.len() == fused_d.len() {
                            let max_diff = cpu_d
                                .iter()
                                .zip(fused_d.iter())
                                .map(|(a, b)| (a - b).abs())
                                .fold(0.0f32, f32::max);
                            let mean_diff = cpu_d
                                .iter()
                                .zip(fused_d.iter())
                                .map(|(a, b)| (a - b).abs())
                                .sum::<f32>()
                                / cpu_d.len() as f32;
                            println!(
                                "    CPU vs Fused {}: max_diff={:.6} mean_diff={:.6}",
                                name, max_diff, mean_diff
                            );
                        }
                    }
                }
            }

            // ── f16 I/O fused benchmark ──
            if gpu.has_f16_io() {
                cooldown("GPU f16");
                println!("\n  [GPU f16 I/O Fused]");
                let compiled = yscv_onnx::compile_gpu_plan_f16(
                    &gpu,
                    &model,
                    &plan,
                    &mut wc,
                    "images",
                    &input_tensor,
                )
                .expect("f16 compile failed");
                println!("    Recorded ops: {}", compiled.ops_count());

                let input_data = input_tensor.data();
                // Warmup
                let _ = yscv_onnx::run_compiled_gpu_f16_fused(&gpu, &compiled, input_data);

                let n = 10;
                let mut times = Vec::new();
                for i in 0..n {
                    let start = std::time::Instant::now();
                    let result = yscv_onnx::run_compiled_gpu_f16_fused(&gpu, &compiled, input_data);
                    let elapsed = start.elapsed();
                    times.push(elapsed);
                    match &result {
                        Ok(out) => {
                            let shape: Vec<_> = out.values().map(|t| t.shape().to_vec()).collect();
                            println!(
                                "    Run {}: {:.1}ms {:?}",
                                i + 1,
                                elapsed.as_secs_f64() * 1000.0,
                                shape
                            );
                        }
                        Err(e) => {
                            println!("    Run {}: FAILED - {}", i + 1, e);
                            break;
                        }
                    }
                }
                if !times.is_empty() {
                    let avg_ms = times.iter().map(|t| t.as_secs_f64()).sum::<f64>()
                        / times.len() as f64
                        * 1000.0;
                    let min_ms = times
                        .iter()
                        .map(|t| t.as_secs_f64())
                        .fold(f64::INFINITY, f64::min)
                        * 1000.0;
                    println!("    Avg: {:.1}ms  Min: {:.1}ms", avg_ms, min_ms);
                }

                // Verify f16 vs CPU
                let f16_out = yscv_onnx::run_compiled_gpu_f16_fused(&gpu, &compiled, input_data)
                    .expect("f16 verify fail");
                for (name, cpu_t) in &cpu_out {
                    if let Some(f16_t) = f16_out.get(name) {
                        let cpu_d = cpu_t.data();
                        let f16_d = f16_t.data();
                        if cpu_d.len() == f16_d.len() {
                            let max_diff = cpu_d
                                .iter()
                                .zip(f16_d.iter())
                                .map(|(a, b)| (a - b).abs())
                                .fold(0.0f32, f32::max);
                            let mean_diff = cpu_d
                                .iter()
                                .zip(f16_d.iter())
                                .map(|(a, b)| (a - b).abs())
                                .sum::<f32>()
                                / cpu_d.len() as f32;
                            println!(
                                "    CPU vs f16 {}: max_diff={:.6} mean_diff={:.6}",
                                name, max_diff, mean_diff
                            );
                        }
                    }
                }
            }
        }
    }
}

fn bench_run(
    model: &yscv_onnx::OnnxModel,
    input_tensor: &Tensor,
    run_fn: impl Fn(
        &yscv_onnx::OnnxModel,
        FxHashMap<String, Tensor>,
    ) -> Result<FxHashMap<String, Tensor>, yscv_onnx::OnnxError>,
) {
    // Warmup
    let mut inputs = FxHashMap::default();
    inputs.insert("images".to_string(), input_tensor.clone());
    let _ = run_fn(model, inputs);

    let n = 5;
    let mut times = Vec::new();
    for i in 0..n {
        let mut inputs = FxHashMap::default();
        inputs.insert("images".to_string(), input_tensor.clone());
        let start = std::time::Instant::now();
        let result = run_fn(model, inputs);
        let elapsed = start.elapsed();
        times.push(elapsed);
        match &result {
            Ok(out) => {
                let shape: Vec<_> = out.values().map(|t| t.shape().to_vec()).collect();
                println!(
                    "    Run {}: {:.1}ms {:?}",
                    i + 1,
                    elapsed.as_secs_f64() * 1000.0,
                    shape
                );
            }
            Err(e) => {
                println!("    Run {}: FAILED - {}", i + 1, e);
                return;
            }
        }
    }
    let avg_ms = times.iter().map(|t| t.as_secs_f64()).sum::<f64>() / n as f64 * 1000.0;
    let min_ms = times
        .iter()
        .map(|t| t.as_secs_f64())
        .fold(f64::INFINITY, f64::min)
        * 1000.0;
    println!("    Avg: {:.1}ms  Min: {:.1}ms", avg_ms, min_ms);
}
