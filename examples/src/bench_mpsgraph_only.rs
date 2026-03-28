//! MPSGraph-only benchmark — no CPU/Metal to avoid thermal throttling on M1 MBA.
//!
//! Usage:
//!   cargo run --release --example bench_mpsgraph_only --features metal-backend -- model.onnx [iters]

use std::time::Instant;

use yscv_onnx::{compile_mpsgraph_plan, load_onnx_model_from_file, run_mpsgraph_plan};
use yscv_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args
        .get(1)
        .expect("usage: bench_mpsgraph_only <model.onnx> [iters]");
    let iters: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(30);

    let model = load_onnx_model_from_file(model_path)?;
    let input_name = model.inputs.first().cloned().unwrap();

    let input_shape: Vec<usize> = if model_path.contains("vball") {
        vec![1, 9, 432, 768]
    } else {
        vec![1, 3, 640, 640]
    };
    println!(
        "{model_path}: input {:?}, {iters} iters (MPSGraph only)",
        input_shape
    );

    let n: usize = input_shape.iter().product();
    let input_data = vec![0.5f32; n];
    let input_tensor = Tensor::from_vec(input_shape, input_data.clone())?;

    let plan = compile_mpsgraph_plan(&model, &input_name, &input_tensor)?;

    // warmup
    for _ in 0..5 {
        let _ = run_mpsgraph_plan(&plan, &input_data);
    }

    let mut times = Vec::new();
    for _ in 0..iters {
        let t0 = Instant::now();
        let _ = run_mpsgraph_plan(&plan, &input_data);
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = times[0];
    let median = times[times.len() / 2];
    let p95 = times[(times.len() as f64 * 0.95) as usize];
    println!("  min={min:.1}ms  median={median:.1}ms  p95={p95:.1}ms");
    Ok(())
}
