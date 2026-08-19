//! Benchmark competitor ONNX runtimes (tract, candle) against yscv.
//! Outputs JSON timing results for automated comparison.

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::time::Instant;

fn main() {
    let models = [
        ("yolov8n", "../../examples/src/slowwork/yolov8n.onnx"),
        ("yolo11n", "../../examples/src/slowwork/yolo11n.onnx"),
    ];

    let warmup = 3;
    let runs = 20;

    println!("{{");
    println!("  \"platform\": \"{}-{}\",", std::env::consts::OS, std::env::consts::ARCH);

    #[cfg(feature = "tract")]
    {
        println!("  \"tract\": {{");
        for (i, (name, path)) in models.iter().enumerate() {
            print!("    \"{}\": ", name);
            match bench_tract(path, warmup, runs) {
                Ok(result) => print!("{}", result),
                Err(e) => print!("{{\"error\": \"{}\"}}", e.to_string().replace('"', "'")),
            }
            if i + 1 < models.len() {
                println!(",");
            } else {
                println!();
            }
        }
        println!("  }}");
    }

    println!("}}");
}

#[cfg(feature = "tract")]
fn bench_tract(model_path: &str, warmup: usize, runs: usize) -> Result<String, Box<dyn std::error::Error>> {
    use tract_onnx::prelude::*;

    let model = tract_onnx::onnx()
        .model_for_path(model_path)?
        .with_input_fact(0, f32::fact([1, 3, 640, 640]).into())?
        .into_optimized()?
        .into_runnable()?;

    let input: tract_onnx::prelude::TValue = tract_ndarray::Array4::<f32>::from_elem((1, 3, 640, 640), 0.5).into_tensor().into();

    // Warmup
    for _ in 0..warmup {
        let _ = model.run(tvec![input.clone()])?;
    }

    // Benchmark
    let mut times_ms = Vec::with_capacity(runs);
    let mut output_shapes = Vec::new();
    for i in 0..runs {
        let start = Instant::now();
        let result = model.run(tvec![input.clone()])?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        times_ms.push(elapsed_ms);

        if i == 0 {
            for t in result.iter() {
                output_shapes.push(format!("{:?}", t.shape()));
            }
        }
    }

    let min = times_ms.iter().cloned().fold(f64::INFINITY, f64::min);
    let avg = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
    let median = {
        let mut sorted = times_ms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[sorted.len() / 2]
    };

    Ok(format!(
        "{{\"min_ms\": {:.1}, \"avg_ms\": {:.1}, \"median_ms\": {:.1}, \"runs\": {}, \"shapes\": {:?}}}",
        min, avg, median, runs, output_shapes
    ))
}
