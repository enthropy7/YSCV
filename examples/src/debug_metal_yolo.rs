//! Diagnostic: find where Metal per-op diverges from CPU for YOLO11n.
//! Compare intermediate DFL head values.

#[cfg(not(all(target_os = "macos", feature = "metal-backend")))]
fn main() {
    eprintln!("debug_metal_yolo requires macOS with the metal-backend feature");
}

#[cfg(all(target_os = "macos", feature = "metal-backend"))]
use yscv_onnx::load_onnx_model_from_file;
#[cfg(all(target_os = "macos", feature = "metal-backend"))]
use yscv_tensor::Tensor;

#[cfg(all(target_os = "macos", feature = "metal-backend"))]
fn main() {
    // Use uniform input for simplicity
    let model_path = "examples/src/slowwork/yolov8n.onnx";
    let model = load_onnx_model_from_file(model_path).expect("load");
    println!("Model: {} ({} nodes)", model_path, model.nodes.len());

    let input_data = vec![0.5f32; 3 * 640 * 640];
    let input_tensor = Tensor::from_vec(vec![1, 3, 640, 640], input_data.clone()).unwrap();

    // CPU inference
    let mut cpu_inputs = std::collections::HashMap::new();
    cpu_inputs.insert("images".to_string(), input_tensor.clone());
    let cpu_out = yscv_onnx::run_onnx_model(&model, cpu_inputs).expect("cpu");
    let cpu_t = cpu_out.values().next().unwrap();
    println!("CPU output: {:?}", cpu_t.shape());

    // Metal per-op
    let plan =
        yscv_onnx::compile_metal_plan(&model, "images", &input_tensor).expect("metal compile");
    println!("Metal ops: {}", plan.ops_count());

    let metal_out = yscv_onnx::run_metal_plan(&plan, &input_data).expect("metal run");
    let metal_t = metal_out.values().next().unwrap();

    // Compare output
    let cd = cpu_t.data();
    let md = metal_t.data();
    let cols = 8400;

    println!("\nPer-row comparison (first 4 = bbox, rest = class scores):");
    for r in 0..84 {
        let cs = &cd[r * cols..(r + 1) * cols];
        let ms = &md[r * cols..(r + 1) * cols];
        let max_diff = cs
            .iter()
            .zip(ms.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let nans = ms.iter().filter(|x| x.is_nan()).count();
        let sat = ms.iter().filter(|x| x.abs() >= 65500.0).count();
        if max_diff > 1.0 || nans > 0 || r < 5 || r == 83 {
            println!(
                "  row {:2}: max_diff={:10.2} nans={} sat={} cpu[0..3]=[{:.2},{:.2},{:.2}] mtl[0..3]=[{:.2},{:.2},{:.2}]",
                r, max_diff, nans, sat, cs[0], cs[1], cs[2], ms[0], ms[1], ms[2],
            );
        }
    }

    // Check if class scores (rows 4-83) have good values
    let mut cls_max_diff = 0.0f32;
    let mut cls_nans = 0;
    for r in 4..84 {
        let cs = &cd[r * cols..(r + 1) * cols];
        let ms = &md[r * cols..(r + 1) * cols];
        for (&c, &m) in cs.iter().zip(ms.iter()) {
            if m.is_nan() {
                cls_nans += 1;
            }
            let d = (c - m).abs();
            if d > cls_max_diff {
                cls_max_diff = d;
            }
        }
    }
    println!(
        "\nClass scores (rows 4-83): max_diff={:.4} nans={}",
        cls_max_diff, cls_nans
    );

    // Check bbox rows specifically
    println!("\nBbox rows detail:");
    for r in 0..4 {
        let cs = &cd[r * cols..(r + 1) * cols];
        let ms = &md[r * cols..(r + 1) * cols];
        let cpu_min = cs.iter().cloned().fold(f32::INFINITY, f32::min);
        let cpu_max = cs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mtl_min = ms
            .iter()
            .filter(|x| !x.is_nan())
            .cloned()
            .fold(f32::INFINITY, f32::min);
        let mtl_max = ms
            .iter()
            .filter(|x| !x.is_nan())
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let names = ["cx", "cy", "w ", "h "];
        println!(
            "  {} cpu=[{:.1}..{:.1}] metal=[{:.1}..{:.1}]",
            names[r], cpu_min, cpu_max, mtl_min, mtl_max
        );
    }
}
