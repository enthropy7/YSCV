//! Visualize YOLO detections from all backends side-by-side.
//!
//! Usage:
//!   cargo run --example yolo_vis --features metal-backend -- <model.onnx> <image> <out_dir>

use yscv_detect::{
    Detection, coco_labels, decode_yolov8_output, decode_yolov11_output, letterbox_preprocess,
    yolov8_coco_config,
};
use yscv_imgproc::{DrawDetection, draw_detections, draw_text_scaled, imread, imwrite};
use yscv_onnx::load_onnx_model_from_file;

fn decode(
    output: &yscv_tensor::Tensor,
    config: &yscv_detect::YoloConfig,
    orig_w: usize,
    orig_h: usize,
) -> Vec<Detection> {
    let out_shape = output.shape();
    if out_shape.len() == 3 && out_shape[1] < out_shape[2] {
        decode_yolov8_output(output, config, orig_w, orig_h)
    } else {
        decode_yolov11_output(output, config, orig_w, orig_h)
    }
}

fn to_draw(dets: &[Detection]) -> Vec<DrawDetection> {
    dets.iter()
        .map(|d| DrawDetection {
            x: d.bbox.x1 as usize,
            y: d.bbox.y1 as usize,
            width: (d.bbox.x2 - d.bbox.x1).max(0.0) as usize,
            height: (d.bbox.y2 - d.bbox.y1).max(0.0) as usize,
            score: d.score,
            class_id: d.class_id,
        })
        .collect()
}

fn annotate_and_save(
    img: &yscv_tensor::Tensor,
    dets: &[Detection],
    backend_name: &str,
    out_path: &str,
) {
    let labels_vec = coco_labels();
    let labels: Vec<&str> = labels_vec.iter().map(|s| s.as_str()).collect();
    let mut img_copy = img.clone();

    let draw_dets = to_draw(dets);
    draw_detections(&mut img_copy, &draw_dets, &labels).unwrap();

    // Draw backend label at top
    let label = format!("{} ({} objects)", backend_name, dets.len());
    draw_text_scaled(&mut img_copy, &label, 10, 10, 2, [1.0, 1.0, 1.0]).unwrap();

    imwrite(out_path, &img_copy).unwrap();
    println!("  Saved: {out_path}");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: yolo_vis <model.onnx> <image> <out_dir>");
        std::process::exit(1);
    }

    let model_path = &args[1];
    let image_path = &args[2];
    let out_dir = &args[3];

    // Load model
    println!("Loading model: {model_path}");
    let model = load_onnx_model_from_file(model_path).expect("Failed to load model");

    // Load image
    println!("Loading image: {image_path}");
    let img = imread(image_path).expect("Failed to load image");
    let shape = img.shape().to_vec();
    let (orig_h, orig_w) = (shape[0], shape[1]);
    println!("  Size: {orig_w}x{orig_h}");

    let config = yolov8_coco_config();
    let (letterboxed, _scale, _pad_x, _pad_y) = letterbox_preprocess(&img, config.input_size);
    let sz = config.input_size;

    // HWC → NCHW
    let hwc_data = letterboxed.data();
    let mut nchw = vec![0.0f32; 3 * sz * sz];
    for y in 0..sz {
        for x in 0..sz {
            let src = (y * sz + x) * 3;
            for c in 0..3 {
                nchw[c * sz * sz + y * sz + x] = hwc_data[src + c];
            }
        }
    }
    let input_tensor =
        yscv_tensor::Tensor::from_vec(vec![1, 3, sz, sz], nchw.clone()).expect("tensor");

    // CPU
    println!("\n=== CPU ===");
    let cpu_dets = {
        let mut inputs = std::collections::HashMap::new();
        inputs.insert("images".to_string(), input_tensor.clone());
        let outputs = yscv_onnx::run_onnx_model(&model, inputs).expect("CPU failed");
        let output = outputs.values().next().expect("no output");
        decode(output, &config, orig_w, orig_h)
    };
    println!("  {} detections", cpu_dets.len());
    annotate_and_save(&img, &cpu_dets, "CPU", &format!("{out_dir}/cpu.png"));

    // Metal per-op
    #[cfg(feature = "metal-backend")]
    {
        println!("\n=== Metal per-op ===");
        let plan = yscv_onnx::compile_metal_plan(&model, "images", &input_tensor)
            .expect("Metal compile failed");
        let result = yscv_onnx::run_metal_plan(&plan, &nchw).expect("Metal run failed");
        let output = result.values().next().expect("no output");
        let metal_dets = decode(output, &config, orig_w, orig_h);
        println!("  {} detections", metal_dets.len());
        annotate_and_save(
            &img,
            &metal_dets,
            "Metal per-op",
            &format!("{out_dir}/metal.png"),
        );
    }

    // MPSGraph
    #[cfg(feature = "metal-backend")]
    {
        println!("\n=== MPSGraph ===");
        match yscv_onnx::compile_mpsgraph_plan(&model, &[("images", &input_tensor)]) {
            Ok(plan) => {
                let result = yscv_onnx::run_mpsgraph_plan(&plan, &[("images", nchw.as_slice())])
                    .expect("MPSGraph run failed");
                let output = result.values().next().expect("no output");
                let mps_dets = decode(output, &config, orig_w, orig_h);
                println!("  {} detections", mps_dets.len());
                annotate_and_save(
                    &img,
                    &mps_dets,
                    "MPSGraph",
                    &format!("{out_dir}/mpsgraph.png"),
                );
            }
            Err(e) => {
                eprintln!("  MPSGraph compile failed: {e}");
            }
        }
    }

    println!("\nDone!");
}
