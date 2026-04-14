//! Example: YOLOv8/v11 object detection with ONNX model.
//!
//! Demonstrates the full detection pipeline:
//! 1. Load an ONNX model (YOLOv8 or YOLOv11)
//! 2. Load and preprocess an image (letterbox)
//! 3. Run inference
//! 4. Decode detections with NMS
//! 5. Print results
//!
//! Usage:
//!   cargo run --example yolo_detect -- <model.onnx> <image.jpg> [--metal] [--mpsgraph]
//!
//! Example with YOLOv8:
//!   cargo run --example yolo_detect -- yolov8n.onnx photo.jpg
//!
//! Example with YOLOv11:
//!   cargo run --example yolo_detect -- yolo11n.onnx photo.jpg
//!
//! Metal per-op backend:
//!   cargo run --example yolo_detect --features metal-backend -- yolo11n.onnx photo.jpg --metal
//!
//! MPSGraph backend:
//!   cargo run --example yolo_detect --features metal-backend -- yolo11n.onnx photo.jpg --mpsgraph
//!
//! To get models:
//!   pip install ultralytics
//!   yolo export model=yolov8n.pt format=onnx   # YOLOv8
//!   yolo export model=yolo11n.pt format=onnx   # YOLOv11

use yscv_detect::{
    Detection, coco_labels, decode_yolov8_output, decode_yolov11_output, letterbox_preprocess,
    yolov8_coco_config,
};
use yscv_imgproc::imread;
use yscv_onnx::load_onnx_model_from_file;

fn decode(
    output: &yscv_tensor::Tensor,
    config: &yscv_detect::YoloConfig,
    orig_w: usize,
    orig_h: usize,
) -> Vec<Detection> {
    let out_shape = output.shape();
    if out_shape.len() == 3 && out_shape[1] < out_shape[2] {
        println!("  Detected YOLOv8 output format");
        decode_yolov8_output(output, config, orig_w, orig_h)
    } else {
        println!("  Detected YOLOv11 output format");
        decode_yolov11_output(output, config, orig_w, orig_h)
    }
}

fn print_detections(detections: &[Detection]) {
    let labels = coco_labels();
    println!("\nDetected {} objects:", detections.len());
    for (i, det) in detections.iter().enumerate() {
        let label = labels
            .get(det.class_id)
            .map(|s| s.as_str())
            .unwrap_or("unknown");
        println!(
            "  [{}] {} ({:.1}%) at ({:.0}, {:.0}, {:.0}, {:.0})",
            i + 1,
            label,
            det.score * 100.0,
            det.bbox.x1,
            det.bbox.y1,
            det.bbox.x2,
            det.bbox.y2,
        );
    }
    if detections.is_empty() {
        println!("  (no objects detected above confidence threshold)");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: yolo_detect <model.onnx> <image.jpg> [--metal] [--mpsgraph]");
        eprintln!();
        eprintln!("Supports both YOLOv8 and YOLOv11 ONNX models.");
        eprintln!("The format is auto-detected from the output tensor shape.");
        eprintln!();
        eprintln!("Flags:");
        eprintln!("  --metal     Use Metal per-op backend (requires --features metal-backend)");
        eprintln!("  --mpsgraph  Use MPSGraph backend (requires --features metal-backend)");
        eprintln!();
        eprintln!("Example:");
        eprintln!("  cargo run --example yolo_detect -- yolov8n.onnx photo.jpg");
        std::process::exit(1);
    }

    let model_path = &args[1];
    let image_path = &args[2];
    let use_metal = args.iter().any(|a| a == "--metal");
    let use_mpsgraph = args.iter().any(|a| a == "--mpsgraph");

    // Step 1: Load ONNX model
    println!("Loading model: {model_path}");
    let model = load_onnx_model_from_file(model_path).expect("Failed to load ONNX model");
    println!("  Nodes: {}", model.nodes.len());

    // Step 2: Load and preprocess image
    println!("Loading image: {image_path}");
    let img = imread(image_path).expect("Failed to load image");
    let shape = img.shape().to_vec();
    let (orig_h, orig_w) = (shape[0], shape[1]);
    println!("  Size: {orig_w}x{orig_h}");

    let config = yolov8_coco_config();
    let (letterboxed, _scale, _pad_x, _pad_y) = letterbox_preprocess(&img, config.input_size);
    let sz = config.input_size;
    println!("  Preprocessed to {sz}x{sz}");

    // Convert HWC [H, W, 3] → NCHW [1, 3, H, W] for ONNX inference
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

    // ── CPU inference ──
    {
        println!("\n=== CPU ===");
        println!("Running inference...");
        let mut inputs = std::collections::HashMap::new();
        inputs.insert("images".to_string(), input_tensor.clone());
        let outputs = yscv_onnx::run_onnx_model(&model, inputs).expect("Inference failed");
        let output = outputs.values().next().expect("no output");
        println!("  Output shape: {:?}", output.shape());

        // Dump raw output tensor for comparison with ORT
        if std::env::var("DUMP_RAW").is_ok() {
            let data = output.data();
            let path = "/tmp/yolo_vis/yscv_raw.bin";
            let mut f = std::fs::File::create(path).expect("create dump file");
            use std::io::Write;
            for &val in data {
                f.write_all(&val.to_le_bytes()).expect("write");
            }
            println!("  Dumped {} floats to {path}", data.len());

            // Also dump preprocessed input for comparison
            let ipath = "/tmp/yolo_vis/yscv_input.bin";
            let mut fi = std::fs::File::create(ipath).expect("create input dump");
            for &val in &nchw {
                fi.write_all(&val.to_le_bytes()).expect("write");
            }
            println!("  Dumped input {} floats to {ipath}", nchw.len());
        }

        let dets = decode(output, &config, orig_w, orig_h);
        print_detections(&dets);
    }

    // ── Metal per-op inference ──
    #[cfg(feature = "metal-backend")]
    if use_metal {
        println!("\n=== Metal per-op ===");
        let plan = yscv_onnx::compile_metal_plan(&model, "images", &input_tensor)
            .expect("Metal compile failed");
        let result = yscv_onnx::run_metal_plan(&plan, &nchw).expect("Metal run failed");
        for (name, t) in &result {
            println!("  Output '{}': {:?}", name, t.shape());
            let dets = decode(t, &config, orig_w, orig_h);
            print_detections(&dets);
        }
    }

    // ── MPSGraph inference ──
    #[cfg(feature = "metal-backend")]
    if use_mpsgraph {
        println!("\n=== MPSGraph ===");
        let plan = yscv_onnx::compile_mpsgraph_plan(&model, &[("images", &input_tensor)])
            .expect("MPSGraph compile failed");
        let result = yscv_onnx::run_mpsgraph_plan(&plan, &[("images", nchw.as_slice())])
            .expect("MPSGraph run failed");
        for (name, t) in &result {
            println!("  Output '{}': {:?}", name, t.shape());
            let dets = decode(t, &config, orig_w, orig_h);
            print_detections(&dets);
        }
    }

    #[cfg(not(feature = "metal-backend"))]
    {
        if use_metal {
            eprintln!("Metal per-op requires --features metal-backend");
        }
        if use_mpsgraph {
            eprintln!("MPSGraph requires --features metal-backend");
        }
    }

    println!("\nDone!");
}
