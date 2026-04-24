# YSCV Cookbook

Practical recipes for common tasks. Each recipe is self-contained — copy, paste, run.

## Table of Contents

1. [Installation](#installation)
2. [Image Processing](#image-processing)
3. [Training a Model](#training-a-model)
4. [ONNX Inference (CPU)](#onnx-inference-cpu)
5. [ONNX Inference (GPU / Metal)](#onnx-inference-gpu--metal)
6. [YOLO Object Detection](#yolo-object-detection)
7. [Detection + Tracking Pipeline](#detection--tracking-pipeline)
8. [Preprocessing Pipeline](#preprocessing-pipeline)
9. [Fine-tuning a Detection Head](#fine-tuning-a-detection-head)
10. [Image Classification with Pretrained Model](#image-classification-with-pretrained-model)
11. [Video Processing](#video-processing)
12. [Cross-Compilation and Deployment](#cross-compilation-and-deployment)
13. [Feature Flags Reference](#feature-flags-reference)
14. [INT4 Quantization](#int4-quantization)
15. [LLM Text Generation](#llm-text-generation)
16. [AV1 Video Decode](#av1-video-decode)
17. [Benchmarking](#benchmarking)
18. [Edge Pipeline](#edge-pipeline)
19. [Rockchip NPU (RKNN)](#rockchip-npu-rknn)

---

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
yscv = "0.1.7"
```

That's it. No Python, no C++ libraries, no system packages required. On macOS, Apple's Accelerate framework is used automatically for BLAS.

For GPU support, add feature flags:

```toml
[dependencies]
yscv = { version = "0.1.7", features = ["gpu"] }           # wgpu (Vulkan/Metal/DX12)
# or
yscv = { version = "0.1.7", features = ["metal-backend"] }  # Metal-native (macOS only, fastest)
```

### Verify it works

```bash
cargo run --example train_linear    # should print "Final loss: 0.00..."
```

---

## Image Processing

### Load, process, save

```rust
use yscv::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load any JPEG/PNG
    let img = imread("photo.jpg")?;
    println!("Size: {}x{}", img.shape()[1], img.shape()[0]); // WxH

    // Grayscale
    let gray = rgb_to_grayscale(&img)?;
    imwrite("gray.png", &gray)?;

    // Gaussian blur
    let blurred = imgproc::gaussian_blur_3x3(&gray)?;
    imwrite("blurred.png", &blurred)?;

    // Edge detection
    let edges = imgproc::canny(&blurred, 0.1, 0.3)?;
    imwrite("edges.png", &edges)?;

    // Binary threshold
    let binary = imgproc::threshold_binary(&gray, 128)?;
    imwrite("binary.png", &binary)?;

    Ok(())
}
```

```bash
cargo run --example image_processing -- photo.jpg ./output/
```

### Resize

```rust
use yscv_imgproc::{imread, imwrite, resize_bilinear, resize_nearest};

let img = imread("photo.jpg")?;
let small = resize_bilinear(&img, 320, 240)?;   // high quality
let fast  = resize_nearest(&img, 320, 240)?;    // 3.3x faster than OpenCV
imwrite("small.png", &small)?;
```

### Morphological operations

```rust
use yscv_imgproc::{dilate_3x3, erode_3x3, rgb_to_grayscale, imread};

let img = imread("photo.jpg")?;
let gray = rgb_to_grayscale(&img)?;
let dilated = dilate_3x3(&gray)?;
let eroded = erode_3x3(&gray)?;
```

---

## Training a Model

### Linear regression (y = 2x + 1)

```rust
use yscv::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let inputs  = Tensor::from_vec(vec![8, 1], vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])?;
    let targets = Tensor::from_vec(vec![8, 1], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])?;

    let mut graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    model.add_linear(
        &mut graph, 1, 1,
        Tensor::from_vec(vec![1, 1], vec![0.1])?,
        Tensor::from_vec(vec![1], vec![0.0])?,
    )?;

    let config = TrainerConfig {
        optimizer: OptimizerKind::Sgd { lr: 0.01, momentum: 0.0 },
        loss: LossKind::Mse,
        epochs: 200,
        batch_size: 8,
        validation_split: None,
    };

    let mut trainer = Trainer::new(config);

    // Optional: stop early if loss plateaus
    trainer.add_callback(Box::new(EarlyStopping::new(10, 0.001, MonitorMode::Min)));

    let result = trainer.fit(&mut model, &mut graph, &inputs, &targets)?;
    println!("Final loss: {:.6}", result.final_loss);
    Ok(())
}
```

```bash
cargo run --example train_linear
```

### CNN classifier

```rust
use yscv::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut graph = Graph::new();
    let mut model = SequentialModel::new(&graph);

    // Conv2d(1 → 4, 3x3) → ReLU → Flatten → Linear(16 → 2)
    model.add_conv2d_zero(1, 4, 3, 3, 1, 1, true)?;
    model.add_relu();
    model.add_flatten();
    model.add_linear_zero(&mut graph, 16, 2)?;  // 4 channels * 2x2 spatial = 16

    // Synthetic data: 8 samples of 4x4 images, 2 classes
    let n = 8;
    let mut input_data = Vec::new();
    let mut target_data = Vec::new();
    for i in 0..n {
        let val = if i < n / 2 { 0.1 } else { 0.9 };
        input_data.extend(std::iter::repeat(val).take(4 * 4));
        if i < n / 2 {
            target_data.extend_from_slice(&[1.0, 0.0]);
        } else {
            target_data.extend_from_slice(&[0.0, 1.0]);
        }
    }

    let inputs  = Tensor::from_vec(vec![n, 4, 4, 1], input_data)?;
    let targets = Tensor::from_vec(vec![n, 2], target_data)?;

    let config = TrainerConfig {
        optimizer: OptimizerKind::Adam { lr: 0.001 },
        loss: LossKind::CrossEntropy,
        epochs: 50,
        batch_size: 8,
        validation_split: None,
    };

    let result = Trainer::new(config).fit(&mut model, &mut graph, &inputs, &targets)?;
    println!("Final loss: {:.6}", result.final_loss);
    Ok(())
}
```

```bash
cargo run --example train_cnn
```

### Available optimizers

```rust
OptimizerKind::Sgd { lr: 0.01, momentum: 0.9 }
OptimizerKind::Adam { lr: 0.001 }
OptimizerKind::AdamW { lr: 0.001 }
OptimizerKind::Lamb { lr: 0.001 }
OptimizerKind::RAdam { lr: 0.001 }
```

### Available losses

```rust
LossKind::Mse
LossKind::CrossEntropy
LossKind::BinaryCrossEntropy
LossKind::Huber
```

---

## ONNX Inference (CPU)

Load any ONNX model and run inference on CPU. No GPU required. Supports 128+ ONNX operators including opset 22.

```rust
use yscv_onnx::{load_onnx_model_from_file, run_onnx_model};
use yscv_tensor::Tensor;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Step 1: Load the ONNX model
    let model = load_onnx_model_from_file("model.onnx")?;
    println!("Nodes: {}", model.nodes.len());

    // Step 2: Create input tensor (match your model's expected shape)
    let input = Tensor::from_vec(
        vec![1, 3, 640, 640],                    // [batch, channels, height, width]
        vec![0.5f32; 1 * 3 * 640 * 640],        // dummy data
    )?;

    // Step 3: Run inference
    let mut inputs = HashMap::new();
    inputs.insert("images".to_string(), input);   // "images" = model's input name
    let outputs = run_onnx_model(&model, inputs)?;

    // Step 4: Read results
    for (name, tensor) in &outputs {
        println!("{}: shape={:?}", name, tensor.shape());
    }

    Ok(())
}
```

CPU performance vs competitors (YOLOv8n 640×640, Apple M1):

| Runtime | YOLOv8n | YOLO11n | Opset 22 |
|---------|---------|---------|----------|
| **yscv** | **30.4ms** | **33.7ms** | Yes |
| onnxruntime 1.19 | 37.4ms | 35.2ms* | No* |
| tract 0.21 | 217.2ms (7× slower) | FAILED | No |

\* onnxruntime requires manual opset downgrade (22→21) for YOLO11n; native opset 22 files fail. yscv handles opset 22 natively. tract fails on YOLO11n and is 7× slower on YOLOv8n.

---

## ONNX Inference (GPU / Metal)

### Option A: wgpu backend (cross-platform)

Works on any platform with Vulkan, Metal, or DX12.

```toml
# Cargo.toml
yscv-onnx = { version = "0.1", features = ["gpu"] }
yscv-kernels = { version = "0.1", features = ["gpu"] }
```

```rust
use yscv_kernels::GpuBackend;
use yscv_onnx::{load_onnx_model_from_file, plan_gpu_execution,
                 GpuWeightCache, run_onnx_model_gpu_cached};
use yscv_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = load_onnx_model_from_file("model.onnx")?;
    let gpu = GpuBackend::new()?;
    println!("GPU: {}", gpu.adapter_name());

    let plan = plan_gpu_execution(&model);
    let mut wc = GpuWeightCache::new();
    let input = Tensor::from_vec(vec![1, 3, 640, 640], vec![0.5f32; 1*3*640*640])?;

    // Warmup (populates weight cache)
    let mut inputs = std::collections::HashMap::new();
    inputs.insert("images".to_string(), input.clone());
    let _ = run_onnx_model_gpu_cached(&gpu, &model, inputs, &mut wc, Some(&plan));

    // Timed run
    let mut inputs = std::collections::HashMap::new();
    inputs.insert("images".to_string(), input);
    let start = std::time::Instant::now();
    let outputs = run_onnx_model_gpu_cached(&gpu, &model, inputs, &mut wc, Some(&plan))?;
    println!("GPU inference: {:.1}ms", start.elapsed().as_secs_f64() * 1000.0);

    for (name, t) in &outputs {
        println!("{}: {:?}", name, t.shape());
    }
    Ok(())
}
```

```bash
cargo run --release --features gpu
```

### Option B: Metal MPSGraph (macOS only — fastest)

MPSGraph compiles the entire ONNX model into a single GPU dispatch and runs it triple-buffered. For sustained inference, the pipelined API (`submit_mpsgraph_plan` + `wait_mpsgraph_plan`) lets CPU-side marshaling overlap GPU work — ~3–5× higher throughput than the sync path.

> **Want the full story?** See [`mpsgraph-guide.md`](mpsgraph-guide.md)
> — standalone guide with sync vs pipelined decision table, multi-input
> models, full API reference, fallback strategy, and troubleshooting.
> This cookbook section is the quick recipe; the guide is everything.

```toml
# Cargo.toml
yscv-onnx = { version = "0.1", features = ["metal-backend"] }
yscv-kernels = { version = "0.1", features = ["metal-backend"] }
```

**Single-shot latency (sync):**

```rust
use yscv_onnx::{
    compile_mpsgraph_plan, run_mpsgraph_plan,
    load_onnx_model_from_file,
};
use yscv_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = load_onnx_model_from_file("model.onnx")?;
    let input_name = model.inputs[0].clone();
    let input = Tensor::from_vec(vec![1, 3, 640, 640], vec![0.5f32; 1*3*640*640])?;

    // Compile the graph once (10-80ms depending on model).
    // Pipeline depth is controlled by YSCV_MPS_PIPELINE (default 3).
    let plan = compile_mpsgraph_plan(&model, &[(input_name.as_str(), &input)])?;

    // Sync run — submit + wait in one call. Multi-input models pass
    // multiple (name, &[f32]) pairs; lookup is by name.
    let input_data = vec![0.5f32; 1 * 3 * 640 * 640];
    let outputs =
        run_mpsgraph_plan(&plan, &[(input_name.as_str(), input_data.as_slice())])?;
    for (name, t) in &outputs {
        println!("{}: {:?}", name, t.shape());
    }
    Ok(())
}
```

**Sustained throughput (pipelined):**

```rust
use std::collections::VecDeque;
use yscv_onnx::{
    compile_mpsgraph_plan, submit_mpsgraph_plan, wait_mpsgraph_plan, InferenceHandle,
};

let plan = compile_mpsgraph_plan(&model, &[(name, &input)])?;
let feeds = [(name, input_data.as_slice())];

// Prime the pipeline — 2 or 3 in-flight frames is the sweet spot.
let depth = 3;
let mut in_flight: VecDeque<InferenceHandle> = (0..depth)
    .map(|_| submit_mpsgraph_plan(&plan, &feeds))
    .collect::<Result<_, _>>()?;

// Steady state: wait on oldest, submit new — GPU is always busy.
for _ in 0..1000 {
    let oldest = in_flight.pop_front().unwrap();
    let outputs = wait_mpsgraph_plan(&plan, oldest)?;
    // … consume outputs …
    in_flight.push_back(submit_mpsgraph_plan(&plan, &feeds)?);
}
// Drain.
while let Some(h) = in_flight.pop_front() {
    let _ = wait_mpsgraph_plan(&plan, h)?;
}
```

The plan allocates `YSCV_MPS_PIPELINE` independent buffer-sets (default `3`, clamped `1..=8`). If the caller tries to submit more outstanding frames than the pipeline depth, `submit_mpsgraph_plan` transparently blocks on the oldest slot's previous command buffer — built-in back-pressure, no buffer aliasing.

```bash
cargo run --release --features metal-backend
# Override pipeline depth:
YSCV_MPS_PIPELINE=4 cargo run --release --features metal-backend
```

### Option C: Metal per-op (fallback for unsupported models)

If MPSGraph doesn't support all ops in your model, use the per-op Metal backend:

```rust
use yscv_onnx::{compile_metal_plan, run_metal_plan, load_onnx_model_from_file};
use yscv_tensor::Tensor;

let model = load_onnx_model_from_file("model.onnx")?;
let input = Tensor::from_vec(vec![1, 3, 640, 640], vec![0.5f32; 1*3*640*640])?;

let plan = compile_metal_plan(&model, "images", &input)?;
let outputs = run_metal_plan(&plan, &vec![0.5f32; 1*3*640*640])?;
```

Metal performance: **3.5ms** on YOLOv8n, **5.0ms** on YOLO11n, **7.8ms** on VballNet (MPSGraph) — 4× faster than CoreML on YOLOv8n (14.2ms), 1.1× faster on VballNet (8.6ms). CoreML fails entirely on YOLO11n.

---

## YOLO Object Detection

End-to-end detection from image to bounding boxes. Works with YOLOv8 and YOLO11 ONNX models.

### Get a model

```bash
pip install ultralytics
yolo export model=yolov8n.pt format=onnx    # YOLOv8 nano
yolo export model=yolo11n.pt format=onnx    # YOLO11 nano
```

### Run detection

```rust
use yscv_detect::{coco_labels, decode_yolov8_output, letterbox_preprocess, yolov8_coco_config};
use yscv_imgproc::imread;
use yscv_onnx::{load_onnx_model_from_file, run_onnx_model};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = load_onnx_model_from_file("yolov8n.onnx")?;
    let img = imread("photo.jpg")?;

    // Letterbox resize to 640x640
    let config = yolov8_coco_config();
    let (input_tensor, scale, pad_x, pad_y) = letterbox_preprocess(&img, config.input_size);

    // Inference
    let mut inputs = std::collections::HashMap::new();
    inputs.insert("images".to_string(), input_tensor);
    let outputs = run_onnx_model(&model, inputs)?;
    let output = outputs.values().next().unwrap();

    // Decode detections
    let detections = decode_yolov8_output(output, &config);
    let labels = coco_labels();

    for det in &detections {
        println!(
            "{}: {:.0}% at ({}, {}, {}x{})",
            labels.get(det.class_id).unwrap_or(&"?"),
            det.confidence * 100.0,
            det.bbox.x, det.bbox.y, det.bbox.width, det.bbox.height
        );
    }
    Ok(())
}
```

```bash
cargo run --example yolo_detect -- yolov8n.onnx photo.jpg
```

The model format (v8 vs v11) is auto-detected from the output tensor shape.

### YOLOv8 vs YOLO11: which model to use

| | YOLOv8n | YOLO11n |
|---|---------|---------|
| Architecture | Standard Conv blocks | DWConv + C2PSA attention |
| FLOPs | 8.7 GFLOP | 6.5 GFLOP |
| Parameters | 3.2M | 2.6M |
| CPU inference | 31.7ms | 34.3ms |
| MPSGraph | 3.5ms | 5.0ms |
| ORT compatible | Yes | No (opset 22) |

**Use YOLOv8n** when you need maximum compatibility (ORT, CoreML, TensorRT all support it) or fastest absolute speed on both CPU and GPU.

**Use YOLO11n** when you want better accuracy per FLOP. YOLO11n uses C2PSA (cross-stage partial + spatial attention) and depthwise separable convolutions — fewer parameters with better feature extraction. Slightly slower than v8n despite fewer FLOPs because attention blocks (MatMul Q*K^T + Softmax) are memory-bound. Only yscv can run it — ORT and tract crash on opset 22.

Both models produce identical output format `[1, 84, 8400]` — 80 COCO classes + 4 bbox coords, 8400 anchor positions across 3 scales (80x80 + 40x40 + 20x20). The `decode` function auto-detects the format.

### Choosing a backend

```
MPSGraph (3.5ms)  →  Metal per-op (12.1ms)  →  CPU (31.7ms)
   fastest              fallback                always works
```

| Backend | When to use |
|---------|-------------|
| **MPSGraph** | Default choice on macOS. Compiles entire model into one GPU dispatch. Fastest by far (3.5ms YOLOv8n). Requires `--features metal-backend`. |
| **Metal per-op** | Fallback when MPSGraph compilation fails (e.g. dynamic reshape chains, unsupported ops). Still 2.6x faster than CPU. Same feature flag. |
| **CPU** | No feature flags needed. Works everywhere. Relative speed vs ORT is model+hardware dependent (see `performance-benchmarks.md` for current measured matrices). Best for Linux/Windows servers, CI, or when GPU isn't available. |
| **wgpu** | Cross-platform GPU via Vulkan/Metal/DX12. Use `--features gpu`. Slower than Metal-native on macOS but works on all platforms with GPU. |

For ONNX CPU kernel routing details (fused Conv paths, asm vs intrinsics,
and runtime A/B env toggles), see
[`onnx-cpu-kernels.md`](onnx-cpu-kernels.md).

**Recommended pattern:**

```rust
// Try MPSGraph → Metal per-op → CPU
#[cfg(feature = "metal-backend")]
{
    match compile_mpsgraph_plan(&model, &[("images", &input)]) {
        Ok(plan) => { /* fastest path */ },
        Err(_) => {
            let plan = compile_metal_plan(&model, "images", &input)?;
            /* metal per-op fallback */
        }
    }
}

#[cfg(not(feature = "metal-backend"))]
{
    let outputs = run_onnx_model(&model, inputs)?;  // CPU
}
```

### Detection quality notes

All backends produce equivalent detections (±1-2 at confidence threshold boundary):
- Metal/MPSGraph use f16 intermediates — may detect ±1 object vs CPU at the 0.25 confidence edge
- CPU uses f32 with platform BLAS (Accelerate on macOS, OpenBLAS on Linux) — minor floating-point differences vs ORT are expected and not a bug
- DFL (Distribution Focal Loss) in the bbox decoder amplifies small numerical differences exponentially, so bbox coordinates can differ by ~30px between runtimes while detecting the same objects

---

## Detection + Tracking Pipeline

Detect objects in video frames and track them across time with DeepSORT or ByteTrack.

```rust
use yscv_detect::{Detection, BoundingBox};
use yscv_track::{ByteTracker, TrackerConfig};

fn main() {
    // Initialize tracker
    let config = TrackerConfig::default();
    let mut tracker = ByteTracker::new(config);

    // For each frame: detect → track
    for frame_id in 0..100 {
        // Your detection code produces these:
        let detections = vec![
            Detection {
                bbox: BoundingBox { x: 100, y: 50, width: 80, height: 120 },
                confidence: 0.92,
                class_id: 0,   // person
            },
        ];

        let tracks = tracker.update(&detections);

        for track in &tracks {
            println!(
                "Frame {}: Track {} class={} at ({}, {})",
                frame_id, track.id, track.class_id,
                track.bbox.x, track.bbox.y
            );
        }
    }
}
```

The full detect → track → recognize pipeline runs in **67µs per frame** (15,000 FPS).

---

## Preprocessing Pipeline

Build composable image transforms for model inference:

```rust
use yscv_model::{Compose, Normalize, Resize, ScaleValues, Transform};
use yscv_tensor::Tensor;

// Build pipeline: resize → scale → normalize
let preprocess = Compose::new()
    .add(Resize::new(224, 224))
    .add(ScaleValues::new(1.0 / 255.0))
    .add(Normalize::new(
        vec![0.485, 0.456, 0.406],    // ImageNet mean
        vec![0.229, 0.224, 0.225],    // ImageNet std
    ));

// Apply to any image tensor
let image = Tensor::from_vec(vec![480, 640, 3], vec![128.0f32; 480*640*3])?;
let processed = preprocess.apply(&image)?;
// processed shape: [224, 224, 3], normalized
```

```bash
cargo run --example image_pipeline
```

---

## Fine-tuning a Detection Head

Train a custom detection head on your data, using a frozen backbone:

```rust
use yscv::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut graph = Graph::new();
    let mut model = SequentialModel::new(&graph);

    // Detection head: feature_map → bbox predictions
    // Input: [batch, 8, 8, 32] from backbone
    // Output: [batch, 8, 8, 5] (cx, cy, w, h, objectness)
    model.add_conv2d_zero(32, 64, 1, 1, 1, 1, true)?;  // pointwise
    model.add_relu();
    model.add_conv2d_zero(64, 5, 1, 1, 1, 1, true)?;   // predict 5 values/cell

    // Your training data (replace with real data)
    let inputs  = Tensor::from_vec(vec![4, 8, 8, 32], vec![0.5; 4*8*8*32])?;
    let targets = Tensor::from_vec(vec![4, 8, 8, 5],  vec![0.0; 4*8*8*5])?;

    let config = TrainerConfig {
        optimizer: OptimizerKind::Adam { lr: 0.001 },
        loss: LossKind::Mse,
        epochs: 100,
        batch_size: 4,
        validation_split: None,
    };

    let result = Trainer::new(config).fit(&mut model, &mut graph, &inputs, &targets)?;
    println!("Final loss: {:.6}", result.final_loss);
    Ok(())
}
```

```bash
cargo run --example yolo_finetune
```

---

## Image Classification with Pretrained Model

Load a pretrained ResNet from the model hub and classify images:

```rust
use yscv_autograd::Graph;
use yscv_imgproc::{imagenet_preprocess, imread};
use yscv_model::{ModelArchitecture, ModelHub, build_resnet, remap_state_dict};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let img = imread("photo.jpg")?;

    // ImageNet preprocessing: resize → center crop → normalize → HWC→CHW
    let input = imagenet_preprocess(&img)?;

    // Build ResNet18 and load pretrained weights
    let mut graph = Graph::new();
    let config = ModelArchitecture::ResNet18.config();
    let model = build_resnet(&mut graph, &config)?;

    let hub = ModelHub::new();  // caches in ~/.yscv/models/
    let weights = hub.load_weights("resnet18")?;
    let mapped = remap_state_dict(&weights, ModelArchitecture::ResNet18);
    // ... load weights into model, run forward pass

    Ok(())
}
```

Available architectures: ResNet18/34/50/101, VGG16/19, MobileNetV2, EfficientNetB0, AlexNet, ViTTiny/Base/Large, DeiTTiny.

```bash
cargo run --example classify_image -- photo.jpg
```

---

## Video Processing

### Decode MP4 frames (H.264 / HEVC / AV1)

Decode any H.264, HEVC, or AV1 MP4 file. The decoder auto-detects the codec from the MP4 container.

```rust
use yscv_video::Mp4VideoReader;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut reader = Mp4VideoReader::open(Path::new("video.mp4"))?;
    println!("NALs: {}, Codec: {:?}", reader.nal_count(), reader.codec());

    let mut frame_count = 0;
    while let Some(frame) = reader.next_frame()? {
        println!("Frame {}: {}x{} keyframe={}", frame_count, frame.width, frame.height, frame.keyframe);
        // frame.rgb8_data is Vec<u8> (RGB8, width * height * 3 bytes)
        frame_count += 1;
    }
    println!("Total: {frame_count} frames decoded");
    Ok(())
}
```

The decoder handles:
- **MP4 container**: avcC/hvcC parameter set extraction, stbl/stco/stsz sample table navigation
- **MKV/WebM container**: EBML demuxer with track/cluster parsing
- **H.264**: Baseline (CAVLC), Main (CABAC), High (CABAC + 8x8 transform), I/P/B slices, weighted prediction, sub-MB partitions, deblocking, SIMD IDCT (NEON + SSE2)
- **HEVC**: Main, Main10 (10-bit, u16 DPB), I/P/B slices, branchless CABAC, BS=0 deblock skip, SAO, CTU quad-tree, tiles parsing, chroma residual
- **AV1**: OBU parser, intra + inter prediction (8-tap Lanczos MC), CDEF, reference frame management, deblocking
- **SIMD**: 29 NEON blocks (aarch64) + 31 SSE2 blocks (x86_64) — full cross-architecture coverage
- **Annex B**: raw H.264/HEVC stream parser

Performance (Apple M-series, single-threaded, vs ffmpeg `-threads 1`, best of 5):

| Video | yscv | ffmpeg | Speedup |
|-------|------|--------|---------|
| H.264 Baseline 1080p (300 frames) | 324ms | 519ms | **1.60×** |
| H.264 High 1080p (300 frames) | 332ms | 760ms | **2.28×** |
| Real camera H.264 1080p60 (1100 frames) | 1187ms | 5372ms | **4.52×** |
| **HEVC Main 1080p P/B (300 frames)** | **575ms** | **806ms** | **1.40×** |
| **HEVC Main 1080p P/B (600 frames)** | **1288ms** | **1808ms** | **1.40×** |
| HEVC Main 1080p I-only (180 frames) | 1538ms | 1483ms | 0.97× |

All HEVC with **full color** (chroma MC + YUV420→RGB). Memory: **27MB RSS** for 41MB file.

**How to reproduce:**
```bash
# Build
cargo build --release --example bench_video_decode

# H.264 benchmark
cargo run --release --example bench_video_decode -- your_video.mp4

# HEVC benchmark
cargo run --release --example bench_video_decode -- your_hevc.mp4

# Compare with ffmpeg
ffmpeg -threads 1 -benchmark -i your_video.mp4 -f null -

# Luma-only (fair vs ffmpeg -f null, skips post-processing)
cargo run --release --example bench_video_decode -- your_video.mp4 --luma-only

# Hardware decode (macOS VideoToolbox)
cargo run --release --features videotoolbox --example bench_video_decode -- your_video.mp4 --hw
```

### Camera capture (requires `native-camera` feature)

```toml
yscv-video = { version = "0.1", features = ["native-camera"] }
```

```rust
use yscv_video::Camera;

let mut cam = Camera::open(0)?;  // device index
loop {
    let frame = cam.capture()?;
    // process frame...
}
```

Supported: V4L2 (Linux), AVFoundation (macOS), MediaFoundation (Windows).

---

## Cross-Compilation and Deployment

### Static binary (no dependencies)

```bash
# Linux x86_64 (musl for fully static)
cargo build --release --target x86_64-unknown-linux-musl

# macOS Apple Silicon
cargo build --release --target aarch64-apple-darwin

# Linux ARM64 (Raspberry Pi, Graviton)
cargo build --release --target aarch64-unknown-linux-gnu

# Windows
cargo build --release --target x86_64-pc-windows-msvc
```

The result is a single binary with zero runtime dependencies. No Python, no Docker, no shared libraries.

### Per-platform optimization

The project includes `.cargo/config.toml` with per-target CPU flags:

| Target | CPU flag | SIMD |
|--------|---------|------|
| macOS ARM | `apple-m1` | NEON |
| Linux ARM (Graviton) | `neoverse-n1` | NEON |
| Linux/Windows x86_64 | `x86-64-v3` | AVX2 |
| Intel Mac | `haswell` | AVX2 + FMA |

All SIMD dispatch is automatic at runtime — the binary includes all code paths and selects the fastest one.

### Deploy to edge devices

```bash
# Cross-compile for Raspberry Pi 4
cross build --release --target aarch64-unknown-linux-gnu

# Copy single binary
scp target/aarch64-unknown-linux-gnu/release/my_app pi@192.168.1.100:~/
```

---

## Feature Flags Reference

| Flag | What | When to use | Platforms |
|------|------|-------------|-----------|
| *(default)* | CPU-only, Accelerate BLAS on macOS | Most cases | All |
| `gpu` | wgpu GPU compute (Vulkan/Metal/DX12) | Cross-platform GPU inference | All |
| `metal-backend` | Metal-native GPU pipeline | Fastest Apple Silicon inference | macOS only |
| `native-camera` | Real camera capture | Live video processing | All |
| `blas` | Hardware BLAS | Enabled by default | All |
| `mkl` | Intel MKL vectorized math | x86 servers with MKL installed | x86/x86_64 |
| `armpl` | ARM Performance Libraries | ARM Linux servers (Graviton, Ampere) | aarch64 Linux |

Combine features as needed:

```bash
# Metal GPU + camera
cargo build --release --features "metal-backend,native-camera"

# wgpu GPU + Intel MKL
cargo build --release --features "gpu,mkl"
```

---

## INT4 Quantization

Quantize an ONNX model's weights to INT4 for reduced memory and faster inference on LLM-style models:

```rust
use yscv_onnx::{load_onnx_model_from_file, quantize_weights_int4, export_onnx_model_to_file};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load the original FP32 model
    let mut model = load_onnx_model_from_file("llama-7b.onnx")?;
    println!("Original initializers: {}", model.initializers.len());

    // Quantize all weight tensors to INT4 (per-channel)
    // This inserts DequantizeLinear nodes automatically
    quantize_weights_int4(&mut model)?;

    // Save the quantized model
    export_onnx_model_to_file(&model, "llama-7b-int4.onnx")?;
    println!("Quantized model saved");

    Ok(())
}
```

INT4 quantization maps each weight channel to [-8, 7] with per-channel scale and zero-point. During inference, the DequantizeLinear nodes unpack values back to f32. Typical model size reduction: 4x compared to FP32.

---

## LLM Text Generation

Generate text from a decoder-only transformer model (GPT-2, LLaMA, Mistral) with KV-cache and sampling:

```rust
use yscv_onnx::{GenerateConfig, generate, load_onnx_model_from_file, run_onnx_model};
use yscv_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = load_onnx_model_from_file("gpt2.onnx")?;

    // Tokenize your prompt (use your tokenizer of choice)
    let prompt_tokens: Vec<u32> = vec![15496, 11, 616, 1438, 318]; // "Hello, my name is"

    let config = GenerateConfig {
        max_tokens: 100,
        temperature: 0.8,
        top_k: 50,
        top_p: 0.9,
        eos_token_id: Some(50256),    // GPT-2 EOS token
        repetition_penalty: 1.2,
    };

    // Generate tokens autoregressively
    let generated = generate(
        &prompt_tokens,
        &config,
        |tokens| {
            // Run the ONNX model for each step
            let input = Tensor::from_vec(
                vec![1, tokens.len()],
                tokens.iter().map(|&t| t as f32).collect(),
            )?;
            let mut inputs = std::collections::HashMap::new();
            inputs.insert("input_ids".to_string(), input);
            let outputs = run_onnx_model(&model, inputs)?;
            let logits = outputs.into_values().next()
                .expect("model should produce at least one output");
            Ok(logits.data().to_vec())
        },
    )?;

    println!("Generated {} tokens: {:?}", generated.len(), generated);
    Ok(())
}
```

The `generate()` function supports:
- **Temperature scaling**: `< 1.0` for sharper distributions, `> 1.0` for more randomness
- **Top-k sampling**: keep only the k most probable next tokens
- **Top-p (nucleus) sampling**: keep tokens until cumulative probability exceeds p
- **Repetition penalty**: penalize tokens that have already appeared
- **EOS stopping**: stop when the end-of-sequence token is generated

---

## AV1 Video Decode

Decode AV1 video frames from an OBU bitstream. The decoder handles both intra (key) frames and inter frames with motion-compensated prediction.

```rust
use yscv_video::Mp4VideoReader;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // AV1 streams are typically in MP4 (av01 codec) or MKV/WebM containers
    let mut reader = Mp4VideoReader::open(Path::new("video_av1.mp4"))?;
    println!("Codec: {:?}", reader.codec());  // Av1

    let mut frame_count = 0;
    while let Some(frame) = reader.next_frame()? {
        println!(
            "Frame {}: {}x{} keyframe={} bit_depth={}",
            frame_count, frame.width, frame.height,
            frame.keyframe, frame.bit_depth
        );
        // frame.rgb8_data is Vec<u8> (RGB8, width * height * 3 bytes)
        frame_count += 1;
    }
    println!("Total: {frame_count} AV1 frames decoded");
    Ok(())
}
```

The AV1 decoder supports:
- **Intra prediction**: DC, vertical, horizontal, smooth, paeth modes
- **Inter prediction**: single-reference motion compensation with 8-tap Lanczos sub-pixel filter
- **CDEF**: directional enhancement filter with primary + secondary filtering
- **Reference frames**: 8-slot buffer with `refresh_frame_flags` management
- **show_existing_frame**: instant output from reference buffer without re-decode

---

## Benchmarking

### Run all benchmarks

```bash
# CPU tensor/kernel benchmarks
cargo run -p yscv-bench --release

# YOLO CPU + GPU benchmark
cargo run --release --example bench_yolo --features gpu

# Metal-native YOLO benchmark (macOS)
cargo run --release --example bench_metal_yolo --features metal-backend

# Metal conv kernel isolation
cargo run --release --example bench_metal_conv --features metal-backend

# H.264 video decode benchmark
cargo run --release --example bench_video_decode -- video.mp4

# Criterion micro-benchmarks (per crate)
cargo bench -p yscv-kernels
cargo bench -p yscv-imgproc
cargo bench -p yscv-tensor
```

### Compare against Python frameworks

```bash
python benchmarks/python/bench_tensor.py    # vs NumPy
python benchmarks/python/bench_kernels.py   # vs PyTorch
python benchmarks/python/bench_opencv.py    # vs OpenCV
```

### Current numbers (Apple M-series, April 2026)

| What | yscv | Competitor | Speedup |
|------|------|-----------|---------|
| YOLOv8n CPU | **30.4ms** | onnxruntime 37.4ms | **1.2x** |
| YOLOv8n MPSGraph | **3.5ms** | CoreML 15.5ms | **4.4x** |
| YOLO11n CPU | **33.7ms** | onnxruntime 35.2ms* | **1.0x** |
| H.264 decode 1080p60 (1100 frames) | **1187ms** | ffmpeg 5372ms | **4.5x** |
| H.264 High 1080p (300 frames) | **332ms** | ffmpeg 760ms | **2.3x** |
| **HEVC 1080p P/B (300 frames)** | **575ms** | **ffmpeg 806ms** | **1.4x** |
| **HEVC 1080p P/B (600 frames)** | **1288ms** | **ffmpeg 1808ms** | **1.4x** |
| sigmoid 921K | 0.217ms | PyTorch 1.296ms | **6.0x** |
| resize nearest u8 | 0.048ms | OpenCV 0.157ms | **3.3x** |
| detect+track pipeline | 0.067ms | — | 15,000 FPS |

### Current numbers (Orange Pi Zero 3, Siamese tracker, April 21, 2026)

Same ONNX model, same input shapes, `--iters 200` for both yscv and ORT:

| Threads | yscv p50 | ORT p50 | yscv vs ORT |
|---:|---:|---:|---:|
| 1 | **461.63 ms** | 499.25 ms | **1.08x faster** |
| 2 | **252.08 ms** | 273.18 ms | **1.08x faster** |
| 3 | **192.91 ms** | 199.41 ms | **1.03x faster** |
| 4 | **150.17 ms** | 164.56 ms | **1.10x faster** |

Full results: [docs/performance-benchmarks.md](performance-benchmarks.md)

---

## Edge Pipeline

Real-time camera-to-encoded-video pipeline using `FramePipeline` (lock-free ring buffer),
V4L2 capture, YUV/RGB conversion, detection overlay, telemetry OSD, and H.264 encoding.

### FramePipeline creation and usage

```rust
use yscv_video::{FramePipeline, run_pipeline, SlotMut, SlotRef};

// 4-slot ring buffer, each slot holds 320*240*3 bytes (RGB8)
let pipeline = FramePipeline::new(4, 320 * 240 * 3);

run_pipeline(
    &pipeline,
    // Stage 1: Capture
    |slot: &mut SlotMut<'_>| {
        slot.set_width(320);
        slot.set_height(240);
        slot.set_pixel_format(2); // RGB8
        // Fill slot.data_mut() with frame pixels...
        true // return false to stop
    },
    // Stage 2: Process (inference)
    |slot: &mut SlotMut<'_>| {
        // Run detection, write results to slot.detections_mut()
    },
    // Stage 3: Output (overlay + encode)
    |slot: &SlotRef<'_>| {
        let _data = slot.data();
        let _dets = slot.detections();
        // Overlay, encode, write...
    },
    100, // max frames
);
```

### V4L2 camera capture (Linux)

```rust
#[cfg(target_os = "linux")]
{
    use yscv_video::{V4l2Camera, V4l2PixelFormat};

    let mut cam = V4l2Camera::open("/dev/video0", 640, 480, V4l2PixelFormat::Yuyv)
        .expect("open camera");
    cam.start_streaming().expect("start streaming");

    // Capture a frame (zero-copy reference to mmap'd kernel buffer)
    let yuyv_data = cam.capture_frame().expect("capture frame");
    // Convert YUYV to RGB8...
}
```

### YUV to RGB conversion

```rust
use yscv_video::{yuv420_to_rgb8, yuyv_to_rgb8};

// YUV420 planar -> RGB8
let y = vec![128u8; 640 * 480];
let u = vec![128u8; 320 * 240];
let v = vec![128u8; 320 * 240];
let rgb = yuv420_to_rgb8(&y, &u, &v, 640, 480).expect("convert");

// YUYV (V4L2 camera output) -> RGB8
let yuyv = vec![128u8; 640 * 480 * 2];
let mut rgb_out = vec![0u8; 640 * 480 * 3];
yuyv_to_rgb8(&yuyv, 640, 480, &mut rgb_out).expect("convert");
```

### Detection overlay

```rust
use yscv_video::overlay_detections;

let mut frame = vec![0u8; 640 * 480 * 3]; // RGB8
let detections = vec![
    (100.0, 50.0, 80.0, 120.0, 0.95, "person"),  // x, y, w, h, confidence, label
    (300.0, 200.0, 60.0, 60.0, 0.87, "car"),
];
overlay_detections(&mut frame, 640, 480, &detections);
```

### H.264 encoding

```rust
use yscv_video::{H264Encoder, h264_encoder::rgb8_to_yuv420};

let width = 320;
let height = 240;
let mut encoder = H264Encoder::new(width as u32, height as u32, 26); // QP=26

let rgb_frame = vec![128u8; width * height * 3];
let yuv = rgb8_to_yuv420(&rgb_frame, width, height);
let nal_data = encoder.encode_frame(&yuv);
// nal_data contains Annex B NAL units (SPS + PPS + IDR slice)
assert!(nal_data.starts_with(&[0x00, 0x00, 0x00, 0x01]));
```

### MAVLink telemetry parsing

```rust
use yscv_video::{
    MavlinkParser, TelemetryData, TelemetryUpdate,
    telemetry_from_mavlink, apply_telemetry_update,
};

let mut parser = MavlinkParser::new();
let mut telemetry = TelemetryData {
    battery_voltage: 12.6,
    battery_current: 0.0,
    altitude_m: 0.0,
    speed_ms: 0.0,
    lat: 0.0,
    lon: 0.0,
    heading_deg: 0.0,
    ai_detections: 0,
};

// Feed raw bytes from a serial port
let raw_bytes: &[u8] = &[]; // from serial read
let messages = parser.feed(raw_bytes);
for msg in &messages {
    if let Some(update) = telemetry_from_mavlink(msg) {
        apply_telemetry_update(&mut telemetry, &update);
    }
}
```

On Linux, use the built-in serial port reader:

```rust
#[cfg(target_os = "linux")]
{
    use yscv_video::MavlinkSerial;

    let mut mav = MavlinkSerial::open("/dev/ttyUSB0", 115200).expect("open serial");
    let messages = mav.read_messages().expect("read");
    // Process messages...
}
```

### Telemetry OSD overlay

```rust
use yscv_video::{TelemetryData, overlay_telemetry};

let mut frame = vec![0u8; 640 * 480 * 3];
let telemetry = TelemetryData {
    battery_voltage: 12.4,
    battery_current: 1.2,
    altitude_m: 45.0,
    speed_ms: 5.3,
    lat: 55.7558,
    lon: 37.6173,
    heading_deg: 127.0,
    ai_detections: 3,
};
overlay_telemetry(&mut frame, 640, 480, &telemetry);
```

### Full pipeline example

Run the complete edge deployment example:

```bash
# Synthetic test frames (all platforms)
cargo run --example edge_pipeline

# Real V4L2 camera (Linux)
cargo run --example edge_pipeline -- --camera /dev/video0

# Save encoded H.264 output
cargo run --example edge_pipeline -- --output out.h264 --frames 300

# With MAVLink telemetry (Linux, serial port)
cargo run --example edge_pipeline -- --camera /dev/video0 --serial /dev/ttyUSB0 --output out.h264

# Custom resolution
cargo run --example edge_pipeline -- --width 640 --height 480 --frames 60
```

The pipeline runs three stages on separate OS threads with zero per-frame allocations.
Typical throughput on synthetic frames: >500 FPS (capture + detect + overlay + H.264 encode).

---

## Rockchip NPU (RKNN)

Real-time NPU inference on Rockchip SoCs (RK3588 / RK3576 / RK3566 / RV1106 /
RV1103) via the `rknn` feature. The binary compiles on any platform; the NPU
path activates at runtime when `librknnrt.so` is present (devices ship it
with rknn-toolkit2-lite).

### Enable the feature

```toml
[dependencies]
yscv-kernels = { version = "0.1", features = ["rknn"] }
```

### Single-context inference

```rust
use yscv_kernels::{rknn_available, RknnBackend};

if !rknn_available() {
    eprintln!("Not on a Rockchip device — falling back to CPU");
    return;
}

let model = std::fs::read("yolov8n.rknn")?;
let rknn = RknnBackend::load(&model)?;
let outputs = rknn.run(&[&rgb_frame_bytes])?;
// outputs is Vec<Tensor> matching the model's output count
```

### Multi-core RK3588 (3 NPU cores in parallel)

Synchronous round-robin (one frame per call, blocks until that core returns):

```rust
use yscv_kernels::{ContextPool, NpuCoreMask};

let pool = ContextPool::new(
    &model,
    &[NpuCoreMask::Core0, NpuCoreMask::Core1, NpuCoreMask::Core2],
)?;
for frame in frames {
    let _outputs = pool.dispatch_roundrobin(&[("images", frame)])?;
}
```

### Pipelined RK3588 (submit/wait, overlap all 3 NPU cores)

Same pattern as the MPSGraph pipelined API on Apple Silicon —
`RknnPipelinedPool` pre-allocates an `RknnMem` per input + output per
slot, exposes non-blocking `submit` and blocking `wait`:

```rust
use std::collections::VecDeque;
use yscv_kernels::{NpuCoreMask, RknnInferenceHandle, RknnPipelinedPool};

let pool = RknnPipelinedPool::new(
    &model,
    &[NpuCoreMask::Core0, NpuCoreMask::Core1, NpuCoreMask::Core2],
)?;

// Prime the pipeline.
let mut in_flight: VecDeque<RknnInferenceHandle> = VecDeque::new();
for frame in frames.by_ref().take(pool.slot_count()) {
    in_flight.push_back(pool.submit(&[("images", &frame)])?);
}

// Steady state: wait on oldest, submit new — every NPU core stays hot.
for frame in frames {
    let oldest = in_flight.pop_front().unwrap();
    let outputs = pool.wait(oldest)?;
    consume(outputs);
    in_flight.push_back(pool.submit(&[("images", &frame)])?);
}

// Drain.
while let Some(h) = in_flight.pop_front() {
    let _ = pool.wait(h)?;
}
```

`RknnInferenceHandle` is `#[must_use]`. Submitting more outstanding
frames than the pool has slots is safe: `submit` transparently waits
on the oldest slot's previous `AsyncFrame` before reusing its buffers.


### Zero-copy V4L2 → NPU (camera straight into NPU buffer)

```rust
use yscv_video::V4l2Camera;

let mut cam = V4l2Camera::open("/dev/video0", 1280, 720,
                                yscv_video::V4l2PixelFormat::Nv12)?;
cam.start_streaming()?;

let (idx, _slice) = cam.capture_frame_indexed()?;
let fd = cam.export_dmabuf(idx)?;        // VIDIOC_EXPBUF
let virt = cam.buffer_mut(idx)?;
let mem = rknn.wrap_fd(fd, virt, 0)?;    // rknn_create_mem_from_fd
rknn.bind_input_by_name(&mem, "images")?;
rknn.run_bound()?;                         // no memcpy at all
```

### Async pipelining (overlap CPU & NPU work)

```rust
let handle = rknn.run_async_bound(/*frame_id=*/ 42)?;
// CPU runs tracking / overlay / encoding while NPU computes…
let outputs = rknn.wait(handle)?;
```

`AsyncFrame::Drop` waits if you forget to call `wait()` — no leaks even on
panic paths.

### On-chip SRAM for hot intermediate tensors (RK3588 / RK3576)

```rust
let mem_info = rknn.mem_size()?;
if mem_info.sram_free_bytes >= 1_048_576 {
    let scratch = rknn.alloc_sram(1 << 20)?;     // 1 MB on-chip
    rknn.bind_internal_mem(&scratch)?;            // skips DDR for hot tensors
}
```

### Custom op with a Rust callback

For ops not in the RKNN built-in set:

```rust
use yscv_kernels::{CustomOp, CustomOpHandler, CustomOpContext, CustomOpTensor};
use std::sync::Arc;

struct ArgmaxOp;
impl CustomOpHandler for ArgmaxOp {
    fn compute(
        &self,
        _ctx: &mut CustomOpContext<'_>,
        ins: &[CustomOpTensor<'_>],
        outs: &[CustomOpTensor<'_>],
    ) -> Result<(), yscv_kernels::KernelError> {
        // SAFETY: SDK guarantees buffers are CPU-mapped for callback duration.
        let src: &[f32] = unsafe {
            let bytes = ins[0].as_bytes();
            std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4)
        };
        let dst: &mut [i32] = unsafe {
            let bytes = outs[0].as_bytes_mut();
            std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut i32, bytes.len() / 4)
        };
        let argmax = src.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i as i32).unwrap_or(0);
        dst[0] = argmax;
        Ok(())
    }
}

let op = CustomOp::cpu("Argmax").with_handler(Arc::new(ArgmaxOp));
let _registration = rknn.register_custom_ops(vec![op])?;
// Hold _registration alive for the rest of the RknnBackend lifetime.
```

Up to 16 simultaneous Rust handlers per process. Pure OpenCL kernels (no
Rust callback) have no slot limit — drop `with_handler(...)` and use
`with_kernel_source(...)` instead.

### Performance profiling

```rust
let rknn = RknnBackend::load_with_perf(&model)?;
rknn.run(&[&rgb])?;
let perf = rknn.perf_detail()?;
for op in &perf.per_op {
    println!("{:>6}us  {} ({})", op.duration_us, op.name, op.op_type);
}
println!("total: {} us", perf.total_us);
```

### On-device ONNX → RKNN compilation (limited)

`librknn_api.so` (toolkit2-lite) supports only **fp16** or **int8 with
calibration** on-device. For full configuration (target SoC, mean/std,
asymmetric quant) use the offline Python `rknn-toolkit2` on the host.

```rust
use yscv_kernels::{compile_onnx_to_rknn, RknnCompileConfig};

// fp16 — no calibration needed
let bytes = compile_onnx_to_rknn(&onnx_model, &RknnCompileConfig::default())?;
std::fs::write("model.rknn", bytes)?;

// int8 with calibration (one preprocessed image path per line in dataset.txt)
let cfg = RknnCompileConfig {
    dataset_path: Some("./calibration.txt".into()),
};
let bytes = compile_onnx_to_rknn(&onnx_model, &cfg)?;
```

See [docs/edge-deployment.md](edge-deployment.md) for the full RKNN guide
including MPP zero-copy, dynamic-shape matmul for LLM inference, and core
allocation strategies.

## Config-driven pipeline (TOML → running system)

`yscv-pipeline` turns a declarative TOML config into a validated
pipeline with real dispatchers per task — no hand-rolled glue. See
[docs/pipeline-config.md](pipeline-config.md) for the schema.

```rust
use yscv_pipeline::{PipelineConfig, run_pipeline};

let cfg = PipelineConfig::from_toml_path("boards/rock4d.toml")?;
// `run_pipeline` validates everything (cycle check, accelerator
// availability, real `.rknn` / `.onnx` magic-byte check via
// validate_models) and builds one AcceleratorDispatcher per task.
let handle = run_pipeline(cfg)?;

// Hot loop — one call walks the task graph from camera to sink.
for frame_bytes in camera_frames {
    let outputs = handle.dispatch_frame(&[("images", frame_bytes)])?;
    // `outputs` is a HashMap<String, Vec<u8>> keyed by
    // "<task_name>.<output_tensor>" for every task's outputs, plus
    // "camera.<name>" and "camera" echoes of the ingress.
    let detector_out = &outputs["detector.output0"];
    // … feed detector_out to post-processing, tracker, etc.
}
```

Recovery + real-time:

```rust
// Transient NPU hang? Ask the pipeline to reset every dispatcher.
if let Err(e) = handle.dispatch_frame(&feeds) {
    eprintln!("dispatch failed: {e}");
    handle.recover_all()?;
}
```

Hot-reload a `.rknn` model in flight (A/B test, swap to a freshly-
quantized variant without stopping the FPV loop):

```rust
use yscv_kernels::RknnPipelinedPool;

let pool: &RknnPipelinedPool = /* from your dispatcher */;
let new_bytes = std::fs::read("models/yolov8n-v2.rknn")?;
pool.reload(&new_bytes)?;  // walks every slot, swaps the model atomically per-slot
```

DVFS hint — kill first-burst latency by pinning CPU governor to
performance (TOML: `[realtime] cpu_governor = "performance"`, or
programmatically):

```rust
use yscv_video::realtime::set_cpu_governor;
let cores_set = set_cpu_governor("performance")?;  // best-effort sysfs writes
println!("performance governor active on {cores_set} cores");
```

With `--features realtime`, `run_pipeline` wires SCHED_FIFO + CPU
affinity + `mlockall` from `[realtime]` in the TOML. Graceful fallback
on dev hosts without `CAP_SYS_NICE`.

The end-to-end reference is
[`examples/src/edge_pipeline_v2.rs`](../examples/src/edge_pipeline_v2.rs) —
it parses a TOML, drives the hot loop against a synthetic camera
generator, reports p50/p95/p99 latency via the new
[`yscv_video::latency_histogram::LatencyHistogram`](../crates/yscv-video/src/latency_histogram.rs),
and exits cleanly on repeated failures.
