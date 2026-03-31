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
14. [Benchmarking](#benchmarking)

---

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
yscv = "0.1"
```

That's it. No Python, no C++ libraries, no system packages required. On macOS, Apple's Accelerate framework is used automatically for BLAS.

For GPU support, add feature flags:

```toml
[dependencies]
yscv = { version = "0.1", features = ["gpu"] }           # wgpu (Vulkan/Metal/DX12)
# or
yscv = { version = "0.1", features = ["metal-backend"] }  # Metal-native (macOS only, fastest)
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

MPSGraph compiles the entire ONNX model into a single GPU dispatch. This is the fastest path on Apple Silicon.

```toml
# Cargo.toml
yscv-onnx = { version = "0.1", features = ["metal-backend"] }
yscv-kernels = { version = "0.1", features = ["metal-backend"] }
```

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

    // Compile the graph (one-time cost, ~10-30ms)
    let plan = compile_mpsgraph_plan(&model, &input_name, &input)?;

    // Run — input as raw f32 slice
    let input_data = vec![0.5f32; 1 * 3 * 640 * 640];
    let outputs = run_mpsgraph_plan(&plan, &input_data)?;
    for (name, t) in &outputs {
        println!("{}: {:?}", name, t.shape());
    }
    Ok(())
}
```

```bash
cargo run --release --features metal-backend
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
| **CPU** | No feature flags needed. Works everywhere. 3.2x faster than ORT. Best for Linux/Windows servers, CI, or when GPU isn't available. |
| **wgpu** | Cross-platform GPU via Vulkan/Metal/DX12. Use `--features gpu`. Slower than Metal-native on macOS but works on all platforms with GPU. |

**Recommended pattern:**

```rust
// Try MPSGraph → Metal per-op → CPU
#[cfg(feature = "metal-backend")]
{
    match compile_mpsgraph_plan(&model, "images", &input) {
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

### Decode MP4 frames (H.264 / HEVC)

Decode any H.264 MP4 file — Baseline (CAVLC), Main, or High profile (CABAC) are all supported. The decoder auto-detects the codec from the MP4 container.

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
- **SIMD**: 29 NEON blocks (aarch64) + 31 SSE2 blocks (x86_64) — full cross-architecture coverage
- **Annex B**: raw H.264/HEVC stream parser

Performance (Apple M-series, single-threaded, vs ffmpeg `-threads 1`, best of 5):

| Video | yscv | ffmpeg | Speedup |
|-------|------|--------|---------|
| H.264 Baseline 1080p (300 frames) | 302ms | 509ms | **1.68×** |
| H.264 High 1080p (300 frames) | 315ms | 750ms | **2.38×** |
| Real camera H.264 1080p60 (1100 frames) | 1195ms | 5332ms | **4.46×** |
| **HEVC Main 1080p P/B (300 frames)** | **480ms** | **811ms** | **1.68×** |
| **HEVC Main 1080p P/B (600 frames)** | **1114ms** | **1797ms** | **1.61×** |
| HEVC Main 1080p I-only (180 frames) | 1486ms | 1487ms | **1.00×** |

Luma-only mode (`--luma-only`, pure decode without post-processing):

| Video | yscv | ffmpeg | Speedup |
|-------|------|--------|---------|
| HEVC P/B 5s | 371ms | 811ms | **2.18×** |
| HEVC P/B 10s | 823ms | 1797ms | **2.18×** |

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

### Current numbers (Apple M-series, March 2026)

| What | yscv | Competitor | Speedup |
|------|------|-----------|---------|
| YOLOv8n CPU | **30.4ms** | onnxruntime 37.4ms | **1.2x** |
| YOLOv8n MPSGraph | **3.5ms** | CoreML 15.5ms | **4.4x** |
| YOLO11n CPU | **33.7ms** | onnxruntime 35.2ms* | **1.0x** |
| H.264 decode 1080p60 (1100 frames) | **1195ms** | ffmpeg 5332ms | **4.5x** |
| H.264 High 1080p (300 frames) | **315ms** | ffmpeg 750ms | **2.4x** |
| **HEVC 1080p P/B (300 frames)** | **480ms** | **ffmpeg 811ms** | **1.7x** |
| **HEVC 1080p P/B (600 frames)** | **1114ms** | **ffmpeg 1797ms** | **1.6x** |
| sigmoid 921K | 0.217ms | PyTorch 1.296ms | **6.0x** |
| resize nearest u8 | 0.048ms | OpenCV 0.157ms | **3.3x** |
| detect+track pipeline | 0.067ms | — | 15,000 FPS |

Full results: [docs/performance-benchmarks.md](performance-benchmarks.md)
