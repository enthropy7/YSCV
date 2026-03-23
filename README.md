# yscv

A complete computer vision and deep learning framework in pure Rust. One `cargo add yscv` gives you image processing (178 ops, faster than OpenCV), neural network training (39 layer types, 5 optimizers), ONNX inference (128+ operators, INT8 quantization), real-time detection + tracking + recognition (67µs per frame), H.264 video decoding, and GPU compute via Vulkan/Metal/DX12 — all in a single statically-linked binary with zero Python or C++ dependencies.

We built this because deploying ML in production shouldn't require Docker containers with PyTorch, CUDA drivers, and a prayer. YSCV compiles to one binary that runs on a Raspberry Pi, a cloud VM, or a factory floor computer. Every hot path has hand-tuned SIMD for ARM and x86 — 295 functions with runtime dispatch. It's faster than NumPy, PyTorch, and OpenCV on every operation we benchmarked (72 wins, 0 losses).

## Quick Start

```toml
[dependencies]
yscv = "0.1"
```

Load an image, process it, save the result — three lines:

```rust
use yscv::prelude::*;

let img = imread("photo.jpg")?;
let gray = rgb_to_grayscale(&img)?;
imwrite("gray.png", &gray)?;
```

## What can you build with this?

**The short answer: anything you'd normally need Python + OpenCV + PyTorch for.**

YSCV covers the full pipeline — from reading pixels off a camera to training a neural network to deploying an optimized model. Here are some real examples:

**Video surveillance and security.** Hook up a camera, detect people, track them across frames, recognize faces — all in 67 microseconds per frame. That's 15,000 FPS on a single CPU core. No GPU needed. Deploy on any ARM or x86 device as a static binary.

**Factory quality control.** Train a defect detection model on your laptop, export to ONNX, quantize to INT8, deploy on cheap edge hardware on the production line. The whole thing runs without internet, without Python, without Docker.

**Retail and traffic analytics.** Count people, track movement paths, measure dwell time. One machine can handle dozens of camera streams because the processing is that fast.

**Medical imaging.** Process X-rays and DICOM images at scale: resize, normalize, detect edges, extract features. Memory-safe by construction — no segfaults on malformed input, ever.

**Robotics and drones.** Cross-compile to ARM, get a 5MB binary with feature detection (ORB, SIFT), optical flow, stereo matching, and homography estimation. No Python runtime on your drone.

**Training on the edge.** Fine-tune a pretrained ResNet or ViT directly on device with Adam optimizer, learning rate scheduling, and gradient clipping. You don't always need a GPU cluster.

## Training a model

Build a CNN, train it, done:

```rust
use yscv::prelude::*;

let mut graph = Graph::new();
let mut model = SequentialModel::new(&graph);
model.add_conv2d_zero(3, 16, 3, 3, 1, 1, true)?;
model.add_relu();
model.add_flatten();
model.add_linear_zero(&mut graph, 16 * 30 * 30, 10)?;

let config = TrainerConfig {
    optimizer: OptimizerKind::Adam { lr: 0.001 },
    loss: LossKind::CrossEntropy,
    epochs: 50,
    batch_size: 32,
    validation_split: Some(0.2),
};

let result = Trainer::new(config).fit(&mut model, &mut graph, &inputs, &targets)?;
println!("Final loss: {:.4}", result.final_loss);
```

13 pretrained architectures available out of the box — ResNet, VGG, MobileNet, EfficientNet, ViT, DeiT. Weights download automatically:

```rust
let hub = ModelHub::new(); // caches in ~/.yscv/models/
let weights = hub.load_weights("resnet50")?;
```

## Object detection

YOLOv8 detection with tracking and recognition:

```rust
use yscv::detect::{detect_yolov8_from_rgb, yolov8_coco_config, non_max_suppression};

let img = imread("scene.jpg")?;
let detections = detect_yolov8_from_rgb(&img, &model, &yolov8_coco_config())?;
let filtered = non_max_suppression(&detections, 0.5);
```

The detect → track → recognize pipeline runs in 67µs per frame end-to-end. DeepSORT and ByteTrack are built in. VP-tree ANN for recognition.

## Performance

We benchmark every hot path against NumPy, PyTorch, and OpenCV. Current score: **72 wins, 4 ties, 0 losses.**

Every operation has hand-tuned SIMD on all platforms — NEON on ARM, AVX/SSE on x86, with optional Intel MKL and ARM Performance Libraries for the last few percent.

| vs PyTorch (CPU, 1 thread) | yscv | PyTorch | Speedup |
|-----------|------|---------|---------|
| sigmoid 921K | 0.217ms | 1.296ms | **6.0×** |
| softmax 512×256 | 0.098ms | 0.216ms | **2.2×** |
| layer_norm 512×256 | 0.065ms | 0.117ms | **1.8×** |
| batch_norm 64² | 0.028ms | 0.045ms | **1.6×** |
| relu 921K | 0.069ms | 0.105ms | **1.5×** |

| vs OpenCV (u8, 640×480) | yscv | OpenCV | Speedup |
|------------------------|------|--------|---------|
| resize nearest | 0.048ms | 0.157ms | **3.3×** |
| resize bilinear | 0.068ms | 0.201ms | **3.0×** |
| sobel 3×3 | 0.074ms | 0.169ms | **2.3×** |
| dilate 3×3 | 0.031ms | 0.047ms | **1.5×** |

Full benchmark results in [docs/performance-benchmarks.md](docs/performance-benchmarks.md).

## What's inside

The framework is split into 14 crates, each doing one thing well:

| Crate | Purpose |
|-------|---------|
| `yscv-tensor` | N-dimensional tensor with 115 ops, f32/f16/bf16, SIMD-accelerated |
| `yscv-kernels` | CPU + GPU compute backends, 295 SIMD functions, 17 GPU shaders |
| `yscv-autograd` | Reverse-mode autodiff with 40+ backward ops |
| `yscv-optim` | SGD, Adam, AdamW, LAMB, RAdam — all with SIMD. 11 LR schedulers |
| `yscv-model` | 39 layer types, Trainer API, model zoo, LoRA, distributed training |
| `yscv-imgproc` | 178 image processing ops (blur, edges, morphology, features, color) |
| `yscv-video` | H.264/HEVC decoder, camera I/O, MP4 parsing |
| `yscv-detect` | YOLOv8 pipeline, NMS, heatmap decoding |
| `yscv-track` | DeepSORT, ByteTrack, Kalman filter |
| `yscv-recognize` | Cosine matching, VP-Tree ANN indexing |
| `yscv-eval` | 41 metrics (mAP, MOTA, HOTA, PSNR, etc.), 11 dataset formats |
| `yscv-onnx` | 128+ op ONNX runtime, INT8 quantization, graph optimizer |

## Building

```bash
cargo build --workspace --release
cargo test --workspace --release      # 1,659 tests
cargo run --example train_cnn         # train a CNN on synthetic data
cargo run --example train_linear      # linear regression
cargo run --example image_processing  # image pipeline demo
```

### System dependencies

YSCV builds with zero required system dependencies. Optional:

- **OpenBLAS** (Linux/Windows) — faster matmul. `apt install libopenblas-dev` or `brew install openblas`. macOS uses Accelerate automatically.
- **protoc** — for ONNX proto generation. Without it, a built-in fallback is used. `apt install protobuf-compiler` or `brew install protobuf`.

### Feature flags

| Flag | What it does |
|------|-------------|
| `gpu` | GPU acceleration via wgpu (Vulkan / Metal / DX12) |
| `native-camera` | Real camera capture (V4L2 / AVFoundation / MediaFoundation) |
| `blas` | Hardware BLAS — Accelerate on macOS, OpenBLAS on Linux/Windows |
| `mkl` | Intel MKL for vectorized math on x86 |
| `armpl` | ARM Performance Libraries on ARM Linux (Graviton, Ampere) |

## Architecture

```
yscv (prelude)
    ↓
┌─────────────┐ ┌──────────┐ ┌───────────┐
│ yscv-model   │ │ yscv-    │ │ yscv-     │
│ (39 layers,  │ │ imgproc  │ │ detect/   │
│  trainer,    │ │ (178 ops)│ │ track/    │
│  zoo, LoRA)  │ │          │ │ recognize │
└──────┬───────┘ └────┬─────┘ └─────┬─────┘
       ↓              ↓             ↓
┌──────────────┐ ┌──────────┐ ┌──────────┐
│ yscv-autograd│ │ yscv-    │ │ yscv-    │
│ (40+ backward│ │ kernels  │ │ video    │
│  ops)        │ │ (SIMD+   │ │ (H.264,  │
└──────┬───────┘ │ GPU)     │ │ HEVC,    │
       ↓         └────┬─────┘ │ camera)  │
┌──────────────┐      ↓       └──────────┘
│ yscv-tensor  │←─────┘
│ (115 ops,    │
│  f32/f16/bf16│
│  SIMD)       │
└──────────────┘
```

## When to use YSCV (and when not to)

**Use YSCV when** you need to deploy a trained model as a single binary without Python. When you're building a real-time CV pipeline on edge hardware. When you want to train a CNN or ViT on moderate-scale data without setting up a GPU cluster. When you need image processing that's faster than OpenCV and memory-safe.

**Don't use YSCV when** you need the Python ML ecosystem — Hugging Face model hub, thousands of community architectures, Jupyter notebook prototyping, dynamic graph debugging with breakpoints. When you're training foundation models across thousands of GPUs with NCCL. For that, use PyTorch.

YSCV is faster than PyTorch on individual operations (72 benchmark wins), but PyTorch has a decade-old ecosystem with millions of pretrained models and research tooling. Different tools for different jobs. Train in PyTorch, export to ONNX, deploy with YSCV — or train directly in YSCV if your model fits within our 39 layer types and you don't need the Python ecosystem.

## License

Licensed under either of [Apache 2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at your option.
