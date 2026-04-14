# Important: Current state of framework is not stable. It is in development. WIP, to be clear. If you found any issues, please report them. You can connect with me via Telegram on my GitHub page or open an issue on GitHub. Stable version will be released soon. Milestone is 0.2.0. 

# yscv
![MSRV](https://img.shields.io/badge/MSRV-1.94-blue?logo=rust)
![Rust](https://img.shields.io/badge/Rust-2024_edition-orange?logo=rust)
![Platforms](https://img.shields.io/badge/platforms-macOS%20%7C%20Linux%20%7C%20Windows%20%7C%20ARM64-lightgrey)
[![GitHub stars](https://img.shields.io/github/stars/enthropy7/yscv?style=flat&logo=github)](https://github.com/enthropy7/yscv/stargazers)
[![CI](https://github.com/enthropy7/yscv/actions/workflows/ci.yml/badge.svg)](https://github.com/enthropy7/yscv/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-1861%20passing%20%7C%201897%20all--features-brightgreen.svg)]()
[![Crates.io](https://img.shields.io/crates/v/yscv)](https://crates.io/crates/yscv)
[![docs.rs](https://img.shields.io/docsrs/yscv)](https://docs.rs/yscv)

A complete computer vision and deep learning framework in pure Rust. One `cargo add yscv` gives you image processing (160 ops, faster than OpenCV), neural network training (39 layer types, 8 optimizers), ONNX inference (128+ operators, INT4/INT8 quantization), LLM generation (KV-cache, RoPE, GQA), real-time detection + tracking + recognition (67µs per frame), H.264/HEVC/AV1 video decoding (4.5× faster than ffmpeg), hardware decode (VideoToolbox/VAAPI/NVDEC/MediaFoundation), and GPU compute via Vulkan/Metal/DX12 — all in a single statically-linked binary with zero Python or C++ dependencies.

> **First time here?** → **[QUICKSTART](QUICKSTART.md)** (5 minutes to a running program) · **[Tutorial](docs/getting-started.md)** (full walkthrough) · **[Cookbook](docs/cookbook.md)** (recipes by task) · **[Edge / Rockchip](docs/edge-deployment.md)** (NPU deployment) · **[Examples](examples/README.md)** (worked code) · **[Troubleshooting](docs/troubleshooting.md)** (when things break) · **[Docs hub](docs/README.md)** (everything else)

We built this because deploying ML in production shouldn't require Docker containers with PyTorch, CUDA drivers, and a prayer. YSCV compiles to one binary that runs on a Raspberry Pi, a cloud VM, or a factory floor computer. Every hot path has hand-tuned SIMD for ARM and x86 — 315 `#[target_feature]`-gated functions with runtime dispatch. It's faster than NumPy, PyTorch, OpenCV, and ffmpeg on every operation we benchmarked (85 wins, 0 losses).

## Quick Start

```toml
[dependencies]
yscv = "0.1.7"
```

Load an image, process it, save the result — three lines:

```rust,ignore
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

```rust,ignore
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

17 pretrained architectures available out of the box — ResNet, VGG, MobileNet, EfficientNet, ViT, DeiT. Weights download automatically:

```rust,ignore
let hub = ModelHub::new(); // caches in ~/.yscv/models/
let weights = hub.load_weights("resnet50")?;
```

## Object detection

YOLOv8 detection with tracking and recognition:

```rust,ignore
use yscv::detect::{detect_yolov8_from_rgb, yolov8_coco_config, non_max_suppression};

let img = imread("scene.jpg")?;
let detections = detect_yolov8_from_rgb(&img, &model, &yolov8_coco_config())?;
let filtered = non_max_suppression(&detections, 0.5);
```

The detect → track → recognize pipeline runs in 67µs per frame end-to-end. DeepSORT and ByteTrack are built in. VP-tree ANN for recognition.

## Performance

We benchmark every hot path against NumPy, PyTorch, OpenCV, onnxruntime, ffmpeg, and CoreML. Current score: **85 wins, ~4 parity, 1 close, 0 losses.** H.264 decode is **4.5× faster than ffmpeg**, HEVC is **1.4× faster** (full color). MPSGraph GPU inference is **3.4× faster than Apple CoreML** on YOLOv8n. 1,861 default tests / 1,897 with all features, across 14 crates.

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

| ONNX Inference | yscv | Competitor | Result |
|----------------------------------|------|-----------|---------|
| YOLOv8n CPU | 32.7ms | onnxruntime 103.4ms | **3.2× faster** |
| YOLOv8n MPSGraph | **4.8ms** | CoreML 16.1ms | **3.4× faster** (CoreML uses ANE hw) |
| VballNet CPU | 124.1ms | onnxruntime 196.7ms | **1.6× faster** |
| VballNet MPSGraph | **7.8ms** | CoreML 8.6ms | **1.1× faster** (CoreML uses AMX hw) |
| YOLO11n CPU | 36.4ms | all competitors FAIL | **WIN** |
| YOLO11n MPSGraph | **5.9ms** | all competitors FAIL | **WIN** |

Full benchmark results in [docs/performance-benchmarks.md](docs/performance-benchmarks.md).

## What's inside

The framework is split into 14 crates, each doing one thing well:

| Crate | Purpose |
|-------|---------|
| `yscv-tensor` | N-dimensional tensor with 115 ops, f32/f16/bf16, SIMD-accelerated |
| `yscv-kernels` | CPU + GPU compute backends, 315 SIMD functions, 61 WGSL + 4 Metal shaders |
| `yscv-autograd` | Reverse-mode autodiff with 61 backward op variants |
| `yscv-optim` | SGD, Adam, AdamW, Adagrad, RAdam, RmsProp, Lamb, Lars + Lookahead, 11 LR schedulers |
| `yscv-model` | 39 layer types, 17 loss functions, Trainer API, model zoo (17 architectures), LoRA |
| `yscv-imgproc` | 160 image processing ops (blur, edges, morphology, features, color) |
| `yscv-video` | H.264/HEVC/AV1 decoder (parallel tile/WPP, weighted prediction, AV1 inter MC), hardware decode, camera I/O, MP4 / MKV parsing, audio metadata |
| `yscv-detect` | YOLOv8/v11 pipeline, NMS, heatmap decoding |
| `yscv-track` | DeepSORT, ByteTrack, Kalman filter, Hungarian assignment, Re-ID |
| `yscv-recognize` | Cosine matching, VP-Tree ANN indexing, Recognizer with enroll/match |
| `yscv-eval` | Classification/detection/tracking/regression/image-quality metrics, 8 dataset adapters |
| `yscv-onnx` | 128+ op ONNX CPU runtime, INT4/INT8 quantization, LLM generation (KV-cache, RoPE, GQA), graph optimizer, Metal/MPSGraph GPU |
| `yscv-cli` | Inference + evaluation CLI: camera diagnostics, dataset eval, pipeline runner |
| `yscv` | Umbrella crate re-exporting the prelude and per-crate APIs |

## Building

```bash
cargo build --workspace --release
cargo test --workspace --release      # 1,861 tests (default)
cargo run --example train_cnn         # train a CNN on synthetic data
cargo run --example train_linear      # linear regression
cargo run --example image_processing  # image pipeline demo
cargo run --example yolo_detect -- model.onnx photo.jpg  # YOLOv8/v11 detection
cargo run --example yolo_finetune     # fine-tune a detection head
```

### GPU / Metal (Apple Silicon)

```bash
# wgpu backend (Vulkan / Metal / DX12 — cross-platform)
cargo run --release --example bench_yolo --features gpu

# Metal-native backend (macOS only — fastest on Apple Silicon)
cargo run --release --example bench_mpsgraph --features metal-backend  # MPSGraph + per-op comparison
cargo run --release --example bench_metal_yolo --features metal-backend
cargo run --release --example bench_mps_gemm  --features metal-backend

# Both features can be combined
cargo test --workspace --features metal-backend
cargo clippy --workspace --features metal-backend
```

The `gpu` feature uses wgpu and works on any platform with Vulkan, Metal, or DX12. The `metal-backend` feature talks to Metal directly via `metal-rs` and provides two backends: **MPSGraph** (whole-model graph compilation, fastest — 4.8 ms YOLOv8n; also exposes a triple-buffered `submit_mpsgraph_plan` / `wait_mpsgraph_plan` pipelined API that overlaps CPU marshaling with GPU compute for 3–5× throughput on sustained inference, multi-input models supported) and **Metal per-op** (individual op dispatch with Winograd + MPS GEMM, fallback for unsupported models). On macOS, `metal-backend` is what you want.

### System dependencies

YSCV builds with zero required system dependencies. Optional:

- **OpenBLAS** (Linux/Windows) — faster matmul. `apt install libopenblas-dev` or `brew install openblas`. macOS uses Accelerate automatically.
- **protoc** — for ONNX proto generation. Without it, a built-in fallback is used. `apt install protobuf-compiler` or `brew install protobuf`.

### Feature flags

| Flag | What it does | Platforms |
|------|-------------|-----------|
| `gpu` | GPU acceleration via wgpu (Vulkan / Metal / DX12) | All |
| `metal-backend` | Metal-native GPU pipeline — fastest on Apple Silicon | macOS only |
| `rknn` | Rockchip NPU via `librknnrt.so` (RK3588 / RK3576 / RV1106) — `dlopen` at runtime, full SDK 2.4.3a0 | Linux ARM64 (Rockchip device) |
| `native-camera` | Real camera capture (V4L2 / AVFoundation / MediaFoundation) | All |
| `blas` | Hardware BLAS — Accelerate on macOS, OpenBLAS on Linux/Windows | All (default) |
| `mkl` | Intel MKL for vectorized math on x86 | x86/x86_64 |
| `armpl` | ARM Performance Libraries on ARM Linux (Graviton, Ampere) | aarch64 Linux |

### Multi-architecture SIMD

All hot paths have hand-tuned SIMD for three architectures with runtime CPU detection:

| | macOS (Apple Silicon) | Linux/Windows (x86_64) | Linux (ARM64) |
|---|---|---|---|
| **f32 tensor ops** | NEON 4× unroll | AVX 4× unroll → SSE fallback | NEON 4× unroll |
| **u8 image ops** | NEON 16B/iter | AVX2 32B/iter → SSE2/SSSE3 16B/iter | NEON 16B/iter |
| **Activations** | NEON 3-term poly | AVX/SSE poly | NEON 3-term poly |
| **MatMul / Conv** | Accelerate BLAS | OpenBLAS (or MKL with `--features mkl`) | OpenBLAS (or ARMPL with `--features armpl`) |
| **Vectorized math** | vDSP (Accelerate) | MKL VML (opt-in) | ARMPL (opt-in) |
| **Threading** | GCD dispatch_apply (~0.3µs) | std::thread::scope (~1µs) | std::thread::scope (~1µs) |
| **GPU inference** | MPSGraph (4.8ms YOLOv8n) | wgpu/Vulkan | wgpu/Vulkan |
| **Softmax** | Fused NEON | Fused AVX/SSE | Fused NEON |
| **Allocator** | mimalloc | mimalloc | mimalloc |

SIMD dispatch is automatic at runtime — no need for `-C target-cpu` flags (though they help: `-C target-cpu=apple-m1` or `-C target-cpu=native` for best codegen). The framework uses `std::is_x86_feature_detected!("avx")` on x86 and compile-time `#[cfg(target_arch = "aarch64")]` on ARM. 315 `#[target_feature]`-gated functions total, all with scalar fallback for WASM/RISC-V/Miri.

### Recommended release profile

For best performance, the workspace already includes an optimized release profile in `Cargo.toml`:

```toml
[profile.release]
lto = "thin"
codegen-units = 1
```

For Apple Silicon specifically, add to `.cargo/config.toml`:

```toml
[target.aarch64-apple-darwin]
rustflags = ["-C", "target-cpu=apple-m1"]
```

## Architecture

```
yscv (prelude)
    ↓
┌─────────────┐ ┌──────────┐ ┌───────────┐
│ yscv-model   │ │ yscv-    │ │ yscv-     │
│ (39 layers,  │ │ imgproc  │ │ detect/   │
│  trainer,    │ │ (160 ops)│ │ track/    │
│  zoo, LoRA)  │ │          │ │ recognize │
└──────┬───────┘ └────┬─────┘ └─────┬─────┘
       ↓              ↓             ↓
┌──────────────┐ ┌──────────┐ ┌──────────┐
│ yscv-autograd│ │ yscv-    │ │ yscv-    │
│ (61 backward │ │ kernels  │ │ video    │
│  op variants)│ │ (SIMD+   │ │ (H.264,  │
└──────┬───────┘ │ GPU)     │ │ HEVC,AV1,│
       ↓         └────┬─────┘ │ camera)  │
┌──────────────┐      ↓       └──────────┘
│ yscv-tensor  │←─────┘
│ (115 ops,    │
│  f32/f16/bf16│
│  SIMD)       │
└──────────────┘
```

## Cookbook

See [docs/cookbook.md](docs/cookbook.md) for practical recipes: image processing, model training, ONNX inference (CPU/GPU/Metal), YOLO detection, tracking, preprocessing pipelines, fine-tuning, video, cross-compilation, and benchmarking.

## When to use YSCV (and when not to)

**Use YSCV when** you need to deploy a trained model as a single binary without Python. When you're building a real-time CV pipeline on edge hardware. When you want to train a CNN or ViT on moderate-scale data without setting up a GPU cluster. When you need image processing that's faster than OpenCV and memory-safe.

**Don't use YSCV when** you need the Python ML ecosystem — Hugging Face model hub, thousands of community architectures, Jupyter notebook prototyping, dynamic graph debugging with breakpoints. When you're training foundation models across thousands of GPUs with NCCL. For that, use PyTorch.

YSCV is faster than PyTorch on individual operations (85 benchmark wins across all competitors, with 0 losses), but PyTorch has a decade-old ecosystem with millions of pretrained models and research tooling. Different tools for different jobs. Train in PyTorch, export to ONNX, deploy with YSCV — or train directly in YSCV if your model fits within our 39 layer types and you don't need the Python ecosystem.

## License

Licensed under either of [Apache 2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at your option.
