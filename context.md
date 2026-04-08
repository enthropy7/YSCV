# yscv — Project Context

This document describes the current state of the yscv framework.

## Architecture overview

yscv is a monorepo Cargo workspace with 14 library crates, 2 application binaries, and an examples crate.

```
yscv (umbrella re-export)
├── yscv-tensor          ← 115 ops in `ops.rs`, f32/f16/bf16
├── yscv-kernels         ← 128 public kernel fns, 50 WGSL + 4 Metal compute shaders
├── yscv-autograd        ← dynamic computation graph, 61 `Op` variants
├── yscv-optim           ← 8 optimizers (SGD/Adam/AdamW/RAdam/RmsProp/Adagrad/Lamb/Lars) + Lookahead, 11 LR schedulers
├── yscv-model           ← 39 layer types, 13 model zoo architectures, 17 losses
├── yscv-imgproc         ← 159 image ops in `ops/`, u8 NEON/SSE/AVX SIMD, GCD/rayon threading
├── yscv-video           ← H.264/HEVC decode (4.5×/1.4× ffmpeg), MP4/MKV demux, HW decode (VT/VAAPI/NVDEC/MF), camera I/O, audio metadata
├── yscv-detect          ← YOLOv8 + YOLOv11 pipelines, NMS, heatmap, RoI align
├── yscv-recognize       ← cosine matching, VP-Tree ANN, Recognizer
├── yscv-track           ← DeepSORT, ByteTrack, Kalman, re-id
├── yscv-eval            ← 37 public metric/eval fns, 8 dataset adapters
├── yscv-onnx            ← 128 op ONNX CPU runtime, quantization, graph optimizer, Metal/MPSGraph GPU
└── yscv-cli             ← CLI for inference, camera diagnostics, dataset evaluation
```

## Codebase metrics

| Metric | Value |
|--------|-------|
| Workspace version | **0.1.7** |
| Tests (passing) | **1,693** |
| `#[target_feature]`-gated SIMD functions | **315** (NEON + SSE/SSE2/SSSE3 + AVX/AVX2) |
| GPU compute shaders | **54** (50 WGSL + 4 Metal) |
| Examples in `examples/src/` | **21** |

## Build and toolchain

- **Edition**: 2024 (`rust-version = "1.92"` in workspace `Cargo.toml`)
- **Release profile**: `lto = "thin"`, `codegen-units = 1`
- **Target CPU flags**: `apple-m1` (macOS ARM), `neoverse-n1` (Linux ARM), `x86-64-v3` (AVX2)
- **BLAS**: Accelerate (macOS), OpenBLAS (Linux/Windows)
- **GPU**: wgpu compute shaders (Vulkan/Metal/DX12) + Metal-native backend (`--features metal-backend`)

## SIMD strategy

Three-tier dispatch with runtime feature detection (315 `#[target_feature]`-gated functions across the workspace):
1. **aarch64 NEON** — compile-time gated, dominant on Apple Silicon and ARM Linux
2. **x86_64 AVX / AVX2 / SSE / SSE2 / SSSE3** — runtime-detected via `is_x86_feature_detected!`
3. **Scalar fallback** — always present for all platforms including RISC-V, WASM, and Miri

All dispatch functions have `#[inline]` for cross-crate inlining.

## Performance vs competitors

Across 100+ benchmarked operations against NumPy 2.0, PyTorch 2.8, OpenCV 4.13, onnxruntime 1.19, and Apple CoreML (Apple M1 MacBook Air, March 2026):

- **80 wins** — faster than all competitors
- **~4 parity** — within 10%
- **0 losses**

Key wins: sigmoid **6.0×** vs PyTorch, relu **6.2×** vs NumPy, resize nearest **3.3×** vs OpenCV, resize bilinear **3.0×** vs OpenCV, sobel u8 **2.3×** vs OpenCV, softmax **2.2×** vs PyTorch, ONNX CPU **3.2×** vs onnxruntime (YOLOv8n), VballNet CPU **1.6×** vs onnxruntime, MPSGraph GPU **3.4× faster** than CoreML on YOLOv8n (4.8ms vs 16.1ms — CoreML uses dedicated Neural Engine hardware), VballNet MPSGraph **7.8ms beats CoreML 8.6ms** (1.1×). YOLO11n: only runtime that runs it (CPU + GPU), competitors all fail.

## Framework features

- **Training**: Trainer API, DataLoader, 8 optimizers + Lookahead, 11 schedulers, 17 losses, mixed precision, LoRA, gradient checkpointing, distributed (AllReduce + pipeline parallel + tensor sharding)
- **Inference**: LinearLayer inference mode, ONNX runtime, quantization (INT8), weight pruning, Flash Attention
- **Vision**: 159 imgproc free functions, FAST/ORB/SIFT/SURF, optical flow, contours, homography
- **Video**: H.264 full decoder, HEVC full pipeline (Main + Main10), MP4 + MKV readers, hardware decode backends (VT/VAAPI/NVDEC/MF), camera I/O via `nokhwa`, audio metadata for AAC/ALAC/Opus/Vorbis/MP3/FLAC
- **Detection + Tracking**: YOLOv8 + YOLOv11 + ByteTrack + DeepSORT, MaskHead for segmentation
- **Model Zoo**: 13 architectures (ResNet18/34/50/101, VGG16/19, MobileNetV2, EfficientNetB0, AlexNet, ViTTiny/Base/Large, DeiTTiny)
- **Evaluation**: 37 public metric/eval functions, 8 dataset adapters

## Testing and CI

- **1,693 tests** across the workspace
- CI: GitHub Actions on Ubuntu/macOS/Windows + ARM64 Linux
- Quality gates: `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test --workspace --release`

## Key documentation

- `docs/performance-benchmarks.md` — benchmark scorecard and methodology
- `docs/architecture.md` — SIMD/threading/crate layer guide
- `docs/ecosystem-capability-matrix.md` — capability map
- `docs/api-stability.md` — versioning policy
- `docs/training-optimizers.md` — training API guide
