# yscv — Project Context

This document describes the current state of the yscv framework.

## Architecture overview

yscv is a monorepo Cargo workspace with 14 library crates, 2 application binaries, and an examples crate.

```
yscv (umbrella re-export)
├── yscv-tensor          ← 115 ops, f32/f16/bf16, 50 SIMD functions
├── yscv-kernels         ← 61 kernel ops, 49 SIMD, 20 GPU WGSL shaders
├── yscv-autograd        ← dynamic computation graph, 40+ backward ops
├── yscv-optim           ← 8 optimizers (SGD/Adam/AdamW/RAdam/RmsProp/Adagrad/Lamb/Lars), 11 LR schedulers
├── yscv-model           ← 39 layer types, 13 model zoo architectures, 17 losses
├── yscv-imgproc         ← 178 image ops, u8 NEON/SSE/AVX SIMD, GCD/rayon threading
├── yscv-video           ← H.264/HEVC decode (23K LOC, 4.5× ffmpeg), MP4/MKV demux, HW decode (VT/VAAPI/NVDEC/MF), camera I/O, audio metadata
├── yscv-detect          ← YOLOv8 pipeline, NMS, heatmap, RoI align
├── yscv-recognize       ← cosine matching, VP-Tree ANN
├── yscv-track           ← DeepSORT, ByteTrack, Kalman, re-id
├── yscv-eval            ← 41 metrics, 11 dataset formats
├── yscv-onnx            ← 128+ op ONNX runtime, quantization, graph optimizer, Metal GPU
└── yscv-cli             ← CLI for inference, benchmarking, evaluation
```

## Codebase metrics

| Metric | Value |
|--------|-------|
| Total Rust LOC | **140,260** |
| .rs files | **397** |
| Tests | **1,678** |
| SIMD functions | **298** (NEON + SSE + AVX) |
| GPU WGSL shaders | **17** |
| Benchmark operations | **100+** |

## Build and toolchain

- **Edition**: 2024 (Rust 1.88+)
- **Release profile**: `lto = "thin"`, `codegen-units = 1`
- **Target CPU flags**: `apple-m1` (macOS ARM), `neoverse-n1` (Linux ARM), `x86-64-v3` (AVX2)
- **BLAS**: Accelerate (macOS), OpenBLAS (Linux/Windows)
- **GPU**: wgpu compute shaders (Vulkan/Metal/DX12) + Metal-native backend (`--features metal-backend`)

## SIMD strategy

Three-tier dispatch with runtime feature detection:
1. **aarch64 NEON** — 106 functions (tensor + kernels + imgproc + optimizers)
2. **x86_64 AVX/AVX2/SSE** — 69 AVX + 25 SSE2 + 10 SSSE3 + 6 AVX2 functions
3. **Scalar fallback** — always present for all platforms including RISC-V, WASM, and Miri

All dispatch functions have `#[inline]` for cross-crate inlining.

## Performance vs competitors

Across 100+ benchmarked operations against NumPy 2.0, PyTorch 2.8, OpenCV 4.13, onnxruntime 1.19, and Apple CoreML (Apple M1 MacBook Air, March 2026):

- **80 wins** — faster than all competitors
- **~4 parity** — within 10%
- **0 losses**

Key wins: sigmoid **6.0×** vs PyTorch, relu **6.2×** vs NumPy, resize nearest **3.3×** vs OpenCV, resize bilinear **3.0×** vs OpenCV, sobel u8 **2.3×** vs OpenCV, softmax **2.2×** vs PyTorch, ONNX CPU **3.2×** vs onnxruntime (YOLOv8n), VballNet CPU **1.6×** vs onnxruntime, MPSGraph GPU **3.4× faster** than CoreML on YOLOv8n (4.8ms vs 16.1ms — CoreML uses dedicated Neural Engine hardware), VballNet MPSGraph **7.8ms beats CoreML 8.6ms** (1.1×). YOLO11n: only runtime that runs it (CPU + GPU), competitors all fail.

## Framework features

- **Training**: Trainer API, DataLoader, 20 optimizers, 11 schedulers, mixed precision, LoRA, gradient checkpointing, distributed (AllReduce + pipeline parallel + tensor sharding)
- **Inference**: LinearLayer inference mode, ONNX runtime, quantization (INT8), weight pruning, Flash Attention
- **Vision**: 178 imgproc ops, FAST/ORB/SIFT/SURF, optical flow, contours, homography
- **Video**: H.264 full decoder, HEVC full pipeline, MP4 reader, camera I/O
- **Detection+Tracking**: YOLOv8 + ByteTrack + DeepSORT, MaskHead for segmentation
- **Model Zoo**: 13 architectures (ResNet18/34/50/101, VGG16/19, MobileNetV2, EfficientNetB0, AlexNet, ViTTiny/Base/Large, DeiTTiny)
- **Evaluation**: 41 metrics, 11 dataset formats

## Testing and CI

- **1,678 tests** across the workspace
- CI: GitHub Actions on Ubuntu/macOS/Windows + ARM64 Linux
- Quality gates: `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test --workspace --release`

## Key documentation

- `docs/performance-benchmarks.md` — benchmark scorecard and methodology
- `docs/architecture.md` — SIMD/threading/crate layer guide
- `docs/ecosystem-capability-matrix.md` — capability map
- `docs/api-stability.md` — versioning policy
- `docs/training-optimizers.md` — training API guide
