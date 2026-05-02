# yscv — Project Context

Current state of the yscv framework.

Project priority is CPU inference on edge devices (Cortex-A SBCs,
low-power x86, drone boards). Other backends — wgpu, MPSGraph, RKNN,
BLAS — are opt-in extensions and keep widening. See
[`README.md`](README.md) and [`AGENTS.md`](AGENTS.md) for the rules
and PR workflow.

## Architecture

18 library crates, 3 apps, 1 examples crate.

```
yscv (umbrella re-export)
├── yscv-tensor          ← N-dimensional tensor, f32/f16/bf16, SIMD ops
├── yscv-kernels         ← CPU + GPU compute backends, SIMD, RKNN NPU
├── yscv-threadpool      ← Work-stealing pool + rayon scope abstraction, opt-in PersistentSection
├── yscv-autograd        ← Reverse-mode autodiff, BackwardOps routing
├── yscv-optim           ← 8 optimizers + Lookahead, 11 LR schedulers
├── yscv-model           ← 39 layers, 17 architectures, LoRA, trainer
├── yscv-imgproc         ← 160 image ops, u8/f32 SIMD
├── yscv-video           ← H.264/HEVC/AV1 decode, V4L2, MJPEG, H.264 encode, MAVLink, overlay, framebuffer
├── yscv-video-mpp       ← Rockchip MPP hardware video encoder
├── yscv-detect          ← YOLOv8/v11, NMS, heatmap, RoI align
├── yscv-recognize       ← Cosine matching, VP-Tree ANN
├── yscv-track           ← DeepSORT, ByteTrack, Kalman
├── yscv-eval            ← Metrics, 8 dataset adapters
├── yscv-onnx            ← 122-op ONNX runtime, INT4/INT8 quantization (incl. fused PW->DW + DW->PW chain actions with SIMD requant epilogue), GPU, Metal, KV-cache, generation
├── yscv-pipeline        ← Pipeline framework: ONNX/Metal/RKNN dispatch, TOML config
└── yscv-cli             ← Inference CLI, camera diagnostics
```

## Metrics

| Metric | Value |
|--------|-------|
| Tests | **1,861** |
| ONNX operators | **122** |
| Tensor methods | **159** |
| Imgproc ops | **160** |
| Model architectures | **17** |
| WGSL + Metal shaders | **61 + 4** |
| SAFETY-commented unsafe blocks | **220** |
| Fuzz targets | **5** |

## Build

- Edition 2024, MSRV 1.94
- Release: `lto = "thin"`, `codegen-units = 1`
- BLAS: Accelerate (macOS), OpenBLAS (Linux)
- GPU: wgpu (Vulkan/Metal/DX12) + Metal-native
- NPU: RKNN (Rockchip, `--features rknn`)
