# yscv — Project Context

Current state of the yscv framework.

## Architecture

16 library crates, 2 apps, 1 examples crate.

```
yscv (umbrella re-export)
├── yscv-tensor          ← N-dimensional tensor, f32/f16/bf16, SIMD ops
├── yscv-kernels         ← CPU + GPU compute backends, SIMD, RKNN NPU
├── yscv-autograd        ← Reverse-mode autodiff, BackwardOps routing
├── yscv-optim           ← 8 optimizers + Lookahead, 11 LR schedulers
├── yscv-model           ← 39 layers, 17 architectures, LoRA, trainer
├── yscv-imgproc         ← 160 image ops, u8/f32 SIMD
├── yscv-video           ← H.264/HEVC/AV1 decode, V4L2, MJPEG, H.264 encode, MAVLink, overlay, framebuffer
├── yscv-detect          ← YOLOv8/v11, NMS, heatmap, RoI align
├── yscv-recognize       ← Cosine matching, VP-Tree ANN
├── yscv-track           ← DeepSORT, ByteTrack, Kalman
├── yscv-eval            ← Metrics, 8 dataset adapters
├── yscv-onnx            ← 128+ op ONNX runtime, INT4/INT8 quantization, GPU, Metal, KV-cache, generation
└── yscv-cli             ← Inference CLI, camera diagnostics
```

## Metrics

| Metric | Value |
|--------|-------|
| Tests | **1,807** |
| ONNX operators | **130** |
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
