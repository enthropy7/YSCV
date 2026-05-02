# yscv

Umbrella crate that re-exports the entire yscv framework. One dependency gives you everything.

```toml
[dependencies]
yscv = "0.1.8"
```

```rust,ignore
use yscv::prelude::*;

let img = imread("photo.jpg")?;
let gray = rgb_to_grayscale(&img)?;
imwrite("gray.png", &gray)?;
```

## Included Crates

The yscv workspace ships 18 library crates; this umbrella re-exports them all behind a single prelude:

| Crate | What it does |
|-------|-------------|
| `yscv-tensor` | 115 `Tensor` ops, f32/f16/bf16, broadcasting, SIMD |
| `yscv-kernels` | CPU/GPU compute backends (61 WGSL + 4 Metal shaders, wgpu + metal-rs) |
| `yscv-autograd` | Tape-based reverse-mode autodiff with 61 `Op` variants |
| `yscv-optim` | 8 optimizers (SGD/Adam/AdamW/RAdam/RmsProp/Adagrad/Lamb/Lars) + Lookahead, 11 LR schedulers |
| `yscv-model` | 39 layer types, 17 loss functions, Trainer API, model zoo (17 architectures), LoRA |
| `yscv-imgproc` | 160 image processing operations (NEON/AVX2/SSE/SSSE3 + scalar) |
| `yscv-video` | H.264/HEVC/AV1 decode (incl. hardware backends), MP4/MKV parse, camera I/O, audio metadata |
| `yscv-detect` | YOLOv8 + YOLOv11 detection, NMS, heatmaps, RoI align |
| `yscv-track` | DeepSORT, ByteTrack, Kalman filter, Hungarian, ReId |
| `yscv-recognize` | Cosine similarity, VP-Tree ANN, `Recognizer` enroll/match |
| `yscv-eval` | Classification/detection/tracking/regression metrics, 8 dataset adapters |
| `yscv-onnx` | 122-op ONNX CPU runtime, INT4/INT8 quantization, LLM generation, graph optimizer, Metal/MPSGraph backend |
| `yscv-pipeline` | TOML-driven multi-accelerator dispatch, RT scheduling, recovery |
| `yscv-video-mpp` | Rockchip MPP hardware encoder (H.264, H.265) |
| `yscv-cli` | CLI for inference, camera diagnostics, dataset evaluation |
| `yscv` | This umbrella crate (`prelude` + re-exports) |
