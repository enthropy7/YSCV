# yscv

Umbrella crate that re-exports the entire yscv framework. One dependency gives you everything.

```toml
[dependencies]
yscv = "0.1"
```

```rust
use yscv::prelude::*;

let img = imread("photo.jpg")?;
let gray = rgb_to_grayscale(&img)?;
imwrite("gray.png", &gray)?;
```

## Included Crates

| Crate | What it does |
|-------|-------------|
| `yscv-tensor` | SIMD tensor ops, f32/f16/bf16, broadcasting |
| `yscv-kernels` | CPU/GPU compute backends (wgpu, Metal) |
| `yscv-imgproc` | 100+ image processing operations |
| `yscv-video` | H.264/HEVC decode, MP4 parse, camera I/O |
| `yscv-detect` | YOLOv8 detection, NMS, heatmaps |
| `yscv-track` | DeepSORT, ByteTrack, Kalman filter |
| `yscv-recognize` | Face/object recognition, VP-Tree ANN |
| `yscv-eval` | mAP, MOTA, HOTA metrics |
| `yscv-onnx` | ONNX runtime, 128+ operators, graph opt |
| `yscv-optim` | SGD, Adam, LAMB, LR schedulers |
| `yscv-model` | 50+ layers, ViT, ResNet, training loop |
| `yscv-autograd` | Tape-based reverse-mode autodiff |
| `yscv-cli` | CLI for inference and benchmarking |
