# yscv examples

Each example is self-contained and runs with a single `cargo run`
command. They're grouped by what you're trying to learn — start at the
top of whichever section matches your goal.

> **First time here?** Read [QUICKSTART.md](../QUICKSTART.md) first —
> it's the 5-minute path to a working program.

---

## Image processing

| Example | What it shows |
|---|---|
| **[image_processing.rs](src/image_processing.rs)** | Load → grayscale → blur → save. The "hello world" of yscv. |
| **[image_pipeline.rs](src/image_pipeline.rs)** | Multi-stage preprocessing chain (resize → normalize → augment) you'd build before feeding a model. |

```bash
cargo run --release --example image_processing
cargo run --release --example image_pipeline
```

## Detection

| Example | What it shows |
|---|---|
| **[classify_image.rs](src/classify_image.rs)** | One-shot ImageNet classification using a pretrained model from the model zoo. |
| **[detect_objects.rs](src/detect_objects.rs)** | YOLO-style detection with bounding boxes. |
| **[yolo_detect.rs](src/yolo_detect.rs)** | YOLOv8/v11 with the full pipeline — preprocess, run, NMS, decode bboxes. Accepts any opset 22 ONNX. |
| **[yolo_vis.rs](src/yolo_vis.rs)** | YOLO + draw bounding boxes onto the image, save annotated output. |
| **[yolo_finetune.rs](src/yolo_finetune.rs)** | Fine-tune a YOLO detection head on your own labelled data. |

```bash
cargo run --release --example yolo_detect -- yolov8n.onnx photo.jpg
cargo run --release --example yolo_vis    -- yolov8n.onnx photo.jpg out.png
```

## Training

| Example | What it shows |
|---|---|
| **[train_linear.rs](src/train_linear.rs)** | Linear regression on synthetic data — autograd + SGD end-to-end in 50 lines. |
| **[train_cnn.rs](src/train_cnn.rs)** | Train a small CNN with the `Trainer` API: optimizer, loss, validation split, epochs. |

```bash
cargo run --release --example train_linear
cargo run --release --example train_cnn
```

## ONNX inference (CPU + GPU)

| Example | What it shows | Features |
|---|---|---|
| **[bench_yolo.rs](src/bench_yolo.rs)** | YOLO inference timing on CPU vs onnxruntime. | none |
| **[bench_gpu.rs](src/bench_gpu.rs)** | wgpu (Vulkan / Metal / DX12) inference benchmark. | `gpu` |
| **[bench_mpsgraph.rs](src/bench_mpsgraph.rs)** | Apple MPSGraph: CPU vs Metal-per-op vs MPSGraph for any ONNX. | `metal-backend` |
| **[bench_mpsgraph_only.rs](src/bench_mpsgraph_only.rs)** | MPSGraph alone, focused on plan compile + dispatch latency. | `metal-backend` |
| **[bench_metal_yolo.rs](src/bench_metal_yolo.rs)** | YOLO on Metal-per-op (Winograd + MPS GEMM). | `metal-backend` |
| **[bench_metal_conv.rs](src/bench_metal_conv.rs)** | Single-op Conv2d benchmark on Metal — for diagnosing kernel perf. | `metal-backend` |
| **[bench_metal_vball.rs](src/bench_metal_vball.rs)** | VballNet (DSConv) inference on Metal vs CPU. | `metal-backend` |
| **[bench_mps_gemm.rs](src/bench_mps_gemm.rs)** | Raw MPS GEMM benchmark — establishes the "ceiling" for matmul-bound models. | `metal-backend` |
| **[debug_metal_yolo.rs](src/debug_metal_yolo.rs)** | Per-op profiling for YOLO on Metal — finds which op is slow. | `metal-backend` |
| **[bench_vball_cpu.rs](src/bench_vball_cpu.rs)** | VballNet CPU inference for comparison against the Metal version. | none |

```bash
cargo run --release --example bench_yolo
cargo run --release --features metal-backend --example bench_mpsgraph
```

## Video

| Example | What it shows |
|---|---|
| **[bench_video_decode.rs](src/bench_video_decode.rs)** | H.264 / HEVC / AV1 decoder benchmark vs ffmpeg. |

```bash
cargo run --release --example bench_video_decode -- video.mp4
```

## Pipeline framework (TOML-driven)

| Example | What it shows | Features |
|---|---|---|
| **[edge_pipeline_v2.rs](src/edge_pipeline_v2.rs)** ⭐ | **Recommended starting point.** Full TOML → pipeline → hot loop with latency reporting + recovery on transient faults. Drop-in replacement for the old `edge_pipeline.rs`. | depends on TOML accelerators |
| **[board_pipeline.rs](src/board_pipeline.rs)** | Validates a TOML config and prints the topologically-sorted task order — useful for sanity-checking a new board file before deploying. | none |
| **[edge_pipeline.rs](src/edge_pipeline.rs)** | Original synthetic FPV pipeline using `ContextPool` directly. Kept for reference; new code should use `edge_pipeline_v2.rs`. | `rknn` (optional) |

```bash
# Validate a TOML config (no real hardware needed):
cargo run --release --example board_pipeline -- boards/rock4d.toml

# Run the full pipeline (build with the features your TOML asks for):
cargo run --release --features "rknn realtime" --example edge_pipeline_v2 -- boards/rock4d.toml 600

# CPU-only smoke test on macOS:
cargo run --release --example edge_pipeline_v2 -- boards/cpu-only.toml 10
```

---

## Choosing a feature combination

The "Features" column above tells you which `--features` flag(s) the
example needs. Defaults:

```bash
cargo run --release --example <name>                              # CPU only
cargo run --release --features gpu --example <name>               # + wgpu cross-platform GPU
cargo run --release --features metal-backend --example <name>     # + Metal-native (macOS)
cargo run --release --features rknn --example <name>              # + Rockchip NPU (Linux ARM)
cargo run --release --features "rknn metal-backend gpu realtime" --example <name>   # everything
```

For build performance, combine features in a single `cargo build` —
incremental compile shares the upstream artifacts.

## Running on real hardware

Most examples accept a `model.onnx` / image / video as a CLI argument.
For inference benchmarks, point them at any opset-22 ONNX:

```bash
# Download a YOLO model (any export from ultralytics works)
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx

cargo run --release --example yolo_detect -- yolov8n.onnx photo.jpg
cargo run --release --features metal-backend --example bench_mpsgraph -- yolov8n.onnx 50
```

For the pipeline framework, see [docs/pipeline-config.md](../docs/pipeline-config.md)
for the TOML schema and sample configs.

## Got stuck?

| Symptom | Doc |
|---|---|
| Build fails, missing system lib | [docs/troubleshooting.md](../docs/troubleshooting.md) |
| `librknnrt.so not found` on Mac/dev host | Expected — RKNN dispatcher only loads on the actual board |
| `validate_models: ModelInvalid` | Wrong file extension or corrupted model — check the magic-byte error message |
| Inference works but slow | [docs/performance-benchmarks.md](../docs/performance-benchmarks.md) for expected numbers per backend |
| Want to write a custom op | [docs/architecture.md](../docs/architecture.md) explains kernel dispatch |
