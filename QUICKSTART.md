# Quickstart

Five minutes from `cargo new` to a working program. Three flavours — pick
the one that matches what you're trying to do, run the commands, copy the
code, you're done.

> Need help choosing? **Want to process images and detect objects** →
> [§1](#1-image-processing--detection-cv-user). **Want to train a model
> from scratch** → [§2](#2-train-a-neural-network-ml-user). **Want to
> deploy on a Rockchip / edge device** → [§3](#3-edge-deployment-rockchip-npu).

---

## 1. Image processing + detection (CV user)

You have an image, you want to do something with it. 3 minutes.

```bash
cargo new my-cv-app && cd my-cv-app
```

Add to `Cargo.toml`:

```toml
[dependencies]
yscv = "0.1.9"

[profile.release]
lto = "thin"
codegen-units = 1
```

`src/main.rs`:

```rust
use yscv::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load → process → save.
    let img = imread("input.jpg")?;
    let gray = rgb_to_grayscale(&img)?;
    let blurred = gaussian_blur(&gray, 5, 1.5)?;
    imwrite("output.png", &blurred)?;
    println!("Saved output.png");
    Ok(())
}
```

```bash
cargo run --release
```

That's it. **160 image-processing operations** are available — find the
ones you need in [docs/cookbook.md](docs/cookbook.md). For YOLO
detection with bbox decode + NMS, see
[§Object detection](docs/cookbook.md#yolo-object-detection).

---

## 2. Train a neural network (ML user)

You have data, you want a model. 5 minutes.

`Cargo.toml` — same as above. `src/main.rs`:

```rust
use yscv::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Build the model — sequential CNN, 3-channel input, 10-class output.
    let mut graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    model.add_conv2d_zero(3, 16, 3, 3, 1, 1, true)?;
    model.add_relu();
    model.add_flatten();
    model.add_linear_zero(&mut graph, 16 * 30 * 30, 10)?;

    // 2. Synthetic data (replace with your own dataset loader).
    let inputs = Tensor::randn(vec![32, 3, 32, 32])?;
    let targets = Tensor::from_vec(vec![32], (0..32).map(|i| (i % 10) as f32).collect())?;

    // 3. Train.
    let result = Trainer::new(TrainerConfig {
        optimizer: OptimizerKind::Adam { lr: 0.001 },
        loss: LossKind::CrossEntropy,
        epochs: 50,
        batch_size: 32,
        validation_split: Some(0.2),
        ..Default::default()
    }).fit(&mut model, &mut graph, &inputs, &targets)?;

    println!("Final loss: {:.4}", result.final_loss);
    Ok(())
}
```

```bash
cargo run --release
```

For real datasets (COCO, ImageFolder, JSONL), see
[docs/dataset-adapters.md](docs/dataset-adapters.md). For pretrained
weights (ResNet, ViT, MobileNet — 17 architectures), see
[§Model zoo in cookbook](docs/cookbook.md#image-classification-with-pretrained-model).

---

## 3. Edge deployment (Rockchip NPU)

You have a Rock4D / RV1106 / RK3576 board and a `.rknn` model. 5 minutes.

```bash
cargo new my-drone-app && cd my-drone-app
```

`Cargo.toml`:

```toml
[dependencies]
yscv-pipeline = { version = "0.1.9", features = ["rknn", "realtime"] }

[profile.release]
lto = "thin"
codegen-units = 1
```

`config.toml` (next to your binary):

```toml
board = "rock4d"

[camera]
device = "/dev/video0"
format = "nv12"
width = 1280
height = 720
fps = 60

[output]
kind = "null"          # or "drm", "v4l2-out", "file" — see pipeline-config.md

[encoder]
kind = "mpp-h264"      # hardware encoder; "soft-h264" works on any host
bitrate_kbps = 8000
profile = "main"

[[tasks]]
name = "detector"
model_path = "yolov8n.rknn"     # or .onnx — auto-compiles to .rknn at startup
accelerator = { kind = "rknn", core = "core0" }
inputs  = [{ name = "images", source = "camera" }]
outputs = []

[realtime]
sched_fifo = true
cpu_governor = "performance"
prio.dispatch = 70
affinity.dispatch = [4, 5, 6]
```

`src/main.rs`:

```rust
use yscv_pipeline::{PipelineConfig, run_pipeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = PipelineConfig::from_toml_path("config.toml")?;
    let handle = run_pipeline(cfg)?;          // validates + builds dispatchers + applies RT

    loop {
        let frame_bytes = capture_camera_frame();         // your camera code
        let outputs = handle.dispatch_frame(&[("images", &frame_bytes)])?;
        let detections = &outputs["detector.output0"];    // raw f32 LE bytes
        process_detections(detections);                   // your post-processing
    }
}

fn capture_camera_frame() -> Vec<u8> { todo!("V4l2Camera or your source") }
fn process_detections(_bytes: &[u8]) { /* parse YOLO output, draw bboxes, etc. */ }
```

```bash
# On dev machine — cross-compile for aarch64 Linux
cargo build --release --target aarch64-unknown-linux-gnu

# Or build directly on the board
cargo build --release --features "rknn realtime"
```

That's it. The framework:

- **Validates** the config + model file (RKNN magic-byte check, optionally
  a full SDK load) before any real-time threads start
- **Auto-compiles** ONNX → RKNN if you point `model_path` at `.onnx`,
  caches the result next to the source
- **Auto-recovers** the NPU on transient hangs (`TIMEOUT`, `CTX_INVALID`)
- **Pipelines** across all 3 NPU cores when configured (3× throughput)
- **Applies** SCHED_FIFO + CPU affinity + `mlockall` + cpufreq governor
  if `[realtime]` is set, gracefully falls back if you don't have
  `CAP_SYS_NICE` / `CAP_SYS_ADMIN`

See [docs/edge-deployment.md](docs/edge-deployment.md) for the full
RKNN guide (DMA-BUF zero-copy, SRAM allocation, MPP zero-copy from
hardware decoder, custom OpenCL ops).

---

## What to read next

| Goal | Doc |
|---|---|
| Full feature catalogue | [README.md](README.md) |
| Step-by-step tutorial through every layer | [docs/getting-started.md](docs/getting-started.md) |
| Recipes for specific tasks | [docs/cookbook.md](docs/cookbook.md) |
| TOML config schema reference | [docs/pipeline-config.md](docs/pipeline-config.md) |
| RKNN / Rockchip deep dive | [docs/edge-deployment.md](docs/edge-deployment.md) |
| Things broken? Look here first | [docs/troubleshooting.md](docs/troubleshooting.md) |
| What every example does | [examples/README.md](examples/README.md) |
| How fast is fast | [docs/performance-benchmarks.md](docs/performance-benchmarks.md) |
| What yscv can do today | [docs/ecosystem-capability-matrix.md](docs/ecosystem-capability-matrix.md) |
