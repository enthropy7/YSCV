# Getting Started with yscv

A progressive tutorial. Start at the top, stop when you have what you
need. Each section is ~5 minutes; each builds on the previous one.

> **Already have a specific goal?** Jump to [QUICKSTART](../QUICKSTART.md)
> for the 5-minute path for CV / training / edge deployment.

---

## What is yscv?

A **pure-Rust computer-vision and ML framework**. Single workspace,
18 crates, zero Python or C++ runtime dependencies. Compiles to one
statically-linked binary that runs the same way on a Mac dev box, a
Linux server, and a Rockchip drone board.

Three things it's optimised for:

1. **Inference deployment** — train a model wherever (PyTorch, your
   own data, etc.), export to ONNX, run with yscv. Faster than
   onnxruntime on CPU + Apple GPU. Auto-compiles to RKNN for Rockchip
   NPUs at startup.
2. **Edge / real-time pipelines** — TOML-driven multi-accelerator
   dispatch, NPU-hang auto-recovery, SCHED_FIFO + CPU affinity +
   DVFS governor wired through one config file.
3. **CV ergonomics** — 160 image-processing ops, full YOLOv8/v11
   pipeline (preprocess + run + NMS + decode), DeepSORT / ByteTrack
   tracker, video codec stack faster than ffmpeg.

It's **not** trying to replace PyTorch for foundation-model training.
It's the deployment + edge inference layer for models you already
have.

---

## Step 1 — Install

You need Rust 1.94 or newer.

```bash
cargo new my-yscv-project && cd my-yscv-project
```

Add to `Cargo.toml`:

```toml
[dependencies]
yscv = "0.1.8"

[profile.release]
lto = "thin"
codegen-units = 1
```

Optional system deps (skip until you need them):

- **OpenBLAS** for faster matmul on Linux/Windows. macOS uses
  Accelerate automatically.
  ```bash
  apt install libopenblas-dev   # Debian/Ubuntu
  brew install openblas         # macOS (rarely needed; Accelerate already there)
  ```
- **protoc** for ONNX proto. Without it, a built-in fallback is used.
  ```bash
  apt install protobuf-compiler
  brew install protobuf
  ```

That's the whole install. Everything else (RKNN runtime, Metal
frameworks) is loaded at runtime via `dlopen` on the platforms that
need them.

---

## Step 2 — First program: image processing

`src/main.rs`:

```rust
use yscv::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let img = imread("input.jpg")?;
    let gray = rgb_to_grayscale(&img)?;
    let blurred = gaussian_blur(&gray, 5, 1.5)?;
    imwrite("output.png", &blurred)?;
    Ok(())
}
```

```bash
cargo run --release
```

Done. The `prelude` module re-exports the most common types — `Tensor`,
`imread`/`imwrite`, every basic op — so you don't have to track
which crate has what. Browse the prelude for the full menu, or read
[docs/cookbook.md](cookbook.md) for recipes by task.

---

## Step 3 — Run a pre-trained ONNX model

Download any opset-22 ONNX (YOLOv8n is a good test):

```bash
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx
```

```rust
use yscv::prelude::*;
use yscv_onnx::{load_onnx_model_from_file, run_onnx_model};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = load_onnx_model_from_file("yolov8n.onnx")?;
    let img = imread("photo.jpg")?;

    // YOLO expects 1×3×640×640 NCHW f32 normalised to [0,1].
    let resized = resize_bilinear(&img, 640, 640)?;
    let nchw = hwc_to_nchw(&resized)?.to_dtype(DType::F32) / 255.0;

    let mut inputs = HashMap::new();
    inputs.insert("images".to_string(), nchw);

    let outputs = run_onnx_model(&model, inputs)?;
    let detections = &outputs["output0"];
    println!("Output shape: {:?}", detections.shape());
    Ok(())
}
```

```bash
cargo run --release
```

This is the **CPU runner** — pure Rust, 122 ONNX ops, NEON/AVX
SIMD. For YOLO post-processing (decoding bbox grid + NMS) see
[examples/src/yolo_detect.rs](../examples/src/yolo_detect.rs) — it's
~150 lines and you can copy-paste the relevant bits.

### Speed it up — Apple Silicon GPU

On macOS, add `metal-backend` to the feature list:

```toml
yscv = { version = "0.1.8", features = ["metal-backend"] }
```

```rust
use yscv_onnx::{compile_mpsgraph_plan, run_mpsgraph_plan};

let plan = compile_mpsgraph_plan(&model, &[("images", &nchw)])?;
let outputs = run_mpsgraph_plan(&plan, &[("images", nchw.data())])?;
```

That's it. ~5× faster than the CPU runner on YOLO. For sustained
throughput across many frames, use the triple-buffered submit/wait
API (or read [`docs/mpsgraph-guide.md`](mpsgraph-guide.md) for the
full MPSGraph walkthrough — when to use sync vs pipelined, multi-input
models, fallback strategy, troubleshooting):

```rust
use yscv_onnx::{submit_mpsgraph_plan, wait_mpsgraph_plan, InferenceHandle};
use std::collections::VecDeque;

let mut in_flight: VecDeque<InferenceHandle> = VecDeque::new();
for _ in 0..3 {
    in_flight.push_back(submit_mpsgraph_plan(&plan, &feeds)?);
}
loop {
    let oldest = in_flight.pop_front().unwrap();
    let outputs = wait_mpsgraph_plan(&plan, oldest)?;
    process(outputs);
    in_flight.push_back(submit_mpsgraph_plan(&plan, &feeds)?);
}
```

This overlaps GPU compute with CPU marshaling — 3× throughput vs sync
on M-series Macs.

---

## Step 4 — Train a model

Replace your `Cargo.toml` deps if you only had `yscv = ".."`; the
prelude already includes training types.

```rust
use yscv::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    model.add_conv2d_zero(3, 16, 3, 3, 1, 1, true)?;
    model.add_relu();
    model.add_flatten();
    model.add_linear_zero(&mut graph, 16 * 30 * 30, 10)?;

    let inputs = Tensor::randn(vec![32, 3, 32, 32])?;
    let targets = Tensor::from_vec(vec![32], (0..32).map(|i| (i % 10) as f32).collect())?;

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

For real datasets:

```rust
let dataset = ImageFolder::open("data/train")?;
// or:
let dataset = JsonlDataset::open("data/train.jsonl")?;
```

See [docs/dataset-adapters.md](dataset-adapters.md) for COCO,
ImageManifest, CSV, ImageFolder, JSONL — the full list.

For pretrained weights (17 architectures: ResNet, ViT, MobileNet,
EfficientNet, DeiT, ...):

```rust
let hub = ModelHub::new();              // caches in ~/.yscv/models/
let weights = hub.load_weights("resnet50")?;
```

For optimizers (8: SGD through LARS), schedulers (11), and the rest
of the training surface, see
[docs/training-optimizers.md](training-optimizers.md).

---

## Step 5 — Detection + tracking + recognition

YOLOv8/v11 detection with bounding-box decode + NMS:

```rust
use yscv::detect::{detect_yolov8_from_rgb, yolov8_coco_config, non_max_suppression};

let img = imread("scene.jpg")?;
let dets = detect_yolov8_from_rgb(&img, &model, &yolov8_coco_config())?;
let kept = non_max_suppression(&dets, 0.5);
for d in &kept {
    println!("{}: {:.2}% at {:?}", d.class_name, d.score * 100.0, d.bbox);
}
```

Add tracking across frames (DeepSORT or ByteTrack):

```rust
use yscv::track::{ByteTracker, ByteTrackerConfig};

let mut tracker = ByteTracker::new(ByteTrackerConfig::default());
for (frame_idx, dets) in detection_stream.enumerate() {
    let tracks = tracker.update(&dets);
    for t in &tracks {
        println!("frame {frame_idx}: track {} at {:?}", t.track_id, t.bbox);
    }
}
```

Add recognition (face / object identity):

```rust
use yscv::recognize::{Recognizer, VpTreeIndex};

let mut rec = Recognizer::new(VpTreeIndex::new(0.6));
rec.enroll("alice", &alice_embedding)?;
rec.enroll("bob",   &bob_embedding)?;
let matched = rec.match_one(&query_embedding);
```

The full detect → track → recognize pipeline runs in **~67 µs per
frame** end-to-end on a single CPU core. That's 15,000 FPS headroom
to stack additional logic.

---

## Step 6 — Build a config-driven pipeline (edge / drone path)

The previous steps are great for prototyping. For a production edge
deployment — multiple models, accelerator dispatch, real-time
priorities — the framework offers a TOML-driven runtime.

`Cargo.toml`:

```toml
[dependencies]
yscv-pipeline = { version = "0.1.8", features = ["rknn", "realtime"] }
```

`config.toml`:

```toml
board = "rock4d"

[camera]
device = "/dev/video0"
format = "nv12"
width = 1280
height = 720
fps = 60

[output]
kind = "drm"          # display via Linux KMS; or "v4l2-out", "file", "null"
connector = "HDMI-A-1"
mode = "720p60"

[encoder]
kind = "mpp-h264"     # hardware H.264 on Rockchip
bitrate_kbps = 8000
profile = "main"

[[tasks]]
name = "detector"
model_path = "yolov8n.onnx"   # auto-compiles to .rknn at startup, caches result
accelerator = { kind = "rknn", core = "core0" }
inputs  = [{ name = "images", source = "camera" }]
outputs = []

[realtime]
sched_fifo   = true
cpu_governor = "performance"
prio.dispatch     = 70
affinity.dispatch = [4, 5, 6]
```

`src/main.rs`:

```rust
use yscv_pipeline::{PipelineConfig, run_pipeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = PipelineConfig::from_toml_path("config.toml")?;
    let handle = run_pipeline(cfg)?;
    // run_pipeline does: validate config → validate models (magic +
    // optional SDK dry-run load) → build dispatchers → apply RT
    // (SCHED_FIFO + affinity + mlockall + governor).

    loop {
        let bytes = capture_camera_frame();
        let outputs = handle.dispatch_frame(&[("images", &bytes)])?;
        process(&outputs["detector.output0"]);
    }
}
```

Same binary runs YOLO on Rock4D through NPU, on RV1106 through NPU,
or on your dev Mac through MPSGraph — only the TOML changes. See:

- [docs/pipeline-config.md](pipeline-config.md) for the full TOML schema
- [docs/edge-deployment.md](edge-deployment.md) for RKNN deep dive
- [examples/src/edge_pipeline_v2.rs](../examples/src/edge_pipeline_v2.rs)
  for the complete reference (synthetic camera, latency reporting,
  recovery, graceful shutdown)

### Recovery, watchdog, hot-reload

```rust
// On any transient failure, recover every dispatcher in one call.
if let Err(e) = handle.dispatch_frame(&feeds) {
    handle.recover_all()?;
}

// Auto-recover on stage budget overruns (needs --features realtime).
let arc_handle = std::sync::Arc::new(handle);
let _wd = arc_handle.clone().spawn_watchdog(stats, std::time::Duration::from_millis(100));

// Hot-swap a model in-flight without stopping the loop.
let pool: &yscv_kernels::RknnPipelinedPool = /* from your dispatcher setup */;
pool.reload(&std::fs::read("models/v2.rknn")?)?;
```

---

## Step 7 — Drop down to the kernel layer (when you need it)

The TOML / dispatcher path is the most ergonomic, but it carries a
small per-call overhead (`HashMap<String, Vec<u8>>` round trips,
boxed trait dispatch). For sub-millisecond hot loops — drone OSD
overlay, inference at 1000+ FPS — you can drop to the kernel layer
directly.

```rust
use yscv_kernels::{RknnPipelinedPool, NpuCoreMask};

// Skip the TOML, skip the dispatcher.
let model = std::fs::read("yolov8n.rknn")?;
let pool = RknnPipelinedPool::new(
    &model,
    &[NpuCoreMask::Core0, NpuCoreMask::Core1, NpuCoreMask::Core2],
)?;

// Three frames in flight at once.
let h0 = pool.submit(&[("images", &frame0)])?;
let h1 = pool.submit(&[("images", &frame1)])?;
let out0 = pool.wait(h0)?;     // already done, NPU was running while CPU marshaled
let out1 = pool.wait(h1)?;
```

Same applies to MPSGraph (`compile_mpsgraph_plan` + `submit_mpsgraph_plan`)
and to the bottom-layer single-context APIs (`RknnBackend::run_async_bound`
/ `wait`). Each layer is a clean escape hatch — you're never trapped
in a high-level API.

---

## Where to next

| If you want to... | Read |
|---|---|
| Browse all 160 image-processing ops | [docs/cookbook.md §image processing](cookbook.md#image-processing) |
| Fine-tune a YOLO on your data | [docs/cookbook.md §fine-tuning a detection head](cookbook.md#fine-tuning-a-detection-head) |
| Decode video faster than ffmpeg | [docs/video-pipeline.md](video-pipeline.md) |
| Cross-compile for ARM Linux | [docs/cookbook.md §cross-compilation](cookbook.md#cross-compilation-and-deployment) |
| Understand the architecture | [docs/architecture.md](architecture.md) |
| Compare against PyTorch / OpenCV / ORT | [docs/performance-benchmarks.md](performance-benchmarks.md) |
| Wire up a custom NPU op | [docs/edge-deployment.md §custom ops](edge-deployment.md) |
| See what's still missing | [docs/ecosystem-capability-matrix.md](ecosystem-capability-matrix.md) |
| Things broken? | [docs/troubleshooting.md](troubleshooting.md) |

---

## How yscv is structured (one-screen overview)

```
yscv (umbrella crate, prelude)
   ↓
[Layer 4 — Domain]   yscv-imgproc · yscv-video · yscv-detect · yscv-track
                     yscv-recognize · yscv-eval · yscv-onnx · yscv-pipeline
                     ↑
[Layer 3 — Train]    yscv-model (Trainer + 39 layers + 17 losses + zoo + LoRA)
                     ↑
[Layer 2 — AD]       yscv-autograd · yscv-optim
                     ↑
[Layer 1 — Compute]  yscv-kernels (CPU SIMD, GPU wgpu, Metal, RKNN NPU)
                     yscv-tensor  (115 ops, f32 / f16 / bf16, SIMD-aligned)
```

Each layer can be used standalone — you can pull `yscv-tensor` alone
for SIMD'd numerics without the rest, or `yscv-pipeline` alone for
edge runtime without the trainer. The umbrella `yscv` crate is just a
re-export convenience for the common case.
