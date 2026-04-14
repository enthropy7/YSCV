# yscv-pipeline

TOML-driven multi-accelerator inference pipeline for edge / FPV / drone
deployments. The runtime sits on top of `yscv-onnx` (CPU + Metal) and
`yscv-kernels::rknn` (Rockchip NPU) and gives you:

- **One declarative config per board** — describe camera, models,
  accelerator assignment, real-time tuning in TOML; ship the same Rust
  binary everywhere
- **Five accelerators with a single `dispatch_frame` API** — CPU
  (yscv-onnx), wgpu (cross-platform GPU), Apple MPSGraph, Rockchip
  RKNN NPU (with auto-recovery + ONNX→RKNN compile-on-startup),
  Rockchip matmul accelerator
- **Real-time wiring** — SCHED_FIFO, CPU affinity, `mlockall`, cpufreq
  governor; graceful fallback on dev hosts without privileges
- **Supervisor watchdog** — auto-recover the pipeline when per-stage
  budget overruns cross threshold
- **Hot-reload** — swap a `.rknn` model in-flight without stopping the
  capture loop

## At a glance

```rust
use yscv_pipeline::{PipelineConfig, run_pipeline};

let cfg = PipelineConfig::from_toml_path("boards/rock4d.toml")?;
let handle = run_pipeline(cfg)?;     // validates + builds dispatchers + applies RT

loop {
    let bytes = camera.capture()?;
    let outs = handle.dispatch_frame(&[("images", &bytes)])?;
    process(&outs["detector.output0"]);
}
```

The TOML — describes WHAT runs WHERE, no Rust code changes between
boards:

```toml
board = "rock4d"

[camera]
device = "/dev/video0"
format = "nv12"
width = 1280
height = 720
fps = 60

[output]      # one of: drm | v4l2-out | file | null
kind = "null"

[encoder]
kind = "mpp-h264"
bitrate_kbps = 8000

[[tasks]]
name = "detector"
model_path = "yolov8n.rknn"   # or .onnx — auto-compiles + caches at startup
accelerator = { kind = "rknn", core = "core0" }
inputs  = [{ name = "images", source = "camera" }]
outputs = []

[realtime]
sched_fifo   = true
cpu_governor = "performance"
prio.dispatch     = 70
affinity.dispatch = [4, 5, 6]
```

## Features

| flag | what it adds |
|---|---|
| `rknn` | Rockchip NPU dispatcher (`RknnPipelinedPool` with auto-recovery); NPU-matmul dispatcher; ONNX→RKNN compile-on-startup. Requires `librknnrt.so` at runtime; build is host-agnostic. |
| `rknn-validate` | Adds a full `RknnBackend::load` dry-run inside `validate_models` on hosts where `librknnrt.so` is loadable. Catches corrupted-but-magic-OK files at config time. |
| `metal-backend` | Apple Silicon MPSGraph dispatcher (lazy compile + plan reset on `recover`). macOS only. |
| `gpu` | wgpu cross-platform GPU dispatcher (Vulkan / Metal / DX12). |
| `realtime` | Wires `[realtime]` from the TOML through `yscv_video::realtime::apply_rt_config_with_governor`. Adds `yscv-video` as a dep + `PipelineHandle::spawn_watchdog`. |

## Five-accelerator dispatcher table

| `accelerator` value | dispatcher | requires |
|---|---|---|
| `kind = "cpu"` | `yscv-onnx` CPU runner | always |
| `kind = "rknn", core = "..."` | `RknnPipelinedPool` single-slot, auto-recovery, ONNX→RKNN compile | `rknn` |
| `kind = "rknn-matmul", m, k, n, dtype` | dedicated NPU matmul (LLM dequant / attention) | `rknn` |
| `kind = "metal-mps"` | `MpsGraphPlan` lazy-compiled | `metal-backend` (macOS) |
| `kind = "gpu"` | `yscv_onnx::run_onnx_model_gpu` | `gpu` |

## Why "one binary, many configs"

Same `cargo build` output runs YOLO on Rock4D through NPU, Siamese
tracker on RV1106 through NPU, ResNet on dev macOS through MPSGraph —
only the TOML changes. Validation is **fail-loud**: TOML asks for an
accelerator that isn't compiled in or whose runtime library is missing
on the host → startup aborts with a clear `ConfigError`. No silent CPU
fallback, no degradation in flight.

## Layered escape hatches

The trait-object dispatcher path is the most ergonomic. Each layer
below it is also public — drop down when you need finer control:

```
TOML config       run_pipeline(cfg) -> PipelineHandle::dispatch_frame
   ↓ (drop the TOML)
AcceleratorDispatcher trait        dispatcher_for(&task)
   ↓ (drop the trait)
RknnPipelinedPool / MpsGraphPlan   submit + wait, multi-slot, async
   ↓ (drop the pool)
RknnBackend / MpsGraphPlan         single-context async or sync
```

`apps/bench/fpv-latency` shows a bottom-layer path — submit + wait
directly against `MpsGraphPlan` / `RknnPipelinedPool`, sub-millisecond
Metal latency without going through the trait object.

## Documentation

- [`docs/pipeline-config.md`](../../docs/pipeline-config.md) — full TOML schema reference
- [`docs/edge-deployment.md`](../../docs/edge-deployment.md) — RKNN deep dive (DMA-BUF, SRAM, MPP zero-copy)
- [`docs/cookbook.md`](../../docs/cookbook.md) — recipes for every dispatcher
- [`docs/getting-started.md`](../../docs/getting-started.md) — progressive tutorial
- [`examples/src/edge_pipeline_v2.rs`](../../examples/src/edge_pipeline_v2.rs) — end-to-end reference

## Recovery, hot-reload, watchdog

```rust
// Manual recovery (call after observing a TIMEOUT-class error):
handle.recover_all()?;

// Background watchdog: auto-recover on per-stage budget overruns.
#[cfg(feature = "realtime")]
let _wd = std::sync::Arc::new(handle).spawn_watchdog(stats, std::time::Duration::from_millis(100));

// Hot-swap the model under one task — needs the bottom-layer pool.
let pool: &yscv_kernels::RknnPipelinedPool = /* from your dispatcher */;
pool.reload(&std::fs::read("models/v2.rknn")?)?;
```

## Tests

22 unit + integration tests covering: config parsing, validation
(file magic + accelerator availability + dry-run model load), topo
sort + cycle detection, dispatcher factory feature-gate behaviour,
and the realtime + watchdog wire-up. 1897 workspace lib tests pass
under `--features "rknn metal-backend gpu realtime rknn-validate"`.

## License

MIT OR Apache-2.0.
