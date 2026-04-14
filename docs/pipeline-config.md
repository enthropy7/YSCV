# Pipeline Config Reference

The `yscv-pipeline` crate consumes a single TOML file describing the
full runtime pipeline: camera, output, encoder, inference task graph,
OSD fields, and real-time scheduler. One config per target.

## Schema

```toml
board = "my-target"       # human-readable label, used in logs

[camera]
device = "/dev/video0"    # V4L2 device node
format = "nv12"           # "nv12" | "yuyv" | "mjpeg" | "rgb"
width = 1280              # > 0
height = 720              # > 0
fps = 60                  # > 0

[output]                  # one of:
kind = "drm"              # DRM/KMS atomic flip on HDMI/DSI
connector = "HDMI-A-1"
mode = "720p60"
# OR
kind = "v4l2-out"         # V4L2 output device (encode → /dev/videoN)
device = "/dev/video10"
# OR
kind = "file"             # write encoded NALs to file (dev/CI)
path = "out.h264"
# OR
kind = "null"             # discard (benchmarks)

[encoder]
kind = "mpp-h264"         # "mpp-h264" | "mpp-hevc" | "soft-h264" (slow)
bitrate_kbps = 8000
profile = "baseline"      # "baseline" | "main" | "high"

[[tasks]]                 # repeat per inference task
name = "detector"
model_path = "models/detector.rknn"
accelerator = { kind = "rknn", core = "core0" }
inputs  = [{ name = "images",  source = "camera" }]
outputs = [{ name = "output0", source = "detections" }]

[osd]
fields = ["fps", "detection_count", "latency"]
glyph_size = 12           # OSD text point size

[realtime]
sched_fifo = true         # try SCHED_FIFO; warns if no CAP_SYS_NICE

[realtime.prio]           # SCHED_FIFO priority per stage (1..=90)
capture = 80
dispatch = 70
wait = 65
encode = 60
output = 55

[realtime.affinity]       # CPU pinning per stage
capture = [4]
dispatch = [5]
wait = [0, 1]
encode = [2]
output = [3]

[realtime]
# Optional: write this governor to every cpufreq sysfs node at startup.
# `"performance"` pins all cores at max frequency, eliminating the
# 5–30ms first-burst latency from DVFS step-up. Requires CAP_SYS_ADMIN
# or root; missing privilege is logged but not fatal.
cpu_governor = "performance"
```

## `accelerator` values

```toml
accelerator = { kind = "cpu" }
accelerator = { kind = "gpu" }                              # wgpu (cross-platform)
accelerator = { kind = "rknn", core = "core0" }             # NPU core 0
accelerator = { kind = "rknn", core = "cores012" }          # All 3 RK3588 cores
accelerator = { kind = "rknn-matmul", m = 1, k = 4096, n = 4096, dtype = "fp16-mm-fp16-to-fp16" }  # Matmul accelerator
accelerator = { kind = "metal-mps" }                        # macOS dev path
```

`core` values: `auto`, `core0`, `core1`, `core2`, `cores01`, `cores02`,
`cores12`, `cores012`, `all`.

## `source` / `sink` syntax

A `TensorBinding`'s `source` (or `sink`, alias) is one of:

- `"camera"` — raw frame from V4L2
- `"detections"` / `"tracking"` / `"overlay"` — pipeline-managed sinks
- `"<task_name>.<output_tensor>"` — chain another task's output

Cycles in this graph are detected at validation time
(`PipelineConfig::validate_self()` returns `ConfigError::CyclicDependency`).

## Validation order

```rust
let cfg = PipelineConfig::from_toml_path("pipeline.toml")?;  // syntax
cfg.validate_self()?;          // structural — width>0, no dupe names, no cycles
cfg.validate_accelerators()?;  // runtime — librknnrt.so loadable, etc.
cfg.validate_models()?;        // per-file read + magic-byte check
// or:
cfg.validate()?;               // all three in order
```

`validate_models` does more than stat — it reads every `.rknn`/`.onnx`
file referenced by a task and runs a magic-byte check (`RKNN`/`RKNF` for
Rockchip models, `0x08` protobuf tag for ONNX). With the optional
`yscv-pipeline/rknn-validate` feature on a host where `librknnrt.so` is
loadable, it also runs a full `RknnBackend::load` — catching
corrupted-but-magic-matching files before real-time threads start.

A typical startup will fail at `validate_accelerators()` if you specify
`accelerator = { kind = "rknn", ... }` on a host without `librknnrt.so` —
that's by design. **No silent CPU fallback.**

## Running a config

After validation, `run_pipeline(cfg)` constructs one
`AcceleratorDispatcher` per task and returns a `PipelineHandle`. The
caller's hot loop drives it:

```rust
use yscv_pipeline::{PipelineConfig, run_pipeline};

let cfg = PipelineConfig::from_toml_path("boards/rock4d.toml")?;
let handle = run_pipeline(cfg)?;  // validates + builds dispatchers
// handle.order:         Vec<String> — topologically sorted task names
// handle.dispatcher_label(name):  Option<&str>   — for logging / OSD
// handle.dispatch_frame(inputs):  Result<HashMap<String, Vec<u8>>>
// handle.recover_all():           Result<()>     — reset every backend
```

Dispatcher-by-accelerator coverage today:

| `accelerator` value | dispatcher | feature flag |
|---|---|---|
| `kind = "cpu"` | CPU ONNX (yscv-onnx runner) | always |
| `kind = "rknn", core = "..."` | `RknnPipelinedPool` single-slot, with NPU-hang auto-recovery + ONNX→RKNN compile-on-load if `model_path` ends in `.onnx` | `rknn` |
| `kind = "metal-mps"` | `MpsGraphPlan` (Apple Silicon, `Mutex`-serialised). First dispatch compiles; `recover()` drops the plan to force recompile | `metal-backend` (macOS) |
| `kind = "gpu"` | wgpu cross-platform GPU via `yscv_onnx::run_onnx_model_gpu` | `gpu` |
| `kind = "rknn-matmul", m, k, n, dtype` | dedicated NPU matmul unit. Shape + dtype in the TOML; pre-allocates A/B/C buffers; inputs named `"a"` and `"b"`; output named `"c"` | `rknn` |

With `--features realtime`, `run_pipeline` applies the TOML's
`[realtime]` config (`sched_fifo` + `affinity` + `mlockall`) to the
driver thread at startup. Graceful fallback on dev hosts without
`CAP_SYS_NICE` / `RLIMIT_MEMLOCK`. See
[`examples/src/edge_pipeline_v2.rs`](../examples/src/edge_pipeline_v2.rs)
for the full driver loop including latency reporting.

## OSD fields

| field | shows |
|---|---|
| `fps` | rolling 1-second average frame rate |
| `latency` | most recent end-to-end ms |
| `detection_count` | bbox count from detector task |
| `battery` | voltage from the telemetry source if wired in |
| `signal` | link quality if wired in |
| `telemetry` | full HUD overlay (heavy) |

## Real-time tuning

| stage | suggested prio | hint |
|---|---|---|
| capture | 80 | highest — never miss a V4L2 DQBUF |
| inference dispatch | 70 | submit work to accelerator |
| inference wait + post | 65 | collect results + NMS |
| encode | 60 | MPP / soft encode submit |
| display | 55 | DRM atomic flip |

Pin to performance cores where the SoC has big.LITTLE topology
(e.g. RK3588: A55 cores 0–3, A76 cores 4–7 — put latency-sensitive
stages on A76).

To grant `CAP_SYS_NICE` without root:

```bash
sudo setcap 'cap_sys_nice=ep' /path/to/your-binary
```

To raise `RLIMIT_MEMLOCK` for `mlockall`:

```bash
ulimit -l unlimited
# or in a systemd unit:
LimitMEMLOCK=infinity
```

## Writing a config

1. Start with a minimal file: `[camera]`, `[output] kind = "null"`,
   `[encoder] kind = "soft-h264"`, zero tasks.
2. Validate: `cargo run --example board_pipeline -- your.toml`
3. Add tasks one at a time. Each `[[tasks]]` block declares one
   inference node with its accelerator + model + I/O bindings.
4. Set `realtime.prio` and `realtime.affinity` to match the SoC's
   CPU topology (use `lscpu -p` to list cores).
5. Run on the actual target with the needed feature flags enabled
   (`--features "rknn gpu"` for Rockchip, etc.).

Validation will fail-loud if anything's wrong, with a message pointing
to the field and the missing accelerator / library / model.
