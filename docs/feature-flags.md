# Feature flags — what they do and how to set them up

Every `yscv-*` crate has Cargo feature flags that gate optional code paths:
GPU backends, hardware codecs, real-time scheduling, platform-specific
acceleration. This is the canonical reference — what each flag does,
what it requires on the host, how to turn it on.

For specific backends with lots of surface area (MPSGraph, wgpu) see
the dedicated guides linked inline.

---

## TL;DR — the map

```
yscv (umbrella)
├── blas            default ON — BLAS matmul dispatch (Accelerate / OpenBLAS)
├── gpu             wgpu cross-platform GPU            → gpu-backend-guide.md
├── metal-backend   Apple MPSGraph (macOS, fastest)    → mpsgraph-guide.md
├── rknn            Rockchip NPU                       → edge-deployment.md
├── rknn-validate   Dry-run RKNN load at config time
├── realtime        SCHED_FIFO + affinity + mlockall + governor
├── native-camera   V4L2 / AVFoundation / MediaFoundation
├── mkl             Intel MKL for x86 matmul
├── armpl           ARM Performance Libraries for aarch64 matmul
├── drm             Linux KMS display output (yscv-video)
├── mpp             Rockchip MPP hardware H.264/H.265 encoder (yscv-video-mpp)
├── videotoolbox    Apple VideoToolbox HW decode (macOS/iOS)
├── vaapi           Intel/AMD VA-API HW decode (Linux)
├── nvdec           NVIDIA NVDEC HW decode (Linux/Windows)
├── mediafoundation Microsoft MediaFoundation HW decode (Windows)
└── profile         Per-op ONNX profiling instrumentation
```

---

## How to set features

In your `Cargo.toml`:

```toml
[dependencies]
yscv = { version = "0.1", features = ["rknn", "realtime"] }
# or:
yscv-onnx = { version = "0.1", features = ["metal-backend"] }
# or per-crate for finer control:
yscv-kernels = { version = "0.1", features = ["gpu"] }
yscv-video   = { version = "0.1", features = ["native-camera", "vaapi"] }
```

On the command line:

```bash
cargo build --release --features "rknn realtime"
cargo run   --release --features "metal-backend gpu" --example bench_mpsgraph
cargo test  --workspace --features "rknn metal-backend gpu realtime rknn-validate"
```

---

## Backend flags (pick what you're deploying on)

These change which execution path `run_onnx_model` / `dispatch_frame` /
`RknnPipelinedPool` route to. Multiple can be combined — the dispatcher
picks at runtime based on your TOML/API choice.

### `blas` — BLAS-accelerated matmul

Default ON in the `yscv-kernels` / `yscv-onnx` crates; default OFF in
the `private/onnx-fps` bench harness. Links a CBLAS implementation for
SGEMM dispatch:

- **macOS** — `Accelerate.framework` (built into OS, zero setup). Uses
  Apple's AMX matrix accelerator; ~2× faster than OpenBLAS here.
- **Linux** — `libopenblas.so.0`. Install with
  `sudo apt install libopenblas-dev` (Debian/Ubuntu) or
  `sudo dnf install openblas-devel` (Fedora).
- **Windows** — OpenBLAS via vcpkg. `vcpkg install openblas:x64-windows`
  and set `OPENBLAS_PATH` or `VCPKG_ROOT` env.

**Turn ON when**:

- Transformer attention (`QKᵀ`, `AV` — big square-ish GEMM).
- ResNet / ViT classification (FC 2048×1000 output).
- LLM K/V projection, output head, any decoder step.
- Graphs where one MatMul is tens of millions of FLOPs and there's no
  adjacent fusion opportunity.
- macOS anywhere — `Accelerate` + AMX is hard to beat and cheap to link.

**Turn OFF when**:

- Conv-heavy graphs (object detectors, Siamese trackers, U-Nets,
  segmentation). Our blocked GEMM fuses Conv+Bias+Residual+Activation
  in registers; BLAS can't — it writes GEMM output first, then a
  separate pass adds bias/residual and applies activation.
- You're on Linux / Windows with OpenBLAS 0.3.x and hit the
  oversubscription wall — our non-BLAS path hands all work to rayon
  and scales cleanly; OpenBLAS's internal threadpool fights rayon
  even with `OPENBLAS_NUM_THREADS=1`.
- You care about single-binary deployment and don't want to carry
  `libopenblas` into a container image.

Measured on the public Siamese tracker (Conv 85 %, DW 9 %, MatMul 5 %),
Zen 4 6C/12T, 2026-04 arc state:

| path | 1T p50 | 6T p50 |
|---|---:|---:|
| yscv **no-BLAS**  | 11.22 ms | **3.17 ms** |
| yscv **with BLAS** | 13.00 ms | 8.75 ms |
| ONNX Runtime 1.24 | 8.07 ms  | 1.74 ms |

BLAS makes this workload **2.76× slower at 6T** because it breaks
the fused-epilogue path (`row_gemm_set_parallel_fused`,
`blocked_gemm` MR=4×24 / MR=12×32 with in-register residual+bias+
activation) — all the R4 / R7 / R9 / A2 landings in the April arc
only fire on the non-BLAS branch.

On Transformer workloads the tradeoff inverts: one big sgemm per
attention head dominates and `Accelerate` / `MKL` walk all over any
hand-tuned blocked GEMM.

```toml
# Library default — keeps BLAS on, good for Transformer / classifier
yscv = "0.1"

# Conv-heavy deployment — no-BLAS is faster, no external lib
yscv = { version = "0.1", default-features = false }
```

See [`performance-benchmarks.md`](performance-benchmarks.md#onnx-siamese-tracker-zen-4-7500f-fp32-cpu)
for the tracker-specific numbers and
[`perf-arc-2026-04.md`](perf-arc-2026-04.md) for why the non-BLAS
path wins on this class of model.

### `metal-backend` — Apple MPSGraph (macOS, fastest on Apple Silicon)

Enables `compile_mpsgraph_plan` / `run_mpsgraph_plan` / pipelined
submit+wait API. Requires macOS 13+, best on Apple Silicon.

**Full guide**: [`mpsgraph-guide.md`](mpsgraph-guide.md) — when to use,
API, pipelining, troubleshooting, performance numbers.

Quick example:
```rust
use yscv_onnx::{compile_mpsgraph_plan, run_mpsgraph_plan};
let plan = compile_mpsgraph_plan(&model, &[("images", &sample)])?;
let outputs = run_mpsgraph_plan(&plan, &[("images", bytes)])?;
```

### `gpu` — wgpu cross-platform GPU (Linux / Windows / macOS)

Vulkan / Metal / DX12 / OpenGL via `wgpu`. Works on AMD, NVIDIA, Intel,
Apple, Adreno, Mali — single Rust binary.

**Full guide**: [`gpu-backend-guide.md`](gpu-backend-guide.md) — platform
selection, compiled plans, multi-GPU, NixOS LD setup, fp16/fp32
toggle, troubleshooting.

Quick example:
```rust
use yscv_onnx::run_onnx_model_gpu;
let outputs = run_onnx_model_gpu(&model, inputs)?;
```

Runtime flags:
- `YSCV_GPU_FP32=1` — disable opportunistic fp16 (`SHADER_F16`),
  force every kernel + buffer through fp32. Output becomes 1-ULP
  identical to the CPU runner; latency rises ~12-15 % on memory-bound
  graphs. Use for accuracy A/B; ship default fp16.
- `WGPU_BACKEND={vulkan|dx12|metal|gl}` — pin the backend (wgpu env;
  default is "best-of"). Useful for forcing software fallback in CI.
- `WGPU_ADAPTER_NAME=...` — pick a specific GPU when multiple are
  visible.

### `rknn` — Rockchip NPU (edge deployment)

Links `librknnrt.so` at runtime via `dlopen`. Builds on any platform;
runs only on Rockchip SoCs with the runtime lib installed. Covers
RK3588 / RK3576 / RV1106.

**Full guide**: [`edge-deployment.md`](edge-deployment.md) — DMA-BUF
zero-copy, SRAM, MPP zero-copy from hardware decoder, dynamic-shape
matmul, custom OpenCL ops.

Quick example:
```rust
use yscv_kernels::{RknnPipelinedPool, NpuCoreMask};
let pool = RknnPipelinedPool::new(&model_bytes,
    &[NpuCoreMask::Core0, NpuCoreMask::Core1, NpuCoreMask::Core2])?;
```

Host setup on the Rockchip board:
```bash
# librknnrt.so from rockchip-linux/rknn-toolkit2 release
sudo cp librknnrt.so /usr/lib/aarch64-linux-gnu/
sudo ldconfig
```

### `rknn-validate` — dry-run model load at startup

Optional companion to `rknn`. When enabled, `PipelineConfig::validate_models`
calls `RknnBackend::load` at config time to catch corrupt-but-magic-OK
`.rknn` files before any RT threads start.

Adds ~100 ms to startup. Worth it in production.

```toml
yscv-pipeline = { version = "0.1", features = ["rknn", "rknn-validate"] }
```

---

## Platform-acceleration flags (silently faster if set)

### `mkl` — Intel MKL for x86 matmul

Replaces OpenBLAS with Intel's MKL library. 2-3× faster matmul on Intel
CPUs (AVX-512 + per-cpu dispatched kernels).

Setup:
```bash
# Install Intel oneAPI Base Toolkit (includes MKL)
# https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html

# Or on Debian/Ubuntu:
sudo apt install intel-mkl

# Point cargo at it:
export MKLROOT=/opt/intel/oneapi/mkl/latest
```

Then:
```toml
yscv = { version = "0.1", features = ["mkl"] }
```

No code changes. Matmul and Conv silently take the MKL path.

**When to use**: Intel servers (Xeon, Core i9), Windows/Linux x86 dev.
**When NOT to use**: Apple Silicon (Accelerate beats MKL), AMD (MKL
historically pessimises on non-Intel, set `MKL_DEBUG_CPU_TYPE=5` to
force AVX2 path).

### `armpl` — ARM Performance Libraries

Same idea, for aarch64 Linux. Arm's hand-tuned NEON/SVE matmul kernels.
Biggest win on AWS Graviton, Ampere, or any server ARM.

Setup:
```bash
# Download from https://developer.arm.com/downloads/-/arm-performance-libraries
# Install to /opt/arm/armpl-xxx
export ARMPL_DIR=/opt/arm/armpl-23.10
```

```toml
yscv = { version = "0.1", features = ["armpl"] }
```

**When to use**: aarch64 Linux servers.
**When NOT to use**: Apple Silicon (use default Accelerate), Rockchip
(use `rknn` for NPU instead of CPU matmul).

---

## Real-time scheduling — `realtime`

Wires Linux real-time scheduling primitives through the pipeline
framework. When set, `PipelineConfig` with `[realtime]` section will
apply:

1. **SCHED_FIFO** — real-time priority for dispatch threads
2. **CPU affinity** — pin threads to specific cores, stop cache bouncing
3. **mlockall** — prevent memory paging (no swap-out glitches mid-flight)
4. **cpufreq governor** — write `"performance"` to every core, kill
   DVFS step-up latency

Setup:

```toml
[dependencies]
yscv-pipeline = { version = "0.1", features = ["realtime"] }
```

In your TOML:
```toml
[realtime]
sched_fifo = true
cpu_governor = "performance"
prio.dispatch = 70
affinity.dispatch = [4, 5, 6]
```

Linux capabilities required:
- **CAP_SYS_NICE** for SCHED_FIFO
- **CAP_SYS_ADMIN** for cpufreq governor writes
- Process memlock limit raised for mlockall

Production setup via systemd unit:
```ini
[Service]
ExecStart=/usr/local/bin/my-pipeline config.toml
AmbientCapabilities=CAP_SYS_NICE CAP_SYS_ADMIN
LimitMEMLOCK=infinity
LimitRTPRIO=99
```

Or via `setcap` on the binary:
```bash
sudo setcap cap_sys_nice,cap_sys_admin+ep ./my-pipeline
```

**Graceful fallback**: if the process lacks a capability, the framework
logs a warning and continues without that specific feature. Your
pipeline runs; real-time guarantees are partial.

Verify what actually applied after startup by checking the stderr log:
```
[yscv-pipeline] realtime partial: sched_fifo=true affinity=true mlockall=true governor_cores=8
```

If `sched_fifo=false`: missing CAP_SYS_NICE. If `mlockall=false`:
memlock limit too low. If `governor_cores=0`: missing CAP_SYS_ADMIN.

On non-Linux hosts (macOS, Windows) the feature compiles but all
primitives return `NotSupported` — safe to leave enabled in cross-
platform code.

---

## Camera capture — `native-camera`

Native platform camera APIs for live capture. Wraps `nokhwa` crate
which handles the three platforms uniformly.

Setup per platform:

- **Linux** — V4L2 always available, no extra install. Permissions:
  user must be in `video` group (`sudo usermod -aG video $USER`).
- **macOS** — AVFoundation, works out of the box. On first access
  macOS prompts for camera permission (`Info.plist` NSCameraUsageDescription
  needed for app bundles; console apps typically prompt in Terminal).
- **Windows** — MediaFoundation, works out of the box. UWP apps need
  `Webcam` capability in the manifest.

Setup in Cargo:
```toml
yscv-video = { version = "0.1", features = ["native-camera"] }
```

Usage:
```rust
use yscv_video::{CameraConfig, V4l2Camera};  // Linux

let mut cam = V4l2Camera::open(CameraConfig {
    device: "/dev/video0".into(),
    width: 1280, height: 720,
    fps: 60,
    format: yscv_video::PixelFormat::Nv12,
})?;
loop {
    let frame = cam.capture_frame()?;
    // process frame…
}
```

**On Rockchip boards**: V4L2 works; for zero-copy to NPU use
`VIDIOC_EXPBUF` to get a DMA-BUF fd, then `RknnBackend::wrap_fd` to
bind it as an NPU input. Full recipe in [`edge-deployment.md`](edge-deployment.md).

---

## Hardware video decode — `videotoolbox` / `vaapi` / `nvdec` / `mediafoundation`

Platform-specific HW decoders on top of the software H.264/HEVC/AV1
decoders. yscv automatically detects availability at runtime; software
decode is always the fallback.

| Flag | Platform | Codecs |
|---|---|---|
| `videotoolbox` | macOS / iOS | H.264, HEVC, ProRes |
| `vaapi` | Linux | H.264, HEVC, AV1 (on Intel ≥ Xe, AMD ≥ RDNA2) |
| `nvdec` | Linux / Windows + NVIDIA | H.264, HEVC, AV1 (Ampere+) |
| `mediafoundation` | Windows | H.264, HEVC, AV1 (hardware-dependent) |

Setup:
```toml
yscv-video = { version = "0.1", features = ["videotoolbox"] }   # macOS
yscv-video = { version = "0.1", features = ["vaapi"] }          # Linux/Intel/AMD
yscv-video = { version = "0.1", features = ["nvdec"] }          # Linux/Windows NVIDIA
yscv-video = { version = "0.1", features = ["mediafoundation"] }# Windows
```

System deps per platform:

- **macOS VideoToolbox**: built-in, no install.
- **Linux VA-API**: `sudo apt install libva-dev intel-media-va-driver`
  (or `mesa-va-drivers` for AMD). Check with `vainfo`.
- **NVIDIA NVDEC**: ships with the NVIDIA driver. `nvidia-smi` must work.
- **Windows MediaFoundation**: built into Windows 10+.

Usage — automatic detection via `HwDecoder`:
```rust
use yscv_video::{HwDecoder, VideoCodec};

let mut decoder = HwDecoder::new(VideoCodec::H264)?;
// Tries VideoToolbox/VAAPI/NVDEC/MediaFoundation in that order, falls
// back to software if none available. Use it the same way either way.
let frame = decoder.decode(packet)?;
```

See [`video-pipeline.md`](video-pipeline.md) for container parsing
(MP4/MKV), zero-copy decode, and performance numbers vs ffmpeg.

---

## Miscellaneous flags

### `drm` — Linux KMS display output (yscv-video)

Direct framebuffer output via DRM/KMS. For headless edge devices
displaying video without X/Wayland. Used by pipeline `output.kind = "drm"`.

Setup:
```bash
# User needs access to /dev/dri/card0
sudo usermod -aG video $USER
```

```toml
yscv-video = { version = "0.1", features = ["drm"] }
```

TOML config:
```toml
[output]
kind = "drm"
connector = "HDMI-A-1"
mode = "720p60"
```

### `mpp` — Rockchip MPP hardware encoder (yscv-video-mpp)

Hardware H.264/H.265 encoding on Rockchip SoCs via
`librockchip_mpp.so` (also `dlopen`). Used by pipeline
`encoder.kind = "mpp-h264"`.

Setup on Rockchip board:
```bash
sudo apt install librockchip-mpp1     # or from rockchip-linux/mpp releases
```

```toml
yscv-video-mpp = { version = "0.1", features = ["mpp"] }
```

### `profile` — per-op ONNX profiling

Instruments every op in the ONNX runner with wall-time measurement.
Useful to find which op is a bottleneck.

```toml
yscv-onnx = { version = "0.1", features = ["profile"] }
```

```rust
use yscv_onnx::profile_onnx_model_cpu;
let times = profile_onnx_model_cpu(&model, inputs)?;
for (op_name, ms) in times.iter().take(10) {
    println!("{op_name}: {ms:.2}ms");
}
```

Overhead is small (~1% per op) but leave it off in production — the
counters persist across runs.

---

## Runtime perf env toggles (ONNX CPU kernels)

Cargo features select backend code at build time. For ONNX CPU
micro-tuning and A/B you also have runtime env toggles.

Most used knobs:

- `YSCV_FUSED_PW_DW_STREAM_OFF=1` — disable PW->DW streaming fused path.
- `YSCV_FUSED_PW_DW_PW2X_OFF=1` — disable NEON PW2X inner-loop variant.
- `YSCV_FUSED_PW_DW_W_TILE=<N>` — override fused PW->DW strip-mining tile.
- `YSCV_FUSED_DW_PW_STREAM_OFF=1` — disable DW->PW streaming fused path.
- `YSCV_FUSED_DW_PW_STREAM_PADDED=1` — enable padded streaming variant.
- `YSCV_DIRECT_CONV_WORK_MAX=<N>` — direct-3x3 routing threshold.
- `YSCV_NO_AARCH64_LOW_K_BLOCKED=1` and
  `YSCV_AARCH64_LOW_K_BLOCKED_MIN_WORK_FMAS=<N>` — low-k blocked matmul route.

Canonical reference (with defaults, scope, and reproduction commands):
[`onnx-cpu-kernels.md`](onnx-cpu-kernels.md).

---

## Combination recipes

```toml
# Pure CPU, zero system deps (works anywhere, slowest)
yscv = { version = "0.1", default-features = false }

# Apple Silicon dev — everything Apple has
yscv = { version = "0.1", features = ["metal-backend", "videotoolbox", "native-camera"] }

# Intel server — best CPU inference
yscv = { version = "0.1", features = ["mkl", "gpu", "vaapi"] }

# NVIDIA server
yscv = { version = "0.1", features = ["gpu", "nvdec"] }

# Rockchip drone / FPV — full edge stack
yscv-pipeline  = { version = "0.1", features = ["rknn", "rknn-validate", "realtime"] }
yscv-video     = { version = "0.1", features = ["drm"] }
yscv-video-mpp = { version = "0.1", features = ["mpp"] }

# Windows desktop
yscv = { version = "0.1", features = ["gpu", "mediafoundation"] }

# Cross-platform CV library (no accelerators, works anywhere including WASM)
yscv = { version = "0.1", default-features = false }
```

---

## Deep-dive guides

For flags with lots of surface area, the table above links out:

- [`mpsgraph-guide.md`](mpsgraph-guide.md) — `metal-backend` (MPSGraph API, pipelining)
- [`gpu-backend-guide.md`](gpu-backend-guide.md) — `gpu` (wgpu, multi-GPU, compiled plans)
- [`edge-deployment.md`](edge-deployment.md) — `rknn` (DMA-BUF, SRAM, custom ops)
- [`pipeline-config.md`](pipeline-config.md) — `realtime` TOML schema
- [`video-pipeline.md`](video-pipeline.md) — HW decode + container parsing

---

## Reporting flag bugs

If a feature works on one platform but not another, open an issue with:

- **Feature list** from your `Cargo.toml`
- `rustc --version`, `uname -a`, GPU / CPU model
- Exact error or symptom
- `cargo tree -e features -p <crate>` output — shows what got resolved
