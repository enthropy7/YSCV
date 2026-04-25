# wgpu GPU backend guide — cross-platform GPU inference

The `gpu` feature gives you **wgpu-backed** GPU inference that runs on
Vulkan (Linux / Android), Metal (macOS / iOS), DX12 (Windows) or
OpenGL (fallback). Works on AMD, NVIDIA, Intel, Apple, Adreno, Mali —
one Rust binary, same code, whatever GPU you have.

On macOS prefer [`mpsgraph-guide.md`](mpsgraph-guide.md) for Apple
Silicon — MPSGraph is faster there. But for Linux/Windows/cross-platform
builds, `gpu` is your path.

---

## TL;DR

```toml
# Cargo.toml
[dependencies]
yscv-onnx   = { version = "0.1", features = ["gpu"] }
yscv-tensor = "0.1"
```

```rust
use yscv_onnx::{load_onnx_model_from_file, run_onnx_model_gpu};
use std::collections::HashMap;

let model = load_onnx_model_from_file("yolov8n.onnx")?;
let mut inputs = HashMap::new();
inputs.insert("images".to_string(), input_tensor);

let outputs = run_onnx_model_gpu(&model, inputs)?;
```

That's the simplest path — one-shot GPU inference.

For sustained throughput, use the compiled-plan path with f16 weight
caching — see [Compiled plans](#compiled-plans).

---

## When to use wgpu vs other backends

| Platform | Recommended backend | Why |
|---|---|---|
| macOS (Apple Silicon) | `metal-backend` (MPSGraph) | 2-4× faster than wgpu via native Apple compiler |
| macOS (Intel) | `gpu` (wgpu → Metal) | AMD GPU support; MPSGraph has Intel Mac issues |
| Linux + NVIDIA | `gpu` (wgpu → Vulkan) | No Python/CUDA toolchain |
| Linux + AMD | `gpu` (wgpu → Vulkan) | AMD compute works first-class via RADV |
| Linux + Intel iGPU | `gpu` (wgpu → Vulkan) | ANV driver, very good perf/watt |
| Windows + any GPU | `gpu` (wgpu → DX12) | Native DirectX 12 path, no CUDA needed |
| Windows + NVIDIA | `gpu` (wgpu → DX12) or `gpu` (Vulkan) | Both work; DX12 slightly more stable |
| Rockchip edge | `rknn` (NPU) | Dedicated accelerator is ~10× the iGPU |
| WSL2 | `gpu` (DX12 via DirectML) | Vulkan on WSL2 is rocky; DX12 is solid |
| CI / headless | CPU runner | wgpu needs GPU hardware or Lavapipe |

---

## How it works

```
┌─────────────────────┐
│   Your ONNX model   │
└──────────┬──────────┘
           │
           │ run_onnx_model_gpu(&model, inputs)
           │   or
           │ compile_gpu_plan_f16 + run_compiled_gpu_f16_fused
           ↓
    ┌──────────────┐
    │  GpuBackend  │   ← one per process, owns WGPU device + queue
    └──────┬───────┘
           │
           │ wgpu compiles to native shader language:
           │   • Vulkan   → SPIR-V
           │   • Metal    → MSL
           │   • DX12     → HLSL / DXIL
           │   • OpenGL   → GLSL (fallback)
           ↓
    ┌──────────────────┐
    │ Native GPU ops   │  ← 61 WGSL shaders in yscv-kernels
    │  (matmul, conv,  │
    │   pool, norm,    │
    │   softmax, ...)  │
    └──────────────────┘
```

- **WGSL shaders** — we ship 61 compute shaders written in WebGPU Shading
  Language. wgpu cross-compiles them at runtime to the native format of
  whatever GPU driver you have.
- **One device, many ops** — `GpuBackend` owns a single `wgpu::Device` +
  `Queue`. All ops share the buffer pool + pipeline cache.
- **Runtime dispatch** — wgpu picks the "best" backend in this order:
  Vulkan > Metal > DX12 > GL. Can be overridden via `WGPU_BACKEND` env
  var (see below).

---

## Simple inference (one-shot path)

```rust
use yscv_onnx::{load_onnx_model_from_file, run_onnx_model_gpu};
use yscv_tensor::Tensor;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = load_onnx_model_from_file("yolov8n.onnx")?;

    let input = Tensor::zeros(vec![1, 3, 640, 640])?;
    let mut inputs = HashMap::new();
    inputs.insert("images".to_string(), input);

    let outputs = run_onnx_model_gpu(&model, inputs)?;
    let detections = outputs.get("output0").unwrap();
    println!("shape: {:?}", detections.shape());
    Ok(())
}
```

Every call reallocates GPU buffers and recompiles kernels. Fine for
scripts; terrible for video.

---

## Compiled plans (sustained throughput)

Like MPSGraph's `compile_mpsgraph_plan`, there's a compile-once path
that reuses weights + shader pipelines:

```rust
use yscv_onnx::{
    load_onnx_model_from_file,
    compile_gpu_plan_f16, run_compiled_gpu_f16_fused,
    GpuWeightCache,
};

let model = load_onnx_model_from_file("yolov8n.onnx")?;

// Weights uploaded to GPU once, f16-compressed, cached for the plan's lifetime.
let mut weight_cache = GpuWeightCache::new();
let plan = compile_gpu_plan_f16(&model, &input_shapes, &mut weight_cache)?;

// Hot loop — weights stay on GPU, only inputs flow in.
loop {
    let outputs = run_compiled_gpu_f16_fused(&plan, &inputs)?;
    process(&outputs);
}
```

f16 weights halve GPU bandwidth vs f32; for memory-bound models (most
CNNs) this alone is 1.5-2×.

---

## Choosing a wgpu backend at runtime

By default wgpu picks whichever of `VULKAN | METAL | DX12 | GL` is
available. Override via env:

```bash
# Force Vulkan (most consistent on Linux)
WGPU_BACKEND=vulkan ./my-app

# Force DX12 on Windows (more stable than Vulkan on some drivers)
WGPU_BACKEND=dx12 ./my-app

# Force software fallback (for CI/testing)
WGPU_BACKEND=gl ./my-app
```

Combined with `WGPU_ADAPTER_NAME` to pick a specific GPU when multiple
are present:

```bash
WGPU_ADAPTER_NAME="NVIDIA GeForce RTX" WGPU_BACKEND=vulkan ./my-app
```

See `wgpu`'s [`Instance::request_adapter`
docs](https://docs.rs/wgpu/latest/wgpu/struct.Instance.html) for the
full env-var list.

---

## Numerical precision (`YSCV_GPU_FP32`)

`GpuBackend::new` opportunistically enables `wgpu::Features::SHADER_F16`
when the adapter advertises it. Gain: ~1.5× throughput on memory-bound
Conv graphs, half the GPU buffer footprint, lower bandwidth to fill.
Cost: outputs drift roughly 1-2 % vs an fp32 reference — typically
invisible after detection NMS / softmax-argmax, but enough to fail
bit-level correctness checks (max-element drift on the public Siamese
tracker: 49.0915 fp32 vs 48.1491 fp16, ~1.0 unit). For accuracy A/B
runs against CPU or ORT reference, force fp32:

```bash
YSCV_GPU_FP32=1 ./my-app
```

| mode | tracker `882` max | drift vs ORT CPU 49.0916 |
|---|---:|---:|
| `gpu` (fp16, default when adapter has SHADER_F16) | 48.1491 | −0.94 |
| `gpu` + `YSCV_GPU_FP32=1` | **49.0915** | **−1 ULP** |
| ORT CUDA EP fp32 | 48.9193 | −0.17 (Tensor-Core implicit downcast) |
| CPU runner (default no-BLAS) | 49.0920 | +1 ULP |

Rule of thumb: ship with default fp16 in production, switch to
`YSCV_GPU_FP32=1` for golden-reference comparison and accuracy
regression tests. The extra 12-15 % latency from fp32 is small and
the outputs become 1-ULP-close to the CPU/ORT reference.

---

## NixOS / non-FHS Linux runtime

Default `wgpu::Instance` uses `dlopen` to load `libvulkan.so` from
the system loader path. Standard glibc distros work out of the box
(libvulkan ends up in `/usr/lib`); on NixOS it doesn't, because
loader libraries live in `/nix/store/.../lib` and the standard ld
search path doesn't see them. Symptom: `GpuBackend::new()` returns
`no GPU adapter found` even though `vulkaninfo` reports devices.

Fix is `LD_LIBRARY_PATH` at process start:

```bash
LD_LIBRARY_PATH=$(nix-build '<nixpkgs>' -A vulkan-loader --no-out-link)/lib:/run/opengl-driver/lib \
  ./my-app
```

Or (less surgical, easier to remember):

```bash
nix-shell -p vulkan-loader --run \
  'LD_LIBRARY_PATH=$buildInputs/lib:/run/opengl-driver/lib ./my-app'
```

The `/run/opengl-driver/lib` part is essential too — that's where
NixOS stages vendor-specific Vulkan ICDs (NVIDIA, Mesa, etc.). On
Ubuntu/Fedora/Arch this whole song is unnecessary.

Verify the adapter list with a tiny probe before chasing further:

```rust
use wgpu::{Instance, InstanceDescriptor, Backends};
let inst = Instance::new(InstanceDescriptor {
    backends: Backends::all(),
    ..InstanceDescriptor::new_without_display_handle()
});
let adapters = pollster::block_on(inst.enumerate_adapters(Backends::all()));
println!("found {} adapter(s)", adapters.len());
```

If `enumerate_adapters` returns 0 with `RUST_LOG=wgpu_hal=info` showing
"vulkan drivers/libraries could not be loaded", it's the LD_LIBRARY_PATH
issue.

---

## Performance vs ORT CUDA EP

wgpu is **not** going to match cuDNN-on-Tensor-Cores. On the public
Siamese tracker, RTX 4060, 1×3×128×128 + 1×3×256×256:

| backend | p50 | min | output drift |
|---|---:|---:|---:|
| ORT CUDA EP (cuDNN + Tensor Cores) | 1.42 ms | 1.40 ms | TF32 / mixed |
| yscv `gpu` fp16 (wgpu Vulkan) | 5.25 ms | 5.18 ms | ~2 % |
| yscv `gpu` fp32 (wgpu Vulkan) | 5.82 ms | 5.72 ms | 1 ULP |
| yscv CPU 6T (Zen 4) | 3.05 ms | 2.84 ms | 1 ULP |
| ORT CPU 1T | 8.07 ms | 8.03 ms | reference |

The 4× gap to CUDA EP is structural, not a bug:

- **cuDNN ships shape-specific kernels.** For each Conv shape it picks
  among ~30 algorithms (Winograd / ImplicitGEMM / Direct / FFT /
  precomputed) tuned for the host SM architecture. wgpu compute
  shaders are vendor-portable WGSL — one kernel, no per-shape variant.
- **Tensor Cores.** RTX 4060 has fourth-gen Tensor Cores doing 4×4
  fp16 MMA per cycle per SM (`wmma::fragment` in CUDA, `coopmat` in
  Vulkan extension). cuDNN uses them. wgpu intentionally does not —
  `Features::COOPERATIVE_MATRIX` is gated behind a non-portable
  Vulkan extension wgpu doesn't surface.
- **Tensor Cores need fp16/bf16 with specific layouts.** cuDNN
  reshapes weights into Tensor-Core-friendly tiles at load time. Our
  WGSL kernels do generic 8×8 / 16×16 tiles in fp32 (or stored fp16
  with f32 accumulator), no MMA path.

If you need closer-to-CUDA perf on NVIDIA, the right tool is ORT
itself with `CUDAExecutionProvider`, or specifically NVIDIA's
TensorRT EP. yscv's `gpu` is for "I want one binary that runs on
Linux + Windows + macOS + Android with whatever GPU is present and
no vendor toolchain". On Apple Silicon, `metal-backend` (MPSGraph)
hits the AMX coprocessor and gets within 1.5× of CUDA EP for
detection workloads — that's the sibling code path documented in
[`mpsgraph-guide.md`](mpsgraph-guide.md).

For the same model the CPU runner is **2× faster** than wgpu fp16
(3.05 vs 5.25 ms) — that's a small Conv-heavy graph where CPU
fork-join dispatch beats GPU launch latency. Crossover where wgpu
starts winning is roughly: batch ≥ 4, total FLOPs ≥ 1 G, or
inference time ≥ 30 ms on CPU. Run a quick A/B before committing.

---

## Multi-GPU

`MultiGpuBackend` splits work across several GPUs for big workloads:

```rust
use yscv_kernels::{MultiGpuBackend, SchedulingStrategy};

let backend = MultiGpuBackend::new_all_gpus()?;
println!("found {} GPUs", backend.gpu_count());

// Strategies:
//   RoundRobin      — each frame goes to the next GPU (simplest, lowest latency)
//   DataParallel    — split the batch dim across GPUs (best for large batches)
//   Manual          — you pick which GPU per op
let scheduled = backend.with_strategy(SchedulingStrategy::RoundRobin);
```

Only useful when you have 2+ discrete GPUs (e.g. NVIDIA multi-GPU
servers). Integrated + discrete mixes work but the integrated GPU
is usually a bottleneck.

---

## Supported ops

61 WGSL shaders cover:

- **Matmul** — tiled f16 GEMM, row/col-major, transposed variants
- **Conv2d** — im2col + GEMM + bias + activation fused
- **Pooling** — max/avg pool 2D
- **Normalization** — BatchNorm, LayerNorm, GroupNorm, RMSNorm
- **Activations** — ReLU, SiLU, GELU, Tanh, Sigmoid, HardSwish, Softmax
- **Elementwise** — add, sub, mul, div (with broadcasting)
- **Shape ops** — Transpose, Reshape, Concat, Slice
- **Reductions** — sum, mean, max, min over axes

Not supported (triggers CPU fallback for that op):

- Exotic quantization (INT4 per-channel, TensorRT-style scales)
- Custom ops (ScatterND with specific indices, RoIAlign variants)
- Dynamic-shape reshape (shape fed from another op's output)

When a fallback happens, `run_onnx_model_gpu` silently routes that
op through the CPU runner and pays one GPU↔CPU copy. Check with:

```rust
use yscv_onnx::{plan_gpu_execution, GpuExecAction};
let plan = plan_gpu_execution(&model, &inputs)?;
for action in &plan.actions {
    match action {
        GpuExecAction::Gpu(op) => println!("✓ GPU: {}", op.op_type),
        GpuExecAction::CpuFallback(op) => println!("✗ CPU (fallback): {}", op.op_type),
    }
}
```

---

## Performance expectations

Rough numbers on representative hardware (fp32, YOLOv8n 640×640):

| GPU | wgpu FPS (sync) | vs CPU runner |
|---|---:|---:|
| NVIDIA RTX 4090 (Vulkan) | ~250 | 8× |
| NVIDIA RTX 3060 (Vulkan) | ~120 | 4× |
| AMD RX 7900 (Vulkan) | ~180 | 6× |
| Intel Arc A770 (Vulkan) | ~90 | 3× |
| Apple M1 Pro (Metal via wgpu) | ~150 | 5× |
| Apple M1 Pro (**Metal via MPSGraph**) | **~400** | **13×** — use MPSGraph on mac |

For compiled f16 path expect +30-60% vs the numbers above.

**Key takeaways:**
- On Apple Silicon, MPSGraph wins by 2-3× over wgpu. Use the dedicated
  `metal-backend` there.
- On Linux/Windows with any GPU, wgpu is the right choice — works out
  of the box, no CUDA/ROCm toolchain required.
- On NVIDIA specifically: wgpu via Vulkan is 70-80% the speed of
  onnxruntime with CUDA EP. You give up that 20% but gain "no driver
  install dance".

---

## Troubleshooting

### "wgpu failed to find an adapter"

No supported GPU driver installed. On Linux:

```bash
# NVIDIA (proprietary)
sudo apt install nvidia-driver-535  # or matching version

# AMD / Intel (Mesa, usually already there)
sudo apt install mesa-vulkan-drivers libvulkan1

# Check it's working:
vulkaninfo | head -20       # should print GPU info
```

On Windows DX12 needs Windows 10 1903+. If your Windows is older,
force Vulkan: `WGPU_BACKEND=vulkan`.

On macOS everything just works (Metal is built-in). If you see this
error, `--features metal-backend` is already going via a different
path; Cargo feature conflict — check your dep tree.

### "GPU compute shader crashes on X op"

wgpu uses runtime shader compilation. A buggy driver can silently
produce wrong output for exotic shapes. First-aid:

```bash
# Try a different wgpu backend
WGPU_BACKEND=dx12 ./my-app     # on Windows instead of Vulkan
WGPU_BACKEND=vulkan ./my-app   # on Windows instead of DX12

# Update your GPU driver
# Fall back to CPU for this model while you report the bug
```

Report with the model file + `vulkaninfo` output to our tracker — we
can often work around driver bugs with alternative shader paths.

### "Inference works but throughput is bad"

Most common cause: GPU↔CPU copy per frame. Use the compiled f16 plan:

```rust
let plan = compile_gpu_plan_f16(&model, &shapes, &mut weight_cache)?;
// Now weights stay on GPU across calls.
```

And pre-upload your input tensor to GPU explicitly:

```rust
use yscv_kernels::GpuBackend;
let gpu = GpuBackend::new()?;
let input_gpu = gpu.upload_f32(&cpu_input)?;     // once
// Use input_gpu across iterations — zero per-frame copy.
```

### "Different results between CPU and GPU"

Two reasons:
1. **f16 quantization** (compiled path only) — weights lose ~3-4 bits
   of precision. Expected. If unacceptable, use f32 path:
   `run_onnx_model_gpu` (no `_f16`).
2. **Non-associative reductions** — GPU reduces in a tree, CPU left-to-
   right. Different rounding order. Expected difference is < 1e-4
   relative; bigger means a bug — report it.

### "Works on my dev box, crashes in Docker / systemd"

Docker: container needs `--device=/dev/dri` + mesa drivers mounted for
Linux GPU access. NVIDIA containers: use `nvidia-container-toolkit`.

Systemd: if your service runs as `root` or without display-user access
in X/Wayland, wgpu may fail to pick up the GPU. `User=your-user` + give
the user access to `/dev/dri/*` nodes.

---

## Feature-flag cheatsheet

```toml
# yscv-onnx alone
yscv-onnx = { version = "0.1", features = ["gpu"] }

# Combined with Metal backend (macOS — metal-backend wins where supported)
yscv-onnx = { version = "0.1", features = ["gpu", "metal-backend"] }

# yscv umbrella
yscv = { version = "0.1", features = ["gpu"] }
```

`gpu` implies `wgpu` + `pollster` (async runtime) + `bytemuck` at build
time. Adds ~20 MB to the debug binary, ~4 MB release (WGSL shaders
inlined as strings).

---

## Next steps

- [`mpsgraph-guide.md`](mpsgraph-guide.md) — Apple-native path (faster on mac)
- [`edge-deployment.md`](edge-deployment.md) — Rockchip NPU (~10× any GPU on that hardware)
- [`onnx-inference.md`](onnx-inference.md) — CPU runner and op coverage details
- [`performance-benchmarks.md`](performance-benchmarks.md) — detailed numbers
