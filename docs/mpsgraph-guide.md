# MPSGraph guide — fastest ONNX inference on Apple Silicon

The `metal-backend` feature gives you **MPSGraph** — Apple's compile-once
run-many GPU inference engine — wrapped in a simple Rust API. On M-series
Macs it's typically **5-15× faster than the CPU runner** and **1.4× faster
than ORT CoreML** on the same model.

This doc covers everything: when to use it, the full API, pipelined mode,
multi-input models, and troubleshooting.

> **New here?** Read [`QUICKSTART.md`](../QUICKSTART.md) first — this guide
> assumes you already have a working ONNX inference loop.

---

## TL;DR

```toml
# Cargo.toml
[dependencies]
yscv-onnx   = { version = "0.1", features = ["metal-backend"] }
yscv-tensor = "0.1"
```

```rust
use yscv_onnx::{compile_mpsgraph_plan, run_mpsgraph_plan, load_onnx_model_from_file};

let model = load_onnx_model_from_file("yolov8n.onnx")?;
let plan = compile_mpsgraph_plan(&model, &[("images", &input_tensor)])?;

// Hot loop: no re-compilation.
loop {
    let outputs = run_mpsgraph_plan(&plan, &[("images", input_bytes)])?;
    process(&outputs);
}
```

That's the sync path. For sustained throughput (drone video, real-time
tracking), use the pipelined path — see [Pipelined mode](#pipelined-mode)
below.

---

## When to use which backend

| You want... | Use | Why |
|---|---|---|
| Develop + test without Metal | CPU runner (`run_onnx_model`) | Zero deps, works everywhere |
| Highest throughput on macOS | **MPSGraph** | 5-15× CPU, 1.4× ORT CoreML |
| Cross-platform GPU (Linux/Windows too) | wgpu (`--features gpu`) | Vulkan / Metal / DX12 |
| Rockchip edge device | RKNN (`--features rknn`) | Hardware NPU, ~1-3 ms per frame |
| Apple Neural Engine specifically | Not supported | Use ORT's CoreMLExecutionProvider — but benchmarks say we're still faster via MPSGraph on most models |

MPSGraph only runs on macOS 13+ with Apple Silicon. On Intel Macs it
falls back to AMD GPU via Metal (slower but functional).

---

## How MPSGraph works under the hood

Three-stage lifecycle:

```
┌─────────────────────┐
│   Your ONNX model   │
└──────────┬──────────┘
           │
           │ compile_mpsgraph_plan(&model, &[(name, shape_tensor), ...])
           │   • Graph optimizations (constant folding, fusion, dead-op elim)
           │   • Kernel selection per GPU generation (M1 vs M4)
           │   • fp16 promotion where numerically safe
           │   • Metal intermediate representation → final GPU shaders
           │   • Pre-allocate triple-buffered input/output MTLBuffers
           ↓
    ┌──────────────┐
    │ MpsGraphPlan │   ← keep this, reuse for every frame
    └──────┬───────┘
           │
           │ run_mpsgraph_plan(&plan, &[(name, bytes), ...])     ← sync
           │   or
           │ submit_mpsgraph_plan(&plan, ...)  → wait(...)       ← async pipelined
           ↓
    ┌──────────────────┐
    │ Output HashMap   │
    │ name → Vec<f32>  │
    └──────────────────┘
```

The heavy lifting is in `compile_mpsgraph_plan`. It runs **once** at
startup (150-300 ms typically). The per-frame cost is just buffer copy
+ GPU dispatch + readback.

---

## Sync API — when you call inference once per frame

Use when: single image, batch scoring, or frame rate already satisfies
your budget.

```rust
use yscv_onnx::{compile_mpsgraph_plan, run_mpsgraph_plan, load_onnx_model_from_file};
use yscv_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = load_onnx_model_from_file("yolov8n.onnx")?;

    // Provide a sample tensor with the exact shape your graph expects.
    // Only the shape is used for compile; values can be zeros.
    let sample = Tensor::zeros(vec![1, 3, 640, 640])?;
    let plan = compile_mpsgraph_plan(&model, &[("images", &sample)])?;

    // Real inputs — as f32 little-endian bytes this time.
    let real_input: Vec<f32> = preprocess("photo.jpg")?;
    let bytes = bytemuck::cast_slice::<f32, u8>(&real_input);

    let outputs = run_mpsgraph_plan(&plan, &[("images", bytes)])?;
    let detections: &Vec<f32> = outputs.get("output0").expect("missing output");

    println!("got {} detection values", detections.len());
    Ok(())
}
```

Key points:

- **Compile once, run many.** Never call `compile_mpsgraph_plan` inside
  your hot loop. Store the `MpsGraphPlan` in your state.
- **Shape-based compile.** The sample tensor at compile time only needs
  the right shape. Values don't matter.
- **Bytes in, Vec\<f32\> out.** The run function takes raw f32 LE bytes
  (easy `bytemuck::cast_slice` from a `Vec<f32>`) and returns
  `HashMap<String, Vec<f32>>` keyed by output tensor name.

### What counts as "same shape"

MPSGraph specialises on the compile-time shape. If your real frame
shape differs — even by one pixel — inference fails. Either:

1. Resize every input to a fixed shape (standard for YOLO / ResNet / etc.)
2. Compile one plan per shape you care about
3. Use the CPU runner for dynamic shapes

Dynamic shapes are a documented gap — see [roadmap](#future-work).

---

## Pipelined mode — when you stream frames

Use when: sustained video rate, drone / FPV, live tracking.

The pipelined API lets you submit frame N+1 while GPU is still computing
frame N. CPU-side preprocessing (decode → resize → normalize) overlaps
with GPU compute. Typical gain: **2-3× sustained FPS** over sync.

```rust
use std::collections::VecDeque;
use yscv_onnx::{compile_mpsgraph_plan, submit_mpsgraph_plan, wait_mpsgraph_plan, InferenceHandle};

let plan = compile_mpsgraph_plan(&model, &[("images", &sample)])?;
const DEPTH: usize = 3;

// Prime the pipeline with DEPTH frames before we start waiting.
let mut in_flight: VecDeque<InferenceHandle> = VecDeque::new();
for _ in 0..DEPTH {
    let frame = camera.capture()?;
    in_flight.push_back(submit_mpsgraph_plan(&plan, &[("images", &frame)])?);
}

// Hot loop: wait for oldest, process, submit next.
loop {
    let oldest = in_flight.pop_front().unwrap();
    let outputs = wait_mpsgraph_plan(&plan, oldest)?;
    process(&outputs);

    let frame = camera.capture()?;
    in_flight.push_back(submit_mpsgraph_plan(&plan, &[("images", &frame)])?);
}
```

At any moment `DEPTH` frames are in flight: N on GPU, N-1 done awaiting
wait, N+1 being marshalled on CPU.

### Pipeline depth tuning

Set via env var before starting your binary:

```bash
YSCV_MPS_PIPELINE=3 ./my-app
```

- `YSCV_MPS_PIPELINE=1` — sync behaviour (same as `run_mpsgraph_plan`)
- `YSCV_MPS_PIPELINE=2` — usually the throughput peak
- `YSCV_MPS_PIPELINE=3` — default — best balance of throughput and
  tail latency
- `YSCV_MPS_PIPELINE=4..=8` — diminishing returns, but tightest p99
  tail latency

Depth > 8 is clamped to 8. Depth < 1 is clamped to 1.

### Back-pressure contract

If the caller submits more frames than the pipeline depth without
waiting, `submit_mpsgraph_plan` **transparently blocks** on the
oldest slot's previous command buffer. This is built-in back-pressure
— you never corrupt buffers or oversubscribe GPU memory, but you also
never see the pipeline "break".

### When to use sync vs pipelined

| Situation | Use |
|---|---|
| One-shot detection (photo analysis) | sync |
| Testing correctness | sync |
| Frame rate < 60 FPS is enough | sync |
| Streaming video / FPV / tracker | **pipelined, depth=3** |
| Tight p99 latency requirement | **pipelined, depth=4-5** |
| Benchmarking peak throughput | **pipelined, depth=2** |

---

## Multi-input models

Siamese trackers, two-tower rankers, transformer encoder-decoder pairs
all have more than one input. The API takes a slice so every input
appears with its name:

```rust
let template = Tensor::zeros(vec![1, 3, 128, 128])?;
let search   = Tensor::zeros(vec![1, 3, 256, 256])?;

let plan = compile_mpsgraph_plan(&model, &[
    ("template_input", &template),
    ("search_input",   &search),
])?;

// At run time:
let outputs = run_mpsgraph_plan(&plan, &[
    ("template_input", template_bytes),
    ("search_input",   search_bytes),
])?;
```

Inputs are matched by name, so order doesn't matter. The compile-time
shapes must match run-time shapes exactly.

---

## Fallback — when MPSGraph can't handle your graph

MPSGraph supports ~20 op kinds natively: Conv, MatMul, Relu, Softmax,
Resize, Concat, BatchNormalization, etc. If your model uses an
unsupported op (e.g. some custom layer, or exotic quantization), the
compile will fail or fall back partially.

Two fallback paths exist:

### 1. Metal per-op runner

Our second-tier Metal backend (`MetalPlan`) supports a broader set of
ops via individual kernel dispatches. Slower than MPSGraph but more
compatible:

```rust
use yscv_onnx::{compile_metal_plan, run_metal_plan};

let plan = compile_metal_plan(&model, &[("images", &sample)])?;
let outputs = run_metal_plan(&plan, &[("images", &input_f32)])?;
```

Use when MPSGraph compile fails with "unsupported op".

### 2. CPU runner

The universal fallback always works:

```rust
use yscv_onnx::run_onnx_model;
let outputs = run_onnx_model(&model, inputs_hashmap)?;
```

### Recommended pattern

Try MPSGraph, fall back to Metal per-op, fall back to CPU:

```rust
fn run_best_available(model: &ModelProto, inputs: &[(&str, &Tensor)]) -> Result<_, _> {
    if let Ok(plan) = compile_mpsgraph_plan(model, inputs) {
        return run_mpsgraph_plan(&plan, bytes_feeds)
            .map(wrap_outputs);
    }
    if let Ok(plan) = compile_metal_plan(model, inputs) {
        return run_metal_plan(&plan, f32_feeds)
            .map(wrap_outputs);
    }
    run_onnx_model(model, hashmap_feeds)
}
```

---

## Full API reference

All functions come from `yscv_onnx`. The `metal-backend` feature must
be enabled.

### `compile_mpsgraph_plan`

```rust
pub fn compile_mpsgraph_plan(
    model: &ModelProto,
    input_shapes: &[(&str, &Tensor)],
) -> Result<MpsGraphPlan, MpsGraphError>
```

Compile an ONNX graph into a Metal executable. Call once at startup.
**150-300 ms typical compile time** (once per plan, forever).

### `run_mpsgraph_plan` — sync

```rust
pub fn run_mpsgraph_plan(
    plan: &MpsGraphPlan,
    inputs: &[(&str, &[u8])],
) -> Result<HashMap<String, Vec<f32>>, MpsGraphError>
```

Synchronously run one frame. Returns outputs keyed by tensor name.
Input `&[u8]` is raw f32 little-endian — use `bytemuck::cast_slice`
from `Vec<f32>`.

### `submit_mpsgraph_plan` — async

```rust
pub fn submit_mpsgraph_plan(
    plan: &MpsGraphPlan,
    inputs: &[(&str, &[u8])],
) -> Result<InferenceHandle, MpsGraphError>
```

Submit a frame to GPU without blocking. Returns a handle to be passed
to `wait_mpsgraph_plan`. Back-pressures when pipeline depth exceeded.

### `wait_mpsgraph_plan` — collect async

```rust
pub fn wait_mpsgraph_plan(
    plan: &MpsGraphPlan,
    handle: InferenceHandle,
) -> Result<HashMap<String, Vec<f32>>, MpsGraphError>
```

Block until the GPU finishes the frame associated with `handle`, then
return outputs. Must be called exactly once per `submit` (else you
leak GPU resources + block the pool).

### `InferenceHandle`

```rust
pub struct InferenceHandle { /* opaque */ }
```

Returned by `submit`, consumed by `wait`. `#[must_use]` — dropping it
without calling `wait` debug-warns and loses the outputs.

---

## Performance expectations

Numbers from actual benchmarks, M-series Apple Silicon, fp32 inputs:

| Model | Sync latency | Pipelined×3 throughput | vs CPU runner |
|---|---:|---:|---:|
| YOLOv8n 640×640 | 4.8 ms | ~450 FPS sustained | 7× faster than CPU |
| YOLO11n 640×640 | 5.9 ms | ~350 FPS sustained | 6× faster than CPU |
| VballNet (DSConv) | 7.8 ms | ~280 FPS sustained | 16× faster than CPU |
| Siamese tracker (128+256) | 1.37 ms | **1510 FPS sustained** | 35× faster than CPU |

**peak burst on Siamese**: 2801 FPS (0.36 ms min latency).

See [`performance-benchmarks.md`](performance-benchmarks.md) for full
methodology + reproduction commands.

---

## Troubleshooting

### "compile_mpsgraph_plan failed: unsupported op 'X'"

MPSGraph doesn't support that op. Options:
1. Check the op name in the error — sometimes it's a fixable metadata
   issue (wrong attribute on a standard op)
2. Try `compile_metal_plan` — covers more ops
3. Fall back to `run_onnx_model` for this model

### "crashed with Metal assertion on Intel Mac"

Intel Macs use AMD GPU via Metal. Some MPSGraph paths only test on
Apple Silicon. Use `compile_metal_plan` instead on Intel Macs.

### "First call is 20× slower than subsequent ones"

Normal. First inference compiles + caches GPU shaders. Always do a
warmup iteration (or five) before measuring.

### "submit/wait order got confused, now the pool is stuck"

Ensure **exactly one wait per submit**. Dropping an `InferenceHandle`
without waiting does NOT reclaim the slot — next submit will block
forever on that slot's fence. Either always wait, or reset the plan
by dropping it.

### "Throughput lower than expected on pipelined path"

Check that your preprocessing runs on CPU (allowing overlap with GPU).
If preprocessing itself is GPU-bound (e.g. you're running resize on
Metal too), the pipeline can't overlap and degrades to sync latency.

### "I want to see what MPSGraph actually did"

Set env var:
```bash
YSCV_MPS_DEBUG=1 ./my-app 2>&1 | head -50
```
Prints the compile step plan + per-op routing decisions.

---

## Future work

Gaps we know about (in roadmap but not implemented):

- **Dynamic shapes** — MPSGraph supports them internally; we don't
  expose the API yet. Workaround: pre-compile multiple plans.
- **Custom ops** — ANE-only ops aren't wired through. Use Metal per-op.
- **f16 inputs/outputs** — Internal compute is f16, but input/output
  conversion is f32. Direct f16 I/O would save ~50% bandwidth on the
  marshal step.
- **Batched submit** — Submit N frames in one call, get back an array
  of handles. Currently each submit is one call.

---

## Next steps

- [`docs/onnx-inference.md`](onnx-inference.md) — CPU runner + quantization
- [`docs/cookbook.md`](cookbook.md) — recipes for YOLO, Siamese, etc.
- [`docs/performance-benchmarks.md`](performance-benchmarks.md) — more numbers, more models
- [`docs/edge-deployment.md`](edge-deployment.md) — Rockchip NPU path (the
  other fast backend)
- [`docs/pipeline-config.md`](pipeline-config.md) — TOML-driven pipeline
  wiring that uses MPSGraph under the hood
