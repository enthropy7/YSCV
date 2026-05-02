# ONNX Inference Guide

How to run ONNX models with yscv — CPU, Metal per-op, and MPSGraph.

## Quick Start (CPU)

```rust
use std::collections::HashMap;
use yscv_onnx::{load_onnx_model_from_file, run_onnx_model};
use yscv_tensor::Tensor;

let model = load_onnx_model_from_file("model.onnx")?;
let input_name = model.inputs[0].clone();

let input = Tensor::zeros(vec![1, 3, 640, 640])?;
let mut inputs = HashMap::new();
inputs.insert(input_name, input);

let outputs = run_onnx_model(&model, inputs)?;
```

CPU inference is fully optimized by default:
- **Multi-threaded**: All ops above 262K elements use rayon parallelism automatically
- **SIMD**: NEON (aarch64), AVX (x86_64), SSE (x86) — runtime detected, no flags needed
- **BLAS**: Accelerate (macOS) or OpenBLAS (Linux) for matmul and standard conv

No special configuration required. Just build with `--release`.

For kernel-level details (fused Conv paths, asm vs intrinsics, and CPU
A/B env toggles), see [`onnx-cpu-kernels.md`](onnx-cpu-kernels.md).

## Metal GPU (macOS)

Build with the `metal-backend` feature:
```bash
cargo run --release --features metal-backend
```

### MPSGraph (recommended — fastest)

MPSGraph compiles the entire ONNX model into an optimized GPU execution plan and runs it as a single dispatch. The plan is triple-buffered by default — each `submit` lands in a different slot, so CPU can marshal frame N+1 while the GPU finishes frame N.

Zero-allocation hot path: `MPSGraphTensorData` objects and their NSArray wrappers are built once at compile time, output buffers are pre-allocated f16 `StorageModeShared`, and the final `f16→f32` widening happens on the CPU via aarch64 `vcvt_f32_f16`.

#### Synchronous API

```rust
use yscv_onnx::{
    compile_mpsgraph_plan, run_mpsgraph_plan,
    load_onnx_model_from_file,
};
use yscv_tensor::Tensor;

let model = load_onnx_model_from_file("model.onnx")?;
let input_name = model.inputs[0].clone();
let input_tensor = Tensor::zeros(vec![1, 3, 640, 640])?;

// Compile (one-time cost, ~10-80ms). `inputs` is a slice of
// (name, tensor) pairs — one entry per ONNX graph input.
let plan = compile_mpsgraph_plan(&model, &[(input_name.as_str(), &input_tensor)])?;

// Run — each input as a named raw f32 slice. Order doesn't matter;
// lookup is by name. Outputs come back as a HashMap.
let input_data = vec![0.5f32; 1 * 3 * 640 * 640];
let outputs = run_mpsgraph_plan(&plan, &[(input_name.as_str(), input_data.as_slice())])?;
for (name, tensor) in &outputs {
    println!("{}: {:?}", name, tensor.shape());
}
```

#### Pipelined (async) API — 3–5× higher throughput

For sustained high-throughput (batch inference, video processing), keep multiple frames in flight:

```rust
use std::collections::VecDeque;
use yscv_onnx::{submit_mpsgraph_plan, wait_mpsgraph_plan, InferenceHandle};

// Prime the pipeline — 2 to 4 frames is usually the sweet spot.
let mut in_flight: VecDeque<InferenceHandle> = VecDeque::new();
let pipeline_depth = 3;
for _ in 0..pipeline_depth {
    in_flight.push_back(submit_mpsgraph_plan(&plan, &feeds)?);
}

// Steady state: submit a new frame and collect the oldest one.
for _ in 0..remaining_frames {
    let oldest = in_flight.pop_front().unwrap();
    let outputs = wait_mpsgraph_plan(&plan, oldest)?;  // GPU is ≥ depth frames back
    process(outputs);
    in_flight.push_back(submit_mpsgraph_plan(&plan, &feeds)?);
}

// Drain the last in-flight frames.
while let Some(h) = in_flight.pop_front() {
    let outputs = wait_mpsgraph_plan(&plan, h)?;
    process(outputs);
}
```

Pipeline depth is set by the `YSCV_MPS_PIPELINE` env var (default `3`, clamped to `1..=8`). The plan allocates that many independent input/output buffer sets; `submit` picks the next one round-robin and only blocks if the caller has more outstanding handles than the pipeline depth (built-in back-pressure — no silent buffer aliasing).

`InferenceHandle` is `#[must_use]`: drop one without calling `wait_mpsgraph_plan` and the next `submit` to that slot will transparently block waiting for the GPU before reusing its buffers.

MPSGraph supports: Conv (including depthwise), Relu, Sigmoid, Exp, Identity, Constant, Add/Sub/Mul/Div, MaxPool, AvgPool, GlobalAvgPool, BatchNorm, Concat, Reshape, Flatten, Transpose, Softmax, Resize, MatMul, Split, Slice, Squeeze, Unsqueeze, Expand.

### Metal Per-Op (fallback)

Per-op Metal compiles the ONNX graph into a sequence of `MetalOp`s dispatched via Metal command buffers. Use this when MPSGraph doesn't support all ops in your model (e.g., dynamic reshape chains in attention blocks).

```rust
use yscv_onnx::{compile_metal_plan, run_metal_plan, load_onnx_model_from_file};
use yscv_tensor::Tensor;

let model = load_onnx_model_from_file("model.onnx")?;
let input_tensor = Tensor::zeros(vec![1, 3, 640, 640])?;

// Compile: builds fused Metal command buffer
let plan = compile_metal_plan(&model, "images", &input_tensor)?;
println!("Compiled: {} ops", plan.ops_count());

// Run: input as raw f32 slice
let input_data = vec![0.5f32; 1 * 3 * 640 * 640];
let outputs = run_metal_plan(&plan, &input_data)?;
```

Per-op Metal benefits:
- Winograd F(4x4, 3x3) for 3x3 convolutions
- MPS GEMM for 1x1 and large convolutions
- f16 inter-op pipeline (halves memory bandwidth)
- Fused Conv+Activation+Residual kernels
- Supports all ONNX ops that the CPU runner supports

### Choosing a backend

| | MPSGraph (pipelined) | MPSGraph (sync) | Metal per-op | CPU |
|---|---|---|---|---|
| Speed | Fastest (0.37-3.5ms p50) | Fast (1.3-5ms) | Fast (22-47ms) | Baseline (33-124ms) |
| Multi-input | ✅ | ✅ | ❌ (single input) | ✅ |
| Op coverage | Most common ops | Most common ops | All ops | All ops |
| Dynamic shapes | Static only | Static only | Static only | Dynamic |
| Platform | macOS (Apple Silicon) | macOS (Apple Silicon) | macOS (Apple Silicon) | All |
| Feature flag | `metal-backend` | `metal-backend` | `metal-backend` | none |

Recommended approach: try pipelined MPSGraph first (for throughput) or sync MPSGraph (for single-shot latency); fall back to Metal per-op if compilation fails; fall back to CPU for non-macOS platforms.

## Benchmark Examples

```bash
# Full comparison: CPU vs Metal per-op vs MPSGraph (all models)
cargo run --release --example bench_mpsgraph --features metal-backend

# Specific model
cargo run --release --example bench_mpsgraph --features metal-backend -- path/to/model.onnx 50

# CPU-only YOLO benchmark
cargo run --release --example bench_yolo
```

## Performance Reference (Apple M1 MacBook Air)

### YOLOv8n (8.7 GFLOP, 640x640 input)

| Backend | Min | Speedup vs CPU |
|---------|-----|---------------|
| CPU | 31.7ms | — |
| Metal per-op | 12.1ms | 2.6x |
| **MPSGraph** | **3.5ms** | **9.1x** |
| onnxruntime CPU | 100.8ms | 0.31x |
| onnxruntime CoreML | 14.2ms | 2.2x |

### YOLO11n (6.5 GFLOP, 640x640 input)

| Backend | Min | Speedup vs CPU |
|---------|-----|---------------|
| CPU | 34.3ms | — |
| Metal per-op | 12.6ms | 2.7x |
| **MPSGraph** | **5.0ms** | **6.9x** |
| onnxruntime | FAIL | Opset 22 unsupported |

### VballNetGrid (16.3 GFLOP, 432x768 input)

| Backend | Min | Speedup vs CPU |
|---------|-----|---------------|
| CPU | 124.1ms | — |
| Metal per-op | 47.3ms | 2.6x |
| **MPSGraph** | **7.8ms** | **15.9x** |
| onnxruntime CPU | 196.7ms | 0.63x |
| onnxruntime CoreML CPU_ONLY | 8.6ms | 14.4x |

## Supported ONNX Operations

122 CPU operators (verified against the dispatch `match` arms in `crates/yscv-onnx/src/runner/mod.rs`), including Conv, ConvTranspose, MatMul, Gemm, BatchNormalization, LayerNormalization, InstanceNormalization, LpNormalization, all pooling variants (MaxPool, AveragePool, GlobalAveragePool), Relu, LeakyRelu, Sigmoid, Tanh, Gelu, Erf, HardSigmoid, Selu, Celu, ThresholdedRelu, Hardmax, Softplus, Softsign, HardSwish, Mish, Softmax, LogSoftmax, Resize, Upsample, Concat, Split, Reshape, Flatten, Transpose, Gather, GatherElements, GatherND, ScatterElements, ScatterND, Slice, Tile, Expand, Shape, Range, OneHot, NonZero, ConstantOfShape, Cast, Pad, Clip, Where, Identity, CumSum, ArgMax, ArgMin, TopK, DepthToSpace, SpaceToDepth, GridSample, RoiAlign, Compress, NonMaxSuppression, LRN, Mean, Sum, Min, Max, Mod, BitShift, IsNaN, IsInf, Sign, Round, sin/cos/tan/asin/acos/atan/sinh/cosh/asinh/acosh/atanh, Equal/Greater/Less/GreaterOrEqual/LessOrEqual, And/Or/Xor/Not, plus quantized variants (QuantizeLinear, DequantizeLinear, QLinearConv, QLinearMatMul, MatMulInteger, ConvInteger, DynamicQuantizeLinear), Einsum, ReduceMean/Sum/Max/Min/Prod/L1/L2, and the fused `Conv_Relu` / `BatchNormalization_Relu` patterns the graph optimizer emits. The Metal/MPSGraph plan compiler in `crates/yscv-onnx/src/runner/metal/record.rs` covers a smaller per-op subset (~17 distinct ops); models that fall outside the Metal subset transparently fall back to the CPU runner.

## Troubleshooting

- **Slow inference**: Ensure you build with `--release`. Debug builds are 10-50x slower.
- **MPSGraph compilation fails**: Falls back to Metal per-op or CPU. Usually caused by unsupported ops like dynamic Shape→Gather→Reshape chains. Run with `RUST_LOG=debug` for details.
- **Metal compilation fails**: Check that your model uses supported ops. Run with `METAL_DEBUG=1` for detailed op tracing.
- **Numerical differences (Metal vs CPU)**: Expected — Metal uses f16 intermediates. Max diff < 0.05 is normal for most models.
