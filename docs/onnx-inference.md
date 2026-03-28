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

## Metal GPU (macOS)

Build with the `metal-backend` feature:
```bash
cargo run --release --features metal-backend
```

### MPSGraph (recommended — fastest)

MPSGraph compiles the entire ONNX model into an optimized GPU execution plan and runs it as a single dispatch. This eliminates per-op encoder transitions and lets Apple's Metal driver fuse operations automatically.

```rust
use yscv_onnx::{
    compile_mpsgraph_plan, run_mpsgraph_plan,
    load_onnx_model_from_file,
};
use yscv_tensor::Tensor;

let model = load_onnx_model_from_file("model.onnx")?;
let input_name = model.inputs[0].clone();
let input_tensor = Tensor::zeros(vec![1, 3, 640, 640])?;

// Compile (one-time cost, ~10-30ms)
let plan = compile_mpsgraph_plan(&model, &input_name, &input_tensor)?;

// Run — input as raw f32 slice, output as named tensors
let input_data = vec![0.5f32; 1 * 3 * 640 * 640];
let outputs = run_mpsgraph_plan(&plan, &input_data)?;
for (name, tensor) in &outputs {
    println!("{}: {:?}", name, tensor.shape());
}
```

MPSGraph supports: Conv (including depthwise), Relu, Sigmoid, Add/Sub/Mul/Div, MaxPool, AvgPool, GlobalAvgPool, BatchNorm, Concat, Reshape, Flatten, Transpose, Softmax, Resize, MatMul, Split, Slice, Squeeze, Unsqueeze, Expand.

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

| | MPSGraph | Metal per-op | CPU |
|---|---------|-------------|-----|
| Speed | Fastest (4.8-7.8ms) | Fast (22-47ms) | Baseline (33-124ms) |
| Op coverage | Most common ops | All ops | All ops |
| Dynamic shapes | Static only | Static only | Dynamic |
| Platform | macOS (Apple Silicon) | macOS (Apple Silicon) | All |
| Feature flag | `metal-backend` | `metal-backend` | none |

Recommended approach: try MPSGraph first, fall back to Metal per-op if compilation fails, fall back to CPU for unsupported platforms.

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
| CPU | 32.7ms | — |
| Metal per-op | 22.1ms | 1.5x |
| **MPSGraph** | **4.8ms** | **6.8x** |
| onnxruntime CPU | 103.4ms | 0.32x |
| onnxruntime CoreML | 16.1ms | 2.0x |

### YOLO11n (6.5 GFLOP, 640x640 input)

| Backend | Min | Speedup vs CPU |
|---------|-----|---------------|
| CPU | 36.4ms | — |
| Metal per-op | 22.6ms | 1.6x |
| **MPSGraph** | **5.9ms** | **6.2x** |
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

128+ operators including Conv, ConvTranspose, MatMul, Gemm, BatchNormalization, GroupNormalization, LayerNormalization, all pooling variants, Relu, Sigmoid, Tanh, SiLU, GELU, Softmax, Resize, Concat, Split, Reshape, Transpose, Gather, Slice, and more. See `crates/yscv-onnx/src/runner/` for the full list.

## Troubleshooting

- **Slow inference**: Ensure you build with `--release`. Debug builds are 10-50x slower.
- **MPSGraph compilation fails**: Falls back to Metal per-op or CPU. Usually caused by unsupported ops like dynamic Shape→Gather→Reshape chains. Run with `RUST_LOG=debug` for details.
- **Metal compilation fails**: Check that your model uses supported ops. Run with `METAL_DEBUG=1` for detailed op tracing.
- **Numerical differences (Metal vs CPU)**: Expected — Metal uses f16 intermediates. Max diff < 0.05 is normal for most models.
