# yscv-onnx

Pure Rust ONNX runtime. 121 CPU operators, graph optimization, multi-thread CPU dispatch, optional wgpu cross-platform GPU (Vulkan / Metal / DX12) and Apple MPSGraph backends.

```rust,ignore
use yscv_onnx::*;

let model = load_onnx_model("yolov8n.onnx")?;

// Multi-threaded (default, all CPU cores):
let runner = OnnxRunner::new(&model)?;

// Explicit thread count (6 physical cores is optimal for Ryzen):
let runner = OnnxRunner::with_threads(&model, 6)?;

// Single-threaded:
let runner = OnnxRunner::with_threads(&model, 1)?;

let input = Tensor::from_vec(vec![1, 3, 640, 640], image_data)?;
let outputs = runner.run(&[("images", &input)])?;
```

## Capabilities

- **Inference API variants**:
  - `run_onnx_model` (owned `HashMap<String, Tensor>`)
  - `run_onnx_model_borrowed` (borrowed `HashMap`)
  - `run_onnx_model_borrowed_slice` (borrowed `&[(&str, &Tensor)]`, no `HashMap` required)
- **128 ONNX CPU operators**: Conv, MatMul, Gemm, Relu/LeakyRelu/Sigmoid/Tanh/Gelu/Erf/Mish/HardSwish/Softmax/LogSoftmax, BatchNormalization/LayerNormalization/InstanceNormalization, MaxPool/AveragePool/GlobalAveragePool, Resize/Upsample, Concat/Split/Reshape/Flatten/Transpose/Gather/GatherElements/GatherND/ScatterElements/ScatterND/Slice/Tile/Expand, Cast/Pad/Clip/Where/Identity/CumSum/ArgMax/ArgMin/TopK, DepthToSpace/SpaceToDepth, GridSample/RoiAlign/NonMaxSuppression, full quantized stack (QuantizeLinear/DequantizeLinear/QLinearConv/QLinearMatMul/MatMulInteger/ConvInteger/DynamicQuantizeLinear), trig + hyperbolic, logical, fused `Conv_Relu` / `BatchNormalization_Relu`, … (verified against the dispatch in `src/runner/mod.rs`)
- **Graph optimizations**: constant folding, Conv+BN fusion, Conv+Relu fusion, dead node elimination
- **Runtime fusion**: Conv+SiLU (Conv→Sigmoid→Mul pattern), Conv+Relu, BN+Relu, Gemm+Relu, Add+Relu (in-place with buffer reuse), Conv+Add residual (in-place, buffer reuse), in-place Add (same-shape last-use)
- **NHWC layout**: Conv outputs stored in NHWC for cache-friendly depthwise/pointwise chains; per-slot layout tracking with automatic NCHW permute when needed. Conv+Add fusion captures NHWC flag before `env.remove()` to prevent layout corruption in residual chains.
- **GEMM store fusion**: bias+activation fused directly in blocked GEMM microkernel store phase (all architectures: AVX+FMA, AVX, SSE, NEON, scalar) via `GemmEpilogue` — eliminates separate memory pass for Conv output
- **Pointwise 1×1 fast path**: 1×1 stride-1 convolutions bypass the im2col+conv loop — reshaped directly to `[N*H*W, C_in] × [C_in, C_out]` matmul with fused bias+activation
- **Depthwise activation fusion**: ReLU fused into depthwise SIMD store (AVX+FMA, AVX, SSE, NEON) — eliminates separate activation pass
- **Bias preload**: bias vectors for common channel counts (16, 24) preloaded into SIMD registers before the row loop, eliminating per-row memory accesses
- **Quantization**: INT4 weight-only (`quantize_weights_int4`) + INT8 symmetric/asymmetric/per-channel inference support; per-tensor activation-statistics collection for PTQ via `CalibrationCollector` (install a `CalibrationScope` before running inference, read aggregated min/max with `snapshot()`); `MinMax → QuantParams` derivation in `quantize::derive` (asymmetric uint8 / symmetric int8 / per-channel int8 / per-channel int4)
- **LLM inference**: autoregressive `generate()` with KV-cache, top-k/top-p sampling, temperature, repetition penalty; RoPE and GroupQueryAttention for decoder-only transformers
- **GPU inference**: wgpu (Vulkan/Metal/DX12) and native Metal/MPSGraph plan compiler with triple-buffered pipelined `submit`/`wait` API (multi-input models, f16 end-to-end, zero-alloc hot path, ~20 op kinds, automatic CPU fallback for unsupported subgraphs)
- **Model export**: save optimized graphs back to ONNX format

## Features

```toml
[features]
gpu = []             # wgpu GPU inference
metal-backend = []   # macOS Metal (MPSGraph)
```

## Tests

166+ tests covering all 128 operator dispatch arms, shape inference, graph optimization, quantization, fusion, and model loading.
