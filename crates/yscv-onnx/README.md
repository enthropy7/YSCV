# yscv-onnx

Pure Rust ONNX runtime. 122 CPU operators, graph optimization, multi-thread CPU dispatch, optional wgpu cross-platform GPU (Vulkan / Metal / DX12) and Apple MPSGraph backends.

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
- **122 ONNX CPU operators**: Conv, MatMul, Gemm, Relu/LeakyRelu/Sigmoid/Tanh/Gelu/Erf/Mish/HardSwish/Softmax/LogSoftmax, BatchNormalization/LayerNormalization/InstanceNormalization, MaxPool/AveragePool/GlobalAveragePool, Resize/Upsample, Concat/Split/Reshape/Flatten/Transpose/Gather/GatherElements/GatherND/ScatterElements/ScatterND/Slice/Tile/Expand, Cast/Pad/Clip/Where/Identity/CumSum/ArgMax/ArgMin/TopK, DepthToSpace/SpaceToDepth, GridSample/RoiAlign/NonMaxSuppression, full quantized stack (QuantizeLinear/DequantizeLinear/QLinearConv/QLinearMatMul/MatMulInteger/ConvInteger/DynamicQuantizeLinear), trig + hyperbolic, logical, fused `Conv_Relu` / `BatchNormalization_Relu`, … (see the op dispatch in `src/runner/dispatch.rs`)
- **Graph optimizations**: constant folding, Conv+BN fusion, Conv+Relu fusion, dead node elimination
- **Runtime fusion**: Conv+SiLU (Conv→Sigmoid→Mul pattern), Conv+Relu, BN+Relu, Gemm+Relu, Add+Relu (in-place with buffer reuse), Conv+Add residual (in-place, buffer reuse), in-place Add (same-shape last-use)
- **NHWC layout**: Conv outputs stored in NHWC for cache-friendly depthwise/pointwise chains; per-slot layout tracking with automatic NCHW permute when needed. Conv+Add fusion captures NHWC flag before `env.remove()` to prevent layout corruption in residual chains.
- **Quantized Conv export**: QDQ/QLinear rewrites serialize Conv weights in standard OIHW form even when the loader had normalised them to internal KHWC, grouped-KHWC, or depthwise-KHWC layouts; yscv restores the fast NHWC layouts on reload while exported models stay ONNX/ORT-compatible.
- **QLinear depthwise fast path**: symmetric `QLinearConv` depthwise 3×3/5×5 stride-1 and measured-win stride-2 tracker weights are packed once at load and executed through reusable runner scratch buffers. NHWC activations route through the multi-arch INT8 depthwise kernel; large NCHW stride-2 tracker layers route through a direct NCHW/KHWC scalar path when that avoids a measured-losing layout materialisation.
- **QLinear pointwise/MatMul prepack**: VNNI-friendly INT8 RHS tensors (`K >= 4`, `N % 16 == 0`) are packed once at load into the AVX-512 VNNI 4×16 layout as well as transposed-B, so explicit QLinearConv pointwise layers do not repack weights per inference.
- **QLinear boundary fast path**: per-tensor `QuantizeLinear` writes internal i8 tensors through scalar/AVX2/AVX-512F/NEON dispatch with packed x86 i8 stores, and matching `DequantizeLinear -> [Relu] -> QuantizeLinear` boundaries stay in the quantized domain at runtime when scale/zero-point match. `QLinearConv` writes i8 outputs into the same side-table, so adjacent quantized ops no longer need f32-coded int8 tensors between them.
- **Quant runtime counters**: `reset_quant_runtime_stats()` /
  `quant_runtime_stats()` report executed QDQ-boundary folds,
  QLinearConv fast/fallback, QLinearMatMul fast/fallback, i8 stores, i8
  materialization, and fused INT8 chain (`quant_chain_executed`) counts.
  Set `YSCV_QUANT_INT8_FAST=0` to disable both the internal QDQ-boundary
  folding and the fused INT8 chain action so the runner falls back to the
  unfused per-op path while still using standard QLinear kernels (used as
  the bitwise reference in the test suite).
- **Fused INT8 PW->DW chain**: `NodeAction::QuantizedPwDw` fuses
  `QLinearConv(pw 1×1) -> DequantizeLinear -> [Relu] -> QuantizeLinear ->
  QLinearConv(dw 3×3 or 5×5)` into a single multi-arch SIMD kernel
  (`int8_fused_pw_dw_3x3`) when all five inputs/outputs use zero
  zero-points, per-tensor scales, and dilations 1. The fused kernel
  streams NHWC i8 PW outputs through a per-worker `kh`-row ring buffer
  (no main-memory boundary tensor between PW and DW), requantises on the
  fly into the next i8, and writes the final NCHW i8 output —
  bitwise-identical to the unfused per-op path.
- **Fused INT8 DW->PW chain**: `NodeAction::QuantizedDwPw` mirrors the
  PW->DW action for the inverted-bottleneck closing pair
  `QLinearConv(dw 3×3 or 5×5) -> DequantizeLinear -> [Relu] ->
  QuantizeLinear -> QLinearConv(pw 1×1)`, dispatched through the
  multi-arch SIMD kernel `int8_fused_dw_pw_3x3`. Each output-row chunk
  computes one DW i32 row from NHWC input directly, requantises to i8,
  and the PW reduction matmuls that row to the chain output
  immediately. The closing-pair PW weight is force-prepacked even when
  `c_out` is not a multiple of 16 (e.g. detection heads with
  `c_out ∈ {1, 4}`); the transposed-B fallback inside
  `pack_i8_b_for_matmul` handles those shapes when the VNNI 4×16
  layout is unavailable.
- **SIMD requant epilogue**: both fused INT8 chain kernels share a
  runtime-dispatched `int8_requant::requant_i32_row_to_i8_dispatch`
  path (AVX-512BW 16-lane / AVX2+SSE4.1 8-lane / NEON 4-lane / scalar
  fallback). Replaces the per-pixel scalar `(a as f32) * composite +
  y_zp` `.round().clamp(-128, 127) as i8` loop in the chain inner
  body. Bitwise-identical to the scalar reference and to the per-op
  QLinearConv requant in `runner/conv/quantized.rs`; the round-half-away-from-zero
  emulation uses a `0.5 - ULP/2 = f32::from_bits(0x3EFFFFFF)` bias to
  avoid the sub-ULP precision pitfall at `v ≈ 0.5`.
- **Residual/fork quant-chain kernels**: `NodeAction::QuantizedForkPair`,
  `QuantizedResidualChain`, and `QuantizedConvDq` cover the residual-style
  tail that cannot be owned by the pure PW->DW / DW->PW pair kernels because
  the midpoint feeds a side branch or a fp32 Conv/Add suffix. Forked PW->DW
  computes the PW matmul once, exposes the graph-visible f32 side tensor for
  residual `Add`, and feeds DW directly from the same NHWC i8 buffer.
  DW-residual suffixes compute DW in int8/i32, preserve QLinear rounding,
  run the fp32 pointwise residual GEMM, and quantize the result inside one
  action. ConvDQ tails compute pointwise QLinearConv and DQ without a
  standalone QLinearConv op. Together these close the residual-style tail
  so no unfused `QLinearConv` is left in the quantized chain
  (`quant_chain_fallback` reaches 0).
- **QLinear multi-thread cleanup**: INT8 depthwise accumulation now has a
  row/pixel-parallel dispatch path, the residual/fork chain glue parallelizes
  NCHW/NHWC layout conversion and requant/dequant rows, and large
  `QuantizeLinear` activation entries split work across the active runner
  pool while still calling the same scalar/AVX2/AVX-512F/NEON quant kernels
  inside each chunk, improving 6T scaling on the quantized path.
- **Tracker quant-chain audit**: `bench_tracker` reports static QLinear
  Conv-chain candidates with the same greedy non-overlapping ownership as
  the runtime plan (`PW->DW`, `DW->PW`, residual/fork suffixes, and split
  `QLinearConv -> DQ` boundaries) next to runtime counters
  (`quant_chain_executed`, `quant_chain_fallback`), making it explicit when a
  quantized model still falls back to per-node QLinearConv instead of planned
  INT8 chain execution.
- **GEMM store fusion**: bias+activation fused directly in blocked GEMM microkernel store phase (all architectures: AVX+FMA, AVX, SSE, NEON, scalar) via `GemmEpilogue` — eliminates separate memory pass for Conv output
- **Pointwise 1×1 fast path**: 1×1 stride-1 convolutions bypass the im2col+conv loop — reshaped directly to `[N*H*W, C_in] × [C_in, C_out]` matmul with fused bias+activation
- **Depthwise activation fusion**: ReLU fused into depthwise SIMD store (AVX+FMA, AVX, SSE, NEON) — eliminates separate activation pass
- **Bias preload**: bias vectors for common channel counts (16, 24) preloaded into SIMD registers before the row loop, eliminating per-row memory accesses
- **Quantization**: INT4 weight-only (`quantize_weights_int4`) + INT8 symmetric/asymmetric/per-channel inference support; per-tensor activation-statistics collection for PTQ via `CalibrationCollector` (install a `CalibrationScope` before running inference, read aggregated min/max with `snapshot()`); `MinMax → QuantParams` derivation in `quantize::derive` (asymmetric uint8 / symmetric int8 / per-channel int8 / per-channel int4)
- **Fused-chain calibration**: calibration runs record local DW/PW intermediates inside zero-env fused Conv chains; normal inference keeps the streaming/no-intermediate path, while PTQ gets the activation stats needed for QLinear/QDQ coverage.
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
