# INT4 quant pipeline plan

This is the implementation plan for turning the current INT4 support from
weight-only stubs into a real no-fp32-fallback runtime path.

## Current state

- INT4 weights can be packed and routed through `packed_int4_gemv_dispatch` /
  `packed_int4_gemm_dispatch` for rank-2 MatMul/Gemm-like shapes.
- The tracker INT8 path now has real chain kernels for PW->DW, DW->PW,
  residual/fork suffixes, and ConvDQ tails. INT4 does not yet have equivalent
  graph-chain ownership.
- Activation edges are still INT8/QLinear-centric; INT4 activation tensors are
  not a runtime representation and should not be introduced until a benchmark
  proves they beat INT8 activations on the target hardware.

## Target contract

- No packed-INT4 node may unpack to fp32 inside the hot loop.
- Every INT4 fast path has scalar + x86 + aarch64 coverage, or stays scalar
  until both SIMD sides are ready.
- INT4 outputs must match the documented reference path within the quantized
  integer contract for the op. Any relaxed accuracy mode needs an explicit env
  gate.
- Bench output must split `int4_fast`, `int4_fallback`, and materialization
  counters so a green run cannot hide fp32 fallback.

## Chunk I4-1: make counters honest

- Add runtime counters for packed INT4 GEMV/GEMM fast path, fallback path, and
  unpack/materialization.
- Extend `kernel_bench` and LLM bench JSON with those counters.
- Add a profile filter that lists any MatMul/Gemm that dequantizes packed INT4
  to fp32.

## Chunk I4-2: finish packed MatMul/GEMM kernels

- Audit `packed_int4_gemv_dispatch` and `packed_int4_gemm_dispatch` shape
  coverage against TinyLlama / Qwen / Phi / Llama layer tables.
- Add missing AVX2/AVX-512 and NEON dot kernels for the common group sizes.
- Keep scalar reference as the bitwise oracle and add shape tests for every
  model-family row in `kernel_bench`.
- Prepack static INT4 RHS at load time and forbid hot-loop repacking.

## Chunk I4-3: LLM graph ownership

- Add loader actions for `MatMul/Gemm -> Add/Bias -> activation` where the RHS
  is packed INT4.
- Fuse bias, residual, and activation into the INT4 matmul epilogue.
- Ensure KV-cache decode uses GEMV kernels and prefill uses GEMM kernels
  without changing graph outputs.

## Chunk I4-4: optional INT4 Conv path

- Do not add INT4 Conv until tracker/LLM profiling shows a real target.
- If needed, start with pointwise Conv as packed INT4 RHS GEMM, not general
  im2col Conv.
- Depthwise INT4 is a separate kernel family and should only land with
  measured wins over INT8 depthwise.

## Chunk I4-5: final gates

- For each model family: scalar parity tests, SIMD parity tests, and 3x200
  release benchmarks.
- Compare against ORT where ORT has an equivalent quantized export; compare
  against llama.cpp/GGUF for LLM decode throughput.
- Update docs with before/after numbers and the exact fallback count. Target:
  `int4_fallback=0` on supported INT4 exports.
