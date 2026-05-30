# ONNX CPU Kernels (Conv/MatMul) — Runtime Map

This note documents the current hot CPU kernel paths used by `yscv-onnx`
for tracker-class models, where assembly is used, where NEON/AVX
intrinsics are used, and which env vars are intended only for A/B.

## Status

For current cross-thread and cross-hardware benchmark numbers (fp32, QDQ,
and explicit QLinear paths), see
[`performance-benchmarks.md`](performance-benchmarks.md). In short: yscv's
fp32 and QDQ-fast paths are competitive with ONNX Runtime on ARM SBCs and
trail it on desktop Zen 4; the explicit-QLinear path is an interoperability
and kernel bring-up target, not the default speed path. The default
quantized path is QDQ-fast with internal quant-domain boundary folding.

## Hot kernel paths

- `FusedPwDw` (pointwise expand -> depthwise 3x3):
  - runner entry: `crates/yscv-onnx/src/runner/conv/fused.rs` (`exec_fused_pw_dw`)
  - kernel: `crates/yscv-kernels/src/ops/fused_pw_dw_3x3/`
  - default: streaming path ON (kill-switch only).
- `FusedDwPw` (depthwise 3x3 -> pointwise 1x1):
  - runner entry: `crates/yscv-onnx/src/runner/conv/fused.rs` (`exec_fused_dw_pw`)
  - kernel: `crates/yscv-kernels/src/ops/conv/mod.rs` (`fused_dw_pw_nhwc_streaming`)
  - default: streaming ON for supported shapes, padded streaming OFF by default.
- pointwise/MatMul heavy ops:
  - `crates/yscv-kernels/src/ops/matmul/`
  - blocked GEMM + row GEMM dispatch, BLAS routing where enabled.
- explicit quantized depthwise:
  - runner entry: `crates/yscv-onnx/src/runner/conv/quantized.rs` (`exec_qlinear_conv`)
  - kernel: `crates/yscv-kernels/src/ops/int8_depthwise.rs`
  - default: symmetric `QLinearConv` depthwise 3x3/5x5 stride-1 and measured-win
    stride-2 tracker weights are packed once at load and executed through
    AVX-512/AVX2/NEON/scalar dispatch. Large NCHW stride-2 tracker
    depthwise layers (`ih > 64`) use a direct NCHW activations +
    KHWC-packed-weights scalar path to avoid a losing NCHW→NHWC layout
    materialisation.
- explicit quantized pointwise/MatMul:
  - runner entry: `crates/yscv-onnx/src/runner/conv/quantized.rs` (`exec_qlinear_conv`)
    and `crates/yscv-onnx/src/runner/linear.rs` (`exec_qlinear_matmul`)
  - kernel: `crates/yscv-kernels/src/ops/int8_matmul.rs`
  - default: VNNI-friendly RHS matrices (`K >= 4`, `N % 16 == 0`) are packed
    once at model load into both transposed-B and AVX-512 VNNI 4x16 layouts.
    The prepacked dispatch reuses the 4x16 bytes and column sums instead of
    allocating/packing them on every inference.
- explicit quantized boundaries:
  - runner entry: `crates/yscv-onnx/src/runner/misc.rs` (`exec_quantize_linear`)
  - kernel: `crates/yscv-kernels/src/ops/quantize.rs`
  - default: per-tensor `QuantizeLinear` uses scalar/AVX-512/AVX2/NEON dispatch
    while preserving yscv's rounded-f32 signed-int8 storage. The runtime
    execution plan also folds matching
    `DequantizeLinear -> [Relu] -> QuantizeLinear` boundaries into a
    quant-domain action when scale and zero-point match, avoiding
    unnecessary fp32 materialisation between adjacent QLinear ops.
    Set `YSCV_QUANT_INT8_FAST=0` to disable this internal boundary
    folding for A/B; explicit QLinearConv kernels still run so the model
    remains valid.
- fused INT8 quant-domain chains:
  - runner entry: `crates/yscv-onnx/src/runner/conv/quantized.rs`
    (`exec_quantized_pw_dw`, `exec_quantized_dw_pw`)
  - kernels: `crates/yscv-kernels/src/ops/int8_fused_pw_dw_3x3.rs`,
    `crates/yscv-kernels/src/ops/int8_fused_dw_pw_3x3.rs`
  - default: `NodeAction::QuantizedPwDw` fuses the inverted-bottleneck
    opening pair `QLinearConv(pw 1x1) -> DequantizeLinear -> [Relu] ->
    QuantizeLinear -> QLinearConv(dw 3x3 or 5x5)` into a single kernel
    call. The kernel streams NHWC i8 PW outputs through a per-worker
    `kh`-row ring buffer so the boundary i8 tensor never round-trips
    through main memory, requantises on the fly into the next i8, and
    writes the final NCHW i8 output. `NodeAction::QuantizedDwPw` is the
    mirror for the closing pair `QLinearConv(dw kxk) -> DequantizeLinear
    -> [Relu] -> QuantizeLinear -> QLinearConv(pw 1x1)`: each output-row
    chunk computes one DW i32 row from NHWC input directly, requantises
    to i8, and the PW reduction matmuls that row to the chain output
    immediately. Both kernels are bitwise-identical to the unfused
    per-op path (zero zero-point gates and per-tensor scales required).
    Multi-arch dispatch: `int8_matmul_prepacked_dispatch` for the PW dot
    (AVX-512-VNNI / AVX-VNNI / NEON SDOT / scalar) plus AVX-512BW /
    AVX2 / NEON widen-mul DW row reducers. The per-row requant epilogue
    (`(acc + bias) as f32 * composite + y_zp`, round-half-away-from-zero,
    clamp to `[-128, 127]`, optional Relu) runs through the shared
    `int8_requant::requant_i32_row_to_i8_dispatch` (AVX-512BW 16-lane /
    AVX2+SSE4.1 8-lane / NEON 4-lane / scalar) and is bitwise-identical
    to the per-op QLinearConv requant in `runner/conv/quantized.rs`. The
    closing-pair PW weight
    is force-prepacked even when `c_out` is not a multiple of 16 (head
    layers like `bbox_pred` `c_out=4` or `cls_pred` `c_out=1`); the
    transposed-B fallback inside `pack_i8_b_for_matmul` handles those
    shapes when the VNNI 4x16 layout is unavailable. Setting
    `YSCV_QUANT_INT8_FAST=0` falls back to the unfused per-op path for
    A/B; the chain matcher still fires at load time but every call
    reverts to per-op execution.

## Assembly vs intrinsics

Assembly microkernels live in:

- `crates/yscv-kernels/src/asm/x86_64_sysv.S`
- `crates/yscv-kernels/src/asm/x86_64_win64.S`
- `crates/yscv-kernels/src/asm/aarch64.S`

Covered families include:

- x86_64: SGEMM 4x8, 4x24 fused epilogue, 4x32 AVX-512, 12x32 AVX-512.
- aarch64: SGEMM 4x24 NEON, 8x12 NEON.

Important: the tracker's fused PW->DW path on aarch64 (`fused_pw_dw_3x3/`)
is currently NEON intrinsics (including PW2X mode), not inline asm.

## Runtime env knobs (A/B / rollback)

These vars are primarily for measurement and controlled rollback, not for
default production tuning:

| Env var | Default | Purpose |
|---|---|---|
| `YSCV_FUSED_PW_DW_STREAM_OFF=1` | unset | Disable `FusedPwDw` streaming fast path. |
| `YSCV_FUSED_PW_DW_PW2X_OFF=1` | unset | Disable NEON two-column PW path (PW2X). |
| `YSCV_FUSED_PW_DW_W_TILE=<N>` | auto | Override strip-mining tile width in `fused_pw_dw_3x3`. |
| `YSCV_FUSED_DW_PW_STREAM_OFF=1` | unset | Disable `FusedDwPw` streaming path. |
| `YSCV_FUSED_DW_PW_STREAM_MT=1` | unset | Allow `FusedDwPw` streaming in multi-thread mode. |
| `YSCV_FUSED_DW_PW_STREAM_PADDED=1` | unset | Enable padded streaming variant for `FusedDwPw`. |
| `YSCV_FUSED_DW_PW_TRUE_FUSED_ON=1` | unset | Force true-fused DM=1 path (kernel-level A/B). |
| `YSCV_FUSED_DW_PW_TRUE_FUSED_OFF=1` | unset | Force-disable true-fused DM=1 path. |
| `YSCV_QUANT_INT8_FAST=0` | unset | Disable internal quant-domain boundary folding while keeping standard QLinear kernels enabled. |
| `YSCV_DIRECT_CONV_WORK_MAX=<N>` | arch/thread auto | Threshold for direct 3x3 conv path. |
| `YSCV_NO_AARCH64_LOW_K_BLOCKED=1` | unset | Disable aarch64 low-k blocked matmul route. |
| `YSCV_AARCH64_LOW_K_BLOCKED_MIN_WORK_FMAS=<N>` | 500000 | Low-k blocked matmul activation threshold. |

## Reproduction commands (tracker)

```bash
cd onnx-fps

for t in 1 2 3 4; do
  ./target/release/onnx-fps \
    --model ../model.onnx \
    --input input.1:1x3x128x128 \
    --input input.249:1x3x256x256 \
    --iters 200 --threads "$t" --text \
    | grep -E '^Threads:|^p50:|^avg:'
done
```

```bash
cd onnx-fps

for t in 1 2 3 4; do
  python3 ./bench_ort_onnx_fps.py \
    --model ../model.onnx \
    --input input.1:1x3x128x128 \
    --input input.249:1x3x256x256 \
    --iters 200 --threads "$t" --text \
    | grep -E '^Threads:|^p50:|^avg:'
done
```
