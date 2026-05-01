# ONNX CPU Kernels (Conv/MatMul) — Runtime Map

This note documents the current hot CPU kernel paths used by `yscv-onnx`
for tracker-class models, where assembly is used, where NEON/AVX
intrinsics are used, and which env vars are intended only for A/B.

## Current public status (Orange Pi Zero 3, 2026-04-21)

Siamese tracker (`model.onnx`, sha256
`6336fbde82e3996128cd18e2141682c7a6b9a7575018ca9ffee974df546f22ab`),
`--iters 200`, same inputs/threads for both engines:

| Threads | yscv p50 | ORT p50 | yscv vs ORT |
|---:|---:|---:|---:|
| 1 | **461.63 ms** | 499.25 ms | **1.08× faster** |
| 2 | **252.08 ms** | 273.18 ms | **1.08× faster** |
| 3 | **192.91 ms** | 199.41 ms | **1.03× faster** |
| 4 | **150.17 ms** | 164.56 ms | **1.10× faster** |

## Current desktop tracker status (Zen 4, 2026-04-29)

Private two-input tracker (`private/private/model.onnx`), synthetic
zero/random benchmark inputs for speed only, `--iters 200`, median p50:

| Model / path | Threads | yscv p50 | ORT p50 | Status |
|---|---:|---:|---:|---|
| fp32 original | 1 | 11.34 ms | **8.03 ms** | yscv 1.41× slower |
| fp32 original | 6 | 2.93 ms | **1.80 ms** | yscv 1.63× slower |
| QDQ-fast export | 1 | 11.60 ms | **8.16 ms** | yscv 1.42× slower |
| QDQ-fast export | 6 | 3.44 ms | **1.84 ms** | yscv 1.87× slower |
| QLinear export before NCHW DW | 1 | 85.81 ms | **29.38 ms** | explicit INT8 still too slow |
| QLinear export after NCHW DW + QDQ boundary fuse + i8 edge storage + VNNI RHS prepack + direct i8 QuantizeLinear packed-store | 1 | **61.35 ms** | 29.38 ms | 29% yscv win vs prior |
| QLinear export before NCHW DW | 6 | 78.74 ms | **10.31 ms** | explicit INT8 still too slow |
| QLinear export after NCHW DW + QDQ boundary fuse + i8 edge storage + VNNI RHS prepack + direct i8 QuantizeLinear packed-store | 6 | **51.95 ms** | 10.31 ms | 34% yscv win vs prior |
| QLinear export + fused PW->DW quant-domain chain (`NodeAction::QuantizedPwDw`, 3x200 iter median, 31 chains/inference) | 1 | **46.63 ms** | 29.38 ms | 24% yscv win vs prior QLinear baseline |
| QLinear export + fused PW->DW quant-domain chain (`NodeAction::QuantizedPwDw`, 3x200 iter median, 31 chains/inference) | 6 | **29.87 ms** | 10.31 ms | 43% yscv win vs prior QLinear baseline |
| QLinear export + fused PW->DW + DW->PW quant-domain chains (`NodeAction::QuantizedPwDw` + `NodeAction::QuantizedDwPw`, 3x200 iter median, 36 chains/inference) | 1 | **45.95 ms** | 29.38 ms | adds the inverted-bottleneck closing pair; total 25% win vs original QLinear |
| QLinear export + fused PW->DW + DW->PW quant-domain chains (`NodeAction::QuantizedPwDw` + `NodeAction::QuantizedDwPw`, 3x200 iter median, 36 chains/inference) | 6 | **26.01 ms** | 10.31 ms | 50% win vs original QLinear baseline at 6T |
| QLinear export + both fused chains + SIMD requant epilogue (`int8_requant::requant_i32_row_to_i8_dispatch` AVX-512BW / AVX2 / NEON, 3x200 iter median, 36 chains/inference) | 1 | **34.23 ms** | 29.38 ms | replaces scalar f32 round/clamp inner loop with 16-lane (AVX-512BW) / 8-lane (AVX2) / 4-lane (NEON) SIMD; 60% total win vs original QLinear at 1T, within **1.17× of ORT** |
| QLinear export + both fused chains + SIMD requant epilogue (`int8_requant::requant_i32_row_to_i8_dispatch` AVX-512BW / AVX2 / NEON, 3x200 iter median, 36 chains/inference) | 6 | **22.37 ms** | 10.31 ms | 71% total win vs original QLinear at 6T |

The QLinear numbers are not the shipping quantized target yet: they are
the explicit standard-ONNX QLinear graph. The default yscv speed path is
still QDQ-fast while QLinear is being used as an interoperability and
kernel bring-up target. Accuracy gates must use representative paired
template/search crops; synthetic random calibration is smoke-only.
The workspace-native `bench_tracker` harness reports quant-runtime
counters; current QDQ-fast runs show `qlinear_conv_fast=0`, while QLinear
runs execute 88 `QLinearConv` fast paths and 49 quant-domain QDQ
boundaries per inference. QLinearConv / QuantizeLinear outputs now live in
an internal i8 side-table until a true fp32 consumer requests materialization;
the tracker QLinear path reports 176 i8 stores and 0 quant materializations
per inference. VNNI-friendly pointwise/MatMul RHS tensors are now packed once
at model load into the 4x16 AVX-512 VNNI layout, avoiding per-inference RHS
packing on explicit QLinearConv pointwise layers. Entry `QuantizeLinear` nodes
that feed QLinear activations now quantize directly into i8 side-table storage
through scalar/AVX2/AVX-512F/NEON dispatch, with packed AVX2/AVX-512 i8 stores, instead of scalar iterator
collection. `bench_tracker` also reports
static QLinear Conv-chain candidates; the current tracker QLinear export has 51
`PW->DW`, `DW->PW`, or residual-style chain candidates. The first production
INT8 chain actions (`NodeAction::QuantizedPwDw` + `NodeAction::QuantizedDwPw`)
now execute 36 fused chains per inference on the desktop tracker
(26 PW->DW opening pairs + 10 DW->PW closing pairs;
`quant_chain_executed=36`, `quant_chain_fallback=15` — the remaining 15
candidates are residual `Conv-Add-Q` patterns out of scope for this arc).
Each fused dispatch removes two `qlinear_conv_fast` hits and one QDQ
boundary on the per-op path.

## Hot kernel paths

- `FusedPwDw` (pointwise expand -> depthwise 3x3):
  - runner entry: `crates/yscv-onnx/src/runner/conv.rs` (`exec_fused_pw_dw`)
  - kernel: `crates/yscv-kernels/src/ops/fused_pw_dw_3x3.rs`
  - default: streaming path ON (kill-switch only).
- `FusedDwPw` (depthwise 3x3 -> pointwise 1x1):
  - runner entry: `crates/yscv-onnx/src/runner/conv.rs` (`exec_fused_dw_pw`)
  - kernel: `crates/yscv-kernels/src/ops/conv.rs` (`fused_dw_pw_nhwc_streaming`)
  - default: streaming ON for supported shapes, padded streaming OFF by default.
- pointwise/MatMul heavy ops:
  - `crates/yscv-kernels/src/ops/matmul.rs`
  - blocked GEMM + row GEMM dispatch, BLAS routing where enabled.
- explicit quantized depthwise:
  - runner entry: `crates/yscv-onnx/src/runner/conv.rs` (`exec_qlinear_conv`)
  - kernel: `crates/yscv-kernels/src/ops/int8_depthwise.rs`
  - default: symmetric `QLinearConv` depthwise 3x3/5x5 stride-1 and measured-win
    stride-2 tracker weights are packed once at load and executed through
    AVX-512/AVX2/NEON/scalar dispatch. Large NCHW stride-2 tracker
    depthwise layers (`ih > 64`) use a direct NCHW activations +
    KHWC-packed-weights scalar path to avoid a losing NCHW→NHWC layout
    materialisation.
- explicit quantized pointwise/MatMul:
  - runner entry: `crates/yscv-onnx/src/runner/conv.rs` (`exec_qlinear_conv`)
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
  - runner entry: `crates/yscv-onnx/src/runner/conv.rs`
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
    to the per-op QLinearConv requant in `runner/conv.rs`. The
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

Important: the tracker's fused PW->DW path on aarch64 (`fused_pw_dw_3x3.rs`)
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
