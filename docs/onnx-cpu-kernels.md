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
