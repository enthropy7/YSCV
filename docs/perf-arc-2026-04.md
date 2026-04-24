# April 2026 perf arc — Siamese tracker on Zen 4

Note: this document is a Zen 4 arc snapshot. Latest public Orange Pi Zero 3
reruns (2026-04-21) are now tracked in `docs/performance-benchmarks.md` and
show yscv ahead of ORT CPU on this model at 1..4 threads.

Nineteen sessions of fp32 CPU optimization work on a Siamese tracker model,
targeting ONNX Runtime 1.24.4 as the reference. The goal throughout was
clear: no int8 / no quantization until yscv beats ORT on fp32 at full
thread count.

This document summarizes the arc — what landed, what failed, what's left.

## Baseline and endpoint

| metric | Pre-S.3 (2026-04-20, start) | Post-R9 (2026-04-20, end) |
|---|---:|---:|
| 1T p50 | ~11,500 µs | 11,427 µs |
| 6T p50 | ~4,180 µs | **3,665 µs** |
| 6T gap vs ORT | 2.38× | **2.10×** |
| cumulative win @ 6T (default-ON) | 0 | **~−953 µs** |

Hardware: AMD Ryzen 5 7500F (Zen 4, 6C/12T), NixOS, OpenBLAS.

---

## Landings (default-ON, shipped)

| step | change | 6T p50 delta | notes |
|---|---|---:|---|
| S.3 | AVX 8×8 NCHW↔NHWC block transpose | −99 µs | first real tracker win of the arc after 5 failed kernel attempts |
| S.4 | `fused_row_epilogue_dispatch` for row-GEMM (residual + bias + activation in one SIMD pass) | noise | shipped as cleanup |
| A2  | `depthwise_conv2d_nhwc_row_avx512` (ZMM 128/64/32/16 + YMM/scalar tail) | −117 µs | DW was 26.7% of cycles, no AVX-512 DW variant existed |
| A3  | First-layer AVX-512 flipped default-ON | −64 µs | was opt-in based on stale Zen-3 FMA latency claim |
| R1  | KHWC-weight fast-path fix for `Conv+Add` fusion | −91 µs | bug: `is_pointwise` read OIHW indices on KHWC-permuted weights; all 24 residual Conv_Add ops missed fusion for the entire arc |
| R3  | Slot-ID cache for `TensorEnv` dispatch | ~0 µs | cleanup; HashMap cost was already below noise at 6T |
| R4  | First-layer 3×3 row-level parallelism | −255 µs | four SIMD variants were all entirely single-threaded; `par_chunks_mut` when `out_h >= 32` |
| R5  | `conv_compute_nhwc` pure-compute split from env-binding wrapper | neutral | enabler for true-fused kernels |
| R6  | `FusedPwDw` plan variant (PW expand → DW 3×3) | neutral | enabler for R7 |
| R7  | Streaming FusedPwDw AVX-512 register-blocked + tiled | −146 µs | first real graph fusion win — ZMM accumulators held across entire inner loop, zero DRAM traffic for ~6 MB PW intermediate |
| R8  | First-layer tile-8 AVX-512 variant | neutral (wall) | per-op 212 → 178 µs but too small to surface at 6T wall |
| R9  | `FusedTransposeMatMul` mirroring ORT's `MatmulTransposeFusion` | −291 µs | graph-level fusion; one Transpose feeding N MatMuls requires `cleanup_transpose=true` on exactly one consumer |

Cumulative default-ON: **−963 µs @ 6T p50**, bitwise-identical outputs
across every step (or 1-ULP-close, documented per step).

---

## Opt-in landings (shipped default OFF, gated behind env)

| step | env gate | reason |
|---|---|---|
| A1.1 / A1.2 | `YSCV_AVX512_SGEMM=1` | strict `m%12==0, n%32==0` — narrow tracker coverage |
| S.1  | `YSCV_POINTWISE_LOW_K=1` | microbench win 1.84-2.33× but per-call pack_b ate the win |
| S.1′ | `YSCV_LOW_K_TILE=1` | proper matmul integration; 1T p50 +212 µs regression |
| R7 (initial AVX2-only) | `YSCV_FUSED_PW_DW_STREAM=1` before the AVX-512 variant | superseded by R7 Part 2 default-ON |

Kept as opt-in because they pass correctness but either regress or don't
help tracker. Not dead code — shape gates may shift on a different model.

---

## Dead-ends (reverted or abandoned)

| step | idea | why it failed |
|---|---|---|
| MR=6 | pure-intrinsics 6×16 GEMM | 1.7-2.3× slower than MR=4 inline-asm; LLVM couldn't replicate double-buffered B loads |
| E1   | native NCHWc pointwise Conv kernel | v4 Goto KC-tiling 30-50× slower than wrapper; single-thread broadcast-limited vs MT blocked GEMM |
| E2   | native NCHWc DW 3×3 s=1 SAME-pad | 1.1-5.9× slower than mature NHWC im2col+SIMD+MT on all 6 tracker shapes |
| R2   | 2D rayon tile batching and `use_blocked` threshold relaxation | variant 1: 0 µs (rayon already handles fine). Variant 2: 4-ULP drift for ~0 p50 win |
| R5 v1-v4 | `FusedDwPw` detection before true-fused kernel existed | +35-56 µs regression; fused action was just two env-thru calls |
| A1.2 tail | split-recurse AVX-512 main + generic tail | +1248 µs regression from duplicate pack-B |
| R8   | first-layer tile-8 at wall-clock | per-op −16% but wall-neutral (~34 µs sequential = ~6 µs wall, below noise) |

**Common theme:** microbench wins don't transfer to system-level wins
when the existing hot path is already mature. Five separate kernel
attempts failed on `row_gemm` / `blocked_gemm` before S.3 broke through
on the **under-tuned** layout-conversion path.

---

## Lessons

1. **Attack under-tuned paths before mature kernels.** Blocked GEMM and
   row-GEMM on Zen 4 are near peak; attacking them directly (MR=6, S.1,
   S.1′) yielded nothing. Attacking the scalar NCHW↔NHWC permute (S.3)
   and the single-threaded first-layer (R4) yielded the biggest wins.

2. **Verify fast-paths actually fire via profile labels.** R1 uncovered
   a silent bug where all 24 Conv+Add residuals bypassed the fusion for
   the entire arc because of a weight-shape index mismatch. One
   `YSCV_RUNNER_PROFILE` run catches this in seconds.

3. **Cleanup refcounts must stay centralized when fusing N:M patterns.**
   R9's critical bug: one Transpose feeding N MatMuls needs cleanup on
   exactly one fused variant. Decrementing N times evicts the pre-transpose
   tensor mid-read.

4. **Rayon > custom threadpool for this workload.** Proven three times
   (Step 3 C, yscv-threadpool, A″ series): work-stealing + tower-parallel
   fork-join + cold-idle backoff is already close to ideal. Custom pools
   hit either deadlocks (session spin-lock), over-serialization, or
   cold-start latency. Rayon stays the default.

5. **Graph-level fusion wins are easier than kernel-level wins when ORT
   has a matching contrib op.** R7 (streaming PW+DW, mirrors ORT's fused
   `FusedConv` kMSDomain) and R9 (`FusedTransposeMatMul`, mirrors
   `MatmulTransposeFusion`) — both delivered real wins because the
   pattern was pre-validated by ORT.

---

## Multi-architecture status

All hot-path kernels ship AVX2 + AVX-512 + NEON + scalar with runtime
dispatch. Known coverage gaps:

| area | status |
|---|---|
| NCHW↔NHWC block transpose (S.3) | **x86 AVX 8×8 only**, no NEON `vtrn`; scalar LLVM-autovec on aarch64 |
| AVX-512 DW row kernel (A2) | **AVX-512 only**; no AVX2 equivalent for non-Zen4 x86 |
| Fused PW+DW streaming NEON (R7) | correct, not perf-tuned |
| NCHWc DW 3×3 NEON (E2 opt-in) | correct, not perf-tuned |
| First-layer AVX-512 MR=16 | no NEON MR=8/16 equivalent |
| `matmul_2d_slices_trans_a` (R9) | BLAS-first; non-BLAS fallback does scratch-transpose + blocked GEMM |
| aarch64 hardware validation | cross-compile + unit tests only; no real ARM hw run |

These are follow-ups (MA-1 through MA-5 in the gap report), not blockers
for the tracker arc. They matter when ARM becomes a primary deployment
target.

---

## What's next (ranked by ROI)

From [gap-report-2026-04-20.md](gap-report-2026-04-20.md):

| # | lever | expected @ 6T | P(win) | effort |
|---|---|---:|---:|---:|
| T1 | First-layer Winograd F(2,3) or deeper register tile | 80-150 µs | 60% | 2-3 sessions |
| T2 | NCHWc layout transformer + native NCHWc Conv | 200-500 µs | 50% | 4-6 sessions |
| T3 | 2-branch batch kernel for first-layer | 60-120 µs | 70% | 2 sessions |
| T4 | TensorEnv arena allocator | 40-80 µs | 75% | 2 sessions |
| T5 | Improved rayon bridge / wait-on-epoch | 60-100 µs | 40% | 3-4 sessions |
| T6 | Non-BLAS specialized transA pack-A | 20-40 µs | 80% | 1 session |

**T1** is highest-confidence. **T2** is highest-potential but repeatedly
failed in different forms (E1, E2, A.1) — would need GotoBLAS-style KC
tiling + pre-packed B.

---

## Status

- Arc in **checkpoint state**, not finished
- 610+ tests green, aarch64 cross-compile clean, no-default-features OK
- All landings bitwise-identical (or 1-ULP-close documented per step)
- ORT parity on fp32 CPU is still the stated goal; int8 explicitly
  off-scope until fp32 gap closes

For the full per-op breakdown and thread sweep, see
[gap-report-2026-04-20.md](gap-report-2026-04-20.md).
