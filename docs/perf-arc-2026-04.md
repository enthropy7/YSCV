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

| metric | Pre-S.3 (2026-04-20, start) | Post-R9 (2026-04-20) | Post-R10 (2026-04-25, current) |
|---|---:|---:|---:|
| 1T p50 | ~11,500 µs | 11,427 µs | 11,220 µs |
| 6T p50 | ~4,180 µs | 3,665 µs | **3,170 µs** |
| 6T gap vs ORT | 2.38× | 2.10× | **1.82×** |
| cumulative win @ 6T (default-ON) | 0 | ~−953 µs | **~−1,450 µs** |
| output bitwise | ref | ref | ref (1-ULP FP drift) |

Hardware: AMD Ryzen 5 7500F (Zen 4, 6C/12T), NixOS, no-BLAS default build.

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
| R10 | `microkernel_4x8_dispatch` residual-tile correctness fix | correctness + latent perf | x86 ASM/SIMD 4×8 variants silently dropped `residual_tile` (Conv+Add shapes with 1×NR=8 tail landed in `microkernel_4x8_avx_fma` without residual → wrong output). Gate the fast paths on `residual_tile.is_none() \|\| !is_last_k` and fall through to `microkernel_4x8_scalar` — output goes 84.78 → 49.092 on the tracker |
| Q1  | INT8 fused PW->DW quant-domain chain (`NodeAction::QuantizedPwDw`, `int8_fused_pw_dw_3x3` kernel) | tracker QLinear 1T 61.35 → 46.63 ms (**−14.7 ms / −24%**), 6T 51.95 → 29.87 ms (**−22.1 ms / −43%**) | first INT8 chain action: streams NHWC i8 PW outputs through a per-worker `kh`-row ring buffer, requantises on the fly, writes NCHW i8 directly. Fires on 31 of 51 chain candidates per inference on the desktop tracker (`quant_chain_executed=31`, `quant_chain_fallback=20`); replaces 62 per-op `qlinear_conv_fast` hits and 31 QDQ boundaries. Bitwise-identical to `YSCV_QUANT_INT8_FAST=0` (unfused) path. Multi-arch: AVX-512-VNNI / AVX-VNNI / NEON SDOT / scalar PW dot + AVX-512BW / AVX2 / NEON widen-mul DW row reducers. |
| Q2  | INT8 fused DW->PW quant-domain chain (`NodeAction::QuantizedDwPw`, `int8_fused_dw_pw_3x3` kernel) | tracker QLinear 1T 46.63 → 45.95 ms (**−0.7 ms**), 6T 29.87 → 26.01 ms (**−3.9 ms / −13%**) | mirror of `Q1` for the closing pair of the inverted bottleneck. One worker per output-row chunk owns DW i32 / DW i8 / PW i32 row scratch; DW reads NHWC input directly (no ring buffer needed), requantises, and the PW reduction consumes the i8 immediately. Fires on the remaining 10 DW->PW chains per inference (`quant_chain_executed=36`, `quant_chain_fallback=15` — the 15 remainder are `Conv-Add-Q` patterns out of scope for this arc). Force-prepacks closing-pair PW weights even when `c_out` isn't a multiple of 16 (e.g. `bbox_pred` `c_out=4`, `cls_pred` `c_out=1`) so the kernel can run end-to-end without per-iteration RHS packing. |
| Q3  | SIMD requant epilogue for both INT8 fused chains (`int8_requant::requant_i32_row_to_i8_dispatch`) | tracker QLinear 1T 45.95 → 34.23 ms (**−11.7 ms / −25%**), 6T 26.01 → 22.37 ms (**−3.6 ms / −14%**) | replaces the per-pixel scalar `(a as f32) * composite + y_zp` `.round().clamp(-128,127) as i8` loop in `int8_fused_pw_dw_3x3` and `int8_fused_dw_pw_3x3` with a runtime-dispatched AVX-512BW 16-lane / AVX2+SSE4.1 8-lane / NEON 4-lane epilogue. Hot path on the desktop tracker because the requant fires twice per fused output row (PW + DW for `Q1`, DW + PW for `Q2`) — ~1.2 M elements per big chain at 1T. Bitwise-identical to the scalar reference, gated by 11 unit tests including a 144-pixel × 16-channel realistic stress test that originally caught a sub-ULP precision bug at `v = 0.49999997` (`+ 0.5` rounds up to `1.0` due to round-to-nearest-even on the half-tie boundary; fixed by biasing with `0.5 − ULP/2 = f32::from_bits(0x3EFFFFFF)` instead of `0.5`). |

Cumulative default-ON: **~−1,450 µs @ 6T p50** on fp32, plus the
explicit-INT8 wins (`Q1` **−22.1 ms @ 6T p50** + `Q2` **−3.9 ms @ 6T
p50** + `Q3` **−3.6 ms @ 6T p50** on the QLinear export, and **−15.4 ms
@ 1T p50** for the chain `Q1 → Q3`).
Bitwise-identical outputs (or 1-ULP FP-ordering drift, documented per
step) on fp32 paths; bitwise-identical i8 on the INT8 chain.

### R10 context

The bug was latent from `be34c23` ("opti") onward — that commit pushed
`residual_tile` through every scalar / NEON tail kernel but missed the
three x86 SIMD variants (`microkernel_4x8_avx_fma`, `microkernel_4x8_avx`,
`microkernel_4x8_sse`) and the `.S` fast path (`sgemm_4x8_set/acc`).
`microkernel_4x8_dispatch` had `#[cfg(not(target_arch = "aarch64"))] let _ = residual_tile;`
at the top, so LLVM happily discarded the residual pointer on x86.

All four tail paths only fire when `jr + 2*NR > nc` in `gebp_kernel_raw` —
i.e. the innermost loop has exactly one NR=8 panel left. Unit tests
never exercised this with `residual=Some`, so the regression walked
into the squashed commit unnoticed. Found via `private/onnx-fps` A/B:
output `882` max was 84.7835 on non-BLAS vs 49.0916 on ORT (and 49.0922
on BLAS, because the BLAS path goes through `blas_sgemm +
apply_epilogue_fallback` — completely bypasses `gebp_kernel_raw`, so
the drop never happened there). Bisect pinned it to `be34c23`, surgical
fix in `microkernel_4x8_dispatch` without reverting any of the
subsequent optimisations.

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
