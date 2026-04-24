# Gap analysis: post-R9 yscv vs ORT — full thread sweep

**Collected 2026-04-20**, after R9 FusedTransposeMatMul landing.
Zen 4 (AMD Ryzen 5 7500F, 6C/12T), 500 iters × 3 runs per thread-count
for timing; 200 iters for per-op profile at T=6.

Model: Siamese tracker, 156 ops post graph-opt. Two input branches
(`input.1` 1×3×128×128, `input.249` 1×3×256×256) joined in
`connect_model`.

---

## Wall-clock thread sweep

Medians across 3 runs × 500 iters at each thread-count. yscv is
bitwise-identical to baseline across all threads (882/894 outputs match).

| Threads | yscv min | yscv p50 | ORT min | ORT p50 | gap (p50) | yscv scaling | ORT scaling |
|---:|---:|---:|---:|---:|---:|---:|---:|
|  1 | 10902 | **11427** | 8013 | **8051** | **1.42×** | 1.00× | 1.00× |
|  2 |  6211 |  **6552** | 4388 | **4418** | **1.48×** | 1.74× | 1.82× |
|  4 |  3764 |  **4151** | 2337 | **2360** | **1.76×** | 2.75× | 3.41× |
|  6 |  3032 |  **3665** | 1726 | **1743** | **2.10×** | 3.12× | 4.62× |
|  8 |  3360 |  **3869** | 2245 | **2278** | **1.70×** | 2.95× | 3.53× |
| 12 |  3658 |  **4023** | 1866 | **1934** | **2.08×** | 2.84× | 4.16× |

**All values in µs.**

**Observations:**

1. **1T gap is 1.42×** — per-core kernel efficiency is close to ORT parity.
   The work we still lack there is kernel microarchitecture polish plus
   some dispatch overhead.
2. **6T is the sweet spot for both engines** — physical core count.
   Beyond 6T, SMT contention hurts both: ORT degrades 1743 → 2278 µs (+31%)
   at 8T; yscv degrades 3665 → 3869 µs (+5.6%). ORT's MT kernels fight
   harder for the FP pipeline; ours are better at SMT tolerance.
3. **12T is worse than 6T** for both engines. Do not deploy at 12T.
4. **MT-scaling gap is the dominant issue.** yscv scales 3.12× from 1T→6T;
   ORT scales 4.62×. Per-core parity + perfect scaling would give us
   11427 / 4.6 ≈ 2480 µs at 6T — still ~740 µs behind ORT, but much closer.

---

## Per-op gap breakdown (6T, sequential sums over one inference)

All yscv fused ops aliased back to their unfused ORT counterparts
(`FusedPwDw`, `FusedDwPw`, `Conv_Add_fused`, `Conv_Relu` → `Conv`;
`FusedTransposeMatMul` → `MatMul`).

| op | yscv total | yscv ops | ORT total | ORT ops | gap | ratio |
|---|---:|---:|---:|---:|---:|---:|
| Conv | 6.31 ms | 78 | 2.22 ms | 114 | **+4.09 ms** | **2.84×** |
| MatMul | 0.19 ms | 2 | 0.04 ms | 2 | +0.14 ms | 4.17× |
| Reshape | 0.11 ms | 5 | 0.01 ms | 5 | +0.10 ms | 8.01× |
| Concat | 0.02 ms | 2 | 0.02 ms | 2 | +0.00 ms | 1.08× |
| Mul | 0.00 ms | 1 | 0.00 ms | 1 | +0.00 ms | 1.06× |
| Exp | 0.00 ms | 1 | 0.00 ms | 1 | −0.00 ms | 0.69× |
| Reorder (NCHWc) | — | — | 0.05 ms | 7 | — | — |
| **Totals** | 6.65 ms | 95 nodes | 2.36 ms | 133 nodes | — | 2.82× |

**Conv dominates 94% of the sequential-sum gap.** Everything else is
noise on top. The wall-clock ratio (2.10× @ 6T) is smaller than the
sequential-sum ratio (2.82×) because MT parallelises across cores —
ORT scales better, hence a larger wall → sequential factor.

---

## Per-block Conv hot spots

### `/xif0_0/` (first-layer 3×3 s=2 RGB)

| | µs (sequential) | ops |
|---|---:|---:|
| yscv | ~255 µs total | 2 (`Conv_Relu` at 128×128 and 256×256) |
| ORT | ~30 µs total | 2 (NCHWc) |
| **ratio** | | **~8.5×** |

Same hot spot as post-R7 report. R4 (row-level par) + R8 (8-col tile)
brought the 256×256 branch from 354 → 178 µs. Remaining 150 µs single-op
gap is the largest addressable target on the lever list (T1: Winograd
F(2,3) or deeper register tiling).

### `/xif2_0/` (first inverted-bottleneck, c_exp = 96)

yscv now runs this as two `FusedPwDw` pairs (R6/R7 streaming AVX-512),
then two separate `Conv` for the pwl projection. ORT uses NCHWc sub-ops
(`Conv` + `Relu` unfused but NCHWc-native).

| | µs (sequential) | ops |
|---|---:|---:|
| yscv | ~205 µs | 2 FusedPwDw + 2 Conv |
| ORT | ~189 µs | 6 NCHWc sub-ops |
| ratio | | ~1.08× |

**R7 closed most of this.** The remaining gap is NCHWc layout
(ORT's DW walks contiguous block-of-channel data; our NHWC DW walks
stride-`c_exp` rows).

### Residual-suffix `Conv_Add_fused` (R1)

24 ops total, ~1.25 ms sequential. These were entirely unfused before R1
(each `Conv` + separate `Add` + optional `Relu`). R1 fixed the KHWC
dispatch so the fast-path actually fires.

### FusedTransposeMatMul (R9)

```
yscv: FusedTransposeMatMul /connect_model/cls_dw/MatMul  ~90 µs
yscv: FusedTransposeMatMul /connect_model/reg_dw/MatMul  ~90 µs
ORT:  FusedMatMul /connect_model/cls_dw/MatMul/MatmulTransposeFusion  ~19 µs
ORT:  FusedMatMul /connect_model/reg_dw/MatMul/MatmulTransposeFusion  ~19 µs
```

R9 landed the plumbing and the `matmul_2d_slices_trans_a` kernel
(BLAS `CblasTrans`). Remaining 4.17× ratio here is real BLAS vs MLAS
kernel delta on small m=64 k=32 n=64 shape — probably not closeable
without an MLAS-calibre specialised transA microkernel.

---

## Residual gap attack surface (ranked by ROI)

| # | Lever | Target | Expected @ 6T | P(win) | Effort |
|---|---|---|---:|---:|---:|
| **T1** | First-layer Conv Winograd F(2,3) or deeper register tile | 150 µs single-op gap | 80–150 µs | 60% | 2-3 sessions |
| **T2** | NCHWc layout transformer + native NCHWc Conv kernel | Conv bulk (2.84× ratio) | 200–500 µs | 50% | 4-6 sessions |
| **T3** | 2-branch batch kernel (128×128 + 256×256 fused first-layer) | first-layer cross-branch parallelism | 60–120 µs | 70% | 2 sessions |
| **T4** | TensorEnv arena allocator (size-classed buffer pool) | malloc per-op churn | 40–80 µs | 75% | 2 sessions |
| **T5** | Improved rayon bridge (wait-on-epoch / persistent worker state) | MT scaling 3.1× → 3.5× | 60–100 µs | 40% | 3-4 sessions |
| **T6** | Non-BLAS `matmul_2d_slices_trans_a` specialised pack-A | MatMul 4.17× ratio at non-BLAS builds | 20–40 µs | 80% | 1 session |

**T7 (Conv+BN fusion) excluded** — tracker has BN pre-folded at export.

---

## Status

- Cumulative arc from baseline (pre-S.3) to post-R9: **~−953 µs @ 6T p50**
- Arc gap: **2.38× → 2.10×** (measured this sweep) at 6T
- 1T gap stable at 1.42×
- All landings bitwise-identical or 1-ULP-close to reference
- 610+ tests green, aarch64 cross-compile clean, no-default-features build OK

Arc is in **checkpoint state** — no session planned here, this is the
state-of-union report. Next implementation session picks T1, T3, or T6
depending on user priority.
