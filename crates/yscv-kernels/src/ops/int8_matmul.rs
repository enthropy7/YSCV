//! INT8 × INT8 → INT32 GEMM.
//!
//! Pure dot-product accumulation: `out[i,j] = Σ_k a[i,k] * b[k,j]` with
//! `a`, `b` ∈ `[-128, 127]` and `out` ∈ `i32`. This is the fast path for
//! symmetric quantization (zero-point == 0 on both operands); asymmetric
//! callers must pre-subtract zero-points or use the f32 fallback.
//!
//! Variants (selected at runtime):
//! - **scalar** — i32 += i8 × i8, always available; reference for tests.
//! - **NEON SDOT** — `sdot` aarch64 instruction emitted via `core::arch::asm!`
//!   so we don't depend on the still-nightly `vdotq_s32` intrinsic.
//!   Requires `target_feature = "dotprod"`.
//! - **NEON SMMLA (i8mm)** — `smmla` 2×2 i32 += dot(2×8 i8, 8×2 i8) for
//!   ARMv8.6+ (Apple M1+, Cortex-X1+, Neoverse N2+), also via inline asm.
//! - **AVX-VNNI** — `_mm256_dpbusd_avx_epi32` with `a XOR 0x80`
//!   bias-shift so unsigned-signed VNNI gives the same result as
//!   signed-signed.
//! - **AVX-512-VNNI** — `_mm512_dpbusd_epi32` with the same bias-shift.
//! - **AVX2 widen-mul** — sign-extend i8→i16, `_mm256_madd_epi16`,
//!   accumulate to i32. Correct on every AVX2 CPU; matches scalar
//!   bitwise.
//!
//! Bitwise behaviour is identical across all variants — they differ only
//! in the order of int32 additions, which is associative for true integers
//! (no overflow guard is needed for k ≤ 2³¹/(127*127) ≈ 130k).

#![allow(unsafe_code, unsafe_op_in_unsafe_fn)]

/// Transpose `b` from row-major `[K × N]` into row-major `[N × K]` so
/// the inner kernel can issue a contiguous K-vector load per output
/// column instead of a strided per-element gather. The transpose itself
/// is O(K·N) and happens once per call — for typical M·K·N with M ≥ 8
/// the saved cache misses pay back the transpose by 1-2 orders of
/// magnitude.
fn transpose_b(b: &[i8], k: usize, n: usize) -> Vec<i8> {
    let mut bt = vec![0_i8; n * k];
    // 8×8 blocking improves throughput vs the naive scalar loop without
    // pulling in arch-specific intrinsics for the transpose itself.
    let bs = 8;
    let kb_full = (k / bs) * bs;
    let nb_full = (n / bs) * bs;
    for kk in (0..kb_full).step_by(bs) {
        for jj in (0..nb_full).step_by(bs) {
            for r in 0..bs {
                for c in 0..bs {
                    bt[(jj + c) * k + kk + r] = b[(kk + r) * n + jj + c];
                }
            }
        }
        for jj in nb_full..n {
            for r in 0..bs {
                bt[jj * k + kk + r] = b[(kk + r) * n + jj];
            }
        }
    }
    for kk in kb_full..k {
        for jj in 0..n {
            bt[jj * k + kk] = b[kk * n + jj];
        }
    }
    bt
}

/// Load-time packed INT8 GEMM RHS.
///
/// Public `int8_matmul_dispatch` accepts `b` as row-major `[K, N]` and
/// transposes it internally because every SIMD backend wants contiguous
/// K-lanes for each output column. That is correct for one-off calls but
/// wasteful for inference graphs where weights are constant. This type
/// stores that transposed `[N, K]` layout once at model load so hot-path
/// QLinearConv/QLinearMatMul can reuse it without heap work or weight
/// repacking per inference.
#[derive(Debug, Clone)]
pub struct PackedI8B {
    k: usize,
    n: usize,
    bt: Vec<i8>,
    vnni_4x16: Option<PackedI8BVnni4x16>,
}

#[derive(Debug, Clone)]
struct PackedI8BVnni4x16 {
    k4_full: usize,
    bp: Vec<i8>,
    col_sum_b_vnni: Vec<i32>,
}

impl PackedI8B {
    #[inline]
    pub fn k(&self) -> usize {
        self.k
    }

    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    #[inline]
    pub fn transposed(&self) -> &[i8] {
        &self.bt
    }
}

/// Pack row-major `b` (`[K, N]`) for repeated INT8 GEMM calls.
pub fn pack_i8_b_for_matmul(b: &[i8], k: usize, n: usize) -> PackedI8B {
    debug_assert_eq!(b.len(), k * n);
    let vnni_4x16 = if n.is_multiple_of(16) && k >= 4 {
        let (bp, k4_full) = pack_b_vnni_4x16(b, k, n);
        let mut col_sum_b_vnni = vec![0_i32; n];
        for j in 0..n {
            let mut s: i32 = 0;
            for kk in 0..k4_full {
                s += b[kk * n + j] as i32;
            }
            col_sum_b_vnni[j] = s;
        }
        Some(PackedI8BVnni4x16 {
            k4_full,
            bp,
            col_sum_b_vnni,
        })
    } else {
        None
    };
    PackedI8B {
        k,
        n,
        bt: transpose_b(b, k, n),
        vnni_4x16,
    }
}

/// Scalar reference. Always correct; use for tests and as fallback when
/// no SIMD path is detected at runtime.
pub fn int8_matmul_scalar(a: &[i8], b: &[i8], m: usize, k: usize, n: usize, out: &mut [i32]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(out.len(), m * n);
    for i in 0..m {
        for j in 0..n {
            let mut acc: i32 = 0;
            for kk in 0..k {
                acc += (a[i * k + kk] as i32) * (b[kk * n + j] as i32);
            }
            out[i * n + j] = acc;
        }
    }
}

/// Scalar reference for prepacked RHS. Always correct; SIMD variants
/// below must match this bit-for-bit.
pub fn int8_matmul_prepacked_scalar(a: &[i8], b: &PackedI8B, m: usize, out: &mut [i32]) {
    let k = b.k;
    let n = b.n;
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(out.len(), m * n);
    let bt = b.transposed();
    for i in 0..m {
        for j in 0..n {
            let mut acc: i32 = 0;
            for kk in 0..k {
                acc += (a[i * k + kk] as i32) * (bt[j * k + kk] as i32);
            }
            out[i * n + j] = acc;
        }
    }
}

/// Issue the aarch64 `sdot Vd.4S, Vn.16B, Vm.16B` instruction. Each
/// i32 lane in `acc` accumulates the dot product of the corresponding
/// 4-byte slice of `a` with the matching slice of `b`. Inline asm
/// because the matching `vdotq_s32` intrinsic is still nightly-only.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn sdot_inline(
    acc: std::arch::aarch64::int32x4_t,
    a: std::arch::aarch64::int8x16_t,
    b: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    use std::arch::aarch64::int32x4_t;
    let mut out: int32x4_t = acc;
    std::arch::asm!(
        "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
        acc = inout(vreg) out,
        a = in(vreg) a,
        b = in(vreg) b,
        options(pure, nomem, nostack)
    );
    out
}

/// Issue the aarch64 `smmla Vd.4S, Vn.16B, Vm.16B` (i8mm) instruction.
/// Computes `acc += [A0·B0, A0·B1, A1·B0, A1·B1]` where Ai/Bj are the
/// two 8-byte halves of `a` and `b`.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon,i8mm")]
unsafe fn smmla_inline(
    acc: std::arch::aarch64::int32x4_t,
    a: std::arch::aarch64::int8x16_t,
    b: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    use std::arch::aarch64::int32x4_t;
    let mut out: int32x4_t = acc;
    std::arch::asm!(
        "smmla {acc:v}.4s, {a:v}.16b, {b:v}.16b",
        acc = inout(vreg) out,
        a = in(vreg) a,
        b = in(vreg) b,
        options(pure, nomem, nostack)
    );
    out
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,i8mm")]
unsafe fn int8_matmul_neon_i8mm(a: &[i8], b: &[i8], m: usize, k: usize, n: usize, out: &mut [i32]) {
    use std::arch::aarch64::*;
    // SMMLA: acc[0..4] += [A0·B0, A0·B1, A1·B0, A1·B1] where Ai is an
    // 8-byte row slice and Bj is an 8-byte column slice. Tile 2×2 in
    // (M, N); fall back to sdot for odd-edge rows/cols.
    let m2 = m & !1;
    let n2 = n & !1;
    let k8 = k & !7;
    for i in (0..m2).step_by(2) {
        for j in (0..n2).step_by(2) {
            let mut acc = vdupq_n_s32(0);
            let mut kk = 0;
            while kk + 8 <= k8 {
                let mut abuf = [0_i8; 16];
                for r in 0..8 {
                    abuf[r] = a[i * k + kk + r];
                    abuf[8 + r] = a[(i + 1) * k + kk + r];
                }
                let av = vld1q_s8(abuf.as_ptr());
                let mut bbuf = [0_i8; 16];
                for r in 0..8 {
                    bbuf[r] = b[(kk + r) * n + j];
                    bbuf[8 + r] = b[(kk + r) * n + j + 1];
                }
                let bv = vld1q_s8(bbuf.as_ptr());
                acc = smmla_inline(acc, av, bv);
                kk += 8;
            }
            let mut buf = [0_i32; 4];
            vst1q_s32(buf.as_mut_ptr(), acc);
            for di in 0..2 {
                for dj in 0..2 {
                    let mut tail = buf[di * 2 + dj];
                    let mut kt = kk;
                    while kt < k {
                        tail += (a[(i + di) * k + kt] as i32) * (b[kt * n + j + dj] as i32);
                        kt += 1;
                    }
                    out[(i + di) * n + j + dj] = tail;
                }
            }
        }
    }
    if m2 < m || n2 < n {
        for i in 0..m {
            for j in 0..n {
                if i < m2 && j < n2 {
                    continue;
                }
                let mut acc = vdupq_n_s32(0);
                let mut kk = 0;
                while kk + 16 <= k {
                    let av = vld1q_s8(a.as_ptr().add(i * k + kk));
                    let mut bbuf = [0_i8; 16];
                    for r in 0..16 {
                        bbuf[r] = b[(kk + r) * n + j];
                    }
                    let bv = vld1q_s8(bbuf.as_ptr());
                    acc = sdot_inline(acc, av, bv);
                    kk += 16;
                }
                let mut tail = vaddvq_s32(acc);
                while kk < k {
                    tail += (a[i * k + kk] as i32) * (b[kk * n + j] as i32);
                    kk += 1;
                }
                out[i * n + j] = tail;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn int8_matmul_prepacked_neon_sdot(a: &[i8], b: &PackedI8B, m: usize, out: &mut [i32]) {
    use std::arch::aarch64::*;
    let k = b.k;
    let n = b.n;
    let bt = b.transposed();
    for i in 0..m {
        for j in 0..n {
            let mut acc = vdupq_n_s32(0);
            let mut kk = 0;
            while kk + 16 <= k {
                let av = vld1q_s8(a.as_ptr().add(i * k + kk));
                let bv = vld1q_s8(bt.as_ptr().add(j * k + kk));
                acc = sdot_inline(acc, av, bv);
                kk += 16;
            }
            let mut tail = vaddvq_s32(acc);
            while kk < k {
                tail += (a[i * k + kk] as i32) * (bt[j * k + kk] as i32);
                kk += 1;
            }
            out[i * n + j] = tail;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn int8_matmul_neon_sdot(a: &[i8], b: &[i8], m: usize, k: usize, n: usize, out: &mut [i32]) {
    use std::arch::aarch64::*;
    let k4 = k & !3;
    for i in 0..m {
        for j in 0..n {
            let mut acc = vdupq_n_s32(0);
            let mut kk = 0;
            while kk + 16 <= k4 {
                let av = vld1q_s8(a.as_ptr().add(i * k + kk));
                let mut bbuf = [0_i8; 16];
                for r in 0..16 {
                    bbuf[r] = b[(kk + r) * n + j];
                }
                let bv = vld1q_s8(bbuf.as_ptr());
                acc = sdot_inline(acc, av, bv);
                kk += 16;
            }
            let mut tail = vaddvq_s32(acc);
            while kk < k {
                tail += (a[i * k + kk] as i32) * (b[kk * n + j] as i32);
                kk += 1;
            }
            out[i * n + j] = tail;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn int8_matmul_avx2_widen(
    a: &[i8],
    b: &[i8],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [i32],
) {
    use std::arch::x86_64::*;
    let bt = transpose_b(b, k, n);
    for i in 0..m {
        for j in 0..n {
            let mut acc = _mm256_setzero_si256();
            let mut kk = 0;
            while kk + 16 <= k {
                let av = _mm_loadu_si128(a.as_ptr().add(i * k + kk) as *const __m128i);
                let bv = _mm_loadu_si128(bt.as_ptr().add(j * k + kk) as *const __m128i);
                let av16 = _mm256_cvtepi8_epi16(av);
                let bv16 = _mm256_cvtepi8_epi16(bv);
                let prod = _mm256_madd_epi16(av16, bv16);
                acc = _mm256_add_epi32(acc, prod);
                kk += 16;
            }
            let mut buf = [0_i32; 8];
            _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, acc);
            let mut tail = buf.iter().sum::<i32>();
            while kk < k {
                tail += (a[i * k + kk] as i32) * (bt[j * k + kk] as i32);
                kk += 1;
            }
            out[i * n + j] = tail;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn int8_matmul_prepacked_avx2_widen(a: &[i8], b: &PackedI8B, m: usize, out: &mut [i32]) {
    use std::arch::x86_64::*;
    let k = b.k;
    let n = b.n;
    let bt = b.transposed();
    for i in 0..m {
        for j in 0..n {
            let mut acc = _mm256_setzero_si256();
            let mut kk = 0;
            while kk + 16 <= k {
                let av = _mm_loadu_si128(a.as_ptr().add(i * k + kk) as *const __m128i);
                let bv = _mm_loadu_si128(bt.as_ptr().add(j * k + kk) as *const __m128i);
                let av16 = _mm256_cvtepi8_epi16(av);
                let bv16 = _mm256_cvtepi8_epi16(bv);
                let prod = _mm256_madd_epi16(av16, bv16);
                acc = _mm256_add_epi32(acc, prod);
                kk += 16;
            }
            let mut buf = [0_i32; 8];
            _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, acc);
            let mut tail = buf.iter().sum::<i32>();
            while kk < k {
                tail += (a[i * k + kk] as i32) * (bt[j * k + kk] as i32);
                kk += 1;
            }
            out[i * n + j] = tail;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avxvnni")]
unsafe fn int8_matmul_avx_vnni(a: &[i8], b: &[i8], m: usize, k: usize, n: usize, out: &mut [i32]) {
    use std::arch::x86_64::*;
    let bt = transpose_b(b, k, n);
    let bias: i32 = 128;
    let bias128 = _mm256_set1_epi8(-128_i8);
    // Per-column sum_b is shape-invariant; precompute once over bt.
    let mut col_sum_b = vec![0_i32; n];
    for j in 0..n {
        let mut s: i32 = 0;
        for kk in 0..k {
            s += bt[j * k + kk] as i32;
        }
        col_sum_b[j] = s;
    }
    let k_vnni = (k / 32) * 32;
    // Per-column sum over the VNNI-covered K prefix. col_sum_b - tail.
    let mut col_sum_b_vnni = vec![0_i32; n];
    for j in 0..n {
        let mut tail = 0_i32;
        for kk in k_vnni..k {
            tail += bt[j * k + kk] as i32;
        }
        col_sum_b_vnni[j] = col_sum_b[j] - tail;
    }
    for i in 0..m {
        for j in 0..n {
            let mut acc = _mm256_setzero_si256();
            let mut kk = 0;
            while kk + 32 <= k {
                let av = _mm256_loadu_si256(a.as_ptr().add(i * k + kk) as *const __m256i);
                let bv = _mm256_loadu_si256(bt.as_ptr().add(j * k + kk) as *const __m256i);
                // a → a + 128 (mod 256, reinterpret i8→u8) so unsigned-signed
                // VNNI gives `dot(a, b) + 128 * sum(b)`. Subtract `128 * sum_b`
                // at the end to get the signed-signed result.
                let av_u = _mm256_xor_si256(av, bias128);
                acc = _mm256_dpbusd_avx_epi32(acc, av_u, bv);
                kk += 32;
            }
            let mut buf = [0_i32; 8];
            _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, acc);
            let mut tail = buf.iter().sum::<i32>() - bias * col_sum_b_vnni[j];
            while kk < k {
                tail += (a[i * k + kk] as i32) * (bt[j * k + kk] as i32);
                kk += 1;
            }
            out[i * n + j] = tail;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avxvnni")]
unsafe fn int8_matmul_prepacked_avx_vnni(a: &[i8], b: &PackedI8B, m: usize, out: &mut [i32]) {
    use std::arch::x86_64::*;
    let k = b.k;
    let n = b.n;
    let bt = b.transposed();
    let bias: i32 = 128;
    let bias128 = _mm256_set1_epi8(-128_i8);
    let k_vnni = (k / 32) * 32;
    let mut col_sum_b_vnni = vec![0_i32; n];
    for j in 0..n {
        let mut s: i32 = 0;
        for kk in 0..k_vnni {
            s += bt[j * k + kk] as i32;
        }
        col_sum_b_vnni[j] = s;
    }
    for i in 0..m {
        for j in 0..n {
            let mut acc = _mm256_setzero_si256();
            let mut kk = 0;
            while kk + 32 <= k {
                let av = _mm256_loadu_si256(a.as_ptr().add(i * k + kk) as *const __m256i);
                let bv = _mm256_loadu_si256(bt.as_ptr().add(j * k + kk) as *const __m256i);
                let av_u = _mm256_xor_si256(av, bias128);
                acc = _mm256_dpbusd_avx_epi32(acc, av_u, bv);
                kk += 32;
            }
            let mut buf = [0_i32; 8];
            _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, acc);
            let mut tail = buf.iter().sum::<i32>() - bias * col_sum_b_vnni[j];
            while kk < k {
                tail += (a[i * k + kk] as i32) * (bt[j * k + kk] as i32);
                kk += 1;
            }
            out[i * n + j] = tail;
        }
    }
}

/// Pack `b` (shape `[K, N]` row-major) into the layout the
/// register-blocked VNNI kernel consumes:
///
/// ```text
/// bp[jp][g][c][r] = b[(g*4 + r) * n + jp*16 + c]
/// ```
///
/// i.e. for each N panel of 16 columns and each K group of 4 rows, the
/// 64 bytes are interleaved so a single ZMM load gives one VNNI input.
/// Caller must guarantee `n % 16 == 0`; K is rounded down to a multiple
/// of 4 (`k4_full`), the K tail handled scalar-side.
fn pack_b_vnni_4x16(b: &[i8], k: usize, n: usize) -> (Vec<i8>, usize) {
    debug_assert_eq!(n % 16, 0);
    let k4_full = (k / 4) * 4;
    let groups = k4_full / 4;
    let n_panels = n / 16;
    let mut bp = vec![0_i8; n_panels * groups * 16 * 4];
    for jp in 0..n_panels {
        for g in 0..groups {
            let dst_panel = jp * groups * 16 * 4;
            for c in 0..16 {
                let n_idx = jp * 16 + c;
                let dst_col = dst_panel + g * 16 * 4 + c * 4;
                for r in 0..4 {
                    let k_idx = g * 4 + r;
                    bp[dst_col + r] = b[k_idx * n + n_idx];
                }
            }
        }
    }
    (bp, k4_full)
}

/// Register-blocked AVX-512-VNNI kernel, MR=4 × NR=16.
///
/// Each MR×NR output tile keeps 4 ZMM accumulators alive across the
/// full K loop. Per K group of 4: one ZMM B load is reused across all
/// 4 A rows (broadcast-and-vpdpbusd), so we issue 4 vpdpbusd per
/// 64-byte B traffic. On Zen 4 vpdpbusd has 1/cycle throughput, B
/// load is L1-bound — kernel is FMA-issue-limited at ~95% of peak.
///
/// Strict shape gate: `m % 4 == 0`, `n % 16 == 0`, `k >= 4`. Caller
/// (`int8_matmul_avx512_vnni`) routes other shapes to the simple
/// transposed-B path.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn int8_matmul_avx512_vnni_blocked(
    a: &[i8],
    b: &[i8],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [i32],
) {
    use std::arch::x86_64::*;
    const MR: usize = 4;
    const NR: usize = 16;

    let (bp, k4_full) = pack_b_vnni_4x16(b, k, n);
    let groups = k4_full / 4;
    let n_panels = n / NR;

    // Per-column sum_b over the K range covered by the VNNI groups.
    // Bias-shift: dpbusd(a XOR 0x80, b) = dot(a, b) + 128 · sum(b),
    // so we subtract `128 · col_sum_b_vnni[j]` after the K loop.
    let mut col_sum_b_vnni = vec![0_i32; n];
    for j in 0..n {
        let mut s: i32 = 0;
        for kk in 0..k4_full {
            s += b[kk * n + j] as i32;
        }
        col_sum_b_vnni[j] = s;
    }
    let bias = 128_i32;
    let bias_xor = _mm512_set1_epi8(-128_i8);

    for ib in (0..m).step_by(MR) {
        for jp in 0..n_panels {
            let mut a0 = _mm512_setzero_si512();
            let mut a1 = _mm512_setzero_si512();
            let mut a2 = _mm512_setzero_si512();
            let mut a3 = _mm512_setzero_si512();
            let bp_panel = jp * groups * NR * 4;
            for g in 0..groups {
                let bv =
                    _mm512_loadu_si512(bp.as_ptr().add(bp_panel + g * NR * 4) as *const __m512i);
                let k_idx = g * 4;
                // Read 4 bytes of A row r at K position k_idx as a single
                // i32; broadcast across 16 lanes; XOR with 0x80808080 to
                // shift signed → unsigned for vpdpbusd.
                let a_p0 = (a.as_ptr().add(ib * k + k_idx) as *const i32).read_unaligned();
                let a_p1 = (a.as_ptr().add((ib + 1) * k + k_idx) as *const i32).read_unaligned();
                let a_p2 = (a.as_ptr().add((ib + 2) * k + k_idx) as *const i32).read_unaligned();
                let a_p3 = (a.as_ptr().add((ib + 3) * k + k_idx) as *const i32).read_unaligned();
                let av0 = _mm512_xor_si512(_mm512_set1_epi32(a_p0), bias_xor);
                let av1 = _mm512_xor_si512(_mm512_set1_epi32(a_p1), bias_xor);
                let av2 = _mm512_xor_si512(_mm512_set1_epi32(a_p2), bias_xor);
                let av3 = _mm512_xor_si512(_mm512_set1_epi32(a_p3), bias_xor);
                a0 = _mm512_dpbusd_epi32(a0, av0, bv);
                a1 = _mm512_dpbusd_epi32(a1, av1, bv);
                a2 = _mm512_dpbusd_epi32(a2, av2, bv);
                a3 = _mm512_dpbusd_epi32(a3, av3, bv);
            }
            let mut buf = [[0_i32; NR]; MR];
            _mm512_storeu_si512(buf[0].as_mut_ptr() as *mut __m512i, a0);
            _mm512_storeu_si512(buf[1].as_mut_ptr() as *mut __m512i, a1);
            _mm512_storeu_si512(buf[2].as_mut_ptr() as *mut __m512i, a2);
            _mm512_storeu_si512(buf[3].as_mut_ptr() as *mut __m512i, a3);
            for r_off in 0..MR {
                for c in 0..NR {
                    let n_idx = jp * NR + c;
                    let mut v = buf[r_off][c] - bias * col_sum_b_vnni[n_idx];
                    // K tail (≤ 3 elements).
                    for kk in k4_full..k {
                        v += (a[(ib + r_off) * k + kk] as i32) * (b[kk * n + n_idx] as i32);
                    }
                    out[(ib + r_off) * n + n_idx] = v;
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn int8_matmul_prepacked_avx512_vnni_blocked(
    a: &[i8],
    b: &PackedI8B,
    m: usize,
    out: &mut [i32],
) {
    use std::arch::x86_64::*;
    const MR: usize = 4;
    const NR: usize = 16;

    let k = b.k;
    let n = b.n;
    let packed = b.vnni_4x16.as_ref().expect("missing VNNI-packed RHS");
    let k4_full = packed.k4_full;
    let groups = k4_full / 4;
    let n_panels = n / NR;
    let bt = b.transposed();
    let bias = 128_i32;
    let bias_xor = _mm512_set1_epi8(-128_i8);

    for ib in (0..m).step_by(MR) {
        for jp in 0..n_panels {
            let mut a0 = _mm512_setzero_si512();
            let mut a1 = _mm512_setzero_si512();
            let mut a2 = _mm512_setzero_si512();
            let mut a3 = _mm512_setzero_si512();
            let bp_panel = jp * groups * NR * 4;
            for g in 0..groups {
                let bv = _mm512_loadu_si512(
                    packed.bp.as_ptr().add(bp_panel + g * NR * 4) as *const __m512i
                );
                let k_idx = g * 4;
                let a_p0 = (a.as_ptr().add(ib * k + k_idx) as *const i32).read_unaligned();
                let a_p1 = (a.as_ptr().add((ib + 1) * k + k_idx) as *const i32).read_unaligned();
                let a_p2 = (a.as_ptr().add((ib + 2) * k + k_idx) as *const i32).read_unaligned();
                let a_p3 = (a.as_ptr().add((ib + 3) * k + k_idx) as *const i32).read_unaligned();
                let av0 = _mm512_xor_si512(_mm512_set1_epi32(a_p0), bias_xor);
                let av1 = _mm512_xor_si512(_mm512_set1_epi32(a_p1), bias_xor);
                let av2 = _mm512_xor_si512(_mm512_set1_epi32(a_p2), bias_xor);
                let av3 = _mm512_xor_si512(_mm512_set1_epi32(a_p3), bias_xor);
                a0 = _mm512_dpbusd_epi32(a0, av0, bv);
                a1 = _mm512_dpbusd_epi32(a1, av1, bv);
                a2 = _mm512_dpbusd_epi32(a2, av2, bv);
                a3 = _mm512_dpbusd_epi32(a3, av3, bv);
            }
            let mut buf = [[0_i32; NR]; MR];
            _mm512_storeu_si512(buf[0].as_mut_ptr() as *mut __m512i, a0);
            _mm512_storeu_si512(buf[1].as_mut_ptr() as *mut __m512i, a1);
            _mm512_storeu_si512(buf[2].as_mut_ptr() as *mut __m512i, a2);
            _mm512_storeu_si512(buf[3].as_mut_ptr() as *mut __m512i, a3);
            for r_off in 0..MR {
                for c in 0..NR {
                    let n_idx = jp * NR + c;
                    let mut v = buf[r_off][c] - bias * packed.col_sum_b_vnni[n_idx];
                    for kk in k4_full..k {
                        v += (a[(ib + r_off) * k + kk] as i32) * (bt[n_idx * k + kk] as i32);
                    }
                    out[(ib + r_off) * n + n_idx] = v;
                }
            }
        }
    }
}

/// Register-blocked AVX-512-VNNI kernel, MR=8 × NR=16. 8 ZMM
/// accumulators + 1 ZMM B + 8 ZMM A broadcasts (held in 8 GP-sourced
/// `set1_epi32` results) fit comfortably in the 32-register file.
///
/// vs MR=4: B-vector reuse doubles (8 vpdpbusd per B load instead of 4),
/// halving the L1 traffic on B for fixed M. Wins on hidden×hidden LLM
/// linears where MR=4 lost to the simple kernel; loses on tiny-K shapes
/// where the extra register pressure exceeds the B-reuse benefit. The
/// dispatch gate picks between MR=8 and MR=4 per shape.
///
/// Strict gate: `m % 8 == 0 && n % 16 == 0 && k >= 4`. Caller routes
/// other shapes to MR=4 or simple paths.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn int8_matmul_avx512_vnni_blocked_mr8(
    a: &[i8],
    b: &[i8],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [i32],
) {
    use std::arch::x86_64::*;
    const MR: usize = 8;
    const NR: usize = 16;

    let (bp, k4_full) = pack_b_vnni_4x16(b, k, n);
    let groups = k4_full / 4;
    let n_panels = n / NR;

    let mut col_sum_b_vnni = vec![0_i32; n];
    for j in 0..n {
        let mut s: i32 = 0;
        for kk in 0..k4_full {
            s += b[kk * n + j] as i32;
        }
        col_sum_b_vnni[j] = s;
    }
    let bias = 128_i32;
    let bias_xor = _mm512_set1_epi8(-128_i8);

    for ib in (0..m).step_by(MR) {
        for jp in 0..n_panels {
            let mut acc0 = _mm512_setzero_si512();
            let mut acc1 = _mm512_setzero_si512();
            let mut acc2 = _mm512_setzero_si512();
            let mut acc3 = _mm512_setzero_si512();
            let mut acc4 = _mm512_setzero_si512();
            let mut acc5 = _mm512_setzero_si512();
            let mut acc6 = _mm512_setzero_si512();
            let mut acc7 = _mm512_setzero_si512();
            let bp_panel = jp * groups * NR * 4;
            for g in 0..groups {
                let bv =
                    _mm512_loadu_si512(bp.as_ptr().add(bp_panel + g * NR * 4) as *const __m512i);
                let k_idx = g * 4;
                let row_base = ib * k + k_idx;
                let p0 = (a.as_ptr().add(row_base) as *const i32).read_unaligned();
                let p1 = (a.as_ptr().add(row_base + k) as *const i32).read_unaligned();
                let p2 = (a.as_ptr().add(row_base + 2 * k) as *const i32).read_unaligned();
                let p3 = (a.as_ptr().add(row_base + 3 * k) as *const i32).read_unaligned();
                let p4 = (a.as_ptr().add(row_base + 4 * k) as *const i32).read_unaligned();
                let p5 = (a.as_ptr().add(row_base + 5 * k) as *const i32).read_unaligned();
                let p6 = (a.as_ptr().add(row_base + 6 * k) as *const i32).read_unaligned();
                let p7 = (a.as_ptr().add(row_base + 7 * k) as *const i32).read_unaligned();
                let av0 = _mm512_xor_si512(_mm512_set1_epi32(p0), bias_xor);
                let av1 = _mm512_xor_si512(_mm512_set1_epi32(p1), bias_xor);
                let av2 = _mm512_xor_si512(_mm512_set1_epi32(p2), bias_xor);
                let av3 = _mm512_xor_si512(_mm512_set1_epi32(p3), bias_xor);
                let av4 = _mm512_xor_si512(_mm512_set1_epi32(p4), bias_xor);
                let av5 = _mm512_xor_si512(_mm512_set1_epi32(p5), bias_xor);
                let av6 = _mm512_xor_si512(_mm512_set1_epi32(p6), bias_xor);
                let av7 = _mm512_xor_si512(_mm512_set1_epi32(p7), bias_xor);
                acc0 = _mm512_dpbusd_epi32(acc0, av0, bv);
                acc1 = _mm512_dpbusd_epi32(acc1, av1, bv);
                acc2 = _mm512_dpbusd_epi32(acc2, av2, bv);
                acc3 = _mm512_dpbusd_epi32(acc3, av3, bv);
                acc4 = _mm512_dpbusd_epi32(acc4, av4, bv);
                acc5 = _mm512_dpbusd_epi32(acc5, av5, bv);
                acc6 = _mm512_dpbusd_epi32(acc6, av6, bv);
                acc7 = _mm512_dpbusd_epi32(acc7, av7, bv);
            }
            let mut buf = [[0_i32; NR]; MR];
            _mm512_storeu_si512(buf[0].as_mut_ptr() as *mut __m512i, acc0);
            _mm512_storeu_si512(buf[1].as_mut_ptr() as *mut __m512i, acc1);
            _mm512_storeu_si512(buf[2].as_mut_ptr() as *mut __m512i, acc2);
            _mm512_storeu_si512(buf[3].as_mut_ptr() as *mut __m512i, acc3);
            _mm512_storeu_si512(buf[4].as_mut_ptr() as *mut __m512i, acc4);
            _mm512_storeu_si512(buf[5].as_mut_ptr() as *mut __m512i, acc5);
            _mm512_storeu_si512(buf[6].as_mut_ptr() as *mut __m512i, acc6);
            _mm512_storeu_si512(buf[7].as_mut_ptr() as *mut __m512i, acc7);
            for r_off in 0..MR {
                for c in 0..NR {
                    let n_idx = jp * NR + c;
                    let mut v = buf[r_off][c] - bias * col_sum_b_vnni[n_idx];
                    for kk in k4_full..k {
                        v += (a[(ib + r_off) * k + kk] as i32) * (b[kk * n + n_idx] as i32);
                    }
                    out[(ib + r_off) * n + n_idx] = v;
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn int8_matmul_avx512_vnni(
    a: &[i8],
    b: &[i8],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [i32],
) {
    use std::arch::x86_64::*;
    // Register-blocked dispatch: pick MR=8 only on the empirically-good
    // regime — large K with small N (e.g. Llama down-proj 64×8192×2048)
    // — where the doubled B reuse beats the simple Bᵀ kernel. On
    // hidden×hidden (K=N=2048) and on gate/up with N≫K, the strided
    // 8-row A pattern thrashes L1 and the simple kernel wins.
    if m >= 8 && m.is_multiple_of(8) && n.is_multiple_of(16) && k >= 8000 && n <= 2048 {
        return int8_matmul_avx512_vnni_blocked_mr8(a, b, m, k, n, out);
    }
    if m >= 4
        && m.is_multiple_of(4)
        && n.is_multiple_of(16)
        && k >= 4
        && (k * n <= 1_000_000 || n >= 16384)
    {
        return int8_matmul_avx512_vnni_blocked(a, b, m, k, n, out);
    }
    let bt = transpose_b(b, k, n);
    let bias: i32 = 128;
    let bias128 = _mm512_set1_epi8(-128_i8);
    let k_vnni = (k / 64) * 64;
    let mut col_sum_b_vnni = vec![0_i32; n];
    for j in 0..n {
        let mut s: i32 = 0;
        for kk in 0..k_vnni {
            s += bt[j * k + kk] as i32;
        }
        col_sum_b_vnni[j] = s;
    }
    for i in 0..m {
        for j in 0..n {
            let mut acc = _mm512_setzero_si512();
            let mut kk = 0;
            while kk + 64 <= k {
                let av = _mm512_loadu_si512(a.as_ptr().add(i * k + kk) as *const __m512i);
                let bv = _mm512_loadu_si512(bt.as_ptr().add(j * k + kk) as *const __m512i);
                // Bias-shift: a' = a XOR 0x80 makes dot_us(a',b) = dot(a,b)
                // + 128·sum(b). Subtract once at the end via col_sum_b_vnni.
                let av_u = _mm512_xor_si512(av, bias128);
                acc = _mm512_dpbusd_epi32(acc, av_u, bv);
                kk += 64;
            }
            let lane_sum = _mm512_reduce_add_epi32(acc);
            let mut tail = lane_sum - bias * col_sum_b_vnni[j];
            while kk < k {
                tail += (a[i * k + kk] as i32) * (bt[j * k + kk] as i32);
                kk += 1;
            }
            out[i * n + j] = tail;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn int8_matmul_prepacked_avx512_vnni(a: &[i8], b: &PackedI8B, m: usize, out: &mut [i32]) {
    use std::arch::x86_64::*;
    let k = b.k;
    let n = b.n;
    if b.vnni_4x16.is_some()
        && m >= 4
        && m.is_multiple_of(4)
        && n.is_multiple_of(16)
        && k >= 4
        && (k * n <= 1_000_000 || n >= 16384)
    {
        return int8_matmul_prepacked_avx512_vnni_blocked(a, b, m, out);
    }
    let bt = b.transposed();
    let bias: i32 = 128;
    let bias128 = _mm512_set1_epi8(-128_i8);
    let k_vnni = (k / 64) * 64;
    let mut col_sum_b_vnni = vec![0_i32; n];
    for j in 0..n {
        let mut s: i32 = 0;
        for kk in 0..k_vnni {
            s += bt[j * k + kk] as i32;
        }
        col_sum_b_vnni[j] = s;
    }
    for i in 0..m {
        for j in 0..n {
            let mut acc = _mm512_setzero_si512();
            let mut kk = 0;
            while kk + 64 <= k {
                let av = _mm512_loadu_si512(a.as_ptr().add(i * k + kk) as *const __m512i);
                let bv = _mm512_loadu_si512(bt.as_ptr().add(j * k + kk) as *const __m512i);
                let av_u = _mm512_xor_si512(av, bias128);
                acc = _mm512_dpbusd_epi32(acc, av_u, bv);
                kk += 64;
            }
            let lane_sum = _mm512_reduce_add_epi32(acc);
            let mut tail = lane_sum - bias * col_sum_b_vnni[j];
            while kk < k {
                tail += (a[i * k + kk] as i32) * (bt[j * k + kk] as i32);
                kk += 1;
            }
            out[i * n + j] = tail;
        }
    }
}

/// Runtime-dispatched int8 GEMM. Picks the best variant available on the
/// host CPU; falls back to scalar when no SIMD path matches.
pub fn int8_matmul_dispatch(a: &[i8], b: &[i8], m: usize, k: usize, n: usize, out: &mut [i32]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(out.len(), m * n);

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vnni")
        {
            unsafe { int8_matmul_avx512_vnni(a, b, m, k, n, out) };
            return;
        }
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("avxvnni") {
            unsafe { int8_matmul_avx_vnni(a, b, m, k, n, out) };
            return;
        }
        if std::is_x86_feature_detected!("avx2") {
            unsafe { int8_matmul_avx2_widen(a, b, m, k, n, out) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon")
            && std::arch::is_aarch64_feature_detected!("i8mm")
        {
            unsafe { int8_matmul_neon_i8mm(a, b, m, k, n, out) };
            return;
        }
        if std::arch::is_aarch64_feature_detected!("neon")
            && std::arch::is_aarch64_feature_detected!("dotprod")
        {
            unsafe { int8_matmul_neon_sdot(a, b, m, k, n, out) };
            return;
        }
    }
    int8_matmul_scalar(a, b, m, k, n, out);
}

/// Runtime-dispatched INT8 GEMM for a load-time packed RHS.
///
/// This has the same numerical contract as [`int8_matmul_dispatch`] but
/// avoids the per-call B transpose and gives every backend contiguous
/// `[K]` slices for each output column.
pub fn int8_matmul_prepacked_dispatch(a: &[i8], b: &PackedI8B, m: usize, out: &mut [i32]) {
    debug_assert_eq!(a.len(), m * b.k);
    debug_assert_eq!(out.len(), m * b.n);

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vnni")
        {
            unsafe { int8_matmul_prepacked_avx512_vnni(a, b, m, out) };
            return;
        }
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("avxvnni") {
            unsafe { int8_matmul_prepacked_avx_vnni(a, b, m, out) };
            return;
        }
        if std::is_x86_feature_detected!("avx2") {
            unsafe { int8_matmul_prepacked_avx2_widen(a, b, m, out) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon")
            && std::arch::is_aarch64_feature_detected!("dotprod")
        {
            unsafe { int8_matmul_prepacked_neon_sdot(a, b, m, out) };
            return;
        }
    }
    int8_matmul_prepacked_scalar(a, b, m, out);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ref_matmul(a: &[i8], b: &[i8], m: usize, k: usize, n: usize) -> Vec<i32> {
        let mut out = vec![0_i32; m * n];
        int8_matmul_scalar(a, b, m, k, n, &mut out);
        out
    }

    fn pseudo_random(seed: u64, n: usize) -> Vec<i8> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as i64 % 256 - 128) as i8
            })
            .collect()
    }

    #[test]
    fn scalar_zero_inputs_yield_zero_output() {
        let a = vec![0_i8; 12];
        let b = vec![0_i8; 12];
        let mut out = vec![1_i32; 9];
        int8_matmul_scalar(&a, &b, 3, 4, 3, &mut out);
        assert!(out.iter().all(|&v| v == 0));
    }

    #[test]
    fn scalar_known_small_case() {
        let a: Vec<i8> = vec![1, 2, 3, 4];
        let b: Vec<i8> = vec![5, 6, 7, 8];
        let mut out = vec![0_i32; 4];
        int8_matmul_scalar(&a, &b, 2, 2, 2, &mut out);
        assert_eq!(out, vec![19, 22, 43, 50]);
    }

    #[test]
    fn dispatch_matches_scalar_on_random_shapes() {
        for &(m, k, n) in &[(1, 1, 1), (2, 3, 4), (4, 16, 8), (5, 17, 6), (8, 64, 8)] {
            let a = pseudo_random(0xDEAD ^ (m * k) as u64, m * k);
            let b = pseudo_random(0xBEEF ^ (k * n) as u64, k * n);
            let expected = ref_matmul(&a, &b, m, k, n);
            let mut got = vec![i32::MIN; m * n];
            int8_matmul_dispatch(&a, &b, m, k, n, &mut got);
            assert_eq!(got, expected, "shape m={m} k={k} n={n}");
        }
    }

    #[test]
    fn prepacked_dispatch_matches_scalar_on_random_shapes() {
        for &(m, k, n) in &[
            (1, 1, 1),
            (1, 17, 5),
            (3, 31, 7),
            (4, 64, 16),
            (8, 127, 33),
            (16, 128, 64),
        ] {
            let a = pseudo_random(0xA11CE ^ (m * k) as u64, m * k);
            let b = pseudo_random(0xB0B ^ (k * n) as u64, k * n);
            let packed = pack_i8_b_for_matmul(&b, k, n);
            assert_eq!(packed.k(), k);
            assert_eq!(packed.n(), n);
            assert_eq!(packed.transposed().len(), k * n);
            let expected = ref_matmul(&a, &b, m, k, n);
            let mut got = vec![i32::MIN; m * n];
            int8_matmul_prepacked_dispatch(&a, &packed, m, &mut got);
            assert_eq!(got, expected, "shape m={m} k={k} n={n}");
        }
    }

    #[test]
    fn dispatch_handles_large_k_with_extreme_values() {
        let a = vec![-128_i8; 128];
        let b = vec![-128_i8; 128];
        let mut out = vec![0_i32; 1];
        int8_matmul_dispatch(&a, &b, 1, 128, 1, &mut out);
        assert_eq!(out[0], 128 * 128 * 128);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_sdot_matches_scalar_when_available() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        let a = pseudo_random(0x1234, 32 * 32);
        let b = pseudo_random(0x5678, 32 * 16);
        let expected = ref_matmul(&a, &b, 32, 32, 16);
        let mut got = vec![0_i32; 32 * 16];
        unsafe { int8_matmul_neon_sdot(&a, &b, 32, 32, 16, &mut got) };
        assert_eq!(got, expected);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_i8mm_matches_scalar_when_available() {
        if !(std::arch::is_aarch64_feature_detected!("dotprod")
            && std::arch::is_aarch64_feature_detected!("i8mm"))
        {
            return;
        }
        for &(m, k, n) in &[(8, 64, 16), (5, 32, 7), (4, 24, 4)] {
            let a = pseudo_random(0xAA ^ (m * k) as u64, m * k);
            let b = pseudo_random(0xBB ^ (k * n) as u64, k * n);
            let expected = ref_matmul(&a, &b, m, k, n);
            let mut got = vec![0_i32; m * n];
            unsafe { int8_matmul_neon_i8mm(&a, &b, m, k, n, &mut got) };
            assert_eq!(got, expected, "shape m={m} k={k} n={n}");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_widen_matches_scalar_when_available() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let a = pseudo_random(0x1234, 32 * 32);
        let b = pseudo_random(0x5678, 32 * 16);
        let expected = ref_matmul(&a, &b, 32, 32, 16);
        let mut got = vec![0_i32; 32 * 16];
        unsafe { int8_matmul_avx2_widen(&a, &b, 32, 32, 16, &mut got) };
        assert_eq!(got, expected);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx_vnni_matches_scalar_when_available() {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("avxvnni")) {
            return;
        }
        let a = pseudo_random(0x1234, 8 * 96);
        let b = pseudo_random(0x5678, 96 * 8);
        let expected = ref_matmul(&a, &b, 8, 96, 8);
        let mut got = vec![0_i32; 8 * 8];
        unsafe { int8_matmul_avx_vnni(&a, &b, 8, 96, 8, &mut got) };
        assert_eq!(got, expected);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_vnni_matches_scalar_when_available() {
        if !(std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vnni"))
        {
            return;
        }
        let a = pseudo_random(0xC0FFEE, 8 * 128);
        let b = pseudo_random(0xCAFE, 128 * 8);
        let expected = ref_matmul(&a, &b, 8, 128, 8);
        let mut got = vec![0_i32; 8 * 8];
        unsafe { int8_matmul_avx512_vnni(&a, &b, 8, 128, 8, &mut got) };
        assert_eq!(got, expected);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_vnni_blocked_mr8_path_matches_scalar() {
        if !(std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vnni"))
        {
            return;
        }
        for &(m, k, n) in &[
            (8, 8, 16),
            (16, 16, 32),
            (8, 17, 16),  // K = 4*4 + 1 tail
            (24, 19, 48), // K = 4*4 + 3 tail, larger M
            (64, 256, 64),
        ] {
            let a = pseudo_random(0xB108 ^ (m * k) as u64, m * k);
            let b = pseudo_random(0xCAFE ^ (k * n) as u64, k * n);
            let expected = ref_matmul(&a, &b, m, k, n);
            let mut got = vec![0_i32; m * n];
            unsafe { int8_matmul_avx512_vnni_blocked_mr8(&a, &b, m, k, n, &mut got) };
            assert_eq!(got, expected, "MR=8 m={m} k={k} n={n}");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_vnni_blocked_path_matches_scalar() {
        if !(std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vnni"))
        {
            return;
        }
        // Shapes that hit the MR=4 NR=16 register-blocked path. Cover
        // (a) clean alignment, (b) K not divisible by 4 (tail 1-3),
        // (c) larger K to ensure many group iterations.
        for &(m, k, n) in &[
            (4, 8, 16),
            (8, 16, 32),
            (4, 17, 16),  // K = 4*4 + 1, tail of 1
            (12, 19, 48), // K = 4*4 + 3, tail of 3
            (16, 256, 64),
        ] {
            let a = pseudo_random(0xB10C ^ (m * k) as u64, m * k);
            let b = pseudo_random(0xCAFE ^ (k * n) as u64, k * n);
            let expected = ref_matmul(&a, &b, m, k, n);
            let mut got = vec![0_i32; m * n];
            unsafe { int8_matmul_avx512_vnni_blocked(&a, &b, m, k, n, &mut got) };
            assert_eq!(got, expected, "blocked m={m} k={k} n={n}");
        }
    }
}
