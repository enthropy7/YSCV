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
    for i in 0..m {
        for j in 0..n {
            let mut acc = _mm256_setzero_si256();
            let mut kk = 0;
            while kk + 16 <= k {
                let av = _mm_loadu_si128(a.as_ptr().add(i * k + kk) as *const __m128i);
                let mut bbuf = [0_i8; 16];
                for r in 0..16 {
                    bbuf[r] = b[(kk + r) * n + j];
                }
                let bv = _mm_loadu_si128(bbuf.as_ptr() as *const __m128i);
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
                tail += (a[i * k + kk] as i32) * (b[kk * n + j] as i32);
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
    let bias: i32 = 128;
    let bias128 = _mm256_set1_epi8(-128_i8);
    for i in 0..m {
        for j in 0..n {
            let mut acc = _mm256_setzero_si256();
            let mut sum_b: i32 = 0;
            let mut kk = 0;
            while kk + 32 <= k {
                let av = _mm256_loadu_si256(a.as_ptr().add(i * k + kk) as *const __m256i);
                let mut bbuf = [0_i8; 32];
                for r in 0..32 {
                    let bv = b[(kk + r) * n + j];
                    bbuf[r] = bv;
                    sum_b += bv as i32;
                }
                let bv = _mm256_loadu_si256(bbuf.as_ptr() as *const __m256i);
                // a → a + 128 (mod 256, reinterpret i8→u8) so unsigned-signed
                // VNNI gives `dot(a, b) + 128 * sum(b)`. Subtract `128 * sum_b`
                // at the end to get the signed-signed result.
                let av_u = _mm256_xor_si256(av, bias128);
                acc = _mm256_dpbusd_avx_epi32(acc, av_u, bv);
                kk += 32;
            }
            let mut buf = [0_i32; 8];
            _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, acc);
            let mut tail = buf.iter().sum::<i32>() - bias * sum_b;
            while kk < k {
                tail += (a[i * k + kk] as i32) * (b[kk * n + j] as i32);
                kk += 1;
            }
            out[i * n + j] = tail;
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
    for i in 0..m {
        for j in 0..n {
            let mut acc = _mm512_setzero_si512();
            let mut kk = 0;
            // Bias-shift: a' = a XOR 0x80 = a + 128 (mod 256, but the
            // i8→u8 reinterpretation lands in the right range). Then
            // dot_unsigned_signed(a', b) = dot(a, b) + 128 * sum(b),
            // so we accumulate 128 * sum_b separately and subtract.
            let bias: i32 = 128;
            let mut sum_b: i32 = 0;
            while kk + 64 <= k {
                let av = _mm512_loadu_si512(a.as_ptr().add(i * k + kk) as *const __m512i);
                let mut bbuf = [0_i8; 64];
                for r in 0..64 {
                    let bv = b[(kk + r) * n + j];
                    bbuf[r] = bv;
                    sum_b += bv as i32;
                }
                let bv = _mm512_loadu_si512(bbuf.as_ptr() as *const __m512i);
                let bias128 = _mm512_set1_epi8(-128_i8);
                let av_u = _mm512_xor_si512(av, bias128);
                acc = _mm512_dpbusd_epi32(acc, av_u, bv);
                kk += 64;
            }
            let lane_sum = _mm512_reduce_add_epi32(acc);
            let mut tail = lane_sum - bias * sum_b;
            while kk < k {
                tail += (a[i * k + kk] as i32) * (b[kk * n + j] as i32);
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
}
