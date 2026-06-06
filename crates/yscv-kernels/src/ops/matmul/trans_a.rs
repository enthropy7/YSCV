//! Transpose+MatMul (FusedTransposeMatMul) kernels: the non-transposed and
//! transposed-A 4-row × NR=16 outer-product tiles (AVX-512/AVX2/NEON) and
//! their dispatchers.

use super::*;

pub(super) fn non_trans_4row_disabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var_os("YSCV_NON_TRANS_4ROW_OFF").is_some())
}

pub(super) fn non_trans_a_4row_dispatch(
    a: &[f32],
    mi_base: usize,
    k: usize,
    b: &[f32],
    n: usize,
    out_4rows: &mut [f32],
) {
    debug_assert_eq!(out_4rows.len(), 4 * n);
    #[cfg(target_arch = "x86_64")]
    {
        if !cfg!(miri) && n.is_multiple_of(16) && crate::host_cpu().features.avx512f {
            #[allow(unsafe_code)]
            unsafe {
                non_trans_a_4row_avx512(a, mi_base, k, b, n, out_4rows);
            }
            return;
        }
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if !cfg!(miri)
            && n.is_multiple_of(8)
            && crate::host_cpu().features.avx
            && crate::host_cpu().features.fma
        {
            #[allow(unsafe_code)]
            unsafe {
                non_trans_a_4row_avx2(a, mi_base, k, b, n, out_4rows);
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if !cfg!(miri) && n.is_multiple_of(4) && crate::host_cpu().features.neon {
            #[allow(unsafe_code)]
            unsafe {
                non_trans_a_4row_neon(a, mi_base, k, b, n, out_4rows);
            }
            return;
        }
    }
    // Scalar fallback: per-row dispatch.
    for (row_off, out_row) in out_4rows.chunks_exact_mut(n).enumerate() {
        #[allow(unsafe_code)]
        unsafe {
            matmul_row_set_dispatch(
                a.as_ptr().add((mi_base + row_off) * k),
                b.as_ptr(),
                out_row.as_mut_ptr(),
                k,
                n,
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_code)]
unsafe fn non_trans_a_4row_avx512(
    a: &[f32],
    mi_base: usize,
    k: usize,
    b: &[f32],
    n: usize,
    out_4rows: &mut [f32],
) {
    use std::arch::x86_64::*;
    unsafe {
        let r0 = out_4rows.as_mut_ptr();
        let r1 = r0.add(n);
        let r2 = r0.add(2 * n);
        let r3 = r0.add(3 * n);
        // A in [M, K] row-major. Base pointers to the 4 rows we're producing.
        let a_row0 = a.as_ptr().add(mi_base * k);
        let a_row1 = a_row0.add(k);
        let a_row2 = a_row0.add(2 * k);
        let a_row3 = a_row0.add(3 * k);
        let blocks = n / 64;
        let tail_chunks = (n % 64) / 16;
        let zero = _mm512_setzero_ps();
        // 4×4 outer-product super-block: 16 acc ZMMs in flight.
        for block in 0..blocks {
            let col = block * 64;
            let mut c00 = zero;
            let mut c01 = zero;
            let mut c02 = zero;
            let mut c03 = zero;
            let mut c10 = zero;
            let mut c11 = zero;
            let mut c12 = zero;
            let mut c13 = zero;
            let mut c20 = zero;
            let mut c21 = zero;
            let mut c22 = zero;
            let mut c23 = zero;
            let mut c30 = zero;
            let mut c31 = zero;
            let mut c32 = zero;
            let mut c33 = zero;
            for ki in 0..k {
                let a0 = _mm512_set1_ps(*a_row0.add(ki));
                let a1 = _mm512_set1_ps(*a_row1.add(ki));
                let a2 = _mm512_set1_ps(*a_row2.add(ki));
                let a3 = _mm512_set1_ps(*a_row3.add(ki));
                let bptr = b.as_ptr().add(ki * n + col);
                let b0 = _mm512_loadu_ps(bptr);
                let b1 = _mm512_loadu_ps(bptr.add(16));
                let b2 = _mm512_loadu_ps(bptr.add(32));
                let b3 = _mm512_loadu_ps(bptr.add(48));
                c00 = _mm512_fmadd_ps(a0, b0, c00);
                c01 = _mm512_fmadd_ps(a0, b1, c01);
                c02 = _mm512_fmadd_ps(a0, b2, c02);
                c03 = _mm512_fmadd_ps(a0, b3, c03);
                c10 = _mm512_fmadd_ps(a1, b0, c10);
                c11 = _mm512_fmadd_ps(a1, b1, c11);
                c12 = _mm512_fmadd_ps(a1, b2, c12);
                c13 = _mm512_fmadd_ps(a1, b3, c13);
                c20 = _mm512_fmadd_ps(a2, b0, c20);
                c21 = _mm512_fmadd_ps(a2, b1, c21);
                c22 = _mm512_fmadd_ps(a2, b2, c22);
                c23 = _mm512_fmadd_ps(a2, b3, c23);
                c30 = _mm512_fmadd_ps(a3, b0, c30);
                c31 = _mm512_fmadd_ps(a3, b1, c31);
                c32 = _mm512_fmadd_ps(a3, b2, c32);
                c33 = _mm512_fmadd_ps(a3, b3, c33);
            }
            _mm512_storeu_ps(r0.add(col), c00);
            _mm512_storeu_ps(r0.add(col + 16), c01);
            _mm512_storeu_ps(r0.add(col + 32), c02);
            _mm512_storeu_ps(r0.add(col + 48), c03);
            _mm512_storeu_ps(r1.add(col), c10);
            _mm512_storeu_ps(r1.add(col + 16), c11);
            _mm512_storeu_ps(r1.add(col + 32), c12);
            _mm512_storeu_ps(r1.add(col + 48), c13);
            _mm512_storeu_ps(r2.add(col), c20);
            _mm512_storeu_ps(r2.add(col + 16), c21);
            _mm512_storeu_ps(r2.add(col + 32), c22);
            _mm512_storeu_ps(r2.add(col + 48), c23);
            _mm512_storeu_ps(r3.add(col), c30);
            _mm512_storeu_ps(r3.add(col + 16), c31);
            _mm512_storeu_ps(r3.add(col + 32), c32);
            _mm512_storeu_ps(r3.add(col + 48), c33);
        }
        // Tail: N%64 in NR=16 chunks (4 acc ZMMs per chunk).
        let tail_base = blocks * 64;
        for chunk in 0..tail_chunks {
            let col = tail_base + chunk * 16;
            let mut c0 = zero;
            let mut c1 = zero;
            let mut c2 = zero;
            let mut c3 = zero;
            for ki in 0..k {
                let a0 = _mm512_set1_ps(*a_row0.add(ki));
                let a1 = _mm512_set1_ps(*a_row1.add(ki));
                let a2 = _mm512_set1_ps(*a_row2.add(ki));
                let a3 = _mm512_set1_ps(*a_row3.add(ki));
                let bv = _mm512_loadu_ps(b.as_ptr().add(ki * n + col));
                c0 = _mm512_fmadd_ps(a0, bv, c0);
                c1 = _mm512_fmadd_ps(a1, bv, c1);
                c2 = _mm512_fmadd_ps(a2, bv, c2);
                c3 = _mm512_fmadd_ps(a3, bv, c3);
            }
            _mm512_storeu_ps(r0.add(col), c0);
            _mm512_storeu_ps(r1.add(col), c1);
            _mm512_storeu_ps(r2.add(col), c2);
            _mm512_storeu_ps(r3.add(col), c3);
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
#[allow(unsafe_code)]
unsafe fn non_trans_a_4row_avx2(
    a: &[f32],
    mi_base: usize,
    k: usize,
    b: &[f32],
    n: usize,
    out_4rows: &mut [f32],
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    unsafe {
        let r0 = out_4rows.as_mut_ptr();
        let r1 = r0.add(n);
        let r2 = r0.add(2 * n);
        let r3 = r0.add(3 * n);
        let a_row0 = a.as_ptr().add(mi_base * k);
        let a_row1 = a_row0.add(k);
        let a_row2 = a_row0.add(2 * k);
        let a_row3 = a_row0.add(3 * k);
        // AVX2: 4 rows × 2 NR=8 panels = 8 acc YMMs + 2 b panels + 4 a broadcasts.
        let blocks = n / 16;
        let tail_chunks = (n % 16) / 8;
        let zero = _mm256_setzero_ps();
        for block in 0..blocks {
            let col = block * 16;
            let mut c00 = zero;
            let mut c01 = zero;
            let mut c10 = zero;
            let mut c11 = zero;
            let mut c20 = zero;
            let mut c21 = zero;
            let mut c30 = zero;
            let mut c31 = zero;
            for ki in 0..k {
                let a0 = _mm256_set1_ps(*a_row0.add(ki));
                let a1 = _mm256_set1_ps(*a_row1.add(ki));
                let a2 = _mm256_set1_ps(*a_row2.add(ki));
                let a3 = _mm256_set1_ps(*a_row3.add(ki));
                let bptr = b.as_ptr().add(ki * n + col);
                let b0 = _mm256_loadu_ps(bptr);
                let b1 = _mm256_loadu_ps(bptr.add(8));
                c00 = _mm256_fmadd_ps(a0, b0, c00);
                c01 = _mm256_fmadd_ps(a0, b1, c01);
                c10 = _mm256_fmadd_ps(a1, b0, c10);
                c11 = _mm256_fmadd_ps(a1, b1, c11);
                c20 = _mm256_fmadd_ps(a2, b0, c20);
                c21 = _mm256_fmadd_ps(a2, b1, c21);
                c30 = _mm256_fmadd_ps(a3, b0, c30);
                c31 = _mm256_fmadd_ps(a3, b1, c31);
            }
            _mm256_storeu_ps(r0.add(col), c00);
            _mm256_storeu_ps(r0.add(col + 8), c01);
            _mm256_storeu_ps(r1.add(col), c10);
            _mm256_storeu_ps(r1.add(col + 8), c11);
            _mm256_storeu_ps(r2.add(col), c20);
            _mm256_storeu_ps(r2.add(col + 8), c21);
            _mm256_storeu_ps(r3.add(col), c30);
            _mm256_storeu_ps(r3.add(col + 8), c31);
        }
        let tail_base = blocks * 16;
        for chunk in 0..tail_chunks {
            let col = tail_base + chunk * 8;
            let mut c0 = zero;
            let mut c1 = zero;
            let mut c2 = zero;
            let mut c3 = zero;
            for ki in 0..k {
                let a0 = _mm256_set1_ps(*a_row0.add(ki));
                let a1 = _mm256_set1_ps(*a_row1.add(ki));
                let a2 = _mm256_set1_ps(*a_row2.add(ki));
                let a3 = _mm256_set1_ps(*a_row3.add(ki));
                let bv = _mm256_loadu_ps(b.as_ptr().add(ki * n + col));
                c0 = _mm256_fmadd_ps(a0, bv, c0);
                c1 = _mm256_fmadd_ps(a1, bv, c1);
                c2 = _mm256_fmadd_ps(a2, bv, c2);
                c3 = _mm256_fmadd_ps(a3, bv, c3);
            }
            _mm256_storeu_ps(r0.add(col), c0);
            _mm256_storeu_ps(r1.add(col), c1);
            _mm256_storeu_ps(r2.add(col), c2);
            _mm256_storeu_ps(r3.add(col), c3);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
unsafe fn non_trans_a_4row_neon(
    a: &[f32],
    mi_base: usize,
    k: usize,
    b: &[f32],
    n: usize,
    out_4rows: &mut [f32],
) {
    unsafe {
        let r0 = out_4rows.as_mut_ptr();
        let r1 = r0.add(n);
        let r2 = r0.add(2 * n);
        let r3 = r0.add(3 * n);
        let a_row0 = a.as_ptr().add(mi_base * k);
        let a_row1 = a_row0.add(k);
        let a_row2 = a_row0.add(2 * k);
        let a_row3 = a_row0.add(3 * k);
        // NEON 4 rows × 4 NR=4 panels = 16 acc q-regs.
        let blocks = n / 16;
        let tail_chunks = (n % 16) / 4;
        let zero = vdupq_n_f32(0.0);
        for block in 0..blocks {
            let col = block * 16;
            let mut c00 = zero;
            let mut c01 = zero;
            let mut c02 = zero;
            let mut c03 = zero;
            let mut c10 = zero;
            let mut c11 = zero;
            let mut c12 = zero;
            let mut c13 = zero;
            let mut c20 = zero;
            let mut c21 = zero;
            let mut c22 = zero;
            let mut c23 = zero;
            let mut c30 = zero;
            let mut c31 = zero;
            let mut c32 = zero;
            let mut c33 = zero;
            for ki in 0..k {
                let a0 = vdupq_n_f32(*a_row0.add(ki));
                let a1 = vdupq_n_f32(*a_row1.add(ki));
                let a2 = vdupq_n_f32(*a_row2.add(ki));
                let a3 = vdupq_n_f32(*a_row3.add(ki));
                let bptr = b.as_ptr().add(ki * n + col);
                let b0 = vld1q_f32(bptr);
                let b1 = vld1q_f32(bptr.add(4));
                let b2 = vld1q_f32(bptr.add(8));
                let b3 = vld1q_f32(bptr.add(12));
                c00 = vfmaq_f32(c00, a0, b0);
                c01 = vfmaq_f32(c01, a0, b1);
                c02 = vfmaq_f32(c02, a0, b2);
                c03 = vfmaq_f32(c03, a0, b3);
                c10 = vfmaq_f32(c10, a1, b0);
                c11 = vfmaq_f32(c11, a1, b1);
                c12 = vfmaq_f32(c12, a1, b2);
                c13 = vfmaq_f32(c13, a1, b3);
                c20 = vfmaq_f32(c20, a2, b0);
                c21 = vfmaq_f32(c21, a2, b1);
                c22 = vfmaq_f32(c22, a2, b2);
                c23 = vfmaq_f32(c23, a2, b3);
                c30 = vfmaq_f32(c30, a3, b0);
                c31 = vfmaq_f32(c31, a3, b1);
                c32 = vfmaq_f32(c32, a3, b2);
                c33 = vfmaq_f32(c33, a3, b3);
            }
            vst1q_f32(r0.add(col), c00);
            vst1q_f32(r0.add(col + 4), c01);
            vst1q_f32(r0.add(col + 8), c02);
            vst1q_f32(r0.add(col + 12), c03);
            vst1q_f32(r1.add(col), c10);
            vst1q_f32(r1.add(col + 4), c11);
            vst1q_f32(r1.add(col + 8), c12);
            vst1q_f32(r1.add(col + 12), c13);
            vst1q_f32(r2.add(col), c20);
            vst1q_f32(r2.add(col + 4), c21);
            vst1q_f32(r2.add(col + 8), c22);
            vst1q_f32(r2.add(col + 12), c23);
            vst1q_f32(r3.add(col), c30);
            vst1q_f32(r3.add(col + 4), c31);
            vst1q_f32(r3.add(col + 8), c32);
            vst1q_f32(r3.add(col + 12), c33);
        }
        let tail_base = blocks * 16;
        for chunk in 0..tail_chunks {
            let col = tail_base + chunk * 4;
            let mut c0 = zero;
            let mut c1 = zero;
            let mut c2 = zero;
            let mut c3 = zero;
            for ki in 0..k {
                let a0 = vdupq_n_f32(*a_row0.add(ki));
                let a1 = vdupq_n_f32(*a_row1.add(ki));
                let a2 = vdupq_n_f32(*a_row2.add(ki));
                let a3 = vdupq_n_f32(*a_row3.add(ki));
                let bv = vld1q_f32(b.as_ptr().add(ki * n + col));
                c0 = vfmaq_f32(c0, a0, bv);
                c1 = vfmaq_f32(c1, a1, bv);
                c2 = vfmaq_f32(c2, a2, bv);
                c3 = vfmaq_f32(c3, a3, bv);
            }
            vst1q_f32(r0.add(col), c0);
            vst1q_f32(r1.add(col), c1);
            vst1q_f32(r2.add(col), c2);
            vst1q_f32(r3.add(col), c3);
        }
    }
}

/// Matmul with transposed left operand. `a_kt` is shape `[K, M]` in
/// row-major memory (i.e. NOT transposed — it's the original A before
/// the Transpose node we've fused into this call). Computes
/// `out[m, n] = sum_k a_kt[k, m] * b[k, n]` — equivalent to
/// `(a_kt^T) @ b` — without materialising the transpose.
///
/// The native direct path handles the transposed-access pattern without
/// materialising the transpose. If it is disabled via env, the fallback
/// materialises the transpose and calls the standard matmul path.
///
/// Used by the Transpose-perm[0,2,1] → MatMul fusion in the runner.
pub fn matmul_2d_slices_trans_a(
    a_kt: &[f32],
    m: usize,
    k: usize,
    b: &[f32],
    n: usize,
    out: &mut [f32],
) {
    debug_assert!(a_kt.len() >= k * m);
    debug_assert!(b.len() >= k * n);
    debug_assert!(out.len() >= m * n);

    if !trans_a_direct_disabled() {
        matmul_2d_slices_trans_a_direct(a_kt, m, k, b, n, out);
        return;
    }

    // Fallback: materialise the transposed view and call the standard path.
    // zero-init + transpose in one pass. The zero-fill cost is a single
    // linear memset; caller is the BLAS-transA fallback path, not a hot
    // loop, so the extra write is negligible. Replaces a prior
    // `with_capacity + set_len` pattern flagged by `clippy::uninit_vec`.
    let mut a_transposed: Vec<f32> = vec![0.0; m * k];
    for mi in 0..m {
        for ki in 0..k {
            a_transposed[mi * k + ki] = a_kt[ki * m + mi];
        }
    }
    matmul_2d_slices_fused(
        &a_transposed,
        m,
        k,
        b,
        n,
        out,
        GemmEpilogue::IDENTITY,
        ParallelMatmulConfig::default(),
        None,
    );
}

fn trans_a_direct_disabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var_os("YSCV_TRANS_A_DIRECT_OFF").is_some())
}

fn ftmm_4row_disabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var_os("YSCV_FTMM_4ROW_OFF").is_some())
}

fn matmul_2d_slices_trans_a_direct(
    a_kt: &[f32],
    m: usize,
    k: usize,
    b: &[f32],
    n: usize,
    out: &mut [f32],
) {
    // 4-row × NR=16 outer-product tile: keeps 16 accumulator ZMMs in flight per
    // inner-K iteration (vs 4 for the 1-row kernel), saturating Zen 4's
    // double-pumped ZMM-FMA pipe (~8-cycle effective latency).
    //
    // Kill switch: `YSCV_FTMM_4ROW_OFF=1` reverts to the single-row dispatch.
    // M%4 tail rows fall back to the existing 1-row kernel unchanged.
    let m4 = m & !3usize;
    if !ftmm_4row_disabled() && m4 > 0 && trans_a_4row_supported(n) {
        let (head, tail) = out.split_at_mut(m4 * n);
        head.par_chunks_mut(4 * n)
            .enumerate()
            .for_each(|(group_idx, chunk_4rows)| {
                let mi_base = group_idx * 4;
                trans_a_4row_dispatch(a_kt, mi_base, m, k, b, n, chunk_4rows);
            });
        if m4 < m {
            tail.par_chunks_mut(n)
                .enumerate()
                .for_each(|(off, out_row)| {
                    trans_a_row_set_dispatch(a_kt, m4 + off, m, k, b, n, out_row);
                });
        }
        return;
    }
    out.par_chunks_mut(n)
        .enumerate()
        .for_each(|(mi, out_row)| trans_a_row_set_dispatch(a_kt, mi, m, k, b, n, out_row));
}

pub(super) fn trans_a_4row_supported(n: usize) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if !cfg!(miri) && n.is_multiple_of(16) && crate::host_cpu().features.avx512f {
            return true;
        }
        if !cfg!(miri)
            && n.is_multiple_of(8)
            && crate::host_cpu().features.avx
            && crate::host_cpu().features.fma
        {
            return true;
        }
    }
    #[cfg(target_arch = "x86")]
    {
        if !cfg!(miri)
            && n.is_multiple_of(8)
            && crate::host_cpu().features.avx
            && crate::host_cpu().features.fma
        {
            return true;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if !cfg!(miri) && n.is_multiple_of(4) && crate::host_cpu().features.neon {
            return true;
        }
    }
    let _ = n;
    false
}

fn trans_a_4row_dispatch(
    a_kt: &[f32],
    mi_base: usize,
    m: usize,
    k: usize,
    b: &[f32],
    n: usize,
    out_4rows: &mut [f32],
) {
    debug_assert_eq!(out_4rows.len(), 4 * n);
    #[cfg(target_arch = "x86_64")]
    {
        if !cfg!(miri) && n.is_multiple_of(16) && crate::host_cpu().features.avx512f {
            #[allow(unsafe_code)]
            unsafe {
                trans_a_4row_avx512(a_kt, mi_base, m, k, b, n, out_4rows);
            }
            return;
        }
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if !cfg!(miri)
            && n.is_multiple_of(8)
            && crate::host_cpu().features.avx
            && crate::host_cpu().features.fma
        {
            #[allow(unsafe_code)]
            unsafe {
                trans_a_4row_avx2(a_kt, mi_base, m, k, b, n, out_4rows);
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if !cfg!(miri) && n.is_multiple_of(4) && crate::host_cpu().features.neon {
            #[allow(unsafe_code)]
            unsafe {
                trans_a_4row_neon(a_kt, mi_base, m, k, b, n, out_4rows);
            }
            return;
        }
    }
    // Scalar fallback: per-row 1-row kernel.
    for (row_off, out_row) in out_4rows.chunks_exact_mut(n).enumerate() {
        trans_a_row_set_scalar(a_kt, mi_base + row_off, m, k, b, n, out_row);
    }
}

fn trans_a_row_set_dispatch(
    a_kt: &[f32],
    mi: usize,
    m: usize,
    k: usize,
    b: &[f32],
    n: usize,
    out_row: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if !cfg!(miri) && n.is_multiple_of(16) && crate::host_cpu().features.avx512f {
            #[allow(unsafe_code)]
            unsafe {
                trans_a_row_set_avx512(a_kt, mi, m, k, b, n, out_row);
            }
            return;
        }
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if !cfg!(miri)
            && n.is_multiple_of(8)
            && crate::host_cpu().features.avx
            && crate::host_cpu().features.fma
        {
            #[allow(unsafe_code)]
            unsafe {
                trans_a_row_set_avx2(a_kt, mi, m, k, b, n, out_row);
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if !cfg!(miri) && n.is_multiple_of(4) && crate::host_cpu().features.neon {
            #[allow(unsafe_code)]
            unsafe {
                trans_a_row_set_neon(a_kt, mi, m, k, b, n, out_row);
            }
            return;
        }
    }
    trans_a_row_set_scalar(a_kt, mi, m, k, b, n, out_row);
}

fn trans_a_row_set_scalar(
    a_kt: &[f32],
    mi: usize,
    m: usize,
    k: usize,
    b: &[f32],
    n: usize,
    out_row: &mut [f32],
) {
    out_row.fill(0.0);
    for ki in 0..k {
        let a = a_kt[ki * m + mi];
        let b_row = &b[ki * n..(ki + 1) * n];
        for j in 0..n {
            out_row[j] += a * b_row[j];
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_code)]
unsafe fn trans_a_row_set_avx512(
    a_kt: &[f32],
    mi: usize,
    m: usize,
    k: usize,
    b: &[f32],
    n: usize,
    out_row: &mut [f32],
) {
    use std::arch::x86_64::*;
    unsafe {
        let blocks = n / 64;
        let tail_chunks = (n % 64) / 16;
        let zero = _mm512_setzero_ps();
        for block in 0..blocks {
            let col = block * 64;
            let mut c0 = zero;
            let mut c1 = zero;
            let mut c2 = zero;
            let mut c3 = zero;
            for ki in 0..k {
                let a = _mm512_set1_ps(*a_kt.as_ptr().add(ki * m + mi));
                let bptr = b.as_ptr().add(ki * n + col);
                c0 = _mm512_fmadd_ps(a, _mm512_loadu_ps(bptr), c0);
                c1 = _mm512_fmadd_ps(a, _mm512_loadu_ps(bptr.add(16)), c1);
                c2 = _mm512_fmadd_ps(a, _mm512_loadu_ps(bptr.add(32)), c2);
                c3 = _mm512_fmadd_ps(a, _mm512_loadu_ps(bptr.add(48)), c3);
            }
            let op = out_row.as_mut_ptr().add(col);
            _mm512_storeu_ps(op, c0);
            _mm512_storeu_ps(op.add(16), c1);
            _mm512_storeu_ps(op.add(32), c2);
            _mm512_storeu_ps(op.add(48), c3);
        }
        let tail_base = blocks * 64;
        for chunk in 0..tail_chunks {
            let col = tail_base + chunk * 16;
            let mut c = zero;
            for ki in 0..k {
                let a = _mm512_set1_ps(*a_kt.as_ptr().add(ki * m + mi));
                c = _mm512_fmadd_ps(a, _mm512_loadu_ps(b.as_ptr().add(ki * n + col)), c);
            }
            _mm512_storeu_ps(out_row.as_mut_ptr().add(col), c);
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
#[allow(unsafe_code)]
unsafe fn trans_a_row_set_avx2(
    a_kt: &[f32],
    mi: usize,
    m: usize,
    k: usize,
    b: &[f32],
    n: usize,
    out_row: &mut [f32],
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    unsafe {
        let blocks = n / 32;
        let tail_chunks = (n % 32) / 8;
        let zero = _mm256_setzero_ps();
        for block in 0..blocks {
            let col = block * 32;
            let mut c0 = zero;
            let mut c1 = zero;
            let mut c2 = zero;
            let mut c3 = zero;
            for ki in 0..k {
                let a = _mm256_set1_ps(*a_kt.as_ptr().add(ki * m + mi));
                let bptr = b.as_ptr().add(ki * n + col);
                c0 = _mm256_fmadd_ps(a, _mm256_loadu_ps(bptr), c0);
                c1 = _mm256_fmadd_ps(a, _mm256_loadu_ps(bptr.add(8)), c1);
                c2 = _mm256_fmadd_ps(a, _mm256_loadu_ps(bptr.add(16)), c2);
                c3 = _mm256_fmadd_ps(a, _mm256_loadu_ps(bptr.add(24)), c3);
            }
            let op = out_row.as_mut_ptr().add(col);
            _mm256_storeu_ps(op, c0);
            _mm256_storeu_ps(op.add(8), c1);
            _mm256_storeu_ps(op.add(16), c2);
            _mm256_storeu_ps(op.add(24), c3);
        }
        let tail_base = blocks * 32;
        for chunk in 0..tail_chunks {
            let col = tail_base + chunk * 8;
            let mut c = zero;
            for ki in 0..k {
                let a = _mm256_set1_ps(*a_kt.as_ptr().add(ki * m + mi));
                c = _mm256_fmadd_ps(a, _mm256_loadu_ps(b.as_ptr().add(ki * n + col)), c);
            }
            _mm256_storeu_ps(out_row.as_mut_ptr().add(col), c);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
unsafe fn trans_a_row_set_neon(
    a_kt: &[f32],
    mi: usize,
    m: usize,
    k: usize,
    b: &[f32],
    n: usize,
    out_row: &mut [f32],
) {
    unsafe {
        let blocks = n / 16;
        let tail_chunks = (n % 16) / 4;
        let zero = vdupq_n_f32(0.0);
        for block in 0..blocks {
            let col = block * 16;
            let mut c0 = zero;
            let mut c1 = zero;
            let mut c2 = zero;
            let mut c3 = zero;
            for ki in 0..k {
                let a = vdupq_n_f32(*a_kt.as_ptr().add(ki * m + mi));
                let bptr = b.as_ptr().add(ki * n + col);
                c0 = vfmaq_f32(c0, a, vld1q_f32(bptr));
                c1 = vfmaq_f32(c1, a, vld1q_f32(bptr.add(4)));
                c2 = vfmaq_f32(c2, a, vld1q_f32(bptr.add(8)));
                c3 = vfmaq_f32(c3, a, vld1q_f32(bptr.add(12)));
            }
            let op = out_row.as_mut_ptr().add(col);
            vst1q_f32(op, c0);
            vst1q_f32(op.add(4), c1);
            vst1q_f32(op.add(8), c2);
            vst1q_f32(op.add(12), c3);
        }
        let tail_base = blocks * 16;
        for chunk in 0..tail_chunks {
            let col = tail_base + chunk * 4;
            let mut c = zero;
            for ki in 0..k {
                let a = vdupq_n_f32(*a_kt.as_ptr().add(ki * m + mi));
                c = vfmaq_f32(c, a, vld1q_f32(b.as_ptr().add(ki * n + col)));
            }
            vst1q_f32(out_row.as_mut_ptr().add(col), c);
        }
    }
}

// ---------------------------------------------------------------------------
// 4-row × NR=16 outer-product tile for FTMM (Transpose+MatMul).
//
// Rationale: the 1-row kernel above uses 4 acc ZMMs per NR=16 panel. Inside
// the inner K loop only ONE accumulator is being updated per FMA — Zen 4's
// double-pumped ZMM-FMA pipe (effective latency ~8 cycles) needs ≥16
// independent in-flight chains to saturate at 2 FMA/cyc peak. The 1-row
// kernel runs at ~0.5 FMA/cyc (matches measured 4.7× over FMA floor).
//
// The 4-row variant updates 16 acc ZMMs per inner K iteration (4 rows × 4
// NR=16 panels) — every accumulator is touched exactly once per K-step, so
// each chain sees a 16-FMA gap between consecutive updates, hiding the
// 8-cycle latency. The 4-row × NR=64 super-block targets N ≥ 64 (tracker
// FTMM shape); N % 64 tail handled in NR=16 chunks (still 4 rows, 4 acc).
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_code)]
unsafe fn trans_a_4row_avx512(
    a_kt: &[f32],
    mi_base: usize,
    m: usize,
    k: usize,
    b: &[f32],
    n: usize,
    out_4rows: &mut [f32],
) {
    use std::arch::x86_64::*;
    unsafe {
        let r0 = out_4rows.as_mut_ptr();
        let r1 = r0.add(n);
        let r2 = r0.add(2 * n);
        let r3 = r0.add(3 * n);
        let blocks = n / 64;
        let tail_chunks = (n % 64) / 16;
        let zero = _mm512_setzero_ps();
        // 4×4 outer-product super-block: 16 acc ZMMs in flight.
        for block in 0..blocks {
            let col = block * 64;
            let mut c00 = zero;
            let mut c01 = zero;
            let mut c02 = zero;
            let mut c03 = zero;
            let mut c10 = zero;
            let mut c11 = zero;
            let mut c12 = zero;
            let mut c13 = zero;
            let mut c20 = zero;
            let mut c21 = zero;
            let mut c22 = zero;
            let mut c23 = zero;
            let mut c30 = zero;
            let mut c31 = zero;
            let mut c32 = zero;
            let mut c33 = zero;
            for ki in 0..k {
                let a_row = a_kt.as_ptr().add(ki * m + mi_base);
                let a0 = _mm512_set1_ps(*a_row);
                let a1 = _mm512_set1_ps(*a_row.add(1));
                let a2 = _mm512_set1_ps(*a_row.add(2));
                let a3 = _mm512_set1_ps(*a_row.add(3));
                let bptr = b.as_ptr().add(ki * n + col);
                let b0 = _mm512_loadu_ps(bptr);
                let b1 = _mm512_loadu_ps(bptr.add(16));
                let b2 = _mm512_loadu_ps(bptr.add(32));
                let b3 = _mm512_loadu_ps(bptr.add(48));
                c00 = _mm512_fmadd_ps(a0, b0, c00);
                c01 = _mm512_fmadd_ps(a0, b1, c01);
                c02 = _mm512_fmadd_ps(a0, b2, c02);
                c03 = _mm512_fmadd_ps(a0, b3, c03);
                c10 = _mm512_fmadd_ps(a1, b0, c10);
                c11 = _mm512_fmadd_ps(a1, b1, c11);
                c12 = _mm512_fmadd_ps(a1, b2, c12);
                c13 = _mm512_fmadd_ps(a1, b3, c13);
                c20 = _mm512_fmadd_ps(a2, b0, c20);
                c21 = _mm512_fmadd_ps(a2, b1, c21);
                c22 = _mm512_fmadd_ps(a2, b2, c22);
                c23 = _mm512_fmadd_ps(a2, b3, c23);
                c30 = _mm512_fmadd_ps(a3, b0, c30);
                c31 = _mm512_fmadd_ps(a3, b1, c31);
                c32 = _mm512_fmadd_ps(a3, b2, c32);
                c33 = _mm512_fmadd_ps(a3, b3, c33);
            }
            _mm512_storeu_ps(r0.add(col), c00);
            _mm512_storeu_ps(r0.add(col + 16), c01);
            _mm512_storeu_ps(r0.add(col + 32), c02);
            _mm512_storeu_ps(r0.add(col + 48), c03);
            _mm512_storeu_ps(r1.add(col), c10);
            _mm512_storeu_ps(r1.add(col + 16), c11);
            _mm512_storeu_ps(r1.add(col + 32), c12);
            _mm512_storeu_ps(r1.add(col + 48), c13);
            _mm512_storeu_ps(r2.add(col), c20);
            _mm512_storeu_ps(r2.add(col + 16), c21);
            _mm512_storeu_ps(r2.add(col + 32), c22);
            _mm512_storeu_ps(r2.add(col + 48), c23);
            _mm512_storeu_ps(r3.add(col), c30);
            _mm512_storeu_ps(r3.add(col + 16), c31);
            _mm512_storeu_ps(r3.add(col + 32), c32);
            _mm512_storeu_ps(r3.add(col + 48), c33);
        }
        // Tail: N%64 in NR=16 chunks (4 acc ZMMs per chunk).
        let tail_base = blocks * 64;
        for chunk in 0..tail_chunks {
            let col = tail_base + chunk * 16;
            let mut c0 = zero;
            let mut c1 = zero;
            let mut c2 = zero;
            let mut c3 = zero;
            for ki in 0..k {
                let a_row = a_kt.as_ptr().add(ki * m + mi_base);
                let a0 = _mm512_set1_ps(*a_row);
                let a1 = _mm512_set1_ps(*a_row.add(1));
                let a2 = _mm512_set1_ps(*a_row.add(2));
                let a3 = _mm512_set1_ps(*a_row.add(3));
                let bv = _mm512_loadu_ps(b.as_ptr().add(ki * n + col));
                c0 = _mm512_fmadd_ps(a0, bv, c0);
                c1 = _mm512_fmadd_ps(a1, bv, c1);
                c2 = _mm512_fmadd_ps(a2, bv, c2);
                c3 = _mm512_fmadd_ps(a3, bv, c3);
            }
            _mm512_storeu_ps(r0.add(col), c0);
            _mm512_storeu_ps(r1.add(col), c1);
            _mm512_storeu_ps(r2.add(col), c2);
            _mm512_storeu_ps(r3.add(col), c3);
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
#[allow(unsafe_code)]
unsafe fn trans_a_4row_avx2(
    a_kt: &[f32],
    mi_base: usize,
    m: usize,
    k: usize,
    b: &[f32],
    n: usize,
    out_4rows: &mut [f32],
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    unsafe {
        let r0 = out_4rows.as_mut_ptr();
        let r1 = r0.add(n);
        let r2 = r0.add(2 * n);
        let r3 = r0.add(3 * n);
        // AVX2 has 16 YMM regs: 4 rows × 2 NR=8 panels = 8 acc YMMs + 2 b
        // panels + 4 a broadcasts = 14 YMM. N % 16 super-block, then NR=8 tail.
        let blocks = n / 16;
        let tail_chunks = (n % 16) / 8;
        let zero = _mm256_setzero_ps();
        for block in 0..blocks {
            let col = block * 16;
            let mut c00 = zero;
            let mut c01 = zero;
            let mut c10 = zero;
            let mut c11 = zero;
            let mut c20 = zero;
            let mut c21 = zero;
            let mut c30 = zero;
            let mut c31 = zero;
            for ki in 0..k {
                let a_row = a_kt.as_ptr().add(ki * m + mi_base);
                let a0 = _mm256_set1_ps(*a_row);
                let a1 = _mm256_set1_ps(*a_row.add(1));
                let a2 = _mm256_set1_ps(*a_row.add(2));
                let a3 = _mm256_set1_ps(*a_row.add(3));
                let bptr = b.as_ptr().add(ki * n + col);
                let b0 = _mm256_loadu_ps(bptr);
                let b1 = _mm256_loadu_ps(bptr.add(8));
                c00 = _mm256_fmadd_ps(a0, b0, c00);
                c01 = _mm256_fmadd_ps(a0, b1, c01);
                c10 = _mm256_fmadd_ps(a1, b0, c10);
                c11 = _mm256_fmadd_ps(a1, b1, c11);
                c20 = _mm256_fmadd_ps(a2, b0, c20);
                c21 = _mm256_fmadd_ps(a2, b1, c21);
                c30 = _mm256_fmadd_ps(a3, b0, c30);
                c31 = _mm256_fmadd_ps(a3, b1, c31);
            }
            _mm256_storeu_ps(r0.add(col), c00);
            _mm256_storeu_ps(r0.add(col + 8), c01);
            _mm256_storeu_ps(r1.add(col), c10);
            _mm256_storeu_ps(r1.add(col + 8), c11);
            _mm256_storeu_ps(r2.add(col), c20);
            _mm256_storeu_ps(r2.add(col + 8), c21);
            _mm256_storeu_ps(r3.add(col), c30);
            _mm256_storeu_ps(r3.add(col + 8), c31);
        }
        let tail_base = blocks * 16;
        for chunk in 0..tail_chunks {
            let col = tail_base + chunk * 8;
            let mut c0 = zero;
            let mut c1 = zero;
            let mut c2 = zero;
            let mut c3 = zero;
            for ki in 0..k {
                let a_row = a_kt.as_ptr().add(ki * m + mi_base);
                let a0 = _mm256_set1_ps(*a_row);
                let a1 = _mm256_set1_ps(*a_row.add(1));
                let a2 = _mm256_set1_ps(*a_row.add(2));
                let a3 = _mm256_set1_ps(*a_row.add(3));
                let bv = _mm256_loadu_ps(b.as_ptr().add(ki * n + col));
                c0 = _mm256_fmadd_ps(a0, bv, c0);
                c1 = _mm256_fmadd_ps(a1, bv, c1);
                c2 = _mm256_fmadd_ps(a2, bv, c2);
                c3 = _mm256_fmadd_ps(a3, bv, c3);
            }
            _mm256_storeu_ps(r0.add(col), c0);
            _mm256_storeu_ps(r1.add(col), c1);
            _mm256_storeu_ps(r2.add(col), c2);
            _mm256_storeu_ps(r3.add(col), c3);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
unsafe fn trans_a_4row_neon(
    a_kt: &[f32],
    mi_base: usize,
    m: usize,
    k: usize,
    b: &[f32],
    n: usize,
    out_4rows: &mut [f32],
) {
    unsafe {
        let r0 = out_4rows.as_mut_ptr();
        let r1 = r0.add(n);
        let r2 = r0.add(2 * n);
        let r3 = r0.add(3 * n);
        // NEON has 32 q-regs: 4 rows × 4 NR=4 panels = 16 acc + 4 b + 4 a = 24.
        let blocks = n / 16;
        let tail_chunks = (n % 16) / 4;
        let zero = vdupq_n_f32(0.0);
        for block in 0..blocks {
            let col = block * 16;
            let mut c00 = zero;
            let mut c01 = zero;
            let mut c02 = zero;
            let mut c03 = zero;
            let mut c10 = zero;
            let mut c11 = zero;
            let mut c12 = zero;
            let mut c13 = zero;
            let mut c20 = zero;
            let mut c21 = zero;
            let mut c22 = zero;
            let mut c23 = zero;
            let mut c30 = zero;
            let mut c31 = zero;
            let mut c32 = zero;
            let mut c33 = zero;
            for ki in 0..k {
                let a_row = a_kt.as_ptr().add(ki * m + mi_base);
                let a0 = vdupq_n_f32(*a_row);
                let a1 = vdupq_n_f32(*a_row.add(1));
                let a2 = vdupq_n_f32(*a_row.add(2));
                let a3 = vdupq_n_f32(*a_row.add(3));
                let bptr = b.as_ptr().add(ki * n + col);
                let b0 = vld1q_f32(bptr);
                let b1 = vld1q_f32(bptr.add(4));
                let b2 = vld1q_f32(bptr.add(8));
                let b3 = vld1q_f32(bptr.add(12));
                c00 = vfmaq_f32(c00, a0, b0);
                c01 = vfmaq_f32(c01, a0, b1);
                c02 = vfmaq_f32(c02, a0, b2);
                c03 = vfmaq_f32(c03, a0, b3);
                c10 = vfmaq_f32(c10, a1, b0);
                c11 = vfmaq_f32(c11, a1, b1);
                c12 = vfmaq_f32(c12, a1, b2);
                c13 = vfmaq_f32(c13, a1, b3);
                c20 = vfmaq_f32(c20, a2, b0);
                c21 = vfmaq_f32(c21, a2, b1);
                c22 = vfmaq_f32(c22, a2, b2);
                c23 = vfmaq_f32(c23, a2, b3);
                c30 = vfmaq_f32(c30, a3, b0);
                c31 = vfmaq_f32(c31, a3, b1);
                c32 = vfmaq_f32(c32, a3, b2);
                c33 = vfmaq_f32(c33, a3, b3);
            }
            vst1q_f32(r0.add(col), c00);
            vst1q_f32(r0.add(col + 4), c01);
            vst1q_f32(r0.add(col + 8), c02);
            vst1q_f32(r0.add(col + 12), c03);
            vst1q_f32(r1.add(col), c10);
            vst1q_f32(r1.add(col + 4), c11);
            vst1q_f32(r1.add(col + 8), c12);
            vst1q_f32(r1.add(col + 12), c13);
            vst1q_f32(r2.add(col), c20);
            vst1q_f32(r2.add(col + 4), c21);
            vst1q_f32(r2.add(col + 8), c22);
            vst1q_f32(r2.add(col + 12), c23);
            vst1q_f32(r3.add(col), c30);
            vst1q_f32(r3.add(col + 4), c31);
            vst1q_f32(r3.add(col + 8), c32);
            vst1q_f32(r3.add(col + 12), c33);
        }
        let tail_base = blocks * 16;
        for chunk in 0..tail_chunks {
            let col = tail_base + chunk * 4;
            let mut c0 = zero;
            let mut c1 = zero;
            let mut c2 = zero;
            let mut c3 = zero;
            for ki in 0..k {
                let a_row = a_kt.as_ptr().add(ki * m + mi_base);
                let a0 = vdupq_n_f32(*a_row);
                let a1 = vdupq_n_f32(*a_row.add(1));
                let a2 = vdupq_n_f32(*a_row.add(2));
                let a3 = vdupq_n_f32(*a_row.add(3));
                let bv = vld1q_f32(b.as_ptr().add(ki * n + col));
                c0 = vfmaq_f32(c0, a0, bv);
                c1 = vfmaq_f32(c1, a1, bv);
                c2 = vfmaq_f32(c2, a2, bv);
                c3 = vfmaq_f32(c3, a3, bv);
            }
            vst1q_f32(r0.add(col), c0);
            vst1q_f32(r1.add(col), c1);
            vst1q_f32(r2.add(col), c2);
            vst1q_f32(r3.add(col), c3);
        }
    }
}

#[cfg(test)]
mod trans_a_tests {
    use super::*;

    fn ref_matmul(a_mk: &[f32], b_kn: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..k {
                    s += a_mk[i * k + p] * b_kn[p * n + j];
                }
                out[i * n + j] = s;
            }
        }
        out
    }

    /// Feed `matmul_2d_slices_trans_a` a (K, M)-laid-out A matrix and
    /// check the output matches the reference where A is transposed to
    /// (M, K) first. Covers the tracker shapes the runner dispatches on.
    fn check_shape(m: usize, k: usize, n: usize) {
        let mut a_mk = vec![0.0f32; m * k];
        for (i, v) in a_mk.iter_mut().enumerate() {
            *v = ((i % 97) as f32) * 0.013 - 0.5;
        }
        // Build the (K, M) layout that the runner hands in.
        let mut a_kt = vec![0.0f32; k * m];
        for mi in 0..m {
            for ki in 0..k {
                a_kt[ki * m + mi] = a_mk[mi * k + ki];
            }
        }
        let mut b = vec![0.0f32; k * n];
        for (i, v) in b.iter_mut().enumerate() {
            *v = ((i % 113) as f32) * 0.009 + 0.25;
        }
        let expected = ref_matmul(&a_mk, &b, m, k, n);
        let mut out = vec![0.0f32; m * n];
        matmul_2d_slices_trans_a(&a_kt, m, k, &b, n, &mut out);
        for (idx, (&g, &e)) in out.iter().zip(expected.iter()).enumerate() {
            let tol = 1e-3 * e.abs().max(1.0);
            assert!(
                (g - e).abs() <= tol,
                "shape m={m} k={k} n={n} idx={idx}: got={g} expected={e}"
            );
        }
    }

    #[test]
    fn trans_a_small_square() {
        check_shape(4, 8, 6);
    }

    #[test]
    fn trans_a_tracker_cls_dw_shape() {
        // Tracker `cls_dw/MatMul` shape: A:[1,32,64]→[1,64,32], B:[1,32,64]
        // post-transpose the fused kernel sees m=64, k=32, n=64.
        check_shape(64, 32, 64);
    }

    #[test]
    fn trans_a_tall_thin() {
        check_shape(128, 96, 4);
    }

    #[test]
    fn trans_a_wide_short() {
        check_shape(4, 96, 128);
    }

    // Coverage: 4-row × NR=16 outer-product kernel. Hits the
    // 4-aligned super-block, the M%4 tail dispatch, and N values that
    // exercise both the NR=64 super-block and NR=16 tail chunks inside
    // the AVX-512 path. Tracker FTMM shape is m=64/k=32/n=64; nearby
    // larger M and odd-M cases verify dispatch boundaries.

    #[test]
    fn trans_a_4row_m80_n64() {
        // M aligned to 4, N % 64 == 0 — pure 4-row super-block path.
        check_shape(80, 32, 64);
    }

    #[test]
    fn trans_a_4row_m96_n80() {
        // N = 80 = 64 + 16 → exercises the NR=16 tail chunk in 4-row kernel.
        check_shape(96, 32, 80);
    }

    #[test]
    fn trans_a_4row_m112_n112() {
        // N = 112 = 64 + 48 → 3 tail chunks at NR=16.
        check_shape(112, 32, 112);
    }

    #[test]
    fn trans_a_4row_tail_m63() {
        // M = 63 → m4 = 60 (15 groups of 4) + 3 tail rows via 1-row kernel.
        check_shape(63, 32, 64);
    }

    #[test]
    fn trans_a_4row_tail_m65() {
        // M = 65 → m4 = 64 + 1 tail row.
        check_shape(65, 32, 64);
    }

    #[test]
    fn trans_a_4row_tail_m66() {
        // M = 66 → m4 = 64 + 2 tail rows.
        check_shape(66, 32, 64);
    }

    /// Compare the 4-row path against the 1-row fallback on the tracker
    /// FTMM shape. Catches any drift between the two implementations
    /// (FP summation order differs slightly, but the kernel-level result
    /// must be within a few ULP for the same K accumulation pattern).
    #[test]
    fn trans_a_4row_matches_1row_baseline() {
        let (m, k, n) = (64usize, 32usize, 64usize);
        let mut a_kt = vec![0.0f32; k * m];
        for (i, v) in a_kt.iter_mut().enumerate() {
            *v = ((i % 137) as f32) * 0.011 - 0.5;
        }
        let mut b = vec![0.0f32; k * n];
        for (i, v) in b.iter_mut().enumerate() {
            *v = ((i % 211) as f32) * 0.007 + 0.125;
        }

        // 4-row path (default).
        let mut out_4row = vec![0.0f32; m * n];
        matmul_2d_slices_trans_a(&a_kt, m, k, &b, n, &mut out_4row);

        // 1-row path via env kill switch. SAFETY: env access in tests is
        // single-threaded; OnceLock cache is hit before this test runs in
        // the broader suite if needed.
        // SAFETY: tests run in process-local env, and `trans_a_direct_disabled`
        // / `ftmm_4row_disabled` cache through OnceLock. To get a clean A/B,
        // call the lower path explicitly without env-flipping.
        let mut out_1row = vec![0.0f32; m * n];
        out_1row
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(mi, out_row)| {
                trans_a_row_set_dispatch(&a_kt, mi, m, k, &b, n, out_row);
            });

        for (idx, (&g, &e)) in out_4row.iter().zip(out_1row.iter()).enumerate() {
            // FP summation order matches between both kernels (same K-step
            // sequence; only the outer-row loop differs). Identity expected.
            assert!(
                (g - e).abs() <= 5e-4 * e.abs().max(1.0),
                "idx={idx} 4row={g} 1row={e}"
            );
        }
    }
}
