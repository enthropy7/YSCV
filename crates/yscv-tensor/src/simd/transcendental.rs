//! Transcendental functions: exp, sin, cos, tan, ln with SIMD dispatch.

use super::*;

// ===========================================================================
// Exp: out[i] = exp(data[i]) -- SIMD polynomial approximation
// ===========================================================================

/// Compute exp(data) into `out` using SIMD polynomial approximation where
/// available, falling back to scalar `f32::exp` otherwise.
#[allow(unsafe_code, unreachable_code)]
#[inline]
pub(crate) fn exp_dispatch(data: &[f32], out: &mut [f32]) {
    debug_assert_eq!(data.len(), out.len());

    if cfg!(miri) {
        exp_scalar(data, out);
        return;
    }

    // macOS aarch64: use Apple Accelerate vvexpf.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let count = data.len() as i32;
        unsafe {
            vvexpf(out.as_mut_ptr(), data.as_ptr(), &count);
        }
        return;
    }

    // x86/x86_64 with MKL: use Intel VML vsExp (heavily optimized).
    #[cfg(all(feature = "mkl", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        let count = data.len() as i32;
        // SAFETY: vsExp reads `count` floats from `data` and writes to `out`.
        unsafe { vsExp(count, data.as_ptr(), out.as_mut_ptr()) };
        return;
    }

    // aarch64 Linux with ARMPL: use ARM Performance Libraries vectorized exp.
    #[cfg(all(feature = "armpl", target_arch = "aarch64", not(target_os = "macos")))]
    {
        let count = data.len() as i32;
        // SAFETY: armpl_svexp_f32 reads `count` floats from `data` and writes to `out`.
        unsafe { armpl_svexp_f32(count, data.as_ptr(), out.as_mut_ptr()) };
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if yscv_cpu::host_cpu().features.avx {
            unsafe { exp_avx(data, out) };
            return;
        }
        if yscv_cpu::host_cpu().features.sse {
            unsafe { exp_sse(data, out) };
            return;
        }
    }

    #[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
    {
        if yscv_cpu::host_cpu().features.neon {
            unsafe { exp_neon(data, out) };
            return;
        }
    }

    exp_scalar(data, out);
}

/// Kept for backward compatibility -- calls `exp_dispatch` with in-place semantics.
#[allow(unsafe_code, dead_code)]
#[inline]
pub(crate) fn exp_inplace_dispatch(data: &mut [f32]) {
    if cfg!(miri) {
        for v in data.iter_mut() {
            *v = v.exp();
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if yscv_cpu::host_cpu().features.avx {
            // SAFETY: input and output alias, but each element is read
            // once then written, so there's no ordering hazard.
            unsafe {
                let ptr = data.as_ptr();
                let len = data.len();
                let slice = std::slice::from_raw_parts(ptr, len);
                exp_avx(slice, data);
            };
            return;
        }
        if yscv_cpu::host_cpu().features.sse {
            unsafe {
                let ptr = data.as_ptr();
                let len = data.len();
                let slice = std::slice::from_raw_parts(ptr, len);
                exp_sse(slice, data);
            };
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if yscv_cpu::host_cpu().features.neon {
            unsafe {
                let ptr = data.as_ptr();
                let len = data.len();
                let slice = std::slice::from_raw_parts(ptr, len);
                exp_neon(slice, data);
            };
            return;
        }
    }

    for v in data.iter_mut() {
        *v = v.exp();
    }
}

fn exp_scalar(data: &[f32], out: &mut [f32]) {
    for i in 0..data.len() {
        out[i] = data[i].exp();
    }
}

// -- SSE fast-exp (4-wide) --

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn fast_exp_sse(x: __m128) -> __m128 {
    let ln2_inv = _mm_set1_ps(std::f32::consts::LOG2_E);
    let ln2_hi = _mm_set1_ps(0.693_359_4);
    let ln2_lo = _mm_set1_ps(-2.121_944_4e-4);

    let c0 = _mm_set1_ps(1.0);
    let c1 = _mm_set1_ps(1.0);
    let c2 = _mm_set1_ps(0.5);
    let c3 = _mm_set1_ps(1.0 / 6.0);
    let c4 = _mm_set1_ps(1.0 / 24.0);
    let c5 = _mm_set1_ps(1.0 / 120.0);
    let c6 = _mm_set1_ps(1.0 / 720.0);

    let x = _mm_max_ps(_mm_set1_ps(-88.0), _mm_min_ps(_mm_set1_ps(88.0), x));

    let n_f = _mm_mul_ps(x, ln2_inv);
    let n_i = _mm_cvtps_epi32(n_f);
    let n_f = _mm_cvtepi32_ps(n_i);

    let r = _mm_sub_ps(
        _mm_sub_ps(x, _mm_mul_ps(n_f, ln2_hi)),
        _mm_mul_ps(n_f, ln2_lo),
    );

    let mut poly = _mm_add_ps(c5, _mm_mul_ps(r, c6));
    poly = _mm_add_ps(c4, _mm_mul_ps(r, poly));
    poly = _mm_add_ps(c3, _mm_mul_ps(r, poly));
    poly = _mm_add_ps(c2, _mm_mul_ps(r, poly));
    poly = _mm_add_ps(c1, _mm_mul_ps(r, poly));
    poly = _mm_add_ps(c0, _mm_mul_ps(r, poly));

    let pow2n = {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::{_mm_add_epi32, _mm_slli_epi32};
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::{_mm_add_epi32, _mm_slli_epi32};
        let bias = _mm_set1_epi32(127);
        _mm_castsi128_ps(_mm_slli_epi32(_mm_add_epi32(n_i, bias), 23))
    };

    _mm_mul_ps(poly, pow2n)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn exp_sse(data: &[f32], out: &mut [f32]) {
    let len = data.len();
    let inp = data.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = 0usize;

    while i + 4 <= len {
        let v = _mm_loadu_ps(inp.add(i));
        _mm_storeu_ps(op.add(i), fast_exp_sse(v));
        i += 4;
    }

    while i < len {
        *op.add(i) = (*inp.add(i)).exp();
        i += 1;
    }
}

// -- AVX fast-exp (8-wide) --

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn fast_exp_avx(x: __m256) -> __m256 {
    let ln2_inv = _mm256_set1_ps(std::f32::consts::LOG2_E);
    let ln2_hi = _mm256_set1_ps(0.693_359_4);
    let ln2_lo = _mm256_set1_ps(-2.121_944_4e-4);

    let c0 = _mm256_set1_ps(1.0);
    let c1 = _mm256_set1_ps(1.0);
    let c2 = _mm256_set1_ps(0.5);
    let c3 = _mm256_set1_ps(1.0 / 6.0);
    let c4 = _mm256_set1_ps(1.0 / 24.0);
    let c5 = _mm256_set1_ps(1.0 / 120.0);
    let c6 = _mm256_set1_ps(1.0 / 720.0);

    let x = _mm256_max_ps(
        _mm256_set1_ps(-88.0),
        _mm256_min_ps(_mm256_set1_ps(88.0), x),
    );

    let n_f = _mm256_mul_ps(x, ln2_inv);
    let n_i = _mm256_cvtps_epi32(n_f);
    let n_f = _mm256_cvtepi32_ps(n_i);

    let r = _mm256_sub_ps(
        _mm256_sub_ps(x, _mm256_mul_ps(n_f, ln2_hi)),
        _mm256_mul_ps(n_f, ln2_lo),
    );

    let mut poly = _mm256_add_ps(c5, _mm256_mul_ps(r, c6));
    poly = _mm256_add_ps(c4, _mm256_mul_ps(r, poly));
    poly = _mm256_add_ps(c3, _mm256_mul_ps(r, poly));
    poly = _mm256_add_ps(c2, _mm256_mul_ps(r, poly));
    poly = _mm256_add_ps(c1, _mm256_mul_ps(r, poly));
    poly = _mm256_add_ps(c0, _mm256_mul_ps(r, poly));

    let bias = _mm256_set1_epi32(127);
    let pow2n = {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::{_mm256_add_epi32, _mm256_slli_epi32};
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::{_mm256_add_epi32, _mm256_slli_epi32};
        _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(n_i, bias), 23))
    };

    _mm256_mul_ps(poly, pow2n)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn exp_avx(data: &[f32], out: &mut [f32]) {
    let len = data.len();
    let inp = data.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = 0usize;

    while i + 8 <= len {
        let v = _mm256_loadu_ps(inp.add(i));
        _mm256_storeu_ps(op.add(i), fast_exp_avx(v));
        i += 8;
    }

    if i < len {
        exp_sse(&data[i..], &mut out[i..]);
    }
}

// -- NEON fast-exp (4-wide) --

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[inline(always)]
unsafe fn fast_exp_neon(x: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::{
        vaddq_s32, vcvtnq_s32_f32, vcvtq_f32_s32, vdupq_n_s32, vfmaq_f32, vreinterpretq_f32_s32,
        vshlq_n_s32,
    };

    let ln2_inv = vdupq_n_f32(std::f32::consts::LOG2_E);
    let ln2_hi = vdupq_n_f32(0.693_359_4);
    let ln2_lo = vdupq_n_f32(-2.121_944_4e-4);

    let c1 = vdupq_n_f32(1.0);
    let c2 = vdupq_n_f32(0.5);
    let c3 = vdupq_n_f32(1.0 / 6.0);
    let c4 = vdupq_n_f32(1.0 / 24.0);
    let c5 = vdupq_n_f32(1.0 / 120.0);
    let c6 = vdupq_n_f32(1.0 / 720.0);

    let x = vmaxq_f32(vdupq_n_f32(-88.0), vminq_f32(vdupq_n_f32(88.0), x));

    let n_f = vmulq_f32(x, ln2_inv);
    let n_i = vcvtnq_s32_f32(n_f);
    let n_f = vcvtq_f32_s32(n_i);

    // r = x - n * ln2  (Cody-Waite two-step)
    let r = vsubq_f32(vsubq_f32(x, vmulq_f32(n_f, ln2_hi)), vmulq_f32(n_f, ln2_lo));

    // Horner: 1 + r*(1 + r*(0.5 + r*(1/6 + r*(1/24 + r*(1/120 + r/720)))))
    let mut poly = vfmaq_f32(c5, r, c6);
    poly = vfmaq_f32(c4, r, poly);
    poly = vfmaq_f32(c3, r, poly);
    poly = vfmaq_f32(c2, r, poly);
    poly = vfmaq_f32(c1, r, poly);
    poly = vfmaq_f32(c1, r, poly); // c0 == c1 == 1.0

    let bias = vdupq_n_s32(127);
    let pow2n = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(n_i, bias)));

    vmulq_f32(poly, pow2n)
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn exp_neon(data: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::{
        vaddq_s32, vcvtnq_s32_f32, vcvtq_f32_s32, vdupq_n_s32, vfmaq_f32, vreinterpretq_f32_s32,
        vshlq_n_s32,
    };

    let len = data.len();
    let inp = data.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = 0usize;

    // Constants hoisted out of the loop
    let ln2_inv = vdupq_n_f32(std::f32::consts::LOG2_E);
    let ln2_hi = vdupq_n_f32(0.693_359_4);
    let ln2_lo = vdupq_n_f32(-2.121_944_4e-4);
    let c1 = vdupq_n_f32(1.0);
    let c2 = vdupq_n_f32(0.5);
    let c3 = vdupq_n_f32(1.0 / 6.0);
    let c4 = vdupq_n_f32(1.0 / 24.0);
    let c5 = vdupq_n_f32(1.0 / 120.0);
    let c6 = vdupq_n_f32(1.0 / 720.0);
    let lo_clamp = vdupq_n_f32(-88.0);
    let hi_clamp = vdupq_n_f32(88.0);
    let bias = vdupq_n_s32(127);

    // 4x interleaved: all polynomial steps run across 4 vectors
    // for maximum instruction-level parallelism on wide pipelines.
    while i + 16 <= len {
        // Load
        let mut x0 = vld1q_f32(inp.add(i));
        let mut x1 = vld1q_f32(inp.add(i + 4));
        let mut x2 = vld1q_f32(inp.add(i + 8));
        let mut x3 = vld1q_f32(inp.add(i + 12));

        // Clamp
        x0 = vmaxq_f32(lo_clamp, vminq_f32(hi_clamp, x0));
        x1 = vmaxq_f32(lo_clamp, vminq_f32(hi_clamp, x1));
        x2 = vmaxq_f32(lo_clamp, vminq_f32(hi_clamp, x2));
        x3 = vmaxq_f32(lo_clamp, vminq_f32(hi_clamp, x3));

        // n = round(x / ln2)
        let n0 = vcvtnq_s32_f32(vmulq_f32(x0, ln2_inv));
        let n1 = vcvtnq_s32_f32(vmulq_f32(x1, ln2_inv));
        let n2 = vcvtnq_s32_f32(vmulq_f32(x2, ln2_inv));
        let n3 = vcvtnq_s32_f32(vmulq_f32(x3, ln2_inv));
        let nf0 = vcvtq_f32_s32(n0);
        let nf1 = vcvtq_f32_s32(n1);
        let nf2 = vcvtq_f32_s32(n2);
        let nf3 = vcvtq_f32_s32(n3);

        // r = x - n * ln2 (Cody-Waite)
        let r0 = vsubq_f32(
            vsubq_f32(x0, vmulq_f32(nf0, ln2_hi)),
            vmulq_f32(nf0, ln2_lo),
        );
        let r1 = vsubq_f32(
            vsubq_f32(x1, vmulq_f32(nf1, ln2_hi)),
            vmulq_f32(nf1, ln2_lo),
        );
        let r2 = vsubq_f32(
            vsubq_f32(x2, vmulq_f32(nf2, ln2_hi)),
            vmulq_f32(nf2, ln2_lo),
        );
        let r3 = vsubq_f32(
            vsubq_f32(x3, vmulq_f32(nf3, ln2_hi)),
            vmulq_f32(nf3, ln2_lo),
        );

        // Horner polynomial: interleaved across all 4 vectors
        let mut p0 = vfmaq_f32(c5, r0, c6);
        let mut p1 = vfmaq_f32(c5, r1, c6);
        let mut p2 = vfmaq_f32(c5, r2, c6);
        let mut p3 = vfmaq_f32(c5, r3, c6);

        p0 = vfmaq_f32(c4, r0, p0);
        p1 = vfmaq_f32(c4, r1, p1);
        p2 = vfmaq_f32(c4, r2, p2);
        p3 = vfmaq_f32(c4, r3, p3);

        p0 = vfmaq_f32(c3, r0, p0);
        p1 = vfmaq_f32(c3, r1, p1);
        p2 = vfmaq_f32(c3, r2, p2);
        p3 = vfmaq_f32(c3, r3, p3);

        p0 = vfmaq_f32(c2, r0, p0);
        p1 = vfmaq_f32(c2, r1, p1);
        p2 = vfmaq_f32(c2, r2, p2);
        p3 = vfmaq_f32(c2, r3, p3);

        p0 = vfmaq_f32(c1, r0, p0);
        p1 = vfmaq_f32(c1, r1, p1);
        p2 = vfmaq_f32(c1, r2, p2);
        p3 = vfmaq_f32(c1, r3, p3);

        p0 = vfmaq_f32(c1, r0, p0); // c0 == c1 == 1.0
        p1 = vfmaq_f32(c1, r1, p1);
        p2 = vfmaq_f32(c1, r2, p2);
        p3 = vfmaq_f32(c1, r3, p3);

        // 2^n via integer bit manipulation
        let pow0 = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(n0, bias)));
        let pow1 = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(n1, bias)));
        let pow2 = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(n2, bias)));
        let pow3 = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(n3, bias)));

        // Store
        vst1q_f32(op.add(i), vmulq_f32(p0, pow0));
        vst1q_f32(op.add(i + 4), vmulq_f32(p1, pow1));
        vst1q_f32(op.add(i + 8), vmulq_f32(p2, pow2));
        vst1q_f32(op.add(i + 12), vmulq_f32(p3, pow3));
        i += 16;
    }

    while i + 4 <= len {
        let v = vld1q_f32(inp.add(i));
        vst1q_f32(op.add(i), fast_exp_neon(v));
        i += 4;
    }

    while i < len {
        *op.add(i) = (*inp.add(i)).exp();
        i += 1;
    }
}

// ===========================================================================
// sin: out[i] = sin(data[i]) -- SIMD polynomial approximation
// ===========================================================================

/// Compute sin(data) into `out` using SIMD polynomial approximation where
/// available, falling back to scalar `f32::sin` otherwise.
#[allow(unsafe_code)]
#[inline]
pub(crate) fn sin_dispatch(data: &[f32], out: &mut [f32]) {
    debug_assert_eq!(data.len(), out.len());

    if cfg!(miri) {
        sin_scalar(data, out);
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if yscv_cpu::host_cpu().features.neon {
            unsafe { sin_neon(data, out) };
            return;
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if yscv_cpu::host_cpu().features.avx {
            unsafe { sin_avx(data, out) };
            return;
        }
        if yscv_cpu::host_cpu().features.sse {
            unsafe { sin_sse(data, out) };
            return;
        }
    }

    sin_scalar(data, out);
}

fn sin_scalar(data: &[f32], out: &mut [f32]) {
    for i in 0..data.len() {
        out[i] = data[i].sin();
    }
}

/// Compute cos(data) into `out` using SIMD polynomial approximation where
/// available, falling back to scalar `f32::cos` otherwise.
#[allow(unsafe_code)]
#[inline]
pub(crate) fn cos_dispatch(data: &[f32], out: &mut [f32]) {
    debug_assert_eq!(data.len(), out.len());

    if cfg!(miri) {
        cos_scalar(data, out);
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if yscv_cpu::host_cpu().features.neon {
            unsafe { cos_neon(data, out) };
            return;
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if yscv_cpu::host_cpu().features.avx {
            unsafe { cos_avx(data, out) };
            return;
        }
        if yscv_cpu::host_cpu().features.sse {
            unsafe { cos_sse(data, out) };
            return;
        }
    }

    cos_scalar(data, out);
}

fn cos_scalar(data: &[f32], out: &mut [f32]) {
    for i in 0..data.len() {
        out[i] = data[i].cos();
    }
}

// -- NEON sin/cos (4-wide) --
//
// Uses Cephes-style range reduction and minimax polynomial:
//   1. Range-reduce x to [-pi, pi] via x = x - round(x / (2*pi)) * 2*pi
//   2. Further reduce to [-pi/2, pi/2] using reflection
//   3. Evaluate minimax polynomial sin(x) ~ x * (1 + x^2 * (c1 + x^2 * (c2 + x^2 * c3)))

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn fast_sin_neon(x: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::{
        vandq_s32, vbslq_f32, vcgtq_f32, vcvtnq_s32_f32, vcvtq_f32_s32, vdupq_n_s32, vfmaq_f32,
        vorrq_s32, vreinterpretq_f32_s32, vreinterpretq_s32_f32,
    };

    // Constants
    let two_pi = vdupq_n_f32(std::f32::consts::TAU); // 2*pi
    let inv_two_pi = vdupq_n_f32(1.0 / std::f32::consts::TAU); // 1/(2*pi)
    let pi = vdupq_n_f32(std::f32::consts::PI);
    let half_pi = vdupq_n_f32(std::f32::consts::FRAC_PI_2);

    // Minimax polynomial coefficients for sin(x) on [-pi/2, pi/2]
    // sin(x) ~ x * (1 + x^2 * (c1 + x^2 * (c2 + x^2 * c3)))
    let c1 = vdupq_n_f32(-1.666_666_6e-1); // -1/6
    let c2 = vdupq_n_f32(8.333_331e-3); // 1/120
    let c3 = vdupq_n_f32(-1.980_741e-4); // -1/5040
    let c4 = vdupq_n_f32(2.601_903e-6); // ~1/362880

    // 1. Range reduce to [-pi, pi]
    let n = vcvtnq_s32_f32(vmulq_f32(x, inv_two_pi));
    let nf = vcvtq_f32_s32(n);
    let x_red = vsubq_f32(x, vmulq_f32(nf, two_pi));

    // 2. Reduce to [-pi/2, pi/2] using reflection:
    //    if x > pi/2:  x = pi - x
    //    if x < -pi/2: x = -pi - x
    let abs_mask_i = vdupq_n_s32(0x7FFF_FFFFu32 as i32);
    let sign_mask_i = vdupq_n_s32(0x8000_0000u32 as i32);
    let abs_x = vreinterpretq_f32_s32(vandq_s32(vreinterpretq_s32_f32(x_red), abs_mask_i));
    let sign_x = vandq_s32(vreinterpretq_s32_f32(x_red), sign_mask_i);
    let signed_pi = vreinterpretq_f32_s32(vorrq_s32(vreinterpretq_s32_f32(pi), sign_x));

    // if abs_x > half_pi, reflect
    let needs_reflect = vcgtq_f32(abs_x, half_pi);
    // reflected = signed_pi - x_red
    let reflected = vsubq_f32(signed_pi, x_red);
    let x_final = vbslq_f32(needs_reflect, reflected, x_red);

    // 3. Evaluate polynomial: sin(x) = x * (1 + x^2*(c1 + x^2*(c2 + x^2*(c3 + x^2*c4))))
    let x2 = vmulq_f32(x_final, x_final);
    let mut poly = vfmaq_f32(c3, x2, c4);
    poly = vfmaq_f32(c2, x2, poly);
    poly = vfmaq_f32(c1, x2, poly);
    poly = vfmaq_f32(vdupq_n_f32(1.0), x2, poly);

    vmulq_f32(x_final, poly)
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn sin_neon(data: &[f32], out: &mut [f32]) {
    let len = data.len();
    let inp = data.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = 0usize;

    while i + 4 <= len {
        let v = vld1q_f32(inp.add(i));
        vst1q_f32(op.add(i), fast_sin_neon(v));
        i += 4;
    }

    while i < len {
        *op.add(i) = (*inp.add(i)).sin();
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn cos_neon(data: &[f32], out: &mut [f32]) {
    let len = data.len();
    let inp = data.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = 0usize;

    let half_pi = vdupq_n_f32(std::f32::consts::FRAC_PI_2);

    while i + 4 <= len {
        let v = vld1q_f32(inp.add(i));
        // cos(x) = sin(x + pi/2)
        vst1q_f32(op.add(i), fast_sin_neon(vaddq_f32(v, half_pi)));
        i += 4;
    }

    while i < len {
        *op.add(i) = (*inp.add(i)).cos();
        i += 1;
    }
}

// -- SSE sin/cos (4-wide) --

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn fast_sin_sse(x: __m128) -> __m128 {
    let two_pi = _mm_set1_ps(std::f32::consts::TAU);
    let inv_two_pi = _mm_set1_ps(1.0 / std::f32::consts::TAU);
    let pi = _mm_set1_ps(std::f32::consts::PI);
    let half_pi = _mm_set1_ps(std::f32::consts::FRAC_PI_2);

    let c1 = _mm_set1_ps(-1.666_666_6e-1);
    let c2 = _mm_set1_ps(8.333_331e-3);
    let c3 = _mm_set1_ps(-1.980_741e-4);
    let c4 = _mm_set1_ps(2.601_903e-6);

    // 1. Range reduce to [-pi, pi]
    let n = _mm_cvtps_epi32(_mm_mul_ps(x, inv_two_pi));
    let nf = _mm_cvtepi32_ps(n);
    let x_red = _mm_sub_ps(x, _mm_mul_ps(nf, two_pi));

    // 2. Reduce to [-pi/2, pi/2]
    let _abs_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFF_FFFFu32 as i32));
    let sign_mask = _mm_set1_ps(-0.0f32);
    let abs_x = {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::_mm_andnot_ps;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::_mm_andnot_ps;
        _mm_andnot_ps(sign_mask, x_red)
    };
    let sign_x = {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::_mm_and_ps;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::_mm_and_ps;
        _mm_and_ps(x_red, sign_mask)
    };
    // signed_pi = pi with sign of x
    let signed_pi = {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::_mm_or_ps;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::_mm_or_ps;
        _mm_or_ps(pi, sign_x)
    };
    // needs_reflect = abs_x > half_pi
    let needs_reflect = {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::_mm_cmpgt_ps;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::_mm_cmpgt_ps;
        _mm_cmpgt_ps(abs_x, half_pi)
    };
    let reflected = _mm_sub_ps(signed_pi, x_red);
    // x_final = needs_reflect ? reflected : x_red
    let x_final = {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::{_mm_and_ps, _mm_andnot_ps, _mm_or_ps};
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::{_mm_and_ps, _mm_andnot_ps, _mm_or_ps};
        _mm_or_ps(
            _mm_and_ps(needs_reflect, reflected),
            _mm_andnot_ps(needs_reflect, x_red),
        )
    };

    // 3. Polynomial
    let x2 = _mm_mul_ps(x_final, x_final);
    let mut poly = _mm_add_ps(c3, _mm_mul_ps(x2, c4));
    poly = _mm_add_ps(c2, _mm_mul_ps(x2, poly));
    poly = _mm_add_ps(c1, _mm_mul_ps(x2, poly));
    poly = _mm_add_ps(_mm_set1_ps(1.0), _mm_mul_ps(x2, poly));

    _mm_mul_ps(x_final, poly)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn sin_sse(data: &[f32], out: &mut [f32]) {
    let len = data.len();
    let inp = data.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = 0usize;

    while i + 4 <= len {
        let v = _mm_loadu_ps(inp.add(i));
        _mm_storeu_ps(op.add(i), fast_sin_sse(v));
        i += 4;
    }

    while i < len {
        *op.add(i) = (*inp.add(i)).sin();
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn cos_sse(data: &[f32], out: &mut [f32]) {
    let len = data.len();
    let inp = data.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = 0usize;

    let half_pi = _mm_set1_ps(std::f32::consts::FRAC_PI_2);

    while i + 4 <= len {
        let v = _mm_loadu_ps(inp.add(i));
        _mm_storeu_ps(op.add(i), fast_sin_sse(_mm_add_ps(v, half_pi)));
        i += 4;
    }

    while i < len {
        *op.add(i) = (*inp.add(i)).cos();
        i += 1;
    }
}

// -- AVX sin/cos (8-wide) --

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn fast_sin_avx(x: __m256) -> __m256 {
    let two_pi = _mm256_set1_ps(std::f32::consts::TAU);
    let inv_two_pi = _mm256_set1_ps(1.0 / std::f32::consts::TAU);
    let pi = _mm256_set1_ps(std::f32::consts::PI);
    let half_pi = _mm256_set1_ps(std::f32::consts::FRAC_PI_2);

    let c1 = _mm256_set1_ps(-1.666_666_6e-1);
    let c2 = _mm256_set1_ps(8.333_331e-3);
    let c3 = _mm256_set1_ps(-1.980_741e-4);
    let c4 = _mm256_set1_ps(2.601_903e-6);

    // 1. Range reduce to [-pi, pi]
    let n = _mm256_cvtps_epi32(_mm256_mul_ps(x, inv_two_pi));
    let nf = _mm256_cvtepi32_ps(n);
    let x_red = _mm256_sub_ps(x, _mm256_mul_ps(nf, two_pi));

    // 2. Reduce to [-pi/2, pi/2]
    let sign_mask = _mm256_set1_ps(-0.0f32);
    let abs_x = _mm256_andnot_ps(sign_mask, x_red);
    let sign_x = {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::_mm256_and_ps;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::_mm256_and_ps;
        _mm256_and_ps(x_red, sign_mask)
    };
    let signed_pi = {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::_mm256_or_ps;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::_mm256_or_ps;
        _mm256_or_ps(pi, sign_x)
    };
    let needs_reflect = {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::_mm256_cmp_ps;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::_mm256_cmp_ps;
        _mm256_cmp_ps::<14>(abs_x, half_pi) // _CMP_GT_OS = 14
    };
    let reflected = _mm256_sub_ps(signed_pi, x_red);
    let x_final = {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::{_mm256_and_ps, _mm256_or_ps};
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::{_mm256_and_ps, _mm256_or_ps};
        _mm256_or_ps(
            _mm256_and_ps(needs_reflect, reflected),
            _mm256_andnot_ps(needs_reflect, x_red),
        )
    };

    // 3. Polynomial
    let x2 = _mm256_mul_ps(x_final, x_final);
    let mut poly = _mm256_add_ps(c3, _mm256_mul_ps(x2, c4));
    poly = _mm256_add_ps(c2, _mm256_mul_ps(x2, poly));
    poly = _mm256_add_ps(c1, _mm256_mul_ps(x2, poly));
    poly = _mm256_add_ps(_mm256_set1_ps(1.0), _mm256_mul_ps(x2, poly));

    _mm256_mul_ps(x_final, poly)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn sin_avx(data: &[f32], out: &mut [f32]) {
    let len = data.len();
    let inp = data.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = 0usize;

    while i + 8 <= len {
        let v = _mm256_loadu_ps(inp.add(i));
        _mm256_storeu_ps(op.add(i), fast_sin_avx(v));
        i += 8;
    }

    if i < len {
        sin_sse(&data[i..], &mut out[i..]);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn cos_avx(data: &[f32], out: &mut [f32]) {
    let len = data.len();
    let inp = data.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = 0usize;

    let half_pi = _mm256_set1_ps(std::f32::consts::FRAC_PI_2);

    while i + 8 <= len {
        let v = _mm256_loadu_ps(inp.add(i));
        _mm256_storeu_ps(op.add(i), fast_sin_avx(_mm256_add_ps(v, half_pi)));
        i += 8;
    }

    if i < len {
        cos_sse(&data[i..], &mut out[i..]);
    }
}

// ===========================================================================
// ln: out[i] = ln(data[i])
// ===========================================================================

/// SIMD-accelerated natural logarithm using IEEE 754 bit decomposition
/// + 5th-order minimax polynomial. Max error ~1.5e-7 (23-bit mantissa limit).
#[allow(unsafe_code)]
#[inline]
pub(crate) fn ln_dispatch(data: &[f32], out: &mut [f32]) {
    debug_assert_eq!(data.len(), out.len());

    if cfg!(miri) {
        for i in 0..data.len() {
            out[i] = data[i].ln();
        }
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if yscv_cpu::host_cpu().features.neon {
            unsafe { ln_neon(data, out) };
            return;
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if yscv_cpu::host_cpu().features.avx {
            unsafe { ln_avx(data, out) };
            return;
        }
        if yscv_cpu::host_cpu().features.sse {
            unsafe { ln_sse(data, out) };
            return;
        }
    }

    for i in 0..data.len() {
        out[i] = data[i].ln();
    }
}

/// NEON polynomial ln using s=(m-1)/(m+1) substitution for fast convergence.
/// Maps mantissa [1,2) -> s in [0, 1/3). 5 terms give < 1e-7 error.
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn ln_neon(data: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::{
        vandq_s32, vcvtq_f32_s32, vdivq_f32, vdupq_n_s32, vfmaq_f32, vorrq_s32,
        vreinterpretq_f32_s32, vreinterpretq_s32_f32, vshrq_n_s32, vsubq_s32,
    };

    let len = data.len();
    let inp = data.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = 0usize;

    // ln(x) = e * ln(2) + ln(m), where x = m * 2^e, m in [1, 2)
    // s = (m-1)/(m+1) maps [1,2) -> [0, 1/3)
    // ln(m) = 2*s*(1 + s^2/3 + s^4/5 + s^6/7 + s^8/9 + s^10/11)
    let mantissa_mask = vdupq_n_s32(0x007F_FFFF);
    let one_bits = vdupq_n_s32(0x3F80_0000u32 as i32);
    let bias = vdupq_n_s32(127);
    let ln2 = vdupq_n_f32(std::f32::consts::LN_2);
    let one = vdupq_n_f32(1.0);
    let two = vdupq_n_f32(2.0);
    // Coefficients: 1/3, 1/5, 1/7, 1/9, 1/11
    let c1 = vdupq_n_f32(1.0 / 3.0);
    let c2 = vdupq_n_f32(1.0 / 5.0);
    let c3 = vdupq_n_f32(1.0 / 7.0);
    let c4 = vdupq_n_f32(1.0 / 9.0);
    let c5 = vdupq_n_f32(1.0 / 11.0);

    while i + 4 <= len {
        let bits = vreinterpretq_s32_f32(vld1q_f32(inp.add(i)));
        let exp_i = vsubq_s32(vshrq_n_s32::<23>(bits), bias);
        let exp_f = vcvtq_f32_s32(exp_i);
        let m = vreinterpretq_f32_s32(vorrq_s32(vandq_s32(bits, mantissa_mask), one_bits));
        // s = (m - 1) / (m + 1)
        let s = vdivq_f32(vsubq_f32(m, one), vaddq_f32(m, one));
        let s2 = vmulq_f32(s, s);
        // Horner: p = 1 + s^2*(c1 + s^2*(c2 + s^2*(c3 + s^2*(c4 + s^2*c5))))
        let mut p = vfmaq_f32(c4, s2, c5);
        p = vfmaq_f32(c3, s2, p);
        p = vfmaq_f32(c2, s2, p);
        p = vfmaq_f32(c1, s2, p);
        p = vfmaq_f32(one, s2, p);
        // ln(m) = 2 * s * p
        let ln_m = vmulq_f32(two, vmulq_f32(s, p));
        // result = e * ln(2) + ln(m)
        let result = vfmaq_f32(ln_m, exp_f, ln2);
        vst1q_f32(op.add(i), result);
        i += 4;
    }

    while i < len {
        *op.add(i) = (*inp.add(i)).ln();
        i += 1;
    }
}

/// SSE polynomial ln using s=(m-1)/(m+1) substitution.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn ln_sse(data: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{
        __m128i, _mm_and_si128, _mm_castps_si128, _mm_castsi128_ps, _mm_cvtepi32_ps, _mm_div_ps,
        _mm_or_si128, _mm_set1_epi32, _mm_srai_epi32, _mm_sub_epi32,
    };
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        __m128i, _mm_and_si128, _mm_castps_si128, _mm_castsi128_ps, _mm_cvtepi32_ps, _mm_div_ps,
        _mm_or_si128, _mm_set1_epi32, _mm_srai_epi32, _mm_sub_epi32,
    };

    let len = data.len();
    let inp = data.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = 0usize;

    let mantissa_mask = _mm_set1_epi32(0x007F_FFFF);
    let one_bits = _mm_set1_epi32(0x3F80_0000u32 as i32);
    let bias = _mm_set1_epi32(127);
    let ln2 = _mm_set1_ps(std::f32::consts::LN_2);
    let one_f = _mm_set1_ps(1.0);
    let two_f = _mm_set1_ps(2.0);
    let k1 = _mm_set1_ps(1.0 / 3.0);
    let k2 = _mm_set1_ps(1.0 / 5.0);
    let k3 = _mm_set1_ps(1.0 / 7.0);
    let k4 = _mm_set1_ps(1.0 / 9.0);
    let k5 = _mm_set1_ps(1.0 / 11.0);

    while i + 4 <= len {
        let bits: __m128i = _mm_castps_si128(_mm_loadu_ps(inp.add(i)));
        let exp_i = _mm_sub_epi32(_mm_srai_epi32::<23>(bits), bias);
        let exp_f = _mm_cvtepi32_ps(exp_i);
        let m = _mm_castsi128_ps(_mm_or_si128(_mm_and_si128(bits, mantissa_mask), one_bits));
        let s = _mm_div_ps(_mm_sub_ps(m, one_f), _mm_add_ps(m, one_f));
        let s2 = _mm_mul_ps(s, s);
        // Horner: p = 1 + s^2*(1/3 + s^2*(1/5 + s^2*(1/7 + s^2*(1/9 + s^2/11))))
        let mut p = _mm_add_ps(k4, _mm_mul_ps(s2, k5));
        p = _mm_add_ps(k3, _mm_mul_ps(s2, p));
        p = _mm_add_ps(k2, _mm_mul_ps(s2, p));
        p = _mm_add_ps(k1, _mm_mul_ps(s2, p));
        p = _mm_add_ps(one_f, _mm_mul_ps(s2, p));
        let ln_m = _mm_mul_ps(two_f, _mm_mul_ps(s, p));
        let result = _mm_add_ps(ln_m, _mm_mul_ps(exp_f, ln2));
        _mm_storeu_ps(op.add(i), result);
        i += 4;
    }

    while i < len {
        *op.add(i) = (*inp.add(i)).ln();
        i += 1;
    }
}

/// AVX polynomial ln using s=(m-1)/(m+1) substitution (8 floats per iteration).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn ln_avx(data: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{
        __m256i, _mm256_and_si256, _mm256_castps_si256, _mm256_cvtepi32_ps, _mm256_div_ps,
        _mm256_or_si256, _mm256_set1_epi32, _mm256_srai_epi32, _mm256_sub_epi32,
    };
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        __m256i, _mm256_and_si256, _mm256_castps_si256, _mm256_cvtepi32_ps, _mm256_div_ps,
        _mm256_or_si256, _mm256_set1_epi32, _mm256_srai_epi32, _mm256_sub_epi32,
    };

    let len = data.len();
    let inp = data.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = 0usize;

    let mantissa_mask = _mm256_set1_epi32(0x007F_FFFF);
    let one_bits = _mm256_set1_epi32(0x3F80_0000u32 as i32);
    let bias = _mm256_set1_epi32(127);
    let ln2 = _mm256_set1_ps(std::f32::consts::LN_2);
    let one_f = _mm256_set1_ps(1.0);
    let two_f = _mm256_set1_ps(2.0);
    let k1 = _mm256_set1_ps(1.0 / 3.0);
    let k2 = _mm256_set1_ps(1.0 / 5.0);
    let k3 = _mm256_set1_ps(1.0 / 7.0);
    let k4 = _mm256_set1_ps(1.0 / 9.0);
    let k5 = _mm256_set1_ps(1.0 / 11.0);

    while i + 8 <= len {
        let bits: __m256i = _mm256_castps_si256(_mm256_loadu_ps(inp.add(i)));
        let exp_i = _mm256_sub_epi32(_mm256_srai_epi32::<23>(bits), bias);
        let exp_f = _mm256_cvtepi32_ps(exp_i);
        let m = _mm256_castsi256_ps(_mm256_or_si256(
            _mm256_and_si256(bits, mantissa_mask),
            one_bits,
        ));
        let s = _mm256_div_ps(_mm256_sub_ps(m, one_f), _mm256_add_ps(m, one_f));
        let s2 = _mm256_mul_ps(s, s);
        // Horner: p = 1 + s^2*(1/3 + s^2*(1/5 + s^2*(1/7 + s^2*(1/9 + s^2/11))))
        let mut p = _mm256_add_ps(k4, _mm256_mul_ps(s2, k5));
        p = _mm256_add_ps(k3, _mm256_mul_ps(s2, p));
        p = _mm256_add_ps(k2, _mm256_mul_ps(s2, p));
        p = _mm256_add_ps(k1, _mm256_mul_ps(s2, p));
        p = _mm256_add_ps(one_f, _mm256_mul_ps(s2, p));
        let ln_m = _mm256_mul_ps(two_f, _mm256_mul_ps(s, p));
        let result = _mm256_add_ps(ln_m, _mm256_mul_ps(exp_f, ln2));
        _mm256_storeu_ps(op.add(i), result);
        i += 8;
    }

    if i < len {
        ln_sse(&data[i..], &mut out[i..]);
    }
}

// ===========================================================================
// tan: out[i] = sin(data[i]) / cos(data[i])
// ===========================================================================

/// SIMD-accelerated tangent using sin/cos dispatchers.
#[allow(unsafe_code)]
#[inline]
pub(crate) fn tan_dispatch(data: &[f32], out: &mut [f32]) {
    debug_assert_eq!(data.len(), out.len());
    // Compute sin and cos in temp buffers, then divide.
    // Use stack allocation for small sizes to avoid heap allocation in hot paths.
    let len = data.len();
    if len <= 256 {
        let mut sin_buf = [0.0f32; 256];
        let mut cos_buf = [0.0f32; 256];
        sin_dispatch(data, &mut sin_buf[..len]);
        cos_dispatch(data, &mut cos_buf[..len]);
        for i in 0..len {
            out[i] = sin_buf[i] / cos_buf[i];
        }
    } else {
        let mut sin_buf = super::super::aligned::AlignedVec::<f32>::uninitialized(len);
        let mut cos_buf = super::super::aligned::AlignedVec::<f32>::uninitialized(len);
        sin_dispatch(data, &mut sin_buf);
        cos_dispatch(data, &mut cos_buf);
        for i in 0..len {
            out[i] = sin_buf[i] / cos_buf[i];
        }
    }
}
