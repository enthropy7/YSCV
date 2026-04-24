// ===========================================================================
// Fast-exp helpers (SSE / AVX / NEON) + exp_slice + sub_exp + tanh
// ===========================================================================

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    float32x4_t, vaddq_f32, vdivq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vmaxq_f32, vminq_f32,
    vmulq_f32, vnegq_f32, vst1q_f32, vsubq_f32,
};
#[cfg(target_arch = "x86")]
use std::arch::x86::{
    __m128, __m256, _mm_add_ps, _mm_castsi128_ps, _mm_cvtepi32_ps, _mm_cvtps_epi32, _mm_loadu_ps,
    _mm_max_ps, _mm_min_ps, _mm_mul_ps, _mm_set1_epi32, _mm_set1_ps, _mm_setzero_ps, _mm_storeu_ps,
    _mm_sub_ps, _mm256_add_ps, _mm256_castsi256_ps, _mm256_cvtepi32_ps, _mm256_cvtps_epi32,
    _mm256_loadu_ps, _mm256_max_ps, _mm256_min_ps, _mm256_mul_ps, _mm256_set1_epi32,
    _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps, _mm256_sub_ps,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m128, __m256, _mm_add_ps, _mm_castsi128_ps, _mm_cvtepi32_ps, _mm_cvtps_epi32, _mm_loadu_ps,
    _mm_max_ps, _mm_min_ps, _mm_mul_ps, _mm_set1_epi32, _mm_set1_ps, _mm_setzero_ps, _mm_storeu_ps,
    _mm_sub_ps, _mm256_add_ps, _mm256_castsi256_ps, _mm256_cvtepi32_ps, _mm256_cvtps_epi32,
    _mm256_loadu_ps, _mm256_max_ps, _mm256_min_ps, _mm256_mul_ps, _mm256_set1_epi32,
    _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps, _mm256_sub_ps,
};

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[allow(unsafe_code)]
unsafe extern "C" {
    fn vvexpf(result: *mut f32, input: *const f32, count: *const i32);
}

#[cfg(all(feature = "mkl", any(target_arch = "x86", target_arch = "x86_64")))]
#[allow(unsafe_code, dead_code)]
unsafe extern "C" {
    fn vsExp(n: i32, a: *const f32, y: *mut f32);
}

#[cfg(all(feature = "armpl", target_arch = "aarch64", not(target_os = "macos")))]
#[allow(unsafe_code, dead_code)]
unsafe extern "C" {
    fn armpl_svexp_f32(n: i32, x: *const f32, y: *mut f32);
}

// ===========================================================================
// SSE fast-exp helper (4-wide)
// ===========================================================================
//
// Uses the classic range-reduction approach:
//   exp(x) = 2^n * exp(r)  where  n = round(x / ln2), r = x - n*ln2
// Then exp(r) is approximated with a degree-4 polynomial on [-ln2/2, ln2/2].

/// Schraudolph 1999 bit-trick exp for SSE: exp(x) ~ reinterpret(int(x * 2^23/ln2) + 127*2^23).
/// WHY: ~3x faster than polynomial, ~1e-3 accuracy is sufficient for sigmoid/tanh where 1/(1+exp) dampens error.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
#[inline]
pub(crate) unsafe fn fast_exp_bittrick_sse(x: __m128) -> __m128 {
    // SSE2 intrinsics used below are always available on x86_64.
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{_mm_add_epi32, _mm_cvtps_epi32, _mm_set1_epi32};
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{_mm_add_epi32, _mm_cvtps_epi32, _mm_set1_epi32};
    // exp(x) ~ reinterpret(int(x * C + B)) where C = 2^23/ln2, B = 127*2^23
    let scale = _mm_set1_ps(12102203.0); // WHY: 2^23/ln(2) maps float to IEEE 754 exponent field
    let offset = _mm_set1_epi32(1065353216); // WHY: 127*2^23 is the IEEE 754 exponent bias in integer form
    let clamp_lo = _mm_set1_ps(-87.0); // WHY: below this exp() produces denormals (underflow)
    let clamp_hi = _mm_set1_ps(88.0); // WHY: above this exp() exceeds f32 max (overflow to inf)
    let x_clamped = _mm_max_ps(_mm_min_ps(x, clamp_hi), clamp_lo);
    let val = _mm_cvtps_epi32(_mm_mul_ps(x_clamped, scale));
    _mm_castsi128_ps(_mm_add_epi32(val, offset))
}

/// Polynomial exp for SSE: range-reduction + 6-term Taylor. Higher accuracy (~1e-6)
/// for standalone exp (softmax, etc.) where precision matters more.
/// WHY 6 terms: 6th-order Taylor series for 2^f on [0,1), max error ~1e-7, good accuracy/speed tradeoff.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
pub(crate) unsafe fn fast_exp_sse(x: __m128) -> __m128 {
    let ln2_inv = _mm_set1_ps(std::f32::consts::LOG2_E);
    let ln2_hi = _mm_set1_ps(0.693_359_4); // upper bits of ln(2)
    let ln2_lo = _mm_set1_ps(-2.121_944_4e-4); // lower bits of ln(2)

    // Polynomial coefficients (Taylor series for exp(r) on [-ln2/2, ln2/2])
    let c0 = _mm_set1_ps(1.0);
    let c1 = _mm_set1_ps(1.0);
    let c2 = _mm_set1_ps(0.5);
    let c3 = _mm_set1_ps(1.0 / 6.0);
    let c4 = _mm_set1_ps(1.0 / 24.0);
    let c5 = _mm_set1_ps(1.0 / 120.0);
    let c6 = _mm_set1_ps(1.0 / 720.0);

    // Clamp input to prevent overflow/underflow
    let x = _mm_max_ps(_mm_set1_ps(-88.0), _mm_min_ps(_mm_set1_ps(88.0), x));

    // n = round(x / ln2)
    let n_f = _mm_mul_ps(x, ln2_inv);
    // Round to nearest integer using convert (rounds to nearest by default)
    let n_i = _mm_cvtps_epi32(n_f);
    let n_f = _mm_cvtepi32_ps(n_i);

    // r = x - n * ln2  (two-step for accuracy)
    let r = _mm_sub_ps(
        _mm_sub_ps(x, _mm_mul_ps(n_f, ln2_hi)),
        _mm_mul_ps(n_f, ln2_lo),
    );

    // Polynomial: c0 + r*(c1 + r*(c2 + r*(c3 + r*(c4 + r*(c5 + r*c6)))))
    let mut poly = _mm_add_ps(c5, _mm_mul_ps(r, c6));
    poly = _mm_add_ps(c4, _mm_mul_ps(r, poly));
    poly = _mm_add_ps(c3, _mm_mul_ps(r, poly));
    poly = _mm_add_ps(c2, _mm_mul_ps(r, poly));
    poly = _mm_add_ps(c1, _mm_mul_ps(r, poly));
    poly = _mm_add_ps(c0, _mm_mul_ps(r, poly));

    // Multiply by 2^n using bit manipulation: reinterpret (n + 127) << 23 as f32.
    // _mm_add_epi32 and _mm_slli_epi32 are SSE2, always available on x86_64.
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

// ===========================================================================
// AVX fast-exp helper (8-wide)
// ===========================================================================

/// Schraudolph 1999 bit-trick exp for AVX: exp(x) ~ reinterpret(int(x * 2^23/ln2) + 127*2^23).
/// WHY: ~3x faster than polynomial, ~1e-3 accuracy is sufficient for sigmoid/tanh where 1/(1+exp) dampens error.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
#[inline]
pub(crate) unsafe fn fast_exp_bittrick_avx(x: __m256) -> __m256 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{_mm256_add_epi32, _mm256_cvtps_epi32, _mm256_set1_epi32};
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{_mm256_add_epi32, _mm256_cvtps_epi32, _mm256_set1_epi32};
    let scale = _mm256_set1_ps(12102203.0); // WHY: 2^23/ln(2) maps float to IEEE 754 exponent field
    let offset = _mm256_set1_epi32(1065353216); // WHY: 127*2^23 is the IEEE 754 exponent bias in integer form
    let clamp_lo = _mm256_set1_ps(-87.0); // WHY: below this exp() produces denormals
    let clamp_hi = _mm256_set1_ps(88.0); // WHY: above this exp() exceeds f32 max
    let x_clamped = _mm256_max_ps(_mm256_min_ps(x, clamp_hi), clamp_lo);
    let val = _mm256_cvtps_epi32(_mm256_mul_ps(x_clamped, scale));
    _mm256_castsi256_ps(_mm256_add_epi32(val, offset))
}

/// Polynomial exp for AVX: range-reduction + 6-term Taylor. Higher accuracy (~1e-6)
/// for standalone exp (softmax, etc.) where precision matters more.
/// WHY 6 terms: 6th-order Taylor series for 2^f on [0,1), max error ~1e-7, good accuracy/speed tradeoff.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
pub(crate) unsafe fn fast_exp_avx(x: __m256) -> __m256 {
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

// ===========================================================================
// NEON fast-exp helper (4-wide)
// ===========================================================================

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn fast_exp_neon(x: float32x4_t) -> float32x4_t {
    use std::arch::aarch64::{
        vaddq_s32, vcvtnq_s32_f32, vcvtq_f32_s32, vreinterpretq_f32_s32, vshlq_n_s32,
    };

    let ln2_inv = vdupq_n_f32(std::f32::consts::LOG2_E);
    let ln2_hi = vdupq_n_f32(0.693_359_4);
    let ln2_lo = vdupq_n_f32(-2.121_944_4e-4);

    let c0 = vdupq_n_f32(1.0);
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

    let r = vsubq_f32(vsubq_f32(x, vmulq_f32(n_f, ln2_hi)), vmulq_f32(n_f, ln2_lo));

    let mut poly = vfmaq_f32(c5, r, c6);
    poly = vfmaq_f32(c4, r, poly);
    poly = vfmaq_f32(c3, r, poly);
    poly = vfmaq_f32(c2, r, poly);
    poly = vfmaq_f32(c1, r, poly);
    poly = vfmaq_f32(c0, r, poly);

    use std::arch::aarch64::vdupq_n_s32;
    let bias = vdupq_n_s32(127);
    let pow2n = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(n_i, bias)));

    vmulq_f32(poly, pow2n)
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
#[inline]
/// Fast exp for sigmoid: range reduction + 3-term Horner + IEEE bit trick.
/// WHY 3 terms: 3rd-order polynomial suffices for sigmoid (1/(1+exp) dampens error); max error ~1e-4.
pub(crate) unsafe fn fast_exp_sigmoid_neon(x: float32x4_t) -> float32x4_t {
    use std::arch::aarch64::{
        vaddq_s32, vcvtnq_s32_f32, vcvtq_f32_s32, vdupq_n_s32, vreinterpretq_f32_s32, vshlq_n_s32,
        vsubq_f32,
    };
    let x = vmaxq_f32(vdupq_n_f32(-88.0), vminq_f32(vdupq_n_f32(88.0), x));
    let n_f = vmulq_f32(x, vdupq_n_f32(std::f32::consts::LOG2_E));
    let n_i = vcvtnq_s32_f32(n_f);
    let r = vsubq_f32(
        x,
        vmulq_f32(vcvtq_f32_s32(n_i), vdupq_n_f32(std::f32::consts::LN_2)),
    );
    let pow2n = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(n_i, vdupq_n_s32(127))));
    let p = vfmaq_f32(vdupq_n_f32(0.5), r, vdupq_n_f32(1.0 / 6.0));
    let p = vfmaq_f32(vdupq_n_f32(1.0), r, p);
    vmulq_f32(vfmaq_f32(vdupq_n_f32(1.0), r, p), pow2n)
}

// ===========================================================================
// Exp slice dispatch + implementations
// ===========================================================================

/// Fast exp approximation applied element-wise: `output[i] = exp(input[i])`.
///
/// Uses a polynomial approximation (degree-4 minimax on [-88, 88]) that is
/// accurate to roughly 1e-4 relative error for the typical NN activation range.
#[allow(unsafe_code, unreachable_code)]
#[inline]
pub fn exp_slice_dispatch(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());

    if cfg!(miri) {
        exp_slice_scalar(input, output);
        return;
    }

    // macOS aarch64: use Apple Accelerate vvexpf (heavily optimized).
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let count = input.len() as i32;
        // SAFETY: vvexpf reads `count` floats from `input` and writes to `output`.
        // Both slices have equal length (debug_assert above).
        unsafe {
            vvexpf(output.as_mut_ptr(), input.as_ptr(), &count);
        }
        return;
    }

    // x86/x86_64 with MKL: use Intel VML vsExp (heavily optimized).
    #[cfg(all(feature = "mkl", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        let count = input.len() as i32;
        // SAFETY: vsExp reads `count` floats from `input` and writes to `output`.
        unsafe { vsExp(count, input.as_ptr(), output.as_mut_ptr()) };
        return;
    }

    // aarch64 Linux with ARMPL: use ARM Performance Libraries vectorized exp.
    #[cfg(all(feature = "armpl", target_arch = "aarch64", not(target_os = "macos")))]
    {
        let count = input.len() as i32;
        // SAFETY: armpl_svexp_f32 reads `count` floats from `input` and writes to `output`.
        unsafe { armpl_svexp_f32(count, input.as_ptr(), output.as_mut_ptr()) };
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                exp_slice_avx(input, output);
            }
            return;
        }
        if std::is_x86_feature_detected!("sse") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                exp_slice_sse(input, output);
            }
            return;
        }
    }

    #[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                exp_slice_neon(input, output);
            }
            return;
        }
    }

    exp_slice_scalar(input, output);
}

/// Fused subtract-and-exp: `output[i] = exp(input[i] - offset)`.
///
/// Combines the max-subtraction and exp steps of softmax into one pass,
/// avoiding an extra read/write of the output buffer.
#[allow(unsafe_code)]
#[inline]
pub fn sub_exp_slice_dispatch(input: &[f32], offset: f32, output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());

    if cfg!(miri) {
        sub_exp_slice_scalar(input, offset, output);
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                sub_exp_slice_avx(input, offset, output);
            }
            return;
        }
        if std::is_x86_feature_detected!("sse") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                sub_exp_slice_sse(input, offset, output);
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                sub_exp_slice_neon(input, offset, output);
            }
            return;
        }
    }

    sub_exp_slice_scalar(input, offset, output);
}

/// Fast tanh applied element-wise: `output[i] = tanh(input[i])`.
///
/// Computed as `2 * sigmoid(2x) - 1`.
#[allow(unsafe_code)]
#[inline]
pub fn tanh_slice_dispatch(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());

    if cfg!(miri) {
        tanh_slice_dispatch_scalar(input, output);
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                tanh_slice_avx(input, output);
            }
            return;
        }
        if std::is_x86_feature_detected!("sse") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                tanh_slice_sse(input, output);
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                tanh_slice_neon(input, output);
            }
            return;
        }
    }

    tanh_slice_dispatch_scalar(input, output);
}

// ===========================================================================
// Scalar fallbacks
// ===========================================================================

pub(super) fn exp_slice_scalar(input: &[f32], output: &mut [f32]) {
    for (o, &v) in output.iter_mut().zip(input.iter()) {
        *o = v.exp();
    }
}

pub(super) fn sub_exp_slice_scalar(input: &[f32], offset: f32, output: &mut [f32]) {
    for (o, &v) in output.iter_mut().zip(input.iter()) {
        *o = (v - offset).exp();
    }
}

pub(super) fn tanh_slice_dispatch_scalar(input: &[f32], output: &mut [f32]) {
    for (o, &v) in output.iter_mut().zip(input.iter()) {
        *o = v.tanh();
    }
}

// ===========================================================================
// Exp slice implementations
// ===========================================================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn exp_slice_sse(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let mut index = 0usize;

    while index + 4 <= len {
        let v = _mm_loadu_ps(in_ptr.add(index));
        let r = fast_exp_sse(v);
        _mm_storeu_ps(out_ptr.add(index), r);
        index += 4;
    }

    while index < len {
        *out_ptr.add(index) = (*in_ptr.add(index)).exp();
        index += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn exp_slice_avx(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let mut index = 0usize;

    // 2x unrolled: process 16 floats per iteration to hide FMA latency.
    while index + 16 <= len {
        // Prefetch next cacheline (64 bytes = 16 floats ahead)
        #[cfg(target_arch = "x86")]
        {
            use std::arch::x86::_mm_prefetch;
            _mm_prefetch::<3>(in_ptr.add(index + 16) as *const i8);
        }
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::_mm_prefetch;
            _mm_prefetch::<3>(in_ptr.add(index + 16) as *const i8);
        }
        let v0 = _mm256_loadu_ps(in_ptr.add(index));
        let v1 = _mm256_loadu_ps(in_ptr.add(index + 8));
        let r0 = fast_exp_avx(v0);
        let r1 = fast_exp_avx(v1);
        _mm256_storeu_ps(out_ptr.add(index), r0);
        _mm256_storeu_ps(out_ptr.add(index + 8), r1);
        index += 16;
    }

    // Handle remaining 8-float chunk
    while index + 8 <= len {
        let v = _mm256_loadu_ps(in_ptr.add(index));
        let r = fast_exp_avx(v);
        _mm256_storeu_ps(out_ptr.add(index), r);
        index += 8;
    }

    if index < len {
        exp_slice_sse(&input[index..], &mut output[index..]);
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, dead_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn exp_slice_neon(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let mut index = 0usize;

    while index + 4 <= len {
        let v = vld1q_f32(in_ptr.add(index));
        let r = fast_exp_neon(v);
        vst1q_f32(out_ptr.add(index), r);
        index += 4;
    }

    while index < len {
        *out_ptr.add(index) = (*in_ptr.add(index)).exp();
        index += 1;
    }
}

// ===========================================================================
// Fused subtract-and-exp: output[i] = exp(input[i] - offset)
// ===========================================================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn sub_exp_slice_sse(input: &[f32], offset: f32, output: &mut [f32]) {
    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let off = _mm_set1_ps(offset);
    let mut index = 0usize;

    while index + 4 <= len {
        let v = _mm_loadu_ps(in_ptr.add(index));
        let shifted = _mm_sub_ps(v, off);
        let r = fast_exp_sse(shifted);
        _mm_storeu_ps(out_ptr.add(index), r);
        index += 4;
    }

    while index < len {
        *out_ptr.add(index) = (*in_ptr.add(index) - offset).exp();
        index += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn sub_exp_slice_avx(input: &[f32], offset: f32, output: &mut [f32]) {
    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let off = _mm256_set1_ps(offset);
    let mut index = 0usize;

    // 2x unrolled: process 16 floats per iteration to hide FMA latency.
    while index + 16 <= len {
        #[cfg(target_arch = "x86")]
        {
            use std::arch::x86::_mm_prefetch;
            _mm_prefetch::<3>(in_ptr.add(index + 16) as *const i8);
        }
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::_mm_prefetch;
            _mm_prefetch::<3>(in_ptr.add(index + 16) as *const i8);
        }
        let v0 = _mm256_loadu_ps(in_ptr.add(index));
        let v1 = _mm256_loadu_ps(in_ptr.add(index + 8));
        let shifted0 = _mm256_sub_ps(v0, off);
        let shifted1 = _mm256_sub_ps(v1, off);
        let r0 = fast_exp_avx(shifted0);
        let r1 = fast_exp_avx(shifted1);
        _mm256_storeu_ps(out_ptr.add(index), r0);
        _mm256_storeu_ps(out_ptr.add(index + 8), r1);
        index += 16;
    }

    // Handle remaining 8-float chunk
    while index + 8 <= len {
        let v = _mm256_loadu_ps(in_ptr.add(index));
        let shifted = _mm256_sub_ps(v, off);
        let r = fast_exp_avx(shifted);
        _mm256_storeu_ps(out_ptr.add(index), r);
        index += 8;
    }

    if index < len {
        sub_exp_slice_sse(&input[index..], offset, &mut output[index..]);
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn sub_exp_slice_neon(input: &[f32], offset: f32, output: &mut [f32]) {
    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let off = vdupq_n_f32(offset);
    let mut index = 0usize;

    while index + 4 <= len {
        let v = vld1q_f32(in_ptr.add(index));
        let shifted = vsubq_f32(v, off);
        let r = fast_exp_neon(shifted);
        vst1q_f32(out_ptr.add(index), r);
        index += 4;
    }

    while index < len {
        *out_ptr.add(index) = (*in_ptr.add(index) - offset).exp();
        index += 1;
    }
}

// ===========================================================================
// Tanh slice implementations: tanh(x) = 2 * sigmoid(2x) - 1
// ===========================================================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn tanh_slice_sse(input: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::_mm_div_ps;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::_mm_div_ps;
    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let two = _mm_set1_ps(2.0);
    let one = _mm_set1_ps(1.0);
    let zero = _mm_setzero_ps();
    let mut index = 0usize;

    while index + 4 <= len {
        let x = _mm_loadu_ps(in_ptr.add(index));
        let two_x = _mm_mul_ps(two, x);
        // sigmoid(2x) = 1 / (1 + exp(-2x))
        let neg_two_x = _mm_sub_ps(zero, two_x);
        // Use polynomial exp (not bit-trick) for tanh -- needs ~1e-4 accuracy
        let exp_neg = fast_exp_sse(neg_two_x);
        let sig = _mm_div_ps(one, _mm_add_ps(one, exp_neg));
        // tanh = 2 * sig - 1
        let result = _mm_sub_ps(_mm_mul_ps(two, sig), one);
        _mm_storeu_ps(out_ptr.add(index), result);
        index += 4;
    }

    while index < len {
        *out_ptr.add(index) = (*in_ptr.add(index)).tanh();
        index += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn tanh_slice_avx(input: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::_mm256_div_ps;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::_mm256_div_ps;
    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let two = _mm256_set1_ps(2.0);
    let one = _mm256_set1_ps(1.0);
    let zero = _mm256_setzero_ps();
    let mut index = 0usize;

    while index + 8 <= len {
        let x = _mm256_loadu_ps(in_ptr.add(index));
        let two_x = _mm256_mul_ps(two, x);
        let neg_two_x = _mm256_sub_ps(zero, two_x);
        // Use polynomial exp (not bit-trick) for tanh -- needs ~1e-4 accuracy
        let exp_neg = fast_exp_avx(neg_two_x);
        let sig = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg));
        let result = _mm256_sub_ps(_mm256_mul_ps(two, sig), one);
        _mm256_storeu_ps(out_ptr.add(index), result);
        index += 8;
    }

    if index < len {
        tanh_slice_sse(&input[index..], &mut output[index..]);
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, dead_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn tanh_slice_neon(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let two = vdupq_n_f32(2.0);
    let one = vdupq_n_f32(1.0);
    let mut index = 0usize;

    // 8x unrolled: 32 elements per iteration, using fast 3-term exp polynomial
    while index + 32 <= len {
        let x0 = vld1q_f32(in_ptr.add(index));
        let x1 = vld1q_f32(in_ptr.add(index + 4));
        let x2 = vld1q_f32(in_ptr.add(index + 8));
        let x3 = vld1q_f32(in_ptr.add(index + 12));
        let x4 = vld1q_f32(in_ptr.add(index + 16));
        let x5 = vld1q_f32(in_ptr.add(index + 20));
        let x6 = vld1q_f32(in_ptr.add(index + 24));
        let x7 = vld1q_f32(in_ptr.add(index + 28));

        // exp(-2x) using fast 3-term polynomial (sufficient for tanh)
        let e0 = fast_exp_sigmoid_neon(vnegq_f32(vmulq_f32(two, x0)));
        let e1 = fast_exp_sigmoid_neon(vnegq_f32(vmulq_f32(two, x1)));
        let e2 = fast_exp_sigmoid_neon(vnegq_f32(vmulq_f32(two, x2)));
        let e3 = fast_exp_sigmoid_neon(vnegq_f32(vmulq_f32(two, x3)));
        let e4 = fast_exp_sigmoid_neon(vnegq_f32(vmulq_f32(two, x4)));
        let e5 = fast_exp_sigmoid_neon(vnegq_f32(vmulq_f32(two, x5)));
        let e6 = fast_exp_sigmoid_neon(vnegq_f32(vmulq_f32(two, x6)));
        let e7 = fast_exp_sigmoid_neon(vnegq_f32(vmulq_f32(two, x7)));

        // tanh(x) = 2 * sigmoid(2x) - 1 = 2/(1+exp(-2x)) - 1
        vst1q_f32(
            out_ptr.add(index),
            vsubq_f32(vdivq_f32(two, vaddq_f32(one, e0)), one),
        );
        vst1q_f32(
            out_ptr.add(index + 4),
            vsubq_f32(vdivq_f32(two, vaddq_f32(one, e1)), one),
        );
        vst1q_f32(
            out_ptr.add(index + 8),
            vsubq_f32(vdivq_f32(two, vaddq_f32(one, e2)), one),
        );
        vst1q_f32(
            out_ptr.add(index + 12),
            vsubq_f32(vdivq_f32(two, vaddq_f32(one, e3)), one),
        );
        vst1q_f32(
            out_ptr.add(index + 16),
            vsubq_f32(vdivq_f32(two, vaddq_f32(one, e4)), one),
        );
        vst1q_f32(
            out_ptr.add(index + 20),
            vsubq_f32(vdivq_f32(two, vaddq_f32(one, e5)), one),
        );
        vst1q_f32(
            out_ptr.add(index + 24),
            vsubq_f32(vdivq_f32(two, vaddq_f32(one, e6)), one),
        );
        vst1q_f32(
            out_ptr.add(index + 28),
            vsubq_f32(vdivq_f32(two, vaddq_f32(one, e7)), one),
        );
        index += 32;
    }

    while index + 4 <= len {
        let x = vld1q_f32(in_ptr.add(index));
        let two_x = vmulq_f32(two, x);
        let neg_two_x = vnegq_f32(two_x);
        let exp_neg = fast_exp_sigmoid_neon(neg_two_x);
        let denom = vaddq_f32(one, exp_neg);
        let result = vsubq_f32(vdivq_f32(two, denom), one);
        vst1q_f32(out_ptr.add(index), result);
        index += 4;
    }

    while index < len {
        *out_ptr.add(index) = (*in_ptr.add(index)).tanh();
        index += 1;
    }
}
