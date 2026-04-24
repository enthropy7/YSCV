// ===========================================================================
// binary_same_shape_dispatch + impls, mul_scalar_inplace + impls
// ===========================================================================

#[cfg(target_arch = "aarch64")]
#[allow(unused_imports)]
use std::arch::aarch64::{vaddq_f32, vdupq_n_f32, vld1q_f32, vmulq_f32, vst1q_f32, vsubq_f32};
#[cfg(target_arch = "x86")]
use std::arch::x86::{
    _mm_add_ps, _mm_loadu_ps, _mm_mul_ps, _mm_storeu_ps, _mm_sub_ps, _mm256_add_ps,
    _mm256_loadu_ps, _mm256_max_ps, _mm256_mul_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    _mm256_sub_ps,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _mm_add_ps, _mm_loadu_ps, _mm_mul_ps, _mm_storeu_ps, _mm_sub_ps, _mm256_add_ps,
    _mm256_loadu_ps, _mm256_max_ps, _mm256_mul_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    _mm256_sub_ps,
};

use super::super::config::BinaryKind;

#[cfg(all(feature = "mkl", any(target_arch = "x86", target_arch = "x86_64")))]
#[allow(unsafe_code, dead_code)]
unsafe extern "C" {
    fn vsAdd(n: i32, a: *const f32, b: *const f32, y: *mut f32);
    fn vsSub(n: i32, a: *const f32, b: *const f32, y: *mut f32);
    fn vsMul(n: i32, a: *const f32, b: *const f32, y: *mut f32);
}

#[cfg(all(feature = "armpl", target_arch = "aarch64", not(target_os = "macos")))]
#[allow(unsafe_code, dead_code)]
unsafe extern "C" {
    fn armpl_svadd_f32(n: i32, a: *const f32, b: *const f32, y: *mut f32);
    fn armpl_svsub_f32(n: i32, a: *const f32, b: *const f32, y: *mut f32);
    fn armpl_svmul_f32(n: i32, a: *const f32, b: *const f32, y: *mut f32);
}

#[cfg(target_os = "macos")]
#[allow(unsafe_code, dead_code)]
unsafe extern "C" {
    fn vDSP_vadd(
        __A: *const f32,
        __IA: i32,
        __B: *const f32,
        __IB: i32,
        __C: *mut f32,
        __IC: i32,
        __N: u32,
    );
    fn vDSP_vsub(
        __B: *const f32,
        __IB: i32,
        __A: *const f32,
        __IA: i32,
        __C: *mut f32,
        __IC: i32,
        __N: u32,
    );
    fn vDSP_vmul(
        __A: *const f32,
        __IA: i32,
        __B: *const f32,
        __IB: i32,
        __C: *mut f32,
        __IC: i32,
        __N: u32,
    );
}

// ===========================================================================
// Dispatchers
// ===========================================================================

#[allow(unsafe_code, unreachable_code)]
#[inline]
pub fn binary_same_shape_dispatch(lhs: &[f32], rhs: &[f32], out: &mut [f32], kind: BinaryKind) {
    debug_assert_eq!(lhs.len(), rhs.len());
    debug_assert_eq!(lhs.len(), out.len());

    if cfg!(miri) {
        // SAFETY: scalar path only reads/writes within equal-sized slice bounds.
        unsafe {
            binary_same_shape_scalar(lhs, rhs, out, kind);
        }
        return;
    }

    // macOS: use vDSP for add/sub/mul (heavily optimized, zero loop overhead).
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let n = lhs.len() as u32;
        // SAFETY: vDSP functions read/write `n` floats from contiguous slices.
        unsafe {
            match kind {
                BinaryKind::Add => {
                    vDSP_vadd(lhs.as_ptr(), 1, rhs.as_ptr(), 1, out.as_mut_ptr(), 1, n)
                }
                // NOTE: vDSP_vsub computes A - B with reversed argument order: vsub(B, ..., A, ..., C, ...)
                BinaryKind::Sub => {
                    vDSP_vsub(rhs.as_ptr(), 1, lhs.as_ptr(), 1, out.as_mut_ptr(), 1, n)
                }
                BinaryKind::Mul => {
                    vDSP_vmul(lhs.as_ptr(), 1, rhs.as_ptr(), 1, out.as_mut_ptr(), 1, n)
                }
            }
        }
        return;
    }

    // x86/x86_64 with MKL: use Intel VML for add/sub/mul (heavily optimized).
    #[cfg(all(feature = "mkl", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        let n = lhs.len() as i32;
        // SAFETY: VML functions read `n` floats from contiguous slices and write to `out`.
        unsafe {
            match kind {
                BinaryKind::Add => vsAdd(n, lhs.as_ptr(), rhs.as_ptr(), out.as_mut_ptr()),
                BinaryKind::Sub => vsSub(n, lhs.as_ptr(), rhs.as_ptr(), out.as_mut_ptr()),
                BinaryKind::Mul => vsMul(n, lhs.as_ptr(), rhs.as_ptr(), out.as_mut_ptr()),
            }
        }
        return;
    }

    // aarch64 Linux with ARMPL: use ARM Performance Libraries for add/sub/mul.
    #[cfg(all(feature = "armpl", target_arch = "aarch64", not(target_os = "macos")))]
    {
        let n = lhs.len() as i32;
        // SAFETY: ARMPL functions read `n` floats from contiguous slices and write to `out`.
        unsafe {
            match kind {
                BinaryKind::Add => armpl_svadd_f32(n, lhs.as_ptr(), rhs.as_ptr(), out.as_mut_ptr()),
                BinaryKind::Sub => armpl_svsub_f32(n, lhs.as_ptr(), rhs.as_ptr(), out.as_mut_ptr()),
                BinaryKind::Mul => armpl_svmul_f32(n, lhs.as_ptr(), rhs.as_ptr(), out.as_mut_ptr()),
            }
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                binary_same_shape_avx(lhs, rhs, out, kind);
            }
            return;
        }
        if std::is_x86_feature_detected!("sse") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                binary_same_shape_sse(lhs, rhs, out, kind);
            }
            return;
        }
    }

    #[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                binary_same_shape_neon(lhs, rhs, out, kind);
            }
            return;
        }
    }

    // SAFETY: scalar path only reads/writes within equal-sized slice bounds.
    unsafe {
        binary_same_shape_scalar(lhs, rhs, out, kind);
    }
}

/// Multiply every element of `data` by `scalar` in-place.
#[allow(unsafe_code, dead_code)]
#[inline]
pub fn mul_scalar_inplace_dispatch(data: &mut [f32], scalar: f32) {
    if cfg!(miri) || data.is_empty() {
        for v in data.iter_mut() {
            *v *= scalar;
        }
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: guarded by runtime feature detection.
            unsafe { mul_scalar_inplace_neon(data, scalar) };
            return;
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            // SAFETY: guarded by runtime feature detection.
            unsafe { mul_scalar_inplace_avx(data, scalar) };
            return;
        }
        if std::is_x86_feature_detected!("sse") {
            // SAFETY: guarded by runtime feature detection.
            unsafe { mul_scalar_inplace_sse(data, scalar) };
            return;
        }
    }

    for v in data.iter_mut() {
        *v *= scalar;
    }
}

// ===========================================================================
// Scalar fallbacks
// ===========================================================================

#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn binary_same_shape_scalar(lhs: &[f32], rhs: &[f32], out: &mut [f32], kind: BinaryKind) {
    let len = lhs.len();
    let left_ptr = lhs.as_ptr();
    let right_ptr = rhs.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut index = 0usize;

    match kind {
        BinaryKind::Add => {
            while index + 8 <= len {
                *out_ptr.add(index) = *left_ptr.add(index) + *right_ptr.add(index);
                *out_ptr.add(index + 1) = *left_ptr.add(index + 1) + *right_ptr.add(index + 1);
                *out_ptr.add(index + 2) = *left_ptr.add(index + 2) + *right_ptr.add(index + 2);
                *out_ptr.add(index + 3) = *left_ptr.add(index + 3) + *right_ptr.add(index + 3);
                *out_ptr.add(index + 4) = *left_ptr.add(index + 4) + *right_ptr.add(index + 4);
                *out_ptr.add(index + 5) = *left_ptr.add(index + 5) + *right_ptr.add(index + 5);
                *out_ptr.add(index + 6) = *left_ptr.add(index + 6) + *right_ptr.add(index + 6);
                *out_ptr.add(index + 7) = *left_ptr.add(index + 7) + *right_ptr.add(index + 7);
                index += 8;
            }
            while index < len {
                *out_ptr.add(index) = *left_ptr.add(index) + *right_ptr.add(index);
                index += 1;
            }
        }
        BinaryKind::Sub => {
            while index + 8 <= len {
                *out_ptr.add(index) = *left_ptr.add(index) - *right_ptr.add(index);
                *out_ptr.add(index + 1) = *left_ptr.add(index + 1) - *right_ptr.add(index + 1);
                *out_ptr.add(index + 2) = *left_ptr.add(index + 2) - *right_ptr.add(index + 2);
                *out_ptr.add(index + 3) = *left_ptr.add(index + 3) - *right_ptr.add(index + 3);
                *out_ptr.add(index + 4) = *left_ptr.add(index + 4) - *right_ptr.add(index + 4);
                *out_ptr.add(index + 5) = *left_ptr.add(index + 5) - *right_ptr.add(index + 5);
                *out_ptr.add(index + 6) = *left_ptr.add(index + 6) - *right_ptr.add(index + 6);
                *out_ptr.add(index + 7) = *left_ptr.add(index + 7) - *right_ptr.add(index + 7);
                index += 8;
            }
            while index < len {
                *out_ptr.add(index) = *left_ptr.add(index) - *right_ptr.add(index);
                index += 1;
            }
        }
        BinaryKind::Mul => {
            while index + 8 <= len {
                *out_ptr.add(index) = *left_ptr.add(index) * *right_ptr.add(index);
                *out_ptr.add(index + 1) = *left_ptr.add(index + 1) * *right_ptr.add(index + 1);
                *out_ptr.add(index + 2) = *left_ptr.add(index + 2) * *right_ptr.add(index + 2);
                *out_ptr.add(index + 3) = *left_ptr.add(index + 3) * *right_ptr.add(index + 3);
                *out_ptr.add(index + 4) = *left_ptr.add(index + 4) * *right_ptr.add(index + 4);
                *out_ptr.add(index + 5) = *left_ptr.add(index + 5) * *right_ptr.add(index + 5);
                *out_ptr.add(index + 6) = *left_ptr.add(index + 6) * *right_ptr.add(index + 6);
                *out_ptr.add(index + 7) = *left_ptr.add(index + 7) * *right_ptr.add(index + 7);
                index += 8;
            }
            while index < len {
                *out_ptr.add(index) = *left_ptr.add(index) * *right_ptr.add(index);
                index += 1;
            }
        }
    }
}

// ===========================================================================
// Binary SIMD implementations
// ===========================================================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn binary_same_shape_sse(lhs: &[f32], rhs: &[f32], out: &mut [f32], kind: BinaryKind) {
    let len = lhs.len();
    let left_ptr = lhs.as_ptr();
    let right_ptr = rhs.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut index = 0usize;

    while index + 4 <= len {
        let left = _mm_loadu_ps(left_ptr.add(index));
        let right = _mm_loadu_ps(right_ptr.add(index));
        let result = match kind {
            BinaryKind::Add => _mm_add_ps(left, right),
            BinaryKind::Sub => _mm_sub_ps(left, right),
            BinaryKind::Mul => _mm_mul_ps(left, right),
        };
        _mm_storeu_ps(out_ptr.add(index), result);
        index += 4;
    }

    if index < len {
        binary_same_shape_scalar(&lhs[index..], &rhs[index..], &mut out[index..], kind);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn binary_same_shape_avx(lhs: &[f32], rhs: &[f32], out: &mut [f32], kind: BinaryKind) {
    let len = lhs.len();
    let left_ptr = lhs.as_ptr();
    let right_ptr = rhs.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut index = 0usize;

    // 4x unrolled: process 32 floats per iteration with software prefetch.
    // Matches vDSP throughput by keeping the OoO pipeline fully saturated.
    match kind {
        BinaryKind::Add => {
            while index + 32 <= len {
                #[cfg(target_arch = "x86")]
                {
                    use std::arch::x86::_mm_prefetch;
                    _mm_prefetch::<3>(left_ptr.add(index + 32) as *const i8);
                    _mm_prefetch::<3>(right_ptr.add(index + 32) as *const i8);
                }
                #[cfg(target_arch = "x86_64")]
                {
                    use std::arch::x86_64::_mm_prefetch;
                    _mm_prefetch::<3>(left_ptr.add(index + 32) as *const i8);
                    _mm_prefetch::<3>(right_ptr.add(index + 32) as *const i8);
                }
                let a0 = _mm256_loadu_ps(left_ptr.add(index));
                let b0 = _mm256_loadu_ps(right_ptr.add(index));
                let a1 = _mm256_loadu_ps(left_ptr.add(index + 8));
                let b1 = _mm256_loadu_ps(right_ptr.add(index + 8));
                _mm256_storeu_ps(out_ptr.add(index), _mm256_add_ps(a0, b0));
                _mm256_storeu_ps(out_ptr.add(index + 8), _mm256_add_ps(a1, b1));
                let a2 = _mm256_loadu_ps(left_ptr.add(index + 16));
                let b2 = _mm256_loadu_ps(right_ptr.add(index + 16));
                let a3 = _mm256_loadu_ps(left_ptr.add(index + 24));
                let b3 = _mm256_loadu_ps(right_ptr.add(index + 24));
                _mm256_storeu_ps(out_ptr.add(index + 16), _mm256_add_ps(a2, b2));
                _mm256_storeu_ps(out_ptr.add(index + 24), _mm256_add_ps(a3, b3));
                index += 32;
            }
        }
        BinaryKind::Sub => {
            while index + 32 <= len {
                #[cfg(target_arch = "x86")]
                {
                    use std::arch::x86::_mm_prefetch;
                    _mm_prefetch::<3>(left_ptr.add(index + 32) as *const i8);
                    _mm_prefetch::<3>(right_ptr.add(index + 32) as *const i8);
                }
                #[cfg(target_arch = "x86_64")]
                {
                    use std::arch::x86_64::_mm_prefetch;
                    _mm_prefetch::<3>(left_ptr.add(index + 32) as *const i8);
                    _mm_prefetch::<3>(right_ptr.add(index + 32) as *const i8);
                }
                let a0 = _mm256_loadu_ps(left_ptr.add(index));
                let b0 = _mm256_loadu_ps(right_ptr.add(index));
                let a1 = _mm256_loadu_ps(left_ptr.add(index + 8));
                let b1 = _mm256_loadu_ps(right_ptr.add(index + 8));
                _mm256_storeu_ps(out_ptr.add(index), _mm256_sub_ps(a0, b0));
                _mm256_storeu_ps(out_ptr.add(index + 8), _mm256_sub_ps(a1, b1));
                let a2 = _mm256_loadu_ps(left_ptr.add(index + 16));
                let b2 = _mm256_loadu_ps(right_ptr.add(index + 16));
                let a3 = _mm256_loadu_ps(left_ptr.add(index + 24));
                let b3 = _mm256_loadu_ps(right_ptr.add(index + 24));
                _mm256_storeu_ps(out_ptr.add(index + 16), _mm256_sub_ps(a2, b2));
                _mm256_storeu_ps(out_ptr.add(index + 24), _mm256_sub_ps(a3, b3));
                index += 32;
            }
        }
        BinaryKind::Mul => {
            while index + 32 <= len {
                #[cfg(target_arch = "x86")]
                {
                    use std::arch::x86::_mm_prefetch;
                    _mm_prefetch::<3>(left_ptr.add(index + 32) as *const i8);
                    _mm_prefetch::<3>(right_ptr.add(index + 32) as *const i8);
                }
                #[cfg(target_arch = "x86_64")]
                {
                    use std::arch::x86_64::_mm_prefetch;
                    _mm_prefetch::<3>(left_ptr.add(index + 32) as *const i8);
                    _mm_prefetch::<3>(right_ptr.add(index + 32) as *const i8);
                }
                let a0 = _mm256_loadu_ps(left_ptr.add(index));
                let b0 = _mm256_loadu_ps(right_ptr.add(index));
                let a1 = _mm256_loadu_ps(left_ptr.add(index + 8));
                let b1 = _mm256_loadu_ps(right_ptr.add(index + 8));
                _mm256_storeu_ps(out_ptr.add(index), _mm256_mul_ps(a0, b0));
                _mm256_storeu_ps(out_ptr.add(index + 8), _mm256_mul_ps(a1, b1));
                let a2 = _mm256_loadu_ps(left_ptr.add(index + 16));
                let b2 = _mm256_loadu_ps(right_ptr.add(index + 16));
                let a3 = _mm256_loadu_ps(left_ptr.add(index + 24));
                let b3 = _mm256_loadu_ps(right_ptr.add(index + 24));
                _mm256_storeu_ps(out_ptr.add(index + 16), _mm256_mul_ps(a2, b2));
                _mm256_storeu_ps(out_ptr.add(index + 24), _mm256_mul_ps(a3, b3));
                index += 32;
            }
        }
    }

    // Handle remaining elements 8 at a time
    while index + 8 <= len {
        let left = _mm256_loadu_ps(left_ptr.add(index));
        let right = _mm256_loadu_ps(right_ptr.add(index));
        let result = match kind {
            BinaryKind::Add => _mm256_add_ps(left, right),
            BinaryKind::Sub => _mm256_sub_ps(left, right),
            BinaryKind::Mul => _mm256_mul_ps(left, right),
        };
        _mm256_storeu_ps(out_ptr.add(index), result);
        index += 8;
    }

    if index < len {
        binary_same_shape_sse(&lhs[index..], &rhs[index..], &mut out[index..], kind);
    }
}

#[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn binary_same_shape_neon(lhs: &[f32], rhs: &[f32], out: &mut [f32], kind: BinaryKind) {
    let len = lhs.len();
    let left_ptr = lhs.as_ptr();
    let right_ptr = rhs.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut index = 0usize;

    while index + 4 <= len {
        let left = vld1q_f32(left_ptr.add(index));
        let right = vld1q_f32(right_ptr.add(index));
        let result = match kind {
            BinaryKind::Add => vaddq_f32(left, right),
            BinaryKind::Sub => vsubq_f32(left, right),
            BinaryKind::Mul => vmulq_f32(left, right),
        };
        vst1q_f32(out_ptr.add(index), result);
        index += 4;
    }

    if index < len {
        binary_same_shape_scalar(&lhs[index..], &rhs[index..], &mut out[index..], kind);
    }
}

// ===========================================================================
// add_relu_inplace: data[i] = max(data[i] + rhs[i], 0) using SIMD
// ===========================================================================

#[allow(unsafe_code, unreachable_code)]
#[inline]
pub fn add_relu_inplace_dispatch(data: &mut [f32], rhs: &[f32]) {
    debug_assert_eq!(data.len(), rhs.len());

    if cfg!(miri) || data.is_empty() {
        for (d, &r) in data.iter_mut().zip(rhs.iter()) {
            let v = *d + r;
            *d = if v > 0.0 { v } else { 0.0 };
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            unsafe { add_relu_inplace_avx(data, rhs) };
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { add_relu_inplace_neon(data, rhs) };
            return;
        }
    }

    for (d, &r) in data.iter_mut().zip(rhs.iter()) {
        let v = *d + r;
        *d = if v > 0.0 { v } else { 0.0 };
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn add_relu_inplace_avx(data: &mut [f32], rhs: &[f32]) {
    let len = data.len();
    let dp = data.as_mut_ptr();
    let rp = rhs.as_ptr();
    let zero = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 32 <= len {
        let s0 = _mm256_add_ps(_mm256_loadu_ps(dp.add(i)), _mm256_loadu_ps(rp.add(i)));
        let s1 = _mm256_add_ps(
            _mm256_loadu_ps(dp.add(i + 8)),
            _mm256_loadu_ps(rp.add(i + 8)),
        );
        _mm256_storeu_ps(dp.add(i), _mm256_max_ps(s0, zero));
        _mm256_storeu_ps(dp.add(i + 8), _mm256_max_ps(s1, zero));
        let s2 = _mm256_add_ps(
            _mm256_loadu_ps(dp.add(i + 16)),
            _mm256_loadu_ps(rp.add(i + 16)),
        );
        let s3 = _mm256_add_ps(
            _mm256_loadu_ps(dp.add(i + 24)),
            _mm256_loadu_ps(rp.add(i + 24)),
        );
        _mm256_storeu_ps(dp.add(i + 16), _mm256_max_ps(s2, zero));
        _mm256_storeu_ps(dp.add(i + 24), _mm256_max_ps(s3, zero));
        i += 32;
    }
    while i + 8 <= len {
        let s = _mm256_add_ps(_mm256_loadu_ps(dp.add(i)), _mm256_loadu_ps(rp.add(i)));
        _mm256_storeu_ps(dp.add(i), _mm256_max_ps(s, zero));
        i += 8;
    }
    while i < len {
        let v = *dp.add(i) + *rp.add(i);
        *dp.add(i) = if v > 0.0 { v } else { 0.0 };
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn add_relu_inplace_neon(data: &mut [f32], rhs: &[f32]) {
    use std::arch::aarch64::vmaxq_f32;
    let len = data.len();
    let dp = data.as_mut_ptr();
    let rp = rhs.as_ptr();
    let zero = vdupq_n_f32(0.0);
    let mut i = 0usize;
    while i + 4 <= len {
        let s = vaddq_f32(vld1q_f32(dp.add(i)), vld1q_f32(rp.add(i)));
        vst1q_f32(dp.add(i), vmaxq_f32(s, zero));
        i += 4;
    }
    while i < len {
        let v = *dp.add(i) + *rp.add(i);
        *dp.add(i) = if v > 0.0 { v } else { 0.0 };
        i += 1;
    }
}

// ===========================================================================
// add_inplace: data[i] += rhs[i] using SIMD
// ===========================================================================

#[allow(unsafe_code, unreachable_code)]
#[inline]
pub fn add_inplace_dispatch(data: &mut [f32], rhs: &[f32]) {
    debug_assert_eq!(data.len(), rhs.len());

    if cfg!(miri) || data.is_empty() {
        for (d, &r) in data.iter_mut().zip(rhs.iter()) {
            *d += r;
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            unsafe { add_inplace_avx(data, rhs) };
            return;
        }
        if std::is_x86_feature_detected!("sse") {
            unsafe { add_inplace_sse(data, rhs) };
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { add_inplace_neon(data, rhs) };
            return;
        }
    }

    for (d, &r) in data.iter_mut().zip(rhs.iter()) {
        *d += r;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn add_inplace_avx(data: &mut [f32], rhs: &[f32]) {
    let len = data.len();
    let dp = data.as_mut_ptr();
    let rp = rhs.as_ptr();
    let mut i = 0usize;
    while i + 32 <= len {
        let a0 = _mm256_loadu_ps(dp.add(i));
        let b0 = _mm256_loadu_ps(rp.add(i));
        let a1 = _mm256_loadu_ps(dp.add(i + 8));
        let b1 = _mm256_loadu_ps(rp.add(i + 8));
        _mm256_storeu_ps(dp.add(i), _mm256_add_ps(a0, b0));
        _mm256_storeu_ps(dp.add(i + 8), _mm256_add_ps(a1, b1));
        let a2 = _mm256_loadu_ps(dp.add(i + 16));
        let b2 = _mm256_loadu_ps(rp.add(i + 16));
        let a3 = _mm256_loadu_ps(dp.add(i + 24));
        let b3 = _mm256_loadu_ps(rp.add(i + 24));
        _mm256_storeu_ps(dp.add(i + 16), _mm256_add_ps(a2, b2));
        _mm256_storeu_ps(dp.add(i + 24), _mm256_add_ps(a3, b3));
        i += 32;
    }
    while i + 8 <= len {
        let a = _mm256_loadu_ps(dp.add(i));
        let b = _mm256_loadu_ps(rp.add(i));
        _mm256_storeu_ps(dp.add(i), _mm256_add_ps(a, b));
        i += 8;
    }
    while i < len {
        *dp.add(i) += *rp.add(i);
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn add_inplace_sse(data: &mut [f32], rhs: &[f32]) {
    let len = data.len();
    let dp = data.as_mut_ptr();
    let rp = rhs.as_ptr();
    let mut i = 0usize;
    while i + 4 <= len {
        let a = _mm_loadu_ps(dp.add(i));
        let b = _mm_loadu_ps(rp.add(i));
        _mm_storeu_ps(dp.add(i), _mm_add_ps(a, b));
        i += 4;
    }
    while i < len {
        *dp.add(i) += *rp.add(i);
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn add_inplace_neon(data: &mut [f32], rhs: &[f32]) {
    let len = data.len();
    let dp = data.as_mut_ptr();
    let rp = rhs.as_ptr();
    let mut i = 0usize;
    while i + 4 <= len {
        let a = vld1q_f32(dp.add(i));
        let b = vld1q_f32(rp.add(i));
        vst1q_f32(dp.add(i), vaddq_f32(a, b));
        i += 4;
    }
    while i < len {
        *dp.add(i) += *rp.add(i);
        i += 1;
    }
}

// ===========================================================================
// mul_scalar_inplace implementations
// ===========================================================================

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn mul_scalar_inplace_neon(data: &mut [f32], scalar: f32) {
    let len = data.len();
    let ptr = data.as_mut_ptr();
    let vs = vdupq_n_f32(scalar);
    let mut i = 0usize;
    while i + 4 <= len {
        let v = vld1q_f32(ptr.add(i));
        vst1q_f32(ptr.add(i), vmulq_f32(v, vs));
        i += 4;
    }
    while i < len {
        *ptr.add(i) *= scalar;
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn mul_scalar_inplace_avx(data: &mut [f32], scalar: f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let len = data.len();
    let ptr = data.as_mut_ptr();
    let vs = _mm256_set1_ps(scalar);
    let mut i = 0usize;
    while i + 8 <= len {
        let v = _mm256_loadu_ps(ptr.add(i));
        _mm256_storeu_ps(ptr.add(i), _mm256_mul_ps(v, vs));
        i += 8;
    }
    // SSE tail
    let vs4 = _mm_set1_ps(scalar);
    while i + 4 <= len {
        let v = _mm_loadu_ps(ptr.add(i));
        _mm_storeu_ps(ptr.add(i), _mm_mul_ps(v, vs4));
        i += 4;
    }
    while i < len {
        *ptr.add(i) *= scalar;
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn mul_scalar_inplace_sse(data: &mut [f32], scalar: f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let len = data.len();
    let ptr = data.as_mut_ptr();
    let vs = _mm_set1_ps(scalar);
    let mut i = 0usize;
    while i + 4 <= len {
        let v = _mm_loadu_ps(ptr.add(i));
        _mm_storeu_ps(ptr.add(i), _mm_mul_ps(v, vs));
        i += 4;
    }
    while i < len {
        *ptr.add(i) *= scalar;
        i += 1;
    }
}
