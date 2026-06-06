//! In-place operations: add, max, relu, add_scalar, mul_scalar with SIMD dispatch.

use super::*;

// ===========================================================================
// In-place add: dst[i] += src[i]
// ===========================================================================

#[allow(unsafe_code)]
#[inline]
pub(crate) fn add_inplace_dispatch(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());

    if cfg!(miri) {
        for i in 0..dst.len() {
            dst[i] += src[i];
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if yscv_cpu::host_cpu().features.avx {
            unsafe { add_inplace_avx(dst, src) };
            return;
        }
        if yscv_cpu::host_cpu().features.sse {
            unsafe { add_inplace_sse(dst, src) };
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if yscv_cpu::host_cpu().features.neon {
            unsafe { add_inplace_neon(dst, src) };
            return;
        }
    }

    for i in 0..dst.len() {
        dst[i] += src[i];
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn add_inplace_sse(dst: &mut [f32], src: &[f32]) {
    let len = dst.len();
    let dp = dst.as_mut_ptr();
    let sp = src.as_ptr();
    let mut i = 0usize;

    while i + 4 <= len {
        let d = _mm_loadu_ps(dp.add(i));
        let s = _mm_loadu_ps(sp.add(i));
        _mm_storeu_ps(dp.add(i), _mm_add_ps(d, s));
        i += 4;
    }

    while i < len {
        *dp.add(i) += *sp.add(i);
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn add_inplace_avx(dst: &mut [f32], src: &[f32]) {
    let len = dst.len();
    let dp = dst.as_mut_ptr();
    let sp = src.as_ptr();
    let mut i = 0usize;

    while i + 8 <= len {
        let d = _mm256_loadu_ps(dp.add(i));
        let s = _mm256_loadu_ps(sp.add(i));
        _mm256_storeu_ps(dp.add(i), _mm256_add_ps(d, s));
        i += 8;
    }

    if i < len {
        add_inplace_sse(&mut dst[i..], &src[i..]);
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn add_inplace_neon(dst: &mut [f32], src: &[f32]) {
    let len = dst.len();
    let dp = dst.as_mut_ptr();
    let sp = src.as_ptr();
    let mut i = 0usize;

    while i + 4 <= len {
        let d = vld1q_f32(dp.add(i));
        let s = vld1q_f32(sp.add(i));
        vst1q_f32(dp.add(i), vaddq_f32(d, s));
        i += 4;
    }

    while i < len {
        *dp.add(i) += *sp.add(i);
        i += 1;
    }
}

// ===========================================================================
// In-place max: dst[i] = max(dst[i], src[i])
// ===========================================================================

#[allow(unsafe_code)]
#[inline]
pub(crate) fn max_inplace_dispatch(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());

    if cfg!(miri) {
        for i in 0..dst.len() {
            dst[i] = dst[i].max(src[i]);
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if yscv_cpu::host_cpu().features.avx {
            unsafe { max_inplace_avx(dst, src) };
            return;
        }
        if yscv_cpu::host_cpu().features.sse {
            unsafe { max_inplace_sse(dst, src) };
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if yscv_cpu::host_cpu().features.neon {
            unsafe { max_inplace_neon(dst, src) };
            return;
        }
    }

    for i in 0..dst.len() {
        dst[i] = dst[i].max(src[i]);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn max_inplace_sse(dst: &mut [f32], src: &[f32]) {
    let len = dst.len();
    let dp = dst.as_mut_ptr();
    let sp = src.as_ptr();
    let mut i = 0usize;

    while i + 4 <= len {
        let d = _mm_loadu_ps(dp.add(i));
        let s = _mm_loadu_ps(sp.add(i));
        _mm_storeu_ps(dp.add(i), _mm_max_ps(d, s));
        i += 4;
    }

    while i < len {
        let d = *dp.add(i);
        let s = *sp.add(i);
        *dp.add(i) = if d > s { d } else { s };
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn max_inplace_avx(dst: &mut [f32], src: &[f32]) {
    let len = dst.len();
    let dp = dst.as_mut_ptr();
    let sp = src.as_ptr();
    let mut i = 0usize;

    while i + 8 <= len {
        let d = _mm256_loadu_ps(dp.add(i));
        let s = _mm256_loadu_ps(sp.add(i));
        _mm256_storeu_ps(dp.add(i), _mm256_max_ps(d, s));
        i += 8;
    }

    if i < len {
        max_inplace_sse(&mut dst[i..], &src[i..]);
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn max_inplace_neon(dst: &mut [f32], src: &[f32]) {
    let len = dst.len();
    let dp = dst.as_mut_ptr();
    let sp = src.as_ptr();
    let mut i = 0usize;

    while i + 4 <= len {
        let d = vld1q_f32(dp.add(i));
        let s = vld1q_f32(sp.add(i));
        vst1q_f32(dp.add(i), vmaxq_f32(d, s));
        i += 4;
    }

    while i < len {
        let d = *dp.add(i);
        let s = *sp.add(i);
        *dp.add(i) = if d > s { d } else { s };
        i += 1;
    }
}

// ===========================================================================
// In-place ReLU: v[i] = max(v[i], 0)
// ===========================================================================

#[allow(unsafe_code)]
#[inline]
pub(crate) fn relu_inplace_dispatch(values: &mut [f32]) {
    if cfg!(miri) {
        for v in values.iter_mut() {
            *v = v.max(0.0);
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if yscv_cpu::host_cpu().features.avx {
            unsafe { relu_inplace_avx(values) };
            return;
        }
        if yscv_cpu::host_cpu().features.sse {
            unsafe { relu_inplace_sse(values) };
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if yscv_cpu::host_cpu().features.neon {
            unsafe { relu_inplace_neon(values) };
            return;
        }
    }

    for v in values.iter_mut() {
        *v = v.max(0.0);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn relu_inplace_sse(values: &mut [f32]) {
    let len = values.len();
    let ptr = values.as_mut_ptr();
    let zero = _mm_setzero_ps();
    let mut i = 0usize;

    while i + 4 <= len {
        let v = _mm_loadu_ps(ptr.add(i));
        _mm_storeu_ps(ptr.add(i), _mm_max_ps(v, zero));
        i += 4;
    }

    while i < len {
        let v = *ptr.add(i);
        *ptr.add(i) = if v > 0.0 { v } else { 0.0 };
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn relu_inplace_avx(values: &mut [f32]) {
    let len = values.len();
    let ptr = values.as_mut_ptr();
    let zero = _mm256_setzero_ps();
    let mut i = 0usize;

    while i + 8 <= len {
        let v = _mm256_loadu_ps(ptr.add(i));
        _mm256_storeu_ps(ptr.add(i), _mm256_max_ps(v, zero));
        i += 8;
    }

    if i < len {
        relu_inplace_sse(&mut values[i..]);
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn relu_inplace_neon(values: &mut [f32]) {
    let len = values.len();
    let ptr = values.as_mut_ptr();
    let zero = vdupq_n_f32(0.0);
    let mut i = 0usize;

    while i + 4 <= len {
        let v = vld1q_f32(ptr.add(i));
        vst1q_f32(ptr.add(i), vmaxq_f32(v, zero));
        i += 4;
    }

    while i < len {
        let v = *ptr.add(i);
        *ptr.add(i) = if v > 0.0 { v } else { 0.0 };
        i += 1;
    }
}

// ===========================================================================
// In-place scalar ops: v[i] += s, v[i] *= s
// ===========================================================================

#[allow(unsafe_code)]
#[inline]
pub(crate) fn add_scalar_inplace_dispatch(values: &mut [f32], s: f32) {
    if cfg!(miri) {
        for v in values.iter_mut() {
            *v += s;
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if yscv_cpu::host_cpu().features.avx {
            unsafe { add_scalar_inplace_avx(values, s) };
            return;
        }
        if yscv_cpu::host_cpu().features.sse {
            unsafe { add_scalar_inplace_sse(values, s) };
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if yscv_cpu::host_cpu().features.neon {
            unsafe { add_scalar_inplace_neon(values, s) };
            return;
        }
    }

    for v in values.iter_mut() {
        *v += s;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn add_scalar_inplace_sse(values: &mut [f32], s: f32) {
    let len = values.len();
    let ptr = values.as_mut_ptr();
    let sv = _mm_set1_ps(s);
    let mut i = 0usize;

    while i + 4 <= len {
        let v = _mm_loadu_ps(ptr.add(i));
        _mm_storeu_ps(ptr.add(i), _mm_add_ps(v, sv));
        i += 4;
    }
    while i < len {
        *ptr.add(i) += s;
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn add_scalar_inplace_avx(values: &mut [f32], s: f32) {
    let len = values.len();
    let ptr = values.as_mut_ptr();
    let sv = _mm256_set1_ps(s);
    let mut i = 0usize;

    while i + 8 <= len {
        let v = _mm256_loadu_ps(ptr.add(i));
        _mm256_storeu_ps(ptr.add(i), _mm256_add_ps(v, sv));
        i += 8;
    }
    if i < len {
        add_scalar_inplace_sse(&mut values[i..], s);
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn add_scalar_inplace_neon(values: &mut [f32], s: f32) {
    let len = values.len();
    let ptr = values.as_mut_ptr();
    let sv = vdupq_n_f32(s);
    let mut i = 0usize;

    while i + 4 <= len {
        let v = vld1q_f32(ptr.add(i));
        vst1q_f32(ptr.add(i), vaddq_f32(v, sv));
        i += 4;
    }
    while i < len {
        *ptr.add(i) += s;
        i += 1;
    }
}

#[allow(unsafe_code)]
#[inline]
pub(crate) fn mul_scalar_inplace_dispatch(values: &mut [f32], s: f32) {
    if cfg!(miri) {
        for v in values.iter_mut() {
            *v *= s;
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if yscv_cpu::host_cpu().features.avx {
            unsafe { mul_scalar_inplace_avx(values, s) };
            return;
        }
        if yscv_cpu::host_cpu().features.sse {
            unsafe { mul_scalar_inplace_sse(values, s) };
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if yscv_cpu::host_cpu().features.neon {
            unsafe { mul_scalar_inplace_neon(values, s) };
            return;
        }
    }

    for v in values.iter_mut() {
        *v *= s;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn mul_scalar_inplace_sse(values: &mut [f32], s: f32) {
    let len = values.len();
    let ptr = values.as_mut_ptr();
    let sv = _mm_set1_ps(s);
    let mut i = 0usize;

    while i + 4 <= len {
        let v = _mm_loadu_ps(ptr.add(i));
        _mm_storeu_ps(ptr.add(i), _mm_mul_ps(v, sv));
        i += 4;
    }
    while i < len {
        *ptr.add(i) *= s;
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn mul_scalar_inplace_avx(values: &mut [f32], s: f32) {
    let len = values.len();
    let ptr = values.as_mut_ptr();
    let sv = _mm256_set1_ps(s);
    let mut i = 0usize;

    while i + 8 <= len {
        let v = _mm256_loadu_ps(ptr.add(i));
        _mm256_storeu_ps(ptr.add(i), _mm256_mul_ps(v, sv));
        i += 8;
    }
    if i < len {
        mul_scalar_inplace_sse(&mut values[i..], s);
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn mul_scalar_inplace_neon(values: &mut [f32], s: f32) {
    let len = values.len();
    let ptr = values.as_mut_ptr();
    let sv = vdupq_n_f32(s);
    let mut i = 0usize;

    while i + 4 <= len {
        let v = vld1q_f32(ptr.add(i));
        vst1q_f32(ptr.add(i), vmulq_f32(v, sv));
        i += 4;
    }
    while i < len {
        *ptr.add(i) *= s;
        i += 1;
    }
}
