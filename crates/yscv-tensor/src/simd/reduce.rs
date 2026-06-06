//! Sum, max, min reductions with SIMD dispatch.

use super::*;

// ===========================================================================
// Sum reduction
// ===========================================================================

#[allow(unsafe_code)]
#[inline]
pub(crate) fn sum_dispatch(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    if cfg!(miri) {
        return sum_scalar(data);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if yscv_cpu::host_cpu().features.avx {
            return unsafe { sum_avx(data) };
        }
        if yscv_cpu::host_cpu().features.sse {
            return unsafe { sum_sse(data) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if yscv_cpu::host_cpu().features.neon {
            return unsafe { sum_neon(data) };
        }
    }

    sum_scalar(data)
}

fn sum_scalar(data: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for &v in data {
        acc += v;
    }
    acc
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn sum_sse(data: &[f32]) -> f32 {
    let len = data.len();
    let ptr = data.as_ptr();
    let mut i = 0usize;
    let mut acc = _mm_setzero_ps();

    while i + 4 <= len {
        acc = _mm_add_ps(acc, _mm_loadu_ps(ptr.add(i)));
        i += 4;
    }

    let mut buf = [0.0f32; 4];
    _mm_storeu_ps(buf.as_mut_ptr(), acc);
    let mut result = buf[0] + buf[1] + buf[2] + buf[3];

    while i < len {
        result += *ptr.add(i);
        i += 1;
    }
    result
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn sum_avx(data: &[f32]) -> f32 {
    let len = data.len();
    let ptr = data.as_ptr();
    let mut i = 0usize;
    let mut acc = _mm256_setzero_ps();

    while i + 8 <= len {
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(ptr.add(i)));
        i += 8;
    }

    let mut buf = [0.0f32; 8];
    _mm256_storeu_ps(buf.as_mut_ptr(), acc);
    let mut result = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];

    while i < len {
        result += *ptr.add(i);
        i += 1;
    }
    result
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn sum_neon(data: &[f32]) -> f32 {
    use std::arch::aarch64::vaddvq_f32;

    let len = data.len();
    let ptr = data.as_ptr();
    let mut i = 0usize;
    let mut acc = vdupq_n_f32(0.0);

    while i + 4 <= len {
        acc = vaddq_f32(acc, vld1q_f32(ptr.add(i)));
        i += 4;
    }

    let mut result = vaddvq_f32(acc);
    while i < len {
        result += *ptr.add(i);
        i += 1;
    }
    result
}

// ===========================================================================
// Max reduction
// ===========================================================================

#[allow(unsafe_code)]
#[inline]
pub(crate) fn max_dispatch(data: &[f32]) -> f32 {
    if data.is_empty() {
        return f32::NEG_INFINITY;
    }
    if cfg!(miri) {
        return max_scalar(data);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if yscv_cpu::host_cpu().features.avx {
            return unsafe { max_avx(data) };
        }
        if yscv_cpu::host_cpu().features.sse {
            return unsafe { max_sse(data) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if yscv_cpu::host_cpu().features.neon {
            return unsafe { max_neon(data) };
        }
    }

    max_scalar(data)
}

pub(super) fn max_scalar(data: &[f32]) -> f32 {
    let mut acc = f32::NEG_INFINITY;
    for &v in data {
        acc = acc.max(v);
    }
    acc
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn max_sse(data: &[f32]) -> f32 {
    let len = data.len();
    let ptr = data.as_ptr();
    let mut i = 0usize;
    let mut acc = _mm_set1_ps(f32::NEG_INFINITY);

    while i + 4 <= len {
        acc = _mm_max_ps(acc, _mm_loadu_ps(ptr.add(i)));
        i += 4;
    }

    let mut buf = [0.0f32; 4];
    _mm_storeu_ps(buf.as_mut_ptr(), acc);
    let mut result = buf[0].max(buf[1]).max(buf[2]).max(buf[3]);

    while i < len {
        result = result.max(*ptr.add(i));
        i += 1;
    }
    result
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn max_avx(data: &[f32]) -> f32 {
    let len = data.len();
    let ptr = data.as_ptr();
    let mut i = 0usize;
    let mut acc = _mm256_set1_ps(f32::NEG_INFINITY);

    while i + 8 <= len {
        acc = _mm256_max_ps(acc, _mm256_loadu_ps(ptr.add(i)));
        i += 8;
    }

    let mut buf = [0.0f32; 8];
    _mm256_storeu_ps(buf.as_mut_ptr(), acc);
    let mut result = buf[0];
    for j in 1..8 {
        result = result.max(buf[j]);
    }

    while i < len {
        result = result.max(*ptr.add(i));
        i += 1;
    }
    result
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn max_neon(data: &[f32]) -> f32 {
    use std::arch::aarch64::vmaxvq_f32;

    let len = data.len();
    let ptr = data.as_ptr();
    let mut i = 0usize;
    let mut acc = vdupq_n_f32(f32::NEG_INFINITY);

    while i + 4 <= len {
        acc = vmaxq_f32(acc, vld1q_f32(ptr.add(i)));
        i += 4;
    }

    let mut result = vmaxvq_f32(acc);
    while i < len {
        result = result.max(*ptr.add(i));
        i += 1;
    }
    result
}

// ===========================================================================
// Min reduction
// ===========================================================================

#[allow(unsafe_code)]
#[inline]
pub(crate) fn min_dispatch(data: &[f32]) -> f32 {
    if data.is_empty() {
        return f32::INFINITY;
    }
    if cfg!(miri) {
        return min_scalar(data);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if yscv_cpu::host_cpu().features.avx {
            return unsafe { min_avx(data) };
        }
        if yscv_cpu::host_cpu().features.sse {
            return unsafe { min_sse(data) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if yscv_cpu::host_cpu().features.neon {
            return unsafe { min_neon(data) };
        }
    }

    min_scalar(data)
}

pub(super) fn min_scalar(data: &[f32]) -> f32 {
    let mut acc = f32::INFINITY;
    for &v in data {
        acc = acc.min(v);
    }
    acc
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn min_sse(data: &[f32]) -> f32 {
    let len = data.len();
    let ptr = data.as_ptr();
    let mut i = 0usize;
    let mut acc = _mm_set1_ps(f32::INFINITY);

    while i + 4 <= len {
        acc = _mm_min_ps(acc, _mm_loadu_ps(ptr.add(i)));
        i += 4;
    }

    let mut buf = [0.0f32; 4];
    _mm_storeu_ps(buf.as_mut_ptr(), acc);
    let mut result = buf[0].min(buf[1]).min(buf[2]).min(buf[3]);

    while i < len {
        result = result.min(*ptr.add(i));
        i += 1;
    }
    result
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn min_avx(data: &[f32]) -> f32 {
    let len = data.len();
    let ptr = data.as_ptr();
    let mut i = 0usize;
    let mut acc = _mm256_set1_ps(f32::INFINITY);

    while i + 8 <= len {
        acc = _mm256_min_ps(acc, _mm256_loadu_ps(ptr.add(i)));
        i += 8;
    }

    let mut buf = [0.0f32; 8];
    _mm256_storeu_ps(buf.as_mut_ptr(), acc);
    let mut result = buf[0];
    for j in 1..8 {
        result = result.min(buf[j]);
    }

    while i < len {
        result = result.min(*ptr.add(i));
        i += 1;
    }
    result
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn min_neon(data: &[f32]) -> f32 {
    use std::arch::aarch64::vminvq_f32;

    let len = data.len();
    let ptr = data.as_ptr();
    let mut i = 0usize;
    let mut acc = vdupq_n_f32(f32::INFINITY);

    while i + 4 <= len {
        acc = vminq_f32(acc, vld1q_f32(ptr.add(i)));
        i += 4;
    }

    let mut result = vminvq_f32(acc);
    while i < len {
        result = result.min(*ptr.add(i));
        i += 1;
    }
    result
}
