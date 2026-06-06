// ===========================================================================
// max_reduce, add_reduce dispatchers + implementations
// ===========================================================================

use super::{SimdDispatchPath, dispatch_path};
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{vaddq_f32, vdupq_n_f32, vld1q_f32, vmaxq_f32};
#[cfg(target_arch = "x86")]
use std::arch::x86::{
    _mm_add_ps, _mm_loadu_ps, _mm_max_ps, _mm_set1_ps, _mm_setzero_ps, _mm_storeu_ps,
    _mm256_add_ps, _mm256_loadu_ps, _mm256_max_ps, _mm256_set1_ps, _mm256_setzero_ps,
    _mm256_storeu_ps,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _mm_add_ps, _mm_loadu_ps, _mm_max_ps, _mm_set1_ps, _mm_setzero_ps, _mm_storeu_ps,
    _mm256_add_ps, _mm256_loadu_ps, _mm256_max_ps, _mm256_set1_ps, _mm256_setzero_ps,
    _mm256_storeu_ps,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _mm512_add_ps, _mm512_loadu_ps, _mm512_max_ps, _mm512_reduce_add_ps, _mm512_reduce_max_ps,
    _mm512_set1_ps, _mm512_setzero_ps,
};

// ===========================================================================
// Dispatchers
// ===========================================================================

/// Find the maximum value in `data`.  Returns `f32::NEG_INFINITY` for empty slices.
#[allow(unsafe_code, dead_code)]
#[inline]
pub fn max_reduce_dispatch(data: &[f32]) -> f32 {
    if data.is_empty() {
        return f32::NEG_INFINITY;
    }

    if cfg!(miri) {
        return max_reduce_scalar(data);
    }

    let path = dispatch_path(true, false);

    #[cfg(target_arch = "x86_64")]
    if path == SimdDispatchPath::Avx512 {
        // SAFETY: guarded by runtime feature detection in `dispatch_path`.
        return unsafe { max_reduce_avx512(data) };
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if path == SimdDispatchPath::Avx {
            // SAFETY: guarded by runtime feature detection in `dispatch_path`.
            return unsafe { max_reduce_avx(data) };
        }
        if path == SimdDispatchPath::Sse {
            // SAFETY: guarded by runtime feature detection in `dispatch_path`.
            return unsafe { max_reduce_sse(data) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if path == SimdDispatchPath::Neon {
            // SAFETY: guarded by runtime feature detection in `dispatch_path`.
            return unsafe { max_reduce_neon(data) };
        }
    }

    max_reduce_scalar(data)
}

/// Sum all values in `data`.  Returns `0.0` for empty slices.
#[allow(unsafe_code, dead_code)]
#[inline]
pub fn add_reduce_dispatch(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }

    if cfg!(miri) {
        return add_reduce_scalar(data);
    }

    let path = dispatch_path(true, false);

    #[cfg(target_arch = "x86_64")]
    if path == SimdDispatchPath::Avx512 {
        // SAFETY: guarded by runtime feature detection in `dispatch_path`.
        return unsafe { add_reduce_avx512(data) };
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if path == SimdDispatchPath::Avx {
            // SAFETY: guarded by runtime feature detection in `dispatch_path`.
            return unsafe { add_reduce_avx(data) };
        }
        if path == SimdDispatchPath::Sse {
            // SAFETY: guarded by runtime feature detection in `dispatch_path`.
            return unsafe { add_reduce_sse(data) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if path == SimdDispatchPath::Neon {
            // SAFETY: guarded by runtime feature detection in `dispatch_path`.
            return unsafe { add_reduce_neon(data) };
        }
    }

    add_reduce_scalar(data)
}

// ===========================================================================
// Scalar fallbacks
// ===========================================================================

#[allow(dead_code)]
pub(super) fn max_reduce_scalar(data: &[f32]) -> f32 {
    let mut acc = f32::NEG_INFINITY;
    for &v in data {
        acc = acc.max(v);
    }
    acc
}

#[allow(dead_code)]
pub(super) fn add_reduce_scalar(data: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for &v in data {
        acc += v;
    }
    acc
}

// ===========================================================================
// Max-reduce implementations
// ===========================================================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn max_reduce_sse(data: &[f32]) -> f32 {
    let len = data.len();
    let ptr = data.as_ptr();
    let mut index = 0usize;
    let mut acc0 = _mm_set1_ps(f32::NEG_INFINITY);
    let mut acc1 = _mm_set1_ps(f32::NEG_INFINITY);
    let mut acc2 = _mm_set1_ps(f32::NEG_INFINITY);
    let mut acc3 = _mm_set1_ps(f32::NEG_INFINITY);

    while index + 16 <= len {
        acc0 = _mm_max_ps(acc0, _mm_loadu_ps(ptr.add(index)));
        acc1 = _mm_max_ps(acc1, _mm_loadu_ps(ptr.add(index + 4)));
        acc2 = _mm_max_ps(acc2, _mm_loadu_ps(ptr.add(index + 8)));
        acc3 = _mm_max_ps(acc3, _mm_loadu_ps(ptr.add(index + 12)));
        index += 16;
    }
    while index + 4 <= len {
        acc0 = _mm_max_ps(acc0, _mm_loadu_ps(ptr.add(index)));
        index += 4;
    }

    // Horizontal max of 4-lane accumulator
    let mut buf = [0.0f32; 4];
    let acc = _mm_max_ps(_mm_max_ps(acc0, acc1), _mm_max_ps(acc2, acc3));
    _mm_storeu_ps(buf.as_mut_ptr(), acc);
    let mut result = buf[0].max(buf[1]).max(buf[2]).max(buf[3]);

    while index < len {
        result = result.max(*ptr.add(index));
        index += 1;
    }
    result
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn max_reduce_avx(data: &[f32]) -> f32 {
    let len = data.len();
    let ptr = data.as_ptr();
    let mut index = 0usize;
    let mut acc0 = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut acc1 = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut acc2 = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut acc3 = _mm256_set1_ps(f32::NEG_INFINITY);

    while index + 32 <= len {
        acc0 = _mm256_max_ps(acc0, _mm256_loadu_ps(ptr.add(index)));
        acc1 = _mm256_max_ps(acc1, _mm256_loadu_ps(ptr.add(index + 8)));
        acc2 = _mm256_max_ps(acc2, _mm256_loadu_ps(ptr.add(index + 16)));
        acc3 = _mm256_max_ps(acc3, _mm256_loadu_ps(ptr.add(index + 24)));
        index += 32;
    }
    while index + 8 <= len {
        acc0 = _mm256_max_ps(acc0, _mm256_loadu_ps(ptr.add(index)));
        index += 8;
    }

    // Horizontal max of 8-lane accumulator
    let mut buf = [0.0f32; 8];
    let acc = _mm256_max_ps(_mm256_max_ps(acc0, acc1), _mm256_max_ps(acc2, acc3));
    _mm256_storeu_ps(buf.as_mut_ptr(), acc);
    let mut result = buf[0];
    for i in 1..8 {
        result = result.max(buf[i]);
    }

    while index < len {
        result = result.max(*ptr.add(index));
        index += 1;
    }
    result
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx512f")]
unsafe fn max_reduce_avx512(data: &[f32]) -> f32 {
    let len = data.len();
    let ptr = data.as_ptr();
    let mut index = 0usize;
    let mut acc0 = _mm512_set1_ps(f32::NEG_INFINITY);
    let mut acc1 = _mm512_set1_ps(f32::NEG_INFINITY);
    let mut acc2 = _mm512_set1_ps(f32::NEG_INFINITY);
    let mut acc3 = _mm512_set1_ps(f32::NEG_INFINITY);

    while index + 64 <= len {
        acc0 = _mm512_max_ps(acc0, _mm512_loadu_ps(ptr.add(index)));
        acc1 = _mm512_max_ps(acc1, _mm512_loadu_ps(ptr.add(index + 16)));
        acc2 = _mm512_max_ps(acc2, _mm512_loadu_ps(ptr.add(index + 32)));
        acc3 = _mm512_max_ps(acc3, _mm512_loadu_ps(ptr.add(index + 48)));
        index += 64;
    }
    while index + 16 <= len {
        acc0 = _mm512_max_ps(acc0, _mm512_loadu_ps(ptr.add(index)));
        index += 16;
    }

    let acc = _mm512_max_ps(_mm512_max_ps(acc0, acc1), _mm512_max_ps(acc2, acc3));
    let mut result = _mm512_reduce_max_ps(acc);
    while index < len {
        result = result.max(*ptr.add(index));
        index += 1;
    }
    result
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, dead_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn max_reduce_neon(data: &[f32]) -> f32 {
    use std::arch::aarch64::vmaxvq_f32;

    let len = data.len();
    let ptr = data.as_ptr();
    let mut index = 0usize;
    let mut acc0 = vdupq_n_f32(f32::NEG_INFINITY);
    let mut acc1 = vdupq_n_f32(f32::NEG_INFINITY);
    let mut acc2 = vdupq_n_f32(f32::NEG_INFINITY);
    let mut acc3 = vdupq_n_f32(f32::NEG_INFINITY);

    while index + 16 <= len {
        acc0 = vmaxq_f32(acc0, vld1q_f32(ptr.add(index)));
        acc1 = vmaxq_f32(acc1, vld1q_f32(ptr.add(index + 4)));
        acc2 = vmaxq_f32(acc2, vld1q_f32(ptr.add(index + 8)));
        acc3 = vmaxq_f32(acc3, vld1q_f32(ptr.add(index + 12)));
        index += 16;
    }
    while index + 4 <= len {
        acc0 = vmaxq_f32(acc0, vld1q_f32(ptr.add(index)));
        index += 4;
    }

    let acc = vmaxq_f32(vmaxq_f32(acc0, acc1), vmaxq_f32(acc2, acc3));
    let mut result = vmaxvq_f32(acc);
    while index < len {
        result = result.max(*ptr.add(index));
        index += 1;
    }
    result
}

// ===========================================================================
// Add-reduce (sum) implementations
// ===========================================================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn add_reduce_sse(data: &[f32]) -> f32 {
    let len = data.len();
    let ptr = data.as_ptr();
    let mut index = 0usize;
    let mut acc0 = _mm_setzero_ps();
    let mut acc1 = _mm_setzero_ps();
    let mut acc2 = _mm_setzero_ps();
    let mut acc3 = _mm_setzero_ps();

    while index + 16 <= len {
        acc0 = _mm_add_ps(acc0, _mm_loadu_ps(ptr.add(index)));
        acc1 = _mm_add_ps(acc1, _mm_loadu_ps(ptr.add(index + 4)));
        acc2 = _mm_add_ps(acc2, _mm_loadu_ps(ptr.add(index + 8)));
        acc3 = _mm_add_ps(acc3, _mm_loadu_ps(ptr.add(index + 12)));
        index += 16;
    }
    while index + 4 <= len {
        acc0 = _mm_add_ps(acc0, _mm_loadu_ps(ptr.add(index)));
        index += 4;
    }

    // Horizontal sum of 4-lane accumulator
    let mut buf = [0.0f32; 4];
    let acc = _mm_add_ps(_mm_add_ps(acc0, acc1), _mm_add_ps(acc2, acc3));
    _mm_storeu_ps(buf.as_mut_ptr(), acc);
    let mut result = buf[0] + buf[1] + buf[2] + buf[3];

    while index < len {
        result += *ptr.add(index);
        index += 1;
    }
    result
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn add_reduce_avx(data: &[f32]) -> f32 {
    let len = data.len();
    let ptr = data.as_ptr();
    let mut index = 0usize;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    while index + 32 <= len {
        acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(ptr.add(index)));
        acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps(ptr.add(index + 8)));
        acc2 = _mm256_add_ps(acc2, _mm256_loadu_ps(ptr.add(index + 16)));
        acc3 = _mm256_add_ps(acc3, _mm256_loadu_ps(ptr.add(index + 24)));
        index += 32;
    }
    while index + 8 <= len {
        acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(ptr.add(index)));
        index += 8;
    }

    let mut buf = [0.0f32; 8];
    let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    _mm256_storeu_ps(buf.as_mut_ptr(), acc);
    let mut result = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];

    while index < len {
        result += *ptr.add(index);
        index += 1;
    }
    result
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx512f")]
unsafe fn add_reduce_avx512(data: &[f32]) -> f32 {
    let len = data.len();
    let ptr = data.as_ptr();
    let mut index = 0usize;
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();

    while index + 64 <= len {
        acc0 = _mm512_add_ps(acc0, _mm512_loadu_ps(ptr.add(index)));
        acc1 = _mm512_add_ps(acc1, _mm512_loadu_ps(ptr.add(index + 16)));
        acc2 = _mm512_add_ps(acc2, _mm512_loadu_ps(ptr.add(index + 32)));
        acc3 = _mm512_add_ps(acc3, _mm512_loadu_ps(ptr.add(index + 48)));
        index += 64;
    }
    while index + 16 <= len {
        acc0 = _mm512_add_ps(acc0, _mm512_loadu_ps(ptr.add(index)));
        index += 16;
    }

    let acc = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
    let mut result = _mm512_reduce_add_ps(acc);
    while index < len {
        result += *ptr.add(index);
        index += 1;
    }
    result
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, dead_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn add_reduce_neon(data: &[f32]) -> f32 {
    use std::arch::aarch64::vaddvq_f32;

    let len = data.len();
    let ptr = data.as_ptr();
    let mut index = 0usize;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);

    while index + 16 <= len {
        acc0 = vaddq_f32(acc0, vld1q_f32(ptr.add(index)));
        acc1 = vaddq_f32(acc1, vld1q_f32(ptr.add(index + 4)));
        acc2 = vaddq_f32(acc2, vld1q_f32(ptr.add(index + 8)));
        acc3 = vaddq_f32(acc3, vld1q_f32(ptr.add(index + 12)));
        index += 16;
    }
    while index + 4 <= len {
        acc0 = vaddq_f32(acc0, vld1q_f32(ptr.add(index)));
        index += 4;
    }

    let acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    let mut result = vaddvq_f32(acc);
    while index < len {
        result += *ptr.add(index);
        index += 1;
    }
    result
}
