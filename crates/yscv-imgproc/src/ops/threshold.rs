use rayon::prelude::*;
use yscv_tensor::{AlignedVec, Tensor};

use super::super::ImgProcError;
use super::super::shape::hwc_shape;

use super::geometry::sobel_3x3_gradients;

const RAYON_THRESHOLD: usize = 4096;

/// Binary threshold: outputs `max_val` where value > threshold, else `0.0`.
#[allow(unsafe_code, clippy::uninit_vec)]
pub fn threshold_binary(
    input: &Tensor,
    threshold: f32,
    max_val: f32,
) -> Result<Tensor, ImgProcError> {
    let (h, w, channels) = hwc_shape(input)?;
    let data = input.data();
    let len = data.len();
    // SAFETY: every element is written by threshold_binary_simd_slice + scalar tail below.
    let mut out = AlignedVec::<f32>::uninitialized(len);

    let row_len = w * channels;

    #[cfg(target_os = "macos")]
    if len >= RAYON_THRESHOLD && !cfg!(miri) {
        let src_ptr = super::SendConstPtr(data.as_ptr());
        let dst_ptr = super::SendPtr(out.as_mut_ptr());
        use super::u8ops::gcd;
        gcd::parallel_for(h, |y| {
            let src =
                unsafe { std::slice::from_raw_parts(src_ptr.ptr().add(y * row_len), row_len) };
            let dst =
                unsafe { std::slice::from_raw_parts_mut(dst_ptr.ptr().add(y * row_len), row_len) };
            threshold_binary_simd_slice(src, dst, threshold, max_val);
        });
        return Tensor::from_aligned(vec![h, w, channels], out).map_err(Into::into);
    }

    if len >= RAYON_THRESHOLD {
        out.par_chunks_mut(row_len)
            .enumerate()
            .for_each(|(y, dst)| {
                let src = &data[y * row_len..(y + 1) * row_len];
                threshold_binary_simd_slice(src, dst, threshold, max_val);
            });
    } else {
        threshold_binary_simd_slice(data, &mut out, threshold, max_val);
    }
    Tensor::from_aligned(vec![h, w, channels], out).map_err(Into::into)
}

/// Inverse binary threshold: outputs `0.0` where value > threshold, else `max_val`.
#[allow(unsafe_code, clippy::uninit_vec)]
pub fn threshold_binary_inv(
    input: &Tensor,
    threshold: f32,
    max_val: f32,
) -> Result<Tensor, ImgProcError> {
    let (h, w, channels) = hwc_shape(input)?;
    let data = input.data();
    let len = data.len();
    // SAFETY: every element is written by threshold_binary_inv_simd_slice + scalar tail below.
    let mut out = AlignedVec::<f32>::uninitialized(len);

    let row_len = w * channels;

    #[cfg(target_os = "macos")]
    if len >= RAYON_THRESHOLD && !cfg!(miri) {
        let src_ptr = super::SendConstPtr(data.as_ptr());
        let dst_ptr = super::SendPtr(out.as_mut_ptr());
        use super::u8ops::gcd;
        gcd::parallel_for(h, |y| {
            let src =
                unsafe { std::slice::from_raw_parts(src_ptr.ptr().add(y * row_len), row_len) };
            let dst =
                unsafe { std::slice::from_raw_parts_mut(dst_ptr.ptr().add(y * row_len), row_len) };
            threshold_binary_inv_simd_slice(src, dst, threshold, max_val);
        });
        return Tensor::from_aligned(vec![h, w, channels], out).map_err(Into::into);
    }

    if len >= RAYON_THRESHOLD {
        out.par_chunks_mut(row_len)
            .enumerate()
            .for_each(|(y, dst)| {
                let src = &data[y * row_len..(y + 1) * row_len];
                threshold_binary_inv_simd_slice(src, dst, threshold, max_val);
            });
    } else {
        threshold_binary_inv_simd_slice(data, &mut out, threshold, max_val);
    }
    Tensor::from_aligned(vec![h, w, channels], out).map_err(Into::into)
}

/// Truncate threshold: caps values above `threshold`.
#[allow(unsafe_code, clippy::uninit_vec)]
pub fn threshold_truncate(input: &Tensor, threshold: f32) -> Result<Tensor, ImgProcError> {
    let (h, w, channels) = hwc_shape(input)?;
    let data = input.data();
    let len = data.len();
    // SAFETY: every element is written by threshold_truncate_simd_slice + scalar tail below.
    let mut out = AlignedVec::<f32>::uninitialized(len);

    let row_len = w * channels;

    #[cfg(target_os = "macos")]
    if len >= RAYON_THRESHOLD && !cfg!(miri) {
        let src_ptr = super::SendConstPtr(data.as_ptr());
        let dst_ptr = super::SendPtr(out.as_mut_ptr());
        use super::u8ops::gcd;
        gcd::parallel_for(h, |y| {
            let src =
                unsafe { std::slice::from_raw_parts(src_ptr.ptr().add(y * row_len), row_len) };
            let dst =
                unsafe { std::slice::from_raw_parts_mut(dst_ptr.ptr().add(y * row_len), row_len) };
            threshold_truncate_simd_slice(src, dst, threshold);
        });
        return Tensor::from_aligned(vec![h, w, channels], out).map_err(Into::into);
    }

    if len >= RAYON_THRESHOLD {
        out.par_chunks_mut(row_len)
            .enumerate()
            .for_each(|(y, dst)| {
                let src = &data[y * row_len..(y + 1) * row_len];
                threshold_truncate_simd_slice(src, dst, threshold);
            });
    } else {
        threshold_truncate_simd_slice(data, &mut out, threshold);
    }
    Tensor::from_aligned(vec![h, w, channels], out).map_err(Into::into)
}

/// SIMD-accelerated binary threshold for an entire slice.
#[allow(unsafe_code)]
#[inline(always)]
fn threshold_binary_simd_slice(src: &[f32], dst: &mut [f32], threshold: f32, max_val: f32) {
    debug_assert_eq!(src.len(), dst.len());
    let len = src.len();
    let mut i = 0usize;

    if !cfg!(miri) {
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                // SAFETY: feature detected; pointers valid for len elements.
                i = unsafe {
                    threshold_binary_neon(src.as_ptr(), dst.as_mut_ptr(), len, threshold, max_val)
                };
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx") {
                // SAFETY: feature detected; pointers valid for len elements.
                i = unsafe {
                    threshold_binary_avx(src.as_ptr(), dst.as_mut_ptr(), len, threshold, max_val)
                };
            } else if std::is_x86_feature_detected!("sse") {
                // SAFETY: feature detected; pointers valid for len elements.
                i = unsafe {
                    threshold_binary_sse(src.as_ptr(), dst.as_mut_ptr(), len, threshold, max_val)
                };
            }
        }
    }

    // Scalar tail
    while i < len {
        dst[i] = if src[i] > threshold { max_val } else { 0.0 };
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn threshold_binary_neon(
    src: *const f32,
    dst: *mut f32,
    len: usize,
    threshold: f32,
    max_val: f32,
) -> usize {
    use std::arch::aarch64::*;
    let thresh_v = vdupq_n_f32(threshold);
    let max_v = vdupq_n_f32(max_val);
    let zero_v = vdupq_n_f32(0.0);
    let mut x = 0usize;
    // Process 32 floats (8×4) per iteration for better throughput
    while x + 32 <= len {
        let v0 = vld1q_f32(src.add(x));
        let v1 = vld1q_f32(src.add(x + 4));
        let v2 = vld1q_f32(src.add(x + 8));
        let v3 = vld1q_f32(src.add(x + 12));
        let v4 = vld1q_f32(src.add(x + 16));
        let v5 = vld1q_f32(src.add(x + 20));
        let v6 = vld1q_f32(src.add(x + 24));
        let v7 = vld1q_f32(src.add(x + 28));
        vst1q_f32(
            dst.add(x),
            vbslq_f32(vcgtq_f32(v0, thresh_v), max_v, zero_v),
        );
        vst1q_f32(
            dst.add(x + 4),
            vbslq_f32(vcgtq_f32(v1, thresh_v), max_v, zero_v),
        );
        vst1q_f32(
            dst.add(x + 8),
            vbslq_f32(vcgtq_f32(v2, thresh_v), max_v, zero_v),
        );
        vst1q_f32(
            dst.add(x + 12),
            vbslq_f32(vcgtq_f32(v3, thresh_v), max_v, zero_v),
        );
        vst1q_f32(
            dst.add(x + 16),
            vbslq_f32(vcgtq_f32(v4, thresh_v), max_v, zero_v),
        );
        vst1q_f32(
            dst.add(x + 20),
            vbslq_f32(vcgtq_f32(v5, thresh_v), max_v, zero_v),
        );
        vst1q_f32(
            dst.add(x + 24),
            vbslq_f32(vcgtq_f32(v6, thresh_v), max_v, zero_v),
        );
        vst1q_f32(
            dst.add(x + 28),
            vbslq_f32(vcgtq_f32(v7, thresh_v), max_v, zero_v),
        );
        x += 32;
    }
    while x + 4 <= len {
        let v = vld1q_f32(src.add(x));
        let mask = vcgtq_f32(v, thresh_v);
        let result = vbslq_f32(mask, max_v, zero_v);
        vst1q_f32(dst.add(x), result);
        x += 4;
    }
    x
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn threshold_binary_avx(
    src: *const f32,
    dst: *mut f32,
    len: usize,
    threshold: f32,
    max_val: f32,
) -> usize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let thresh_v = _mm256_set1_ps(threshold);
    let max_v = _mm256_set1_ps(max_val);
    let mut x = 0usize;
    // 4× unrolled: 32 elements per iteration
    while x + 32 <= len {
        let v0 = _mm256_loadu_ps(src.add(x));
        let v1 = _mm256_loadu_ps(src.add(x + 8));
        let v2 = _mm256_loadu_ps(src.add(x + 16));
        let v3 = _mm256_loadu_ps(src.add(x + 24));
        _mm256_storeu_ps(
            dst.add(x),
            _mm256_and_ps(_mm256_cmp_ps::<14>(v0, thresh_v), max_v),
        );
        _mm256_storeu_ps(
            dst.add(x + 8),
            _mm256_and_ps(_mm256_cmp_ps::<14>(v1, thresh_v), max_v),
        );
        _mm256_storeu_ps(
            dst.add(x + 16),
            _mm256_and_ps(_mm256_cmp_ps::<14>(v2, thresh_v), max_v),
        );
        _mm256_storeu_ps(
            dst.add(x + 24),
            _mm256_and_ps(_mm256_cmp_ps::<14>(v3, thresh_v), max_v),
        );
        x += 32;
    }
    while x + 8 <= len {
        _mm256_storeu_ps(
            dst.add(x),
            _mm256_and_ps(
                _mm256_cmp_ps::<14>(_mm256_loadu_ps(src.add(x)), thresh_v),
                max_v,
            ),
        );
        x += 8;
    }
    x
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn threshold_binary_sse(
    src: *const f32,
    dst: *mut f32,
    len: usize,
    threshold: f32,
    max_val: f32,
) -> usize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let thresh_v = _mm_set1_ps(threshold);
    let max_v = _mm_set1_ps(max_val);
    let mut x = 0usize;
    while x + 4 <= len {
        let v = _mm_loadu_ps(src.add(x));
        let mask = _mm_cmpgt_ps(v, thresh_v);
        let result = _mm_and_ps(mask, max_v);
        _mm_storeu_ps(dst.add(x), result);
        x += 4;
    }
    x
}

/// SIMD-accelerated inverse binary threshold for an entire slice.
#[allow(unsafe_code)]
#[inline(always)]
fn threshold_binary_inv_simd_slice(src: &[f32], dst: &mut [f32], threshold: f32, max_val: f32) {
    debug_assert_eq!(src.len(), dst.len());
    let len = src.len();
    let mut i = 0usize;

    if !cfg!(miri) {
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                // SAFETY: feature detected; pointers valid for len elements.
                i = unsafe {
                    threshold_binary_inv_neon(
                        src.as_ptr(),
                        dst.as_mut_ptr(),
                        len,
                        threshold,
                        max_val,
                    )
                };
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx") {
                // SAFETY: feature detected; pointers valid for len elements.
                i = unsafe {
                    threshold_binary_inv_avx(
                        src.as_ptr(),
                        dst.as_mut_ptr(),
                        len,
                        threshold,
                        max_val,
                    )
                };
            } else if std::is_x86_feature_detected!("sse") {
                // SAFETY: feature detected; pointers valid for len elements.
                i = unsafe {
                    threshold_binary_inv_sse(
                        src.as_ptr(),
                        dst.as_mut_ptr(),
                        len,
                        threshold,
                        max_val,
                    )
                };
            }
        }
    }

    // Scalar tail
    while i < len {
        dst[i] = if src[i] > threshold { 0.0 } else { max_val };
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn threshold_binary_inv_neon(
    src: *const f32,
    dst: *mut f32,
    len: usize,
    threshold: f32,
    max_val: f32,
) -> usize {
    use std::arch::aarch64::*;
    let thresh_v = vdupq_n_f32(threshold);
    let max_v = vdupq_n_f32(max_val);
    let zero_v = vdupq_n_f32(0.0);
    let mut x = 0usize;
    // Process 16 floats (4x4) per iteration for better throughput
    while x + 16 <= len {
        let v0 = vld1q_f32(src.add(x));
        let v1 = vld1q_f32(src.add(x + 4));
        let v2 = vld1q_f32(src.add(x + 8));
        let v3 = vld1q_f32(src.add(x + 12));
        vst1q_f32(
            dst.add(x),
            vbslq_f32(vcgtq_f32(v0, thresh_v), zero_v, max_v),
        );
        vst1q_f32(
            dst.add(x + 4),
            vbslq_f32(vcgtq_f32(v1, thresh_v), zero_v, max_v),
        );
        vst1q_f32(
            dst.add(x + 8),
            vbslq_f32(vcgtq_f32(v2, thresh_v), zero_v, max_v),
        );
        vst1q_f32(
            dst.add(x + 12),
            vbslq_f32(vcgtq_f32(v3, thresh_v), zero_v, max_v),
        );
        x += 16;
    }
    while x + 4 <= len {
        let v = vld1q_f32(src.add(x));
        let mask = vcgtq_f32(v, thresh_v);
        let result = vbslq_f32(mask, zero_v, max_v);
        vst1q_f32(dst.add(x), result);
        x += 4;
    }
    x
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn threshold_binary_inv_avx(
    src: *const f32,
    dst: *mut f32,
    len: usize,
    threshold: f32,
    max_val: f32,
) -> usize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let thresh_v = _mm256_set1_ps(threshold);
    let max_v = _mm256_set1_ps(max_val);
    let mut x = 0usize;
    while x + 8 <= len {
        let v = _mm256_loadu_ps(src.add(x));
        // _CMP_GT_OQ = 14: greater-than, ordered, quiet
        let mask = _mm256_cmp_ps::<14>(v, thresh_v);
        let result = _mm256_andnot_ps(mask, max_v);
        _mm256_storeu_ps(dst.add(x), result);
        x += 8;
    }
    x
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn threshold_binary_inv_sse(
    src: *const f32,
    dst: *mut f32,
    len: usize,
    threshold: f32,
    max_val: f32,
) -> usize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let thresh_v = _mm_set1_ps(threshold);
    let max_v = _mm_set1_ps(max_val);
    let mut x = 0usize;
    while x + 4 <= len {
        let v = _mm_loadu_ps(src.add(x));
        let mask = _mm_cmpgt_ps(v, thresh_v);
        let result = _mm_andnot_ps(mask, max_v);
        _mm_storeu_ps(dst.add(x), result);
        x += 4;
    }
    x
}

/// SIMD-accelerated truncate threshold for an entire slice.
#[allow(unsafe_code)]
#[inline(always)]
fn threshold_truncate_simd_slice(src: &[f32], dst: &mut [f32], threshold: f32) {
    debug_assert_eq!(src.len(), dst.len());
    let len = src.len();
    let mut i = 0usize;

    if !cfg!(miri) {
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                // SAFETY: feature detected; pointers valid for len elements.
                i = unsafe {
                    threshold_truncate_neon(src.as_ptr(), dst.as_mut_ptr(), len, threshold)
                };
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx") {
                // SAFETY: feature detected; pointers valid for len elements.
                i = unsafe {
                    threshold_truncate_avx(src.as_ptr(), dst.as_mut_ptr(), len, threshold)
                };
            } else if std::is_x86_feature_detected!("sse") {
                // SAFETY: feature detected; pointers valid for len elements.
                i = unsafe {
                    threshold_truncate_sse(src.as_ptr(), dst.as_mut_ptr(), len, threshold)
                };
            }
        }
    }

    // Scalar tail
    while i < len {
        dst[i] = if src[i] > threshold {
            threshold
        } else {
            src[i]
        };
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn threshold_truncate_neon(
    src: *const f32,
    dst: *mut f32,
    len: usize,
    threshold: f32,
) -> usize {
    use std::arch::aarch64::*;
    let thresh_v = vdupq_n_f32(threshold);
    let mut x = 0usize;
    // Process 16 floats (4x4) per iteration for better throughput
    while x + 16 <= len {
        let v0 = vld1q_f32(src.add(x));
        let v1 = vld1q_f32(src.add(x + 4));
        let v2 = vld1q_f32(src.add(x + 8));
        let v3 = vld1q_f32(src.add(x + 12));
        vst1q_f32(dst.add(x), vminq_f32(v0, thresh_v));
        vst1q_f32(dst.add(x + 4), vminq_f32(v1, thresh_v));
        vst1q_f32(dst.add(x + 8), vminq_f32(v2, thresh_v));
        vst1q_f32(dst.add(x + 12), vminq_f32(v3, thresh_v));
        x += 16;
    }
    while x + 4 <= len {
        let v = vld1q_f32(src.add(x));
        vst1q_f32(dst.add(x), vminq_f32(v, thresh_v));
        x += 4;
    }
    x
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn threshold_truncate_avx(
    src: *const f32,
    dst: *mut f32,
    len: usize,
    threshold: f32,
) -> usize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let thresh_v = _mm256_set1_ps(threshold);
    let mut x = 0usize;
    while x + 8 <= len {
        let v = _mm256_loadu_ps(src.add(x));
        let result = _mm256_min_ps(v, thresh_v);
        _mm256_storeu_ps(dst.add(x), result);
        x += 8;
    }
    x
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn threshold_truncate_sse(
    src: *const f32,
    dst: *mut f32,
    len: usize,
    threshold: f32,
) -> usize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let thresh_v = _mm_set1_ps(threshold);
    let mut x = 0usize;
    while x + 4 <= len {
        let v = _mm_loadu_ps(src.add(x));
        let result = _mm_min_ps(v, thresh_v);
        _mm_storeu_ps(dst.add(x), result);
        x += 4;
    }
    x
}

/// Otsu threshold for single-channel `[H, W, 1]` images.
/// Returns `(threshold, thresholded_image)`.
pub fn threshold_otsu(input: &Tensor, max_val: f32) -> Result<(f32, Tensor), ImgProcError> {
    let (_h, _w, channels) = hwc_shape(input)?;
    if channels != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: channels,
        });
    }

    // Build 256-bin histogram over [0, 1] range
    let data = input.data();
    let total = data.len() as f32;
    let mut hist = [0u32; 256];
    for &v in data {
        let bin = (v.clamp(0.0, 1.0) * 255.0) as usize;
        hist[bin.min(255)] += 1;
    }

    let mut sum_total = 0.0f64;
    for (i, &count) in hist.iter().enumerate() {
        sum_total += i as f64 * count as f64;
    }

    let mut sum_bg = 0.0f64;
    let mut weight_bg = 0.0f64;
    let mut max_variance = 0.0f64;
    let mut best_t = 0usize;

    for (t, &count) in hist.iter().enumerate() {
        weight_bg += count as f64;
        if weight_bg == 0.0 {
            continue;
        }
        let weight_fg = total as f64 - weight_bg;
        if weight_fg == 0.0 {
            break;
        }

        sum_bg += t as f64 * count as f64;
        let mean_bg = sum_bg / weight_bg;
        let mean_fg = (sum_total - sum_bg) / weight_fg;
        let diff = mean_bg - mean_fg;
        let variance = weight_bg * weight_fg * diff * diff;

        if variance > max_variance {
            max_variance = variance;
            best_t = t;
        }
    }

    let threshold = best_t as f32 / 255.0;
    let thresholded = threshold_binary(input, threshold, max_val)?;
    Ok((threshold, thresholded))
}

/// Canny edge detection on a single-channel HWC image.
///
/// Steps: Sobel gradients -> non-maximum suppression -> double-threshold hysteresis.
/// Returns a binary edge map with values `0.0` or `1.0`.
/// Reusable scratch buffers for [`canny_with_scratch`].
///
/// Pre-allocating these avoids per-call allocation overhead in hot loops
/// (e.g., processing video frames).
pub struct CannyScratch {
    magnitude: Vec<f32>,
    direction: Vec<u8>,
    nms: Vec<f32>,
    edges: Vec<u8>,
    queue: Vec<usize>,
}

impl CannyScratch {
    /// Creates empty scratch (buffers grow on first use).
    pub fn new() -> Self {
        Self {
            magnitude: Vec::new(),
            direction: Vec::new(),
            nms: Vec::new(),
            edges: Vec::new(),
            queue: Vec::new(),
        }
    }

    fn ensure_capacity(&mut self, len: usize) {
        self.magnitude.resize(len, 0.0);
        self.direction.resize(len, 0);
        self.nms.resize(len, 0.0);
        self.edges.resize(len, 0);
    }
}

impl Default for CannyScratch {
    fn default() -> Self {
        Self::new()
    }
}

pub fn canny(input: &Tensor, low_thresh: f32, high_thresh: f32) -> Result<Tensor, ImgProcError> {
    let mut scratch = CannyScratch::new();
    canny_with_scratch(input, low_thresh, high_thresh, &mut scratch)
}

/// Canny edge detection with reusable scratch buffers.
///
/// Optimized with:
/// - Alpha-max-beta-min magnitude approximation (no sqrt)
/// - Fast gradient direction via sign comparison (no atan2)
/// - Single-pass BFS hysteresis (no iterative convergence loop)
pub fn canny_with_scratch(
    input: &Tensor,
    low_thresh: f32,
    high_thresh: f32,
    scratch: &mut CannyScratch,
) -> Result<Tensor, ImgProcError> {
    let (_h, _w, channels) = hwc_shape(input)?;
    if channels != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: channels,
        });
    }

    let (h, w, _) = hwc_shape(input)?;
    let (gx, gy) = sobel_3x3_gradients(input)?;
    let gx_data = gx.data();
    let gy_data = gy.data();
    let len = h * w;

    scratch.ensure_capacity(len);
    let magnitude = &mut scratch.magnitude;
    let direction = &mut scratch.direction;
    let nms = &mut scratch.nms;
    let edges = &mut scratch.edges;

    // Pass 1: Compute magnitude (alpha-max-beta-min) and direction (sign-based)
    // Alpha-max-beta-min: mag ≈ max(|dx|,|dy|) + 0.414 * min(|dx|,|dy|)
    // ~4% max error vs sqrt, but avoids expensive per-pixel sqrt
    for i in 0..len {
        let dx = gx_data[i];
        let dy = gy_data[i];
        let adx = dx.abs();
        let ady = dy.abs();
        let (big, small) = if adx > ady { (adx, ady) } else { (ady, adx) };
        magnitude[i] = big + 0.414 * small;

        // Fast direction quantization using sign comparisons:
        // 0 = horizontal (|dx| > 2.414*|dy|)
        // 2 = vertical (|dy| > 2.414*|dx|)
        // 1 = diagonal /  (dx*dy < 0, roughly 45°)
        // 3 = diagonal \  (dx*dy > 0, roughly 135°)
        // Threshold 2.414 = tan(67.5°), using a fast approximation
        direction[i] = if ady * 5.0 < adx * 2.0 {
            // |dy|/|dx| < 0.4 → horizontal
            0
        } else if adx * 5.0 < ady * 2.0 {
            // |dx|/|dy| < 0.4 → vertical
            2
        } else if (dx > 0.0) == (dy > 0.0) {
            // Same sign → NE-SW diagonal
            1
        } else {
            // Different sign → NW-SE diagonal
            3
        };
    }

    // Pass 2: Non-maximum suppression
    for v in nms.iter_mut() {
        *v = 0.0;
    }
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let idx = y * w + x;
            let mag = magnitude[idx];
            let (n1, n2) = match direction[idx] {
                0 => (magnitude[y * w + x - 1], magnitude[y * w + x + 1]),
                1 => (
                    magnitude[(y - 1) * w + x + 1],
                    magnitude[(y + 1) * w + x - 1],
                ),
                2 => (magnitude[(y - 1) * w + x], magnitude[(y + 1) * w + x]),
                _ => (
                    magnitude[(y - 1) * w + x - 1],
                    magnitude[(y + 1) * w + x + 1],
                ),
            };
            if mag >= n1 && mag >= n2 {
                nms[idx] = mag;
            }
        }
    }

    // Pass 3: Double threshold + BFS hysteresis (single pass, no convergence loop)
    for v in edges.iter_mut() {
        *v = 0;
    }
    scratch.queue.clear();

    // Seed the queue with strong edges
    for i in 0..len {
        if nms[i] >= high_thresh {
            edges[i] = 2;
            scratch.queue.push(i);
        } else if nms[i] >= low_thresh {
            edges[i] = 1;
        }
    }

    // BFS: propagate from strong edges to connected weak edges
    let mut head = 0;
    while head < scratch.queue.len() {
        let idx = scratch.queue[head];
        head += 1;
        let y = idx / w;
        let x = idx % w;
        if y == 0 || y >= h - 1 || x == 0 || x >= w - 1 {
            continue;
        }
        // Check 8 neighbors
        for dy in [-1isize, 0, 1] {
            for dx in [-1isize, 0, 1] {
                if dy == 0 && dx == 0 {
                    continue;
                }
                let ny = (y as isize + dy) as usize;
                let nx = (x as isize + dx) as usize;
                let ni = ny * w + nx;
                if edges[ni] == 1 {
                    edges[ni] = 2;
                    scratch.queue.push(ni);
                }
            }
        }
    }

    let out: Vec<f32> = edges
        .iter()
        .map(|&e| if e == 2 { 1.0 } else { 0.0 })
        .collect();
    Tensor::from_vec(vec![h, w, 1], out).map_err(Into::into)
}

/// Adaptive threshold using a local mean window.
///
/// For each pixel, threshold = local_mean - constant. Pixels above threshold get `max_val`.
/// Operates on single-channel HWC input. `block_size` must be odd and > 0.
pub fn adaptive_threshold_mean(
    input: &Tensor,
    max_val: f32,
    block_size: usize,
    constant: f32,
) -> Result<Tensor, ImgProcError> {
    let (h, w, channels) = hwc_shape(input)?;
    if channels != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: channels,
        });
    }
    if block_size == 0 || block_size.is_multiple_of(2) {
        return Err(ImgProcError::InvalidBlockSize { block_size });
    }

    let data = input.data();
    let half = (block_size / 2) as isize;
    let mut out = vec![0.0f32; h * w];

    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0f32;
            let mut count = 0u32;
            for ky in -half..=half {
                for kx in -half..=half {
                    let sy = y as isize + ky;
                    let sx = x as isize + kx;
                    if sy >= 0 && sy < h as isize && sx >= 0 && sx < w as isize {
                        sum += data[sy as usize * w + sx as usize];
                        count += 1;
                    }
                }
            }
            let local_mean = sum / count as f32;
            let threshold = local_mean - constant;
            out[y * w + x] = if data[y * w + x] > threshold {
                max_val
            } else {
                0.0
            };
        }
    }

    Tensor::from_vec(vec![h, w, 1], out).map_err(Into::into)
}

/// Adaptive threshold using a Gaussian-weighted local window.
///
/// Operates on single-channel HWC input. `block_size` must be odd and > 0.
pub fn adaptive_threshold_gaussian(
    input: &Tensor,
    max_val: f32,
    block_size: usize,
    constant: f32,
) -> Result<Tensor, ImgProcError> {
    let (h, w, channels) = hwc_shape(input)?;
    if channels != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: channels,
        });
    }
    if block_size == 0 || block_size.is_multiple_of(2) {
        return Err(ImgProcError::InvalidBlockSize { block_size });
    }

    let half = block_size / 2;
    let sigma = 0.3 * ((block_size as f64 - 1.0) * 0.5 - 1.0) + 0.8;
    let sigma2 = sigma * sigma;

    let mut kernel = vec![0.0f64; block_size * block_size];
    let mut ksum = 0.0f64;
    for ky in 0..block_size {
        for kx in 0..block_size {
            let dy = ky as f64 - half as f64;
            let dx = kx as f64 - half as f64;
            let val = (-(dy * dy + dx * dx) / (2.0 * sigma2)).exp();
            kernel[ky * block_size + kx] = val;
            ksum += val;
        }
    }
    for v in &mut kernel {
        *v /= ksum;
    }

    let data = input.data();
    let half_i = half as isize;
    let mut out = vec![0.0f32; h * w];

    for y in 0..h {
        for x in 0..w {
            let mut wsum = 0.0f64;
            let mut wnorm = 0.0f64;
            for ky in -half_i..=half_i {
                for kx in -half_i..=half_i {
                    let sy = y as isize + ky;
                    let sx = x as isize + kx;
                    if sy >= 0 && sy < h as isize && sx >= 0 && sx < w as isize {
                        let kw =
                            kernel[(ky + half_i) as usize * block_size + (kx + half_i) as usize];
                        wsum += data[sy as usize * w + sx as usize] as f64 * kw;
                        wnorm += kw;
                    }
                }
            }
            let local_mean = (wsum / wnorm) as f32;
            let threshold = local_mean - constant;
            out[y * w + x] = if data[y * w + x] > threshold {
                max_val
            } else {
                0.0
            };
        }
    }

    Tensor::from_vec(vec![h, w, 1], out).map_err(Into::into)
}

/// Connected-component labeling on a binary single-channel HWC image (4-connectivity).
///
/// Input pixels > 0 are foreground. Returns `(label_map, num_labels)` where
/// each connected component has a unique positive integer label, background is 0.
pub fn connected_components_4(input: &Tensor) -> Result<(Tensor, usize), ImgProcError> {
    let (h, w, channels) = hwc_shape(input)?;
    if channels != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: channels,
        });
    }

    let data = input.data();
    let len = h * w;
    let mut labels = vec![0u32; len];
    let mut next_label = 1u32;
    let mut equivalences: Vec<u32> = vec![0];

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if data[idx] <= 0.0 {
                continue;
            }
            let left = if x > 0 { labels[y * w + x - 1] } else { 0 };
            let above = if y > 0 { labels[(y - 1) * w + x] } else { 0 };

            match (left > 0, above > 0) {
                (false, false) => {
                    labels[idx] = next_label;
                    equivalences.push(next_label);
                    next_label += 1;
                }
                (true, false) => labels[idx] = left,
                (false, true) => labels[idx] = above,
                (true, true) => {
                    let rl = find_root(&equivalences, left);
                    let ra = find_root(&equivalences, above);
                    labels[idx] = rl.min(ra);
                    if rl != ra {
                        let (lo, hi) = if rl < ra { (rl, ra) } else { (ra, rl) };
                        equivalences[hi as usize] = lo;
                    }
                }
            }
        }
    }

    let mut canonical = vec![0u32; next_label as usize];
    let mut label_count = 0u32;
    #[allow(clippy::needless_range_loop)]
    for i in 1..next_label as usize {
        let root = find_root(&equivalences, i as u32);
        if root == i as u32 {
            label_count += 1;
            canonical[i] = label_count;
        }
    }
    #[allow(clippy::needless_range_loop)]
    for i in 1..next_label as usize {
        let root = find_root(&equivalences, i as u32);
        canonical[i] = canonical[root as usize];
    }

    let out: Vec<f32> = labels
        .iter()
        .map(|&l| {
            if l == 0 {
                0.0
            } else {
                canonical[l as usize] as f32
            }
        })
        .collect();

    Ok((Tensor::from_vec(vec![h, w, 1], out)?, label_count as usize))
}

pub(crate) fn find_root(equiv: &[u32], mut label: u32) -> u32 {
    while equiv[label as usize] != label {
        label = equiv[label as usize];
    }
    label
}
