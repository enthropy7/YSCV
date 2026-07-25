use rayon::prelude::*;
use yscv_tensor::Tensor;

use super::super::ImgProcError;
use super::super::shape::hwc_shape;
use super::filter::border_coords_3x3;

/// SIMD 3x3 max (dilate) for single-channel row. Returns first x NOT processed.
#[allow(unsafe_code)]
fn dilate_simd_row_c1(
    row0: &[f32],
    row1: &[f32],
    row2: &[f32],
    out: &mut [f32],
    w: usize,
) -> usize {
    if w < 6 {
        return 1;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if yscv_cpu::host_cpu().features.neon {
            return unsafe { dilate_neon_row_c1(row0, row1, row2, out, w) };
        }
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if yscv_cpu::host_cpu().features.avx {
            return unsafe { dilate_avx_row_c1(row0, row1, row2, out, w) };
        }
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if yscv_cpu::host_cpu().features.sse {
            return unsafe { dilate_sse_row_c1(row0, row1, row2, out, w) };
        }
    }
    1
}

/// SIMD 3x3 max (dilate) for multi-channel (HWC) row.
/// Operates on flat row data of length `row_len = w * channels`.
/// Horizontal neighbors are at offsets `-channels, 0, +channels` in the flat array.
/// Returns the flat index up to which processing was done (interior pixels only,
/// i.e., from `channels` to `row_len - channels`).
#[allow(unsafe_code)]
fn dilate_simd_row_mc(
    row0: &[f32],
    row1: &[f32],
    row2: &[f32],
    out: &mut [f32],
    row_len: usize,
    channels: usize,
) -> usize {
    if row_len < channels * 3 + 4 {
        return channels;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if yscv_cpu::host_cpu().features.neon {
            return unsafe { dilate_neon_row_mc(row0, row1, row2, out, row_len, channels) };
        }
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if yscv_cpu::host_cpu().features.avx {
            return unsafe { dilate_avx_row_mc(row0, row1, row2, out, row_len, channels) };
        }
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if yscv_cpu::host_cpu().features.sse {
            return unsafe { dilate_sse_row_mc(row0, row1, row2, out, row_len, channels) };
        }
    }
    channels
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn dilate_neon_row_mc(
    row0: &[f32],
    row1: &[f32],
    row2: &[f32],
    out: &mut [f32],
    row_len: usize,
    channels: usize,
) -> usize {
    use std::arch::aarch64::*;
    let mut i = channels;
    let end = row_len - channels;
    while i + 4 <= end {
        let r0l = vld1q_f32(row0.as_ptr().add(i - channels));
        let r0m = vld1q_f32(row0.as_ptr().add(i));
        let r0r = vld1q_f32(row0.as_ptr().add(i + channels));
        let r1l = vld1q_f32(row1.as_ptr().add(i - channels));
        let r1m = vld1q_f32(row1.as_ptr().add(i));
        let r1r = vld1q_f32(row1.as_ptr().add(i + channels));
        let r2l = vld1q_f32(row2.as_ptr().add(i - channels));
        let r2m = vld1q_f32(row2.as_ptr().add(i));
        let r2r = vld1q_f32(row2.as_ptr().add(i + channels));

        let m0 = vmaxq_f32(vmaxq_f32(r0l, r0m), r0r);
        let m1 = vmaxq_f32(vmaxq_f32(r1l, r1m), r1r);
        let m2 = vmaxq_f32(vmaxq_f32(r2l, r2m), r2r);
        vst1q_f32(out.as_mut_ptr().add(i), vmaxq_f32(vmaxq_f32(m0, m1), m2));
        i += 4;
    }
    i
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn dilate_sse_row_mc(
    row0: &[f32],
    row1: &[f32],
    row2: &[f32],
    out: &mut [f32],
    row_len: usize,
    channels: usize,
) -> usize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut i = channels;
    let end = row_len - channels;
    while i + 4 <= end {
        let r0l = _mm_loadu_ps(row0.as_ptr().add(i - channels));
        let r0m = _mm_loadu_ps(row0.as_ptr().add(i));
        let r0r = _mm_loadu_ps(row0.as_ptr().add(i + channels));
        let r1l = _mm_loadu_ps(row1.as_ptr().add(i - channels));
        let r1m = _mm_loadu_ps(row1.as_ptr().add(i));
        let r1r = _mm_loadu_ps(row1.as_ptr().add(i + channels));
        let r2l = _mm_loadu_ps(row2.as_ptr().add(i - channels));
        let r2m = _mm_loadu_ps(row2.as_ptr().add(i));
        let r2r = _mm_loadu_ps(row2.as_ptr().add(i + channels));

        let m0 = _mm_max_ps(_mm_max_ps(r0l, r0m), r0r);
        let m1 = _mm_max_ps(_mm_max_ps(r1l, r1m), r1r);
        let m2 = _mm_max_ps(_mm_max_ps(r2l, r2m), r2r);
        _mm_storeu_ps(out.as_mut_ptr().add(i), _mm_max_ps(_mm_max_ps(m0, m1), m2));
        i += 4;
    }
    i
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn dilate_avx_row_mc(
    row0: &[f32],
    row1: &[f32],
    row2: &[f32],
    out: &mut [f32],
    row_len: usize,
    channels: usize,
) -> usize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut i = channels;
    let end = row_len - channels;
    while i + 8 <= end {
        let r0l = _mm256_loadu_ps(row0.as_ptr().add(i - channels));
        let r0m = _mm256_loadu_ps(row0.as_ptr().add(i));
        let r0r = _mm256_loadu_ps(row0.as_ptr().add(i + channels));
        let r1l = _mm256_loadu_ps(row1.as_ptr().add(i - channels));
        let r1m = _mm256_loadu_ps(row1.as_ptr().add(i));
        let r1r = _mm256_loadu_ps(row1.as_ptr().add(i + channels));
        let r2l = _mm256_loadu_ps(row2.as_ptr().add(i - channels));
        let r2m = _mm256_loadu_ps(row2.as_ptr().add(i));
        let r2r = _mm256_loadu_ps(row2.as_ptr().add(i + channels));

        let m0 = _mm256_max_ps(_mm256_max_ps(r0l, r0m), r0r);
        let m1 = _mm256_max_ps(_mm256_max_ps(r1l, r1m), r1r);
        let m2 = _mm256_max_ps(_mm256_max_ps(r2l, r2m), r2r);
        _mm256_storeu_ps(
            out.as_mut_ptr().add(i),
            _mm256_max_ps(_mm256_max_ps(m0, m1), m2),
        );
        i += 8;
    }
    i
}

/// SIMD 3x3 min (erode) for multi-channel (HWC) row.
#[allow(unsafe_code)]
fn erode_simd_row_mc(
    row0: &[f32],
    row1: &[f32],
    row2: &[f32],
    out: &mut [f32],
    row_len: usize,
    channels: usize,
) -> usize {
    if row_len < channels * 3 + 4 {
        return channels;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if yscv_cpu::host_cpu().features.neon {
            return unsafe { erode_neon_row_mc(row0, row1, row2, out, row_len, channels) };
        }
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if yscv_cpu::host_cpu().features.avx {
            return unsafe { erode_avx_row_mc(row0, row1, row2, out, row_len, channels) };
        }
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if yscv_cpu::host_cpu().features.sse {
            return unsafe { erode_sse_row_mc(row0, row1, row2, out, row_len, channels) };
        }
    }
    channels
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn erode_neon_row_mc(
    row0: &[f32],
    row1: &[f32],
    row2: &[f32],
    out: &mut [f32],
    row_len: usize,
    channels: usize,
) -> usize {
    use std::arch::aarch64::*;
    let mut i = channels;
    let end = row_len - channels;
    while i + 4 <= end {
        let r0l = vld1q_f32(row0.as_ptr().add(i - channels));
        let r0m = vld1q_f32(row0.as_ptr().add(i));
        let r0r = vld1q_f32(row0.as_ptr().add(i + channels));
        let r1l = vld1q_f32(row1.as_ptr().add(i - channels));
        let r1m = vld1q_f32(row1.as_ptr().add(i));
        let r1r = vld1q_f32(row1.as_ptr().add(i + channels));
        let r2l = vld1q_f32(row2.as_ptr().add(i - channels));
        let r2m = vld1q_f32(row2.as_ptr().add(i));
        let r2r = vld1q_f32(row2.as_ptr().add(i + channels));

        let m0 = vminq_f32(vminq_f32(r0l, r0m), r0r);
        let m1 = vminq_f32(vminq_f32(r1l, r1m), r1r);
        let m2 = vminq_f32(vminq_f32(r2l, r2m), r2r);
        vst1q_f32(out.as_mut_ptr().add(i), vminq_f32(vminq_f32(m0, m1), m2));
        i += 4;
    }
    i
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn erode_sse_row_mc(
    row0: &[f32],
    row1: &[f32],
    row2: &[f32],
    out: &mut [f32],
    row_len: usize,
    channels: usize,
) -> usize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut i = channels;
    let end = row_len - channels;
    while i + 4 <= end {
        let r0l = _mm_loadu_ps(row0.as_ptr().add(i - channels));
        let r0m = _mm_loadu_ps(row0.as_ptr().add(i));
        let r0r = _mm_loadu_ps(row0.as_ptr().add(i + channels));
        let r1l = _mm_loadu_ps(row1.as_ptr().add(i - channels));
        let r1m = _mm_loadu_ps(row1.as_ptr().add(i));
        let r1r = _mm_loadu_ps(row1.as_ptr().add(i + channels));
        let r2l = _mm_loadu_ps(row2.as_ptr().add(i - channels));
        let r2m = _mm_loadu_ps(row2.as_ptr().add(i));
        let r2r = _mm_loadu_ps(row2.as_ptr().add(i + channels));

        let m0 = _mm_min_ps(_mm_min_ps(r0l, r0m), r0r);
        let m1 = _mm_min_ps(_mm_min_ps(r1l, r1m), r1r);
        let m2 = _mm_min_ps(_mm_min_ps(r2l, r2m), r2r);
        _mm_storeu_ps(out.as_mut_ptr().add(i), _mm_min_ps(_mm_min_ps(m0, m1), m2));
        i += 4;
    }
    i
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn erode_avx_row_mc(
    row0: &[f32],
    row1: &[f32],
    row2: &[f32],
    out: &mut [f32],
    row_len: usize,
    channels: usize,
) -> usize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut i = channels;
    let end = row_len - channels;
    while i + 8 <= end {
        let r0l = _mm256_loadu_ps(row0.as_ptr().add(i - channels));
        let r0m = _mm256_loadu_ps(row0.as_ptr().add(i));
        let r0r = _mm256_loadu_ps(row0.as_ptr().add(i + channels));
        let r1l = _mm256_loadu_ps(row1.as_ptr().add(i - channels));
        let r1m = _mm256_loadu_ps(row1.as_ptr().add(i));
        let r1r = _mm256_loadu_ps(row1.as_ptr().add(i + channels));
        let r2l = _mm256_loadu_ps(row2.as_ptr().add(i - channels));
        let r2m = _mm256_loadu_ps(row2.as_ptr().add(i));
        let r2r = _mm256_loadu_ps(row2.as_ptr().add(i + channels));

        let m0 = _mm256_min_ps(_mm256_min_ps(r0l, r0m), r0r);
        let m1 = _mm256_min_ps(_mm256_min_ps(r1l, r1m), r1r);
        let m2 = _mm256_min_ps(_mm256_min_ps(r2l, r2m), r2r);
        _mm256_storeu_ps(
            out.as_mut_ptr().add(i),
            _mm256_min_ps(_mm256_min_ps(m0, m1), m2),
        );
        i += 8;
    }
    i
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn dilate_neon_row_c1(
    row0: &[f32],
    row1: &[f32],
    row2: &[f32],
    out: &mut [f32],
    w: usize,
) -> usize {
    use std::arch::aarch64::*;
    let mut x = 1usize;
    while x + 5 <= w {
        let r0l = vld1q_f32(row0.as_ptr().add(x - 1));
        let r0m = vld1q_f32(row0.as_ptr().add(x));
        let r0r = vld1q_f32(row0.as_ptr().add(x + 1));
        let r1l = vld1q_f32(row1.as_ptr().add(x - 1));
        let r1m = vld1q_f32(row1.as_ptr().add(x));
        let r1r = vld1q_f32(row1.as_ptr().add(x + 1));
        let r2l = vld1q_f32(row2.as_ptr().add(x - 1));
        let r2m = vld1q_f32(row2.as_ptr().add(x));
        let r2r = vld1q_f32(row2.as_ptr().add(x + 1));

        let m0 = vmaxq_f32(vmaxq_f32(r0l, r0m), r0r);
        let m1 = vmaxq_f32(vmaxq_f32(r1l, r1m), r1r);
        let m2 = vmaxq_f32(vmaxq_f32(r2l, r2m), r2r);
        let result = vmaxq_f32(vmaxq_f32(m0, m1), m2);

        vst1q_f32(out.as_mut_ptr().add(x), result);
        x += 4;
    }
    x
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn dilate_sse_row_c1(
    row0: &[f32],
    row1: &[f32],
    row2: &[f32],
    out: &mut [f32],
    w: usize,
) -> usize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut x = 1usize;
    while x + 5 <= w {
        let r0l = _mm_loadu_ps(row0.as_ptr().add(x - 1));
        let r0m = _mm_loadu_ps(row0.as_ptr().add(x));
        let r0r = _mm_loadu_ps(row0.as_ptr().add(x + 1));
        let r1l = _mm_loadu_ps(row1.as_ptr().add(x - 1));
        let r1m = _mm_loadu_ps(row1.as_ptr().add(x));
        let r1r = _mm_loadu_ps(row1.as_ptr().add(x + 1));
        let r2l = _mm_loadu_ps(row2.as_ptr().add(x - 1));
        let r2m = _mm_loadu_ps(row2.as_ptr().add(x));
        let r2r = _mm_loadu_ps(row2.as_ptr().add(x + 1));

        let m0 = _mm_max_ps(_mm_max_ps(r0l, r0m), r0r);
        let m1 = _mm_max_ps(_mm_max_ps(r1l, r1m), r1r);
        let m2 = _mm_max_ps(_mm_max_ps(r2l, r2m), r2r);
        let result = _mm_max_ps(_mm_max_ps(m0, m1), m2);

        _mm_storeu_ps(out.as_mut_ptr().add(x), result);
        x += 4;
    }
    x
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn dilate_avx_row_c1(
    row0: &[f32],
    row1: &[f32],
    row2: &[f32],
    out: &mut [f32],
    w: usize,
) -> usize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut x = 1usize;
    while x + 9 <= w {
        let r0l = _mm256_loadu_ps(row0.as_ptr().add(x - 1));
        let r0m = _mm256_loadu_ps(row0.as_ptr().add(x));
        let r0r = _mm256_loadu_ps(row0.as_ptr().add(x + 1));
        let r1l = _mm256_loadu_ps(row1.as_ptr().add(x - 1));
        let r1m = _mm256_loadu_ps(row1.as_ptr().add(x));
        let r1r = _mm256_loadu_ps(row1.as_ptr().add(x + 1));
        let r2l = _mm256_loadu_ps(row2.as_ptr().add(x - 1));
        let r2m = _mm256_loadu_ps(row2.as_ptr().add(x));
        let r2r = _mm256_loadu_ps(row2.as_ptr().add(x + 1));

        let m0 = _mm256_max_ps(_mm256_max_ps(r0l, r0m), r0r);
        let m1 = _mm256_max_ps(_mm256_max_ps(r1l, r1m), r1r);
        let m2 = _mm256_max_ps(_mm256_max_ps(r2l, r2m), r2r);
        let result = _mm256_max_ps(_mm256_max_ps(m0, m1), m2);

        _mm256_storeu_ps(out.as_mut_ptr().add(x), result);
        x += 8;
    }
    x
}

/// SIMD 3x3 min (erode) for single-channel row. Returns first x NOT processed.
#[allow(unsafe_code)]
fn erode_simd_row_c1(row0: &[f32], row1: &[f32], row2: &[f32], out: &mut [f32], w: usize) -> usize {
    if w < 6 {
        return 1;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if yscv_cpu::host_cpu().features.neon {
            return unsafe { erode_neon_row_c1(row0, row1, row2, out, w) };
        }
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if yscv_cpu::host_cpu().features.avx {
            return unsafe { erode_avx_row_c1(row0, row1, row2, out, w) };
        }
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if yscv_cpu::host_cpu().features.sse {
            return unsafe { erode_sse_row_c1(row0, row1, row2, out, w) };
        }
    }
    1
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn erode_neon_row_c1(
    row0: &[f32],
    row1: &[f32],
    row2: &[f32],
    out: &mut [f32],
    w: usize,
) -> usize {
    use std::arch::aarch64::*;
    let mut x = 1usize;
    while x + 5 <= w {
        let r0l = vld1q_f32(row0.as_ptr().add(x - 1));
        let r0m = vld1q_f32(row0.as_ptr().add(x));
        let r0r = vld1q_f32(row0.as_ptr().add(x + 1));
        let r1l = vld1q_f32(row1.as_ptr().add(x - 1));
        let r1m = vld1q_f32(row1.as_ptr().add(x));
        let r1r = vld1q_f32(row1.as_ptr().add(x + 1));
        let r2l = vld1q_f32(row2.as_ptr().add(x - 1));
        let r2m = vld1q_f32(row2.as_ptr().add(x));
        let r2r = vld1q_f32(row2.as_ptr().add(x + 1));

        let m0 = vminq_f32(vminq_f32(r0l, r0m), r0r);
        let m1 = vminq_f32(vminq_f32(r1l, r1m), r1r);
        let m2 = vminq_f32(vminq_f32(r2l, r2m), r2r);
        let result = vminq_f32(vminq_f32(m0, m1), m2);

        vst1q_f32(out.as_mut_ptr().add(x), result);
        x += 4;
    }
    x
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn erode_sse_row_c1(
    row0: &[f32],
    row1: &[f32],
    row2: &[f32],
    out: &mut [f32],
    w: usize,
) -> usize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut x = 1usize;
    while x + 5 <= w {
        let r0l = _mm_loadu_ps(row0.as_ptr().add(x - 1));
        let r0m = _mm_loadu_ps(row0.as_ptr().add(x));
        let r0r = _mm_loadu_ps(row0.as_ptr().add(x + 1));
        let r1l = _mm_loadu_ps(row1.as_ptr().add(x - 1));
        let r1m = _mm_loadu_ps(row1.as_ptr().add(x));
        let r1r = _mm_loadu_ps(row1.as_ptr().add(x + 1));
        let r2l = _mm_loadu_ps(row2.as_ptr().add(x - 1));
        let r2m = _mm_loadu_ps(row2.as_ptr().add(x));
        let r2r = _mm_loadu_ps(row2.as_ptr().add(x + 1));

        let m0 = _mm_min_ps(_mm_min_ps(r0l, r0m), r0r);
        let m1 = _mm_min_ps(_mm_min_ps(r1l, r1m), r1r);
        let m2 = _mm_min_ps(_mm_min_ps(r2l, r2m), r2r);
        let result = _mm_min_ps(_mm_min_ps(m0, m1), m2);

        _mm_storeu_ps(out.as_mut_ptr().add(x), result);
        x += 4;
    }
    x
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn erode_avx_row_c1(
    row0: &[f32],
    row1: &[f32],
    row2: &[f32],
    out: &mut [f32],
    w: usize,
) -> usize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut x = 1usize;
    while x + 9 <= w {
        let r0l = _mm256_loadu_ps(row0.as_ptr().add(x - 1));
        let r0m = _mm256_loadu_ps(row0.as_ptr().add(x));
        let r0r = _mm256_loadu_ps(row0.as_ptr().add(x + 1));
        let r1l = _mm256_loadu_ps(row1.as_ptr().add(x - 1));
        let r1m = _mm256_loadu_ps(row1.as_ptr().add(x));
        let r1r = _mm256_loadu_ps(row1.as_ptr().add(x + 1));
        let r2l = _mm256_loadu_ps(row2.as_ptr().add(x - 1));
        let r2m = _mm256_loadu_ps(row2.as_ptr().add(x));
        let r2r = _mm256_loadu_ps(row2.as_ptr().add(x + 1));

        let m0 = _mm256_min_ps(_mm256_min_ps(r0l, r0m), r0r);
        let m1 = _mm256_min_ps(_mm256_min_ps(r1l, r1m), r1r);
        let m2 = _mm256_min_ps(_mm256_min_ps(r2l, r2m), r2r);
        let result = _mm256_min_ps(_mm256_min_ps(m0, m1), m2);

        _mm256_storeu_ps(out.as_mut_ptr().add(x), result);
        x += 8;
    }
    x
}

/// Applies a 3x3 grayscale/RGB dilation (local maximum per channel).
///
/// Border handling uses only in-bounds neighbors.
#[allow(unsafe_code)]
pub fn dilate_3x3(input: &Tensor) -> Result<Tensor, ImgProcError> {
    let (h, w, channels) = hwc_shape(input)?;
    let total = h * w * channels;
    let mut out = vec![0.0f32; total];
    let data = input.data();
    let row_len = w * channels;
    let interior_h = h.saturating_sub(2);

    // Interior pixels (y=1..h-1, x=1..w-1): no bounds checks
    let compute_interior_row = |y: usize, row: &mut [f32]| {
        if !cfg!(miri) {
            // SIMD fast path for any channel count.
            // For HWC layout, flat row data is [R0 G0 B0 R1 G1 B1 ...].
            // Horizontal neighbors for pixel x, channel c are at flat offsets
            // (x-1)*C+c, x*C+c, (x+1)*C+c — which is offset -C, 0, +C in flat space.
            // SIMD loads at these offsets naturally compare same channels.
            let row0 = &data[(y - 1) * row_len..y * row_len];
            let row1 = &data[y * row_len..(y + 1) * row_len];
            let row2 = &data[(y + 1) * row_len..(y + 2) * row_len];

            let done = if channels == 1 {
                dilate_simd_row_c1(row0, row1, row2, row, w)
            } else {
                dilate_simd_row_mc(row0, row1, row2, row, row_len, channels)
            };

            // Scalar fallback for remaining interior elements
            if channels == 1 {
                for x in done..w.saturating_sub(1) {
                    if x == 0 {
                        continue;
                    }
                    let mut m = row0[x - 1];
                    m = m.max(row0[x]).max(row0[x + 1]);
                    m = m.max(row1[x - 1]).max(row1[x]).max(row1[x + 1]);
                    m = m.max(row2[x - 1]).max(row2[x]).max(row2[x + 1]);
                    row[x] = m;
                }
            } else {
                // Convert flat index back to pixel x, handle remaining pixels
                let start_x = done / channels;
                for x in start_x..w.saturating_sub(1) {
                    if x == 0 {
                        continue;
                    }
                    for c in 0..channels {
                        let i = x * channels + c;
                        let mut m = row0[i - channels];
                        m = m.max(row0[i]).max(row0[i + channels]);
                        m = m
                            .max(row1[i - channels])
                            .max(row1[i])
                            .max(row1[i + channels]);
                        m = m
                            .max(row2[i - channels])
                            .max(row2[i])
                            .max(row2[i + channels]);
                        row[i] = m;
                    }
                }
            }
            return;
        }

        if channels == 1 {
            for x in 1..w.saturating_sub(1) {
                let row0 = &data[(y - 1) * w..y * w];
                let row1 = &data[y * w..(y + 1) * w];
                let row2 = &data[(y + 1) * w..(y + 2) * w];
                let mut m = row0[x - 1];
                m = m.max(row0[x]).max(row0[x + 1]);
                m = m.max(row1[x - 1]).max(row1[x]).max(row1[x + 1]);
                m = m.max(row2[x - 1]).max(row2[x]).max(row2[x + 1]);
                row[x] = m;
            }
        } else {
            for x in 1..w.saturating_sub(1) {
                for c in 0..channels {
                    let r0 = ((y - 1) * w + x - 1) * channels + c;
                    let r1 = (y * w + x - 1) * channels + c;
                    let r2 = ((y + 1) * w + x - 1) * channels + c;
                    let mut max_value = data[r0];
                    max_value = max_value.max(data[r0 + channels]);
                    max_value = max_value.max(data[r0 + 2 * channels]);
                    max_value = max_value.max(data[r1]);
                    max_value = max_value.max(data[r1 + channels]);
                    max_value = max_value.max(data[r1 + 2 * channels]);
                    max_value = max_value.max(data[r2]);
                    max_value = max_value.max(data[r2 + channels]);
                    max_value = max_value.max(data[r2 + 2 * channels]);
                    row[x * channels + c] = max_value;
                }
            }
        }
    };

    if interior_h > 0 {
        let pixels = h * w;

        #[cfg(target_os = "macos")]
        let use_gcd = pixels > 4096 && !cfg!(miri);
        #[cfg(not(target_os = "macos"))]
        let use_gcd = false;

        if use_gcd {
            #[cfg(target_os = "macos")]
            {
                let out_ptr = super::SendPtr(out.as_mut_ptr());
                use super::u8ops::gcd;
                gcd::parallel_for(interior_h, |i| {
                    let y = i + 1;
                    // SAFETY: each row writes to a disjoint slice of out.
                    let row = unsafe {
                        std::slice::from_raw_parts_mut(out_ptr.ptr().add(y * row_len), row_len)
                    };
                    compute_interior_row(y, row);
                });
            }
        } else if pixels > 4096 {
            let interior_out = &mut out[row_len..row_len + interior_h * row_len];
            interior_out
                .par_chunks_mut(row_len)
                .enumerate()
                .for_each(|(i, row)| compute_interior_row(i + 1, row));
        } else {
            let interior_out = &mut out[row_len..row_len + interior_h * row_len];
            interior_out
                .chunks_mut(row_len)
                .enumerate()
                .for_each(|(i, row)| compute_interior_row(i + 1, row));
        }
    }

    // Border pixels: bounds-checked path
    let border = border_coords_3x3(h, w);
    for (y, x) in border {
        for c in 0..channels {
            let mut max_value = f32::NEG_INFINITY;
            for ky in -1isize..=1 {
                for kx in -1isize..=1 {
                    let sy = y as isize + ky;
                    let sx = x as isize + kx;
                    if sy < 0 || sx < 0 || sy >= h as isize || sx >= w as isize {
                        continue;
                    }
                    let src = ((sy as usize) * w + sx as usize) * channels + c;
                    max_value = max_value.max(data[src]);
                }
            }
            let dst = (y * w + x) * channels + c;
            out[dst] = max_value;
        }
    }

    Tensor::from_vec(vec![h, w, channels], out).map_err(Into::into)
}

/// Applies a 3x3 grayscale/RGB erosion (local minimum per channel).
///
/// Border handling uses only in-bounds neighbors.
pub fn erode_3x3(input: &Tensor) -> Result<Tensor, ImgProcError> {
    let (h, w, channels) = hwc_shape(input)?;
    let total = h * w * channels;
    let mut out = vec![0.0f32; total];
    let data = input.data();
    let row_len = w * channels;
    let interior_h = h.saturating_sub(2);

    // Interior pixels (y=1..h-1, x=1..w-1): no bounds checks
    let compute_interior_row = |y: usize, row: &mut [f32]| {
        if !cfg!(miri) {
            let row0 = &data[(y - 1) * row_len..y * row_len];
            let row1 = &data[y * row_len..(y + 1) * row_len];
            let row2 = &data[(y + 1) * row_len..(y + 2) * row_len];

            let done = if channels == 1 {
                erode_simd_row_c1(row0, row1, row2, row, w)
            } else {
                erode_simd_row_mc(row0, row1, row2, row, row_len, channels)
            };

            if channels == 1 {
                for x in done..w.saturating_sub(1) {
                    if x == 0 {
                        continue;
                    }
                    let mut m = row0[x - 1];
                    m = m.min(row0[x]).min(row0[x + 1]);
                    m = m.min(row1[x - 1]).min(row1[x]).min(row1[x + 1]);
                    m = m.min(row2[x - 1]).min(row2[x]).min(row2[x + 1]);
                    row[x] = m;
                }
            } else {
                let start_x = done / channels;
                for x in start_x..w.saturating_sub(1) {
                    if x == 0 {
                        continue;
                    }
                    for c in 0..channels {
                        let i = x * channels + c;
                        let mut m = row0[i - channels];
                        m = m.min(row0[i]).min(row0[i + channels]);
                        m = m
                            .min(row1[i - channels])
                            .min(row1[i])
                            .min(row1[i + channels]);
                        m = m
                            .min(row2[i - channels])
                            .min(row2[i])
                            .min(row2[i + channels]);
                        row[i] = m;
                    }
                }
            }
            return;
        }

        if channels == 1 {
            for x in 1..w.saturating_sub(1) {
                let row0 = &data[(y - 1) * w..y * w];
                let row1 = &data[y * w..(y + 1) * w];
                let row2 = &data[(y + 1) * w..(y + 2) * w];
                let mut m = row0[x - 1];
                m = m.min(row0[x]).min(row0[x + 1]);
                m = m.min(row1[x - 1]).min(row1[x]).min(row1[x + 1]);
                m = m.min(row2[x - 1]).min(row2[x]).min(row2[x + 1]);
                row[x] = m;
            }
        } else {
            for x in 1..w.saturating_sub(1) {
                for c in 0..channels {
                    let r0 = ((y - 1) * w + x - 1) * channels + c;
                    let r1 = (y * w + x - 1) * channels + c;
                    let r2 = ((y + 1) * w + x - 1) * channels + c;
                    let mut min_value = data[r0];
                    min_value = min_value.min(data[r0 + channels]);
                    min_value = min_value.min(data[r0 + 2 * channels]);
                    min_value = min_value.min(data[r1]);
                    min_value = min_value.min(data[r1 + channels]);
                    min_value = min_value.min(data[r1 + 2 * channels]);
                    min_value = min_value.min(data[r2]);
                    min_value = min_value.min(data[r2 + channels]);
                    min_value = min_value.min(data[r2 + 2 * channels]);
                    row[x * channels + c] = min_value;
                }
            }
        }
    };

    if interior_h > 0 {
        let pixels = h * w;

        #[cfg(target_os = "macos")]
        let use_gcd = pixels > 4096 && !cfg!(miri);
        #[cfg(not(target_os = "macos"))]
        let use_gcd = false;

        if use_gcd {
            #[cfg(target_os = "macos")]
            {
                let out_ptr = super::SendPtr(out.as_mut_ptr());
                use super::u8ops::gcd;
                gcd::parallel_for(interior_h, |i| {
                    let y = i + 1;
                    // SAFETY: each row writes to a disjoint slice of out.
                    let row = unsafe {
                        std::slice::from_raw_parts_mut(out_ptr.ptr().add(y * row_len), row_len)
                    };
                    compute_interior_row(y, row);
                });
            }
        } else if pixels > 4096 {
            let interior_out = &mut out[row_len..row_len + interior_h * row_len];
            interior_out
                .par_chunks_mut(row_len)
                .enumerate()
                .for_each(|(i, row)| compute_interior_row(i + 1, row));
        } else {
            let interior_out = &mut out[row_len..row_len + interior_h * row_len];
            interior_out
                .chunks_mut(row_len)
                .enumerate()
                .for_each(|(i, row)| compute_interior_row(i + 1, row));
        }
    }

    // Border pixels: bounds-checked path
    let border = border_coords_3x3(h, w);
    for (y, x) in border {
        for c in 0..channels {
            let mut min_value = f32::INFINITY;
            for ky in -1isize..=1 {
                for kx in -1isize..=1 {
                    let sy = y as isize + ky;
                    let sx = x as isize + kx;
                    if sy < 0 || sx < 0 || sy >= h as isize || sx >= w as isize {
                        continue;
                    }
                    let src = ((sy as usize) * w + sx as usize) * channels + c;
                    min_value = min_value.min(data[src]);
                }
            }
            let dst = (y * w + x) * channels + c;
            out[dst] = min_value;
        }
    }

    Tensor::from_vec(vec![h, w, channels], out).map_err(Into::into)
}

/// Applies a 3x3 opening (`erode` followed by `dilate`) per channel.
pub fn opening_3x3(input: &Tensor) -> Result<Tensor, ImgProcError> {
    let eroded = erode_3x3(input)?;
    dilate_3x3(&eroded)
}

/// Applies a 3x3 closing (`dilate` followed by `erode`) per channel.
pub fn closing_3x3(input: &Tensor) -> Result<Tensor, ImgProcError> {
    let dilated = dilate_3x3(input)?;
    erode_3x3(&dilated)
}

/// Applies a 3x3 morphological gradient (`dilate - erode`) per channel.
pub fn morph_gradient_3x3(input: &Tensor) -> Result<Tensor, ImgProcError> {
    let dilated = dilate_3x3(input)?;
    let eroded = erode_3x3(input)?;

    let mut out = vec![0.0f32; input.len()];
    for (idx, value) in out.iter_mut().enumerate() {
        *value = dilated.data()[idx] - eroded.data()[idx];
    }

    let (h, w, channels) = hwc_shape(input)?;
    Tensor::from_vec(vec![h, w, channels], out).map_err(Into::into)
}

/// Dilate a single-channel `[H, W, 1]` image with an arbitrary structuring element.
///
/// `kernel` is `[kh, kw, 1]` where nonzero values indicate active elements.
pub fn dilate(input: &Tensor, kernel: &Tensor) -> Result<Tensor, ImgProcError> {
    let (h, w, c) = hwc_shape(input)?;
    if c != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: c,
        });
    }
    let (kh, kw, kc) = hwc_shape(kernel)?;
    if kc != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: kc,
        });
    }
    let data = input.data();
    let kern = kernel.data();
    let rh = kh / 2;
    let rw = kw / 2;
    let mut out = vec![0.0f32; h * w];

    for y in 0..h {
        for x in 0..w {
            let mut max_val = f32::NEG_INFINITY;
            for ky in 0..kh {
                for kx in 0..kw {
                    if kern[ky * kw + kx] <= 0.0 {
                        continue;
                    }
                    let ny = y as i32 + ky as i32 - rh as i32;
                    let nx = x as i32 + kx as i32 - rw as i32;
                    if ny >= 0 && ny < h as i32 && nx >= 0 && nx < w as i32 {
                        max_val = max_val.max(data[ny as usize * w + nx as usize]);
                    }
                }
            }
            out[y * w + x] = if max_val.is_finite() { max_val } else { 0.0 };
        }
    }

    Tensor::from_vec(vec![h, w, 1], out).map_err(Into::into)
}

/// Top-hat transform: `input - opening(input)`.
///
/// Extracts bright features smaller than the 3x3 structuring element.
pub fn morph_tophat(input: &Tensor) -> Result<Tensor, ImgProcError> {
    let opened = opening_3x3(input)?;
    let mut out = vec![0.0f32; input.len()];
    for (i, v) in out.iter_mut().enumerate() {
        *v = input.data()[i] - opened.data()[i];
    }
    let (h, w, c) = hwc_shape(input)?;
    Tensor::from_vec(vec![h, w, c], out).map_err(Into::into)
}

/// Black-hat transform: `closing(input) - input`.
///
/// Extracts dark features smaller than the 3x3 structuring element.
pub fn morph_blackhat(input: &Tensor) -> Result<Tensor, ImgProcError> {
    let closed = closing_3x3(input)?;
    let mut out = vec![0.0f32; input.len()];
    for (i, v) in out.iter_mut().enumerate() {
        *v = closed.data()[i] - input.data()[i];
    }
    let (h, w, c) = hwc_shape(input)?;
    Tensor::from_vec(vec![h, w, c], out).map_err(Into::into)
}

/// Zhang-Suen thinning algorithm on a binary single-channel `[H, W, 1]` image.
///
/// Pixels > 0.5 are foreground. Iteratively removes boundary pixels to produce
/// a one-pixel-wide skeleton. Returns a binary image with 1.0 for skeleton pixels.
pub fn skeletonize(input: &Tensor) -> Result<Tensor, ImgProcError> {
    let (h, w, c) = hwc_shape(input)?;
    if c != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: c,
        });
    }
    let data = input.data();
    let mut img: Vec<u8> = data
        .iter()
        .map(|&v| if v > 0.5 { 1u8 } else { 0u8 })
        .collect();

    loop {
        let mut changed = false;

        // Sub-iteration 1
        let mut markers = vec![false; h * w];
        for y in 1..h.saturating_sub(1) {
            for x in 1..w.saturating_sub(1) {
                if img[y * w + x] == 0 {
                    continue;
                }
                let p = zhang_suen_neighbors(&img, w, x, y);
                let b = p.iter().map(|&v| v as u32).sum::<u32>();
                let a = zhang_suen_transitions(&p);
                if (2..=6).contains(&b)
                    && a == 1
                    && p[0] * p[2] * p[4] == 0
                    && p[2] * p[4] * p[6] == 0
                {
                    markers[y * w + x] = true;
                }
            }
        }
        for i in 0..h * w {
            if markers[i] {
                img[i] = 0;
                changed = true;
            }
        }

        // Sub-iteration 2
        markers.fill(false);
        for y in 1..h.saturating_sub(1) {
            for x in 1..w.saturating_sub(1) {
                if img[y * w + x] == 0 {
                    continue;
                }
                let p = zhang_suen_neighbors(&img, w, x, y);
                let b = p.iter().map(|&v| v as u32).sum::<u32>();
                let a = zhang_suen_transitions(&p);
                if (2..=6).contains(&b)
                    && a == 1
                    && p[0] * p[2] * p[6] == 0
                    && p[0] * p[4] * p[6] == 0
                {
                    markers[y * w + x] = true;
                }
            }
        }
        for i in 0..h * w {
            if markers[i] {
                img[i] = 0;
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    let out: Vec<f32> = img.iter().map(|&v| v as f32).collect();
    Tensor::from_vec(vec![h, w, 1], out).map_err(Into::into)
}

/// Returns the 8 neighbors P2..P9 in Zhang-Suen order.
/// P2=N, P3=NE, P4=E, P5=SE, P6=S, P7=SW, P8=W, P9=NW
fn zhang_suen_neighbors(img: &[u8], w: usize, x: usize, y: usize) -> [u8; 8] {
    [
        img[(y - 1) * w + x],     // P2 (N)
        img[(y - 1) * w + x + 1], // P3 (NE)
        img[y * w + x + 1],       // P4 (E)
        img[(y + 1) * w + x + 1], // P5 (SE)
        img[(y + 1) * w + x],     // P6 (S)
        img[(y + 1) * w + x - 1], // P7 (SW)
        img[y * w + x - 1],       // P8 (W)
        img[(y - 1) * w + x - 1], // P9 (NW)
    ]
}

/// Number of 0->1 transitions in the circular sequence P2..P9..P2.
fn zhang_suen_transitions(p: &[u8; 8]) -> u32 {
    let mut count = 0u32;
    for i in 0..8 {
        if p[i] == 0 && p[(i + 1) % 8] == 1 {
            count += 1;
        }
    }
    count
}

/// Removes connected components with area less than `min_size`.
///
/// Takes a single-channel `[H, W, 1]` binary image. Pixels > 0.5 are foreground.
/// Connected components (4-connected) with fewer than `min_size` foreground pixels
/// are set to 0.0.
pub fn remove_small_objects(input: &Tensor, min_size: usize) -> Result<Tensor, ImgProcError> {
    let (h, w, c) = hwc_shape(input)?;
    if c != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: c,
        });
    }
    let data = input.data();
    let mut labels = vec![0u32; h * w];
    let mut label_id = 0u32;
    let mut label_sizes: Vec<usize> = Vec::new();

    // BFS-based connected components (4-connected)
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if data[idx] <= 0.5 || labels[idx] != 0 {
                continue;
            }
            label_id += 1;
            let mut queue = vec![(x, y)];
            labels[idx] = label_id;
            let mut size = 0usize;
            while let Some((cx, cy)) = queue.pop() {
                size += 1;
                for &(dx, dy) in &[(0isize, -1isize), (0, 1), (-1, 0), (1, 0)] {
                    let nx = cx as isize + dx;
                    let ny = cy as isize + dy;
                    if nx >= 0 && nx < w as isize && ny >= 0 && ny < h as isize {
                        let nidx = ny as usize * w + nx as usize;
                        if data[nidx] > 0.5 && labels[nidx] == 0 {
                            labels[nidx] = label_id;
                            queue.push((nx as usize, ny as usize));
                        }
                    }
                }
            }
            label_sizes.push(size);
        }
    }

    let mut out: Vec<f32> = data.to_vec();
    for i in 0..h * w {
        if labels[i] > 0 && label_sizes[(labels[i] - 1) as usize] < min_size {
            out[i] = 0.0;
        }
    }

    Tensor::from_vec(vec![h, w, 1], out).map_err(Into::into)
}

/// Erode a single-channel `[H, W, 1]` image with an arbitrary structuring element.
pub fn erode(input: &Tensor, kernel: &Tensor) -> Result<Tensor, ImgProcError> {
    let (h, w, c) = hwc_shape(input)?;
    if c != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: c,
        });
    }
    let (kh, kw, kc) = hwc_shape(kernel)?;
    if kc != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: kc,
        });
    }
    let data = input.data();
    let kern = kernel.data();
    let rh = kh / 2;
    let rw = kw / 2;
    let mut out = vec![0.0f32; h * w];

    for y in 0..h {
        for x in 0..w {
            let mut min_val = f32::INFINITY;
            for ky in 0..kh {
                for kx in 0..kw {
                    if kern[ky * kw + kx] <= 0.0 {
                        continue;
                    }
                    let ny = y as i32 + ky as i32 - rh as i32;
                    let nx = x as i32 + kx as i32 - rw as i32;
                    if ny >= 0 && ny < h as i32 && nx >= 0 && nx < w as i32 {
                        min_val = min_val.min(data[ny as usize * w + nx as usize]);
                    }
                }
            }
            out[y * w + x] = if min_val.is_finite() { min_val } else { 0.0 };
        }
    }

    Tensor::from_vec(vec![h, w, 1], out).map_err(Into::into)
}

// ── Box-kernel morphology: separable sliding min/max, O(n) per pass ──────

/// van Herk / Gil-Werman max/min filter: три линейных прохода на направление,
/// O(1) на элемент независимо от размера ядра. Границы — виртуальный паддинг
/// identity-значением (для max это -inf, для min +inf), что эквивалентно
/// клампу окна к краям изображения.
fn box_filter_2d(
    input: &Tensor,
    ksize: usize,
    pick: fn(f32, f32) -> f32,
    identity: f32,
) -> Result<Tensor, ImgProcError> {
    let (h, w, c) = hwc_shape(input)?;
    if c != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: c,
        });
    }
    if ksize == 0 || ksize.is_multiple_of(2) {
        return Err(ImgProcError::InvalidBlockSize { block_size: ksize });
    }
    let r = ksize / 2;
    let data = input.data();

    // ── горизонтальный проход (строка с паддингом r по краям) ──
    let padded_w = w + 2 * r;
    let mut row = vec![identity; padded_w];
    let mut left = vec![identity; padded_w]; // блочный префикс
    let mut right = vec![identity; padded_w]; // блочный суффикс
    let mut tmp = vec![0.0f32; h * w];
    for y in 0..h {
        row[r..r + w].copy_from_slice(&data[y * w..(y + 1) * w]);
        for (i, &v) in row.iter().enumerate() {
            left[i] = if i % ksize == 0 {
                v
            } else {
                pick(left[i - 1], v)
            };
        }
        for i in (0..padded_w).rev() {
            right[i] = if (i + 1) % ksize == 0 || i + 1 == padded_w {
                row[i]
            } else {
                pick(right[i + 1], row[i])
            };
        }
        let out_row = &mut tmp[y * w..(y + 1) * w];
        for x in 0..w {
            // окно в паддинговых координатах: [x, x + 2r]
            out_row[x] = pick(right[x], left[x + 2 * r]);
        }
    }

    // ── вертикальный проход: row-major стриминг, блоки по строкам ──
    let padded_h = h + 2 * r;
    let mut left_v = vec![identity; padded_h * w];
    let mut right_v = vec![identity; padded_h * w];
    let at = |buf: &Vec<f32>, py: usize, x: usize| buf[py * w + x];
    // источник в паддинговых координатах
    let src = |py: usize, x: usize| -> f32 {
        if py < r || py >= r + h {
            identity
        } else {
            tmp[(py - r) * w + x]
        }
    };
    for py in 0..padded_h {
        for x in 0..w {
            let v = src(py, x);
            left_v[py * w + x] = if py % ksize == 0 {
                v
            } else {
                pick(at(&left_v, py - 1, x), v)
            };
        }
    }
    for py in (0..padded_h).rev() {
        for x in 0..w {
            let v = src(py, x);
            right_v[py * w + x] = if (py + 1) % ksize == 0 || py + 1 == padded_h {
                v
            } else {
                pick(at(&right_v, py + 1, x), v)
            };
        }
    }
    let mut out = vec![0.0f32; h * w];
    for y in 0..h {
        for x in 0..w {
            out[y * w + x] = pick(at(&right_v, y, x), at(&left_v, y + 2 * r, x));
        }
    }
    Tensor::from_vec(vec![h, w, 1], out).map_err(Into::into)
}

/// Dilation with an all-ones square kernel of odd size `ksize`.
///
/// Identical результату [`dilate`] с ones-ядром (границы клампятся), но
/// сепарабельно и за O(n) на проход вместо O(n·k²) — на 720p и ядре 5×5
/// это порядок с лишним разницы.
pub fn dilate_box(input: &Tensor, ksize: usize) -> Result<Tensor, ImgProcError> {
    box_filter_2d(input, ksize, f32::max, f32::NEG_INFINITY)
}

/// Erosion with an all-ones square kernel of odd size `ksize`.
/// See [`dilate_box`] for semantics and complexity.
pub fn erode_box(input: &Tensor, ksize: usize) -> Result<Tensor, ImgProcError> {
    box_filter_2d(input, ksize, f32::min, f32::INFINITY)
}

// ── Bit-packed binary morphology ────────────────────────────────────
//
// A boolean mask needs one bit per pixel, not one float. Packing rows into
// `u64` words shrinks a 720p mask from 3.7 MB to 115 KB — it stops being a
// memory-bandwidth problem and starts fitting in L2, which matters most on
// the small ARM boards this runs on. Both passes are then plain word-wise
// OR, and a window of length L is built by recursive doubling in
// `O(log L)` passes instead of `O(L)`.

/// One bit per pixel, rows padded to whole `u64` words.
///
/// Invariant: bits at `x >= width` are always zero. Erosion relies on it —
/// complementing a row must not turn padding into foreground.
struct PackedMask {
    width: usize,
    height: usize,
    words_per_row: usize,
    data: Vec<u64>,
}

impl PackedMask {
    fn new(width: usize, height: usize) -> Self {
        let words_per_row = width.div_ceil(64);
        Self {
            width,
            height,
            words_per_row,
            data: vec![0u64; words_per_row * height],
        }
    }

    fn pack(values: &[f32], width: usize, height: usize) -> Self {
        let mut mask = Self::new(width, height);
        for y in 0..height {
            let row = &values[y * width..(y + 1) * width];
            let words = &mut mask.data[y * mask.words_per_row..(y + 1) * mask.words_per_row];
            pack_row(row, words);
        }
        mask
    }

    fn unpack(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.width * self.height];
        for y in 0..self.height {
            let words = &self.data[y * self.words_per_row..(y + 1) * self.words_per_row];
            let row = &mut out[y * self.width..(y + 1) * self.width];
            unpack_row(words, row);
        }
        out
    }

    /// Mask off the bits past `width` in the last word of every row.
    fn clear_padding(&mut self) {
        let used = self.width % 64;
        if used == 0 {
            return;
        }
        let keep = (1u64 << used) - 1;
        for row in self.data.chunks_exact_mut(self.words_per_row) {
            if let Some(last) = row.last_mut() {
                *last &= keep;
            }
        }
    }

    fn complement(&mut self) {
        for word in &mut self.data {
            *word = !*word;
        }
        self.clear_padding();
    }
}

/// Pack one row of floats into bits (`> 0.5` is foreground).
///
/// The round trip through floats is what a bit-packed mask actually costs,
/// so both directions run wide: 16 pixels per iteration through a movemask
/// on x86 and a weighted horizontal add on NEON, with a scalar tail.
fn pack_row(row: &[f32], words: &mut [u64]) {
    words.fill(0);
    let mut index = 0usize;

    #[cfg(target_arch = "aarch64")]
    if yscv_cpu::host_cpu().features.neon {
        // SAFETY: ISA guard (feature detection) above.
        index = unsafe { pack_row_neon(row, words) };
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if yscv_cpu::host_cpu().features.sse {
        // SAFETY: ISA guard (feature detection) above.
        index = unsafe { pack_row_sse(row, words) };
    }

    for (x, &value) in row.iter().enumerate().skip(index) {
        if value > 0.5 {
            words[x / 64] |= 1u64 << (x % 64);
        }
    }
}

/// Expand packed bits back into `0.0`/`1.0` floats.
fn unpack_row(words: &[u64], row: &mut [f32]) {
    let mut index = 0usize;

    #[cfg(target_arch = "aarch64")]
    if yscv_cpu::host_cpu().features.neon {
        // SAFETY: ISA guard (feature detection) above.
        index = unsafe { unpack_row_neon(words, row) };
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if yscv_cpu::host_cpu().features.sse2 {
        // SAFETY: ISA guard (feature detection) above.
        index = unsafe { unpack_row_sse2(words, row) };
    }

    for (x, value) in row.iter_mut().enumerate().skip(index) {
        *value = f32::from(u8::from(words[x / 64] >> (x % 64) & 1 == 1));
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn pack_row_neon(row: &[f32], words: &mut [u64]) -> usize {
    use std::arch::aarch64::*;
    // Each lane contributes its own bit, so a horizontal add of the masked
    // weights is exactly the 4-bit nibble for those pixels.
    let weights = vld1q_u32([1u32, 2, 4, 8].as_ptr());
    let threshold = vdupq_n_f32(0.5);
    let ptr = row.as_ptr();
    let mut index = 0usize;
    while index + 64 <= row.len() {
        let mut word = 0u64;
        for group in 0..16 {
            let lanes = vld1q_f32(ptr.add(index + group * 4));
            let mask = vcgtq_f32(lanes, threshold);
            let nibble = u64::from(vaddvq_u32(vandq_u32(mask, weights)));
            word |= nibble << (group * 4);
        }
        words[index / 64] = word;
        index += 64;
    }
    index
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn unpack_row_neon(words: &[u64], row: &mut [f32]) -> usize {
    use std::arch::aarch64::*;
    let weights = vld1q_u32([1u32, 2, 4, 8].as_ptr());
    let ones = vdupq_n_f32(1.0);
    let zeros = vdupq_n_f32(0.0);
    let ptr = row.as_mut_ptr();
    let mut index = 0usize;
    while index + 64 <= row.len() {
        let word = words[index / 64];
        for group in 0..16 {
            let nibble = (word >> (group * 4)) as u32 & 0xF;
            let selected = vtstq_u32(vdupq_n_u32(nibble), weights);
            vst1q_f32(ptr.add(index + group * 4), vbslq_f32(selected, ones, zeros));
        }
        index += 64;
    }
    index
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn pack_row_sse(row: &[f32], words: &mut [u64]) -> usize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let threshold = _mm_set1_ps(0.5);
    let ptr = row.as_ptr();
    let mut index = 0usize;
    while index + 64 <= row.len() {
        let mut word = 0u64;
        for group in 0..16 {
            let lanes = _mm_loadu_ps(ptr.add(index + group * 4));
            let mask = _mm_cmpgt_ps(lanes, threshold);
            #[allow(clippy::cast_sign_loss)]
            let nibble = _mm_movemask_ps(mask) as u32 as u64;
            word |= nibble << (group * 4);
        }
        words[index / 64] = word;
        index += 64;
    }
    index
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse2")]
unsafe fn unpack_row_sse2(words: &[u64], row: &mut [f32]) -> usize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let weights = _mm_set_epi32(8, 4, 2, 1);
    let ones = _mm_set1_ps(1.0);
    let ptr = row.as_mut_ptr();
    let mut index = 0usize;
    while index + 64 <= row.len() {
        let word = words[index / 64];
        for group in 0..16 {
            #[allow(clippy::cast_possible_wrap)]
            let nibble = ((word >> (group * 4)) & 0xF) as i32;
            let spread = _mm_set1_epi32(nibble);
            let selected = _mm_and_si128(spread, weights);
            let mask = _mm_cmpeq_epi32(selected, weights);
            _mm_storeu_ps(
                ptr.add(index + group * 4),
                _mm_and_ps(_mm_castsi128_ps(mask), ones),
            );
        }
        index += 64;
    }
    index
}

/// `dst |= src` shifted `shift` pixels towards larger x (bits move up).
fn or_shift_up(dst: &mut [u64], src: &[u64], shift: usize) {
    let (words, bits) = (shift / 64, shift % 64);
    for index in (0..dst.len()).rev() {
        if index < words {
            break;
        }
        let low = src[index - words];
        let mut value = low << bits;
        if bits > 0 && index > words {
            value |= src[index - words - 1] >> (64 - bits);
        }
        dst[index] |= value;
    }
}

/// `dst |= src` shifted `shift` pixels towards smaller x (bits move down).
fn or_shift_down(dst: &mut [u64], src: &[u64], shift: usize) {
    let (words, bits) = (shift / 64, shift % 64);
    for index in 0..dst.len() {
        let Some(&high) = src.get(index + words) else {
            break;
        };
        let mut value = high >> bits;
        if bits > 0
            && let Some(&next) = src.get(index + words + 1)
        {
            value |= next << (64 - bits);
        }
        dst[index] |= value;
    }
}

/// One-sided horizontal OR-window of `length` pixels, built by doubling:
/// after step `i` every bit holds the OR of up to `2^i` neighbours in the
/// chosen direction. `up` selects growth towards larger x.
fn horizontal_window(mask: &mut PackedMask, length: usize, up: bool) {
    let mut covered = 1usize;
    let mut scratch = vec![0u64; mask.words_per_row];
    while covered < length {
        let step = covered.min(length - covered);
        for row in mask.data.chunks_exact_mut(mask.words_per_row) {
            scratch.copy_from_slice(row);
            if up {
                or_shift_up(row, &scratch, step);
            } else {
                or_shift_down(row, &scratch, step);
            }
        }
        covered += step;
    }
    mask.clear_padding();
}

/// One-sided vertical OR-window of `length` rows, same doubling scheme.
fn vertical_window(mask: &mut PackedMask, length: usize, down: bool) {
    let stride = mask.words_per_row;
    let mut covered = 1usize;
    let mut scratch = vec![0u64; mask.data.len()];
    while covered < length {
        let step = covered.min(length - covered);
        scratch.copy_from_slice(&mask.data);
        for y in 0..mask.height {
            let source = if down {
                y.checked_add(step)
            } else {
                y.checked_sub(step)
            };
            let Some(source) = source.filter(|&row| row < mask.height) else {
                continue;
            };
            let (dst_start, src_start) = (y * stride, source * stride);
            for offset in 0..stride {
                mask.data[dst_start + offset] |= scratch[src_start + offset];
            }
        }
        covered += step;
    }
}

/// Dilate in place with a `ksize`×`ksize` ones kernel; outside the image
/// counts as background.
fn dilate_packed(mask: &mut PackedMask, ksize: usize) {
    let radius = ksize / 2;
    if radius == 0 {
        return;
    }
    // Centred window = the union of two one-sided windows of radius+1.
    let mut mirrored = PackedMask {
        width: mask.width,
        height: mask.height,
        words_per_row: mask.words_per_row,
        data: mask.data.clone(),
    };
    horizontal_window(mask, radius + 1, true);
    horizontal_window(&mut mirrored, radius + 1, false);
    for (word, other) in mask.data.iter_mut().zip(&mirrored.data) {
        *word |= *other;
    }

    mirrored.data.copy_from_slice(&mask.data);
    vertical_window(mask, radius + 1, true);
    vertical_window(&mut mirrored, radius + 1, false);
    for (word, other) in mask.data.iter_mut().zip(&mirrored.data) {
        *word |= *other;
    }
}

/// Erode in place. Morphological duality: eroding is dilating the
/// complement, and because padding stays zero the complement treats
/// out-of-image pixels as foreground — exactly the clamped-window border
/// semantics of OpenCV.
fn erode_packed(mask: &mut PackedMask, ksize: usize) {
    mask.complement();
    dilate_packed(mask, ksize);
    mask.complement();
}

fn packed_from_tensor(input: &Tensor, ksize: usize) -> Result<PackedMask, ImgProcError> {
    let (h, w, c) = hwc_shape(input)?;
    if c != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: c,
        });
    }
    if ksize == 0 || ksize.is_multiple_of(2) {
        return Err(ImgProcError::InvalidBlockSize { block_size: ksize });
    }
    Ok(PackedMask::pack(input.data(), w, h))
}

fn tensor_from_packed(mask: &PackedMask) -> Result<Tensor, ImgProcError> {
    Tensor::from_vec(vec![mask.height, mask.width, 1], mask.unpack()).map_err(Into::into)
}

/// Binary dilation with a ksize×ksize ones kernel: a pixel is set when the
/// clamped window contains any foreground pixel. Input: `[H, W, 1]`,
/// foreground > 0.5; output 0.0/1.0. Runs on a bit-packed copy, so cost is
/// `O(log ksize)` word passes rather than per-pixel work.
pub fn dilate_binary_box(input: &Tensor, ksize: usize) -> Result<Tensor, ImgProcError> {
    let mut mask = packed_from_tensor(input, ksize)?;
    dilate_packed(&mut mask, ksize);
    tensor_from_packed(&mask)
}

/// Binary erosion with a ksize×ksize ones kernel: a pixel survives when
/// every in-bounds window pixel is foreground (OpenCV border semantics — the
/// window is clamped at image edges). See [`dilate_binary_box`].
pub fn erode_binary_box(input: &Tensor, ksize: usize) -> Result<Tensor, ImgProcError> {
    let mut mask = packed_from_tensor(input, ksize)?;
    erode_packed(&mut mask, ksize);
    tensor_from_packed(&mask)
}

/// Binary opening (erode then dilate) with one pack/unpack round trip.
///
/// Chaining [`erode_binary_box`] and [`dilate_binary_box`] converts the whole
/// image to floats and back twice; this keeps the mask packed in between.
pub fn open_binary_box(input: &Tensor, ksize: usize) -> Result<Tensor, ImgProcError> {
    let mut mask = packed_from_tensor(input, ksize)?;
    erode_packed(&mut mask, ksize);
    dilate_packed(&mut mask, ksize);
    tensor_from_packed(&mask)
}

/// Binary closing (dilate then erode) with one pack/unpack round trip.
/// See [`open_binary_box`].
pub fn close_binary_box(input: &Tensor, ksize: usize) -> Result<Tensor, ImgProcError> {
    let mut mask = packed_from_tensor(input, ksize)?;
    dilate_packed(&mut mask, ksize);
    erode_packed(&mut mask, ksize);
    tensor_from_packed(&mask)
}
