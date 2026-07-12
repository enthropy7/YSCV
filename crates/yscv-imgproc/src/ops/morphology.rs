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

// ── Binary box morphology: sliding-window counts, ~2 ops/pixel ───────────

/// Число foreground-пикселей (>0.5) в клампнутом окне ksize×ksize вокруг
/// каждого пикселя — два прохода префиксных сумм, row-major.
fn binary_box_counts(
    input: &Tensor,
    ksize: usize,
) -> Result<(Vec<u32>, usize, usize), ImgProcError> {
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

    // горизонтальные оконные суммы через префикс по строке; середина строки —
    // безиндексные zip-итераторы (векторизуются), кромки — отдельно
    let mut row_sums = vec![0u32; h * w];
    let mut prefix = vec![0u32; w + 1];
    for y in 0..h {
        let row = &data[y * w..(y + 1) * w];
        let mut acc = 0u32;
        for (p, &v) in prefix[1..].iter_mut().zip(row) {
            acc += u32::from(v > 0.5);
            *p = acc;
        }
        let dst = &mut row_sums[y * w..(y + 1) * w];
        if w > 2 * r {
            for (x, d) in dst.iter_mut().enumerate().take(r) {
                *d = prefix[(x + r + 1).min(w)];
            }
            let (hi_part, lo_part) = (&prefix[2 * r + 1..=w], &prefix[..w - 2 * r]);
            for ((d, &hi), &lo) in dst[r..w - r].iter_mut().zip(hi_part).zip(lo_part) {
                *d = hi - lo;
            }
            let total = prefix[w];
            for (x, d) in dst.iter_mut().enumerate().skip(w - r) {
                *d = total - prefix[x - r];
            }
        } else {
            for (x, d) in dst.iter_mut().enumerate() {
                let hi = (x + r + 1).min(w);
                let lo = x.saturating_sub(r);
                *d = prefix[hi] - prefix[lo];
            }
        }
    }
    // вертикальное суммирование окном по столбцам — скользящий аккумулятор строк
    let mut out = vec![0u32; h * w];
    let mut acc = vec![0u32; w];
    // стартовое окно для y=0: строки [0, r]
    for row_sum in row_sums.chunks_exact(w).take(r + 1) {
        for (a, &v) in acc.iter_mut().zip(row_sum) {
            *a += v;
        }
    }
    for y in 0..h {
        out[y * w..(y + 1) * w].copy_from_slice(&acc);
        // сдвиг окна: добавить строку y+r+1, убрать y-r
        if y + r + 1 < h {
            let add = &row_sums[(y + r + 1) * w..(y + r + 2) * w];
            for (a, &v) in acc.iter_mut().zip(add) {
                *a += v;
            }
        }
        if y >= r {
            let sub = &row_sums[(y - r) * w..(y - r + 1) * w];
            for (a, &v) in acc.iter_mut().zip(sub) {
                *a -= v;
            }
        }
    }
    Ok((out, h, w))
}

/// Binary dilation with a ksize×ksize ones kernel: pixel is set when the
/// clamped window contains any foreground pixel. Input: `[H, W, 1]`,
/// foreground > 0.5; output 0.0/1.0. ~2 ops per pixel via sliding counts —
/// the fastest path for boolean masks.
pub fn dilate_binary_box(input: &Tensor, ksize: usize) -> Result<Tensor, ImgProcError> {
    let (counts, h, w) = binary_box_counts(input, ksize)?;
    let out: Vec<f32> = counts.iter().map(|&c| f32::from(u8::from(c > 0))).collect();
    Tensor::from_vec(vec![h, w, 1], out).map_err(Into::into)
}

/// Binary erosion with a ksize×ksize ones kernel: pixel survives when every
/// in-bounds window pixel is foreground (OpenCV border semantics — the
/// window is clamped at image edges). See [`dilate_binary_box`].
pub fn erode_binary_box(input: &Tensor, ksize: usize) -> Result<Tensor, ImgProcError> {
    let (counts, h, w) = binary_box_counts(input, ksize)?;
    let r = ksize / 2;
    let mut out = vec![0.0f32; h * w];
    for y in 0..h {
        let wy = (y + r + 1).min(h) - y.saturating_sub(r);
        let row_counts = &counts[y * w..(y + 1) * w];
        let out_row = &mut out[y * w..(y + 1) * w];
        if w > 2 * r {
            for (x, (o, &c)) in out_row.iter_mut().zip(row_counts).enumerate().take(r) {
                let wx = (x + r + 1).min(w) - x.saturating_sub(r);
                *o = f32::from(u8::from(c as usize == wx * wy));
            }
            let full = (ksize * wy) as u32;
            for (o, &c) in out_row[r..w - r].iter_mut().zip(&row_counts[r..w - r]) {
                *o = f32::from(u8::from(c == full));
            }
            for (x, (o, &c)) in out_row.iter_mut().zip(row_counts).enumerate().skip(w - r) {
                let wx = (x + r + 1).min(w) - x.saturating_sub(r);
                *o = f32::from(u8::from(c as usize == wx * wy));
            }
        } else {
            for (x, (o, &c)) in out_row.iter_mut().zip(row_counts).enumerate() {
                let wx = (x + r + 1).min(w) - x.saturating_sub(r);
                *o = f32::from(u8::from(c as usize == wx * wy));
            }
        }
    }
    Tensor::from_vec(vec![h, w, 1], out).map_err(Into::into)
}
