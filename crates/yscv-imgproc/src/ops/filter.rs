use rayon::prelude::*;
use yscv_tensor::Tensor;

use super::super::ImgProcError;
use super::super::shape::hwc_shape;

/// SIMD 3x3 box blur for single-channel interior row. Returns first x NOT processed.
#[allow(unsafe_code)]
fn box_blur_simd_row_c1(
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
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { box_blur_neon_row_c1(row0, row1, row2, out, w) };
        }
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("avx") {
        return unsafe { box_blur_avx_row_c1(row0, row1, row2, out, w) };
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("sse") {
            return unsafe { box_blur_sse_row_c1(row0, row1, row2, out, w) };
        }
    }
    1
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn box_blur_neon_row_c1(
    row0: &[f32],
    row1: &[f32],
    row2: &[f32],
    out: &mut [f32],
    w: usize,
) -> usize {
    use std::arch::aarch64::*;
    let inv9 = vdupq_n_f32(1.0 / 9.0);
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

        let sum = vaddq_f32(
            vaddq_f32(vaddq_f32(r0l, r0m), vaddq_f32(r0r, r1l)),
            vaddq_f32(vaddq_f32(r1m, r1r), vaddq_f32(r2l, vaddq_f32(r2m, r2r))),
        );
        vst1q_f32(out.as_mut_ptr().add(x), vmulq_f32(sum, inv9));
        x += 4;
    }
    x
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn box_blur_avx_row_c1(
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

    let inv9 = _mm256_set1_ps(1.0 / 9.0);
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

        let sum = _mm256_add_ps(
            _mm256_add_ps(_mm256_add_ps(r0l, r0m), _mm256_add_ps(r0r, r1l)),
            _mm256_add_ps(
                _mm256_add_ps(r1m, r1r),
                _mm256_add_ps(r2l, _mm256_add_ps(r2m, r2r)),
            ),
        );
        _mm256_storeu_ps(out.as_mut_ptr().add(x), _mm256_mul_ps(sum, inv9));
        x += 8;
    }
    x
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn box_blur_sse_row_c1(
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

    let inv9 = _mm_set1_ps(1.0 / 9.0);
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

        let sum = _mm_add_ps(
            _mm_add_ps(_mm_add_ps(r0l, r0m), _mm_add_ps(r0r, r1l)),
            _mm_add_ps(_mm_add_ps(r1m, r1r), _mm_add_ps(r2l, _mm_add_ps(r2m, r2r))),
        );
        _mm_storeu_ps(out.as_mut_ptr().add(x), _mm_mul_ps(sum, inv9));
        x += 4;
    }
    x
}

/// Applies zero-padded 3x3 box blur over each channel.
#[allow(unsafe_code, clippy::uninit_vec)]
pub fn box_blur_3x3(input: &Tensor) -> Result<Tensor, ImgProcError> {
    let (h, w, channels) = hwc_shape(input)?;
    let data = input.data();
    let row_len = w * channels;
    let total = h * w * channels;
    // SAFETY: every element is written by compute_row (SIMD + scalar) below.
    let mut out: Vec<f32> = Vec::with_capacity(total);
    unsafe {
        out.set_len(total);
    }

    let compute_row = |y: usize, row: &mut [f32]| {
        // SIMD fast path for single-channel interior rows
        if channels == 1 && y > 0 && y < h - 1 && !cfg!(miri) {
            let row0 = &data[(y - 1) * w..y * w];
            let row1 = &data[y * w..(y + 1) * w];
            let row2 = &data[(y + 1) * w..(y + 2) * w];
            let done = box_blur_simd_row_c1(row0, row1, row2, row, w);
            // Scalar tail for interior
            for x in done..w.saturating_sub(1) {
                if x == 0 {
                    continue;
                }
                let sum = row0[x - 1]
                    + row0[x]
                    + row0[x + 1]
                    + row1[x - 1]
                    + row1[x]
                    + row1[x + 1]
                    + row2[x - 1]
                    + row2[x]
                    + row2[x + 1];
                row[x] = sum / 9.0;
            }
            // Border pixels x=0 and x=w-1 still need bounds-checked path
            // x=0
            {
                let mut acc = 0.0f32;
                let mut count = 0.0f32;
                for ky in -1isize..=1 {
                    let sy = y as isize + ky;
                    if sy < 0 || sy >= h as isize {
                        continue;
                    }
                    for kx in 0isize..=1 {
                        acc += data[(sy as usize) * w + kx as usize];
                        count += 1.0;
                    }
                }
                row[0] = acc / count;
            }
            // x=w-1
            if w > 1 {
                let mut acc = 0.0f32;
                let mut count = 0.0f32;
                for ky in -1isize..=1 {
                    let sy = y as isize + ky;
                    if sy < 0 || sy >= h as isize {
                        continue;
                    }
                    for kx in (w as isize - 2)..=(w as isize - 1) {
                        if kx >= 0 {
                            acc += data[(sy as usize) * w + kx as usize];
                            count += 1.0;
                        }
                    }
                }
                row[w - 1] = acc / count;
            }
            return;
        }

        for x in 0..w {
            for c in 0..channels {
                let mut acc = 0.0f32;
                let mut count = 0.0f32;
                for ky in -1isize..=1 {
                    for kx in -1isize..=1 {
                        let sy = y as isize + ky;
                        let sx = x as isize + kx;
                        if sy < 0 || sx < 0 || sy >= h as isize || sx >= w as isize {
                            continue;
                        }
                        let src = ((sy as usize) * w + sx as usize) * channels + c;
                        acc += data[src];
                        count += 1.0;
                    }
                }
                row[x * channels + c] = acc / count;
            }
        }
    };

    let pixels = h * w;

    #[cfg(target_os = "macos")]
    if pixels > 4096 && !cfg!(miri) {
        let out_ptr = super::SendPtr(out.as_mut_ptr());
        use super::u8ops::gcd;
        gcd::parallel_for(h, |y| {
            // SAFETY: each row writes to a disjoint slice of out.
            let row =
                unsafe { std::slice::from_raw_parts_mut(out_ptr.ptr().add(y * row_len), row_len) };
            compute_row(y, row);
        });
        return Tensor::from_vec(vec![h, w, channels], out).map_err(Into::into);
    }

    if pixels > 4096 {
        out.par_chunks_mut(row_len)
            .enumerate()
            .for_each(|(y, row)| compute_row(y, row));
    } else {
        out.chunks_mut(row_len)
            .enumerate()
            .for_each(|(y, row)| compute_row(y, row));
    }

    Tensor::from_vec(vec![h, w, channels], out).map_err(Into::into)
}

// ── Gaussian blur (separable) ─────────────────────────────────────

/// SIMD `[1,2,1]` horizontal pass for c=1: `out[x] = (src[x-1] + 2*src[x] + src[x+1]) * 0.25`
#[allow(unsafe_code)]
fn gauss_h_simd_row_c1(src: &[f32], out: &mut [f32], w: usize) -> usize {
    if w < 6 {
        return 1;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { gauss_h_neon_row_c1(src, out, w) };
        }
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("avx") {
        return unsafe { gauss_h_avx_row_c1(src, out, w) };
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("sse") {
            return unsafe { gauss_h_sse_row_c1(src, out, w) };
        }
    }
    1
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn gauss_h_neon_row_c1(src: &[f32], out: &mut [f32], w: usize) -> usize {
    use std::arch::aarch64::*;
    let two = vdupq_n_f32(2.0);
    let quarter = vdupq_n_f32(0.25);
    let mut x = 1usize;
    while x + 5 <= w {
        let left = vld1q_f32(src.as_ptr().add(x - 1));
        let center = vld1q_f32(src.as_ptr().add(x));
        let right = vld1q_f32(src.as_ptr().add(x + 1));
        let sum = vaddq_f32(vaddq_f32(left, right), vmulq_f32(center, two));
        vst1q_f32(out.as_mut_ptr().add(x), vmulq_f32(sum, quarter));
        x += 4;
    }
    x
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn gauss_h_avx_row_c1(src: &[f32], out: &mut [f32], w: usize) -> usize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let two = _mm256_set1_ps(2.0);
    let quarter = _mm256_set1_ps(0.25);
    let mut x = 1usize;
    while x + 9 <= w {
        let left = _mm256_loadu_ps(src.as_ptr().add(x - 1));
        let center = _mm256_loadu_ps(src.as_ptr().add(x));
        let right = _mm256_loadu_ps(src.as_ptr().add(x + 1));
        let sum = _mm256_add_ps(_mm256_add_ps(left, right), _mm256_mul_ps(center, two));
        _mm256_storeu_ps(out.as_mut_ptr().add(x), _mm256_mul_ps(sum, quarter));
        x += 8;
    }
    x
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn gauss_h_sse_row_c1(src: &[f32], out: &mut [f32], w: usize) -> usize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let two = _mm_set1_ps(2.0);
    let quarter = _mm_set1_ps(0.25);
    let mut x = 1usize;
    while x + 5 <= w {
        let left = _mm_loadu_ps(src.as_ptr().add(x - 1));
        let center = _mm_loadu_ps(src.as_ptr().add(x));
        let right = _mm_loadu_ps(src.as_ptr().add(x + 1));
        let sum = _mm_add_ps(_mm_add_ps(left, right), _mm_mul_ps(center, two));
        _mm_storeu_ps(out.as_mut_ptr().add(x), _mm_mul_ps(sum, quarter));
        x += 4;
    }
    x
}

/// Applies zero-padded 3x3 Gaussian blur per channel.
/// Kernel: `[[1,2,1],[2,4,2],[1,2,1]] / 16`.
///
/// Uses separable decomposition: horizontal `[1,2,1]`/4 then vertical `[1,2,1]`/4.
#[allow(unsafe_code, clippy::uninit_vec)]
pub fn gaussian_blur_3x3(input: &Tensor) -> Result<Tensor, ImgProcError> {
    let (h, w, channels) = hwc_shape(input)?;
    let data = input.data();
    let total = h * w * channels;
    // Horizontal pass: [1, 2, 1] / 4
    // SAFETY: every element is written by the horizontal pass below.
    let mut tmp = Vec::with_capacity(total);
    unsafe {
        tmp.set_len(total);
    }

    for y in 0..h {
        // SIMD fast path for c=1
        if channels == 1 && !cfg!(miri) {
            let src = &data[y * w..(y + 1) * w];
            let dst = &mut tmp[y * w..(y + 1) * w];
            // Left border
            {
                let center = src[0];
                let right = src[1.min(w - 1)];
                dst[0] = (center * 2.0 + right) / 4.0;
            }
            let done = gauss_h_simd_row_c1(src, dst, w);
            // Scalar tail
            for x in done..w.saturating_sub(1) {
                if x == 0 {
                    continue;
                }
                dst[x] = (src[x - 1] + src[x] * 2.0 + src[x + 1]) * 0.25;
            }
            // Right border
            if w > 1 {
                dst[w - 1] = (src[w - 2] + src[w - 1] * 2.0) / 4.0;
            }
            continue;
        }

        for c in 0..channels {
            // Left border (x=0)
            {
                let center = data[(y * w) * channels + c];
                let right = data[(y * w + 1.min(w - 1)) * channels + c];
                tmp[(y * w) * channels + c] = (center * 2.0 + right) / 4.0;
            }
            // Interior (no bounds checks needed)
            for x in 1..w.saturating_sub(1) {
                let base = y * w;
                let left = data[(base + x - 1) * channels + c];
                let center = data[(base + x) * channels + c];
                let right = data[(base + x + 1) * channels + c];
                tmp[(base + x) * channels + c] = (left + center * 2.0 + right) * 0.25;
            }
            // Right border (x=w-1)
            if w > 1 {
                let base = y * w;
                let left = data[(base + w - 2) * channels + c];
                let center = data[(base + w - 1) * channels + c];
                tmp[(base + w - 1) * channels + c] = (left + center * 2.0) / 4.0;
            }
        }
    }
    // Vertical pass: [1, 2, 1] / 4
    // SAFETY: every element is written by compute_row below.
    let mut out: Vec<f32> = Vec::with_capacity(total);
    unsafe {
        out.set_len(total);
    }
    let row_len = w * channels;

    let compute_row = |y: usize, row: &mut [f32]| {
        // SIMD fast path for c=1 interior rows
        if channels == 1 && y > 0 && y < h - 1 && !cfg!(miri) {
            let above = &tmp[(y - 1) * w..y * w];
            let center = &tmp[y * w..(y + 1) * w];
            let below = &tmp[(y + 1) * w..(y + 2) * w];
            let done = gauss_v_simd_row_c1(above, center, below, row, w);
            for x in done..w {
                row[x] = (above[x] + center[x] * 2.0 + below[x]) * 0.25;
            }
            return;
        }

        for x in 0..w {
            for c in 0..channels {
                let val = if y == 0 {
                    let center = tmp[x * channels + c];
                    let below = tmp[(1.min(h - 1) * w + x) * channels + c];
                    (center * 2.0 + below) / 4.0
                } else if y == h - 1 && h > 1 {
                    let above = tmp[((h - 2) * w + x) * channels + c];
                    let center = tmp[((h - 1) * w + x) * channels + c];
                    (above + center * 2.0) / 4.0
                } else {
                    let above = tmp[((y - 1) * w + x) * channels + c];
                    let center = tmp[(y * w + x) * channels + c];
                    let below = tmp[((y + 1) * w + x) * channels + c];
                    (above + center * 2.0 + below) * 0.25
                };
                row[x * channels + c] = val;
            }
        }
    };

    let pixels = h * w;

    #[cfg(target_os = "macos")]
    if pixels > 4096 && !cfg!(miri) {
        let out_ptr = super::SendPtr(out.as_mut_ptr());
        use super::u8ops::gcd;
        gcd::parallel_for(h, |y| {
            // SAFETY: each row writes to a disjoint slice of out.
            let row =
                unsafe { std::slice::from_raw_parts_mut(out_ptr.ptr().add(y * row_len), row_len) };
            compute_row(y, row);
        });
        return Tensor::from_vec(vec![h, w, channels], out).map_err(Into::into);
    }

    if pixels > 4096 {
        out.par_chunks_mut(row_len)
            .enumerate()
            .for_each(|(y, row)| compute_row(y, row));
    } else {
        out.chunks_mut(row_len)
            .enumerate()
            .for_each(|(y, row)| compute_row(y, row));
    }

    Tensor::from_vec(vec![h, w, channels], out).map_err(Into::into)
}

/// SIMD vertical `[1,2,1]`/4 pass for c=1
#[allow(unsafe_code)]
fn gauss_v_simd_row_c1(
    above: &[f32],
    center: &[f32],
    below: &[f32],
    out: &mut [f32],
    w: usize,
) -> usize {
    if w < 4 {
        return 0;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { gauss_v_neon_c1(above, center, below, out, w) };
        }
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("avx") {
        return unsafe { gauss_v_avx_c1(above, center, below, out, w) };
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("sse") {
            return unsafe { gauss_v_sse_c1(above, center, below, out, w) };
        }
    }
    0
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn gauss_v_neon_c1(
    above: &[f32],
    center: &[f32],
    below: &[f32],
    out: &mut [f32],
    w: usize,
) -> usize {
    use std::arch::aarch64::*;
    let two = vdupq_n_f32(2.0);
    let quarter = vdupq_n_f32(0.25);
    let mut x = 0usize;
    while x + 4 <= w {
        let a = vld1q_f32(above.as_ptr().add(x));
        let c = vld1q_f32(center.as_ptr().add(x));
        let b = vld1q_f32(below.as_ptr().add(x));
        let sum = vaddq_f32(vaddq_f32(a, b), vmulq_f32(c, two));
        vst1q_f32(out.as_mut_ptr().add(x), vmulq_f32(sum, quarter));
        x += 4;
    }
    x
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn gauss_v_avx_c1(
    above: &[f32],
    center: &[f32],
    below: &[f32],
    out: &mut [f32],
    w: usize,
) -> usize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let two = _mm256_set1_ps(2.0);
    let quarter = _mm256_set1_ps(0.25);
    let mut x = 0usize;
    while x + 8 <= w {
        let a = _mm256_loadu_ps(above.as_ptr().add(x));
        let c = _mm256_loadu_ps(center.as_ptr().add(x));
        let b = _mm256_loadu_ps(below.as_ptr().add(x));
        let sum = _mm256_add_ps(_mm256_add_ps(a, b), _mm256_mul_ps(c, two));
        _mm256_storeu_ps(out.as_mut_ptr().add(x), _mm256_mul_ps(sum, quarter));
        x += 8;
    }
    x
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn gauss_v_sse_c1(
    above: &[f32],
    center: &[f32],
    below: &[f32],
    out: &mut [f32],
    w: usize,
) -> usize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let two = _mm_set1_ps(2.0);
    let quarter = _mm_set1_ps(0.25);
    let mut x = 0usize;
    while x + 4 <= w {
        let a = _mm_loadu_ps(above.as_ptr().add(x));
        let c = _mm_loadu_ps(center.as_ptr().add(x));
        let b = _mm_loadu_ps(below.as_ptr().add(x));
        let sum = _mm_add_ps(_mm_add_ps(a, b), _mm_mul_ps(c, two));
        _mm_storeu_ps(out.as_mut_ptr().add(x), _mm_mul_ps(sum, quarter));
        x += 4;
    }
    x
}

/// Applies zero-padded 5x5 Gaussian blur per channel.
///
/// Uses separable decomposition: horizontal `[1,4,6,4,1]`/16 then vertical `[1,4,6,4,1]`/16.
pub fn gaussian_blur_5x5(input: &Tensor) -> Result<Tensor, ImgProcError> {
    let (h, w, channels) = hwc_shape(input)?;
    let data = input.data();
    let k: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];

    // Horizontal pass
    let mut tmp = vec![0.0f32; h * w * channels];
    for y in 0..h {
        for x in 0..w {
            for c in 0..channels {
                let base = y * w;
                let mut acc = 0.0f32;
                for i in 0..5 {
                    let sx = (x as isize + i as isize - 2).clamp(0, w as isize - 1) as usize;
                    acc += data[(base + sx) * channels + c] * k[i];
                }
                tmp[(base + x) * channels + c] = acc;
            }
        }
    }
    // Vertical pass
    let mut out = vec![0.0f32; h * w * channels];
    let row_len = w * channels;

    let compute_row = |y: usize, row: &mut [f32]| {
        for x in 0..w {
            for c in 0..channels {
                let mut acc = 0.0f32;
                for i in 0..5 {
                    let sy = (y as isize + i as isize - 2).clamp(0, h as isize - 1) as usize;
                    acc += tmp[(sy * w + x) * channels + c] * k[i];
                }
                row[x * channels + c] = acc;
            }
        }
    };

    if h * w > 4096 {
        out.par_chunks_mut(row_len)
            .enumerate()
            .for_each(|(y, row)| compute_row(y, row));
    } else {
        out.chunks_mut(row_len)
            .enumerate()
            .for_each(|(y, row)| compute_row(y, row));
    }

    Tensor::from_vec(vec![h, w, channels], out).map_err(Into::into)
}

// ── Generic 3x3 kernel (with interior/border split) ───────────────

pub(crate) fn apply_kernel_3x3(
    input: &Tensor,
    kernel: &[[f32; 3]; 3],
) -> Result<Tensor, ImgProcError> {
    let (h, w, channels) = hwc_shape(input)?;
    let data = input.data();
    let mut out = vec![0.0f32; h * w * channels];

    // Interior pixels: no bounds checks needed (y=1..h-1, x=1..w-1)
    let row_len = w * channels;
    let interior_h = h.saturating_sub(2); // number of interior rows

    let compute_interior_row = |y: usize, row: &mut [f32]| {
        for x in 1..w.saturating_sub(1) {
            for c in 0..channels {
                let mut acc = 0.0f32;
                let r0 = ((y - 1) * w + x - 1) * channels + c;
                let r1 = (y * w + x - 1) * channels + c;
                let r2 = ((y + 1) * w + x - 1) * channels + c;
                acc += data[r0] * kernel[0][0];
                acc += data[r0 + channels] * kernel[0][1];
                acc += data[r0 + 2 * channels] * kernel[0][2];
                acc += data[r1] * kernel[1][0];
                acc += data[r1 + channels] * kernel[1][1];
                acc += data[r1 + 2 * channels] * kernel[1][2];
                acc += data[r2] * kernel[2][0];
                acc += data[r2 + channels] * kernel[2][1];
                acc += data[r2 + 2 * channels] * kernel[2][2];
                row[x * channels + c] = acc;
            }
        }
    };

    if interior_h > 0 {
        // Slice out rows 1..h-1
        let interior_out = &mut out[row_len..row_len + interior_h * row_len];
        if h * w > 4096 {
            interior_out
                .par_chunks_mut(row_len)
                .enumerate()
                .for_each(|(i, row)| compute_interior_row(i + 1, row));
        } else {
            interior_out
                .chunks_mut(row_len)
                .enumerate()
                .for_each(|(i, row)| compute_interior_row(i + 1, row));
        }
    }

    // Border pixels: use bounds-checked path (top/bottom rows, left/right columns)
    let border_pixels = border_coords_3x3(h, w);
    for (y, x) in border_pixels {
        for c in 0..channels {
            let mut acc = 0.0f32;
            for ky in -1isize..=1 {
                for kx in -1isize..=1 {
                    let sy = y as isize + ky;
                    let sx = x as isize + kx;
                    if sy < 0 || sx < 0 || sy >= h as isize || sx >= w as isize {
                        continue;
                    }
                    let src = ((sy as usize) * w + sx as usize) * channels + c;
                    acc += data[src] * kernel[(ky + 1) as usize][(kx + 1) as usize];
                }
            }
            out[(y * w + x) * channels + c] = acc;
        }
    }

    Tensor::from_vec(vec![h, w, channels], out).map_err(Into::into)
}

/// Returns an iterator of (y, x) border pixel coordinates for a 3x3 kernel.
pub(crate) fn border_coords_3x3(h: usize, w: usize) -> Vec<(usize, usize)> {
    let mut coords = Vec::with_capacity(2 * w + 2 * h);
    // Top row
    for x in 0..w {
        coords.push((0, x));
    }
    // Bottom row (if h > 1)
    if h > 1 {
        for x in 0..w {
            coords.push((h - 1, x));
        }
    }
    // Left and right columns (excluding corners already added)
    for y in 1..h.saturating_sub(1) {
        coords.push((y, 0));
        if w > 1 {
            coords.push((y, w - 1));
        }
    }
    coords
}

/// Applies 3x3 Laplacian edge detection per channel.
pub fn laplacian_3x3(input: &Tensor) -> Result<Tensor, ImgProcError> {
    const KERNEL: [[f32; 3]; 3] = [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]];
    apply_kernel_3x3(input, &KERNEL)
}

// ── Median blur (sorting network for N=9) ─────────────────────────

/// Conditional swap at indices `a` and `b` in the array.
#[inline(always)]
fn cswap(v: &mut [f32; 9], a: usize, b: usize) {
    if v[a] > v[b] {
        v.swap(a, b);
    }
}

/// Finds the median of exactly 9 f32 values using an optimal sorting network.
/// Uses 25 compare-and-swap operations (optimal for N=9).
#[inline(always)]
fn median9(v: &mut [f32; 9]) -> f32 {
    // Optimal 9-element sorting network (25 comparisons)
    cswap(v, 0, 1);
    cswap(v, 3, 4);
    cswap(v, 6, 7);
    cswap(v, 1, 2);
    cswap(v, 4, 5);
    cswap(v, 7, 8);
    cswap(v, 0, 1);
    cswap(v, 3, 4);
    cswap(v, 6, 7);
    cswap(v, 0, 3);
    cswap(v, 3, 6);
    cswap(v, 0, 3);
    cswap(v, 1, 4);
    cswap(v, 4, 7);
    cswap(v, 1, 4);
    cswap(v, 2, 5);
    cswap(v, 5, 8);
    cswap(v, 2, 5);
    cswap(v, 1, 3);
    cswap(v, 5, 7);
    cswap(v, 2, 6);
    cswap(v, 4, 6);
    cswap(v, 2, 4);
    cswap(v, 2, 3);
    cswap(v, 5, 6);
    v[4]
}

/// Applies 3x3 median filter per channel.
pub fn median_blur_3x3(input: &Tensor) -> Result<Tensor, ImgProcError> {
    let (h, w, channels) = hwc_shape(input)?;
    let data = input.data();
    let mut out = vec![0.0f32; h * w * channels];
    let mut neighborhood = [0.0f32; 9];

    // Interior pixels: full 9-element neighborhood, use sorting network
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            for c in 0..channels {
                let r0 = ((y - 1) * w + x - 1) * channels + c;
                let r1 = (y * w + x - 1) * channels + c;
                let r2 = ((y + 1) * w + x - 1) * channels + c;
                neighborhood[0] = data[r0];
                neighborhood[1] = data[r0 + channels];
                neighborhood[2] = data[r0 + 2 * channels];
                neighborhood[3] = data[r1];
                neighborhood[4] = data[r1 + channels];
                neighborhood[5] = data[r1 + 2 * channels];
                neighborhood[6] = data[r2];
                neighborhood[7] = data[r2 + channels];
                neighborhood[8] = data[r2 + 2 * channels];
                out[(y * w + x) * channels + c] = median9(&mut neighborhood);
            }
        }
    }

    // Border pixels: variable neighborhood size, use sort
    let border = border_coords_3x3(h, w);
    for (y, x) in border {
        for c in 0..channels {
            let mut count = 0usize;
            for ky in -1isize..=1 {
                for kx in -1isize..=1 {
                    let sy = y as isize + ky;
                    let sx = x as isize + kx;
                    if sy < 0 || sx < 0 || sy >= h as isize || sx >= w as isize {
                        continue;
                    }
                    let src = ((sy as usize) * w + sx as usize) * channels + c;
                    neighborhood[count] = data[src];
                    count += 1;
                }
            }
            let slice = &mut neighborhood[..count];
            slice.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            out[(y * w + x) * channels + c] = slice[count / 2];
        }
    }

    Tensor::from_vec(vec![h, w, channels], out).map_err(Into::into)
}

/// Applies NxN median filter on a single-channel `[H, W, 1]` image.
///
/// `kernel_size` must be odd and >= 1. Border pixels use replicate (clamp)
/// padding so every pixel gets a full NxN neighborhood.
pub fn median_filter(input: &Tensor, kernel_size: usize) -> Result<Tensor, ImgProcError> {
    if kernel_size == 0 || kernel_size.is_multiple_of(2) {
        return Err(ImgProcError::InvalidBlockSize {
            block_size: kernel_size,
        });
    }
    let (h, w, c) = hwc_shape(input)?;
    if c != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: c,
        });
    }
    let data = input.data();
    let radius = (kernel_size / 2) as isize;
    let mut out = vec![0.0f32; h * w];
    let mut neighborhood = vec![0.0f32; kernel_size * kernel_size];

    for y in 0..h {
        for x in 0..w {
            let mut count = 0usize;
            for ky in -radius..=radius {
                for kx in -radius..=radius {
                    let sy = (y as isize + ky).clamp(0, h as isize - 1) as usize;
                    let sx = (x as isize + kx).clamp(0, w as isize - 1) as usize;
                    neighborhood[count] = data[sy * w + sx];
                    count += 1;
                }
            }
            let slice = &mut neighborhood[..count];
            slice.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            out[y * w + x] = slice[count / 2];
        }
    }

    Tensor::from_vec(vec![h, w, 1], out).map_err(Into::into)
}

/// NEON-accelerated bilateral filter for an entire interior row.
/// Processes pixels x in [x_start, x_end) where all neighbors are in bounds.
/// Uses double-batch (8 neighbors at a time) with interleaved LUT lookups.
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn bilateral_neon_row(
    data: &[f32],
    w: usize,
    y: usize,
    x_start: usize,
    x_end: usize,
    radius: i32,
    diameter: usize,
    spatial_lut: &[f32],
    color_lut: &[f32; 256],
    row_out: &mut [f32],
) {
    use std::arch::aarch64::*;

    let scale_255 = vdupq_n_f32(255.0);
    let max255 = vdupq_n_u32(255);
    let clut = color_lut.as_ptr();

    for x in x_start..x_end {
        let center = *data.get_unchecked(y * w + x);
        let center_v = vdupq_n_f32(center);
        let mut sum_v = vdupq_n_f32(0.0);
        let mut wsum_v = vdupq_n_f32(0.0);
        let mut sum_s = 0.0f32;
        let mut wsum_s = 0.0f32;

        for dy in -radius..=radius {
            let ny = (y as i32 + dy) as usize;
            let row_ptr = data.as_ptr().add(ny * w + x - (radius as usize));
            let sp_ptr = spatial_lut
                .as_ptr()
                .add(((dy + radius) as usize) * diameter);

            let mut dx = 0usize;

            // Process 8 neighbors at a time (2 NEON batches interleaved)
            while dx + 8 <= diameter {
                // Batch 1
                let n1 = vld1q_f32(row_ptr.add(dx));
                let sp1 = vld1q_f32(sp_ptr.add(dx));
                let diff1 = vabsq_f32(vsubq_f32(n1, center_v));
                let idx1 = vminq_u32(vcvtq_u32_f32(vmulq_f32(diff1, scale_255)), max255);
                let mut ia1 = [0u32; 4];
                vst1q_u32(ia1.as_mut_ptr(), idx1);

                // Batch 2
                let n2 = vld1q_f32(row_ptr.add(dx + 4));
                let sp2 = vld1q_f32(sp_ptr.add(dx + 4));
                let diff2 = vabsq_f32(vsubq_f32(n2, center_v));
                let idx2 = vminq_u32(vcvtq_u32_f32(vmulq_f32(diff2, scale_255)), max255);
                let mut ia2 = [0u32; 4];
                vst1q_u32(ia2.as_mut_ptr(), idx2);

                // Interleaved LUT lookups (while CPU works on store above)
                let cw1_arr = [
                    *clut.add(ia1[0] as usize),
                    *clut.add(ia1[1] as usize),
                    *clut.add(ia1[2] as usize),
                    *clut.add(ia1[3] as usize),
                ];
                let cw2_arr = [
                    *clut.add(ia2[0] as usize),
                    *clut.add(ia2[1] as usize),
                    *clut.add(ia2[2] as usize),
                    *clut.add(ia2[3] as usize),
                ];

                let wt1 = vmulq_f32(sp1, vld1q_f32(cw1_arr.as_ptr()));
                let wt2 = vmulq_f32(sp2, vld1q_f32(cw2_arr.as_ptr()));
                sum_v = vfmaq_f32(sum_v, n1, wt1);
                sum_v = vfmaq_f32(sum_v, n2, wt2);
                wsum_v = vaddq_f32(wsum_v, vaddq_f32(wt1, wt2));

                dx += 8;
            }

            // Remaining 4-element batch
            while dx + 4 <= diameter {
                let neighbors = vld1q_f32(row_ptr.add(dx));
                let spatial_w = vld1q_f32(sp_ptr.add(dx));
                let diff = vabsq_f32(vsubq_f32(neighbors, center_v));
                let idx_u32 = vminq_u32(vcvtq_u32_f32(vmulq_f32(diff, scale_255)), max255);
                let mut idx_arr = [0u32; 4];
                vst1q_u32(idx_arr.as_mut_ptr(), idx_u32);

                let cw_arr = [
                    *clut.add(idx_arr[0] as usize),
                    *clut.add(idx_arr[1] as usize),
                    *clut.add(idx_arr[2] as usize),
                    *clut.add(idx_arr[3] as usize),
                ];
                let wt = vmulq_f32(spatial_w, vld1q_f32(cw_arr.as_ptr()));
                sum_v = vfmaq_f32(sum_v, neighbors, wt);
                wsum_v = vaddq_f32(wsum_v, wt);
                dx += 4;
            }

            // Scalar tail
            while dx < diameter {
                let neighbor = *row_ptr.add(dx);
                let color_diff = (neighbor - center).abs();
                let color_idx = ((color_diff * 255.0) as usize).min(255);
                let wt = *sp_ptr.add(dx) * *clut.add(color_idx);
                sum_s += neighbor * wt;
                wsum_s += wt;
                dx += 1;
            }
        }

        let total_sum = vaddvq_f32(sum_v) + sum_s;
        let total_wsum = vaddvq_f32(wsum_v) + wsum_s;
        *row_out.get_unchecked_mut(x) = if total_wsum > 0.0 {
            total_sum / total_wsum
        } else {
            center
        };
    }
}

/// SSE2-accelerated bilateral filter for an entire interior row.
/// Mirrors `bilateral_neon_row`: processes pixels x in [x_start, x_end) where
/// all neighbors are in bounds. Uses a 256-entry color-weight LUT with
/// Schraudolph-free table lookup (same as NEON path).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse2")]
unsafe fn bilateral_sse_row(
    data: &[f32],
    w: usize,
    y: usize,
    x_start: usize,
    x_end: usize,
    radius: i32,
    diameter: usize,
    spatial_lut: &[f32],
    color_lut: &[f32; 256],
    row_out: &mut [f32],
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let scale_255 = _mm_set1_ps(255.0);
    let max_idx = 255i32;
    let clut = color_lut.as_ptr();

    for x in x_start..x_end {
        let center = *data.get_unchecked(y * w + x);
        let center_v = _mm_set1_ps(center);
        let mut sum_v = _mm_setzero_ps();
        let mut wsum_v = _mm_setzero_ps();
        let mut sum_s = 0.0f32;
        let mut wsum_s = 0.0f32;

        for dy in -radius..=radius {
            let ny = (y as i32 + dy) as usize;
            let row_ptr = data.as_ptr().add(ny * w + x - (radius as usize));
            let sp_ptr = spatial_lut
                .as_ptr()
                .add(((dy + radius) as usize) * diameter);

            let mut dx = 0usize;

            // Process 4 neighbors at a time
            while dx + 4 <= diameter {
                let neighbors = _mm_loadu_ps(row_ptr.add(dx));
                let spatial_w = _mm_loadu_ps(sp_ptr.add(dx));

                // |neighbor - center|
                let diff = _mm_sub_ps(neighbors, center_v);
                // SSE doesn't have vabsq, use max(diff, -diff)
                let neg_diff = _mm_sub_ps(_mm_setzero_ps(), diff);
                let abs_diff = _mm_max_ps(diff, neg_diff);

                // Convert to LUT index: clamp(abs_diff * 255, 0, 255)
                let scaled = _mm_mul_ps(abs_diff, scale_255);
                // Convert to int (truncate)
                let idx_i32 = _mm_cvttps_epi32(scaled);

                // Extract indices and clamp to [0, 255]
                // SSE2 doesn't have _mm_extract_epi32, use a union or store
                let mut idx_arr = [0i32; 4];
                _mm_storeu_si128(idx_arr.as_mut_ptr() as *mut __m128i, idx_i32);
                idx_arr[0] = idx_arr[0].min(max_idx).max(0);
                idx_arr[1] = idx_arr[1].min(max_idx).max(0);
                idx_arr[2] = idx_arr[2].min(max_idx).max(0);
                idx_arr[3] = idx_arr[3].min(max_idx).max(0);

                // Gather color weights from LUT
                let cw_arr = [
                    *clut.add(idx_arr[0] as usize),
                    *clut.add(idx_arr[1] as usize),
                    *clut.add(idx_arr[2] as usize),
                    *clut.add(idx_arr[3] as usize),
                ];

                let color_w = _mm_loadu_ps(cw_arr.as_ptr());
                let wt = _mm_mul_ps(spatial_w, color_w);

                // sum += neighbor * wt;  wsum += wt
                sum_v = _mm_add_ps(sum_v, _mm_mul_ps(neighbors, wt));
                wsum_v = _mm_add_ps(wsum_v, wt);

                dx += 4;
            }

            // Scalar tail
            while dx < diameter {
                let neighbor = *row_ptr.add(dx);
                let color_diff = (neighbor - center).abs();
                let color_idx = ((color_diff * 255.0) as usize).min(255);
                let wt = *sp_ptr.add(dx) * *clut.add(color_idx);
                sum_s += neighbor * wt;
                wsum_s += wt;
                dx += 1;
            }
        }

        // Horizontal sum of SSE vectors
        // sum_v = [a, b, c, d] -> a+b+c+d
        let hi = _mm_movehl_ps(sum_v, sum_v); // [c, d, c, d]
        let sum_lo = _mm_add_ps(sum_v, hi); // [a+c, b+d, ...]
        let sum_shuf = _mm_shuffle_ps(sum_lo, sum_lo, 1); // [b+d, ...]
        let total_sum_v = _mm_add_ss(sum_lo, sum_shuf);

        let hi_w = _mm_movehl_ps(wsum_v, wsum_v);
        let wsum_lo = _mm_add_ps(wsum_v, hi_w);
        let wsum_shuf = _mm_shuffle_ps(wsum_lo, wsum_lo, 1);
        let total_wsum_v = _mm_add_ss(wsum_lo, wsum_shuf);

        let total_sum = _mm_cvtss_f32(total_sum_v) + sum_s;
        let total_wsum = _mm_cvtss_f32(total_wsum_v) + wsum_s;

        *row_out.get_unchecked_mut(x) = if total_wsum > 0.0 {
            total_sum / total_wsum
        } else {
            center
        };
    }
}

/// Bilateral filter on a single-channel `[H, W, 1]` image.
///
/// Preserves edges while smoothing. `d` is the spatial kernel radius,
/// `sigma_color` controls color similarity range, `sigma_space` controls spatial decay.
///
/// Uses rayon parallelism + NEON SIMD on aarch64 / SSE2 on x86 for high performance.
#[allow(unsafe_code)]
pub fn bilateral_filter(
    input: &Tensor,
    d: usize,
    sigma_color: f32,
    sigma_space: f32,
) -> Result<Tensor, ImgProcError> {
    let (h, w, c) = hwc_shape(input)?;
    if c != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: c,
        });
    }
    let data = input.data();
    let mut out = vec![0.0f32; h * w];
    let radius = d as i32;
    let color_coeff = -0.5 / (sigma_color * sigma_color);
    let space_coeff = -0.5 / (sigma_space * sigma_space);

    // Precompute spatial weight LUT: spatial_lut[(dy+radius)*diameter + (dx+radius)]
    let diameter = (2 * radius + 1) as usize;
    let mut spatial_lut = vec![0.0f32; diameter * diameter];
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let spatial_dist_sq = (dy * dy + dx * dx) as f32;
            let idx = ((dy + radius) as usize) * diameter + (dx + radius) as usize;
            spatial_lut[idx] = (space_coeff * spatial_dist_sq).exp();
        }
    }

    // Precompute color weight LUT: color_lut[i] = exp(color_coeff * (i/255)^2) for i in 0..256
    let mut color_lut = [0.0f32; 256];
    for i in 0..256 {
        let diff = i as f32 / 255.0;
        color_lut[i] = (color_coeff * diff * diff).exp();
    }

    let radius_u = d;

    // Process pixel at (y, x) — scalar fallback (used for borders and non-NEON platforms)
    let process_pixel_scalar = |y: usize, x: usize| -> f32 {
        let center = data[y * w + x];
        let mut sum = 0.0f32;
        let mut weight_sum = 0.0f32;
        for dy in -radius..=radius {
            let ny = y as i32 + dy;
            if ny < 0 || ny >= h as i32 {
                continue;
            }
            let ny = ny as usize;
            let spatial_row_off = ((dy + radius) as usize) * diameter;
            for dx in -radius..=radius {
                let nx = x as i32 + dx;
                if nx < 0 || nx >= w as i32 {
                    continue;
                }
                let neighbor = data[ny * w + nx as usize];
                let color_diff = (neighbor - center).abs();
                let color_idx = ((color_diff * 255.0) as usize).min(255);
                let spatial_idx = spatial_row_off + (dx + radius) as usize;
                let wt = spatial_lut[spatial_idx] * color_lut[color_idx];
                sum += neighbor * wt;
                weight_sum += wt;
            }
        }
        if weight_sum > 0.0 {
            sum / weight_sum
        } else {
            center
        }
    };

    // Check SIMD availability once, outside the hot loop
    #[cfg(target_arch = "aarch64")]
    let use_neon = !cfg!(miri) && std::arch::is_aarch64_feature_detected!("neon");
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let use_sse2 = !cfg!(miri) && std::is_x86_feature_detected!("sse2");

    // Interior x range: [radius_u, w - radius_u)
    let x_start = radius_u;
    let x_end = w.saturating_sub(radius_u);

    // Parallel processing: each row is independent
    let compute_row = |y: usize, row_out: &mut [f32]| {
        let is_interior_y = y >= radius_u && y + radius_u < h;

        if is_interior_y {
            // Border pixels on left
            for x in 0..x_start {
                row_out[x] = process_pixel_scalar(y, x);
            }
            // Interior pixels — SIMD fast path
            #[cfg(target_arch = "aarch64")]
            if use_neon {
                unsafe {
                    bilateral_neon_row(
                        data,
                        w,
                        y,
                        x_start,
                        x_end,
                        radius,
                        diameter,
                        &spatial_lut,
                        &color_lut,
                        row_out,
                    );
                }
            } else {
                for x in x_start..x_end {
                    row_out[x] = process_pixel_scalar(y, x);
                }
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if use_sse2 {
                unsafe {
                    bilateral_sse_row(
                        data,
                        w,
                        y,
                        x_start,
                        x_end,
                        radius,
                        diameter,
                        &spatial_lut,
                        &color_lut,
                        row_out,
                    );
                }
            } else {
                for x in x_start..x_end {
                    row_out[x] = process_pixel_scalar(y, x);
                }
            }
            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
            {
                for x in x_start..x_end {
                    row_out[x] = process_pixel_scalar(y, x);
                }
            }
            // Border pixels on right
            for x in x_end..w {
                row_out[x] = process_pixel_scalar(y, x);
            }
        } else {
            // Entire row is border
            for x in 0..w {
                row_out[x] = process_pixel_scalar(y, x);
            }
        }
    };

    let pixels = h * w;

    #[cfg(target_os = "macos")]
    if pixels > 4096 && !cfg!(miri) {
        let out_ptr = super::SendPtr(out.as_mut_ptr());
        use super::u8ops::gcd;
        gcd::parallel_for(h, |y| {
            let row = unsafe { std::slice::from_raw_parts_mut(out_ptr.ptr().add(y * w), w) };
            compute_row(y, row);
        });
        return Tensor::from_vec(vec![h, w, 1], out).map_err(Into::into);
    }

    if pixels > 4096 {
        out.par_chunks_mut(w)
            .enumerate()
            .for_each(|(y, row)| compute_row(y, row));
    } else {
        out.chunks_mut(w)
            .enumerate()
            .for_each(|(y, row)| compute_row(y, row));
    }

    Tensor::from_vec(vec![h, w, 1], out).map_err(Into::into)
}

/// 2D convolution of a single-channel `[H, W, 1]` image with an arbitrary kernel `[KH, KW, 1]`.
///
/// Zero-pads the borders. Output has the same shape as input.
pub fn filter2d(input: &Tensor, kernel: &Tensor) -> Result<Tensor, ImgProcError> {
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
            let mut sum = 0.0f32;
            for ky in 0..kh {
                for kx in 0..kw {
                    let ny = y as i32 + ky as i32 - rh as i32;
                    let nx = x as i32 + kx as i32 - rw as i32;
                    if ny >= 0 && ny < h as i32 && nx >= 0 && nx < w as i32 {
                        sum += data[ny as usize * w + nx as usize] * kern[ky * kw + kx];
                    }
                }
            }
            out[y * w + x] = sum;
        }
    }

    Tensor::from_vec(vec![h, w, 1], out).map_err(Into::into)
}
