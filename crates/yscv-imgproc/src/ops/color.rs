//! # Safety contract
//!
//! Unsafe code categories:
//! 1. **SIMD intrinsics (NEON / AVX / SSE)** — ISA guard via runtime detection or `#[target_feature]`.
//! 2. **`SendConstPtr` / `SendPtr` for rayon** — each chunk writes non-overlapping rows.
//! 3. **`get_unchecked` in inner loops** — indices bounded by validated image dimensions.

use rayon::prelude::*;
use yscv_tensor::{AlignedVec, Tensor};

use super::super::ImgProcError;
use super::super::shape::hwc_shape;

/// Converts RGB image `[H, W, 3]` to grayscale `[H, W, 1]`.
#[allow(unsafe_code, clippy::uninit_vec)]
pub fn rgb_to_grayscale(input: &Tensor) -> Result<Tensor, ImgProcError> {
    let (h, w, c) = hwc_shape(input)?;
    if c != 3 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 3,
            got: c,
        });
    }

    let total = h * w;
    let data = input.data();

    // Use uninit aligned buffer — SIMD + scalar tail will write every element.
    let mut out = AlignedVec::<f32>::uninitialized(total);

    // For large images, parallelize across row chunks to saturate memory bandwidth.
    if total >= 4096 && !cfg!(miri) {
        let n_chunks = 8usize.min(h);
        let chunk_h = h.div_ceil(n_chunks);
        let src_ptr = super::SendConstPtr(data.as_ptr());
        let dst_ptr = super::SendPtr(out.as_mut_ptr());
        let w_c = w;
        let data_len = data.len();
        // SAFETY: each chunk writes to non-overlapping region
        (0..n_chunks).into_par_iter().for_each(|chunk| {
            let y_start = chunk * chunk_h;
            let y_end = ((chunk + 1) * chunk_h).min(h);
            let n = (y_end - y_start) * w_c;
            let src_off = y_start * w_c * 3;
            let dst_off = y_start * w_c;
            // SAFETY: pointer and length from validated image data; parallel chunks are non-overlapping.
            let chunk_src = unsafe { std::slice::from_raw_parts(src_ptr.ptr(), data_len) };
            let chunk_src = &chunk_src[src_off..src_off + n * 3];
            // SAFETY: dst_off + n bounded by output allocation; chunks are disjoint.
            let chunk_dst =
                unsafe { std::slice::from_raw_parts_mut(dst_ptr.ptr().add(dst_off), n) };
            let done = grayscale_simd_row(chunk_src, chunk_dst);
            for i in done..n {
                let base = i * 3;
                chunk_dst[i] = 0.299 * chunk_src[base]
                    + 0.587 * chunk_src[base + 1]
                    + 0.114 * chunk_src[base + 2];
            }
        });
    } else {
        // Small image — flat single-threaded
        let done = if !cfg!(miri) {
            grayscale_simd_row(data, &mut out)
        } else {
            0
        };
        for x in done..total {
            let base = x * 3;
            out[x] = 0.299 * data[base] + 0.587 * data[base + 1] + 0.114 * data[base + 2];
        }
    }

    Tensor::from_aligned(vec![h, w, 1], out).map_err(Into::into)
}

/// Returns number of pixels processed by SIMD.
#[allow(unsafe_code)]
fn grayscale_simd_row(src: &[f32], dst: &mut [f32]) -> usize {
    let w = dst.len();
    if w < 4 {
        return 0;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: ISA guard (feature detection) above.
            return unsafe { grayscale_neon_row(src, dst) };
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            // SAFETY: ISA guard (feature detection) above.
            return unsafe { grayscale_avx_row(src, dst) };
        }
        if std::is_x86_feature_detected!("sse") {
            // SAFETY: ISA guard (feature detection) above.
            return unsafe { grayscale_sse_row(src, dst) };
        }
    }

    0
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn grayscale_neon_row(src: &[f32], dst: &mut [f32]) -> usize {
    use std::arch::aarch64::{vdupq_n_f32, vfmaq_f32, vld3q_f32, vmulq_f32, vst1q_f32};

    let w = dst.len();
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();
    let v_r = vdupq_n_f32(0.299);
    let v_g = vdupq_n_f32(0.587);
    let v_b = vdupq_n_f32(0.114);
    let mut x = 0usize;

    // vld3q_f32 deinterleaves 12 floats into 3 × float32x4
    while x + 4 <= w {
        let rgb = vld3q_f32(sp.add(x * 3));
        let gray = vfmaq_f32(vfmaq_f32(vmulq_f32(rgb.0, v_r), rgb.1, v_g), rgb.2, v_b);
        vst1q_f32(dp.add(x), gray);
        x += 4;
    }
    x
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn grayscale_avx_row(src: &[f32], dst: &mut [f32]) -> usize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let w = dst.len();
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();
    let v_r = _mm256_set1_ps(0.299);
    let v_g = _mm256_set1_ps(0.587);
    let v_b = _mm256_set1_ps(0.114);
    let mut x = 0usize;

    // Manual deinterleave: load 24 floats (8 RGB pixels), extract R/G/B lanes via scalar gather.
    // AVX has no float deinterleave intrinsic equivalent to NEON's vld3q_f32, so we use
    // _mm256_set_ps with individual scalar loads, same approach as the SSE version but for 8 pixels.
    while x + 8 <= w {
        let p = sp.add(x * 3);
        let r = _mm256_set_ps(
            *p.add(21), // r7
            *p.add(18), // r6
            *p.add(15), // r5
            *p.add(12), // r4
            *p.add(9),  // r3
            *p.add(6),  // r2
            *p.add(3),  // r1
            *p.add(0),  // r0
        );
        let g = _mm256_set_ps(
            *p.add(22), // g7
            *p.add(19), // g6
            *p.add(16), // g5
            *p.add(13), // g4
            *p.add(10), // g3
            *p.add(7),  // g2
            *p.add(4),  // g1
            *p.add(1),  // g0
        );
        let b = _mm256_set_ps(
            *p.add(23), // b7
            *p.add(20), // b6
            *p.add(17), // b5
            *p.add(14), // b4
            *p.add(11), // b3
            *p.add(8),  // b2
            *p.add(5),  // b1
            *p.add(2),  // b0
        );

        // gray = r*0.299 + g*0.587 + b*0.114
        let gray = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(r, v_r), _mm256_mul_ps(g, v_g)),
            _mm256_mul_ps(b, v_b),
        );
        _mm256_storeu_ps(dp.add(x), gray);
        x += 8;
    }
    x
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn grayscale_sse_row(src: &[f32], dst: &mut [f32]) -> usize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let w = dst.len();
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();
    let v_r = _mm_set1_ps(0.299);
    let v_g = _mm_set1_ps(0.587);
    let v_b = _mm_set1_ps(0.114);
    let mut x = 0usize;

    // Manual deinterleave: load 12 floats, extract R/G/B lanes
    while x + 4 <= w {
        let p = sp.add(x * 3);
        // Load [r0,g0,b0,r1], [g1,b1,r2,g2], [b2,r3,g3,b3]
        let _v0 = _mm_loadu_ps(p);
        let _v1 = _mm_loadu_ps(p.add(4));
        let _v2 = _mm_loadu_ps(p.add(8));

        // Shuffle to extract R, G, B channels
        // R = [r0, r1, r2, r3] = v0[0], v0[3], v1[2], v2[1]
        // G = [g0, g1, g2, g3] = v0[1], v1[0], v1[3], v2[2]
        // B = [b0, b1, b2, b3] = v0[2], v1[1], v2[0], v2[3]
        let r = _mm_set_ps(
            *p.add(9), // r3
            *p.add(6), // r2
            *p.add(3), // r1
            *p.add(0), // r0
        );
        let g = _mm_set_ps(
            *p.add(10), // g3
            *p.add(7),  // g2
            *p.add(4),  // g1
            *p.add(1),  // g0
        );
        let b = _mm_set_ps(
            *p.add(11), // b3
            *p.add(8),  // b2
            *p.add(5),  // b1
            *p.add(2),  // b0
        );

        // gray = r*0.299 + g*0.587 + b*0.114
        let gray = _mm_add_ps(
            _mm_add_ps(_mm_mul_ps(r, v_r), _mm_mul_ps(g, v_g)),
            _mm_mul_ps(b, v_b),
        );
        _mm_storeu_ps(dp.add(x), gray);
        x += 4;
    }
    x
}

/// Converts RGB `[H, W, 3]` to HSV `[H, W, 3]`.
/// H is in `[0, 1]` (scaled from [0, 360]), S and V are in `[0, 1]`.
#[allow(unsafe_code)]
pub fn rgb_to_hsv(input: &Tensor) -> Result<Tensor, ImgProcError> {
    let (h, w, c) = hwc_shape(input)?;
    if c != 3 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 3,
            got: c,
        });
    }

    let data = input.data();
    let pixels = h * w;
    let mut out = AlignedVec::<f32>::uninitialized(pixels * 3);
    let row_stride = w * 3;

    #[inline(always)]
    fn hsv_pixel(r: f32, g: f32, b: f32, dst: &mut [f32]) {
        let max_c = r.max(g).max(b);
        let min_c = r.min(g).min(b);
        let delta = max_c - min_c;
        dst[2] = max_c; // V
        dst[1] = if max_c > 0.0 { delta / max_c } else { 0.0 }; // S
        dst[0] = if delta < 1e-8 {
            0.0
        } else if (max_c - r).abs() < 1e-8 {
            ((g - b) / delta).rem_euclid(6.0) / 6.0
        } else if (max_c - g).abs() < 1e-8 {
            ((b - r) / delta + 2.0) / 6.0
        } else {
            ((r - g) / delta + 4.0) / 6.0
        }; // H
    }

    // Row-parallel dispatch (not per-pixel — avoids rayon scheduling overhead)
    if pixels > 4096 {
        let _src_ptr = super::SendConstPtr(data.as_ptr());
        let _dst_ptr = super::SendPtr(out.as_mut_ptr());

        #[cfg(target_os = "macos")]
        {
            super::u8ops::gcd::parallel_for(h, |y| {
                // SAFETY: pointer and length from validated image data; rows are non-overlapping.
                let src = unsafe {
                    std::slice::from_raw_parts(_src_ptr.ptr().add(y * row_stride), row_stride)
                };
                // SAFETY: pointer and length from validated image data; rows are non-overlapping.
                let dst = unsafe {
                    std::slice::from_raw_parts_mut(_dst_ptr.ptr().add(y * row_stride), row_stride)
                };
                let done = if !cfg!(miri) {
                    hsv_simd_row(src, dst)
                } else {
                    0
                };
                for i in done..w {
                    hsv_pixel(
                        src[i * 3],
                        src[i * 3 + 1],
                        src[i * 3 + 2],
                        &mut dst[i * 3..i * 3 + 3],
                    );
                }
            });
        }
        #[cfg(not(target_os = "macos"))]
        {
            out.par_chunks_mut(row_stride)
                .enumerate()
                .for_each(|(y, dst_row)| {
                    let src_row = &data[y * row_stride..(y + 1) * row_stride];
                    let done = if !cfg!(miri) {
                        hsv_simd_row(src_row, dst_row)
                    } else {
                        0
                    };
                    for i in done..w {
                        hsv_pixel(
                            src_row[i * 3],
                            src_row[i * 3 + 1],
                            src_row[i * 3 + 2],
                            &mut dst_row[i * 3..i * 3 + 3],
                        );
                    }
                });
        }
    } else {
        let done = if !cfg!(miri) {
            hsv_simd_row(data, &mut out)
        } else {
            0
        };
        for i in done..pixels {
            hsv_pixel(
                data[i * 3],
                data[i * 3 + 1],
                data[i * 3 + 2],
                &mut out[i * 3..i * 3 + 3],
            );
        }
    }

    Tensor::from_aligned(vec![h, w, 3], out).map_err(Into::into)
}

/// Returns number of pixels processed by SIMD.
#[allow(unsafe_code)]
fn hsv_simd_row(src: &[f32], dst: &mut [f32]) -> usize {
    let w = dst.len() / 3;
    if w < 4 {
        return 0;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: ISA guard (feature detection) above.
            return unsafe { hsv_neon_row(src, dst) };
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            // SAFETY: ISA guard (feature detection) above.
            return unsafe { hsv_avx_row(src, dst) };
        }
        if std::is_x86_feature_detected!("sse") {
            // SAFETY: ISA guard (feature detection) above.
            return unsafe { hsv_sse_row(src, dst) };
        }
    }

    0
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn hsv_neon_row(src: &[f32], dst: &mut [f32]) -> usize {
    use std::arch::aarch64::*;

    let w = dst.len() / 3;
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();

    let v_zero = vdupq_n_f32(0.0);
    let v_eps = vdupq_n_f32(1e-8);
    let v_two = vdupq_n_f32(2.0);
    let v_four = vdupq_n_f32(4.0);
    let v_six = vdupq_n_f32(6.0);
    let v_inv6 = vdupq_n_f32(1.0 / 6.0);

    let mut x = 0usize;
    while x + 4 <= w {
        let rgb = vld3q_f32(sp.add(x * 3));
        let r = rgb.0;
        let g = rgb.1;
        let b = rgb.2;

        let max_c = vmaxq_f32(vmaxq_f32(r, g), b);
        let min_c = vminq_f32(vminq_f32(r, g), b);
        let delta = vsubq_f32(max_c, min_c);

        // V = max
        let v_val = max_c;

        // S = delta / max (0 where max == 0)
        let max_gt_zero = vcgtq_f32(max_c, v_zero);
        let s_val = vbslq_f32(max_gt_zero, vdivq_f32(delta, max_c), v_zero);

        // H computation:
        // delta < eps -> H = 0
        // max == r -> H = ((g - b) / delta) mod 6
        // max == g -> H = (b - r) / delta + 2
        // max == b -> H = (r - g) / delta + 4
        // Then H /= 6
        let delta_big = vcgtq_f32(delta, v_eps);

        // Compute all three H candidates
        let h_r = vdivq_f32(vsubq_f32(g, b), delta); // (g - b) / delta
        let h_g = vaddq_f32(vdivq_f32(vsubq_f32(b, r), delta), v_two); // (b - r) / delta + 2
        let h_b = vaddq_f32(vdivq_f32(vsubq_f32(r, g), delta), v_four); // (r - g) / delta + 4

        // Select based on which channel is max
        // Start with h_b (default), override with h_g where max==g, override with h_r where max==r
        let max_is_r = vceqq_f32(max_c, r);
        let max_is_g = vceqq_f32(max_c, g);

        let h_raw = vbslq_f32(max_is_r, h_r, vbslq_f32(max_is_g, h_g, h_b));

        // rem_euclid(6): add 6 where negative, then divide by 6
        let h_neg = vcltq_f32(h_raw, v_zero);
        let h_mod = vbslq_f32(h_neg, vaddq_f32(h_raw, v_six), h_raw);
        let h_scaled = vmulq_f32(h_mod, v_inv6);

        // Zero out H where delta is tiny
        let h_val = vbslq_f32(delta_big, h_scaled, v_zero);

        vst3q_f32(dp.add(x * 3), float32x4x3_t(h_val, s_val, v_val));
        x += 4;
    }
    x
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn hsv_avx_row(src: &[f32], dst: &mut [f32]) -> usize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let w = dst.len() / 3;
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();

    let v_zero = _mm256_setzero_ps();
    let v_eps = _mm256_set1_ps(1e-8);
    let v_two = _mm256_set1_ps(2.0);
    let v_four = _mm256_set1_ps(4.0);
    let v_six = _mm256_set1_ps(6.0);
    let v_inv6 = _mm256_set1_ps(1.0 / 6.0);
    let v_one = _mm256_set1_ps(1.0);

    let mut x = 0usize;
    while x + 8 <= w {
        let p = sp.add(x * 3);
        // Gather R, G, B from interleaved RGB (8 pixels = 24 floats)
        let r = _mm256_set_ps(
            *p.add(21),
            *p.add(18),
            *p.add(15),
            *p.add(12),
            *p.add(9),
            *p.add(6),
            *p.add(3),
            *p.add(0),
        );
        let g = _mm256_set_ps(
            *p.add(22),
            *p.add(19),
            *p.add(16),
            *p.add(13),
            *p.add(10),
            *p.add(7),
            *p.add(4),
            *p.add(1),
        );
        let b = _mm256_set_ps(
            *p.add(23),
            *p.add(20),
            *p.add(17),
            *p.add(14),
            *p.add(11),
            *p.add(8),
            *p.add(5),
            *p.add(2),
        );

        let max_c = _mm256_max_ps(_mm256_max_ps(r, g), b);
        let min_c = _mm256_min_ps(_mm256_min_ps(r, g), b);
        let delta = _mm256_sub_ps(max_c, min_c);

        // V = max
        let v_val = max_c;

        // S = delta / max (0 where max == 0)
        let max_gt_zero = _mm256_cmp_ps(max_c, v_zero, _CMP_GT_OQ);
        let s_raw = _mm256_div_ps(delta, max_c);
        let s_val = _mm256_and_ps(max_gt_zero, s_raw);

        // H computation
        let delta_big = _mm256_cmp_ps(delta, v_eps, _CMP_GT_OQ);

        // Safe division
        let safe_delta = _mm256_or_ps(
            _mm256_and_ps(delta_big, delta),
            _mm256_andnot_ps(delta_big, v_one),
        );

        let h_r = _mm256_div_ps(_mm256_sub_ps(g, b), safe_delta);
        let h_g = _mm256_add_ps(_mm256_div_ps(_mm256_sub_ps(b, r), safe_delta), v_two);
        let h_b = _mm256_add_ps(_mm256_div_ps(_mm256_sub_ps(r, g), safe_delta), v_four);

        let max_is_r = _mm256_cmp_ps(max_c, r, _CMP_EQ_OQ);
        let max_is_g = _mm256_cmp_ps(max_c, g, _CMP_EQ_OQ);

        let h1 = _mm256_or_ps(
            _mm256_and_ps(max_is_g, h_g),
            _mm256_andnot_ps(max_is_g, h_b),
        );
        let h_raw = _mm256_or_ps(_mm256_and_ps(max_is_r, h_r), _mm256_andnot_ps(max_is_r, h1));

        // rem_euclid(6): add 6 where negative
        let h_neg = _mm256_cmp_ps(h_raw, v_zero, _CMP_LT_OQ);
        let h_mod = _mm256_or_ps(
            _mm256_and_ps(h_neg, _mm256_add_ps(h_raw, v_six)),
            _mm256_andnot_ps(h_neg, h_raw),
        );
        let h_scaled = _mm256_mul_ps(h_mod, v_inv6);
        let h_val = _mm256_and_ps(delta_big, h_scaled);

        // Scatter back to interleaved HSV
        let ha: [f32; 8] = std::mem::transmute(h_val);
        let sa: [f32; 8] = std::mem::transmute(s_val);
        let va: [f32; 8] = std::mem::transmute(v_val);
        let op = dp.add(x * 3);
        for j in 0..8 {
            *op.add(j * 3) = ha[j];
            *op.add(j * 3 + 1) = sa[j];
            *op.add(j * 3 + 2) = va[j];
        }
        x += 8;
    }
    x
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn hsv_sse_row(src: &[f32], dst: &mut [f32]) -> usize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let w = dst.len() / 3;
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();

    let v_zero = _mm_setzero_ps();
    let v_eps = _mm_set1_ps(1e-8);
    let v_two = _mm_set1_ps(2.0);
    let v_four = _mm_set1_ps(4.0);
    let v_six = _mm_set1_ps(6.0);
    let v_inv6 = _mm_set1_ps(1.0 / 6.0);

    let mut x = 0usize;
    while x + 4 <= w {
        let p = sp.add(x * 3);
        // Gather R, G, B from interleaved RGB
        let r = _mm_set_ps(*p.add(9), *p.add(6), *p.add(3), *p.add(0));
        let g = _mm_set_ps(*p.add(10), *p.add(7), *p.add(4), *p.add(1));
        let b = _mm_set_ps(*p.add(11), *p.add(8), *p.add(5), *p.add(2));

        let max_c = _mm_max_ps(_mm_max_ps(r, g), b);
        let min_c = _mm_min_ps(_mm_min_ps(r, g), b);
        let delta = _mm_sub_ps(max_c, min_c);

        // V = max
        let v_val = max_c;

        // S = delta / max (0 where max == 0)
        let max_gt_zero = _mm_cmpgt_ps(max_c, v_zero);
        let s_raw = _mm_div_ps(delta, max_c);
        let s_val = _mm_and_ps(max_gt_zero, s_raw);

        // H computation
        let delta_big = _mm_cmpgt_ps(delta, v_eps);

        // Safe division: use delta where > eps, else 1.0 to avoid div-by-zero
        let safe_delta = _mm_or_ps(
            _mm_and_ps(delta_big, delta),
            _mm_andnot_ps(delta_big, _mm_set1_ps(1.0)),
        );

        let h_r = _mm_div_ps(_mm_sub_ps(g, b), safe_delta);
        let h_g = _mm_add_ps(_mm_div_ps(_mm_sub_ps(b, r), safe_delta), v_two);
        let h_b = _mm_add_ps(_mm_div_ps(_mm_sub_ps(r, g), safe_delta), v_four);

        // Select: max==r -> h_r, max==g -> h_g, else h_b
        let max_is_r = _mm_cmpeq_ps(max_c, r);
        let max_is_g = _mm_cmpeq_ps(max_c, g);

        // Start with h_b, blend in h_g where max==g, blend in h_r where max==r
        let h1 = _mm_or_ps(_mm_and_ps(max_is_g, h_g), _mm_andnot_ps(max_is_g, h_b));
        let h_raw = _mm_or_ps(_mm_and_ps(max_is_r, h_r), _mm_andnot_ps(max_is_r, h1));

        // rem_euclid(6): add 6 where negative
        let h_neg = _mm_cmplt_ps(h_raw, v_zero);
        let h_mod = _mm_or_ps(
            _mm_and_ps(h_neg, _mm_add_ps(h_raw, v_six)),
            _mm_andnot_ps(h_neg, h_raw),
        );
        let h_scaled = _mm_mul_ps(h_mod, v_inv6);

        // Zero out H where delta is tiny
        let h_val = _mm_and_ps(delta_big, h_scaled);

        // Scatter back to interleaved HSV
        let ha: [f32; 4] = std::mem::transmute(h_val);
        let sa: [f32; 4] = std::mem::transmute(s_val);
        let va: [f32; 4] = std::mem::transmute(v_val);
        let op = dp.add(x * 3);
        for j in 0..4 {
            *op.add(j * 3) = ha[j];
            *op.add(j * 3 + 1) = sa[j];
            *op.add(j * 3 + 2) = va[j];
        }
        x += 4;
    }
    x
}

/// Converts HSV `[H, W, 3]` to RGB `[H, W, 3]`.
/// Expects H in `[0, 1]`, S in `[0, 1]`, V in `[0, 1]`.
pub fn hsv_to_rgb(input: &Tensor) -> Result<Tensor, ImgProcError> {
    let (h, w, c) = hwc_shape(input)?;
    if c != 3 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 3,
            got: c,
        });
    }

    let data = input.data();
    let pixels = h * w;
    let mut out = vec![0.0f32; pixels * 3];

    for i in 0..pixels {
        let hue = data[i * 3];
        let s = data[i * 3 + 1];
        let v = data[i * 3 + 2];

        let h6 = hue * 6.0;
        let sector = h6.floor() as i32;
        let f = h6 - sector as f32;
        let p = v * (1.0 - s);
        let q = v * (1.0 - s * f);
        let t = v * (1.0 - s * (1.0 - f));

        let (r, g, b) = match sector % 6 {
            0 => (v, t, p),
            1 => (q, v, p),
            2 => (p, v, t),
            3 => (p, q, v),
            4 => (t, p, v),
            _ => (v, p, q),
        };

        out[i * 3] = r;
        out[i * 3 + 1] = g;
        out[i * 3 + 2] = b;
    }

    Tensor::from_vec(vec![h, w, 3], out).map_err(Into::into)
}

/// Pre-computed sRGB → linear LUT (256 entries).
/// Maps byte values [0..255] to linear f32 values via `srgb_to_linear(i/255)`.
fn srgb_lut() -> &'static [f32; 256] {
    use std::sync::OnceLock;
    static LUT: OnceLock<[f32; 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut table = [0.0f32; 256];
        for i in 0..256 {
            table[i] = srgb_to_linear(i as f64 / 255.0) as f32;
        }
        table
    })
}

/// Fast sRGB → linear for f32 values in [0,1] using the 256-entry LUT with linear interpolation.
#[inline]
fn srgb_to_linear_fast(v: f32) -> f32 {
    let lut = srgb_lut();
    let scaled = v * 255.0;
    let lo = (scaled as usize).min(254);
    let frac = scaled - lo as f32;
    lut[lo] * (1.0 - frac) + lut[lo + 1] * frac
}

/// Converts RGB `[H, W, 3]` to CIE LAB `[H, W, 3]`.
///
/// Input RGB values are expected in `[0, 1]`.
/// Output: L in `[0, 100]`, a and b roughly in `[-128, 127]`.
///
/// Uses fixed-point integer math for the XYZ matrix multiply when inputs are
/// quantizable to u8 (the common imread path), and rayon row parallelism for
/// large images (>4096 pixels).
#[allow(unsafe_code, clippy::uninit_vec)]
pub fn rgb_to_lab(input: &Tensor) -> Result<Tensor, ImgProcError> {
    let (h, w, c) = hwc_shape(input)?;
    if c != 3 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 3,
            got: c,
        });
    }

    let data = input.data();
    let pixels = h * w;

    // For large images, use parallel row processing. Each row is independent.
    if pixels > 4096 {
        let mut out = AlignedVec::<f32>::uninitialized(pixels * 3);
        // Force LUT initialization before parallel section.
        let _ = srgb_lut();
        let _ = lab_f_lut();

        // Pre-compute direct u8->lab_f(XYZ/Xn) LUTs: for each u8 value, store the
        // contribution to fx/fy/fz after sRGB decode and matrix multiply.
        // This precomputes srgb_fp values and XYZ matrix rows.
        let srgb = srgb_lut();

        // Precompute sRGB -> linear f32 (already have this in srgb_lut)
        // But also precompute XYZ contributions per channel per u8 value.
        // xyz_r[i] = srgb[i] * M_row, etc. This replaces the 3 muls + 2 adds per pixel per XYZ row.
        const INV_XN: f32 = 1.0 / 0.95047;
        const INV_ZN: f32 = 1.0 / 1.08883;

        // For each u8 value, precompute the XYZ contribution scaled by white point.
        // x_contrib_r[i] = srgb[i] * 0.4124564 * INV_XN, etc.
        let mut x_r = [0.0f32; 256];
        let mut x_g = [0.0f32; 256];
        let mut x_b = [0.0f32; 256];
        let mut y_r = [0.0f32; 256];
        let mut y_g = [0.0f32; 256];
        let mut y_b = [0.0f32; 256];
        let mut z_r = [0.0f32; 256];
        let mut z_g = [0.0f32; 256];
        let mut z_b = [0.0f32; 256];
        for i in 0..256 {
            let lin = srgb[i];
            x_r[i] = lin * 0.4124564 * INV_XN;
            x_g[i] = lin * 0.3575761 * INV_XN;
            x_b[i] = lin * 0.1804375 * INV_XN;
            y_r[i] = lin * 0.2126729;
            y_g[i] = lin * 0.7151522;
            y_b[i] = lin * 0.0721750;
            z_r[i] = lin * 0.0193339 * INV_ZN;
            z_g[i] = lin * 0.119_192 * INV_ZN;
            z_b[i] = lin * 0.9503041 * INV_ZN;
        }

        let src_ptr = super::SendConstPtr(data.as_ptr());
        let dst_ptr = super::SendPtr(out.as_mut_ptr());
        let row_stride = w * 3;

        let process_row = |row: usize| {
            // SAFETY: pointer and length from validated image data; rows are non-overlapping.
            let src_row = unsafe {
                std::slice::from_raw_parts(src_ptr.ptr().add(row * row_stride), row_stride)
            };
            // SAFETY: pointer and length from validated image data; rows are non-overlapping.
            let out_row = unsafe {
                std::slice::from_raw_parts_mut(dst_ptr.ptr().add(row * row_stride), row_stride)
            };
            for px in 0..w {
                let base = px * 3;
                // SAFETY: base+2 < row_stride = w*3, indices ri/gi/bi clamped to 0..255 by u8 cast.
                let ri = unsafe { (src_row.get_unchecked(base) * 255.0) as u8 as usize };
                let gi = unsafe { (src_row.get_unchecked(base + 1) * 255.0) as u8 as usize };
                let bi = unsafe { (src_row.get_unchecked(base + 2) * 255.0) as u8 as usize };

                // XYZ / white_point via precomputed per-channel LUTs (3 adds per component)
                // SAFETY: ri/gi/bi are u8 values (0..255); LUT size is 256.
                let xn = unsafe {
                    x_r.get_unchecked(ri) + x_g.get_unchecked(gi) + x_b.get_unchecked(bi)
                };
                // SAFETY: ri/gi/bi are u8 values (0..255); LUT size is 256.
                let yn = unsafe {
                    y_r.get_unchecked(ri) + y_g.get_unchecked(gi) + y_b.get_unchecked(bi)
                };
                // SAFETY: ri/gi/bi are u8 values (0..255); LUT size is 256.
                let zn = unsafe {
                    z_r.get_unchecked(ri) + z_g.get_unchecked(gi) + z_b.get_unchecked(bi)
                };

                let fx = lab_f_fast(xn);
                let fy = lab_f_fast(yn);
                let fz = lab_f_fast(zn);

                // SAFETY: base+2 < row_stride = w*3, bounds checked by px < w loop.
                unsafe {
                    *out_row.get_unchecked_mut(base) = 116.0 * fy - 16.0;
                    *out_row.get_unchecked_mut(base + 1) = 500.0 * (fx - fy);
                    *out_row.get_unchecked_mut(base + 2) = 200.0 * (fy - fz);
                }
            }
        };

        #[cfg(target_os = "macos")]
        if !cfg!(miri) {
            use super::u8ops::gcd;
            gcd::parallel_for(h, process_row);
            return Tensor::from_aligned(vec![h, w, 3], out).map_err(Into::into);
        }

        out.par_chunks_mut(w * 3)
            .enumerate()
            .for_each(|(row, _out_row)| {
                process_row(row);
            });

        return Tensor::from_aligned(vec![h, w, 3], out).map_err(Into::into);
    }

    // Small image path: single-threaded SIMD + scalar tail.
    let mut out = AlignedVec::<f32>::uninitialized(pixels * 3);

    // D65 white point
    const XN: f32 = 0.95047;
    const YN: f32 = 1.0;
    const ZN: f32 = 1.08883;

    let done = if !cfg!(miri) {
        lab_simd_row(data, &mut out)
    } else {
        0
    };

    // Scalar tail — uses LUT for gamma, f32 math for speed (matching SIMD path).
    for i in done..pixels {
        let lin_r = srgb_to_linear_fast(data[i * 3]);
        let lin_g = srgb_to_linear_fast(data[i * 3 + 1]);
        let lin_b = srgb_to_linear_fast(data[i * 3 + 2]);

        let x = 0.4124564_f32 * lin_r + 0.3575761 * lin_g + 0.1804375 * lin_b;
        let y = 0.2126729_f32 * lin_r + 0.7151522 * lin_g + 0.0721750 * lin_b;
        let z = 0.0193339_f32 * lin_r + 0.119_192 * lin_g + 0.9503041 * lin_b;

        let fx = lab_f_fast(x / XN);
        let fy = lab_f_fast(y / YN);
        let fz = lab_f_fast(z / ZN);

        out[i * 3] = 116.0 * fy - 16.0;
        out[i * 3 + 1] = 500.0 * (fx - fy);
        out[i * 3 + 2] = 200.0 * (fy - fz);
    }

    Tensor::from_aligned(vec![h, w, 3], out).map_err(Into::into)
}

/// Returns number of pixels processed by SIMD.
#[allow(unsafe_code)]
fn lab_simd_row(src: &[f32], dst: &mut [f32]) -> usize {
    let w = dst.len() / 3;
    if w < 4 {
        return 0;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: ISA guard (feature detection) above.
            return unsafe { lab_neon_row(src, dst) };
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            // SAFETY: ISA guard (feature detection) above.
            return unsafe { lab_avx_row(src, dst) };
        }
        if std::is_x86_feature_detected!("sse") {
            // SAFETY: ISA guard (feature detection) above.
            return unsafe { lab_sse_row(src, dst) };
        }
    }

    0
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn lab_neon_row(src: &[f32], dst: &mut [f32]) -> usize {
    use std::arch::aarch64::{
        vdupq_n_f32, vfmaq_f32, vgetq_lane_f32, vld1q_f32, vld3q_f32, vmulq_f32, vst1q_f32,
    };

    let lut = srgb_lut();
    let w = dst.len() / 3;
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();

    // XYZ matrix coefficients
    let m00 = vdupq_n_f32(0.4124564);
    let m01 = vdupq_n_f32(0.3575761);
    let m02 = vdupq_n_f32(0.1804375);
    let m10 = vdupq_n_f32(0.2126729);
    let m11 = vdupq_n_f32(0.7151522);
    let m12 = vdupq_n_f32(0.0721750);
    let m20 = vdupq_n_f32(0.0193339);
    let m21 = vdupq_n_f32(0.119_192);
    let m22 = vdupq_n_f32(0.9503041);

    // White point reciprocals
    let inv_xn = vdupq_n_f32(1.0 / 0.95047);
    let inv_yn = vdupq_n_f32(1.0);
    let inv_zn = vdupq_n_f32(1.0 / 1.08883);

    let v255 = vdupq_n_f32(255.0);

    let mut x = 0usize;
    while x + 4 <= w {
        let rgb = vld3q_f32(sp.add(x * 3));

        // Gamma decode via LUT: scale to [0,255], floor-index, interpolate
        let r_scaled = vmulq_f32(rgb.0, v255);
        let g_scaled = vmulq_f32(rgb.1, v255);
        let b_scaled = vmulq_f32(rgb.2, v255);

        // Linearize 4 pixels via LUT lookup (scalar, then reload as NEON)
        let mut lin_r_arr = [0.0f32; 4];
        let mut lin_g_arr = [0.0f32; 4];
        let mut lin_b_arr = [0.0f32; 4];
        for j in 0..4 {
            let rs = vgetq_lane_f32::<0>(match j {
                0 => r_scaled,
                1 => {
                    let t = r_scaled;
                    std::arch::aarch64::vextq_f32(t, t, 1)
                }
                2 => {
                    let t = r_scaled;
                    std::arch::aarch64::vextq_f32(t, t, 2)
                }
                _ => {
                    let t = r_scaled;
                    std::arch::aarch64::vextq_f32(t, t, 3)
                }
            });
            let gs = vgetq_lane_f32::<0>(match j {
                0 => g_scaled,
                1 => {
                    let t = g_scaled;
                    std::arch::aarch64::vextq_f32(t, t, 1)
                }
                2 => {
                    let t = g_scaled;
                    std::arch::aarch64::vextq_f32(t, t, 2)
                }
                _ => {
                    let t = g_scaled;
                    std::arch::aarch64::vextq_f32(t, t, 3)
                }
            });
            let bs = vgetq_lane_f32::<0>(match j {
                0 => b_scaled,
                1 => {
                    let t = b_scaled;
                    std::arch::aarch64::vextq_f32(t, t, 1)
                }
                2 => {
                    let t = b_scaled;
                    std::arch::aarch64::vextq_f32(t, t, 2)
                }
                _ => {
                    let t = b_scaled;
                    std::arch::aarch64::vextq_f32(t, t, 3)
                }
            });
            let ri = (rs as usize).min(254);
            let gi = (gs as usize).min(254);
            let bi = (bs as usize).min(254);
            let rf = rs - ri as f32;
            let gf = gs - gi as f32;
            let bf = bs - bi as f32;
            lin_r_arr[j] = lut[ri] * (1.0 - rf) + lut[ri + 1] * rf;
            lin_g_arr[j] = lut[gi] * (1.0 - gf) + lut[gi + 1] * gf;
            lin_b_arr[j] = lut[bi] * (1.0 - bf) + lut[bi + 1] * bf;
        }

        let lin_r = vld1q_f32(lin_r_arr.as_ptr());
        let lin_g = vld1q_f32(lin_g_arr.as_ptr());
        let lin_b = vld1q_f32(lin_b_arr.as_ptr());

        // RGB → XYZ matrix multiply (SIMD FMA, 4 pixels at a time)
        let vx = vfmaq_f32(vfmaq_f32(vmulq_f32(lin_r, m00), lin_g, m01), lin_b, m02);
        let vy = vfmaq_f32(vfmaq_f32(vmulq_f32(lin_r, m10), lin_g, m11), lin_b, m12);
        let vz = vfmaq_f32(vfmaq_f32(vmulq_f32(lin_r, m20), lin_g, m21), lin_b, m22);

        // Normalize by white point
        let nx = vmulq_f32(vx, inv_xn);
        let ny = vmulq_f32(vy, inv_yn);
        let nz = vmulq_f32(vz, inv_zn);

        // Store normalized XYZ, then do scalar lab_f + LAB conversion
        let mut nx_arr = [0.0f32; 4];
        let mut ny_arr = [0.0f32; 4];
        let mut nz_arr = [0.0f32; 4];
        vst1q_f32(nx_arr.as_mut_ptr(), nx);
        vst1q_f32(ny_arr.as_mut_ptr(), ny);
        vst1q_f32(nz_arr.as_mut_ptr(), nz);

        for j in 0..4 {
            let fx = lab_f_fast(nx_arr[j]);
            let fy = lab_f_fast(ny_arr[j]);
            let fz = lab_f_fast(nz_arr[j]);
            let idx = (x + j) * 3;
            *dp.add(idx) = 116.0 * fy - 16.0;
            *dp.add(idx + 1) = 500.0 * (fx - fy);
            *dp.add(idx + 2) = 200.0 * (fy - fz);
        }

        x += 4;
    }
    x
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn lab_avx_row(src: &[f32], dst: &mut [f32]) -> usize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let _lut = srgb_lut();
    let w = dst.len() / 3;
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();

    let m00 = _mm256_set1_ps(0.4124564);
    let m01 = _mm256_set1_ps(0.3575761);
    let m02 = _mm256_set1_ps(0.1804375);
    let m10 = _mm256_set1_ps(0.2126729);
    let m11 = _mm256_set1_ps(0.7151522);
    let m12 = _mm256_set1_ps(0.0721750);
    let m20 = _mm256_set1_ps(0.0193339);
    let m21 = _mm256_set1_ps(0.119_192);
    let m22 = _mm256_set1_ps(0.9503041);

    let inv_xn = 1.0f32 / 0.95047;
    let inv_yn = 1.0f32;
    let inv_zn = 1.0f32 / 1.08883;

    let mut x = 0usize;
    while x + 8 <= w {
        let p = sp.add(x * 3);

        // Gather R, G, B from interleaved RGB (8 pixels)
        let r_raw = _mm256_set_ps(
            *p.add(21),
            *p.add(18),
            *p.add(15),
            *p.add(12),
            *p.add(9),
            *p.add(6),
            *p.add(3),
            *p.add(0),
        );
        let g_raw = _mm256_set_ps(
            *p.add(22),
            *p.add(19),
            *p.add(16),
            *p.add(13),
            *p.add(10),
            *p.add(7),
            *p.add(4),
            *p.add(1),
        );
        let b_raw = _mm256_set_ps(
            *p.add(23),
            *p.add(20),
            *p.add(17),
            *p.add(14),
            *p.add(11),
            *p.add(8),
            *p.add(5),
            *p.add(2),
        );

        // Gamma decode via LUT (scalar per lane)
        let ra: [f32; 8] = std::mem::transmute(r_raw);
        let ga: [f32; 8] = std::mem::transmute(g_raw);
        let ba: [f32; 8] = std::mem::transmute(b_raw);
        let mut lin_r_arr = [0.0f32; 8];
        let mut lin_g_arr = [0.0f32; 8];
        let mut lin_b_arr = [0.0f32; 8];
        for j in 0..8 {
            lin_r_arr[j] = srgb_to_linear_fast(ra[j]);
            lin_g_arr[j] = srgb_to_linear_fast(ga[j]);
            lin_b_arr[j] = srgb_to_linear_fast(ba[j]);
        }
        let lin_r = _mm256_loadu_ps(lin_r_arr.as_ptr());
        let lin_g = _mm256_loadu_ps(lin_g_arr.as_ptr());
        let lin_b = _mm256_loadu_ps(lin_b_arr.as_ptr());

        // XYZ matrix multiply (AVX)
        let vx = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(lin_r, m00), _mm256_mul_ps(lin_g, m01)),
            _mm256_mul_ps(lin_b, m02),
        );
        let vy = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(lin_r, m10), _mm256_mul_ps(lin_g, m11)),
            _mm256_mul_ps(lin_b, m12),
        );
        let vz = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(lin_r, m20), _mm256_mul_ps(lin_g, m21)),
            _mm256_mul_ps(lin_b, m22),
        );

        let xa: [f32; 8] = std::mem::transmute(vx);
        let ya: [f32; 8] = std::mem::transmute(vy);
        let za: [f32; 8] = std::mem::transmute(vz);

        // Fast lab_f via LUT (scalar)
        let op = dp.add(x * 3);
        for j in 0..8 {
            let fx = lab_f_fast(xa[j] * inv_xn);
            let fy = lab_f_fast(ya[j] * inv_yn);
            let fz = lab_f_fast(za[j] * inv_zn);
            *op.add(j * 3) = 116.0 * fy - 16.0;
            *op.add(j * 3 + 1) = 500.0 * (fx - fy);
            *op.add(j * 3 + 2) = 200.0 * (fy - fz);
        }

        x += 8;
    }
    x
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn lab_sse_row(src: &[f32], dst: &mut [f32]) -> usize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let _lut = srgb_lut();
    let w = dst.len() / 3;
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();

    let m00 = _mm_set1_ps(0.4124564);
    let m01 = _mm_set1_ps(0.3575761);
    let m02 = _mm_set1_ps(0.1804375);
    let m10 = _mm_set1_ps(0.2126729);
    let m11 = _mm_set1_ps(0.7151522);
    let m12 = _mm_set1_ps(0.0721750);
    let m20 = _mm_set1_ps(0.0193339);
    let m21 = _mm_set1_ps(0.119_192);
    let m22 = _mm_set1_ps(0.9503041);

    let inv_xn = 1.0f32 / 0.95047;
    let inv_yn = 1.0f32;
    let inv_zn = 1.0f32 / 1.08883;

    let mut x = 0usize;
    while x + 4 <= w {
        let p = sp.add(x * 3);

        // Gather R, G, B
        let r_raw = _mm_set_ps(*p.add(9), *p.add(6), *p.add(3), *p.add(0));
        let g_raw = _mm_set_ps(*p.add(10), *p.add(7), *p.add(4), *p.add(1));
        let b_raw = _mm_set_ps(*p.add(11), *p.add(8), *p.add(5), *p.add(2));

        // Gamma decode via LUT (scalar per lane)
        let ra: [f32; 4] = std::mem::transmute(r_raw);
        let ga: [f32; 4] = std::mem::transmute(g_raw);
        let ba: [f32; 4] = std::mem::transmute(b_raw);
        let mut lin_r_arr = [0.0f32; 4];
        let mut lin_g_arr = [0.0f32; 4];
        let mut lin_b_arr = [0.0f32; 4];
        for j in 0..4 {
            lin_r_arr[j] = srgb_to_linear_fast(ra[j]);
            lin_g_arr[j] = srgb_to_linear_fast(ga[j]);
            lin_b_arr[j] = srgb_to_linear_fast(ba[j]);
        }
        let lin_r = _mm_loadu_ps(lin_r_arr.as_ptr());
        let lin_g = _mm_loadu_ps(lin_g_arr.as_ptr());
        let lin_b = _mm_loadu_ps(lin_b_arr.as_ptr());

        // XYZ matrix multiply (SSE)
        let vx = _mm_add_ps(
            _mm_add_ps(_mm_mul_ps(lin_r, m00), _mm_mul_ps(lin_g, m01)),
            _mm_mul_ps(lin_b, m02),
        );
        let vy = _mm_add_ps(
            _mm_add_ps(_mm_mul_ps(lin_r, m10), _mm_mul_ps(lin_g, m11)),
            _mm_mul_ps(lin_b, m12),
        );
        let vz = _mm_add_ps(
            _mm_add_ps(_mm_mul_ps(lin_r, m20), _mm_mul_ps(lin_g, m21)),
            _mm_mul_ps(lin_b, m22),
        );

        let xa: [f32; 4] = std::mem::transmute(vx);
        let ya: [f32; 4] = std::mem::transmute(vy);
        let za: [f32; 4] = std::mem::transmute(vz);

        // Fast lab_f via LUT (no cbrt)
        let op = dp.add(x * 3);
        for j in 0..4 {
            let fx = lab_f_fast(xa[j] * inv_xn);
            let fy = lab_f_fast(ya[j] * inv_yn);
            let fz = lab_f_fast(za[j] * inv_zn);
            *op.add(j * 3) = 116.0 * fy - 16.0;
            *op.add(j * 3 + 1) = 500.0 * (fx - fy);
            *op.add(j * 3 + 2) = 200.0 * (fy - fz);
        }

        x += 4;
    }
    x
}

/// Converts CIE LAB `[H, W, 3]` to RGB `[H, W, 3]`.
///
/// Input: L in `[0, 100]`, a and b roughly in `[-128, 127]`.
/// Output RGB values are clamped to `[0, 1]`.
pub fn lab_to_rgb(input: &Tensor) -> Result<Tensor, ImgProcError> {
    let (h, w, c) = hwc_shape(input)?;
    if c != 3 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 3,
            got: c,
        });
    }

    let data = input.data();
    let pixels = h * w;
    let mut out = vec![0.0f32; pixels * 3];

    const XN: f64 = 0.95047;
    const YN: f64 = 1.0;
    const ZN: f64 = 1.08883;

    for i in 0..pixels {
        let l = data[i * 3] as f64;
        let a = data[i * 3 + 1] as f64;
        let b_val = data[i * 3 + 2] as f64;

        // LAB → XYZ
        let fy = (l + 16.0) / 116.0;
        let fx = a / 500.0 + fy;
        let fz = fy - b_val / 200.0;

        let x = XN * lab_f_inv(fx);
        let y = YN * lab_f_inv(fy);
        let z = ZN * lab_f_inv(fz);

        // XYZ → linear RGB (inverse sRGB D65 matrix)
        let lin_r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
        let lin_g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z;
        let lin_b = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z;

        // Linear RGB → sRGB gamma encode, then clamp
        out[i * 3] = linear_to_srgb(lin_r).clamp(0.0, 1.0) as f32;
        out[i * 3 + 1] = linear_to_srgb(lin_g).clamp(0.0, 1.0) as f32;
        out[i * 3 + 2] = linear_to_srgb(lin_b).clamp(0.0, 1.0) as f32;
    }

    Tensor::from_vec(vec![h, w, 3], out).map_err(Into::into)
}

/// Converts RGB `[H, W, 3]` to YUV `[H, W, 3]` using BT.601 coefficients.
///
/// Input RGB values are expected in `[0, 1]`.
#[allow(unsafe_code, clippy::uninit_vec)]
pub fn rgb_to_yuv(input: &Tensor) -> Result<Tensor, ImgProcError> {
    let (h, w, c) = hwc_shape(input)?;
    if c != 3 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 3,
            got: c,
        });
    }

    let data = input.data();
    let pixels = h * w;
    let row_stride = w * 3;

    // For large images, parallelise across rows
    #[cfg(target_os = "macos")]
    if pixels > 4096 && !cfg!(miri) {
        let mut out = AlignedVec::<f32>::uninitialized(pixels * 3);
        let out_ptr = super::SendPtr(out.as_mut_ptr());
        use super::u8ops::gcd;
        gcd::parallel_for(h, |y| {
            let src_row = &data[y * row_stride..(y + 1) * row_stride];
            // SAFETY: pointer and length from validated image data; rows are non-overlapping.
            let dst_row = unsafe {
                std::slice::from_raw_parts_mut(out_ptr.ptr().add(y * row_stride), row_stride)
            };
            let done = if !cfg!(miri) {
                yuv_simd_row(src_row, dst_row)
            } else {
                0
            };
            for i in done..w {
                let r = src_row[i * 3];
                let g = src_row[i * 3 + 1];
                let b = src_row[i * 3 + 2];
                dst_row[i * 3] = 0.299 * r + 0.587 * g + 0.114 * b;
                dst_row[i * 3 + 1] = -0.14713 * r - 0.28886 * g + 0.436 * b;
                dst_row[i * 3 + 2] = 0.615 * r - 0.51499 * g - 0.10001 * b;
            }
        });
        return Tensor::from_aligned(vec![h, w, 3], out).map_err(Into::into);
    }

    #[cfg(not(target_os = "macos"))]
    if pixels > 4096 && !cfg!(miri) {
        let mut out = vec![0.0f32; pixels * 3];
        out.par_chunks_mut(row_stride)
            .enumerate()
            .for_each(|(y, dst_row)| {
                let src_row = &data[y * row_stride..(y + 1) * row_stride];
                let done = yuv_simd_row(src_row, dst_row);
                for i in done..w {
                    let r = src_row[i * 3];
                    let g = src_row[i * 3 + 1];
                    let b = src_row[i * 3 + 2];
                    dst_row[i * 3] = 0.299 * r + 0.587 * g + 0.114 * b;
                    dst_row[i * 3 + 1] = -0.14713 * r - 0.28886 * g + 0.436 * b;
                    dst_row[i * 3 + 2] = 0.615 * r - 0.51499 * g - 0.10001 * b;
                }
            });
        return Tensor::from_vec(vec![h, w, 3], out).map_err(Into::into);
    }

    let mut out = AlignedVec::<f32>::uninitialized(pixels * 3);
    let done = if !cfg!(miri) {
        yuv_simd_row(data, &mut out)
    } else {
        0
    };
    for i in done..pixels {
        let r = data[i * 3];
        let g = data[i * 3 + 1];
        let b = data[i * 3 + 2];
        out[i * 3] = 0.299 * r + 0.587 * g + 0.114 * b;
        out[i * 3 + 1] = -0.14713 * r - 0.28886 * g + 0.436 * b;
        out[i * 3 + 2] = 0.615 * r - 0.51499 * g - 0.10001 * b;
    }

    Tensor::from_aligned(vec![h, w, 3], out).map_err(Into::into)
}

/// Returns number of pixels processed by SIMD.
#[allow(unsafe_code)]
fn yuv_simd_row(src: &[f32], dst: &mut [f32]) -> usize {
    let w = dst.len() / 3;
    if w < 4 {
        return 0;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: ISA guard (feature detection) above.
            return unsafe { yuv_neon_row(src, dst) };
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            // SAFETY: ISA guard (feature detection) above.
            return unsafe { yuv_avx_row(src, dst) };
        }
        if std::is_x86_feature_detected!("sse") {
            // SAFETY: ISA guard (feature detection) above.
            return unsafe { yuv_sse_row(src, dst) };
        }
    }

    0
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn yuv_neon_row(src: &[f32], dst: &mut [f32]) -> usize {
    use std::arch::aarch64::{
        float32x4x3_t, vdupq_n_f32, vfmaq_f32, vld3q_f32, vmulq_f32, vst3q_f32,
    };

    let w = dst.len() / 3;
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();

    // Y coefficients
    let vy_r = vdupq_n_f32(0.299);
    let vy_g = vdupq_n_f32(0.587);
    let vy_b = vdupq_n_f32(0.114);
    // U coefficients
    let vu_r = vdupq_n_f32(-0.14713);
    let vu_g = vdupq_n_f32(-0.28886);
    let vu_b = vdupq_n_f32(0.436);
    // V coefficients
    let vv_r = vdupq_n_f32(0.615);
    let vv_g = vdupq_n_f32(-0.51499);
    let vv_b = vdupq_n_f32(-0.10001);

    let mut x = 0usize;
    while x + 4 <= w {
        let rgb = vld3q_f32(sp.add(x * 3));
        let y = vfmaq_f32(vfmaq_f32(vmulq_f32(rgb.0, vy_r), rgb.1, vy_g), rgb.2, vy_b);
        let u = vfmaq_f32(vfmaq_f32(vmulq_f32(rgb.0, vu_r), rgb.1, vu_g), rgb.2, vu_b);
        let v = vfmaq_f32(vfmaq_f32(vmulq_f32(rgb.0, vv_r), rgb.1, vv_g), rgb.2, vv_b);
        vst3q_f32(dp.add(x * 3), float32x4x3_t(y, u, v));
        x += 4;
    }
    x
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn yuv_avx_row(src: &[f32], dst: &mut [f32]) -> usize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let w = dst.len() / 3;
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();

    let vy_r = _mm256_set1_ps(0.299);
    let vy_g = _mm256_set1_ps(0.587);
    let vy_b = _mm256_set1_ps(0.114);
    let vu_r = _mm256_set1_ps(-0.14713);
    let vu_g = _mm256_set1_ps(-0.28886);
    let vu_b = _mm256_set1_ps(0.436);
    let vv_r = _mm256_set1_ps(0.615);
    let vv_g = _mm256_set1_ps(-0.51499);
    let vv_b = _mm256_set1_ps(-0.10001);

    let mut x = 0usize;
    while x + 8 <= w {
        let p = sp.add(x * 3);
        let r = _mm256_set_ps(
            *p.add(21),
            *p.add(18),
            *p.add(15),
            *p.add(12),
            *p.add(9),
            *p.add(6),
            *p.add(3),
            *p.add(0),
        );
        let g = _mm256_set_ps(
            *p.add(22),
            *p.add(19),
            *p.add(16),
            *p.add(13),
            *p.add(10),
            *p.add(7),
            *p.add(4),
            *p.add(1),
        );
        let b = _mm256_set_ps(
            *p.add(23),
            *p.add(20),
            *p.add(17),
            *p.add(14),
            *p.add(11),
            *p.add(8),
            *p.add(5),
            *p.add(2),
        );

        let y = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(r, vy_r), _mm256_mul_ps(g, vy_g)),
            _mm256_mul_ps(b, vy_b),
        );
        let u = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(r, vu_r), _mm256_mul_ps(g, vu_g)),
            _mm256_mul_ps(b, vu_b),
        );
        let v = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(r, vv_r), _mm256_mul_ps(g, vv_g)),
            _mm256_mul_ps(b, vv_b),
        );

        // Scatter back to interleaved YUV
        let ya: [f32; 8] = std::mem::transmute(y);
        let ua: [f32; 8] = std::mem::transmute(u);
        let va: [f32; 8] = std::mem::transmute(v);
        let op = dp.add(x * 3);
        for j in 0..8 {
            *op.add(j * 3) = ya[j];
            *op.add(j * 3 + 1) = ua[j];
            *op.add(j * 3 + 2) = va[j];
        }
        x += 8;
    }
    x
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn yuv_sse_row(src: &[f32], dst: &mut [f32]) -> usize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let w = dst.len() / 3;
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();

    let vy_r = _mm_set1_ps(0.299);
    let vy_g = _mm_set1_ps(0.587);
    let vy_b = _mm_set1_ps(0.114);
    let vu_r = _mm_set1_ps(-0.14713);
    let vu_g = _mm_set1_ps(-0.28886);
    let vu_b = _mm_set1_ps(0.436);
    let vv_r = _mm_set1_ps(0.615);
    let vv_g = _mm_set1_ps(-0.51499);
    let vv_b = _mm_set1_ps(-0.10001);

    let mut x = 0usize;
    while x + 4 <= w {
        let p = sp.add(x * 3);
        let r = _mm_set_ps(*p.add(9), *p.add(6), *p.add(3), *p.add(0));
        let g = _mm_set_ps(*p.add(10), *p.add(7), *p.add(4), *p.add(1));
        let b = _mm_set_ps(*p.add(11), *p.add(8), *p.add(5), *p.add(2));

        let y = _mm_add_ps(
            _mm_add_ps(_mm_mul_ps(r, vy_r), _mm_mul_ps(g, vy_g)),
            _mm_mul_ps(b, vy_b),
        );
        let u = _mm_add_ps(
            _mm_add_ps(_mm_mul_ps(r, vu_r), _mm_mul_ps(g, vu_g)),
            _mm_mul_ps(b, vu_b),
        );
        let v = _mm_add_ps(
            _mm_add_ps(_mm_mul_ps(r, vv_r), _mm_mul_ps(g, vv_g)),
            _mm_mul_ps(b, vv_b),
        );

        // Interleave Y, U, V back to [y0,u0,v0, y1,u1,v1, y2,u2,v2, y3,u3,v3]
        let op = dp.add(x * 3);
        // No SSE interleave intrinsic for 3-channel, store per-pixel
        let ya: [f32; 4] = std::mem::transmute(y);
        let ua: [f32; 4] = std::mem::transmute(u);
        let va: [f32; 4] = std::mem::transmute(v);
        for j in 0..4 {
            *op.add(j * 3) = ya[j];
            *op.add(j * 3 + 1) = ua[j];
            *op.add(j * 3 + 2) = va[j];
        }
        x += 4;
    }
    x
}

/// Converts YUV `[H, W, 3]` to RGB `[H, W, 3]` using BT.601 coefficients.
///
/// Output RGB values are not clamped; caller may clamp to `[0, 1]` if needed.
pub fn yuv_to_rgb(input: &Tensor) -> Result<Tensor, ImgProcError> {
    let (h, w, c) = hwc_shape(input)?;
    if c != 3 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 3,
            got: c,
        });
    }

    let data = input.data();
    let pixels = h * w;
    let mut out = vec![0.0f32; pixels * 3];

    let compute = |chunk: &mut [f32], i: usize| {
        let y = data[i * 3];
        let u = data[i * 3 + 1];
        let v = data[i * 3 + 2];
        chunk[0] = y + 1.13983 * v;
        chunk[1] = y - 0.39465 * u - 0.58060 * v;
        chunk[2] = y + 2.03211 * u;
    };

    if pixels > 4096 {
        out.par_chunks_mut(3)
            .enumerate()
            .for_each(|(i, chunk)| compute(chunk, i));
    } else {
        out.chunks_mut(3)
            .enumerate()
            .for_each(|(i, chunk)| compute(chunk, i));
    }

    Tensor::from_vec(vec![h, w, 3], out).map_err(Into::into)
}

// --- Helper functions for LAB conversion ---

/// sRGB gamma decode: nonlinear sRGB → linear.
fn srgb_to_linear(v: f64) -> f64 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// sRGB gamma encode: linear → nonlinear sRGB.
fn linear_to_srgb(v: f64) -> f64 {
    if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// CIE LAB f function (reference, used for LUT construction).
fn lab_f(t: f64) -> f64 {
    const DELTA: f64 = 6.0 / 29.0;
    if t > DELTA * DELTA * DELTA {
        t.cbrt()
    } else {
        t / (3.0 * DELTA * DELTA) + 4.0 / 29.0
    }
}

/// Pre-computed LUT for `lab_f()` over the normalized [0, ~2.0] range.
/// 4096 entries mapping `i / LAB_F_LUT_SCALE` → `lab_f(i / LAB_F_LUT_SCALE)`.
/// With 4096 entries at scale 2048, the nearest-index lookup has max error ~0.00024,
/// which is negligible for u8-quantized input images.
const LAB_F_LUT_SIZE: usize = 4096;
const LAB_F_LUT_SCALE: f64 = 2048.0; // max input ≈ 4095/2048 ≈ 2.0, covers all valid XYZ/Xn

fn lab_f_lut() -> &'static [f32; LAB_F_LUT_SIZE] {
    use std::sync::OnceLock;
    static LUT: OnceLock<Box<[f32; LAB_F_LUT_SIZE]>> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut table = Box::new([0.0f32; LAB_F_LUT_SIZE]);
        for i in 0..LAB_F_LUT_SIZE {
            table[i] = lab_f(i as f64 / LAB_F_LUT_SCALE) as f32;
        }
        table
    })
}

/// Fast `lab_f()` via LUT with linear interpolation.
/// With 4096 entries at scale 2048.0, interpolation adds just one FMA.
#[inline]
fn lab_f_fast(t: f32) -> f32 {
    let lut = lab_f_lut();
    let scaled = t * LAB_F_LUT_SCALE as f32;
    let lo = (scaled as usize).min(LAB_F_LUT_SIZE - 2);
    let frac = scaled - lo as f32;
    // SAFETY: lo clamped to 0..LAB_F_LUT_SIZE-2; lo+1 < LAB_F_LUT_SIZE.
    unsafe {
        let a = *lut.get_unchecked(lo);
        // a + frac * (b - a) = lerp with one FMA
        a + frac * (*lut.get_unchecked(lo + 1) - a)
    }
}

/// Inverse CIE LAB f function.
fn lab_f_inv(t: f64) -> f64 {
    const DELTA: f64 = 6.0 / 29.0;
    if t > DELTA {
        t * t * t
    } else {
        3.0 * DELTA * DELTA * (t - 4.0 / 29.0)
    }
}

/// Converts RGB `[H, W, 3]` to BGR `[H, W, 3]` (channel swap).
#[allow(unsafe_code, clippy::uninit_vec)]
pub fn rgb_to_bgr(input: &Tensor) -> Result<Tensor, ImgProcError> {
    let (h, w, c) = hwc_shape(input)?;
    if c != 3 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 3,
            got: c,
        });
    }

    let data = input.data();
    let pixels = h * w;
    let row_stride = w * 3;

    // For large images, parallelise across rows
    #[cfg(target_os = "macos")]
    if pixels > 4096 && !cfg!(miri) {
        let mut out = AlignedVec::<f32>::uninitialized(pixels * 3);
        let out_ptr = super::SendPtr(out.as_mut_ptr());
        use super::u8ops::gcd;
        gcd::parallel_for(h, |y| {
            let src_row = &data[y * row_stride..(y + 1) * row_stride];
            // SAFETY: pointer and length from validated image data; rows are non-overlapping.
            let dst_row = unsafe {
                std::slice::from_raw_parts_mut(out_ptr.ptr().add(y * row_stride), row_stride)
            };
            let done = if !cfg!(miri) {
                bgr_simd_row(src_row, dst_row)
            } else {
                0
            };
            for i in done..w {
                dst_row[i * 3] = src_row[i * 3 + 2];
                dst_row[i * 3 + 1] = src_row[i * 3 + 1];
                dst_row[i * 3 + 2] = src_row[i * 3];
            }
        });
        return Tensor::from_aligned(vec![h, w, 3], out).map_err(Into::into);
    }

    #[cfg(not(target_os = "macos"))]
    if pixels > 4096 && !cfg!(miri) {
        let mut out = vec![0.0f32; pixels * 3];
        out.par_chunks_mut(row_stride)
            .enumerate()
            .for_each(|(y, dst_row)| {
                let src_row = &data[y * row_stride..(y + 1) * row_stride];
                let done = bgr_simd_row(src_row, dst_row);
                for i in done..w {
                    dst_row[i * 3] = src_row[i * 3 + 2];
                    dst_row[i * 3 + 1] = src_row[i * 3 + 1];
                    dst_row[i * 3 + 2] = src_row[i * 3];
                }
            });
        return Tensor::from_vec(vec![h, w, 3], out).map_err(Into::into);
    }

    let mut out = AlignedVec::<f32>::uninitialized(pixels * 3);
    let done = if !cfg!(miri) {
        bgr_simd_row(data, &mut out)
    } else {
        0
    };
    for i in done..pixels {
        out[i * 3] = data[i * 3 + 2];
        out[i * 3 + 1] = data[i * 3 + 1];
        out[i * 3 + 2] = data[i * 3];
    }

    Tensor::from_aligned(vec![h, w, 3], out).map_err(Into::into)
}

/// Returns number of pixels processed by SIMD.
#[allow(unsafe_code)]
fn bgr_simd_row(src: &[f32], dst: &mut [f32]) -> usize {
    let w = dst.len() / 3;
    if w < 4 {
        return 0;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: ISA guard (feature detection) above.
            return unsafe { bgr_neon_row(src, dst) };
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            // SAFETY: ISA guard (feature detection) above.
            return unsafe { bgr_avx_row(src, dst) };
        }
        if std::is_x86_feature_detected!("sse") {
            // SAFETY: ISA guard (feature detection) above.
            return unsafe { bgr_sse_row(src, dst) };
        }
    }

    0
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn bgr_neon_row(src: &[f32], dst: &mut [f32]) -> usize {
    use std::arch::aarch64::{float32x4x3_t, vld3q_f32, vst3q_f32};

    let w = dst.len() / 3;
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();
    let mut x = 0usize;

    while x + 4 <= w {
        let rgb = vld3q_f32(sp.add(x * 3));
        // Swap R and B, keep G
        vst3q_f32(dp.add(x * 3), float32x4x3_t(rgb.2, rgb.1, rgb.0));
        x += 4;
    }
    x
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn bgr_avx_row(src: &[f32], dst: &mut [f32]) -> usize {
    #[cfg(target_arch = "x86")]
    #[allow(unused_imports)]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    #[allow(unused_imports)]
    use std::arch::x86_64::*;

    let w = dst.len() / 3;
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();
    let mut x = 0usize;

    // Process 8 pixels at a time: gather R, G, B then scatter as B, G, R
    while x + 8 <= w {
        let p = sp.add(x * 3);
        let op = dp.add(x * 3);
        for j in 0..8usize {
            *op.add(j * 3) = *p.add(j * 3 + 2);
            *op.add(j * 3 + 1) = *p.add(j * 3 + 1);
            *op.add(j * 3 + 2) = *p.add(j * 3);
        }
        x += 8;
    }
    x
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn bgr_sse_row(src: &[f32], dst: &mut [f32]) -> usize {
    let w = dst.len() / 3;
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();
    let mut x = 0usize;

    // No SSE deinterleave for 3-channel, do scalar gather/scatter
    while x + 4 <= w {
        let p = sp.add(x * 3);
        let op = dp.add(x * 3);
        for j in 0..4usize {
            *op.add(j * 3) = *p.add(j * 3 + 2);
            *op.add(j * 3 + 1) = *p.add(j * 3 + 1);
            *op.add(j * 3 + 2) = *p.add(j * 3);
        }
        x += 4;
    }
    x
}
