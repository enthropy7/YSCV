//! Canny edge detection (u8 pipeline).
#![allow(unsafe_code)]

use super::super::ImgProcError;
use super::u8ops::{ImageU8, RAYON_THRESHOLD};
use rayon::prelude::*;

// ============================================================================
// Canny edge detection (u8 pipeline)
// ============================================================================

/// Canny edge detection on single-channel u8 image.
/// Full pipeline: Gaussian blur → Sobel L1 magnitude + direction → NMS + threshold → iterative hysteresis.
/// `low_thresh` and `high_thresh` are in `[0, 255]`.
///
/// Optimizations vs naive pipeline:
/// - Inlined separable Gaussian blur (avoids ImageU8 allocation + function call overhead)
/// - Fused Sobel magnitude + 2-bit direction encoding (NEON SIMD, 8 pixels/iteration)
/// - NMS uses precomputed directions (no Sobel recomputation — saves ~50% of Sobel work)
/// - L1 norm (`|dx|+|dy|`) instead of L2 — matches OpenCV default, enables pure SIMD magnitude
/// - Integer angle quantization: `5*|dy| vs 2*|dx|` instead of atan2 or float ratios
/// - Fused mag/dir into 3-row ring buffers + inline NMS (cache-hot, avoids h*w mag/dir allocs)
/// - Pre-sized BFS queue (from NMS strong count) + flat offset array (no nested dy/dx loops)
/// - SIMD final output copy (NEON 16-byte threshold instead of scalar weak-pixel zeroing)
pub fn canny_u8(input: &ImageU8, low_thresh: u8, high_thresh: u8) -> Result<ImageU8, ImgProcError> {
    if input.channels() != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: input.channels(),
        });
    }
    let (h, w) = (input.height(), input.width());
    if h < 3 || w < 3 {
        return Ok(ImageU8::zeros(h, w, 1));
    }

    let raw = input.data();
    let mut nms_out = vec![0u8; h * w];
    let low = low_thresh as u16;
    let high = high_thresh as u16;

    let use_rayon = h * w >= RAYON_THRESHOLD && !cfg!(miri) && h > 30;

    // ========================================================================
    // 3×3 Sobel → NMS pipeline (matches OpenCV default: no Gaussian pre-blur).
    // ========================================================================
    let strong_positions: Vec<u32> = if use_rayon {
        // ==================================================================
        // Strip-based parallel canny: 3×3 Sobel→NMS with mag/dir rings.
        // No gauss_h ring — reads raw u8 directly. ~5.7KB per strip.
        // ==================================================================

        let n_cpus = rayon::current_num_threads().max(1);
        let nms_rows = h - 2;
        let strip_h = (nms_rows / n_cpus).max(30);
        let mut strips: Vec<(usize, usize)> = Vec::new();
        {
            let mut s = 1usize;
            while s <= h - 2 {
                let e = (s + strip_h).min(h - 1);
                strips.push((s, e));
                s = e;
            }
        }

        let nms_ptr = super::SendPtr(nms_out.as_mut_ptr());
        let strong_per_strip: Vec<Vec<u32>> = strips
            .par_iter()
            .map(|&(nms_start, nms_end)| {
                let mut mag_ring = vec![0u16; 3 * w];
                let mut dir_ring = vec![0u8; 3 * w];
                let mut local_strong: Vec<u32> = Vec::new();

                let sobel_first = nms_start.saturating_sub(1).max(1);
                let sobel_last = nms_end.min(h - 2);

                for y in sobel_first..=sobel_last {
                    let above = &raw[(y - 1) * w..y * w];
                    let center = &raw[y * w..(y + 1) * w];
                    let below = &raw[(y + 1) * w..(y + 2) * w];
                    let ms = (y % 3) * w;
                    canny_sobel3x3_row(
                        above,
                        center,
                        below,
                        &mut mag_ring[ms..ms + w],
                        &mut dir_ring[ms..ms + w],
                        w,
                    );

                    if y > sobel_first {
                        let nms_y = y - 1;
                        if nms_y >= nms_start && nms_y < nms_end {
                            let a_s = ((y - 2) % 3) * w;
                            let c_s = ((y - 1) % 3) * w;
                            let b_s = (y % 3) * w;
                            let nms_row = unsafe {
                                std::slice::from_raw_parts_mut(
                                    nms_ptr.ptr().add(nms_y * w),
                                    w,
                                )
                            };
                            canny_nms_dir_row(
                                &mag_ring[a_s..a_s + w],
                                &mag_ring[c_s..c_s + w],
                                &mag_ring[b_s..b_s + w],
                                &dir_ring[c_s..c_s + w],
                                nms_row,
                                w,
                                low,
                                high,
                            );
                            collect_strong_positions(nms_row, nms_y, w, &mut local_strong);
                        }
                    }
                }

                if sobel_last >= nms_start && sobel_last < nms_end {
                    let nms_y = sobel_last;
                    let c_s = (nms_y % 3) * w;
                    let a_s = ((nms_y.wrapping_sub(1)) % 3) * w;
                    let b_s = ((nms_y + 1) % 3) * w;
                    mag_ring[b_s..b_s + w].fill(0);
                    let nms_row = unsafe {
                        std::slice::from_raw_parts_mut(nms_ptr.ptr().add(nms_y * w), w)
                    };
                    canny_nms_dir_row(
                        &mag_ring[a_s..a_s + w],
                        &mag_ring[c_s..c_s + w],
                        &mag_ring[b_s..b_s + w],
                        &dir_ring[c_s..c_s + w],
                        nms_row,
                        w,
                        low,
                        high,
                    );
                    collect_strong_positions(nms_row, nms_y, w, &mut local_strong);
                }

                local_strong
            })
            .collect();

        strong_per_strip.into_iter().flatten().collect()
    } else {
        // ==================================================================
        // Sequential path: 3×3 Sobel with mag/dir ring buffers.
        // ==================================================================
        let mut mag_ring = vec![0u16; 3 * w];
        let mut dir_ring = vec![0u8; 3 * w];
        let mut strong_positions: Vec<u32> = Vec::new();

        for y in 1..h - 1 {
            let above = &raw[(y - 1) * w..y * w];
            let center = &raw[y * w..(y + 1) * w];
            let below = &raw[(y + 1) * w..(y + 2) * w];
            let ms = (y % 3) * w;
            canny_sobel3x3_row(
                above,
                center,
                below,
                &mut mag_ring[ms..ms + w],
                &mut dir_ring[ms..ms + w],
                w,
            );

            if y >= 2 {
                let nms_y = y - 1;
                let a_s = ((y - 2) % 3) * w;
                let c_s = ((y - 1) % 3) * w;
                let b_s = (y % 3) * w;
                let nms_row_start = nms_y * w;
                let nms_row = &mut nms_out[nms_row_start..nms_row_start + w];
                canny_nms_dir_row(
                    &mag_ring[a_s..a_s + w],
                    &mag_ring[c_s..c_s + w],
                    &mag_ring[b_s..b_s + w],
                    &dir_ring[c_s..c_s + w],
                    nms_row,
                    w,
                    low,
                    high,
                );
                collect_strong_positions(nms_row, nms_y, w, &mut strong_positions);
            }
        }

        // NMS for last row
        {
            let nms_y = h - 2;
            let c_s = (nms_y % 3) * w;
            let a_s = ((nms_y.wrapping_sub(1)) % 3) * w;
            let b_s = ((nms_y + 1) % 3) * w;
            mag_ring[b_s..b_s + w].fill(0);
            let nms_row_start = nms_y * w;
            let nms_row = &mut nms_out[nms_row_start..nms_row_start + w];
            canny_nms_dir_row(
                &mag_ring[a_s..a_s + w],
                &mag_ring[c_s..c_s + w],
                &mag_ring[b_s..b_s + w],
                &dir_ring[c_s..c_s + w],
                nms_row,
                w,
                low,
                high,
            );
            collect_strong_positions(nms_row, nms_y, w, &mut strong_positions);
        }

        strong_positions
    };

    // ========================================================================
    // DFS hysteresis — stack-based (Vec::pop is faster than VecDeque::pop_front).
    // ========================================================================
    let mut stack = strong_positions;

    // Flat offset array for 8-connected neighbors
    let offsets: [isize; 8] = [
        -(w as isize) - 1,
        -(w as isize),
        -(w as isize) + 1,
        -1,
        1,
        w as isize - 1,
        w as isize,
        w as isize + 1,
    ];

    while let Some(idx) = stack.pop() {
        let idx = idx as usize;
        for &off in &offsets {
            let ni = idx as isize + off;
            if ni < w as isize || ni >= ((h - 1) * w) as isize {
                continue;
            }
            let ni = ni as usize;
            unsafe {
                if *nms_out.get_unchecked(ni) == 1 {
                    *nms_out.get_unchecked_mut(ni) = 255;
                    stack.push(ni as u32);
                }
            }
        }
    }

    // ========================================================================
    // Opt 1C: In-place SIMD threshold on nms_out — convert weak (1) to 0,
    // keep strong (255) as 255. Eliminates separate out_data allocation.
    // ========================================================================
    let total = h * w;

    #[cfg(target_arch = "aarch64")]
    if !cfg!(miri) {
        use std::arch::aarch64::*;
        let mut i = 0;
        unsafe {
            let all_ff = vdupq_n_u8(255);
            while i + 16 <= total {
                let v = vld1q_u8(nms_out.as_ptr().add(i));
                let mask = vceqq_u8(v, all_ff);
                let result = vandq_u8(v, mask);
                vst1q_u8(nms_out.as_mut_ptr().add(i), result);
                i += 16;
            }
        }
        // scalar tail
        for j in i..total {
            if nms_out[j] != 255 {
                nms_out[j] = 0;
            }
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for j in 0..total {
            if nms_out[j] != 255 {
                nms_out[j] = 0;
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    if cfg!(miri) {
        for j in 0..total {
            if nms_out[j] != 255 {
                nms_out[j] = 0;
            }
        }
    }

    ImageU8::new(nms_out, h, w, 1).ok_or(ImgProcError::InvalidChannelCount {
        expected: 1,
        got: input.channels(),
    })
}

/// NEON-accelerated scan: collect positions where `nms_row[x] == 255`.
/// Skips 16 bytes at a time when no strong edges present.
#[inline]
fn collect_strong_positions(nms_row: &[u8], nms_y: usize, w: usize, out: &mut Vec<u32>) {
    let base = nms_y * w;
    #[allow(unused_mut)]
    let mut x = 1usize;

    #[cfg(target_arch = "aarch64")]
    if !cfg!(miri) && w >= 18 {
        use std::arch::aarch64::*;
        let target = unsafe { vdupq_n_u8(255) };
        let p = nms_row.as_ptr();
        while x + 16 <= w - 1 {
            let v = unsafe { vld1q_u8(p.add(x)) };
            let mask = unsafe { vceqq_u8(v, target) };
            if unsafe { vmaxvq_u8(mask) } != 0 {
                // At least one strong edge in this group
                for i in 0..16 {
                    if nms_row[x + i] == 255 {
                        out.push((base + x + i) as u32);
                    }
                }
            }
            x += 16;
        }
    }

    for xx in x..w - 1 {
        if nms_row[xx] == 255 {
            out.push((base + xx) as u32);
        }
    }
}

/// 3×3 Sobel directly on raw u8 rows. Matches OpenCV's default Canny (no Gaussian pre-blur).
#[inline]
fn canny_sobel3x3_row(
    above: &[u8],
    center: &[u8],
    below: &[u8],
    mag_row: &mut [u16],
    dir_row: &mut [u8],
    w: usize,
) {
    let mut done = 1usize;
    if !cfg!(miri) {
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                done =
                    unsafe { canny_sobel3x3_row_neon(above, center, below, mag_row, dir_row, w) };
            }
        }
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("ssse3") {
                done = unsafe { canny_sobel3x3_row_sse(above, center, below, mag_row, dir_row, w) };
            }
        }
    }
    for x in done..w.saturating_sub(1) {
        let al = above[x - 1] as i32;
        let ac = above[x] as i32;
        let ar = above[x + 1] as i32;
        let cl = center[x - 1] as i32;
        let cr = center[x + 1] as i32;
        let bl = below[x - 1] as i32;
        let bc = below[x] as i32;
        let br = below[x + 1] as i32;
        let gx = (ar - al) + 2 * (cr - cl) + (br - bl);
        let gy = (bl + 2 * bc + br) - (al + 2 * ac + ar);
        let ax = gx.unsigned_abs();
        let ay = gy.unsigned_abs();
        mag_row[x] = (ax + ay) as u16;
        dir_row[x] = if ay * 5 < ax * 2 {
            0
        } else if ay * 2 > ax * 5 {
            2
        } else if (gx > 0 && gy > 0) || (gx < 0 && gy < 0) {
            1
        } else {
            3
        };
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn canny_sobel3x3_row_neon(
    above: &[u8],
    center: &[u8],
    below: &[u8],
    mag_row: &mut [u16],
    dir_row: &mut [u8],
    w: usize,
) -> usize {
    use std::arch::aarch64::*;
    if w < 18 {
        return 1;
    }
    let ap = above.as_ptr();
    let cp = center.as_ptr();
    let bp = below.as_ptr();
    let mp = mag_row.as_mut_ptr();
    let dp = dir_row.as_mut_ptr();
    let mut x = 1usize;

    while x + 16 <= w - 1 {
        let al_v = vld1q_u8(ap.add(x - 1));
        let ac_v = vld1q_u8(ap.add(x));
        let ar_v = vld1q_u8(ap.add(x + 1));
        let cl_v = vld1q_u8(cp.add(x - 1));
        let cr_v = vld1q_u8(cp.add(x + 1));
        let bl_v = vld1q_u8(bp.add(x - 1));
        let bc_v = vld1q_u8(bp.add(x));
        let br_v = vld1q_u8(bp.add(x + 1));

        // Low 8 pixels
        let (al, ac, ar) = (
            vmovl_u8(vget_low_u8(al_v)),
            vmovl_u8(vget_low_u8(ac_v)),
            vmovl_u8(vget_low_u8(ar_v)),
        );
        let (cl, cr) = (vmovl_u8(vget_low_u8(cl_v)), vmovl_u8(vget_low_u8(cr_v)));
        let (bl, bc, br) = (
            vmovl_u8(vget_low_u8(bl_v)),
            vmovl_u8(vget_low_u8(bc_v)),
            vmovl_u8(vget_low_u8(br_v)),
        );
        let gx = vaddq_s16(
            vaddq_s16(
                vsubq_s16(vreinterpretq_s16_u16(ar), vreinterpretq_s16_u16(al)),
                vsubq_s16(vreinterpretq_s16_u16(br), vreinterpretq_s16_u16(bl)),
            ),
            vshlq_n_s16::<1>(vsubq_s16(
                vreinterpretq_s16_u16(cr),
                vreinterpretq_s16_u16(cl),
            )),
        );
        let gy = vsubq_s16(
            vaddq_s16(
                vaddq_s16(vreinterpretq_s16_u16(bl), vreinterpretq_s16_u16(br)),
                vshlq_n_s16::<1>(vreinterpretq_s16_u16(bc)),
            ),
            vaddq_s16(
                vaddq_s16(vreinterpretq_s16_u16(al), vreinterpretq_s16_u16(ar)),
                vshlq_n_s16::<1>(vreinterpretq_s16_u16(ac)),
            ),
        );
        let agx = vreinterpretq_u16_s16(vabsq_s16(gx));
        let agy = vreinterpretq_u16_s16(vabsq_s16(gy));
        vst1q_u16(mp.add(x), vaddq_u16(agx, agy));
        let h = vcgtq_u16(vshlq_n_u16::<1>(agx), vaddq_u16(vshlq_n_u16::<2>(agy), agy));
        let v = vcgtq_u16(vshlq_n_u16::<1>(agy), vaddq_u16(vshlq_n_u16::<2>(agx), agx));
        let s = vcgeq_s16(veorq_s16(gx, gy), vdupq_n_s16(0));
        vst1_u8(
            dp.add(x),
            vbsl_u8(
                vmovn_u16(h),
                vdup_n_u8(0),
                vbsl_u8(
                    vmovn_u16(v),
                    vdup_n_u8(2),
                    vbsl_u8(vmovn_u16(s), vdup_n_u8(1), vdup_n_u8(3)),
                ),
            ),
        );

        // High 8 pixels
        let (al, ac, ar) = (
            vmovl_u8(vget_high_u8(al_v)),
            vmovl_u8(vget_high_u8(ac_v)),
            vmovl_u8(vget_high_u8(ar_v)),
        );
        let (cl, cr) = (vmovl_u8(vget_high_u8(cl_v)), vmovl_u8(vget_high_u8(cr_v)));
        let (bl, bc, br) = (
            vmovl_u8(vget_high_u8(bl_v)),
            vmovl_u8(vget_high_u8(bc_v)),
            vmovl_u8(vget_high_u8(br_v)),
        );
        let gx = vaddq_s16(
            vaddq_s16(
                vsubq_s16(vreinterpretq_s16_u16(ar), vreinterpretq_s16_u16(al)),
                vsubq_s16(vreinterpretq_s16_u16(br), vreinterpretq_s16_u16(bl)),
            ),
            vshlq_n_s16::<1>(vsubq_s16(
                vreinterpretq_s16_u16(cr),
                vreinterpretq_s16_u16(cl),
            )),
        );
        let gy = vsubq_s16(
            vaddq_s16(
                vaddq_s16(vreinterpretq_s16_u16(bl), vreinterpretq_s16_u16(br)),
                vshlq_n_s16::<1>(vreinterpretq_s16_u16(bc)),
            ),
            vaddq_s16(
                vaddq_s16(vreinterpretq_s16_u16(al), vreinterpretq_s16_u16(ar)),
                vshlq_n_s16::<1>(vreinterpretq_s16_u16(ac)),
            ),
        );
        let agx = vreinterpretq_u16_s16(vabsq_s16(gx));
        let agy = vreinterpretq_u16_s16(vabsq_s16(gy));
        vst1q_u16(mp.add(x + 8), vaddq_u16(agx, agy));
        let h = vcgtq_u16(vshlq_n_u16::<1>(agx), vaddq_u16(vshlq_n_u16::<2>(agy), agy));
        let v = vcgtq_u16(vshlq_n_u16::<1>(agy), vaddq_u16(vshlq_n_u16::<2>(agx), agx));
        let s = vcgeq_s16(veorq_s16(gx, gy), vdupq_n_s16(0));
        vst1_u8(
            dp.add(x + 8),
            vbsl_u8(
                vmovn_u16(h),
                vdup_n_u8(0),
                vbsl_u8(
                    vmovn_u16(v),
                    vdup_n_u8(2),
                    vbsl_u8(vmovn_u16(s), vdup_n_u8(1), vdup_n_u8(3)),
                ),
            ),
        );
        x += 16;
    }
    while x + 8 <= w - 1 {
        let (al, ac, ar) = (
            vmovl_u8(vld1_u8(ap.add(x - 1))),
            vmovl_u8(vld1_u8(ap.add(x))),
            vmovl_u8(vld1_u8(ap.add(x + 1))),
        );
        let (cl, cr) = (
            vmovl_u8(vld1_u8(cp.add(x - 1))),
            vmovl_u8(vld1_u8(cp.add(x + 1))),
        );
        let (bl, bc, br) = (
            vmovl_u8(vld1_u8(bp.add(x - 1))),
            vmovl_u8(vld1_u8(bp.add(x))),
            vmovl_u8(vld1_u8(bp.add(x + 1))),
        );
        let gx = vaddq_s16(
            vaddq_s16(
                vsubq_s16(vreinterpretq_s16_u16(ar), vreinterpretq_s16_u16(al)),
                vsubq_s16(vreinterpretq_s16_u16(br), vreinterpretq_s16_u16(bl)),
            ),
            vshlq_n_s16::<1>(vsubq_s16(
                vreinterpretq_s16_u16(cr),
                vreinterpretq_s16_u16(cl),
            )),
        );
        let gy = vsubq_s16(
            vaddq_s16(
                vaddq_s16(vreinterpretq_s16_u16(bl), vreinterpretq_s16_u16(br)),
                vshlq_n_s16::<1>(vreinterpretq_s16_u16(bc)),
            ),
            vaddq_s16(
                vaddq_s16(vreinterpretq_s16_u16(al), vreinterpretq_s16_u16(ar)),
                vshlq_n_s16::<1>(vreinterpretq_s16_u16(ac)),
            ),
        );
        let agx = vreinterpretq_u16_s16(vabsq_s16(gx));
        let agy = vreinterpretq_u16_s16(vabsq_s16(gy));
        vst1q_u16(mp.add(x), vaddq_u16(agx, agy));
        let h = vcgtq_u16(vshlq_n_u16::<1>(agx), vaddq_u16(vshlq_n_u16::<2>(agy), agy));
        let v = vcgtq_u16(vshlq_n_u16::<1>(agy), vaddq_u16(vshlq_n_u16::<2>(agx), agx));
        let s = vcgeq_s16(veorq_s16(gx, gy), vdupq_n_s16(0));
        vst1_u8(
            dp.add(x),
            vbsl_u8(
                vmovn_u16(h),
                vdup_n_u8(0),
                vbsl_u8(
                    vmovn_u16(v),
                    vdup_n_u8(2),
                    vbsl_u8(vmovn_u16(s), vdup_n_u8(1), vdup_n_u8(3)),
                ),
            ),
        );
        x += 8;
    }
    x
}

/// SSE canny sobel row — computes magnitude + direction for 16 pixels/iter.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn canny_sobel3x3_row_sse(
    above: &[u8],
    center: &[u8],
    below: &[u8],
    mag_row: &mut [u16],
    dir_row: &mut [u8],
    w: usize,
) -> usize {
    use std::arch::x86_64::*;
    if w < 18 {
        return 1;
    }
    let ap = above.as_ptr();
    let cp = center.as_ptr();
    let bp = below.as_ptr();
    let mp = mag_row.as_mut_ptr();
    let dp = dir_row.as_mut_ptr();
    let zero = _mm_setzero_si128();
    let mut x = 1usize;

    while x + 8 <= w - 1 {
        let al = _mm_loadl_epi64(ap.add(x - 1) as *const __m128i);
        let ac = _mm_loadl_epi64(ap.add(x) as *const __m128i);
        let ar = _mm_loadl_epi64(ap.add(x + 1) as *const __m128i);
        let cl = _mm_loadl_epi64(cp.add(x - 1) as *const __m128i);
        let cr = _mm_loadl_epi64(cp.add(x + 1) as *const __m128i);
        let bl = _mm_loadl_epi64(bp.add(x - 1) as *const __m128i);
        let bc = _mm_loadl_epi64(bp.add(x) as *const __m128i);
        let br = _mm_loadl_epi64(bp.add(x + 1) as *const __m128i);

        let (al, ac, ar) = (
            _mm_unpacklo_epi8(al, zero),
            _mm_unpacklo_epi8(ac, zero),
            _mm_unpacklo_epi8(ar, zero),
        );
        let (cl, cr) = (_mm_unpacklo_epi8(cl, zero), _mm_unpacklo_epi8(cr, zero));
        let (bl, bc, br) = (
            _mm_unpacklo_epi8(bl, zero),
            _mm_unpacklo_epi8(bc, zero),
            _mm_unpacklo_epi8(br, zero),
        );

        // gx = (ar - al) + 2*(cr - cl) + (br - bl)
        let gx = _mm_add_epi16(
            _mm_add_epi16(_mm_sub_epi16(ar, al), _mm_sub_epi16(br, bl)),
            _mm_slli_epi16(_mm_sub_epi16(cr, cl), 1),
        );
        // gy = (bl + 2*bc + br) - (al + 2*ac + ar)
        let gy = _mm_sub_epi16(
            _mm_add_epi16(_mm_add_epi16(bl, br), _mm_slli_epi16(bc, 1)),
            _mm_add_epi16(_mm_add_epi16(al, ar), _mm_slli_epi16(ac, 1)),
        );

        let agx = _mm_abs_epi16(gx);
        let agy = _mm_abs_epi16(gy);
        _mm_storeu_si128(mp.add(x) as *mut __m128i, _mm_add_epi16(agx, agy));

        // Direction: h = 2*agx > 5*agy, v = 2*agy > 5*agx
        let agx2 = _mm_slli_epi16(agx, 1);
        let agy2 = _mm_slli_epi16(agy, 1);
        let agx5 = _mm_add_epi16(_mm_slli_epi16(agx, 2), agx);
        let agy5 = _mm_add_epi16(_mm_slli_epi16(agy, 2), agy);
        let h = _mm_cmpgt_epi16(agx2, agy5); // horizontal
        let v = _mm_cmpgt_epi16(agy2, agx5); // vertical
        let same_sign = _mm_cmpgt_epi16(_mm_xor_si128(gx, gy), _mm_set1_epi16(-1)); // gx^gy >= 0

        // Pack direction: h→0, v→2, same_sign→1, else→3
        let _d0 = _mm_set1_epi16(3); // default=3
        let d1 = _mm_or_si128(
            _mm_and_si128(same_sign, _mm_set1_epi16(1)),
            _mm_andnot_si128(same_sign, _mm_set1_epi16(3)),
        );
        let d2 = _mm_or_si128(_mm_and_si128(v, _mm_set1_epi16(2)), _mm_andnot_si128(v, d1));
        let d_final = _mm_or_si128(
            _mm_and_si128(h, _mm_setzero_si128()),
            _mm_andnot_si128(h, d2),
        );
        // Pack i16→u8 (only low 8 bytes matter)
        let packed = _mm_packus_epi16(d_final, zero);
        _mm_storel_epi64(dp.add(x) as *mut __m128i, packed);
        x += 8;
    }
    x
}

/// NMS + threshold classification using precomputed direction codes.
/// Direction encoding: 0=horizontal, 1=diagonal45(\), 2=vertical, 3=diagonal135(/).
/// Output: 0 = suppressed, 1 = weak edge candidate, 255 = strong edge.
/// Returns the count of strong edges (value == 255) found in this row.
/// Opt 1B: dispatches to NEON SIMD when available, with scalar tail.
#[inline]
fn canny_nms_dir_row(
    mag_above: &[u16],
    mag_cur: &[u16],
    mag_below: &[u16],
    dir_row: &[u8],
    nms_row: &mut [u8],
    w: usize,
    low: u16,
    high: u16,
) -> usize {
    let mut done = 1usize;
    let mut strong_count = 0usize;

    if !cfg!(miri) {
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                let (d, sc) = unsafe {
                    canny_nms_dir_row_neon(
                        mag_above, mag_cur, mag_below, dir_row, nms_row, w, low, high,
                    )
                };
                done = d;
                strong_count = sc;
            }
        }
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse2") {
                let (d, sc) = unsafe {
                    canny_nms_dir_row_sse(
                        mag_above, mag_cur, mag_below, dir_row, nms_row, w, low, high,
                    )
                };
                done = d;
                strong_count = sc;
            }
        }
    }

    // Scalar tail
    for x in done..w - 1 {
        let m = mag_cur[x];
        if m < low {
            continue;
        }

        let (n1, n2) = match dir_row[x] {
            0 => (mag_cur[x - 1], mag_cur[x + 1]),     // horizontal
            1 => (mag_above[x - 1], mag_below[x + 1]), // diagonal 45 (\)
            2 => (mag_above[x], mag_below[x]),         // vertical
            _ => (mag_above[x + 1], mag_below[x - 1]), // diagonal 135 (/)
        };

        if m >= n1 && m >= n2 {
            if m >= high {
                nms_row[x] = 255;
                strong_count += 1;
            } else {
                nms_row[x] = 1;
            }
        }
    }

    strong_count
}

/// SSE2: NMS + threshold for 8 pixels at a time.
/// Direction-based neighbor selection via compare + blend.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn canny_nms_dir_row_sse(
    mag_above: &[u16],
    mag_cur: &[u16],
    mag_below: &[u16],
    dir_row: &[u8],
    nms_row: &mut [u8],
    w: usize,
    low: u16,
    high: u16,
) -> (usize, usize) {
    use std::arch::x86_64::*;

    if w < 10 {
        return (1, 0);
    }

    let ma = mag_above.as_ptr();
    let mc = mag_cur.as_ptr();
    let mb = mag_below.as_ptr();
    let dp = dir_row.as_ptr();
    let np = nms_row.as_mut_ptr();

    let v_low = _mm_set1_epi16(low as i16);
    let v_high = _mm_set1_epi16(high as i16);
    let v0 = _mm_setzero_si128();
    let v1_16 = _mm_set1_epi16(1);
    let v2_16 = _mm_set1_epi16(2);
    let v255_16 = _mm_set1_epi16(255);
    let zero = _mm_setzero_si128();

    let mut x = 1usize;
    let mut strong_count = 0usize;

    while x + 8 <= w - 1 {
        let m = _mm_loadu_si128(mc.add(x) as *const __m128i);

        let cur_left = _mm_loadu_si128(mc.add(x - 1) as *const __m128i);
        let cur_right = _mm_loadu_si128(mc.add(x + 1) as *const __m128i);
        let above_left = _mm_loadu_si128(ma.add(x - 1) as *const __m128i);
        let above_center = _mm_loadu_si128(ma.add(x) as *const __m128i);
        let above_right = _mm_loadu_si128(ma.add(x + 1) as *const __m128i);
        let below_left = _mm_loadu_si128(mb.add(x - 1) as *const __m128i);
        let below_center = _mm_loadu_si128(mb.add(x) as *const __m128i);
        let below_right = _mm_loadu_si128(mb.add(x + 1) as *const __m128i);

        // Direction: widen u8 → u16
        let d = _mm_unpacklo_epi8(_mm_loadl_epi64(dp.add(x) as *const __m128i), zero);

        let is_0 = _mm_cmpeq_epi16(d, v0);
        let is_1 = _mm_cmpeq_epi16(d, v1_16);
        let is_2 = _mm_cmpeq_epi16(d, v2_16);

        // n1: dir0→cur_left, dir1→above_left, dir2→above_center, else→above_right
        let n1 = _mm_or_si128(
            _mm_and_si128(is_0, cur_left),
            _mm_or_si128(
                _mm_and_si128(is_1, above_left),
                _mm_or_si128(
                    _mm_and_si128(is_2, above_center),
                    _mm_andnot_si128(_mm_or_si128(_mm_or_si128(is_0, is_1), is_2), above_right),
                ),
            ),
        );

        // n2: dir0→cur_right, dir1→below_right, dir2→below_center, else→below_left
        let n2 = _mm_or_si128(
            _mm_and_si128(is_0, cur_right),
            _mm_or_si128(
                _mm_and_si128(is_1, below_right),
                _mm_or_si128(
                    _mm_and_si128(is_2, below_center),
                    _mm_andnot_si128(_mm_or_si128(_mm_or_si128(is_0, is_1), is_2), below_left),
                ),
            ),
        );

        // NMS: keep if m >= n1 AND m >= n2 AND m >= low
        // SSE2 has no unsigned compare, so use: a >= b ⟺ !(a < b) ⟺ !(b > a)
        // But for u16 compare we can use subtraction trick or saturating sub:
        // a >= b ⟺ _mm_subs_epu16(b, a) == 0
        let ge_n1 = _mm_cmpeq_epi16(_mm_subs_epu16(n1, m), zero);
        let ge_n2 = _mm_cmpeq_epi16(_mm_subs_epu16(n2, m), zero);
        let ge_low = _mm_cmpeq_epi16(_mm_subs_epu16(v_low, m), zero);
        let keep = _mm_and_si128(_mm_and_si128(ge_n1, ge_n2), ge_low);

        // Strong: keep AND m >= high
        let ge_high = _mm_cmpeq_epi16(_mm_subs_epu16(v_high, m), zero);
        let is_strong = _mm_and_si128(keep, ge_high);

        // Result: strong→255, weak(keep & !strong)→1, else→0
        let weak_only = _mm_andnot_si128(is_strong, keep);
        let result = _mm_or_si128(
            _mm_and_si128(is_strong, v255_16),
            _mm_and_si128(weak_only, v1_16),
        );

        // Pack u16→u8 and store
        let result_u8 = _mm_packus_epi16(result, zero);
        _mm_storel_epi64(np.add(x) as *mut __m128i, result_u8);

        // Count strong edges
        let strong_u8 = _mm_packus_epi16(_mm_srli_epi16(is_strong, 15), zero);
        // Horizontal sum of 8 bytes
        let sum = _mm_sad_epu8(strong_u8, zero);
        strong_count += _mm_extract_epi16(sum, 0) as usize;

        x += 8;
    }

    (x, strong_count)
}

/// NEON SIMD: NMS + threshold for 8 pixels at a time.
/// Uses direction-based neighbor selection via vbslq_u16.
/// Returns (done_x, strong_count).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn canny_nms_dir_row_neon(
    mag_above: &[u16],
    mag_cur: &[u16],
    mag_below: &[u16],
    dir_row: &[u8],
    nms_row: &mut [u8],
    w: usize,
    low: u16,
    high: u16,
) -> (usize, usize) {
    use std::arch::aarch64::*;

    if w < 10 {
        return (1, 0);
    }

    let ma = mag_above.as_ptr();
    let mc = mag_cur.as_ptr();
    let mb = mag_below.as_ptr();
    let dp = dir_row.as_ptr();
    let np = nms_row.as_mut_ptr();

    let v_low = vdupq_n_u16(low);
    let v_high = vdupq_n_u16(high);
    let v0 = vdupq_n_u16(0);
    let v1 = vdupq_n_u16(1);
    let v2 = vdupq_n_u16(2);
    let v255_u16 = vdupq_n_u16(255);
    let v1_u16 = vdupq_n_u16(1);

    let mut x = 1usize;
    let mut strong_count = 0usize;

    while x + 8 <= w - 1 {
        // Load magnitude center
        let m = vld1q_u16(mc.add(x));

        // Load shifted magnitudes for all three rows
        let cur_left = vld1q_u16(mc.add(x - 1));
        let cur_right = vld1q_u16(mc.add(x + 1));
        let above_left = vld1q_u16(ma.add(x - 1));
        let above_center = vld1q_u16(ma.add(x));
        let above_right = vld1q_u16(ma.add(x + 1));
        let below_left = vld1q_u16(mb.add(x - 1));
        let below_center = vld1q_u16(mb.add(x));
        let below_right = vld1q_u16(mb.add(x + 1));

        // Load direction, widen to u16
        let d = vmovl_u8(vld1_u8(dp.add(x)));

        // Direction masks
        let is_0 = vceqq_u16(d, v0);
        let is_1 = vceqq_u16(d, v1);
        let is_2 = vceqq_u16(d, v2);
        // is_3 = everything else (handled by default in vbsl chain)

        // Select n1 based on direction
        // dir=0: cur_left, dir=1: above_left, dir=2: above_center, dir=3: above_right
        let n1 = vbslq_u16(
            is_0,
            cur_left,
            vbslq_u16(is_1, above_left, vbslq_u16(is_2, above_center, above_right)),
        );

        // Select n2 based on direction
        // dir=0: cur_right, dir=1: below_right, dir=2: below_center, dir=3: below_left
        let n2 = vbslq_u16(
            is_0,
            cur_right,
            vbslq_u16(is_1, below_right, vbslq_u16(is_2, below_center, below_left)),
        );

        // NMS: keep if m >= n1 AND m >= n2
        let ge_n1 = vcgeq_u16(m, n1);
        let ge_n2 = vcgeq_u16(m, n2);
        let keep = vandq_u16(ge_n1, ge_n2);

        // Also must be >= low to keep at all
        let ge_low = vcgeq_u16(m, v_low);
        let keep = vandq_u16(keep, ge_low);

        // Threshold: strong if m >= high, weak otherwise (but kept)
        let is_strong = vandq_u16(keep, vcgeq_u16(m, v_high));

        // Output: 255 for strong, 1 for weak-only (keep & !strong), 0 otherwise
        let result = vbslq_u16(is_strong, v255_u16, vbslq_u16(keep, v1_u16, vdupq_n_u16(0)));

        // Narrow to u8 and store
        let result_u8 = vmovn_u16(result);
        vst1_u8(np.add(x), result_u8);

        // Count strong edges: narrow is_strong to u8 (0xFF or 0x00),
        // shift to get 1 or 0, horizontal sum via vaddv_u8
        let strong_u8 = vmovn_u16(is_strong);
        let ones = vshr_n_u8::<7>(strong_u8);
        strong_count += vaddv_u8(ones) as usize;

        x += 8;
    }

    (x, strong_count)
}
