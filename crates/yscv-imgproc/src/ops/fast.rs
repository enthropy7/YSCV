//! # Safety contract
//!
//! Unsafe code categories:
//! 1. **SIMD intrinsics (NEON / SSE)** — ISA guard via runtime detection; used for cardinal
//!    early-rejection test (4 pixels at a time).
//! 2. **`get_unchecked`** — pixel indices are within the 3-pixel border margin enforced by
//!    `y_start=3, y_end=h-3, x_start=3, x_end=w-3`.

use yscv_tensor::Tensor;

use super::super::ImgProcError;
use super::super::shape::hwc_shape;

/// A detected keypoint with orientation and scale information.
#[derive(Debug, Clone)]
pub struct Keypoint {
    pub x: f32,
    pub y: f32,
    pub response: f32,
    pub angle: f32,
    pub octave: usize,
}

/// Bresenham circle of radius 3: 16 pixel offsets (dx, dy).
const CIRCLE: [(i32, i32); 16] = [
    (0, -3),
    (1, -3),
    (2, -2),
    (3, -1),
    (3, 0),
    (3, 1),
    (2, 2),
    (1, 3),
    (0, 3),
    (-1, 3),
    (-2, 2),
    (-3, 1),
    (-3, 0),
    (-3, -1),
    (-2, -2),
    (-1, -3),
];

/// Precompute circle pixel offsets as flat `isize` offsets from center.
#[inline]
fn circle_offsets(w: usize) -> [isize; 16] {
    let ws = w as isize;
    let mut offsets = [0isize; 16];
    for (i, &(dx, dy)) in CIRCLE.iter().enumerate() {
        offsets[i] = dy as isize * ws + dx as isize;
    }
    offsets
}

/// Check if a 16-bit bitmask has 9 or more contiguous set bits (circular).
/// Returns the length of the longest contiguous run, or 0 if < 9.
#[inline]
fn contiguous_run_from_mask(mask: u32) -> usize {
    if mask == 0 {
        return 0;
    }
    // Double the bits to handle wrap-around: bits 0..15 repeated at 16..31
    let doubled = mask | (mask << 16);
    let mut best = 0u32;
    let mut run = 0u32;
    // Only need to scan 32 bits
    for i in 0..32 {
        if (doubled >> i) & 1 != 0 {
            run += 1;
            if run > best {
                best = run;
            }
        } else {
            run = 0;
        }
    }
    best.min(16) as usize
}

/// FAST-9 corner detection on a single-channel `[H, W, 1]` image.
///
/// Examines 16 pixels on a circle of radius 3 around each pixel.
/// A corner exists if 9 contiguous pixels are all brighter or all darker
/// than the center by at least `threshold`.
///
/// If `non_max` is true, non-maximum suppression is applied in a 3x3 neighbourhood.
///
/// Uses SIMD to batch the cardinal early-rejection test (4 pixels at a time),
/// eliminating ~90% of non-corner pixels with minimal work.
#[allow(unsafe_code)]
pub fn fast9_detect(
    image: &Tensor,
    threshold: f32,
    non_max: bool,
) -> Result<Vec<Keypoint>, ImgProcError> {
    let (h, w, c) = hwc_shape(image)?;
    if c != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: c,
        });
    }
    Ok(fast9_detect_raw(image.data(), h, w, threshold, non_max))
}

/// FAST-9 corner detection on raw f32 data — no Tensor allocation needed.
pub fn fast9_detect_raw(
    data: &[f32],
    h: usize,
    w: usize,
    threshold: f32,
    non_max: bool,
) -> Vec<Keypoint> {
    let offsets = circle_offsets(w);

    // Cardinal directions for early rejection: N(0), E(4), S(8), W(12)
    let card = [offsets[0], offsets[4], offsets[8], offsets[12]];

    let y_start = 3;
    let y_end = h.saturating_sub(3);
    let x_start = 3;
    let x_end = w.saturating_sub(3);

    // Parallel FAST: each row processed independently, results merged.
    let n_rows = y_end.saturating_sub(y_start);
    let row_corners: Vec<Vec<Keypoint>> = {
        use std::sync::Mutex;
        let results: Vec<Mutex<Vec<Keypoint>>> =
            (0..n_rows).map(|_| Mutex::new(Vec::new())).collect();

        use super::u8ops::gcd;
        gcd::parallel_for(n_rows, |row_idx| {
            let y = y_start + row_idx;
            let mut row_kps = Vec::new();

            let row_base = y * w;
            let mut x = x_start;

            #[cfg(target_arch = "aarch64")]
            if std::arch::is_aarch64_feature_detected!("neon") {
                while x + 4 <= x_end {
                    // SAFETY: ISA guard (feature detection) above; indices bounded by border.
                    let pass_mask =
                        unsafe { fast9_cardinal_check_neon(data, row_base + x, &card, threshold) };
                    if pass_mask == 0 {
                        x += 4;
                        continue;
                    }
                    for i in 0..4 {
                        if (pass_mask >> i) & 1 != 0 {
                            let cx = x + i;
                            let idx = row_base + cx;
                            // SAFETY: bounds checked by border range [border, w-border).
                            let max_run =
                                unsafe { fast9_full_check(data, idx, &offsets, threshold) };
                            if max_run >= 9 {
                                row_kps.push(Keypoint {
                                    x: cx as f32,
                                    y: y as f32,
                                    response: max_run as f32,
                                    angle: 0.0,
                                    octave: 0,
                                });
                            }
                        }
                    }
                    x += 4;
                }
            }

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if std::is_x86_feature_detected!("sse") {
                while x + 4 <= x_end {
                    // SAFETY: ISA guard (feature detection) above; indices bounded by border.
                    let pass_mask =
                        unsafe { fast9_cardinal_check_sse(data, row_base + x, &card, threshold) };
                    if pass_mask == 0 {
                        x += 4;
                        continue;
                    }
                    for i in 0..4 {
                        if (pass_mask >> i) & 1 != 0 {
                            let cx = x + i;
                            let idx = row_base + cx;
                            // SAFETY: bounds checked by border range [border, w-border).
                            let max_run =
                                unsafe { fast9_full_check(data, idx, &offsets, threshold) };
                            if max_run >= 9 {
                                row_kps.push(Keypoint {
                                    x: cx as f32,
                                    y: y as f32,
                                    response: max_run as f32,
                                    angle: 0.0,
                                    octave: 0,
                                });
                            }
                        }
                    }
                    x += 4;
                }
            }

            while x < x_end {
                let idx = row_base + x;
                // SAFETY: bounds checked by border range [border, w-border).
                let center = unsafe { *data.get_unchecked(idx) };
                let bright_thresh = center + threshold;
                let dark_thresh = center - threshold;
                let mut bright_count = 0u32;
                let mut dark_count = 0u32;
                for &co in &card {
                    // SAFETY: bounds checked by border range; cardinal offsets within 3-pixel radius.
                    let v = unsafe { *data.get_unchecked((idx as isize + co) as usize) };
                    bright_count += (v > bright_thresh) as u32;
                    dark_count += (v < dark_thresh) as u32;
                }
                if bright_count < 3 && dark_count < 3 {
                    x += 1;
                    continue;
                }
                // SAFETY: bounds checked by border range [border, w-border).
                let max_run = unsafe { fast9_full_check(data, idx, &offsets, threshold) };
                if max_run >= 9 {
                    row_kps.push(Keypoint {
                        x: x as f32,
                        y: y as f32,
                        response: max_run as f32,
                        angle: 0.0,
                        octave: 0,
                    });
                }
                x += 1;
            }

            *results[row_idx].lock().unwrap_or_else(|e| e.into_inner()) = row_kps;
        });

        results
            .into_iter()
            .map(|m| m.into_inner().unwrap_or_else(|e| e.into_inner()))
            .collect()
    };

    let mut corners: Vec<Keypoint> = row_corners.into_iter().flatten().collect();

    // Dead code: old sequential loop replaced by parallel version above.
    #[allow(unreachable_code)]
    if false {
        let y = y_start;
        for _y in y_start..y_end {
            let row_base = y * w;
            let mut x = x_start;

            // SIMD batch: check 4 consecutive center pixels at a time
            // This vectorizes the cardinal early-rejection test
            #[cfg(target_arch = "aarch64")]
            if std::arch::is_aarch64_feature_detected!("neon") {
                while x + 4 <= x_end {
                    // SAFETY: ISA guard (feature detection) above; indices bounded by border.
                    let pass_mask =
                        unsafe { fast9_cardinal_check_neon(data, row_base + x, &card, threshold) };
                    // If no pixels passed cardinal check, skip all 4
                    if pass_mask == 0 {
                        x += 4;
                        continue;
                    }
                    // Process pixels that passed cardinal check individually
                    for i in 0..4 {
                        if (pass_mask >> i) & 1 != 0 {
                            let cx = x + i;
                            let idx = row_base + cx;
                            // SAFETY: bounds checked by border range [border, w-border).
                            let max_run =
                                unsafe { fast9_full_check(data, idx, &offsets, threshold) };
                            if max_run >= 9 {
                                corners.push(Keypoint {
                                    x: cx as f32,
                                    y: y as f32,
                                    response: max_run as f32,
                                    angle: 0.0,
                                    octave: 0,
                                });
                            }
                        }
                    }
                    x += 4;
                }
            }

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if std::is_x86_feature_detected!("sse") {
                while x + 4 <= x_end {
                    // SAFETY: ISA guard (feature detection) above; indices bounded by border.
                    let pass_mask =
                        unsafe { fast9_cardinal_check_sse(data, row_base + x, &card, threshold) };
                    if pass_mask == 0 {
                        x += 4;
                        continue;
                    }
                    for i in 0..4 {
                        if (pass_mask >> i) & 1 != 0 {
                            let cx = x + i;
                            let idx = row_base + cx;
                            // SAFETY: bounds checked by border range [border, w-border).
                            let max_run =
                                unsafe { fast9_full_check(data, idx, &offsets, threshold) };
                            if max_run >= 9 {
                                corners.push(Keypoint {
                                    x: cx as f32,
                                    y: y as f32,
                                    response: max_run as f32,
                                    angle: 0.0,
                                    octave: 0,
                                });
                            }
                        }
                    }
                    x += 4;
                }
            }

            // Scalar tail for remaining pixels
            while x < x_end {
                let idx = row_base + x;
                // SAFETY: bounds checked by border range [border, w-border).
                let center = unsafe { *data.get_unchecked(idx) };
                let bright_thresh = center + threshold;
                let dark_thresh = center - threshold;

                let mut bright_count = 0u32;
                let mut dark_count = 0u32;
                for &co in &card {
                    // SAFETY: bounds checked by border range; cardinal offsets within 3-pixel radius.
                    let v = unsafe { *data.get_unchecked((idx as isize + co) as usize) };
                    bright_count += (v > bright_thresh) as u32;
                    dark_count += (v < dark_thresh) as u32;
                }
                if bright_count < 3 && dark_count < 3 {
                    x += 1;
                    continue;
                }

                // SAFETY: bounds checked by border range [border, w-border).
                let max_run = unsafe { fast9_full_check(data, idx, &offsets, threshold) };
                if max_run >= 9 {
                    corners.push(Keypoint {
                        x: x as f32,
                        y: y as f32,
                        response: max_run as f32,
                        angle: 0.0,
                        octave: 0,
                    });
                }
                x += 1;
            }
        }
    } // if false

    if non_max {
        let mut response_map = vec![0.0f32; h * w];
        for kp in &corners {
            let ix = kp.x as usize;
            let iy = kp.y as usize;
            response_map[iy * w + ix] = kp.response;
        }
        corners.retain(|kp| {
            let ix = kp.x as usize;
            let iy = kp.y as usize;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dy == 0 && dx == 0 {
                        continue;
                    }
                    let ny = (iy as i32 + dy) as usize;
                    let nx = (ix as i32 + dx) as usize;
                    if ny < h && nx < w && response_map[ny * w + nx] > kp.response {
                        return false;
                    }
                }
            }
            true
        });
    }

    corners
}

/// SIMD cardinal check for 4 consecutive center pixels (NEON).
/// Returns a 4-bit mask: bit i is set if pixel i passes cardinal test.
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn fast9_cardinal_check_neon(
    data: &[f32],
    base_idx: usize,
    card: &[isize; 4],
    threshold: f32,
) -> u32 {
    use std::arch::aarch64::*;

    let ptr = data.as_ptr();
    let thresh = vdupq_n_f32(threshold);
    let neg_thresh = vdupq_n_f32(-threshold);
    let three = vdupq_n_u32(3);

    // Load 4 consecutive center pixels
    let centers = vld1q_f32(ptr.add(base_idx));
    let bright_thresh = vaddq_f32(centers, thresh);
    let dark_thresh = vaddq_f32(centers, neg_thresh);

    // For each cardinal direction, load the 4 corresponding circle pixels
    // (consecutive since centers are consecutive)
    let mut bright_cnt = vdupq_n_u32(0);
    let mut dark_cnt = vdupq_n_u32(0);

    for &co in card.iter() {
        let circle_px = vld1q_f32(ptr.add((base_idx as isize + co) as usize));
        // brighter: circle > bright_thresh
        let b = vcgtq_f32(circle_px, bright_thresh);
        bright_cnt = vsubq_u32(bright_cnt, vreinterpretq_u32_f32(vreinterpretq_f32_u32(b)));
        // darker: circle < dark_thresh
        let d = vcltq_f32(circle_px, dark_thresh);
        dark_cnt = vsubq_u32(dark_cnt, vreinterpretq_u32_f32(vreinterpretq_f32_u32(d)));
    }

    // Check if bright_cnt >= 3 OR dark_cnt >= 3 for each pixel
    let bright_pass = vcgeq_u32(bright_cnt, three);
    let dark_pass = vcgeq_u32(dark_cnt, three);
    let pass = vorrq_u32(bright_pass, dark_pass);

    // Extract to 4-bit mask
    let mut mask = 0u32;
    if vgetq_lane_u32(pass, 0) != 0 {
        mask |= 1;
    }
    if vgetq_lane_u32(pass, 1) != 0 {
        mask |= 2;
    }
    if vgetq_lane_u32(pass, 2) != 0 {
        mask |= 4;
    }
    if vgetq_lane_u32(pass, 3) != 0 {
        mask |= 8;
    }
    mask
}

/// SIMD cardinal check for 4 consecutive center pixels (SSE).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn fast9_cardinal_check_sse(
    data: &[f32],
    base_idx: usize,
    card: &[isize; 4],
    threshold: f32,
) -> u32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let ptr = data.as_ptr();
    let thresh = _mm_set1_ps(threshold);
    let neg_thresh = _mm_set1_ps(-threshold);
    let _zero = _mm_setzero_ps();

    let centers = _mm_loadu_ps(ptr.add(base_idx));
    let bright_thresh = _mm_add_ps(centers, thresh);
    let dark_thresh = _mm_add_ps(centers, neg_thresh);

    // Count cardinals that pass each threshold
    let mut bright_cnt = _mm_setzero_ps();
    let mut dark_cnt = _mm_setzero_ps();
    let one_bits = _mm_set1_ps(1.0);

    for &co in card.iter() {
        let circle_px = _mm_loadu_ps(ptr.add((base_idx as isize + co) as usize));
        // cmpgt returns 0xFFFFFFFF for true, so AND with 1.0 gives 1.0 for true
        let b = _mm_and_ps(_mm_cmpgt_ps(circle_px, bright_thresh), one_bits);
        bright_cnt = _mm_add_ps(bright_cnt, b);
        let d = _mm_and_ps(_mm_cmplt_ps(circle_px, dark_thresh), one_bits);
        dark_cnt = _mm_add_ps(dark_cnt, d);
    }

    let three = _mm_set1_ps(3.0);
    let bright_pass = _mm_cmpge_ps(bright_cnt, three);
    let dark_pass = _mm_cmpge_ps(dark_cnt, three);
    let pass = _mm_or_ps(bright_pass, dark_pass);

    _mm_movemask_ps(pass) as u32
}

/// Full FAST-9 check: build bitmasks and find max contiguous run.
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn fast9_full_check(
    data: &[f32],
    idx: usize,
    offsets: &[isize; 16],
    threshold: f32,
) -> usize {
    let center = *data.get_unchecked(idx);
    let bright_thresh = center + threshold;
    let dark_thresh = center - threshold;

    let mut bright_mask = 0u32;
    let mut dark_mask = 0u32;
    for i in 0..16 {
        let v = *data.get_unchecked((idx as isize + offsets[i]) as usize);
        if v > bright_thresh {
            bright_mask |= 1 << i;
        }
        if v < dark_thresh {
            dark_mask |= 1 << i;
        }
    }

    let bright_run = contiguous_run_from_mask(bright_mask);
    let dark_run = contiguous_run_from_mask(dark_mask);
    bright_run.max(dark_run)
}

/// Compute the intensity centroid orientation for a keypoint.
/// Uses moments in a circular patch of given radius around (kx, ky).
pub(crate) fn intensity_centroid_angle(
    data: &[f32],
    w: usize,
    h: usize,
    kx: usize,
    ky: usize,
    radius: i32,
) -> f32 {
    let mut m01: f32 = 0.0;
    let mut m10: f32 = 0.0;
    for dy in -radius..=radius {
        let max_dx = ((radius * radius - dy * dy) as f32).sqrt() as i32;
        for dx in -max_dx..=max_dx {
            let py = ky as i32 + dy;
            let px = kx as i32 + dx;
            if py >= 0 && py < h as i32 && px >= 0 && px < w as i32 {
                let v = data[py as usize * w + px as usize];
                m10 += dx as f32 * v;
                m01 += dy as f32 * v;
            }
        }
    }
    m01.atan2(m10)
}

/// Find the maximum number of contiguous `true` values in a circular 16-element array.
#[allow(dead_code)]
fn max_consecutive(flags: &[bool; 16]) -> usize {
    let mut best = 0usize;
    let mut count = 0usize;
    // Scan twice around the circle to handle wrap-around
    for i in 0..32 {
        if flags[i % 16] {
            count += 1;
            if count > best {
                best = count;
            }
        } else {
            count = 0;
        }
    }
    best.min(16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast9_detects_corner() {
        // Create 30x30 image with an L-shaped corner: bright pixels on two edges
        let mut data = vec![0.0f32; 30 * 30];
        // Horizontal bright bar y=15, x=10..20
        for x in 10..20 {
            data[15 * 30 + x] = 1.0;
        }
        // Vertical bright bar x=10, y=10..20
        for y in 10..20 {
            data[y * 30 + 10] = 1.0;
        }
        let img = Tensor::from_vec(vec![30, 30, 1], data).unwrap();
        let kps = fast9_detect(&img, 0.3, false).unwrap();
        assert!(!kps.is_empty(), "should detect corners near the L-shape");
    }

    #[test]
    fn test_fast9_no_corners_on_flat() {
        let img = Tensor::from_vec(vec![20, 20, 1], vec![0.5; 400]).unwrap();
        let kps = fast9_detect(&img, 0.1, false).unwrap();
        assert!(kps.is_empty(), "flat image should produce no corners");
    }

    #[test]
    fn test_fast9_threshold() {
        // Bright dot on dark background — high threshold should detect fewer
        let mut data = vec![0.0f32; 30 * 30];
        data[15 * 30 + 15] = 1.0;
        for &(dx, dy) in &CIRCLE {
            let px = (15 + dx) as usize;
            let py = (15 + dy) as usize;
            data[py * 30 + px] = 0.6;
        }
        let img = Tensor::from_vec(vec![30, 30, 1], data.clone()).unwrap();
        let low = fast9_detect(&img, 0.1, false).unwrap();
        let high = fast9_detect(&img, 0.8, false).unwrap();
        assert!(
            high.len() <= low.len(),
            "higher threshold should produce fewer or equal corners: low={} high={}",
            low.len(),
            high.len()
        );
    }
}
