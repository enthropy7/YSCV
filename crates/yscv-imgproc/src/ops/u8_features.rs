//! u8 feature operations: FAST-9, distance transform, warp perspective, bilateral filter.
//!
//! # Safety contract
//!
//! Unsafe code in this module falls into these categories:
//!
//! 1. **`SendConstPtr` / `SendPtr` slice reconstruction for rayon** — pointer derived from
//!    a slice that outlives the parallel scope; each thread accesses non-overlapping rows.
//!
//! 2. **SIMD intrinsics (NEON / SSE2 / AVX2)** — ISA availability guaranteed by runtime
//!    `is_aarch64_feature_detected!` / `is_x86_feature_detected!` guards or `#[target_feature]`.
//!
//! 3. **Pointer arithmetic in distance transform** — loop bounds ensure offsets stay within
//!    the `h * w` allocation.
//!
//! 4. **`get_unchecked` in bilateral filter inner loops** — indices are bounded by the
//!    kernel diameter and image dimensions validated at function entry.
#![allow(unsafe_code)]

#[cfg(target_os = "macos")]
use super::u8ops::gcd;
use super::u8ops::{ImageU8, RAYON_THRESHOLD};
use rayon::prelude::*;

// ============================================================================
// FAST-9 corner detection (u8, single-channel)
// ============================================================================

/// Bresenham circle offsets for radius 3, given image width `w`.
/// Returns 16 offsets in row-major order starting from N, going clockwise.
#[inline]
fn fast9_circle_offsets(w: isize) -> [isize; 16] {
    [
        -3 * w,     // 0:  N
        -3 * w + 1, // 1:  NNE
        -2 * w + 2, // 2:  NE
        -w + 3,     // 3:  ENE
        3,          // 4:  E
        w + 3,      // 5:  ESE
        2 * w + 2,  // 6:  SE
        3 * w + 1,  // 7:  SSE
        3 * w,      // 8:  S
        3 * w - 1,  // 9:  SSW
        2 * w - 2,  // 10: SW
        w - 3,      // 11: WSW
        -3,         // 12: W
        -w - 3,     // 13: WNW
        -2 * w - 2, // 14: NW
        -3 * w - 1, // 15: NNW
    ]
}

/// Check if there are 9 contiguous pixels in a 16-element circular mask
/// that are all set. `bits` is a 16-bit bitmask (one bit per circle position).
#[inline]
fn has_9_contiguous(bits: u32) -> bool {
    // Duplicate the ring to handle wrap-around: 32 bits = bits | (bits << 16)
    let ring = bits | (bits << 16);
    // AND shifted versions to find runs of contiguous 1-bits.
    let mut m = ring;
    m &= m >> 1; // runs of 2+
    m &= m >> 2; // runs of 4+
    m &= m >> 4; // runs of 8+
    m &= ring >> 8; // runs of 9+
    m != 0
}

/// Compute FAST corner response as the sum of absolute differences
/// between the center and the circle pixels that exceed the threshold.
#[inline]
fn fast9_response(src: &[u8], center_idx: usize, offsets: &[isize; 16], threshold: u8) -> u16 {
    let c = src[center_idx] as i16;
    let t = threshold as i16;
    let mut score: u16 = 0;
    for &off in offsets.iter() {
        let p = src[(center_idx as isize + off) as usize] as i16;
        let diff = (p - c).abs();
        if diff > t {
            score += diff as u16;
        }
    }
    score
}

/// FAST-9 corner detection on u8 single-channel image.
/// Returns keypoints as (x, y) coordinates.
pub fn fast9_detect_u8(image: &ImageU8, threshold: u8, nms: bool) -> Vec<(u16, u16)> {
    if image.channels() != 1 {
        return vec![];
    }
    let (h, w) = (image.height(), image.width());
    if h < 7 || w < 7 {
        return vec![];
    }
    let src = image.data();
    let wi = w as isize;
    let offsets = fast9_circle_offsets(wi);

    // Collect candidate corners with their responses
    let border = 3usize;
    let use_rayon = (h - 2 * border) * (w - 2 * border) >= RAYON_THRESHOLD && !cfg!(miri);

    let candidates: Vec<(u16, u16, u16)> = if use_rayon {
        let src_ptr = super::SendConstPtr(src.as_ptr());
        (border..h - border)
            .into_par_iter()
            .flat_map(|y| {
                // SAFETY: (category 1) src_ptr from slice that outlives par_iter; h*w is the slice length.
                let src = unsafe { std::slice::from_raw_parts(src_ptr.ptr(), h * w) };
                let mut row_corners = Vec::new();
                fast9_detect_row(src, y, w, border, &offsets, threshold, &mut row_corners);
                row_corners
            })
            .collect()
    } else {
        let mut corners = Vec::new();
        for y in border..h - border {
            fast9_detect_row(src, y, w, border, &offsets, threshold, &mut corners);
        }
        corners
    };

    if !nms || candidates.is_empty() {
        return candidates.iter().map(|&(x, y, _)| (x, y)).collect();
    }

    // Non-maximum suppression: suppress corners that have a neighbor with higher response
    // Build a response map for quick lookup
    let mut response_map = vec![0u16; h * w];
    for &(x, y, resp) in &candidates {
        response_map[y as usize * w + x as usize] = resp;
    }

    let mut result = Vec::with_capacity(candidates.len());
    for &(x, y, resp) in &candidates {
        let xi = x as usize;
        let yi = y as usize;
        let mut is_max = true;
        'outer: for dy in 0..3usize {
            let ny = yi + dy;
            if ny < 1 {
                continue;
            }
            let ny = ny - 1;
            if ny >= h {
                continue;
            }
            for dx in 0..3usize {
                let nx = xi + dx;
                if nx < 1 {
                    continue;
                }
                let nx = nx - 1;
                if nx >= w {
                    continue;
                }
                if nx == xi && ny == yi {
                    continue;
                }
                if response_map[ny * w + nx] > resp {
                    is_max = false;
                    break 'outer;
                }
            }
        }
        if is_max {
            result.push((x, y));
        }
    }
    result
}

/// Process one row for FAST-9 detection.
fn fast9_detect_row(
    src: &[u8],
    y: usize,
    w: usize,
    border: usize,
    offsets: &[isize; 16],
    threshold: u8,
    out: &mut Vec<(u16, u16, u16)>,
) {
    let t = threshold;
    let row_start = y * w;

    // NEON fast path: check 16 consecutive center pixels at once for early rejection
    #[cfg(target_arch = "aarch64")]
    if !cfg!(miri) && std::arch::is_aarch64_feature_detected!("neon") {
        use std::arch::aarch64::*;
        let sp = src.as_ptr();
        // SAFETY: (category 2) NEON guaranteed by feature detection guard above.
        let t_v = unsafe { vdupq_n_u8(t) };
        let _three = unsafe { vdupq_n_u8(3) };
        let mut x = border;

        while x + 16 <= w - border {
            let idx = row_start + x;
            // SAFETY: ISA guard (feature detection) above; idx+16 <= w-border bounded by loop condition.
            let centers = unsafe { vld1q_u8(sp.add(idx)) };
            let _c_hi = unsafe { vqaddq_u8(centers, t_v) };
            let _c_lo = unsafe { vqsubq_u8(centers, t_v) };

            // Load N, E, S, W for 16 pixels (each at different offsets)
            // But offsets are relative to center — need gather, which NEON can't do efficiently.
            // Instead, check if ANY of the 16 could be a corner using max/min of the row.
            // Skip the batch if the entire row is uniform (common case).
            // SAFETY: ISA guard (feature detection) above; operating on loaded register.
            let row_max = unsafe { vmaxvq_u8(centers) };
            let row_min = unsafe { vminvq_u8(centers) };
            if row_max - row_min < t {
                // Entire 16-pixel batch has < threshold variation → no corners
                x += 16;
                continue;
            }

            // Fall through to per-pixel scalar for this batch
            for xx in x..x + 16 {
                fast9_check_pixel(src, row_start + xx, xx, y, offsets, t, out);
            }
            x += 16;
        }

        // Scalar tail
        for xx in x..w - border {
            fast9_check_pixel(src, row_start + xx, xx, y, offsets, t, out);
        }
        return;
    }

    // Scalar fallback
    for x in border..w - border {
        fast9_check_pixel(src, row_start + x, x, y, offsets, t, out);
    }
}

#[inline]
fn fast9_check_pixel(
    src: &[u8],
    idx: usize,
    x: usize,
    y: usize,
    offsets: &[isize; 16],
    t: u8,
    out: &mut Vec<(u16, u16, u16)>,
) {
    let center = src[idx];
    let c_hi = center.saturating_add(t);
    let c_lo = center.saturating_sub(t);

    // Early rejection: N, E, S, W
    let pn = src[(idx as isize + offsets[0]) as usize];
    let pe = src[(idx as isize + offsets[4]) as usize];
    let ps = src[(idx as isize + offsets[8]) as usize];
    let pw = src[(idx as isize + offsets[12]) as usize];

    let mut n_bright = 0u8;
    let mut n_dark = 0u8;
    if pn > c_hi {
        n_bright += 1;
    } else if pn < c_lo {
        n_dark += 1;
    }
    if pe > c_hi {
        n_bright += 1;
    } else if pe < c_lo {
        n_dark += 1;
    }
    if ps > c_hi {
        n_bright += 1;
    } else if ps < c_lo {
        n_dark += 1;
    }
    if pw > c_hi {
        n_bright += 1;
    } else if pw < c_lo {
        n_dark += 1;
    }
    if n_bright < 3 && n_dark < 3 {
        return;
    }

    // Full 16-pixel test
    let mut bright_bits: u32 = 0;
    let mut dark_bits: u32 = 0;
    for (i, &off) in offsets.iter().enumerate() {
        let p = src[(idx as isize + off) as usize];
        if p > c_hi {
            bright_bits |= 1 << i;
        }
        if p < c_lo {
            dark_bits |= 1 << i;
        }
    }

    if has_9_contiguous(bright_bits) || has_9_contiguous(dark_bits) {
        let resp = fast9_response(src, idx, offsets, t);
        out.push((x as u16, y as u16, resp));
    }
}

// ============================================================================
// Distance transform (u8 binary → u16 L1 distance)
// ============================================================================

/// L1 distance transform on binary u8 image. Non-zero pixels get distance 0,
/// zero pixels get the L1 distance to the nearest non-zero pixel.
/// Returns result as a flat `Vec<u16>` (max distance = h + w).
pub fn distance_transform_u8(image: &ImageU8) -> Vec<u16> {
    if image.channels() != 1 {
        return vec![];
    }
    let (h, w) = (image.height(), image.width());
    let src = image.data();
    let cap = (h + w) as u16;

    // Initialize: non-zero src → 0, zero src → cap
    let total = h * w;
    let mut dist = vec![0u16; total];

    // SIMD initialization: compare src[i] with 0, select 0 or cap
    {
        let sp = src.as_ptr();
        let dp: *mut u16 = dist.as_mut_ptr();
        #[allow(unused_mut)]
        let mut i = 0usize;

        #[cfg(target_arch = "aarch64")]
        if !cfg!(miri) && std::arch::is_aarch64_feature_detected!("neon") {
            use std::arch::aarch64::*;
            // SAFETY: ISA guard (feature detection) above; i+16 <= total bounds checked.
            unsafe {
                let zero_v = vdupq_n_u8(0);
                let cap_v = vdupq_n_u16(cap);
                let zero16 = vdupq_n_u16(0);
                while i + 16 <= total {
                    let s = vld1q_u8(sp.add(i));
                    let mask = vceqq_u8(s, zero_v);
                    let mask_lo = vmovl_u8(vget_low_u8(mask));
                    let val_lo = vbslq_u16(mask_lo, cap_v, zero16);
                    vst1q_u16(dp.add(i), val_lo);
                    let mask_hi = vmovl_u8(vget_high_u8(mask));
                    let val_hi = vbslq_u16(mask_hi, cap_v, zero16);
                    vst1q_u16(dp.add(i + 8), val_hi);
                    i += 16;
                }
            }
        }

        for j in i..total {
            // SAFETY: j < total = h*w; sp/dp from validated slice pointers.
            unsafe {
                *dp.add(j) = if *sp.add(j) != 0 { 0 } else { cap };
            }
        }
    }

    // Forward pass: top-to-bottom, then left-to-right
    // Vertical: d[y][x] = min(d[y][x], d[y-1][x] + 1)
    #[cfg(target_arch = "aarch64")]
    let has_neon = !cfg!(miri) && std::arch::is_aarch64_feature_detected!("neon");
    #[cfg(target_arch = "x86_64")]
    let has_sse = !cfg!(miri) && is_x86_feature_detected!("sse2");

    // Exactly 2 passes — same as OpenCV:
    // Pass 1 (forward): for each pixel, propagate from top and left
    // Pass 2 (backward): for each pixel, propagate from bottom and right
    //
    // Each pass combines vertical + horizontal in one scan.

    // Forward: row 0 left-to-right only, then rows 1..h do vertical+horizontal
    #[allow(unsafe_code)]
    // SAFETY: dp offsets 0..w within dist allocation.
    unsafe {
        let dp = dist.as_mut_ptr();
        for i in 1..w {
            let left_plus1 = (*dp.add(i - 1)).saturating_add(1);
            let cur_val = dp.add(i);
            if left_plus1 < *cur_val {
                *cur_val = left_plus1;
            }
        }
    }
    for y in 1..h {
        let cur = y * w;
        let prev = (y - 1) * w;

        // Vertical: min(current, above + 1) — SIMD
        let mut x = 0usize;
        #[cfg(target_arch = "aarch64")]
        if has_neon {
            // SAFETY: ISA guard (feature detection) above.
            x = unsafe { dt_vert_fwd_neon(&mut dist, prev, cur, w) };
        }
        #[cfg(target_arch = "x86_64")]
        if has_sse {
            // SAFETY: ISA guard (feature detection) above.
            x = unsafe { dt_vert_fwd_sse(&mut dist, prev, cur, w) };
        }
        for i in x..w {
            dist[cur + i] = dist[cur + i].min(dist[prev + i].saturating_add(1));
        }

        #[allow(unsafe_code)]
        // SAFETY: dp offsets 0..w within dist row; cur = y*w bounds checked by loop.
        unsafe {
            let dp = dist.as_mut_ptr().add(cur);
            for i in 1..w {
                let left_plus1 = (*dp.add(i - 1)).saturating_add(1);
                let cur_val = dp.add(i);
                if left_plus1 < *cur_val {
                    *cur_val = left_plus1;
                }
            }
        }
    }

    // Backward: last row right-to-left, then rows h-2..0 do vertical+horizontal
    #[allow(unsafe_code)]
    // SAFETY: dp offsets 0..w within last row of dist allocation.
    unsafe {
        let dp = dist.as_mut_ptr().add((h - 1) * w);
        for i in (0..w.saturating_sub(1)).rev() {
            let right_plus1 = (*dp.add(i + 1)).saturating_add(1);
            let cur_val = dp.add(i);
            if right_plus1 < *cur_val {
                *cur_val = right_plus1;
            }
        }
    }
    for y in (0..h.saturating_sub(1)).rev() {
        let cur = y * w;
        let next = (y + 1) * w;

        // Vertical SIMD
        let mut x = 0usize;
        #[cfg(target_arch = "aarch64")]
        if has_neon {
            // SAFETY: ISA guard (feature detection) above.
            x = unsafe { dt_vert_bwd_neon(&mut dist, next, cur, w) };
        }
        #[cfg(target_arch = "x86_64")]
        if has_sse {
            // SAFETY: ISA guard (feature detection) above.
            x = unsafe { dt_vert_bwd_sse(&mut dist, next, cur, w) };
        }
        for i in x..w {
            dist[cur + i] = dist[cur + i].min(dist[next + i].saturating_add(1));
        }

        #[allow(unsafe_code)]
        // SAFETY: dp offsets 0..w within dist row; cur = y*w bounds checked by loop.
        unsafe {
            let dp = dist.as_mut_ptr().add(cur);
            for i in (0..w.saturating_sub(1)).rev() {
                let right_plus1 = (*dp.add(i + 1)).saturating_add(1);
                let cur_val = dp.add(i);
                if right_plus1 < *cur_val {
                    *cur_val = right_plus1;
                }
            }
        }
    }

    dist
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn dt_vert_fwd_neon(dist: &mut [u16], prev_off: usize, cur_off: usize, w: usize) -> usize {
    use std::arch::aarch64::*;
    let dp = dist.as_mut_ptr();
    let one = vdupq_n_u16(1);
    let mut x = 0usize;
    while x + 8 <= w {
        let prev = vld1q_u16(dp.add(prev_off + x));
        let cur = vld1q_u16(dp.add(cur_off + x));
        let prev_plus1 = vqaddq_u16(prev, one);
        let result = vminq_u16(cur, prev_plus1);
        vst1q_u16(dp.add(cur_off + x), result);
        x += 8;
    }
    x
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn dt_vert_bwd_neon(dist: &mut [u16], next_off: usize, cur_off: usize, w: usize) -> usize {
    use std::arch::aarch64::*;
    let dp = dist.as_mut_ptr();
    let one = vdupq_n_u16(1);
    let mut x = 0usize;
    while x + 8 <= w {
        let next = vld1q_u16(dp.add(next_off + x));
        let cur = vld1q_u16(dp.add(cur_off + x));
        let next_plus1 = vqaddq_u16(next, one);
        let result = vminq_u16(cur, next_plus1);
        vst1q_u16(dp.add(cur_off + x), result);
        x += 8;
    }
    x
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn dt_vert_fwd_sse(dist: &mut [u16], prev_off: usize, cur_off: usize, w: usize) -> usize {
    use std::arch::x86_64::*;
    let dp = dist.as_mut_ptr();
    let one = _mm_set1_epi16(1);
    let mut x = 0usize;
    while x + 8 <= w {
        let prev = _mm_loadu_si128(dp.add(prev_off + x) as *const __m128i);
        let cur = _mm_loadu_si128(dp.add(cur_off + x) as *const __m128i);
        let prev_plus1 = _mm_adds_epu16(prev, one);
        // SSE2 has no min_epu16, but we can use: min(a,b) = a + b - max(a,b)
        // Or use the signed comparison trick: offset both by 0x8000
        let offset = _mm_set1_epi16(-0x8000i16);
        let a_signed = _mm_add_epi16(cur, offset);
        let b_signed = _mm_add_epi16(prev_plus1, offset);
        let min_signed = _mm_min_epi16(a_signed, b_signed);
        let result = _mm_sub_epi16(min_signed, offset);
        _mm_storeu_si128(dp.add(cur_off + x) as *mut __m128i, result);
        x += 8;
    }
    x
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn dt_vert_bwd_sse(dist: &mut [u16], next_off: usize, cur_off: usize, w: usize) -> usize {
    use std::arch::x86_64::*;
    let dp = dist.as_mut_ptr();
    let one = _mm_set1_epi16(1);
    let mut x = 0usize;
    while x + 8 <= w {
        let next = _mm_loadu_si128(dp.add(next_off + x) as *const __m128i);
        let cur = _mm_loadu_si128(dp.add(cur_off + x) as *const __m128i);
        let next_plus1 = _mm_adds_epu16(next, one);
        let offset = _mm_set1_epi16(-0x8000i16);
        let a_signed = _mm_add_epi16(cur, offset);
        let b_signed = _mm_add_epi16(next_plus1, offset);
        let min_signed = _mm_min_epi16(a_signed, b_signed);
        let result = _mm_sub_epi16(min_signed, offset);
        _mm_storeu_si128(dp.add(cur_off + x) as *mut __m128i, result);
        x += 8;
    }
    x
}

// ============================================================================
// Warp perspective (u8, single-channel)
// ============================================================================

/// Warp u8 single-channel image using a 3x3 homography matrix.
/// Uses bilinear interpolation. Out-of-bounds pixels are set to 0.
pub fn warp_perspective_u8(
    image: &ImageU8,
    h_matrix: &[f64; 9],
    out_h: usize,
    out_w: usize,
) -> ImageU8 {
    let (ih, iw) = (image.height(), image.width());
    let channels = image.channels();
    let src = image.data();
    let mut out = vec![0u8; out_h * out_w * channels];

    let h = *h_matrix;
    let max_x = (iw as f64) - 1.0001;
    let max_y = (ih as f64) - 1.0001;

    let use_rayon = out_h * out_w >= RAYON_THRESHOLD && !cfg!(miri);

    if use_rayon {
        let src_ptr = super::SendConstPtr(src.as_ptr());
        let out_ptr = super::SendPtr(out.as_mut_ptr());
        (0..out_h).into_par_iter().for_each(|oy| {
            // SAFETY: pointer and length from validated image data; parallel rows are non-overlapping.
            let src = unsafe { std::slice::from_raw_parts(src_ptr.ptr(), ih * iw * channels) };
            let out_row = unsafe {
                std::slice::from_raw_parts_mut(
                    out_ptr.ptr().add(oy * out_w * channels),
                    out_w * channels,
                )
            };
            warp_perspective_row(src, out_row, ih, iw, channels, &h, max_x, max_y, oy, out_w);
        });
    } else {
        for oy in 0..out_h {
            let row_start = oy * out_w * channels;
            let row_end = row_start + out_w * channels;
            warp_perspective_row(
                src,
                &mut out[row_start..row_end],
                ih,
                iw,
                channels,
                &h,
                max_x,
                max_y,
                oy,
                out_w,
            );
        }
    }

    ImageU8::new(out, out_h, out_w, channels).expect("output dimensions match data")
}

/// Process one output row for warp_perspective_u8. Uses f32 + unsafe for speed.
#[allow(unsafe_code)]
fn warp_perspective_row(
    src: &[u8],
    dst: &mut [u8],
    ih: usize,
    iw: usize,
    channels: usize,
    h: &[f64; 9],
    max_x: f64,
    max_y: f64,
    oy: usize,
    out_w: usize,
) {
    // Convert to f32 for faster arithmetic (sufficient for 480×640)
    let h0 = h[0] as f32;
    let h1 = h[1] as f32;
    let h2 = h[2] as f32;
    let h3 = h[3] as f32;
    let h4 = h[4] as f32;
    let h5 = h[5] as f32;
    let h6 = h[6] as f32;
    let h7 = h[7] as f32;
    let h8 = h[8] as f32;
    let oy_f = oy as f32;
    let max_xf = max_x as f32;
    let max_yf = max_y as f32;

    let base_x = h1 * oy_f + h2;
    let base_y = h4 * oy_f + h5;
    let base_w = h7 * oy_f + h8;

    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();

    if channels == 1 {
        // Single-channel fast path: no inner channel loop
        for ox in 0..out_w {
            let ox_f = ox as f32;
            let denom = h6 * ox_f + base_w;
            if denom.abs() < 1e-7 {
                continue;
            }
            let inv_w = 1.0 / denom;
            let sx = (h0 * ox_f + base_x) * inv_w;
            let sy = (h3 * ox_f + base_y) * inv_w;

            if sx < 0.0 || sy < 0.0 || sx > max_xf || sy > max_yf {
                continue;
            }

            let sxi = sx as usize;
            let syi = sy as usize;
            let sx1 = (sxi + 1).min(iw - 1);
            let sy1 = (syi + 1).min(ih - 1);

            let fx = ((sx - sxi as f32) * 256.0) as u16;
            let fy = ((sy - syi as f32) * 256.0) as u16;

            // SAFETY: bounds checked by sx/sy clamp and min(iw-1)/min(ih-1) above.
            unsafe {
                let p00 = *sp.add(syi * iw + sxi) as u16;
                let p10 = *sp.add(syi * iw + sx1) as u16;
                let p01 = *sp.add(sy1 * iw + sxi) as u16;
                let p11 = *sp.add(sy1 * iw + sx1) as u16;

                let top = (p00 * (256 - fx) + p10 * fx) >> 8;
                let bot = (p01 * (256 - fx) + p11 * fx) >> 8;
                *dp.add(ox) = ((top * (256 - fy) + bot * fy) >> 8) as u8;
            }
        }
    } else {
        for ox in 0..out_w {
            let ox_f = ox as f32;
            let denom = h6 * ox_f + base_w;
            if denom.abs() < 1e-7 {
                continue;
            }
            let inv_w = 1.0 / denom;
            let sx = (h0 * ox_f + base_x) * inv_w;
            let sy = (h3 * ox_f + base_y) * inv_w;

            if sx < 0.0 || sy < 0.0 || sx > max_xf || sy > max_yf {
                continue;
            }

            let sxi = sx as usize;
            let syi = sy as usize;
            let sx1 = (sxi + 1).min(iw - 1);
            let sy1 = (syi + 1).min(ih - 1);

            let fx = ((sx - sxi as f32) * 256.0) as u16;
            let fy = ((sy - syi as f32) * 256.0) as u16;
            let fx_inv = 256 - fx;
            let fy_inv = 256 - fy;

            for c in 0..channels {
                // SAFETY: bounds checked by sx/sy clamp and min(iw-1)/min(ih-1) above.
                unsafe {
                    let p00 = *sp.add((syi * iw + sxi) * channels + c) as u16;
                    let p10 = *sp.add((syi * iw + sx1) * channels + c) as u16;
                    let p01 = *sp.add((sy1 * iw + sxi) * channels + c) as u16;
                    let p11 = *sp.add((sy1 * iw + sx1) * channels + c) as u16;

                    let top = (p00 * fx_inv + p10 * fx) >> 8;
                    let bot = (p01 * fx_inv + p11 * fx) >> 8;
                    *dp.add(ox * channels + c) = ((top * fy_inv + bot * fy) >> 8) as u8;
                }
            }
        }
    }
}

// ============================================================================
// Bilateral filter (u8, single-channel)
// ============================================================================

/// Bilateral filter operating directly on u8 pixel data.
///
/// The key advantage over the f32 path:
/// - `vabdq_u8` computes 16 absolute differences in ONE instruction
/// - 16 pixels per NEON register vs 4 for f32
/// - No float conversion needed for the diff computation
/// - Memory bandwidth: 1 byte vs 4 bytes per pixel
///
/// `channels` must be 1 for now.
pub fn bilateral_filter_u8(
    src: &[u8],
    width: usize,
    height: usize,
    channels: usize,
    diameter: usize,
    sigma_color: f32,
    sigma_space: f32,
) -> Vec<u8> {
    assert_eq!(
        channels, 1,
        "bilateral_filter_u8: only single-channel supported"
    );
    assert!(
        width > 0 && height > 0,
        "bilateral_filter_u8: zero dimensions"
    );
    assert_eq!(
        src.len(),
        width * height,
        "bilateral_filter_u8: data length mismatch"
    );

    let d = diameter;
    let radius = d / 2;

    // Precompute spatial LUT: exp(-dist²/(2σs²))
    let inv_2sigma_space_sq = -1.0 / (2.0 * sigma_space * sigma_space);
    let spatial_lut: Vec<f32> = (0..d * d)
        .map(|idx| {
            let dy = (idx / d) as f32 - radius as f32;
            let dx = (idx % d) as f32 - radius as f32;
            let dist_sq = dx * dx + dy * dy;
            (dist_sq * inv_2sigma_space_sq).exp()
        })
        .collect();

    // Precompute color LUT: exp(-i²/(2σc²)) for i=0..255
    let inv_2sigma_color_sq = -1.0 / (2.0 * sigma_color * sigma_color);
    let color_lut: Vec<f32> = (0..256)
        .map(|i| {
            let diff = i as f32;
            (diff * diff * inv_2sigma_color_sq).exp()
        })
        .collect();

    // Early-exit threshold: skip neighbors with negligible weight
    let max_color_diff = color_lut.iter().rposition(|&w| w >= 0.001).unwrap_or(0) as u8;

    // Combined weight table: combined[offset_idx * 256 + diff] = spatial * color
    // Eliminates per-pixel multiply. Size: 121 * 256 * 4 = 124KB
    let n_offsets = d * d;
    let mut combined_lut = vec![0.0f32; n_offsets * 256];
    for offset_idx in 0..n_offsets {
        let sw = spatial_lut[offset_idx];
        for diff in 0..256 {
            combined_lut[offset_idx * 256 + diff] = sw * color_lut[diff];
        }
    }

    let mut out = vec![0u8; width * height];

    // Dispatch to NEON path if available
    #[cfg(target_arch = "aarch64")]
    if !cfg!(miri) && std::arch::is_aarch64_feature_detected!("neon") {
        bilateral_u8_parallel(
            src,
            &mut out,
            width,
            height,
            d,
            radius,
            &spatial_lut,
            &color_lut,
            max_color_diff,
            &combined_lut,
        );
        return out;
    }

    // Dispatch to AVX2 or SSE path if available
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if !cfg!(miri) {
        if std::is_x86_feature_detected!("avx2") {
            bilateral_u8_parallel_x86::<true>(
                src,
                &mut out,
                width,
                height,
                d,
                radius,
                &spatial_lut,
                &color_lut,
                max_color_diff,
                &combined_lut,
            );
            return out;
        }
        if std::is_x86_feature_detected!("sse2") {
            bilateral_u8_parallel_x86::<false>(
                src,
                &mut out,
                width,
                height,
                d,
                radius,
                &spatial_lut,
                &color_lut,
                max_color_diff,
                &combined_lut,
            );
            return out;
        }
    }

    // Scalar fallback (also used for non-aarch64)
    bilateral_u8_parallel_scalar(
        src,
        &mut out,
        width,
        height,
        d,
        radius,
        &spatial_lut,
        &color_lut,
        max_color_diff,
    );
    out
}

/// Parallel dispatch for bilateral filter (NEON path).
#[cfg(target_arch = "aarch64")]
fn bilateral_u8_parallel(
    src: &[u8],
    out: &mut [u8],
    width: usize,
    height: usize,
    d: usize,
    radius: usize,
    spatial_lut: &[f32],
    color_lut: &[f32],
    max_color_diff: u8,
    combined_lut: &[f32],
) {
    let sp = super::SendConstPtr(src.as_ptr());
    let dp = super::SendPtr(out.as_mut_ptr());
    let spatial_ptr = super::SendConstPtr(spatial_lut.as_ptr());
    let color_ptr = super::SendConstPtr(color_lut.as_ptr());
    let combined_ptr = super::SendConstPtr(combined_lut.as_ptr());
    let combined_len = combined_lut.len();

    #[cfg(target_os = "macos")]
    if height * width >= RAYON_THRESHOLD {
        gcd::parallel_for(height, |y| {
            let sp = sp.ptr();
            let dp = dp.ptr();
            // SAFETY: pointer and length from validated LUT allocations.
            let spatial_lut = unsafe { std::slice::from_raw_parts(spatial_ptr.ptr(), d * d) };
            let color_lut = unsafe { std::slice::from_raw_parts(color_ptr.ptr(), 256) };
            let combined_lut =
                unsafe { std::slice::from_raw_parts(combined_ptr.ptr(), combined_len) };
            // SAFETY: ISA guard (feature detection) above; pointers from validated image data.
            unsafe {
                bilateral_u8_row_neon(
                    sp,
                    dp,
                    width,
                    height,
                    d,
                    radius,
                    spatial_lut,
                    color_lut,
                    max_color_diff,
                    combined_lut,
                    y,
                );
            }
        });
        return;
    }

    #[cfg(not(target_os = "macos"))]
    if height * width >= RAYON_THRESHOLD {
        (0..height).into_par_iter().for_each(|y| {
            let sp = sp.ptr();
            let dp = dp.ptr();
            // SAFETY: pointer and length from validated LUT allocations.
            let spatial_lut = unsafe { std::slice::from_raw_parts(spatial_ptr.ptr(), d * d) };
            let color_lut = unsafe { std::slice::from_raw_parts(color_ptr.ptr(), 256) };
            let combined_lut =
                unsafe { std::slice::from_raw_parts(combined_ptr.ptr(), combined_len) };
            // SAFETY: ISA guard (feature detection) above; pointers from validated image data.
            unsafe {
                bilateral_u8_row_neon(
                    sp,
                    dp,
                    width,
                    height,
                    d,
                    radius,
                    spatial_lut,
                    color_lut,
                    max_color_diff,
                    combined_lut,
                    y,
                );
            }
        });
        return;
    }

    // Sequential fallback for small images
    for y in 0..height {
        // SAFETY: ISA guard (feature detection) above; pointers from validated image data.
        unsafe {
            bilateral_u8_row_neon(
                sp.ptr(),
                dp.ptr(),
                width,
                height,
                d,
                radius,
                spatial_lut,
                color_lut,
                max_color_diff,
                combined_lut,
                y,
            );
        }
    }
}

/// Scalar parallel dispatch (non-NEON platforms).
fn bilateral_u8_parallel_scalar(
    src: &[u8],
    out: &mut [u8],
    width: usize,
    height: usize,
    d: usize,
    radius: usize,
    spatial_lut: &[f32],
    color_lut: &[f32],
    max_color_diff: u8,
) {
    let sp = super::SendConstPtr(src.as_ptr());
    let dp = super::SendPtr(out.as_mut_ptr());
    let spatial_ptr = super::SendConstPtr(spatial_lut.as_ptr());
    let color_ptr = super::SendConstPtr(color_lut.as_ptr());

    let process_row = |y: usize| {
        // SAFETY: pointer and length from validated image data; parallel rows are non-overlapping.
        let src = unsafe { std::slice::from_raw_parts(sp.ptr(), width * height) };
        let dst = unsafe { std::slice::from_raw_parts_mut(dp.ptr(), width * height) };
        // SAFETY: pointer and length from validated LUT allocations.
        let spatial_lut = unsafe { std::slice::from_raw_parts(spatial_ptr.ptr(), d * d) };
        let color_lut = unsafe { std::slice::from_raw_parts(color_ptr.ptr(), 256) };

        for x in 0..width {
            let center = src[y * width + x];
            let mut sum = 0.0f32;
            let mut wsum = 0.0f32;

            for dy_idx in 0..d {
                let ny = y as isize + dy_idx as isize - radius as isize;
                let ny = ny.clamp(0, (height - 1) as isize) as usize;
                for dx_idx in 0..d {
                    let nx = x as isize + dx_idx as isize - radius as isize;
                    let nx = nx.clamp(0, (width - 1) as isize) as usize;

                    let neighbor = src[ny * width + nx];
                    let color_diff = (neighbor as i16 - center as i16).unsigned_abs() as u8;
                    if color_diff > max_color_diff {
                        continue;
                    }
                    let w = spatial_lut[dy_idx * d + dx_idx] * color_lut[color_diff as usize];
                    sum += neighbor as f32 * w;
                    wsum += w;
                }
            }
            dst[y * width + x] = (sum / wsum + 0.5) as u8;
        }
    };

    #[cfg(target_os = "macos")]
    if height * width >= RAYON_THRESHOLD {
        gcd::parallel_for(height, process_row);
        return;
    }

    if height * width >= RAYON_THRESHOLD {
        (0..height).into_par_iter().for_each(process_row);
        return;
    }

    for y in 0..height {
        process_row(y);
    }
}

// ===========================================================================
// x86 SSE2 / AVX2 bilateral filter support
// ===========================================================================

/// Parallel dispatch for bilateral filter (x86 SSE2/AVX2 path).
/// `USE_AVX2` selects between the AVX2 (32-byte) and SSE2 (16-byte) inner kernels.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bilateral_u8_parallel_x86<const USE_AVX2: bool>(
    src: &[u8],
    out: &mut [u8],
    width: usize,
    height: usize,
    d: usize,
    radius: usize,
    spatial_lut: &[f32],
    color_lut: &[f32],
    max_color_diff: u8,
    combined_lut: &[f32],
) {
    let sp = super::SendConstPtr(src.as_ptr());
    let dp = super::SendPtr(out.as_mut_ptr());
    let spatial_ptr = super::SendConstPtr(spatial_lut.as_ptr());
    let color_ptr = super::SendConstPtr(color_lut.as_ptr());
    let combined_ptr = super::SendConstPtr(combined_lut.as_ptr());
    let combined_len = combined_lut.len();

    let process_row = |y: usize| {
        let sp = sp.ptr();
        let dp = dp.ptr();
        // SAFETY: pointer and length from validated LUT allocations.
        let spatial_lut = unsafe { std::slice::from_raw_parts(spatial_ptr.ptr(), d * d) };
        let color_lut = unsafe { std::slice::from_raw_parts(color_ptr.ptr(), 256) };
        let combined_lut = unsafe { std::slice::from_raw_parts(combined_ptr.ptr(), combined_len) };
        if USE_AVX2 {
            // SAFETY: ISA guard (AVX2 feature detection) in caller.
            unsafe {
                bilateral_u8_row_avx2(
                    sp,
                    dp,
                    width,
                    height,
                    d,
                    radius,
                    spatial_lut,
                    color_lut,
                    max_color_diff,
                    combined_lut,
                    y,
                );
            }
        } else {
            // SAFETY: ISA guard (SSE2 feature detection) in caller.
            unsafe {
                bilateral_u8_row_sse(
                    sp,
                    dp,
                    width,
                    height,
                    d,
                    radius,
                    spatial_lut,
                    color_lut,
                    max_color_diff,
                    combined_lut,
                    y,
                );
            }
        }
    };

    #[cfg(target_os = "macos")]
    if height * width >= RAYON_THRESHOLD {
        gcd::parallel_for(height, process_row);
        return;
    }

    #[cfg(not(target_os = "macos"))]
    if height * width >= RAYON_THRESHOLD {
        (0..height).into_par_iter().for_each(process_row);
        return;
    }

    for y in 0..height {
        process_row(y);
    }
}

/// SSE2-accelerated bilateral filter for a single row.
///
/// Uses `_mm_sad_epu8` to compute 16 absolute differences at once (sum of abs diff),
/// then gathers weights from the combined LUT and accumulates in f32.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn bilateral_u8_row_sse(
    src: *const u8,
    dst: *mut u8,
    width: usize,
    height: usize,
    d: usize,
    radius: usize,
    spatial_lut: &[f32],
    color_lut: &[f32],
    _max_color_diff: u8,
    combined_lut: &[f32],
    y: usize,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let r = radius;
    let row_base = y * width;
    let mut sum_buf = vec![0.0f32; width];
    let mut wsum_buf = vec![0.0f32; width];
    let center_row = src.add(row_base);

    for dy_idx in 0..d {
        let ny =
            (y as isize + dy_idx as isize - r as isize).clamp(0, (height - 1) as isize) as usize;
        let neighbor_row_base = src.add(ny * width);
        let spatial_row_base = dy_idx * d;

        for dx_idx in 0..d {
            let dx_offset = dx_idx as isize - r as isize;
            let spatial_w = *spatial_lut.get_unchecked(spatial_row_base + dx_idx);
            if spatial_w == 0.0 {
                continue;
            }

            let offset_idx = spatial_row_base + dx_idx;
            let clut_base = combined_lut.as_ptr().add(offset_idx * 256);

            let mut x = 0usize;
            // Process 16 pixels at a time using SSE2 _mm_sad_epu8
            while x + 16 <= width {
                let nx_start = x as isize + dx_offset;

                if nx_start >= 0 && (nx_start as usize + 16) <= width {
                    let center_v = _mm_loadu_si128(center_row.add(x) as *const __m128i);
                    let neighbor_v =
                        _mm_loadu_si128(neighbor_row_base.add(nx_start as usize) as *const __m128i);

                    // Compute absolute differences: max(a,b) - min(a,b)
                    let max_v = _mm_max_epu8(center_v, neighbor_v);
                    let min_v = _mm_min_epu8(center_v, neighbor_v);
                    let diff_v = _mm_sub_epi8(max_v, min_v);

                    // Extract diff bytes for LUT lookup
                    let mut diff_arr = [0u8; 16];
                    _mm_storeu_si128(diff_arr.as_mut_ptr() as *mut __m128i, diff_v);

                    // Extract neighbor bytes for weighted sum
                    let mut nb_arr = [0u8; 16];
                    _mm_storeu_si128(nb_arr.as_mut_ptr() as *mut __m128i, neighbor_v);

                    // Process 4 groups of 4 pixels using SSE f32
                    for g in 0..4 {
                        let base = g * 4;
                        let w0 = *clut_base.add(diff_arr[base] as usize);
                        let w1 = *clut_base.add(diff_arr[base + 1] as usize);
                        let w2 = *clut_base.add(diff_arr[base + 2] as usize);
                        let w3 = *clut_base.add(diff_arr[base + 3] as usize);
                        let w_vec = _mm_set_ps(w3, w2, w1, w0);

                        let n_vec = _mm_set_ps(
                            nb_arr[base + 3] as f32,
                            nb_arr[base + 2] as f32,
                            nb_arr[base + 1] as f32,
                            nb_arr[base] as f32,
                        );

                        let sp = sum_buf.as_mut_ptr().add(x + base);
                        let wp = wsum_buf.as_mut_ptr().add(x + base);
                        let old_s = _mm_loadu_ps(sp);
                        let old_w = _mm_loadu_ps(wp);
                        _mm_storeu_ps(sp, _mm_add_ps(old_s, _mm_mul_ps(n_vec, w_vec)));
                        _mm_storeu_ps(wp, _mm_add_ps(old_w, w_vec));
                    }
                } else {
                    // Slow path for border pixels
                    for i in 0..16 {
                        let px = x + i;
                        let nx = (px as isize + dx_offset).clamp(0, (width - 1) as isize) as usize;
                        let center = *center_row.add(px);
                        let neighbor = *neighbor_row_base.add(nx);
                        let cd = (neighbor as i16 - center as i16).unsigned_abs() as usize;
                        let w = spatial_w * *color_lut.get_unchecked(cd);
                        *sum_buf.get_unchecked_mut(px) += neighbor as f32 * w;
                        *wsum_buf.get_unchecked_mut(px) += w;
                    }
                }
                x += 16;
            }

            while x < width {
                let nx = (x as isize + dx_offset).clamp(0, (width - 1) as isize) as usize;
                let center = *center_row.add(x);
                let neighbor = *neighbor_row_base.add(nx);
                let cd = (neighbor as i16 - center as i16).unsigned_abs() as usize;
                let w = spatial_w * *color_lut.get_unchecked(cd);
                *sum_buf.get_unchecked_mut(x) += neighbor as f32 * w;
                *wsum_buf.get_unchecked_mut(x) += w;
                x += 1;
            }
        }
    }

    // Final normalization
    let dst_row = dst.add(row_base);
    for x in 0..width {
        let s = *sum_buf.get_unchecked(x);
        let ws = *wsum_buf.get_unchecked(x);
        *dst_row.add(x) = if ws > 0.0 {
            (s / ws + 0.5) as u8
        } else {
            *center_row.add(x)
        };
    }
}

/// AVX2-accelerated bilateral filter for a single row.
///
/// Uses `_mm256_sad_epu8` to compute 32 absolute differences at once,
/// then gathers weights from the combined LUT and accumulates in f32.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn bilateral_u8_row_avx2(
    src: *const u8,
    dst: *mut u8,
    width: usize,
    height: usize,
    d: usize,
    radius: usize,
    spatial_lut: &[f32],
    color_lut: &[f32],
    _max_color_diff: u8,
    combined_lut: &[f32],
    y: usize,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let r = radius;
    let row_base = y * width;
    let mut sum_buf = vec![0.0f32; width];
    let mut wsum_buf = vec![0.0f32; width];
    let center_row = src.add(row_base);

    for dy_idx in 0..d {
        let ny =
            (y as isize + dy_idx as isize - r as isize).clamp(0, (height - 1) as isize) as usize;
        let neighbor_row_base = src.add(ny * width);
        let spatial_row_base = dy_idx * d;

        for dx_idx in 0..d {
            let dx_offset = dx_idx as isize - r as isize;
            let spatial_w = *spatial_lut.get_unchecked(spatial_row_base + dx_idx);
            if spatial_w == 0.0 {
                continue;
            }

            let offset_idx = spatial_row_base + dx_idx;
            let clut_base = combined_lut.as_ptr().add(offset_idx * 256);

            let mut x = 0usize;
            // Process 32 pixels at a time using AVX2 _mm256_sad_epu8
            while x + 32 <= width {
                let nx_start = x as isize + dx_offset;

                if nx_start >= 0 && (nx_start as usize + 32) <= width {
                    let center_v = _mm256_loadu_si256(center_row.add(x) as *const __m256i);
                    let neighbor_v = _mm256_loadu_si256(
                        neighbor_row_base.add(nx_start as usize) as *const __m256i
                    );

                    // Compute absolute differences: max(a,b) - min(a,b)
                    let max_v = _mm256_max_epu8(center_v, neighbor_v);
                    let min_v = _mm256_min_epu8(center_v, neighbor_v);
                    let diff_v = _mm256_sub_epi8(max_v, min_v);

                    // Extract diff bytes for LUT lookup
                    let mut diff_arr = [0u8; 32];
                    _mm256_storeu_si256(diff_arr.as_mut_ptr() as *mut __m256i, diff_v);

                    // Extract neighbor bytes for weighted sum
                    let mut nb_arr = [0u8; 32];
                    _mm256_storeu_si256(nb_arr.as_mut_ptr() as *mut __m256i, neighbor_v);

                    // Process 4 groups of 8 pixels using AVX f32
                    for g in 0..4 {
                        let base = g * 8;

                        let w_vec = _mm256_set_ps(
                            *clut_base.add(diff_arr[base + 7] as usize),
                            *clut_base.add(diff_arr[base + 6] as usize),
                            *clut_base.add(diff_arr[base + 5] as usize),
                            *clut_base.add(diff_arr[base + 4] as usize),
                            *clut_base.add(diff_arr[base + 3] as usize),
                            *clut_base.add(diff_arr[base + 2] as usize),
                            *clut_base.add(diff_arr[base + 1] as usize),
                            *clut_base.add(diff_arr[base] as usize),
                        );

                        let n_vec = _mm256_set_ps(
                            nb_arr[base + 7] as f32,
                            nb_arr[base + 6] as f32,
                            nb_arr[base + 5] as f32,
                            nb_arr[base + 4] as f32,
                            nb_arr[base + 3] as f32,
                            nb_arr[base + 2] as f32,
                            nb_arr[base + 1] as f32,
                            nb_arr[base] as f32,
                        );

                        let sp = sum_buf.as_mut_ptr().add(x + base);
                        let wp = wsum_buf.as_mut_ptr().add(x + base);
                        let old_s = _mm256_loadu_ps(sp);
                        let old_w = _mm256_loadu_ps(wp);
                        _mm256_storeu_ps(sp, _mm256_add_ps(old_s, _mm256_mul_ps(n_vec, w_vec)));
                        _mm256_storeu_ps(wp, _mm256_add_ps(old_w, w_vec));
                    }
                } else {
                    // Slow path for border pixels
                    for i in 0..32 {
                        let px = x + i;
                        let nx = (px as isize + dx_offset).clamp(0, (width - 1) as isize) as usize;
                        let center = *center_row.add(px);
                        let neighbor = *neighbor_row_base.add(nx);
                        let cd = (neighbor as i16 - center as i16).unsigned_abs() as usize;
                        let w = spatial_w * *color_lut.get_unchecked(cd);
                        *sum_buf.get_unchecked_mut(px) += neighbor as f32 * w;
                        *wsum_buf.get_unchecked_mut(px) += w;
                    }
                }
                x += 32;
            }

            // Handle remaining pixels with scalar
            while x < width {
                let nx = (x as isize + dx_offset).clamp(0, (width - 1) as isize) as usize;
                let center = *center_row.add(x);
                let neighbor = *neighbor_row_base.add(nx);
                let cd = (neighbor as i16 - center as i16).unsigned_abs() as usize;
                let w = spatial_w * *color_lut.get_unchecked(cd);
                *sum_buf.get_unchecked_mut(x) += neighbor as f32 * w;
                *wsum_buf.get_unchecked_mut(x) += w;
                x += 1;
            }
        }
    }

    // Final normalization
    let dst_row = dst.add(row_base);
    for x in 0..width {
        let s = *sum_buf.get_unchecked(x);
        let ws = *wsum_buf.get_unchecked(x);
        *dst_row.add(x) = if ws > 0.0 {
            (s / ws + 0.5) as u8
        } else {
            *center_row.add(x)
        };
    }
}

/// NEON-accelerated bilateral filter for a single row — columnar/row-vectorized.
///
/// Instead of iterating per-pixel then over the d×d window, this processes
/// all pixels in the row simultaneously for each (dy, dx) offset. For a given
/// offset the spatial weight is constant across all pixels, so we only need to
/// compute per-pixel color weights and accumulate.
///
/// Inner loop uses NEON:
/// - `vabdq_u8` computes 16 absolute differences in one instruction
/// - Scalar gather for color LUT (unavoidable — no NEON gather)
/// - f32 FMA accumulation with 4-wide NEON vectors
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn bilateral_u8_row_neon(
    src: *const u8,
    dst: *mut u8,
    width: usize,
    height: usize,
    d: usize,
    radius: usize,
    spatial_lut: &[f32],
    color_lut: &[f32],
    max_color_diff: u8,
    combined_lut: &[f32],
    y: usize,
) {
    use std::arch::aarch64::*;

    let r = radius;
    let row_base = y * width;

    // Accumulation buffers for the entire row
    let mut sum_buf = vec![0.0f32; width];
    let mut wsum_buf = vec![0.0f32; width];

    // Center row pointer
    let center_row = src.add(row_base);

    let _max_cd = max_color_diff;

    // For each (dy, dx) offset in the window, process ALL pixels at once
    for dy_idx in 0..d {
        let ny =
            (y as isize + dy_idx as isize - r as isize).clamp(0, (height - 1) as isize) as usize;
        let neighbor_row_base = src.add(ny * width);
        let spatial_row_base = dy_idx * d;

        for dx_idx in 0..d {
            let dx_offset = dx_idx as isize - r as isize;
            let spatial_w = *spatial_lut.get_unchecked(spatial_row_base + dx_idx);
            if spatial_w == 0.0 {
                continue;
            }

            let offset_idx = spatial_row_base + dx_idx;
            let clut_base = combined_lut.as_ptr().add(offset_idx * 256);

            // Process pixels in chunks of 16 (NEON u8 lane width)
            let mut x = 0usize;
            while x + 16 <= width {
                // Load 16 center pixels
                let center_v = vld1q_u8(center_row.add(x));

                // Compute clamped neighbor x-coordinates and load neighbor pixels.
                // For interior columns we can load directly; for borders we must clamp.
                let nx_start = x as isize + dx_offset;

                // Fast path: if the entire 16-pixel span is within bounds, direct load
                if nx_start >= 0 && (nx_start as usize + 16) <= width {
                    let neighbor_v = vld1q_u8(neighbor_row_base.add(nx_start as usize));

                    // 16 absolute diffs in one instruction
                    let diff_v = vabdq_u8(neighbor_v, center_v);

                    let diff_arr: [u8; 16] = std::mem::transmute(diff_v);
                    let _clut = color_lut.as_ptr();

                    // Widen neighbors to f32 via NEON (no scalar conversion)
                    let nb_lo8 = vget_low_u8(neighbor_v);
                    let nb_hi8 = vget_high_u8(neighbor_v);
                    let nb16_lo = vmovl_u8(nb_lo8);
                    let nb16_hi = vmovl_u8(nb_hi8);
                    let nf0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(nb16_lo)));
                    let nf4 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(nb16_lo)));
                    let nf8 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(nb16_hi)));
                    let nf12 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(nb16_hi)));

                    // Process 4 groups of 4 pixels — use combined LUT
                    // (spatial * color precomputed, no multiply needed)
                    macro_rules! process_group {
                        ($off:expr, $nf:expr) => {{
                            let w_arr: [f32; 4] = [
                                *clut_base.add(diff_arr[$off] as usize),
                                *clut_base.add(diff_arr[$off + 1] as usize),
                                *clut_base.add(diff_arr[$off + 2] as usize),
                                *clut_base.add(diff_arr[$off + 3] as usize),
                            ];
                            let w_vec = vld1q_f32(w_arr.as_ptr());
                            let sp = sum_buf.as_mut_ptr().add(x + $off);
                            let wp = wsum_buf.as_mut_ptr().add(x + $off);
                            vst1q_f32(sp, vfmaq_f32(vld1q_f32(sp), $nf, w_vec));
                            vst1q_f32(wp, vaddq_f32(vld1q_f32(wp), w_vec));
                        }};
                    }
                    process_group!(0, nf0);
                    process_group!(4, nf4);
                    process_group!(8, nf8);
                    process_group!(12, nf12);
                } else {
                    // Slow path for border pixels — scalar per pixel (branchless)
                    for i in 0..16 {
                        let px = x + i;
                        let nx = (px as isize + dx_offset).clamp(0, (width - 1) as isize) as usize;
                        let center = *center_row.add(px);
                        let neighbor = *neighbor_row_base.add(nx);
                        let cd = (neighbor as i16 - center as i16).unsigned_abs() as usize;
                        let w = spatial_w * *color_lut.get_unchecked(cd);
                        *sum_buf.get_unchecked_mut(px) += neighbor as f32 * w;
                        *wsum_buf.get_unchecked_mut(px) += w;
                    }
                }

                x += 16;
            }

            // Handle remaining pixels (width not divisible by 16) — branchless
            while x < width {
                let nx = (x as isize + dx_offset).clamp(0, (width - 1) as isize) as usize;
                let center = *center_row.add(x);
                let neighbor = *neighbor_row_base.add(nx);
                let cd = (neighbor as i16 - center as i16).unsigned_abs() as usize;
                let w = spatial_w * *color_lut.get_unchecked(cd);
                *sum_buf.get_unchecked_mut(x) += neighbor as f32 * w;
                *wsum_buf.get_unchecked_mut(x) += w;
                x += 1;
            }
        }
    }

    // Final normalization pass — write output row
    let dst_row = dst.add(row_base);
    let mut x = 0usize;
    while x + 4 <= width {
        let s = vld1q_f32(sum_buf.as_ptr().add(x));
        let ws = vld1q_f32(wsum_buf.as_ptr().add(x));

        // Check for zero weights
        let zero = vdupq_n_f32(0.0);
        let half_v = vdupq_n_f32(0.5);

        // Compute sum / wsum + 0.5, clamp to [0, 255]
        // Use reciprocal estimate + Newton-Raphson for fast divide
        let recip = vrecpeq_f32(ws);
        let recip = vmulq_f32(vrecpsq_f32(ws, recip), recip); // one NR step
        let result = vfmaq_f32(half_v, s, recip);

        // Clamp to 0..255 and convert to u32 then narrow to u8
        let clamped = vminq_f32(vmaxq_f32(result, zero), vdupq_n_f32(255.0));
        let u32v = vcvtq_u32_f32(clamped);

        // Narrow u32 -> u16 -> u8 and store 4 bytes
        let u16v = vmovn_u32(u32v);
        // We need to pair with something for vmovn_u16; just duplicate
        let u16v_paired = vcombine_u16(u16v, u16v);
        let u8v = vmovn_u16(u16v_paired);

        // Store 4 bytes (low 4 lanes of the 8-byte result)
        // Use scalar stores since we only have 4 valid pixels
        let ws0 = vgetq_lane_f32(ws, 0);
        let ws1 = vgetq_lane_f32(ws, 1);
        let ws2 = vgetq_lane_f32(ws, 2);
        let ws3 = vgetq_lane_f32(ws, 3);

        *dst_row.add(x) = if ws0 > 0.0 {
            vget_lane_u8(u8v, 0)
        } else {
            *center_row.add(x)
        };
        *dst_row.add(x + 1) = if ws1 > 0.0 {
            vget_lane_u8(u8v, 1)
        } else {
            *center_row.add(x + 1)
        };
        *dst_row.add(x + 2) = if ws2 > 0.0 {
            vget_lane_u8(u8v, 2)
        } else {
            *center_row.add(x + 2)
        };
        *dst_row.add(x + 3) = if ws3 > 0.0 {
            vget_lane_u8(u8v, 3)
        } else {
            *center_row.add(x + 3)
        };

        x += 4;
    }

    // Remaining pixels
    while x < width {
        let s = *sum_buf.get_unchecked(x);
        let ws = *wsum_buf.get_unchecked(x);
        *dst_row.add(x) = if ws > 0.0 {
            (s / ws + 0.5) as u8
        } else {
            *center_row.add(x)
        };
        x += 1;
    }
}
