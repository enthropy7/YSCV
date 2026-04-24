//! # Safety contract
//!
//! Unsafe code categories:
//! 1. **SIMD intrinsics (NEON / SSE)** — ISA guard via runtime detection or `#[target_feature]`.
//! 2. **`SendConstPtr` / `SendPtr` for rayon** — each chunk writes non-overlapping rows.
//! 3. **`get_unchecked` in FAST detection** — 3-pixel border margin ensures valid access.
//! 4. **Pointer arithmetic in distance transform** — offsets bounded by `h * w` allocation.

use rayon::prelude::*;
use yscv_tensor::Tensor;

use super::super::ImgProcError;
use super::super::shape::hwc_shape;

/// Detected corner/interest point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HarrisKeypoint {
    pub x: usize,
    pub y: usize,
    pub response: f32,
}

/// Harris corner detector on a single-channel `[H, W, 1]` image.
///
/// Returns corners whose Harris response exceeds `threshold`.
/// `k` is the Harris sensitivity parameter (typical 0.04-0.06).
#[allow(unsafe_code)]
pub fn harris_corners(
    input: &Tensor,
    block_size: usize,
    k: f32,
    threshold: f32,
) -> Result<Vec<HarrisKeypoint>, ImgProcError> {
    let (h, w, c) = hwc_shape(input)?;
    if c != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: c,
        });
    }
    let data = input.data();

    // Interleaved structure tensor products: [sxx, sxy, syy] per pixel
    let n = h * w;
    let mut prods = vec![0.0f32; n * 3]; // interleaved [sxx0,sxy0,syy0, sxx1,sxy1,syy1, ...]

    #[cfg(target_arch = "aarch64")]
    if !cfg!(miri) && std::arch::is_aarch64_feature_detected!("neon") {
        // SAFETY: ISA guard (feature detection) above.
        unsafe {
            sobel_products_interleaved_neon(data, &mut prods, h, w);
        }
    } else {
        sobel_products_interleaved_scalar(data, &mut prods, h, w);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if !cfg!(miri) && std::is_x86_feature_detected!("sse") {
            // SAFETY: ISA guard (feature detection) above.
            unsafe {
                sobel_products_interleaved_sse(data, &mut prods, h, w);
            }
        } else {
            sobel_products_interleaved_scalar(data, &mut prods, h, w);
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        sobel_products_interleaved_scalar(data, &mut prods, h, w);
    }

    let half = (block_size / 2) as i32;
    let r = half as usize;

    // Separable box filter on interleaved buffer.
    // Horizontal pass: sliding window on 3 channels at once
    let mut h_prods = vec![0.0f32; n * 3];
    for y in 0..h {
        let row3 = y * w * 3;
        let mut r0 = 0.0f32;
        let mut r1 = 0.0f32;
        let mut r2 = 0.0f32;
        for x in 0..r.min(w) {
            let i = row3 + x * 3;
            r0 += prods[i];
            r1 += prods[i + 1];
            r2 += prods[i + 2];
        }
        for x in 0..w {
            if x + r < w {
                let i = row3 + (x + r) * 3;
                r0 += prods[i];
                r1 += prods[i + 1];
                r2 += prods[i + 2];
            }
            let o = row3 + x * 3;
            h_prods[o] = r0;
            h_prods[o + 1] = r1;
            h_prods[o + 2] = r2;
            if x >= r {
                let i = row3 + (x - r) * 3;
                r0 -= prods[i];
                r1 -= prods[i + 1];
                r2 -= prods[i + 2];
            }
        }
    }

    // Vertical pass: interleaved column accumulators (w*3 elements)
    let stride3 = w * 3;
    let mut col = vec![0.0f32; stride3];

    for y in 0..r.min(h) {
        vec_add_row(&mut col, &h_prods[y * stride3..(y + 1) * stride3]);
    }

    // Reuse prods for the final box-filtered result
    for y in 0..h {
        if y + r < h {
            let bot = (y + r) * stride3;
            vec_add_row(&mut col, &h_prods[bot..bot + stride3]);
        }
        let row = y * stride3;
        prods[row..row + stride3].copy_from_slice(&col);
        if y >= r {
            let top = (y - r) * stride3;
            vec_sub_row(&mut col, &h_prods[top..top + stride3]);
        }
    }

    // prods now contains interleaved box-filtered [sxx,sxy,syy] per pixel.
    // Compute Harris response in parallel using rayon.
    let margin = r.max(1);
    let row_range: Vec<usize> = (margin..h.saturating_sub(margin)).collect();
    let corners: Vec<HarrisKeypoint> = row_range
        .par_iter()
        .flat_map(|&y| {
            let mut row_corners = Vec::new();
            for x in margin..w.saturating_sub(margin) {
                let i = (y * w + x) * 3;
                let a = prods[i]; // sxx
                let b = prods[i + 1]; // sxy
                let c = prods[i + 2]; // syy
                let det = a * c - b * b;
                let trace = a + c;
                let response = det - k * trace * trace;
                if response > threshold {
                    row_corners.push(HarrisKeypoint { x, y, response });
                }
            }
            row_corners
        })
        .collect();

    // Non-maximum suppression within 3x3 neighborhood
    let mut response_map = vec![0.0f32; h * w];
    for corner in &corners {
        response_map[corner.y * w + corner.x] = corner.response;
    }
    let corners: Vec<HarrisKeypoint> = corners
        .into_iter()
        .filter(|corner| {
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dy == 0 && dx == 0 {
                        continue;
                    }
                    let ny = (corner.y as i32 + dy) as usize;
                    let nx = (corner.x as i32 + dx) as usize;
                    if ny < h && nx < w && response_map[ny * w + nx] > corner.response {
                        return false;
                    }
                }
            }
            true
        })
        .collect();

    Ok(corners)
}

// ── SIMD row-vector helpers ────────────────────────────────────────

/// acc[i] += src[i] for all i, NEON-accelerated on aarch64.
#[allow(unsafe_code)]
#[inline]
fn vec_add_row(acc: &mut [f32], src: &[f32]) {
    let n = acc.len().min(src.len());
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    if !cfg!(miri) {
        // SAFETY: ISA guard (cfg aarch64 guarantees NEON); i+4 <= n bounds checked.
        unsafe {
            use std::arch::aarch64::*;
            let ap = acc.as_mut_ptr();
            let sp = src.as_ptr();
            while i + 4 <= n {
                let a = vld1q_f32(ap.add(i));
                let b = vld1q_f32(sp.add(i));
                vst1q_f32(ap.add(i), vaddq_f32(a, b));
                i += 4;
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if !cfg!(miri) {
        // SAFETY: ISA guard (feature detection) inside; i+4 <= n bounds checked.
        unsafe {
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            if std::is_x86_feature_detected!("sse") {
                let ap = acc.as_mut_ptr();
                let sp = src.as_ptr();
                while i + 4 <= n {
                    let a = _mm_loadu_ps(ap.add(i));
                    let b = _mm_loadu_ps(sp.add(i));
                    _mm_storeu_ps(ap.add(i), _mm_add_ps(a, b));
                    i += 4;
                }
            }
        }
    }

    while i < n {
        acc[i] += src[i];
        i += 1;
    }
}

/// acc[i] -= src[i] for all i, NEON-accelerated on aarch64.
#[allow(unsafe_code)]
#[inline]
fn vec_sub_row(acc: &mut [f32], src: &[f32]) {
    let n = acc.len().min(src.len());
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    if !cfg!(miri) {
        // SAFETY: ISA guard (cfg aarch64 guarantees NEON); i+4 <= n bounds checked.
        unsafe {
            use std::arch::aarch64::*;
            let ap = acc.as_mut_ptr();
            let sp = src.as_ptr();
            while i + 4 <= n {
                let a = vld1q_f32(ap.add(i));
                let b = vld1q_f32(sp.add(i));
                vst1q_f32(ap.add(i), vsubq_f32(a, b));
                i += 4;
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if !cfg!(miri) {
        // SAFETY: ISA guard (feature detection) inside; i+4 <= n bounds checked.
        unsafe {
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            if std::is_x86_feature_detected!("sse") {
                let ap = acc.as_mut_ptr();
                let sp = src.as_ptr();
                while i + 4 <= n {
                    let a = _mm_loadu_ps(ap.add(i));
                    let b = _mm_loadu_ps(sp.add(i));
                    _mm_storeu_ps(ap.add(i), _mm_sub_ps(a, b));
                    i += 4;
                }
            }
        }
    }

    while i < n {
        acc[i] -= src[i];
        i += 1;
    }
}

// ── Sobel gradient helpers for Harris ──────────────────────────────

/// Scalar fused Sobel gradient + interleaved product computation.
/// Stores [Ix², Ix*Iy, Iy²] interleaved at each pixel position.
#[inline]
fn sobel_products_interleaved_scalar(data: &[f32], prods: &mut [f32], h: usize, w: usize) {
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let p =
                |dy: i32, dx: i32| data[(y as i32 + dy) as usize * w + (x as i32 + dx) as usize];
            let gx = -p(-1, -1) + p(-1, 1) - 2.0 * p(0, -1) + 2.0 * p(0, 1) - p(1, -1) + p(1, 1);
            let gy = -p(-1, -1) - 2.0 * p(-1, 0) - p(-1, 1) + p(1, -1) + 2.0 * p(1, 0) + p(1, 1);
            let idx = (y * w + x) * 3;
            prods[idx] = gx * gx;
            prods[idx + 1] = gx * gy;
            prods[idx + 2] = gy * gy;
        }
    }
}

/// NEON SIMD fused Sobel gradient + interleaved product computation.
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn sobel_products_interleaved_neon(data: &[f32], prods: &mut [f32], h: usize, w: usize) {
    use std::arch::aarch64::*;

    let two = vdupq_n_f32(2.0);

    for y in 1..h - 1 {
        let row_above = (y - 1) * w;
        let row_curr = y * w;
        let row_below = (y + 1) * w;

        let dp = data.as_ptr();
        let pp = prods.as_mut_ptr();

        let mut x = 1usize;

        while x + 5 <= w {
            let r0l = vld1q_f32(dp.add(row_above + x - 1));
            let r0c = vld1q_f32(dp.add(row_above + x));
            let r0r = vld1q_f32(dp.add(row_above + x + 1));
            let r1l = vld1q_f32(dp.add(row_curr + x - 1));
            let r1r = vld1q_f32(dp.add(row_curr + x + 1));
            let r2l = vld1q_f32(dp.add(row_below + x - 1));
            let r2c = vld1q_f32(dp.add(row_below + x));
            let r2r = vld1q_f32(dp.add(row_below + x + 1));

            let gx = vaddq_f32(
                vaddq_f32(vsubq_f32(r0r, r0l), vsubq_f32(r2r, r2l)),
                vmulq_f32(vsubq_f32(r1r, r1l), two),
            );
            let gy = vaddq_f32(
                vaddq_f32(vsubq_f32(r2l, r0l), vsubq_f32(r2r, r0r)),
                vmulq_f32(vsubq_f32(r2c, r0c), two),
            );

            // Interleaved store: [sxx0,sxy0,syy0, sxx1,sxy1,syy1, sxx2,sxy2,syy2, sxx3,sxy3,syy3]
            let gxx = vmulq_f32(gx, gx);
            let gxy = vmulq_f32(gx, gy);
            let gyy = vmulq_f32(gy, gy);
            vst3q_f32(pp.add((row_curr + x) * 3), float32x4x3_t(gxx, gxy, gyy));

            x += 4;
        }

        // Scalar tail
        while x < w - 1 {
            let p = |dy: i32, dx: i32| {
                *dp.add(((y as i32 + dy) as usize) * w + ((x as i32 + dx) as usize))
            };
            let gx = -p(-1, -1) + p(-1, 1) - 2.0 * p(0, -1) + 2.0 * p(0, 1) - p(1, -1) + p(1, 1);
            let gy = -p(-1, -1) - 2.0 * p(-1, 0) - p(-1, 1) + p(1, -1) + 2.0 * p(1, 0) + p(1, 1);
            let idx = (row_curr + x) * 3;
            *pp.add(idx) = gx * gx;
            *pp.add(idx + 1) = gx * gy;
            *pp.add(idx + 2) = gy * gy;
            x += 1;
        }
    }
}

/// SSE SIMD fused Sobel gradient + interleaved product computation.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn sobel_products_interleaved_sse(data: &[f32], prods: &mut [f32], h: usize, w: usize) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let two = _mm_set1_ps(2.0);

    for y in 1..h - 1 {
        let row_above = (y - 1) * w;
        let row_curr = y * w;
        let row_below = (y + 1) * w;

        let dp = data.as_ptr();
        let pp = prods.as_mut_ptr();

        let mut x = 1usize;

        while x + 5 <= w {
            let r0l = _mm_loadu_ps(dp.add(row_above + x - 1));
            let r0c = _mm_loadu_ps(dp.add(row_above + x));
            let r0r = _mm_loadu_ps(dp.add(row_above + x + 1));
            let r1l = _mm_loadu_ps(dp.add(row_curr + x - 1));
            let r1r = _mm_loadu_ps(dp.add(row_curr + x + 1));
            let r2l = _mm_loadu_ps(dp.add(row_below + x - 1));
            let r2c = _mm_loadu_ps(dp.add(row_below + x));
            let r2r = _mm_loadu_ps(dp.add(row_below + x + 1));

            // gx = (r0r - r0l) + (r2r - r2l) + 2*(r1r - r1l)
            let gx = _mm_add_ps(
                _mm_add_ps(_mm_sub_ps(r0r, r0l), _mm_sub_ps(r2r, r2l)),
                _mm_mul_ps(_mm_sub_ps(r1r, r1l), two),
            );
            // gy = (r2l - r0l) + (r2r - r0r) + 2*(r2c - r0c)
            let gy = _mm_add_ps(
                _mm_add_ps(_mm_sub_ps(r2l, r0l), _mm_sub_ps(r2r, r0r)),
                _mm_mul_ps(_mm_sub_ps(r2c, r0c), two),
            );

            let gxx = _mm_mul_ps(gx, gx);
            let gxy = _mm_mul_ps(gx, gy);
            let gyy = _mm_mul_ps(gy, gy);

            // SSE has no vst3q equivalent, so store interleaved via scalar extracts
            // Extract 4 lanes and store interleaved [sxx, sxy, syy] per pixel
            let mut gxx_arr = [0.0f32; 4];
            let mut gxy_arr = [0.0f32; 4];
            let mut gyy_arr = [0.0f32; 4];
            _mm_storeu_ps(gxx_arr.as_mut_ptr(), gxx);
            _mm_storeu_ps(gxy_arr.as_mut_ptr(), gxy);
            _mm_storeu_ps(gyy_arr.as_mut_ptr(), gyy);

            for k in 0..4 {
                let idx = (row_curr + x + k) * 3;
                *pp.add(idx) = gxx_arr[k];
                *pp.add(idx + 1) = gxy_arr[k];
                *pp.add(idx + 2) = gyy_arr[k];
            }

            x += 4;
        }

        // Scalar tail
        while x < w - 1 {
            let p = |dy: i32, dx: i32| {
                *dp.add(((y as i32 + dy) as usize) * w + ((x as i32 + dx) as usize))
            };
            let gx = -p(-1, -1) + p(-1, 1) - 2.0 * p(0, -1) + 2.0 * p(0, 1) - p(1, -1) + p(1, 1);
            let gy = -p(-1, -1) - 2.0 * p(-1, 0) - p(-1, 1) + p(1, -1) + 2.0 * p(1, 0) + p(1, 1);
            let idx = (row_curr + x) * 3;
            *pp.add(idx) = gx * gx;
            *pp.add(idx + 1) = gx * gy;
            *pp.add(idx + 2) = gy * gy;
            x += 1;
        }
    }
}

// ── FAST corner detection ──────────────────────────────────────────

/// FAST-9 corner detector on a single-channel `[H, W, 1]` image.
///
/// Tests 16 pixels on a Bresenham circle of radius 3. A pixel is a corner
/// if at least `min_consecutive` contiguous circle pixels are all brighter
/// (or all darker) than the center by at least `threshold`.
///
/// Uses precomputed offsets and bitmask-based contiguity checking for speed.
#[allow(unsafe_code)]
pub fn fast_corners(
    input: &Tensor,
    threshold: f32,
    min_consecutive: usize,
) -> Result<Vec<HarrisKeypoint>, ImgProcError> {
    let (h, w, c) = hwc_shape(input)?;
    if c != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: c,
        });
    }
    let data = input.data();

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

    // Precompute flat offsets from center pixel
    let ws = w as isize;
    let mut offsets = [0isize; 16];
    for (i, &(dx, dy)) in CIRCLE.iter().enumerate() {
        offsets[i] = dy as isize * ws + dx as isize;
    }

    // Cardinal offsets for early rejection: N(0), E(4), S(8), W(12)
    let card = [offsets[0], offsets[4], offsets[8], offsets[12]];

    let n = min_consecutive.min(16);
    let mut corners = Vec::new();

    for y in 3..h.saturating_sub(3) {
        let row_base = y * w;
        for x in 3..w.saturating_sub(3) {
            let idx = row_base + x;
            // SAFETY: bounds checked by loop range [3, w-3) and [3, h-3).
            let center = unsafe { *data.get_unchecked(idx) };
            let bright_thresh = center + threshold;
            let dark_thresh = center - threshold;

            // Quick reject: check cardinal pixels N, E, S, W
            let mut bc = 0u32;
            let mut dc = 0u32;
            for &co in &card {
                // SAFETY: bounds checked by 3-pixel border; cardinal offsets within radius 3.
                let v = unsafe { *data.get_unchecked((idx as isize + co) as usize) };
                bc += (v > bright_thresh) as u32;
                dc += (v < dark_thresh) as u32;
            }
            // Need at least 3 of 4 cardinals passing for any 9-run to exist
            let min_card = if n >= 9 { 3 } else { n.min(4) };
            if (bc as usize) < min_card && (dc as usize) < min_card {
                continue;
            }

            // Build bitmasks
            let mut bright_mask = 0u32;
            let mut dark_mask = 0u32;
            for i in 0..16 {
                // SAFETY: bounds checked by 3-pixel border; circle offsets within radius 3.
                let v = unsafe { *data.get_unchecked((idx as isize + offsets[i]) as usize) };
                if v > bright_thresh {
                    bright_mask |= 1 << i;
                }
                if v < dark_thresh {
                    dark_mask |= 1 << i;
                }
            }

            let is_corner =
                has_consecutive_mask(bright_mask, n) || has_consecutive_mask(dark_mask, n);
            if is_corner {
                // Compute score: sum of absolute differences
                let mut score = 0.0f32;
                for i in 0..16 {
                    // SAFETY: bounds checked by 3-pixel border; circle offsets within radius 3.
                    let v = unsafe { *data.get_unchecked((idx as isize + offsets[i]) as usize) };
                    score += (v - center).abs();
                }
                corners.push(HarrisKeypoint {
                    x,
                    y,
                    response: score,
                });
            }
        }
    }

    Ok(corners)
}

/// Check if a 16-bit circular bitmask has `n` contiguous set bits.
pub(crate) fn has_consecutive_mask(mask: u32, n: usize) -> bool {
    if n == 0 {
        return true;
    }
    if mask == 0 {
        return false;
    }
    let doubled = mask | (mask << 16);
    let mut run = 0u32;
    for i in 0..32 {
        if (doubled >> i) & 1 != 0 {
            run += 1;
            if run >= n as u32 {
                return true;
            }
        } else {
            run = 0;
        }
    }
    false
}

#[allow(dead_code)]
pub(crate) fn has_consecutive(flags: &[bool; 16], n: usize) -> bool {
    if n == 0 {
        return true;
    }
    let mut count = 0usize;
    for i in 0..32 {
        if flags[i % 16] {
            count += 1;
            if count >= n {
                return true;
            }
        } else {
            count = 0;
        }
    }
    false
}

// ── Hough line detection ───────────────────────────────────────────

/// Detected line in Hough parameter space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HoughLine {
    pub rho: f32,
    pub theta: f32,
    pub votes: u32,
}

/// Standard Hough line transform on a binary/edge single-channel `[H, W, 1]` image.
///
/// `rho_resolution` is the distance resolution in pixels (typically 1.0).
/// `theta_resolution` is the angular resolution in radians (typically pi/180).
/// `vote_threshold` is the minimum accumulator count for a line.
pub fn hough_lines(
    input: &Tensor,
    rho_resolution: f32,
    theta_resolution: f32,
    vote_threshold: u32,
) -> Result<Vec<HoughLine>, ImgProcError> {
    let (h, w, c) = hwc_shape(input)?;
    if c != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: c,
        });
    }
    let data = input.data();
    let diag = ((h * h + w * w) as f32).sqrt();
    let max_rho = diag;
    let num_rho = (2.0 * max_rho / rho_resolution) as usize + 1;
    let num_theta = (std::f32::consts::PI / theta_resolution) as usize;

    let mut accumulator = vec![0u32; num_rho * num_theta];

    let sins: Vec<f32> = (0..num_theta)
        .map(|t| (t as f32 * theta_resolution).sin())
        .collect();
    let coss: Vec<f32> = (0..num_theta)
        .map(|t| (t as f32 * theta_resolution).cos())
        .collect();

    for y in 0..h {
        for x in 0..w {
            if data[y * w + x] > 0.5 {
                for t in 0..num_theta {
                    let rho = x as f32 * coss[t] + y as f32 * sins[t];
                    let rho_idx = ((rho + max_rho) / rho_resolution) as usize;
                    if rho_idx < num_rho {
                        accumulator[rho_idx * num_theta + t] += 1;
                    }
                }
            }
        }
    }

    let mut lines = Vec::new();
    for rho_idx in 0..num_rho {
        for t in 0..num_theta {
            let votes = accumulator[rho_idx * num_theta + t];
            if votes >= vote_threshold {
                let rho = rho_idx as f32 * rho_resolution - max_rho;
                let theta = t as f32 * theta_resolution;
                lines.push(HoughLine { rho, theta, votes });
            }
        }
    }

    lines.sort_by_key(|line| std::cmp::Reverse(line.votes));
    Ok(lines)
}

// ── Image pyramid ──────────────────────────────────────────────────

/// Builds a Gaussian image pyramid by repeated 2x downsampling.
///
/// Input is `[H, W, C]`. Returns a vector of progressively smaller images.
/// Level 0 is the original image.
pub fn gaussian_pyramid(input: &Tensor, levels: usize) -> Result<Vec<Tensor>, ImgProcError> {
    let mut pyramid = Vec::with_capacity(levels + 1);
    pyramid.push(input.clone());

    for _ in 0..levels {
        // SAFETY: pyramid always has at least one element (input.clone() pushed above).
        let prev = pyramid.last().expect("pyramid must be non-empty");
        let (ph, pw, pc) = hwc_shape(prev)?;
        if ph < 2 || pw < 2 {
            break;
        }
        let nh = ph / 2;
        let nw = pw / 2;
        let prev_data = prev.data();
        let mut down = vec![0.0f32; nh * nw * pc];
        for y in 0..nh {
            for x in 0..nw {
                let sy = y * 2;
                let sx = x * 2;
                for c in 0..pc {
                    let v00 = prev_data[(sy * pw + sx) * pc + c];
                    let v01 = prev_data[(sy * pw + (sx + 1).min(pw - 1)) * pc + c];
                    let v10 = prev_data[((sy + 1).min(ph - 1) * pw + sx) * pc + c];
                    let v11 =
                        prev_data[((sy + 1).min(ph - 1) * pw + (sx + 1).min(pw - 1)) * pc + c];
                    down[(y * nw + x) * pc + c] = (v00 + v01 + v10 + v11) * 0.25;
                }
            }
        }
        let t = Tensor::from_vec(vec![nh, nw, pc], down)?;
        pyramid.push(t);
    }

    Ok(pyramid)
}

// ── Distance transform ─────────────────────────────────────────────

/// L1 distance transform on a binary single-channel `[H, W, 1]` image.
///
/// Input pixels > 0.5 are foreground. Output is the distance from each
/// foreground pixel to the nearest background pixel (two-pass L1).
///
/// Uses SIMD (NEON/SSE) to accelerate the vertical propagation step,
/// which processes 4 f32 values at a time.
#[allow(unsafe_code)]
/// L1 Manhattan distance transform. Two interleaved passes (like OpenCV):
/// each pass handles both vertical (SIMD) and horizontal (scalar) in one sweep.
/// Conditional stores skip writes when value unchanged (huge win on sparse inputs).
pub fn distance_transform(input: &Tensor) -> Result<Tensor, ImgProcError> {
    let (h, w, c) = hwc_shape(input)?;
    if c != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: c,
        });
    }
    let data = input.data();
    let inf = (h + w) as f32;

    // Fast path: if no foreground pixels, distance is 0 everywhere.
    let has_foreground = data.iter().any(|&v| v > 0.5);
    if !has_foreground {
        return Tensor::from_vec(vec![h, w, 1], vec![0.0f32; h * w]).map_err(Into::into);
    }

    let mut dist = vec![0.0f32; h * w];
    for i in 0..h * w {
        if data[i] > 0.5 {
            // SAFETY: i < h*w = dist.len().
            unsafe {
                *dist.as_mut_ptr().add(i) = inf;
            }
        }
    }

    // === Forward pass: top→bottom, vertical SIMD + horizontal register-scan ===
    // Key optimization: keep running min in a REGISTER, not memory.
    // Eliminates store→load forwarding latency (4→1 cycle per element).
    {
        let p = dist.as_mut_ptr();
        // SAFETY: pointer offsets 0..w within dist allocation.
        unsafe {
            let mut run = *p; // register
            for x in 1..w {
                run += 1.0;
                let cur = *p.add(x);
                if run < cur {
                    *p.add(x) = run;
                } else {
                    run = cur;
                }
            }
        }
    }
    for y in 1..h {
        dt_vertical_min_forward(&mut dist, (y - 1) * w, y * w, w);
        // SAFETY: pointer offset y*w within dist allocation (y < h).
        let p = unsafe { dist.as_mut_ptr().add(y * w) };
        // SAFETY: pointer offsets 0..w within row starting at p.
        unsafe {
            let mut run = *p;
            for x in 1..w {
                run += 1.0;
                let cur = *p.add(x);
                if run < cur {
                    *p.add(x) = run;
                } else {
                    run = cur;
                }
            }
        }
    }

    // === Backward pass: bottom→top, register-scan R→L ===
    {
        // SAFETY: pointer offset (h-1)*w within dist allocation.
        let p = unsafe { dist.as_mut_ptr().add((h - 1) * w) };
        // SAFETY: pointer offsets 0..w within row starting at p.
        unsafe {
            let mut run = *p.add(w - 1);
            for x in (0..w.saturating_sub(1)).rev() {
                run += 1.0;
                let cur = *p.add(x);
                if run < cur {
                    *p.add(x) = run;
                } else {
                    run = cur;
                }
            }
        }
    }
    for y in (0..h.saturating_sub(1)).rev() {
        dt_vertical_min_forward(&mut dist, (y + 1) * w, y * w, w);
        // SAFETY: pointer offset y*w within dist allocation (y < h).
        let p = unsafe { dist.as_mut_ptr().add(y * w) };
        // SAFETY: pointer offsets 0..w within row starting at p.
        unsafe {
            let mut run = *p.add(w - 1);
            for x in (0..w.saturating_sub(1)).rev() {
                run += 1.0;
                let cur = *p.add(x);
                if run < cur {
                    *p.add(x) = run;
                } else {
                    run = cur;
                }
            }
        }
    }

    Tensor::from_vec(vec![h, w, 1], dist).map_err(Into::into)
}

/// SIMD-accelerated vertical min step: cur[x] = min(cur[x], src[x] + 1) for all x.
#[allow(unsafe_code)]
fn dt_vertical_min_forward(dist: &mut [f32], src_start: usize, cur_start: usize, w: usize) {
    let mut x = 0usize;

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: ISA guard (feature detection) above.
            x = unsafe { dt_vertical_neon(dist, src_start, cur_start, w) };
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("sse") {
            // SAFETY: ISA guard (feature detection) above.
            x = unsafe { dt_vertical_sse(dist, src_start, cur_start, w) };
        }
    }

    // Scalar tail
    while x < w {
        let src_val = dist[src_start + x] + 1.0;
        if src_val < dist[cur_start + x] {
            dist[cur_start + x] = src_val;
        }
        x += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn dt_vertical_neon(
    dist: &mut [f32],
    src_start: usize,
    cur_start: usize,
    w: usize,
) -> usize {
    use std::arch::aarch64::*;
    let one = vdupq_n_f32(1.0);
    let sp = dist.as_ptr().add(src_start);
    let cp = dist.as_mut_ptr().add(cur_start);
    let mut x = 0usize;
    // 4× unrolled: 16 elements per iteration
    while x + 16 <= w {
        let s0 = vaddq_f32(vld1q_f32(sp.add(x)), one);
        let s1 = vaddq_f32(vld1q_f32(sp.add(x + 4)), one);
        let s2 = vaddq_f32(vld1q_f32(sp.add(x + 8)), one);
        let s3 = vaddq_f32(vld1q_f32(sp.add(x + 12)), one);
        vst1q_f32(cp.add(x), vminq_f32(vld1q_f32(cp.add(x)), s0));
        vst1q_f32(cp.add(x + 4), vminq_f32(vld1q_f32(cp.add(x + 4)), s1));
        vst1q_f32(cp.add(x + 8), vminq_f32(vld1q_f32(cp.add(x + 8)), s2));
        vst1q_f32(cp.add(x + 12), vminq_f32(vld1q_f32(cp.add(x + 12)), s3));
        x += 16;
    }
    while x + 4 <= w {
        let s = vaddq_f32(vld1q_f32(sp.add(x)), one);
        vst1q_f32(cp.add(x), vminq_f32(vld1q_f32(cp.add(x)), s));
        x += 4;
    }
    x
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn dt_vertical_sse(dist: &mut [f32], src_start: usize, cur_start: usize, w: usize) -> usize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let one = _mm_set1_ps(1.0);
    let ptr = dist.as_mut_ptr();
    let mut x = 0usize;
    while x + 4 <= w {
        let src = _mm_loadu_ps(ptr.add(src_start + x));
        let cur = _mm_loadu_ps(ptr.add(cur_start + x));
        let src_plus_one = _mm_add_ps(src, one);
        let result = _mm_min_ps(cur, src_plus_one);
        _mm_storeu_ps(ptr.add(cur_start + x), result);
        x += 4;
    }
    x
}

// ── ORB feature descriptors ────────────────────────────────────────

/// ORB descriptor: 256-bit binary descriptor stored as 32 bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrbDescriptor {
    pub keypoint: (usize, usize),
    pub bits: [u8; 32],
}

const ORB_PATTERN_LEN: usize = 256;

fn orb_pattern() -> Vec<(i32, i32, i32, i32)> {
    let mut pattern = Vec::with_capacity(ORB_PATTERN_LEN);
    let mut seed: u32 = 0x1234_5678;
    for _ in 0..ORB_PATTERN_LEN {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        let a_x = ((seed % 31) as i32) - 15;
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        let a_y = ((seed % 31) as i32) - 15;
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        let b_x = ((seed % 31) as i32) - 15;
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        let b_y = ((seed % 31) as i32) - 15;
        pattern.push((a_x, a_y, b_x, b_y));
    }
    pattern
}

/// Computes ORB descriptors for keypoints on a grayscale `[H,W,1]` image.
pub fn orb_descriptors(
    image: &Tensor,
    keypoints: &[(usize, usize)],
    patch_radius: usize,
) -> Result<Vec<OrbDescriptor>, ImgProcError> {
    let (h, w, c) = hwc_shape(image)?;
    if c != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: c,
        });
    }
    let data = image.data();
    let pattern = orb_pattern();
    let mut descriptors = Vec::new();
    for &(kx, ky) in keypoints {
        if kx < patch_radius
            || ky < patch_radius
            || kx + patch_radius >= w
            || ky + patch_radius >= h
        {
            continue;
        }
        let mut bits = [0u8; 32];
        for (i, &(ax, ay, bx, by)) in pattern.iter().enumerate() {
            let pa = data[(ky as i32 + ay) as usize * w + (kx as i32 + ax) as usize];
            let pb = data[(ky as i32 + by) as usize * w + (kx as i32 + bx) as usize];
            if pa < pb {
                bits[i / 8] |= 1 << (i % 8);
            }
        }
        descriptors.push(OrbDescriptor {
            keypoint: (kx, ky),
            bits,
        });
    }
    Ok(descriptors)
}

/// Hamming distance between two ORB descriptors.
pub fn orb_hamming_distance(a: &OrbDescriptor, b: &OrbDescriptor) -> u32 {
    a.bits
        .iter()
        .zip(b.bits.iter())
        .map(|(&x, &y)| (x ^ y).count_ones())
        .sum()
}

/// Brute-force ORB descriptor matching. Returns `(idx_a, idx_b, distance)`.
pub fn orb_match(
    desc_a: &[OrbDescriptor],
    desc_b: &[OrbDescriptor],
    max_distance: u32,
) -> Vec<(usize, usize, u32)> {
    let mut matches = Vec::new();
    for (i, da) in desc_a.iter().enumerate() {
        let mut best_j = 0;
        let mut best_dist = u32::MAX;
        for (j, db) in desc_b.iter().enumerate() {
            let d = orb_hamming_distance(da, db);
            if d < best_dist {
                best_dist = d;
                best_j = j;
            }
        }
        if best_dist <= max_distance {
            matches.push((i, best_j, best_dist));
        }
    }
    matches
}

// ── Shi-Tomasi good features to track ──────────────────────────────

/// Shi-Tomasi corner detection (minimum eigenvalue approach).
///
/// Detects up to `max_corners` strong corners on a grayscale `[H, W, 1]` image.
/// `quality_level` (0..1) sets the threshold relative to the strongest corner response.
/// `min_distance` enforces minimum Euclidean distance between returned corners via
/// greedy non-maximum suppression.
///
/// Returns a vector of `(row, col)` coordinates.
pub fn good_features_to_track(
    img: &Tensor,
    max_corners: usize,
    quality_level: f32,
    min_distance: f32,
) -> Result<Vec<(usize, usize)>, ImgProcError> {
    let (h, w, c) = hwc_shape(img)?;
    if c != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: c,
        });
    }
    let data = img.data();

    // Compute gradients using [-1, 0, 1] kernel
    let mut ix = vec![0.0f32; h * w];
    let mut iy = vec![0.0f32; h * w];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            ix[y * w + x] = data[y * w + x + 1] - data[y * w + x - 1];
            iy[y * w + x] = data[(y + 1) * w + x] - data[(y - 1) * w + x];
        }
    }

    // Compute structure tensor elements with 3x3 window sum, then min eigenvalue
    let mut min_eig = vec![0.0f32; h * w];
    let mut max_eig_val: f32 = 0.0;

    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let (mut sxx, mut sxy, mut syy) = (0.0f32, 0.0f32, 0.0f32);
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let py = (y as i32 + dy) as usize;
                    let px = (x as i32 + dx) as usize;
                    let gx = ix[py * w + px];
                    let gy = iy[py * w + px];
                    sxx += gx * gx;
                    sxy += gx * gy;
                    syy += gy * gy;
                }
            }
            // Min eigenvalue of 2x2 matrix [[sxx, sxy], [sxy, syy]]
            let trace = sxx + syy;
            let det = sxx * syy - sxy * sxy;
            let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();
            let lambda_min = (trace - disc) * 0.5;
            min_eig[y * w + x] = lambda_min;
            if lambda_min > max_eig_val {
                max_eig_val = lambda_min;
            }
        }
    }

    // Threshold
    let thresh = quality_level * max_eig_val;

    // Collect candidates above threshold
    let mut candidates: Vec<(usize, usize, f32)> = Vec::new();
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let e = min_eig[y * w + x];
            if e > thresh {
                candidates.push((y, x, e));
            }
        }
    }

    // Sort by strength (strongest first)
    candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    // Greedy non-maximum suppression by min_distance
    let min_dist_sq = min_distance * min_distance;
    let mut selected: Vec<(usize, usize)> = Vec::new();
    for (r, c_col, _) in &candidates {
        if selected.len() >= max_corners {
            break;
        }
        let too_close = selected.iter().any(|&(sr, sc)| {
            let dr = *r as f32 - sr as f32;
            let dc = *c_col as f32 - sc as f32;
            dr * dr + dc * dc < min_dist_sq
        });
        if !too_close {
            selected.push((*r, *c_col));
        }
    }

    Ok(selected)
}

// ── Sub-pixel corner refinement ────────────────────────────────────

/// Refines corner locations to sub-pixel accuracy using gradient-based method.
///
/// For each corner in `corners` (given as `(row, col)`), a window of `±win_size`
/// is used to solve a 2x2 linear system for the sub-pixel offset.
///
/// Returns refined `(row, col)` as `f32` coordinates.
pub fn corner_sub_pix(
    img: &Tensor,
    corners: &[(usize, usize)],
    win_size: usize,
) -> Result<Vec<(f32, f32)>, ImgProcError> {
    let (h, w, c) = hwc_shape(img)?;
    if c != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: c,
        });
    }
    let data = img.data();

    let mut refined = Vec::with_capacity(corners.len());

    for &(row, col) in corners {
        // Check if window fits inside the image (need 1 extra for gradient)
        if row < win_size + 1
            || row + win_size + 1 >= h
            || col < win_size + 1
            || col + win_size + 1 >= w
        {
            // Can't refine, return original position
            refined.push((row as f32, col as f32));
            continue;
        }

        // Build the normal equations for sub-pixel shift:
        // For each pixel in the window, compute gradient g = (gx, gy)
        // and the dot product g . (p - corner). We solve:
        //   A * delta = b
        // where A = sum(g * g^T), b = sum(g * g^T * (p - corner))
        // This is equivalent to minimizing the squared gradient projection error.
        let mut a00 = 0.0f32; // sum gx*gx
        let mut a01 = 0.0f32; // sum gx*gy
        let mut a11 = 0.0f32; // sum gy*gy
        let mut b0 = 0.0f32; // sum gx*gx*(px-col) + gx*gy*(py-row)
        let mut b1 = 0.0f32; // sum gx*gy*(px-col) + gy*gy*(py-row)

        let ws = win_size as i32;
        for dy in -ws..=ws {
            for dx in -ws..=ws {
                let py = (row as i32 + dy) as usize;
                let px = (col as i32 + dx) as usize;

                let gx = data[py * w + px + 1] - data[py * w + px - 1];
                let gy = data[(py + 1) * w + px] - data[(py - 1) * w + px];

                a00 += gx * gx;
                a01 += gx * gy;
                a11 += gy * gy;

                let dxf = dx as f32;
                let dyf = dy as f32;
                b0 += gx * gx * dxf + gx * gy * dyf;
                b1 += gx * gy * dxf + gy * gy * dyf;
            }
        }

        // Solve 2x2 system: [[a00, a01], [a01, a11]] * [dcol, drow] = [b0, b1]
        let det = a00 * a11 - a01 * a01;
        if det.abs() < 1e-10 {
            // Singular, keep original
            refined.push((row as f32, col as f32));
        } else {
            let inv_det = 1.0 / det;
            let dcol = (a11 * b0 - a01 * b1) * inv_det;
            let drow = (-a01 * b0 + a00 * b1) * inv_det;

            // Clamp offset to ±win_size to avoid wild jumps
            let dcol = dcol.clamp(-(win_size as f32), win_size as f32);
            let drow = drow.clamp(-(win_size as f32), win_size as f32);

            refined.push((row as f32 + drow, col as f32 + dcol));
        }
    }

    Ok(refined)
}

// ── Gradient orientation ───────────────────────────────────────────

/// Computes image gradients and orientation for descriptor construction.
/// Returns a gradient magnitude + orientation pair per pixel.
pub fn compute_gradient_orientation(image: &Tensor) -> Result<(Tensor, Tensor), ImgProcError> {
    let (h, w, c) = hwc_shape(image)?;
    if c != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: c,
        });
    }
    let data = image.data();
    let mut mag = vec![0.0f32; h * w];
    let mut ori = vec![0.0f32; h * w];

    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let gx = data[y * w + x + 1] - data[y * w + x - 1];
            let gy = data[(y + 1) * w + x] - data[(y - 1) * w + x];
            mag[y * w + x] = (gx * gx + gy * gy).sqrt();
            ori[y * w + x] = gy.atan2(gx);
        }
    }

    let mag_t = Tensor::from_vec(vec![h, w, 1], mag)?;
    let ori_t = Tensor::from_vec(vec![h, w, 1], ori)?;
    Ok((mag_t, ori_t))
}

/// Histogram of Oriented Gradients (HOG) descriptor for a single cell.
///
/// `cell` is a grayscale patch `[cell_h, cell_w, 1]`, returns 9-bin orientation histogram.
pub fn hog_cell_descriptor(cell: &Tensor) -> Result<Vec<f32>, ImgProcError> {
    let (ch, cw, cc) = hwc_shape(cell)?;
    if cc != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: cc,
        });
    }
    let data = cell.data();
    let mut bins = [0.0f32; 9];
    let bin_width = std::f32::consts::PI / 9.0;

    for y in 1..ch.saturating_sub(1) {
        for x in 1..cw.saturating_sub(1) {
            let gx = data[y * cw + x + 1] - data[y * cw + x - 1];
            let gy = data[(y + 1) * cw + x] - data[(y - 1) * cw + x];
            let mag = (gx * gx + gy * gy).sqrt();
            let angle = gy.atan2(gx).rem_euclid(std::f32::consts::PI);
            let bin = ((angle / bin_width) as usize).min(8);
            bins[bin] += mag;
        }
    }

    // L2-normalize
    let norm = bins.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
    Ok(bins.iter().map(|v| v / norm).collect())
}

// ── SIFT-like descriptor ───────────────────────────────────────────

/// SIFT-like 128-element descriptor for a keypoint on a grayscale `[H,W,1]` image.
///
/// Computes a 4x4 grid of 8-bin orientation histograms in a 16x16 patch around each keypoint.
/// Returns 128-float L2-normalized descriptor per keypoint.
/// Keypoints too close to border are skipped (returned as all zeros).
pub fn sift_descriptor(
    image: &Tensor,
    keypoints: &[(usize, usize)],
) -> Result<Vec<[f32; 128]>, ImgProcError> {
    let (h, w, c) = hwc_shape(image)?;
    if c != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: c,
        });
    }
    let data = image.data();
    let half_patch = 8; // 16x16 patch, radius 8

    let mut descriptors = Vec::with_capacity(keypoints.len());

    for &(ky, kx) in keypoints {
        let mut desc = [0.0f32; 128];

        if ky < half_patch || ky + half_patch >= h || kx < half_patch || kx + half_patch >= w {
            descriptors.push(desc);
            continue;
        }

        // 4x4 grid of 4x4 cells, each producing 8-bin histogram
        for gy in 0..4 {
            for gx in 0..4 {
                let cell_y = (ky - half_patch) + gy * 4;
                let cell_x = (kx - half_patch) + gx * 4;
                let bin_offset = (gy * 4 + gx) * 8;

                for cy in 0..4 {
                    for cx in 0..4 {
                        let py = cell_y + cy;
                        let px = cell_x + cx;
                        if py < 1 || py + 1 >= h || px < 1 || px + 1 >= w {
                            continue;
                        }
                        let gx_val = data[py * w + px + 1] - data[py * w + px - 1];
                        let gy_val = data[(py + 1) * w + px] - data[(py - 1) * w + px];
                        let mag = (gx_val * gx_val + gy_val * gy_val).sqrt();
                        let angle = gy_val.atan2(gx_val).rem_euclid(2.0 * std::f32::consts::PI);
                        let bin = ((angle / (2.0 * std::f32::consts::PI) * 8.0) as usize).min(7);
                        desc[bin_offset + bin] += mag;
                    }
                }
            }
        }

        // L2-normalize, clamp, re-normalize
        let norm1 = desc.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-7);
        for v in &mut desc {
            *v /= norm1;
        }
        for v in &mut desc {
            *v = v.min(0.2);
        }
        let norm2 = desc.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-7);
        for v in &mut desc {
            *v /= norm2;
        }

        descriptors.push(desc);
    }

    Ok(descriptors)
}

/// Matches SIFT descriptors using Euclidean distance with Lowe's ratio test.
///
/// Returns `(idx_a, idx_b, distance)` pairs where the best match ratio is below `ratio_threshold`.
pub fn sift_match(
    desc_a: &[[f32; 128]],
    desc_b: &[[f32; 128]],
    ratio_threshold: f32,
) -> Vec<(usize, usize, f32)> {
    let mut matches = Vec::new();

    for (ia, da) in desc_a.iter().enumerate() {
        let mut best_dist = f32::MAX;
        let mut second_dist = f32::MAX;
        let mut best_idx = 0;

        for (ib, db) in desc_b.iter().enumerate() {
            let dist: f32 = da
                .iter()
                .zip(db.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt();
            if dist < best_dist {
                second_dist = best_dist;
                best_dist = dist;
                best_idx = ib;
            } else if dist < second_dist {
                second_dist = dist;
            }
        }

        if second_dist > 0.0 && best_dist / second_dist < ratio_threshold {
            matches.push((ia, best_idx, best_dist));
        }
    }

    matches
}

// ── Blob detection (Laplacian of Gaussian) ─────────────────────────

/// Laplacian-of-Gaussian blob detection on a grayscale `[H, W, 1]` image.
///
/// Builds a scale space by applying Gaussian blur at `num_sigma` scales
/// linearly spaced between `min_sigma` and `max_sigma`, computing the
/// Laplacian at each scale (approximated with second differences), and
/// scale-normalising by `sigma²`.
///
/// Local maxima in the 3×3×3 scale-space neighbourhood that exceed
/// `threshold` are returned as `(row, col, sigma)`.
pub fn blob_log(
    img: &Tensor,
    min_sigma: f32,
    max_sigma: f32,
    num_sigma: usize,
    threshold: f32,
) -> Result<Vec<(usize, usize, f32)>, ImgProcError> {
    let (h, w, c) = hwc_shape(img)?;
    if c != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: c,
        });
    }
    if num_sigma == 0 {
        return Ok(Vec::new());
    }

    // Build sigma values
    let sigmas: Vec<f32> = if num_sigma == 1 {
        vec![min_sigma]
    } else {
        (0..num_sigma)
            .map(|i| min_sigma + (max_sigma - min_sigma) * i as f32 / (num_sigma - 1) as f32)
            .collect()
    };

    // Build scale-space: for each sigma, blur then compute Laplacian, then scale-normalise
    let data = img.data();
    let n = h * w;
    let mut scale_space: Vec<Vec<f32>> = Vec::with_capacity(num_sigma);

    for &sigma in &sigmas {
        // Gaussian blur with given sigma using separable convolution
        let blurred = gaussian_blur_sigma(data, h, w, sigma);

        // Approximate Laplacian with second differences: d²/dx² + d²/dy²
        let mut lap = vec![0.0f32; n];
        for y in 1..h.saturating_sub(1) {
            for x in 1..w.saturating_sub(1) {
                let idx = y * w + x;
                let dxx = blurred[idx + 1] - 2.0 * blurred[idx] + blurred[idx - 1];
                let dyy = blurred[idx + w] - 2.0 * blurred[idx] + blurred[idx - w];
                // Scale-normalise by sigma²
                lap[idx] = (dxx + dyy).abs() * sigma * sigma;
            }
        }
        scale_space.push(lap);
    }

    // Find local maxima in 3×3×3 neighbourhood (exclude border region proportional to sigma)
    let mut blobs = Vec::new();
    for s in 0..num_sigma {
        let border = (sigmas[s] * 3.0).ceil() as usize + 1;
        for y in border..h.saturating_sub(border) {
            for x in border..w.saturating_sub(border) {
                let val = scale_space[s][y * w + x];
                if val < threshold {
                    continue;
                }
                let mut is_max = true;
                'outer: for ds in -1i32..=1 {
                    let si = s as i32 + ds;
                    if si < 0 || si >= num_sigma as i32 {
                        continue;
                    }
                    let si = si as usize;
                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            if ds == 0 && dy == 0 && dx == 0 {
                                continue;
                            }
                            let ny = (y as i32 + dy) as usize;
                            let nx = (x as i32 + dx) as usize;
                            if ny < h && nx < w && scale_space[si][ny * w + nx] >= val {
                                is_max = false;
                                break 'outer;
                            }
                        }
                    }
                }
                if is_max {
                    blobs.push((y, x, sigmas[s]));
                }
            }
        }
    }

    Ok(blobs)
}

/// Separable Gaussian blur with arbitrary sigma on flat single-channel data.
fn gaussian_blur_sigma(data: &[f32], h: usize, w: usize, sigma: f32) -> Vec<f32> {
    let radius = (sigma * 3.0).ceil() as usize;
    let size = 2 * radius + 1;

    // Build 1D Gaussian kernel
    let mut kernel = vec![0.0f32; size];
    let denom = 2.0 * sigma * sigma;
    let mut sum = 0.0f32;
    for i in 0..size {
        let x = i as f32 - radius as f32;
        kernel[i] = (-x * x / denom).exp();
        sum += kernel[i];
    }
    for v in &mut kernel {
        *v /= sum;
    }

    let n = h * w;
    // Horizontal pass
    let mut tmp = vec![0.0f32; n];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for k in 0..size {
                let sx = x as i32 + k as i32 - radius as i32;
                if sx >= 0 && sx < w as i32 {
                    acc += data[y * w + sx as usize] * kernel[k];
                }
            }
            tmp[y * w + x] = acc;
        }
    }
    // Vertical pass
    let mut out = vec![0.0f32; n];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for k in 0..size {
                let sy = y as i32 + k as i32 - radius as i32;
                if sy >= 0 && sy < h as i32 {
                    acc += tmp[sy as usize * w + x] * kernel[k];
                }
            }
            out[y * w + x] = acc;
        }
    }
    out
}

// ── Hough circle transform ─────────────────────────────────────────

/// Hough circle transform on a binary/edge single-channel `[H, W, 1]` image.
///
/// For each edge pixel (value > 0.5), votes are cast into an accumulator for
/// circles of every radius in `[min_radius, max_radius]`. Returns
/// `(center_row, center_col, radius)` for accumulator peaks above `threshold`.
pub fn hough_circles(
    img: &Tensor,
    min_radius: usize,
    max_radius: usize,
    threshold: usize,
) -> Result<Vec<(usize, usize, usize)>, ImgProcError> {
    let (h, w, c) = hwc_shape(img)?;
    if c != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: c,
        });
    }
    if min_radius > max_radius || max_radius == 0 {
        return Ok(Vec::new());
    }
    let data = img.data();
    let num_radii = max_radius - min_radius + 1;

    // Accumulator: [radius_idx][y * w + x]
    let mut acc = vec![vec![0u32; h * w]; num_radii];

    // Pre-compute circle offsets for each radius
    for ri in 0..num_radii {
        let r = min_radius + ri;
        // Generate circle points using angular discretisation
        let circumference = (2.0 * std::f32::consts::PI * r as f32).ceil() as usize;
        let num_steps = circumference.max(36);
        let mut visited = std::collections::HashSet::new();

        for step in 0..num_steps {
            let angle = 2.0 * std::f32::consts::PI * step as f32 / num_steps as f32;
            let dx = (r as f32 * angle.cos()).round() as i32;
            let dy = (r as f32 * angle.sin()).round() as i32;
            visited.insert((dy, dx));
        }

        // Vote
        for y in 0..h {
            for x in 0..w {
                if data[y * w + x] <= 0.5 {
                    continue;
                }
                for &(dy, dx) in &visited {
                    let cy = y as i32 - dy;
                    let cx = x as i32 - dx;
                    if cy >= 0 && cy < h as i32 && cx >= 0 && cx < w as i32 {
                        acc[ri][cy as usize * w + cx as usize] += 1;
                    }
                }
            }
        }
    }

    // Extract peaks above threshold with 3×3×3 non-maximum suppression
    let mut circles = Vec::new();
    for ri in 0..num_radii {
        for y in 0..h {
            for x in 0..w {
                let votes = acc[ri][y * w + x];
                if votes < threshold as u32 {
                    continue;
                }
                let mut is_max = true;
                'peak: for dri in -1i32..=1 {
                    let nri = ri as i32 + dri;
                    if nri < 0 || nri >= num_radii as i32 {
                        continue;
                    }
                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            if dri == 0 && dy == 0 && dx == 0 {
                                continue;
                            }
                            let ny = y as i32 + dy;
                            let nx = x as i32 + dx;
                            if ny >= 0
                                && ny < h as i32
                                && nx >= 0
                                && nx < w as i32
                                && acc[nri as usize][ny as usize * w + nx as usize] >= votes
                            {
                                is_max = false;
                                break 'peak;
                            }
                        }
                    }
                }
                if is_max {
                    circles.push((y, x, min_radius + ri));
                }
            }
        }
    }

    // Sort by votes descending
    circles.sort_by(|a, b| {
        let va = acc[a.2 - min_radius][a.0 * w + a.1];
        let vb = acc[b.2 - min_radius][b.0 * w + b.1];
        vb.cmp(&va)
    });

    Ok(circles)
}
