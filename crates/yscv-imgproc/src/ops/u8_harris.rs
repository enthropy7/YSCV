//! Harris corner detector (u8-native, integer Sobel + integer structure tensor).
#![allow(unsafe_code)]

#[cfg(not(target_os = "macos"))]
use super::u8ops::RAYON_THRESHOLD;
#[cfg(not(target_os = "macos"))]
use rayon::prelude::*;

// ============================================================================
// Harris corner detector (u8-native, integer Sobel + integer structure tensor)
// ============================================================================

/// Harris corner detection operating directly on u8 grayscale image data.
///
/// Computes integer Sobel gradients (i16), accumulates structure tensor in i32,
/// and only converts to f32 for the final Harris response `det - k * trace²`.
/// Uses NEON SIMD for gradient computation and GCD/rayon for row parallelism.
///
/// Returns a list of `(x, y, response)` corners above `threshold` after 3×3 NMS.
pub fn harris_corners_u8(
    src: &[u8],
    width: usize,
    height: usize,
    block_size: usize,
    k: f32,
    threshold: f32,
) -> Vec<(usize, usize, f32)> {
    if height < 3 || width < 3 || src.len() != width * height {
        return Vec::new();
    }

    let half = block_size / 2;
    let npix = width * height;

    // Step 1: Compute Sobel gradients Ix, Iy as i16
    let mut ix = vec![0i16; npix];
    let mut iy = vec![0i16; npix];

    let interior_h = height - 2;

    #[cfg(target_os = "macos")]
    {
        let src_ptr = super::SendConstPtr(src.as_ptr());
        let ix_ptr = super::SendPtr(ix.as_mut_ptr());
        let iy_ptr = super::SendPtr(iy.as_mut_ptr());
        super::u8ops::gcd::parallel_for(interior_h, |i| {
            let y = i + 1;
            let sp = src_ptr.ptr();
            let ixp = ix_ptr.ptr();
            let iyp = iy_ptr.ptr();
            let row0 = unsafe { core::slice::from_raw_parts(sp.add((y - 1) * width), width) };
            let row1 = unsafe { core::slice::from_raw_parts(sp.add(y * width), width) };
            let row2 = unsafe { core::slice::from_raw_parts(sp.add((y + 1) * width), width) };
            let dix = unsafe { core::slice::from_raw_parts_mut(ixp.add(y * width), width) };
            let diy = unsafe { core::slice::from_raw_parts_mut(iyp.add(y * width), width) };
            let done = harris_sobel_row_simd(row0, row1, row2, dix, diy, width);
            harris_sobel_row_scalar(row0, row1, row2, dix, diy, width, done);
        });
    }

    #[cfg(not(target_os = "macos"))]
    {
        let use_rayon = npix >= RAYON_THRESHOLD && !cfg!(miri);
        if use_rayon && interior_h > 1 {
            let ix_interior = &mut ix[width..width + interior_h * width];
            let iy_interior = &mut iy[width..width + interior_h * width];
            ix_interior
                .par_chunks_mut(width)
                .zip(iy_interior.par_chunks_mut(width))
                .enumerate()
                .for_each(|(i, (dix, diy))| {
                    let y = i + 1;
                    let row0 = &src[(y - 1) * width..y * width];
                    let row1 = &src[y * width..(y + 1) * width];
                    let row2 = &src[(y + 1) * width..(y + 2) * width];
                    let done = harris_sobel_row_simd(row0, row1, row2, dix, diy, width);
                    harris_sobel_row_scalar(row0, row1, row2, dix, diy, width, done);
                });
        } else {
            for y in 1..height - 1 {
                let row0 = &src[(y - 1) * width..y * width];
                let row1 = &src[y * width..(y + 1) * width];
                let row2 = &src[(y + 1) * width..(y + 2) * width];
                let dix = &mut ix[y * width..(y + 1) * width];
                let diy = &mut iy[y * width..(y + 1) * width];
                let done = harris_sobel_row_simd(row0, row1, row2, dix, diy, width);
                harris_sobel_row_scalar(row0, row1, row2, dix, diy, width, done);
            }
        }
    }

    // Step 2+3: Build structure tensor sums and compute Harris response
    let margin = half + 1; // need half for block + 1 for sobel border
    if height <= 2 * margin || width <= 2 * margin {
        return Vec::new();
    }
    let resp_h = height - 2 * margin;
    let resp_w = width - 2 * margin;
    let mut response = vec![0.0f32; resp_h * resp_w];

    #[cfg(target_os = "macos")]
    {
        let ix_ptr = super::SendConstPtr(ix.as_ptr());
        let iy_ptr = super::SendConstPtr(iy.as_ptr());
        let resp_ptr = super::SendPtr(response.as_mut_ptr());
        super::u8ops::gcd::parallel_for(resp_h, |ry| {
            let y = ry + margin;
            let ixp = ix_ptr.ptr();
            let iyp = iy_ptr.ptr();
            let rp = resp_ptr.ptr();
            let dst = unsafe { core::slice::from_raw_parts_mut(rp.add(ry * resp_w), resp_w) };
            let ixs = unsafe { core::slice::from_raw_parts(ixp, npix) };
            let iys = unsafe { core::slice::from_raw_parts(iyp, npix) };
            harris_response_row(ixs, iys, dst, y, margin, width, half, k);
        });
    }

    #[cfg(not(target_os = "macos"))]
    {
        let use_rayon = npix >= RAYON_THRESHOLD && !cfg!(miri);
        if use_rayon && resp_h > 1 {
            response
                .par_chunks_mut(resp_w)
                .enumerate()
                .for_each(|(ry, dst)| {
                    let y = ry + margin;
                    harris_response_row(&ix, &iy, dst, y, margin, width, half, k);
                });
        } else {
            for ry in 0..resp_h {
                let y = ry + margin;
                let dst = &mut response[ry * resp_w..(ry + 1) * resp_w];
                harris_response_row(&ix, &iy, dst, y, margin, width, half, k);
            }
        }
    }

    // Step 4: NMS in 3x3 neighborhood + threshold
    let mut corners = Vec::new();
    for ry in 1..resp_h.saturating_sub(1) {
        for rx in 1..resp_w.saturating_sub(1) {
            let val = response[ry * resp_w + rx];
            if val <= threshold {
                continue;
            }
            let mut is_max = true;
            'nms: for dy in 0..3usize {
                for dx in 0..3usize {
                    if dy == 1 && dx == 1 {
                        continue;
                    }
                    if response[(ry + dy - 1) * resp_w + (rx + dx - 1)] >= val {
                        is_max = false;
                        break 'nms;
                    }
                }
            }
            if is_max {
                corners.push((rx + margin, ry + margin, val));
            }
        }
    }

    corners
}

/// Compute Harris response for one row of the response image.
#[inline]
fn harris_response_row(
    ix: &[i16],
    iy: &[i16],
    dst: &mut [f32],
    y: usize,
    margin: usize,
    width: usize,
    half: usize,
    k: f32,
) {
    for rx in 0..dst.len() {
        let x = rx + margin;
        let mut sxx: i32 = 0;
        let mut sxy: i32 = 0;
        let mut syy: i32 = 0;

        for by in (y - half)..=(y + half) {
            let row_off = by * width;
            for bx in (x - half)..=(x + half) {
                let gx = ix[row_off + bx] as i32;
                let gy = iy[row_off + bx] as i32;
                sxx += gx * gx;
                sxy += gx * gy;
                syy += gy * gy;
            }
        }

        let sxx_f = sxx as f32;
        let sxy_f = sxy as f32;
        let syy_f = syy as f32;
        let det = sxx_f * syy_f - sxy_f * sxy_f;
        let trace = sxx_f + syy_f;
        dst[rx] = det - k * trace * trace;
    }
}

/// SIMD dispatch for Sobel gradient row (writes Ix, Iy as i16).
/// Returns the x position where SIMD stopped (scalar handles the rest).
#[inline]
fn harris_sobel_row_simd(
    row0: &[u8],
    row1: &[u8],
    row2: &[u8],
    dix: &mut [i16],
    diy: &mut [i16],
    w: usize,
) -> usize {
    if cfg!(miri) || w < 10 {
        return 1;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { harris_sobel_neon(row0, row1, row2, dix, diy, w) };
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("sse2") {
            return unsafe { harris_sobel_sse(row0, row1, row2, dix, diy, w) };
        }
    }
    1
}

/// Scalar Sobel gradient computation for remaining pixels in a row.
#[inline]
fn harris_sobel_row_scalar(
    row0: &[u8],
    row1: &[u8],
    row2: &[u8],
    dix: &mut [i16],
    diy: &mut [i16],
    w: usize,
    start: usize,
) {
    for x in start..w - 1 {
        if x == 0 {
            continue;
        }
        dix[x] = row0[x + 1] as i16 - row0[x - 1] as i16
            + 2 * (row1[x + 1] as i16 - row1[x - 1] as i16)
            + row2[x + 1] as i16
            - row2[x - 1] as i16;
        diy[x] = row2[x - 1] as i16 + 2 * row2[x] as i16 + row2[x + 1] as i16
            - row0[x - 1] as i16
            - 2 * row0[x] as i16
            - row0[x + 1] as i16;
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn harris_sobel_neon(
    row0: &[u8],
    row1: &[u8],
    row2: &[u8],
    dix: &mut [i16],
    diy: &mut [i16],
    w: usize,
) -> usize {
    use std::arch::aarch64::*;

    let p0 = row0.as_ptr();
    let p1 = row1.as_ptr();
    let p2 = row2.as_ptr();
    let oix = dix.as_mut_ptr();
    let oiy = diy.as_mut_ptr();

    // Process 8 pixels at a time (i16 lanes = 8 per 128-bit register)
    let mut x = 1usize;
    while x + 9 <= w {
        // Load 8 u8 values at offsets x-1, x, x+1 for each row
        let r0_left = vld1_u8(p0.add(x - 1));
        let r0_mid = vld1_u8(p0.add(x));
        let r0_right = vld1_u8(p0.add(x + 1));
        let r1_left = vld1_u8(p1.add(x - 1));
        let r1_right = vld1_u8(p1.add(x + 1));
        let r2_left = vld1_u8(p2.add(x - 1));
        let r2_mid = vld1_u8(p2.add(x));
        let r2_right = vld1_u8(p2.add(x + 1));

        // Widen to u16 for subtraction without overflow
        let r0l = vmovl_u8(r0_left);
        let r0r = vmovl_u8(r0_right);
        let r1l = vmovl_u8(r1_left);
        let r1r = vmovl_u8(r1_right);
        let r2l = vmovl_u8(r2_left);
        let r2r = vmovl_u8(r2_right);
        let r0m = vmovl_u8(r0_mid);
        let r2m = vmovl_u8(r2_mid);

        // Ix = [-1,0,1; -2,0,2; -1,0,1]
        let gx_0 = vreinterpretq_s16_u16(vsubq_u16(r0r, r0l));
        let gx_1 = vreinterpretq_s16_u16(vsubq_u16(r1r, r1l));
        let gx_2 = vreinterpretq_s16_u16(vsubq_u16(r2r, r2l));
        let gx = vaddq_s16(vaddq_s16(gx_0, gx_2), vshlq_n_s16::<1>(gx_1));

        // Iy = [-1,-2,-1; 0,0,0; 1,2,1]
        let bot = vaddq_s16(
            vreinterpretq_s16_u16(vaddq_u16(r2l, r2r)),
            vshlq_n_s16::<1>(vreinterpretq_s16_u16(r2m)),
        );
        let top = vaddq_s16(
            vreinterpretq_s16_u16(vaddq_u16(r0l, r0r)),
            vshlq_n_s16::<1>(vreinterpretq_s16_u16(r0m)),
        );
        let gy = vsubq_s16(bot, top);

        vst1q_s16(oix.add(x), gx);
        vst1q_s16(oiy.add(x), gy);

        x += 8;
    }
    x
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse2")]
unsafe fn harris_sobel_sse(
    row0: &[u8],
    row1: &[u8],
    row2: &[u8],
    dix: &mut [i16],
    diy: &mut [i16],
    w: usize,
) -> usize {
    use std::arch::x86_64::*;

    let p0 = row0.as_ptr();
    let p1 = row1.as_ptr();
    let p2 = row2.as_ptr();
    let oix = dix.as_mut_ptr();
    let oiy = diy.as_mut_ptr();
    let zero = _mm_setzero_si128();

    // Process 8 pixels at a time (i16 = 8 per 128-bit register)
    let mut x = 1usize;
    while x + 9 <= w {
        // Load 8 u8 values at offsets x-1, x, x+1 for each row
        let r0_left = _mm_loadl_epi64(p0.add(x - 1) as *const __m128i);
        let r0_mid = _mm_loadl_epi64(p0.add(x) as *const __m128i);
        let r0_right = _mm_loadl_epi64(p0.add(x + 1) as *const __m128i);
        let r1_left = _mm_loadl_epi64(p1.add(x - 1) as *const __m128i);
        let r1_right = _mm_loadl_epi64(p1.add(x + 1) as *const __m128i);
        let r2_left = _mm_loadl_epi64(p2.add(x - 1) as *const __m128i);
        let r2_mid = _mm_loadl_epi64(p2.add(x) as *const __m128i);
        let r2_right = _mm_loadl_epi64(p2.add(x + 1) as *const __m128i);

        // Widen to u16 via unpack with zero
        let r0l = _mm_unpacklo_epi8(r0_left, zero);
        let r0r = _mm_unpacklo_epi8(r0_right, zero);
        let r1l = _mm_unpacklo_epi8(r1_left, zero);
        let r1r = _mm_unpacklo_epi8(r1_right, zero);
        let r2l = _mm_unpacklo_epi8(r2_left, zero);
        let r2r = _mm_unpacklo_epi8(r2_right, zero);
        let r0m = _mm_unpacklo_epi8(r0_mid, zero);
        let r2m = _mm_unpacklo_epi8(r2_mid, zero);

        // Ix = [-1,0,1; -2,0,2; -1,0,1]
        let gx_0 = _mm_sub_epi16(r0r, r0l);
        let gx_1 = _mm_sub_epi16(r1r, r1l);
        let gx_2 = _mm_sub_epi16(r2r, r2l);
        let gx = _mm_add_epi16(_mm_add_epi16(gx_0, gx_2), _mm_slli_epi16(gx_1, 1));

        // Iy = [-1,-2,-1; 0,0,0; 1,2,1]
        let bot = _mm_add_epi16(_mm_add_epi16(r2l, r2r), _mm_slli_epi16(r2m, 1));
        let top = _mm_add_epi16(_mm_add_epi16(r0l, r0r), _mm_slli_epi16(r0m, 1));
        let gy = _mm_sub_epi16(bot, top);

        _mm_storeu_si128(oix.add(x) as *mut __m128i, gx);
        _mm_storeu_si128(oiy.add(x) as *mut __m128i, gy);

        x += 8;
    }
    x
}
