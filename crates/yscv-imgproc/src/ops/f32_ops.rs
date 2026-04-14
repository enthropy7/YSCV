//! f32 image operations — direct slice processing, zero Tensor overhead.
//!
//! # Safety contract
//!
//! Unsafe code categories:
//! 1. **SIMD intrinsics (NEON / SSE)** — ISA guard via `#[target_feature]` or runtime detection.
//! 2. **`SendConstPtr` / `SendPtr` for rayon** — non-overlapping chunk access.
//! 3. **Pointer arithmetic in row kernels** — bounded by image width/height validated at entry.
#![allow(unsafe_code)]

#[cfg(target_os = "macos")]
use super::u8ops::gcd;
use super::u8ops::{ImageF32, RAYON_THRESHOLD};
use rayon::prelude::*;

// ============================================================================
// f32 image operations — direct slice processing, zero Tensor overhead
// ============================================================================

// ── Grayscale f32 ──────────────────────────────────────────────────────────

/// Converts RGB f32 image `[H,W,3]` to grayscale `[H,W,1]`.
/// Uses BT.601: gray = 0.299*R + 0.587*G + 0.114*B.
#[allow(clippy::uninit_vec)]
pub fn grayscale_f32(input: &ImageF32) -> Option<ImageF32> {
    if input.channels() != 3 {
        return None;
    }
    let (h, w) = (input.height(), input.width());
    let src = input.data();
    let total = h * w;

    let mut out: Vec<f32> = Vec::with_capacity(total);
    #[allow(unsafe_code)]
    // SAFETY: every element written by SIMD + scalar tail + GCD chunks.
    unsafe {
        out.set_len(total);
    }

    // GCD parallel in chunks (bandwidth-bound — coarse chunks reduce dispatch overhead)
    #[cfg(target_os = "macos")]
    if total >= RAYON_THRESHOLD && !cfg!(miri) {
        let n_chunks = 8usize.min(h);
        let chunk_h = h.div_ceil(n_chunks);
        let sp = super::SendConstPtr(src.as_ptr());
        let dp = super::SendPtr(out.as_mut_ptr());
        gcd::parallel_for(n_chunks, |chunk| {
            let sp = sp.ptr();
            let dp = dp.ptr();
            let y_start = chunk * chunk_h;
            let y_end = ((chunk + 1) * chunk_h).min(h);
            let n = (y_end - y_start) * w;
            let src_off = y_start * w * 3;
            let dst_off = y_start * w;
            // SAFETY: pointer and length from validated image data; parallel chunks are non-overlapping.
            let chunk_src = unsafe { core::slice::from_raw_parts(sp.add(src_off), n * 3) };
            let chunk_dst = unsafe { core::slice::from_raw_parts_mut(dp.add(dst_off), n) };
            let done = grayscale_f32_simd(chunk_src, chunk_dst, n);
            for i in done..n {
                let base = i * 3;
                chunk_dst[i] = 0.299 * chunk_src[base]
                    + 0.587 * chunk_src[base + 1]
                    + 0.114 * chunk_src[base + 2];
            }
        });
        return ImageF32::new(out, h, w, 1);
    }

    // Flat single-threaded
    let done = grayscale_f32_simd(src, &mut out, total);
    for i in done..total {
        let base = i * 3;
        out[i] = 0.299 * src[base] + 0.587 * src[base + 1] + 0.114 * src[base + 2];
    }

    ImageF32::new(out, h, w, 1)
}

fn grayscale_f32_simd(src: &[f32], dst: &mut [f32], total: usize) -> usize {
    if cfg!(miri) || total < 4 {
        return 0;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: ISA guard (feature detection) above.
            return unsafe { grayscale_f32_neon(src, dst, total) };
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse2") {
            // SAFETY: ISA guard (feature detection) above.
            return unsafe { grayscale_f32_sse(src, dst, total) };
        }
    }
    0
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn grayscale_f32_neon(src: &[f32], dst: &mut [f32], total: usize) -> usize {
    use std::arch::aarch64::*;
    let coeff_r = vdupq_n_f32(0.299);
    let coeff_g = vdupq_n_f32(0.587);
    let coeff_b = vdupq_n_f32(0.114);
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();
    let mut i = 0usize;
    while i + 4 <= total {
        let rgb = vld3q_f32(sp.add(i * 3));
        let gray = vfmaq_f32(
            vfmaq_f32(vmulq_f32(rgb.0, coeff_r), rgb.1, coeff_g),
            rgb.2,
            coeff_b,
        );
        vst1q_f32(dp.add(i), gray);
        i += 4;
    }
    i
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn grayscale_f32_sse(src: &[f32], dst: &mut [f32], total: usize) -> usize {
    use std::arch::x86_64::*;
    let coeff_r = _mm_set1_ps(0.299);
    let coeff_g = _mm_set1_ps(0.587);
    let coeff_b = _mm_set1_ps(0.114);
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();
    let mut i = 0usize;
    // Process 4 pixels: 12 floats from RGB, deinterleave manually
    while i + 4 <= total {
        let v0 = _mm_loadu_ps(sp.add(i * 3)); // R0 G0 B0 R1
        let v1 = _mm_loadu_ps(sp.add(i * 3 + 4)); // G1 B1 R2 G2
        let v2 = _mm_loadu_ps(sp.add(i * 3 + 8)); // B2 R3 G3 B3

        // Deinterleave: R = [R0,R1,R2,R3], G = [G0,G1,G2,G3], B = [B0,B1,B2,B3]
        // v0 = [R0,G0,B0,R1], v1 = [G1,B1,R2,G2], v2 = [B2,R3,G3,B3]
        let t0 = _mm_shuffle_ps(v0, v1, 0b01_00_11_00); // [R0, R1, G1, B1] -> need [R0,R1,_,_]
        let _r = _mm_shuffle_ps(t0, v2, 0b01_11_10_00); // [R0, R1, R2, R3] — actually let me do this properly

        // Proper deinterleave for 3-channel f32:
        // R0=v0[0], R1=v0[3], R2=v1[2], R3=v2[1]
        // G0=v0[1], G1=v1[0], G2=v1[3], G3=v2[2]
        // B0=v0[2], B1=v1[1], B2=v2[0], B3=v2[3]
        let r0r1 = _mm_shuffle_ps(v0, v0, 0b11_11_00_00); // [R0,R0,R1,R1]
        let r2r3 = _mm_shuffle_ps(v1, v2, 0b01_01_10_10); // [R2,R2,R3,R3]
        let r = _mm_shuffle_ps(r0r1, r2r3, 0b10_00_10_00); // [R0,R1,R2,R3]

        let g0g1 = _mm_shuffle_ps(v0, v1, 0b00_00_01_01); // [G0,G0,G1,G1]
        let g2g3 = _mm_shuffle_ps(v1, v2, 0b10_10_11_11); // [G2,G2,G3,G3]
        let g = _mm_shuffle_ps(g0g1, g2g3, 0b10_00_10_00); // [G0,G1,G2,G3]

        let b0b1 = _mm_shuffle_ps(v0, v1, 0b01_01_10_10); // [B0,B0,B1,B1]
        let b2b3 = _mm_shuffle_ps(v2, v2, 0b11_11_00_00); // [B2,B2,B3,B3]
        let b = _mm_shuffle_ps(b0b1, b2b3, 0b10_00_10_00); // [B0,B1,B2,B3]

        let gray = _mm_add_ps(
            _mm_add_ps(_mm_mul_ps(r, coeff_r), _mm_mul_ps(g, coeff_g)),
            _mm_mul_ps(b, coeff_b),
        );
        _mm_storeu_ps(dp.add(i), gray);
        i += 4;
    }
    i
}

// ── Gaussian blur 3x3 f32 ─────────────────────────────────────────────────

/// 3x3 Gaussian blur on single-channel f32 image.
/// Direct `[1,2,1]`x`[1,2,1]`/16 kernel. Replicate border.
pub fn gaussian_blur_3x3_f32(input: &ImageF32) -> Option<ImageF32> {
    if input.channels() != 1 {
        return None;
    }
    let (h, w) = (input.height(), input.width());
    if h < 3 || w < 3 {
        return Some(input.clone());
    }
    let src = input.data();

    let mut out = vec![0.0f32; h * w];

    #[cfg(target_arch = "aarch64")]
    if !cfg!(miri) && std::arch::is_aarch64_feature_detected!("neon") {
        // SAFETY: ISA guard (feature detection) above.
        unsafe {
            gauss_3x3_direct_f32_neon(src, &mut out, h, w);
        }
        return ImageF32::new(out, h, w, 1);
    }

    #[cfg(target_arch = "x86_64")]
    if !cfg!(miri) && is_x86_feature_detected!("sse2") {
        // SAFETY: ISA guard (feature detection) above.
        unsafe {
            gauss_3x3_direct_f32_sse(src, &mut out, h, w);
        }
        return ImageF32::new(out, h, w, 1);
    }

    // Scalar fallback
    for y in 0..h {
        let ay = if y > 0 { y - 1 } else { 0 };
        let by = if y + 1 < h { y + 1 } else { h - 1 };
        let top = &src[ay * w..(ay + 1) * w];
        let mid = &src[y * w..(y + 1) * w];
        let bot = &src[by * w..(by + 1) * w];
        let dst = &mut out[y * w..(y + 1) * w];
        for x in 0..w {
            let lx = if x > 0 { x - 1 } else { 0 };
            let rx = if x + 1 < w { x + 1 } else { w - 1 };
            let v = top[lx]
                + 2.0 * top[x]
                + top[rx]
                + 2.0 * (mid[lx] + 2.0 * mid[x] + mid[rx])
                + bot[lx]
                + 2.0 * bot[x]
                + bot[rx];
            dst[x] = v / 16.0;
        }
    }
    ImageF32::new(out, h, w, 1)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn gauss_row_f32_neon(
    top: *const f32,
    mid: *const f32,
    bot: *const f32,
    dst: *mut f32,
    w: usize,
) {
    use std::arch::aarch64::*;
    let inv16 = vdupq_n_f32(1.0 / 16.0);
    let two = vdupq_n_f32(2.0);

    let border_t = vdupq_n_f32(*top);
    let border_m = vdupq_n_f32(*mid);
    let border_b = vdupq_n_f32(*bot);
    let mut prev_t = border_t;
    let mut prev_m = border_m;
    let mut prev_b = border_b;
    let mut cur_t = vld1q_f32(top);
    let mut cur_m = vld1q_f32(mid);
    let mut cur_b = vld1q_f32(bot);

    let mut x = 0usize;
    let main_end = w.saturating_sub(4);

    while x < main_end {
        let nxt_t = vld1q_f32(top.add(x + 4));
        let nxt_m = vld1q_f32(mid.add(x + 4));
        let nxt_b = vld1q_f32(bot.add(x + 4));

        // Horizontal: left + 2*center + right for each row
        let left_t = vextq_f32::<3>(prev_t, cur_t);
        let right_t = vextq_f32::<1>(cur_t, nxt_t);
        let ht = vfmaq_f32(vaddq_f32(left_t, right_t), cur_t, two);

        let left_m = vextq_f32::<3>(prev_m, cur_m);
        let right_m = vextq_f32::<1>(cur_m, nxt_m);
        let hm = vfmaq_f32(vaddq_f32(left_m, right_m), cur_m, two);

        let left_b = vextq_f32::<3>(prev_b, cur_b);
        let right_b = vextq_f32::<1>(cur_b, nxt_b);
        let hb = vfmaq_f32(vaddq_f32(left_b, right_b), cur_b, two);

        // Vertical: top + 2*mid + bot, then /16
        let v = vfmaq_f32(vaddq_f32(ht, hb), hm, two);
        vst1q_f32(dst.add(x), vmulq_f32(v, inv16));

        prev_t = cur_t;
        cur_t = nxt_t;
        prev_m = cur_m;
        cur_m = nxt_m;
        prev_b = cur_b;
        cur_b = nxt_b;
        x += 4;
    }

    // Tail
    if x < w {
        let border_t_end = vdupq_n_f32(*top.add(w - 1));
        let border_m_end = vdupq_n_f32(*mid.add(w - 1));
        let border_b_end = vdupq_n_f32(*bot.add(w - 1));

        let left_t = vextq_f32::<3>(prev_t, cur_t);
        let right_t = vextq_f32::<1>(cur_t, border_t_end);
        let ht = vfmaq_f32(vaddq_f32(left_t, right_t), cur_t, two);

        let left_m = vextq_f32::<3>(prev_m, cur_m);
        let right_m = vextq_f32::<1>(cur_m, border_m_end);
        let hm = vfmaq_f32(vaddq_f32(left_m, right_m), cur_m, two);

        let left_b = vextq_f32::<3>(prev_b, cur_b);
        let right_b = vextq_f32::<1>(cur_b, border_b_end);
        let hb = vfmaq_f32(vaddq_f32(left_b, right_b), cur_b, two);

        let v = vfmaq_f32(vaddq_f32(ht, hb), hm, two);
        vst1q_f32(dst.add(x), vmulq_f32(v, inv16));
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn gauss_3x3_direct_f32_neon(src: &[f32], out: &mut [f32], h: usize, w: usize) {
    let sp = src.as_ptr();
    let dp = out.as_mut_ptr();

    #[cfg(target_os = "macos")]
    if h * w >= RAYON_THRESHOLD {
        let sp = sp as usize;
        let dp = dp as usize;
        gcd::parallel_for(h, |y| {
            let sp = sp as *const f32;
            let dp = dp as *mut f32;
            let top = if y > 0 { sp.add((y - 1) * w) } else { sp };
            let mid = sp.add(y * w);
            let bot = if y + 1 < h {
                sp.add((y + 1) * w)
            } else {
                sp.add((h - 1) * w)
            };
            gauss_row_f32_neon(top, mid, bot, dp.add(y * w), w);
        });
        return;
    }

    if h * w >= RAYON_THRESHOLD {
        let sp = sp as usize;
        let dp = dp as usize;
        (0..h).into_par_iter().for_each(|y| {
            let sp = sp as *const f32;
            let dp = dp as *mut f32;
            let top = if y > 0 { sp.add((y - 1) * w) } else { sp };
            let mid = sp.add(y * w);
            let bot = if y + 1 < h {
                sp.add((y + 1) * w)
            } else {
                sp.add((h - 1) * w)
            };
            gauss_row_f32_neon(top, mid, bot, dp.add(y * w), w);
        });
        return;
    }

    gauss_row_f32_neon(sp, sp, if h > 1 { sp.add(w) } else { sp }, dp, w);
    for y in 1..h.saturating_sub(1) {
        gauss_row_f32_neon(
            sp.add((y - 1) * w),
            sp.add(y * w),
            sp.add((y + 1) * w),
            dp.add(y * w),
            w,
        );
    }
    if h > 1 {
        gauss_row_f32_neon(
            sp.add((h - 2) * w),
            sp.add((h - 1) * w),
            sp.add((h - 1) * w),
            dp.add((h - 1) * w),
            w,
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn gauss_row_f32_sse(
    top: *const f32,
    mid: *const f32,
    bot: *const f32,
    dst: *mut f32,
    w: usize,
) {
    use std::arch::x86_64::*;
    let inv16 = _mm_set1_ps(1.0 / 16.0);
    let two = _mm_set1_ps(2.0);

    let mut x = 0usize;
    // Process interior pixels, scalar handles borders
    if w >= 6 {
        // First pixel (x=0) done by scalar
        x = 1;
        while x + 4 <= w.saturating_sub(1) {
            let t_l = _mm_loadu_ps(top.add(x - 1));
            let t_c = _mm_loadu_ps(top.add(x));
            let t_r = _mm_loadu_ps(top.add(x + 1));
            let ht = _mm_add_ps(_mm_add_ps(t_l, t_r), _mm_mul_ps(t_c, two));

            let m_l = _mm_loadu_ps(mid.add(x - 1));
            let m_c = _mm_loadu_ps(mid.add(x));
            let m_r = _mm_loadu_ps(mid.add(x + 1));
            let hm = _mm_add_ps(_mm_add_ps(m_l, m_r), _mm_mul_ps(m_c, two));

            let b_l = _mm_loadu_ps(bot.add(x - 1));
            let b_c = _mm_loadu_ps(bot.add(x));
            let b_r = _mm_loadu_ps(bot.add(x + 1));
            let hb = _mm_add_ps(_mm_add_ps(b_l, b_r), _mm_mul_ps(b_c, two));

            let v = _mm_add_ps(_mm_add_ps(ht, hb), _mm_mul_ps(hm, two));
            _mm_storeu_ps(dst.add(x), _mm_mul_ps(v, inv16));
            x += 4;
        }
    }
    // Scalar for remaining and border pixels
    let border_start = if x == 0 { 0 } else { x };
    for xx in border_start..w {
        let lx = if xx > 0 { xx - 1 } else { 0 };
        let rx = if xx + 1 < w { xx + 1 } else { w - 1 };
        let v = *top.add(lx)
            + 2.0 * *top.add(xx)
            + *top.add(rx)
            + 2.0 * (*mid.add(lx) + 2.0 * *mid.add(xx) + *mid.add(rx))
            + *bot.add(lx)
            + 2.0 * *bot.add(xx)
            + *bot.add(rx);
        *dst.add(xx) = v / 16.0;
    }
    // Also handle x=0 if we skipped it
    if border_start == 1 {
        let lx = 0usize;
        let rx = 1usize.min(w - 1);
        let v = *top.add(lx)
            + 2.0 * *top.add(0)
            + *top.add(rx)
            + 2.0 * (*mid.add(lx) + 2.0 * *mid.add(0) + *mid.add(rx))
            + *bot.add(lx)
            + 2.0 * *bot.add(0)
            + *bot.add(rx);
        *dst.add(0) = v / 16.0;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn gauss_3x3_direct_f32_sse(src: &[f32], out: &mut [f32], h: usize, w: usize) {
    let sp = src.as_ptr();
    let dp = out.as_mut_ptr();

    if h * w >= RAYON_THRESHOLD {
        let sp = sp as usize;
        let dp = dp as usize;
        (0..h).into_par_iter().for_each(|y| {
            let sp = sp as *const f32;
            let dp = dp as *mut f32;
            let top = if y > 0 { sp.add((y - 1) * w) } else { sp };
            let mid = sp.add(y * w);
            let bot = if y + 1 < h {
                sp.add((y + 1) * w)
            } else {
                sp.add((h - 1) * w)
            };
            gauss_row_f32_sse(top, mid, bot, dp.add(y * w), w);
        });
        return;
    }

    gauss_row_f32_sse(sp, sp, if h > 1 { sp.add(w) } else { sp }, dp, w);
    for y in 1..h.saturating_sub(1) {
        gauss_row_f32_sse(
            sp.add((y - 1) * w),
            sp.add(y * w),
            sp.add((y + 1) * w),
            dp.add(y * w),
            w,
        );
    }
    if h > 1 {
        gauss_row_f32_sse(
            sp.add((h - 2) * w),
            sp.add((h - 1) * w),
            sp.add((h - 1) * w),
            dp.add((h - 1) * w),
            w,
        );
    }
}

// ── Box blur 3x3 f32 ──────────────────────────────────────────────────────

/// 3x3 box blur on single-channel f32 image.
/// Direct `[1,1,1]`x`[1,1,1]`/9 kernel. Replicate border.
pub fn box_blur_3x3_f32(input: &ImageF32) -> Option<ImageF32> {
    if input.channels() != 1 {
        return None;
    }
    let (h, w) = (input.height(), input.width());
    if h < 3 || w < 3 {
        return Some(input.clone());
    }
    let src = input.data();

    let mut out = vec![0.0f32; h * w];

    #[cfg(target_arch = "aarch64")]
    if !cfg!(miri) && std::arch::is_aarch64_feature_detected!("neon") {
        // SAFETY: ISA guard (feature detection) above.
        unsafe {
            box_3x3_direct_f32_neon(src, &mut out, h, w);
        }
        return ImageF32::new(out, h, w, 1);
    }

    #[cfg(target_arch = "x86_64")]
    if !cfg!(miri) && is_x86_feature_detected!("sse2") {
        // SAFETY: ISA guard (feature detection) above.
        unsafe {
            box_3x3_direct_f32_sse(src, &mut out, h, w);
        }
        return ImageF32::new(out, h, w, 1);
    }

    // Scalar fallback
    let inv9 = 1.0f32 / 9.0;
    for y in 0..h {
        let ay = if y > 0 { y - 1 } else { 0 };
        let by = if y + 1 < h { y + 1 } else { h - 1 };
        let top = &src[ay * w..(ay + 1) * w];
        let mid = &src[y * w..(y + 1) * w];
        let bot = &src[by * w..(by + 1) * w];
        let dst = &mut out[y * w..(y + 1) * w];
        for x in 0..w {
            let lx = if x > 0 { x - 1 } else { 0 };
            let rx = if x + 1 < w { x + 1 } else { w - 1 };
            let v = top[lx]
                + top[x]
                + top[rx]
                + mid[lx]
                + mid[x]
                + mid[rx]
                + bot[lx]
                + bot[x]
                + bot[rx];
            dst[x] = v * inv9;
        }
    }
    ImageF32::new(out, h, w, 1)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn box_row_f32_neon(
    top: *const f32,
    mid: *const f32,
    bot: *const f32,
    dst: *mut f32,
    w: usize,
) {
    use std::arch::aarch64::*;
    let inv9 = vdupq_n_f32(1.0 / 9.0);

    let border_t = vdupq_n_f32(*top);
    let border_m = vdupq_n_f32(*mid);
    let border_b = vdupq_n_f32(*bot);
    let mut prev_t = border_t;
    let mut prev_m = border_m;
    let mut prev_b = border_b;
    let mut cur_t = vld1q_f32(top);
    let mut cur_m = vld1q_f32(mid);
    let mut cur_b = vld1q_f32(bot);

    let mut x = 0usize;
    let main_end = w.saturating_sub(4);

    while x < main_end {
        let nxt_t = vld1q_f32(top.add(x + 4));
        let nxt_m = vld1q_f32(mid.add(x + 4));
        let nxt_b = vld1q_f32(bot.add(x + 4));

        let ht = vaddq_f32(
            vaddq_f32(vextq_f32::<3>(prev_t, cur_t), cur_t),
            vextq_f32::<1>(cur_t, nxt_t),
        );
        let hm = vaddq_f32(
            vaddq_f32(vextq_f32::<3>(prev_m, cur_m), cur_m),
            vextq_f32::<1>(cur_m, nxt_m),
        );
        let hb = vaddq_f32(
            vaddq_f32(vextq_f32::<3>(prev_b, cur_b), cur_b),
            vextq_f32::<1>(cur_b, nxt_b),
        );

        let v = vaddq_f32(vaddq_f32(ht, hm), hb);
        vst1q_f32(dst.add(x), vmulq_f32(v, inv9));

        prev_t = cur_t;
        cur_t = nxt_t;
        prev_m = cur_m;
        cur_m = nxt_m;
        prev_b = cur_b;
        cur_b = nxt_b;
        x += 4;
    }
    if x < w {
        let border_t_end = vdupq_n_f32(*top.add(w - 1));
        let border_m_end = vdupq_n_f32(*mid.add(w - 1));
        let border_b_end = vdupq_n_f32(*bot.add(w - 1));

        let ht = vaddq_f32(
            vaddq_f32(vextq_f32::<3>(prev_t, cur_t), cur_t),
            vextq_f32::<1>(cur_t, border_t_end),
        );
        let hm = vaddq_f32(
            vaddq_f32(vextq_f32::<3>(prev_m, cur_m), cur_m),
            vextq_f32::<1>(cur_m, border_m_end),
        );
        let hb = vaddq_f32(
            vaddq_f32(vextq_f32::<3>(prev_b, cur_b), cur_b),
            vextq_f32::<1>(cur_b, border_b_end),
        );

        let v = vaddq_f32(vaddq_f32(ht, hm), hb);
        vst1q_f32(dst.add(x), vmulq_f32(v, inv9));
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn box_3x3_direct_f32_neon(src: &[f32], out: &mut [f32], h: usize, w: usize) {
    let sp = src.as_ptr();
    let dp = out.as_mut_ptr();

    #[cfg(target_os = "macos")]
    if h * w >= RAYON_THRESHOLD {
        let sp = sp as usize;
        let dp = dp as usize;
        gcd::parallel_for(h, |y| {
            let sp = sp as *const f32;
            let dp = dp as *mut f32;
            let top = if y > 0 { sp.add((y - 1) * w) } else { sp };
            let mid = sp.add(y * w);
            let bot = if y + 1 < h {
                sp.add((y + 1) * w)
            } else {
                sp.add((h - 1) * w)
            };
            box_row_f32_neon(top, mid, bot, dp.add(y * w), w);
        });
        return;
    }

    if h * w >= RAYON_THRESHOLD {
        let sp = sp as usize;
        let dp = dp as usize;
        (0..h).into_par_iter().for_each(|y| {
            let sp = sp as *const f32;
            let dp = dp as *mut f32;
            let top = if y > 0 { sp.add((y - 1) * w) } else { sp };
            let mid = sp.add(y * w);
            let bot = if y + 1 < h {
                sp.add((y + 1) * w)
            } else {
                sp.add((h - 1) * w)
            };
            box_row_f32_neon(top, mid, bot, dp.add(y * w), w);
        });
        return;
    }

    box_row_f32_neon(sp, sp, if h > 1 { sp.add(w) } else { sp }, dp, w);
    for y in 1..h.saturating_sub(1) {
        box_row_f32_neon(
            sp.add((y - 1) * w),
            sp.add(y * w),
            sp.add((y + 1) * w),
            dp.add(y * w),
            w,
        );
    }
    if h > 1 {
        box_row_f32_neon(
            sp.add((h - 2) * w),
            sp.add((h - 1) * w),
            sp.add((h - 1) * w),
            dp.add((h - 1) * w),
            w,
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn box_row_f32_sse(
    top: *const f32,
    mid: *const f32,
    bot: *const f32,
    dst: *mut f32,
    w: usize,
) {
    use std::arch::x86_64::*;
    let inv9 = _mm_set1_ps(1.0 / 9.0);

    let mut x = 0usize;
    if w >= 6 {
        x = 1;
        while x + 4 <= w.saturating_sub(1) {
            let ht = _mm_add_ps(
                _mm_add_ps(_mm_loadu_ps(top.add(x - 1)), _mm_loadu_ps(top.add(x))),
                _mm_loadu_ps(top.add(x + 1)),
            );
            let hm = _mm_add_ps(
                _mm_add_ps(_mm_loadu_ps(mid.add(x - 1)), _mm_loadu_ps(mid.add(x))),
                _mm_loadu_ps(mid.add(x + 1)),
            );
            let hb = _mm_add_ps(
                _mm_add_ps(_mm_loadu_ps(bot.add(x - 1)), _mm_loadu_ps(bot.add(x))),
                _mm_loadu_ps(bot.add(x + 1)),
            );
            let v = _mm_add_ps(_mm_add_ps(ht, hm), hb);
            _mm_storeu_ps(dst.add(x), _mm_mul_ps(v, inv9));
            x += 4;
        }
    }
    let border_start = if x == 0 { 0 } else { x };
    for xx in border_start..w {
        let lx = if xx > 0 { xx - 1 } else { 0 };
        let rx = if xx + 1 < w { xx + 1 } else { w - 1 };
        let v = *top.add(lx)
            + *top.add(xx)
            + *top.add(rx)
            + *mid.add(lx)
            + *mid.add(xx)
            + *mid.add(rx)
            + *bot.add(lx)
            + *bot.add(xx)
            + *bot.add(rx);
        *dst.add(xx) = v / 9.0;
    }
    if border_start == 1 {
        let lx = 0usize;
        let rx = 1usize.min(w - 1);
        let v = *top.add(lx)
            + *top.add(0)
            + *top.add(rx)
            + *mid.add(lx)
            + *mid.add(0)
            + *mid.add(rx)
            + *bot.add(lx)
            + *bot.add(0)
            + *bot.add(rx);
        *dst.add(0) = v / 9.0;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn box_3x3_direct_f32_sse(src: &[f32], out: &mut [f32], h: usize, w: usize) {
    let sp = src.as_ptr();
    let dp = out.as_mut_ptr();

    if h * w >= RAYON_THRESHOLD {
        let sp = sp as usize;
        let dp = dp as usize;
        (0..h).into_par_iter().for_each(|y| {
            let sp = sp as *const f32;
            let dp = dp as *mut f32;
            let top = if y > 0 { sp.add((y - 1) * w) } else { sp };
            let mid = sp.add(y * w);
            let bot = if y + 1 < h {
                sp.add((y + 1) * w)
            } else {
                sp.add((h - 1) * w)
            };
            box_row_f32_sse(top, mid, bot, dp.add(y * w), w);
        });
        return;
    }

    box_row_f32_sse(sp, sp, if h > 1 { sp.add(w) } else { sp }, dp, w);
    for y in 1..h.saturating_sub(1) {
        box_row_f32_sse(
            sp.add((y - 1) * w),
            sp.add(y * w),
            sp.add((y + 1) * w),
            dp.add(y * w),
            w,
        );
    }
    if h > 1 {
        box_row_f32_sse(
            sp.add((h - 2) * w),
            sp.add((h - 1) * w),
            sp.add((h - 1) * w),
            dp.add((h - 1) * w),
            w,
        );
    }
}

// ── Dilate 3x3 f32 ────────────────────────────────────────────────────────

/// 3x3 dilation (max) on single-channel f32 image.
pub fn dilate_3x3_f32(input: &ImageF32) -> Option<ImageF32> {
    if input.channels() != 1 {
        return None;
    }
    let (h, w) = (input.height(), input.width());
    if h < 3 || w < 3 {
        return Some(input.clone());
    }
    let src = input.data();

    let mut out = vec![0.0f32; h * w];

    #[cfg(target_arch = "aarch64")]
    if !cfg!(miri) && std::arch::is_aarch64_feature_detected!("neon") {
        // SAFETY: ISA guard (feature detection) above.
        unsafe {
            dilate_3x3_direct_f32_neon(src, &mut out, h, w);
        }
        return ImageF32::new(out, h, w, 1);
    }

    #[cfg(target_arch = "x86_64")]
    if !cfg!(miri) && is_x86_feature_detected!("sse2") {
        // SAFETY: ISA guard (feature detection) above.
        unsafe {
            dilate_3x3_direct_f32_sse(src, &mut out, h, w);
        }
        return ImageF32::new(out, h, w, 1);
    }

    // Scalar fallback
    let border = f32::NEG_INFINITY;
    for y in 0..h {
        let ay = if y > 0 { y - 1 } else { 0 };
        let by = if y + 1 < h { y + 1 } else { h - 1 };
        let top = &src[ay * w..(ay + 1) * w];
        let mid = &src[y * w..(y + 1) * w];
        let bot = &src[by * w..(by + 1) * w];
        let dst = &mut out[y * w..(y + 1) * w];
        for x in 0..w {
            let lx = if x > 0 { x - 1 } else { 0 };
            let rx = if x + 1 < w { x + 1 } else { w - 1 };
            let mut mx = top[lx].max(top[x]).max(top[rx]);
            mx = mx.max(mid[lx]).max(mid[x]).max(mid[rx]);
            mx = mx.max(bot[lx]).max(bot[x]).max(bot[rx]);
            dst[x] = mx;
        }
    }
    let _ = border;
    ImageF32::new(out, h, w, 1)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn dilate_row_f32_neon(
    top: *const f32,
    mid: *const f32,
    bot: *const f32,
    dst: *mut f32,
    w: usize,
) {
    use std::arch::aarch64::*;

    let border_v = vdupq_n_f32(f32::NEG_INFINITY);
    let mut prev_t = border_v;
    let mut prev_m = border_v;
    let mut prev_b = border_v;
    let mut cur_t = vld1q_f32(top);
    let mut cur_m = vld1q_f32(mid);
    let mut cur_b = vld1q_f32(bot);

    let mut x = 0usize;
    let main_end = w.saturating_sub(4);
    while x < main_end {
        let nxt_t = vld1q_f32(top.add(x + 4));
        let nxt_m = vld1q_f32(mid.add(x + 4));
        let nxt_b = vld1q_f32(bot.add(x + 4));

        let r0 = vmaxq_f32(
            vmaxq_f32(vextq_f32::<3>(prev_t, cur_t), cur_t),
            vextq_f32::<1>(cur_t, nxt_t),
        );
        let r1 = vmaxq_f32(
            vmaxq_f32(vextq_f32::<3>(prev_m, cur_m), cur_m),
            vextq_f32::<1>(cur_m, nxt_m),
        );
        let r2 = vmaxq_f32(
            vmaxq_f32(vextq_f32::<3>(prev_b, cur_b), cur_b),
            vextq_f32::<1>(cur_b, nxt_b),
        );
        vst1q_f32(dst.add(x), vmaxq_f32(vmaxq_f32(r0, r1), r2));

        prev_t = cur_t;
        cur_t = nxt_t;
        prev_m = cur_m;
        cur_m = nxt_m;
        prev_b = cur_b;
        cur_b = nxt_b;
        x += 4;
    }
    if x < w {
        let r0 = vmaxq_f32(
            vmaxq_f32(vextq_f32::<3>(prev_t, cur_t), cur_t),
            vextq_f32::<1>(cur_t, border_v),
        );
        let r1 = vmaxq_f32(
            vmaxq_f32(vextq_f32::<3>(prev_m, cur_m), cur_m),
            vextq_f32::<1>(cur_m, border_v),
        );
        let r2 = vmaxq_f32(
            vmaxq_f32(vextq_f32::<3>(prev_b, cur_b), cur_b),
            vextq_f32::<1>(cur_b, border_v),
        );
        vst1q_f32(dst.add(x), vmaxq_f32(vmaxq_f32(r0, r1), r2));
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn dilate_3x3_direct_f32_neon(src: &[f32], out: &mut [f32], h: usize, w: usize) {
    let sp = src.as_ptr();
    let dp = out.as_mut_ptr();
    // Use row 0 as border (replicate) for dilate
    // Actually for dilate border should be -inf, but for image data replicate is standard.
    // We'll use the same row-replicate as gaussian for consistency.

    #[cfg(target_os = "macos")]
    if h * w >= RAYON_THRESHOLD {
        let sp = sp as usize;
        let dp = dp as usize;
        gcd::parallel_for(h, |y| {
            let sp = sp as *const f32;
            let dp = dp as *mut f32;
            let top = if y > 0 { sp.add((y - 1) * w) } else { sp };
            let mid = sp.add(y * w);
            let bot = if y + 1 < h {
                sp.add((y + 1) * w)
            } else {
                sp.add((h - 1) * w)
            };
            dilate_row_f32_neon(top, mid, bot, dp.add(y * w), w);
        });
        return;
    }

    if h * w >= RAYON_THRESHOLD {
        let sp = sp as usize;
        let dp = dp as usize;
        (0..h).into_par_iter().for_each(|y| {
            let sp = sp as *const f32;
            let dp = dp as *mut f32;
            let top = if y > 0 { sp.add((y - 1) * w) } else { sp };
            let mid = sp.add(y * w);
            let bot = if y + 1 < h {
                sp.add((y + 1) * w)
            } else {
                sp.add((h - 1) * w)
            };
            dilate_row_f32_neon(top, mid, bot, dp.add(y * w), w);
        });
        return;
    }

    dilate_row_f32_neon(sp, sp, if h > 1 { sp.add(w) } else { sp }, dp, w);
    for y in 1..h.saturating_sub(1) {
        dilate_row_f32_neon(
            sp.add((y - 1) * w),
            sp.add(y * w),
            sp.add((y + 1) * w),
            dp.add(y * w),
            w,
        );
    }
    if h > 1 {
        dilate_row_f32_neon(
            sp.add((h - 2) * w),
            sp.add((h - 1) * w),
            sp.add((h - 1) * w),
            dp.add((h - 1) * w),
            w,
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn dilate_row_f32_sse(
    top: *const f32,
    mid: *const f32,
    bot: *const f32,
    dst: *mut f32,
    w: usize,
) {
    use std::arch::x86_64::*;
    let mut x = 0usize;
    if w >= 6 {
        x = 1;
        while x + 4 <= w.saturating_sub(1) {
            let r0 = _mm_max_ps(
                _mm_max_ps(_mm_loadu_ps(top.add(x - 1)), _mm_loadu_ps(top.add(x))),
                _mm_loadu_ps(top.add(x + 1)),
            );
            let r1 = _mm_max_ps(
                _mm_max_ps(_mm_loadu_ps(mid.add(x - 1)), _mm_loadu_ps(mid.add(x))),
                _mm_loadu_ps(mid.add(x + 1)),
            );
            let r2 = _mm_max_ps(
                _mm_max_ps(_mm_loadu_ps(bot.add(x - 1)), _mm_loadu_ps(bot.add(x))),
                _mm_loadu_ps(bot.add(x + 1)),
            );
            _mm_storeu_ps(dst.add(x), _mm_max_ps(_mm_max_ps(r0, r1), r2));
            x += 4;
        }
    }
    let border_start = if x == 0 { 0 } else { x };
    for xx in border_start..w {
        let lx = if xx > 0 { xx - 1 } else { 0 };
        let rx = if xx + 1 < w { xx + 1 } else { w - 1 };
        let mut mx = (*top.add(lx)).max(*top.add(xx)).max(*top.add(rx));
        mx = mx.max(*mid.add(lx)).max(*mid.add(xx)).max(*mid.add(rx));
        mx = mx.max(*bot.add(lx)).max(*bot.add(xx)).max(*bot.add(rx));
        *dst.add(xx) = mx;
    }
    if border_start == 1 {
        let lx = 0usize;
        let rx = 1usize.min(w - 1);
        let mut mx = (*top.add(lx)).max(*top.add(0)).max(*top.add(rx));
        mx = mx.max(*mid.add(lx)).max(*mid.add(0)).max(*mid.add(rx));
        mx = mx.max(*bot.add(lx)).max(*bot.add(0)).max(*bot.add(rx));
        *dst.add(0) = mx;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn dilate_3x3_direct_f32_sse(src: &[f32], out: &mut [f32], h: usize, w: usize) {
    let sp = src.as_ptr();
    let dp = out.as_mut_ptr();

    if h * w >= RAYON_THRESHOLD {
        let sp = sp as usize;
        let dp = dp as usize;
        (0..h).into_par_iter().for_each(|y| {
            let sp = sp as *const f32;
            let dp = dp as *mut f32;
            let top = if y > 0 { sp.add((y - 1) * w) } else { sp };
            let mid = sp.add(y * w);
            let bot = if y + 1 < h {
                sp.add((y + 1) * w)
            } else {
                sp.add((h - 1) * w)
            };
            dilate_row_f32_sse(top, mid, bot, dp.add(y * w), w);
        });
        return;
    }

    dilate_row_f32_sse(sp, sp, if h > 1 { sp.add(w) } else { sp }, dp, w);
    for y in 1..h.saturating_sub(1) {
        dilate_row_f32_sse(
            sp.add((y - 1) * w),
            sp.add(y * w),
            sp.add((y + 1) * w),
            dp.add(y * w),
            w,
        );
    }
    if h > 1 {
        dilate_row_f32_sse(
            sp.add((h - 2) * w),
            sp.add((h - 1) * w),
            sp.add((h - 1) * w),
            dp.add((h - 1) * w),
            w,
        );
    }
}

// ── Sobel 3x3 f32 ─────────────────────────────────────────────────────────

/// 3x3 Sobel gradient magnitude on single-channel f32 image.
/// Returns `|Gx| + |Gy|` approximation. Border pixels are zero.
pub fn sobel_3x3_f32(input: &ImageF32) -> Option<ImageF32> {
    if input.channels() != 1 {
        return None;
    }
    let (h, w) = (input.height(), input.width());
    if h < 3 || w < 3 {
        return Some(ImageF32::zeros(h, w, 1));
    }
    let src = input.data();

    let mut out = vec![0.0f32; h * w];

    let use_rayon = h * w >= RAYON_THRESHOLD && !cfg!(miri);
    let interior_h = h - 2;

    let process_row = |row0: &[f32], row1: &[f32], row2: &[f32], dst: &mut [f32]| {
        let done = sobel_f32_simd_row(row0, row1, row2, dst, w);
        for x in done..w - 1 {
            if x == 0 {
                continue;
            }
            let gx = row0[x + 1] - row0[x - 1] + 2.0 * (row1[x + 1] - row1[x - 1]) + row2[x + 1]
                - row2[x - 1];
            let gy = row2[x - 1] + 2.0 * row2[x] + row2[x + 1]
                - row0[x - 1]
                - 2.0 * row0[x]
                - row0[x + 1];
            dst[x] = gx.abs() + gy.abs();
        }
    };

    if use_rayon && interior_h > 1 {
        out[w..w + interior_h * w]
            .par_chunks_mut(w)
            .enumerate()
            .for_each(|(i, dst)| {
                let y = i + 1;
                let row0 = &src[(y - 1) * w..y * w];
                let row1 = &src[y * w..(y + 1) * w];
                let row2 = &src[(y + 1) * w..(y + 2) * w];
                process_row(row0, row1, row2, dst);
            });
    } else {
        for y in 1..h - 1 {
            let row0 = &src[(y - 1) * w..y * w];
            let row1 = &src[y * w..(y + 1) * w];
            let row2 = &src[(y + 1) * w..(y + 2) * w];
            let dst = &mut out[y * w..(y + 1) * w];
            process_row(row0, row1, row2, dst);
        }
    }

    ImageF32::new(out, h, w, 1)
}

fn sobel_f32_simd_row(
    row0: &[f32],
    row1: &[f32],
    row2: &[f32],
    dst: &mut [f32],
    w: usize,
) -> usize {
    if cfg!(miri) || w < 6 {
        return 1;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: ISA guard (feature detection) above.
            return unsafe { sobel_f32_neon_row(row0, row1, row2, dst, w) };
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse2") {
            // SAFETY: ISA guard (feature detection) above.
            return unsafe { sobel_f32_sse_row(row0, row1, row2, dst, w) };
        }
    }
    1
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn sobel_f32_neon_row(
    row0: &[f32],
    row1: &[f32],
    row2: &[f32],
    dst: &mut [f32],
    w: usize,
) -> usize {
    use std::arch::aarch64::*;
    let two = vdupq_n_f32(2.0);
    let p0 = row0.as_ptr();
    let p1 = row1.as_ptr();
    let p2 = row2.as_ptr();
    let dp = dst.as_mut_ptr();

    let mut x = 1usize;
    while x + 4 <= w - 1 {
        let t_l = vld1q_f32(p0.add(x - 1));
        let t_c = vld1q_f32(p0.add(x));
        let t_r = vld1q_f32(p0.add(x + 1));
        let m_l = vld1q_f32(p1.add(x - 1));
        let m_r = vld1q_f32(p1.add(x + 1));
        let b_l = vld1q_f32(p2.add(x - 1));
        let b_c = vld1q_f32(p2.add(x));
        let b_r = vld1q_f32(p2.add(x + 1));

        // Gx = t_r - t_l + 2*(m_r - m_l) + b_r - b_l
        let gx = vaddq_f32(
            vaddq_f32(vsubq_f32(t_r, t_l), vsubq_f32(b_r, b_l)),
            vmulq_f32(vsubq_f32(m_r, m_l), two),
        );
        // Gy = b_l + 2*b_c + b_r - t_l - 2*t_c - t_r
        let gy = vsubq_f32(
            vfmaq_f32(vaddq_f32(b_l, b_r), b_c, two),
            vfmaq_f32(vaddq_f32(t_l, t_r), t_c, two),
        );
        let mag = vaddq_f32(vabsq_f32(gx), vabsq_f32(gy));
        vst1q_f32(dp.add(x), mag);
        x += 4;
    }
    x
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn sobel_f32_sse_row(
    row0: &[f32],
    row1: &[f32],
    row2: &[f32],
    dst: &mut [f32],
    w: usize,
) -> usize {
    use std::arch::x86_64::*;
    let two = _mm_set1_ps(2.0);
    let sign_mask = _mm_set1_ps(-0.0); // sign bit mask for abs
    let p0 = row0.as_ptr();
    let p1 = row1.as_ptr();
    let p2 = row2.as_ptr();
    let dp = dst.as_mut_ptr();

    let mut x = 1usize;
    while x + 4 <= w - 1 {
        let t_l = _mm_loadu_ps(p0.add(x - 1));
        let t_c = _mm_loadu_ps(p0.add(x));
        let t_r = _mm_loadu_ps(p0.add(x + 1));
        let m_l = _mm_loadu_ps(p1.add(x - 1));
        let m_r = _mm_loadu_ps(p1.add(x + 1));
        let b_l = _mm_loadu_ps(p2.add(x - 1));
        let b_c = _mm_loadu_ps(p2.add(x));
        let b_r = _mm_loadu_ps(p2.add(x + 1));

        let gx = _mm_add_ps(
            _mm_add_ps(_mm_sub_ps(t_r, t_l), _mm_sub_ps(b_r, b_l)),
            _mm_mul_ps(_mm_sub_ps(m_r, m_l), two),
        );
        let gy = _mm_sub_ps(
            _mm_add_ps(_mm_add_ps(b_l, b_r), _mm_mul_ps(b_c, two)),
            _mm_add_ps(_mm_add_ps(t_l, t_r), _mm_mul_ps(t_c, two)),
        );
        let abs_gx = _mm_andnot_ps(sign_mask, gx);
        let abs_gy = _mm_andnot_ps(sign_mask, gy);
        _mm_storeu_ps(dp.add(x), _mm_add_ps(abs_gx, abs_gy));
        x += 4;
    }
    x
}

// ── Threshold binary f32 ──────────────────────────────────────────────────

/// Binary threshold on single-channel f32 image.
/// Pixels > thresh become max_val, otherwise 0.
#[allow(clippy::uninit_vec)]
pub fn threshold_binary_f32(input: &ImageF32, thresh: f32, max_val: f32) -> Option<ImageF32> {
    if input.channels() != 1 {
        return None;
    }
    let (h, w) = (input.height(), input.width());
    let src = input.data();
    let total = h * w;

    let mut out: Vec<f32> = Vec::with_capacity(total);
    #[allow(unsafe_code)]
    // SAFETY: every element written by SIMD + scalar tail below.
    unsafe {
        out.set_len(total);
    }

    // GCD parallel in chunks (bandwidth-bound, trivial compute)
    #[cfg(target_os = "macos")]
    if total >= RAYON_THRESHOLD && !cfg!(miri) {
        let n_chunks = 8usize.min(h);
        let chunk_h = h.div_ceil(n_chunks);
        let sp = super::SendConstPtr(src.as_ptr());
        let dp = super::SendPtr(out.as_mut_ptr());
        gcd::parallel_for(n_chunks, |chunk| {
            let sp = sp.ptr();
            let dp = dp.ptr();
            let start = chunk * chunk_h * w;
            let end = (((chunk + 1) * chunk_h).min(h)) * w;
            let n = end - start;
            // SAFETY: pointer and length from validated image data; parallel chunks are non-overlapping.
            let chunk_src = unsafe { core::slice::from_raw_parts(sp.add(start), n) };
            let chunk_dst = unsafe { core::slice::from_raw_parts_mut(dp.add(start), n) };
            let done = threshold_f32_simd(chunk_src, chunk_dst, n, thresh, max_val);
            for i in done..n {
                chunk_dst[i] = if chunk_src[i] > thresh { max_val } else { 0.0 };
            }
        });
        return ImageF32::new(out, h, w, 1);
    }

    let done = threshold_f32_simd(src, &mut out, total, thresh, max_val);
    for i in done..total {
        out[i] = if src[i] > thresh { max_val } else { 0.0 };
    }

    ImageF32::new(out, h, w, 1)
}

fn threshold_f32_simd(
    src: &[f32],
    dst: &mut [f32],
    total: usize,
    thresh: f32,
    max_val: f32,
) -> usize {
    if cfg!(miri) || total < 4 {
        return 0;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: ISA guard (feature detection) above.
            return unsafe { threshold_f32_neon(src, dst, total, thresh, max_val) };
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse2") {
            // SAFETY: ISA guard (feature detection) above.
            return unsafe { threshold_f32_sse(src, dst, total, thresh, max_val) };
        }
    }
    0
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn threshold_f32_neon(
    src: &[f32],
    dst: &mut [f32],
    total: usize,
    thresh: f32,
    max_val: f32,
) -> usize {
    use std::arch::aarch64::*;
    let thresh_v = vdupq_n_f32(thresh);
    let max_v = vdupq_n_f32(max_val);
    let zero = vdupq_n_f32(0.0);
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();
    let mut i = 0usize;
    while i + 4 <= total {
        let v = vld1q_f32(sp.add(i));
        let mask = vcgtq_f32(v, thresh_v); // v > thresh → all-ones
        let result = vbslq_f32(mask, max_v, zero);
        vst1q_f32(dp.add(i), result);
        i += 4;
    }
    i
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn threshold_f32_sse(
    src: &[f32],
    dst: &mut [f32],
    total: usize,
    thresh: f32,
    max_val: f32,
) -> usize {
    use std::arch::x86_64::*;
    let thresh_v = _mm_set1_ps(thresh);
    let max_v = _mm_set1_ps(max_val);
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();
    let mut i = 0usize;
    while i + 4 <= total {
        let v = _mm_loadu_ps(sp.add(i));
        let mask = _mm_cmpgt_ps(v, thresh_v); // v > thresh → all-ones
        let result = _mm_and_ps(mask, max_v);
        _mm_storeu_ps(dp.add(i), result);
        i += 4;
    }
    i
}
