//! u8 filter operations: grayscale, dilate, erode, gaussian, box_blur, sobel, median.
//!
//! # Safety contract
//!
//! Unsafe code categories:
//! 1. **SIMD intrinsics (NEON / SSE2)** — ISA guard via runtime detection or `#[target_feature]`.
//! 2. **`SendConstPtr` / `SendPtr` for rayon** — each chunk writes non-overlapping rows.
//! 3. **Pointer arithmetic in row kernels** — bounded by image width validated at entry.
#![allow(unsafe_code)]

use super::u8ops::{ImageU8, RAYON_THRESHOLD, gcd};
use rayon::prelude::*;

pub fn grayscale_u8(input: &ImageU8) -> Option<ImageU8> {
    if input.channels() != 3 {
        return None;
    }
    let (h, w) = (input.height(), input.width());
    let src = input.data();
    let mut out = vec![0u8; h * w];
    let total = h * w;

    // For large images, parallelize across row chunks to amortize dispatch overhead.
    // Each chunk processes multiple rows to ensure sufficient work per task.
    // For grayscale, the NEON inner loop is very fast (~1 cycle/pixel), so
    // parallelism only helps for large images. Use a higher threshold than the
    // global RAYON_THRESHOLD to avoid dispatch overhead dominating.
    // WHY 500K: grayscale NEON runs ~1 cycle/pixel; below 500K pixels, dispatch overhead dominates.
    const GRAY_PAR_THRESHOLD: usize = 500_000;
    const MIN_PIXELS_PER_CHUNK: usize = 65536; // ~64K pixels per chunk
    if total >= GRAY_PAR_THRESHOLD && !cfg!(miri) {
        let rows_per_chunk = (MIN_PIXELS_PER_CHUNK / w).max(1);
        let num_chunks = h.div_ceil(rows_per_chunk);
        let sp = super::SendConstPtr(src.as_ptr());
        let dp = super::SendPtr(out.as_mut_ptr());
        let w_c = w;
        let h_c = h;
        gcd::parallel_for(num_chunks, |chunk_idx| {
            let y_start = chunk_idx * rows_per_chunk;
            let y_end = (y_start + rows_per_chunk).min(h_c);
            let chunk_pixels = (y_end - y_start) * w_c;
            // SAFETY: pointer and length from validated image data; parallel chunks are non-overlapping.
            let src_chunk = unsafe {
                std::slice::from_raw_parts(sp.ptr().add(y_start * w_c * 3), chunk_pixels * 3)
            };
            // SAFETY: pointer and length from validated image data; parallel chunks are non-overlapping.
            let dst_chunk = unsafe {
                std::slice::from_raw_parts_mut(dp.ptr().add(y_start * w_c), chunk_pixels)
            };
            let done = grayscale_u8_simd_row(src_chunk, dst_chunk);
            for x in done..chunk_pixels {
                let base = x * 3;
                let r = src_chunk[base] as u16;
                let g = src_chunk[base + 1] as u16;
                let b = src_chunk[base + 2] as u16;
                // WHY 77,150,29: BT.601 luma (0.299R+0.587G+0.114B) in Q8 fixed-point (77/256~0.301, 150/256~0.586, 29/256~0.113).
                dst_chunk[x] = ((77 * r + 150 * g + 29 * b + 128) >> 8) as u8;
            }
        });
        return ImageU8::new(out, h, w, 1);
    }

    // Flat processing — optimal for small images (no threading overhead).
    let done = grayscale_u8_simd_row(src, &mut out);
    for x in done..total {
        let base = x * 3;
        let r = src[base] as u16;
        let g = src[base + 1] as u16;
        let b = src[base + 2] as u16;
        out[x] = ((77 * r + 150 * g + 29 * b + 128) >> 8) as u8;
    }

    ImageU8::new(out, h, w, 1)
}

fn grayscale_u8_simd_row(src: &[u8], dst: &mut [u8]) -> usize {
    if cfg!(miri) {
        return 0;
    }
    let w = dst.len();
    if w < 8 {
        return 0;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: ISA guard (feature detection) above.
            return unsafe { grayscale_u8_neon(src, dst, w) };
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: ISA guard (feature detection) above.
            return unsafe { grayscale_u8_avx2(src, dst, w) };
        }
        if is_x86_feature_detected!("ssse3") {
            // SAFETY: ISA guard (feature detection) above.
            return unsafe { grayscale_u8_sse(src, dst, w) };
        }
    }
    0
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn grayscale_u8_neon(src: &[u8], dst: &mut [u8], w: usize) -> usize {
    use std::arch::aarch64::*;

    // Integer approximation: gray = (77*R + 150*G + 29*B + 128) >> 8
    // which approximates BT.601: 0.299*R + 0.587*G + 0.114*B
    // The +128 provides rounding to nearest instead of truncation.

    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();
    let mut x = 0usize;

    // Splat coefficient vectors once. We use vmulq_u16/vmlaq_u16 with splatted
    // vectors because vmulq_n_u16 is not available in std::arch::aarch64.
    let coeff_r = vdupq_n_u16(77);
    let coeff_g = vdupq_n_u16(150);
    let coeff_b = vdupq_n_u16(29);

    // Inline helper: convert 16 RGB pixels to grayscale
    #[inline(always)]
    unsafe fn gray16(
        sp: *const u8,
        dp: *mut u8,
        x: usize,
        coeff_r: uint16x8_t,
        coeff_g: uint16x8_t,
        coeff_b: uint16x8_t,
    ) {
        let rgb = vld3q_u8(sp.add(x * 3));
        let r_lo = vmovl_u8(vget_low_u8(rgb.0));
        let g_lo = vmovl_u8(vget_low_u8(rgb.1));
        let b_lo = vmovl_u8(vget_low_u8(rgb.2));
        let acc_lo = vmlaq_u16(
            vmlaq_u16(vmulq_u16(r_lo, coeff_r), g_lo, coeff_g),
            b_lo,
            coeff_b,
        );
        let r_hi = vmovl_u8(vget_high_u8(rgb.0));
        let g_hi = vmovl_u8(vget_high_u8(rgb.1));
        let b_hi = vmovl_u8(vget_high_u8(rgb.2));
        let acc_hi = vmlaq_u16(
            vmlaq_u16(vmulq_u16(r_hi, coeff_r), g_hi, coeff_g),
            b_hi,
            coeff_b,
        );
        vst1q_u8(
            dp.add(x),
            vcombine_u8(vrshrn_n_u16::<8>(acc_lo), vrshrn_n_u16::<8>(acc_hi)),
        );
    }

    // Process 64 pixels per iteration (4x unrolled) — hides 3-4 cycle multiply
    // latency by allowing the CPU to pipeline loads across 4 batches.
    while x + 64 <= w {
        gray16(sp, dp, x, coeff_r, coeff_g, coeff_b);
        gray16(sp, dp, x + 16, coeff_r, coeff_g, coeff_b);
        gray16(sp, dp, x + 32, coeff_r, coeff_g, coeff_b);
        gray16(sp, dp, x + 48, coeff_r, coeff_g, coeff_b);
        x += 64;
    }

    // Handle remaining pixels in 16-pixel chunks
    while x + 16 <= w {
        gray16(sp, dp, x, coeff_r, coeff_g, coeff_b);
        x += 16;
    }

    // Process remaining 8 pixels if available using vld3_u8
    if x + 8 <= w {
        let rgb = vld3_u8(sp.add(x * 3));
        let r8 = vmovl_u8(rgb.0);
        let g8 = vmovl_u8(rgb.1);
        let b8 = vmovl_u8(rgb.2);

        let acc = vmlaq_u16(vmlaq_u16(vmulq_u16(r8, coeff_r), g8, coeff_g), b8, coeff_b);
        let result = vrshrn_n_u16::<8>(acc);
        vst1_u8(dp.add(x), result);
        x += 8;
    }

    x
}

// ============================================================================
// Dilate 3x3 (u8, single-channel)
// ============================================================================

/// 3x3 dilation on single-channel u8 image.
pub fn dilate_3x3_u8(input: &ImageU8) -> Option<ImageU8> {
    morph_3x3_separable(input, true)
}

// ============================================================================
// Erode 3x3 (u8, single-channel)
// ============================================================================

/// 3x3 erosion on single-channel u8 image.
pub fn erode_3x3_u8(input: &ImageU8) -> Option<ImageU8> {
    morph_3x3_separable(input, false)
}

/// Fused single-pass 3x3 morphology using a ring buffer of 3 H-pass rows.
/// Dispatches once to a specialized NEON function (max or min) to avoid
/// per-row dynamic dispatch overhead.
fn morph_3x3_separable(input: &ImageU8, is_dilate: bool) -> Option<ImageU8> {
    if input.channels() != 1 {
        return None;
    }
    let (h, w) = (input.height(), input.width());
    if h < 3 || w < 3 {
        return Some(input.clone());
    }
    let src = input.data();

    // WHY 0 for dilate, 255 for erode: identity elements of max/min — max(x,0)=x, min(x,255)=x.
    let border = if is_dilate { 0u8 } else { 255u8 };
    let mut out = vec![0u8; h * w];

    #[cfg(target_arch = "aarch64")]
    if !cfg!(miri) && std::arch::is_aarch64_feature_detected!("neon") {
        // SAFETY: ISA guard (feature detection) above.
        unsafe {
            if is_dilate {
                morph_3x3_direct_neon::<true>(src, &mut out, h, w);
            } else {
                morph_3x3_direct_neon::<false>(src, &mut out, h, w);
            }
        }
        return ImageU8::new(out, h, w, 1);
    }

    #[cfg(target_arch = "x86_64")]
    if !cfg!(miri) && is_x86_feature_detected!("avx2") {
        // SAFETY: ISA guard (feature detection) above.
        unsafe {
            if is_dilate {
                morph_3x3_direct_avx2::<true>(src, &mut out, h, w);
            } else {
                morph_3x3_direct_avx2::<false>(src, &mut out, h, w);
            }
        }
        return ImageU8::new(out, h, w, 1);
    }

    #[cfg(target_arch = "x86_64")]
    if !cfg!(miri) && is_x86_feature_detected!("ssse3") {
        // SAFETY: ISA guard (feature detection) above.
        unsafe {
            if is_dilate {
                morph_3x3_direct_sse::<true>(src, &mut out, h, w);
            } else {
                morph_3x3_direct_sse::<false>(src, &mut out, h, w);
            }
        }
        return ImageU8::new(out, h, w, 1);
    }

    // Scalar fallback
    let cmp: fn(u8, u8) -> u8 = if is_dilate { u8::max } else { u8::min };
    let mut ring = vec![border; 3 * w];
    let border_row = vec![border; w];

    let morph_h_row = |row: &[u8], dst: &mut [u8]| {
        dst[0] = cmp(cmp(border, row[0]), row[1.min(w - 1)]);
        for x in 1..w.saturating_sub(1) {
            dst[x] = cmp(cmp(row[x - 1], row[x]), row[x + 1]);
        }
        if w > 1 {
            dst[w - 1] = cmp(cmp(row[w - 2], row[w - 1]), border);
        }
    };

    morph_h_row(&src[0..w], &mut ring[0..w]);
    for y in 0..h {
        let slot_next = ((y + 1) % 3) * w;
        if y + 1 < h {
            morph_h_row(
                &src[(y + 1) * w..(y + 2) * w],
                &mut ring[slot_next..slot_next + w],
            );
        } else {
            ring[slot_next..slot_next + w].fill(border);
        }
        let above = if y == 0 {
            &border_row[..]
        } else {
            let s = ((y - 1) % 3) * w;
            &ring[s..s + w]
        };
        let center_s = (y % 3) * w;
        let center = &ring[center_s..center_s + w];
        let below_s = ((y + 1) % 3) * w;
        let below = &ring[below_s..below_s + w];
        let out_row = &mut out[y * w..(y + 1) * w];
        for x in 0..w {
            out_row[x] = cmp(cmp(above[x], center[x]), below[x]);
        }
    }
    ImageU8::new(out, h, w, 1)
}

/// Process one morph row — branchless NEON inner loop with 2x unrolling.
/// Zero branches in steady state.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn morph_row_neon<const IS_DILATE: bool>(
    top: *const u8,
    mid: *const u8,
    bot: *const u8,
    dst: *mut u8,
    w: usize,
) {
    use std::arch::aarch64::*;

    #[inline(always)]
    unsafe fn cmp<const D: bool>(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        if D { vmaxq_u8(a, b) } else { vminq_u8(a, b) }
    }

    // Scalar fallback for very narrow images where SIMD would read/write past buffer
    if w < 16 {
        let border_val: u8 = if IS_DILATE { 0 } else { 255 };
        for x in 0..w {
            let mut best = border_val;
            for row in [top, mid, bot] {
                let l = if x > 0 { *row.add(x - 1) } else { border_val };
                let c = *row.add(x);
                let r = if x + 1 < w {
                    *row.add(x + 1)
                } else {
                    border_val
                };
                let row_best = if IS_DILATE {
                    l.max(c).max(r)
                } else {
                    l.min(c).min(r)
                };
                best = if IS_DILATE {
                    best.max(row_best)
                } else {
                    best.min(row_best)
                };
            }
            *dst.add(x) = best;
        }
        return;
    }

    let border_v = vdupq_n_u8(if IS_DILATE { 0 } else { 255 });
    let mut prev_t = border_v;
    let mut prev_m = border_v;
    let mut prev_b = border_v;
    let mut cur_t = vld1q_u8(top);
    let mut cur_m = vld1q_u8(mid);
    let mut cur_b = vld1q_u8(bot);

    let mut x = 0usize;
    let main_end = w.saturating_sub(16);
    // 2x unrolled main loop — hides vext latency by interleaving two iterations
    let unroll_end = if main_end >= 32 { main_end - 16 } else { 0 };
    while x < unroll_end {
        // Iteration 1
        let nxt_t = vld1q_u8(top.add(x + 16));
        let nxt_m = vld1q_u8(mid.add(x + 16));
        let nxt_b = vld1q_u8(bot.add(x + 16));

        let r0 = cmp::<IS_DILATE>(
            cmp::<IS_DILATE>(vextq_u8::<15>(prev_t, cur_t), cur_t),
            vextq_u8::<1>(cur_t, nxt_t),
        );
        let r1 = cmp::<IS_DILATE>(
            cmp::<IS_DILATE>(vextq_u8::<15>(prev_m, cur_m), cur_m),
            vextq_u8::<1>(cur_m, nxt_m),
        );
        let r2 = cmp::<IS_DILATE>(
            cmp::<IS_DILATE>(vextq_u8::<15>(prev_b, cur_b), cur_b),
            vextq_u8::<1>(cur_b, nxt_b),
        );
        vst1q_u8(dst.add(x), cmp::<IS_DILATE>(cmp::<IS_DILATE>(r0, r1), r2));

        // Iteration 2 — interleaved to hide load/vext latency
        let nxt2_t = vld1q_u8(top.add(x + 32));
        let nxt2_m = vld1q_u8(mid.add(x + 32));
        let nxt2_b = vld1q_u8(bot.add(x + 32));

        let s0 = cmp::<IS_DILATE>(
            cmp::<IS_DILATE>(vextq_u8::<15>(cur_t, nxt_t), nxt_t),
            vextq_u8::<1>(nxt_t, nxt2_t),
        );
        let s1 = cmp::<IS_DILATE>(
            cmp::<IS_DILATE>(vextq_u8::<15>(cur_m, nxt_m), nxt_m),
            vextq_u8::<1>(nxt_m, nxt2_m),
        );
        let s2 = cmp::<IS_DILATE>(
            cmp::<IS_DILATE>(vextq_u8::<15>(cur_b, nxt_b), nxt_b),
            vextq_u8::<1>(nxt_b, nxt2_b),
        );
        vst1q_u8(
            dst.add(x + 16),
            cmp::<IS_DILATE>(cmp::<IS_DILATE>(s0, s1), s2),
        );

        prev_t = nxt_t;
        cur_t = nxt2_t;
        prev_m = nxt_m;
        cur_m = nxt2_m;
        prev_b = nxt_b;
        cur_b = nxt2_b;
        x += 32;
    }
    // Remaining 16-pixel chunks
    while x < main_end {
        let nxt_t = vld1q_u8(top.add(x + 16));
        let nxt_m = vld1q_u8(mid.add(x + 16));
        let nxt_b = vld1q_u8(bot.add(x + 16));

        let r0 = cmp::<IS_DILATE>(
            cmp::<IS_DILATE>(vextq_u8::<15>(prev_t, cur_t), cur_t),
            vextq_u8::<1>(cur_t, nxt_t),
        );
        let r1 = cmp::<IS_DILATE>(
            cmp::<IS_DILATE>(vextq_u8::<15>(prev_m, cur_m), cur_m),
            vextq_u8::<1>(cur_m, nxt_m),
        );
        let r2 = cmp::<IS_DILATE>(
            cmp::<IS_DILATE>(vextq_u8::<15>(prev_b, cur_b), cur_b),
            vextq_u8::<1>(cur_b, nxt_b),
        );
        vst1q_u8(dst.add(x), cmp::<IS_DILATE>(cmp::<IS_DILATE>(r0, r1), r2));

        prev_t = cur_t;
        cur_t = nxt_t;
        prev_m = cur_m;
        cur_m = nxt_m;
        prev_b = cur_b;
        cur_b = nxt_b;
        x += 16;
    }
    if x < w {
        let r0 = cmp::<IS_DILATE>(
            cmp::<IS_DILATE>(vextq_u8::<15>(prev_t, cur_t), cur_t),
            vextq_u8::<1>(cur_t, border_v),
        );
        let r1 = cmp::<IS_DILATE>(
            cmp::<IS_DILATE>(vextq_u8::<15>(prev_m, cur_m), cur_m),
            vextq_u8::<1>(cur_m, border_v),
        );
        let r2 = cmp::<IS_DILATE>(
            cmp::<IS_DILATE>(vextq_u8::<15>(prev_b, cur_b), cur_b),
            vextq_u8::<1>(cur_b, border_v),
        );
        vst1q_u8(dst.add(x), cmp::<IS_DILATE>(cmp::<IS_DILATE>(r0, r1), r2));
    }
}

/// Direct 3×3 morphology — GCD parallel on macOS, sequential fallback.
/// Eliminates border_row allocation by using vdupq_n_u8 border vectors directly.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn morph_3x3_direct_neon<const IS_DILATE: bool>(
    src: &[u8],
    out: &mut [u8],
    h: usize,
    w: usize,
) {
    let border: u8 = if IS_DILATE { 0 } else { 255 };
    let sp = src.as_ptr();
    let dp = out.as_mut_ptr();
    // Stack-allocated border row for narrow images, heap for wide.
    // Pad to w+32 so NEON 2x-unrolled reads (x+32) are safe.
    let border_heap;
    let border_stack = [border; 1024];
    let bp = if w + 32 <= 1024 {
        border_stack.as_ptr()
    } else {
        border_heap = vec![border; w + 32];
        border_heap.as_ptr()
    };

    // Use chunked parallelism: group rows into chunks to amortize dispatch overhead.
    const MIN_ROWS_PER_CHUNK: usize = 32;

    #[inline(always)]
    unsafe fn process_chunk<const IS_DILATE: bool>(
        sp: *const u8,
        bp: *const u8,
        dp: *mut u8,
        y_start: usize,
        y_end: usize,
        h: usize,
        w: usize,
    ) {
        // Handle first row with border check
        if y_start == 0 && y_end > 0 {
            let bot = if h > 1 { sp.add(w) } else { bp };
            morph_row_neon::<IS_DILATE>(bp, sp, bot, dp, w);
        }
        // Interior rows — no branch per row (hot path)
        let interior_start = if y_start == 0 { 1 } else { y_start };
        let interior_end = if y_end == h { h - 1 } else { y_end };
        for y in interior_start..interior_end {
            morph_row_neon::<IS_DILATE>(
                sp.add((y - 1) * w),
                sp.add(y * w),
                sp.add((y + 1) * w),
                dp.add(y * w),
                w,
            );
        }
        // Handle last row with border check
        if y_end == h && h > 1 {
            morph_row_neon::<IS_DILATE>(
                sp.add((h - 2) * w),
                sp.add((h - 1) * w),
                bp,
                dp.add((h - 1) * w),
                w,
            );
        }
    }

    // Sequential NEON is faster for 640x480 (307K pixels).
    // GCD dispatch overhead (~0.3µs × chunks) exceeds the parallelism gain at this size.
    const MORPH_PAR_THRESHOLD: usize = 500_000;
    if h * w >= MORPH_PAR_THRESHOLD {
        let rows_per_chunk = MIN_ROWS_PER_CHUNK.max(1);
        let num_chunks = h.div_ceil(rows_per_chunk);
        let sp = sp as usize;
        let dp = dp as usize;
        let bp = bp as usize;
        gcd::parallel_for(num_chunks, |chunk_idx| {
            let sp = sp as *const u8;
            let dp = dp as *mut u8;
            let bp = bp as *const u8;
            let y_start = chunk_idx * rows_per_chunk;
            let y_end = (y_start + rows_per_chunk).min(h);
            process_chunk::<IS_DILATE>(sp, bp, dp, y_start, y_end, h, w);
        });
        return;
    }

    // Sequential fallback (small images)
    process_chunk::<IS_DILATE>(sp, bp, dp, 0, h, h, w);
}

// ============================================================================
// x86_64 AVX2 implementations
// ============================================================================

/// AVX2 morph row — loads at offsets -1, 0, +1 and max/min them directly.
/// Processes 32 bytes at a time, avoids cross-lane alignment issues.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn morph_row_avx2<const IS_DILATE: bool>(
    top: *const u8,
    mid: *const u8,
    bot: *const u8,
    dst: *mut u8,
    w: usize,
) {
    use std::arch::x86_64::*;

    #[inline(always)]
    unsafe fn cmp<const D: bool>(a: __m256i, b: __m256i) -> __m256i {
        if D {
            _mm256_max_epu8(a, b)
        } else {
            _mm256_min_epu8(a, b)
        }
    }

    // For narrow images, fall back to SSE or scalar
    if w < 32 {
        let border_val: u8 = if IS_DILATE { 0 } else { 255 };
        for x in 0..w {
            let mut best = border_val;
            for row in [top, mid, bot] {
                let l = if x > 0 { *row.add(x - 1) } else { border_val };
                let c = *row.add(x);
                let r = if x + 1 < w {
                    *row.add(x + 1)
                } else {
                    border_val
                };
                let row_best = if IS_DILATE {
                    l.max(c).max(r)
                } else {
                    l.min(c).min(r)
                };
                best = if IS_DILATE {
                    best.max(row_best)
                } else {
                    best.min(row_best)
                };
            }
            *dst.add(x) = best;
        }
        return;
    }

    let border_val: u8 = if IS_DILATE { 0 } else { 255 };
    let _border_v = _mm256_set1_epi8(border_val as i8);

    // First pixel column is handled specially (left neighbor is border)
    // We process the bulk using offset loads: load at ptr-1, ptr, ptr+1
    let mut x = 0usize;
    let main_end = if w >= 33 { w - 32 } else { 0 };

    // Handle x=0: left neighbors are border
    if main_end > 0 {
        // For x=0, left load would be at ptr-1 which is out of bounds.
        // Use SSE-style approach for first chunk: manually handle border.
        let center_t = _mm256_loadu_si256(top as *const __m256i);
        let center_m = _mm256_loadu_si256(mid as *const __m256i);
        let center_b = _mm256_loadu_si256(bot as *const __m256i);
        let right_t = _mm256_loadu_si256(top.add(1) as *const __m256i);
        let right_m = _mm256_loadu_si256(mid.add(1) as *const __m256i);
        let right_b = _mm256_loadu_si256(bot.add(1) as *const __m256i);

        // For left: shift center right by 1, insert border byte at position 0
        // Use _mm256_alignr_epi8 with border vector — but it's lane-wise.
        // Simpler: blend byte 0 from border, rest from loadu at (ptr-1+1)=ptr
        // Actually just build left by: [border, center[0], center[1], ..., center[30]]
        // Easiest: use palignr on 128-bit halves
        let bv128 = _mm_set1_epi8(border_val as i8);
        let ct_lo = _mm256_castsi256_si128(center_t);
        let ct_hi = _mm256_extracti128_si256(center_t, 1);
        let left_t_lo = _mm_alignr_epi8(ct_lo, bv128, 15); // [border_val, ct[0]..ct[14]]
        let left_t_hi = _mm_alignr_epi8(ct_hi, ct_lo, 15); // [ct[15], ct[16]..ct[30]]
        let left_t = _mm256_set_m128i(left_t_hi, left_t_lo);

        let cm_lo = _mm256_castsi256_si128(center_m);
        let cm_hi = _mm256_extracti128_si256(center_m, 1);
        let left_m_lo = _mm_alignr_epi8(cm_lo, bv128, 15);
        let left_m_hi = _mm_alignr_epi8(cm_hi, cm_lo, 15);
        let left_m = _mm256_set_m128i(left_m_hi, left_m_lo);

        let cb_lo = _mm256_castsi256_si128(center_b);
        let cb_hi = _mm256_extracti128_si256(center_b, 1);
        let left_b_lo = _mm_alignr_epi8(cb_lo, bv128, 15);
        let left_b_hi = _mm_alignr_epi8(cb_hi, cb_lo, 15);
        let left_b = _mm256_set_m128i(left_b_hi, left_b_lo);

        let h_t = cmp::<IS_DILATE>(cmp::<IS_DILATE>(left_t, center_t), right_t);
        let h_m = cmp::<IS_DILATE>(cmp::<IS_DILATE>(left_m, center_m), right_m);
        let h_b = cmp::<IS_DILATE>(cmp::<IS_DILATE>(left_b, center_b), right_b);
        _mm256_storeu_si256(
            dst as *mut __m256i,
            cmp::<IS_DILATE>(cmp::<IS_DILATE>(h_t, h_m), h_b),
        );
        x = 1; // Next chunk starts at x=1
    }

    // Main loop: loads at x-1, x, x+1 are all safe
    while x + 32 <= main_end + 1 {
        let left_t = _mm256_loadu_si256(top.add(x - 1) as *const __m256i);
        let center_t = _mm256_loadu_si256(top.add(x) as *const __m256i);
        let right_t = _mm256_loadu_si256(top.add(x + 1) as *const __m256i);
        let left_m = _mm256_loadu_si256(mid.add(x - 1) as *const __m256i);
        let center_m = _mm256_loadu_si256(mid.add(x) as *const __m256i);
        let right_m = _mm256_loadu_si256(mid.add(x + 1) as *const __m256i);
        let left_b = _mm256_loadu_si256(bot.add(x - 1) as *const __m256i);
        let center_b = _mm256_loadu_si256(bot.add(x) as *const __m256i);
        let right_b = _mm256_loadu_si256(bot.add(x + 1) as *const __m256i);

        let h_t = cmp::<IS_DILATE>(cmp::<IS_DILATE>(left_t, center_t), right_t);
        let h_m = cmp::<IS_DILATE>(cmp::<IS_DILATE>(left_m, center_m), right_m);
        let h_b = cmp::<IS_DILATE>(cmp::<IS_DILATE>(left_b, center_b), right_b);
        _mm256_storeu_si256(
            dst.add(x) as *mut __m256i,
            cmp::<IS_DILATE>(cmp::<IS_DILATE>(h_t, h_m), h_b),
        );
        x += 32;
    }

    // Tail: process last chunk ending at w, handling right border
    if x < w {
        // Process the last (w - x) pixels. Use overlapping store at w-32 if possible.
        let tail_start = w.saturating_sub(32);
        if tail_start >= 1 {
            let left_t = _mm256_loadu_si256(top.add(tail_start - 1) as *const __m256i);
            let center_t = _mm256_loadu_si256(top.add(tail_start) as *const __m256i);
            let left_m = _mm256_loadu_si256(mid.add(tail_start - 1) as *const __m256i);
            let center_m = _mm256_loadu_si256(mid.add(tail_start) as *const __m256i);
            let left_b = _mm256_loadu_si256(bot.add(tail_start - 1) as *const __m256i);
            let center_b = _mm256_loadu_si256(bot.add(tail_start) as *const __m256i);

            // Right: for pixels at w-1, right neighbor is border
            // Load at tail_start+1, but the last byte (position w) may need border
            // Since tail_start = w-32, tail_start+1 would read up to w-32+1+31 = w,
            // which is one past the last valid index. We need to handle this carefully.
            // Load right normally (reads up to tail_start+32 which may be w+1 — out of bounds for last byte)
            // Instead, build right with border for the last position
            let right_t = if tail_start + 32 < w {
                _mm256_loadu_si256(top.add(tail_start + 1) as *const __m256i)
            } else {
                // Safe: tail_start+1 reads 32 bytes = tail_start+33 bytes from start,
                // but w = tail_start+32, so tail_start+1+31 = w which is fine for the
                // last 31 pixels, but position 31 in this vector corresponds to w
                // which needs border value. Use a masked approach:
                let mut buf = [border_val; 32];
                let avail = w - tail_start - 1; // how many valid right-neighbors
                std::ptr::copy_nonoverlapping(
                    top.add(tail_start + 1),
                    buf.as_mut_ptr(),
                    avail.min(32),
                );
                _mm256_loadu_si256(buf.as_ptr() as *const __m256i)
            };
            let right_m = if tail_start + 32 < w {
                _mm256_loadu_si256(mid.add(tail_start + 1) as *const __m256i)
            } else {
                let mut buf = [border_val; 32];
                let avail = w - tail_start - 1;
                std::ptr::copy_nonoverlapping(
                    mid.add(tail_start + 1),
                    buf.as_mut_ptr(),
                    avail.min(32),
                );
                _mm256_loadu_si256(buf.as_ptr() as *const __m256i)
            };
            let right_b = if tail_start + 32 < w {
                _mm256_loadu_si256(bot.add(tail_start + 1) as *const __m256i)
            } else {
                let mut buf = [border_val; 32];
                let avail = w - tail_start - 1;
                std::ptr::copy_nonoverlapping(
                    bot.add(tail_start + 1),
                    buf.as_mut_ptr(),
                    avail.min(32),
                );
                _mm256_loadu_si256(buf.as_ptr() as *const __m256i)
            };

            let h_t = cmp::<IS_DILATE>(cmp::<IS_DILATE>(left_t, center_t), right_t);
            let h_m = cmp::<IS_DILATE>(cmp::<IS_DILATE>(left_m, center_m), right_m);
            let h_b = cmp::<IS_DILATE>(cmp::<IS_DILATE>(left_b, center_b), right_b);
            _mm256_storeu_si256(
                dst.add(tail_start) as *mut __m256i,
                cmp::<IS_DILATE>(cmp::<IS_DILATE>(h_t, h_m), h_b),
            );
        } else {
            // Very narrow tail — scalar
            for px in x..w {
                let mut best = border_val;
                for row in [top, mid, bot] {
                    let l = if px > 0 { *row.add(px - 1) } else { border_val };
                    let c = *row.add(px);
                    let r = if px + 1 < w {
                        *row.add(px + 1)
                    } else {
                        border_val
                    };
                    let row_best = if IS_DILATE {
                        l.max(c).max(r)
                    } else {
                        l.min(c).min(r)
                    };
                    best = if IS_DILATE {
                        best.max(row_best)
                    } else {
                        best.min(row_best)
                    };
                }
                *dst.add(px) = best;
            }
        }
    }
}

/// Direct 3x3 morphology on x86_64 with AVX2 — rayon parallel.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn morph_3x3_direct_avx2<const IS_DILATE: bool>(
    src: &[u8],
    out: &mut [u8],
    h: usize,
    w: usize,
) {
    let border: u8 = if IS_DILATE { 0 } else { 255 };
    let sp = src.as_ptr();
    let dp = out.as_mut_ptr();
    let border_row = vec![border; w + 32]; // +32 padding for safe AVX2 reads
    let bp = border_row.as_ptr();

    if h * w >= RAYON_THRESHOLD {
        let sp = sp as usize;
        let dp = dp as usize;
        let bp = bp as usize;
        (0..h).into_par_iter().for_each(|y| {
            let sp = sp as *const u8;
            let dp = dp as *mut u8;
            let bp = bp as *const u8;
            let top = if y > 0 { sp.add((y - 1) * w) } else { bp };
            let mid = sp.add(y * w);
            let bot = if y + 1 < h { sp.add((y + 1) * w) } else { bp };
            morph_row_avx2::<IS_DILATE>(top, mid, bot, dp.add(y * w), w);
        });
        return;
    }

    morph_row_avx2::<IS_DILATE>(bp, sp, if h > 1 { sp.add(w) } else { bp }, dp, w);
    for y in 1..h.saturating_sub(1) {
        morph_row_avx2::<IS_DILATE>(
            sp.add((y - 1) * w),
            sp.add(y * w),
            sp.add((y + 1) * w),
            dp.add(y * w),
            w,
        );
    }
    if h > 1 {
        morph_row_avx2::<IS_DILATE>(
            sp.add((h - 2) * w),
            sp.add((h - 1) * w),
            bp,
            dp.add((h - 1) * w),
            w,
        );
    }
}

// ============================================================================
// x86_64 SSE/SSSE3 implementations
// ============================================================================

/// SSE morph row — _mm_alignr_epi8 for horizontal shifts (SSSE3), _mm_max/min_epu8 (SSE2).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn morph_row_sse<const IS_DILATE: bool>(
    top: *const u8,
    mid: *const u8,
    bot: *const u8,
    dst: *mut u8,
    w: usize,
) {
    use std::arch::x86_64::*;

    #[inline(always)]
    unsafe fn cmp<const D: bool>(a: __m128i, b: __m128i) -> __m128i {
        if D {
            _mm_max_epu8(a, b)
        } else {
            _mm_min_epu8(a, b)
        }
    }

    let border_v = _mm_set1_epi8(if IS_DILATE { 0 } else { -1i8 }); // 0 or 255
    let mut prev_t = border_v;
    let mut prev_m = border_v;
    let mut prev_b = border_v;
    let mut cur_t = _mm_loadu_si128(top as *const __m128i);
    let mut cur_m = _mm_loadu_si128(mid as *const __m128i);
    let mut cur_b = _mm_loadu_si128(bot as *const __m128i);

    let mut x = 0usize;
    let main_end = w.saturating_sub(16);
    while x < main_end {
        let nxt_t = _mm_loadu_si128(top.add(x + 16) as *const __m128i);
        let nxt_m = _mm_loadu_si128(mid.add(x + 16) as *const __m128i);
        let nxt_b = _mm_loadu_si128(bot.add(x + 16) as *const __m128i);

        // _mm_alignr_epi8::<15>(cur, prev) = [prev[15], cur[0..15]] = left shift
        // _mm_alignr_epi8::<1>(nxt, cur) = [cur[1..16], nxt[0]] = right shift
        let r0 = cmp::<IS_DILATE>(
            cmp::<IS_DILATE>(_mm_alignr_epi8::<15>(cur_t, prev_t), cur_t),
            _mm_alignr_epi8::<1>(nxt_t, cur_t),
        );
        let r1 = cmp::<IS_DILATE>(
            cmp::<IS_DILATE>(_mm_alignr_epi8::<15>(cur_m, prev_m), cur_m),
            _mm_alignr_epi8::<1>(nxt_m, cur_m),
        );
        let r2 = cmp::<IS_DILATE>(
            cmp::<IS_DILATE>(_mm_alignr_epi8::<15>(cur_b, prev_b), cur_b),
            _mm_alignr_epi8::<1>(nxt_b, cur_b),
        );
        _mm_storeu_si128(
            dst.add(x) as *mut __m128i,
            cmp::<IS_DILATE>(cmp::<IS_DILATE>(r0, r1), r2),
        );

        prev_t = cur_t;
        cur_t = nxt_t;
        prev_m = cur_m;
        cur_m = nxt_m;
        prev_b = cur_b;
        cur_b = nxt_b;
        x += 16;
    }
    if x < w {
        let r0 = cmp::<IS_DILATE>(
            cmp::<IS_DILATE>(_mm_alignr_epi8::<15>(cur_t, prev_t), cur_t),
            _mm_alignr_epi8::<1>(border_v, cur_t),
        );
        let r1 = cmp::<IS_DILATE>(
            cmp::<IS_DILATE>(_mm_alignr_epi8::<15>(cur_m, prev_m), cur_m),
            _mm_alignr_epi8::<1>(border_v, cur_m),
        );
        let r2 = cmp::<IS_DILATE>(
            cmp::<IS_DILATE>(_mm_alignr_epi8::<15>(cur_b, prev_b), cur_b),
            _mm_alignr_epi8::<1>(border_v, cur_b),
        );
        _mm_storeu_si128(
            dst.add(x) as *mut __m128i,
            cmp::<IS_DILATE>(cmp::<IS_DILATE>(r0, r1), r2),
        );
    }
}

/// Direct 3×3 morphology on x86_64 — rayon parallel.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn morph_3x3_direct_sse<const IS_DILATE: bool>(
    src: &[u8],
    out: &mut [u8],
    h: usize,
    w: usize,
) {
    let border: u8 = if IS_DILATE { 0 } else { 255 };
    let sp = src.as_ptr();
    let dp = out.as_mut_ptr();
    let border_row = vec![border; w];
    let bp = border_row.as_ptr();

    if h * w >= RAYON_THRESHOLD {
        let sp = sp as usize;
        let dp = dp as usize;
        let bp = bp as usize;
        (0..h).into_par_iter().for_each(|y| {
            let sp = sp as *const u8;
            let dp = dp as *mut u8;
            let bp = bp as *const u8;
            let top = if y > 0 { sp.add((y - 1) * w) } else { bp };
            let mid = sp.add(y * w);
            let bot = if y + 1 < h { sp.add((y + 1) * w) } else { bp };
            morph_row_sse::<IS_DILATE>(top, mid, bot, dp.add(y * w), w);
        });
        return;
    }

    morph_row_sse::<IS_DILATE>(bp, sp, if h > 1 { sp.add(w) } else { bp }, dp, w);
    for y in 1..h.saturating_sub(1) {
        morph_row_sse::<IS_DILATE>(
            sp.add((y - 1) * w),
            sp.add(y * w),
            sp.add((y + 1) * w),
            dp.add(y * w),
            w,
        );
    }
    if h > 1 {
        morph_row_sse::<IS_DILATE>(
            sp.add((h - 2) * w),
            sp.add((h - 1) * w),
            bp,
            dp.add((h - 1) * w),
            w,
        );
    }
}

/// SSE gaussian row -- direct 3x3 `[1,2,1]`x`[1,2,1]`/16, SSSE3 alignr for shifts.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn gauss_row_sse(top: *const u8, mid: *const u8, bot: *const u8, dst: *mut u8, w: usize) {
    use std::arch::x86_64::*;

    #[inline(always)]
    unsafe fn hsum(left: __m128i, center: __m128i, right: __m128i) -> (__m128i, __m128i) {
        let zero = _mm_setzero_si128();
        // Low 8 pixels
        let l_lo = _mm_unpacklo_epi8(left, zero);
        let c_lo = _mm_unpacklo_epi8(center, zero);
        let r_lo = _mm_unpacklo_epi8(right, zero);
        let lo = _mm_add_epi16(_mm_add_epi16(l_lo, r_lo), _mm_slli_epi16(c_lo, 1));
        // High 8 pixels
        let l_hi = _mm_unpackhi_epi8(left, zero);
        let c_hi = _mm_unpackhi_epi8(center, zero);
        let r_hi = _mm_unpackhi_epi8(right, zero);
        let hi = _mm_add_epi16(_mm_add_epi16(l_hi, r_hi), _mm_slli_epi16(c_hi, 1));
        (lo, hi)
    }

    let mut prev_t = _mm_set1_epi8(*top as i8);
    let mut prev_m = _mm_set1_epi8(*mid as i8);
    let mut prev_b = _mm_set1_epi8(*bot as i8);
    let mut cur_t = _mm_loadu_si128(top as *const __m128i);
    let mut cur_m = _mm_loadu_si128(mid as *const __m128i);
    let mut cur_b = _mm_loadu_si128(bot as *const __m128i);

    let round = _mm_set1_epi16(8); // +8 for rounding >>4
    let mut x = 0usize;
    let main_end = w.saturating_sub(16);

    while x < main_end {
        let nxt_t = _mm_loadu_si128(top.add(x + 16) as *const __m128i);
        let nxt_m = _mm_loadu_si128(mid.add(x + 16) as *const __m128i);
        let nxt_b = _mm_loadu_si128(bot.add(x + 16) as *const __m128i);

        let (ht_lo, ht_hi) = hsum(
            _mm_alignr_epi8::<15>(cur_t, prev_t),
            cur_t,
            _mm_alignr_epi8::<1>(nxt_t, cur_t),
        );
        let (hm_lo, hm_hi) = hsum(
            _mm_alignr_epi8::<15>(cur_m, prev_m),
            cur_m,
            _mm_alignr_epi8::<1>(nxt_m, cur_m),
        );
        let (hb_lo, hb_hi) = hsum(
            _mm_alignr_epi8::<15>(cur_b, prev_b),
            cur_b,
            _mm_alignr_epi8::<1>(nxt_b, cur_b),
        );

        let v_lo = _mm_add_epi16(_mm_add_epi16(ht_lo, hb_lo), _mm_slli_epi16(hm_lo, 1));
        let v_hi = _mm_add_epi16(_mm_add_epi16(ht_hi, hb_hi), _mm_slli_epi16(hm_hi, 1));

        // >>4 with rounding, pack to u8
        let r_lo = _mm_srli_epi16(_mm_add_epi16(v_lo, round), 4);
        let r_hi = _mm_srli_epi16(_mm_add_epi16(v_hi, round), 4);
        _mm_storeu_si128(dst.add(x) as *mut __m128i, _mm_packus_epi16(r_lo, r_hi));

        prev_t = cur_t;
        cur_t = nxt_t;
        prev_m = cur_m;
        cur_m = nxt_m;
        prev_b = cur_b;
        cur_b = nxt_b;
        x += 16;
    }

    if x < w {
        let border_t = _mm_set1_epi8(*top.add(w - 1) as i8);
        let border_m = _mm_set1_epi8(*mid.add(w - 1) as i8);
        let border_b = _mm_set1_epi8(*bot.add(w - 1) as i8);

        let (ht_lo, ht_hi) = hsum(
            _mm_alignr_epi8::<15>(cur_t, prev_t),
            cur_t,
            _mm_alignr_epi8::<1>(border_t, cur_t),
        );
        let (hm_lo, hm_hi) = hsum(
            _mm_alignr_epi8::<15>(cur_m, prev_m),
            cur_m,
            _mm_alignr_epi8::<1>(border_m, cur_m),
        );
        let (hb_lo, hb_hi) = hsum(
            _mm_alignr_epi8::<15>(cur_b, prev_b),
            cur_b,
            _mm_alignr_epi8::<1>(border_b, cur_b),
        );

        let v_lo = _mm_add_epi16(_mm_add_epi16(ht_lo, hb_lo), _mm_slli_epi16(hm_lo, 1));
        let v_hi = _mm_add_epi16(_mm_add_epi16(ht_hi, hb_hi), _mm_slli_epi16(hm_hi, 1));

        let r_lo = _mm_srli_epi16(_mm_add_epi16(v_lo, round), 4);
        let r_hi = _mm_srli_epi16(_mm_add_epi16(v_hi, round), 4);
        _mm_storeu_si128(dst.add(x) as *mut __m128i, _mm_packus_epi16(r_lo, r_hi));
    }
}

/// Direct 3×3 gaussian on x86_64 — rayon parallel.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn gauss_3x3_direct_sse(src: &[u8], out: &mut [u8], h: usize, w: usize) {
    let sp = src.as_ptr();
    let dp = out.as_mut_ptr();

    if h * w >= RAYON_THRESHOLD {
        let sp = sp as usize;
        let dp = dp as usize;
        (0..h).into_par_iter().for_each(|y| {
            let sp = sp as *const u8;
            let dp = dp as *mut u8;
            let top = if y > 0 { sp.add((y - 1) * w) } else { sp };
            let mid = sp.add(y * w);
            let bot = if y + 1 < h {
                sp.add((y + 1) * w)
            } else {
                sp.add((h - 1) * w)
            };
            gauss_row_sse(top, mid, bot, dp.add(y * w), w);
        });
        return;
    }

    gauss_row_sse(sp, sp, if h > 1 { sp.add(w) } else { sp }, dp, w);
    for y in 1..h.saturating_sub(1) {
        gauss_row_sse(
            sp.add((y - 1) * w),
            sp.add(y * w),
            sp.add((y + 1) * w),
            dp.add(y * w),
            w,
        );
    }
    if h > 1 {
        gauss_row_sse(
            sp.add((h - 2) * w),
            sp.add((h - 1) * w),
            sp.add((h - 1) * w),
            dp.add((h - 1) * w),
            w,
        );
    }
}

/// AVX2 grayscale row — processes 16 RGB pixels per iteration using 128-bit
/// SSSE3 deinterleave extended to two halves, then AVX2 multiply-add.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn grayscale_u8_avx2(src: &[u8], dst: &mut [u8], total: usize) -> usize {
    use std::arch::x86_64::*;

    let zero = _mm256_setzero_si256();
    let coeff_rg = _mm256_set1_epi16((150i16 << 8) | 77); // G=150, R=77 in maddubs order
    let coeff_b = _mm256_set1_epi16(29);
    let round = _mm256_set1_epi16(128);

    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();
    let mut x = 0usize;

    // SSSE3 shuffle masks for 16-byte deinterleave (reused for each 128-bit half)
    let shuf_r0 = _mm_setr_epi8(0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let shuf_r1 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14, -1, -1, -1, -1, -1);
    let shuf_r2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 4, 7, 10, 13);

    let shuf_g0 = _mm_setr_epi8(1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let shuf_g1 = _mm_setr_epi8(-1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1);
    let shuf_g2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14);

    let shuf_b0 = _mm_setr_epi8(2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let shuf_b1 = _mm_setr_epi8(-1, -1, -1, -1, -1, 1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1);
    let shuf_b2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15);

    // Process 32 pixels per iteration (96 source bytes) using two 16-pixel SSE deinterleaves
    while x + 32 <= total {
        // First 16 pixels
        let v0a = _mm_loadu_si128(sp.add(x * 3) as *const __m128i);
        let v1a = _mm_loadu_si128(sp.add(x * 3 + 16) as *const __m128i);
        let v2a = _mm_loadu_si128(sp.add(x * 3 + 32) as *const __m128i);

        let r_a = _mm_or_si128(
            _mm_or_si128(
                _mm_shuffle_epi8(v0a, shuf_r0),
                _mm_shuffle_epi8(v1a, shuf_r1),
            ),
            _mm_shuffle_epi8(v2a, shuf_r2),
        );
        let g_a = _mm_or_si128(
            _mm_or_si128(
                _mm_shuffle_epi8(v0a, shuf_g0),
                _mm_shuffle_epi8(v1a, shuf_g1),
            ),
            _mm_shuffle_epi8(v2a, shuf_g2),
        );
        let b_a = _mm_or_si128(
            _mm_or_si128(
                _mm_shuffle_epi8(v0a, shuf_b0),
                _mm_shuffle_epi8(v1a, shuf_b1),
            ),
            _mm_shuffle_epi8(v2a, shuf_b2),
        );

        // Second 16 pixels
        let v0b = _mm_loadu_si128(sp.add((x + 16) * 3) as *const __m128i);
        let v1b = _mm_loadu_si128(sp.add((x + 16) * 3 + 16) as *const __m128i);
        let v2b = _mm_loadu_si128(sp.add((x + 16) * 3 + 32) as *const __m128i);

        let r_b = _mm_or_si128(
            _mm_or_si128(
                _mm_shuffle_epi8(v0b, shuf_r0),
                _mm_shuffle_epi8(v1b, shuf_r1),
            ),
            _mm_shuffle_epi8(v2b, shuf_r2),
        );
        let g_b = _mm_or_si128(
            _mm_or_si128(
                _mm_shuffle_epi8(v0b, shuf_g0),
                _mm_shuffle_epi8(v1b, shuf_g1),
            ),
            _mm_shuffle_epi8(v2b, shuf_g2),
        );
        let b_b = _mm_or_si128(
            _mm_or_si128(
                _mm_shuffle_epi8(v0b, shuf_b0),
                _mm_shuffle_epi8(v1b, shuf_b1),
            ),
            _mm_shuffle_epi8(v2b, shuf_b2),
        );

        // Combine into 256-bit registers: [first16 | second16]
        let r = _mm256_set_m128i(r_b, r_a);
        let g = _mm256_set_m128i(g_b, g_a);
        let b = _mm256_set_m128i(b_b, b_a);

        // Interleave R,G pairs for _mm256_maddubs_epi16: [r0,g0, r1,g1, ...]
        let rg_lo = _mm256_unpacklo_epi8(r, g); // [r0,g0, r1,g1, ...r7,g7 | r16,g16, ...]
        let rg_hi = _mm256_unpackhi_epi8(r, g);

        // maddubs: treats first arg as u8, second as i8
        // result_lo[i] = r[2i]*77 + g[2i]*150 (as u16)
        let acc_lo = _mm256_maddubs_epi16(rg_lo, coeff_rg);
        let acc_hi = _mm256_maddubs_epi16(rg_hi, coeff_rg);

        // Add blue contribution: widen b to u16 and multiply by 29
        let b_lo = _mm256_unpacklo_epi8(b, zero);
        let b_hi = _mm256_unpackhi_epi8(b, zero);
        let acc_lo = _mm256_add_epi16(
            acc_lo,
            _mm256_add_epi16(_mm256_mullo_epi16(b_lo, coeff_b), round),
        );
        let acc_hi = _mm256_add_epi16(
            acc_hi,
            _mm256_add_epi16(_mm256_mullo_epi16(b_hi, coeff_b), round),
        );

        // >> 8 and pack to u8
        let gray_lo = _mm256_srli_epi16(acc_lo, 8);
        let gray_hi = _mm256_srli_epi16(acc_hi, 8);
        let packed = _mm256_packus_epi16(gray_lo, gray_hi);

        // packus interleaves 128-bit lanes: need to fix order with permute
        // packus gives: [lo0..lo7, hi0..hi7 | lo8..lo15, hi8..hi15]
        // We want: [lo0..lo7, lo8..lo15, hi0..hi7, hi8..hi15]
        let result = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);
        _mm256_storeu_si256(dp.add(x) as *mut __m256i, result);
        x += 32;
    }

    // Handle remaining 16-pixel chunk using SSE path logic
    while x + 16 <= total {
        let v0 = _mm_loadu_si128(sp.add(x * 3) as *const __m128i);
        let v1 = _mm_loadu_si128(sp.add(x * 3 + 16) as *const __m128i);
        let v2 = _mm_loadu_si128(sp.add(x * 3 + 32) as *const __m128i);

        let r = _mm_or_si128(
            _mm_or_si128(_mm_shuffle_epi8(v0, shuf_r0), _mm_shuffle_epi8(v1, shuf_r1)),
            _mm_shuffle_epi8(v2, shuf_r2),
        );
        let g = _mm_or_si128(
            _mm_or_si128(_mm_shuffle_epi8(v0, shuf_g0), _mm_shuffle_epi8(v1, shuf_g1)),
            _mm_shuffle_epi8(v2, shuf_g2),
        );
        let b = _mm_or_si128(
            _mm_or_si128(_mm_shuffle_epi8(v0, shuf_b0), _mm_shuffle_epi8(v1, shuf_b1)),
            _mm_shuffle_epi8(v2, shuf_b2),
        );

        let zero128 = _mm_setzero_si128();
        let coeff_r128 = _mm_set1_epi16(77);
        let coeff_g128 = _mm_set1_epi16(150);
        let coeff_b128 = _mm_set1_epi16(29);
        let round128 = _mm_set1_epi16(128);

        let r_lo = _mm_unpacklo_epi8(r, zero128);
        let g_lo = _mm_unpacklo_epi8(g, zero128);
        let b_lo = _mm_unpacklo_epi8(b, zero128);
        let acc_lo = _mm_add_epi16(
            _mm_add_epi16(
                _mm_mullo_epi16(r_lo, coeff_r128),
                _mm_mullo_epi16(g_lo, coeff_g128),
            ),
            _mm_add_epi16(_mm_mullo_epi16(b_lo, coeff_b128), round128),
        );
        let gray_lo = _mm_srli_epi16(acc_lo, 8);

        let r_hi = _mm_unpackhi_epi8(r, zero128);
        let g_hi = _mm_unpackhi_epi8(g, zero128);
        let b_hi = _mm_unpackhi_epi8(b, zero128);
        let acc_hi = _mm_add_epi16(
            _mm_add_epi16(
                _mm_mullo_epi16(r_hi, coeff_r128),
                _mm_mullo_epi16(g_hi, coeff_g128),
            ),
            _mm_add_epi16(_mm_mullo_epi16(b_hi, coeff_b128), round128),
        );
        let gray_hi = _mm_srli_epi16(acc_hi, 8);

        _mm_storeu_si128(
            dp.add(x) as *mut __m128i,
            _mm_packus_epi16(gray_lo, gray_hi),
        );
        x += 16;
    }
    x
}

/// SSE grayscale row — manual RGB deinterleave + weighted sum.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn grayscale_u8_sse(src: &[u8], dst: &mut [u8], total: usize) -> usize {
    use std::arch::x86_64::*;

    let zero = _mm_setzero_si128();
    let coeff_r = _mm_set1_epi16(77);
    let coeff_g = _mm_set1_epi16(150);
    let coeff_b = _mm_set1_epi16(29);
    let round = _mm_set1_epi16(128);

    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();
    let mut x = 0usize;

    // Process 16 pixels per iteration (48 source bytes)
    while x + 16 <= total {
        // Load 48 bytes = 16 RGB pixels, manually deinterleave
        let v0 = _mm_loadu_si128(sp.add(x * 3) as *const __m128i); // bytes 0-15
        let v1 = _mm_loadu_si128(sp.add(x * 3 + 16) as *const __m128i); // bytes 16-31
        let v2 = _mm_loadu_si128(sp.add(x * 3 + 32) as *const __m128i); // bytes 32-47

        // Deinterleave RGB using SSSE3 shuffle
        // Gather R bytes: positions 0,3,6,9,12,15 from v0, 2,5,8,11,14 from v1, 1,4,7,10,13 from v2
        let shuf_r0 = _mm_setr_epi8(0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        let shuf_r1 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14, -1, -1, -1, -1, -1);
        let shuf_r2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 4, 7, 10, 13);
        let r = _mm_or_si128(
            _mm_or_si128(_mm_shuffle_epi8(v0, shuf_r0), _mm_shuffle_epi8(v1, shuf_r1)),
            _mm_shuffle_epi8(v2, shuf_r2),
        );

        let shuf_g0 = _mm_setr_epi8(1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        let shuf_g1 = _mm_setr_epi8(-1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1);
        let shuf_g2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14);
        let g = _mm_or_si128(
            _mm_or_si128(_mm_shuffle_epi8(v0, shuf_g0), _mm_shuffle_epi8(v1, shuf_g1)),
            _mm_shuffle_epi8(v2, shuf_g2),
        );

        let shuf_b0 = _mm_setr_epi8(2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        let shuf_b1 = _mm_setr_epi8(-1, -1, -1, -1, -1, 1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1);
        let shuf_b2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15);
        let b = _mm_or_si128(
            _mm_or_si128(_mm_shuffle_epi8(v0, shuf_b0), _mm_shuffle_epi8(v1, shuf_b1)),
            _mm_shuffle_epi8(v2, shuf_b2),
        );

        // Low 8: widen, weighted sum, narrow
        let r_lo = _mm_unpacklo_epi8(r, zero);
        let g_lo = _mm_unpacklo_epi8(g, zero);
        let b_lo = _mm_unpacklo_epi8(b, zero);
        let acc_lo = _mm_add_epi16(
            _mm_add_epi16(
                _mm_mullo_epi16(r_lo, coeff_r),
                _mm_mullo_epi16(g_lo, coeff_g),
            ),
            _mm_add_epi16(_mm_mullo_epi16(b_lo, coeff_b), round),
        );
        let gray_lo = _mm_srli_epi16(acc_lo, 8);

        // High 8
        let r_hi = _mm_unpackhi_epi8(r, zero);
        let g_hi = _mm_unpackhi_epi8(g, zero);
        let b_hi = _mm_unpackhi_epi8(b, zero);
        let acc_hi = _mm_add_epi16(
            _mm_add_epi16(
                _mm_mullo_epi16(r_hi, coeff_r),
                _mm_mullo_epi16(g_hi, coeff_g),
            ),
            _mm_add_epi16(_mm_mullo_epi16(b_hi, coeff_b), round),
        );
        let gray_hi = _mm_srli_epi16(acc_hi, 8);

        _mm_storeu_si128(
            dp.add(x) as *mut __m128i,
            _mm_packus_epi16(gray_lo, gray_hi),
        );
        x += 16;
    }
    x
}

/// SSE median row — sorting network with min/max on 3×3 neighborhood.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn median_row_sse(
    row0: *const u8,
    row1: *const u8,
    row2: *const u8,
    dst: *mut u8,
    w: usize,
) -> usize {
    use std::arch::x86_64::*;

    #[inline(always)]
    unsafe fn sort2(a: &mut __m128i, b: &mut __m128i) {
        let mn = _mm_min_epu8(*a, *b);
        let mx = _mm_max_epu8(*a, *b);
        *a = mn;
        *b = mx;
    }

    let mut x = 1usize;
    while x + 16 < w {
        let mut v0 = _mm_loadu_si128(row0.add(x - 1) as *const __m128i);
        let mut v1 = _mm_loadu_si128(row0.add(x) as *const __m128i);
        let mut v2 = _mm_loadu_si128(row0.add(x + 1) as *const __m128i);
        let mut v3 = _mm_loadu_si128(row1.add(x - 1) as *const __m128i);
        let mut v4 = _mm_loadu_si128(row1.add(x) as *const __m128i);
        let mut v5 = _mm_loadu_si128(row1.add(x + 1) as *const __m128i);
        let mut v6 = _mm_loadu_si128(row2.add(x - 1) as *const __m128i);
        let mut v7 = _mm_loadu_si128(row2.add(x) as *const __m128i);
        let mut v8 = _mm_loadu_si128(row2.add(x + 1) as *const __m128i);

        // Sort each row triplet
        sort2(&mut v0, &mut v1);
        sort2(&mut v3, &mut v4);
        sort2(&mut v6, &mut v7);
        sort2(&mut v1, &mut v2);
        sort2(&mut v4, &mut v5);
        sort2(&mut v7, &mut v8);
        sort2(&mut v0, &mut v1);
        sort2(&mut v3, &mut v4);
        sort2(&mut v6, &mut v7);

        // Max of mins
        v3 = _mm_max_epu8(v0, v3);
        v3 = _mm_max_epu8(v3, v6);
        // Min of maxes
        v5 = _mm_min_epu8(v2, v5);
        v5 = _mm_min_epu8(v5, v8);
        // Median of (v3, v4, v5)
        sort2(&mut v3, &mut v4);
        sort2(&mut v4, &mut v5);
        sort2(&mut v3, &mut v4);

        _mm_storeu_si128(dst.add(x) as *mut __m128i, v4);
        x += 16;
    }
    x
}

/// SSE sobel row — gradient magnitude using SSSE3.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn sobel_row_sse(
    row0: *const u8,
    row1: *const u8,
    row2: *const u8,
    dst: *mut u8,
    w: usize,
) -> usize {
    use std::arch::x86_64::*;

    let zero = _mm_setzero_si128();
    let mut x = 1usize;

    while x + 16 < w {
        // Load neighbors
        let t_l = _mm_loadu_si128(row0.add(x - 1) as *const __m128i);
        let t_c = _mm_loadu_si128(row0.add(x) as *const __m128i);
        let t_r = _mm_loadu_si128(row0.add(x + 1) as *const __m128i);
        let b_l = _mm_loadu_si128(row2.add(x - 1) as *const __m128i);
        let b_c = _mm_loadu_si128(row2.add(x) as *const __m128i);
        let b_r = _mm_loadu_si128(row2.add(x + 1) as *const __m128i);
        let m_l = _mm_loadu_si128(row1.add(x - 1) as *const __m128i);
        let m_r = _mm_loadu_si128(row1.add(x + 1) as *const __m128i);

        // Process low 8 pixels
        let gx_lo = _mm_sub_epi16(
            _mm_add_epi16(
                _mm_add_epi16(_mm_unpacklo_epi8(t_r, zero), _mm_unpacklo_epi8(b_r, zero)),
                _mm_slli_epi16(_mm_unpacklo_epi8(m_r, zero), 1),
            ),
            _mm_add_epi16(
                _mm_add_epi16(_mm_unpacklo_epi8(t_l, zero), _mm_unpacklo_epi8(b_l, zero)),
                _mm_slli_epi16(_mm_unpacklo_epi8(m_l, zero), 1),
            ),
        );
        let gy_lo = _mm_sub_epi16(
            _mm_add_epi16(
                _mm_add_epi16(_mm_unpacklo_epi8(b_l, zero), _mm_unpacklo_epi8(b_r, zero)),
                _mm_slli_epi16(_mm_unpacklo_epi8(b_c, zero), 1),
            ),
            _mm_add_epi16(
                _mm_add_epi16(_mm_unpacklo_epi8(t_l, zero), _mm_unpacklo_epi8(t_r, zero)),
                _mm_slli_epi16(_mm_unpacklo_epi8(t_c, zero), 1),
            ),
        );
        let abs_lo = _mm_add_epi16(_mm_abs_epi16(gx_lo), _mm_abs_epi16(gy_lo));

        // Process high 8 pixels
        let gx_hi = _mm_sub_epi16(
            _mm_add_epi16(
                _mm_add_epi16(_mm_unpackhi_epi8(t_r, zero), _mm_unpackhi_epi8(b_r, zero)),
                _mm_slli_epi16(_mm_unpackhi_epi8(m_r, zero), 1),
            ),
            _mm_add_epi16(
                _mm_add_epi16(_mm_unpackhi_epi8(t_l, zero), _mm_unpackhi_epi8(b_l, zero)),
                _mm_slli_epi16(_mm_unpackhi_epi8(m_l, zero), 1),
            ),
        );
        let gy_hi = _mm_sub_epi16(
            _mm_add_epi16(
                _mm_add_epi16(_mm_unpackhi_epi8(b_l, zero), _mm_unpackhi_epi8(b_r, zero)),
                _mm_slli_epi16(_mm_unpackhi_epi8(b_c, zero), 1),
            ),
            _mm_add_epi16(
                _mm_add_epi16(_mm_unpackhi_epi8(t_l, zero), _mm_unpackhi_epi8(t_r, zero)),
                _mm_slli_epi16(_mm_unpackhi_epi8(t_c, zero), 1),
            ),
        );
        let abs_hi = _mm_add_epi16(_mm_abs_epi16(gx_hi), _mm_abs_epi16(gy_hi));

        _mm_storeu_si128(dst.add(x) as *mut __m128i, _mm_packus_epi16(abs_lo, abs_hi));
        x += 16;
    }
    x
}

/// NEON sobel row — gradient magnitude using NEON intrinsics.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn sobel_row_neon(
    row0: *const u8,
    row1: *const u8,
    row2: *const u8,
    dst: *mut u8,
    w: usize,
) -> usize {
    use std::arch::aarch64::*;

    let mut x = 1usize;

    while x + 16 <= w - 1 {
        // Load 16 u8 at offsets x-1, x, x+1 for each row
        let t_l = vld1q_u8(row0.add(x - 1));
        let t_c = vld1q_u8(row0.add(x));
        let t_r = vld1q_u8(row0.add(x + 1));
        let b_l = vld1q_u8(row2.add(x - 1));
        let b_c = vld1q_u8(row2.add(x));
        let b_r = vld1q_u8(row2.add(x + 1));
        let m_l = vld1q_u8(row1.add(x - 1));
        let m_r = vld1q_u8(row1.add(x + 1));

        // Process low 8 pixels: widen to u16 then reinterpret as s16
        let gx_lo = vsubq_s16(
            vaddq_s16(
                vaddq_s16(
                    vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(t_r))),
                    vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(b_r))),
                ),
                vshlq_n_s16::<1>(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(m_r)))),
            ),
            vaddq_s16(
                vaddq_s16(
                    vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(t_l))),
                    vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(b_l))),
                ),
                vshlq_n_s16::<1>(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(m_l)))),
            ),
        );
        let gy_lo = vsubq_s16(
            vaddq_s16(
                vaddq_s16(
                    vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(b_l))),
                    vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(b_r))),
                ),
                vshlq_n_s16::<1>(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(b_c)))),
            ),
            vaddq_s16(
                vaddq_s16(
                    vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(t_l))),
                    vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(t_r))),
                ),
                vshlq_n_s16::<1>(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(t_c)))),
            ),
        );
        let abs_lo = vreinterpretq_u16_s16(vqaddq_s16(vabsq_s16(gx_lo), vabsq_s16(gy_lo)));

        // Process high 8 pixels
        let gx_hi = vsubq_s16(
            vaddq_s16(
                vaddq_s16(
                    vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(t_r))),
                    vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(b_r))),
                ),
                vshlq_n_s16::<1>(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(m_r)))),
            ),
            vaddq_s16(
                vaddq_s16(
                    vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(t_l))),
                    vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(b_l))),
                ),
                vshlq_n_s16::<1>(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(m_l)))),
            ),
        );
        let gy_hi = vsubq_s16(
            vaddq_s16(
                vaddq_s16(
                    vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(b_l))),
                    vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(b_r))),
                ),
                vshlq_n_s16::<1>(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(b_c)))),
            ),
            vaddq_s16(
                vaddq_s16(
                    vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(t_l))),
                    vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(t_r))),
                ),
                vshlq_n_s16::<1>(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(t_c)))),
            ),
        );
        let abs_hi = vreinterpretq_u16_s16(vqaddq_s16(vabsq_s16(gx_hi), vabsq_s16(gy_hi)));

        // Pack u16 → u8 with saturation and store
        let result = vcombine_u8(vqmovn_u16(abs_lo), vqmovn_u16(abs_hi));
        vst1q_u8(dst.add(x), result);
        x += 16;
    }
    x
}

// ============================================================================
// Gaussian blur 3x3 (u8, single-channel)
// ============================================================================

/// 3x3 Gaussian blur on single-channel u8 image.
/// Uses `[1,2,1]` separable kernel with u16 intermediate.
/// Tiled processing: H-pass and V-pass are fused within tiles of 8 rows,
/// so the u16 tile buffer (10 rows * w * 2 bytes) fits in L1 cache.
/// Rayon parallelism across tiles when image is large enough.
pub fn gaussian_blur_3x3_u8(input: &ImageU8) -> Option<ImageU8> {
    if input.channels() != 1 {
        return None;
    }
    let (h, w) = (input.height(), input.width());
    if w == 0 || h == 0 {
        return Some(input.clone());
    }
    let src = input.data();
    let mut out = vec![0u8; h * w];

    // Direct 3×3 gaussian [1,2,1]×[1,2,1]/16 — no intermediate buffer.
    // Same approach as morph: vextq for horizontal shifts, GCD for parallelism.
    #[cfg(target_arch = "aarch64")]
    if !cfg!(miri) && std::arch::is_aarch64_feature_detected!("neon") {
        // SAFETY: ISA guard (feature detection) above.
        unsafe {
            gauss_3x3_direct_neon(src, &mut out, h, w);
        }
        return ImageU8::new(out, h, w, 1);
    }

    #[cfg(target_arch = "x86_64")]
    if !cfg!(miri) && is_x86_feature_detected!("ssse3") {
        // SAFETY: ISA guard (feature detection) above.
        unsafe {
            gauss_3x3_direct_sse(src, &mut out, h, w);
        }
        return ImageU8::new(out, h, w, 1);
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
            let v = top[lx] as u16
                + 2 * top[x] as u16
                + top[rx] as u16
                + 2 * (mid[lx] as u16 + 2 * mid[x] as u16 + mid[rx] as u16)
                + bot[lx] as u16
                + 2 * bot[x] as u16
                + bot[rx] as u16;
            dst[x] = ((v + 8) >> 4) as u8;
        }
    }
    ImageU8::new(out, h, w, 1)
}

/// Direct 3×3 gaussian row — vextq for horizontal neighbors, all u16 arithmetic.
/// Kernel weights `[1,2,1]`x`[1,2,1]`, max sum = 4080, fits u16. Divide by >>4 with rounding.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn gauss_row_neon(top: *const u8, mid: *const u8, bot: *const u8, dst: *mut u8, w: usize) {
    use std::arch::aarch64::*;

    // Inline helper: compute h_sum = left + 2*center + right in u16 for low/high halves
    #[inline(always)]
    unsafe fn hsum(
        left: uint8x16_t,
        center: uint8x16_t,
        right: uint8x16_t,
    ) -> (uint16x8_t, uint16x8_t) {
        let lo = vaddq_u16(
            vaddw_u8(vshll_n_u8::<1>(vget_low_u8(center)), vget_low_u8(left)),
            vmovl_u8(vget_low_u8(right)),
        );
        let hi = vaddq_u16(
            vaddw_u8(vshll_n_u8::<1>(vget_high_u8(center)), vget_high_u8(left)),
            vmovl_u8(vget_high_u8(right)),
        );
        (lo, hi)
    }

    let mut prev_t = vdupq_n_u8(*top);
    let mut prev_m = vdupq_n_u8(*mid);
    let mut prev_b = vdupq_n_u8(*bot);
    let mut cur_t = vld1q_u8(top);
    let mut cur_m = vld1q_u8(mid);
    let mut cur_b = vld1q_u8(bot);

    let mut x = 0usize;
    let main_end = w.saturating_sub(16);

    while x < main_end {
        let nxt_t = vld1q_u8(top.add(x + 16));
        let nxt_m = vld1q_u8(mid.add(x + 16));
        let nxt_b = vld1q_u8(bot.add(x + 16));

        let (ht_lo, ht_hi) = hsum(
            vextq_u8::<15>(prev_t, cur_t),
            cur_t,
            vextq_u8::<1>(cur_t, nxt_t),
        );
        let (hm_lo, hm_hi) = hsum(
            vextq_u8::<15>(prev_m, cur_m),
            cur_m,
            vextq_u8::<1>(cur_m, nxt_m),
        );
        let (hb_lo, hb_hi) = hsum(
            vextq_u8::<15>(prev_b, cur_b),
            cur_b,
            vextq_u8::<1>(cur_b, nxt_b),
        );

        let v_lo = vaddq_u16(vaddq_u16(ht_lo, hb_lo), vshlq_n_u16::<1>(hm_lo));
        let v_hi = vaddq_u16(vaddq_u16(ht_hi, hb_hi), vshlq_n_u16::<1>(hm_hi));

        vst1q_u8(
            dst.add(x),
            vcombine_u8(vrshrn_n_u16::<4>(v_lo), vrshrn_n_u16::<4>(v_hi)),
        );

        prev_t = cur_t;
        cur_t = nxt_t;
        prev_m = cur_m;
        cur_m = nxt_m;
        prev_b = cur_b;
        cur_b = nxt_b;
        x += 16;
    }

    // Tail with right-edge replicate border
    if x < w {
        let border_t = vdupq_n_u8(*top.add(w - 1));
        let border_m = vdupq_n_u8(*mid.add(w - 1));
        let border_b = vdupq_n_u8(*bot.add(w - 1));

        let (ht_lo, ht_hi) = hsum(
            vextq_u8::<15>(prev_t, cur_t),
            cur_t,
            vextq_u8::<1>(cur_t, border_t),
        );
        let (hm_lo, hm_hi) = hsum(
            vextq_u8::<15>(prev_m, cur_m),
            cur_m,
            vextq_u8::<1>(cur_m, border_m),
        );
        let (hb_lo, hb_hi) = hsum(
            vextq_u8::<15>(prev_b, cur_b),
            cur_b,
            vextq_u8::<1>(cur_b, border_b),
        );

        let v_lo = vaddq_u16(vaddq_u16(ht_lo, hb_lo), vshlq_n_u16::<1>(hm_lo));
        let v_hi = vaddq_u16(vaddq_u16(ht_hi, hb_hi), vshlq_n_u16::<1>(hm_hi));

        vst1q_u8(
            dst.add(x),
            vcombine_u8(vrshrn_n_u16::<4>(v_lo), vrshrn_n_u16::<4>(v_hi)),
        );
    }
}

/// Direct 3×3 gaussian — GCD parallel on macOS, sequential fallback.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn gauss_3x3_direct_neon(src: &[u8], out: &mut [u8], h: usize, w: usize) {
    let sp = src.as_ptr();
    let dp = out.as_mut_ptr();

    #[cfg(target_os = "macos")]
    if h * w >= RAYON_THRESHOLD {
        let sp = sp as usize;
        let dp = dp as usize;
        gcd::parallel_for(h, |y| {
            let sp = sp as *const u8;
            let dp = dp as *mut u8;
            let top = if y > 0 { sp.add((y - 1) * w) } else { sp };
            let mid = sp.add(y * w);
            let bot = if y + 1 < h {
                sp.add((y + 1) * w)
            } else {
                sp.add((h - 1) * w)
            };
            gauss_row_neon(top, mid, bot, dp.add(y * w), w);
        });
        return;
    }

    // Rayon fallback (non-macOS or no GCD)
    if h * w >= RAYON_THRESHOLD {
        let sp = sp as usize;
        let dp = dp as usize;
        (0..h).into_par_iter().for_each(|y| {
            let sp = sp as *const u8;
            let dp = dp as *mut u8;
            let top = if y > 0 { sp.add((y - 1) * w) } else { sp };
            let mid = sp.add(y * w);
            let bot = if y + 1 < h {
                sp.add((y + 1) * w)
            } else {
                sp.add((h - 1) * w)
            };
            gauss_row_neon(top, mid, bot, dp.add(y * w), w);
        });
        return;
    }

    // Sequential fallback (small images)
    gauss_row_neon(sp, sp, if h > 1 { sp.add(w) } else { sp }, dp, w);
    for y in 1..h.saturating_sub(1) {
        gauss_row_neon(
            sp.add((y - 1) * w),
            sp.add(y * w),
            sp.add((y + 1) * w),
            dp.add(y * w),
            w,
        );
    }
    if h > 1 {
        gauss_row_neon(
            sp.add((h - 2) * w),
            sp.add((h - 1) * w),
            sp.add((h - 1) * w),
            dp.add((h - 1) * w),
            w,
        );
    }
}

// ============================================================================
// Box blur 3x3 (u8, single-channel)
// ============================================================================

/// 3x3 box blur on single-channel u8 image.
/// Direct single-pass approach using vextq for horizontal shifts (no intermediate buffer).
/// Division by 9: `(sum * 3641 + 16384) >> 15` (3641/32768 ≈ 1/9.005).
/// Edge pixels use replicate (clamp) border.
pub fn box_blur_3x3_u8(input: &ImageU8) -> Option<ImageU8> {
    if input.channels() != 1 {
        return None;
    }
    let (h, w) = (input.height(), input.width());
    if w == 0 || h == 0 {
        return Some(input.clone());
    }
    let src = input.data();
    let mut out = vec![0u8; h * w];

    #[cfg(target_arch = "aarch64")]
    if !cfg!(miri) && std::arch::is_aarch64_feature_detected!("neon") {
        // SAFETY: (category 1) NEON guaranteed by feature detection; src/out same h*w allocation.
        unsafe {
            box_3x3_direct_neon(src, &mut out, h, w);
        }
        // Fix border pixels with accurate variable-count averaging
        if h >= 2 && w >= 2 {
            border_blur(src, &mut out, h, w);
        }
        return ImageU8::new(out, h, w, 1);
    }

    #[cfg(target_arch = "x86_64")]
    if !cfg!(miri) && is_x86_feature_detected!("sse2") {
        // SAFETY: (category 1) SSE2 guaranteed by feature detection; src/out same h*w allocation.
        unsafe {
            box_3x3_direct_sse(src, &mut out, h, w);
        }
        // Fix border pixels with accurate variable-count averaging
        if h >= 2 && w >= 2 {
            border_blur(src, &mut out, h, w);
        }
        return ImageU8::new(out, h, w, 1);
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
            let sum = top[lx] as u32
                + top[x] as u32
                + top[rx] as u32
                + mid[lx] as u32
                + mid[x] as u32
                + mid[rx] as u32
                + bot[lx] as u32
                + bot[x] as u32
                + bot[rx] as u32;
            dst[x] = ((sum * 3641) >> 15).min(255) as u8;
        }
    }
    ImageU8::new(out, h, w, 1)
}

/// Direct 3×3 box blur row — NEON single-pass using vextq for horizontal shifts.
/// Kernel: [1,1,1] x [1,1,1] / 9. Max u16 sum = 255*9 = 2295, fits u16.
/// Division by 9 via (sum * 3641) >> 15.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn box_row_neon(top: *const u8, mid: *const u8, bot: *const u8, dst: *mut u8, w: usize) {
    use std::arch::aarch64::*;

    // h_sum = left + center + right (uniform [1,1,1] kernel)
    // Uses vaddl_u8 (widening add: u8+u8->u16) + vaddw_u8 (add wide: u16+u8->u16)
    // to minimize instruction count: 2 instructions per 8 pixels.
    #[inline(always)]
    unsafe fn hsum(
        left: uint8x16_t,
        center: uint8x16_t,
        right: uint8x16_t,
    ) -> (uint16x8_t, uint16x8_t) {
        let lo = vaddw_u8(
            vaddl_u8(vget_low_u8(left), vget_low_u8(center)),
            vget_low_u8(right),
        );
        let hi = vaddw_u8(
            vaddl_u8(vget_high_u8(left), vget_high_u8(center)),
            vget_high_u8(right),
        );
        (lo, hi)
    }

    // Division by 9: (sum * 3641) >> 15
    // Process 8 u16 values, output 8 u8 values (via widening multiply to u32, shift, narrow)
    // Division by 9 using vqdmulhq_s16: computes (a * b) >> 15 (saturating).
    // With factor=3641: (sum * 3641) >> 15 ≈ sum / 9.005.
    // sum max = 2295 (fits i16), 2295*3641 = 8,355,495 fits i32.
    // Single instruction per 8 values (vs 4 instructions with vmull+vshrn).
    let div9_factor = vdupq_n_s16(3641);

    let mut prev_t = vdupq_n_u8(*top);
    let mut prev_m = vdupq_n_u8(*mid);
    let mut prev_b = vdupq_n_u8(*bot);
    let mut cur_t = vld1q_u8(top);
    let mut cur_m = vld1q_u8(mid);
    let mut cur_b = vld1q_u8(bot);

    let mut x = 0usize;
    let main_end = w.saturating_sub(16);

    while x < main_end {
        let nxt_t = vld1q_u8(top.add(x + 16));
        let nxt_m = vld1q_u8(mid.add(x + 16));
        let nxt_b = vld1q_u8(bot.add(x + 16));

        let (ht_lo, ht_hi) = hsum(
            vextq_u8::<15>(prev_t, cur_t),
            cur_t,
            vextq_u8::<1>(cur_t, nxt_t),
        );
        let (hm_lo, hm_hi) = hsum(
            vextq_u8::<15>(prev_m, cur_m),
            cur_m,
            vextq_u8::<1>(cur_m, nxt_m),
        );
        let (hb_lo, hb_hi) = hsum(
            vextq_u8::<15>(prev_b, cur_b),
            cur_b,
            vextq_u8::<1>(cur_b, nxt_b),
        );

        let v_lo = vaddq_u16(vaddq_u16(ht_lo, hm_lo), hb_lo);
        let v_hi = vaddq_u16(vaddq_u16(ht_hi, hm_hi), hb_hi);

        // (sum * 3641) >> 15 via vqdmulhq_s16 — single instruction per 8 values
        let d_lo = vqdmulhq_s16(vreinterpretq_s16_u16(v_lo), div9_factor);
        let d_hi = vqdmulhq_s16(vreinterpretq_s16_u16(v_hi), div9_factor);
        vst1q_u8(
            dst.add(x),
            vcombine_u8(vqmovun_s16(d_lo), vqmovun_s16(d_hi)),
        );

        prev_t = cur_t;
        cur_t = nxt_t;
        prev_m = cur_m;
        cur_m = nxt_m;
        prev_b = cur_b;
        cur_b = nxt_b;
        x += 16;
    }

    // Tail with right-edge replicate border
    if x < w {
        let border_t = vdupq_n_u8(*top.add(w - 1));
        let border_m = vdupq_n_u8(*mid.add(w - 1));
        let border_b = vdupq_n_u8(*bot.add(w - 1));

        let (ht_lo, ht_hi) = hsum(
            vextq_u8::<15>(prev_t, cur_t),
            cur_t,
            vextq_u8::<1>(cur_t, border_t),
        );
        let (hm_lo, hm_hi) = hsum(
            vextq_u8::<15>(prev_m, cur_m),
            cur_m,
            vextq_u8::<1>(cur_m, border_m),
        );
        let (hb_lo, hb_hi) = hsum(
            vextq_u8::<15>(prev_b, cur_b),
            cur_b,
            vextq_u8::<1>(cur_b, border_b),
        );

        let v_lo = vaddq_u16(vaddq_u16(ht_lo, hm_lo), hb_lo);
        let v_hi = vaddq_u16(vaddq_u16(ht_hi, hm_hi), hb_hi);

        // (sum * 3641) >> 15 via vqdmulhq_s16 — single instruction per 8 values
        let d_lo = vqdmulhq_s16(vreinterpretq_s16_u16(v_lo), div9_factor);
        let d_hi = vqdmulhq_s16(vreinterpretq_s16_u16(v_hi), div9_factor);
        vst1q_u8(
            dst.add(x),
            vcombine_u8(vqmovun_s16(d_lo), vqmovun_s16(d_hi)),
        );
    }
}

/// Direct 3×3 box blur — sequential for moderate images, chunked parallel for large.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn box_3x3_direct_neon(src: &[u8], out: &mut [u8], h: usize, w: usize) {
    let sp = src.as_ptr();
    let dp = out.as_mut_ptr();

    // Chunked parallelism for moderate-to-large images.
    const MIN_ROWS_PER_CHUNK: usize = 32;
    if h * w >= RAYON_THRESHOLD {
        let num_chunks = h.div_ceil(MIN_ROWS_PER_CHUNK);
        let sp = sp as usize;
        let dp = dp as usize;
        gcd::parallel_for(num_chunks, |chunk_idx| {
            let sp = sp as *const u8;
            let dp = dp as *mut u8;
            let y_start = chunk_idx * MIN_ROWS_PER_CHUNK;
            let y_end = (y_start + MIN_ROWS_PER_CHUNK).min(h);
            for y in y_start..y_end {
                let top = if y > 0 { sp.add((y - 1) * w) } else { sp };
                let mid = sp.add(y * w);
                let bot = if y + 1 < h {
                    sp.add((y + 1) * w)
                } else {
                    sp.add((h - 1) * w)
                };
                box_row_neon(top, mid, bot, dp.add(y * w), w);
            }
        });
        return;
    }

    // Sequential fallback
    for y in 0..h {
        let top = if y > 0 { sp.add((y - 1) * w) } else { sp };
        let mid = sp.add(y * w);
        let bot = if y + 1 < h {
            sp.add((y + 1) * w)
        } else {
            sp.add((h - 1) * w)
        };
        box_row_neon(top, mid, bot, dp.add(y * w), w);
    }
}

/// Direct 3×3 box blur on x86_64 — uses existing tiled approach.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn box_3x3_direct_sse(src: &[u8], out: &mut [u8], h: usize, w: usize) {
    // Use the tiled separable approach for SSE since it already exists and works.
    const TILE_H: usize = 8;
    let max_buf_rows = TILE_H + 2;
    let mut tile_buf = vec![0u16; max_buf_rows * w];

    let box_h_row = |row: &[u8], dst: &mut [u16]| {
        let r = if w > 1 { row[1] } else { row[0] };
        dst[0] = row[0] as u16 + row[0] as u16 + r as u16;
        let done = box_h_u8_simd(row, dst, w);
        for x in done..w.saturating_sub(1) {
            if x == 0 {
                continue;
            }
            dst[x] = row[x - 1] as u16 + row[x] as u16 + row[x + 1] as u16;
        }
        if w > 1 {
            dst[w - 1] = row[w - 2] as u16 + row[w - 1] as u16 + row[w - 1] as u16;
        }
    };

    let box_v_row = |above: &[u16], center: &[u16], below: &[u16], dst: &mut [u8]| {
        let done = box_v_u16_simd(above, center, below, dst, w);
        for x in done..w {
            let sum = above[x] as u32 + center[x] as u32 + below[x] as u32;
            dst[x] = ((sum * 3641) >> 15).min(255) as u8;
        }
    };

    for tile_y in (0..h).step_by(TILE_H) {
        let tile_end = (tile_y + TILE_H).min(h);
        let src_start = tile_y.saturating_sub(1);
        let src_end = (tile_end + 1).min(h);

        for (i, sy) in (src_start..src_end).enumerate() {
            let src_row = &src[sy * w..(sy + 1) * w];
            let dst_row = &mut tile_buf[i * w..(i + 1) * w];
            box_h_row(src_row, dst_row);
        }

        for oy in tile_y..tile_end {
            let ay = if oy > 0 { oy - 1 } else { 0 };
            let by = if oy + 1 < h { oy + 1 } else { h - 1 };
            let above_idx = ay - src_start;
            let center_idx = oy - src_start;
            let below_idx = by - src_start;
            let above = &tile_buf[above_idx * w..(above_idx + 1) * w];
            let center = &tile_buf[center_idx * w..(center_idx + 1) * w];
            let below = &tile_buf[below_idx * w..(below_idx + 1) * w];
            let out_row = &mut out[oy * w..(oy + 1) * w];
            box_v_row(above, center, below, out_row);
        }
    }
}

/// Shared border handling (variable-count average for edge pixels).
fn border_blur(src: &[u8], out: &mut [u8], h: usize, w: usize) {
    for y in [0, h - 1] {
        for x in 0..w {
            let mut sum = 0u16;
            let mut count = 0u16;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let ny = y as i32 + dy;
                    let nx = x as i32 + dx;
                    if ny >= 0 && ny < h as i32 && nx >= 0 && nx < w as i32 {
                        sum += src[ny as usize * w + nx as usize] as u16;
                        count += 1;
                    }
                }
            }
            out[y * w + x] = (sum / count) as u8;
        }
    }
    for y in 1..h - 1 {
        for x in [0, w - 1] {
            let mut sum = 0u16;
            let mut count = 0u16;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let ny = y as i32 + dy;
                    let nx = x as i32 + dx;
                    if ny >= 0 && ny < h as i32 && nx >= 0 && nx < w as i32 {
                        sum += src[ny as usize * w + nx as usize] as u16;
                        count += 1;
                    }
                }
            }
            out[y * w + x] = (sum / count) as u8;
        }
    }
}

#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
fn box_h_u8_simd(src: &[u8], dst: &mut [u16], w: usize) -> usize {
    if cfg!(miri) || w < 18 {
        return 1;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: ISA guard (feature detection) above.
            return unsafe { box_h_u8_neon(src, dst, w) };
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse2") {
            // SAFETY: ISA guard (feature detection) above.
            return unsafe { box_h_u8_sse(src, dst, w) };
        }
    }
    1
}

/// SSE2 horizontal `[1,1,1]` pass -- 16 pixels per iteration.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn box_h_u8_sse(src: &[u8], dst: &mut [u16], w: usize) -> usize {
    use std::arch::x86_64::*;
    let ptr = src.as_ptr();
    let out = dst.as_mut_ptr();
    let zero = _mm_setzero_si128();
    let mut x = 1usize;
    while x + 17 <= w {
        let left = _mm_loadu_si128(ptr.add(x - 1) as *const __m128i);
        let center = _mm_loadu_si128(ptr.add(x) as *const __m128i);
        let right = _mm_loadu_si128(ptr.add(x + 1) as *const __m128i);

        let l_lo = _mm_unpacklo_epi8(left, zero);
        let c_lo = _mm_unpacklo_epi8(center, zero);
        let r_lo = _mm_unpacklo_epi8(right, zero);
        _mm_storeu_si128(
            out.add(x) as *mut __m128i,
            _mm_add_epi16(_mm_add_epi16(l_lo, c_lo), r_lo),
        );

        let l_hi = _mm_unpackhi_epi8(left, zero);
        let c_hi = _mm_unpackhi_epi8(center, zero);
        let r_hi = _mm_unpackhi_epi8(right, zero);
        _mm_storeu_si128(
            out.add(x + 8) as *mut __m128i,
            _mm_add_epi16(_mm_add_epi16(l_hi, c_hi), r_hi),
        );

        x += 16;
    }
    while x + 9 <= w {
        let left = _mm_loadl_epi64(ptr.add(x - 1) as *const __m128i);
        let center = _mm_loadl_epi64(ptr.add(x) as *const __m128i);
        let right = _mm_loadl_epi64(ptr.add(x + 1) as *const __m128i);
        let sum = _mm_add_epi16(
            _mm_add_epi16(
                _mm_unpacklo_epi8(left, zero),
                _mm_unpacklo_epi8(center, zero),
            ),
            _mm_unpacklo_epi8(right, zero),
        );
        _mm_storeu_si128(out.add(x) as *mut __m128i, sum);
        x += 8;
    }
    x
}

/// NEON horizontal `[1,1,1]` pass -- 16 pixels per iteration.
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_op_in_unsafe_fn, dead_code)]
#[target_feature(enable = "neon")]
unsafe fn box_h_u8_neon(src: &[u8], dst: &mut [u16], w: usize) -> usize {
    use std::arch::aarch64::*;
    let ptr = src.as_ptr();
    let out = dst.as_mut_ptr();
    let mut x = 1usize;
    while x + 17 <= w {
        let left = vld1q_u8(ptr.add(x - 1));
        let center = vld1q_u8(ptr.add(x));
        let right = vld1q_u8(ptr.add(x + 1));

        let l_lo = vmovl_u8(vget_low_u8(left));
        let c_lo = vmovl_u8(vget_low_u8(center));
        let r_lo = vmovl_u8(vget_low_u8(right));
        vst1q_u16(out.add(x), vaddq_u16(vaddq_u16(l_lo, c_lo), r_lo));

        let l_hi = vmovl_u8(vget_high_u8(left));
        let c_hi = vmovl_u8(vget_high_u8(center));
        let r_hi = vmovl_u8(vget_high_u8(right));
        vst1q_u16(out.add(x + 8), vaddq_u16(vaddq_u16(l_hi, c_hi), r_hi));

        x += 16;
    }
    while x + 9 <= w {
        let left = vld1_u8(ptr.add(x - 1));
        let center = vld1_u8(ptr.add(x));
        let right = vld1_u8(ptr.add(x + 1));
        let sum = vaddq_u16(vaddq_u16(vmovl_u8(left), vmovl_u8(center)), vmovl_u8(right));
        vst1q_u16(out.add(x), sum);
        x += 8;
    }
    x
}

#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
fn box_v_u16_simd(above: &[u16], center: &[u16], below: &[u16], dst: &mut [u8], w: usize) -> usize {
    if cfg!(miri) || w < 8 {
        return 0;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: ISA guard (feature detection) above.
            return unsafe { box_v_u16_neon(above, center, below, dst, w) };
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse2") {
            // SAFETY: ISA guard (feature detection) above.
            return unsafe { box_v_u16_sse(above, center, below, dst, w) };
        }
    }
    0
}

/// SSE2 vertical `[1,1,1]`/9 pass -- 8 pixels per iteration.
/// Divides by 9: (sum * 3641) >> 15. Max sum = 2295, 2295*3641 fits u32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn box_v_u16_sse(
    above: &[u16],
    center: &[u16],
    below: &[u16],
    dst: &mut [u8],
    w: usize,
) -> usize {
    use std::arch::x86_64::*;
    let ap = above.as_ptr();
    let cp = center.as_ptr();
    let bp = below.as_ptr();
    let op = dst.as_mut_ptr();
    let factor = _mm_set1_epi16(3641);
    let mut x = 0usize;

    while x + 16 <= w {
        // First 8 pixels
        let a0 = _mm_loadu_si128(ap.add(x) as *const __m128i);
        let c0 = _mm_loadu_si128(cp.add(x) as *const __m128i);
        let b0 = _mm_loadu_si128(bp.add(x) as *const __m128i);
        let sum0 = _mm_add_epi16(_mm_add_epi16(a0, c0), b0);
        // (sum * 3641) >> 15 via mulhi: _mm_mulhi_epu16 gives (a*b)>>16,
        // but we need >>15, so we shift sum left by 1 first: (sum<<1 * 3641)>>16 = (sum*3641)>>15
        let r0 = _mm_mulhi_epu16(_mm_slli_epi16(sum0, 1), factor);

        // Second 8 pixels
        let a1 = _mm_loadu_si128(ap.add(x + 8) as *const __m128i);
        let c1 = _mm_loadu_si128(cp.add(x + 8) as *const __m128i);
        let b1 = _mm_loadu_si128(bp.add(x + 8) as *const __m128i);
        let sum1 = _mm_add_epi16(_mm_add_epi16(a1, c1), b1);
        let r1 = _mm_mulhi_epu16(_mm_slli_epi16(sum1, 1), factor);

        _mm_storeu_si128(op.add(x) as *mut __m128i, _mm_packus_epi16(r0, r1));
        x += 16;
    }
    while x + 8 <= w {
        let a = _mm_loadu_si128(ap.add(x) as *const __m128i);
        let c = _mm_loadu_si128(cp.add(x) as *const __m128i);
        let b = _mm_loadu_si128(bp.add(x) as *const __m128i);
        let sum = _mm_add_epi16(_mm_add_epi16(a, c), b);
        let r = _mm_mulhi_epu16(_mm_slli_epi16(sum, 1), factor);
        _mm_storel_epi64(op.add(x) as *mut __m128i, _mm_packus_epi16(r, r));
        x += 8;
    }
    x
}

/// NEON vertical `[1,1,1]`/9 pass -- 8 pixels per iteration.
/// Divides by 9 via: (sum * 3641) >> 15.
/// u16 sum max = 765*3 = 2295; 2295*3641 = 8,355,495 fits u32.
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_op_in_unsafe_fn, dead_code)]
#[target_feature(enable = "neon")]
unsafe fn box_v_u16_neon(
    above: &[u16],
    center: &[u16],
    below: &[u16],
    dst: &mut [u8],
    w: usize,
) -> usize {
    use std::arch::aarch64::*;
    let ap = above.as_ptr();
    let cp = center.as_ptr();
    let bp = below.as_ptr();
    let op = dst.as_mut_ptr();
    let factor = vdup_n_u16(3641);
    let mut x = 0usize;
    while x + 16 <= w {
        // First 8
        let a0 = vld1q_u16(ap.add(x));
        let c0 = vld1q_u16(cp.add(x));
        let b0 = vld1q_u16(bp.add(x));
        let sum0 = vaddq_u16(vaddq_u16(a0, c0), b0);
        let lo0 = vshrn_n_u32(vmull_u16(vget_low_u16(sum0), factor), 15);
        let hi0 = vshrn_n_u32(vmull_u16(vget_high_u16(sum0), factor), 15);
        let r0 = vqmovn_u16(vcombine_u16(lo0, hi0));

        // Second 8
        let a1 = vld1q_u16(ap.add(x + 8));
        let c1 = vld1q_u16(cp.add(x + 8));
        let b1 = vld1q_u16(bp.add(x + 8));
        let sum1 = vaddq_u16(vaddq_u16(a1, c1), b1);
        let lo1 = vshrn_n_u32(vmull_u16(vget_low_u16(sum1), factor), 15);
        let hi1 = vshrn_n_u32(vmull_u16(vget_high_u16(sum1), factor), 15);
        let r1 = vqmovn_u16(vcombine_u16(lo1, hi1));

        vst1q_u8(op.add(x), vcombine_u8(r0, r1));
        x += 16;
    }
    while x + 8 <= w {
        let a = vld1q_u16(ap.add(x));
        let c = vld1q_u16(cp.add(x));
        let b = vld1q_u16(bp.add(x));
        let sum = vaddq_u16(vaddq_u16(a, c), b);
        let lo = vshrn_n_u32(vmull_u16(vget_low_u16(sum), factor), 15);
        let hi = vshrn_n_u32(vmull_u16(vget_high_u16(sum), factor), 15);
        let result = vqmovn_u16(vcombine_u16(lo, hi));
        vst1_u8(op.add(x), result);
        x += 8;
    }
    x
}

// ============================================================================
// Sobel 3x3 magnitude (u8 → u8, single-channel)
// ============================================================================

/// 3x3 Sobel gradient magnitude on single-channel u8 image.
/// Returns gradient magnitude clamped to `[0, 255]`.
pub fn sobel_3x3_magnitude_u8(input: &ImageU8) -> Option<ImageU8> {
    if input.channels() != 1 {
        return None;
    }
    let (h, w) = (input.height(), input.width());
    if h < 3 || w < 3 {
        return Some(input.clone());
    }
    let src = input.data();
    let mut out = vec![0u8; h * w];

    let use_rayon = h * w >= RAYON_THRESHOLD && !cfg!(miri);
    let interior_h = h - 2;

    let process_sobel_row = |row0: &[u8], row1: &[u8], row2: &[u8], dst: &mut [u8]| {
        let mut done = 1usize;
        #[cfg(target_arch = "aarch64")]
        if !cfg!(miri) && std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: ISA guard (feature detection) above.
            done = unsafe {
                sobel_row_neon(
                    row0.as_ptr(),
                    row1.as_ptr(),
                    row2.as_ptr(),
                    dst.as_mut_ptr(),
                    w,
                )
            };
        }
        #[cfg(target_arch = "x86_64")]
        if !cfg!(miri) && is_x86_feature_detected!("ssse3") {
            // SAFETY: ISA guard (feature detection) above.
            done = unsafe {
                sobel_row_sse(
                    row0.as_ptr(),
                    row1.as_ptr(),
                    row2.as_ptr(),
                    dst.as_mut_ptr(),
                    w,
                )
            };
        }
        sobel_row_scalar_from(row0, row1, row2, dst, w, done);
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
                process_sobel_row(row0, row1, row2, dst);
            });
    } else {
        for y in 1..h - 1 {
            let row0 = &src[(y - 1) * w..y * w];
            let row1 = &src[y * w..(y + 1) * w];
            let row2 = &src[(y + 1) * w..(y + 2) * w];
            let dst = &mut out[y * w..(y + 1) * w];
            process_sobel_row(row0, row1, row2, dst);
        }
    }

    ImageU8::new(out, h, w, 1)
}

#[inline]
fn sobel_row_scalar_from(
    row0: &[u8],
    row1: &[u8],
    row2: &[u8],
    dst: &mut [u8],
    w: usize,
    start: usize,
) {
    for x in start..w - 1 {
        if x == 0 {
            continue;
        }
        let gx = row0[x + 1] as i16 - row0[x - 1] as i16
            + 2 * (row1[x + 1] as i16 - row1[x - 1] as i16)
            + row2[x + 1] as i16
            - row2[x - 1] as i16;
        let gy = row2[x - 1] as i16 + 2 * row2[x] as i16 + row2[x + 1] as i16
            - row0[x - 1] as i16
            - 2 * row0[x] as i16
            - row0[x + 1] as i16;
        let ax = gx.unsigned_abs();
        let ay = gy.unsigned_abs();
        let mag = ax.max(ay) + ax.min(ay) / 2;
        dst[x] = mag.min(255) as u8;
    }
}

// ============================================================================
// Median blur 3x3 (u8, single-channel) — O(1) histogram sliding window
// ============================================================================

/// 3x3 median filter on single-channel u8 image.
/// 3x3 median blur on single-channel u8 image.
/// Uses a SIMD sorting network (NEON) to find the median of 9 elements,
/// processing 16 pixels at a time. Scalar fallback for non-NEON and border pixels.
pub fn median_blur_3x3_u8(input: &ImageU8) -> Option<ImageU8> {
    if input.channels() != 1 {
        return None;
    }
    let (h, w) = (input.height(), input.width());
    if h < 3 || w < 3 {
        return Some(input.clone());
    }
    let src = input.data();
    let mut out = vec![0u8; h * w];

    let use_rayon = h * w >= RAYON_THRESHOLD && !cfg!(miri);
    let interior_h = h - 2;

    #[cfg(target_os = "macos")]
    if use_rayon && interior_h > 1 {
        let src_ptr = super::SendConstPtr(src.as_ptr());
        let out_ptr = super::SendPtr(out.as_mut_ptr());
        gcd::parallel_for(interior_h, |i| {
            let sp = src_ptr.ptr();
            let dp = out_ptr.ptr();
            let y = i + 1;
            // SAFETY: pointer and length from validated image data; rows are non-overlapping.
            let row0 = unsafe { core::slice::from_raw_parts(sp.add((y - 1) * w), w) };
            let row1 = unsafe { core::slice::from_raw_parts(sp.add(y * w), w) };
            let row2 = unsafe { core::slice::from_raw_parts(sp.add((y + 1) * w), w) };
            let dst = unsafe { core::slice::from_raw_parts_mut(dp.add(y * w), w) };
            let done = median_u8_simd_row(row0, row1, row2, dst, w);
            median_row_scalar(row0, row1, row2, dst, w, done);
        });
        return ImageU8::new(out, h, w, 1);
    }

    if use_rayon && interior_h > 1 {
        out[w..w + interior_h * w]
            .par_chunks_mut(w)
            .enumerate()
            .for_each(|(i, dst)| {
                let y = i + 1;
                let row0 = &src[(y - 1) * w..y * w];
                let row1 = &src[y * w..(y + 1) * w];
                let row2 = &src[(y + 1) * w..(y + 2) * w];
                let done = median_u8_simd_row(row0, row1, row2, dst, w);
                median_row_scalar(row0, row1, row2, dst, w, done);
            });
    } else {
        for y in 1..h - 1 {
            let row0 = &src[(y - 1) * w..y * w];
            let row1 = &src[y * w..(y + 1) * w];
            let row2 = &src[(y + 1) * w..(y + 2) * w];
            let dst = &mut out[y * w..(y + 1) * w];
            let done = median_u8_simd_row(row0, row1, row2, dst, w);
            median_row_scalar(row0, row1, row2, dst, w, done);
        }
    }

    // Border: zero (same as sobel)
    ImageU8::new(out, h, w, 1)
}

/// Scalar median for remaining/all pixels in a row. Processes x in `[start_x, w-2]`.
#[inline]
fn median_row_scalar(
    row0: &[u8],
    row1: &[u8],
    row2: &[u8],
    dst: &mut [u8],
    w: usize,
    start_x: usize,
) {
    for x in start_x..w - 1 {
        if x == 0 {
            continue;
        }
        let mut v = [
            row0[x - 1],
            row0[x],
            row0[x + 1],
            row1[x - 1],
            row1[x],
            row1[x + 1],
            row2[x - 1],
            row2[x],
            row2[x + 1],
        ];
        // Sorting network for median of 9 (scalar version)
        macro_rules! cas {
            ($a:expr, $b:expr) => {
                if v[$a] > v[$b] {
                    v.swap($a, $b);
                }
            };
        }
        // Sort each triplet (rows)
        cas!(0, 1);
        cas!(3, 4);
        cas!(6, 7);
        cas!(1, 2);
        cas!(4, 5);
        cas!(7, 8);
        cas!(0, 1);
        cas!(3, 4);
        cas!(6, 7);
        // Max of mins (v[0], v[3], v[6] are sorted-row mins)
        v[3] = v[0].max(v[3]);
        v[3] = v[3].max(v[6]);
        // Min of maxes (v[2], v[5], v[8] are sorted-row maxes)
        v[5] = v[2].min(v[5]);
        v[5] = v[5].min(v[8]);
        // Median from (v[3], v[4], v[5])
        if v[3] > v[4] {
            v.swap(3, 4);
        }
        if v[4] > v[5] {
            v.swap(4, 5);
        }
        if v[3] > v[4] {
            v.swap(3, 4);
        }
        dst[x] = v[4];
    }
}

/// SIMD dispatch for median row. Returns the x position where SIMD stopped.
fn median_u8_simd_row(row0: &[u8], row1: &[u8], row2: &[u8], out: &mut [u8], w: usize) -> usize {
    if cfg!(miri) || w < 18 {
        return 1;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: ISA guard (feature detection) above.
            return unsafe { median_u8_neon_row(row0, row1, row2, out, w) };
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse2") {
            // SAFETY: ISA guard (feature detection) above.
            return unsafe {
                median_row_sse(
                    row0.as_ptr(),
                    row1.as_ptr(),
                    row2.as_ptr(),
                    out.as_mut_ptr(),
                    w,
                )
            };
        }
    }
    1
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn median_u8_neon_row(
    row0: &[u8],
    row1: &[u8],
    row2: &[u8],
    out: &mut [u8],
    w: usize,
) -> usize {
    use std::arch::aarch64::*;

    #[inline(always)]
    unsafe fn cas(a: &mut uint8x16_t, b: &mut uint8x16_t) {
        let mn = vminq_u8(*a, *b);
        let mx = vmaxq_u8(*a, *b);
        *a = mn;
        *b = mx;
    }

    let p0 = row0.as_ptr();
    let p1 = row1.as_ptr();
    let p2 = row2.as_ptr();
    let op = out.as_mut_ptr();

    let mut x = 1usize;
    while x + 17 <= w {
        let mut a0 = vld1q_u8(p0.add(x - 1));
        let mut a1 = vld1q_u8(p0.add(x));
        let mut a2 = vld1q_u8(p0.add(x + 1));
        let mut a3 = vld1q_u8(p1.add(x - 1));
        let mut a4 = vld1q_u8(p1.add(x));
        let mut a5 = vld1q_u8(p1.add(x + 1));
        let mut a6 = vld1q_u8(p2.add(x - 1));
        let mut a7 = vld1q_u8(p2.add(x));
        let mut a8 = vld1q_u8(p2.add(x + 1));

        cas(&mut a0, &mut a1);
        cas(&mut a3, &mut a4);
        cas(&mut a6, &mut a7);
        cas(&mut a1, &mut a2);
        cas(&mut a4, &mut a5);
        cas(&mut a7, &mut a8);
        cas(&mut a0, &mut a1);
        cas(&mut a3, &mut a4);
        cas(&mut a6, &mut a7);

        a3 = vmaxq_u8(a0, a3);
        a3 = vmaxq_u8(a3, a6);
        a5 = vminq_u8(a2, a5);
        a5 = vminq_u8(a5, a8);

        cas(&mut a3, &mut a4);
        cas(&mut a4, &mut a5);
        cas(&mut a3, &mut a4);

        vst1q_u8(op.add(x), a4);
        x += 16;
    }
    x
}
