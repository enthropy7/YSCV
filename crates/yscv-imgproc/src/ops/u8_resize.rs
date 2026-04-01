//! Resize operations (u8, any channels): nearest-neighbour and bilinear.
#![allow(unsafe_code)]

use super::u8ops::{ImageU8, RAYON_THRESHOLD};
use rayon::prelude::*;

// ============================================================================
// Resize nearest-neighbour (u8, any channels)
// ============================================================================

/// Nearest-neighbour resize on u8 image `[H,W,C]` to target dimensions.
///
/// Optimisations:
/// - Precomputed x/y source-index maps (avoids per-pixel division).
/// - Row deduplication: when consecutive output rows map to the same source row
///   the entire row is `memcpy`'d instead of re-gathered.
/// - NEON gather via `TBL` for the x-remap inner loop (aarch64).
/// - Rayon parallelism for images above `RAYON_THRESHOLD` pixels.
pub fn resize_nearest_u8(input: &ImageU8, out_h: usize, out_w: usize) -> Option<ImageU8> {
    let (in_h, in_w, channels) = (input.height(), input.width(), input.channels());
    if out_h == 0 || out_w == 0 || in_h == 0 || in_w == 0 {
        return None;
    }
    let data = input.data();

    // Precompute source indices for each output column/row
    let x_map: Vec<usize> = (0..out_w).map(|x| x * in_w / out_w).collect();
    let y_map: Vec<usize> = (0..out_h).map(|y| y * in_h / out_h).collect();

    let out_row_bytes = out_w * channels;
    let in_row_stride = in_w * channels;
    let total = out_h * out_row_bytes;
    let mut out = vec![0u8; total];

    let use_rayon = out_h * out_w >= RAYON_THRESHOLD && !cfg!(miri);

    // Build a single "template row" gather, then replicate for duplicate y-mappings.
    if use_rayon {
        // Identify groups of consecutive output rows sharing the same source row.
        // Each chunk is (out_y_start, out_y_end, src_y).
        let out_base = super::SendPtr(out.as_mut_ptr());
        let x_map_ref = &x_map;

        // Parallel over output rows
        let chunk_rows = (out_h / (rayon::current_num_threads().max(1) * 2)).max(8);
        let mut chunks: Vec<(usize, usize)> = Vec::new();
        let mut s = 0;
        while s < out_h {
            let e = (s + chunk_rows).min(out_h);
            chunks.push((s, e));
            s = e;
        }

        chunks.par_iter().for_each(|&(start, end)| {
            let mut prev_src_y = usize::MAX;
            let mut prev_out_row: usize = 0; // offset of previously gathered row

            for oy in start..end {
                let src_y = y_map[oy];
                let dst_off = oy * out_row_bytes;

                if src_y == prev_src_y && oy > start {
                    // Duplicate row — memcpy from the previously computed row.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            (out_base.ptr() as *const u8).add(prev_out_row),
                            out_base.ptr().add(dst_off),
                            out_row_bytes,
                        );
                    }
                } else {
                    let src_row = &data[src_y * in_row_stride..];
                    let dst_row = unsafe {
                        std::slice::from_raw_parts_mut(out_base.ptr().add(dst_off), out_row_bytes)
                    };
                    resize_nearest_row(src_row, dst_row, x_map_ref, channels);
                    prev_src_y = src_y;
                    prev_out_row = dst_off;
                }
            }
        });
    } else {
        // Sequential with row deduplication
        let mut prev_src_y = usize::MAX;
        let mut prev_row_start: usize = 0;

        for oy in 0..out_h {
            let src_y = y_map[oy];
            let dst_off = oy * out_row_bytes;

            if src_y == prev_src_y && oy > 0 {
                out.copy_within(prev_row_start..prev_row_start + out_row_bytes, dst_off);
            } else {
                let src_row = &data[src_y * in_row_stride..];
                let dst_row = &mut out[dst_off..dst_off + out_row_bytes];
                resize_nearest_row(src_row, dst_row, &x_map, channels);
                prev_src_y = src_y;
                prev_row_start = dst_off;
            }
        }
    }

    ImageU8::new(out, out_h, out_w, channels)
}

/// Gather one output row using the precomputed x_map.
/// On aarch64 with NEON, uses vectorised byte-copy for single-channel images.
#[inline]
fn resize_nearest_row(src_row: &[u8], dst_row: &mut [u8], x_map: &[usize], channels: usize) {
    #[cfg(target_arch = "aarch64")]
    if channels == 1 && !cfg!(miri) && std::arch::is_aarch64_feature_detected!("neon") {
        unsafe {
            resize_nearest_row_1ch_neon(src_row, dst_row, x_map);
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if channels == 1 && !cfg!(miri) {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                resize_nearest_row_1ch_avx2(src_row, dst_row, x_map);
            }
            return;
        }
        if is_x86_feature_detected!("ssse3") {
            unsafe {
                resize_nearest_row_1ch_sse(src_row, dst_row, x_map);
            }
            return;
        }
    }

    // Scalar path (works for any channel count)
    match channels {
        1 => {
            for (ox, &sx) in x_map.iter().enumerate() {
                dst_row[ox] = src_row[sx];
            }
        }
        3 => {
            for (ox, &sx) in x_map.iter().enumerate() {
                let si = sx * 3;
                let di = ox * 3;
                dst_row[di] = src_row[si];
                dst_row[di + 1] = src_row[si + 1];
                dst_row[di + 2] = src_row[si + 2];
            }
        }
        4 => {
            for (ox, &sx) in x_map.iter().enumerate() {
                let si = sx * 4;
                let di = ox * 4;
                dst_row[di..di + 4].copy_from_slice(&src_row[si..si + 4]);
            }
        }
        _ => {
            for (ox, &sx) in x_map.iter().enumerate() {
                let si = sx * channels;
                let di = ox * channels;
                dst_row[di..di + channels].copy_from_slice(&src_row[si..si + channels]);
            }
        }
    }
}

/// NEON-accelerated single-channel nearest row gather.
/// Uses `vld1q_u8` + `vqtbl1q_u8` (TBL1) for groups of 16 output pixels that
/// span ≤16 source bytes, falling back to scalar for the remainder.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn resize_nearest_row_1ch_neon(src_row: &[u8], dst_row: &mut [u8], x_map: &[usize]) {
    use std::arch::aarch64::*;

    let out_w = x_map.len();
    let src_len = src_row.len();
    let mut ox = 0usize;

    // Process groups of 16 output pixels
    while ox + 16 <= out_w {
        let base = x_map[ox];
        let span_end = x_map[ox + 15];
        let span = span_end - base;

        if span < 16 && base + 16 <= src_len {
            // All 16 source indices fit inside one 16-byte NEON register
            let src_v = vld1q_u8(src_row.as_ptr().add(base));
            let mut idx = [0u8; 16];
            for k in 0..16 {
                idx[k] = (x_map[ox + k] - base) as u8;
            }
            let idx_v = vld1q_u8(idx.as_ptr());
            let gathered = vqtbl1q_u8(src_v, idx_v);
            vst1q_u8(dst_row.as_mut_ptr().add(ox), gathered);
            ox += 16;
        } else {
            // Span too wide or near end — scalar fallback for this group
            for k in 0..16 {
                *dst_row.as_mut_ptr().add(ox + k) = *src_row.as_ptr().add(x_map[ox + k]);
            }
            ox += 16;
        }
    }

    // Scalar tail
    while ox < out_w {
        *dst_row.as_mut_ptr().add(ox) = *src_row.as_ptr().add(x_map[ox]);
        ox += 1;
    }
}

/// SSSE3-accelerated single-channel nearest row gather.
/// Uses `_mm_shuffle_epi8` for groups of 16 output pixels that
/// span <= 16 source bytes, falling back to scalar for wider spans.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn resize_nearest_row_1ch_sse(src_row: &[u8], dst_row: &mut [u8], x_map: &[usize]) {
    use std::arch::x86_64::*;

    let out_w = x_map.len();
    let src_len = src_row.len();
    let mut ox = 0usize;

    // Process groups of 16 output pixels
    while ox + 16 <= out_w {
        let base = x_map[ox];
        let span_end = x_map[ox + 15];
        let span = span_end - base;

        if span < 16 && base + 16 <= src_len {
            // All 16 source indices fit inside one 16-byte SSE register
            let src_v = _mm_loadu_si128(src_row.as_ptr().add(base) as *const __m128i);
            let mut idx = [0u8; 16];
            for k in 0..16 {
                idx[k] = (x_map[ox + k] - base) as u8;
            }
            let idx_v = _mm_loadu_si128(idx.as_ptr() as *const __m128i);
            let gathered = _mm_shuffle_epi8(src_v, idx_v);
            _mm_storeu_si128(dst_row.as_mut_ptr().add(ox) as *mut __m128i, gathered);
            ox += 16;
        } else {
            // Span too wide or near end — scalar fallback for this group
            for k in 0..16 {
                *dst_row.as_mut_ptr().add(ox + k) = *src_row.as_ptr().add(x_map[ox + k]);
            }
            ox += 16;
        }
    }

    // Scalar tail
    while ox < out_w {
        *dst_row.as_mut_ptr().add(ox) = *src_row.as_ptr().add(x_map[ox]);
        ox += 1;
    }
}

/// AVX2-accelerated single-channel nearest row gather.
/// Uses `_mm256_shuffle_epi8` for groups of 32 output pixels that span
/// <= 16 source bytes per 128-bit lane, falling back to 16-pixel SSE
/// shuffle or scalar for wider spans.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn resize_nearest_row_1ch_avx2(src_row: &[u8], dst_row: &mut [u8], x_map: &[usize]) {
    use std::arch::x86_64::*;

    let out_w = x_map.len();
    let src_len = src_row.len();
    let mut ox = 0usize;

    // Try 16-pixel chunks (AVX2 shuffle is lane-wise, so we use two 16-byte shuffles)
    while ox + 16 <= out_w {
        let base = x_map[ox];
        let span_end = x_map[ox + 15];
        let span = span_end - base;

        if span < 16 && base + 16 <= src_len {
            let src_v = _mm_loadu_si128(src_row.as_ptr().add(base) as *const __m128i);
            let mut idx = [0u8; 16];
            for k in 0..16 {
                idx[k] = (x_map[ox + k] - base) as u8;
            }
            let idx_v = _mm_loadu_si128(idx.as_ptr() as *const __m128i);
            let gathered = _mm_shuffle_epi8(src_v, idx_v);
            _mm_storeu_si128(dst_row.as_mut_ptr().add(ox) as *mut __m128i, gathered);
            ox += 16;
        } else {
            for k in 0..16 {
                *dst_row.as_mut_ptr().add(ox + k) = *src_row.as_ptr().add(x_map[ox + k]);
            }
            ox += 16;
        }
    }

    // Scalar tail
    while ox < out_w {
        *dst_row.as_mut_ptr().add(ox) = *src_row.as_ptr().add(x_map[ox]);
        ox += 1;
    }
}

// ============================================================================
// Resize bilinear (u8, any channels)
// ============================================================================

/// Bilinear resize on u8 image `[H,W,C]` to target dimensions.
/// Uses fixed-point arithmetic (8-bit fractional) for speed.
/// Single-channel: NEON gather + blend, 8 pixels at a time.
/// Multi-channel: separable H-pass (scalar gather into u16 buf) then
/// NEON-vectorized V-pass on contiguous data, with row caching.
pub fn resize_bilinear_u8(input: &ImageU8, out_h: usize, out_w: usize) -> Option<ImageU8> {
    let (h, w, c) = (input.height(), input.width(), input.channels());
    if h == 0 || w == 0 || out_h == 0 || out_w == 0 {
        return None;
    }
    let src = input.data();

    // Fast path: single-channel ~2× downscale — skip all precomputation overhead.
    // Detects when output is exactly half the input and every source index is stride-2.
    if c == 1 && out_h >= 2 && out_w >= 2 {
        let y_ratio = ((h - 1) as u64 * 65536) / (out_h - 1).max(1) as u64;
        let x_ratio = ((w - 1) as u64 * 65536) / (out_w - 1).max(1) as u64;
        // Check stride-2: first two sx values must be 0,2 and last must be (out_w-1)*2
        let sx0 = 0usize;
        let sx1_check = (x_ratio >> 16) as usize;
        let sx_last = (((out_w - 1) as u64 * x_ratio) >> 16) as usize;
        if sx0 == 0 && sx1_check == 2 && sx_last == (out_w - 1) * 2 {
            let mut out = vec![0u8; out_h * out_w];
            // Precompute only fx_arr (needed by NEON kernel)
            let mut fx_arr = vec![0u16; out_w];
            for ox in 0..out_w {
                let x_fp = ox as u64 * x_ratio;
                fx_arr[ox] = ((x_fp & 0xFFFF) >> 8) as u16;
            }
            resize_bilinear_u8_1ch_stride2(src, &mut out, h, w, out_h, out_w, y_ratio, &fx_arr);
            return ImageU8::new(out, out_h, out_w, 1);
        }
    }

    let mut out = vec![0u8; out_h * out_w * c];

    // Fixed-point scale factors (16-bit fractional)
    let y_ratio = if out_h > 1 {
        ((h - 1) as u64 * 65536) / (out_h - 1).max(1) as u64
    } else {
        0
    };
    let x_ratio = if out_w > 1 {
        ((w - 1) as u64 * 65536) / (out_w - 1).max(1) as u64
    } else {
        0
    };

    let use_rayon = out_h * out_w >= RAYON_THRESHOLD && !cfg!(miri);
    let row_stride = out_w * c;

    // Pre-compute x-coordinates and fractional weights for each output column.
    // This avoids redundant computation per row.
    let mut sx_arr = vec![0u32; out_w];
    let mut sx1_arr = vec![0u32; out_w];
    let mut fx_arr = vec![0u16; out_w];
    for ox in 0..out_w {
        let x_fp = ox as u64 * x_ratio;
        let sx = (x_fp >> 16) as usize;
        let fx = ((x_fp & 0xFFFF) >> 8) as u16; // 8-bit fractional
        sx_arr[ox] = sx as u32;
        sx1_arr[ox] = (sx + 1).min(w - 1) as u32;
        fx_arr[ox] = fx;
    }

    // Precompute replicated fx array for c=3 NEON H-pass: [fx0,fx0,fx0, fx1,fx1,fx1, ...]
    // Amortized over all rows. Padded to out_w*3+8 for safe NEON loads.
    let fx_rep: Vec<u16> = if c == 3 {
        let mut rep = vec![0u16; out_w * 3 + 8]; // +8 padding for safe vld1q_u16 near end
        for ox in 0..out_w {
            let fx = fx_arr[ox];
            rep[ox * 3] = fx;
            rep[ox * 3 + 1] = fx;
            rep[ox * 3 + 2] = fx;
        }
        rep
    } else {
        Vec::new()
    };

    // Precompute byte offsets: sx*c and sx1*c to eliminate per-pixel multiply in H-pass.
    let sx_byte_arr: Vec<usize> = sx_arr.iter().map(|&sx| sx as usize * c).collect();
    let sx1_byte_arr: Vec<usize> = sx1_arr.iter().map(|&sx1| sx1 as usize * c).collect();

    if c > 1 {
        // Separable multi-channel: H-pass into u16 buffers, then V-pass.
        // H-pass: hbuf[ox*c+ch] = src[sx0*c+ch]*(256-fx) + src[sx1*c+ch]*fx  (u16)
        // V-pass: out = (hbuf0[i]*(256-fy) + hbuf1[i]*fy + 32768) >> 16      (u8)
        // V-pass is contiguous -> full NEON vectorization, no gather.
        // Row caching: consecutive output rows sharing (sy, sy1) skip H-pass.
        let hbuf_len = out_w * c;

        // Pre-compute (sy, sy1, fy) for each output row
        let mut row_params: Vec<(usize, usize, u32)> = Vec::with_capacity(out_h);
        for oy in 0..out_h {
            let y_fp = oy as u64 * y_ratio;
            let sy = (y_fp >> 16) as usize;
            let fy = ((y_fp & 0xFFFF) >> 8) as u32;
            let sy1 = (sy + 1).min(h - 1);
            row_params.push((sy, sy1, fy));
        }

        if use_rayon {
            let out_base = super::SendPtr(out.as_mut_ptr());
            // Split into chunks aligned to reduce rayon overhead.
            // Use ~2× CPU count for good load balancing without excessive overhead.
            let n_threads = rayon::current_num_threads().max(1);
            let chunk_size = (out_h / (n_threads * 2)).max(8);
            let mut super_groups: Vec<(usize, usize)> = Vec::new();
            let mut s = 0;
            while s < out_h {
                let e = (s + chunk_size).min(out_h);
                super_groups.push((s, e));
                s = e;
            }

            super_groups.par_iter().for_each(|&(sg_start, sg_end)| {
                let src_stride = w * c;
                let mut hbuf0 = vec![0u16; hbuf_len];
                let mut hbuf1 = vec![0u16; hbuf_len];
                let mut cached_sy = usize::MAX;
                let mut cached_sy1 = usize::MAX;

                for oy in sg_start..sg_end {
                    let (sy, sy1, fy) = row_params[oy];

                    if sy != cached_sy || sy1 != cached_sy1 {
                        if sy == cached_sy {
                            // Only sy1 changed — recompute hbuf1 only
                            let src_row1 = &src[sy1 * src_stride..sy1 * src_stride + src_stride];
                            unsafe {
                                resize_hpass_multi_ch(
                                    src_row1,
                                    &mut hbuf1,
                                    &sx_byte_arr,
                                    &sx1_byte_arr,
                                    &fx_arr,
                                    &fx_rep,
                                    out_w,
                                    c,
                                );
                            }
                        } else if sy == cached_sy1 {
                            // New row 0 was old row 1 — swap and compute only new row 1
                            std::mem::swap(&mut hbuf0, &mut hbuf1);
                            let src_row1 = &src[sy1 * src_stride..sy1 * src_stride + src_stride];
                            unsafe {
                                resize_hpass_multi_ch(
                                    src_row1,
                                    &mut hbuf1,
                                    &sx_byte_arr,
                                    &sx1_byte_arr,
                                    &fx_arr,
                                    &fx_rep,
                                    out_w,
                                    c,
                                );
                            }
                        } else {
                            let src_row0 = &src[sy * src_stride..sy * src_stride + src_stride];
                            let src_row1 = &src[sy1 * src_stride..sy1 * src_stride + src_stride];
                            unsafe {
                                resize_hpass_multi_ch(
                                    src_row0,
                                    &mut hbuf0,
                                    &sx_byte_arr,
                                    &sx1_byte_arr,
                                    &fx_arr,
                                    &fx_rep,
                                    out_w,
                                    c,
                                );
                                resize_hpass_multi_ch(
                                    src_row1,
                                    &mut hbuf1,
                                    &sx_byte_arr,
                                    &sx1_byte_arr,
                                    &fx_arr,
                                    &fx_rep,
                                    out_w,
                                    c,
                                );
                            }
                        }
                        cached_sy = sy;
                        cached_sy1 = sy1;
                    }

                    let dst_off = oy * row_stride;
                    let dst_row = unsafe {
                        std::slice::from_raw_parts_mut(out_base.ptr().add(dst_off), row_stride)
                    };
                    unsafe {
                        resize_vpass_multi_ch(&hbuf0, &hbuf1, dst_row, fy as u16, hbuf_len);
                    }
                }
            });
        } else {
            // Sequential with row caching
            let mut cached_sy: usize = usize::MAX;
            let mut cached_sy1: usize = usize::MAX;
            let mut hbuf0 = vec![0u16; hbuf_len];
            let mut hbuf1 = vec![0u16; hbuf_len];

            for oy in 0..out_h {
                let (sy, sy1, fy) = row_params[oy];

                if sy != cached_sy || sy1 != cached_sy1 {
                    let src_stride = w * c;
                    if sy == cached_sy {
                        let src_row1 = &src[sy1 * src_stride..sy1 * src_stride + src_stride];
                        unsafe {
                            resize_hpass_multi_ch(
                                src_row1,
                                &mut hbuf1,
                                &sx_byte_arr,
                                &sx1_byte_arr,
                                &fx_arr,
                                &fx_rep,
                                out_w,
                                c,
                            );
                        }
                    } else if sy == cached_sy1 {
                        std::mem::swap(&mut hbuf0, &mut hbuf1);
                        let src_row1 = &src[sy1 * src_stride..sy1 * src_stride + src_stride];
                        unsafe {
                            resize_hpass_multi_ch(
                                src_row1,
                                &mut hbuf1,
                                &sx_byte_arr,
                                &sx1_byte_arr,
                                &fx_arr,
                                &fx_rep,
                                out_w,
                                c,
                            );
                        }
                    } else {
                        let src_row0 = &src[sy * src_stride..sy * src_stride + src_stride];
                        let src_row1 = &src[sy1 * src_stride..sy1 * src_stride + src_stride];
                        unsafe {
                            resize_hpass_multi_ch(
                                src_row0,
                                &mut hbuf0,
                                &sx_byte_arr,
                                &sx1_byte_arr,
                                &fx_arr,
                                &fx_rep,
                                out_w,
                                c,
                            );
                            resize_hpass_multi_ch(
                                src_row1,
                                &mut hbuf1,
                                &sx_byte_arr,
                                &sx1_byte_arr,
                                &fx_arr,
                                &fx_rep,
                                out_w,
                                c,
                            );
                        }
                    }
                    cached_sy = sy;
                    cached_sy1 = sy1;
                }

                let dst_row = &mut out[oy * row_stride..(oy + 1) * row_stride];
                unsafe {
                    resize_vpass_multi_ch(&hbuf0, &hbuf1, dst_row, fy as u16, hbuf_len);
                }
            }
        }
    } else {
        // Single-channel: check for stride-2 fast path (~2× downscale).
        // When source indices are exactly [0,2,4,...], vld2q_u8 replaces TBL2 gather.
        let is_stride2 = out_w >= 2
            && sx_arr[0] == 0
            && sx_arr[1] == 2
            && sx_arr[out_w - 1] as usize == (out_w - 1) * 2;

        if is_stride2 {
            resize_bilinear_u8_1ch_stride2(src, &mut out, h, w, out_h, out_w, y_ratio, &fx_arr);
        } else {
            resize_bilinear_u8_1ch_fused(
                src, &mut out, h, w, out_h, out_w, y_ratio, &sx_arr, &sx1_arr, &fx_arr,
            );
        }
    }

    ImageU8::new(out, out_h, out_w, c)
}

/// Fast path for ~2× downscale 1ch: vld2q_u8 stride-2 load, full-image NEON kernel.
/// Single function call processes all rows — no per-row feature detection or slice setup.
fn resize_bilinear_u8_1ch_stride2(
    src: &[u8],
    out: &mut [u8],
    h: usize,
    w: usize,
    out_h: usize,
    out_w: usize,
    y_ratio: u64,
    fx_arr: &[u16],
) {
    #[cfg(target_arch = "aarch64")]
    if !cfg!(miri) && std::arch::is_aarch64_feature_detected!("neon") {
        unsafe {
            resize_1ch_stride2_neon_full(
                src.as_ptr(),
                out.as_mut_ptr(),
                fx_arr.as_ptr(),
                h,
                w,
                out_h,
                out_w,
                y_ratio,
            );
        }
        return;
    }

    // Scalar fallback
    for oy in 0..out_h {
        let y_fp = oy as u64 * y_ratio;
        let sy = (y_fp >> 16) as usize;
        let fy = ((y_fp & 0xFFFF) >> 8) as u32;
        let fyi = 256 - fy;
        let sy1 = (sy + 1).min(h - 1);
        for ox in 0..out_w {
            let sx = ox * 2;
            let sx1 = (sx + 1).min(w - 1);
            let fx = fx_arr[ox] as u32;
            let fxi = 256 - fx;
            let top = src[sy * w + sx] as u32 * fxi + src[sy * w + sx1] as u32 * fx;
            let bot = src[sy1 * w + sx] as u32 * fxi + src[sy1 * w + sx1] as u32 * fx;
            out[oy * out_w + ox] = ((top * fyi + bot * fy + 32768) >> 16).min(255) as u8;
        }
    }
}

/// Full-image NEON stride-2 resize: zero per-row overhead.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn resize_1ch_stride2_neon_full(
    src: *const u8,
    dst: *mut u8,
    fx_ptr: *const u16,
    h: usize,
    w: usize,
    out_h: usize,
    out_w: usize,
    y_ratio: u64,
) {
    use std::arch::aarch64::*;

    let v256 = vdupq_n_u16(256);
    let round = vdupq_n_u16(128);
    let neon_end = out_w & !15; // round down to multiple of 16

    for oy in 0..out_h {
        let y_fp = oy as u64 * y_ratio;
        let sy = (y_fp >> 16) as usize;
        let fy = ((y_fp & 0xFFFF) >> 8) as u16;
        let sy1 = (sy + 1).min(h - 1);
        let r0 = src.add(sy * w);
        let r1 = src.add(sy1 * w);
        let dp = dst.add(oy * out_w);

        let fy_v = vdupq_n_u16(fy);
        let fyi_v = vdupq_n_u16(256 - fy);

        let mut ox = 0usize;
        while ox < neon_end {
            let sx = ox * 2;
            let pair0 = vld2q_u8(r0.add(sx));
            let pair1 = vld2q_u8(r1.add(sx));

            let fx_lo = vld1q_u16(fx_ptr.add(ox));
            let fxi_lo = vsubq_u16(v256, fx_lo);
            let top_lo = vshrn_n_u16::<8>(vaddq_u16(
                vmulq_u16(fxi_lo, vmovl_u8(vget_low_u8(pair0.0))),
                vmulq_u16(fx_lo, vmovl_u8(vget_low_u8(pair0.1))),
            ));
            let bot_lo = vshrn_n_u16::<8>(vaddq_u16(
                vmulq_u16(fxi_lo, vmovl_u8(vget_low_u8(pair1.0))),
                vmulq_u16(fx_lo, vmovl_u8(vget_low_u8(pair1.1))),
            ));
            let rlo = vshrn_n_u16::<8>(vaddq_u16(
                vaddq_u16(
                    vmulq_u16(fyi_v, vmovl_u8(top_lo)),
                    vmulq_u16(fy_v, vmovl_u8(bot_lo)),
                ),
                round,
            ));

            let fx_hi = vld1q_u16(fx_ptr.add(ox + 8));
            let fxi_hi = vsubq_u16(v256, fx_hi);
            let top_hi = vshrn_n_u16::<8>(vaddq_u16(
                vmulq_u16(fxi_hi, vmovl_u8(vget_high_u8(pair0.0))),
                vmulq_u16(fx_hi, vmovl_u8(vget_high_u8(pair0.1))),
            ));
            let bot_hi = vshrn_n_u16::<8>(vaddq_u16(
                vmulq_u16(fxi_hi, vmovl_u8(vget_high_u8(pair1.0))),
                vmulq_u16(fx_hi, vmovl_u8(vget_high_u8(pair1.1))),
            ));
            let rhi = vshrn_n_u16::<8>(vaddq_u16(
                vaddq_u16(
                    vmulq_u16(fyi_v, vmovl_u8(top_hi)),
                    vmulq_u16(fy_v, vmovl_u8(bot_hi)),
                ),
                round,
            ));

            vst1q_u8(dp.add(ox), vcombine_u8(rlo, rhi));
            ox += 16;
        }

        // Scalar tail
        let fy32 = fy as u32;
        let fyi32 = 256 - fy32;
        while ox < out_w {
            let sx = ox * 2;
            let sx1 = (sx + 1).min(w - 1);
            let fx = *fx_ptr.add(ox) as u32;
            let fxi = 256 - fx;
            let top = *r0.add(sx) as u32 * fxi + *r0.add(sx1) as u32 * fx;
            let bot = *r1.add(sx) as u32 * fxi + *r1.add(sx1) as u32 * fx;
            *dp.add(ox) = ((top * fyi32 + bot * fy32 + 32768) >> 16) as u8;
            ox += 1;
        }
    }
}

/// Fused single-channel bilinear resize with TBL2 gather (16 pixels/iteration).
fn resize_bilinear_u8_1ch_fused(
    src: &[u8],
    out: &mut [u8],
    h: usize,
    w: usize,
    out_h: usize,
    out_w: usize,
    y_ratio: u64,
    sx_arr: &[u32],
    sx1_arr: &[u32],
    fx_arr: &[u16],
) {
    // Precompute TBL2 gather indices for groups of 16 output pixels.
    // For ~2× downscale: 16 output pixels span ~32 source bytes → fits TBL2.
    let n16 = out_w / 16;
    let mut base16 = vec![0u32; n16];
    let mut idx16_0 = vec![[0u8; 16]; n16]; // sx offsets
    let mut idx16_1 = vec![[0u8; 16]; n16]; // sx+1 offsets
    let mut use_tbl2 = true;

    for g in 0..n16 {
        let ox = g * 16;
        let b = sx_arr[ox];
        base16[g] = b;
        let max_idx = sx1_arr[ox + 15];
        if max_idx - b >= 32 {
            use_tbl2 = false;
            break;
        }
        for k in 0..16 {
            idx16_0[g][k] = (sx_arr[ox + k] - b) as u8;
            idx16_1[g][k] = (sx1_arr[ox + k] - b) as u8;
        }
    }

    // Also precompute 8-pixel groups for the tail (n16*16..out_w)
    let tail_start = n16 * 16;
    let n8_tail = (out_w - tail_start) / 8;
    let mut base8 = vec![0u32; n8_tail];
    let mut idx8_0 = vec![[0u8; 8]; n8_tail];
    let mut idx8_1 = vec![[0u8; 8]; n8_tail];
    let mut use_tbl1_tail = use_tbl2; // only if tbl2 path is active

    if use_tbl2 {
        for g in 0..n8_tail {
            let ox = tail_start + g * 8;
            let b = sx_arr[ox];
            base8[g] = b;
            let max_idx = sx1_arr[ox + 7];
            if max_idx - b >= 16 {
                use_tbl1_tail = false;
                break;
            }
            for k in 0..8 {
                idx8_0[g][k] = (sx_arr[ox + k] - b) as u8;
                idx8_1[g][k] = (sx1_arr[ox + k] - b) as u8;
            }
        }
    }

    for oy in 0..out_h {
        let y_fp = oy as u64 * y_ratio;
        let sy = (y_fp >> 16) as usize;
        let fy = ((y_fp & 0xFFFF) >> 8) as u32;
        let fy_inv = 256 - fy;
        let sy1 = (sy + 1).min(h - 1);
        let src_row0 = &src[sy * w..];
        let src_row1 = &src[sy1 * w..];
        let dst_row = &mut out[oy * out_w..(oy + 1) * out_w];

        let mut done = 0usize;

        #[cfg(target_arch = "aarch64")]
        if use_tbl2 && !cfg!(miri) && std::arch::is_aarch64_feature_detected!("neon") {
            done = unsafe {
                resize_1ch_fused_neon16(
                    src_row0,
                    src_row1,
                    dst_row,
                    &base16,
                    &idx16_0,
                    &idx16_1,
                    &base8,
                    &idx8_0,
                    &idx8_1,
                    fx_arr,
                    fy as u16,
                    n16,
                    n8_tail,
                    tail_start,
                    use_tbl1_tail,
                )
            };
        }

        #[cfg(target_arch = "x86_64")]
        if use_tbl1_tail && !cfg!(miri) && is_x86_feature_detected!("ssse3") {
            done = unsafe {
                resize_1ch_fused_sse(
                    src_row0, src_row1, dst_row, &base16, &idx16_0, &idx16_1, &base8, &idx8_0,
                    &idx8_1, fx_arr, fy as u16, n16, n8_tail, tail_start,
                )
            };
        }

        for ox in done..out_w {
            let sx = sx_arr[ox] as usize;
            let sx1 = sx1_arr[ox] as usize;
            let fx = fx_arr[ox] as u32;
            let fx_inv = 256 - fx;
            let p00 = src_row0[sx] as u32;
            let p01 = src_row0[sx1] as u32;
            let p10 = src_row1[sx] as u32;
            let p11 = src_row1[sx1] as u32;
            let top = p00 * fx_inv + p01 * fx;
            let bot = p10 * fx_inv + p11 * fx;
            dst_row[ox] = ((top * fy_inv + bot * fy + 32768) >> 16).min(255) as u8;
        }
    }
}

/// SSE resize 1ch: SSSE3 shuffle for 8-pixel gather, SSE2 blend.
/// 16-pixel groups processed as two 8-pixel halves.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn resize_1ch_fused_sse(
    src_row0: &[u8],
    src_row1: &[u8],
    dst_row: &mut [u8],
    base16: &[u32],
    idx16_0: &[[u8; 16]],
    idx16_1: &[[u8; 16]],
    base8: &[u32],
    idx8_0: &[[u8; 8]],
    idx8_1: &[[u8; 8]],
    fx_arr: &[u16],
    fy: u16,
    n16: usize,
    n8_tail: usize,
    tail_start: usize,
) -> usize {
    use std::arch::x86_64::*;

    let r0 = src_row0.as_ptr();
    let r1 = src_row1.as_ptr();
    let dp = dst_row.as_mut_ptr();
    let fy_v = _mm_set1_epi16(fy as i16);
    let fyi_v = _mm_set1_epi16((256 - fy as u32) as i16);
    let v256 = _mm_set1_epi16(256);
    let round = _mm_set1_epi16(128);
    let zero = _mm_setzero_si128();

    // 16-pixel groups: process as two 8-pixel halves using _mm_shuffle_epi8
    for g in 0..n16 {
        let ox = g * 16;
        let base = base16[g] as usize;

        // Low 8 pixels
        let i0_lo = _mm_loadl_epi64(idx16_0[g].as_ptr() as *const __m128i);
        let i1_lo = _mm_loadl_epi64(idx16_1[g].as_ptr() as *const __m128i);
        let row0_v = _mm_loadu_si128(r0.add(base) as *const __m128i);
        let row1_v = _mm_loadu_si128(r1.add(base) as *const __m128i);
        let g00_lo = _mm_shuffle_epi8(row0_v, i0_lo);
        let g01_lo = _mm_shuffle_epi8(row0_v, i1_lo);
        let g10_lo = _mm_shuffle_epi8(row1_v, i0_lo);
        let g11_lo = _mm_shuffle_epi8(row1_v, i1_lo);

        let fx_lo = _mm_loadu_si128(fx_arr.as_ptr().add(ox) as *const __m128i);
        let fxi_lo = _mm_sub_epi16(v256, fx_lo);
        let top_lo = _mm_srli_epi16(
            _mm_add_epi16(
                _mm_mullo_epi16(fxi_lo, _mm_unpacklo_epi8(g00_lo, zero)),
                _mm_mullo_epi16(fx_lo, _mm_unpacklo_epi8(g01_lo, zero)),
            ),
            8,
        );
        let bot_lo = _mm_srli_epi16(
            _mm_add_epi16(
                _mm_mullo_epi16(fxi_lo, _mm_unpacklo_epi8(g10_lo, zero)),
                _mm_mullo_epi16(fx_lo, _mm_unpacklo_epi8(g11_lo, zero)),
            ),
            8,
        );
        let rlo = _mm_srli_epi16(
            _mm_add_epi16(
                _mm_add_epi16(
                    _mm_mullo_epi16(fyi_v, top_lo),
                    _mm_mullo_epi16(fy_v, bot_lo),
                ),
                round,
            ),
            8,
        );

        // High 8 pixels (indices 8..16 may reference bytes 16..31, need second load)
        let i0_hi = _mm_loadl_epi64(idx16_0[g].as_ptr().add(8) as *const __m128i);
        let i1_hi = _mm_loadl_epi64(idx16_1[g].as_ptr().add(8) as *const __m128i);
        let row0_v2 = _mm_loadu_si128(r0.add(base + 16) as *const __m128i);
        let row1_v2 = _mm_loadu_si128(r1.add(base + 16) as *const __m128i);
        // Indices >= 16 need second table. Subtract 16 and use second vector.
        let sub16 = _mm_set1_epi8(16);
        let needs_hi = _mm_cmpgt_epi8(i0_hi, _mm_set1_epi8(15));
        let i0_hi_adj = _mm_sub_epi8(i0_hi, _mm_and_si128(needs_hi, sub16));
        let g00_hi_lo = _mm_shuffle_epi8(row0_v, i0_hi_adj);
        let g00_hi_hi = _mm_shuffle_epi8(row0_v2, i0_hi_adj);
        let g00_hi = _mm_or_si128(
            _mm_andnot_si128(needs_hi, g00_hi_lo),
            _mm_and_si128(needs_hi, g00_hi_hi),
        );

        let needs_hi1 = _mm_cmpgt_epi8(i1_hi, _mm_set1_epi8(15));
        let i1_hi_adj = _mm_sub_epi8(i1_hi, _mm_and_si128(needs_hi1, sub16));
        let g01_hi_lo = _mm_shuffle_epi8(row0_v, i1_hi_adj);
        let g01_hi_hi = _mm_shuffle_epi8(row0_v2, i1_hi_adj);
        let g01_hi = _mm_or_si128(
            _mm_andnot_si128(needs_hi1, g01_hi_lo),
            _mm_and_si128(needs_hi1, g01_hi_hi),
        );

        let g10_hi_lo = _mm_shuffle_epi8(row1_v, i0_hi_adj);
        let g10_hi_hi = _mm_shuffle_epi8(row1_v2, i0_hi_adj);
        let g10_hi = _mm_or_si128(
            _mm_andnot_si128(needs_hi, g10_hi_lo),
            _mm_and_si128(needs_hi, g10_hi_hi),
        );

        let g11_hi_lo = _mm_shuffle_epi8(row1_v, i1_hi_adj);
        let g11_hi_hi = _mm_shuffle_epi8(row1_v2, i1_hi_adj);
        let g11_hi = _mm_or_si128(
            _mm_andnot_si128(needs_hi1, g11_hi_lo),
            _mm_and_si128(needs_hi1, g11_hi_hi),
        );

        let fx_hi = _mm_loadu_si128(fx_arr.as_ptr().add(ox + 8) as *const __m128i);
        let fxi_hi = _mm_sub_epi16(v256, fx_hi);
        let top_hi = _mm_srli_epi16(
            _mm_add_epi16(
                _mm_mullo_epi16(fxi_hi, _mm_unpacklo_epi8(g00_hi, zero)),
                _mm_mullo_epi16(fx_hi, _mm_unpacklo_epi8(g01_hi, zero)),
            ),
            8,
        );
        let bot_hi = _mm_srli_epi16(
            _mm_add_epi16(
                _mm_mullo_epi16(fxi_hi, _mm_unpacklo_epi8(g10_hi, zero)),
                _mm_mullo_epi16(fx_hi, _mm_unpacklo_epi8(g11_hi, zero)),
            ),
            8,
        );
        let rhi = _mm_srli_epi16(
            _mm_add_epi16(
                _mm_add_epi16(
                    _mm_mullo_epi16(fyi_v, top_hi),
                    _mm_mullo_epi16(fy_v, bot_hi),
                ),
                round,
            ),
            8,
        );

        _mm_storeu_si128(dp.add(ox) as *mut __m128i, _mm_packus_epi16(rlo, rhi));
    }

    // 8-pixel tail groups using _mm_shuffle_epi8
    for g in 0..n8_tail {
        let ox = tail_start + g * 8;
        let base = base8[g] as usize;
        let i0 = _mm_loadl_epi64(idx8_0[g].as_ptr() as *const __m128i);
        let i1 = _mm_loadl_epi64(idx8_1[g].as_ptr() as *const __m128i);

        let row0_v = _mm_loadu_si128(r0.add(base) as *const __m128i);
        let row1_v = _mm_loadu_si128(r1.add(base) as *const __m128i);
        let p00 = _mm_unpacklo_epi8(_mm_shuffle_epi8(row0_v, i0), zero);
        let p01 = _mm_unpacklo_epi8(_mm_shuffle_epi8(row0_v, i1), zero);
        let p10 = _mm_unpacklo_epi8(_mm_shuffle_epi8(row1_v, i0), zero);
        let p11 = _mm_unpacklo_epi8(_mm_shuffle_epi8(row1_v, i1), zero);

        let fx = _mm_loadu_si128(fx_arr.as_ptr().add(ox) as *const __m128i);
        let fxi = _mm_sub_epi16(v256, fx);
        let top = _mm_srli_epi16(
            _mm_add_epi16(_mm_mullo_epi16(fxi, p00), _mm_mullo_epi16(fx, p01)),
            8,
        );
        let bot = _mm_srli_epi16(
            _mm_add_epi16(_mm_mullo_epi16(fxi, p10), _mm_mullo_epi16(fx, p11)),
            8,
        );
        let res = _mm_srli_epi16(
            _mm_add_epi16(
                _mm_add_epi16(_mm_mullo_epi16(fyi_v, top), _mm_mullo_epi16(fy_v, bot)),
                round,
            ),
            8,
        );
        _mm_storel_epi64(dp.add(ox) as *mut __m128i, _mm_packus_epi16(res, zero));
    }
    tail_start + n8_tail * 8
}

/// Fused H+V bilinear: processes 16 pixels/iter via TBL2 (32-byte gather).
/// Uses u16-only V-blend: pre-narrows H-blend result to u8, then blends vertically
/// in u16 space. Trades ~0.5 LSB precision for 2× V-blend throughput.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn resize_1ch_fused_neon16(
    src_row0: &[u8],
    src_row1: &[u8],
    dst_row: &mut [u8],
    base16: &[u32],
    idx16_0: &[[u8; 16]],
    idx16_1: &[[u8; 16]],
    base8: &[u32],
    idx8_0: &[[u8; 8]],
    idx8_1: &[[u8; 8]],
    fx_arr: &[u16],
    fy: u16,
    n16: usize,
    n8_tail: usize,
    tail_start: usize,
    use_tbl1_tail: bool,
) -> usize {
    use std::arch::aarch64::*;

    let r0 = src_row0.as_ptr();
    let r1 = src_row1.as_ptr();
    let dp = dst_row.as_mut_ptr();
    let fy_v16 = vdupq_n_u16(fy);
    let fy_inv16 = vdupq_n_u16(256 - fy);
    let v256 = vdupq_n_u16(256);
    let round8 = vdupq_n_u16(128);

    // 16-pixel groups using TBL2
    for g in 0..n16 {
        let ox = g * 16;
        let base = base16[g] as usize;
        let i0 = vld1q_u8(idx16_0[g].as_ptr());
        let i1 = vld1q_u8(idx16_1[g].as_ptr());

        let tbl0 = uint8x16x2_t(vld1q_u8(r0.add(base)), vld1q_u8(r0.add(base + 16)));
        let tbl1 = uint8x16x2_t(vld1q_u8(r1.add(base)), vld1q_u8(r1.add(base + 16)));

        let g00 = vqtbl2q_u8(tbl0, i0);
        let g01 = vqtbl2q_u8(tbl0, i1);
        let g10 = vqtbl2q_u8(tbl1, i0);
        let g11 = vqtbl2q_u8(tbl1, i1);

        // Low 8 pixels: H-blend in u16, narrow to u8, V-blend in u16
        let fx_lo = vld1q_u16(fx_arr.as_ptr().add(ox));
        let fxi_lo = vsubq_u16(v256, fx_lo);
        // top = (p00*(256-fx) + p01*fx) >> 8  → u8
        let top_lo = vshrn_n_u16::<8>(vaddq_u16(
            vmulq_u16(fxi_lo, vmovl_u8(vget_low_u8(g00))),
            vmulq_u16(fx_lo, vmovl_u8(vget_low_u8(g01))),
        ));
        let bot_lo = vshrn_n_u16::<8>(vaddq_u16(
            vmulq_u16(fxi_lo, vmovl_u8(vget_low_u8(g10))),
            vmulq_u16(fx_lo, vmovl_u8(vget_low_u8(g11))),
        ));
        // V-blend: (top*(256-fy) + bot*fy + 128) >> 8
        let rlo = vshrn_n_u16::<8>(vaddq_u16(
            vaddq_u16(
                vmulq_u16(fy_inv16, vmovl_u8(top_lo)),
                vmulq_u16(fy_v16, vmovl_u8(bot_lo)),
            ),
            round8,
        ));

        // High 8 pixels
        let fx_hi = vld1q_u16(fx_arr.as_ptr().add(ox + 8));
        let fxi_hi = vsubq_u16(v256, fx_hi);
        let top_hi = vshrn_n_u16::<8>(vaddq_u16(
            vmulq_u16(fxi_hi, vmovl_u8(vget_high_u8(g00))),
            vmulq_u16(fx_hi, vmovl_u8(vget_high_u8(g01))),
        ));
        let bot_hi = vshrn_n_u16::<8>(vaddq_u16(
            vmulq_u16(fxi_hi, vmovl_u8(vget_high_u8(g10))),
            vmulq_u16(fx_hi, vmovl_u8(vget_high_u8(g11))),
        ));
        let rhi = vshrn_n_u16::<8>(vaddq_u16(
            vaddq_u16(
                vmulq_u16(fy_inv16, vmovl_u8(top_hi)),
                vmulq_u16(fy_v16, vmovl_u8(bot_hi)),
            ),
            round8,
        ));

        vst1q_u8(dp.add(ox), vcombine_u8(rlo, rhi));
    }

    // 8-pixel tail groups using TBL1
    if use_tbl1_tail {
        for g in 0..n8_tail {
            let ox = tail_start + g * 8;
            let base = base8[g] as usize;
            let i0 = vld1_u8(idx8_0[g].as_ptr());
            let i1 = vld1_u8(idx8_1[g].as_ptr());

            let row0_v = vld1q_u8(r0.add(base));
            let row1_v = vld1q_u8(r1.add(base));
            let p00 = vmovl_u8(vqtbl1_u8(row0_v, i0));
            let p01 = vmovl_u8(vqtbl1_u8(row0_v, i1));
            let p10 = vmovl_u8(vqtbl1_u8(row1_v, i0));
            let p11 = vmovl_u8(vqtbl1_u8(row1_v, i1));

            let fx = vld1q_u16(fx_arr.as_ptr().add(ox));
            let fxi = vsubq_u16(v256, fx);
            let top = vshrn_n_u16::<8>(vaddq_u16(vmulq_u16(fxi, p00), vmulq_u16(fx, p01)));
            let bot = vshrn_n_u16::<8>(vaddq_u16(vmulq_u16(fxi, p10), vmulq_u16(fx, p11)));
            let res = vshrn_n_u16::<8>(vaddq_u16(
                vaddq_u16(
                    vmulq_u16(fy_inv16, vmovl_u8(top)),
                    vmulq_u16(fy_v16, vmovl_u8(bot)),
                ),
                round8,
            ));
            vst1_u8(dp.add(ox), res);
        }
        return tail_start + n8_tail * 8;
    }

    n16 * 16
}

/// Horizontal pass for separable bilinear resize (multi-channel).
/// `hbuf[ox*c+ch] = src[sx0*c+ch]*(256-fx) + src[sx1*c+ch]*fx`  (u16, max 65280)
/// fx_rep: replicated fx array `[fx0,fx0,fx0, fx1,fx1,fx1, ...]` for c=3 NEON path.
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn resize_hpass_multi_ch(
    src_row: &[u8],
    hbuf: &mut [u16],
    sx_byte: &[usize],
    sx1_byte: &[usize],
    fx_arr: &[u16],
    fx_rep: &[u16],
    out_w: usize,
    c: usize,
) {
    let sp = src_row.as_ptr();
    let hp = hbuf.as_mut_ptr();

    if c == 3 {
        #[cfg(target_arch = "aarch64")]
        {
            if !cfg!(miri) && std::arch::is_aarch64_feature_detected!("neon") {
                resize_hpass_rgb_neon(sp, hp, sx_byte, sx1_byte, fx_rep, out_w);
                return;
            }
        }
        #[cfg(target_arch = "x86_64")]
        {
            if !cfg!(miri) && is_x86_feature_detected!("sse2") {
                resize_hpass_rgb_sse(sp, hp, sx_byte, sx1_byte, fx_rep, out_w);
                return;
            }
        }
        for ox in 0..out_w {
            let s0 = *sx_byte.get_unchecked(ox);
            let s1 = *sx1_byte.get_unchecked(ox);
            let fx = *fx_arr.get_unchecked(ox);
            let fx_inv = 256 - fx;
            let d = ox * 3;
            *hp.add(d) = (*sp.add(s0) as u16) * fx_inv + (*sp.add(s1) as u16) * fx;
            *hp.add(d + 1) = (*sp.add(s0 + 1) as u16) * fx_inv + (*sp.add(s1 + 1) as u16) * fx;
            *hp.add(d + 2) = (*sp.add(s0 + 2) as u16) * fx_inv + (*sp.add(s1 + 2) as u16) * fx;
        }
    } else if c == 4 {
        for ox in 0..out_w {
            let s0 = *sx_byte.get_unchecked(ox);
            let s1 = *sx1_byte.get_unchecked(ox);
            let fx = *fx_arr.get_unchecked(ox);
            let fx_inv = 256 - fx;
            let d = ox * 4;
            *hp.add(d) = (*sp.add(s0) as u16) * fx_inv + (*sp.add(s1) as u16) * fx;
            *hp.add(d + 1) = (*sp.add(s0 + 1) as u16) * fx_inv + (*sp.add(s1 + 1) as u16) * fx;
            *hp.add(d + 2) = (*sp.add(s0 + 2) as u16) * fx_inv + (*sp.add(s1 + 2) as u16) * fx;
            *hp.add(d + 3) = (*sp.add(s0 + 3) as u16) * fx_inv + (*sp.add(s1 + 3) as u16) * fx;
        }
    } else {
        for ox in 0..out_w {
            let s0 = *sx_byte.get_unchecked(ox);
            let s1 = *sx1_byte.get_unchecked(ox);
            let fx = *fx_arr.get_unchecked(ox);
            let fx_inv = 256 - fx;
            for ch in 0..c {
                let d = ox * c + ch;
                *hp.add(d) = (*sp.add(s0 + ch) as u16) * fx_inv + (*sp.add(s1 + ch) as u16) * fx;
            }
        }
    }
}

/// NEON-accelerated horizontal pass for RGB (c=3) bilinear resize.
/// SSE2 H-pass for RGB resize: scalar gather + SSE blend (8 pixels/iter).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn resize_hpass_rgb_sse(
    sp: *const u8,
    hp: *mut u16,
    sx_byte: &[usize],
    sx1_byte: &[usize],
    fx_rep: &[u16],
    out_w: usize,
) {
    use std::arch::x86_64::*;

    let mut ox = 0usize;
    let frp = fx_rep.as_ptr();
    let v256 = _mm_set1_epi16(256);
    let zero = _mm_setzero_si128();

    while ox + 8 <= out_w {
        let mut left = [0u8; 24];
        let mut right = [0u8; 24];
        for k in 0..8usize {
            let s0 = *sx_byte.get_unchecked(ox + k);
            let s1 = *sx1_byte.get_unchecked(ox + k);
            left[k * 3] = *sp.add(s0);
            left[k * 3 + 1] = *sp.add(s0 + 1);
            left[k * 3 + 2] = *sp.add(s0 + 2);
            right[k * 3] = *sp.add(s1);
            right[k * 3 + 1] = *sp.add(s1 + 1);
            right[k * 3 + 2] = *sp.add(s1 + 2);
        }
        let d = ox * 3;
        for chunk in 0..3usize {
            let off = chunk * 8;
            let f = _mm_loadu_si128(frp.add(d + off) as *const __m128i);
            let fi = _mm_sub_epi16(v256, f);
            let l = _mm_unpacklo_epi8(
                _mm_loadl_epi64(left.as_ptr().add(off) as *const __m128i),
                zero,
            );
            let r = _mm_unpacklo_epi8(
                _mm_loadl_epi64(right.as_ptr().add(off) as *const __m128i),
                zero,
            );
            _mm_storeu_si128(
                hp.add(d + off) as *mut __m128i,
                _mm_add_epi16(_mm_mullo_epi16(l, fi), _mm_mullo_epi16(r, f)),
            );
        }
        ox += 8;
    }
    while ox < out_w {
        let s0 = *sx_byte.get_unchecked(ox);
        let s1 = *sx1_byte.get_unchecked(ox);
        let d = ox * 3;
        let fx = *frp.add(d);
        let fx_inv = 256 - fx;
        *hp.add(d) = (*sp.add(s0) as u16) * fx_inv + (*sp.add(s1) as u16) * fx;
        *hp.add(d + 1) = (*sp.add(s0 + 1) as u16) * fx_inv + (*sp.add(s1 + 1) as u16) * fx;
        *hp.add(d + 2) = (*sp.add(s0 + 2) as u16) * fx_inv + (*sp.add(s1 + 2) as u16) * fx;
        ox += 1;
    }
}

/// Uses precomputed replicated fx array and byte offsets to avoid per-pixel multiplies.
/// Processes 8 output pixels (24 channels = 3 × u16x8) per iteration for full store utilization.
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn resize_hpass_rgb_neon(
    sp: *const u8,
    hp: *mut u16,
    sx_byte: &[usize],
    sx1_byte: &[usize],
    fx_rep: &[u16],
    out_w: usize,
) {
    use std::arch::aarch64::*;

    let mut ox = 0usize;
    let frp = fx_rep.as_ptr();

    // Process 8 output pixels per iteration (24 channels = 3 × u16x8, all full stores)
    while ox + 8 <= out_w {
        // Gather 8 pixel pairs using precomputed byte offsets
        let mut left = [0u8; 24];
        let mut right = [0u8; 24];

        for k in 0..8usize {
            let s0 = *sx_byte.get_unchecked(ox + k);
            let s1 = *sx1_byte.get_unchecked(ox + k);
            left[k * 3] = *sp.add(s0);
            left[k * 3 + 1] = *sp.add(s0 + 1);
            left[k * 3 + 2] = *sp.add(s0 + 2);
            right[k * 3] = *sp.add(s1);
            right[k * 3 + 1] = *sp.add(s1 + 1);
            right[k * 3 + 2] = *sp.add(s1 + 2);
        }

        let d = ox * 3;

        // Chunk 0: channels 0..8 (pixels 0-1 full + pixel 2 partial)
        let f0 = vld1q_u16(frp.add(d));
        let fi0 = vsubq_u16(vdupq_n_u16(256), f0);
        let l0 = vmovl_u8(vld1_u8(left.as_ptr()));
        let r0 = vmovl_u8(vld1_u8(right.as_ptr()));
        vst1q_u16(hp.add(d), vaddq_u16(vmulq_u16(l0, fi0), vmulq_u16(r0, f0)));

        // Chunk 1: channels 8..16 (pixel 2 last + pixels 3-4 + pixel 5 partial)
        let f1 = vld1q_u16(frp.add(d + 8));
        let fi1 = vsubq_u16(vdupq_n_u16(256), f1);
        let l1 = vmovl_u8(vld1_u8(left.as_ptr().add(8)));
        let r1 = vmovl_u8(vld1_u8(right.as_ptr().add(8)));
        vst1q_u16(
            hp.add(d + 8),
            vaddq_u16(vmulq_u16(l1, fi1), vmulq_u16(r1, f1)),
        );

        // Chunk 2: channels 16..24 (pixel 5 rest + pixels 6-7)
        let f2 = vld1q_u16(frp.add(d + 16));
        let fi2 = vsubq_u16(vdupq_n_u16(256), f2);
        let l2 = vmovl_u8(vld1_u8(left.as_ptr().add(16)));
        let r2 = vmovl_u8(vld1_u8(right.as_ptr().add(16)));
        vst1q_u16(
            hp.add(d + 16),
            vaddq_u16(vmulq_u16(l2, fi2), vmulq_u16(r2, f2)),
        );

        ox += 8;
    }

    // 4-pixel remainder
    while ox + 4 <= out_w {
        let mut left = [0u8; 16];
        let mut right = [0u8; 16];

        for k in 0..4usize {
            let s0 = *sx_byte.get_unchecked(ox + k);
            let s1 = *sx1_byte.get_unchecked(ox + k);
            left[k * 3] = *sp.add(s0);
            left[k * 3 + 1] = *sp.add(s0 + 1);
            left[k * 3 + 2] = *sp.add(s0 + 2);
            right[k * 3] = *sp.add(s1);
            right[k * 3 + 1] = *sp.add(s1 + 1);
            right[k * 3 + 2] = *sp.add(s1 + 2);
        }

        let d = ox * 3;
        let f0 = vld1q_u16(frp.add(d));
        let fi0 = vsubq_u16(vdupq_n_u16(256), f0);
        let l0 = vmovl_u8(vld1_u8(left.as_ptr()));
        let r0 = vmovl_u8(vld1_u8(right.as_ptr()));
        vst1q_u16(hp.add(d), vaddq_u16(vmulq_u16(l0, fi0), vmulq_u16(r0, f0)));

        let f1 = vld1q_u16(frp.add(d + 8));
        let fi1 = vsubq_u16(vdupq_n_u16(256), f1);
        let l1 = vmovl_u8(vld1_u8(left.as_ptr().add(8)));
        let r1 = vmovl_u8(vld1_u8(right.as_ptr().add(8)));
        let res1 = vaddq_u16(vmulq_u16(l1, fi1), vmulq_u16(r1, f1));
        let mut tmp = [0u16; 8];
        vst1q_u16(tmp.as_mut_ptr(), res1);
        *hp.add(d + 8) = tmp[0];
        *hp.add(d + 9) = tmp[1];
        *hp.add(d + 10) = tmp[2];
        *hp.add(d + 11) = tmp[3];

        ox += 4;
    }

    // Scalar tail for remaining 0-3 pixels
    while ox < out_w {
        let s0 = *sx_byte.get_unchecked(ox);
        let s1 = *sx1_byte.get_unchecked(ox);
        let d = ox * 3;
        let fx = *frp.add(d);
        let fx_inv = 256 - fx;
        *hp.add(d) = (*sp.add(s0) as u16) * fx_inv + (*sp.add(s1) as u16) * fx;
        *hp.add(d + 1) = (*sp.add(s0 + 1) as u16) * fx_inv + (*sp.add(s1 + 1) as u16) * fx;
        *hp.add(d + 2) = (*sp.add(s0 + 2) as u16) * fx_inv + (*sp.add(s1 + 2) as u16) * fx;
        ox += 1;
    }
}

/// Vertical pass for separable bilinear resize (multi-channel).
/// `out[i] = (hbuf0[i]*(256-fy) + hbuf1[i]*fy + 32768) >> 16`
#[inline]
unsafe fn resize_vpass_multi_ch(hbuf0: &[u16], hbuf1: &[u16], dst: &mut [u8], fy: u16, len: usize) {
    unsafe {
        if cfg!(miri) {
            resize_vpass_scalar(hbuf0, hbuf1, dst, fy, len);
            return;
        }

        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                resize_vpass_neon(hbuf0, hbuf1, dst, fy, len);
                return;
            }
        }

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse2") {
                resize_vpass_sse(hbuf0, hbuf1, dst, fy, len);
                return;
            }
        }

        resize_vpass_scalar(hbuf0, hbuf1, dst, fy, len);
    }
}

#[inline]
unsafe fn resize_vpass_scalar(hbuf0: &[u16], hbuf1: &[u16], dst: &mut [u8], fy: u16, len: usize) {
    unsafe {
        let fy32 = fy as u32;
        let fy_inv = 256 - fy32;
        for i in 0..len {
            let v0 = *hbuf0.get_unchecked(i) as u32;
            let v1 = *hbuf1.get_unchecked(i) as u32;
            let val = (v0 * fy_inv + v1 * fy32 + 32768) >> 16;
            *dst.get_unchecked_mut(i) = val as u8;
        }
    }
}

/// SSE2 V-pass: blend two u16 H-pass rows → u8 output.
/// `val = (hbuf0[i] * (256-fy) + hbuf1[i] * fy + 32768) >> 16`
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn resize_vpass_sse(hbuf0: &[u16], hbuf1: &[u16], dst: &mut [u8], fy: u16, len: usize) {
    use std::arch::x86_64::*;

    let fy_v = _mm_set1_epi32(fy as i32);
    let fyi_v = _mm_set1_epi32(256 - fy as i32);
    let round = _mm_set1_epi32(32768);
    let zero = _mm_setzero_si128();

    let h0 = hbuf0.as_ptr();
    let h1 = hbuf1.as_ptr();
    let dp = dst.as_mut_ptr();
    let mut i = 0usize;

    while i + 16 <= len {
        // First 8 u16 → 8 u8
        let a0 = _mm_loadu_si128(h0.add(i) as *const __m128i);
        let b0 = _mm_loadu_si128(h1.add(i) as *const __m128i);

        let a0_lo = _mm_unpacklo_epi16(a0, zero); // u32
        let a0_hi = _mm_unpackhi_epi16(a0, zero);
        let b0_lo = _mm_unpacklo_epi16(b0, zero);
        let b0_hi = _mm_unpackhi_epi16(b0, zero);

        let r0_lo = _mm_srli_epi32(
            _mm_add_epi32(
                _mm_add_epi32(_mm_mullo_epi32(a0_lo, fyi_v), _mm_mullo_epi32(b0_lo, fy_v)),
                round,
            ),
            16,
        );
        let r0_hi = _mm_srli_epi32(
            _mm_add_epi32(
                _mm_add_epi32(_mm_mullo_epi32(a0_hi, fyi_v), _mm_mullo_epi32(b0_hi, fy_v)),
                round,
            ),
            16,
        );
        let r0_16 = _mm_packs_epi32(r0_lo, r0_hi);

        // Second 8
        let a1 = _mm_loadu_si128(h0.add(i + 8) as *const __m128i);
        let b1 = _mm_loadu_si128(h1.add(i + 8) as *const __m128i);

        let a1_lo = _mm_unpacklo_epi16(a1, zero);
        let a1_hi = _mm_unpackhi_epi16(a1, zero);
        let b1_lo = _mm_unpacklo_epi16(b1, zero);
        let b1_hi = _mm_unpackhi_epi16(b1, zero);

        let r1_lo = _mm_srli_epi32(
            _mm_add_epi32(
                _mm_add_epi32(_mm_mullo_epi32(a1_lo, fyi_v), _mm_mullo_epi32(b1_lo, fy_v)),
                round,
            ),
            16,
        );
        let r1_hi = _mm_srli_epi32(
            _mm_add_epi32(
                _mm_add_epi32(_mm_mullo_epi32(a1_hi, fyi_v), _mm_mullo_epi32(b1_hi, fy_v)),
                round,
            ),
            16,
        );
        let r1_16 = _mm_packs_epi32(r1_lo, r1_hi);

        _mm_storeu_si128(dp.add(i) as *mut __m128i, _mm_packus_epi16(r0_16, r1_16));
        i += 16;
    }

    // Scalar tail
    let fy32 = fy as u32;
    let fy_inv = 256 - fy32;
    while i < len {
        let v0 = *h0.add(i) as u32;
        let v1 = *h1.add(i) as u32;
        *dp.add(i) = ((v0 * fy_inv + v1 * fy32 + 32768) >> 16) as u8;
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn resize_vpass_neon(hbuf0: &[u16], hbuf1: &[u16], dst: &mut [u8], fy: u16, len: usize) {
    use std::arch::aarch64::*;

    let fy_v = vdupq_n_u32(fy as u32);
    let fy_inv_v = vdupq_n_u32(256 - fy as u32);
    let round = vdupq_n_u32(32768);

    let h0 = hbuf0.as_ptr();
    let h1 = hbuf1.as_ptr();
    let dp = dst.as_mut_ptr();
    let mut i = 0usize;

    // Process 16 channel values at a time -> 16 u8 output bytes
    while i + 16 <= len {
        let a0 = vld1q_u16(h0.add(i));
        let b0 = vld1q_u16(h1.add(i));
        let a1 = vld1q_u16(h0.add(i + 8));
        let b1 = vld1q_u16(h1.add(i + 8));

        // First 8: widen to u32, blend, shift
        let a0_lo = vmovl_u16(vget_low_u16(a0));
        let a0_hi = vmovl_u16(vget_high_u16(a0));
        let b0_lo = vmovl_u16(vget_low_u16(b0));
        let b0_hi = vmovl_u16(vget_high_u16(b0));

        let r0_lo = vshrq_n_u32(
            vaddq_u32(
                vaddq_u32(vmulq_u32(a0_lo, fy_inv_v), vmulq_u32(b0_lo, fy_v)),
                round,
            ),
            16,
        );
        let r0_hi = vshrq_n_u32(
            vaddq_u32(
                vaddq_u32(vmulq_u32(a0_hi, fy_inv_v), vmulq_u32(b0_hi, fy_v)),
                round,
            ),
            16,
        );
        let r0_16 = vcombine_u16(vmovn_u32(r0_lo), vmovn_u32(r0_hi));

        // Second 8
        let a1_lo = vmovl_u16(vget_low_u16(a1));
        let a1_hi = vmovl_u16(vget_high_u16(a1));
        let b1_lo = vmovl_u16(vget_low_u16(b1));
        let b1_hi = vmovl_u16(vget_high_u16(b1));

        let r1_lo = vshrq_n_u32(
            vaddq_u32(
                vaddq_u32(vmulq_u32(a1_lo, fy_inv_v), vmulq_u32(b1_lo, fy_v)),
                round,
            ),
            16,
        );
        let r1_hi = vshrq_n_u32(
            vaddq_u32(
                vaddq_u32(vmulq_u32(a1_hi, fy_inv_v), vmulq_u32(b1_hi, fy_v)),
                round,
            ),
            16,
        );
        let r1_16 = vcombine_u16(vmovn_u32(r1_lo), vmovn_u32(r1_hi));

        // Narrow u16 -> u8 (16 values)
        let res8 = vcombine_u8(vqmovn_u16(r0_16), vqmovn_u16(r1_16));
        vst1q_u8(dp.add(i), res8);

        i += 16;
    }

    // Process 8 at a time
    while i + 8 <= len {
        let a0 = vld1q_u16(h0.add(i));
        let b0 = vld1q_u16(h1.add(i));

        let a0_lo = vmovl_u16(vget_low_u16(a0));
        let a0_hi = vmovl_u16(vget_high_u16(a0));
        let b0_lo = vmovl_u16(vget_low_u16(b0));
        let b0_hi = vmovl_u16(vget_high_u16(b0));

        let r_lo = vshrq_n_u32(
            vaddq_u32(
                vaddq_u32(vmulq_u32(a0_lo, fy_inv_v), vmulq_u32(b0_lo, fy_v)),
                round,
            ),
            16,
        );
        let r_hi = vshrq_n_u32(
            vaddq_u32(
                vaddq_u32(vmulq_u32(a0_hi, fy_inv_v), vmulq_u32(b0_hi, fy_v)),
                round,
            ),
            16,
        );
        let r_16 = vcombine_u16(vmovn_u32(r_lo), vmovn_u32(r_hi));
        let res8 = vqmovn_u16(r_16);
        vst1_u8(dp.add(i), res8);

        i += 8;
    }

    // Scalar tail
    let fy32 = fy as u32;
    let fy_inv32 = 256 - fy32;
    while i < len {
        let v0 = *h0.add(i) as u32;
        let v1 = *h1.add(i) as u32;
        let val = (v0 * fy_inv32 + v1 * fy32 + 32768) >> 16;
        *dp.add(i) = val as u8;
        i += 1;
    }
}
