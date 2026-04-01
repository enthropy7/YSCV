//! H.264 YUV-to-RGB conversion, deinterlacing, and FMO helpers.

use super::h264_params::{Pps, Sps};
use crate::VideoError;

// ---------------------------------------------------------------------------
// Interlaced (MBAFF/PAFF) field-pair deinterlacing
// ---------------------------------------------------------------------------

/// Deinterlaces a field pair by interleaving top-field and bottom-field rows.
///
/// `top_field` and `bottom_field` each contain `height` rows of `width * 3` bytes
/// (RGB8). The output frame has `height * 2` rows where even rows come from the
/// top field and odd rows come from the bottom field.
pub fn deinterlace_fields(
    top_field: &[u8],
    bottom_field: &[u8],
    width: usize,
    height: usize,
) -> Vec<u8> {
    let row_bytes = width * 3; // RGB
    let mut frame = vec![0u8; height * 2 * row_bytes];
    for y in 0..height {
        // Even rows from top field
        let dst_even_start = y * 2 * row_bytes;
        let src_top_start = y * row_bytes;
        if src_top_start + row_bytes <= top_field.len() && dst_even_start + row_bytes <= frame.len()
        {
            frame[dst_even_start..dst_even_start + row_bytes]
                .copy_from_slice(&top_field[src_top_start..src_top_start + row_bytes]);
        }
        // Odd rows from bottom field
        let dst_odd_start = (y * 2 + 1) * row_bytes;
        let src_bot_start = y * row_bytes;
        if src_bot_start + row_bytes <= bottom_field.len()
            && dst_odd_start + row_bytes <= frame.len()
        {
            frame[dst_odd_start..dst_odd_start + row_bytes]
                .copy_from_slice(&bottom_field[src_bot_start..src_bot_start + row_bytes]);
        }
    }
    frame
}

// ---------------------------------------------------------------------------
// FMO (Flexible Macroblock Ordering) — slice group map generation
// ---------------------------------------------------------------------------

/// Generates the macroblock-to-slice-group mapping for FMO.
///
/// When `num_slice_groups <= 1`, all MBs belong to group 0 (raster scan order,
/// the default non-FMO case). Otherwise the mapping is determined by
/// `slice_group_map_type` (0–6) as specified in ITU-T H.264 section 8.2.2.
pub fn generate_slice_group_map(pps: &Pps, sps: &Sps) -> Vec<u8> {
    let pic_width = sps.pic_width_in_mbs as usize;
    let pic_height = sps.pic_height_in_map_units as usize;
    let num_mbs = pic_width * pic_height;
    let mut map = vec![0u8; num_mbs];

    if pps.num_slice_groups <= 1 {
        return map; // all MBs in group 0
    }

    let num_groups = pps.num_slice_groups as usize;

    match pps.slice_group_map_type {
        0 => {
            // Interleaved: run_length based cyclic assignment
            let mut i = 0;
            loop {
                if i >= num_mbs {
                    break;
                }
                for group in 0..num_groups {
                    let run = if group < pps.run_length_minus1.len() {
                        pps.run_length_minus1[group] as usize + 1
                    } else {
                        1
                    };
                    for _ in 0..run {
                        if i >= num_mbs {
                            break;
                        }
                        map[i] = group as u8;
                        i += 1;
                    }
                }
            }
        }
        1 => {
            // Dispersed: modular mapping
            for i in 0..num_mbs {
                let x = i % pic_width;
                let y = i / pic_width;
                let group = ((x + ((y * num_groups) / 2)) % num_groups) as u8;
                map[i] = group;
            }
        }
        2 => {
            // Foreground with left-over: rectangular regions
            // Initially all MBs in the last group (background)
            let bg_group = (num_groups - 1) as u8;
            for m in map.iter_mut() {
                *m = bg_group;
            }
            // Assign foreground regions (highest group index has priority)
            for group in (0..num_groups.saturating_sub(1)).rev() {
                if group >= pps.top_left.len() || group >= pps.bottom_right.len() {
                    continue;
                }
                let tl = pps.top_left[group] as usize;
                let br = pps.bottom_right[group] as usize;
                let tl_x = tl % pic_width;
                let tl_y = tl / pic_width;
                let br_x = br % pic_width;
                let br_y = br / pic_width;
                for y in tl_y..=br_y.min(pic_height.saturating_sub(1)) {
                    for x in tl_x..=br_x.min(pic_width.saturating_sub(1)) {
                        let idx = y * pic_width + x;
                        if idx < num_mbs {
                            map[idx] = group as u8;
                        }
                    }
                }
            }
        }
        3..=5 => {
            // Box-out / raster-scan / wipe: evolving slice groups
            // These types use slice_group_change_rate to determine a moving
            // boundary. For a single-frame decode the boundary position comes
            // from `slice_group_change_cycle` in the slice header. As a
            // simplification we map the first `change_rate` MBs to group 0
            // and the rest to group 1.
            let change = (pps.slice_group_change_rate as usize).min(num_mbs);
            for (i, m) in map.iter_mut().enumerate() {
                *m = if i < change { 0 } else { 1 };
            }
        }
        6 => {
            // Explicit: per-MB group IDs stored in PPS
            for (i, m) in map.iter_mut().enumerate() {
                if i < pps.slice_group_id.len() {
                    *m = pps.slice_group_id[i] as u8;
                }
            }
        }
        _ => {
            // Unknown type — fall back to single group
        }
    }

    map
}

// ---------------------------------------------------------------------------
// Chroma format helpers (High 4:2:2 / 4:4:4 profile support)
// ---------------------------------------------------------------------------

/// Returns the chroma plane dimensions `(chroma_width, chroma_height)` given
/// the luma dimensions and `chroma_format_idc` from the SPS.
///
/// - 0 = monochrome (no chroma planes)
/// - 1 = YUV 4:2:0 (default, half width and half height)
/// - 2 = YUV 4:2:2 (half width, full height)
/// - 3 = YUV 4:4:4 (full width, full height)
pub fn chroma_dimensions(width: usize, height: usize, chroma_format: u32) -> (usize, usize) {
    match chroma_format {
        0 => (0, 0),                  // monochrome
        1 => (width / 2, height / 2), // 4:2:0
        2 => (width / 2, height),     // 4:2:2
        3 => (width, height),         // 4:4:4
        _ => (width / 2, height / 2), // default to 4:2:0
    }
}

/// Converts YUV 4:2:2 planar to RGB8 interleaved using BT.601 coefficients.
///
/// Chroma planes are half-width, full-height relative to luma.
pub fn yuv422_to_rgb8(
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    width: usize,
    height: usize,
) -> Result<Vec<u8>, VideoError> {
    let expected_y = width * height;
    let expected_uv = (width / 2) * height;

    if y_plane.len() < expected_y {
        return Err(VideoError::Codec(format!(
            "Y plane too small: expected {expected_y}, got {}",
            y_plane.len()
        )));
    }
    if u_plane.len() < expected_uv || v_plane.len() < expected_uv {
        return Err(VideoError::Codec(format!(
            "UV planes too small for 4:2:2: expected {expected_uv}, got U={} V={}",
            u_plane.len(),
            v_plane.len()
        )));
    }

    let mut rgb = vec![0u8; width * height * 3];
    let uv_stride = width / 2;

    for row in 0..height {
        let y_off = row * width;
        let uv_off = row * uv_stride;

        for col in 0..width {
            let y_val = y_plane[y_off + col] as i16;
            let u_val = u_plane[uv_off + col / 2] as i16 - 128;
            let v_val = v_plane[uv_off + col / 2] as i16 - 128;

            let r = y_val + ((v_val * 179) >> 7);
            let g = y_val - ((u_val * 44 + v_val * 91) >> 7);
            let b = y_val + ((u_val * 227) >> 7);

            let idx = (row * width + col) * 3;
            rgb[idx] = r.clamp(0, 255) as u8;
            rgb[idx + 1] = g.clamp(0, 255) as u8;
            rgb[idx + 2] = b.clamp(0, 255) as u8;
        }
    }

    Ok(rgb)
}

/// Converts YUV 4:4:4 planar to RGB8 interleaved using BT.601 coefficients.
///
/// All three planes have the same dimensions (no chroma subsampling).
pub fn yuv444_to_rgb8(
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    width: usize,
    height: usize,
) -> Result<Vec<u8>, VideoError> {
    let expected = width * height;

    if y_plane.len() < expected {
        return Err(VideoError::Codec(format!(
            "Y plane too small: expected {expected}, got {}",
            y_plane.len()
        )));
    }
    if u_plane.len() < expected || v_plane.len() < expected {
        return Err(VideoError::Codec(format!(
            "UV planes too small for 4:4:4: expected {expected}, got U={} V={}",
            u_plane.len(),
            v_plane.len()
        )));
    }

    let mut rgb = vec![0u8; width * height * 3];

    for i in 0..expected {
        let y_val = y_plane[i] as i16;
        let u_val = u_plane[i] as i16 - 128;
        let v_val = v_plane[i] as i16 - 128;

        let r = y_val + ((v_val * 179) >> 7);
        let g = y_val - ((u_val * 44 + v_val * 91) >> 7);
        let b = y_val + ((u_val * 227) >> 7);

        let idx = i * 3;
        rgb[idx] = r.clamp(0, 255) as u8;
        rgb[idx + 1] = g.clamp(0, 255) as u8;
        rgb[idx + 2] = b.clamp(0, 255) as u8;
    }

    Ok(rgb)
}

/// Converts a monochrome (luma-only) plane to RGB8 (grayscale).
pub fn mono_to_rgb8(y_plane: &[u8], width: usize, height: usize) -> Result<Vec<u8>, VideoError> {
    let expected = width * height;
    if y_plane.len() < expected {
        return Err(VideoError::Codec(format!(
            "Y plane too small: expected {expected}, got {}",
            y_plane.len()
        )));
    }
    let mut rgb = vec![0u8; expected * 3];
    for i in 0..expected {
        let v = y_plane[i];
        let idx = i * 3;
        rgb[idx] = v;
        rgb[idx + 1] = v;
        rgb[idx + 2] = v;
    }
    Ok(rgb)
}

/// Dispatches YUV-to-RGB conversion based on `chroma_format_idc`.
pub(crate) fn yuv_to_rgb8_by_format(
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    width: usize,
    height: usize,
    chroma_format_idc: u32,
) -> Result<Vec<u8>, VideoError> {
    match chroma_format_idc {
        0 => mono_to_rgb8(y_plane, width, height),
        1 => yuv420_to_rgb8(y_plane, u_plane, v_plane, width, height),
        2 => yuv422_to_rgb8(y_plane, u_plane, v_plane, width, height),
        3 => yuv444_to_rgb8(y_plane, u_plane, v_plane, width, height),
        _ => yuv420_to_rgb8(y_plane, u_plane, v_plane, width, height),
    }
}
// ---------------------------------------------------------------------------
// YUV to RGB conversion
// ---------------------------------------------------------------------------

/// Converts YUV 4:2:0 planar to RGB8 interleaved using BT.601 coefficients.
///
/// Input: separate Y, U, V planes. Y is `width * height`, U and V are `(width/2) * (height/2)`.
/// Output: RGB8 interleaved, `width * height * 3` bytes.
///
/// Uses SIMD (NEON on aarch64, SSE2 on x86_64) with fixed-point i16 arithmetic
/// and multi-threaded row processing for high throughput.
#[allow(unsafe_code)]
pub fn yuv420_to_rgb8(
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    width: usize,
    height: usize,
) -> Result<Vec<u8>, VideoError> {
    let expected_y = width * height;
    let expected_uv = (width / 2) * (height / 2);

    if y_plane.len() < expected_y {
        return Err(VideoError::Codec(format!(
            "Y plane too small: expected {expected_y}, got {}",
            y_plane.len()
        )));
    }
    if u_plane.len() < expected_uv || v_plane.len() < expected_uv {
        return Err(VideoError::Codec(format!(
            "UV planes too small: expected {expected_uv}, got U={} V={}",
            u_plane.len(),
            v_plane.len()
        )));
    }

    let mut rgb = vec![0u8; width * height * 3];
    let uv_stride = width / 2;

    if height < 4 {
        // Single-threaded path for very small images.
        yuv420_to_rgb8_rows(
            y_plane, u_plane, v_plane, &mut rgb, width, uv_stride, 0, height,
        );
    } else {
        // Use rayon par_chunks_mut for near-zero thread dispatch overhead
        // (rayon reuses a warm thread pool vs std::thread::scope which spawns
        // new threads each call).
        use rayon::prelude::*;

        let row_bytes = width * 3;
        rgb.par_chunks_mut(row_bytes)
            .enumerate()
            .for_each(|(row_idx, row_slice)| {
                yuv420_to_rgb8_rows(
                    y_plane,
                    u_plane,
                    v_plane,
                    row_slice,
                    width,
                    uv_stride,
                    row_idx,
                    row_idx + 1,
                );
            });
    }

    Ok(rgb)
}

/// Like [`yuv420_to_rgb8`] but writes into a caller-supplied buffer,
/// avoiding per-frame allocation. `rgb_out` must be at least `width * height * 3` bytes.
pub fn yuv420_to_rgb8_into(
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    width: usize,
    height: usize,
    rgb_out: &mut Vec<u8>,
) -> Result<(), VideoError> {
    let expected_y = width * height;
    let expected_uv = (width / 2) * (height / 2);

    if y_plane.len() < expected_y {
        return Err(VideoError::Codec(format!(
            "Y plane too small: expected {expected_y}, got {}",
            y_plane.len()
        )));
    }
    if u_plane.len() < expected_uv || v_plane.len() < expected_uv {
        return Err(VideoError::Codec(format!(
            "UV planes too small: expected {expected_uv}, got U={} V={}",
            u_plane.len(),
            v_plane.len()
        )));
    }

    let total = width * height * 3;
    rgb_out.resize(total, 0);
    let uv_stride = width / 2;

    if height < 4 {
        yuv420_to_rgb8_rows(
            y_plane, u_plane, v_plane, rgb_out, width, uv_stride, 0, height,
        );
    } else {
        use rayon::prelude::*;
        let row_bytes = width * 3;
        rgb_out
            .par_chunks_mut(row_bytes)
            .enumerate()
            .for_each(|(row_idx, row_slice)| {
                yuv420_to_rgb8_rows(
                    y_plane,
                    u_plane,
                    v_plane,
                    row_slice,
                    width,
                    uv_stride,
                    row_idx,
                    row_idx + 1,
                );
            });
    }

    Ok(())
}

/// Convert rows `start_row..end_row` from YUV420 to RGB8.
/// `rgb_out` starts at the byte corresponding to `start_row`.
#[inline]
#[allow(unsafe_code)]
fn yuv420_to_rgb8_rows(
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    rgb_out: &mut [u8],
    width: usize,
    uv_stride: usize,
    start_row: usize,
    end_row: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: feature detected at runtime.
            unsafe {
                yuv420_to_rgb8_rows_neon(
                    y_plane, u_plane, v_plane, rgb_out, width, uv_stride, start_row, end_row,
                );
            }
            return;
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                yuv420_to_rgb8_rows_avx2(
                    y_plane, u_plane, v_plane, rgb_out, width, uv_stride, start_row, end_row,
                );
            }
            return;
        }
        if is_x86_feature_detected!("sse2") {
            unsafe {
                yuv420_to_rgb8_rows_sse2(
                    y_plane, u_plane, v_plane, rgb_out, width, uv_stride, start_row, end_row,
                );
            }
            return;
        }
    }

    yuv420_to_rgb8_rows_scalar(
        y_plane, u_plane, v_plane, rgb_out, width, uv_stride, start_row, end_row,
    );
}

/// Scalar fallback for YUV420→RGB8 conversion.
#[inline]
fn yuv420_to_rgb8_rows_scalar(
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    rgb_out: &mut [u8],
    width: usize,
    uv_stride: usize,
    start_row: usize,
    end_row: usize,
) {
    // BT.601 fixed-point constants (Q7, fits in i16 without overflow):
    // 1.402 * 128 ≈ 179, 0.344 * 128 ≈ 44, 0.714 * 128 ≈ 91, 1.772 * 128 ≈ 227
    // R = Y + (V-128)*179 >> 7
    // G = Y - ((U-128)*44 + (V-128)*91) >> 7
    // B = Y + (U-128)*227 >> 7
    for row in start_row..end_row {
        let out_row = row - start_row;
        let y_row_off = row * width;
        let uv_row_off = (row / 2) * uv_stride;

        for col in 0..width {
            let y_val = y_plane[y_row_off + col] as i16;
            let u_val = u_plane[uv_row_off + col / 2] as i16 - 128;
            let v_val = v_plane[uv_row_off + col / 2] as i16 - 128;

            let r = y_val + ((v_val * 179) >> 7);
            let g = y_val - ((u_val * 44 + v_val * 91) >> 7);
            let b = y_val + ((u_val * 227) >> 7);

            let idx = (out_row * width + col) * 3;
            rgb_out[idx] = r.clamp(0, 255) as u8;
            rgb_out[idx + 1] = g.clamp(0, 255) as u8;
            rgb_out[idx + 2] = b.clamp(0, 255) as u8;
        }
    }
}

/// NEON-accelerated YUV420→RGB8 conversion (aarch64).
/// Processes 8 pixels per iteration using i16 fixed-point arithmetic.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn yuv420_to_rgb8_rows_neon(
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    rgb_out: &mut [u8],
    width: usize,
    uv_stride: usize,
    start_row: usize,
    end_row: usize,
) {
    use std::arch::aarch64::*;

    // BT.601 fixed-point Q7 constants (fit in i16 without overflow)
    let c_179 = vdupq_n_s16(179); // 1.402 * 128
    let c_44 = vdupq_n_s16(44); // 0.344 * 128
    let c_91 = vdupq_n_s16(91); // 0.714 * 128
    let c_227 = vdupq_n_s16(227); // 1.772 * 128
    let c_128 = vdupq_n_s16(128);

    for row in start_row..end_row {
        let out_row = row - start_row;
        let y_row_ptr = y_plane.as_ptr().add(row * width);
        let uv_row = (row / 2) * uv_stride;
        let u_row_ptr = u_plane.as_ptr().add(uv_row);
        let v_row_ptr = v_plane.as_ptr().add(uv_row);
        let rgb_row_ptr = rgb_out.as_mut_ptr().add(out_row * width * 3);

        let mut col = 0usize;

        // Process 16 pixels per iteration (16 Y, 8 U, 8 V)
        while col + 16 <= width {
            // Load 16 Y values
            let y16 = vld1q_u8(y_row_ptr.add(col));
            // Load 8 U and 8 V values, each covers 16 horizontal pixels
            let u8_vals = vld1_u8(u_row_ptr.add(col / 2));
            let v8_vals = vld1_u8(v_row_ptr.add(col / 2));

            // Duplicate each U/V to cover 2 pixels horizontally → 16 values
            let u16_dup = vcombine_u8(vzip1_u8(u8_vals, u8_vals), vzip2_u8(u8_vals, u8_vals));
            let v16_dup = vcombine_u8(vzip1_u8(v8_vals, v8_vals), vzip2_u8(v8_vals, v8_vals));

            // Process low 8 pixels
            let y_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(y16)));
            let u_lo = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(u16_dup))), c_128);
            let v_lo = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v16_dup))), c_128);

            // R = Y + (V * 359) >> 8
            let r_lo = vaddq_s16(y_lo, vshrq_n_s16::<7>(vmulq_s16(v_lo, c_179)));
            // G = Y - ((U * 88 + V * 183) >> 8)
            let g_lo = vsubq_s16(
                y_lo,
                vshrq_n_s16::<7>(vaddq_s16(vmulq_s16(u_lo, c_44), vmulq_s16(v_lo, c_91))),
            );
            // B = Y + (U * 454) >> 8
            let b_lo = vaddq_s16(y_lo, vshrq_n_s16::<7>(vmulq_s16(u_lo, c_227)));

            // Saturate to u8
            let r_lo_u8 = vqmovun_s16(r_lo);
            let g_lo_u8 = vqmovun_s16(g_lo);
            let b_lo_u8 = vqmovun_s16(b_lo);

            // Process high 8 pixels
            let y_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(y16)));
            let u_hi = vsubq_s16(
                vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(u16_dup))),
                c_128,
            );
            let v_hi = vsubq_s16(
                vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v16_dup))),
                c_128,
            );

            let r_hi = vaddq_s16(y_hi, vshrq_n_s16::<7>(vmulq_s16(v_hi, c_179)));
            let g_hi = vsubq_s16(
                y_hi,
                vshrq_n_s16::<7>(vaddq_s16(vmulq_s16(u_hi, c_44), vmulq_s16(v_hi, c_91))),
            );
            let b_hi = vaddq_s16(y_hi, vshrq_n_s16::<7>(vmulq_s16(u_hi, c_227)));

            let r_hi_u8 = vqmovun_s16(r_hi);
            let g_hi_u8 = vqmovun_s16(g_hi);
            let b_hi_u8 = vqmovun_s16(b_hi);

            // Interleave R, G, B into RGB8 and store
            let rgb_lo = uint8x8x3_t(r_lo_u8, g_lo_u8, b_lo_u8);
            vst3_u8(rgb_row_ptr.add(col * 3), rgb_lo);

            let rgb_hi = uint8x8x3_t(r_hi_u8, g_hi_u8, b_hi_u8);
            vst3_u8(rgb_row_ptr.add((col + 8) * 3), rgb_hi);

            col += 16;
        }

        // Process 8 pixels
        if col + 8 <= width {
            let y8_vals = vld1_u8(y_row_ptr.add(col));
            let u4_vals_raw = u_row_ptr.add(col / 2);
            let v4_vals_raw = v_row_ptr.add(col / 2);

            // Load 4 U/V values manually and duplicate
            let mut u_buf = [0u8; 8];
            let mut v_buf = [0u8; 8];
            for i in 0..4 {
                u_buf[i * 2] = *u4_vals_raw.add(i);
                u_buf[i * 2 + 1] = *u4_vals_raw.add(i);
                v_buf[i * 2] = *v4_vals_raw.add(i);
                v_buf[i * 2 + 1] = *v4_vals_raw.add(i);
            }
            let u8_dup = vld1_u8(u_buf.as_ptr());
            let v8_dup = vld1_u8(v_buf.as_ptr());

            let y_i16 = vreinterpretq_s16_u16(vmovl_u8(y8_vals));
            let u_i16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(u8_dup)), c_128);
            let v_i16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(v8_dup)), c_128);

            let r = vaddq_s16(y_i16, vshrq_n_s16::<7>(vmulq_s16(v_i16, c_179)));
            let g = vsubq_s16(
                y_i16,
                vshrq_n_s16::<7>(vaddq_s16(vmulq_s16(u_i16, c_44), vmulq_s16(v_i16, c_91))),
            );
            let b = vaddq_s16(y_i16, vshrq_n_s16::<7>(vmulq_s16(u_i16, c_227)));

            let r_u8 = vqmovun_s16(r);
            let g_u8 = vqmovun_s16(g);
            let b_u8 = vqmovun_s16(b);

            let rgb = uint8x8x3_t(r_u8, g_u8, b_u8);
            vst3_u8(rgb_row_ptr.add(col * 3), rgb);

            col += 8;
        }

        // Scalar tail for remaining pixels
        while col < width {
            let y_val = *y_row_ptr.add(col) as i16;
            let u_val = *u_row_ptr.add(col / 2) as i16 - 128;
            let v_val = *v_row_ptr.add(col / 2) as i16 - 128;

            let r = y_val + ((v_val * 179) >> 7);
            let g = y_val - (((u_val * 44) + (v_val * 91)) >> 7);
            let b = y_val + ((u_val * 227) >> 7);

            let idx = col * 3;
            *rgb_row_ptr.add(idx) = r.clamp(0, 255) as u8;
            *rgb_row_ptr.add(idx + 1) = g.clamp(0, 255) as u8;
            *rgb_row_ptr.add(idx + 2) = b.clamp(0, 255) as u8;

            col += 1;
        }
    }
}

/// AVX2-accelerated YUV420→RGB8 conversion (x86_64).
/// Processes 16 pixels per iteration using i16 fixed-point arithmetic.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn yuv420_to_rgb8_rows_avx2(
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    rgb_out: &mut [u8],
    width: usize,
    uv_stride: usize,
    start_row: usize,
    end_row: usize,
) {
    use std::arch::x86_64::*;

    // BT.601 fixed-point Q7 constants
    let c_179 = _mm256_set1_epi16(179);
    let c_44 = _mm256_set1_epi16(44);
    let c_91 = _mm256_set1_epi16(91);
    let c_227 = _mm256_set1_epi16(227);
    let c_128 = _mm256_set1_epi16(128);
    let zero = _mm256_setzero_si256();

    for row in start_row..end_row {
        let out_row = row - start_row;
        let y_row_ptr = y_plane.as_ptr().add(row * width);
        let uv_row = (row / 2) * uv_stride;
        let u_row_ptr = u_plane.as_ptr().add(uv_row);
        let v_row_ptr = v_plane.as_ptr().add(uv_row);
        let rgb_row_ptr = rgb_out.as_mut_ptr().add(out_row * width * 3);

        let mut col = 0usize;

        // Process 16 pixels per iteration (16 Y, 8 U, 8 V)
        while col + 16 <= width {
            // Load 16 Y values into the low 128 bits, widen to i16 in 256 bits
            let y16 = _mm_loadu_si128(y_row_ptr.add(col) as *const __m128i);
            let y_lo = _mm256_cvtepu8_epi16(y16);

            // Load 8 U/V values, duplicate each for 2 horizontal pixels → 16 values
            let mut u_buf = [0u8; 16];
            let mut v_buf = [0u8; 16];
            for i in 0..8 {
                u_buf[i * 2] = *u_row_ptr.add(col / 2 + i);
                u_buf[i * 2 + 1] = *u_row_ptr.add(col / 2 + i);
                v_buf[i * 2] = *v_row_ptr.add(col / 2 + i);
                v_buf[i * 2 + 1] = *v_row_ptr.add(col / 2 + i);
            }
            let u16_raw = _mm_loadu_si128(u_buf.as_ptr() as *const __m128i);
            let v16_raw = _mm_loadu_si128(v_buf.as_ptr() as *const __m128i);

            let u_i16 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(u16_raw), c_128);
            let v_i16 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(v16_raw), c_128);

            // R = Y + (V * 179) >> 7
            let r = _mm256_add_epi16(
                y_lo,
                _mm256_srai_epi16::<7>(_mm256_mullo_epi16(v_i16, c_179)),
            );
            // G = Y - ((U * 44 + V * 91) >> 7)
            let g = _mm256_sub_epi16(
                y_lo,
                _mm256_srai_epi16::<7>(_mm256_add_epi16(
                    _mm256_mullo_epi16(u_i16, c_44),
                    _mm256_mullo_epi16(v_i16, c_91),
                )),
            );
            // B = Y + (U * 227) >> 7
            let b = _mm256_add_epi16(
                y_lo,
                _mm256_srai_epi16::<7>(_mm256_mullo_epi16(u_i16, c_227)),
            );

            // Saturating pack i16 → u8 (packus packs lanes independently, then
            // vpermute corrects the cross-lane ordering)
            let r_packed = _mm256_packus_epi16(r, zero);
            let g_packed = _mm256_packus_epi16(g, zero);
            let b_packed = _mm256_packus_epi16(b, zero);

            // Extract lower 16 bytes (the valid u8 results) after fixing lane order
            let r_perm = _mm256_permute4x64_epi64::<0xD8>(r_packed);
            let g_perm = _mm256_permute4x64_epi64::<0xD8>(g_packed);
            let b_perm = _mm256_permute4x64_epi64::<0xD8>(b_packed);

            let r_lo128 = _mm256_castsi256_si128(r_perm);
            let g_lo128 = _mm256_castsi256_si128(g_perm);
            let b_lo128 = _mm256_castsi256_si128(b_perm);

            // Interleave and store RGB (manual interleave since x86 has no vst3)
            let mut r_arr = [0u8; 16];
            let mut g_arr = [0u8; 16];
            let mut b_arr = [0u8; 16];
            _mm_storeu_si128(r_arr.as_mut_ptr() as *mut __m128i, r_lo128);
            _mm_storeu_si128(g_arr.as_mut_ptr() as *mut __m128i, g_lo128);
            _mm_storeu_si128(b_arr.as_mut_ptr() as *mut __m128i, b_lo128);

            let mut rgb_buf = [0u8; 48];
            for i in 0..16 {
                rgb_buf[i * 3] = r_arr[i];
                rgb_buf[i * 3 + 1] = g_arr[i];
                rgb_buf[i * 3 + 2] = b_arr[i];
            }
            std::ptr::copy_nonoverlapping(rgb_buf.as_ptr(), rgb_row_ptr.add(col * 3), 48);

            col += 16;
        }

        // Process 8 pixels using 128-bit subset
        while col + 8 <= width {
            let y8 = _mm_loadl_epi64(y_row_ptr.add(col) as *const __m128i);
            let zero128 = _mm_setzero_si128();
            let y_i16 = _mm_unpacklo_epi8(y8, zero128);

            let c_179_128 = _mm_set1_epi16(179);
            let c_44_128 = _mm_set1_epi16(44);
            let c_91_128 = _mm_set1_epi16(91);
            let c_227_128 = _mm_set1_epi16(227);
            let c_128_128 = _mm_set1_epi16(128);

            let mut u_buf = [0u8; 8];
            let mut v_buf = [0u8; 8];
            for i in 0..4 {
                u_buf[i * 2] = *u_row_ptr.add(col / 2 + i);
                u_buf[i * 2 + 1] = *u_row_ptr.add(col / 2 + i);
                v_buf[i * 2] = *v_row_ptr.add(col / 2 + i);
                v_buf[i * 2 + 1] = *v_row_ptr.add(col / 2 + i);
            }
            let u8_dup = _mm_loadl_epi64(u_buf.as_ptr() as *const __m128i);
            let v8_dup = _mm_loadl_epi64(v_buf.as_ptr() as *const __m128i);

            let u_i16 = _mm_sub_epi16(_mm_unpacklo_epi8(u8_dup, zero128), c_128_128);
            let v_i16 = _mm_sub_epi16(_mm_unpacklo_epi8(v8_dup, zero128), c_128_128);

            let r = _mm_add_epi16(
                y_i16,
                _mm_srai_epi16::<7>(_mm_mullo_epi16(v_i16, c_179_128)),
            );
            let g = _mm_sub_epi16(
                y_i16,
                _mm_srai_epi16::<7>(_mm_add_epi16(
                    _mm_mullo_epi16(u_i16, c_44_128),
                    _mm_mullo_epi16(v_i16, c_91_128),
                )),
            );
            let b = _mm_add_epi16(
                y_i16,
                _mm_srai_epi16::<7>(_mm_mullo_epi16(u_i16, c_227_128)),
            );

            let r_u8 = _mm_packus_epi16(r, zero128);
            let g_u8 = _mm_packus_epi16(g, zero128);
            let b_u8 = _mm_packus_epi16(b, zero128);

            let mut r_arr = [0u8; 8];
            let mut g_arr = [0u8; 8];
            let mut b_arr = [0u8; 8];
            _mm_storel_epi64(r_arr.as_mut_ptr() as *mut __m128i, r_u8);
            _mm_storel_epi64(g_arr.as_mut_ptr() as *mut __m128i, g_u8);
            _mm_storel_epi64(b_arr.as_mut_ptr() as *mut __m128i, b_u8);

            let mut rgb_buf = [0u8; 24];
            for i in 0..8 {
                rgb_buf[i * 3] = r_arr[i];
                rgb_buf[i * 3 + 1] = g_arr[i];
                rgb_buf[i * 3 + 2] = b_arr[i];
            }
            std::ptr::copy_nonoverlapping(rgb_buf.as_ptr(), rgb_row_ptr.add(col * 3), 24);

            col += 8;
        }

        // Scalar tail
        while col < width {
            let y_val = *y_row_ptr.add(col) as i16;
            let u_val = *u_row_ptr.add(col / 2) as i16 - 128;
            let v_val = *v_row_ptr.add(col / 2) as i16 - 128;

            let r = y_val + ((v_val * 179) >> 7);
            let g = y_val - (((u_val * 44) + (v_val * 91)) >> 7);
            let b = y_val + ((u_val * 227) >> 7);

            let idx = col * 3;
            *rgb_row_ptr.add(idx) = r.clamp(0, 255) as u8;
            *rgb_row_ptr.add(idx + 1) = g.clamp(0, 255) as u8;
            *rgb_row_ptr.add(idx + 2) = b.clamp(0, 255) as u8;

            col += 1;
        }
    }
}

/// SSE2-accelerated YUV420→RGB8 conversion (x86_64).
/// Processes 8 pixels per iteration using i16 fixed-point arithmetic.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn yuv420_to_rgb8_rows_sse2(
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    rgb_out: &mut [u8],
    width: usize,
    uv_stride: usize,
    start_row: usize,
    end_row: usize,
) {
    use std::arch::x86_64::*;

    // BT.601 fixed-point Q7 constants (fit in i16 without overflow)
    let c_179 = _mm_set1_epi16(179); // 1.402 * 128
    let c_44 = _mm_set1_epi16(44); // 0.344 * 128
    let c_91 = _mm_set1_epi16(91); // 0.714 * 128
    let c_227 = _mm_set1_epi16(227); // 1.772 * 128
    let c_128 = _mm_set1_epi16(128);
    let zero = _mm_setzero_si128();

    for row in start_row..end_row {
        let out_row = row - start_row;
        let y_row_ptr = y_plane.as_ptr().add(row * width);
        let uv_row = (row / 2) * uv_stride;
        let u_row_ptr = u_plane.as_ptr().add(uv_row);
        let v_row_ptr = v_plane.as_ptr().add(uv_row);
        let rgb_row_ptr = rgb_out.as_mut_ptr().add(out_row * width * 3);

        let mut col = 0usize;

        // Process 8 pixels per iteration
        while col + 8 <= width {
            // Load 8 Y values, widen to i16
            let y8 = _mm_loadl_epi64(y_row_ptr.add(col) as *const __m128i);
            let y_i16 = _mm_unpacklo_epi8(y8, zero);

            // Load 4 U/V values, duplicate each for 2 horizontal pixels
            let mut u_buf = [0u8; 8];
            let mut v_buf = [0u8; 8];
            for i in 0..4 {
                u_buf[i * 2] = *u_row_ptr.add(col / 2 + i);
                u_buf[i * 2 + 1] = *u_row_ptr.add(col / 2 + i);
                v_buf[i * 2] = *v_row_ptr.add(col / 2 + i);
                v_buf[i * 2 + 1] = *v_row_ptr.add(col / 2 + i);
            }
            let u8_dup = _mm_loadl_epi64(u_buf.as_ptr() as *const __m128i);
            let v8_dup = _mm_loadl_epi64(v_buf.as_ptr() as *const __m128i);

            let u_i16 = _mm_sub_epi16(_mm_unpacklo_epi8(u8_dup, zero), c_128);
            let v_i16 = _mm_sub_epi16(_mm_unpacklo_epi8(v8_dup, zero), c_128);

            // R = Y + (V * 359) >> 8
            let r = _mm_add_epi16(y_i16, _mm_srai_epi16::<7>(_mm_mullo_epi16(v_i16, c_179)));
            // G = Y - ((U * 44 + V * 91) >> 7)
            let g = _mm_sub_epi16(
                y_i16,
                _mm_srai_epi16::<7>(_mm_add_epi16(
                    _mm_mullo_epi16(u_i16, c_44),
                    _mm_mullo_epi16(v_i16, c_91),
                )),
            );
            // B = Y + (U * 227) >> 7
            let b = _mm_add_epi16(y_i16, _mm_srai_epi16::<7>(_mm_mullo_epi16(u_i16, c_227)));

            // Saturating pack i16 → u8
            let r_u8 = _mm_packus_epi16(r, zero); // low 8 bytes valid
            let g_u8 = _mm_packus_epi16(g, zero);
            let b_u8 = _mm_packus_epi16(b, zero);

            // Interleave and store RGB (no vst3 on SSE, do it manually)
            let mut rgb_buf = [0u8; 24];
            let mut r_arr = [0u8; 8];
            let mut g_arr = [0u8; 8];
            let mut b_arr = [0u8; 8];
            _mm_storel_epi64(r_arr.as_mut_ptr() as *mut __m128i, r_u8);
            _mm_storel_epi64(g_arr.as_mut_ptr() as *mut __m128i, g_u8);
            _mm_storel_epi64(b_arr.as_mut_ptr() as *mut __m128i, b_u8);
            for i in 0..8 {
                rgb_buf[i * 3] = r_arr[i];
                rgb_buf[i * 3 + 1] = g_arr[i];
                rgb_buf[i * 3 + 2] = b_arr[i];
            }
            std::ptr::copy_nonoverlapping(rgb_buf.as_ptr(), rgb_row_ptr.add(col * 3), 24);

            col += 8;
        }

        // Scalar tail
        while col < width {
            let y_val = *y_row_ptr.add(col) as i16;
            let u_val = *u_row_ptr.add(col / 2) as i16 - 128;
            let v_val = *v_row_ptr.add(col / 2) as i16 - 128;

            let r = y_val + ((v_val * 179) >> 7);
            let g = y_val - (((u_val * 44) + (v_val * 91)) >> 7);
            let b = y_val + ((u_val * 227) >> 7);

            let idx = col * 3;
            *rgb_row_ptr.add(idx) = r.clamp(0, 255) as u8;
            *rgb_row_ptr.add(idx + 1) = g.clamp(0, 255) as u8;
            *rgb_row_ptr.add(idx + 2) = b.clamp(0, 255) as u8;

            col += 1;
        }
    }
}
