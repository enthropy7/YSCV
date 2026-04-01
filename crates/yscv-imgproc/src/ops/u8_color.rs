//! u8 color operations: RGB to HSV, histogram, CLAHE.
#![allow(unsafe_code)]

use super::u8ops::{RAYON_THRESHOLD, gcd};

// ============================================================================
// RGB to HSV (u8-native, integer-only hot loop)
// ============================================================================

/// Converts RGB u8 image to HSV u8 image.
///
/// Output convention matches OpenCV: H in \[0,180), S in \[0,255], V in \[0,255].
/// Uses integer division via precomputed lookup table — no float in the hot loop.
pub fn rgb_to_hsv_u8(src: &[u8], width: usize, height: usize) -> Vec<u8> {
    let npixels = width * height;
    assert!(src.len() >= npixels * 3, "rgb_to_hsv_u8: src too short");

    // Static LUTs — computed once, reused across calls
    use std::sync::OnceLock;
    static SDIV: OnceLock<[u16; 256]> = OnceLock::new();
    static HDIV: OnceLock<[u16; 256]> = OnceLock::new();
    let sdiv = SDIV.get_or_init(|| {
        let mut t = [0u16; 256];
        for v in 1..256 {
            t[v] = ((255 * 256 + v / 2) / v) as u16;
        }
        t
    });
    let hdiv = HDIV.get_or_init(|| {
        let mut t = [0u16; 256];
        for d in 1..256 {
            t[d] = ((30 * 256 + d / 2) / d) as u16;
        }
        t
    });

    let mut out = vec![0u8; npixels * 3];

    if npixels >= RAYON_THRESHOLD {
        let out_base = out.as_mut_ptr() as usize;
        gcd::parallel_for(height, |y| {
            let row_src = &src[y * width * 3..(y + 1) * width * 3];
            let row_dst_start = y * width * 3;
            // SAFETY: each row writes to non-overlapping region; pointer derived from as_mut_ptr
            let out_ptr = unsafe {
                std::slice::from_raw_parts_mut((out_base as *mut u8).add(row_dst_start), width * 3)
            };
            rgb_to_hsv_u8_row(row_src, out_ptr, width, sdiv, hdiv);
        });
    } else {
        rgb_to_hsv_u8_row(src, &mut out, npixels, sdiv, hdiv);
    }

    out
}

#[inline(always)]
fn rgb_to_hsv_u8_pixel_write(
    src: &[u8],
    dst: &mut [u8],
    idx: usize,
    sdiv: &[u16; 256],
    hdiv: &[u16; 256],
) {
    let base = idx * 3;
    let r = unsafe { *src.get_unchecked(base) } as i32;
    let g = unsafe { *src.get_unchecked(base + 1) } as i32;
    let b = unsafe { *src.get_unchecked(base + 2) } as i32;

    // Branchless max/min
    let v = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = v - min;

    unsafe {
        *dst.get_unchecked_mut(base + 2) = v as u8;
    }

    if delta == 0 {
        unsafe {
            *dst.get_unchecked_mut(base) = 0;
            *dst.get_unchecked_mut(base + 1) = 0;
        }
        return;
    }

    // S via LUT
    let s = ((delta as u32 * unsafe { *sdiv.get_unchecked(v as usize) } as u32) >> 8) as u8;
    unsafe {
        *dst.get_unchecked_mut(base + 1) = s;
    }

    // H via LUT — branchless sector selection using masks
    let hd = unsafe { *hdiv.get_unchecked(delta as usize) } as i32;
    // Use OpenCV's approach: compute all three candidate diffs, select via mask
    let vr = (g - b) * hd; // when r is max
    let vg = (b - r) * hd + (60 << 8); // when g is max
    let vb = (r - g) * hd + (120 << 8); // when b is max

    // Select: if r==v use vr, elif g==v use vg, else vb
    // This compiles to conditional moves (no branches)
    let h_shifted = if r == v {
        vr
    } else if g == v {
        vg
    } else {
        vb
    };
    let h_raw = h_shifted >> 8;

    // Normalize to [0,180)
    let h = h_raw + ((h_raw >> 31) & 180);
    let h = h - (if h >= 180 { 180 } else { 0 });
    unsafe {
        *dst.get_unchecked_mut(base) = h as u8;
    }
}

#[inline]
fn rgb_to_hsv_u8_row(
    src: &[u8],
    dst: &mut [u8],
    count: usize,
    sdiv: &[u16; 256],
    hdiv: &[u16; 256],
) {
    let mut start = 0usize;

    // NEON path: deinterleave RGB, vectorized max/min/delta/V/S, scalar H
    #[cfg(target_arch = "aarch64")]
    if !cfg!(miri) && std::arch::is_aarch64_feature_detected!("neon") {
        start = unsafe { rgb_to_hsv_u8_neon_row(src, dst, count, sdiv, hdiv) };
    }

    #[cfg(target_arch = "x86_64")]
    if !cfg!(miri) && std::is_x86_feature_detected!("sse4.1") {
        start = unsafe { rgb_to_hsv_u8_sse_row(src, dst, count, sdiv, hdiv) };
    }

    for i in start..count {
        rgb_to_hsv_u8_pixel_write(src, dst, i, sdiv, hdiv);
    }
}

/// NEON HSV path: uses vld3q_u8 to deinterleave 16 RGB pixels,
/// vmaxq_u8/vminq_u8 for vectorized max/min (V, delta),
/// then scalar H/S via LUT.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn rgb_to_hsv_u8_neon_row(
    src: &[u8],
    dst: &mut [u8],
    count: usize,
    _sdiv: &[u16; 256],
    _hdiv: &[u16; 256],
) -> usize {
    use std::arch::aarch64::*;

    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();
    let mut i = 0usize;

    // Fully vectorized: process 8 pixels using u16 arithmetic, zero scalar
    while i + 8 <= count {
        let base = i * 3;
        let rgb = vld3_u8(sp.add(base)); // 8 pixels
        let rv = rgb.0;
        let gv = rgb.1;
        let bv = rgb.2;

        let v_val = vmax_u8(vmax_u8(rv, gv), bv);
        let delta = vsub_u8(v_val, vmin_u8(vmin_u8(rv, gv), bv));

        // S = delta * 255 / v — use f32 NEON division (4 at a time)
        let d_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(delta))));
        let d_hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8(delta))));
        let v_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(v_val))));
        let v_hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8(v_val))));
        let c255 = vdupq_n_f32(255.0);
        let one = vdupq_n_f32(1.0);
        let s_lo = vcvtq_u32_f32(vdivq_f32(vmulq_f32(d_lo, c255), vmaxq_f32(v_lo, one)));
        let s_hi = vcvtq_u32_f32(vdivq_f32(vmulq_f32(d_hi, c255), vmaxq_f32(v_hi, one)));
        let s_val = vmovn_u16(vcombine_u16(vmovn_u32(s_lo), vmovn_u32(s_hi)));

        // H: use f32 NEON — compute all 3 candidates, select via mask
        // diff_rg = (g-b)/delta, diff_gb = (b-r)/delta, diff_br = (r-g)/delta
        let r16 = vreinterpretq_s16_u16(vmovl_u8(rv));
        let g16 = vreinterpretq_s16_u16(vmovl_u8(gv));
        let b16 = vreinterpretq_s16_u16(vmovl_u8(bv));
        let d16 = vmaxq_s16(vreinterpretq_s16_u16(vmovl_u8(delta)), vdupq_n_s16(1));

        // (g-b), (b-r), (r-g) as f32
        let gb_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vsubq_s16(g16, b16))));
        let gb_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vsubq_s16(g16, b16))));
        let br_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vsubq_s16(b16, r16))));
        let br_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vsubq_s16(b16, r16))));
        let rg_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vsubq_s16(r16, g16))));
        let rg_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vsubq_s16(r16, g16))));
        let df_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(d16)));
        let df_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(d16)));

        // h_r = 30*(g-b)/delta, h_g = 30*(b-r)/delta + 60, h_b = 30*(r-g)/delta + 120
        let c30 = vdupq_n_f32(30.0);
        let c60 = vdupq_n_f32(60.0);
        let c120 = vdupq_n_f32(120.0);
        let c180 = vdupq_n_f32(180.0);
        let zero_f = vdupq_n_f32(0.0);

        let hr_lo = vdivq_f32(vmulq_f32(gb_lo, c30), df_lo);
        let hr_hi = vdivq_f32(vmulq_f32(gb_hi, c30), df_hi);
        let hg_lo = vaddq_f32(vdivq_f32(vmulq_f32(br_lo, c30), df_lo), c60);
        let hg_hi = vaddq_f32(vdivq_f32(vmulq_f32(br_hi, c30), df_hi), c60);
        let hb_lo = vaddq_f32(vdivq_f32(vmulq_f32(rg_lo, c30), df_lo), c120);
        let hb_hi = vaddq_f32(vdivq_f32(vmulq_f32(rg_hi, c30), df_hi), c120);

        // Masks: r_is_max, g_is_max (8-pixel uint8x8_t)
        let r_mask = vceq_u8(rv, v_val);
        let g_mask = vand_u8(vceq_u8(gv, v_val), vmvn_u8(r_mask));

        // Widen u8 mask → u16 → u32 for f32 vbslq select
        let r_m16 = vmovl_u8(r_mask); // uint16x8
        let r_lo_m = vmovl_u16(vget_low_u16(r_m16));
        let r_hi_m = vmovl_u16(vget_high_u16(r_m16));
        let g_m16 = vmovl_u8(g_mask);
        let g_lo_m = vmovl_u16(vget_low_u16(g_m16));
        let g_hi_m = vmovl_u16(vget_high_u16(g_m16));

        // Select: h = r_mask ? hr : (g_mask ? hg : hb)
        let h_lo = vbslq_f32(r_lo_m, hr_lo, vbslq_f32(g_lo_m, hg_lo, hb_lo));
        let h_hi = vbslq_f32(r_hi_m, hr_hi, vbslq_f32(g_hi_m, hg_hi, hb_hi));

        // Normalize: if h < 0 { h += 180 }
        let neg_lo = vcltq_f32(h_lo, zero_f);
        let neg_hi = vcltq_f32(h_hi, zero_f);
        let h_lo = vbslq_f32(neg_lo, vaddq_f32(h_lo, c180), h_lo);
        let h_hi = vbslq_f32(neg_hi, vaddq_f32(h_hi, c180), h_hi);

        // Zero H where delta==0
        let dz = vceq_u8(delta, vdup_n_u8(0));
        let dz_m16 = vmovl_u8(dz);
        let dz_lo_m = vmovl_u16(vget_low_u16(dz_m16));
        let dz_hi_m = vmovl_u16(vget_high_u16(dz_m16));
        let h_lo = vbslq_f32(dz_lo_m, zero_f, h_lo);
        let h_hi = vbslq_f32(dz_hi_m, zero_f, h_hi);

        // Convert back to u8
        let h_u32_lo = vcvtq_u32_f32(h_lo);
        let h_u32_hi = vcvtq_u32_f32(h_hi);
        let h_u8 = vmovn_u16(vcombine_u16(vmovn_u32(h_u32_lo), vmovn_u32(h_u32_hi)));

        // Zero S where delta==0
        let s_val = vbic_u8(s_val, dz);

        vst3_u8(dp.add(base), uint8x8x3_t(h_u8, s_val, v_val));
        i += 8;
    }
    i
}

/// SSE HSV path: process 4 pixels at a time using f32 SSE division.
/// Mirrors the NEON approach: deinterleave RGB, compute V=max, delta=max-min,
/// S via f32 div, H via f32 div + masks, pack back to u8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn rgb_to_hsv_u8_sse_row(
    src: &[u8],
    dst: &mut [u8],
    count: usize,
    _sdiv: &[u16; 256],
    _hdiv: &[u16; 256],
) -> usize {
    use std::arch::x86_64::*;

    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();
    let mut i = 0usize;

    // Process 4 pixels at a time using f32x4 SSE for divisions
    while i + 4 <= count {
        let base = i * 3;

        // Gather 4 RGB pixels into separate r/g/b arrays
        let mut rv = [0u8; 4];
        let mut gv = [0u8; 4];
        let mut bv = [0u8; 4];
        for j in 0..4 {
            rv[j] = *sp.add(base + j * 3);
            gv[j] = *sp.add(base + j * 3 + 1);
            bv[j] = *sp.add(base + j * 3 + 2);
        }

        // Convert to f32 vectors
        let rf = _mm_set_ps(rv[3] as f32, rv[2] as f32, rv[1] as f32, rv[0] as f32);
        let gf = _mm_set_ps(gv[3] as f32, gv[2] as f32, gv[1] as f32, gv[0] as f32);
        let bf = _mm_set_ps(bv[3] as f32, bv[2] as f32, bv[1] as f32, bv[0] as f32);

        // V = max(r, g, b)
        let v_f = _mm_max_ps(_mm_max_ps(rf, gf), bf);
        // min = min(r, g, b)
        let min_f = _mm_min_ps(_mm_min_ps(rf, gf), bf);
        // delta = V - min
        let delta_f = _mm_sub_ps(v_f, min_f);

        let zero_f = _mm_setzero_ps();
        let one_f = _mm_set1_ps(1.0);
        let c255 = _mm_set1_ps(255.0);
        let c30 = _mm_set1_ps(30.0);
        let c60 = _mm_set1_ps(60.0);
        let c120 = _mm_set1_ps(120.0);
        let c180 = _mm_set1_ps(180.0);

        // delta_safe = max(delta, 1.0) to avoid div by zero
        let delta_safe = _mm_max_ps(delta_f, one_f);
        // v_safe = max(v, 1.0) to avoid div by zero
        let v_safe = _mm_max_ps(v_f, one_f);

        // S = delta * 255 / v
        let s_f = _mm_div_ps(_mm_mul_ps(delta_f, c255), v_safe);

        // H candidates: hr = 30*(g-b)/delta, hg = 30*(b-r)/delta+60, hb = 30*(r-g)/delta+120
        let hr = _mm_div_ps(_mm_mul_ps(_mm_sub_ps(gf, bf), c30), delta_safe);
        let hg = _mm_add_ps(
            _mm_div_ps(_mm_mul_ps(_mm_sub_ps(bf, rf), c30), delta_safe),
            c60,
        );
        let hb = _mm_add_ps(
            _mm_div_ps(_mm_mul_ps(_mm_sub_ps(rf, gf), c30), delta_safe),
            c120,
        );

        // Masks: r_is_max, g_is_max
        let r_mask = _mm_cmpeq_ps(rf, v_f); // r == max
        let g_mask = _mm_andnot_ps(r_mask, _mm_cmpeq_ps(gf, v_f)); // g == max && !(r == max)

        // Select: h = r_mask ? hr : (g_mask ? hg : hb)
        let h_sel = _mm_blendv_ps(_mm_blendv_ps(hb, hg, g_mask), hr, r_mask);

        // if h < 0 { h += 180 }
        let neg_mask = _mm_cmplt_ps(h_sel, zero_f);
        let h_norm = _mm_blendv_ps(h_sel, _mm_add_ps(h_sel, c180), neg_mask);

        // Zero H and S where delta == 0
        let dz_mask = _mm_cmpeq_ps(delta_f, zero_f);
        let h_final = _mm_andnot_ps(dz_mask, h_norm);
        let s_final = _mm_andnot_ps(dz_mask, s_f);

        // Convert back to u8 and store interleaved HSV
        // Extract individual float values
        let mut h_arr = [0f32; 4];
        let mut s_arr = [0f32; 4];
        let mut v_arr = [0f32; 4];
        _mm_storeu_ps(h_arr.as_mut_ptr(), h_final);
        _mm_storeu_ps(s_arr.as_mut_ptr(), s_final);
        _mm_storeu_ps(v_arr.as_mut_ptr(), v_f);

        for j in 0..4 {
            *dp.add(base + j * 3) = h_arr[j] as u8;
            *dp.add(base + j * 3 + 1) = s_arr[j] as u8;
            *dp.add(base + j * 3 + 2) = v_arr[j] as u8;
        }

        i += 4;
    }
    i
}

// ============================================================================
// Histogram u8 (direct binning, no float)
// ============================================================================

/// Computes a 256-bin histogram from raw u8 data.
///
/// Direct u8 binning with no float conversion. Uses multiple thread-local
/// histograms via GCD parallel_for and merges at end.
pub fn histogram_u8(src: &[u8], len: usize) -> [u32; 256] {
    assert!(src.len() >= len, "histogram_u8: src too short");

    if len < RAYON_THRESHOLD {
        let mut hist = [0u32; 256];
        for i in 0..len {
            hist[src[i] as usize] += 1;
        }
        return hist;
    }

    // Use 4 thread-local histograms to reduce contention
    let n_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
        .min(8);
    let chunk_size = len.div_ceil(n_threads);

    // Allocate thread-local histograms
    let local_hists: Vec<std::sync::Mutex<[u32; 256]>> = (0..n_threads)
        .map(|_| std::sync::Mutex::new([0u32; 256]))
        .collect();

    gcd::parallel_for(n_threads, |t| {
        let start = t * chunk_size;
        let end = (start + chunk_size).min(len);
        if start >= end {
            return;
        }

        let mut local = [0u32; 256];
        // Process 4 at a time to reduce loop overhead
        let chunk = &src[start..end];
        let n = chunk.len();
        let n4 = n & !3;
        for i in (0..n4).step_by(4) {
            local[chunk[i] as usize] += 1;
            local[chunk[i + 1] as usize] += 1;
            local[chunk[i + 2] as usize] += 1;
            local[chunk[i + 3] as usize] += 1;
        }
        for i in n4..n {
            local[chunk[i] as usize] += 1;
        }

        *local_hists[t].lock().unwrap_or_else(|e| e.into_inner()) = local;
    });

    // Merge all thread-local histograms
    let mut hist = [0u32; 256];
    for lh in &local_hists {
        let local = lh.lock().unwrap_or_else(|e| e.into_inner());
        for i in 0..256 {
            hist[i] += local[i];
        }
    }
    hist
}

// ============================================================================
// CLAHE u8 (Contrast Limited Adaptive Histogram Equalization, u8-native)
// ============================================================================

/// CLAHE operating entirely in u8 domain — no float conversion in the hot loop.
///
/// Uses direct u8 histogram per tile, integer fixed-point bilinear interpolation,
/// and GCD parallelism on both tile histogram computation and final interpolation.
///
/// Arguments:
/// - `src`: grayscale u8 pixel data (row-major, single channel)
/// - `width`, `height`: image dimensions
/// - `tile_rows`, `tile_cols`: number of tiles in each dimension
/// - `clip_limit`: histogram clip limit (as float, converted to integer internally)
pub fn clahe_u8(
    src: &[u8],
    width: usize,
    height: usize,
    tile_rows: usize,
    tile_cols: usize,
    clip_limit: f32,
) -> Vec<u8> {
    let npixels = width * height;
    assert!(src.len() >= npixels, "clahe_u8: src too short");
    assert!(tile_rows >= 1 && tile_cols >= 1);

    let tile_h = height / tile_rows;
    let tile_w = width / tile_cols;
    if tile_h == 0 || tile_w == 0 {
        return src[..npixels].to_vec();
    }

    let n_tiles = tile_rows * tile_cols;
    let tile_pixels = tile_h * tile_w;

    // Integer clip limit: scale from float to per-tile count
    let clip = ((clip_limit * tile_pixels as f32) / 256.0).max(1.0) as u32;

    // Compute LUT maps for each tile: maps[tile_idx * 256 + val] = mapped_val
    let mut maps = vec![0u8; n_tiles * 256];
    let maps_base = maps.as_mut_ptr() as usize;

    // Phase 1: compute histogram + clip + CDF for each tile (parallel)
    {
        gcd::parallel_for(n_tiles, |tile_idx| {
            let tr = tile_idx / tile_cols;
            let tc = tile_idx % tile_cols;
            let y0 = tr * tile_h;
            let x0 = tc * tile_w;

            // Build histogram for this tile
            let mut hist = [0u32; 256];
            for dy in 0..tile_h {
                let row_start = (y0 + dy) * width + x0;
                let row = &src[row_start..row_start + tile_w];
                for &val in row {
                    hist[val as usize] += 1;
                }
            }

            // Clip histogram and redistribute excess
            let mut excess = 0u32;
            for h in hist.iter_mut() {
                if *h > clip {
                    excess += *h - clip;
                    *h = clip;
                }
            }
            let avg_inc = excess / 256;
            let remainder = (excess - avg_inc * 256) as usize;
            for h in hist.iter_mut() {
                *h += avg_inc;
            }
            // Distribute remainder evenly
            let step = if remainder > 0 { 256 / remainder } else { 256 };
            let mut idx = 0;
            for _ in 0..remainder {
                hist[idx] += 1;
                idx = (idx + step) % 256;
            }

            // Build CDF and map
            let mut cdf = [0u32; 256];
            cdf[0] = hist[0];
            for i in 1..256 {
                cdf[i] = cdf[i - 1] + hist[i];
            }

            // cdf_min = first non-zero CDF
            let cdf_min = cdf.iter().copied().find(|&c| c > 0).unwrap_or(0);
            let denom = tile_pixels as u32 - cdf_min;

            // SAFETY: each tile writes to non-overlapping region of maps
            let map = unsafe {
                std::slice::from_raw_parts_mut((maps_base as *mut u8).add(tile_idx * 256), 256)
            };

            if denom == 0 {
                for i in 0..256 {
                    map[i] = i as u8;
                }
            } else {
                for i in 0..256 {
                    let val = ((cdf[i].saturating_sub(cdf_min)) as u64 * 255 / denom as u64) as u8;
                    map[i] = val;
                }
            }
        });
    }

    // Phase 2: interpolate between tile maps for each pixel (parallel by row)
    let mut out = vec![0u8; npixels];
    let out_base = out.as_mut_ptr() as usize;

    // Precompute x-dependent tile indices and weights (shared across all rows)
    let half_w = tile_w / 2;
    let half_h = tile_h / 2;
    // Pack tc0, tc1, wx0, wx1 per x into arrays
    let mut x_tc0 = vec![0u16; width];
    let mut x_tc1 = vec![0u16; width];
    let mut x_wx0 = vec![0u16; width];
    let mut x_wx1 = vec![0u16; width];
    for x in 0..width {
        let fx = x.saturating_sub(half_w);
        let tc0 = (fx / tile_w).min(tile_cols - 1);
        let tc1 = (tc0 + 1).min(tile_cols - 1);
        let cx0 = tc0 * tile_w + half_w;
        let wx1 = if tc0 == tc1 {
            0u32
        } else {
            (((x.saturating_sub(cx0)) as u32 * 256) / tile_w as u32).min(256)
        };
        x_tc0[x] = tc0 as u16;
        x_tc1[x] = tc1 as u16;
        x_wx0[x] = (256 - wx1) as u16;
        x_wx1[x] = wx1 as u16;
    }

    {
        let maps_ref = &maps;
        let src_ref = src;
        let xtc0 = &x_tc0;
        let xtc1 = &x_tc1;
        let xwx0 = &x_wx0;
        let xwx1 = &x_wx1;

        gcd::parallel_for(height, |y| {
            let fy = if y < half_h {
                0
            } else if y >= tile_rows * tile_h - half_h {
                (tile_rows - 1) * tile_h
            } else {
                y - half_h
            };

            let tr0 = (fy / tile_h).min(tile_rows - 1);
            let tr1 = (tr0 + 1).min(tile_rows - 1);

            let cy0 = tr0 * tile_h + half_h;
            let wy1 = if tr0 == tr1 {
                0u32
            } else {
                (((y.saturating_sub(cy0)) as u32 * 256) / tile_h as u32).min(256)
            };
            let wy0 = 256 - wy1;

            // Precompute row-level tile map base offsets
            let base00 = tr0 * tile_cols;
            let base10 = tr1 * tile_cols;

            let row_start = y * width;
            // SAFETY: each row writes to non-overlapping region; pointer derived from as_mut_ptr
            let dst_row = unsafe {
                std::slice::from_raw_parts_mut((out_base as *mut u8).add(row_start), width)
            };

            for x in 0..width {
                let tc0 = unsafe { *xtc0.get_unchecked(x) } as usize;
                let tc1 = unsafe { *xtc1.get_unchecked(x) } as usize;
                let wx0 = unsafe { *xwx0.get_unchecked(x) } as u32;
                let wx1 = unsafe { *xwx1.get_unchecked(x) } as u32;

                let val = unsafe { *src_ref.get_unchecked(row_start + x) } as usize;

                let m00 = unsafe { *maps_ref.get_unchecked((base00 + tc0) * 256 + val) } as u32;
                let m01 = unsafe { *maps_ref.get_unchecked((base00 + tc1) * 256 + val) } as u32;
                let m10 = unsafe { *maps_ref.get_unchecked((base10 + tc0) * 256 + val) } as u32;
                let m11 = unsafe { *maps_ref.get_unchecked((base10 + tc1) * 256 + val) } as u32;

                let top = m00 * wx0 + m01 * wx1;
                let bot = m10 * wx0 + m11 * wx1;
                let result = (top * wy0 + bot * wy1 + 32768) >> 16;

                unsafe {
                    *dst_row.get_unchecked_mut(x) = result as u8;
                }
            }
        });
    }

    out
}
