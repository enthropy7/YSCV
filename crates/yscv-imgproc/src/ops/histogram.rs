use yscv_tensor::Tensor;

use super::super::ImgProcError;
use super::super::shape::hwc_shape;

/// Computes a 256-bin histogram for a single-channel `[H, W, 1]` image.
/// Assumes values in `[0, 1]` range.
pub fn histogram_256(input: &Tensor) -> Result<[u32; 256], ImgProcError> {
    let (_h, _w, c) = hwc_shape(input)?;
    if c != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: c,
        });
    }

    let data = input.data();
    let len = data.len();

    // Pre-convert to u8 for branch-free binning
    let mut u8_buf = vec![0u8; len];
    for (dst, &v) in u8_buf.iter_mut().zip(data.iter()) {
        *dst = (v.clamp(0.0, 1.0) * 255.0) as u8;
    }

    // Parallel binning with thread-local histograms for large images
    if len >= 65536 {
        let num_chunks = rayon::current_num_threads().clamp(1, 8);
        let chunk_size = len.div_ceil(num_chunks);
        let mut local_hists = vec![0u32; num_chunks * 256];

        rayon::scope(|s| {
            for (chunk_idx, chunk) in u8_buf.chunks(chunk_size).enumerate() {
                let hist_slice = &mut local_hists[chunk_idx * 256..(chunk_idx + 1) * 256];
                // SAFETY: each thread writes to its own disjoint slice
                let hist_ptr = super::SendPtr(hist_slice.as_mut_ptr());
                s.spawn(move |_| {
                    let hist = unsafe { std::slice::from_raw_parts_mut(hist_ptr.ptr(), 256) };
                    histogram_bin_chunk(chunk, hist);
                });
            }
        });

        // Merge
        let mut hist = [0u32; 256];
        for chunk_idx in 0..num_chunks {
            for i in 0..256 {
                hist[i] += local_hists[chunk_idx * 256 + i];
            }
        }
        Ok(hist)
    } else {
        let mut hist = [0u32; 256];
        histogram_bin_chunk(&u8_buf, &mut hist);
        Ok(hist)
    }
}

/// Bins a slice of `u8` pixel values into a 256-bin histogram.
///
/// On aarch64 with NEON, loads 16 bytes per iteration with `vld1q_u8` and
/// extracts individual lanes for scatter-add, reducing load overhead.
#[allow(unsafe_code)]
#[inline]
fn histogram_bin_chunk(data: &[u8], hist: &mut [u32]) {
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    if !cfg!(miri) {
        unsafe {
            use std::arch::aarch64::*;
            let ptr = data.as_ptr();
            let len = data.len();
            // Use two sub-histograms to reduce store-to-load forwarding stalls
            // when consecutive pixels map to the same bin.
            let mut hist2 = [0u32; 256];
            while i + 16 <= len {
                let v = vld1q_u8(ptr.add(i));
                // Extract each byte and increment the corresponding bin.
                // Alternate between hist and hist2 to reduce dependency stalls.
                hist[vgetq_lane_u8::<0>(v) as usize] += 1;
                hist2[vgetq_lane_u8::<1>(v) as usize] += 1;
                hist[vgetq_lane_u8::<2>(v) as usize] += 1;
                hist2[vgetq_lane_u8::<3>(v) as usize] += 1;
                hist[vgetq_lane_u8::<4>(v) as usize] += 1;
                hist2[vgetq_lane_u8::<5>(v) as usize] += 1;
                hist[vgetq_lane_u8::<6>(v) as usize] += 1;
                hist2[vgetq_lane_u8::<7>(v) as usize] += 1;
                hist[vgetq_lane_u8::<8>(v) as usize] += 1;
                hist2[vgetq_lane_u8::<9>(v) as usize] += 1;
                hist[vgetq_lane_u8::<10>(v) as usize] += 1;
                hist2[vgetq_lane_u8::<11>(v) as usize] += 1;
                hist[vgetq_lane_u8::<12>(v) as usize] += 1;
                hist2[vgetq_lane_u8::<13>(v) as usize] += 1;
                hist[vgetq_lane_u8::<14>(v) as usize] += 1;
                hist2[vgetq_lane_u8::<15>(v) as usize] += 1;
                i += 16;
            }
            // Merge sub-histogram
            for j in 0..256 {
                hist[j] += hist2[j];
            }
        }
    }

    // Scalar tail (also used as fallback on non-aarch64 / Miri)
    while i < data.len() {
        hist[data[i] as usize] += 1;
        i += 1;
    }
}

/// Histogram equalization for single-channel `[H, W, 1]` images with values in `[0, 1]`.
pub fn histogram_equalize(input: &Tensor) -> Result<Tensor, ImgProcError> {
    let (h, w, c) = hwc_shape(input)?;
    if c != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: c,
        });
    }

    let hist = histogram_256(input)?;
    let total = (h * w) as f32;

    // Build CDF
    let mut cdf = [0.0f32; 256];
    let mut running = 0u32;
    for (i, &count) in hist.iter().enumerate() {
        running += count;
        cdf[i] = running as f32 / total;
    }

    let out: Vec<f32> = input
        .data()
        .iter()
        .map(|&v| {
            let bin = (v.clamp(0.0, 1.0) * 255.0) as usize;
            cdf[bin.min(255)]
        })
        .collect();

    Tensor::from_vec(vec![h, w, 1], out).map_err(Into::into)
}

/// Computes the integral (summed-area table) image from a `[H, W, 1]` tensor.
///
/// Uses a two-pass approach:
/// 1. Horizontal prefix sums per row (parallelized for large images)
/// 2. Vertical accumulation using SIMD
#[allow(unsafe_code)]
pub fn integral_image(input: &Tensor) -> Result<Tensor, ImgProcError> {
    let (h, w, c) = hwc_shape(input)?;
    if c != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: c,
        });
    }
    let data = input.data();
    let mut sat = vec![0.0f32; h * w];

    // Pass 1: horizontal prefix sums per row
    if h > 1 && w * h >= 4096 {
        // Parallel per-row prefix sums for large images
        use rayon::prelude::*;
        sat.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            let src_off = y * w;
            if w > 0 {
                row[0] = data[src_off];
                for x in 1..w {
                    row[x] = row[x - 1] + data[src_off + x];
                }
            }
        });
    } else {
        for y in 0..h {
            let off = y * w;
            if w > 0 {
                sat[off] = data[off];
                for x in 1..w {
                    sat[off + x] = sat[off + x - 1] + data[off + x];
                }
            }
        }
    }

    // Pass 2: vertical accumulation (serial across rows, SIMD within row)
    for y in 1..h {
        let prev = (y - 1) * w;
        let cur = y * w;
        integral_add_row(&mut sat, prev, cur, w);
    }

    Tensor::from_vec(vec![h, w, 1], sat).map_err(Into::into)
}

/// Adds `sat[prev_off..prev_off+w]` into `sat[cur_off..cur_off+w]` using SIMD.
#[allow(unsafe_code)]
#[inline]
fn integral_add_row(sat: &mut [f32], prev_off: usize, cur_off: usize, w: usize) {
    let mut x = 0;

    #[cfg(target_arch = "aarch64")]
    if !cfg!(miri) {
        unsafe {
            use std::arch::aarch64::*;
            let p = sat.as_mut_ptr();
            while x + 4 <= w {
                let a = vld1q_f32(p.add(cur_off + x));
                let b = vld1q_f32(p.add(prev_off + x));
                vst1q_f32(p.add(cur_off + x), vaddq_f32(a, b));
                x += 4;
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if !cfg!(miri) {
        unsafe {
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            if std::is_x86_feature_detected!("sse") {
                let p = sat.as_mut_ptr();
                while x + 4 <= w {
                    let a = _mm_loadu_ps(p.add(cur_off + x));
                    let b = _mm_loadu_ps(p.add(prev_off + x));
                    _mm_storeu_ps(p.add(cur_off + x), _mm_add_ps(a, b));
                    x += 4;
                }
            }
        }
    }

    // Scalar tail
    while x < w {
        sat[cur_off + x] += sat[prev_off + x];
        x += 1;
    }
}

/// Applies CLAHE to a grayscale `[H, W, 1]` image.
///
/// `tile_h` and `tile_w` define the grid size. `clip_limit` caps bin counts
/// before redistribution (typical: 2.0-4.0).
pub fn clahe(
    input: &Tensor,
    tile_h: usize,
    tile_w: usize,
    clip_limit: f32,
) -> Result<Tensor, ImgProcError> {
    let (h, w, c) = hwc_shape(input)?;
    if c != 1 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 1,
            got: c,
        });
    }
    if tile_h == 0 || tile_w == 0 {
        return Err(ImgProcError::InvalidBlockSize { block_size: 0 });
    }

    let src = input.data();
    let mut out = vec![0.0f32; h * w];

    let grid_rows = tile_h.min(h);
    let grid_cols = tile_w.min(w);
    let cell_h = h / grid_rows;
    let cell_w = w / grid_cols;
    let n_tiles = grid_rows * grid_cols;

    // Pre-convert entire image to u8 once (avoid per-pixel float→u8 in hot loops)
    let mut src_u8 = vec![0u8; h * w];
    for (dst, &v) in src_u8.iter_mut().zip(src.iter()) {
        *dst = (v.clamp(0.0, 1.0) * 255.0) as u8;
    }

    // Flat map storage: maps[tile_idx * 256 + val]
    let mut maps = vec![0u8; n_tiles * 256];

    // Compute tile maps — each tile is independent, parallelize across tiles.
    {
        use super::u8ops::gcd;
        let maps_ptr = super::SendPtr(maps.as_mut_ptr());
        let src_u8_ptr = super::SendConstPtr(src_u8.as_ptr());
        gcd::parallel_for(n_tiles, |tile_idx| {
            let gr = tile_idx / grid_cols;
            let gc = tile_idx % grid_cols;
            let y0 = gr * cell_h;
            let x0 = gc * cell_w;
            let y1 = if gr == grid_rows - 1 { h } else { y0 + cell_h };
            let x1 = if gc == grid_cols - 1 { w } else { x0 + cell_w };
            let n_pixels = (y1 - y0) * (x1 - x0);

            // SAFETY: each tile writes to its own non-overlapping 256-byte slice.
            let map = unsafe {
                std::slice::from_raw_parts_mut(maps_ptr.ptr().add(tile_idx * 256), 256)
            };
            let src_u8 = unsafe { std::slice::from_raw_parts(src_u8_ptr.ptr(), h * w) };

            let mut hist = [0u32; 256];
            for y in y0..y1 {
                for x in x0..x1 {
                    hist[src_u8[y * w + x] as usize] += 1;
                }
            }

            let clip = (clip_limit * n_pixels as f32 / 256.0).max(1.0) as u32;
            let mut excess = 0u32;
            for h_bin in hist.iter_mut() {
                if *h_bin > clip {
                    excess += *h_bin - clip;
                    *h_bin = clip;
                }
            }
            let bonus = excess / 256;
            let leftover = (excess % 256) as usize;
            for h_bin in hist.iter_mut() {
                *h_bin += bonus;
            }
            for i in 0..leftover {
                hist[i] += 1;
            }

            let scale = 255.0 / n_pixels.max(1) as f32;
            let mut csum = 0u32;
            for (i, &count) in hist.iter().enumerate() {
                csum += count;
                map[i] = (csum as f32 * scale).round().min(255.0) as u8;
            }
        });
    }

    // Pre-compute reciprocals for interpolation
    let inv_cell_h = 1.0 / cell_h as f32;
    let inv_cell_w = 1.0 / cell_w as f32;
    let gr_max = grid_rows.saturating_sub(2);
    let gc_max = grid_cols.saturating_sub(2);

    // Interpolation pass — rows are independent, parallelize across rows.
    {
        use super::u8ops::gcd;
        let out_ptr = super::SendPtr(out.as_mut_ptr());
        let maps_ptr = super::SendConstPtr(maps.as_ptr());
        let src_u8_ptr = super::SendConstPtr(src_u8.as_ptr());
        gcd::parallel_for(h, |y| {
            // SAFETY: each row writes to non-overlapping out[y*w..(y+1)*w].
            let out_row =
                unsafe { std::slice::from_raw_parts_mut(out_ptr.ptr().add(y * w), w) };
            let maps = unsafe { std::slice::from_raw_parts(maps_ptr.ptr(), n_tiles * 256) };
            let src_u8 = unsafe { std::slice::from_raw_parts(src_u8_ptr.ptr(), h * w) };

            let gy = (y as f32 * inv_cell_h - 0.5).max(0.0);
            let gr0 = (gy as usize).min(gr_max);
            let gr1 = (gr0 + 1).min(grid_rows - 1);
            let fy = gy - gr0 as f32;
            let m00_base = gr0 * grid_cols;
            let m10_base = gr1 * grid_cols;

            for x in 0..w {
                let val = src_u8[y * w + x] as usize;
                let gx = (x as f32 * inv_cell_w - 0.5).max(0.0);
                let gc0 = (gx as usize).min(gc_max);
                let gc1 = (gc0 + 1).min(grid_cols - 1);
                let fx = gx - gc0 as f32;

                let v00 = maps[(m00_base + gc0) * 256 + val] as f32;
                let v01 = maps[(m00_base + gc1) * 256 + val] as f32;
                let v10 = maps[(m10_base + gc0) * 256 + val] as f32;
                let v11 = maps[(m10_base + gc1) * 256 + val] as f32;

                let top = v00 + fx * (v01 - v00);
                let bot = v10 + fx * (v11 - v10);
                out_row[x] = (top + fy * (bot - top)) * (1.0 / 255.0);
            }
        });
    }

    Tensor::from_vec(vec![h, w, 1], out).map_err(Into::into)
}
