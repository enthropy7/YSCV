use rayon::prelude::*;
use yscv_tensor::Tensor;

use super::super::ImgProcError;
use super::super::shape::hwc_shape;

/// Resizes HWC tensor with nearest-neighbor sampling.
///
/// Uses precomputed x-index map, row duplication for repeated source rows,
/// and rayon parallelism for large images.
pub fn resize_nearest(input: &Tensor, out_h: usize, out_w: usize) -> Result<Tensor, ImgProcError> {
    let (in_h, in_w, channels) = hwc_shape(input)?;
    if out_h == 0 || out_w == 0 {
        return Err(ImgProcError::InvalidSize {
            height: out_h,
            width: out_w,
        });
    }

    let data = input.data();

    // Precompute source x-index for each output x (avoids repeated integer division)
    let x_map: Vec<usize> = (0..out_w).map(|x| x * in_w / out_w).collect();

    // Precompute source y-index for each output y
    let y_map: Vec<usize> = (0..out_h).map(|y| y * in_h / out_h).collect();

    let total = out_h * out_w * channels;
    // SAFETY: every element is written by compute_row + row duplication below.
    let mut out = Vec::with_capacity(total);
    #[allow(unsafe_code, clippy::uninit_vec)]
    unsafe {
        out.set_len(total);
    }
    let row_len = out_w * channels;

    // Precompute flat source offsets for each output column (avoids multiply in hot loop)
    let x_src_offsets: Vec<usize> = x_map.iter().map(|&sx| sx * channels).collect();

    // Build a single remapped row for a given source row
    let compute_row = |src_y: usize, row: &mut [f32]| {
        let src_row_base = src_y * in_w * channels;
        if channels == 1 {
            for (x, &src_x) in x_map.iter().enumerate() {
                row[x] = data[src_row_base + src_x];
            }
        } else if channels == 3 {
            // Optimized path for RGB: copy 3 floats per pixel without copy_from_slice overhead
            for (x, &src_off) in x_src_offsets.iter().enumerate() {
                let s = src_row_base + src_off;
                let d = x * 3;
                // Direct element copy is faster than copy_from_slice for small fixed sizes
                // (avoids bounds check + memcpy call overhead)
                unsafe {
                    let sp = data.as_ptr().add(s);
                    let dp = row.as_mut_ptr().add(d);
                    *dp = *sp;
                    *dp.add(1) = *sp.add(1);
                    *dp.add(2) = *sp.add(2);
                }
            }
        } else {
            for (x, &src_off) in x_src_offsets.iter().enumerate() {
                let src_base = src_row_base + src_off;
                let dst_base = x * channels;
                row[dst_base..(dst_base + channels)]
                    .copy_from_slice(&data[src_base..(src_base + channels)]);
            }
        }
    };

    // Build groups of consecutive output rows that map to the same source row.
    // Within each group, only the first row is computed; the rest are memcpy'd.
    // This is critical for upscaling where many output rows share a source row.
    let mut groups: Vec<(usize, usize)> = Vec::new(); // (start, end) of each group
    {
        let mut y = 0;
        while y < out_h {
            let src_y = y_map[y];
            let start = y;
            y += 1;
            while y < out_h && y_map[y] == src_y {
                y += 1;
            }
            groups.push((start, y));
        }
    }

    if out_h * out_w <= 4096 || groups.len() < 4 {
        // Sequential path with row duplication
        for &(start, end) in &groups {
            compute_row(
                y_map[start],
                &mut out[start * row_len..(start + 1) * row_len],
            );
            let first_row_start = start * row_len;
            for y in (start + 1)..end {
                let (head, tail) = out.split_at_mut(y * row_len);
                tail[..row_len].copy_from_slice(&head[first_row_start..first_row_start + row_len]);
            }
        }
    } else {
        // Parallel path with row duplication preserved.
        // Each group is processed independently: compute first row, memcpy duplicates.
        // This keeps row deduplication while parallelizing across groups.
        let out_addr = super::SendPtr(out.as_mut_ptr());
        let row_len_c = row_len;

        #[cfg(target_os = "macos")]
        {
            use super::u8ops::gcd;
            let groups_ref = &groups;
            gcd::parallel_for(groups.len(), |gi| {
                let (start, end) = groups_ref[gi];
                let group_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        out_addr.ptr().add(start * row_len_c),
                        (end - start) * row_len_c,
                    )
                };
                compute_row(y_map[start], &mut group_slice[..row_len_c]);
                for dup in 1..(end - start) {
                    let (head, tail) = group_slice.split_at_mut(dup * row_len_c);
                    tail[..row_len_c].copy_from_slice(&head[..row_len_c]);
                }
            });
        }

        #[cfg(not(target_os = "macos"))]
        {
            groups.par_iter().for_each(|&(start, end)| {
                // SAFETY: groups are non-overlapping ranges, each writes to [start*row_len..(end)*row_len)
                let group_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        out_addr.ptr().add(start * row_len_c),
                        (end - start) * row_len_c,
                    )
                };
                compute_row(y_map[start], &mut group_slice[..row_len_c]);
                for dup in 1..(end - start) {
                    let (head, tail) = group_slice.split_at_mut(dup * row_len_c);
                    tail[..row_len_c].copy_from_slice(&head[..row_len_c]);
                }
            });
        }
    }

    Tensor::from_vec(vec![out_h, out_w, channels], out).map_err(Into::into)
}

/// Resizes HWC tensor with bilinear interpolation.
pub fn resize_bilinear(input: &Tensor, out_h: usize, out_w: usize) -> Result<Tensor, ImgProcError> {
    let (in_h, in_w, channels) = hwc_shape(input)?;
    if out_h == 0 || out_w == 0 {
        return Err(ImgProcError::InvalidSize {
            height: out_h,
            width: out_w,
        });
    }

    let mut out = vec![0.0f32; out_h * out_w * channels];
    let scale_y = if out_h > 1 {
        (in_h as f32 - 1.0) / (out_h as f32 - 1.0)
    } else {
        0.0
    };
    let scale_x = if out_w > 1 {
        (in_w as f32 - 1.0) / (out_w as f32 - 1.0)
    } else {
        0.0
    };

    let data = input.data();
    let row_len = out_w * channels;

    // Precompute x-coordinate mapping: (x0, x1, fx) for each output column.
    // Avoids repeated float multiply + floor per row.
    let x_lut: Vec<(usize, usize, f32)> = (0..out_w)
        .map(|x| {
            let src_x = x as f32 * scale_x;
            let x0 = src_x.floor() as usize;
            let x1 = (x0 + 1).min(in_w - 1);
            let fx = src_x - x0 as f32;
            (x0, x1, fx)
        })
        .collect();

    let compute_row = |y: usize, row: &mut [f32]| {
        let src_y = y as f32 * scale_y;
        let y0 = src_y.floor() as usize;
        let y1 = (y0 + 1).min(in_h - 1);
        let fy = src_y - y0 as f32;
        let fy_inv = 1.0 - fy;
        let row0_off = y0 * in_w;
        let row1_off = y1 * in_w;

        for (x, &(x0, x1, fx)) in x_lut.iter().enumerate() {
            let fx_inv = 1.0 - fx;
            for c in 0..channels {
                let v00 = data[(row0_off + x0) * channels + c];
                let v01 = data[(row0_off + x1) * channels + c];
                let v10 = data[(row1_off + x0) * channels + c];
                let v11 = data[(row1_off + x1) * channels + c];

                let top = v00 * fx_inv + v01 * fx;
                let bot = v10 * fx_inv + v11 * fx;
                row[x * channels + c] = top * fy_inv + bot * fy;
            }
        }
    };

    if out_h * out_w > 4096 {
        out.par_chunks_mut(row_len)
            .enumerate()
            .for_each(|(y, row)| compute_row(y, row));
    } else {
        out.chunks_mut(row_len)
            .enumerate()
            .for_each(|(y, row)| compute_row(y, row));
    }

    Tensor::from_vec(vec![out_h, out_w, channels], out).map_err(Into::into)
}
