use rayon::prelude::*;
use yscv_tensor::Tensor;

use super::super::ImgProcError;
use super::super::shape::hwc_shape;
use super::filter::border_coords_3x3;

/// Computes 3x3 Sobel gradients (`gx`, `gy`) per channel.
///
/// Border handling uses only in-bounds neighbors.
#[allow(unsafe_code)]
pub fn sobel_3x3_gradients(input: &Tensor) -> Result<(Tensor, Tensor), ImgProcError> {
    let (h, w, channels) = hwc_shape(input)?;
    let len = h * w * channels;
    let mut out_gx = vec![0.0f32; len];
    let mut out_gy = vec![0.0f32; len];

    const SOBEL_X: [[f32; 3]; 3] = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];
    const SOBEL_Y: [[f32; 3]; 3] = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];

    let data = input.data();
    let row_len = w * channels;
    let interior_h = h.saturating_sub(2);

    // Interior pixels (y=1..h-1, x=1..w-1): no bounds checks
    let compute_interior_row = |y: usize, gx_row: &mut [f32], gy_row: &mut [f32]| {
        // For single-channel: use SIMD path
        if channels == 1 && !cfg!(miri) {
            let row0 = &data[(y - 1) * w..y * w];
            let row1 = &data[y * w..(y + 1) * w];
            let row2 = &data[(y + 1) * w..(y + 2) * w];
            let done = sobel_simd_row_c1(row0, row1, row2, gx_row, gy_row, w);
            // Scalar tail
            for x in done..w.saturating_sub(1) {
                if x == 0 {
                    continue;
                }
                let gx =
                    row0[x + 1] - row0[x - 1] + 2.0 * (row1[x + 1] - row1[x - 1]) + row2[x + 1]
                        - row2[x - 1];
                let gy = row2[x - 1] + 2.0 * row2[x] + row2[x + 1]
                    - row0[x - 1]
                    - 2.0 * row0[x]
                    - row0[x + 1];
                gx_row[x] = gx;
                gy_row[x] = gy;
            }
            return;
        }

        for x in 1..w.saturating_sub(1) {
            for c in 0..channels {
                let r0 = ((y - 1) * w + x - 1) * channels + c;
                let r1 = (y * w + x - 1) * channels + c;
                let r2 = ((y + 1) * w + x - 1) * channels + c;
                let mut gx = 0.0f32;
                let mut gy = 0.0f32;
                gx += data[r0] * SOBEL_X[0][0];
                gx += data[r0 + 2 * channels] * SOBEL_X[0][2];
                gx += data[r1] * SOBEL_X[1][0];
                gx += data[r1 + 2 * channels] * SOBEL_X[1][2];
                gx += data[r2] * SOBEL_X[2][0];
                gx += data[r2 + 2 * channels] * SOBEL_X[2][2];

                gy += data[r0] * SOBEL_Y[0][0];
                gy += data[r0 + channels] * SOBEL_Y[0][1];
                gy += data[r0 + 2 * channels] * SOBEL_Y[0][2];
                gy += data[r2] * SOBEL_Y[2][0];
                gy += data[r2 + channels] * SOBEL_Y[2][1];
                gy += data[r2 + 2 * channels] * SOBEL_Y[2][2];

                gx_row[x * channels + c] = gx;
                gy_row[x * channels + c] = gy;
            }
        }
    };

    if interior_h > 0 {
        let pixels = h * w;
        let gx_interior = &mut out_gx[row_len..row_len + interior_h * row_len];
        let gy_interior = &mut out_gy[row_len..row_len + interior_h * row_len];
        #[cfg(target_os = "macos")]
        let use_gcd = pixels > 4096 && !cfg!(miri);
        #[cfg(not(target_os = "macos"))]
        let use_gcd = false;

        if use_gcd {
            #[cfg(target_os = "macos")]
            {
                let gx_ptr = super::SendPtr(gx_interior.as_mut_ptr());
                let gy_ptr = super::SendPtr(gy_interior.as_mut_ptr());
                use super::u8ops::gcd;
                gcd::parallel_for(interior_h, |i| {
                    let y = i + 1;
                    // SAFETY: each row writes to a disjoint slice.
                    let gx_row = unsafe {
                        std::slice::from_raw_parts_mut(
                            gx_ptr.ptr().add(i * row_len),
                            row_len,
                        )
                    };
                    let gy_row = unsafe {
                        std::slice::from_raw_parts_mut(
                            gy_ptr.ptr().add(i * row_len),
                            row_len,
                        )
                    };
                    compute_interior_row(y, gx_row, gy_row);
                });
            }
        } else if pixels > 4096 {
            gx_interior
                .par_chunks_mut(row_len)
                .zip(gy_interior.par_chunks_mut(row_len))
                .enumerate()
                .for_each(|(i, (gx_row, gy_row))| {
                    compute_interior_row(i + 1, gx_row, gy_row);
                });
        } else {
            gx_interior
                .chunks_mut(row_len)
                .zip(gy_interior.chunks_mut(row_len))
                .enumerate()
                .for_each(|(i, (gx_row, gy_row))| {
                    compute_interior_row(i + 1, gx_row, gy_row);
                });
        }
    }

    // Border pixels: bounds-checked path
    let border = border_coords_3x3(h, w);
    for (y, x) in border {
        for c in 0..channels {
            let mut gx = 0.0f32;
            let mut gy = 0.0f32;
            for ky in -1isize..=1 {
                for kx in -1isize..=1 {
                    let sy = y as isize + ky;
                    let sx = x as isize + kx;
                    if sy < 0 || sx < 0 || sy >= h as isize || sx >= w as isize {
                        continue;
                    }
                    let src = ((sy as usize) * w + sx as usize) * channels + c;
                    let kernel_y = (ky + 1) as usize;
                    let kernel_x = (kx + 1) as usize;
                    let value = data[src];
                    gx += value * SOBEL_X[kernel_y][kernel_x];
                    gy += value * SOBEL_Y[kernel_y][kernel_x];
                }
            }
            let dst = (y * w + x) * channels + c;
            out_gx[dst] = gx;
            out_gy[dst] = gy;
        }
    }

    Ok((
        Tensor::from_vec(vec![h, w, channels], out_gx)?,
        Tensor::from_vec(vec![h, w, channels], out_gy)?,
    ))
}

/// SIMD sobel for single-channel row. Returns first x that was NOT processed.
#[allow(unsafe_code)]
fn sobel_simd_row_c1(
    row0: &[f32],
    row1: &[f32],
    row2: &[f32],
    gx_out: &mut [f32],
    gy_out: &mut [f32],
    w: usize,
) -> usize {
    if w < 6 {
        return 1; // need at least x=1..4 with x+1 < w
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { sobel_neon_row_c1(row0, row1, row2, gx_out, gy_out, w) };
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            return unsafe { sobel_avx_row_c1(row0, row1, row2, gx_out, gy_out, w) };
        }
        if std::is_x86_feature_detected!("sse") {
            return unsafe { sobel_sse_row_c1(row0, row1, row2, gx_out, gy_out, w) };
        }
    }

    1
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn sobel_neon_row_c1(
    row0: &[f32],
    row1: &[f32],
    row2: &[f32],
    gx_out: &mut [f32],
    gy_out: &mut [f32],
    w: usize,
) -> usize {
    use std::arch::aarch64::*;
    let two = vdupq_n_f32(2.0);
    let mut x = 1usize;
    // Need x+4 <= w-1, i.e. x+5 <= w
    while x + 5 <= w {
        // Load overlapping windows from each row
        let r0l = vld1q_f32(row0.as_ptr().add(x - 1)); // [x-1,x,x+1,x+2]
        let r0m = vld1q_f32(row0.as_ptr().add(x)); // [x,x+1,x+2,x+3]
        let r0r = vld1q_f32(row0.as_ptr().add(x + 1)); // [x+1,x+2,x+3,x+4]
        let r1l = vld1q_f32(row1.as_ptr().add(x - 1));
        let r1r = vld1q_f32(row1.as_ptr().add(x + 1));
        let r2l = vld1q_f32(row2.as_ptr().add(x - 1));
        let r2m = vld1q_f32(row2.as_ptr().add(x));
        let r2r = vld1q_f32(row2.as_ptr().add(x + 1));

        // gx = (r0r - r0l) + 2*(r1r - r1l) + (r2r - r2l)
        let dx0 = vsubq_f32(r0r, r0l);
        let dx1 = vsubq_f32(r1r, r1l);
        let dx2 = vsubq_f32(r2r, r2l);
        let gx = vaddq_f32(vaddq_f32(dx0, dx2), vmulq_f32(dx1, two));

        // gy = (r2l + 2*r2m + r2r) - (r0l + 2*r0m + r0r)
        let sy0 = vaddq_f32(vaddq_f32(r0l, r0r), vmulq_f32(r0m, two));
        let sy2 = vaddq_f32(vaddq_f32(r2l, r2r), vmulq_f32(r2m, two));
        let gy = vsubq_f32(sy2, sy0);

        vst1q_f32(gx_out.as_mut_ptr().add(x), gx);
        vst1q_f32(gy_out.as_mut_ptr().add(x), gy);
        x += 4;
    }
    x
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn sobel_avx_row_c1(
    row0: &[f32],
    row1: &[f32],
    row2: &[f32],
    gx_out: &mut [f32],
    gy_out: &mut [f32],
    w: usize,
) -> usize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let two = _mm256_set1_ps(2.0);
    let mut x = 1usize;
    // Need x+8 <= w-1, i.e. x+9 <= w
    while x + 9 <= w {
        let r0l = _mm256_loadu_ps(row0.as_ptr().add(x - 1));
        let r0m = _mm256_loadu_ps(row0.as_ptr().add(x));
        let r0r = _mm256_loadu_ps(row0.as_ptr().add(x + 1));
        let r1l = _mm256_loadu_ps(row1.as_ptr().add(x - 1));
        let r1r = _mm256_loadu_ps(row1.as_ptr().add(x + 1));
        let r2l = _mm256_loadu_ps(row2.as_ptr().add(x - 1));
        let r2m = _mm256_loadu_ps(row2.as_ptr().add(x));
        let r2r = _mm256_loadu_ps(row2.as_ptr().add(x + 1));

        let dx0 = _mm256_sub_ps(r0r, r0l);
        let dx1 = _mm256_sub_ps(r1r, r1l);
        let dx2 = _mm256_sub_ps(r2r, r2l);
        let gx = _mm256_add_ps(_mm256_add_ps(dx0, dx2), _mm256_mul_ps(dx1, two));

        let sy0 = _mm256_add_ps(_mm256_add_ps(r0l, r0r), _mm256_mul_ps(r0m, two));
        let sy2 = _mm256_add_ps(_mm256_add_ps(r2l, r2r), _mm256_mul_ps(r2m, two));
        let gy = _mm256_sub_ps(sy2, sy0);

        _mm256_storeu_ps(gx_out.as_mut_ptr().add(x), gx);
        _mm256_storeu_ps(gy_out.as_mut_ptr().add(x), gy);
        x += 8;
    }
    x
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn sobel_sse_row_c1(
    row0: &[f32],
    row1: &[f32],
    row2: &[f32],
    gx_out: &mut [f32],
    gy_out: &mut [f32],
    w: usize,
) -> usize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let two = _mm_set1_ps(2.0);
    let mut x = 1usize;
    while x + 5 <= w {
        let r0l = _mm_loadu_ps(row0.as_ptr().add(x - 1));
        let r0m = _mm_loadu_ps(row0.as_ptr().add(x));
        let r0r = _mm_loadu_ps(row0.as_ptr().add(x + 1));
        let r1l = _mm_loadu_ps(row1.as_ptr().add(x - 1));
        let r1r = _mm_loadu_ps(row1.as_ptr().add(x + 1));
        let r2l = _mm_loadu_ps(row2.as_ptr().add(x - 1));
        let r2m = _mm_loadu_ps(row2.as_ptr().add(x));
        let r2r = _mm_loadu_ps(row2.as_ptr().add(x + 1));

        let dx0 = _mm_sub_ps(r0r, r0l);
        let dx1 = _mm_sub_ps(r1r, r1l);
        let dx2 = _mm_sub_ps(r2r, r2l);
        let gx = _mm_add_ps(_mm_add_ps(dx0, dx2), _mm_mul_ps(dx1, two));

        let sy0 = _mm_add_ps(_mm_add_ps(r0l, r0r), _mm_mul_ps(r0m, two));
        let sy2 = _mm_add_ps(_mm_add_ps(r2l, r2r), _mm_mul_ps(r2m, two));
        let gy = _mm_sub_ps(sy2, sy0);

        _mm_storeu_ps(gx_out.as_mut_ptr().add(x), gx);
        _mm_storeu_ps(gy_out.as_mut_ptr().add(x), gy);
        x += 4;
    }
    x
}

/// Computes 3x3 Sobel gradient magnitude `sqrt(gx^2 + gy^2)` per channel.
#[allow(unsafe_code, clippy::uninit_vec)]
pub fn sobel_3x3_magnitude(input: &Tensor) -> Result<Tensor, ImgProcError> {
    let (gx, gy) = sobel_3x3_gradients(input)?;
    let gx_data = gx.data();
    let gy_data = gy.data();
    let total = gx.len();
    // SAFETY: every element is written by compute_chunk below.
    let mut out: Vec<f32> = Vec::with_capacity(total);
    unsafe {
        out.set_len(total);
    }

    let compute_chunk = |chunk: &mut [f32], start: usize| {
        let end = start + chunk.len();
        let mut i = start;

        if !cfg!(miri) {
            i = start + magnitude_simd(&gx_data[start..end], &gy_data[start..end], chunk);
        }

        while i < end {
            let x = gx_data[i];
            let y = gy_data[i];
            chunk[i - start] = (x * x + y * y).sqrt();
            i += 1;
        }
    };

    #[cfg(target_os = "macos")]
    if total > 4096 && !cfg!(miri) {
        let shape = gx.shape().to_vec();
        let (h, w, _channels) = (shape[0], shape[1], shape[2]);
        let row_len = w * shape[2];
        let gx_ptr = super::SendConstPtr(gx_data.as_ptr());
        let gy_ptr = super::SendConstPtr(gy_data.as_ptr());
        let out_ptr = super::SendPtr(out.as_mut_ptr());
        use super::u8ops::gcd;
        gcd::parallel_for(h, |y| {
            let start = y * row_len;
            // SAFETY: each row writes to a disjoint slice.
            let gx_slice =
                unsafe { std::slice::from_raw_parts(gx_ptr.ptr().add(start), row_len) };
            let gy_slice =
                unsafe { std::slice::from_raw_parts(gy_ptr.ptr().add(start), row_len) };
            let dst = unsafe {
                std::slice::from_raw_parts_mut(out_ptr.ptr().add(start), row_len)
            };
            let mut i = magnitude_simd(gx_slice, gy_slice, dst);
            while i < row_len {
                let x = gx_slice[i];
                let yv = gy_slice[i];
                dst[i] = (x * x + yv * yv).sqrt();
                i += 1;
            }
        });
        return Tensor::from_vec(shape, out).map_err(Into::into);
    }

    if total > 4096 {
        out.par_chunks_mut(1024)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                compute_chunk(chunk, chunk_idx * 1024);
            });
    } else {
        compute_chunk(&mut out, 0);
    }

    Tensor::from_vec(gx.shape().to_vec(), out).map_err(Into::into)
}

/// SIMD magnitude: out[i] = sqrt(gx[i]² + gy[i]²). Returns count processed.
#[allow(unsafe_code)]
fn magnitude_simd(gx: &[f32], gy: &[f32], out: &mut [f32]) -> usize {
    let len = gx.len().min(gy.len()).min(out.len());
    if len < 4 {
        return 0;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { magnitude_neon(gx, gy, out, len) };
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            return unsafe { magnitude_avx(gx, gy, out, len) };
        }
        if std::is_x86_feature_detected!("sse") {
            return unsafe { magnitude_sse(gx, gy, out, len) };
        }
    }

    0
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn magnitude_neon(gx: &[f32], gy: &[f32], out: &mut [f32], len: usize) -> usize {
    use std::arch::aarch64::*;
    let gxp = gx.as_ptr();
    let gyp = gy.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = 0usize;
    while i + 4 <= len {
        let x = vld1q_f32(gxp.add(i));
        let y = vld1q_f32(gyp.add(i));
        let sq = vaddq_f32(vmulq_f32(x, x), vmulq_f32(y, y));
        vst1q_f32(op.add(i), vsqrtq_f32(sq));
        i += 4;
    }
    i
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn magnitude_avx(gx: &[f32], gy: &[f32], out: &mut [f32], len: usize) -> usize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let gxp = gx.as_ptr();
    let gyp = gy.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = 0usize;
    while i + 8 <= len {
        let x = _mm256_loadu_ps(gxp.add(i));
        let y = _mm256_loadu_ps(gyp.add(i));
        let sq = _mm256_add_ps(_mm256_mul_ps(x, x), _mm256_mul_ps(y, y));
        _mm256_storeu_ps(op.add(i), _mm256_sqrt_ps(sq));
        i += 8;
    }
    i
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn magnitude_sse(gx: &[f32], gy: &[f32], out: &mut [f32], len: usize) -> usize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let gxp = gx.as_ptr();
    let gyp = gy.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = 0usize;
    while i + 4 <= len {
        let x = _mm_loadu_ps(gxp.add(i));
        let y = _mm_loadu_ps(gyp.add(i));
        let sq = _mm_add_ps(_mm_mul_ps(x, x), _mm_mul_ps(y, y));
        _mm_storeu_ps(op.add(i), _mm_sqrt_ps(sq));
        i += 4;
    }
    i
}

/// Computes 3x3 Scharr gradients (`gx`, `gy`) per channel.
///
/// Scharr kernels have better rotational symmetry than Sobel:
/// - Horizontal: `[[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]`
/// - Vertical: transpose of horizontal
pub fn scharr_3x3_gradients(input: &Tensor) -> Result<(Tensor, Tensor), ImgProcError> {
    let (h, w, channels) = hwc_shape(input)?;
    let len = h * w * channels;
    let mut out_gx = vec![0.0f32; len];
    let mut out_gy = vec![0.0f32; len];

    const SCHARR_X: [[f32; 3]; 3] = [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]];
    const SCHARR_Y: [[f32; 3]; 3] = [[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]];

    let data = input.data();
    let row_len = w * channels;
    let interior_h = h.saturating_sub(2);

    // Interior pixels (y=1..h-1, x=1..w-1): no bounds checks
    let compute_interior_row = |y: usize, gx_row: &mut [f32], gy_row: &mut [f32]| {
        for x in 1..w.saturating_sub(1) {
            for c in 0..channels {
                let r0 = ((y - 1) * w + x - 1) * channels + c;
                let r1 = (y * w + x - 1) * channels + c;
                let r2 = ((y + 1) * w + x - 1) * channels + c;
                let mut gx = 0.0f32;
                let mut gy = 0.0f32;
                gx += data[r0] * SCHARR_X[0][0];
                gx += data[r0 + 2 * channels] * SCHARR_X[0][2];
                gx += data[r1] * SCHARR_X[1][0];
                gx += data[r1 + 2 * channels] * SCHARR_X[1][2];
                gx += data[r2] * SCHARR_X[2][0];
                gx += data[r2 + 2 * channels] * SCHARR_X[2][2];

                gy += data[r0] * SCHARR_Y[0][0];
                gy += data[r0 + channels] * SCHARR_Y[0][1];
                gy += data[r0 + 2 * channels] * SCHARR_Y[0][2];
                gy += data[r2] * SCHARR_Y[2][0];
                gy += data[r2 + channels] * SCHARR_Y[2][1];
                gy += data[r2 + 2 * channels] * SCHARR_Y[2][2];

                gx_row[x * channels + c] = gx;
                gy_row[x * channels + c] = gy;
            }
        }
    };

    if interior_h > 0 {
        let gx_interior = &mut out_gx[row_len..row_len + interior_h * row_len];
        let gy_interior = &mut out_gy[row_len..row_len + interior_h * row_len];
        if h * w > 4096 {
            gx_interior
                .par_chunks_mut(row_len)
                .zip(gy_interior.par_chunks_mut(row_len))
                .enumerate()
                .for_each(|(i, (gx_row, gy_row))| {
                    compute_interior_row(i + 1, gx_row, gy_row);
                });
        } else {
            gx_interior
                .chunks_mut(row_len)
                .zip(gy_interior.chunks_mut(row_len))
                .enumerate()
                .for_each(|(i, (gx_row, gy_row))| {
                    compute_interior_row(i + 1, gx_row, gy_row);
                });
        }
    }

    // Border pixels: bounds-checked path
    let border = border_coords_3x3(h, w);
    for (y, x) in border {
        for c in 0..channels {
            let mut gx = 0.0f32;
            let mut gy = 0.0f32;
            for ky in -1isize..=1 {
                for kx in -1isize..=1 {
                    let sy = y as isize + ky;
                    let sx = x as isize + kx;
                    if sy < 0 || sx < 0 || sy >= h as isize || sx >= w as isize {
                        continue;
                    }
                    let src = ((sy as usize) * w + sx as usize) * channels + c;
                    let kernel_y = (ky + 1) as usize;
                    let kernel_x = (kx + 1) as usize;
                    let value = data[src];
                    gx += value * SCHARR_X[kernel_y][kernel_x];
                    gy += value * SCHARR_Y[kernel_y][kernel_x];
                }
            }
            let dst = (y * w + x) * channels + c;
            out_gx[dst] = gx;
            out_gy[dst] = gy;
        }
    }

    let shape = vec![h, w, channels];
    let gx = Tensor::from_vec(shape.clone(), out_gx)?;
    let gy = Tensor::from_vec(shape, out_gy)?;
    Ok((gx, gy))
}

/// Computes the Scharr gradient magnitude: `sqrt(gx^2 + gy^2)`.
pub fn scharr_3x3_magnitude(input: &Tensor) -> Result<Tensor, ImgProcError> {
    let (gx, gy) = scharr_3x3_gradients(input)?;
    let gx_data = gx.data();
    let gy_data = gy.data();
    let mut out = vec![0.0f32; gx.len()];

    if out.len() > 4096 {
        out.par_chunks_mut(1024)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let start = chunk_idx * 1024;
                for (j, v) in chunk.iter_mut().enumerate() {
                    let i = start + j;
                    let x = gx_data[i];
                    let y = gy_data[i];
                    *v = (x * x + y * y).sqrt();
                }
            });
    } else {
        for (idx, value) in out.iter_mut().enumerate() {
            let x = gx_data[idx];
            let y = gy_data[idx];
            *value = (x * x + y * y).sqrt();
        }
    }

    Tensor::from_vec(gx.shape().to_vec(), out).map_err(Into::into)
}

/// Flips an HWC tensor horizontally (mirror over vertical axis).
pub fn flip_horizontal(input: &Tensor) -> Result<Tensor, ImgProcError> {
    let (h, w, channels) = hwc_shape(input)?;
    let mut out = vec![0.0f32; input.len()];

    for y in 0..h {
        for x in 0..w {
            let src_x = w - 1 - x;
            let dst_base = (y * w + x) * channels;
            let src_base = (y * w + src_x) * channels;
            out[dst_base..(dst_base + channels)]
                .copy_from_slice(&input.data()[src_base..(src_base + channels)]);
        }
    }

    Tensor::from_vec(vec![h, w, channels], out).map_err(Into::into)
}

/// Flips an HWC tensor vertically (mirror over horizontal axis).
pub fn flip_vertical(input: &Tensor) -> Result<Tensor, ImgProcError> {
    let (h, w, channels) = hwc_shape(input)?;
    let mut out = vec![0.0f32; input.len()];

    for y in 0..h {
        let src_y = h - 1 - y;
        for x in 0..w {
            let dst_base = (y * w + x) * channels;
            let src_base = (src_y * w + x) * channels;
            out[dst_base..(dst_base + channels)]
                .copy_from_slice(&input.data()[src_base..(src_base + channels)]);
        }
    }

    Tensor::from_vec(vec![h, w, channels], out).map_err(Into::into)
}

/// Rotates an HWC tensor 90 degrees clockwise.
pub fn rotate90_cw(input: &Tensor) -> Result<Tensor, ImgProcError> {
    let (in_h, in_w, channels) = hwc_shape(input)?;
    let out_h = in_w;
    let out_w = in_h;
    let mut out = vec![0.0f32; input.len()];

    for y in 0..in_h {
        for x in 0..in_w {
            let dst_y = x;
            let dst_x = in_h - 1 - y;
            let src_base = (y * in_w + x) * channels;
            let dst_base = (dst_y * out_w + dst_x) * channels;
            out[dst_base..(dst_base + channels)]
                .copy_from_slice(&input.data()[src_base..(src_base + channels)]);
        }
    }

    Tensor::from_vec(vec![out_h, out_w, channels], out).map_err(Into::into)
}

/// Pads an HWC tensor with a constant value on all sides.
pub fn pad_constant(
    input: &Tensor,
    pad_top: usize,
    pad_bottom: usize,
    pad_left: usize,
    pad_right: usize,
    value: f32,
) -> Result<Tensor, ImgProcError> {
    let (in_h, in_w, channels) = hwc_shape(input)?;
    let out_h = in_h + pad_top + pad_bottom;
    let out_w = in_w + pad_left + pad_right;
    let mut out = vec![value; out_h * out_w * channels];

    let data = input.data();
    for y in 0..in_h {
        for x in 0..in_w {
            let src_base = (y * in_w + x) * channels;
            let dst_base = ((y + pad_top) * out_w + x + pad_left) * channels;
            out[dst_base..dst_base + channels]
                .copy_from_slice(&data[src_base..src_base + channels]);
        }
    }

    Tensor::from_vec(vec![out_h, out_w, channels], out).map_err(Into::into)
}

/// Crops an HWC tensor to `[crop_h, crop_w, C]` starting at `(top, left)`.
pub fn crop(
    input: &Tensor,
    top: usize,
    left: usize,
    crop_h: usize,
    crop_w: usize,
) -> Result<Tensor, ImgProcError> {
    let (in_h, in_w, channels) = hwc_shape(input)?;
    if top + crop_h > in_h || left + crop_w > in_w || crop_h == 0 || crop_w == 0 {
        return Err(ImgProcError::InvalidSize {
            height: crop_h,
            width: crop_w,
        });
    }

    let data = input.data();
    let mut out = vec![0.0f32; crop_h * crop_w * channels];
    for y in 0..crop_h {
        for x in 0..crop_w {
            let src_base = ((y + top) * in_w + x + left) * channels;
            let dst_base = (y * crop_w + x) * channels;
            out[dst_base..dst_base + channels]
                .copy_from_slice(&data[src_base..src_base + channels]);
        }
    }

    Tensor::from_vec(vec![crop_h, crop_w, channels], out).map_err(Into::into)
}

/// Applies a 2x3 affine transformation to an HWC image using bilinear interpolation.
///
/// `matrix` is `[a00, a01, tx, a10, a11, ty]` mapping `dst -> src`:
/// `src_x = a00*dst_x + a01*dst_y + tx`, `src_y = a10*dst_x + a11*dst_y + ty`.
/// Out-of-bounds pixels are filled with `border_value`.
pub fn warp_affine(
    input: &Tensor,
    out_h: usize,
    out_w: usize,
    matrix: &[f32; 6],
    border_value: f32,
) -> Result<Tensor, ImgProcError> {
    let (h, w, channels) = hwc_shape(input)?;
    if out_h == 0 || out_w == 0 {
        return Err(ImgProcError::InvalidOutputDimensions { out_h, out_w });
    }

    let data = input.data();
    let mut out = vec![border_value; out_h * out_w * channels];
    let [a00, a01, tx, a10, a11, ty] = *matrix;

    for dy in 0..out_h {
        for dx in 0..out_w {
            let src_xf = a00 * dx as f32 + a01 * dy as f32 + tx;
            let src_yf = a10 * dx as f32 + a11 * dy as f32 + ty;

            let x0 = src_xf.floor() as isize;
            let y0 = src_yf.floor() as isize;
            let x1 = x0 + 1;
            let y1 = y0 + 1;

            if x0 < 0 || y0 < 0 || x1 >= w as isize || y1 >= h as isize {
                continue;
            }

            let fx = src_xf - x0 as f32;
            let fy = src_yf - y0 as f32;

            let x0u = x0 as usize;
            let y0u = y0 as usize;
            let x1u = x1 as usize;
            let y1u = y1 as usize;

            for c in 0..channels {
                let v00 = data[(y0u * w + x0u) * channels + c];
                let v01 = data[(y0u * w + x1u) * channels + c];
                let v10 = data[(y1u * w + x0u) * channels + c];
                let v11 = data[(y1u * w + x1u) * channels + c];

                let val = v00 * (1.0 - fx) * (1.0 - fy)
                    + v01 * fx * (1.0 - fy)
                    + v10 * (1.0 - fx) * fy
                    + v11 * fx * fy;

                out[(dy * out_w + dx) * channels + c] = val;
            }
        }
    }

    Tensor::from_vec(vec![out_h, out_w, channels], out).map_err(Into::into)
}

/// Applies a 3x3 perspective (homography) transformation to an `[H, W, C]` image.
///
/// `transform` is a 3x3 row-major matrix (9 elements).
///
/// Uses SIMD (NEON/SSE) to process 4 output pixels at a time for the coordinate
/// transformation and bilinear interpolation (single-channel fast path).
#[allow(clippy::too_many_arguments, unsafe_code)]
pub fn warp_perspective(
    input: &Tensor,
    transform: &[f32; 9],
    out_h: usize,
    out_w: usize,
    border_value: f32,
) -> Result<Tensor, ImgProcError> {
    let (ih, iw, channels) = hwc_shape(input)?;
    let in_data = input.data();

    let inv = invert_3x3(transform)
        .ok_or(ImgProcError::InvalidOutputDimensions { out_h: 0, out_w: 0 })?;

    let mut out = vec![border_value; out_h * out_w * channels];

    // For single-channel images, use SIMD-optimized path
    if channels == 1 {
        warp_perspective_c1(in_data, ih, iw, &inv, &mut out, out_h, out_w, border_value);
    } else {
        warp_perspective_scalar(in_data, ih, iw, channels, &inv, &mut out, out_h, out_w);
    }

    Tensor::from_vec(vec![out_h, out_w, channels], out).map_err(Into::into)
}

/// Multi-channel warp perspective, parallelized across output rows.
#[allow(unsafe_code)]
fn warp_perspective_scalar(
    in_data: &[f32],
    ih: usize,
    iw: usize,
    channels: usize,
    inv: &[f32; 9],
    out: &mut [f32],
    out_h: usize,
    out_w: usize,
) {
    use super::u8ops::gcd;
    let out_ptr = super::SendPtr(out.as_mut_ptr());
    let in_ptr = super::SendConstPtr(in_data.as_ptr());
    let in_len = in_data.len();
    let row_stride = out_w * channels;
    let inv = *inv; // copy for closure capture

    gcd::parallel_for(out_h, |dy| {
        // SAFETY: each row writes to non-overlapping out[dy*row_stride..(dy+1)*row_stride].
        let out_row = unsafe {
            std::slice::from_raw_parts_mut(out_ptr.ptr().add(dy * row_stride), row_stride)
        };
        let in_data = unsafe { std::slice::from_raw_parts(in_ptr.ptr(), in_len) };

        let yf = dy as f32 + 0.5;
        let base_num_x = inv[1] * yf + inv[2];
        let base_num_y = inv[4] * yf + inv[5];
        let base_den = inv[7] * yf + inv[8];

        for dx in 0..out_w {
            let xf = dx as f32 + 0.5;
            let denom = inv[6] * xf + base_den;
            if denom.abs() < 1e-10 {
                continue;
            }
            let inv_denom = 1.0 / denom;
            let sx = (inv[0] * xf + base_num_x) * inv_denom - 0.5;
            let sy = (inv[3] * xf + base_num_y) * inv_denom - 0.5;

            if sx < 0.0 || sy < 0.0 || sx >= (iw - 1) as f32 || sy >= (ih - 1) as f32 {
                continue;
            }
            let x0 = sx.floor() as usize;
            let y0 = sy.floor() as usize;
            let x1 = (x0 + 1).min(iw - 1);
            let y1 = (y0 + 1).min(ih - 1);
            let fx = sx - x0 as f32;
            let fy = sy - y0 as f32;

            for c in 0..channels {
                let v00 = in_data[(y0 * iw + x0) * channels + c];
                let v10 = in_data[(y0 * iw + x1) * channels + c];
                let v01 = in_data[(y1 * iw + x0) * channels + c];
                let v11 = in_data[(y1 * iw + x1) * channels + c];
                let val = v00 * (1.0 - fx) * (1.0 - fy)
                    + v10 * fx * (1.0 - fy)
                    + v01 * (1.0 - fx) * fy
                    + v11 * fx * fy;
                out_row[dx * channels + c] = val;
            }
        }
    });
}

/// SIMD-optimized single-channel warp perspective, parallelized across rows.
/// Processes 4 output pixels per iteration using NEON/SSE for coordinate
/// transformation and bilinear interpolation.
#[allow(unsafe_code)]
fn warp_perspective_c1(
    in_data: &[f32],
    ih: usize,
    iw: usize,
    inv: &[f32; 9],
    out: &mut [f32],
    out_h: usize,
    out_w: usize,
    border_value: f32,
) {
    use super::u8ops::gcd;
    let iw_f = (iw - 1) as f32;
    let ih_f = (ih - 1) as f32;
    let out_ptr = super::SendPtr(out.as_mut_ptr());
    let in_ptr = super::SendConstPtr(in_data.as_ptr());
    let in_len = in_data.len();
    let inv = *inv;

    gcd::parallel_for(out_h, |dy| {
        let out_row =
            unsafe { std::slice::from_raw_parts_mut(out_ptr.ptr().add(dy * out_w), out_w) };
        let in_data = unsafe { std::slice::from_raw_parts(in_ptr.ptr(), in_len) };

        let yf = dy as f32 + 0.5;
        let base_num_x = inv[1] * yf + inv[2];
        let base_num_y = inv[4] * yf + inv[5];
        let base_den = inv[7] * yf + inv[8];

        let mut dx = 0usize;

        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                dx = unsafe {
                    warp_perspective_c1_neon_row(
                        in_data,
                        ih,
                        iw,
                        &inv,
                        out_row,
                        out_w,
                        0,
                        yf,
                        base_num_x,
                        base_num_y,
                        base_den,
                        iw_f,
                        ih_f,
                        border_value,
                    )
                };
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse") {
                dx = unsafe {
                    warp_perspective_c1_sse_row(
                        in_data,
                        ih,
                        iw,
                        &inv,
                        out_row,
                        out_w,
                        0,
                        yf,
                        base_num_x,
                        base_num_y,
                        base_den,
                        iw_f,
                        ih_f,
                        border_value,
                    )
                };
            }
        }

        // Scalar tail
        while dx < out_w {
            let xf = dx as f32 + 0.5;
            let denom = inv[6] * xf + base_den;
            if denom.abs() < 1e-10 {
                dx += 1;
                continue;
            }
            let inv_denom = 1.0 / denom;
            let sx = (inv[0] * xf + base_num_x) * inv_denom - 0.5;
            let sy = (inv[3] * xf + base_num_y) * inv_denom - 0.5;

            if sx >= 0.0 && sy >= 0.0 && sx < iw_f && sy < ih_f {
                let x0 = sx.floor() as usize;
                let y0 = sy.floor() as usize;
                let x1 = x0 + 1;
                let y1 = y0 + 1;
                let fx = sx - x0 as f32;
                let fy = sy - y0 as f32;
                let v00 = in_data[y0 * iw + x0];
                let v10 = in_data[y0 * iw + x1];
                let v01 = in_data[y1 * iw + x0];
                let v11 = in_data[y1 * iw + x1];
                out_row[dx] = v00 * (1.0 - fx) * (1.0 - fy)
                    + v10 * fx * (1.0 - fy)
                    + v01 * (1.0 - fx) * fy
                    + v11 * fx * fy;
            }
            dx += 1;
        }
    });
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
unsafe fn warp_perspective_c1_neon_row(
    in_data: &[f32],
    _ih: usize,
    iw: usize,
    inv: &[f32; 9],
    out: &mut [f32],
    out_w: usize,
    dy: usize,
    _yf: f32,
    base_num_x: f32,
    base_num_y: f32,
    base_den: f32,
    iw_f: f32,
    ih_f: f32,
    border_value: f32,
) -> usize {
    use std::arch::aarch64::*;

    let inv0 = vdupq_n_f32(inv[0]);
    let inv3 = vdupq_n_f32(inv[3]);
    let inv6 = vdupq_n_f32(inv[6]);
    let base_nx = vdupq_n_f32(base_num_x);
    let base_ny = vdupq_n_f32(base_num_y);
    let base_d = vdupq_n_f32(base_den);
    let half = vdupq_n_f32(0.5);
    let zero = vdupq_n_f32(0.0);
    let iw_max = vdupq_n_f32(iw_f);
    let ih_max = vdupq_n_f32(ih_f);
    let iw_s = vdupq_n_f32(iw as f32);
    let border = vdupq_n_f32(border_value);

    let row_start = dy * out_w;
    let out_ptr = out.as_mut_ptr().add(row_start);
    let in_ptr = in_data.as_ptr();

    // Precompute xf increment (4 per iteration)
    let xf_init = [0.5f32, 1.5, 2.5, 3.5];
    let mut xf = vaddq_f32(vdupq_n_f32(0.0), vld1q_f32(xf_init.as_ptr()));
    let xf_step = vdupq_n_f32(4.0);

    let mut dx = 0usize;
    while dx + 4 <= out_w {
        // denom = inv[6] * xf + base_den
        let denom = vmlaq_f32(base_d, inv6, xf);
        // Fast reciprocal: 2 iterations of Newton-Raphson for ~24-bit precision
        let recip_est = vrecpeq_f32(denom);
        let inv_denom = vmulq_f32(recip_est, vrecpsq_f32(denom, recip_est));

        // sx = (inv[0] * xf + base_num_x) / denom - 0.5
        let num_x = vmlaq_f32(base_nx, inv0, xf);
        let sx = vsubq_f32(vmulq_f32(num_x, inv_denom), half);

        // sy = (inv[3] * xf + base_num_y) / denom - 0.5
        let num_y = vmlaq_f32(base_ny, inv3, xf);
        let sy = vsubq_f32(vmulq_f32(num_y, inv_denom), half);

        // Bounds check: sx >= 0 && sy >= 0 && sx < iw_f && sy < ih_f
        let in_bounds = vandq_u32(
            vandq_u32(vcgeq_f32(sx, zero), vcgeq_f32(sy, zero)),
            vandq_u32(vcltq_f32(sx, iw_max), vcltq_f32(sy, ih_max)),
        );

        // If no pixels are in bounds, skip
        let mask_bits = vgetq_lane_u32(in_bounds, 0)
            | vgetq_lane_u32(in_bounds, 1)
            | vgetq_lane_u32(in_bounds, 2)
            | vgetq_lane_u32(in_bounds, 3);
        if mask_bits == 0 {
            xf = vaddq_f32(xf, xf_step);
            dx += 4;
            continue;
        }

        // Floor to get integer coords
        let sx_floor = vrndmq_f32(sx);
        let sy_floor = vrndmq_f32(sy);
        let fx = vsubq_f32(sx, sx_floor);
        let fy = vsubq_f32(sy, sy_floor);

        // Compute linear indices: y0*iw + x0
        let idx_base = vmlaq_f32(sx_floor, sy_floor, iw_s);

        // Process each of 4 pixels (scatter-gather for bilinear)
        let mut result = border;
        let fx_arr: [f32; 4] = std::mem::transmute(fx);
        let fy_arr: [f32; 4] = std::mem::transmute(fy);
        let idx_arr: [f32; 4] = std::mem::transmute(idx_base);
        let mask_arr: [u32; 4] = std::mem::transmute(in_bounds);
        let mut res_arr: [f32; 4] = std::mem::transmute(result);

        for i in 0..4 {
            if mask_arr[i] != 0 {
                let base = idx_arr[i] as usize;
                let v00 = *in_ptr.add(base);
                let v10 = *in_ptr.add(base + 1);
                let v01 = *in_ptr.add(base + iw);
                let v11 = *in_ptr.add(base + iw + 1);
                let fxi = fx_arr[i];
                let fyi = fy_arr[i];
                res_arr[i] = v00 * (1.0 - fxi) * (1.0 - fyi)
                    + v10 * fxi * (1.0 - fyi)
                    + v01 * (1.0 - fxi) * fyi
                    + v11 * fxi * fyi;
            }
        }
        result = vld1q_f32(res_arr.as_ptr());
        vst1q_f32(out_ptr.add(dx), result);

        xf = vaddq_f32(xf, xf_step);
        dx += 4;
    }
    dx
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]
#[target_feature(enable = "sse")]
unsafe fn warp_perspective_c1_sse_row(
    in_data: &[f32],
    _ih: usize,
    iw: usize,
    inv: &[f32; 9],
    out: &mut [f32],
    out_w: usize,
    dy: usize,
    _yf: f32,
    base_num_x: f32,
    base_num_y: f32,
    base_den: f32,
    iw_f: f32,
    ih_f: f32,
    border_value: f32,
) -> usize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let inv0 = _mm_set1_ps(inv[0]);
    let inv3 = _mm_set1_ps(inv[3]);
    let inv6 = _mm_set1_ps(inv[6]);
    let base_nx = _mm_set1_ps(base_num_x);
    let base_ny = _mm_set1_ps(base_num_y);
    let base_d = _mm_set1_ps(base_den);
    let half = _mm_set1_ps(0.5);
    let zero = _mm_setzero_ps();
    let iw_max = _mm_set1_ps(iw_f);
    let ih_max = _mm_set1_ps(ih_f);
    let one = _mm_set1_ps(1.0);
    let iw_s = _mm_set1_ps(iw as f32);
    let _border = _mm_set1_ps(border_value);

    let row_start = dy * out_w;
    let out_ptr = out.as_mut_ptr().add(row_start);
    let in_ptr = in_data.as_ptr();

    let mut dx = 0usize;
    while dx + 4 <= out_w {
        let xf = _mm_set_ps(
            dx as f32 + 3.5,
            dx as f32 + 2.5,
            dx as f32 + 1.5,
            dx as f32 + 0.5,
        );

        // denom = inv[6] * xf + base_den
        let denom = _mm_add_ps(_mm_mul_ps(inv6, xf), base_d);
        let inv_denom = _mm_div_ps(one, denom);

        // sx = (inv[0] * xf + base_num_x) / denom - 0.5
        let num_x = _mm_add_ps(_mm_mul_ps(inv0, xf), base_nx);
        let sx = _mm_sub_ps(_mm_mul_ps(num_x, inv_denom), half);

        // sy = (inv[3] * xf + base_num_y) / denom - 0.5
        let num_y = _mm_add_ps(_mm_mul_ps(inv3, xf), base_ny);
        let sy = _mm_sub_ps(_mm_mul_ps(num_y, inv_denom), half);

        // Bounds check
        let in_bounds = _mm_and_ps(
            _mm_and_ps(_mm_cmpge_ps(sx, zero), _mm_cmpge_ps(sy, zero)),
            _mm_and_ps(_mm_cmplt_ps(sx, iw_max), _mm_cmplt_ps(sy, ih_max)),
        );

        let mask_bits = _mm_movemask_ps(in_bounds);
        if mask_bits == 0 {
            dx += 4;
            continue;
        }

        // Since we checked bounds (sx >= 0 and sy >= 0), simple truncation = floor
        let sx_floor = _mm_cvtepi32_ps(_mm_cvttps_epi32(sx));
        let sy_floor = _mm_cvtepi32_ps(_mm_cvttps_epi32(sy));
        let fx = _mm_sub_ps(sx, sx_floor);
        let fy = _mm_sub_ps(sy, sy_floor);

        // Index base: y0 * iw + x0
        let idx_base = _mm_add_ps(_mm_mul_ps(sy_floor, iw_s), sx_floor);

        // Scatter-gather for bilinear interpolation
        let mut res_arr = [border_value; 4];
        let fx_arr: [f32; 4] = std::mem::transmute(fx);
        let fy_arr: [f32; 4] = std::mem::transmute(fy);
        let idx_arr: [f32; 4] = std::mem::transmute(idx_base);

        for i in 0..4 {
            if (mask_bits >> i) & 1 != 0 {
                let base = idx_arr[i] as usize;
                let v00 = *in_ptr.add(base);
                let v10 = *in_ptr.add(base + 1);
                let v01 = *in_ptr.add(base + iw);
                let v11 = *in_ptr.add(base + iw + 1);
                let fxi = fx_arr[i];
                let fyi = fy_arr[i];
                res_arr[i] = v00 * (1.0 - fxi) * (1.0 - fyi)
                    + v10 * fxi * (1.0 - fyi)
                    + v01 * (1.0 - fxi) * fyi
                    + v11 * fxi * fyi;
            }
        }
        _mm_storeu_ps(out_ptr.add(dx), _mm_loadu_ps(res_arr.as_ptr()));

        dx += 4;
    }
    dx
}

pub(crate) fn invert_3x3(m: &[f32; 9]) -> Option<[f32; 9]> {
    let det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if det.abs() < 1e-10 {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        (m[4] * m[8] - m[5] * m[7]) * inv_det,
        (m[2] * m[7] - m[1] * m[8]) * inv_det,
        (m[1] * m[5] - m[2] * m[4]) * inv_det,
        (m[5] * m[6] - m[3] * m[8]) * inv_det,
        (m[0] * m[8] - m[2] * m[6]) * inv_det,
        (m[2] * m[3] - m[0] * m[5]) * inv_det,
        (m[3] * m[7] - m[4] * m[6]) * inv_det,
        (m[1] * m[6] - m[0] * m[7]) * inv_det,
        (m[0] * m[4] - m[1] * m[3]) * inv_det,
    ])
}
