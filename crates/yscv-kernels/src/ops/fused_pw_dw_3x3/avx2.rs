//! AVX2 + FMA fused PW-expand→DW kernels.

use super::{Dw5RowCtx, Dw5TileCtx, PwTileCtx};
use std::arch::x86_64::*;

/// AVX2 PW row: `c_exp` vectorised in YMM groups of 8. Requires
/// `c_exp % 8 == 0`; caller guarantees via `select_variant`.
pub(super) fn compute_pw_row_avx2(
    src_row: &[f32],
    pw_weight: &[f32],
    pw_bias: Option<&[f32]>,
    dst_row: &mut [f32],
    iw_start: usize,
    iw_end: usize,
    c_in: usize,
    c_exp: usize,
    relu: bool,
) {
    // SAFETY: caller verified `avx2`+`fma` available and
    // `c_exp % 8 == 0`; slices sized per contract.
    #[allow(unsafe_code)]
    unsafe {
        compute_pw_row_avx2_inner(
            src_row, pw_weight, pw_bias, dst_row, iw_start, iw_end, c_in, c_exp, relu,
        )
    }
}

#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_code)]
unsafe fn compute_pw_row_avx2_inner(
    src_row: &[f32],
    pw_weight: &[f32],
    pw_bias: Option<&[f32]>,
    dst_row: &mut [f32],
    iw_start: usize,
    iw_end: usize,
    c_in: usize,
    c_exp: usize,
    relu: bool,
) {
    unsafe {
        let zero = _mm256_setzero_ps();
        let c_exp_chunks = c_exp / 8;

        for iw in iw_start..iw_end {
            let src_off = iw * c_in;
            let dst_off = iw * c_exp;
            let dst_ptr = dst_row.as_mut_ptr().add(dst_off);

            // Init dst with bias (or zero).
            if let Some(b) = pw_bias {
                let bp = b.as_ptr();
                for ck in 0..c_exp_chunks {
                    let v = _mm256_loadu_ps(bp.add(ck * 8));
                    _mm256_storeu_ps(dst_ptr.add(ck * 8), v);
                }
            } else {
                for ck in 0..c_exp_chunks {
                    _mm256_storeu_ps(dst_ptr.add(ck * 8), zero);
                }
            }

            // Accumulate c_in × c_exp matmul, 8 c_exp lanes at a time.
            for ci in 0..c_in {
                let x = _mm256_set1_ps(*src_row.as_ptr().add(src_off + ci));
                let w_row = pw_weight.as_ptr().add(ci * c_exp);
                for ck in 0..c_exp_chunks {
                    let w = _mm256_loadu_ps(w_row.add(ck * 8));
                    let d = _mm256_loadu_ps(dst_ptr.add(ck * 8));
                    let r = _mm256_fmadd_ps(x, w, d);
                    _mm256_storeu_ps(dst_ptr.add(ck * 8), r);
                }
            }

            if relu {
                for ck in 0..c_exp_chunks {
                    let d = _mm256_loadu_ps(dst_ptr.add(ck * 8));
                    let r = _mm256_max_ps(d, zero);
                    _mm256_storeu_ps(dst_ptr.add(ck * 8), r);
                }
            }
        }
    }
}

/// AVX2 DW 3×3 row: C_exp vectorised in YMM groups of 8.
#[allow(clippy::too_many_arguments)]
pub(super) fn compute_dw_row_avx2(
    row0: Option<&[f32]>,
    row1: Option<&[f32]>,
    row2: Option<&[f32]>,
    dw_weight: &[f32],
    dw_bias: Option<&[f32]>,
    out_row: &mut [f32],
    in_w: usize,
    ow_start: usize,
    ow_end: usize,
    c_exp: usize,
    stride: usize,
    pad: usize,
    relu: bool,
) {
    // SAFETY: caller verified avx2+fma available and c_exp % 8 == 0.
    #[allow(unsafe_code)]
    unsafe {
        compute_dw_row_avx2_inner(
            row0, row1, row2, dw_weight, dw_bias, out_row, in_w, ow_start, ow_end, c_exp, stride,
            pad, relu,
        )
    }
}

#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_code, clippy::too_many_arguments)]
unsafe fn compute_dw_row_avx2_inner(
    row0: Option<&[f32]>,
    row1: Option<&[f32]>,
    row2: Option<&[f32]>,
    dw_weight: &[f32],
    dw_bias: Option<&[f32]>,
    out_row: &mut [f32],
    in_w: usize,
    ow_start: usize,
    ow_end: usize,
    c_exp: usize,
    stride: usize,
    pad: usize,
    relu: bool,
) {
    unsafe {
        let zero = _mm256_setzero_ps();
        let c_exp_chunks = c_exp / 8;
        let w_ky_stride = 3 * c_exp;
        let w_kx_stride = c_exp;

        let rows = [row0, row1, row2];

        for ow in ow_start..ow_end {
            let iw0 = (ow as i32) * (stride as i32) - (pad as i32);
            let dst_off = ow * c_exp;
            let dst_ptr = out_row.as_mut_ptr().add(dst_off);

            // Init with bias or zero.
            if let Some(b) = dw_bias {
                let bp = b.as_ptr();
                for ck in 0..c_exp_chunks {
                    _mm256_storeu_ps(dst_ptr.add(ck * 8), _mm256_loadu_ps(bp.add(ck * 8)));
                }
            } else {
                for ck in 0..c_exp_chunks {
                    _mm256_storeu_ps(dst_ptr.add(ck * 8), zero);
                }
            }

            // 9 taps; skip OOB (row None or iw OOB).
            for ky in 0..3usize {
                let pw_row = match rows[ky] {
                    Some(r) => r,
                    None => continue,
                };
                for kx in 0..3usize {
                    let iw = iw0 + kx as i32;
                    if iw < 0 || (iw as usize) >= in_w {
                        continue;
                    }
                    let iw_u = iw as usize;
                    let pw_ptr = pw_row.as_ptr().add(iw_u * c_exp);
                    let w_ptr = dw_weight.as_ptr().add(ky * w_ky_stride + kx * w_kx_stride);
                    for ck in 0..c_exp_chunks {
                        let x = _mm256_loadu_ps(pw_ptr.add(ck * 8));
                        let w = _mm256_loadu_ps(w_ptr.add(ck * 8));
                        let d = _mm256_loadu_ps(dst_ptr.add(ck * 8));
                        let r = _mm256_fmadd_ps(x, w, d);
                        _mm256_storeu_ps(dst_ptr.add(ck * 8), r);
                    }
                }
            }

            if relu {
                for ck in 0..c_exp_chunks {
                    let d = _mm256_loadu_ps(dst_ptr.add(ck * 8));
                    let r = _mm256_max_ps(d, zero);
                    _mm256_storeu_ps(dst_ptr.add(ck * 8), r);
                }
            }
        }
    }
}

pub(super) fn compute_dw5_row_avx2(ctx: Dw5RowCtx<'_, '_>) {
    #[allow(unsafe_code)]
    unsafe {
        compute_dw5_row_avx2_inner(ctx)
    }
}

#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_code)]
unsafe fn compute_dw5_row_avx2_inner(ctx: Dw5RowCtx<'_, '_>) {
    unsafe {
        let Dw5RowCtx {
            rows,
            dw_weight,
            dw_bias,
            out_row,
            in_w,
            ow_start,
            ow_end,
            c_exp,
            stride,
            pad,
            relu,
        } = ctx;
        let zero = _mm256_setzero_ps();
        let c_exp_chunks = c_exp / 8;
        let w_ky_stride = 5 * c_exp;
        let w_kx_stride = c_exp;

        for ow in ow_start..ow_end {
            let iw0 = (ow as i32) * (stride as i32) - (pad as i32);
            let dst_ptr = out_row.as_mut_ptr().add(ow * c_exp);

            if let Some(b) = dw_bias {
                let bp = b.as_ptr();
                for ck in 0..c_exp_chunks {
                    _mm256_storeu_ps(dst_ptr.add(ck * 8), _mm256_loadu_ps(bp.add(ck * 8)));
                }
            } else {
                for ck in 0..c_exp_chunks {
                    _mm256_storeu_ps(dst_ptr.add(ck * 8), zero);
                }
            }

            for (ky, row) in rows.iter().enumerate() {
                let pw_row = match row {
                    Some(r) => *r,
                    None => continue,
                };
                for kx in 0..5usize {
                    let iw = iw0 + kx as i32;
                    if iw < 0 || (iw as usize) >= in_w {
                        continue;
                    }
                    let pw_ptr = pw_row.as_ptr().add(iw as usize * c_exp);
                    let w_ptr = dw_weight.as_ptr().add(ky * w_ky_stride + kx * w_kx_stride);
                    for ck in 0..c_exp_chunks {
                        let x = _mm256_loadu_ps(pw_ptr.add(ck * 8));
                        let w = _mm256_loadu_ps(w_ptr.add(ck * 8));
                        let d = _mm256_loadu_ps(dst_ptr.add(ck * 8));
                        _mm256_storeu_ps(dst_ptr.add(ck * 8), _mm256_fmadd_ps(x, w, d));
                    }
                }
            }

            if relu {
                for ck in 0..c_exp_chunks {
                    let d = _mm256_loadu_ps(dst_ptr.add(ck * 8));
                    _mm256_storeu_ps(dst_ptr.add(ck * 8), _mm256_max_ps(d, zero));
                }
            }
        }
    }
}

pub(super) fn compute_pw_tile_avx2(ctx: PwTileCtx<'_, '_>) {
    #[allow(unsafe_code)]
    unsafe {
        compute_pw_tile_avx2_inner(ctx)
    }
}

#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_code)]
unsafe fn compute_pw_tile_avx2_inner(ctx: PwTileCtx<'_, '_>) {
    unsafe {
        let PwTileCtx {
            src_row,
            pw_weight,
            pw_bias,
            dst_row,
            iw_start,
            iw_end,
            c_in,
            c_exp,
            oc_start,
            c_tile,
            relu,
        } = ctx;
        let zero = _mm256_setzero_ps();
        let chunks = c_tile / 8;
        let mut iw = iw_start;
        while iw + 1 < iw_end {
            let dst_ptr0 = dst_row.as_mut_ptr().add(iw * c_tile);
            let dst_ptr1 = dst_row.as_mut_ptr().add((iw + 1) * c_tile);
            if let Some(b) = pw_bias {
                let bp = b.as_ptr().add(oc_start);
                for ck in 0..chunks {
                    let bv = _mm256_loadu_ps(bp.add(ck * 8));
                    _mm256_storeu_ps(dst_ptr0.add(ck * 8), bv);
                    _mm256_storeu_ps(dst_ptr1.add(ck * 8), bv);
                }
            } else {
                for ck in 0..chunks {
                    _mm256_storeu_ps(dst_ptr0.add(ck * 8), zero);
                    _mm256_storeu_ps(dst_ptr1.add(ck * 8), zero);
                }
            }
            for ci in 0..c_in {
                let x0 = _mm256_set1_ps(*src_row.as_ptr().add(iw * c_in + ci));
                let x1 = _mm256_set1_ps(*src_row.as_ptr().add((iw + 1) * c_in + ci));
                let wp = pw_weight.as_ptr().add(ci * c_exp + oc_start);
                for ck in 0..chunks {
                    let w = _mm256_loadu_ps(wp.add(ck * 8));
                    let d0 = _mm256_loadu_ps(dst_ptr0.add(ck * 8));
                    let d1 = _mm256_loadu_ps(dst_ptr1.add(ck * 8));
                    _mm256_storeu_ps(dst_ptr0.add(ck * 8), _mm256_fmadd_ps(x0, w, d0));
                    _mm256_storeu_ps(dst_ptr1.add(ck * 8), _mm256_fmadd_ps(x1, w, d1));
                }
            }
            if relu {
                for ck in 0..chunks {
                    let d0 = _mm256_loadu_ps(dst_ptr0.add(ck * 8));
                    let d1 = _mm256_loadu_ps(dst_ptr1.add(ck * 8));
                    _mm256_storeu_ps(dst_ptr0.add(ck * 8), _mm256_max_ps(d0, zero));
                    _mm256_storeu_ps(dst_ptr1.add(ck * 8), _mm256_max_ps(d1, zero));
                }
            }
            iw += 2;
        }
        while iw < iw_end {
            let dst_ptr = dst_row.as_mut_ptr().add(iw * c_tile);
            if let Some(b) = pw_bias {
                let bp = b.as_ptr().add(oc_start);
                for ck in 0..chunks {
                    _mm256_storeu_ps(dst_ptr.add(ck * 8), _mm256_loadu_ps(bp.add(ck * 8)));
                }
            } else {
                for ck in 0..chunks {
                    _mm256_storeu_ps(dst_ptr.add(ck * 8), zero);
                }
            }
            for ci in 0..c_in {
                let x = _mm256_set1_ps(*src_row.as_ptr().add(iw * c_in + ci));
                let wp = pw_weight.as_ptr().add(ci * c_exp + oc_start);
                for ck in 0..chunks {
                    let w = _mm256_loadu_ps(wp.add(ck * 8));
                    let d = _mm256_loadu_ps(dst_ptr.add(ck * 8));
                    _mm256_storeu_ps(dst_ptr.add(ck * 8), _mm256_fmadd_ps(x, w, d));
                }
            }
            if relu {
                for ck in 0..chunks {
                    let d = _mm256_loadu_ps(dst_ptr.add(ck * 8));
                    _mm256_storeu_ps(dst_ptr.add(ck * 8), _mm256_max_ps(d, zero));
                }
            }
            iw += 1;
        }
    }
}

pub(super) fn compute_dw5_tile_avx2(ctx: Dw5TileCtx<'_, '_>) {
    #[allow(unsafe_code)]
    unsafe {
        compute_dw5_tile_avx2_inner(ctx)
    }
}

#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_code)]
unsafe fn compute_dw5_tile_avx2_inner(ctx: Dw5TileCtx<'_, '_>) {
    unsafe {
        let Dw5TileCtx {
            rows,
            dw_weight,
            dw_bias,
            out_row,
            in_w,
            ow_start,
            ow_end,
            c_exp,
            oc_start,
            c_tile,
            stride,
            pad,
            relu,
        } = ctx;
        let zero = _mm256_setzero_ps();
        let chunks = c_tile / 8;
        let w_ky_stride = 5 * c_exp;
        let w_kx_stride = c_exp;
        for ow in ow_start..ow_end {
            let iw0 = (ow as i32) * (stride as i32) - (pad as i32);
            let dst_ptr = out_row.as_mut_ptr().add(ow * c_exp + oc_start);
            if let Some(b) = dw_bias {
                let bp = b.as_ptr().add(oc_start);
                for ck in 0..chunks {
                    _mm256_storeu_ps(dst_ptr.add(ck * 8), _mm256_loadu_ps(bp.add(ck * 8)));
                }
            } else {
                for ck in 0..chunks {
                    _mm256_storeu_ps(dst_ptr.add(ck * 8), zero);
                }
            }
            for (ky, row) in rows.iter().enumerate() {
                let pw_row = match row {
                    Some(r) => *r,
                    None => continue,
                };
                for kx in 0..5usize {
                    let iw = iw0 + kx as i32;
                    if iw < 0 || (iw as usize) >= in_w {
                        continue;
                    }
                    let pw_ptr = pw_row.as_ptr().add(iw as usize * c_tile);
                    let w_ptr = dw_weight
                        .as_ptr()
                        .add(ky * w_ky_stride + kx * w_kx_stride + oc_start);
                    for ck in 0..chunks {
                        let x = _mm256_loadu_ps(pw_ptr.add(ck * 8));
                        let w = _mm256_loadu_ps(w_ptr.add(ck * 8));
                        let d = _mm256_loadu_ps(dst_ptr.add(ck * 8));
                        _mm256_storeu_ps(dst_ptr.add(ck * 8), _mm256_fmadd_ps(x, w, d));
                    }
                }
            }
            if relu {
                for ck in 0..chunks {
                    let d = _mm256_loadu_ps(dst_ptr.add(ck * 8));
                    _mm256_storeu_ps(dst_ptr.add(ck * 8), _mm256_max_ps(d, zero));
                }
            }
        }
    }
}
