//! Scalar reference fused PW-expand→DW kernels (all-arch fallback).

use super::{Dw5RowCtx, Dw5TileCtx, PwTileCtx};

/// Scalar PW row: for each output column `iw`, compute `c_exp`
/// outputs from `c_in` inputs. Applies bias + Relu.
pub(super) fn compute_pw_row_scalar(
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
    for iw in iw_start..iw_end {
        let src_off = iw * c_in;
        let dst_off = iw * c_exp;
        // Initialise `dst[dst_off..dst_off+c_exp]` from bias or 0.
        if let Some(b) = pw_bias {
            dst_row[dst_off..dst_off + c_exp].copy_from_slice(&b[..c_exp]);
        } else {
            dst_row[dst_off..dst_off + c_exp].fill(0.0);
        }
        // Accumulate c_in × c_exp matmul.
        for ci in 0..c_in {
            let x = src_row[src_off + ci];
            let w_row = &pw_weight[ci * c_exp..(ci + 1) * c_exp];
            for ce in 0..c_exp {
                dst_row[dst_off + ce] += x * w_row[ce];
            }
        }
        if relu {
            for v in &mut dst_row[dst_off..dst_off + c_exp] {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
        }
    }
}

/// Scalar DW 3×3 row: reads up to three PW rows (each Option<&[f32]>
/// — None for padding rows), writes one DW output row.
#[allow(clippy::too_many_arguments)]
pub(super) fn compute_dw_row_scalar(
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
    // `dw_weight` layout: `[KY=3][KX=3][C_exp]` flattened.
    let w_ky_stride = 3 * c_exp;
    let w_kx_stride = c_exp;
    // Safe Option → slice wrapper; maps None to a zero slice of
    // length c_exp (allocated once).
    for ow in ow_start..ow_end {
        let iw0 = (ow as i32) * (stride as i32) - (pad as i32);
        let dst_off = ow * c_exp;
        // Init acc with bias.
        if let Some(b) = dw_bias {
            out_row[dst_off..dst_off + c_exp].copy_from_slice(&b[..c_exp]);
        } else {
            out_row[dst_off..dst_off + c_exp].fill(0.0);
        }

        // Iterate over 9 kernel taps (ky, kx). Skip OOB input
        // positions via `row{0,1,2}` being None or `iw` OOB.
        let rows = [row0, row1, row2];
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
                let pw_off = iw_u * c_exp;
                let w_off = ky * w_ky_stride + kx * w_kx_stride;
                for ce in 0..c_exp {
                    out_row[dst_off + ce] += pw_row[pw_off + ce] * dw_weight[w_off + ce];
                }
            }
        }
        if relu {
            for v in &mut out_row[dst_off..dst_off + c_exp] {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
        }
    }
}

pub(super) fn compute_dw5_row_scalar(ctx: Dw5RowCtx<'_, '_>) {
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
    let w_ky_stride = 5 * c_exp;
    let w_kx_stride = c_exp;
    for ow in ow_start..ow_end {
        let iw0 = (ow as i32) * (stride as i32) - (pad as i32);
        let dst_off = ow * c_exp;
        if let Some(b) = dw_bias {
            out_row[dst_off..dst_off + c_exp].copy_from_slice(&b[..c_exp]);
        } else {
            out_row[dst_off..dst_off + c_exp].fill(0.0);
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
                let pw_off = iw as usize * c_exp;
                let w_off = ky * w_ky_stride + kx * w_kx_stride;
                for ce in 0..c_exp {
                    out_row[dst_off + ce] += pw_row[pw_off + ce] * dw_weight[w_off + ce];
                }
            }
        }
        if relu {
            for v in &mut out_row[dst_off..dst_off + c_exp] {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
        }
    }
}

pub(super) fn compute_pw_tile_scalar(ctx: PwTileCtx<'_, '_>) {
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
    for iw in iw_start..iw_end {
        let src_off = iw * c_in;
        let dst_off = iw * c_tile;
        for co in 0..c_tile {
            let oc = oc_start + co;
            dst_row[dst_off + co] = pw_bias.map(|b| b[oc]).unwrap_or(0.0);
        }
        for ci in 0..c_in {
            let x = src_row[src_off + ci];
            let w_row = &pw_weight[ci * c_exp + oc_start..ci * c_exp + oc_start + c_tile];
            for co in 0..c_tile {
                dst_row[dst_off + co] += x * w_row[co];
            }
        }
        if relu {
            for v in &mut dst_row[dst_off..dst_off + c_tile] {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
        }
    }
}

pub(super) fn compute_dw5_tile_scalar(ctx: Dw5TileCtx<'_, '_>) {
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
    let w_ky_stride = 5 * c_exp;
    let w_kx_stride = c_exp;
    for ow in ow_start..ow_end {
        let iw0 = (ow as i32) * (stride as i32) - (pad as i32);
        let dst_off = ow * c_exp + oc_start;
        for co in 0..c_tile {
            let oc = oc_start + co;
            out_row[dst_off + co] = dw_bias.map(|b| b[oc]).unwrap_or(0.0);
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
                let pw_off = iw as usize * c_tile;
                let w_off = ky * w_ky_stride + kx * w_kx_stride + oc_start;
                for co in 0..c_tile {
                    out_row[dst_off + co] += pw_row[pw_off + co] * dw_weight[w_off + co];
                }
            }
        }
        if relu {
            for v in &mut out_row[dst_off..dst_off + c_tile] {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
        }
    }
}
