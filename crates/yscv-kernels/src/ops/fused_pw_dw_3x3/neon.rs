//! aarch64 NEON fused PW-expand→DW kernels.

use super::{Dw5RowCtx, Dw5TileCtx, PwTileCtx};
use std::arch::aarch64::*;
const NEON_TILE_CHUNKS: usize = 8;

#[inline]
fn pw_2x_disabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // Default ON. This env is only a kill-switch for A/B and quick rollback.
    *CACHED.get_or_init(|| std::env::var_os("YSCV_FUSED_PW_DW_PW2X_OFF").is_some())
}

/// NEON PW row: C_exp vectorised in q-reg groups of 4.
pub(super) fn compute_pw_row_neon(
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
    // SAFETY: NEON is mandatory on aarch64 (ARMv8), caller verified
    // c_exp % 4 == 0 via select_variant.
    #[allow(unsafe_code)]
    unsafe {
        compute_pw_row_neon_inner(
            src_row, pw_weight, pw_bias, dst_row, iw_start, iw_end, c_in, c_exp, relu,
        )
    }
}

#[target_feature(enable = "neon")]
#[allow(unsafe_code, clippy::too_many_arguments)]
unsafe fn compute_pw_row_neon_inner(
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
        let zero = vdupq_n_f32(0.0);
        let c_exp_chunks = c_exp / 4;
        let full_tiles = c_exp_chunks / NEON_TILE_CHUNKS;
        let residual_chunks = c_exp_chunks % NEON_TILE_CHUNKS;
        let use_2x = !pw_2x_disabled();
        let mut iw = iw_start;

        // 2-column PW block: reuse the same PW weights for two adjacent
        // output columns to halve weight-stream pressure in the hot loop.
        if use_2x {
            while iw + 1 < iw_end {
                let src_off0 = iw * c_in;
                let src_off1 = (iw + 1) * c_in;
                let dst_ptr0 = dst_row.as_mut_ptr().add(iw * c_exp);
                let dst_ptr1 = dst_row.as_mut_ptr().add((iw + 1) * c_exp);

                for tile in 0..full_tiles {
                    let lane_off = tile * NEON_TILE_CHUNKS * 4;
                    let mut acc0: [float32x4_t; NEON_TILE_CHUNKS] = [zero; NEON_TILE_CHUNKS];
                    let mut acc1: [float32x4_t; NEON_TILE_CHUNKS] = [zero; NEON_TILE_CHUNKS];

                    if let Some(b) = pw_bias {
                        let bp = b.as_ptr().add(lane_off);
                        for k in 0..NEON_TILE_CHUNKS {
                            let vb = vld1q_f32(bp.add(k * 4));
                            acc0[k] = vb;
                            acc1[k] = vb;
                        }
                    }

                    for ci in 0..c_in {
                        let x0 = vdupq_n_f32(*src_row.as_ptr().add(src_off0 + ci));
                        let x1 = vdupq_n_f32(*src_row.as_ptr().add(src_off1 + ci));
                        let w_base = pw_weight.as_ptr().add(ci * c_exp + lane_off);
                        for k in 0..NEON_TILE_CHUNKS {
                            let w = vld1q_f32(w_base.add(k * 4));
                            acc0[k] = vfmaq_f32(acc0[k], x0, w);
                            acc1[k] = vfmaq_f32(acc1[k], x1, w);
                        }
                    }

                    if relu {
                        for k in 0..NEON_TILE_CHUNKS {
                            acc0[k] = vmaxq_f32(acc0[k], zero);
                            acc1[k] = vmaxq_f32(acc1[k], zero);
                        }
                    }
                    for k in 0..NEON_TILE_CHUNKS {
                        let off = lane_off + k * 4;
                        vst1q_f32(dst_ptr0.add(off), acc0[k]);
                        vst1q_f32(dst_ptr1.add(off), acc1[k]);
                    }
                }

                if residual_chunks > 0 {
                    let lane_off = full_tiles * NEON_TILE_CHUNKS * 4;
                    let mut acc0: [float32x4_t; NEON_TILE_CHUNKS] = [zero; NEON_TILE_CHUNKS];
                    let mut acc1: [float32x4_t; NEON_TILE_CHUNKS] = [zero; NEON_TILE_CHUNKS];

                    if let Some(b) = pw_bias {
                        let bp = b.as_ptr().add(lane_off);
                        for k in 0..residual_chunks {
                            let vb = vld1q_f32(bp.add(k * 4));
                            acc0[k] = vb;
                            acc1[k] = vb;
                        }
                    }

                    for ci in 0..c_in {
                        let x0 = vdupq_n_f32(*src_row.as_ptr().add(src_off0 + ci));
                        let x1 = vdupq_n_f32(*src_row.as_ptr().add(src_off1 + ci));
                        let w_base = pw_weight.as_ptr().add(ci * c_exp + lane_off);
                        for k in 0..residual_chunks {
                            let w = vld1q_f32(w_base.add(k * 4));
                            acc0[k] = vfmaq_f32(acc0[k], x0, w);
                            acc1[k] = vfmaq_f32(acc1[k], x1, w);
                        }
                    }

                    if relu {
                        for k in 0..residual_chunks {
                            acc0[k] = vmaxq_f32(acc0[k], zero);
                            acc1[k] = vmaxq_f32(acc1[k], zero);
                        }
                    }
                    for k in 0..residual_chunks {
                        let off = lane_off + k * 4;
                        vst1q_f32(dst_ptr0.add(off), acc0[k]);
                        vst1q_f32(dst_ptr1.add(off), acc1[k]);
                    }
                }
                iw += 2;
            }
        }

        while iw < iw_end {
            let src_off = iw * c_in;
            let dst_off = iw * c_exp;
            let dst_ptr = dst_row.as_mut_ptr().add(dst_off);

            for tile in 0..full_tiles {
                let lane_off = tile * NEON_TILE_CHUNKS * 4;
                let mut acc: [float32x4_t; NEON_TILE_CHUNKS] = [zero; NEON_TILE_CHUNKS];

                if let Some(b) = pw_bias {
                    let bp = b.as_ptr().add(lane_off);
                    for k in 0..NEON_TILE_CHUNKS {
                        acc[k] = vld1q_f32(bp.add(k * 4));
                    }
                }

                for ci in 0..c_in {
                    let x = vdupq_n_f32(*src_row.as_ptr().add(src_off + ci));
                    let w_base = pw_weight.as_ptr().add(ci * c_exp + lane_off);
                    for k in 0..NEON_TILE_CHUNKS {
                        let w = vld1q_f32(w_base.add(k * 4));
                        acc[k] = vfmaq_f32(acc[k], x, w);
                    }
                }

                if relu {
                    for k in 0..NEON_TILE_CHUNKS {
                        acc[k] = vmaxq_f32(acc[k], zero);
                    }
                }
                for k in 0..NEON_TILE_CHUNKS {
                    vst1q_f32(dst_ptr.add(lane_off + k * 4), acc[k]);
                }
            }

            if residual_chunks > 0 {
                let lane_off = full_tiles * NEON_TILE_CHUNKS * 4;
                let mut acc: [float32x4_t; NEON_TILE_CHUNKS] = [zero; NEON_TILE_CHUNKS];

                if let Some(b) = pw_bias {
                    let bp = b.as_ptr().add(lane_off);
                    for k in 0..residual_chunks {
                        acc[k] = vld1q_f32(bp.add(k * 4));
                    }
                }

                for ci in 0..c_in {
                    let x = vdupq_n_f32(*src_row.as_ptr().add(src_off + ci));
                    let w_base = pw_weight.as_ptr().add(ci * c_exp + lane_off);
                    for k in 0..residual_chunks {
                        let w = vld1q_f32(w_base.add(k * 4));
                        acc[k] = vfmaq_f32(acc[k], x, w);
                    }
                }

                if relu {
                    for k in 0..residual_chunks {
                        acc[k] = vmaxq_f32(acc[k], zero);
                    }
                }
                for k in 0..residual_chunks {
                    vst1q_f32(dst_ptr.add(lane_off + k * 4), acc[k]);
                }
            }
            iw += 1;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn compute_dw_row_neon(
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
    #[allow(unsafe_code)]
    unsafe {
        compute_dw_row_neon_inner(
            row0, row1, row2, dw_weight, dw_bias, out_row, in_w, ow_start, ow_end, c_exp, stride,
            pad, relu,
        )
    }
}

#[target_feature(enable = "neon")]
#[allow(unsafe_code, clippy::too_many_arguments)]
unsafe fn compute_dw_row_neon_inner(
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
        let zero = vdupq_n_f32(0.0);
        let c_exp_chunks = c_exp / 4;
        let full_tiles = c_exp_chunks / NEON_TILE_CHUNKS;
        let residual_chunks = c_exp_chunks % NEON_TILE_CHUNKS;
        let w_ky_stride = 3 * c_exp;
        let w_kx_stride = c_exp;
        let rows = [row0, row1, row2];

        for ow in ow_start..ow_end {
            let iw0 = (ow as i32) * (stride as i32) - (pad as i32);
            let dst_off = ow * c_exp;
            let dst_ptr = out_row.as_mut_ptr().add(dst_off);

            for tile in 0..full_tiles {
                let lane_off = tile * NEON_TILE_CHUNKS * 4;
                let mut acc: [float32x4_t; NEON_TILE_CHUNKS] = [zero; NEON_TILE_CHUNKS];

                if let Some(b) = dw_bias {
                    let bp = b.as_ptr().add(lane_off);
                    for k in 0..NEON_TILE_CHUNKS {
                        acc[k] = vld1q_f32(bp.add(k * 4));
                    }
                }

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
                        let pw_ptr = pw_row.as_ptr().add(iw_u * c_exp + lane_off);
                        let w_ptr = dw_weight
                            .as_ptr()
                            .add(ky * w_ky_stride + kx * w_kx_stride + lane_off);
                        for k in 0..NEON_TILE_CHUNKS {
                            let x = vld1q_f32(pw_ptr.add(k * 4));
                            let w = vld1q_f32(w_ptr.add(k * 4));
                            acc[k] = vfmaq_f32(acc[k], x, w);
                        }
                    }
                }

                if relu {
                    for k in 0..NEON_TILE_CHUNKS {
                        acc[k] = vmaxq_f32(acc[k], zero);
                    }
                }
                for k in 0..NEON_TILE_CHUNKS {
                    vst1q_f32(dst_ptr.add(lane_off + k * 4), acc[k]);
                }
            }

            if residual_chunks > 0 {
                let lane_off = full_tiles * NEON_TILE_CHUNKS * 4;
                let mut acc: [float32x4_t; NEON_TILE_CHUNKS] = [zero; NEON_TILE_CHUNKS];

                if let Some(b) = dw_bias {
                    let bp = b.as_ptr().add(lane_off);
                    for k in 0..residual_chunks {
                        acc[k] = vld1q_f32(bp.add(k * 4));
                    }
                }

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
                        let pw_ptr = pw_row.as_ptr().add(iw_u * c_exp + lane_off);
                        let w_ptr = dw_weight
                            .as_ptr()
                            .add(ky * w_ky_stride + kx * w_kx_stride + lane_off);
                        for k in 0..residual_chunks {
                            let x = vld1q_f32(pw_ptr.add(k * 4));
                            let w = vld1q_f32(w_ptr.add(k * 4));
                            acc[k] = vfmaq_f32(acc[k], x, w);
                        }
                    }
                }

                if relu {
                    for k in 0..residual_chunks {
                        acc[k] = vmaxq_f32(acc[k], zero);
                    }
                }
                for k in 0..residual_chunks {
                    vst1q_f32(dst_ptr.add(lane_off + k * 4), acc[k]);
                }
            }
        }
    }
}

pub(super) fn compute_dw5_row_neon(ctx: Dw5RowCtx<'_, '_>) {
    #[allow(unsafe_code)]
    unsafe {
        compute_dw5_row_neon_inner(ctx)
    }
}

#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
unsafe fn compute_dw5_row_neon_inner(ctx: Dw5RowCtx<'_, '_>) {
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
        let zero = vdupq_n_f32(0.0);
        let c_exp_chunks = c_exp / 4;
        let full_tiles = c_exp_chunks / NEON_TILE_CHUNKS;
        let residual_chunks = c_exp_chunks % NEON_TILE_CHUNKS;
        let w_ky_stride = 5 * c_exp;
        let w_kx_stride = c_exp;

        for ow in ow_start..ow_end {
            let iw0 = (ow as i32) * (stride as i32) - (pad as i32);
            let dst_ptr = out_row.as_mut_ptr().add(ow * c_exp);

            for tile in 0..full_tiles {
                let lane_off = tile * NEON_TILE_CHUNKS * 4;
                let mut acc: [float32x4_t; NEON_TILE_CHUNKS] = [zero; NEON_TILE_CHUNKS];
                if let Some(b) = dw_bias {
                    let bp = b.as_ptr().add(lane_off);
                    for k in 0..NEON_TILE_CHUNKS {
                        acc[k] = vld1q_f32(bp.add(k * 4));
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
                        let pw_ptr = pw_row.as_ptr().add(iw as usize * c_exp + lane_off);
                        let w_ptr = dw_weight
                            .as_ptr()
                            .add(ky * w_ky_stride + kx * w_kx_stride + lane_off);
                        for k in 0..NEON_TILE_CHUNKS {
                            let x = vld1q_f32(pw_ptr.add(k * 4));
                            let w = vld1q_f32(w_ptr.add(k * 4));
                            acc[k] = vfmaq_f32(acc[k], x, w);
                        }
                    }
                }

                if relu {
                    for k in 0..NEON_TILE_CHUNKS {
                        acc[k] = vmaxq_f32(acc[k], zero);
                    }
                }
                for k in 0..NEON_TILE_CHUNKS {
                    vst1q_f32(dst_ptr.add(lane_off + k * 4), acc[k]);
                }
            }

            if residual_chunks > 0 {
                let lane_off = full_tiles * NEON_TILE_CHUNKS * 4;
                let mut acc: [float32x4_t; NEON_TILE_CHUNKS] = [zero; NEON_TILE_CHUNKS];
                if let Some(b) = dw_bias {
                    let bp = b.as_ptr().add(lane_off);
                    for k in 0..residual_chunks {
                        acc[k] = vld1q_f32(bp.add(k * 4));
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
                        let pw_ptr = pw_row.as_ptr().add(iw as usize * c_exp + lane_off);
                        let w_ptr = dw_weight
                            .as_ptr()
                            .add(ky * w_ky_stride + kx * w_kx_stride + lane_off);
                        for k in 0..residual_chunks {
                            let x = vld1q_f32(pw_ptr.add(k * 4));
                            let w = vld1q_f32(w_ptr.add(k * 4));
                            acc[k] = vfmaq_f32(acc[k], x, w);
                        }
                    }
                }

                if relu {
                    for k in 0..residual_chunks {
                        acc[k] = vmaxq_f32(acc[k], zero);
                    }
                }
                for k in 0..residual_chunks {
                    vst1q_f32(dst_ptr.add(lane_off + k * 4), acc[k]);
                }
            }
        }
    }
}

pub(super) fn compute_pw_tile_neon(ctx: PwTileCtx<'_, '_>) {
    #[allow(unsafe_code)]
    unsafe {
        compute_pw_tile_neon_inner(ctx)
    }
}

#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
unsafe fn compute_pw_tile_neon_inner(ctx: PwTileCtx<'_, '_>) {
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
        let zero = vdupq_n_f32(0.0);
        let chunks = c_tile / 4;
        let full_tiles = chunks / NEON_TILE_CHUNKS;
        let residual_chunks = chunks % NEON_TILE_CHUNKS;
        for iw in iw_start..iw_end {
            let dst_ptr = dst_row.as_mut_ptr().add(iw * c_tile);
            for tile in 0..full_tiles {
                let tile_off = tile * NEON_TILE_CHUNKS * 4;
                let mut acc: [float32x4_t; NEON_TILE_CHUNKS] = [zero; NEON_TILE_CHUNKS];
                if let Some(b) = pw_bias {
                    let bp = b.as_ptr().add(oc_start + tile_off);
                    for k in 0..NEON_TILE_CHUNKS {
                        acc[k] = vld1q_f32(bp.add(k * 4));
                    }
                }
                for ci in 0..c_in {
                    let x = vdupq_n_f32(*src_row.as_ptr().add(iw * c_in + ci));
                    let wp = pw_weight.as_ptr().add(ci * c_exp + oc_start + tile_off);
                    for k in 0..NEON_TILE_CHUNKS {
                        let w = vld1q_f32(wp.add(k * 4));
                        acc[k] = vfmaq_f32(acc[k], x, w);
                    }
                }
                if relu {
                    for k in 0..NEON_TILE_CHUNKS {
                        acc[k] = vmaxq_f32(acc[k], zero);
                    }
                }
                for k in 0..NEON_TILE_CHUNKS {
                    vst1q_f32(dst_ptr.add(tile_off + k * 4), acc[k]);
                }
            }

            if residual_chunks > 0 {
                let tile_off = full_tiles * NEON_TILE_CHUNKS * 4;
                let mut acc: [float32x4_t; NEON_TILE_CHUNKS] = [zero; NEON_TILE_CHUNKS];
                if let Some(b) = pw_bias {
                    let bp = b.as_ptr().add(oc_start + tile_off);
                    for k in 0..residual_chunks {
                        acc[k] = vld1q_f32(bp.add(k * 4));
                    }
                }
                for ci in 0..c_in {
                    let x = vdupq_n_f32(*src_row.as_ptr().add(iw * c_in + ci));
                    let wp = pw_weight.as_ptr().add(ci * c_exp + oc_start + tile_off);
                    for k in 0..residual_chunks {
                        let w = vld1q_f32(wp.add(k * 4));
                        acc[k] = vfmaq_f32(acc[k], x, w);
                    }
                }
                if relu {
                    for k in 0..residual_chunks {
                        acc[k] = vmaxq_f32(acc[k], zero);
                    }
                }
                for k in 0..residual_chunks {
                    vst1q_f32(dst_ptr.add(tile_off + k * 4), acc[k]);
                }
            }
        }
    }
}

pub(super) fn compute_dw5_tile_neon(ctx: Dw5TileCtx<'_, '_>) {
    #[allow(unsafe_code)]
    unsafe {
        compute_dw5_tile_neon_inner(ctx)
    }
}

#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
unsafe fn compute_dw5_tile_neon_inner(ctx: Dw5TileCtx<'_, '_>) {
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
        let zero = vdupq_n_f32(0.0);
        let chunks = c_tile / 4;
        let full_tiles = chunks / NEON_TILE_CHUNKS;
        let residual_chunks = chunks % NEON_TILE_CHUNKS;
        let w_ky_stride = 5 * c_exp;
        let w_kx_stride = c_exp;
        for ow in ow_start..ow_end {
            let iw0 = (ow as i32) * (stride as i32) - (pad as i32);
            let dst_ptr = out_row.as_mut_ptr().add(ow * c_exp + oc_start);
            for tile in 0..full_tiles {
                let tile_off = tile * NEON_TILE_CHUNKS * 4;
                let mut acc: [float32x4_t; NEON_TILE_CHUNKS] = [zero; NEON_TILE_CHUNKS];
                if let Some(b) = dw_bias {
                    let bp = b.as_ptr().add(oc_start + tile_off);
                    for k in 0..NEON_TILE_CHUNKS {
                        acc[k] = vld1q_f32(bp.add(k * 4));
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
                        let pw_ptr = pw_row.as_ptr().add(iw as usize * c_tile + tile_off);
                        let w_ptr = dw_weight
                            .as_ptr()
                            .add(ky * w_ky_stride + kx * w_kx_stride + oc_start + tile_off);
                        for k in 0..NEON_TILE_CHUNKS {
                            let x = vld1q_f32(pw_ptr.add(k * 4));
                            let w = vld1q_f32(w_ptr.add(k * 4));
                            acc[k] = vfmaq_f32(acc[k], x, w);
                        }
                    }
                }
                if relu {
                    for k in 0..NEON_TILE_CHUNKS {
                        acc[k] = vmaxq_f32(acc[k], zero);
                    }
                }
                for k in 0..NEON_TILE_CHUNKS {
                    vst1q_f32(dst_ptr.add(tile_off + k * 4), acc[k]);
                }
            }

            if residual_chunks > 0 {
                let tile_off = full_tiles * NEON_TILE_CHUNKS * 4;
                let mut acc: [float32x4_t; NEON_TILE_CHUNKS] = [zero; NEON_TILE_CHUNKS];
                if let Some(b) = dw_bias {
                    let bp = b.as_ptr().add(oc_start + tile_off);
                    for k in 0..residual_chunks {
                        acc[k] = vld1q_f32(bp.add(k * 4));
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
                        let pw_ptr = pw_row.as_ptr().add(iw as usize * c_tile + tile_off);
                        let w_ptr = dw_weight
                            .as_ptr()
                            .add(ky * w_ky_stride + kx * w_kx_stride + oc_start + tile_off);
                        for k in 0..residual_chunks {
                            let x = vld1q_f32(pw_ptr.add(k * 4));
                            let w = vld1q_f32(w_ptr.add(k * 4));
                            acc[k] = vfmaq_f32(acc[k], x, w);
                        }
                    }
                }
                if relu {
                    for k in 0..residual_chunks {
                        acc[k] = vmaxq_f32(acc[k], zero);
                    }
                }
                for k in 0..residual_chunks {
                    vst1q_f32(dst_ptr.add(tile_off + k * 4), acc[k]);
                }
            }
        }
    }
}
