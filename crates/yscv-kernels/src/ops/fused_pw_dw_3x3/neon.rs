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

#[inline]
fn pw_gemm_disabled() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var_os("YSCV_PW_GEMM_OFF").is_some())
}

#[inline]
fn dw5_asm_disabled() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var_os("YSCV_DW5_ASM_OFF").is_some())
}

#[inline]
fn pw_gemm_max_threads() -> usize {
    // Default 4: the pipelined 8×8 GEMM (cached B-pack) now wins at 4T too
    // (−6.5 ms on the A53), unlike the old 4×16 it replaced. Bit-identical to
    // the broadcast path (same in-order FMA k-sum). Env override for other core
    // counts.
    static C: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *C.get_or_init(|| {
        std::env::var("YSCV_PW_GEMM_MAX_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4)
    })
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
    // The streaming broadcast PW runs ~2.4× below the blocked GEMM on the same
    // [in_w, c_in]×[c_in, c_exp] shape — route the PW row through the GEMM
    // (sequential, bias+Relu folded into the epilogue). Gated by thread count:
    // historically the sequential GEMM's per-call A-pack alloc lost to the
    // alloc-free broadcast at 4T (`YSCV_PW_GEMM_MAX_THREADS` tunes the cutoff).
    if !pw_gemm_disabled()
        && iw_end > iw_start
        && rayon::current_num_threads() <= pw_gemm_max_threads()
    {
        let m = iw_end - iw_start;
        let a = &src_row[iw_start * c_in..iw_end * c_in];
        let out = &mut dst_row[iw_start * c_exp..iw_end * c_exp];
        let act = if relu {
            super::super::conv::Activation::Relu
        } else {
            super::super::conv::Activation::None
        };
        let ep = super::super::matmul::GemmEpilogue::new(pw_bias.map(|b| b.as_ptr()), act);
        super::super::matmul::matmul_2d_slices_blocked_fused(a, m, c_in, pw_weight, c_exp, out, ep);
        return;
    }
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

        // Column-reuse asm interior path: stride 2, all 3 ring rows present.
        let interior_rows: Option<[*const f32; 3]> = if stride == 2 && !dw5_asm_disabled() {
            let mut ps = [core::ptr::null::<f32>(); 3];
            let mut ok = true;
            for (k, r) in rows.iter().enumerate() {
                match r {
                    Some(s) => ps[k] = s.as_ptr(),
                    None => ok = false,
                }
            }
            if ok { Some(ps) } else { None }
        } else {
            None
        };

        // 4-wide spatial in 16-channel blocks: one weight quad feeds 4 output
        // columns (16 accumulators), cutting the per-column weight reload. Same
        // technique as the 5×5 path; bit-identical FMA order per accumulator.
        let mut owi = ow_start;
        if c_exp.is_multiple_of(16) {
            while owi + 4 <= ow_end {
                // Interior 4-column stride-2 tile (all 3×3 taps in bounds):
                // hand-asm column-reuse kernel, 8 channels per call.
                if let Some(rp) = interior_rows
                    && owi * 2 >= pad
                    && (owi + 3) * 2 + 2 < in_w + pad
                {
                    let relu_u = relu as usize;
                    let wbytes = c_exp * 4;
                    let wbase = dw_weight.as_ptr();
                    let obase = out_row.as_mut_ptr();
                    let base_col = owi * 2 - pad;
                    let mut ch = 0usize;
                    while ch < c_exp {
                        let roff: [*const f32; 3] =
                            std::array::from_fn(|k| rp[k].add(base_col * c_exp + ch));
                        let biasp = dw_bias
                            .map(|b| b.as_ptr().add(ch))
                            .unwrap_or(core::ptr::null());
                        super::super::matmul::dw3s2_creuse_neon(
                            roff.as_ptr(),
                            wbase.add(ch),
                            obase.add(owi * c_exp + ch),
                            c_exp,
                            biasp,
                            relu_u,
                            wbytes,
                        );
                        ch += 8;
                    }
                    owi += 4;
                    continue;
                }
                let iw0a = (owi as i32) * (stride as i32) - (pad as i32);
                let iw0b = ((owi + 1) as i32) * (stride as i32) - (pad as i32);
                let iw0c = ((owi + 2) as i32) * (stride as i32) - (pad as i32);
                let iw0d = ((owi + 3) as i32) * (stride as i32) - (pad as i32);
                let mut ch = 0usize;
                while ch < c_exp {
                    let (bb0, bb1, bb2, bb3) = if let Some(b) = dw_bias {
                        let bp = b.as_ptr().add(ch);
                        (
                            vld1q_f32(bp),
                            vld1q_f32(bp.add(4)),
                            vld1q_f32(bp.add(8)),
                            vld1q_f32(bp.add(12)),
                        )
                    } else {
                        (zero, zero, zero, zero)
                    };
                    let (mut a00, mut a01, mut a02, mut a03) = (bb0, bb1, bb2, bb3);
                    let (mut a10, mut a11, mut a12, mut a13) = (bb0, bb1, bb2, bb3);
                    let (mut a20, mut a21, mut a22, mut a23) = (bb0, bb1, bb2, bb3);
                    let (mut a30, mut a31, mut a32, mut a33) = (bb0, bb1, bb2, bb3);
                    for ky in 0..3usize {
                        let pwp = match rows[ky] {
                            Some(r) => r.as_ptr(),
                            None => continue,
                        };
                        for kx in 0..3usize {
                            let wp = dw_weight
                                .as_ptr()
                                .add(ky * w_ky_stride + kx * w_kx_stride + ch);
                            let w0 = vld1q_f32(wp);
                            let w1 = vld1q_f32(wp.add(4));
                            let w2 = vld1q_f32(wp.add(8));
                            let w3 = vld1q_f32(wp.add(12));
                            let iwa = iw0a + kx as i32;
                            if iwa >= 0 && (iwa as usize) < in_w {
                                let p = pwp.add(iwa as usize * c_exp + ch);
                                a00 = vfmaq_f32(a00, vld1q_f32(p), w0);
                                a01 = vfmaq_f32(a01, vld1q_f32(p.add(4)), w1);
                                a02 = vfmaq_f32(a02, vld1q_f32(p.add(8)), w2);
                                a03 = vfmaq_f32(a03, vld1q_f32(p.add(12)), w3);
                            }
                            let iwb = iw0b + kx as i32;
                            if iwb >= 0 && (iwb as usize) < in_w {
                                let p = pwp.add(iwb as usize * c_exp + ch);
                                a10 = vfmaq_f32(a10, vld1q_f32(p), w0);
                                a11 = vfmaq_f32(a11, vld1q_f32(p.add(4)), w1);
                                a12 = vfmaq_f32(a12, vld1q_f32(p.add(8)), w2);
                                a13 = vfmaq_f32(a13, vld1q_f32(p.add(12)), w3);
                            }
                            let iwc = iw0c + kx as i32;
                            if iwc >= 0 && (iwc as usize) < in_w {
                                let p = pwp.add(iwc as usize * c_exp + ch);
                                a20 = vfmaq_f32(a20, vld1q_f32(p), w0);
                                a21 = vfmaq_f32(a21, vld1q_f32(p.add(4)), w1);
                                a22 = vfmaq_f32(a22, vld1q_f32(p.add(8)), w2);
                                a23 = vfmaq_f32(a23, vld1q_f32(p.add(12)), w3);
                            }
                            let iwd = iw0d + kx as i32;
                            if iwd >= 0 && (iwd as usize) < in_w {
                                let p = pwp.add(iwd as usize * c_exp + ch);
                                a30 = vfmaq_f32(a30, vld1q_f32(p), w0);
                                a31 = vfmaq_f32(a31, vld1q_f32(p.add(4)), w1);
                                a32 = vfmaq_f32(a32, vld1q_f32(p.add(8)), w2);
                                a33 = vfmaq_f32(a33, vld1q_f32(p.add(12)), w3);
                            }
                        }
                    }
                    if relu {
                        a00 = vmaxq_f32(a00, zero);
                        a01 = vmaxq_f32(a01, zero);
                        a02 = vmaxq_f32(a02, zero);
                        a03 = vmaxq_f32(a03, zero);
                        a10 = vmaxq_f32(a10, zero);
                        a11 = vmaxq_f32(a11, zero);
                        a12 = vmaxq_f32(a12, zero);
                        a13 = vmaxq_f32(a13, zero);
                        a20 = vmaxq_f32(a20, zero);
                        a21 = vmaxq_f32(a21, zero);
                        a22 = vmaxq_f32(a22, zero);
                        a23 = vmaxq_f32(a23, zero);
                        a30 = vmaxq_f32(a30, zero);
                        a31 = vmaxq_f32(a31, zero);
                        a32 = vmaxq_f32(a32, zero);
                        a33 = vmaxq_f32(a33, zero);
                    }
                    let d = out_row.as_mut_ptr();
                    let d0 = d.add(owi * c_exp + ch);
                    vst1q_f32(d0, a00);
                    vst1q_f32(d0.add(4), a01);
                    vst1q_f32(d0.add(8), a02);
                    vst1q_f32(d0.add(12), a03);
                    let d1 = d.add((owi + 1) * c_exp + ch);
                    vst1q_f32(d1, a10);
                    vst1q_f32(d1.add(4), a11);
                    vst1q_f32(d1.add(8), a12);
                    vst1q_f32(d1.add(12), a13);
                    let d2 = d.add((owi + 2) * c_exp + ch);
                    vst1q_f32(d2, a20);
                    vst1q_f32(d2.add(4), a21);
                    vst1q_f32(d2.add(8), a22);
                    vst1q_f32(d2.add(12), a23);
                    let d3 = d.add((owi + 3) * c_exp + ch);
                    vst1q_f32(d3, a30);
                    vst1q_f32(d3.add(4), a31);
                    vst1q_f32(d3.add(8), a32);
                    vst1q_f32(d3.add(12), a33);
                    ch += 16;
                }
                owi += 4;
            }
        }

        for ow in owi..ow_end {
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

/// Shared read-only zero row for the DW vertical-padding (border) rows, so the
/// column-reuse asm can run on the top/bottom rows. Sized once to cover the
/// tracker's largest DW input row; larger shapes return `None` (fall back to the
/// per-tap intrinsic, which is correctness-equivalent).
fn dw5_zero_row(min_len: usize) -> Option<*const f32> {
    use std::sync::OnceLock;
    const ZERO_LEN: usize = 256 * 1024;
    static ZERO: OnceLock<Vec<f32>> = OnceLock::new();
    if min_len > ZERO_LEN {
        return None;
    }
    Some(ZERO.get_or_init(|| vec![0.0f32; ZERO_LEN]).as_ptr())
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

        // 4-wide spatial: process 4 output columns per pass over the 25 taps in
        // 16-channel (4 q-reg) blocks. One weight quad feeds all 4 columns,
        // cutting the per-column weight reload and the 2:1 load:FMA toward
        // 1.25:1. 16 accumulators + 4 weight + 4 input fit the 32 q-regs (the
        // 8-chunk tile can't hold 4 columns). FMA order per accumulator is
        // unchanged, so bit-identical to the 1-wide path. Channel tail / odd
        // columns fall to the per-column loop below.
        //
        // Column-reuse asm interior path needs all 5 ring rows present (no
        // vertical padding) and unit stride; collect the row bases once.
        // Vertical-padding rows (top/bottom border) point at a shared zero row so
        // the column-reuse asm runs on the border too (input 0 → contributes 0,
        // bit-identical to the per-tap path that skips a `None` row). Without it,
        // ~44% of a 16×16 (xif4) DW falls to the slower per-tap intrinsic.
        let zrow = dw5_zero_row(in_w * c_exp);
        let interior_rows: Option<[*const f32; 5]> = if stride == 1 && !dw5_asm_disabled() {
            let mut ps = [core::ptr::null::<f32>(); 5];
            let mut ok = true;
            for (k, r) in rows.iter().enumerate() {
                match r {
                    Some(s) => ps[k] = s.as_ptr(),
                    None => match zrow {
                        Some(z) => ps[k] = z,
                        None => ok = false,
                    },
                }
            }
            if ok { Some(ps) } else { None }
        } else {
            None
        };

        let mut owi = ow_start;
        if c_exp.is_multiple_of(16) {
            while owi + 4 <= ow_end {
                // Interior 4-column tile (all 5×5 taps in bounds): hand-asm
                // column-reuse kernel, 8 channels per call. Borders / padded
                // rows fall through to the per-tap-bounds intrinsic below.
                if let Some(rp) = interior_rows
                    && owi >= pad
                    && owi + 8 <= in_w + pad
                {
                    let relu_u = relu as usize;
                    let wbytes = c_exp * 4;
                    let wbase = dw_weight.as_ptr();
                    let obase = out_row.as_mut_ptr();
                    let mut ch = 0usize;
                    while ch < c_exp {
                        let roff: [*const f32; 5] =
                            std::array::from_fn(|k| rp[k].add((owi - pad) * c_exp + ch));
                        let biasp = dw_bias
                            .map(|b| b.as_ptr().add(ch))
                            .unwrap_or(core::ptr::null());
                        super::super::matmul::dw5_creuse_neon(
                            roff.as_ptr(),
                            wbase.add(ch),
                            obase.add(owi * c_exp + ch),
                            c_exp,
                            biasp,
                            relu_u,
                            wbytes,
                        );
                        ch += 8;
                    }
                    owi += 4;
                    continue;
                }
                let iw0a = (owi as i32) * (stride as i32) - (pad as i32);
                let iw0b = ((owi + 1) as i32) * (stride as i32) - (pad as i32);
                let iw0c = ((owi + 2) as i32) * (stride as i32) - (pad as i32);
                let iw0d = ((owi + 3) as i32) * (stride as i32) - (pad as i32);
                let mut ch = 0usize;
                while ch < c_exp {
                    let (bb0, bb1, bb2, bb3) = if let Some(b) = dw_bias {
                        let bp = b.as_ptr().add(ch);
                        (
                            vld1q_f32(bp),
                            vld1q_f32(bp.add(4)),
                            vld1q_f32(bp.add(8)),
                            vld1q_f32(bp.add(12)),
                        )
                    } else {
                        (zero, zero, zero, zero)
                    };
                    let (mut a00, mut a01, mut a02, mut a03) = (bb0, bb1, bb2, bb3);
                    let (mut a10, mut a11, mut a12, mut a13) = (bb0, bb1, bb2, bb3);
                    let (mut a20, mut a21, mut a22, mut a23) = (bb0, bb1, bb2, bb3);
                    let (mut a30, mut a31, mut a32, mut a33) = (bb0, bb1, bb2, bb3);
                    for (ky, row) in rows.iter().enumerate() {
                        let pwp = match row {
                            Some(r) => r.as_ptr(),
                            None => continue,
                        };
                        for kx in 0..5usize {
                            let wp = dw_weight
                                .as_ptr()
                                .add(ky * w_ky_stride + kx * w_kx_stride + ch);
                            let w0 = vld1q_f32(wp);
                            let w1 = vld1q_f32(wp.add(4));
                            let w2 = vld1q_f32(wp.add(8));
                            let w3 = vld1q_f32(wp.add(12));
                            let iwa = iw0a + kx as i32;
                            if iwa >= 0 && (iwa as usize) < in_w {
                                let p = pwp.add(iwa as usize * c_exp + ch);
                                a00 = vfmaq_f32(a00, vld1q_f32(p), w0);
                                a01 = vfmaq_f32(a01, vld1q_f32(p.add(4)), w1);
                                a02 = vfmaq_f32(a02, vld1q_f32(p.add(8)), w2);
                                a03 = vfmaq_f32(a03, vld1q_f32(p.add(12)), w3);
                            }
                            let iwb = iw0b + kx as i32;
                            if iwb >= 0 && (iwb as usize) < in_w {
                                let p = pwp.add(iwb as usize * c_exp + ch);
                                a10 = vfmaq_f32(a10, vld1q_f32(p), w0);
                                a11 = vfmaq_f32(a11, vld1q_f32(p.add(4)), w1);
                                a12 = vfmaq_f32(a12, vld1q_f32(p.add(8)), w2);
                                a13 = vfmaq_f32(a13, vld1q_f32(p.add(12)), w3);
                            }
                            let iwc = iw0c + kx as i32;
                            if iwc >= 0 && (iwc as usize) < in_w {
                                let p = pwp.add(iwc as usize * c_exp + ch);
                                a20 = vfmaq_f32(a20, vld1q_f32(p), w0);
                                a21 = vfmaq_f32(a21, vld1q_f32(p.add(4)), w1);
                                a22 = vfmaq_f32(a22, vld1q_f32(p.add(8)), w2);
                                a23 = vfmaq_f32(a23, vld1q_f32(p.add(12)), w3);
                            }
                            let iwd = iw0d + kx as i32;
                            if iwd >= 0 && (iwd as usize) < in_w {
                                let p = pwp.add(iwd as usize * c_exp + ch);
                                a30 = vfmaq_f32(a30, vld1q_f32(p), w0);
                                a31 = vfmaq_f32(a31, vld1q_f32(p.add(4)), w1);
                                a32 = vfmaq_f32(a32, vld1q_f32(p.add(8)), w2);
                                a33 = vfmaq_f32(a33, vld1q_f32(p.add(12)), w3);
                            }
                        }
                    }
                    if relu {
                        a00 = vmaxq_f32(a00, zero);
                        a01 = vmaxq_f32(a01, zero);
                        a02 = vmaxq_f32(a02, zero);
                        a03 = vmaxq_f32(a03, zero);
                        a10 = vmaxq_f32(a10, zero);
                        a11 = vmaxq_f32(a11, zero);
                        a12 = vmaxq_f32(a12, zero);
                        a13 = vmaxq_f32(a13, zero);
                        a20 = vmaxq_f32(a20, zero);
                        a21 = vmaxq_f32(a21, zero);
                        a22 = vmaxq_f32(a22, zero);
                        a23 = vmaxq_f32(a23, zero);
                        a30 = vmaxq_f32(a30, zero);
                        a31 = vmaxq_f32(a31, zero);
                        a32 = vmaxq_f32(a32, zero);
                        a33 = vmaxq_f32(a33, zero);
                    }
                    let d = out_row.as_mut_ptr();
                    let d0 = d.add(owi * c_exp + ch);
                    vst1q_f32(d0, a00);
                    vst1q_f32(d0.add(4), a01);
                    vst1q_f32(d0.add(8), a02);
                    vst1q_f32(d0.add(12), a03);
                    let d1 = d.add((owi + 1) * c_exp + ch);
                    vst1q_f32(d1, a10);
                    vst1q_f32(d1.add(4), a11);
                    vst1q_f32(d1.add(8), a12);
                    vst1q_f32(d1.add(12), a13);
                    let d2 = d.add((owi + 2) * c_exp + ch);
                    vst1q_f32(d2, a20);
                    vst1q_f32(d2.add(4), a21);
                    vst1q_f32(d2.add(8), a22);
                    vst1q_f32(d2.add(12), a23);
                    let d3 = d.add((owi + 3) * c_exp + ch);
                    vst1q_f32(d3, a30);
                    vst1q_f32(d3.add(4), a31);
                    vst1q_f32(d3.add(8), a32);
                    vst1q_f32(d3.add(12), a33);
                    ch += 16;
                }
                owi += 4;
            }
        }

        for ow in owi..ow_end {
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

        // 4-wide spatial in 16-channel blocks (one weight quad feeds 4 output
        // columns) — the column-tiled 5×5 DW used for wide-c_exp blocks like
        // xif4 [16,16,672]. Mirrors the row-kernel 4-wide; pw stride is c_tile,
        // weights/bias carry the oc_start offset, output is c_exp-strided.
        let mut owi = ow_start;
        if c_tile.is_multiple_of(16) {
            while owi + 4 <= ow_end {
                let iw0a = (owi as i32) * (stride as i32) - (pad as i32);
                let iw0b = ((owi + 1) as i32) * (stride as i32) - (pad as i32);
                let iw0c = ((owi + 2) as i32) * (stride as i32) - (pad as i32);
                let iw0d = ((owi + 3) as i32) * (stride as i32) - (pad as i32);
                let mut ch = 0usize;
                while ch < c_tile {
                    let (bb0, bb1, bb2, bb3) = if let Some(b) = dw_bias {
                        let bp = b.as_ptr().add(oc_start + ch);
                        (
                            vld1q_f32(bp),
                            vld1q_f32(bp.add(4)),
                            vld1q_f32(bp.add(8)),
                            vld1q_f32(bp.add(12)),
                        )
                    } else {
                        (zero, zero, zero, zero)
                    };
                    let (mut a00, mut a01, mut a02, mut a03) = (bb0, bb1, bb2, bb3);
                    let (mut a10, mut a11, mut a12, mut a13) = (bb0, bb1, bb2, bb3);
                    let (mut a20, mut a21, mut a22, mut a23) = (bb0, bb1, bb2, bb3);
                    let (mut a30, mut a31, mut a32, mut a33) = (bb0, bb1, bb2, bb3);
                    for (ky, row) in rows.iter().enumerate() {
                        let pwp = match row {
                            Some(r) => r.as_ptr(),
                            None => continue,
                        };
                        for kx in 0..5usize {
                            let wp = dw_weight
                                .as_ptr()
                                .add(ky * w_ky_stride + kx * w_kx_stride + oc_start + ch);
                            let w0 = vld1q_f32(wp);
                            let w1 = vld1q_f32(wp.add(4));
                            let w2 = vld1q_f32(wp.add(8));
                            let w3 = vld1q_f32(wp.add(12));
                            let iwa = iw0a + kx as i32;
                            if iwa >= 0 && (iwa as usize) < in_w {
                                let p = pwp.add(iwa as usize * c_tile + ch);
                                a00 = vfmaq_f32(a00, vld1q_f32(p), w0);
                                a01 = vfmaq_f32(a01, vld1q_f32(p.add(4)), w1);
                                a02 = vfmaq_f32(a02, vld1q_f32(p.add(8)), w2);
                                a03 = vfmaq_f32(a03, vld1q_f32(p.add(12)), w3);
                            }
                            let iwb = iw0b + kx as i32;
                            if iwb >= 0 && (iwb as usize) < in_w {
                                let p = pwp.add(iwb as usize * c_tile + ch);
                                a10 = vfmaq_f32(a10, vld1q_f32(p), w0);
                                a11 = vfmaq_f32(a11, vld1q_f32(p.add(4)), w1);
                                a12 = vfmaq_f32(a12, vld1q_f32(p.add(8)), w2);
                                a13 = vfmaq_f32(a13, vld1q_f32(p.add(12)), w3);
                            }
                            let iwc = iw0c + kx as i32;
                            if iwc >= 0 && (iwc as usize) < in_w {
                                let p = pwp.add(iwc as usize * c_tile + ch);
                                a20 = vfmaq_f32(a20, vld1q_f32(p), w0);
                                a21 = vfmaq_f32(a21, vld1q_f32(p.add(4)), w1);
                                a22 = vfmaq_f32(a22, vld1q_f32(p.add(8)), w2);
                                a23 = vfmaq_f32(a23, vld1q_f32(p.add(12)), w3);
                            }
                            let iwd = iw0d + kx as i32;
                            if iwd >= 0 && (iwd as usize) < in_w {
                                let p = pwp.add(iwd as usize * c_tile + ch);
                                a30 = vfmaq_f32(a30, vld1q_f32(p), w0);
                                a31 = vfmaq_f32(a31, vld1q_f32(p.add(4)), w1);
                                a32 = vfmaq_f32(a32, vld1q_f32(p.add(8)), w2);
                                a33 = vfmaq_f32(a33, vld1q_f32(p.add(12)), w3);
                            }
                        }
                    }
                    if relu {
                        a00 = vmaxq_f32(a00, zero);
                        a01 = vmaxq_f32(a01, zero);
                        a02 = vmaxq_f32(a02, zero);
                        a03 = vmaxq_f32(a03, zero);
                        a10 = vmaxq_f32(a10, zero);
                        a11 = vmaxq_f32(a11, zero);
                        a12 = vmaxq_f32(a12, zero);
                        a13 = vmaxq_f32(a13, zero);
                        a20 = vmaxq_f32(a20, zero);
                        a21 = vmaxq_f32(a21, zero);
                        a22 = vmaxq_f32(a22, zero);
                        a23 = vmaxq_f32(a23, zero);
                        a30 = vmaxq_f32(a30, zero);
                        a31 = vmaxq_f32(a31, zero);
                        a32 = vmaxq_f32(a32, zero);
                        a33 = vmaxq_f32(a33, zero);
                    }
                    let d = out_row.as_mut_ptr();
                    let d0 = d.add(owi * c_exp + oc_start + ch);
                    vst1q_f32(d0, a00);
                    vst1q_f32(d0.add(4), a01);
                    vst1q_f32(d0.add(8), a02);
                    vst1q_f32(d0.add(12), a03);
                    let d1 = d.add((owi + 1) * c_exp + oc_start + ch);
                    vst1q_f32(d1, a10);
                    vst1q_f32(d1.add(4), a11);
                    vst1q_f32(d1.add(8), a12);
                    vst1q_f32(d1.add(12), a13);
                    let d2 = d.add((owi + 2) * c_exp + oc_start + ch);
                    vst1q_f32(d2, a20);
                    vst1q_f32(d2.add(4), a21);
                    vst1q_f32(d2.add(8), a22);
                    vst1q_f32(d2.add(12), a23);
                    let d3 = d.add((owi + 3) * c_exp + oc_start + ch);
                    vst1q_f32(d3, a30);
                    vst1q_f32(d3.add(4), a31);
                    vst1q_f32(d3.add(8), a32);
                    vst1q_f32(d3.add(12), a33);
                    ch += 16;
                }
                owi += 4;
            }
        }

        for ow in owi..ow_end {
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
