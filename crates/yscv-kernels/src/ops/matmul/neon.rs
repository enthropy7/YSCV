//! aarch64 NEON register-tiled GEMM microkernels (4x8 / 4x16 / 4x24),
//! dispatched from gebp_kernel_raw on aarch64 targets.

use super::*;

// ---------------------------------------------------------------------------
// NEON micro-kernel (aarch64)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
pub(super) unsafe fn microkernel_4x8_neon(
    a_panel: *const f32,
    b_panel: *const f32,
    c: *mut f32,
    ldc: usize,
    kc: usize,
    accumulate: bool,
    epilogue: GemmEpilogue,
    is_last_k: bool,
    col_offset: usize,
    residual_tile: Option<*const f32>,
) {
    let mut c00: float32x4_t = vdupq_n_f32(0.0);
    let mut c01: float32x4_t = vdupq_n_f32(0.0);
    let mut c10: float32x4_t = vdupq_n_f32(0.0);
    let mut c11: float32x4_t = vdupq_n_f32(0.0);
    let mut c20: float32x4_t = vdupq_n_f32(0.0);
    let mut c21: float32x4_t = vdupq_n_f32(0.0);
    let mut c30: float32x4_t = vdupq_n_f32(0.0);
    let mut c31: float32x4_t = vdupq_n_f32(0.0);

    for p in 0..kc {
        let a_off = p * MR;
        let b_off = p * NR;

        let a0 = vdupq_n_f32(*a_panel.add(a_off));
        let a1 = vdupq_n_f32(*a_panel.add(a_off + 1));
        let a2 = vdupq_n_f32(*a_panel.add(a_off + 2));
        let a3 = vdupq_n_f32(*a_panel.add(a_off + 3));

        let b0 = vld1q_f32(b_panel.add(b_off));
        let b1 = vld1q_f32(b_panel.add(b_off + 4));

        c00 = vfmaq_f32(c00, a0, b0);
        c01 = vfmaq_f32(c01, a0, b1);
        c10 = vfmaq_f32(c10, a1, b0);
        c11 = vfmaq_f32(c11, a1, b1);
        c20 = vfmaq_f32(c20, a2, b0);
        c21 = vfmaq_f32(c21, a2, b1);
        c30 = vfmaq_f32(c30, a3, b0);
        c31 = vfmaq_f32(c31, a3, b1);
    }

    let cp0 = c;
    let cp1 = c.add(ldc);
    let cp2 = c.add(2 * ldc);
    let cp3 = c.add(3 * ldc);

    if accumulate {
        c00 = vaddq_f32(vld1q_f32(cp0), c00);
        c01 = vaddq_f32(vld1q_f32(cp0.add(4)), c01);
        c10 = vaddq_f32(vld1q_f32(cp1), c10);
        c11 = vaddq_f32(vld1q_f32(cp1.add(4)), c11);
        c20 = vaddq_f32(vld1q_f32(cp2), c20);
        c21 = vaddq_f32(vld1q_f32(cp2.add(4)), c21);
        c30 = vaddq_f32(vld1q_f32(cp3), c30);
        c31 = vaddq_f32(vld1q_f32(cp3.add(4)), c31);
    }

    if is_last_k {
        if let Some(bias) = epilogue.bias {
            let bv0 = vld1q_f32(bias.add(col_offset));
            let bv1 = vld1q_f32(bias.add(col_offset + 4));
            c00 = vaddq_f32(c00, bv0);
            c01 = vaddq_f32(c01, bv1);
            c10 = vaddq_f32(c10, bv0);
            c11 = vaddq_f32(c11, bv1);
            c20 = vaddq_f32(c20, bv0);
            c21 = vaddq_f32(c21, bv1);
            c30 = vaddq_f32(c30, bv0);
            c31 = vaddq_f32(c31, bv1);
        }
        if let Some(res_tile) = residual_tile {
            let rp0 = res_tile;
            let rp1 = res_tile.add(ldc);
            let rp2 = res_tile.add(2 * ldc);
            let rp3 = res_tile.add(3 * ldc);
            c00 = vaddq_f32(c00, vld1q_f32(rp0));
            c01 = vaddq_f32(c01, vld1q_f32(rp0.add(4)));
            c10 = vaddq_f32(c10, vld1q_f32(rp1));
            c11 = vaddq_f32(c11, vld1q_f32(rp1.add(4)));
            c20 = vaddq_f32(c20, vld1q_f32(rp2));
            c21 = vaddq_f32(c21, vld1q_f32(rp2.add(4)));
            c30 = vaddq_f32(c30, vld1q_f32(rp3));
            c31 = vaddq_f32(c31, vld1q_f32(rp3.add(4)));
        }
        match epilogue.activation {
            Activation::Relu => {
                let zero = vdupq_n_f32(0.0);
                c00 = vmaxq_f32(c00, zero);
                c01 = vmaxq_f32(c01, zero);
                c10 = vmaxq_f32(c10, zero);
                c11 = vmaxq_f32(c11, zero);
                c20 = vmaxq_f32(c20, zero);
                c21 = vmaxq_f32(c21, zero);
                c30 = vmaxq_f32(c30, zero);
                c31 = vmaxq_f32(c31, zero);
            }
            Activation::Silu => {
                c00 = silu_neon(c00);
                c01 = silu_neon(c01);
                c10 = silu_neon(c10);
                c11 = silu_neon(c11);
                c20 = silu_neon(c20);
                c21 = silu_neon(c21);
                c30 = silu_neon(c30);
                c31 = silu_neon(c31);
            }
            Activation::None => {}
        }
    }

    vst1q_f32(cp0, c00);
    vst1q_f32(cp0.add(4), c01);
    vst1q_f32(cp1, c10);
    vst1q_f32(cp1.add(4), c11);
    vst1q_f32(cp2, c20);
    vst1q_f32(cp2.add(4), c21);
    vst1q_f32(cp3, c30);
    vst1q_f32(cp3.add(4), c31);
}

// ---------------------------------------------------------------------------
// Paired 4×16 NEON micro-kernel: two adjacent NR=8 panels at once.
// 16 f32x4 accumulators (4 rows × 4 quarters) + 4 b loads + 1 a broadcast per row.
// aarch64 has 32 v-registers, so this fits comfortably.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
pub(super) unsafe fn microkernel_4x16_neon(
    a_panel: *const f32,
    b_panel_0: *const f32,
    b_panel_1: *const f32,
    c: *mut f32,
    ldc: usize,
    kc: usize,
    accumulate: bool,
    epilogue: GemmEpilogue,
    is_last_k: bool,
    col_offset: usize,
    residual_tile: Option<*const f32>,
) {
    let mut c00: float32x4_t = vdupq_n_f32(0.0);
    let mut c01: float32x4_t = vdupq_n_f32(0.0);
    let mut c02: float32x4_t = vdupq_n_f32(0.0);
    let mut c03: float32x4_t = vdupq_n_f32(0.0);
    let mut c10: float32x4_t = vdupq_n_f32(0.0);
    let mut c11: float32x4_t = vdupq_n_f32(0.0);
    let mut c12: float32x4_t = vdupq_n_f32(0.0);
    let mut c13: float32x4_t = vdupq_n_f32(0.0);
    let mut c20: float32x4_t = vdupq_n_f32(0.0);
    let mut c21: float32x4_t = vdupq_n_f32(0.0);
    let mut c22: float32x4_t = vdupq_n_f32(0.0);
    let mut c23: float32x4_t = vdupq_n_f32(0.0);
    let mut c30: float32x4_t = vdupq_n_f32(0.0);
    let mut c31: float32x4_t = vdupq_n_f32(0.0);
    let mut c32: float32x4_t = vdupq_n_f32(0.0);
    let mut c33: float32x4_t = vdupq_n_f32(0.0);

    for p in 0..kc {
        let a_off = p * MR;
        let b_off = p * NR;

        let a0 = vdupq_n_f32(*a_panel.add(a_off));
        let a1 = vdupq_n_f32(*a_panel.add(a_off + 1));
        let a2 = vdupq_n_f32(*a_panel.add(a_off + 2));
        let a3 = vdupq_n_f32(*a_panel.add(a_off + 3));

        let b0 = vld1q_f32(b_panel_0.add(b_off));
        let b1 = vld1q_f32(b_panel_0.add(b_off + 4));
        let b2 = vld1q_f32(b_panel_1.add(b_off));
        let b3 = vld1q_f32(b_panel_1.add(b_off + 4));

        c00 = vfmaq_f32(c00, a0, b0);
        c01 = vfmaq_f32(c01, a0, b1);
        c02 = vfmaq_f32(c02, a0, b2);
        c03 = vfmaq_f32(c03, a0, b3);
        c10 = vfmaq_f32(c10, a1, b0);
        c11 = vfmaq_f32(c11, a1, b1);
        c12 = vfmaq_f32(c12, a1, b2);
        c13 = vfmaq_f32(c13, a1, b3);
        c20 = vfmaq_f32(c20, a2, b0);
        c21 = vfmaq_f32(c21, a2, b1);
        c22 = vfmaq_f32(c22, a2, b2);
        c23 = vfmaq_f32(c23, a2, b3);
        c30 = vfmaq_f32(c30, a3, b0);
        c31 = vfmaq_f32(c31, a3, b1);
        c32 = vfmaq_f32(c32, a3, b2);
        c33 = vfmaq_f32(c33, a3, b3);
    }

    let cp0 = c;
    let cp1 = c.add(ldc);
    let cp2 = c.add(2 * ldc);
    let cp3 = c.add(3 * ldc);

    if accumulate {
        c00 = vaddq_f32(vld1q_f32(cp0), c00);
        c01 = vaddq_f32(vld1q_f32(cp0.add(4)), c01);
        c02 = vaddq_f32(vld1q_f32(cp0.add(8)), c02);
        c03 = vaddq_f32(vld1q_f32(cp0.add(12)), c03);
        c10 = vaddq_f32(vld1q_f32(cp1), c10);
        c11 = vaddq_f32(vld1q_f32(cp1.add(4)), c11);
        c12 = vaddq_f32(vld1q_f32(cp1.add(8)), c12);
        c13 = vaddq_f32(vld1q_f32(cp1.add(12)), c13);
        c20 = vaddq_f32(vld1q_f32(cp2), c20);
        c21 = vaddq_f32(vld1q_f32(cp2.add(4)), c21);
        c22 = vaddq_f32(vld1q_f32(cp2.add(8)), c22);
        c23 = vaddq_f32(vld1q_f32(cp2.add(12)), c23);
        c30 = vaddq_f32(vld1q_f32(cp3), c30);
        c31 = vaddq_f32(vld1q_f32(cp3.add(4)), c31);
        c32 = vaddq_f32(vld1q_f32(cp3.add(8)), c32);
        c33 = vaddq_f32(vld1q_f32(cp3.add(12)), c33);
    }

    if is_last_k {
        if let Some(bias) = epilogue.bias {
            let bv0 = vld1q_f32(bias.add(col_offset));
            let bv1 = vld1q_f32(bias.add(col_offset + 4));
            let bv2 = vld1q_f32(bias.add(col_offset + 8));
            let bv3 = vld1q_f32(bias.add(col_offset + 12));
            c00 = vaddq_f32(c00, bv0);
            c01 = vaddq_f32(c01, bv1);
            c02 = vaddq_f32(c02, bv2);
            c03 = vaddq_f32(c03, bv3);
            c10 = vaddq_f32(c10, bv0);
            c11 = vaddq_f32(c11, bv1);
            c12 = vaddq_f32(c12, bv2);
            c13 = vaddq_f32(c13, bv3);
            c20 = vaddq_f32(c20, bv0);
            c21 = vaddq_f32(c21, bv1);
            c22 = vaddq_f32(c22, bv2);
            c23 = vaddq_f32(c23, bv3);
            c30 = vaddq_f32(c30, bv0);
            c31 = vaddq_f32(c31, bv1);
            c32 = vaddq_f32(c32, bv2);
            c33 = vaddq_f32(c33, bv3);
        }
        if let Some(res_tile) = residual_tile {
            let rp0 = res_tile;
            let rp1 = res_tile.add(ldc);
            let rp2 = res_tile.add(2 * ldc);
            let rp3 = res_tile.add(3 * ldc);
            c00 = vaddq_f32(c00, vld1q_f32(rp0));
            c01 = vaddq_f32(c01, vld1q_f32(rp0.add(4)));
            c02 = vaddq_f32(c02, vld1q_f32(rp0.add(8)));
            c03 = vaddq_f32(c03, vld1q_f32(rp0.add(12)));
            c10 = vaddq_f32(c10, vld1q_f32(rp1));
            c11 = vaddq_f32(c11, vld1q_f32(rp1.add(4)));
            c12 = vaddq_f32(c12, vld1q_f32(rp1.add(8)));
            c13 = vaddq_f32(c13, vld1q_f32(rp1.add(12)));
            c20 = vaddq_f32(c20, vld1q_f32(rp2));
            c21 = vaddq_f32(c21, vld1q_f32(rp2.add(4)));
            c22 = vaddq_f32(c22, vld1q_f32(rp2.add(8)));
            c23 = vaddq_f32(c23, vld1q_f32(rp2.add(12)));
            c30 = vaddq_f32(c30, vld1q_f32(rp3));
            c31 = vaddq_f32(c31, vld1q_f32(rp3.add(4)));
            c32 = vaddq_f32(c32, vld1q_f32(rp3.add(8)));
            c33 = vaddq_f32(c33, vld1q_f32(rp3.add(12)));
        }
        match epilogue.activation {
            Activation::Relu => {
                let zero = vdupq_n_f32(0.0);
                c00 = vmaxq_f32(c00, zero);
                c01 = vmaxq_f32(c01, zero);
                c02 = vmaxq_f32(c02, zero);
                c03 = vmaxq_f32(c03, zero);
                c10 = vmaxq_f32(c10, zero);
                c11 = vmaxq_f32(c11, zero);
                c12 = vmaxq_f32(c12, zero);
                c13 = vmaxq_f32(c13, zero);
                c20 = vmaxq_f32(c20, zero);
                c21 = vmaxq_f32(c21, zero);
                c22 = vmaxq_f32(c22, zero);
                c23 = vmaxq_f32(c23, zero);
                c30 = vmaxq_f32(c30, zero);
                c31 = vmaxq_f32(c31, zero);
                c32 = vmaxq_f32(c32, zero);
                c33 = vmaxq_f32(c33, zero);
            }
            Activation::Silu => {
                c00 = silu_neon(c00);
                c01 = silu_neon(c01);
                c02 = silu_neon(c02);
                c03 = silu_neon(c03);
                c10 = silu_neon(c10);
                c11 = silu_neon(c11);
                c12 = silu_neon(c12);
                c13 = silu_neon(c13);
                c20 = silu_neon(c20);
                c21 = silu_neon(c21);
                c22 = silu_neon(c22);
                c23 = silu_neon(c23);
                c30 = silu_neon(c30);
                c31 = silu_neon(c31);
                c32 = silu_neon(c32);
                c33 = silu_neon(c33);
            }
            Activation::None => {}
        }
    }

    vst1q_f32(cp0, c00);
    vst1q_f32(cp0.add(4), c01);
    vst1q_f32(cp0.add(8), c02);
    vst1q_f32(cp0.add(12), c03);
    vst1q_f32(cp1, c10);
    vst1q_f32(cp1.add(4), c11);
    vst1q_f32(cp1.add(8), c12);
    vst1q_f32(cp1.add(12), c13);
    vst1q_f32(cp2, c20);
    vst1q_f32(cp2.add(4), c21);
    vst1q_f32(cp2.add(8), c22);
    vst1q_f32(cp2.add(12), c23);
    vst1q_f32(cp3, c30);
    vst1q_f32(cp3.add(4), c31);
    vst1q_f32(cp3.add(8), c32);
    vst1q_f32(cp3.add(12), c33);
}

// ---------------------------------------------------------------------------
// Tripled 4×24 NEON micro-kernel: three adjacent NR=8 panels at once.
// 24 f32x4 accumulators (4 rows × 6 quarters) + 6 b loads + 1 a broadcast
// = 31 v-registers. aarch64 has exactly 32, so this is the widest NEON
// variant possible. Matches the x86 4×24 dispatch branch for the common
// "wide N" pointwise Conv case on ARM.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
pub(super) unsafe fn microkernel_4x24_neon(
    a_panel: *const f32,
    b_panel_0: *const f32,
    b_panel_1: *const f32,
    b_panel_2: *const f32,
    c: *mut f32,
    ldc: usize,
    kc: usize,
    accumulate: bool,
    epilogue: GemmEpilogue,
    is_last_k: bool,
    col_offset: usize,
    residual_tile: Option<*const f32>,
) {
    // 4 rows × 6 f32x4 quarters = 24 accumulators.
    let mut c00 = vdupq_n_f32(0.0);
    let mut c01 = vdupq_n_f32(0.0);
    let mut c02 = vdupq_n_f32(0.0);
    let mut c03 = vdupq_n_f32(0.0);
    let mut c04 = vdupq_n_f32(0.0);
    let mut c05 = vdupq_n_f32(0.0);
    let mut c10 = vdupq_n_f32(0.0);
    let mut c11 = vdupq_n_f32(0.0);
    let mut c12 = vdupq_n_f32(0.0);
    let mut c13 = vdupq_n_f32(0.0);
    let mut c14 = vdupq_n_f32(0.0);
    let mut c15 = vdupq_n_f32(0.0);
    let mut c20 = vdupq_n_f32(0.0);
    let mut c21 = vdupq_n_f32(0.0);
    let mut c22 = vdupq_n_f32(0.0);
    let mut c23 = vdupq_n_f32(0.0);
    let mut c24 = vdupq_n_f32(0.0);
    let mut c25 = vdupq_n_f32(0.0);
    let mut c30 = vdupq_n_f32(0.0);
    let mut c31 = vdupq_n_f32(0.0);
    let mut c32 = vdupq_n_f32(0.0);
    let mut c33 = vdupq_n_f32(0.0);
    let mut c34 = vdupq_n_f32(0.0);
    let mut c35 = vdupq_n_f32(0.0);

    for p in 0..kc {
        let a_off = p * MR;
        let b_off = p * NR;

        let a0 = vdupq_n_f32(*a_panel.add(a_off));
        let a1 = vdupq_n_f32(*a_panel.add(a_off + 1));
        let a2 = vdupq_n_f32(*a_panel.add(a_off + 2));
        let a3 = vdupq_n_f32(*a_panel.add(a_off + 3));

        let b0 = vld1q_f32(b_panel_0.add(b_off));
        let b1 = vld1q_f32(b_panel_0.add(b_off + 4));
        let b2 = vld1q_f32(b_panel_1.add(b_off));
        let b3 = vld1q_f32(b_panel_1.add(b_off + 4));
        let b4 = vld1q_f32(b_panel_2.add(b_off));
        let b5 = vld1q_f32(b_panel_2.add(b_off + 4));

        c00 = vfmaq_f32(c00, a0, b0);
        c01 = vfmaq_f32(c01, a0, b1);
        c02 = vfmaq_f32(c02, a0, b2);
        c03 = vfmaq_f32(c03, a0, b3);
        c04 = vfmaq_f32(c04, a0, b4);
        c05 = vfmaq_f32(c05, a0, b5);
        c10 = vfmaq_f32(c10, a1, b0);
        c11 = vfmaq_f32(c11, a1, b1);
        c12 = vfmaq_f32(c12, a1, b2);
        c13 = vfmaq_f32(c13, a1, b3);
        c14 = vfmaq_f32(c14, a1, b4);
        c15 = vfmaq_f32(c15, a1, b5);
        c20 = vfmaq_f32(c20, a2, b0);
        c21 = vfmaq_f32(c21, a2, b1);
        c22 = vfmaq_f32(c22, a2, b2);
        c23 = vfmaq_f32(c23, a2, b3);
        c24 = vfmaq_f32(c24, a2, b4);
        c25 = vfmaq_f32(c25, a2, b5);
        c30 = vfmaq_f32(c30, a3, b0);
        c31 = vfmaq_f32(c31, a3, b1);
        c32 = vfmaq_f32(c32, a3, b2);
        c33 = vfmaq_f32(c33, a3, b3);
        c34 = vfmaq_f32(c34, a3, b4);
        c35 = vfmaq_f32(c35, a3, b5);
    }

    let cp0 = c;
    let cp1 = c.add(ldc);
    let cp2 = c.add(2 * ldc);
    let cp3 = c.add(3 * ldc);

    if accumulate {
        c00 = vaddq_f32(vld1q_f32(cp0), c00);
        c01 = vaddq_f32(vld1q_f32(cp0.add(4)), c01);
        c02 = vaddq_f32(vld1q_f32(cp0.add(8)), c02);
        c03 = vaddq_f32(vld1q_f32(cp0.add(12)), c03);
        c04 = vaddq_f32(vld1q_f32(cp0.add(16)), c04);
        c05 = vaddq_f32(vld1q_f32(cp0.add(20)), c05);
        c10 = vaddq_f32(vld1q_f32(cp1), c10);
        c11 = vaddq_f32(vld1q_f32(cp1.add(4)), c11);
        c12 = vaddq_f32(vld1q_f32(cp1.add(8)), c12);
        c13 = vaddq_f32(vld1q_f32(cp1.add(12)), c13);
        c14 = vaddq_f32(vld1q_f32(cp1.add(16)), c14);
        c15 = vaddq_f32(vld1q_f32(cp1.add(20)), c15);
        c20 = vaddq_f32(vld1q_f32(cp2), c20);
        c21 = vaddq_f32(vld1q_f32(cp2.add(4)), c21);
        c22 = vaddq_f32(vld1q_f32(cp2.add(8)), c22);
        c23 = vaddq_f32(vld1q_f32(cp2.add(12)), c23);
        c24 = vaddq_f32(vld1q_f32(cp2.add(16)), c24);
        c25 = vaddq_f32(vld1q_f32(cp2.add(20)), c25);
        c30 = vaddq_f32(vld1q_f32(cp3), c30);
        c31 = vaddq_f32(vld1q_f32(cp3.add(4)), c31);
        c32 = vaddq_f32(vld1q_f32(cp3.add(8)), c32);
        c33 = vaddq_f32(vld1q_f32(cp3.add(12)), c33);
        c34 = vaddq_f32(vld1q_f32(cp3.add(16)), c34);
        c35 = vaddq_f32(vld1q_f32(cp3.add(20)), c35);
    }

    if is_last_k {
        if let Some(bias) = epilogue.bias {
            let bv0 = vld1q_f32(bias.add(col_offset));
            let bv1 = vld1q_f32(bias.add(col_offset + 4));
            let bv2 = vld1q_f32(bias.add(col_offset + 8));
            let bv3 = vld1q_f32(bias.add(col_offset + 12));
            let bv4 = vld1q_f32(bias.add(col_offset + 16));
            let bv5 = vld1q_f32(bias.add(col_offset + 20));
            c00 = vaddq_f32(c00, bv0);
            c01 = vaddq_f32(c01, bv1);
            c02 = vaddq_f32(c02, bv2);
            c03 = vaddq_f32(c03, bv3);
            c04 = vaddq_f32(c04, bv4);
            c05 = vaddq_f32(c05, bv5);
            c10 = vaddq_f32(c10, bv0);
            c11 = vaddq_f32(c11, bv1);
            c12 = vaddq_f32(c12, bv2);
            c13 = vaddq_f32(c13, bv3);
            c14 = vaddq_f32(c14, bv4);
            c15 = vaddq_f32(c15, bv5);
            c20 = vaddq_f32(c20, bv0);
            c21 = vaddq_f32(c21, bv1);
            c22 = vaddq_f32(c22, bv2);
            c23 = vaddq_f32(c23, bv3);
            c24 = vaddq_f32(c24, bv4);
            c25 = vaddq_f32(c25, bv5);
            c30 = vaddq_f32(c30, bv0);
            c31 = vaddq_f32(c31, bv1);
            c32 = vaddq_f32(c32, bv2);
            c33 = vaddq_f32(c33, bv3);
            c34 = vaddq_f32(c34, bv4);
            c35 = vaddq_f32(c35, bv5);
        }
        if let Some(res_tile) = residual_tile {
            let rp0 = res_tile;
            let rp1 = res_tile.add(ldc);
            let rp2 = res_tile.add(2 * ldc);
            let rp3 = res_tile.add(3 * ldc);
            c00 = vaddq_f32(c00, vld1q_f32(rp0));
            c01 = vaddq_f32(c01, vld1q_f32(rp0.add(4)));
            c02 = vaddq_f32(c02, vld1q_f32(rp0.add(8)));
            c03 = vaddq_f32(c03, vld1q_f32(rp0.add(12)));
            c04 = vaddq_f32(c04, vld1q_f32(rp0.add(16)));
            c05 = vaddq_f32(c05, vld1q_f32(rp0.add(20)));
            c10 = vaddq_f32(c10, vld1q_f32(rp1));
            c11 = vaddq_f32(c11, vld1q_f32(rp1.add(4)));
            c12 = vaddq_f32(c12, vld1q_f32(rp1.add(8)));
            c13 = vaddq_f32(c13, vld1q_f32(rp1.add(12)));
            c14 = vaddq_f32(c14, vld1q_f32(rp1.add(16)));
            c15 = vaddq_f32(c15, vld1q_f32(rp1.add(20)));
            c20 = vaddq_f32(c20, vld1q_f32(rp2));
            c21 = vaddq_f32(c21, vld1q_f32(rp2.add(4)));
            c22 = vaddq_f32(c22, vld1q_f32(rp2.add(8)));
            c23 = vaddq_f32(c23, vld1q_f32(rp2.add(12)));
            c24 = vaddq_f32(c24, vld1q_f32(rp2.add(16)));
            c25 = vaddq_f32(c25, vld1q_f32(rp2.add(20)));
            c30 = vaddq_f32(c30, vld1q_f32(rp3));
            c31 = vaddq_f32(c31, vld1q_f32(rp3.add(4)));
            c32 = vaddq_f32(c32, vld1q_f32(rp3.add(8)));
            c33 = vaddq_f32(c33, vld1q_f32(rp3.add(12)));
            c34 = vaddq_f32(c34, vld1q_f32(rp3.add(16)));
            c35 = vaddq_f32(c35, vld1q_f32(rp3.add(20)));
        }
        match epilogue.activation {
            Activation::Relu => {
                let z = vdupq_n_f32(0.0);
                c00 = vmaxq_f32(c00, z);
                c01 = vmaxq_f32(c01, z);
                c02 = vmaxq_f32(c02, z);
                c03 = vmaxq_f32(c03, z);
                c04 = vmaxq_f32(c04, z);
                c05 = vmaxq_f32(c05, z);
                c10 = vmaxq_f32(c10, z);
                c11 = vmaxq_f32(c11, z);
                c12 = vmaxq_f32(c12, z);
                c13 = vmaxq_f32(c13, z);
                c14 = vmaxq_f32(c14, z);
                c15 = vmaxq_f32(c15, z);
                c20 = vmaxq_f32(c20, z);
                c21 = vmaxq_f32(c21, z);
                c22 = vmaxq_f32(c22, z);
                c23 = vmaxq_f32(c23, z);
                c24 = vmaxq_f32(c24, z);
                c25 = vmaxq_f32(c25, z);
                c30 = vmaxq_f32(c30, z);
                c31 = vmaxq_f32(c31, z);
                c32 = vmaxq_f32(c32, z);
                c33 = vmaxq_f32(c33, z);
                c34 = vmaxq_f32(c34, z);
                c35 = vmaxq_f32(c35, z);
            }
            Activation::Silu => {
                c00 = silu_neon(c00);
                c01 = silu_neon(c01);
                c02 = silu_neon(c02);
                c03 = silu_neon(c03);
                c04 = silu_neon(c04);
                c05 = silu_neon(c05);
                c10 = silu_neon(c10);
                c11 = silu_neon(c11);
                c12 = silu_neon(c12);
                c13 = silu_neon(c13);
                c14 = silu_neon(c14);
                c15 = silu_neon(c15);
                c20 = silu_neon(c20);
                c21 = silu_neon(c21);
                c22 = silu_neon(c22);
                c23 = silu_neon(c23);
                c24 = silu_neon(c24);
                c25 = silu_neon(c25);
                c30 = silu_neon(c30);
                c31 = silu_neon(c31);
                c32 = silu_neon(c32);
                c33 = silu_neon(c33);
                c34 = silu_neon(c34);
                c35 = silu_neon(c35);
            }
            Activation::None => {}
        }
    }

    vst1q_f32(cp0, c00);
    vst1q_f32(cp0.add(4), c01);
    vst1q_f32(cp0.add(8), c02);
    vst1q_f32(cp0.add(12), c03);
    vst1q_f32(cp0.add(16), c04);
    vst1q_f32(cp0.add(20), c05);
    vst1q_f32(cp1, c10);
    vst1q_f32(cp1.add(4), c11);
    vst1q_f32(cp1.add(8), c12);
    vst1q_f32(cp1.add(12), c13);
    vst1q_f32(cp1.add(16), c14);
    vst1q_f32(cp1.add(20), c15);
    vst1q_f32(cp2, c20);
    vst1q_f32(cp2.add(4), c21);
    vst1q_f32(cp2.add(8), c22);
    vst1q_f32(cp2.add(12), c23);
    vst1q_f32(cp2.add(16), c24);
    vst1q_f32(cp2.add(20), c25);
    vst1q_f32(cp3, c30);
    vst1q_f32(cp3.add(4), c31);
    vst1q_f32(cp3.add(8), c32);
    vst1q_f32(cp3.add(12), c33);
    vst1q_f32(cp3.add(16), c34);
    vst1q_f32(cp3.add(20), c35);
}

/// SiLU using NEON: x / (1 + exp(-x)), uses fast sigmoid helper.
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn silu_neon(x: float32x4_t) -> float32x4_t {
    let one = vdupq_n_f32(1.0);
    let neg_x = vnegq_f32(x);
    let exp_neg = super::super::simd::exp::fast_exp_sigmoid_neon(neg_x);
    let denom = vaddq_f32(one, exp_neg);
    vdivq_f32(x, denom)
}
