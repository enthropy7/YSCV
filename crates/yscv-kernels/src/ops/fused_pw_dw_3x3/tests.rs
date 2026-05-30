//! Correctness tests for the fused PW-DW / PW-DW-PW-reduce kernels.

use super::*;

fn seeded(n: usize, seed: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32 + seed) * 0.013).sin()).collect()
}

/// Runs the fused kernel and the scalar reference path via the
/// variant-selector's `scalar` fallback (forced by passing a
/// C_exp that would still go through SIMD, but we compare against
/// an independent scalar execution by calling the same kernel
/// with a C_exp that defeats the SIMD check is not trivial; instead
/// we compare fused output against a sequential PW then DW compute
/// performed in-test).
fn run_sequential_reference(
    input: &[f32],
    pw_weight: &[f32],
    pw_bias: Option<&[f32]>,
    dw_weight: &[f32],
    dw_bias: Option<&[f32]>,
    batch: usize,
    in_h: usize,
    in_w: usize,
    c_in: usize,
    c_exp: usize,
    stride: usize,
    kernel_size: usize,
    pad: usize,
    pw_relu: bool,
    dw_relu: bool,
) -> Vec<f32> {
    // PW step: produce [batch, in_h, in_w, c_exp] tensor.
    let mut pw_out = vec![0.0f32; batch * in_h * in_w * c_exp];
    for ni in 0..batch {
        for ih in 0..in_h {
            for iw in 0..in_w {
                let src_off = ((ni * in_h + ih) * in_w + iw) * c_in;
                let dst_off = ((ni * in_h + ih) * in_w + iw) * c_exp;
                for ce in 0..c_exp {
                    let mut acc = pw_bias.map(|b| b[ce]).unwrap_or(0.0);
                    for ci in 0..c_in {
                        acc += input[src_off + ci] * pw_weight[ci * c_exp + ce];
                    }
                    if pw_relu && acc < 0.0 {
                        acc = 0.0;
                    }
                    pw_out[dst_off + ce] = acc;
                }
            }
        }
    }

    // DW step: [batch, in_h, in_w, c_exp] → [batch, out_h, out_w, c_exp].
    let out_h = (in_h + 2 * pad - kernel_size) / stride + 1;
    let out_w = (in_w + 2 * pad - kernel_size) / stride + 1;
    let mut dw_out = vec![0.0f32; batch * out_h * out_w * c_exp];
    for ni in 0..batch {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let ih0 = (oh as i32) * (stride as i32) - (pad as i32);
                let iw0 = (ow as i32) * (stride as i32) - (pad as i32);
                let dst_off = ((ni * out_h + oh) * out_w + ow) * c_exp;
                for ce in 0..c_exp {
                    let mut acc = dw_bias.map(|b| b[ce]).unwrap_or(0.0);
                    for ky in 0..kernel_size {
                        let ih = ih0 + ky as i32;
                        if ih < 0 || (ih as usize) >= in_h {
                            continue;
                        }
                        for kx in 0..kernel_size {
                            let iw = iw0 + kx as i32;
                            if iw < 0 || (iw as usize) >= in_w {
                                continue;
                            }
                            let pw_off = ((ni * in_h + ih as usize) * in_w + iw as usize) * c_exp;
                            let w_off = (ky * kernel_size + kx) * c_exp;
                            acc += pw_out[pw_off + ce] * dw_weight[w_off + ce];
                        }
                    }
                    if dw_relu && acc < 0.0 {
                        acc = 0.0;
                    }
                    dw_out[dst_off + ce] = acc;
                }
            }
        }
    }
    dw_out
}

fn case(
    batch: usize,
    in_h: usize,
    in_w: usize,
    c_in: usize,
    c_exp: usize,
    stride: usize,
    relu: bool,
) {
    let input = seeded(batch * in_h * in_w * c_in, 0.3);
    let pw_weight = seeded(c_in * c_exp, 1.7);
    let pw_bias: Vec<f32> = (0..c_exp).map(|i| 0.125 * (i as f32 - 4.0)).collect();
    let dw_weight = seeded(9 * c_exp, 2.1);
    let dw_bias: Vec<f32> = (0..c_exp).map(|i| 0.0625 * (i as f32 - 2.0)).collect();

    let pad = 1;
    let out_h = (in_h + 2 * pad - 3) / stride + 1;
    let out_w = (in_w + 2 * pad - 3) / stride + 1;
    let mut fused_out = vec![0.0f32; batch * out_h * out_w * c_exp];

    let act = if relu {
        Activation::Relu
    } else {
        Activation::None
    };
    fused_pw_expand_dw_3x3(
        &input,
        &pw_weight,
        Some(&pw_bias),
        &dw_weight,
        Some(&dw_bias),
        &mut fused_out,
        batch,
        in_h,
        in_w,
        c_in,
        c_exp,
        stride,
        act,
        act,
        None,
    );

    let ref_out = run_sequential_reference(
        &input,
        &pw_weight,
        Some(&pw_bias),
        &dw_weight,
        Some(&dw_bias),
        batch,
        in_h,
        in_w,
        c_in,
        c_exp,
        stride,
        3,
        pad,
        relu,
        relu,
    );

    for (i, (a, b)) in fused_out.iter().zip(ref_out.iter()).enumerate() {
        let delta = (a - b).abs();
        assert!(
            delta < 1e-4,
            "mismatch at {i}: fused={a} ref={b} delta={delta}"
        );
    }
}

#[test]
fn tracker_xif2_0_stride2_c16_c96_relu() {
    // /xif2_0/pw/conv_1 + /xif2_0/dw/conv_1 shape.
    case(1, 128, 128, 16, 96, 2, true);
}

#[test]
fn tracker_stride1_c24_c144_relu() {
    // /xif3_0 series shape (stride=1 sibling).
    case(1, 64, 64, 24, 144, 1, true);
}

#[test]
fn small_shape_stride1_c8_c16_no_relu() {
    case(1, 5, 5, 8, 16, 1, false);
}

#[test]
fn small_shape_stride2_c4_c32_relu() {
    case(1, 7, 7, 4, 32, 2, true);
}

#[test]
fn c_exp_not_multiple_of_8_falls_back_to_scalar() {
    // c_exp=12: AVX2 chunks of 8 won't match; variant selector
    // falls through to scalar on x86_64. NEON needs c_exp % 4 == 0,
    // so 12 works there; on aarch64 this exercises NEON.
    case(1, 7, 7, 4, 12, 1, true);
}

#[test]
fn batch_gt_1() {
    case(2, 8, 8, 8, 16, 2, true);
}

/// Tracker's `/xif4_5/pw/conv_1` + `/xif4_5/dw/conv_1` shape
/// (`c_exp=672`). On x86_64 w/ AVX-512 this exercises the tiled
/// kernel (672/16 = 42 chunks > 16, so above the register-blocked
/// path's ceiling). Small spatial (16×16) keeps the test fast.
#[test]
fn tracker_xif4_5_stride1_c112_c672_relu() {
    case(1, 16, 16, 112, 672, 1, true);
}

/// Fullrow kernel P=8 path: in_w=8, c_exp=672 (tiled packed + fullrow).
/// Exercises the P=8 monomorphization and the residual tile ci-stride.
#[test]
fn tracker_xif4_5_stride1_c112_c672_relu_inw8() {
    case(1, 8, 8, 112, 672, 1, true);
}

/// c_exp at the residual boundary: 288 = 18 chunks = 2 full tiles
/// + 2 residual. Exercises the tiled path's residual handling.
#[test]
fn tiled_residual_c8_c288_stride2_relu() {
    case(1, 8, 8, 8, 288, 2, true);
}

/// 4×6 PW tile path with out_w=64 (tracker xif2_0 shape). On AVX-512
/// this routes through `compute_pw_row_avx512_4x6_inner`:
/// c_exp=96 → 6 chunks = 1 full 4-OC tile + 2-chunk OC tail;
/// out_w=64 → 10 full 6-OW tiles + 4-col OW tail (falls back to
/// 6-ZMM path for the tail).
#[test]
fn pw_4x6_tile_c96_ow64_stride1() {
    case(1, 64, 64, 16, 96, 1, true);
}

/// OW tail exercised: out_w=37 → 6 full tiles + 1-col tail.
/// in_w=73 pad=1 stride=1 → out_w=73; need out_w=37 → set
/// in_w=37, stride=1 (with SAME-pad that gives out_w=37).
#[test]
fn pw_4x6_tile_c128_ow37_tail() {
    case(1, 16, 37, 8, 128, 1, false);
}

/// OC-tail of exactly 1 chunk: c_exp=80 → 5 chunks = 1 full 4-OC
/// tile + 1-chunk tail. Exercises the `oc_tail_chunks == 1` branch.
#[test]
fn pw_4x6_tile_c80_oc_tail() {
    case(1, 12, 12, 8, 80, 1, true);
}

fn case_5x5(
    batch: usize,
    in_h: usize,
    in_w: usize,
    c_in: usize,
    c_exp: usize,
    stride: usize,
    relu: bool,
) {
    let input = seeded(batch * in_h * in_w * c_in, 0.9);
    let pw_weight = seeded(c_in * c_exp, 1.3);
    let pw_bias: Vec<f32> = (0..c_exp).map(|i| 0.03125 * (i as f32 - 5.0)).collect();
    let dw_weight = seeded(25 * c_exp, 2.7);
    let dw_bias: Vec<f32> = (0..c_exp).map(|i| 0.015625 * (i as f32 - 3.0)).collect();
    let pad = 2;
    let out_h = (in_h + 2 * pad - 5) / stride + 1;
    let out_w = (in_w + 2 * pad - 5) / stride + 1;
    let mut fused_out = vec![0.0f32; batch * out_h * out_w * c_exp];
    let act = if relu {
        Activation::Relu
    } else {
        Activation::None
    };

    fused_pw_expand_dw_5x5(FusedPwDw5x5 {
        input: &input,
        pw_weight: &pw_weight,
        pw_bias: Some(&pw_bias),
        dw_weight: &dw_weight,
        dw_bias: Some(&dw_bias),
        output: &mut fused_out,
        batch,
        in_h,
        in_w,
        c_in,
        c_exp,
        stride,
        pw_activation: act,
        dw_activation: act,
        thread_pool: None,
    });

    let ref_out = run_sequential_reference(
        &input,
        &pw_weight,
        Some(&pw_bias),
        &dw_weight,
        Some(&dw_bias),
        batch,
        in_h,
        in_w,
        c_in,
        c_exp,
        stride,
        5,
        pad,
        relu,
        relu,
    );

    for (i, (a, b)) in fused_out.iter().zip(ref_out.iter()).enumerate() {
        let delta = (a - b).abs();
        assert!(
            delta < 1e-4,
            "5x5 mismatch at {i}: fused={a} ref={b} delta={delta}"
        );
    }
}

#[test]
fn tracker_5x5_stride2_c24_c144_relu() {
    case_5x5(1, 32, 32, 24, 144, 2, true);
}

#[test]
fn tracker_5x5_stride1_c112_c672_relu() {
    case_5x5(1, 8, 8, 112, 672, 1, true);
}
