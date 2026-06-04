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
        // 3e-4: the GEMM expand path folds bias at the end of the FMA chain
        // (like the 8×12/4×16 kernels), vs the scalar reference's bias-in-sum;
        // on the deep c_exp=672 reduction this drifts ~1e-4 from the reference.
        assert!(
            delta < 3e-4,
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
        // 3e-4: see case() — GEMM expand folds bias at the FMA-chain end.
        assert!(
            delta < 3e-4,
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

// in_w=16 unit stride exercises the column-reuse asm interior fast path
// (in_w=8 above is too small to enter it) end-to-end against the reference.
#[test]
fn tracker_5x5_stride1_inw16_c112_c672_relu() {
    case_5x5(1, 16, 16, 112, 672, 1, true);
}

#[test]
fn tracker_5x5_stride1_inw16_c24_c144_norelu() {
    case_5x5(1, 16, 16, 24, 144, 1, false);
}

// Hand-asm column-reuse 3×3 stride-2 DW microkernel (aarch64). Bit-parity vs a
// fused-FMA reference on the xif2_0 interior shape (in_w=128, c_exp=96) + GFLOPS
// against the NEON intrinsic.
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
#[test]
fn dw3s2_creuse_asm_parity_and_timing() {
    use super::super::matmul::dw3s2_creuse_neon;

    let c_exp = 96usize;
    let in_w = 128usize;
    let pad = 1usize;
    let stride = 2usize;

    let rows_data: Vec<Vec<f32>> = (0..3)
        .map(|r| seeded(in_w * c_exp, r as f32 * 11.0))
        .collect();
    let weight = seeded(9 * c_exp, 101.0);
    let bias = seeded(c_exp, 3.0);

    // correctness: interior tile at owi=2, ch block 0..8
    let owi = 2usize;
    let ch = 0usize;
    let mut out = vec![0.0f32; ((in_w + 2 * pad - 3) / stride + 1) * c_exp];
    let base_col = 2 * owi - pad; // input col for (p=0, kx=0)
    let rows_off: [*const f32; 3] =
        std::array::from_fn(|ky| unsafe { rows_data[ky].as_ptr().add(base_col * c_exp + ch) });
    unsafe {
        dw3s2_creuse_neon(
            rows_off.as_ptr(),
            weight.as_ptr().add(ch),
            out.as_mut_ptr().add(owi * c_exp + ch),
            c_exp,
            bias.as_ptr().add(ch),
            1,
            c_exp * 4,
        );
    }
    for p in 0..4 {
        for c in 0..8 {
            let mut acc = bias[ch + c];
            for ky in 0..3 {
                for kx in 0..3 {
                    let icol = 2 * (owi + p) - pad + kx;
                    acc = rows_data[ky][icol * c_exp + ch + c]
                        .mul_add(weight[(ky * 3 + kx) * c_exp + ch + c], acc);
                }
            }
            let acc = acc.max(0.0);
            let got = out[(owi + p) * c_exp + ch + c];
            assert_eq!(got, acc, "dw3s2 mismatch p={p} c={c}: got={got} ref={acc}");
        }
    }

    // timing: interior tiles of one output row
    let out_w = (in_w + 2 * pad - 3) / stride + 1;
    let lo = 1usize; // 2*1-1=1 >= 0
    let hi = (in_w - 9 + pad) / 2; // 2*owi-pad+8 <= in_w-1
    let n_tiles = (hi.saturating_sub(lo)) / 4;
    let n_chblk = c_exp / 8;
    let iters = 3000usize;
    let row_bases: [*const f32; 3] = std::array::from_fn(|ky| rows_data[ky].as_ptr());
    let calls = (iters * n_tiles * n_chblk) as f64;
    let flops = calls * 4.0 * 8.0 * 9.0 * 2.0;

    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        for t in 0..n_tiles {
            let owt = lo + t * 4;
            for cb in 0..n_chblk {
                let chb = cb * 8;
                let roff: [*const f32; 3] = std::array::from_fn(|ky| unsafe {
                    row_bases[ky].add((2 * owt - pad) * c_exp + chb)
                });
                unsafe {
                    dw3s2_creuse_neon(
                        roff.as_ptr(),
                        weight.as_ptr().add(chb),
                        out.as_mut_ptr().add(owt * c_exp + chb),
                        c_exp,
                        bias.as_ptr().add(chb),
                        1,
                        c_exp * 4,
                    );
                }
            }
        }
    }
    let dt = t0.elapsed().as_secs_f64();
    eprintln!(
        "dw3s2_creuse asm:  {:.2} GF/s  ({:.3} ms, {:.1} ns/call)",
        flops / dt / 1e9,
        dt * 1e3,
        dt / calls * 1e9
    );

    let interior_cols = (n_tiles * 4) as f64;
    let t1 = std::time::Instant::now();
    for _ in 0..iters {
        super::neon::compute_dw_row_neon(
            Some(&rows_data[0]),
            Some(&rows_data[1]),
            Some(&rows_data[2]),
            &weight,
            Some(&bias),
            &mut out,
            in_w,
            lo,
            lo + n_tiles * 4,
            c_exp,
            stride,
            pad,
            true,
        );
    }
    let dt1 = t1.elapsed().as_secs_f64();
    let flops1 = iters as f64 * interior_cols * c_exp as f64 * 9.0 * 2.0;
    eprintln!(
        "dw3s2 intrinsic:   {:.2} GF/s  ({:.3} ms)",
        flops1 / dt1 / 1e9,
        dt1 * 1e3
    );
    let _ = out_w;
}

// Hand-asm column-reuse 5×5 DW microkernel (aarch64). Validates bit-parity
// against a scalar reference on the xif4_5 interior shape and reports GFLOPS.
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
#[test]
fn dw5_creuse_asm_parity_and_timing() {
    use super::super::matmul::dw5_creuse_neon;

    let c_exp = 672usize;
    let in_w = 16usize;
    let pad = 2usize;
    let ksz = 5usize;

    // 5 ring input rows, weights (25 taps), bias.
    let rows_data: Vec<Vec<f32>> = (0..ksz)
        .map(|r| seeded(in_w * c_exp, r as f32 * 7.0))
        .collect();
    let weight = seeded(ksz * ksz * c_exp, 101.0);
    let bias = seeded(c_exp, 3.0);

    // ---- correctness: one interior tile (cols 2..6) × ch block 0..8 ----
    let ow_tile = pad; // first interior column → input col (ow_tile - pad) = 0
    let ch = 0usize;
    let mut out = vec![0.0f32; in_w * c_exp];
    let rows_off: [*const f32; 5] = std::array::from_fn(|ky| unsafe {
        rows_data[ky].as_ptr().add((ow_tile - pad) * c_exp + ch)
    });
    unsafe {
        dw5_creuse_neon(
            rows_off.as_ptr(),
            weight.as_ptr().add(ch),
            out.as_mut_ptr().add(ow_tile * c_exp + ch),
            c_exp,
            bias.as_ptr().add(ch),
            1,
            c_exp * 4, // natural KH·KW·C tap stride (bytes)
        );
    }
    // Reference uses the SAME fused-FMA accumulation order as the kernel
    // (ky-outer, kx-inner, one mul_add per tap) → bit-exact, not approximate.
    for p in 0..4 {
        for c in 0..8 {
            let mut acc = bias[ch + c];
            for ky in 0..ksz {
                for kx in 0..ksz {
                    let icol = ow_tile - pad + kx + p;
                    acc = rows_data[ky][icol * c_exp + ch + c]
                        .mul_add(weight[(ky * ksz + kx) * c_exp + ch + c], acc);
                }
            }
            let acc = acc.max(0.0);
            let got = out[(ow_tile + p) * c_exp + ch + c];
            assert_eq!(got, acc, "creuse mismatch p={p} c={c}: got={got} ref={acc}");
        }
    }

    // ---- timing: full interior of one output row, GFLOPS ----
    let interior_lo = pad;
    let interior_hi = in_w - pad; // exclusive; cols [pad, in_w-pad)
    let n_tiles = (interior_hi - interior_lo) / 4;
    let n_chblk = c_exp / 8;
    let iters = 3000usize;
    let row_bases: [*const f32; 5] = std::array::from_fn(|ky| rows_data[ky].as_ptr());

    let calls = (iters * n_tiles * n_chblk) as f64;
    let flops = calls * 4.0 * 8.0 * 25.0 * 2.0;

    // (a) natural KH·KW·C weight layout — each tap is c_exp floats apart.
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        for t in 0..n_tiles {
            let owt = interior_lo + t * 4;
            for cb in 0..n_chblk {
                let chb = cb * 8;
                let roff: [*const f32; 5] = std::array::from_fn(|ky| unsafe {
                    row_bases[ky].add((owt - pad) * c_exp + chb)
                });
                unsafe {
                    dw5_creuse_neon(
                        roff.as_ptr(),
                        weight.as_ptr().add(chb),
                        out.as_mut_ptr().add(owt * c_exp + chb),
                        c_exp,
                        bias.as_ptr().add(chb),
                        1,
                        c_exp * 4,
                    );
                }
            }
        }
    }
    let dt = t0.elapsed().as_secs_f64();
    eprintln!(
        "dw5_creuse strided: {:.2} GF/s  ({:.3} ms, {:.1} ns/call)",
        flops / dt / 1e9,
        dt * 1e3,
        dt / calls * 1e9
    );

    // (b) pre-packed [c_exp/8][25][8] weight tiles — taps contiguous (32 B),
    //     so the per-channel-block 800 B weight stays L1-resident across tiles.
    let mut wpack = vec![0.0f32; n_chblk * 25 * 8];
    for cb in 0..n_chblk {
        for tap in 0..25 {
            for c in 0..8 {
                wpack[cb * 200 + tap * 8 + c] = weight[tap * c_exp + cb * 8 + c];
            }
        }
    }
    // packed-path correctness: same tile/block as the natural-layout check.
    let mut out_p = vec![0.0f32; in_w * c_exp];
    unsafe {
        dw5_creuse_neon(
            rows_off.as_ptr(),
            wpack.as_ptr(), // cb=0
            out_p.as_mut_ptr().add(ow_tile * c_exp + ch),
            c_exp,
            bias.as_ptr().add(ch),
            1,
            32,
        );
    }
    for p in 0..4 {
        for c in 0..8 {
            let a = out[(ow_tile + p) * c_exp + ch + c];
            let b = out_p[(ow_tile + p) * c_exp + ch + c];
            assert!(
                (a - b).abs() < 1e-4,
                "packed mismatch p={p} c={c}: {a} vs {b}"
            );
        }
    }
    let t2 = std::time::Instant::now();
    for _ in 0..iters {
        for t in 0..n_tiles {
            let owt = interior_lo + t * 4;
            for cb in 0..n_chblk {
                let chb = cb * 8;
                let roff: [*const f32; 5] = std::array::from_fn(|ky| unsafe {
                    row_bases[ky].add((owt - pad) * c_exp + chb)
                });
                unsafe {
                    dw5_creuse_neon(
                        roff.as_ptr(),
                        wpack.as_ptr().add(cb * 200),
                        out.as_mut_ptr().add(owt * c_exp + chb),
                        c_exp,
                        bias.as_ptr().add(chb),
                        1,
                        32,
                    );
                }
            }
        }
    }
    let dt2 = t2.elapsed().as_secs_f64();
    eprintln!(
        "dw5_creuse packed:  {:.2} GF/s  ({:.3} ms, {:.1} ns/call)",
        flops / dt2 / 1e9,
        dt2 * 1e3,
        dt2 / calls * 1e9
    );

    // ---- baseline: existing NEON intrinsic over the same interior row ----
    let rows_ref: [Option<&[f32]>; 5] = std::array::from_fn(|ky| Some(rows_data[ky].as_slice()));
    let t1 = std::time::Instant::now();
    for _ in 0..iters {
        let ctx = super::Dw5RowCtx {
            rows: rows_ref,
            dw_weight: &weight,
            dw_bias: Some(&bias),
            out_row: &mut out,
            in_w,
            ow_start: interior_lo,
            ow_end: interior_hi,
            c_exp,
            stride: 1,
            pad,
            relu: true,
        };
        super::neon::compute_dw5_row_neon(ctx);
    }
    let dt1 = t1.elapsed().as_secs_f64();
    let cols = (interior_hi - interior_lo) as f64;
    let flops1 = iters as f64 * cols * c_exp as f64 * 25.0 * 2.0;
    eprintln!(
        "dw5 intrinsic:    {:.2} GF/s  ({:.3} ms, {:.1} ns/col-row)",
        flops1 / dt1 / 1e9,
        dt1 * 1e3,
        dt1 / iters as f64 / cols * 1e9
    );
}
