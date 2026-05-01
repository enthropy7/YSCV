//! Criterion micro-benchmarks for the INT8 fused PW->DW chain kernel.
//!
//! Shapes mirror the tracker's `/xif*_0/pw/conv_1 + /xif*_0/dw/conv_1`
//! pairs that motivated the new fused action — the first chunk of the
//! INT8 chain arc targets `/xif2_0` (c_in=16, c_exp=96, ih=128, s=2).
//! Compares fused dispatch against a reference that runs the same work
//! as two separate primitives (PW prepacked GEMM rows + DW NHWC i32) so
//! the per-iter savings of the fused kernel are visible on top of the
//! same SIMD primitives.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use yscv_kernels::{
    DepthwiseI8Params, Int8FusedPwDwParams, depthwise_i8_i32_nhwc_dispatch,
    int8_fused_pw_dw_dispatch, int8_matmul_prepacked_dispatch, pack_i8_b_for_matmul,
};

fn pseudo_i8(seed: u64, n: usize) -> Vec<i8> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as i64 % 64 - 32) as i8
        })
        .collect()
}

fn pseudo_i32(seed: u64, n: usize, range: i32) -> Vec<i32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as i64 % (range as i64 * 2) - range as i64) as i32
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn make_params(
    in_h: usize,
    in_w: usize,
    c_in: usize,
    c_exp: usize,
    kh: usize,
    stride: usize,
    pw_relu: bool,
) -> Int8FusedPwDwParams {
    let pad = (kh - 1) / 2;
    let out_h = (in_h + 2 * pad - kh) / stride + 1;
    let out_w = (in_w + 2 * pad - kh) / stride + 1;
    Int8FusedPwDwParams {
        batch: 1,
        in_h,
        in_w,
        c_in,
        c_exp,
        kh,
        stride,
        pad,
        out_h,
        out_w,
        pw_relu,
        pw_composite: 0.0125,
        dw_composite: 0.0089,
        dw_y_zp: 0.0,
    }
}

fn bench_fused(c: &mut Criterion) {
    let mut group = c.benchmark_group("int8_fused_pw_dw_3x3");

    // Tracker hot-list shapes. PW input width × c_in shrinks per stage as
    // /xif{2,3,4}_0 strides through the input pyramid.
    let cases = [
        ("xif2_0_128x128_c16_to_c96_s2", 128, 128, 16, 96, 3, 2, true),
        ("xif3_0_64x64_c24_to_c144_s2", 64, 64, 24, 144, 3, 2, true),
        ("xif4_0_32x32_c32_to_c192_s1", 32, 32, 32, 192, 3, 1, true),
    ];

    for (label, ih, iw, c_in, c_exp, kh, stride, relu) in cases {
        let p = make_params(ih, iw, c_in, c_exp, kh, stride, relu);

        let input = pseudo_i8(0xA1, p.input_len());
        let pw_weight = pseudo_i8(0xB2, p.c_in * p.c_exp);
        let dw_weight = pseudo_i8(0xC3, p.dw_weight_len());
        let pw_bias = pseudo_i32(0xD4, p.c_exp, 1024);
        let dw_bias = pseudo_i32(0xE5, p.c_exp, 1024);
        let packed = pack_i8_b_for_matmul(&pw_weight, p.c_in, p.c_exp);

        let mut output = vec![0_i8; p.output_len()];
        group.bench_function(format!("fused_{label}"), |b| {
            b.iter(|| {
                int8_fused_pw_dw_dispatch(
                    black_box(&input),
                    black_box(&packed),
                    Some(black_box(&pw_bias)),
                    black_box(&dw_weight),
                    Some(black_box(&dw_bias)),
                    p,
                    black_box(&mut output),
                    None,
                );
                black_box(&output);
            });
        });

        // Reference: compute PW for the entire NHWC tensor + DW separately
        // using the same dispatch primitives the fused kernel calls into.
        // Captures the cost the per-op chain pays for the layout marshall
        // + intermediate i8 buffer + scalar requant epilogue.
        let mut pw_acc = vec![0_i32; p.batch * p.in_h * p.in_w * p.c_exp];
        let mut pw_out_i8 = vec![0_i8; p.batch * p.in_h * p.in_w * p.c_exp];
        let mut dw_acc = vec![0_i32; p.batch * p.out_h * p.out_w * p.c_exp];
        let dw_params = DepthwiseI8Params {
            batch: p.batch,
            in_h: p.in_h,
            in_w: p.in_w,
            channels: p.c_exp,
            kernel: p.kh,
            stride_h: p.stride,
            stride_w: p.stride,
            pad_top: p.pad,
            pad_left: p.pad,
            out_h: p.out_h,
            out_w: p.out_w,
        };
        group.bench_function(format!("unfused_{label}"), |b| {
            b.iter(|| {
                int8_matmul_prepacked_dispatch(
                    black_box(&input),
                    black_box(&packed),
                    p.batch * p.in_h * p.in_w,
                    black_box(&mut pw_acc),
                );
                for (acc, out) in pw_acc.iter().zip(pw_out_i8.iter_mut()) {
                    let v = (*acc as f32) * p.pw_composite;
                    *out = v.round().clamp(-128.0, 127.0) as i8;
                }
                depthwise_i8_i32_nhwc_dispatch(
                    black_box(&pw_out_i8),
                    black_box(&dw_weight),
                    dw_params,
                    black_box(&mut dw_acc),
                );
                black_box(&dw_acc);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_fused);
criterion_main!(benches);
