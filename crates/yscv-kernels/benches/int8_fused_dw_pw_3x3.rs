//! Criterion micro-benchmarks for the INT8 fused DW->PW chain kernel.
//!
//! Shapes mirror the tracker's `/xif*_0/dw/conv_2 + /xif*_0/pwl/conv`
//! closing pairs of the inverted bottleneck. Reference path runs the same
//! work as two separate primitives (DW NHWC i32 + PW prepacked GEMM rows)
//! so the per-iter savings of the fused kernel are visible on top of the
//! same SIMD primitives.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use yscv_kernels::{
    DepthwiseI8Params, Int8FusedDwPwParams, depthwise_i8_i32_nhwc_dispatch,
    int8_fused_dw_pw_dispatch, int8_matmul_prepacked_dispatch, pack_i8_b_for_matmul,
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
    c_out: usize,
    kh: usize,
    stride: usize,
    dw_relu: bool,
) -> Int8FusedDwPwParams {
    let pad = (kh - 1) / 2;
    let out_h = (in_h + 2 * pad - kh) / stride + 1;
    let out_w = (in_w + 2 * pad - kh) / stride + 1;
    Int8FusedDwPwParams {
        batch: 1,
        in_h,
        in_w,
        c_in,
        c_out,
        kh,
        stride,
        pad,
        out_h,
        out_w,
        dw_relu,
        dw_composite: 0.012,
        pw_composite: 0.0085,
        pw_y_zp: 0.0,
    }
}

fn bench_fused(c: &mut Criterion) {
    let mut group = c.benchmark_group("int8_fused_dw_pw_3x3");

    // Tracker closing-pair shapes. After the PW expand, DW operates on
    // the expanded `c_exp` channels; the closing PW reduces back to
    // bottleneck width. ih/iw match the post-stride spatial size of
    // each /xif*_0 stage.
    let cases = [
        ("xif2_0_64x64_c96_to_c24_s1", 64, 64, 96, 24, 3, 1, true),
        ("xif3_0_32x32_c144_to_c32_s1", 32, 32, 144, 32, 3, 1, true),
        ("xif4_0_16x16_c192_to_c64_s1", 16, 16, 192, 64, 3, 1, true),
    ];

    for (label, ih, iw, c_in, c_out, kh, stride, relu) in cases {
        let p = make_params(ih, iw, c_in, c_out, kh, stride, relu);

        let input = pseudo_i8(0xA1, p.input_len());
        let dw_weight = pseudo_i8(0xB2, p.dw_weight_len());
        let pw_weight = pseudo_i8(0xC3, p.c_in * p.c_out);
        let dw_bias = pseudo_i32(0xD4, p.c_in, 1024);
        let pw_bias = pseudo_i32(0xE5, p.c_out, 1024);
        let packed = pack_i8_b_for_matmul(&pw_weight, p.c_in, p.c_out);

        let mut output = vec![0_i8; p.output_len()];
        group.bench_function(format!("fused_{label}"), |b| {
            b.iter(|| {
                int8_fused_dw_pw_dispatch(
                    black_box(&input),
                    black_box(&dw_weight),
                    Some(black_box(&dw_bias)),
                    black_box(&packed),
                    Some(black_box(&pw_bias)),
                    p,
                    black_box(&mut output),
                    None,
                );
                black_box(&output);
            });
        });

        // Reference: full DW then full PW using the same dispatch
        // primitives the fused kernel calls into.
        let mut dw_acc = vec![0_i32; p.batch * p.out_h * p.out_w * p.c_in];
        let mut dw_out_i8 = vec![0_i8; p.batch * p.out_h * p.out_w * p.c_in];
        let mut pw_acc = vec![0_i32; p.batch * p.out_h * p.out_w * p.c_out];
        let dw_params = DepthwiseI8Params {
            batch: p.batch,
            in_h: p.in_h,
            in_w: p.in_w,
            channels: p.c_in,
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
                depthwise_i8_i32_nhwc_dispatch(
                    black_box(&input),
                    black_box(&dw_weight),
                    dw_params,
                    black_box(&mut dw_acc),
                );
                for (acc, out) in dw_acc.iter().zip(dw_out_i8.iter_mut()) {
                    let v = (*acc as f32) * p.dw_composite;
                    let mut q = v.round().clamp(-128.0, 127.0) as i8;
                    if p.dw_relu && q < 0 {
                        q = 0;
                    }
                    *out = q;
                }
                int8_matmul_prepacked_dispatch(
                    black_box(&dw_out_i8),
                    black_box(&packed),
                    p.batch * p.out_h * p.out_w,
                    black_box(&mut pw_acc),
                );
                black_box(&pw_acc);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_fused);
criterion_main!(benches);
