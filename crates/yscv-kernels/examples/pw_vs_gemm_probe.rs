//! Probe: streaming PW-expand broadcast kernel vs the blocked-GEMM (8x12 asm)
//! on the xif2_0 expand shape. Tells us whether the broadcast path is the
//! bottleneck and whether routing the streaming PW through the GEMM is worth it.
#![allow(unsafe_code)]

use std::time::Instant;

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn broadcast_pw(src: &[f32], w: &[f32], dst: &mut [f32], m: usize, k: usize, n: usize) {
    use std::arch::aarch64::*;
    let chunks = n / 4;
    for row in 0..m {
        let sp = src.as_ptr().add(row * k);
        let dp = dst.as_mut_ptr().add(row * n);
        let mut acc = vec![vdupq_n_f32(0.0); chunks];
        for ci in 0..k {
            let x = vdupq_n_f32(*sp.add(ci));
            let wb = w.as_ptr().add(ci * n);
            for c in 0..chunks {
                acc[c] = vfmaq_f32(acc[c], x, vld1q_f32(wb.add(c * 4)));
            }
        }
        for c in 0..chunks {
            vst1q_f32(dp.add(c * 4), acc[c]);
        }
    }
}

fn main() {
    let (m, k, n) = (128usize, 16usize, 96usize); // xif2_0 expand: in_w × cin × cexp
    let src: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.013).sin()).collect();
    let w: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.017).cos()).collect();
    let mut dst = vec![0.0f32; m * n];
    let flop = (m * k * n * 2) as f64;
    let iters = 5000;

    // Warm + time broadcast
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { broadcast_pw(&src, &w, &mut dst, m, k, n) };
        let t = Instant::now();
        for _ in 0..iters {
            unsafe { broadcast_pw(&src, &w, &mut dst, m, k, n) };
        }
        let s = t.elapsed().as_secs_f64();
        println!(
            "broadcast PW : {:.2} us/call  {:.2} GFLOP/s",
            s / iters as f64 * 1e6,
            flop * iters as f64 / s / 1e9
        );
    }

    // Time blocked GEMM (8x12 asm under the hood)
    let mut dstg = vec![0.0f32; m * n];
    yscv_kernels::matmul_2d_slices(&src, m, k, &w, n, &mut dstg);
    let t = Instant::now();
    for _ in 0..iters {
        yscv_kernels::matmul_2d_slices(&src, m, k, &w, n, &mut dstg);
    }
    let s = t.elapsed().as_secs_f64();
    println!(
        "blocked GEMM : {:.2} us/call  {:.2} GFLOP/s",
        s / iters as f64 * 1e6,
        flop * iters as f64 / s / 1e9
    );
}
