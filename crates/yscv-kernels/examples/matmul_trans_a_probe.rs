//! Microbench for `matmul_2d_slices_trans_a` on the tracker's
//! FusedTransposeMatMul shape (M=64, K=32, N=64) and surrounding shapes.
//!
//! The kernel uses a 4-row × 16-acc-ZMM outer-product tile (16 accumulator
//! chains in flight, pipe-saturated) rather than a 1-row × 4-acc tile.
//!
//! This probe times the kernel in isolation, A/B'd via the `YSCV_FTMM_4ROW_OFF`
//! kill switch. Caller can run either way — single binary invocation reports
//! the configuration the env was set to at process start.
//!
//! Usage:
//!   cargo run --release --no-default-features -p yscv-kernels --example matmul_trans_a_probe
//!   YSCV_FTMM_4ROW_OFF=1 cargo run --release --no-default-features -p yscv-kernels --example matmul_trans_a_probe

use std::time::Instant;
use yscv_kernels::matmul_2d_slices_trans_a;

fn fill(buf: &mut [f32], seed: u64) {
    let mut s = seed.wrapping_add(0xdeadbeef);
    for v in buf.iter_mut() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *v = ((s >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0;
    }
}

fn time_shape(label: &str, m: usize, k: usize, n: usize, iters: usize, warm: usize) {
    let mut a_kt = vec![0.0f32; k * m];
    let mut b = vec![0.0f32; k * n];
    let mut out = vec![0.0f32; m * n];
    fill(&mut a_kt, 1);
    fill(&mut b, 2);

    for _ in 0..warm {
        matmul_2d_slices_trans_a(&a_kt, m, k, &b, n, &mut out);
        std::hint::black_box(&out[0]);
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        matmul_2d_slices_trans_a(&a_kt, m, k, &b, n, &mut out);
        std::hint::black_box(&out[0]);
    }
    let us_per = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    let flops = 2.0 * m as f64 * k as f64 * n as f64;
    let gflops = flops / us_per / 1e3;
    println!("  {label:48}  M={m:4} K={k:4} N={n:4}  {us_per:7.2}µs  {gflops:6.1} GFLOPS");
}

fn main() {
    let mode = if std::env::var_os("YSCV_FTMM_4ROW_OFF").is_some() {
        "1-row"
    } else {
        "4-row"
    };
    println!("FTMM kernel mode: {mode}");
    println!(
        "  {:48}  {:>13}  {:>9}  {:>10}",
        "label", "shape", "µs/op", "GFLOPS"
    );

    // Tracker hot shapes — matches `trans_a_tracker_cls_dw_shape` test.
    // The original profile measured ~140 µs/call on the 1-row kernel.
    time_shape(
        "cls_dw FTMM (M=64 K=32 N=64) — tracker hot",
        64,
        32,
        64,
        50_000,
        5_000,
    );

    // Sweep around the tracker shape to characterise the curve.
    time_shape("M=32 K=32 N=64", 32, 32, 64, 50_000, 5_000);
    time_shape("M=128 K=32 N=64", 128, 32, 64, 20_000, 2_000);
    time_shape("M=64 K=64 N=64", 64, 64, 64, 20_000, 2_000);
    time_shape("M=64 K=32 N=128", 64, 32, 128, 20_000, 2_000);
    time_shape("M=64 K=128 N=64", 64, 128, 64, 20_000, 2_000);

    // Larger (M=64 K=256 N=256) — covers the earlier microbench config
    // so old numbers remain comparable.
    time_shape("M=64 K=256 N=256 — legacy probe", 64, 256, 256, 5_000, 500);
}
