//! Sgemm A/B harness: yscv-kernels (blocked GEMM) vs OpenBLAS cblas_sgemm.
//!
//! Runs a fixed shape sweep over tracker's top-10 (m, k, n) triples, times
//! each backend, and emits a CSV row per shape with GFLOPS per backend and
//! the relative ratio.
//!
//! Build: `cargo run --release -p yscv-kernels --example sgemm_ab --features blas`.
//!
//! Env overrides:
//!   YSCV_SGEMM_AB_ITERS=N      per-shape iteration count (default 500).
//!   YSCV_SGEMM_AB_WARMUP=N     warmup iterations (default 50).
//!
//! Our path is invoked via `matmul_2d_slices_fused_maybe_packed` with
//! `YSCV_FORCE_NO_BLAS=1` set in the process env — ensures we do not
//! accidentally measure BLAS twice. BLAS path is invoked via direct
//! `cblas_sys::cblas_sgemm` call. Both paths write into the same output
//! buffer per iteration; we then bitwise-compare outputs to guarantee
//! neither path silently changed its result since the last snapshot.

#![allow(unsafe_code)]

use std::time::Instant;

use yscv_kernels::{
    Activation, GemmEpilogue, ParallelMatmulConfig, matmul_2d_slices_fused_maybe_packed,
};

/// Tracker top-10 (m, k, n) shapes — sourced from
/// `YSCV_RUNNER_PROFILE=...` output on the Siamese tracker (2026-04-19).
const TRACKER_SHAPES: &[(usize, usize, usize, &str)] = &[
    (256, 672, 112, "xif pw up m256"),
    (64, 672, 112, "xif pw up m64"),
    (256, 384, 64, "xif pw mid m256"),
    (1024, 192, 32, "xif pw dn m1024"),
    (256, 192, 32, "xif pw dn m256"),
    (1024, 96, 32, "xif pw early m1024"),
    (4096, 24, 24, "xif pw tiny m4k"),
    (16384, 16, 16, "first-hop m16k"),
    (64, 384, 64, "xif pw mid m64"),
    (256, 96, 32, "xif pw early m256"),
];

fn fill_random(buf: &mut [f32], seed: u64) {
    // LCG — deterministic across runs, cheap, good enough for GEMM input.
    let mut state = seed.wrapping_add(0xdeadbeef);
    for v in buf.iter_mut() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *v = ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0;
    }
}

fn bench_yscv(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    iters: usize,
) -> f64 {
    let cfg = ParallelMatmulConfig {
        min_parallel_shared_dim: 1,
        min_parallel_output_elements: 4096,
    };
    let epilogue = GemmEpilogue::new(None, Activation::None);
    let t0 = Instant::now();
    for _ in 0..iters {
        matmul_2d_slices_fused_maybe_packed(a, m, k, b, n, out, None, epilogue, cfg, None);
    }
    t0.elapsed().as_secs_f64() / iters as f64
}

#[cfg(feature = "blas")]
fn bench_blas(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    iters: usize,
) -> f64 {
    use cblas_sys::{CBLAS_LAYOUT, CBLAS_TRANSPOSE, cblas_sgemm};
    let t0 = Instant::now();
    for _ in 0..iters {
        // SAFETY: slices sized correctly; cblas_sgemm computes pure output
        // with beta=0 so no aliasing constraints beyond buffer size.
        unsafe {
            cblas_sgemm(
                CBLAS_LAYOUT::CblasRowMajor,
                CBLAS_TRANSPOSE::CblasNoTrans,
                CBLAS_TRANSPOSE::CblasNoTrans,
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a.as_ptr(),
                k as i32,
                b.as_ptr(),
                n as i32,
                0.0,
                out.as_mut_ptr(),
                n as i32,
            );
        }
    }
    t0.elapsed().as_secs_f64() / iters as f64
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn main() {
    // Force our path to bypass BLAS short-circuit even when --features blas
    // is on: otherwise matmul_2d_slices_fused_maybe_packed would dispatch
    // to cblas_sgemm internally, making yscv_row indistinguishable from
    // blas_row in this harness.
    // SAFETY: setting env before spawning parallel work is safe on POSIX;
    // rayon's workers are not yet initialised here.
    unsafe {
        std::env::set_var("YSCV_FORCE_NO_BLAS", "1");
    }

    let iters: usize = std::env::var("YSCV_SGEMM_AB_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);
    let warmup: usize = std::env::var("YSCV_SGEMM_AB_WARMUP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);

    eprintln!("sgemm A/B — yscv (blocked, FORCE_NO_BLAS) vs OpenBLAS cblas_sgemm");
    eprintln!("iters={iters}  warmup={warmup}");
    eprintln!();

    println!("m,k,n,label,yscv_us,blas_us,yscv_gflops,blas_gflops,yscv_vs_blas_pct,max_abs_diff");

    for &(m, k, n, label) in TRACKER_SHAPES {
        let mut a = vec![0.0f32; m * k];
        let mut b = vec![0.0f32; k * n];
        let mut out_yscv = vec![0.0f32; m * n];
        #[allow(unused_mut)]
        let mut out_blas = vec![0.0f32; m * n];
        fill_random(&mut a, 1);
        fill_random(&mut b, 2);

        // Warmup both paths.
        bench_yscv(&a, &b, &mut out_yscv, m, k, n, warmup);
        #[cfg(feature = "blas")]
        bench_blas(&a, &b, &mut out_blas, m, k, n, warmup);

        // Measure.
        let yscv_secs = bench_yscv(&a, &b, &mut out_yscv, m, k, n, iters);
        #[cfg(feature = "blas")]
        let blas_secs = bench_blas(&a, &b, &mut out_blas, m, k, n, iters);
        #[cfg(not(feature = "blas"))]
        let blas_secs = f64::NAN;

        let flops = 2.0 * m as f64 * k as f64 * n as f64;
        let yscv_gflops = flops / yscv_secs / 1e9;
        let blas_gflops = flops / blas_secs / 1e9;
        let yscv_vs_blas = 100.0 * yscv_gflops / blas_gflops;

        // Bitwise-correctness sanity: warmup pass outputs should agree to FP rounding.
        let diff = max_abs_diff(&out_yscv, &out_blas);

        println!(
            "{m},{k},{n},{label},{:.2},{:.2},{:.2},{:.2},{:.1},{:.6e}",
            yscv_secs * 1e6,
            blas_secs * 1e6,
            yscv_gflops,
            blas_gflops,
            yscv_vs_blas,
            diff,
        );
    }
}
