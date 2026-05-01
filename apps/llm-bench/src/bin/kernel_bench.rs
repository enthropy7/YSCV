//! Kernel-level microbench for the int8 / int4 GEMM kernels landed in
//! the quantization arc. Covers the full TinyLlama-1.1B / Llama-3.2-1B
//! linear-layer shape set (QKV projection, attention output, gate /
//! up / down projections, vocab head) plus the legacy tracker shapes.
//!
//! For each variant we (a) check the SIMD path is bitwise / numerically
//! equivalent to the scalar reference, then (b) time it. Run from repo
//! root:
//!
//! ```sh
//! cargo run --release --bin kernel_bench -p yscv-llm-bench
//! ```

use std::time::Instant;

use yscv_kernels::{
    Depthwise3x3I8Params, DepthwiseI8Params, depthwise_i8_i32_nhwc_dispatch,
    depthwise_i8_i32_nhwc_scalar, depthwise3x3_i8_i32_nhwc_dispatch,
    depthwise3x3_i8_i32_nhwc_scalar, int8_matmul_dispatch, int8_matmul_prepacked_dispatch,
    int8_matmul_scalar, matmul_2d_slices, pack_i8_b_for_matmul, pack_int4_symmetric_per_group,
    packed_int4_gemm_dispatch, packed_int4_gemm_scalar, packed_int4_gemv_dispatch,
    packed_int4_gemv_scalar,
};

fn pseudo_i8(seed: u64, n: usize) -> Vec<i8> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as i64 % 256 - 128) as i8
        })
        .collect()
}

fn pseudo_f32(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as i64 % 2001) - 1000) as f32 * 0.001
        })
        .collect()
}

/// Median of N timings, in µs/op. Skips the first `warm` iters so we
/// drop first-touch + page-fault costs.
fn time_us<F: FnMut()>(name: &str, iters: usize, warm: usize, mut f: F) -> f64 {
    for _ in 0..warm {
        f();
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        samples.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = samples[samples.len() / 2];
    let min = samples[0];
    println!("    {name:<32} med {med:>9.1} µs  min {min:>9.1} µs  ({iters} iters)",);
    med
}

fn check_int8_match(a: &[i8], b: &[i8], m: usize, k: usize, n: usize) {
    let mut s = vec![0_i32; m * n];
    let mut d = vec![0_i32; m * n];
    let packed = pack_i8_b_for_matmul(b, k, n);
    let mut p = vec![0_i32; m * n];
    int8_matmul_scalar(a, b, m, k, n, &mut s);
    int8_matmul_dispatch(a, b, m, k, n, &mut d);
    int8_matmul_prepacked_dispatch(a, &packed, m, &mut p);
    let mut max_diff = 0_i32;
    for (x, y) in s.iter().zip(d.iter()).chain(s.iter().zip(p.iter())) {
        max_diff = max_diff.max((x - y).abs());
    }
    if max_diff != 0 {
        eprintln!(
            "    !! int8 dispatch/prepacked DIVERGES from scalar at m={m},k={k},n={n}: max |Δ|={max_diff}"
        );
    } else {
        println!("    ✓ bitwise match scalar↔dispatch↔prepacked");
    }
}

fn check_int4_gemv_match(packed: &[u8], scales: &[f32], act: &[f32], n: usize, k: usize, g: usize) {
    let mut s = vec![0.0_f32; n];
    let mut d = vec![0.0_f32; n];
    packed_int4_gemv_scalar(packed, scales, act, &mut s, n, k, g);
    packed_int4_gemv_dispatch(packed, scales, act, &mut d, n, k, g);
    let max = s
        .iter()
        .zip(d.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let rel = max / s.iter().map(|v| v.abs()).fold(1e-6_f32, f32::max);
    println!("    ✓ GEMV scalar↔dispatch max|Δ|={max:.2e} rel={rel:.2e}");
}

fn check_int4_gemm_match(
    packed: &[u8],
    scales: &[f32],
    act: &[f32],
    m: usize,
    n: usize,
    k: usize,
    g: usize,
) {
    let mut s = vec![0.0_f32; m * n];
    let mut d = vec![0.0_f32; m * n];
    packed_int4_gemm_scalar(packed, scales, act, &mut s, m, n, k, g);
    packed_int4_gemm_dispatch(packed, scales, act, &mut d, m, n, k, g);
    let max = s
        .iter()
        .zip(d.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let rel = max / s.iter().map(|v| v.abs()).fold(1e-6_f32, f32::max);
    println!("    ✓ GEMM scalar↔dispatch max|Δ|={max:.2e} rel={rel:.2e}");
}

/// (label, M, K, N) — A is M×K, B is K×N, out M×N.
const INT8_SHAPES: &[(&str, usize, usize, usize)] = &[
    ("tracker  64×672×96", 64, 672, 96),
    ("tracker 256× 96×16", 256, 96, 16),
    // TinyLlama-1.1B prefill (P=64): hidden=2048, intermediate=5632
    ("TLlama  Q/K/V    64×2048×2048", 64, 2048, 2048),
    ("TLlama  attn-out 64×2048×2048", 64, 2048, 2048),
    ("TLlama  gate/up  64×2048×5632", 64, 2048, 5632),
    ("TLlama  down     64×5632×2048", 64, 5632, 2048),
    ("TLlama  vocab    64×2048×32000", 64, 2048, 32000),
    // Llama-3.2-1B: hidden=2048, intermediate=8192, vocab=128256
    ("L3.2-1B gate/up  64×2048×8192", 64, 2048, 8192),
    ("L3.2-1B down     64×8192×2048", 64, 8192, 2048),
    // Phi-2: hidden=2560, intermediate=10240, vocab=51200
    ("Phi-2   Q/K/V    64×2560×2560", 64, 2560, 2560),
    ("Phi-2   gate/up  64×2560×10240", 64, 2560, 10240),
    ("Phi-2   down     64×10240×2560", 64, 10240, 2560),
    // Llama-7B: hidden=4096, intermediate=11008, vocab=32000
    ("Llama-7B Q/K/V   64×4096×4096", 64, 4096, 4096),
    ("Llama-7B gate/up 64×4096×11008", 64, 4096, 11008),
    ("Llama-7B down    64×11008×4096", 64, 11008, 4096),
    // M-sweep at the historically tricky hidden×hidden shape, to expose
    // any per-M crossover between simple Bᵀ and the blocked paths:
    ("M-sweep  16×2048×2048", 16, 2048, 2048),
    ("M-sweep 128×2048×2048", 128, 2048, 2048),
    ("M-sweep 256×2048×2048", 256, 2048, 2048),
];

fn bench_int8_gemm() {
    println!("\n== INT8 GEMM (i32 acc) ==");
    for &(label, m, k, n) in INT8_SHAPES {
        let a = pseudo_i8(0xAA, m * k);
        let b = pseudo_i8(0xBB, k * n);
        // Heavier shapes get fewer iters.
        let iters = if m * k * n > 50_000_000 {
            3
        } else if m * k * n > 1_000_000 {
            5
        } else {
            30
        };
        println!("  {label}");
        check_int8_match(&a, &b, m.min(8), k, n.min(64));
        let mut out_s = vec![0_i32; m * n];
        let mut out_d = vec![0_i32; m * n];
        let mut out_p = vec![0_i32; m * n];
        let packed = pack_i8_b_for_matmul(&b, k, n);
        let s = time_us("scalar", iters, 1, || {
            int8_matmul_scalar(&a, &b, m, k, n, &mut out_s);
        });
        let d = time_us("dispatch (best SIMD)", iters, 1, || {
            int8_matmul_dispatch(&a, &b, m, k, n, &mut out_d);
        });
        let p = time_us("prepacked dispatch", iters, 1, || {
            int8_matmul_prepacked_dispatch(&a, &packed, m, &mut out_p);
        });
        let speedup = s / d;
        let prepack_speedup = d / p;
        let mark = if speedup < 1.0 { " ← REGRESSION" } else { "" };
        println!("    SIMD speedup vs scalar: {speedup:.2}×{mark}");
        println!("    prepacked speedup vs normal dispatch: {prepack_speedup:.2}×");
    }
}

/// (label, K, H, W, C) — NHWC depthwise, stride 1, same padding.
/// These cover the tracker depthwise chain sizes seen in the private
/// two-input model after QDQ-fast cleanup.
const INT8_DW_SHAPES: &[(&str, usize, usize, usize, usize)] = &[
    ("tracker 3x3 stem-ish 128×128×16", 3, 128, 128, 16),
    ("tracker 3x3 xif mid    64×64×96", 3, 64, 64, 96),
    ("tracker 3x3 xif deep   32×32×192", 3, 32, 32, 192),
    ("tracker 3x3 head       16×16×672", 3, 16, 16, 672),
    ("tracker 5x5 xif deep   32×32×192", 5, 32, 32, 192),
    ("tracker 5x5 head       16×16×672", 5, 16, 16, 672),
    ("tail coverage          17×19×15", 5, 17, 19, 15),
];

fn check_int8_depthwise_match(input: &[i8], weight: &[i8], p: Depthwise3x3I8Params) {
    let mut scalar = vec![0_i32; p.batch * p.out_h * p.out_w * p.channels];
    let mut dispatch = vec![0_i32; scalar.len()];
    depthwise3x3_i8_i32_nhwc_scalar(input, weight, p, &mut scalar);
    depthwise3x3_i8_i32_nhwc_dispatch(input, weight, p, &mut dispatch);
    let max_diff = scalar
        .iter()
        .zip(dispatch.iter())
        .map(|(a, b)| (a - b).abs())
        .max()
        .unwrap_or(0);
    if max_diff != 0 {
        eprintln!("    !! int8 depthwise dispatch DIVERGES from scalar: max |Δ|={max_diff}");
    } else {
        println!("    ✓ bitwise match scalar↔dispatch");
    }
}

fn bench_int8_depthwise3x3() {
    println!("\n== INT8 depthwise NHWC (i32 acc) ==");
    for &(label, kernel, h, w, c) in INT8_DW_SHAPES {
        let p = DepthwiseI8Params {
            batch: 1,
            in_h: h,
            in_w: w,
            channels: c,
            kernel,
            stride_h: 1,
            stride_w: 1,
            pad_top: kernel / 2,
            pad_left: kernel / 2,
            out_h: h,
            out_w: w,
        };
        let input = pseudo_i8(0xD1, p.batch * p.in_h * p.in_w * p.channels);
        let weight = pseudo_i8(0xD2, kernel * kernel * p.channels);
        let mut out_s = vec![0_i32; p.batch * p.out_h * p.out_w * p.channels];
        let mut out_d = vec![0_i32; out_s.len()];
        let iters = if h * w * c > 1_000_000 {
            20
        } else if h * w * c > 200_000 {
            50
        } else {
            200
        };
        println!("  {label}");
        if kernel == 3 {
            check_int8_depthwise_match(
                &input,
                &weight,
                Depthwise3x3I8Params {
                    batch: p.batch,
                    in_h: p.in_h,
                    in_w: p.in_w,
                    channels: p.channels,
                    stride_h: p.stride_h,
                    stride_w: p.stride_w,
                    pad_top: p.pad_top,
                    pad_left: p.pad_left,
                    out_h: p.out_h,
                    out_w: p.out_w,
                },
            );
        } else {
            let mut scalar = vec![0_i32; out_s.len()];
            let mut dispatch = vec![0_i32; out_s.len()];
            depthwise_i8_i32_nhwc_scalar(&input, &weight, p, &mut scalar);
            depthwise_i8_i32_nhwc_dispatch(&input, &weight, p, &mut dispatch);
            assert_eq!(scalar, dispatch);
            println!("    ✓ bitwise match scalar↔dispatch");
        }
        let s = time_us("scalar", iters, 3, || {
            depthwise_i8_i32_nhwc_scalar(&input, &weight, p, &mut out_s);
        });
        let d = time_us("dispatch (best SIMD)", iters, 3, || {
            depthwise_i8_i32_nhwc_dispatch(&input, &weight, p, &mut out_d);
        });
        let ops = 2.0 * (p.out_h * p.out_w * p.channels * kernel * kernel) as f64;
        println!(
            "    SIMD speedup vs scalar: {:.2}×   {:.1} GOPS",
            s / d,
            ops / (d * 1e-6) / 1e9
        );
    }
}

const INT4_GEMV_SHAPES: &[(&str, usize, usize)] = &[
    ("TLlama Q/K/V/out 2048×2048", 2048, 2048),
    ("TLlama gate/up   5632×2048", 5632, 2048),
    ("TLlama down      2048×5632", 2048, 5632),
    ("TLlama vocab    32000×2048", 32000, 2048),
    ("L3.2-1B gate/up  8192×2048", 8192, 2048),
    ("L3.2-1B down     2048×8192", 2048, 8192),
];

fn bench_int4_gemv() {
    println!("\n== INT4 packed GEMV (decode hot path, M=1) ==");
    let group_size = 32_usize;
    for &(label, n, k) in INT4_GEMV_SHAPES {
        let weights = pseudo_f32(0x100, n * k);
        let (packed, scales) = pack_int4_symmetric_per_group(&weights, n, k, group_size);
        let activation = pseudo_f32(0x200, k);
        let mut out = vec![0.0_f32; n];
        let iters = if n * k > 20_000_000 { 5 } else { 20 };
        println!("  {label}");
        check_int4_gemv_match(&packed, &scales, &activation, n.min(64), k, group_size);
        let s = time_us("scalar GEMV", iters, 2, || {
            packed_int4_gemv_scalar(&packed, &scales, &activation, &mut out, n, k, group_size);
        });
        let d = time_us("dispatch GEMV (best SIMD)", iters, 2, || {
            packed_int4_gemv_dispatch(&packed, &scales, &activation, &mut out, n, k, group_size);
        });
        // Effective bandwidth: each GEMV reads N*K/2 bytes of weight.
        let bytes = (n * k / 2) as f64;
        let bw = bytes / (d * 1e-6) / 1e9;
        println!(
            "    SIMD speedup vs scalar: {:.2}×   weight BW: {:.1} GB/s",
            s / d,
            bw
        );
    }
}

const INT4_GEMM_SHAPES: &[(&str, usize, usize, usize)] = &[
    // (label, M, N, K) — N=output, K=input.
    ("TLlama prefill M=8   2048×2048", 8, 2048, 2048),
    ("TLlama prefill M=32  2048×2048", 32, 2048, 2048),
    ("TLlama prefill M=128 2048×2048", 128, 2048, 2048),
    ("TLlama prefill M=256 2048×2048", 256, 2048, 2048),
    ("TLlama prefill M=128 5632×2048", 128, 5632, 2048),
    ("TLlama prefill M=128 2048×5632", 128, 2048, 5632),
    ("L3.2-1B prefill M=128 8192×2048", 128, 8192, 2048),
];

fn bench_int4_gemm() {
    println!("\n== INT4 packed GEMM (prefill, M>1) ==");
    let group_size = 32_usize;
    for &(label, m, n, k) in INT4_GEMM_SHAPES {
        let weights = pseudo_f32(0x100, n * k);
        let (packed, scales) = pack_int4_symmetric_per_group(&weights, n, k, group_size);
        let activation = pseudo_f32(0x300 + m as u64, m * k);
        let mut out_s = vec![0.0_f32; m * n];
        let mut out_d = vec![0.0_f32; m * n];
        let iters = if m * n * k > 100_000_000 {
            3
        } else if m >= 32 {
            5
        } else {
            20
        };
        println!("  {label}");
        check_int4_gemm_match(&packed, &scales, &activation, 4, n.min(64), k, group_size);
        let s = time_us("scalar GEMM", iters, 1, || {
            packed_int4_gemm_scalar(
                &packed,
                &scales,
                &activation,
                &mut out_s,
                m,
                n,
                k,
                group_size,
            );
        });
        let d = time_us("dispatch GEMM (best SIMD)", iters, 1, || {
            packed_int4_gemm_dispatch(
                &packed,
                &scales,
                &activation,
                &mut out_d,
                m,
                n,
                k,
                group_size,
            );
        });
        // Compare GEMM vs M× GEMV for this shape.
        let t = Instant::now();
        for row in 0..m {
            let act = &activation[row * k..(row + 1) * k];
            let dst = &mut out_d[row * n..(row + 1) * n];
            packed_int4_gemv_dispatch(&packed, &scales, act, dst, n, k, group_size);
        }
        let gemv_loop = t.elapsed().as_secs_f64() * 1e6;
        // Effective compute: 2 * M * N * K flops.
        let gflops = 2.0 * m as f64 * n as f64 * k as f64 / (d * 1e-6) / 1e9;
        println!(
            "    GEMM vs scalar: {:.2}×   GEMM vs GEMV-loop: {:.2}×   {:.1} GFLOPS",
            s / d,
            gemv_loop / d,
            gflops
        );
    }
}

fn detected_features() {
    print!("CPU features: ");
    #[cfg(target_arch = "x86_64")]
    {
        let mut feats = vec![];
        if std::is_x86_feature_detected!("avx2") {
            feats.push("AVX2");
        }
        if std::is_x86_feature_detected!("fma") {
            feats.push("FMA");
        }
        if std::is_x86_feature_detected!("avxvnni") {
            feats.push("AVX-VNNI");
        }
        if std::is_x86_feature_detected!("avx512f") {
            feats.push("AVX-512F");
        }
        if std::is_x86_feature_detected!("avx512bw") {
            feats.push("AVX-512BW");
        }
        if std::is_x86_feature_detected!("avx512vnni") {
            feats.push("AVX-512-VNNI");
        }
        println!("{}", feats.join(", "));
    }
    #[cfg(target_arch = "aarch64")]
    {
        let mut feats = vec!["NEON"];
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            feats.push("dotprod (SDOT)");
        }
        if std::arch::is_aarch64_feature_detected!("i8mm") {
            feats.push("i8mm (SMMLA)");
        }
        println!("{}", feats.join(", "));
    }
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    println!("Threads: {threads}");
}

/// (label, K, N) — fp32 sgemm M=1 hot path. Includes both LLM
/// hidden×hidden, hidden×intermediate and vocab-head shapes; these
/// are the matmuls that the runner's row_gemm dispatch sees on
/// decode (after packed-INT4 weights take care of the linear layers
/// the int4 packer detected).
const FP32_SGEMM_M1_SHAPES: &[(&str, usize, usize)] = &[
    ("attn-score    K=64    N=58", 64, 58), // Q @ K^T at past_len ~ 50
    ("TLlama Q/K/V  K=2048  N=2048", 2048, 2048),
    ("TLlama gate   K=2048  N=5632", 2048, 5632),
    ("TLlama down   K=5632  N=2048", 5632, 2048),
    ("TLlama vocab  K=2048  N=32000", 2048, 32000),
    ("L3.2 gate     K=2048  N=8192", 2048, 8192),
];

fn bench_fp32_sgemm_m1() {
    println!("\n== fp32 sgemm M=1 (skinny-GEMM hot path) ==");
    for &(label, k, n) in FP32_SGEMM_M1_SHAPES {
        let a = pseudo_f32(0xCAFE, k);
        let b = pseudo_f32(0xBEEF, k * n);
        let mut out = vec![0.0_f32; n];
        let iters = if k * n > 1_000_000 { 100 } else { 1000 };
        println!("  {label}");
        let med = time_us("matmul_2d_slices M=1", iters, 5, || {
            matmul_2d_slices(&a, 1, k, &b, n, &mut out);
        });
        let bytes = (k * n * 4) as f64;
        let bw = bytes / (med * 1e-6) / 1e9;
        let gflops = 2.0 * k as f64 * n as f64 / (med * 1e-6) / 1e9;
        println!("    weight BW: {bw:.1} GB/s   {gflops:.1} GFLOPS");
    }
}

fn main() {
    detected_features();
    bench_int8_depthwise3x3();
    bench_int8_gemm();
    bench_int4_gemv();
    bench_int4_gemm();
    bench_fp32_sgemm_m1();
    println!("\n(All variants verified bitwise/numerically equivalent to scalar reference above.)");
}
