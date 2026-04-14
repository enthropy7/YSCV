//! Benchmark: Metal simdgroup_matrix vs Metal basic (=same algo as WGSL) vs CPU.
//! Isolates: (1) naga overhead, (2) simdgroup_matrix benefit.

fn main() {
    let metal = yscv_kernels::MetalConv::new().expect("Metal init failed (see error above)");
    println!("Metal device: {}", metal.device_name());

    let layers: Vec<(
        &str,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        u32,
    )> = vec![
        (
            "3×640→16 s2",
            1,
            640,
            640,
            3,
            320,
            320,
            16,
            3,
            3,
            2,
            2,
            1,
            1,
            2,
        ),
        (
            "16×320→32 s2",
            1,
            320,
            320,
            16,
            160,
            160,
            32,
            3,
            3,
            2,
            2,
            1,
            1,
            2,
        ),
        (
            "32×160→16 3×3",
            1,
            160,
            160,
            32,
            160,
            160,
            16,
            3,
            3,
            1,
            1,
            1,
            1,
            2,
        ),
        (
            "64×80→64 3×3",
            1,
            80,
            80,
            64,
            80,
            80,
            64,
            3,
            3,
            1,
            1,
            1,
            1,
            2,
        ),
        (
            "80×80→80 3×3",
            1,
            80,
            80,
            80,
            80,
            80,
            80,
            3,
            3,
            1,
            1,
            1,
            1,
            2,
        ),
        (
            "128×40→128 3×3",
            1,
            40,
            40,
            128,
            40,
            40,
            128,
            3,
            3,
            1,
            1,
            1,
            1,
            2,
        ),
        (
            "256×20→256 3×3",
            1,
            20,
            20,
            256,
            20,
            20,
            256,
            3,
            3,
            1,
            1,
            1,
            1,
            2,
        ),
        (
            "64×80→64 1×1",
            1,
            80,
            80,
            64,
            80,
            80,
            64,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
        ),
        (
            "128×40→128 1×1",
            1,
            40,
            40,
            128,
            40,
            40,
            128,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
        ),
    ];

    // ── Per-layer comparison ──
    println!(
        "\n{:<22} {:>6} {:>6} {:>4} {:>8} {:>8} {:>8}",
        "Layer", "M", "K", "N", "simd ms", "basic ms", "ratio"
    );
    println!("{}", "-".repeat(75));

    let mut bufs = Vec::new();
    for (_, batch, ih, iw, ic, oh, ow, oc, kh, kw, sh, sw, pad_h, pad_w, act) in &layers {
        let m = batch * oh * ow;
        let k = kh * kw * ic;
        let n = *oc;

        let params = yscv_kernels::metal_backend::metal_conv::ConvParams {
            m: m as u32,
            n_out: n as u32,
            k: k as u32,
            act: *act,
            ih: *ih as u32,
            iw: *iw as u32,
            ic: *ic as u32,
            oh: *oh as u32,
            ow: *ow as u32,
            kh: *kh as u32,
            kw: *kw as u32,
            sh: *sh as u32,
            sw: *sw as u32,
            pad_h: *pad_h as u32,
            pad_w: *pad_w as u32,
            batch: *batch as u32,
            out_stride: n as u32,
            out_offset: 0,
            in_stride: *ic as u32,
            in_offset: 0,
            has_residual: 0,
            _pad: 0,
        };

        let input_data: Vec<f32> = (0..batch * ih * iw * ic)
            .map(|i| ((i as f32 * 0.001) % 1.0) - 0.5)
            .collect();
        let weight_data: Vec<f32> = (0..k * n)
            .map(|i| ((i as f32 * 0.01) % 1.0) - 0.5)
            .collect();
        let bias_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();

        let buf_input = metal.buffer_from_f32(&input_data);
        let buf_weight = metal.buffer_from_f32(&weight_data);
        let buf_bias = metal.buffer_from_f32(&bias_data);
        let buf_out_simd = metal.output_buffer(m * n);
        let buf_out_basic = metal.output_buffer(m * n);

        bufs.push((
            buf_input,
            buf_weight,
            buf_bias,
            buf_out_simd,
            buf_out_basic,
            params,
            m,
            n,
        ));
    }

    let mut total_simd = 0.0f64;
    let mut total_basic = 0.0f64;
    for (i, (name, ..)) in layers.iter().enumerate() {
        let (ref input, ref weight, ref bias, ref out_simd, ref out_basic, ref params, m, n) =
            bufs[i];

        let t_simd = metal.bench_conv_gemm(input, weight, bias, out_simd, params, 3, 10);
        let t_basic = metal.bench_conv_gemm_basic(input, weight, bias, out_basic, params, 3, 10);

        let min_simd = t_simd.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let min_basic = t_basic.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let ratio = min_basic / min_simd;

        // Verify basic correctness
        let simd_out = metal.read_buffer_f32(out_simd, m * n);
        let basic_out = metal.read_buffer_f32(out_basic, m * n);
        let max_diff = simd_out
            .iter()
            .zip(basic_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        total_simd += min_simd;
        total_basic += min_basic;

        println!(
            "{:<22} {:>6} {:>6} {:>4} {:>8.2} {:>8.2} {:>7.2}x d={:.4}",
            name, params.m, params.k, params.n_out, min_simd, min_basic, ratio, max_diff
        );
    }
    println!(
        "TOTAL (per-layer)      {:>6} {:>6} {:>4} {:>8.2} {:>8.2} {:>7.2}x",
        "",
        "",
        "",
        total_simd,
        total_basic,
        total_basic / total_simd
    );

    // ── Batched comparison ──
    println!("\n── Batched (single command buffer) ──");

    let dispatches_simd: Vec<_> = bufs
        .iter()
        .map(|(i, w, b, o, _, p, ..)| (i.clone(), w.clone(), b.clone(), o.clone(), *p))
        .collect();
    let dispatches_basic: Vec<_> = bufs
        .iter()
        .map(|(i, w, b, _, o, p, ..)| (i.clone(), w.clone(), b.clone(), o.clone(), *p))
        .collect();

    // Warmup
    for _ in 0..3 {
        metal.dispatch_batch_conv_gemm(&dispatches_simd);
        metal.dispatch_batch_conv_gemm_basic(&dispatches_basic);
    }

    let mut simd_times = Vec::new();
    let mut basic_times = Vec::new();
    for _ in 0..10 {
        let s = std::time::Instant::now();
        metal.dispatch_batch_conv_gemm(&dispatches_simd);
        simd_times.push(s.elapsed().as_secs_f64() * 1000.0);

        let s = std::time::Instant::now();
        metal.dispatch_batch_conv_gemm_basic(&dispatches_basic);
        basic_times.push(s.elapsed().as_secs_f64() * 1000.0);
    }

    let min_simd = simd_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let min_basic = basic_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    println!("  Simdgroup: min={:.2}ms", min_simd);
    println!("  Basic:     min={:.2}ms", min_basic);
    println!("  Ratio:     {:.2}x", min_basic / min_simd);
}
