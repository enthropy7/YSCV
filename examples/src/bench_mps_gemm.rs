//! Benchmark: MPS GEMM vs custom Metal matmul/conv kernel on representative YOLOv8n sizes.
//! Tests pure GEMM (1x1 conv equivalent) performance.

fn main() {
    use yscv_kernels::metal_backend::metal_conv::{ConvParams, MetalInference, mps_gemm_f16};

    let inf = MetalInference::new().expect("Metal init failed (see error above)");
    println!("Metal device: {}", inf.device_name());

    // Representative 1x1 conv sizes from YOLOv8n (M=batch*oh*ow, K=ic, N=oc)
    // Plus representative 3x3 sizes as pure GEMM (M=batch*oh*ow, K=kh*kw*ic, N=oc)
    let sizes: Vec<(&str, usize, usize, usize)> = vec![
        // 1x1 convs (pure GEMM)
        ("1x1: 6400×64→64", 6400, 64, 64),
        ("1x1: 6400×64→128", 6400, 64, 128),
        ("1x1: 1600×128→128", 1600, 128, 128),
        ("1x1: 1600×128→256", 1600, 128, 256),
        ("1x1: 400×256→256", 400, 256, 256),
        ("1x1: 400×256→512", 400, 256, 512),
        // 3x3 convs as GEMM (im2col layout: K = kh*kw*ic)
        ("3x3: 25600×48→32", 25600, 48, 32),
        ("3x3: 6400×288→64", 6400, 288, 64),
        ("3x3: 6400×576→64", 6400, 576, 64),
        ("3x3: 1600×576→128", 1600, 576, 128),
        ("3x3: 1600×1152→128", 1600, 1152, 128),
        ("3x3: 400×1152→256", 400, 1152, 256),
        ("3x3: 400×2304→256", 400, 2304, 256),
    ];

    let warmup = 5;
    let runs = 20;

    println!(
        "\n{:<25} {:>6} {:>6} {:>6} {:>7} {:>10} {:>10} {:>7}",
        "Layer", "M", "K", "N", "MFLOP", "custom ms", "MPS ms", "speedup"
    );
    println!("{}", "-".repeat(95));

    for (name, m, k, n) in &sizes {
        let m = *m;
        let k = *k;
        let n = *n;
        let mflop = 2.0 * m as f64 * k as f64 * n as f64 / 1e6;

        // Create f16 buffers
        let a_f32: Vec<f32> = (0..m * k)
            .map(|i| ((i as f32 * 0.001) % 1.0) - 0.5)
            .collect();
        let b_f32: Vec<f32> = (0..k * n)
            .map(|i| ((i as f32 * 0.01) % 1.0) - 0.5)
            .collect();

        let a_buf = inf.buffer_from_f32_as_f16(&a_f32);
        let b_buf = inf.buffer_from_f32_as_f16(&b_f32);
        let out_custom = inf.output_buffer_f16(m * n);
        let out_mps = inf.output_buffer_f16(m * n);

        // Benchmark custom matmul_f16io kernel
        // Use conv_gemm with 1x1 params for fair comparison (that's what the pipeline uses)
        let bias_f32: Vec<f32> = vec![0.0; n];
        let bias_buf = inf.buffer_from_f32_as_f16(&bias_f32);
        let params = ConvParams {
            m: m as u32,
            n_out: n as u32,
            k: k as u32,
            act: 0,
            ih: 1,
            iw: m as u32,
            ic: k as u32,
            oh: 1,
            ow: m as u32,
            kh: 1,
            kw: 1,
            sh: 1,
            sw: 1,
            pad_h: 0,
            pad_w: 0,
            batch: 1,
            out_stride: n as u32,
            out_offset: 0,
            in_stride: k as u32,
            in_offset: 0,
            has_residual: 0,
            _pad: 0,
        };

        // Warmup custom
        for _ in 0..warmup {
            yscv_kernels::metal_backend::metal_conv::autoreleasepool(|| {
                let cmd = inf.queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                let me = yscv_kernels::metal_backend::metal_conv::MetalEncoder::new(enc, &inf);
                me.conv_gemm_f16io(&a_buf, &b_buf, &bias_buf, &out_custom, &params);
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
            });
        }
        // Measure custom
        let mut custom_times = Vec::new();
        for _ in 0..runs {
            yscv_kernels::metal_backend::metal_conv::autoreleasepool(|| {
                let t0 = std::time::Instant::now();
                let cmd = inf.queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                let me = yscv_kernels::metal_backend::metal_conv::MetalEncoder::new(enc, &inf);
                me.conv_gemm_f16io(&a_buf, &b_buf, &bias_buf, &out_custom, &params);
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
                custom_times.push(t0.elapsed().as_secs_f64() * 1000.0);
            });
        }

        // Warmup MPS
        for _ in 0..warmup {
            let _ = mps_gemm_f16(
                &inf.device,
                &inf.queue,
                &a_buf,
                &b_buf,
                &out_mps,
                m as u32,
                n as u32,
                k as u32,
                1.0,
                0.0,
                false,
                false,
            );
        }
        // Measure MPS
        let mut mps_times = Vec::new();
        for _ in 0..runs {
            let t0 = std::time::Instant::now();
            let _ = mps_gemm_f16(
                &inf.device,
                &inf.queue,
                &a_buf,
                &b_buf,
                &out_mps,
                m as u32,
                n as u32,
                k as u32,
                1.0,
                0.0,
                false,
                false,
            );
            mps_times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }

        // Verify correctness (compare a few elements)
        let out_c = inf.read_buffer_f16_as_f32(&out_custom, m * n);
        let out_m = inf.read_buffer_f16_as_f32(&out_mps, m * n);
        let max_diff = out_c
            .iter()
            .zip(out_m.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        custom_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        mps_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let custom_median = custom_times[runs / 2];
        let mps_median = mps_times[runs / 2];
        let speedup = custom_median / mps_median;

        println!(
            "{:<25} {:>6} {:>6} {:>6} {:>7.1} {:>10.3} {:>10.3} {:>6.2}× {}",
            name,
            m,
            k,
            n,
            mflop,
            custom_median,
            mps_median,
            speedup,
            if max_diff > 1.0 {
                format!("DIFF={:.2}", max_diff)
            } else {
                format!("ok({:.3})", max_diff)
            }
        );
    }
}
