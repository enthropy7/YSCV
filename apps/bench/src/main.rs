use std::time::Instant;

use yscv_autograd::Graph;
use yscv_tensor::Tensor;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn bench<F: FnOnce()>(name: &str, f: F) {
    use std::io::Write;
    let start = Instant::now();
    f();
    let elapsed = start.elapsed();
    println!("  {:<40} {:>10.3}ms", name, elapsed.as_secs_f64() * 1000.0);
    std::io::stdout().flush().ok();
}

fn bench_n<F: FnMut()>(name: &str, n: usize, mut f: F) {
    use std::io::Write;
    // Warm up
    f();
    let mut best = std::time::Duration::from_secs(999);
    for _ in 0..n {
        let start = Instant::now();
        f();
        let elapsed = start.elapsed();
        if elapsed < best {
            best = elapsed;
        }
    }
    println!(
        "  {:<40} {:>10.3}ms (best of {})",
        name,
        best.as_secs_f64() * 1000.0,
        n
    );
    std::io::stdout().flush().ok();
}

fn main() {
    println!("yscv framework benchmark");
    println!("========================");
    println!("Dispatch: {}", yscv_kernels::dispatch_report());
    println!();
    println!("  {:<40} {:>13}", "Operation", "Time (ms)");
    println!("  {}", "-".repeat(53));

    // --- Tensor ops (best-of-100, vs NumPy) ---
    let a1m = Tensor::from_vec(
        vec![1_000_000],
        (0..1_000_000).map(|i| i as f32 * 0.001).collect(),
    )
    .unwrap();
    let b1m = Tensor::from_vec(
        vec![1_000_000],
        (0..1_000_000)
            .map(|i| (1_000_000 - i) as f32 * 0.001)
            .collect(),
    )
    .unwrap();

    bench_n("add_1M", 100, || {
        let _c = a1m.add(&b1m).unwrap();
    });
    bench_n("mul_1M", 100, || {
        let _c = a1m.mul(&b1m).unwrap();
    });
    bench_n("sub_1M", 100, || {
        let _c = a1m.sub(&b1m).unwrap();
    });
    let mut out_1m = Tensor::zeros(vec![1_000_000]).unwrap();
    bench_n("add_1M_into", 100, || {
        Tensor::add_into(&mut out_1m, &a1m, &b1m);
    });
    bench_n("mul_1M_into", 100, || {
        Tensor::mul_into(&mut out_1m, &a1m, &b1m);
    });
    bench_n("sub_1M_into", 100, || {
        Tensor::sub_into(&mut out_1m, &a1m, &b1m);
    });
    bench_n("sum_1M", 100, || {
        let _s = a1m.sum();
    });
    bench_n("max_1M", 100, || {
        let _m = a1m.max_value();
    });
    bench_n("min_1M", 100, || {
        let _m = a1m.min_value();
    });
    bench_n("mean_1M", 100, || {
        let _m = a1m.mean();
    });
    bench_n("exp_1M", 100, || {
        let _e = a1m.exp();
    });
    bench_n("relu_1M", 100, || {
        let _r = yscv_kernels::relu(&a1m);
    });
    let mut relu_output = Tensor::zeros(vec![1_000_000]).unwrap();
    bench_n("relu_1M_out", 100, || {
        yscv_kernels::relu_out(&a1m, &mut relu_output);
    });
    // Raw SIMD relu — pure kernel, no Tensor overhead
    let input_raw = a1m.data();
    let mut raw_buf = vec![0.0f32; 1_000_000];
    bench_n("relu_1M_raw", 100, || {
        yscv_kernels::relu_to_slice_dispatch(input_raw, &mut raw_buf);
    });
    // Re-populate with mixed-sign data so relu has work to do each iteration.
    let mut relu_persistent = Tensor::from_vec(
        vec![1_000_000],
        (0..1_000_000)
            .map(|i| (i as f32 - 500_000.0) * 0.001)
            .collect(),
    )
    .unwrap();
    bench_n("relu_1M_inplace_nocopy", 100, || {
        yscv_kernels::relu_inplace(&mut relu_persistent);
    });
    bench_n("argmax_1M", 100, || {
        let _a = a1m.argmax();
    });
    bench_n("argmin_1M", 100, || {
        let _a = a1m.argmin();
    });

    // --- Tensor unary ops ---
    println!();
    println!("  --- tensor unary ops (best of 100) ---");
    println!("  {}", "-".repeat(53));

    bench_n("abs_1M", 100, || {
        let _r = a1m.abs();
    });
    bench_n("neg_1M", 100, || {
        let _r = a1m.neg();
    });
    bench_n("sqrt_1M", 100, || {
        let _r = a1m.sqrt();
    });
    bench_n("sin_1M", 100, || {
        let _r = a1m.sin();
    });
    bench_n("cos_1M", 100, || {
        let _r = a1m.cos();
    });
    bench_n("ln_1M", 100, || {
        let _r = a1m.ln();
    });
    bench_n("floor_1M", 100, || {
        let _r = a1m.floor();
    });
    bench_n("ceil_1M", 100, || {
        let _r = a1m.ceil();
    });
    bench_n("round_1M", 100, || {
        let _r = a1m.round();
    });
    bench_n("sign_1M", 100, || {
        let _r = a1m.sign();
    });
    bench_n("reciprocal_1M", 100, || {
        let _r = a1m.reciprocal();
    });
    bench_n("clamp_1M", 100, || {
        let _r = a1m.clamp(-1.0, 1.0);
    });

    // --- Tensor comparison ops ---
    println!();
    println!("  --- tensor comparison ops (best of 100) ---");
    println!("  {}", "-".repeat(53));

    let zero_1m = Tensor::zeros(vec![1_000_000]).unwrap();
    bench_n("gt_tensor_1M", 100, || {
        let _r = a1m.gt_tensor(&zero_1m).unwrap();
    });
    bench_n("eq_tensor_1M", 100, || {
        let _r = a1m.eq_tensor(&b1m).unwrap();
    });
    bench_n("lt_tensor_1M", 100, || {
        let _r = a1m.lt_tensor(&zero_1m).unwrap();
    });
    let mut cmp_out_1m = Tensor::zeros(vec![1_000_000]).unwrap();
    bench_n("gt_tensor_1M_into", 100, || {
        a1m.gt_tensor_into(&zero_1m, &mut cmp_out_1m);
    });
    bench_n("eq_tensor_1M_into", 100, || {
        a1m.eq_tensor_into(&b1m, &mut cmp_out_1m);
    });
    bench_n("lt_tensor_1M_into", 100, || {
        a1m.lt_tensor_into(&zero_1m, &mut cmp_out_1m);
    });

    // --- Matmul (vs NumPy/BLAS) ---
    let m256 = Tensor::zeros(vec![256, 256]).unwrap();
    let n256 = Tensor::zeros(vec![256, 256]).unwrap();
    bench_n("matmul_256x256", 100, || {
        let _c = yscv_kernels::matmul_2d(&m256, &n256).unwrap();
    });
    let m512 = Tensor::zeros(vec![512, 512]).unwrap();
    let n512 = Tensor::zeros(vec![512, 512]).unwrap();
    bench_n("matmul_512x512", 100, || {
        let _c = yscv_kernels::matmul_2d(&m512, &n512).unwrap();
    });

    // --- Conv2d ---
    let conv_input = Tensor::zeros(vec![1, 32, 32, 3]).unwrap();
    let conv_kernel = Tensor::zeros(vec![3, 3, 3, 16]).unwrap();
    bench_n("conv2d_nhwc_32x32", 100, || {
        let _c = yscv_kernels::conv2d_nhwc(&conv_input, &conv_kernel, None, 1, 1).unwrap();
    });

    // --- Activations (vs PyTorch) ---
    bench_n("sigmoid_1M", 100, || {
        let _s = yscv_kernels::sigmoid(&a1m);
    });

    // --- Activations (additional) ---
    bench_n("tanh_1M", 100, || {
        let _t = yscv_kernels::tanh_act(&a1m);
    });
    bench_n("gelu_1M", 100, || {
        let _g = yscv_kernels::gelu(&a1m);
    });
    bench_n("silu_1M", 100, || {
        let _s = yscv_kernels::silu(&a1m);
    });

    // --- Softmax / Normalization ---
    let sm_input = Tensor::zeros(vec![32, 1000]).unwrap();
    bench_n("softmax_32x1000", 100, || {
        let _s = yscv_kernels::softmax_last_dim(&sm_input).unwrap();
    });
    bench_n("log_softmax_32x1000", 100, || {
        let _s = yscv_kernels::log_softmax_last_dim(&sm_input).unwrap();
    });

    // Layer norm: 32x256
    {
        let ln_input = Tensor::zeros(vec![32, 256]).unwrap();
        let ln_gamma = Tensor::ones(vec![256]).unwrap();
        let ln_beta = Tensor::zeros(vec![256]).unwrap();
        bench_n("layer_norm_32x256", 100, || {
            let _l = yscv_kernels::layer_norm_last_dim(
                &ln_input,
                yscv_kernels::LayerNormLastDimParams {
                    gamma: &ln_gamma,
                    beta: &ln_beta,
                    epsilon: 1e-5,
                },
            )
            .unwrap();
        });
    }

    // Batch norm: 1x64x64x3 (NHWC)
    {
        let bn_input = Tensor::zeros(vec![1, 64, 64, 3]).unwrap();
        let bn_gamma = Tensor::ones(vec![3]).unwrap();
        let bn_beta = Tensor::zeros(vec![3]).unwrap();
        let bn_mean = Tensor::zeros(vec![3]).unwrap();
        let bn_var = Tensor::ones(vec![3]).unwrap();
        bench_n("batch_norm_1x64x64x3", 100, || {
            let _b = yscv_kernels::batch_norm2d_nhwc(
                &bn_input,
                yscv_kernels::BatchNorm2dParams {
                    gamma: &bn_gamma,
                    beta: &bn_beta,
                    mean: &bn_mean,
                    variance: &bn_var,
                    epsilon: 1e-5,
                },
            )
            .unwrap();
        });
    }

    // --- Axis reductions ---
    let mat512 = Tensor::ones(vec![512, 512]).unwrap();
    bench_n("sum_axis0_512x512", 100, || {
        let _s = mat512.sum_axis(0);
    });
    bench_n("mean_axis1_512x512", 100, || {
        let _m = mat512.mean_axis(1);
    });
    bench_n("max_axis0_512x512", 100, || {
        let _m = mat512.max_axis(0).unwrap();
    });

    // --- Transpose ---
    bench_n("transpose_2d_512x512", 100, || {
        let _t = mat512.transpose_2d();
    });

    // --- f32 imgproc (best-of-100, like u8 ops) ---
    println!();
    println!("  --- f32 imgproc ops (best of 100) ---");
    println!("  {}", "-".repeat(53));

    let rgb_img = Tensor::zeros(vec![480, 640, 3]).unwrap();
    let gray_img = Tensor::zeros(vec![480, 640, 1]).unwrap();

    bench_n("f32_grayscale_480x640x3", 100, || {
        let _g = yscv_imgproc::rgb_to_grayscale(&rgb_img).unwrap();
    });
    bench_n("f32_gaussian_blur_3x3_480x640x1", 100, || {
        let _b = yscv_imgproc::gaussian_blur_3x3(&gray_img).unwrap();
    });
    bench_n("f32_box_blur_3x3_480x640x1", 100, || {
        let _b = yscv_imgproc::box_blur_3x3(&gray_img).unwrap();
    });
    bench_n("f32_sobel_3x3_480x640x1", 100, || {
        let _s = yscv_imgproc::sobel_3x3_magnitude(&gray_img).unwrap();
    });
    bench_n("f32_dilate_3x3_480x640x1", 100, || {
        let _d = yscv_imgproc::dilate_3x3(&gray_img).unwrap();
    });
    bench_n("f32_resize_480x640x3_to_240x320", 100, || {
        let img = Tensor::zeros(vec![480, 640, 3]).unwrap();
        let _r = yscv_imgproc::resize_bilinear(&img, 240, 320).unwrap();
    });
    bench_n("f32_threshold_480x640x1", 100, || {
        let _t = yscv_imgproc::threshold_binary(&gray_img, 0.5, 1.0);
    });

    // --- Fast f32 ops (ImageF32, no Tensor overhead) ---
    println!();
    println!("  --- fast f32 ops (ImageF32, best of 100) ---");
    println!("  {}", "-".repeat(53));

    let gray_f32 = yscv_imgproc::ImageF32::zeros(480, 640, 1);
    let rgb_f32 = yscv_imgproc::ImageF32::zeros(480, 640, 3);

    bench_n("f32_fast_grayscale_480x640x3", 100, || {
        let _g = yscv_imgproc::grayscale_f32(&rgb_f32).unwrap();
    });
    bench_n("f32_fast_gauss_3x3_480x640x1", 100, || {
        let _b = yscv_imgproc::gaussian_blur_3x3_f32(&gray_f32).unwrap();
    });
    bench_n("f32_fast_box_3x3_480x640x1", 100, || {
        let _b = yscv_imgproc::box_blur_3x3_f32(&gray_f32).unwrap();
    });
    bench_n("f32_fast_dilate_3x3_480x640x1", 100, || {
        let _d = yscv_imgproc::dilate_3x3_f32(&gray_f32).unwrap();
    });
    bench_n("f32_fast_sobel_3x3_480x640x1", 100, || {
        let _s = yscv_imgproc::sobel_3x3_f32(&gray_f32).unwrap();
    });
    bench_n("f32_fast_threshold_480x640x1", 100, || {
        let _t = yscv_imgproc::threshold_binary_f32(&gray_f32, 0.5, 1.0).unwrap();
    });

    // 19. Autograd forward+backward: matmul + relu + mean
    let n = 64;
    let data_a: Vec<f32> = (0..n * n).map(|i| (i as f32) * 0.001).collect();
    let data_b: Vec<f32> = (0..n * n).map(|i| ((n * n - i) as f32) * 0.001).collect();
    bench("autograd_matmul_relu_mean_64x64", || {
        let mut graph = Graph::new();
        let a = graph.variable(Tensor::from_vec(vec![n, n], data_a.clone()).unwrap());
        let b = graph.variable(Tensor::from_vec(vec![n, n], data_b.clone()).unwrap());
        let c = graph.matmul_2d(a, b).unwrap();
        let d = graph.relu(c).unwrap();
        let loss = graph.mean(d).unwrap();
        graph.backward(loss).unwrap();
    });

    // 20. Autograd forward+backward: 128x128 matmul + relu + mean
    let n2 = 128;
    let data_a2: Vec<f32> = (0..n2 * n2).map(|i| (i as f32) * 0.001).collect();
    let data_b2: Vec<f32> = (0..n2 * n2)
        .map(|i| ((n2 * n2 - i) as f32) * 0.001)
        .collect();
    bench("autograd_matmul_relu_mean_128x128", || {
        let mut graph = Graph::new();
        let a = graph.variable(Tensor::from_vec(vec![n2, n2], data_a2.clone()).unwrap());
        let b = graph.variable(Tensor::from_vec(vec![n2, n2], data_b2.clone()).unwrap());
        let c = graph.matmul_2d(a, b).unwrap();
        let d = graph.relu(c).unwrap();
        let loss = graph.mean(d).unwrap();
        graph.backward(loss).unwrap();
    });

    // 21. Matmul 512x512 (tests blocked tiling)
    let lhs512 = Tensor::zeros(vec![512, 512]).unwrap();
    let rhs512 = Tensor::zeros(vec![512, 512]).unwrap();
    bench("matmul_2d_512x512", || {
        let _c = yscv_kernels::matmul_2d(&lhs512, &rhs512).unwrap();
    });

    // 22. Transpose 512x512
    let t_input =
        Tensor::from_vec(vec![512, 512], (0..512 * 512).map(|i| i as f32).collect()).unwrap();
    bench("transpose_2d_512x512", || {
        let _t = t_input.transpose_2d().unwrap();
    });

    // 23. Autograd forward+backward: 256x256 matmul + relu + mean
    let n3 = 256;
    let data_a3: Vec<f32> = (0..n3 * n3).map(|i| (i as f32) * 0.001).collect();
    let data_b3: Vec<f32> = (0..n3 * n3)
        .map(|i| ((n3 * n3 - i) as f32) * 0.001)
        .collect();
    bench("autograd_matmul_relu_mean_256x256", || {
        let mut graph = Graph::new();
        let a = graph.variable(Tensor::from_vec(vec![n3, n3], data_a3.clone()).unwrap());
        let b = graph.variable(Tensor::from_vec(vec![n3, n3], data_b3.clone()).unwrap());
        let c = graph.matmul_2d(a, b).unwrap();
        let d = graph.relu(c).unwrap();
        let loss = graph.mean(d).unwrap();
        graph.backward(loss).unwrap();
    });

    // 24. RGB to HSV: [480,640,3]
    bench("rgb_to_hsv_480x640x3", || {
        let _h = yscv_imgproc::rgb_to_hsv(&rgb_img).unwrap();
    });

    // 25. Box blur 3x3: [480,640,1]
    bench("box_blur_3x3_480x640x1", || {
        let _b = yscv_imgproc::box_blur_3x3(&gray_img).unwrap();
    });

    // ==================== u8 ops ====================
    println!();
    println!("  --- u8 ops (16 pixels/SIMD register) ---");
    println!("  {}", "-".repeat(53));

    let rgb_u8 = yscv_imgproc::ImageU8::zeros(480, 640, 3);
    let gray_u8 = yscv_imgproc::ImageU8::zeros(480, 640, 1);

    // CPU frequency warm-up: spin for 50ms so Apple Silicon ramps to max clock
    let warmup_start = Instant::now();
    while warmup_start.elapsed().as_millis() < 50 {
        std::hint::black_box(yscv_imgproc::grayscale_u8(&rgb_u8));
    }

    bench_n("u8_grayscale_480x640x3", 100, || {
        let _g = yscv_imgproc::grayscale_u8(&rgb_u8).unwrap();
    });
    bench_n("u8_dilate_3x3_480x640x1", 100, || {
        let _d = yscv_imgproc::dilate_3x3_u8(&gray_u8).unwrap();
    });
    bench_n("u8_erode_3x3_480x640x1", 100, || {
        let _e = yscv_imgproc::erode_3x3_u8(&gray_u8).unwrap();
    });
    bench_n("u8_gaussian_blur_3x3_480x640x1", 100, || {
        let _b = yscv_imgproc::gaussian_blur_3x3_u8(&gray_u8).unwrap();
    });
    bench_n("u8_box_blur_3x3_480x640x1", 100, || {
        let _b = yscv_imgproc::box_blur_3x3_u8(&gray_u8).unwrap();
    });
    bench_n("u8_sobel_3x3_480x640x1", 100, || {
        let _s = yscv_imgproc::sobel_3x3_magnitude_u8(&gray_u8).unwrap();
    });
    bench_n("u8_median_blur_3x3_480x640x1", 100, || {
        let _m = yscv_imgproc::median_blur_3x3_u8(&gray_u8).unwrap();
    });
    bench_n("u8_canny_480x640x1", 100, || {
        let _e = yscv_imgproc::canny_u8(&gray_u8, 30, 100).unwrap();
    });
    let rgb_u8_big = yscv_imgproc::ImageU8::zeros(1080, 1920, 3);
    bench_n("u8_resize_1080p_to_720p_rgb", 100, || {
        let _r = yscv_imgproc::resize_bilinear_u8(&rgb_u8_big, 720, 1280).unwrap();
    });
    bench_n("u8_resize_480x640_to_240x320_gray", 100, || {
        let _r = yscv_imgproc::resize_bilinear_u8(&gray_u8, 240, 320).unwrap();
    });
    let gray_u8_480 = yscv_imgproc::ImageU8::zeros(480, 640, 1);
    bench_n("fast_u8_480x640", 100, || {
        let _corners = yscv_imgproc::fast9_detect_u8(&gray_u8_480, 20, true);
    });
    bench_n("distance_transform_u8_480x640", 100, || {
        let _dt = yscv_imgproc::distance_transform_u8(&gray_u8_480);
    });
    let h_matrix_u8 = [1.0, 0.1, 10.0, -0.1, 1.0, 5.0, 0.0001, 0.0002, 1.0];
    bench_n("warp_u8_480x640", 100, || {
        let _w = yscv_imgproc::warp_perspective_u8(&gray_u8_480, &h_matrix_u8, 480, 640);
    });
    {
        let img_data = vec![128u8; 480 * 640];
        bench_n("bilateral_u8_480x640", 20, || {
            let _b = yscv_imgproc::bilateral_filter_u8(&img_data, 640, 480, 1, 5, 75.0, 75.0);
        });
    }

    // --- In-place operations (no allocation overhead) ---
    println!();
    println!("  --- in-place ops ---");
    println!("  {}", "-".repeat(53));

    let mut a_clone = a1m.clone();
    bench_n("add_1M_inplace", 100, || {
        a_clone.add_inplace(&b1m);
    });

    // --- Video: YUV420 → RGB8 ---
    println!();
    println!("  --- video ops ---");
    println!("  {}", "-".repeat(53));

    let y_1080p = vec![128u8; 1920 * 1080];
    let u_1080p = vec![128u8; 960 * 540];
    let v_1080p = vec![128u8; 960 * 540];
    bench_n("yuv420_to_rgb8_1080p", 100, || {
        let _rgb = yscv_video::yuv420_to_rgb8(&y_1080p, &u_1080p, &v_1080p, 1920, 1080);
    });

    // --- Detection/tracking ---
    println!();
    println!("  --- detect/track ops ---");
    println!("  {}", "-".repeat(53));

    use yscv_detect::{BoundingBox, Detection};
    use yscv_track::{Tracker, TrackerConfig};

    let track_config = TrackerConfig {
        match_iou_threshold: 0.3,
        max_missed_frames: 5,
        max_tracks: 256,
    };
    let mut tracker = Tracker::new(track_config).unwrap();
    let track_dets: Vec<Detection> = (0..32)
        .map(|i| Detection {
            bbox: BoundingBox {
                x1: i as f32 * 20.0,
                y1: 10.0,
                x2: i as f32 * 20.0 + 15.0,
                y2: 25.0,
            },
            score: 0.9,
            class_id: 0,
        })
        .collect();
    bench_n("tracker_update_32det", 100, || {
        let _tracked = tracker.update(&track_dets);
    });

    // --- Additional tensor/kernel ops ---
    println!();
    println!("  --- additional tensor/kernel ops ---");
    println!("  {}", "-".repeat(53));

    // Large matmul 1024x1024
    let m1k = Tensor::zeros(vec![1024, 1024]).unwrap();
    let n1k = Tensor::zeros(vec![1024, 1024]).unwrap();
    bench_n("matmul_1024x1024", 20, || {
        let _c = yscv_kernels::matmul_2d(&m1k, &n1k).unwrap();
    });

    // Even larger matmul 2048x2048
    let m2k = Tensor::zeros(vec![2048, 2048]).unwrap();
    let n2k = Tensor::zeros(vec![2048, 2048]).unwrap();
    bench_n("matmul_2048x2048", 5, || {
        let _c = yscv_kernels::matmul_2d(&m2k, &n2k).unwrap();
    });

    // Cat 10x1000
    let cat_parts: Vec<Tensor> = (0..10)
        .map(|_| Tensor::zeros(vec![1000]).unwrap())
        .collect();
    let cat_refs: Vec<&Tensor> = cat_parts.iter().collect();
    bench_n("cat_10x1000", 100, || {
        let _c = Tensor::cat(&cat_refs, 0).unwrap();
    });

    // Topk 100k, k=100
    let big_tensor = Tensor::from_vec(
        vec![100_000],
        (0..100_000).map(|i| i as f32 * 0.001).collect(),
    )
    .unwrap();
    bench_n("topk_100k_k100", 100, || {
        let _t = big_tensor.topk(100).unwrap();
    });

    // Sort 100k
    bench_n("sort_100k", 20, || {
        let _s = big_tensor.sort(0, false).unwrap();
    });

    // Scaled dot-product attention (2D, unbatched): seq=128, d_k=64
    let attn_q = Tensor::zeros(vec![128, 64]).unwrap();
    let attn_k = Tensor::zeros(vec![128, 64]).unwrap();
    let attn_v = Tensor::zeros(vec![128, 64]).unwrap();
    bench_n("attention_128x64", 100, || {
        let _a =
            yscv_kernels::scaled_dot_product_attention(&attn_q, &attn_k, &attn_v, None).unwrap();
    });

    // --- Optimizer steps ---
    println!();
    println!("  --- optimizer steps ---");
    println!("  {}", "-".repeat(53));

    // SGD step on 1024x1024 weight matrix
    {
        let mut sgd = yscv_optim::Sgd::new(0.01)
            .unwrap()
            .with_momentum(0.9)
            .unwrap();
        let mut weights = Tensor::zeros(vec![1024, 1024]).unwrap();
        let grad = Tensor::from_vec(
            vec![1024, 1024],
            (0..1024 * 1024).map(|i| i as f32 * 1e-6).collect(),
        )
        .unwrap();
        bench_n("sgd_step_1024x1024", 100, || {
            sgd.step(0, &mut weights, &grad).unwrap();
        });
    }

    // Adam step on 1024x1024 weight matrix
    {
        let mut adam = yscv_optim::Adam::new(0.001).unwrap();
        let mut weights = Tensor::zeros(vec![1024, 1024]).unwrap();
        let grad = Tensor::from_vec(
            vec![1024, 1024],
            (0..1024 * 1024).map(|i| i as f32 * 1e-6).collect(),
        )
        .unwrap();
        bench_n("adam_step_1024x1024", 100, || {
            adam.step(0, &mut weights, &grad).unwrap();
        });
    }

    // --- Unique yscv operations (no Python equivalent) ---
    println!();
    println!("  --- unique yscv ops ---");
    println!("  {}", "-".repeat(53));

    // NMS: 1000 boxes
    {
        let boxes: Vec<Detection> = (0..1000)
            .map(|i| Detection {
                bbox: BoundingBox {
                    x1: (i % 30) as f32 * 20.0,
                    y1: (i / 30) as f32 * 20.0,
                    x2: (i % 30) as f32 * 20.0 + 15.0,
                    y2: (i / 30) as f32 * 20.0 + 15.0,
                },
                score: 1.0 - i as f32 * 0.001,
                class_id: 0,
            })
            .collect();
        bench_n("nms_1000_boxes", 100, || {
            let _kept = yscv_detect::non_max_suppression(&boxes, 0.5, 1000);
        });
    }

    // IoU computation
    {
        let a = BoundingBox {
            x1: 0.0,
            y1: 0.0,
            x2: 10.0,
            y2: 10.0,
        };
        let b = BoundingBox {
            x1: 5.0,
            y1: 5.0,
            x2: 15.0,
            y2: 15.0,
        };
        bench_n("iou_single", 100, || {
            std::hint::black_box(yscv_detect::iou(a, b));
        });
    }

    // H.264 SPS parsing
    {
        let sps_nal = vec![
            0x42, 0x00, 0x1e, 0xab, 0x40, 0x50, 0x1e, 0xd0, 0x40, 0x00, 0x00, 0x03, 0x00, 0x40,
        ];
        bench_n("h264_parse_sps", 100, || {
            let _sps = yscv_video::parse_sps(&sps_nal);
        });
    }

    // Kalman filter predict+update cycle
    {
        use yscv_track::KalmanFilter;
        let bbox = BoundingBox {
            x1: 100.0,
            y1: 100.0,
            x2: 150.0,
            y2: 150.0,
        };
        let measurement = [130.0f32, 130.0, 50.0, 50.0];
        bench_n("kalman_predict_update", 100, || {
            let mut kf = KalmanFilter::new(bbox);
            kf.predict();
            kf.update(measurement);
            std::hint::black_box(kf.bbox());
        });
    }

    // FAST corners: 480x640 grayscale
    {
        let gray_for_fast = Tensor::zeros(vec![480, 640, 1]).unwrap();
        bench_n("fast_corners_480x640", 100, || {
            let _corners = yscv_imgproc::fast9_detect(&gray_for_fast, 20.0 / 255.0, true);
        });
    }

    // ORB features: 480x640 grayscale
    {
        let gray_for_orb = Tensor::zeros(vec![480, 640, 1]).unwrap();
        let orb_config = yscv_imgproc::OrbConfig::default();
        bench_n("orb_detect_480x640", 100, || {
            let _feats = yscv_imgproc::detect_orb(&gray_for_orb, &orb_config);
        });
    }

    // Histogram: 480x640 grayscale
    {
        let gray_for_hist = Tensor::zeros(vec![480, 640, 1]).unwrap();
        bench_n("histogram_480x640", 100, || {
            let _h = yscv_imgproc::histogram_256(&gray_for_hist);
        });
    }

    // CLAHE: 480x640 grayscale
    {
        let gray_for_clahe = Tensor::zeros(vec![480, 640, 1]).unwrap();
        bench_n("clahe_480x640", 100, || {
            let _c = yscv_imgproc::clahe(&gray_for_clahe, 8, 8, 40.0);
        });
    }

    // Distance transform: 480x640 grayscale
    {
        let gray_for_dt = Tensor::zeros(vec![480, 640, 1]).unwrap();
        bench_n("distance_transform_480x640", 100, || {
            let _d = yscv_imgproc::distance_transform(&gray_for_dt);
        });
    }

    // Soft-NMS: 1000 boxes
    {
        bench_n("soft_nms_1000_boxes", 100, || {
            let mut boxes: Vec<Detection> = (0..1000)
                .map(|i| Detection {
                    bbox: BoundingBox {
                        x1: (i % 30) as f32 * 20.0,
                        y1: (i / 30) as f32 * 20.0,
                        x2: (i % 30) as f32 * 20.0 + 15.0,
                        y2: (i / 30) as f32 * 20.0 + 15.0,
                    },
                    score: 1.0 - i as f32 * 0.001,
                    class_id: 0,
                })
                .collect();
            yscv_detect::soft_nms(&mut boxes, 0.5, 0.01);
        });
    }

    // RoI Pool: 64x64x16 feature map, 32 rois, 7x7 output
    {
        let feature_map = Tensor::zeros(vec![64, 64, 16]).unwrap();
        let rois: Vec<(f32, f32, f32, f32)> = (0..32)
            .map(|i| {
                let x1 = (i % 8) as f32 * 7.0;
                let y1 = (i / 8) as f32 * 7.0;
                (x1, y1, x1 + 14.0, y1 + 14.0)
            })
            .collect();
        bench_n("roi_pool_64x64x16_32rois", 100, || {
            let _r = yscv_detect::roi_pool(&feature_map, &rois, (7, 7));
        });
    }

    // Warp perspective: 480x640 grayscale
    {
        let gray_for_warp = Tensor::zeros(vec![480, 640, 1]).unwrap();
        // Identity-ish transform (slight rotation)
        let transform: [f32; 9] = [0.99, -0.01, 5.0, 0.01, 0.99, 3.0, 0.0, 0.0, 1.0];
        bench_n("warp_perspective_480x640", 100, || {
            let _w = yscv_imgproc::warp_perspective(&gray_for_warp, &transform, 480, 640, 0.0);
        });
    }

    // --- additional imgproc ops ---
    println!();
    println!("  --- additional imgproc ops ---");
    println!("  -----------------------------------------------------");

    // Optical flow (Farneback): 64x64 grayscale (small due to O(n²))
    {
        let prev = Tensor::from_vec(vec![64, 64], vec![0.5f32; 64 * 64]).unwrap();
        let mut next_data = vec![0.5f32; 64 * 64];
        // Add slight shift
        for y in 0..64 {
            for x in 1..64 {
                next_data[y * 64 + x] = 0.5 + (x as f32 - 32.0) * 0.01;
            }
        }
        let next = Tensor::from_vec(vec![64, 64], next_data).unwrap();
        let config = yscv_imgproc::FarnebackConfig {
            levels: 2,
            iterations: 2,
            win_size: 5,
            ..Default::default()
        };
        bench_n("farneback_flow_64x64", 20, || {
            let _r = yscv_imgproc::farneback_flow(&prev, &next, &config);
        });
    }

    // Harris corners: 480x640 grayscale (f32)
    {
        let gray = Tensor::from_vec(
            vec![480, 640, 1],
            (0..480 * 640).map(|i| ((i % 256) as f32) / 255.0).collect(),
        )
        .unwrap();
        bench_n("harris_corners_480x640", 20, || {
            let _c = yscv_imgproc::harris_corners(&gray, 3, 0.04, 0.01);
        });
    }

    // Harris corners u8: 480x640 grayscale (u8-native, integer Sobel)
    {
        let gray_data: Vec<u8> = (0..480 * 640).map(|i| (i % 256) as u8).collect();
        bench_n("harris_u8_480x640", 100, || {
            let _h = yscv_imgproc::harris_corners_u8(&gray_data, 640, 480, 3, 0.04, 100.0);
        });
    }

    // Histogram equalization: 480x640 grayscale
    {
        let gray = Tensor::from_vec(
            vec![480, 640, 1],
            (0..480 * 640).map(|i| ((i % 256) as f32) / 255.0).collect(),
        )
        .unwrap();
        bench_n("histogram_equalize_480x640", 100, || {
            let _h = yscv_imgproc::histogram_equalize(&gray);
        });
    }

    // --- kernel ops ---
    println!();
    println!("  --- kernel ops ---");
    println!("  -----------------------------------------------------");

    // Flash attention: 128 queries × 64 d_k vs standard attention
    {
        let q = Tensor::from_vec(vec![128, 64], vec![0.1f32; 128 * 64]).unwrap();
        let k = Tensor::from_vec(vec![128, 64], vec![0.2f32; 128 * 64]).unwrap();
        let v = Tensor::from_vec(vec![128, 64], vec![0.3f32; 128 * 64]).unwrap();
        bench_n("attention_standard_128x64", 100, || {
            let _a = yscv_kernels::scaled_dot_product_attention(&q, &k, &v, None);
        });
        bench_n("attention_flash_128x64", 100, || {
            let _a = yscv_kernels::flash_attention(&q, &k, &v, None);
        });
    }

    // Conv3d: [1, 8, 16, 16, 3] with [3, 3, 3, 3, 16] kernel
    {
        let input = vec![0.5f32; 8 * 16 * 16 * 3];
        let kernel = vec![0.1f32; 3 * 3 * 3 * 3 * 16];
        bench_n("conv3d_8x16x16_3x3x3", 20, || {
            let _c = yscv_kernels::conv3d(
                &input,
                &[1, 8, 16, 16, 3],
                &kernel,
                &[3, 3, 3, 3, 16],
                (1, 1, 1),
                (0, 0, 0),
            );
        });
    }

    // --- color space ops ---
    println!();
    println!("  --- color space ops ---");
    println!("  -----------------------------------------------------");
    {
        let rgb = Tensor::from_vec(
            vec![480, 640, 3],
            (0..480 * 640 * 3)
                .map(|i| ((i % 256) as f32) / 255.0)
                .collect(),
        )
        .unwrap();
        bench_n("rgb_to_lab_480x640", 20, || {
            let _l = yscv_imgproc::rgb_to_lab(&rgb);
        });
        bench_n("rgb_to_yuv_480x640", 100, || {
            let _y = yscv_imgproc::rgb_to_yuv(&rgb);
        });
        bench_n("rgb_to_bgr_480x640", 100, || {
            let _b = yscv_imgproc::rgb_to_bgr(&rgb);
        });
    }

    // --- contour/geometry ops ---
    println!();
    println!("  --- contour/geometry ops ---");
    println!("  -----------------------------------------------------");
    {
        let binary = Tensor::from_vec(
            vec![240, 320, 1],
            (0..240 * 320)
                .map(|i| {
                    if (i / 320 + i % 320) % 40 < 20 {
                        1.0f32
                    } else {
                        0.0
                    }
                })
                .collect(),
        )
        .unwrap();
        bench_n("find_contours_240x320", 20, || {
            let _c = yscv_imgproc::find_contours(&binary);
        });
        bench_n("bilateral_filter_480x640", 5, || {
            let gray = Tensor::from_vec(
                vec![480, 640, 1],
                (0..480 * 640).map(|i| ((i % 256) as f32) / 255.0).collect(),
            )
            .unwrap();
            let _b = yscv_imgproc::bilateral_filter(&gray, 5, 75.0, 75.0);
        });
    }

    // --- hough + optical flow ---
    println!();
    println!("  --- hough + optical flow ---");
    println!("  -----------------------------------------------------");
    {
        // Create a simple edge image for hough
        let mut edge_data = vec![0.0f32; 240 * 320];
        for y in 100..110 {
            for x in 0..320 {
                edge_data[y * 320 + x] = 1.0;
            }
        }
        let edges = Tensor::from_vec(vec![240, 320, 1], edge_data).unwrap();
        bench_n("hough_lines_240x320", 20, || {
            let _h = yscv_imgproc::hough_lines(&edges, 1.0, std::f32::consts::PI / 180.0, 50);
        });

        let prev = Tensor::from_vec(
            vec![64, 64, 1],
            (0..64 * 64).map(|i| ((i % 256) as f32) / 255.0).collect(),
        )
        .unwrap();
        let next = Tensor::from_vec(
            vec![64, 64, 1],
            (0..64 * 64)
                .map(|i| (((i + 3) % 256) as f32) / 255.0)
                .collect(),
        )
        .unwrap();
        let points: Vec<(usize, usize)> = (0..50).map(|i| (10 + i % 44, 10 + i / 5)).collect();
        bench_n("lucas_kanade_64x64_50pts", 20, || {
            let _f = yscv_imgproc::lucas_kanade_optical_flow(&prev, &next, &points, 15);
        });
    }

    // --- u8-native ops (vs OpenCV) ---
    println!();
    println!("  --- u8-native ops (vs OpenCV) ---");
    println!("  -----------------------------------------------------");

    {
        let rgb_data: Vec<u8> = (0..480 * 640 * 3).map(|i| (i % 256) as u8).collect();
        let gray_data: Vec<u8> = (0..480 * 640).map(|i| (i % 256) as u8).collect();

        bench_n("hsv_u8_480x640", 100, || {
            let _h = yscv_imgproc::rgb_to_hsv_u8(&rgb_data, 640, 480);
        });
        bench_n("histogram_u8_480x640", 100, || {
            let _h = yscv_imgproc::histogram_u8(&gray_data, 480 * 640);
        });
        bench_n("clahe_u8_480x640", 100, || {
            let _c = yscv_imgproc::clahe_u8(&gray_data, 640, 480, 8, 8, 40.0);
        });
    }

    println!();
    println!("Done.");
}
