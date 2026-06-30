use std::num::NonZeroUsize;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use yscv_kernels::{
    Backend, BatchNorm2dParams, LayerNormLastDimParams, ParallelElementwiseConfig,
    ParallelMatmulConfig, SeparableConv2dParams, ThreadedCpuBackend, ThreadedCpuBackendConfig, add,
    avg_pool2d_nhwc, batch_norm2d_nhwc, conv2d_nhwc, conv2d_nhwc_indirect_padded,
    conv2d_nhwc_padded, depthwise_conv2d_nhwc, layer_norm_last_dim, log_softmax_last_dim,
    logsumexp_last_dim, matmul_2d, matmul_2d_sequential, max_pool2d_nhwc, relu,
    separable_conv2d_nhwc, sigmoid, softmax_last_dim,
};
use yscv_tensor::Tensor;

fn build_tensor(shape: &[usize], seed: f32) -> Tensor {
    let len = shape.iter().copied().product::<usize>();
    let mut data = Vec::with_capacity(len);
    for idx in 0..len {
        data.push(((idx % 251) as f32 * 0.0041 + seed).fract());
    }
    Tensor::from_vec(shape.to_vec(), data).expect("valid tensor")
}

fn bench_matmul_modes(c: &mut Criterion) {
    let lhs_square = build_tensor(&[128, 128], 0.11);
    let rhs_square = build_tensor(&[128, 128], 0.37);

    let lhs_rect = build_tensor(&[96, 192], 0.23);
    let rhs_rect = build_tensor(&[192, 64], 0.59);

    let threaded_backend = ThreadedCpuBackend::with_full_config(
        NonZeroUsize::new(2).expect("benchmark thread count should be non-zero"),
        ThreadedCpuBackendConfig {
            matmul: ParallelMatmulConfig::default(),
            elementwise: ParallelElementwiseConfig::disabled(),
        },
    )
    .expect("thread pool should initialize");
    let no_parallel = ParallelMatmulConfig::disabled();

    let mut group = c.benchmark_group("kernels_matmul_modes");

    group.bench_function("square_128", |b| {
        b.iter(|| {
            let out =
                matmul_2d(black_box(&lhs_square), black_box(&rhs_square)).expect("matmul square");
            black_box(out);
        });
    });

    group.bench_function("square_128_sequential", |b| {
        b.iter(|| {
            let out = matmul_2d_sequential(black_box(&lhs_square), black_box(&rhs_square))
                .expect("matmul square sequential");
            black_box(out);
        });
    });

    group.bench_function("square_128_threaded_2", |b| {
        b.iter(|| {
            let out = threaded_backend
                .matmul_2d(black_box(&lhs_square), black_box(&rhs_square))
                .expect("matmul square threaded");
            black_box(out);
        });
    });

    group.bench_function("rect_96x192x64", |b| {
        b.iter(|| {
            let out = matmul_2d(black_box(&lhs_rect), black_box(&rhs_rect)).expect("matmul rect");
            black_box(out);
        });
    });

    group.bench_function("rect_96x192x64_forced_sequential", |b| {
        b.iter(|| {
            let out = yscv_kernels::matmul_2d_with_config(
                black_box(&lhs_rect),
                black_box(&rhs_rect),
                no_parallel,
            )
            .expect("matmul rect forced sequential");
            black_box(out);
        });
    });

    group.finish();
}

fn bench_relu_mode(c: &mut Criterion) {
    let input = build_tensor(&[480, 640, 3], 0.44);
    let threaded_backend = ThreadedCpuBackend::with_full_config(
        NonZeroUsize::new(2).expect("benchmark thread count should be non-zero"),
        ThreadedCpuBackendConfig {
            matmul: ParallelMatmulConfig::disabled(),
            elementwise: ParallelElementwiseConfig {
                min_parallel_elements: 1,
            },
        },
    )
    .expect("thread pool should initialize");

    let mut group = c.benchmark_group("kernels_relu_modes");
    group.bench_function("relu_640x480x3", |b| {
        b.iter(|| {
            let out = relu(black_box(&input));
            black_box(out);
        });
    });
    group.bench_function("relu_640x480x3_threaded_2", |b| {
        b.iter(|| {
            let out = threaded_backend.relu(black_box(&input));
            black_box(out);
        });
    });
    group.finish();
}

fn bench_sigmoid_modes(c: &mut Criterion) {
    let input = build_tensor(&[480, 640, 3], 0.27);
    let threaded_backend = ThreadedCpuBackend::with_full_config(
        NonZeroUsize::new(2).expect("benchmark thread count should be non-zero"),
        ThreadedCpuBackendConfig {
            matmul: ParallelMatmulConfig::disabled(),
            elementwise: ParallelElementwiseConfig {
                min_parallel_elements: 1,
            },
        },
    )
    .expect("thread pool should initialize");

    let mut group = c.benchmark_group("kernels_sigmoid_modes");
    group.bench_function("sigmoid_640x480x3", |b| {
        b.iter(|| {
            let out = sigmoid(black_box(&input));
            black_box(out);
        });
    });
    group.bench_function("sigmoid_640x480x3_threaded_2", |b| {
        b.iter(|| {
            let out = threaded_backend.sigmoid(black_box(&input));
            black_box(out);
        });
    });
    group.finish();
}

fn bench_elementwise_modes(c: &mut Criterion) {
    let lhs = build_tensor(&[480, 640, 3], 0.17);
    let rhs = build_tensor(&[480, 640, 3], 0.63);
    let rhs_broadcast = Tensor::from_vec(vec![3], vec![0.1, 0.2, 0.3]).expect("valid tensor");

    let threaded_backend = ThreadedCpuBackend::with_full_config(
        NonZeroUsize::new(2).expect("benchmark thread count should be non-zero"),
        ThreadedCpuBackendConfig {
            matmul: ParallelMatmulConfig::disabled(),
            elementwise: ParallelElementwiseConfig {
                min_parallel_elements: 1,
            },
        },
    )
    .expect("thread pool should initialize");

    let mut group = c.benchmark_group("kernels_elementwise_modes");
    group.sample_size(10);
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(2));
    group.bench_function("add_same_shape", |b| {
        b.iter(|| {
            let out = add(black_box(&lhs), black_box(&rhs)).expect("add same shape");
            black_box(out);
        });
    });
    group.bench_function("add_same_shape_threaded_2", |b| {
        b.iter(|| {
            let out = threaded_backend
                .add(black_box(&lhs), black_box(&rhs))
                .expect("add same shape threaded");
            black_box(out);
        });
    });
    group.bench_function("add_broadcast", |b| {
        b.iter(|| {
            let out = add(black_box(&lhs), black_box(&rhs_broadcast)).expect("add broadcast");
            black_box(out);
        });
    });
    group.finish();
}

fn bench_pool_modes(c: &mut Criterion) {
    let input = build_tensor(&[2, 120, 160, 3], 0.33);

    let threaded_backend = ThreadedCpuBackend::with_full_config(
        NonZeroUsize::new(2).expect("benchmark thread count should be non-zero"),
        ThreadedCpuBackendConfig {
            matmul: ParallelMatmulConfig::disabled(),
            elementwise: ParallelElementwiseConfig {
                min_parallel_elements: 1,
            },
        },
    )
    .expect("thread pool should initialize");

    let mut group = c.benchmark_group("kernels_pool_modes");
    group.sample_size(10);
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(2));
    group.bench_function("max_pool2d_nhwc_2x2_s2", |b| {
        b.iter(|| {
            let out = max_pool2d_nhwc(black_box(&input), 2, 2, 2, 2).expect("max pool");
            black_box(out);
        });
    });
    group.bench_function("avg_pool2d_nhwc_2x2_s2", |b| {
        b.iter(|| {
            let out = avg_pool2d_nhwc(black_box(&input), 2, 2, 2, 2).expect("avg pool");
            black_box(out);
        });
    });
    group.bench_function("max_pool2d_nhwc_2x2_s2_threaded_2", |b| {
        b.iter(|| {
            let out = threaded_backend
                .max_pool2d_nhwc(black_box(&input), 2, 2, 2, 2)
                .expect("max pool threaded");
            black_box(out);
        });
    });
    group.bench_function("avg_pool2d_nhwc_2x2_s2_threaded_2", |b| {
        b.iter(|| {
            let out = threaded_backend
                .avg_pool2d_nhwc(black_box(&input), 2, 2, 2, 2)
                .expect("avg pool threaded");
            black_box(out);
        });
    });
    group.finish();
}

fn bench_conv_modes(c: &mut Criterion) {
    let input = build_tensor(&[1, 32, 32, 8], 0.42);
    let kernel = build_tensor(&[3, 3, 8, 16], 0.86);
    let bias = build_tensor(&[16], 0.19);

    let threaded_backend = ThreadedCpuBackend::with_full_config(
        NonZeroUsize::new(2).expect("benchmark thread count should be non-zero"),
        ThreadedCpuBackendConfig {
            matmul: ParallelMatmulConfig::disabled(),
            elementwise: ParallelElementwiseConfig {
                min_parallel_elements: 1,
            },
        },
    )
    .expect("thread pool should initialize");

    let mut group = c.benchmark_group("kernels_conv_modes");
    group.bench_function("conv2d_nhwc_3x3_s1", |b| {
        b.iter(|| {
            let out = conv2d_nhwc(
                black_box(&input),
                black_box(&kernel),
                Some(black_box(&bias)),
                1,
                1,
            )
            .expect("conv2d");
            black_box(out);
        });
    });
    group.bench_function("conv2d_nhwc_3x3_s1_threaded_2", |b| {
        b.iter(|| {
            let out = threaded_backend
                .conv2d_nhwc(
                    black_box(&input),
                    black_box(&kernel),
                    Some(black_box(&bias)),
                    1,
                    1,
                )
                .expect("conv2d threaded");
            black_box(out);
        });
    });
    group.finish();
}

fn bench_winograd_conv_modes(c: &mut Criterion) {
    let small_input = build_tensor(&[1, 32, 32, 8], 0.43);
    let small_kernel = build_tensor(&[3, 3, 8, 16], 0.87);
    let small_bias = build_tensor(&[16], 0.21);

    let yolo_p3_input = build_tensor(&[1, 80, 80, 128], 0.49);
    let yolo_p3_kernel = build_tensor(&[3, 3, 128, 256], 0.83);
    let yolo_p3_bias = build_tensor(&[256], 0.27);

    let mut group = c.benchmark_group("kernels_winograd_conv_modes");
    group.bench_function("winograd_3x3_s1_32x32x8_to16", |b| {
        b.iter(|| {
            let out = conv2d_nhwc_padded(
                black_box(&small_input),
                black_box(&small_kernel),
                Some(black_box(&small_bias)),
                1,
                1,
                0,
                0,
                0,
                0,
                yscv_kernels::Activation::Relu,
            )
            .expect("winograd conv2d small");
            black_box(out);
        });
    });
    group.bench_function("winograd_3x3_s1_yolo_p3_80x80x128_to256", |b| {
        b.iter(|| {
            let out = conv2d_nhwc_padded(
                black_box(&yolo_p3_input),
                black_box(&yolo_p3_kernel),
                Some(black_box(&yolo_p3_bias)),
                1,
                1,
                0,
                0,
                0,
                0,
                yscv_kernels::Activation::Relu,
            )
            .expect("winograd conv2d yolo p3");
            black_box(out);
        });
    });
    group.bench_function("indirect_3x3_s1_yolo_p3_80x80x128_to256", |b| {
        b.iter(|| {
            let out = conv2d_nhwc_indirect_padded(
                black_box(&yolo_p3_input),
                black_box(&yolo_p3_kernel),
                Some(black_box(&yolo_p3_bias)),
                1,
                1,
                0,
                0,
                0,
                0,
                yscv_kernels::Activation::Relu,
            )
            .expect("indirect conv2d yolo p3");
            black_box(out);
        });
    });
    group.finish();
}

fn bench_depthwise_conv_modes(c: &mut Criterion) {
    let input = build_tensor(&[1, 32, 32, 8], 0.28);
    let kernel = build_tensor(&[3, 3, 8, 2], 0.74);
    let bias = build_tensor(&[16], 0.52);

    let threaded_backend = ThreadedCpuBackend::with_full_config(
        NonZeroUsize::new(2).expect("benchmark thread count should be non-zero"),
        ThreadedCpuBackendConfig {
            matmul: ParallelMatmulConfig::disabled(),
            elementwise: ParallelElementwiseConfig {
                min_parallel_elements: 1,
            },
        },
    )
    .expect("thread pool should initialize");

    let mut group = c.benchmark_group("kernels_depthwise_conv_modes");
    group.bench_function("depthwise_conv2d_nhwc_3x3_s1_dm2", |b| {
        b.iter(|| {
            let out = depthwise_conv2d_nhwc(
                black_box(&input),
                black_box(&kernel),
                Some(black_box(&bias)),
                1,
                1,
            )
            .expect("depthwise conv2d");
            black_box(out);
        });
    });
    group.bench_function("depthwise_conv2d_nhwc_3x3_s1_dm2_threaded_2", |b| {
        b.iter(|| {
            let out = threaded_backend
                .depthwise_conv2d_nhwc(
                    black_box(&input),
                    black_box(&kernel),
                    Some(black_box(&bias)),
                    1,
                    1,
                )
                .expect("depthwise conv2d threaded");
            black_box(out);
        });
    });
    group.finish();
}

fn bench_separable_conv_modes(c: &mut Criterion) {
    let input = build_tensor(&[1, 32, 32, 8], 0.39);
    let depthwise_kernel = build_tensor(&[3, 3, 8, 2], 0.77);
    let depthwise_bias = build_tensor(&[16], 0.25);
    let pointwise_kernel = build_tensor(&[1, 1, 16, 16], 0.62);
    let pointwise_bias = build_tensor(&[16], 0.46);

    let threaded_backend = ThreadedCpuBackend::with_full_config(
        NonZeroUsize::new(2).expect("benchmark thread count should be non-zero"),
        ThreadedCpuBackendConfig {
            matmul: ParallelMatmulConfig::disabled(),
            elementwise: ParallelElementwiseConfig {
                min_parallel_elements: 1,
            },
        },
    )
    .expect("thread pool should initialize");

    let mut group = c.benchmark_group("kernels_separable_conv_modes");
    group.bench_function("separable_conv2d_nhwc_3x3_s1_dm2_pw16", |b| {
        b.iter(|| {
            let out = separable_conv2d_nhwc(
                black_box(&input),
                SeparableConv2dParams {
                    depthwise_kernel: black_box(&depthwise_kernel),
                    depthwise_bias: Some(black_box(&depthwise_bias)),
                    pointwise_kernel: black_box(&pointwise_kernel),
                    pointwise_bias: Some(black_box(&pointwise_bias)),
                },
                1,
                1,
            )
            .expect("separable conv2d");
            black_box(out);
        });
    });
    group.bench_function("separable_conv2d_nhwc_3x3_s1_dm2_pw16_threaded_2", |b| {
        b.iter(|| {
            let out = threaded_backend
                .separable_conv2d_nhwc(
                    black_box(&input),
                    SeparableConv2dParams {
                        depthwise_kernel: black_box(&depthwise_kernel),
                        depthwise_bias: Some(black_box(&depthwise_bias)),
                        pointwise_kernel: black_box(&pointwise_kernel),
                        pointwise_bias: Some(black_box(&pointwise_bias)),
                    },
                    1,
                    1,
                )
                .expect("separable conv2d threaded");
            black_box(out);
        });
    });
    group.finish();
}

fn bench_batch_norm_modes(c: &mut Criterion) {
    let input = build_tensor(&[2, 64, 64, 16], 0.57);
    let gamma = build_tensor(&[16], 0.29);
    let beta = build_tensor(&[16], 0.73);
    let mean = build_tensor(&[16], 0.34);
    let variance = build_tensor(&[16], 1.17);

    let threaded_backend = ThreadedCpuBackend::with_full_config(
        NonZeroUsize::new(2).expect("benchmark thread count should be non-zero"),
        ThreadedCpuBackendConfig {
            matmul: ParallelMatmulConfig::disabled(),
            elementwise: ParallelElementwiseConfig {
                min_parallel_elements: 1,
            },
        },
    )
    .expect("thread pool should initialize");

    let mut group = c.benchmark_group("kernels_batch_norm_modes");
    group.sample_size(10);
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(2));
    group.bench_function("batch_norm2d_nhwc_128x128x16", |b| {
        b.iter(|| {
            let out = batch_norm2d_nhwc(
                black_box(&input),
                BatchNorm2dParams {
                    gamma: black_box(&gamma),
                    beta: black_box(&beta),
                    mean: black_box(&mean),
                    variance: black_box(&variance),
                    epsilon: black_box(1e-5),
                },
            )
            .expect("batch norm");
            black_box(out);
        });
    });
    group.bench_function("batch_norm2d_nhwc_128x128x16_threaded_2", |b| {
        b.iter(|| {
            let out = threaded_backend
                .batch_norm2d_nhwc(
                    black_box(&input),
                    BatchNorm2dParams {
                        gamma: black_box(&gamma),
                        beta: black_box(&beta),
                        mean: black_box(&mean),
                        variance: black_box(&variance),
                        epsilon: black_box(1e-5),
                    },
                )
                .expect("batch norm threaded");
            black_box(out);
        });
    });
    group.finish();
}

fn bench_softmax_modes(c: &mut Criterion) {
    let input = build_tensor(&[512, 256], 0.64);
    let threaded_backend = ThreadedCpuBackend::with_full_config(
        NonZeroUsize::new(2).expect("benchmark thread count should be non-zero"),
        ThreadedCpuBackendConfig {
            matmul: ParallelMatmulConfig::disabled(),
            elementwise: ParallelElementwiseConfig {
                min_parallel_elements: 1,
            },
        },
    )
    .expect("thread pool should initialize");

    let mut group = c.benchmark_group("kernels_softmax_modes");
    group.bench_function("softmax_last_dim_512x256", |b| {
        b.iter(|| {
            let out = softmax_last_dim(black_box(&input)).expect("softmax");
            black_box(out);
        });
    });
    group.bench_function("softmax_last_dim_512x256_threaded_2", |b| {
        b.iter(|| {
            let out = threaded_backend
                .softmax_last_dim(black_box(&input))
                .expect("softmax threaded");
            black_box(out);
        });
    });
    group.finish();
}

fn bench_layer_norm_modes(c: &mut Criterion) {
    let input = build_tensor(&[512, 256], 0.42);
    let gamma = build_tensor(&[256], 0.31);
    let beta = build_tensor(&[256], 0.79);
    let threaded_backend = ThreadedCpuBackend::with_full_config(
        NonZeroUsize::new(2).expect("benchmark thread count should be non-zero"),
        ThreadedCpuBackendConfig {
            matmul: ParallelMatmulConfig::disabled(),
            elementwise: ParallelElementwiseConfig {
                min_parallel_elements: 1,
            },
        },
    )
    .expect("thread pool should initialize");

    let mut group = c.benchmark_group("kernels_layer_norm_modes");
    group.bench_function("layer_norm_last_dim_512x256", |b| {
        b.iter(|| {
            let out = layer_norm_last_dim(
                black_box(&input),
                LayerNormLastDimParams {
                    gamma: black_box(&gamma),
                    beta: black_box(&beta),
                    epsilon: black_box(1e-5),
                },
            )
            .expect("layer norm");
            black_box(out);
        });
    });
    group.bench_function("layer_norm_last_dim_512x256_threaded_2", |b| {
        b.iter(|| {
            let out = threaded_backend
                .layer_norm_last_dim(
                    black_box(&input),
                    LayerNormLastDimParams {
                        gamma: black_box(&gamma),
                        beta: black_box(&beta),
                        epsilon: black_box(1e-5),
                    },
                )
                .expect("layer norm threaded");
            black_box(out);
        });
    });
    group.finish();
}

fn bench_log_softmax_modes(c: &mut Criterion) {
    let input = build_tensor(&[512, 256], 0.57);
    let threaded_backend = ThreadedCpuBackend::with_full_config(
        NonZeroUsize::new(2).expect("benchmark thread count should be non-zero"),
        ThreadedCpuBackendConfig {
            matmul: ParallelMatmulConfig::disabled(),
            elementwise: ParallelElementwiseConfig {
                min_parallel_elements: 1,
            },
        },
    )
    .expect("thread pool should initialize");

    let mut group = c.benchmark_group("kernels_log_softmax_modes");
    group.bench_function("log_softmax_last_dim_512x256", |b| {
        b.iter(|| {
            let out = log_softmax_last_dim(black_box(&input)).expect("log softmax");
            black_box(out);
        });
    });
    group.bench_function("log_softmax_last_dim_512x256_threaded_2", |b| {
        b.iter(|| {
            let out = threaded_backend
                .log_softmax_last_dim(black_box(&input))
                .expect("log softmax threaded");
            black_box(out);
        });
    });
    group.finish();
}

fn bench_logsumexp_modes(c: &mut Criterion) {
    let input = build_tensor(&[512, 256], 0.35);
    let threaded_backend = ThreadedCpuBackend::with_full_config(
        NonZeroUsize::new(2).expect("benchmark thread count should be non-zero"),
        ThreadedCpuBackendConfig {
            matmul: ParallelMatmulConfig::disabled(),
            elementwise: ParallelElementwiseConfig {
                min_parallel_elements: 1,
            },
        },
    )
    .expect("thread pool should initialize");

    let mut group = c.benchmark_group("kernels_logsumexp_modes");
    group.bench_function("logsumexp_last_dim_512x256", |b| {
        b.iter(|| {
            let out = logsumexp_last_dim(black_box(&input)).expect("logsumexp");
            black_box(out);
        });
    });
    group.bench_function("logsumexp_last_dim_512x256_threaded_2", |b| {
        b.iter(|| {
            let out = threaded_backend
                .logsumexp_last_dim(black_box(&input))
                .expect("logsumexp threaded");
            black_box(out);
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_matmul_modes,
    bench_relu_mode,
    bench_sigmoid_modes,
    bench_elementwise_modes,
    bench_pool_modes,
    bench_conv_modes,
    bench_winograd_conv_modes,
    bench_depthwise_conv_modes,
    bench_separable_conv_modes,
    bench_batch_norm_modes,
    bench_softmax_modes,
    bench_layer_norm_modes,
    bench_log_softmax_modes,
    bench_logsumexp_modes
);
criterion_main!(benches);
