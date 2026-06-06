use std::time::Instant;

use yscv_kernels::{
    BatchNorm2dParams, LayerNormLastDimParams, ParallelElementwiseConfig, add_out,
    add_reduce_dispatch, add_with_config, batch_norm2d_nhwc, exp_with_config, gelu,
    layer_norm_last_dim, log_softmax_last_dim, log_softmax_last_dim_out, max_reduce_dispatch,
    mul_out, relu, relu_out_with_config, relu_to_slice_dispatch, sigmoid, sigmoid_with_config,
    silu, silu_inplace, softmax_last_dim, softmax_last_dim_out, sub_with_config, tanh_act,
    tanh_act_with_config,
};
use yscv_tensor::Tensor;

#[derive(Debug, Clone, Copy)]
struct Stats {
    min_us: u128,
    p50_us: u128,
    avg_us: u128,
}

#[derive(Debug, Clone)]
struct BenchArgs {
    warmup: usize,
    iters: usize,
    filter: Option<String>,
}

impl BenchArgs {
    fn parse() -> Self {
        let mut warmup = 10usize;
        let mut iters = 200usize;
        let mut filter = None;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--warmup" => {
                    let value = args.next().expect("--warmup needs a value");
                    warmup = value.parse().expect("--warmup must be a usize");
                }
                "--iters" => {
                    let value = args.next().expect("--iters needs a value");
                    iters = value.parse().expect("--iters must be a usize");
                }
                "--filter" => {
                    filter = args.next();
                }
                value => {
                    filter = Some(value.to_string());
                }
            }
        }
        Self {
            warmup,
            iters,
            filter,
        }
    }

    fn should_run(&self, name: &str) -> bool {
        self.filter
            .as_ref()
            .is_none_or(|needle| name.contains(needle))
    }
}

fn bench<F>(args: &BenchArgs, name: &str, mut f: F)
where
    F: FnMut(),
{
    if !args.should_run(name) {
        return;
    }

    for _ in 0..args.warmup {
        f();
    }

    let mut samples = Vec::with_capacity(args.iters);
    for _ in 0..args.iters {
        let start = Instant::now();
        f();
        samples.push(start.elapsed().as_micros());
    }
    samples.sort_unstable();
    let sum: u128 = samples.iter().copied().sum();
    let stats = Stats {
        min_us: samples[0],
        p50_us: samples[samples.len() / 2],
        avg_us: sum / samples.len() as u128,
    };

    println!(
        "{name:<32} min={:>6}us p50={:>6}us avg={:>6}us",
        stats.min_us, stats.p50_us, stats.avg_us
    );
}

fn input_1m() -> Tensor {
    Tensor::from_vec(
        vec![1_000_000],
        (0..1_000_000)
            .map(|i| ((i as f32 * 0.001).sin() * 6.0) - 3.0)
            .collect(),
    )
    .unwrap()
}

fn input_921k() -> Tensor {
    Tensor::from_vec(
        vec![921_600],
        (0..921_600)
            .map(|i| ((i as f32 * 0.001).cos() * 6.0) - 3.0)
            .collect(),
    )
    .unwrap()
}

fn main() {
    let args = BenchArgs::parse();
    let cpu = yscv_kernels::host_cpu();
    println!("host_cpu={cpu:?}");
    println!(
        "available_parallelism={} RAYON_NUM_THREADS={}",
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "unset".to_string())
    );
    println!("operation                         min       p50       avg");
    println!("{}", "-".repeat(62));

    let a1m = input_1m();
    let b1m = Tensor::from_vec(
        vec![1_000_000],
        (0..1_000_000)
            .map(|i| ((i as f32 * 0.0007).cos() * 4.0) + 1.0)
            .collect(),
    )
    .unwrap();
    let a921k = input_921k();

    println!(
        "iters={} warmup={} filter={}",
        args.iters,
        args.warmup,
        args.filter.as_deref().unwrap_or("none")
    );
    bench(&args, "add_1M", || {
        let _ = yscv_kernels::add(&a1m, &b1m).unwrap();
    });
    let mut add_out_1m = Tensor::zeros(vec![1_000_000]).unwrap();
    bench(&args, "add_1M_out_reuse", || {
        add_out(&a1m, &b1m, &mut add_out_1m).unwrap();
    });
    bench(&args, "mul_1M", || {
        let _ = yscv_kernels::mul(&a1m, &b1m).unwrap();
    });
    let mut mul_out_1m = Tensor::zeros(vec![1_000_000]).unwrap();
    bench(&args, "mul_1M_out_reuse", || {
        mul_out(&a1m, &b1m, &mut mul_out_1m).unwrap();
    });
    bench(&args, "exp_1M", || {
        let _ = exp_with_config(&a1m, ParallelElementwiseConfig::default());
    });
    bench(&args, "sum_1M_raw_slice", || {
        std::hint::black_box(add_reduce_dispatch(a1m.data()));
    });
    bench(&args, "max_1M_raw_slice", || {
        std::hint::black_box(max_reduce_dispatch(a1m.data()));
    });

    let mat_1024 = Tensor::from_vec(
        vec![1024, 1024],
        (0..1_048_576)
            .map(|i| ((i as f32 * 0.0009).sin() * 3.0) - 1.0)
            .collect(),
    )
    .unwrap();
    let row_1024 = Tensor::from_vec(
        vec![1024],
        (0..1024)
            .map(|i| ((i as f32 * 0.007).cos() * 2.0) + 0.5)
            .collect(),
    )
    .unwrap();
    bench(&args, "add_broadcast_1024x1024_by_1024", || {
        let _ =
            add_with_config(&mat_1024, &row_1024, ParallelElementwiseConfig::default()).unwrap();
    });
    bench(&args, "sub_broadcast_1024_by_1024x1024", || {
        let _ =
            sub_with_config(&row_1024, &mat_1024, ParallelElementwiseConfig::default()).unwrap();
    });
    bench(&args, "relu_921K", || {
        let _ = relu(&a921k);
    });
    let mut relu_out_921k = Tensor::zeros(vec![921_600]).unwrap();
    bench(&args, "relu_921K_out_reuse", || {
        relu_out_with_config(
            &a921k,
            &mut relu_out_921k,
            ParallelElementwiseConfig::default(),
        )
        .unwrap();
    });
    let mut relu_raw = vec![0.0f32; a921k.data().len()];
    bench(&args, "relu_921K_raw_slice", || {
        relu_to_slice_dispatch(a921k.data(), &mut relu_raw);
    });
    bench(&args, "sigmoid_921K", || {
        let _ = sigmoid(&a921k);
    });
    bench(&args, "sigmoid_921K_disabled", || {
        let _ = sigmoid_with_config(&a921k, ParallelElementwiseConfig::disabled());
    });
    bench(&args, "tanh_1M", || {
        let _ = tanh_act(&a1m);
    });
    bench(&args, "tanh_1M_disabled", || {
        let _ = tanh_act_with_config(&a1m, ParallelElementwiseConfig::disabled());
    });
    bench(&args, "gelu_1M", || {
        let _ = gelu(&a1m);
    });
    bench(&args, "silu_1M", || {
        let _ = silu(&a1m);
    });
    let silu_base = input_1m();
    bench(&args, "silu_1M_inplace_clone", || {
        let mut t = silu_base.clone();
        silu_inplace(&mut t);
    });

    let sm_32_1000 = Tensor::from_vec(
        vec![32, 1000],
        (0..32_000)
            .map(|i| ((i as f32 * 0.013).sin() * 2.0) - 1.0)
            .collect(),
    )
    .unwrap();
    bench(&args, "softmax_32x1000", || {
        let _ = softmax_last_dim(&sm_32_1000).unwrap();
    });
    let mut sm_32_1000_out = Tensor::zeros(vec![32, 1000]).unwrap();
    bench(&args, "softmax_32x1000_out_reuse", || {
        softmax_last_dim_out(&sm_32_1000, &mut sm_32_1000_out).unwrap();
    });
    bench(&args, "log_softmax_32x1000", || {
        let _ = log_softmax_last_dim(&sm_32_1000).unwrap();
    });
    let mut lsm_32_1000_out = Tensor::zeros(vec![32, 1000]).unwrap();
    bench(&args, "log_softmax_32x1000_out_reuse", || {
        log_softmax_last_dim_out(&sm_32_1000, &mut lsm_32_1000_out).unwrap();
    });

    let sm_512_256 = Tensor::from_vec(
        vec![512, 256],
        (0..131_072)
            .map(|i| ((i as f32 * 0.007).cos() * 2.0) - 1.0)
            .collect(),
    )
    .unwrap();
    bench(&args, "softmax_512x256", || {
        let _ = softmax_last_dim(&sm_512_256).unwrap();
    });
    let mut sm_512_256_out = Tensor::zeros(vec![512, 256]).unwrap();
    bench(&args, "softmax_512x256_out_reuse", || {
        softmax_last_dim_out(&sm_512_256, &mut sm_512_256_out).unwrap();
    });

    let ln_input = Tensor::from_vec(
        vec![512, 256],
        (0..131_072)
            .map(|i| ((i as f32 * 0.005).sin() * 2.0) - 1.0)
            .collect(),
    )
    .unwrap();
    let ln_gamma = Tensor::ones(vec![256]).unwrap();
    let ln_beta = Tensor::zeros(vec![256]).unwrap();
    bench(&args, "layer_norm_512x256", || {
        let _ = layer_norm_last_dim(
            &ln_input,
            LayerNormLastDimParams {
                gamma: &ln_gamma,
                beta: &ln_beta,
                epsilon: 1e-5,
            },
        )
        .unwrap();
    });

    let bn_input = Tensor::from_vec(
        vec![1, 64, 64, 3],
        (0..12_288)
            .map(|i| ((i as f32 * 0.011).sin() * 2.0) - 1.0)
            .collect(),
    )
    .unwrap();
    let bn_gamma = Tensor::ones(vec![3]).unwrap();
    let bn_beta = Tensor::zeros(vec![3]).unwrap();
    let bn_mean = Tensor::zeros(vec![3]).unwrap();
    let bn_var = Tensor::ones(vec![3]).unwrap();
    bench(&args, "batch_norm_1x64x64x3", || {
        let _ = batch_norm2d_nhwc(
            &bn_input,
            BatchNorm2dParams {
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
