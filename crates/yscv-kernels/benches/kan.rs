#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use yscv_kernels::{KanLinear, matmul_2d_slices, matmul_row_dispatch, relu_slice_dispatch};

fn values(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| (((i % 251) as f32 * 0.0041 + seed).fract() - 0.5) * 0.2)
        .collect()
}

/// A KAN and an MLP with (about) the same number of parameters, as trained on the sensor
/// datasets: `kan` hidden widths with grid 5, `mlp` hidden widths with ReLU.
struct Pair {
    name: &'static str,
    kan: Vec<KanLinear>,
    mlp: Vec<(Vec<f32>, Vec<f32>, usize, usize)>,
    input: Vec<f32>,
}

fn pair(
    name: &'static str,
    n_in: usize,
    kan_hidden: usize,
    mlp_hidden: usize,
    n_out: usize,
) -> Pair {
    let kan_sizes = [n_in, kan_hidden, n_out];
    let kan = kan_sizes
        .windows(2)
        .enumerate()
        .map(|(l, w)| {
            let (a, b) = (w[0], w[1]);
            KanLinear::new(
                a,
                b,
                5,
                (-1.0, 1.0),
                &values(a * b, 0.1 + l as f32),
                &values(a * b * 8, 0.3),
                &values(a * b, 0.7),
            )
            .expect("valid kan layer")
        })
        .collect();
    let mlp_sizes = [n_in, mlp_hidden, mlp_hidden, n_out];
    let mlp = mlp_sizes
        .windows(2)
        .enumerate()
        .map(|(l, w)| {
            (
                values(w[0] * w[1], 0.2 + l as f32),
                values(w[1], 0.9),
                w[0],
                w[1],
            )
        })
        .collect();
    let input = (0..n_in)
        .map(|i| ((i * 37 % 101) as f32 / 101.0) * 2.0 - 1.0)
        .collect();
    Pair {
        name,
        kan,
        mlp,
        input,
    }
}

fn kan_forward(layers: &[KanLinear], input: &[f32]) -> Vec<f32> {
    let mut x = input.to_vec();
    for layer in layers {
        let mut y = vec![0.0; layer.out_features()];
        layer.forward(&x, 1, &mut y).expect("kan forward");
        x = y;
    }
    x
}

fn mlp_forward(
    layers: &[(Vec<f32>, Vec<f32>, usize, usize)],
    input: &[f32],
    row: bool,
) -> Vec<f32> {
    let mut x = input.to_vec();
    for (l, (w, b, k, n)) in layers.iter().enumerate() {
        let mut y = vec![0.0; *n];
        if row {
            // SAFETY: `x` holds k floats, `w` k · n, `y` n; no aliasing.
            #[allow(unsafe_code)]
            unsafe {
                matmul_row_dispatch(x.as_ptr(), w.as_ptr(), y.as_mut_ptr(), *k, *n)
            };
        } else {
            matmul_2d_slices(&x, 1, *k, w, *n, &mut y);
        }
        y.iter_mut().zip(b).for_each(|(v, bias)| *v += bias);
        if l + 1 < layers.len() {
            relu_slice_dispatch(&mut y);
        }
        x = y;
    }
    x
}

fn bench_kan_vs_mlp(c: &mut Criterion) {
    // parameter-matched networks (≈ 40K parameters) of the HAR (561 features, 6 classes) and
    // DSADS (270 features, 19 classes) sensor tasks
    for p in [pair("har", 561, 7, 63, 6), pair("dsads", 270, 14, 102, 19)] {
        let mut group = c.benchmark_group(format!("kan_vs_mlp_{}", p.name));
        group.bench_function("kan", |b| {
            b.iter(|| kan_forward(black_box(&p.kan), black_box(&p.input)))
        });
        group.bench_function("mlp_gemm", |b| {
            b.iter(|| mlp_forward(black_box(&p.mlp), black_box(&p.input), false))
        });
        group.bench_function("mlp_row", |b| {
            b.iter(|| mlp_forward(black_box(&p.mlp), black_box(&p.input), true))
        });
        group.finish();
    }
}

criterion_group!(benches, bench_kan_vs_mlp);
criterion_main!(benches);
