#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use yscv_autograd::Graph;
use yscv_tensor::Tensor;

fn tensor_from_shape(shape: &[usize], seed: f32) -> Tensor {
    let len = shape.iter().copied().product::<usize>();
    let mut data = Vec::with_capacity(len);
    for idx in 0..len {
        data.push(((idx % 1024) as f32 * 0.00271 + seed).fract());
    }
    Tensor::from_vec(shape.to_vec(), data).expect("valid tensor shape/data")
}

fn bench_autograd_backward_modes(c: &mut Criterion) {
    let mut group = c.benchmark_group("autograd_backward_modes");

    group.bench_function("matmul_relu_sum_32x32", |b| {
        let mut graph = Graph::new();
        let input = graph.variable(tensor_from_shape(&[32, 32], 0.11));
        let weights = graph.variable(tensor_from_shape(&[32, 32], 0.37));
        let persistent = graph.node_count();

        b.iter(|| {
            graph.zero_grads();
            graph
                .truncate(persistent)
                .expect("truncate to persistent nodes");
            let matmul = graph
                .matmul_2d(input, weights)
                .expect("matmul should succeed");
            let activated = graph.relu(matmul).expect("relu should succeed");
            let loss = graph.sum(activated).expect("sum should succeed");
            graph.backward(loss).expect("backward should succeed");

            let first_grad = graph
                .grad(input)
                .expect("query grad should succeed")
                .expect("input grad should exist")
                .data()[0];
            black_box(first_grad);
        });
    });

    group.bench_function("broadcast_add_mul_sum_64x64", |b| {
        let mut graph = Graph::new();
        let input = graph.variable(tensor_from_shape(&[64, 64], 0.19));
        let bias = graph.variable(tensor_from_shape(&[64], 0.43));
        let scale = graph.constant(tensor_from_shape(&[64, 64], 0.71));
        let persistent = graph.node_count();

        b.iter(|| {
            graph.zero_grads();
            graph
                .truncate(persistent)
                .expect("truncate to persistent nodes");
            let shifted = graph.add(input, bias).expect("add should succeed");
            let scaled = graph.mul(shifted, scale).expect("mul should succeed");
            let loss = graph.sum(scaled).expect("sum should succeed");
            graph.backward(loss).expect("backward should succeed");

            let bias_grad = graph
                .grad(bias)
                .expect("query grad should succeed")
                .expect("bias grad should exist")
                .data()[0];
            black_box(bias_grad);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_autograd_backward_modes);
criterion_main!(benches);
