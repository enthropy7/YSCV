#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use yscv_tensor::Tensor;

fn build_tensor(shape: &[usize], seed: f32) -> Tensor {
    let len = shape.iter().copied().product::<usize>();
    let mut data = Vec::with_capacity(len);
    for idx in 0..len {
        data.push(((idx % 257) as f32 * 0.0039 + seed).fract());
    }
    Tensor::from_vec(shape.to_vec(), data).expect("valid tensor shape")
}

fn bench_add_modes(c: &mut Criterion) {
    let shape = [180usize, 320usize, 3usize];
    let lhs = build_tensor(&shape, 0.13);
    let rhs_same = build_tensor(&shape, 0.71);
    let rhs_broadcast = build_tensor(&[3], 0.27);

    let mut group = c.benchmark_group("tensor_add_modes");
    group.bench_function("same_shape_fast_path", |b| {
        b.iter(|| {
            let out = lhs.add(black_box(&rhs_same)).expect("add same shape");
            black_box(out);
        });
    });
    group.bench_function("broadcast_path", |b| {
        b.iter(|| {
            let out = lhs.add(black_box(&rhs_broadcast)).expect("add broadcast");
            black_box(out);
        });
    });
    group.finish();
}

fn bench_mul_modes(c: &mut Criterion) {
    let shape = [180usize, 320usize, 3usize];
    let lhs = build_tensor(&shape, 0.41);
    let rhs_same = build_tensor(&shape, 0.09);
    let rhs_broadcast = build_tensor(&[3], 0.88);

    let mut group = c.benchmark_group("tensor_mul_modes");
    group.bench_function("same_shape_fast_path", |b| {
        b.iter(|| {
            let out = lhs.mul(black_box(&rhs_same)).expect("mul same shape");
            black_box(out);
        });
    });
    group.bench_function("broadcast_path", |b| {
        b.iter(|| {
            let out = lhs.mul(black_box(&rhs_broadcast)).expect("mul broadcast");
            black_box(out);
        });
    });
    group.finish();
}

criterion_group!(benches, bench_add_modes, bench_mul_modes);
criterion_main!(benches);
