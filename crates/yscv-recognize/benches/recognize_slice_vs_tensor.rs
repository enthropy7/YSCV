#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use yscv_recognize::Recognizer;
use yscv_tensor::Tensor;

fn build_recognizer(entry_count: usize) -> Recognizer {
    let mut recognizer = Recognizer::new(0.25).expect("valid threshold");
    for idx in 0..entry_count {
        let x = (idx as f32 + 1.0) / (entry_count as f32 + 3.0);
        let y = (idx as f32 + 2.0) / (entry_count as f32 + 5.0);
        let z = (idx as f32 + 3.0) / (entry_count as f32 + 7.0);
        recognizer
            .enroll(
                format!("id-{idx}"),
                Tensor::from_vec(vec![3], vec![x, y, z]).expect("valid embedding"),
            )
            .expect("unique identity");
    }
    recognizer
}

fn bench_recognition_modes(c: &mut Criterion) {
    let recognizer = build_recognizer(128);
    let query_slice = [0.47f32, 0.28f32, 0.11f32];
    let query_tensor = Tensor::from_vec(vec![3], query_slice.to_vec()).expect("valid embedding");

    let mut group = c.benchmark_group("recognize_query_modes");
    group.bench_function("slice_stack_embedding", |b| {
        b.iter(|| {
            let out = recognizer
                .recognize_slice(black_box(&query_slice))
                .expect("slice recognition");
            black_box(out);
        });
    });
    group.bench_function("tensor_reused", |b| {
        b.iter(|| {
            let out = recognizer
                .recognize(black_box(&query_tensor))
                .expect("tensor recognition");
            black_box(out);
        });
    });
    group.bench_function("tensor_alloc_per_call", |b| {
        b.iter(|| {
            let query = Tensor::from_vec(
                vec![3],
                vec![query_slice[0], query_slice[1], query_slice[2]],
            )
            .expect("valid embedding");
            let out = recognizer
                .recognize(black_box(&query))
                .expect("tensor recognition");
            black_box(out);
        });
    });
    group.finish();
}

criterion_group!(benches, bench_recognition_modes);
criterion_main!(benches);
