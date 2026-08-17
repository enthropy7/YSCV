#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use yscv_video::normalize_rgb8_to_f32_inplace;

fn synthetic_rgb8(width: usize, height: usize) -> Vec<u8> {
    let len = width.saturating_mul(height).saturating_mul(3);
    let mut bytes = Vec::with_capacity(len);
    // Deterministic non-uniform RGB sequence.
    for idx in 0..len {
        bytes.push(((idx * 17 + 91) % 256) as u8);
    }
    bytes
}

fn bench_normalize_rgb8_modes(c: &mut Criterion) {
    let rgb8 = synthetic_rgb8(640, 480);

    let mut group = c.benchmark_group("video_normalize_rgb8_modes");

    group.bench_function("normalize_alloc_output", |b| {
        b.iter(|| {
            let mut out = vec![0.0f32; rgb8.len()];
            normalize_rgb8_to_f32_inplace(black_box(&rgb8), black_box(&mut out))
                .expect("valid output buffer");
            black_box(out[0]);
        });
    });

    let mut reused_out = vec![0.0f32; rgb8.len()];
    group.bench_function("normalize_reused_output", |b| {
        b.iter(|| {
            normalize_rgb8_to_f32_inplace(black_box(&rgb8), black_box(&mut reused_out))
                .expect("valid output buffer");
            black_box(reused_out[0]);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_normalize_rgb8_modes);
criterion_main!(benches);
