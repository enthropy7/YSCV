use criterion::{Criterion, black_box, criterion_group, criterion_main};
use yscv_imgproc::{
    closing_3x3, dilate_3x3, erode_3x3, morph_gradient_3x3, opening_3x3, sobel_3x3_gradients,
    sobel_3x3_magnitude,
};
use yscv_tensor::Tensor;

fn rgb_image(width: usize, height: usize, seed: f32) -> Tensor {
    let mut data = Vec::with_capacity(width * height * 3);
    for idx in 0..(width * height * 3) {
        data.push(((idx % 251) as f32 * 0.0041 + seed).fract());
    }
    Tensor::from_vec(vec![height, width, 3], data).expect("valid image")
}

// ---------------------------------------------------------------------------
// f32 morphology + sobel — heavy allocators, separate process
// ---------------------------------------------------------------------------

fn bench_imgproc_morph(c: &mut Criterion) {
    let rgb_640_480 = rgb_image(640, 480, 0.17);

    let mut group = c.benchmark_group("imgproc_morph");
    group.sample_size(10);
    group.warm_up_time(std::time::Duration::from_millis(200));
    group.measurement_time(std::time::Duration::from_millis(500));

    group.bench_function("dilate_3x3_640x480_rgb", |b| {
        b.iter(|| {
            let out = dilate_3x3(black_box(&rgb_640_480)).expect("dilate");
            black_box(out);
        });
    });

    group.bench_function("erode_3x3_640x480_rgb", |b| {
        b.iter(|| {
            let out = erode_3x3(black_box(&rgb_640_480)).expect("erode");
            black_box(out);
        });
    });

    group.bench_function("opening_3x3_640x480_rgb", |b| {
        b.iter(|| {
            let out = opening_3x3(black_box(&rgb_640_480)).expect("opening");
            black_box(out);
        });
    });

    group.bench_function("closing_3x3_640x480_rgb", |b| {
        b.iter(|| {
            let out = closing_3x3(black_box(&rgb_640_480)).expect("closing");
            black_box(out);
        });
    });

    group.bench_function("morph_gradient_3x3_640x480_rgb", |b| {
        b.iter(|| {
            let out = morph_gradient_3x3(black_box(&rgb_640_480)).expect("morph gradient");
            black_box(out);
        });
    });

    group.bench_function("sobel_3x3_gradients_640x480_rgb", |b| {
        b.iter(|| {
            let out = sobel_3x3_gradients(black_box(&rgb_640_480)).expect("sobel gradients");
            black_box(out);
        });
    });

    group.bench_function("sobel_3x3_magnitude_640x480_rgb", |b| {
        b.iter(|| {
            let out = sobel_3x3_magnitude(black_box(&rgb_640_480)).expect("sobel magnitude");
            black_box(out);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_imgproc_morph);
criterion_main!(benches);
