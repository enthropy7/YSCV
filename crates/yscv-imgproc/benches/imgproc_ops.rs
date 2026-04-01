use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use yscv_imgproc::{
    box_blur_3x3, flip_horizontal, flip_vertical, normalize, resize_nearest, rgb_to_grayscale,
    rotate90_cw,
};
use yscv_tensor::Tensor;

fn rgb_image(width: usize, height: usize, seed: f32) -> Tensor {
    let mut data = Vec::with_capacity(width * height * 3);
    for idx in 0..(width * height * 3) {
        data.push(((idx % 251) as f32 * 0.0041 + seed).fract());
    }
    Tensor::from_vec(vec![height, width, 3], data).expect("valid image")
}

fn bench_imgproc_ops(c: &mut Criterion) {
    let rgb_640_480 = rgb_image(640, 480, 0.17);
    let rgb_320_240 = rgb_image(320, 240, 0.42);

    let mut group = c.benchmark_group("imgproc_ops");
    group.sample_size(10);
    group.warm_up_time(std::time::Duration::from_millis(200));
    group.measurement_time(std::time::Duration::from_millis(500));

    group.bench_function("rgb_to_grayscale_640x480", |b| {
        b.iter(|| {
            let out = rgb_to_grayscale(black_box(&rgb_640_480)).expect("grayscale");
            black_box(out);
        });
    });

    group.bench_function("resize_nearest_320x240_to_640x480", |b| {
        b.iter(|| {
            let out = resize_nearest(black_box(&rgb_320_240), 480, 640).expect("resize");
            black_box(out);
        });
    });

    group.bench_function("normalize_640x480_rgb", |b| {
        let mean = [0.485f32, 0.456, 0.406];
        let std = [0.229f32, 0.224, 0.225];
        b.iter_batched(
            || rgb_640_480.clone(),
            |input| {
                let out = normalize(black_box(&input), black_box(&mean), black_box(&std))
                    .expect("normalize");
                black_box(out);
            },
            BatchSize::LargeInput,
        );
    });

    group.bench_function("box_blur_3x3_640x480_rgb", |b| {
        b.iter(|| {
            let out = box_blur_3x3(black_box(&rgb_640_480)).expect("blur");
            black_box(out);
        });
    });

    group.bench_function("flip_horizontal_640x480_rgb", |b| {
        b.iter(|| {
            let out = flip_horizontal(black_box(&rgb_640_480)).expect("flip horizontal");
            black_box(out);
        });
    });

    group.bench_function("flip_vertical_640x480_rgb", |b| {
        b.iter(|| {
            let out = flip_vertical(black_box(&rgb_640_480)).expect("flip vertical");
            black_box(out);
        });
    });

    group.bench_function("rotate90_cw_640x480_rgb", |b| {
        b.iter(|| {
            let out = rotate90_cw(black_box(&rgb_640_480)).expect("rotate90 cw");
            black_box(out);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_imgproc_ops);
criterion_main!(benches);
