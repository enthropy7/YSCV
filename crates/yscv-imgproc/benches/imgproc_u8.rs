use criterion::{Criterion, black_box, criterion_group, criterion_main};
use yscv_imgproc::{
    ImageU8, box_blur_3x3_u8, dilate_3x3_u8, erode_3x3_u8, grayscale_u8, resize_bilinear_u8,
    resize_nearest_u8, sobel_3x3_magnitude_u8,
};

fn gray_u8_image(width: usize, height: usize) -> ImageU8 {
    let data: Vec<u8> = (0..width * height).map(|i| (i % 251) as u8).collect();
    ImageU8::new(data, height, width, 1).expect("valid u8 image")
}

fn rgb_u8_image(width: usize, height: usize) -> ImageU8 {
    let data: Vec<u8> = (0..width * height * 3).map(|i| (i % 251) as u8).collect();
    ImageU8::new(data, height, width, 3).expect("valid u8 image")
}

fn bench_imgproc_u8_ops(c: &mut Criterion) {
    let rgb_640_480 = rgb_u8_image(640, 480);
    let gray_640_480 = gray_u8_image(640, 480);
    let gray_320_240 = gray_u8_image(320, 240);

    let mut group = c.benchmark_group("imgproc_u8_ops");
    group.sample_size(10);
    group.warm_up_time(std::time::Duration::from_millis(200));
    group.measurement_time(std::time::Duration::from_millis(500));

    group.bench_function("grayscale_u8_640x480", |b| {
        b.iter(|| {
            let out = grayscale_u8(black_box(&rgb_640_480)).expect("grayscale u8");
            black_box(out);
        });
    });

    group.bench_function("resize_nearest_u8_320x240_to_640x480", |b| {
        b.iter(|| {
            let out =
                resize_nearest_u8(black_box(&gray_320_240), 480, 640).expect("resize nearest u8");
            black_box(out);
        });
    });

    group.bench_function("resize_bilinear_u8_320x240_to_640x480", |b| {
        b.iter(|| {
            let out =
                resize_bilinear_u8(black_box(&gray_320_240), 480, 640).expect("resize bilinear u8");
            black_box(out);
        });
    });

    group.bench_function("dilate_3x3_u8_640x480_gray", |b| {
        b.iter(|| {
            let out = dilate_3x3_u8(black_box(&gray_640_480)).expect("dilate u8");
            black_box(out);
        });
    });

    group.bench_function("erode_3x3_u8_640x480_gray", |b| {
        b.iter(|| {
            let out = erode_3x3_u8(black_box(&gray_640_480)).expect("erode u8");
            black_box(out);
        });
    });

    group.bench_function("box_blur_3x3_u8_640x480_gray", |b| {
        b.iter(|| {
            let out = box_blur_3x3_u8(black_box(&gray_640_480)).expect("box blur u8");
            black_box(out);
        });
    });

    group.bench_function("sobel_3x3_u8_640x480_gray", |b| {
        b.iter(|| {
            let out = sobel_3x3_magnitude_u8(black_box(&gray_640_480)).expect("sobel u8");
            black_box(out);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_imgproc_u8_ops);
criterion_main!(benches);
