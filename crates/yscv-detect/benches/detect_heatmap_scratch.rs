#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use yscv_detect::{
    FrameFaceDetectScratch, FramePeopleDetectScratch, HeatmapDetectScratch,
    Rgb8PeopleDetectScratch, detect_faces_from_frame, detect_faces_from_frame_with_scratch,
    detect_from_heatmap, detect_from_heatmap_with_scratch, detect_people_from_frame,
    detect_people_from_frame_with_scratch, detect_people_from_rgb8,
    detect_people_from_rgb8_with_scratch,
};
use yscv_tensor::Tensor;
use yscv_video::Frame;

fn build_heatmap(height: usize, width: usize) -> Tensor {
    let mut data = vec![0.02f32; height * width];
    for y in (8..height).step_by(18) {
        for x in (10..width).step_by(20) {
            for yy in y..(y + 4).min(height) {
                for xx in x..(x + 5).min(width) {
                    let idx = yy * width + xx;
                    data[idx] = 0.9;
                }
            }
        }
    }
    Tensor::from_vec(vec![height, width, 1], data).expect("valid heatmap")
}

fn build_rgb8(width: usize, height: usize) -> Vec<u8> {
    let mut rgb8 = vec![16u8; width * height * 3];
    for y in (8..height).step_by(18) {
        for x in (10..width).step_by(20) {
            for yy in y..(y + 4).min(height) {
                for xx in x..(x + 5).min(width) {
                    let base = (yy * width + xx) * 3;
                    rgb8[base] = 240;
                    rgb8[base + 1] = 240;
                    rgb8[base + 2] = 240;
                }
            }
        }
    }
    rgb8
}

fn build_rgb_frame(width: usize, height: usize) -> Frame {
    let rgb8 = build_rgb8(width, height);
    let mut data = vec![0.0f32; width * height * 3];
    const SCALE: f32 = 1.0 / 255.0;
    for (rgb, out) in rgb8.chunks_exact(3).zip(data.chunks_exact_mut(3)) {
        out[0] = rgb[0] as f32 * SCALE;
        out[1] = rgb[1] as f32 * SCALE;
        out[2] = rgb[2] as f32 * SCALE;
    }
    let image = Tensor::from_vec(vec![height, width, 3], data).expect("valid image");
    Frame::new(0, 0, image).expect("valid frame")
}

fn build_face_frame(width: usize, height: usize) -> Frame {
    let background = [0.10f32, 0.10f32, 0.12f32];
    let skin = [0.78f32, 0.60f32, 0.46f32];
    let mut data = vec![0.0f32; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let base = (y * width + x) * 3;
            let in_face_patch = ((18..74).contains(&y) && (20..66).contains(&x))
                || ((28..96).contains(&y) && (88..134).contains(&x));
            let color = if in_face_patch { skin } else { background };
            data[base] = color[0];
            data[base + 1] = color[1];
            data[base + 2] = color[2];
        }
    }
    let image = Tensor::from_vec(vec![height, width, 3], data).expect("valid image");
    Frame::new(0, 0, image).expect("valid frame")
}

fn bench_heatmap_scratch(c: &mut Criterion) {
    let heatmap = build_heatmap(128, 128);
    let mut scratch = HeatmapDetectScratch::default();

    let mut group = c.benchmark_group("detect_heatmap_modes");
    group.bench_function("fresh_internal_buffers", |b| {
        b.iter(|| {
            let out = detect_from_heatmap(black_box(&heatmap), 0.5, 8, 0.4, 64)
                .expect("detection should succeed");
            black_box(out.len());
        });
    });
    group.bench_function("reused_scratch_buffers", |b| {
        b.iter(|| {
            let out = detect_from_heatmap_with_scratch(
                black_box(&heatmap),
                0.5,
                8,
                0.4,
                64,
                black_box(&mut scratch),
            )
            .expect("detection should succeed");
            black_box(out.len());
        });
    });
    group.finish();
}

fn bench_rgb8_people_scratch(c: &mut Criterion) {
    let width = 160usize;
    let height = 120usize;
    let rgb8 = build_rgb8(width, height);
    let mut scratch = Rgb8PeopleDetectScratch::default();

    let mut group = c.benchmark_group("detect_people_rgb8_modes");
    group.bench_function("fresh_internal_scratch", |b| {
        b.iter(|| {
            let out = detect_people_from_rgb8(
                black_box(width),
                black_box(height),
                black_box(&rgb8),
                0.5,
                8,
                0.4,
                64,
            )
            .expect("detection should succeed");
            black_box(out.len());
        });
    });
    group.bench_function("reused_external_scratch", |b| {
        b.iter(|| {
            let out = detect_people_from_rgb8_with_scratch(
                (black_box(width), black_box(height)),
                black_box(&rgb8),
                0.5,
                8,
                0.4,
                64,
                black_box(&mut scratch),
            )
            .expect("detection should succeed");
            black_box(out.len());
        });
    });
    group.finish();
}

fn bench_frame_people_scratch(c: &mut Criterion) {
    let frame = build_rgb_frame(160, 120);
    let mut scratch = FramePeopleDetectScratch::default();

    let mut group = c.benchmark_group("detect_people_frame_modes");
    group.bench_function("fresh_internal_scratch", |b| {
        b.iter(|| {
            let out = detect_people_from_frame(black_box(&frame), 0.5, 8, 0.4, 64)
                .expect("detection should succeed");
            black_box(out.len());
        });
    });
    group.bench_function("reused_external_scratch", |b| {
        b.iter(|| {
            let out = detect_people_from_frame_with_scratch(
                black_box(&frame),
                0.5,
                8,
                0.4,
                64,
                black_box(&mut scratch),
            )
            .expect("detection should succeed");
            black_box(out.len());
        });
    });
    group.finish();
}

fn bench_frame_faces_scratch(c: &mut Criterion) {
    let frame = build_face_frame(160, 120);
    let mut scratch = FrameFaceDetectScratch::default();

    let mut group = c.benchmark_group("detect_faces_frame_modes");
    group.bench_function("fresh_internal_scratch", |b| {
        b.iter(|| {
            let out = detect_faces_from_frame(black_box(&frame), 0.35, 64, 0.4, 64)
                .expect("detection should succeed");
            black_box(out.len());
        });
    });
    group.bench_function("reused_external_scratch", |b| {
        b.iter(|| {
            let out = detect_faces_from_frame_with_scratch(
                black_box(&frame),
                0.35,
                64,
                0.4,
                64,
                black_box(&mut scratch),
            )
            .expect("detection should succeed");
            black_box(out.len());
        });
    });
    group.finish();
}

fn bench_detect(c: &mut Criterion) {
    bench_heatmap_scratch(c);
    bench_rgb8_people_scratch(c);
    bench_frame_people_scratch(c);
    bench_frame_faces_scratch(c);
}

criterion_group!(benches, bench_detect);
criterion_main!(benches);
