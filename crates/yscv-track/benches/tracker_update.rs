#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use yscv_detect::{BoundingBox, CLASS_ID_PERSON, Detection};
use yscv_track::{Tracker, TrackerConfig};

fn build_frame_detections(frame_idx: usize, count: usize) -> Vec<Detection> {
    let mut out = Vec::with_capacity(count);
    let dx = (frame_idx % 5) as f32 * 0.15;
    let dy = (frame_idx % 7) as f32 * 0.10;
    for idx in 0..count {
        let row = idx / 8;
        let col = idx % 8;
        let x1 = col as f32 * 8.0 + dx;
        let y1 = row as f32 * 8.0 + dy;
        out.push(Detection {
            bbox: BoundingBox {
                x1,
                y1,
                x2: x1 + 4.0,
                y2: y1 + 4.0,
            },
            score: 0.85,
            class_id: CLASS_ID_PERSON,
        });
    }
    out
}

fn build_sequence(frames: usize, count: usize) -> Vec<Vec<Detection>> {
    (0..frames)
        .map(|frame_idx| build_frame_detections(frame_idx, count))
        .collect()
}

fn benchmark_tracker_update(c: &mut Criterion) {
    let config = TrackerConfig {
        match_iou_threshold: 0.2,
        max_missed_frames: 3,
        max_tracks: 512,
    };
    let sequence = build_sequence(90, 32);

    let mut group = c.benchmark_group("tracker_update_modes");
    group.bench_function("update_alloc_output", |b| {
        b.iter_batched(
            || Tracker::new(config).expect("valid tracker config"),
            |mut tracker| {
                for detections in &sequence {
                    let out = tracker.update(black_box(detections));
                    black_box(out.len());
                }
            },
            BatchSize::SmallInput,
        );
    });
    group.bench_function("update_into_reused_output", |b| {
        b.iter_batched(
            || Tracker::new(config).expect("valid tracker config"),
            |mut tracker| {
                let mut out = Vec::new();
                for detections in &sequence {
                    tracker.update_into(black_box(detections), &mut out);
                    black_box(out.len());
                }
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(benches, benchmark_tracker_update);
criterion_main!(benches);
