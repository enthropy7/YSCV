#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::time::Duration;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use yscv_detect::{BoundingBox, Detection};
use yscv_eval::{
    DetectionDatasetFrame, DetectionEvalConfig, GroundTruthTrack, LabeledBox,
    PipelineBenchmarkThresholds, PipelineDurations, StageThresholds, TrackingDatasetFrame,
    TrackingEvalConfig, detection_frames_as_view, evaluate_detections,
    evaluate_detections_from_dataset, evaluate_tracking, evaluate_tracking_from_dataset,
    summarize_pipeline_durations, tracking_frames_as_view, validate_pipeline_benchmark_thresholds,
};
use yscv_track::TrackedDetection;

fn make_bbox(x: f32, y: f32, width: f32, height: f32) -> BoundingBox {
    BoundingBox {
        x1: x,
        y1: y,
        x2: x + width,
        y2: y + height,
    }
}

fn build_detection_dataset(
    frame_count: usize,
    gt_per_frame: usize,
    pred_per_frame: usize,
) -> Vec<DetectionDatasetFrame> {
    let mut frames = Vec::with_capacity(frame_count);

    for frame_idx in 0..frame_count {
        let mut ground_truth = Vec::with_capacity(gt_per_frame);
        for obj_idx in 0..gt_per_frame {
            let x = ((frame_idx * 13 + obj_idx * 17) % 96) as f32;
            let y = ((frame_idx * 11 + obj_idx * 19) % 96) as f32;
            ground_truth.push(LabeledBox {
                bbox: make_bbox(x, y, 14.0, 18.0),
                class_id: obj_idx % 2,
            });
        }

        let mut predictions = Vec::with_capacity(pred_per_frame);
        for pred_idx in 0..pred_per_frame {
            let gt = ground_truth[pred_idx % gt_per_frame];
            let jitter = if pred_idx % 3 == 0 { 0.0 } else { 1.0 };
            let class_id = if pred_idx % 5 == 0 {
                (gt.class_id + 1) % 2
            } else {
                gt.class_id
            };
            let score = 0.35 + ((frame_idx * 7 + pred_idx * 9) % 60) as f32 * 0.01;
            predictions.push(Detection {
                bbox: make_bbox(
                    gt.bbox.x1 + jitter,
                    gt.bbox.y1 + jitter,
                    gt.bbox.width(),
                    gt.bbox.height(),
                ),
                score,
                class_id,
            });
        }

        frames.push(DetectionDatasetFrame {
            ground_truth,
            predictions,
        });
    }

    frames
}

fn build_tracking_dataset(
    frame_count: usize,
    tracks_per_frame: usize,
) -> Vec<TrackingDatasetFrame> {
    let mut frames = Vec::with_capacity(frame_count);

    for frame_idx in 0..frame_count {
        let mut ground_truth = Vec::with_capacity(tracks_per_frame);
        let mut predictions = Vec::with_capacity(tracks_per_frame + 1);

        for track_idx in 0..tracks_per_frame {
            let x = (track_idx * 20) as f32 + frame_idx as f32 * 0.75;
            let y = (track_idx * 12) as f32 + frame_idx as f32 * 0.45;
            let class_id = track_idx % 2;

            ground_truth.push(GroundTruthTrack {
                object_id: track_idx as u64 + 1,
                bbox: make_bbox(x, y, 16.0, 20.0),
                class_id,
            });

            let track_id = if frame_idx % 31 == 0 && track_idx == 0 {
                9_999
            } else {
                track_idx as u64 + 100
            };
            let pred_jitter = if track_idx % 4 == 0 { 0.6 } else { 0.0 };
            predictions.push(TrackedDetection {
                track_id,
                detection: Detection {
                    bbox: make_bbox(x + pred_jitter, y + pred_jitter, 16.0, 20.0),
                    score: 0.9,
                    class_id,
                },
            });
        }

        predictions.push(TrackedDetection {
            track_id: frame_idx as u64 + 5_000,
            detection: Detection {
                bbox: make_bbox(220.0, 160.0, 10.0, 10.0),
                score: 0.55,
                class_id: 1,
            },
        });

        frames.push(TrackingDatasetFrame {
            ground_truth,
            predictions,
        });
    }

    frames
}

fn build_pipeline_durations(
    frame_count: usize,
) -> (Vec<Duration>, Vec<Duration>, Vec<Duration>, Vec<Duration>) {
    let mut detect = Vec::with_capacity(frame_count);
    let mut track = Vec::with_capacity(frame_count);
    let mut recognize = Vec::with_capacity(frame_count);
    let mut end_to_end = Vec::with_capacity(frame_count);

    for idx in 0..frame_count {
        let detect_us = 1_100 + (idx % 47) as u64 * 17;
        let track_us = 380 + (idx % 31) as u64 * 9;
        let recognize_us = 1_600 + (idx % 53) as u64 * 21;
        let overhead_us = 250 + (idx % 19) as u64 * 5;
        detect.push(Duration::from_micros(detect_us));
        track.push(Duration::from_micros(track_us));
        recognize.push(Duration::from_micros(recognize_us));
        end_to_end.push(Duration::from_micros(
            detect_us + track_us + recognize_us + overhead_us,
        ));
    }

    (detect, track, recognize, end_to_end)
}

fn bench_detection_eval_modes(c: &mut Criterion) {
    let dataset = build_detection_dataset(120, 8, 12);
    let views = detection_frames_as_view(&dataset);
    let config = DetectionEvalConfig {
        iou_threshold: 0.5,
        score_threshold: 0.4,
    };

    let mut group = c.benchmark_group("eval_detection_modes");

    group.bench_function("evaluate_detections_view_120f", |b| {
        b.iter(|| {
            let metrics = evaluate_detections(black_box(&views), config)
                .expect("detection eval should succeed");
            black_box(metrics.f1);
        });
    });

    group.bench_function("evaluate_detections_dataset_120f", |b| {
        b.iter(|| {
            let metrics = evaluate_detections_from_dataset(black_box(&dataset), config)
                .expect("detection eval should succeed");
            black_box(metrics.f1);
        });
    });

    group.finish();
}

fn bench_tracking_eval_modes(c: &mut Criterion) {
    let dataset = build_tracking_dataset(120, 8);
    let views = tracking_frames_as_view(&dataset);
    let config = TrackingEvalConfig { iou_threshold: 0.5 };

    let mut group = c.benchmark_group("eval_tracking_modes");

    group.bench_function("evaluate_tracking_view_120f", |b| {
        b.iter(|| {
            let metrics =
                evaluate_tracking(black_box(&views), config).expect("tracking eval should succeed");
            black_box(metrics.mota);
        });
    });

    group.bench_function("evaluate_tracking_dataset_120f", |b| {
        b.iter(|| {
            let metrics = evaluate_tracking_from_dataset(black_box(&dataset), config)
                .expect("tracking eval should succeed");
            black_box(metrics.mota);
        });
    });

    group.finish();
}

fn bench_pipeline_eval_modes(c: &mut Criterion) {
    let (detect, track, recognize, end_to_end) = build_pipeline_durations(512);
    let report = summarize_pipeline_durations(PipelineDurations {
        detect: &detect,
        track: &track,
        recognize: &recognize,
        end_to_end: &end_to_end,
    })
    .expect("pipeline summarize should succeed");
    let thresholds = PipelineBenchmarkThresholds {
        detect: StageThresholds {
            min_fps: Some(400.0),
            max_mean_ms: Some(3.0),
            max_p95_ms: Some(4.0),
        },
        track: StageThresholds {
            min_fps: Some(900.0),
            max_mean_ms: Some(1.0),
            max_p95_ms: Some(1.1),
        },
        recognize: StageThresholds {
            min_fps: Some(300.0),
            max_mean_ms: Some(4.0),
            max_p95_ms: Some(5.0),
        },
        end_to_end: StageThresholds {
            min_fps: Some(180.0),
            max_mean_ms: Some(8.0),
            max_p95_ms: Some(9.0),
        },
    };

    let mut group = c.benchmark_group("eval_pipeline_modes");

    group.bench_function("summarize_pipeline_durations_512f", |b| {
        b.iter(|| {
            let report = summarize_pipeline_durations(PipelineDurations {
                detect: black_box(&detect),
                track: black_box(&track),
                recognize: black_box(&recognize),
                end_to_end: black_box(&end_to_end),
            })
            .expect("pipeline summarize should succeed");
            black_box(report.end_to_end.p95_ms);
        });
    });

    group.bench_function("validate_pipeline_thresholds_512f", |b| {
        b.iter(|| {
            let violations =
                validate_pipeline_benchmark_thresholds(black_box(&report), black_box(&thresholds));
            black_box(violations.len());
        });
    });

    group.finish();
}

fn bench_eval(c: &mut Criterion) {
    bench_detection_eval_modes(c);
    bench_tracking_eval_modes(c);
    bench_pipeline_eval_modes(c);
}

criterion_group!(benches, bench_eval);
criterion_main!(benches);
