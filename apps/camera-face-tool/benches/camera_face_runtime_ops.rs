#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use serde_json::{Value, json};
use yscv_detect::{
    BoundingBox, CLASS_ID_FACE, Rgb8FaceDetectScratch, detect_faces_from_rgb8_with_scratch,
};
use yscv_recognize::Recognizer;
use yscv_tensor::Tensor;
use yscv_track::{TrackedDetection, Tracker, TrackerConfig};

fn bbox_embedding_components(bbox: BoundingBox, frame_width: f32, frame_height: f32) -> [f32; 3] {
    let cx = ((bbox.x1 + bbox.x2) * 0.5) / frame_width;
    let cy = ((bbox.y1 + bbox.y2) * 0.5) / frame_height;
    let area = bbox.area() / (frame_width * frame_height);
    [cx, cy, area]
}

fn build_faces_rgb8(width: usize, height: usize) -> Vec<u8> {
    let background = [26u8, 26u8, 31u8];
    let skin = [199u8, 153u8, 117u8];
    let mut rgb8 = vec![0u8; width * height * 3];

    for y in 0..height {
        for x in 0..width {
            let base = (y * width + x) * 3;
            let color = if ((16..74).contains(&y) && (18..66).contains(&x))
                || ((30..98).contains(&y) && (90..140).contains(&x))
            {
                skin
            } else {
                background
            };
            rgb8[base] = color[0];
            rgb8[base + 1] = color[1];
            rgb8[base + 2] = color[2];
        }
    }

    rgb8
}

fn bench_camera_face_runtime_modes(c: &mut Criterion) {
    let width = 160usize;
    let height = 120usize;
    let rgb8 = build_faces_rgb8(width, height);

    let mut face_scratch = Rgb8FaceDetectScratch::default();
    let mut tracker = Tracker::new(TrackerConfig {
        match_iou_threshold: 0.3,
        max_missed_frames: 10,
        max_tracks: 64,
    })
    .expect("valid tracker config");
    let mut recognizer = Recognizer::new(0.6).expect("valid threshold");
    recognizer
        .enroll(
            "face-ref",
            Tensor::from_vec(vec![3], vec![0.27, 0.38, 0.10]).expect("valid tensor"),
        )
        .expect("enroll face");

    let mut tracked: Vec<TrackedDetection> = Vec::new();
    let mut tracked_event_records: Vec<Value> = Vec::new();

    let mut group = c.benchmark_group("camera_face_runtime_modes");

    group.bench_function("face_detect_track_recognize_rgb8_160x120", |b| {
        b.iter(|| {
            let detections = detect_faces_from_rgb8_with_scratch(
                (black_box(width), black_box(height)),
                black_box(&rgb8),
                0.35,
                64,
                0.4,
                64,
                black_box(&mut face_scratch),
            )
            .expect("face detection");

            tracker.update_into(&detections, &mut tracked);

            let mut known_count = 0usize;
            for item in &tracked {
                let embedding =
                    bbox_embedding_components(item.detection.bbox, width as f32, height as f32);
                let recognition = recognizer.recognize_slice(&embedding).expect("recognition");
                if recognition.identity.is_some() {
                    known_count += 1;
                }
            }

            black_box(known_count);
            black_box(tracker.count_by_class(CLASS_ID_FACE));
        });
    });

    group.bench_function(
        "face_detect_track_recognize_event_payload_rgb8_160x120",
        |b| {
            b.iter(|| {
                let detections = detect_faces_from_rgb8_with_scratch(
                    (black_box(width), black_box(height)),
                    black_box(&rgb8),
                    0.35,
                    64,
                    0.4,
                    64,
                    black_box(&mut face_scratch),
                )
                .expect("face detection");

                tracker.update_into(&detections, &mut tracked);

                tracked_event_records.clear();
                if tracked_event_records.capacity() < tracked.len() {
                    tracked_event_records.reserve(tracked.len() - tracked_event_records.capacity());
                }

                for (idx, item) in tracked.iter().enumerate() {
                    let embedding =
                        bbox_embedding_components(item.detection.bbox, width as f32, height as f32);
                    let recognition = recognizer.recognize_slice(&embedding).expect("recognition");
                    tracked_event_records.push(json!({
                        "det_index": idx,
                        "track_id": item.track_id,
                        "class_id": item.detection.class_id,
                        "score": item.detection.score,
                        "identity": recognition.identity,
                        "similarity": recognition.score,
                        "bbox": {
                            "x1": item.detection.bbox.x1,
                            "y1": item.detection.bbox.y1,
                            "x2": item.detection.bbox.x2,
                            "y2": item.detection.bbox.y2,
                        },
                    }));
                }

                black_box(tracked_event_records.len());
                black_box(tracker.count_by_class(CLASS_ID_FACE));
            });
        },
    );

    group.finish();
}

criterion_group!(benches, bench_camera_face_runtime_modes);
criterion_main!(benches);
