#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use yscv_detect::{
    BoundingBox, CLASS_ID_FACE, CLASS_ID_PERSON, Rgb8FaceDetectScratch, Rgb8PeopleDetectScratch,
    detect_faces_from_rgb8_with_scratch, detect_people_from_rgb8_with_scratch,
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

fn build_people_rgb8(width: usize, height: usize) -> Vec<u8> {
    let mut rgb8 = vec![18u8; width * height * 3];
    for y in 20..80 {
        for x in 24..54 {
            let base = (y * width + x) * 3;
            rgb8[base] = 235;
            rgb8[base + 1] = 235;
            rgb8[base + 2] = 235;
        }
    }
    for y in 30..96 {
        for x in 92..124 {
            let base = (y * width + x) * 3;
            rgb8[base] = 228;
            rgb8[base + 1] = 228;
            rgb8[base + 2] = 228;
        }
    }
    rgb8
}

fn build_faces_rgb8(width: usize, height: usize) -> Vec<u8> {
    let background = [26u8, 26u8, 31u8];
    let skin = [199u8, 153u8, 117u8];
    let mut rgb8 = vec![0u8; width * height * 3];

    for y in 0..height {
        for x in 0..width {
            let base = (y * width + x) * 3;
            let in_face_patch = ((18..74).contains(&y) && (20..66).contains(&x))
                || ((28..96).contains(&y) && (88..134).contains(&x));
            let color = if in_face_patch { skin } else { background };
            rgb8[base] = color[0];
            rgb8[base + 1] = color[1];
            rgb8[base + 2] = color[2];
        }
    }

    rgb8
}

fn bench_cli_runtime_modes(c: &mut Criterion) {
    let width = 160usize;
    let height = 120usize;

    let people_rgb8 = build_people_rgb8(width, height);
    let mut people_scratch = Rgb8PeopleDetectScratch::default();
    let mut people_tracker = Tracker::new(TrackerConfig {
        match_iou_threshold: 0.3,
        max_missed_frames: 10,
        max_tracks: 64,
    })
    .expect("valid tracker config");
    let mut people_recognizer = Recognizer::new(0.6).expect("valid threshold");
    people_recognizer
        .enroll(
            "person-ref",
            Tensor::from_vec(vec![3], vec![0.25, 0.40, 0.12]).expect("valid tensor"),
        )
        .expect("enroll person");
    let mut tracked_people: Vec<TrackedDetection> = Vec::new();

    let faces_rgb8 = build_faces_rgb8(width, height);
    let mut face_scratch = Rgb8FaceDetectScratch::default();
    let mut face_tracker = Tracker::new(TrackerConfig {
        match_iou_threshold: 0.3,
        max_missed_frames: 10,
        max_tracks: 64,
    })
    .expect("valid tracker config");
    let mut face_recognizer = Recognizer::new(0.6).expect("valid threshold");
    face_recognizer
        .enroll(
            "face-ref",
            Tensor::from_vec(vec![3], vec![0.27, 0.38, 0.10]).expect("valid tensor"),
        )
        .expect("enroll face");
    let mut tracked_faces: Vec<TrackedDetection> = Vec::new();

    let mut group = c.benchmark_group("cli_runtime_modes");

    group.bench_function("people_detect_track_recognize_rgb8_160x120", |b| {
        b.iter(|| {
            let detections = detect_people_from_rgb8_with_scratch(
                (black_box(width), black_box(height)),
                black_box(&people_rgb8),
                0.5,
                64,
                0.4,
                64,
                black_box(&mut people_scratch),
            )
            .expect("people detection");

            people_tracker.update_into(&detections, &mut tracked_people);

            let mut known_count = 0usize;
            for item in &tracked_people {
                let embedding =
                    bbox_embedding_components(item.detection.bbox, width as f32, height as f32);
                let recognition = people_recognizer
                    .recognize_slice(&embedding)
                    .expect("recognition");
                if recognition.identity.is_some() {
                    known_count += 1;
                }
            }

            black_box(known_count);
            black_box(people_tracker.count_by_class(CLASS_ID_PERSON));
        });
    });

    group.bench_function("face_detect_track_recognize_rgb8_160x120", |b| {
        b.iter(|| {
            let detections = detect_faces_from_rgb8_with_scratch(
                (black_box(width), black_box(height)),
                black_box(&faces_rgb8),
                0.35,
                64,
                0.4,
                64,
                black_box(&mut face_scratch),
            )
            .expect("face detection");

            face_tracker.update_into(&detections, &mut tracked_faces);

            let mut known_count = 0usize;
            for item in &tracked_faces {
                let embedding =
                    bbox_embedding_components(item.detection.bbox, width as f32, height as f32);
                let recognition = face_recognizer
                    .recognize_slice(&embedding)
                    .expect("recognition");
                if recognition.identity.is_some() {
                    known_count += 1;
                }
            }

            black_box(known_count);
            black_box(face_tracker.count_by_class(CLASS_ID_FACE));
        });
    });

    group.finish();
}

criterion_group!(benches, bench_cli_runtime_modes);
criterion_main!(benches);
