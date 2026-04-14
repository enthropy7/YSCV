use crate::{Track, Tracker, TrackerConfig};
use yscv_detect::{BoundingBox, CLASS_ID_FACE, CLASS_ID_PERSON, Detection};

fn det(x1: f32, y1: f32, x2: f32, y2: f32, score: f32) -> Detection {
    det_with_class(x1, y1, x2, y2, score, CLASS_ID_PERSON)
}

fn det_with_class(x1: f32, y1: f32, x2: f32, y2: f32, score: f32, class_id: usize) -> Detection {
    Detection {
        bbox: BoundingBox { x1, y1, x2, y2 },
        score,
        class_id,
    }
}

#[test]
fn tracker_keeps_stable_ids_for_smooth_motion() {
    let mut tracker = Tracker::new(TrackerConfig {
        match_iou_threshold: 0.2,
        max_missed_frames: 3,
        max_tracks: 16,
    })
    .expect("valid tracker config");

    let out1 = tracker.update(&[det(1.0, 1.0, 3.0, 3.0, 0.9)]);
    assert_eq!(out1.len(), 1);
    let id1 = out1[0].track_id;

    let out2 = tracker.update(&[det(1.2, 1.1, 3.2, 3.1, 0.8)]);
    assert_eq!(out2.len(), 1);
    assert_eq!(out2[0].track_id, id1);
}

#[test]
fn tracker_creates_new_track_for_far_apart_detection() {
    let mut tracker = Tracker::new(TrackerConfig {
        match_iou_threshold: 0.3,
        max_missed_frames: 3,
        max_tracks: 16,
    })
    .expect("valid tracker config");

    let out1 = tracker.update(&[det(0.0, 0.0, 2.0, 2.0, 0.9)]);
    let id1 = out1[0].track_id;
    let out2 = tracker.update(&[det(5.0, 5.0, 7.0, 7.0, 0.9)]);
    let id2 = out2[0].track_id;
    assert_ne!(id1, id2);
}

#[test]
fn tracker_removes_stale_tracks_after_miss_budget() {
    let mut tracker = Tracker::new(TrackerConfig {
        match_iou_threshold: 0.3,
        max_missed_frames: 1,
        max_tracks: 16,
    })
    .expect("valid tracker config");

    tracker.update(&[det(0.0, 0.0, 1.0, 1.0, 0.9)]);
    assert_eq!(tracker.active_tracks().len(), 1);

    tracker.update(&[]);
    assert_eq!(tracker.active_tracks().len(), 1);

    tracker.update(&[]);
    assert_eq!(tracker.active_tracks().len(), 0);
}

#[test]
fn tracker_reuses_id_after_short_occlusion_with_motion_prediction() {
    let mut tracker = Tracker::new(TrackerConfig {
        match_iou_threshold: 0.3,
        max_missed_frames: 2,
        max_tracks: 16,
    })
    .expect("valid tracker config");

    let initial = tracker.update(&[det(0.0, 0.0, 2.0, 2.0, 0.9)]);
    assert_eq!(initial.len(), 1);
    let track_id = initial[0].track_id;

    let next = tracker.update(&[det(1.0, 0.0, 3.0, 2.0, 0.9)]);
    assert_eq!(next[0].track_id, track_id);

    tracker.update(&[]);

    let recovered = tracker.update(&[det(3.0, 0.0, 5.0, 2.0, 0.9)]);
    assert_eq!(recovered.len(), 1);
    assert_eq!(recovered[0].track_id, track_id);
}

#[test]
fn tracker_people_count_reflects_active_tracks() {
    let mut tracker = Tracker::new(TrackerConfig::default()).expect("default config valid");
    tracker.update(&[det(0.0, 0.0, 2.0, 2.0, 0.9), det(3.0, 3.0, 5.0, 5.0, 0.8)]);
    assert_eq!(tracker.people_count(), 2);
}

#[test]
fn tracker_count_by_class_reflects_active_tracks() {
    let mut tracker = Tracker::new(TrackerConfig::default()).expect("default config valid");
    tracker.update(&[
        det_with_class(0.0, 0.0, 2.0, 2.0, 0.9, CLASS_ID_PERSON),
        det_with_class(3.0, 3.0, 5.0, 5.0, 0.8, CLASS_ID_FACE),
        det_with_class(6.0, 6.0, 8.0, 8.0, 0.7, CLASS_ID_FACE),
    ]);
    assert_eq!(tracker.count_by_class(CLASS_ID_PERSON), 1);
    assert_eq!(tracker.count_by_class(CLASS_ID_FACE), 2);
}

#[test]
fn config_validation_rejects_invalid_values() {
    assert!(
        Tracker::new(TrackerConfig {
            match_iou_threshold: 1.5,
            max_missed_frames: 1,
            max_tracks: 16,
        })
        .is_err()
    );

    assert!(
        Tracker::new(TrackerConfig {
            match_iou_threshold: 0.5,
            max_missed_frames: 1,
            max_tracks: 0,
        })
        .is_err()
    );
}

#[test]
fn active_tracks_exposes_track_state() {
    let mut tracker = Tracker::new(TrackerConfig::default()).expect("default config valid");
    let out = tracker.update(&[det(1.0, 1.0, 2.0, 2.0, 0.7)]);
    let track_id = out[0].track_id;
    let tracks = tracker.active_tracks();
    assert_eq!(tracks.len(), 1);
    assert_eq!(
        tracks[0],
        Track {
            id: track_id,
            bbox: BoundingBox {
                x1: 1.0,
                y1: 1.0,
                x2: 2.0,
                y2: 2.0
            },
            score: 0.7,
            class_id: 0,
            age: 1,
            hits: 1,
            missed_frames: 0
        }
    );
}

#[test]
fn update_into_matches_update_output() {
    let config = TrackerConfig {
        match_iou_threshold: 0.2,
        max_missed_frames: 3,
        max_tracks: 16,
    };

    let detections = [det(1.0, 1.0, 3.0, 3.0, 0.9), det(4.0, 1.0, 6.0, 3.0, 0.8)];

    let mut tracker_a = Tracker::new(config).expect("valid tracker config");
    let out_a = tracker_a.update(&detections);

    let mut tracker_b = Tracker::new(config).expect("valid tracker config");
    let mut out_b = Vec::new();
    tracker_b.update_into(&detections, &mut out_b);

    assert_eq!(out_a, out_b);
}
