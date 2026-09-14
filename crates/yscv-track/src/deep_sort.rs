//! DeepSORT-style multi-object tracker with appearance features and cascade matching.

use yscv_detect::{Detection, iou};

use crate::KalmanFilter;
use crate::hungarian::hungarian_assignment;

/// Configuration for the DeepSORT tracker.
#[derive(Debug, Clone)]
pub struct DeepSortConfig {
    /// Maximum cosine distance for appearance matching.
    pub max_cosine_distance: f32,
    /// Maximum IoU distance for fallback matching.
    pub max_iou_distance: f32,
    /// Number of frames to keep a track alive without detection.
    pub max_age: usize,
    /// Number of consecutive hits to confirm a track.
    pub n_init: usize,
}

impl Default for DeepSortConfig {
    fn default() -> Self {
        Self {
            max_cosine_distance: 0.3,
            max_iou_distance: 0.7,
            max_age: 30,
            n_init: 3,
        }
    }
}

/// Track state in the DeepSORT lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackState {
    /// Track has been created but not yet confirmed.
    Tentative,
    /// Track has been confirmed (enough consecutive hits).
    Confirmed,
    /// Track has been marked for deletion.
    Deleted,
}

/// A tracked object with appearance features.
#[derive(Debug, Clone)]
pub struct DeepSortTrack {
    /// Unique track identifier.
    pub id: usize,
    /// Current track state.
    pub state: TrackState,
    /// Kalman filter for motion prediction.
    pub kalman: KalmanFilter,
    /// Feature history for appearance matching (last N features).
    pub features: Vec<Vec<f32>>,
    /// Total number of frames this track has been matched.
    pub hits: usize,
    /// Total number of frames since track creation.
    pub age: usize,
    /// Number of consecutive frames without a matching detection.
    pub time_since_update: usize,
}

/// DeepSORT multi-object tracker.
pub struct DeepSortTracker {
    config: DeepSortConfig,
    tracks: Vec<DeepSortTrack>,
    next_id: usize,
}

impl DeepSortTracker {
    /// Create a new DeepSORT tracker with the given configuration.
    pub const fn new(config: DeepSortConfig) -> Self {
        Self {
            config,
            tracks: Vec::new(),
            next_id: 1,
        }
    }

    /// Predict the next state for all tracks using their Kalman filters.
    pub fn predict(&mut self) {
        for track in &mut self.tracks {
            track.kalman.predict();
            track.age += 1;
            track.time_since_update += 1;
        }
    }

    /// Main update step: match detections to tracks and update state.
    ///
    /// `detections` — the current frame's detections.
    /// `features` — optional appearance feature vectors, one per detection.
    pub fn update(&mut self, detections: &[Detection], features: Option<&[Vec<f32>]>) {
        // Split track indices into confirmed and unconfirmed.
        let mut confirmed_indices: Vec<usize> = Vec::new();
        let mut unconfirmed_indices: Vec<usize> = Vec::new();
        for (i, track) in self.tracks.iter().enumerate() {
            match track.state {
                TrackState::Confirmed => confirmed_indices.push(i),
                TrackState::Tentative => unconfirmed_indices.push(i),
                TrackState::Deleted => {}
            }
        }

        let n_dets = detections.len();
        let mut matched_tracks: Vec<bool> = vec![false; self.tracks.len()];
        let mut matched_dets: Vec<bool> = vec![false; n_dets];

        // ── Stage 1: Cascade matching (appearance) on confirmed tracks ──
        if let Some(feats) = features
            && !confirmed_indices.is_empty()
            && !detections.is_empty()
        {
            let n_tracks = confirmed_indices.len();
            let mut cost_matrix = vec![vec![0.0_f32; n_dets]; n_tracks];
            for (ti, &track_idx) in confirmed_indices.iter().enumerate() {
                let track = &self.tracks[track_idx];
                for dj in 0..n_dets {
                    if track.features.is_empty() {
                        // No features yet — use a high cost so IoU matching picks it up.
                        cost_matrix[ti][dj] = self.config.max_cosine_distance + 1.0;
                    } else {
                        cost_matrix[ti][dj] = min_cosine_distance(&feats[dj], &track.features);
                    }
                }
            }

            let assignments = hungarian_assignment(&cost_matrix);
            for (ti, dj) in assignments {
                if cost_matrix[ti][dj] <= self.config.max_cosine_distance {
                    let track_idx = confirmed_indices[ti];
                    matched_tracks[track_idx] = true;
                    matched_dets[dj] = true;
                    self.update_track(track_idx, &detections[dj], Some(&feats[dj]));
                }
            }
        }

        // ── Stage 2: IoU matching on remaining tracks vs remaining detections ──
        // Collect unmatched track indices (confirmed that weren't matched + all unconfirmed).
        let mut iou_track_indices: Vec<usize> = Vec::new();
        for &ti in &confirmed_indices {
            if !matched_tracks[ti] {
                iou_track_indices.push(ti);
            }
        }
        iou_track_indices.extend_from_slice(&unconfirmed_indices);

        let unmatched_det_indices: Vec<usize> = (0..n_dets).filter(|&d| !matched_dets[d]).collect();

        if !iou_track_indices.is_empty() && !unmatched_det_indices.is_empty() {
            let n_t = iou_track_indices.len();
            let n_d = unmatched_det_indices.len();
            let mut cost_matrix = vec![vec![0.0_f32; n_d]; n_t];
            for (ti, &track_idx) in iou_track_indices.iter().enumerate() {
                let predicted = self.tracks[track_idx].kalman.bbox();
                for (dj, &det_idx) in unmatched_det_indices.iter().enumerate() {
                    let iou_val = iou(predicted, detections[det_idx].bbox);
                    cost_matrix[ti][dj] = 1.0 - iou_val; // IoU distance
                }
            }

            let assignments = hungarian_assignment(&cost_matrix);
            for (ti, dj) in assignments {
                if cost_matrix[ti][dj] <= self.config.max_iou_distance {
                    let track_idx = iou_track_indices[ti];
                    let det_idx = unmatched_det_indices[dj];
                    matched_tracks[track_idx] = true;
                    matched_dets[det_idx] = true;
                    let feat = features.map(|f| &f[det_idx] as &[f32]);
                    self.update_track(track_idx, &detections[det_idx], feat);
                }
            }
        }

        // ── Create new tracks for unmatched detections ──
        for det_idx in 0..n_dets {
            if matched_dets[det_idx] {
                continue;
            }
            let feat = features.map(|f| f[det_idx].clone());
            self.create_track(&detections[det_idx], feat);
        }

        // ── Mark unmatched tracks ──
        for (i, track) in self.tracks.iter_mut().enumerate() {
            if matched_tracks.get(i).copied().unwrap_or(false) {
                continue;
            }
            if track.state == TrackState::Deleted {
                continue;
            }
            // For newly created tracks (not in matched_tracks vec), skip.
            if i >= matched_tracks.len() {
                continue;
            }
            if track.state == TrackState::Tentative && track.time_since_update > 0 {
                track.state = TrackState::Deleted;
            } else if track.time_since_update > self.config.max_age {
                track.state = TrackState::Deleted;
            }
        }

        // Remove deleted tracks.
        self.tracks.retain(|t| t.state != TrackState::Deleted);
    }

    /// Get all active (non-deleted) tracks.
    pub fn tracks(&self) -> &[DeepSortTrack] {
        &self.tracks
    }

    /// Get only confirmed tracks.
    pub fn confirmed_tracks(&self) -> Vec<&DeepSortTrack> {
        self.tracks
            .iter()
            .filter(|t| t.state == TrackState::Confirmed)
            .collect()
    }

    fn update_track(&mut self, track_idx: usize, detection: &Detection, feature: Option<&[f32]>) {
        let track = &mut self.tracks[track_idx];
        let bbox = detection.bbox;
        let cx = (bbox.x1 + bbox.x2) * 0.5;
        let cy = (bbox.y1 + bbox.y2) * 0.5;
        let w = bbox.width();
        let h = bbox.height();
        track.kalman.update([cx, cy, w, h]);
        track.hits += 1;
        track.time_since_update = 0;
        if let Some(feat) = feature {
            track.features.push(feat.to_vec());
            // Keep only last 100 features.
            if track.features.len() > 100 {
                track.features.remove(0);
            }
        }
        if track.state == TrackState::Tentative && track.hits >= self.config.n_init {
            track.state = TrackState::Confirmed;
        }
    }

    fn create_track(&mut self, detection: &Detection, feature: Option<Vec<f32>>) {
        let id = self.next_id;
        self.next_id += 1;
        let kalman = KalmanFilter::new(detection.bbox);
        let mut features = Vec::new();
        if let Some(f) = feature {
            features.push(f);
        }
        self.tracks.push(DeepSortTrack {
            id,
            state: TrackState::Tentative,
            kalman,
            features,
            hits: 1,
            age: 1,
            time_since_update: 0,
        });
    }
}

/// Cosine distance between two feature vectors.
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 {
        return 1.0; // Maximum distance if either vector is zero.
    }
    1.0 - (dot / denom)
}

/// Minimum cosine distance between a feature and a gallery of features.
fn min_cosine_distance(feature: &[f32], gallery: &[Vec<f32>]) -> f32 {
    gallery
        .iter()
        .map(|g| cosine_distance(feature, g))
        .fold(f32::INFINITY, f32::min)
}

#[cfg(test)]
mod tests {
    use super::*;
    use yscv_detect::BoundingBox;

    fn make_detection(x1: f32, y1: f32, x2: f32, y2: f32) -> Detection {
        Detection {
            bbox: BoundingBox { x1, y1, x2, y2 },
            score: 0.9,
            class_id: 0,
        }
    }

    #[test]
    fn test_deep_sort_creation() {
        let tracker = DeepSortTracker::new(DeepSortConfig::default());
        assert!(tracker.tracks().is_empty());
        assert!(tracker.confirmed_tracks().is_empty());
    }

    #[test]
    fn test_deep_sort_single_detection() {
        let mut tracker = DeepSortTracker::new(DeepSortConfig::default());
        let dets = [make_detection(10.0, 10.0, 50.0, 50.0)];
        tracker.predict();
        tracker.update(&dets, None);
        assert_eq!(tracker.tracks().len(), 1);
        assert_eq!(tracker.tracks()[0].state, TrackState::Tentative);
    }

    #[test]
    fn test_deep_sort_track_confirmation() {
        let config = DeepSortConfig {
            n_init: 3,
            ..DeepSortConfig::default()
        };
        let mut tracker = DeepSortTracker::new(config);
        let det = make_detection(10.0, 10.0, 50.0, 50.0);

        // First detection creates tentative track.
        tracker.predict();
        tracker.update(&[det], None);
        assert_eq!(tracker.tracks()[0].state, TrackState::Tentative);

        // Second hit.
        tracker.predict();
        tracker.update(&[det], None);
        assert_eq!(tracker.tracks()[0].state, TrackState::Tentative);

        // Third hit → confirmed.
        tracker.predict();
        tracker.update(&[det], None);
        assert_eq!(tracker.tracks()[0].state, TrackState::Confirmed);
    }

    #[test]
    fn test_deep_sort_track_deletion() {
        let config = DeepSortConfig {
            max_age: 2,
            n_init: 1,
            ..DeepSortConfig::default()
        };
        let mut tracker = DeepSortTracker::new(config);
        let det = make_detection(10.0, 10.0, 50.0, 50.0);

        // Create and confirm a track.
        tracker.predict();
        tracker.update(&[det], None);
        assert_eq!(tracker.tracks().len(), 1);

        // No detections for max_age+1 frames → deleted.
        for _ in 0..4 {
            tracker.predict();
            tracker.update(&[], None);
        }
        assert!(tracker.tracks().is_empty());
    }

    #[test]
    fn test_deep_sort_iou_matching() {
        let config = DeepSortConfig {
            n_init: 1,
            max_age: 5,
            ..DeepSortConfig::default()
        };
        let mut tracker = DeepSortTracker::new(config);

        // Frame 1: two detections.
        let dets1 = [
            make_detection(10.0, 10.0, 50.0, 50.0),
            make_detection(100.0, 100.0, 150.0, 150.0),
        ];
        tracker.predict();
        tracker.update(&dets1, None);
        assert_eq!(tracker.tracks().len(), 2);
        let id0 = tracker.tracks()[0].id;
        let id1 = tracker.tracks()[1].id;

        // Frame 2: same detections, slightly moved.
        let dets2 = [
            make_detection(12.0, 12.0, 52.0, 52.0),
            make_detection(102.0, 102.0, 152.0, 152.0),
        ];
        tracker.predict();
        tracker.update(&dets2, None);
        assert_eq!(tracker.tracks().len(), 2);

        // Track IDs should be preserved (same objects matched).
        let ids: Vec<usize> = tracker.tracks().iter().map(|t| t.id).collect();
        assert!(ids.contains(&id0));
        assert!(ids.contains(&id1));
    }

    #[test]
    fn test_cosine_distance() {
        // Identical vectors → distance 0.
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_distance(&a, &b) - 0.0).abs() < 1e-6);

        // Orthogonal vectors → distance 1.
        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_distance(&a, &c) - 1.0).abs() < 1e-6);

        // Opposite vectors → distance 2.
        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine_distance(&a, &d) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_deep_sort_multiple_objects_tracked() {
        let config = DeepSortConfig {
            n_init: 1,
            max_age: 5,
            ..DeepSortConfig::default()
        };
        let mut tracker = DeepSortTracker::new(config);

        // Three detections far apart.
        let dets = [
            make_detection(10.0, 10.0, 50.0, 50.0),
            make_detection(100.0, 100.0, 140.0, 140.0),
            make_detection(200.0, 200.0, 240.0, 240.0),
        ];
        tracker.predict();
        tracker.update(&dets, None);
        assert_eq!(tracker.tracks().len(), 3);

        let ids: Vec<usize> = tracker.tracks().iter().map(|t| t.id).collect();
        // All IDs should be unique.
        let mut unique = ids.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), 3);
    }

    #[test]
    fn test_deep_sort_occlusion_and_reappearance() {
        // Confirmed tracks survive occlusion up to max_age.
        let config = DeepSortConfig {
            n_init: 2,
            max_age: 5,
            ..DeepSortConfig::default()
        };
        let mut tracker = DeepSortTracker::new(config);

        let det = make_detection(10.0, 10.0, 50.0, 50.0);
        // First frame: creates tentative track.
        tracker.predict();
        tracker.update(&[det], None);
        assert_eq!(tracker.tracks()[0].state, TrackState::Tentative);
        let original_id = tracker.tracks()[0].id;

        // Second frame: confirms the track (hits=2 >= n_init=2).
        tracker.predict();
        tracker.update(&[det], None);
        assert_eq!(tracker.tracks()[0].state, TrackState::Confirmed);

        // Object disappears for 3 frames (within max_age=5).
        for _ in 0..3 {
            tracker.predict();
            tracker.update(&[], None);
        }
        // Confirmed track should still exist (time_since_update=3 <= max_age=5).
        assert!(!tracker.tracks().is_empty());

        // Object reappears at same position.
        tracker.predict();
        tracker.update(&[det], None);
        let ids: Vec<usize> = tracker.tracks().iter().map(|t| t.id).collect();
        assert!(ids.contains(&original_id));
    }

    #[test]
    fn test_deep_sort_id_stability_smooth_motion() {
        let config = DeepSortConfig {
            n_init: 1,
            max_age: 5,
            ..DeepSortConfig::default()
        };
        let mut tracker = DeepSortTracker::new(config);

        // Object moves smoothly across 5 frames.
        let positions = [
            (10.0, 10.0, 50.0, 50.0),
            (12.0, 12.0, 52.0, 52.0),
            (14.0, 14.0, 54.0, 54.0),
            (16.0, 16.0, 56.0, 56.0),
            (18.0, 18.0, 58.0, 58.0),
        ];

        tracker.predict();
        tracker.update(
            &[make_detection(
                positions[0].0,
                positions[0].1,
                positions[0].2,
                positions[0].3,
            )],
            None,
        );
        let original_id = tracker.tracks()[0].id;

        for &(x1, y1, x2, y2) in &positions[1..] {
            tracker.predict();
            tracker.update(&[make_detection(x1, y1, x2, y2)], None);
            assert_eq!(tracker.tracks().len(), 1);
            assert_eq!(tracker.tracks()[0].id, original_id);
        }
    }

    #[test]
    fn test_deep_sort_deletion_after_max_age() {
        let config = DeepSortConfig {
            max_age: 3,
            n_init: 1,
            ..DeepSortConfig::default()
        };
        let mut tracker = DeepSortTracker::new(config);

        tracker.predict();
        tracker.update(&[make_detection(10.0, 10.0, 50.0, 50.0)], None);
        assert_eq!(tracker.tracks().len(), 1);

        // Exactly max_age frames without detection.
        for _ in 0..3 {
            tracker.predict();
            tracker.update(&[], None);
        }
        // Should still be alive (time_since_update==3, max_age==3, deletion is >).
        // One more frame to exceed max_age.
        tracker.predict();
        tracker.update(&[], None);
        assert!(tracker.tracks().is_empty());
    }

    #[test]
    fn test_deep_sort_new_track_far_apart() {
        let config = DeepSortConfig {
            n_init: 1,
            max_age: 5,
            ..DeepSortConfig::default()
        };
        let mut tracker = DeepSortTracker::new(config);

        // Create and confirm a track with two hits.
        let det1 = make_detection(10.0, 10.0, 50.0, 50.0);
        tracker.predict();
        tracker.update(&[det1], None);
        let id1 = tracker.tracks()[0].id;
        tracker.predict();
        tracker.update(&[det1], None);

        // Detection very far away: original track unmatched, new track created.
        tracker.predict();
        tracker.update(&[make_detection(500.0, 500.0, 540.0, 540.0)], None);

        let ids: Vec<usize> = tracker.tracks().iter().map(|t| t.id).collect();
        assert!(ids.len() >= 2);
        assert!(
            ids.iter().any(|&id| id != id1),
            "Should have created a new track"
        );
    }

    #[test]
    fn test_deep_sort_empty_detections_ages_tracks() {
        let config = DeepSortConfig {
            n_init: 1,
            max_age: 10,
            ..DeepSortConfig::default()
        };
        let mut tracker = DeepSortTracker::new(config);

        let det = make_detection(10.0, 10.0, 50.0, 50.0);
        // Create and confirm track with two hits so it survives a miss.
        tracker.predict();
        tracker.update(&[det], None);
        tracker.predict();
        tracker.update(&[det], None);
        assert_eq!(tracker.tracks()[0].time_since_update, 0);
        assert_eq!(tracker.tracks()[0].state, TrackState::Confirmed);

        // Empty frame should increment time_since_update for confirmed tracks.
        tracker.predict();
        tracker.update(&[], None);
        assert_eq!(tracker.tracks().len(), 1);
        assert!(tracker.tracks()[0].time_since_update > 0);
    }

    #[test]
    fn test_deep_sort_single_detection_stable_id() {
        let config = DeepSortConfig {
            n_init: 1,
            max_age: 5,
            ..DeepSortConfig::default()
        };
        let mut tracker = DeepSortTracker::new(config);

        let det = make_detection(20.0, 20.0, 60.0, 60.0);

        tracker.predict();
        tracker.update(&[det], None);
        let id = tracker.tracks()[0].id;

        // Repeat same detection for 10 frames.
        for _ in 0..10 {
            tracker.predict();
            tracker.update(&[det], None);
            assert_eq!(tracker.tracks().len(), 1);
            assert_eq!(tracker.tracks()[0].id, id);
        }
    }

    #[test]
    fn test_deep_sort_overlapping_detections() {
        let config = DeepSortConfig {
            n_init: 1,
            max_age: 5,
            ..DeepSortConfig::default()
        };
        let mut tracker = DeepSortTracker::new(config);

        // Multiple heavily overlapping detections.
        let dets = [
            make_detection(10.0, 10.0, 50.0, 50.0),
            make_detection(12.0, 12.0, 52.0, 52.0),
            make_detection(14.0, 14.0, 54.0, 54.0),
        ];
        tracker.predict();
        tracker.update(&dets, None);
        // Each detection should create a track (all are unmatched initially).
        assert_eq!(tracker.tracks().len(), 3);
    }

    #[test]
    fn test_deep_sort_config_variations_max_age() {
        // Very short max_age.
        let config = DeepSortConfig {
            max_age: 1,
            n_init: 1,
            ..DeepSortConfig::default()
        };
        let mut tracker = DeepSortTracker::new(config);

        tracker.predict();
        tracker.update(&[make_detection(10.0, 10.0, 50.0, 50.0)], None);

        // Two empty frames should delete with max_age=1.
        tracker.predict();
        tracker.update(&[], None);
        tracker.predict();
        tracker.update(&[], None);
        assert!(tracker.tracks().is_empty());
    }

    #[test]
    fn test_deep_sort_config_variations_n_init() {
        // Need 5 hits to confirm.
        let config = DeepSortConfig {
            n_init: 5,
            max_age: 30,
            ..DeepSortConfig::default()
        };
        let mut tracker = DeepSortTracker::new(config);
        let det = make_detection(10.0, 10.0, 50.0, 50.0);

        for i in 0..5 {
            tracker.predict();
            tracker.update(&[det], None);
            if i < 4 {
                assert_eq!(tracker.tracks()[0].state, TrackState::Tentative);
            }
        }
        assert_eq!(tracker.tracks()[0].state, TrackState::Confirmed);
    }

    #[test]
    fn test_deep_sort_appearance_matching_with_features() {
        let config = DeepSortConfig {
            n_init: 1,
            max_age: 5,
            max_cosine_distance: 0.5,
            ..DeepSortConfig::default()
        };
        let mut tracker = DeepSortTracker::new(config);

        let det = make_detection(10.0, 10.0, 50.0, 50.0);
        let feat = vec![vec![1.0, 0.0, 0.0]];
        tracker.predict();
        tracker.update(&[det], Some(&feat));
        let id = tracker.tracks()[0].id;
        assert!(!tracker.tracks()[0].features.is_empty());

        // Same feature, slightly moved detection.
        let det2 = make_detection(12.0, 12.0, 52.0, 52.0);
        let feat2 = vec![vec![0.99, 0.1, 0.0]]; // similar feature
        tracker.predict();
        tracker.update(&[det2], Some(&feat2));
        assert_eq!(tracker.tracks()[0].id, id);
    }

    #[test]
    fn test_deep_sort_confirmed_tracks_filter() {
        let config = DeepSortConfig {
            n_init: 3,
            max_age: 10,
            ..DeepSortConfig::default()
        };
        let mut tracker = DeepSortTracker::new(config);
        let det = make_detection(10.0, 10.0, 50.0, 50.0);

        // After 1 hit: tentative.
        tracker.predict();
        tracker.update(&[det], None);
        assert!(tracker.confirmed_tracks().is_empty());

        // After 3 hits: confirmed.
        tracker.predict();
        tracker.update(&[det], None);
        tracker.predict();
        tracker.update(&[det], None);
        assert_eq!(tracker.confirmed_tracks().len(), 1);
    }

    #[test]
    fn test_min_cosine_distance_gallery() {
        let gallery = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let query = vec![0.9, 0.1, 0.0]; // closer to gallery[0]
        let dist = min_cosine_distance(&query, &gallery);
        // Should be close to 0 (matching gallery[0]).
        assert!(dist < 0.2);
    }
}
