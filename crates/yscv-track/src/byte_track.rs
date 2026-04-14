//! ByteTrack multi-object tracker with two-stage association.
//!
//! ByteTrack splits detections into high- and low-confidence groups, matching
//! high-confidence detections to tracks first, then attempting to recover
//! unmatched tracks with low-confidence detections.

use yscv_detect::{BoundingBox, Detection, iou};

use crate::types::TrackedDetection;

/// Internal track representation for ByteTrack.
#[derive(Debug, Clone)]
struct ByteTrack {
    id: usize,
    bbox: BoundingBox,
    score: f32,
    class_id: usize,
    age: usize,
    hits: usize,
}

/// ByteTrack multi-object tracker.
///
/// Uses two-stage association: first matching high-confidence detections to
/// existing tracks, then trying to match low-confidence detections to
/// remaining unmatched tracks.
pub struct ByteTracker {
    next_id: usize,
    tracks: Vec<ByteTrack>,
    high_threshold: f32,
    low_threshold: f32,
    iou_threshold: f32,
    max_age: usize,
}

impl ByteTracker {
    /// Create a new `ByteTracker`.
    ///
    /// - `high_threshold`: minimum score to be considered high-confidence (e.g. 0.5).
    /// - `low_threshold`: minimum score to be considered at all (e.g. 0.1).
    /// - `iou_threshold`: minimum IoU for a detection-track match (e.g. 0.3).
    /// - `max_age`: frames a track survives without a match before deletion.
    pub fn new(
        high_threshold: f32,
        low_threshold: f32,
        iou_threshold: f32,
        max_age: usize,
    ) -> Self {
        Self {
            next_id: 1,
            tracks: Vec::new(),
            high_threshold,
            low_threshold,
            iou_threshold,
            max_age,
        }
    }

    /// Update the tracker with a new frame of detections.
    ///
    /// Returns the list of currently tracked detections with their track IDs.
    pub fn update(&mut self, detections: &[Detection]) -> Vec<TrackedDetection> {
        // 1. Split detections into high and low confidence.
        let mut high: Vec<usize> = Vec::new();
        let mut low: Vec<usize> = Vec::new();
        for (i, det) in detections.iter().enumerate() {
            if det.score >= self.high_threshold {
                high.push(i);
            } else if det.score >= self.low_threshold {
                low.push(i);
            }
        }

        let all_track_indices: Vec<usize> = (0..self.tracks.len()).collect();
        let mut matched_tracks: Vec<bool> = vec![false; self.tracks.len()];
        let mut matched_dets: Vec<bool> = vec![false; detections.len()];

        // 2. Match high-confidence detections to existing tracks (greedy by best IoU).
        let assignments1 = greedy_match(
            &self.tracks,
            detections,
            &all_track_indices,
            &high,
            self.iou_threshold,
        );

        for &(ti, di) in &assignments1 {
            matched_tracks[ti] = true;
            matched_dets[di] = true;
            self.tracks[ti].bbox = detections[di].bbox;
            self.tracks[ti].score = detections[di].score;
            self.tracks[ti].class_id = detections[di].class_id;
            self.tracks[ti].age = 0;
            self.tracks[ti].hits += 1;
        }

        // 3. Match low-confidence detections to remaining unmatched tracks.
        let unmatched_track_indices: Vec<usize> = (0..self.tracks.len())
            .filter(|&i| !matched_tracks[i])
            .collect();

        let assignments2 = greedy_match(
            &self.tracks,
            detections,
            &unmatched_track_indices,
            &low,
            self.iou_threshold,
        );

        for &(ti, di) in &assignments2 {
            matched_tracks[ti] = true;
            matched_dets[di] = true;
            self.tracks[ti].bbox = detections[di].bbox;
            self.tracks[ti].score = detections[di].score;
            self.tracks[ti].class_id = detections[di].class_id;
            self.tracks[ti].age = 0;
            self.tracks[ti].hits += 1;
        }

        // 4. Create new tracks for unmatched high-confidence detections.
        for &di in &high {
            if !matched_dets[di] {
                let id = self.next_id;
                self.next_id += 1;
                self.tracks.push(ByteTrack {
                    id,
                    bbox: detections[di].bbox,
                    score: detections[di].score,
                    class_id: detections[di].class_id,
                    age: 0,
                    hits: 1,
                });
            }
        }

        // 5. Age unmatched tracks, remove those exceeding max_age.
        for (i, track) in self.tracks.iter_mut().enumerate() {
            if i < matched_tracks.len() && !matched_tracks[i] {
                track.age += 1;
            }
        }
        self.tracks.retain(|t| t.age <= self.max_age);

        // 6. Return all active tracks as TrackedDetection.
        self.tracks
            .iter()
            .map(|t| TrackedDetection {
                track_id: t.id as u64,
                detection: Detection {
                    bbox: t.bbox,
                    score: t.score,
                    class_id: t.class_id,
                },
            })
            .collect()
    }

    /// Return the number of currently active tracks.
    pub fn active_track_count(&self) -> usize {
        self.tracks.len()
    }
}

/// Greedy IoU matching: for each track, find the detection with the highest IoU
/// at or above `iou_threshold`. Returns a list of (track_index, detection_index) pairs.
fn greedy_match(
    tracks: &[ByteTrack],
    detections: &[Detection],
    track_indices: &[usize],
    det_indices: &[usize],
    iou_threshold: f32,
) -> Vec<(usize, usize)> {
    let mut used_dets = vec![false; detections.len()];
    let mut assignments = Vec::new();

    for &ti in track_indices {
        let mut best_iou = iou_threshold;
        let mut best_di: Option<usize> = None;
        for &di in det_indices {
            if used_dets[di] {
                continue;
            }
            let iou_val = iou(tracks[ti].bbox, detections[di].bbox);
            if iou_val >= best_iou {
                best_iou = iou_val;
                best_di = Some(di);
            }
        }
        if let Some(di) = best_di {
            used_dets[di] = true;
            assignments.push((ti, di));
        }
    }

    assignments
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x1: f32, y1: f32, x2: f32, y2: f32, score: f32) -> Detection {
        Detection {
            bbox: BoundingBox { x1, y1, x2, y2 },
            score,
            class_id: 0,
        }
    }

    #[test]
    fn byte_track_creates_tracks() {
        let mut tracker = ByteTracker::new(0.5, 0.1, 0.3, 3);
        let dets = [
            det(10.0, 10.0, 50.0, 50.0, 0.9),
            det(100.0, 100.0, 150.0, 150.0, 0.8),
        ];
        let tracked = tracker.update(&dets);
        assert_eq!(tracked.len(), 2);
        assert_eq!(tracker.active_track_count(), 2);
    }

    #[test]
    fn byte_track_maintains_ids() {
        let mut tracker = ByteTracker::new(0.5, 0.1, 0.3, 3);
        let dets1 = [
            det(10.0, 10.0, 50.0, 50.0, 0.9),
            det(100.0, 100.0, 150.0, 150.0, 0.8),
        ];
        let tracked1 = tracker.update(&dets1);
        let id0 = tracked1[0].track_id;
        let id1 = tracked1[1].track_id;

        // Same objects, slightly moved.
        let dets2 = [
            det(12.0, 12.0, 52.0, 52.0, 0.9),
            det(102.0, 102.0, 152.0, 152.0, 0.85),
        ];
        let tracked2 = tracker.update(&dets2);
        assert_eq!(tracked2.len(), 2);

        let ids: Vec<u64> = tracked2.iter().map(|t| t.track_id).collect();
        assert!(ids.contains(&id0));
        assert!(ids.contains(&id1));
    }

    #[test]
    fn byte_track_removes_old_tracks() {
        let max_age = 2;
        let mut tracker = ByteTracker::new(0.5, 0.1, 0.3, max_age);

        // Create a track.
        let dets = [det(10.0, 10.0, 50.0, 50.0, 0.9)];
        tracker.update(&dets);
        assert_eq!(tracker.active_track_count(), 1);

        // No detections for max_age + 1 frames -> track should be removed.
        for _ in 0..=max_age {
            tracker.update(&[]);
        }
        assert_eq!(tracker.active_track_count(), 0);
    }

    #[test]
    fn byte_track_low_confidence_association() {
        let mut tracker = ByteTracker::new(0.5, 0.1, 0.3, 3);

        // Frame 1: high-confidence detection creates a track.
        let dets1 = [det(10.0, 10.0, 50.0, 50.0, 0.9)];
        let tracked1 = tracker.update(&dets1);
        let id = tracked1[0].track_id;

        // Frame 2: same object appears with low confidence (below high, above low).
        let dets2 = [det(12.0, 12.0, 52.0, 52.0, 0.3)];
        let tracked2 = tracker.update(&dets2);

        // Should match the existing track, not create a new one.
        assert_eq!(tracked2.len(), 1);
        assert_eq!(tracked2[0].track_id, id);
    }

    #[test]
    fn byte_track_new_track_for_new_object() {
        let mut tracker = ByteTracker::new(0.5, 0.1, 0.3, 3);

        // Frame 1: one object.
        let dets1 = [det(10.0, 10.0, 50.0, 50.0, 0.9)];
        let tracked1 = tracker.update(&dets1);
        assert_eq!(tracked1.len(), 1);
        let id1 = tracked1[0].track_id;

        // Frame 2: original object + a new far-away object.
        let dets2 = [
            det(12.0, 12.0, 52.0, 52.0, 0.9),
            det(200.0, 200.0, 250.0, 250.0, 0.8),
        ];
        let tracked2 = tracker.update(&dets2);
        assert_eq!(tracked2.len(), 2);

        let ids: Vec<u64> = tracked2.iter().map(|t| t.track_id).collect();
        assert!(ids.contains(&id1));
        // The new object should have a different track_id.
        let new_id = ids
            .iter()
            .find(|&&id| id != id1)
            .expect("second track should exist");
        assert_ne!(*new_id, id1);
    }

    #[test]
    fn byte_track_three_objects_simultaneously() {
        let mut tracker = ByteTracker::new(0.5, 0.1, 0.3, 5);
        let dets = [
            det(10.0, 10.0, 50.0, 50.0, 0.9),
            det(100.0, 100.0, 140.0, 140.0, 0.8),
            det(200.0, 200.0, 240.0, 240.0, 0.7),
        ];
        let tracked = tracker.update(&dets);
        assert_eq!(tracked.len(), 3);
        assert_eq!(tracker.active_track_count(), 3);

        // All IDs unique.
        let ids: Vec<u64> = tracked.iter().map(|t| t.track_id).collect();
        let mut unique = ids.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), 3);
    }

    #[test]
    fn byte_track_empty_detections_ages_tracks() {
        let mut tracker = ByteTracker::new(0.5, 0.1, 0.3, 5);
        tracker.update(&[det(10.0, 10.0, 50.0, 50.0, 0.9)]);
        assert_eq!(tracker.active_track_count(), 1);

        // Empty frame: track should still exist but age.
        let result = tracker.update(&[]);
        assert_eq!(tracker.active_track_count(), 1);
        assert_eq!(result.len(), 1); // track still reported
    }

    #[test]
    fn byte_track_single_detection_stable_id() {
        let mut tracker = ByteTracker::new(0.5, 0.1, 0.3, 5);
        let d = det(20.0, 20.0, 60.0, 60.0, 0.9);
        let first = tracker.update(&[d]);
        let id = first[0].track_id;

        for _ in 0..10 {
            let tracked = tracker.update(&[d]);
            assert_eq!(tracked.len(), 1);
            assert_eq!(tracked[0].track_id, id);
        }
    }

    #[test]
    fn byte_track_id_stability_smooth_motion() {
        let mut tracker = ByteTracker::new(0.5, 0.1, 0.3, 5);
        let first = tracker.update(&[det(10.0, 10.0, 50.0, 50.0, 0.9)]);
        let id = first[0].track_id;

        // Smooth small moves.
        let positions = [
            (12.0, 12.0, 52.0, 52.0),
            (14.0, 14.0, 54.0, 54.0),
            (16.0, 16.0, 56.0, 56.0),
            (18.0, 18.0, 58.0, 58.0),
        ];
        for (x1, y1, x2, y2) in positions {
            let tracked = tracker.update(&[det(x1, y1, x2, y2, 0.9)]);
            assert_eq!(tracked.len(), 1);
            assert_eq!(tracked[0].track_id, id);
        }
    }

    #[test]
    fn byte_track_iou_matching_overlapping_bboxes() {
        let mut tracker = ByteTracker::new(0.5, 0.1, 0.3, 5);

        // Two overlapping detections.
        let dets1 = [
            det(10.0, 10.0, 50.0, 50.0, 0.9),
            det(30.0, 30.0, 70.0, 70.0, 0.8),
        ];
        let tracked1 = tracker.update(&dets1);
        assert_eq!(tracked1.len(), 2);
        let id_a = tracked1[0].track_id;
        let id_b = tracked1[1].track_id;
        assert_ne!(id_a, id_b);

        // Slightly move each, should still match correctly.
        let dets2 = [
            det(11.0, 11.0, 51.0, 51.0, 0.9),
            det(31.0, 31.0, 71.0, 71.0, 0.8),
        ];
        let tracked2 = tracker.update(&dets2);
        let ids2: Vec<u64> = tracked2.iter().map(|t| t.track_id).collect();
        assert!(ids2.contains(&id_a));
        assert!(ids2.contains(&id_b));
    }

    #[test]
    fn byte_track_low_vs_high_confidence_precedence() {
        // High-confidence detection should match first, low-confidence second.
        let mut tracker = ByteTracker::new(0.5, 0.1, 0.3, 5);

        // Create two tracks with high-confidence detections.
        let dets1 = [
            det(10.0, 10.0, 50.0, 50.0, 0.9),
            det(100.0, 100.0, 140.0, 140.0, 0.8),
        ];
        let tracked1 = tracker.update(&dets1);
        let id_a = tracked1[0].track_id;
        let id_b = tracked1[1].track_id;

        // Frame 2: track A has high-confidence, track B only has low-confidence.
        let dets2 = [
            det(12.0, 12.0, 52.0, 52.0, 0.9),     // high — should match track A
            det(102.0, 102.0, 142.0, 142.0, 0.2), // low — should match track B
        ];
        let tracked2 = tracker.update(&dets2);
        assert_eq!(tracked2.len(), 2);
        let ids2: Vec<u64> = tracked2.iter().map(|t| t.track_id).collect();
        assert!(ids2.contains(&id_a));
        assert!(ids2.contains(&id_b));
    }

    #[test]
    fn byte_track_low_confidence_no_new_track() {
        // Low-confidence detections should NOT create new tracks.
        let mut tracker = ByteTracker::new(0.5, 0.1, 0.3, 5);

        // Only low-confidence detection.
        let dets = [det(10.0, 10.0, 50.0, 50.0, 0.2)];
        let tracked = tracker.update(&dets);
        assert_eq!(tracked.len(), 0);
        assert_eq!(tracker.active_track_count(), 0);
    }

    #[test]
    fn byte_track_below_low_threshold_ignored() {
        // Detection below low_threshold should be completely ignored.
        let mut tracker = ByteTracker::new(0.5, 0.1, 0.3, 5);

        let dets = [det(10.0, 10.0, 50.0, 50.0, 0.05)]; // below low_threshold=0.1
        let tracked = tracker.update(&dets);
        assert_eq!(tracked.len(), 0);
        assert_eq!(tracker.active_track_count(), 0);
    }

    #[test]
    fn byte_track_config_different_max_age() {
        // max_age=0 means tracks die immediately when unmatched.
        let mut tracker = ByteTracker::new(0.5, 0.1, 0.3, 0);
        tracker.update(&[det(10.0, 10.0, 50.0, 50.0, 0.9)]);
        assert_eq!(tracker.active_track_count(), 1);

        // One empty frame → gone immediately.
        tracker.update(&[]);
        assert_eq!(tracker.active_track_count(), 0);
    }

    #[test]
    fn byte_track_config_different_iou_threshold() {
        // Very high IoU threshold: only exact overlaps match.
        let mut tracker = ByteTracker::new(0.5, 0.1, 0.95, 5);
        tracker.update(&[det(10.0, 10.0, 50.0, 50.0, 0.9)]);

        // Slightly moved → IoU below 0.95 → new track created, old one unmatched.
        tracker.update(&[det(15.0, 15.0, 55.0, 55.0, 0.9)]);
        // Should have the old track (unmatched) plus a new one.
        assert_eq!(tracker.active_track_count(), 2);
    }

    #[test]
    fn byte_track_more_tracks_than_detections() {
        let mut tracker = ByteTracker::new(0.5, 0.1, 0.3, 5);

        // Create 3 tracks.
        let dets = [
            det(10.0, 10.0, 50.0, 50.0, 0.9),
            det(100.0, 100.0, 140.0, 140.0, 0.8),
            det(200.0, 200.0, 240.0, 240.0, 0.7),
        ];
        tracker.update(&dets);
        assert_eq!(tracker.active_track_count(), 3);

        // Only 1 detection: 2 tracks unmatched, should age but still alive.
        tracker.update(&[det(12.0, 12.0, 52.0, 52.0, 0.9)]);
        // All 3 tracks still exist since max_age=5 and only 1 frame missed.
        assert_eq!(tracker.active_track_count(), 3);
    }

    #[test]
    fn byte_track_more_detections_than_tracks() {
        let mut tracker = ByteTracker::new(0.5, 0.1, 0.3, 5);

        // Create 1 track.
        tracker.update(&[det(10.0, 10.0, 50.0, 50.0, 0.9)]);
        assert_eq!(tracker.active_track_count(), 1);

        // 3 detections: 1 match + 2 new tracks.
        let dets = [
            det(12.0, 12.0, 52.0, 52.0, 0.9),
            det(100.0, 100.0, 140.0, 140.0, 0.8),
            det(200.0, 200.0, 240.0, 240.0, 0.7),
        ];
        tracker.update(&dets);
        assert_eq!(tracker.active_track_count(), 3);
    }
}
