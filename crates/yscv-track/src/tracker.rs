use yscv_detect::{BoundingBox, CLASS_ID_PERSON, Detection, iou};

use crate::motion::{
    MotionState, apply_motion, bbox_size_similarity, normalized_center_distance,
    update_motion_state,
};
use crate::{Track, TrackError, TrackedDetection, TrackerConfig};

#[derive(Debug, Clone)]
pub struct Tracker {
    config: TrackerConfig,
    next_track_id: u64,
    tracks: Vec<Track>,
    motion: Vec<MotionState>,
    pair_candidates: Vec<(f32, usize, usize)>,
    track_taken: Vec<bool>,
    det_taken: Vec<bool>,
    det_to_track: Vec<Option<u64>>,
}

impl Tracker {
    /// Creates a tracker with validated configuration.
    pub fn new(config: TrackerConfig) -> Result<Self, TrackError> {
        config.validate()?;
        Ok(Self {
            config,
            next_track_id: 1,
            tracks: Vec::new(),
            motion: Vec::new(),
            pair_candidates: Vec::new(),
            track_taken: Vec::new(),
            det_taken: Vec::new(),
            det_to_track: Vec::new(),
        })
    }

    /// Updates tracker state for one frame and returns tracked detections.
    ///
    /// For allocation-sensitive runtime loops, prefer [`Tracker::update_into`]
    /// with a caller-owned output buffer that can be reused across frames.
    pub fn update(&mut self, detections: &[Detection]) -> Vec<TrackedDetection> {
        let mut out = Vec::with_capacity(detections.len());
        self.update_into(detections, &mut out);
        out
    }

    /// Updates tracker state for one frame and writes tracked detections into `out`.
    ///
    /// This API allows callers to reuse `out` across frames and avoid
    /// allocating a fresh output vector for each `update` call.
    pub fn update_into(&mut self, detections: &[Detection], out: &mut Vec<TrackedDetection>) {
        debug_assert_eq!(self.tracks.len(), self.motion.len());

        self.pair_candidates.clear();
        for (track_idx, track) in self.tracks.iter().enumerate() {
            let predicted = self.predict_bbox(track_idx, track);
            for (det_idx, det) in detections.iter().enumerate() {
                if track.class_id != det.class_id {
                    continue;
                }
                if let Some(match_score) = self.match_score(
                    track.missed_frames,
                    predicted,
                    det.bbox,
                    self.config.match_iou_threshold,
                ) {
                    self.pair_candidates.push((match_score, track_idx, det_idx));
                }
            }
        }
        self.pair_candidates.sort_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| left.1.cmp(&right.1))
                .then_with(|| left.2.cmp(&right.2))
        });

        self.track_taken.clear();
        self.track_taken.resize(self.tracks.len(), false);
        self.det_taken.clear();
        self.det_taken.resize(detections.len(), false);
        self.det_to_track.clear();
        self.det_to_track.resize(detections.len(), None);

        for (_match_score, track_idx, det_idx) in self.pair_candidates.iter().copied() {
            if self.track_taken[track_idx] || self.det_taken[det_idx] {
                continue;
            }
            self.track_taken[track_idx] = true;
            self.det_taken[det_idx] = true;
            let det = detections[det_idx];
            let track = &mut self.tracks[track_idx];
            let previous_bbox = track.bbox;
            track.bbox = det.bbox;
            track.score = det.score;
            track.class_id = det.class_id;
            track.age += 1;
            track.hits += 1;
            track.missed_frames = 0;
            update_motion_state(&mut self.motion[track_idx], previous_bbox, det.bbox);
            self.det_to_track[det_idx] = Some(track.id);
        }

        for (idx, track) in self.tracks.iter_mut().enumerate() {
            if !self.track_taken[idx] {
                track.bbox = apply_motion(track.bbox, &self.motion[idx], 1.0);
                track.age += 1;
                track.missed_frames += 1;
            }
        }

        let mut write = 0usize;
        for read in 0..self.tracks.len() {
            if self.tracks[read].missed_frames <= self.config.max_missed_frames {
                if write != read {
                    self.tracks[write] = self.tracks[read];
                    self.motion[write] = self.motion[read];
                }
                write += 1;
            }
        }
        self.tracks.truncate(write);
        self.motion.truncate(write);

        for (det_idx, det) in detections.iter().enumerate() {
            if self.det_taken[det_idx] {
                continue;
            }
            if self.tracks.len() >= self.config.max_tracks {
                break;
            }
            let track_id = self.alloc_track_id();
            self.tracks.push(Track {
                id: track_id,
                bbox: det.bbox,
                score: det.score,
                class_id: det.class_id,
                age: 1,
                hits: 1,
                missed_frames: 0,
            });
            self.motion.push(MotionState::default());
            self.det_to_track[det_idx] = Some(track_id);
        }

        out.clear();
        if out.capacity() < detections.len() {
            out.reserve(detections.len() - out.capacity());
        }
        for (det_idx, det) in detections.iter().enumerate() {
            if let Some(track_id) = self.det_to_track[det_idx] {
                out.push(TrackedDetection {
                    track_id,
                    detection: *det,
                });
            }
        }
    }

    /// Returns active tracks owned by this tracker.
    pub fn active_tracks(&self) -> &[Track] {
        &self.tracks
    }

    /// Counts active tracks for the provided class id.
    pub fn count_by_class(&self, class_id: usize) -> usize {
        self.tracks
            .iter()
            .filter(|track| track.class_id == class_id)
            .count()
    }

    /// Counts active person-class tracks.
    pub fn people_count(&self) -> usize {
        self.count_by_class(CLASS_ID_PERSON)
    }

    fn alloc_track_id(&mut self) -> u64 {
        let id = self.next_track_id;
        self.next_track_id += 1;
        id
    }

    fn predict_bbox(&self, track_idx: usize, track: &Track) -> BoundingBox {
        if track.missed_frames == 0 {
            return track.bbox;
        }
        apply_motion(track.bbox, &self.motion[track_idx], 1.0)
    }

    fn match_score(
        &self,
        missed_frames: u32,
        predicted: BoundingBox,
        detection: BoundingBox,
        iou_threshold: f32,
    ) -> Option<f32> {
        let overlap = iou(predicted, detection);
        if overlap >= iou_threshold {
            return Some(overlap);
        }

        let center_distance = normalized_center_distance(predicted, detection);
        let size_similarity = bbox_size_similarity(predicted, detection);
        let proximity_score = 1.0 / (1.0 + center_distance);
        let blended = 0.6 * overlap + 0.4 * proximity_score;

        if missed_frames > 0 && center_distance <= 2.0 && size_similarity >= 0.5 && blended >= 0.35
        {
            Some(blended)
        } else {
            None
        }
    }
}
