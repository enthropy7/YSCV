use rustc_hash::FxHashMap;

use yscv_detect::{BoundingBox, iou};
use yscv_track::TrackedDetection;

use crate::EvalError;
use crate::util::{harmonic_mean, safe_ratio, validate_iou_threshold};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroundTruthTrack {
    pub object_id: u64,
    pub bbox: BoundingBox,
    pub class_id: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackingFrame<'a> {
    pub ground_truth: &'a [GroundTruthTrack],
    pub predictions: &'a [TrackedDetection],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackingEvalConfig {
    pub iou_threshold: f32,
}

impl Default for TrackingEvalConfig {
    fn default() -> Self {
        Self { iou_threshold: 0.5 }
    }
}

impl TrackingEvalConfig {
    pub fn validate(&self) -> Result<(), EvalError> {
        validate_iou_threshold(self.iou_threshold)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackingMetrics {
    pub total_ground_truth: u64,
    pub matches: u64,
    pub false_positives: u64,
    pub false_negatives: u64,
    pub id_switches: u64,
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
    pub mota: f32,
    pub motp: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrackingDatasetFrame {
    pub ground_truth: Vec<GroundTruthTrack>,
    pub predictions: Vec<TrackedDetection>,
}

impl TrackingDatasetFrame {
    pub fn as_view(&self) -> TrackingFrame<'_> {
        TrackingFrame {
            ground_truth: &self.ground_truth,
            predictions: &self.predictions,
        }
    }
}

pub fn tracking_frames_as_view(frames: &[TrackingDatasetFrame]) -> Vec<TrackingFrame<'_>> {
    frames.iter().map(TrackingDatasetFrame::as_view).collect()
}

pub fn evaluate_tracking_from_dataset(
    frames: &[TrackingDatasetFrame],
    config: TrackingEvalConfig,
) -> Result<TrackingMetrics, EvalError> {
    let borrowed = tracking_frames_as_view(frames);
    evaluate_tracking(&borrowed, config)
}

/// Greedy IoU matching for a single frame. Returns vec of (gt_index, pred_index, iou).
fn greedy_iou_match(frame: &TrackingFrame<'_>, iou_threshold: f32) -> Vec<(usize, usize, f32)> {
    let mut candidates = Vec::new();
    for (gt_idx, gt) in frame.ground_truth.iter().enumerate() {
        for (pred_idx, prediction) in frame.predictions.iter().enumerate() {
            if gt.class_id != prediction.detection.class_id {
                continue;
            }
            let overlap = iou(gt.bbox, prediction.detection.bbox);
            if overlap >= iou_threshold {
                candidates.push((overlap, gt_idx, pred_idx));
            }
        }
    }
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0));

    let mut gt_taken = vec![false; frame.ground_truth.len()];
    let mut pred_taken = vec![false; frame.predictions.len()];
    let mut matches = Vec::new();

    for (overlap, gt_idx, pred_idx) in candidates {
        if gt_taken[gt_idx] || pred_taken[pred_idx] {
            continue;
        }
        gt_taken[gt_idx] = true;
        pred_taken[pred_idx] = true;
        matches.push((gt_idx, pred_idx, overlap));
    }

    matches
}

/// Identity F1 score: measures how well predicted IDs match GT IDs across frames.
pub fn idf1(frames: &[TrackingFrame<'_>], config: TrackingEvalConfig) -> Result<f32, EvalError> {
    config.validate()?;

    // Count co-occurrences of (gt_object_id, pred_track_id) pairs
    let mut cooccurrence: FxHashMap<(u64, u64), u64> = FxHashMap::default();
    let mut gt_appearances: FxHashMap<u64, u64> = FxHashMap::default();
    let mut pred_appearances: FxHashMap<u64, u64> = FxHashMap::default();

    for frame in frames {
        for gt in frame.ground_truth {
            *gt_appearances.entry(gt.object_id).or_insert(0) += 1;
        }
        for pred in frame.predictions {
            *pred_appearances.entry(pred.track_id).or_insert(0) += 1;
        }

        let matches = greedy_iou_match(frame, config.iou_threshold);
        for (gt_idx, pred_idx, _) in matches {
            let gt_id = frame.ground_truth[gt_idx].object_id;
            let pred_id = frame.predictions[pred_idx].track_id;
            *cooccurrence.entry((gt_id, pred_id)).or_insert(0) += 1;
        }
    }

    let total_gt: u64 = gt_appearances.values().sum();
    let total_pred: u64 = pred_appearances.values().sum();

    if total_gt == 0 && total_pred == 0 {
        return Ok(0.0);
    }

    // For each GT object, find the best-matching predicted track (most co-occurrences)
    let mut best_for_gt: FxHashMap<u64, (u64, u64)> = FxHashMap::default(); // gt_id -> (pred_id, count)
    for (&(gt_id, pred_id), &count) in &cooccurrence {
        let entry = best_for_gt.entry(gt_id).or_insert((pred_id, 0));
        if count > entry.1 {
            *entry = (pred_id, count);
        }
    }

    let idtp: u64 = best_for_gt.values().map(|(_, count)| count).sum();
    let idfn = total_gt - idtp;
    let idfp = total_pred - idtp;

    let denom = 2 * idtp + idfp + idfn;
    if denom == 0 {
        return Ok(0.0);
    }

    Ok((2 * idtp) as f32 / denom as f32)
}

/// Higher Order Tracking Accuracy.
pub fn hota(frames: &[TrackingFrame<'_>], config: TrackingEvalConfig) -> Result<f32, EvalError> {
    config.validate()?;

    // Collect all per-frame TP matches and detection counts
    let mut all_matches: Vec<(u64, u64)> = Vec::new(); // (gt_object_id, pred_track_id) per TP
    let mut total_tp = 0u64;
    let mut total_fp = 0u64;
    let mut total_fn = 0u64;

    // Track which pred_id maps to which gt_id per frame (for association computation)
    let mut pred_to_gt_per_frame: Vec<FxHashMap<u64, u64>> = Vec::new();
    let mut gt_to_pred_per_frame: Vec<FxHashMap<u64, u64>> = Vec::new();

    for frame in frames {
        let matches = greedy_iou_match(frame, config.iou_threshold);
        let tp = matches.len() as u64;
        let fn_count = frame.ground_truth.len() as u64 - tp;
        let fp_count = frame.predictions.len() as u64 - tp;

        total_tp += tp;
        total_fp += fp_count;
        total_fn += fn_count;

        let mut pred_to_gt = FxHashMap::default();
        let mut gt_to_pred = FxHashMap::default();
        for &(gt_idx, pred_idx, _) in &matches {
            let gt_id = frame.ground_truth[gt_idx].object_id;
            let pred_id = frame.predictions[pred_idx].track_id;
            all_matches.push((gt_id, pred_id));
            pred_to_gt.insert(pred_id, gt_id);
            gt_to_pred.insert(gt_id, pred_id);
        }
        pred_to_gt_per_frame.push(pred_to_gt);
        gt_to_pred_per_frame.push(gt_to_pred);
    }

    if total_tp == 0 {
        return Ok(0.0);
    }

    let det_a = total_tp as f32 / (total_tp + total_fp + total_fn) as f32;

    // Compute association accuracy for each TP match
    let mut ass_a_sum = 0.0f32;
    let num_frames = frames.len();

    for &(gt_id, pred_id) in &all_matches {
        let mut tpa = 0u64;
        let mut fpa = 0u64;
        let mut fna = 0u64;

        for f in 0..num_frames {
            let pred_matched_gt = pred_to_gt_per_frame[f].get(&pred_id);
            let gt_matched_pred = gt_to_pred_per_frame[f].get(&gt_id);

            match (pred_matched_gt, gt_matched_pred) {
                (Some(&matched_gt), Some(&matched_pred))
                    if matched_gt == gt_id && matched_pred == pred_id =>
                {
                    tpa += 1;
                }
                _ => {
                    // FPA: pred_id matched a different gt (or any gt)
                    if let Some(&matched_gt) = pred_matched_gt
                        && matched_gt != gt_id
                    {
                        fpa += 1;
                    }
                    // FNA: gt_id matched a different pred
                    if let Some(&matched_pred) = gt_matched_pred
                        && matched_pred != pred_id
                    {
                        fna += 1;
                    }
                }
            }
        }

        let denom = tpa + fpa + fna;
        if denom > 0 {
            ass_a_sum += tpa as f32 / denom as f32;
        }
    }

    let ass_a = ass_a_sum / all_matches.len() as f32;
    Ok((det_a * ass_a).sqrt())
}

pub fn evaluate_tracking(
    frames: &[TrackingFrame<'_>],
    config: TrackingEvalConfig,
) -> Result<TrackingMetrics, EvalError> {
    config.validate()?;

    let mut total_ground_truth = 0u64;
    let mut matches = 0u64;
    let mut false_positives = 0u64;
    let mut false_negatives = 0u64;
    let mut id_switches = 0u64;
    let mut iou_sum = 0.0f32;
    let mut last_assignment: FxHashMap<u64, u64> = FxHashMap::default();

    for frame in frames {
        total_ground_truth += frame.ground_truth.len() as u64;

        let mut candidates = Vec::new();
        for (gt_idx, gt) in frame.ground_truth.iter().enumerate() {
            for (pred_idx, prediction) in frame.predictions.iter().enumerate() {
                if gt.class_id != prediction.detection.class_id {
                    continue;
                }
                let overlap = iou(gt.bbox, prediction.detection.bbox);
                if overlap >= config.iou_threshold {
                    candidates.push((overlap, gt_idx, pred_idx));
                }
            }
        }
        candidates.sort_by(|a, b| b.0.total_cmp(&a.0));

        let mut gt_taken = vec![false; frame.ground_truth.len()];
        let mut pred_taken = vec![false; frame.predictions.len()];

        for (overlap, gt_idx, pred_idx) in candidates {
            if gt_taken[gt_idx] || pred_taken[pred_idx] {
                continue;
            }

            gt_taken[gt_idx] = true;
            pred_taken[pred_idx] = true;
            matches += 1;
            iou_sum += overlap;

            let gt_id = frame.ground_truth[gt_idx].object_id;
            let pred_id = frame.predictions[pred_idx].track_id;
            if let Some(previous_pred_id) = last_assignment.get(&gt_id)
                && *previous_pred_id != pred_id
            {
                id_switches += 1;
            }
            last_assignment.insert(gt_id, pred_id);
        }

        false_negatives += gt_taken.iter().filter(|matched| !**matched).count() as u64;
        false_positives += pred_taken.iter().filter(|matched| !**matched).count() as u64;
    }

    let precision = safe_ratio(matches, matches + false_positives);
    let recall = safe_ratio(matches, matches + false_negatives);
    let f1 = harmonic_mean(precision, recall);
    let motp = if matches == 0 {
        0.0
    } else {
        iou_sum / matches as f32
    };
    let mota = if total_ground_truth == 0 {
        0.0
    } else {
        1.0 - ((false_negatives + false_positives + id_switches) as f32 / total_ground_truth as f32)
    };

    Ok(TrackingMetrics {
        total_ground_truth,
        matches,
        false_positives,
        false_negatives,
        id_switches,
        precision,
        recall,
        f1,
        mota,
        motp,
    })
}
