use crate::{BoundingBox, DetectError, Detection};

/// Computes IoU for two axis-aligned boxes.
pub fn iou(a: BoundingBox, b: BoundingBox) -> f32 {
    let inter_x1 = a.x1.max(b.x1);
    let inter_y1 = a.y1.max(b.y1);
    let inter_x2 = a.x2.min(b.x2);
    let inter_y2 = a.y2.min(b.y2);

    let inter_w = (inter_x2 - inter_x1).max(0.0);
    let inter_h = (inter_y2 - inter_y1).max(0.0);
    let inter = inter_w * inter_h;
    let union = a.area() + b.area() - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

/// Standard score-sorted NMS.
pub fn non_max_suppression(
    detections: &[Detection],
    iou_threshold: f32,
    max_detections: usize,
) -> Vec<Detection> {
    let mut sorted = detections
        .iter()
        .copied()
        .filter(is_finite_detection)
        .collect::<Vec<_>>();
    sorted.sort_by(|a, b| b.score.total_cmp(&a.score));

    let mut selected: Vec<Detection> = Vec::new();
    for candidate in sorted {
        if selected.len() >= max_detections {
            break;
        }
        if candidate.bbox.area() <= 0.0 {
            continue;
        }
        let mut suppressed = false;
        for chosen in &selected {
            if chosen.class_id == candidate.class_id
                && iou(chosen.bbox, candidate.bbox) > iou_threshold
            {
                suppressed = true;
                break;
            }
        }
        if !suppressed {
            selected.push(candidate);
        }
    }
    selected
}

pub(crate) fn validate_nms_args(
    iou_threshold: f32,
    max_detections: usize,
) -> Result<(), DetectError> {
    if !iou_threshold.is_finite() || !(0.0..=1.0).contains(&iou_threshold) {
        return Err(DetectError::InvalidIouThreshold { iou_threshold });
    }
    if max_detections == 0 {
        return Err(DetectError::InvalidMaxDetections { max_detections });
    }
    Ok(())
}

fn is_finite_detection(detection: &Detection) -> bool {
    detection.score.is_finite()
        && detection.bbox.x1.is_finite()
        && detection.bbox.y1.is_finite()
        && detection.bbox.x2.is_finite()
        && detection.bbox.y2.is_finite()
}

/// Soft-NMS with Gaussian decay.
///
/// Instead of hard suppression, overlapping detections have their scores
/// decayed by `score *= exp(-(iou² / sigma))`. Detections whose score
/// falls below `score_threshold` are removed. The vector is modified in
/// place.
pub fn soft_nms(detections: &mut Vec<Detection>, sigma: f32, score_threshold: f32) {
    // Filter out non-finite detections first.
    detections.retain(is_finite_detection);

    // Process each position: pick the highest-scoring remaining detection,
    // swap it to the current position, then decay all subsequent detections.
    let mut i = 0;
    while i < detections.len() {
        // Find the index of the max-score detection in [i..].
        let mut max_idx = i;
        for j in (i + 1)..detections.len() {
            if detections[j].score > detections[max_idx].score {
                max_idx = j;
            }
        }
        detections.swap(i, max_idx);

        // Decay scores of all subsequent detections based on IoU with detections[i].
        let current = detections[i];
        let mut j = i + 1;
        while j < detections.len() {
            let overlap = iou(current.bbox, detections[j].bbox);
            detections[j].score *= (-overlap * overlap / sigma).exp();
            if detections[j].score < score_threshold {
                detections.swap_remove(j);
                // Don't increment j; the swapped element needs checking too.
            } else {
                j += 1;
            }
        }
        i += 1;
    }
}

/// Per-class (batched) NMS.
///
/// Groups detections by `class_id`, runs standard `non_max_suppression` on
/// each group independently, then merges and returns results sorted by score
/// descending.
pub fn batched_nms(detections: &[Detection], iou_threshold: f32) -> Vec<Detection> {
    use std::collections::HashMap;

    let mut by_class: HashMap<usize, Vec<Detection>> = HashMap::new();
    for det in detections {
        by_class.entry(det.class_id).or_default().push(*det);
    }

    let mut results: Vec<Detection> = Vec::new();
    for class_dets in by_class.values() {
        let kept = non_max_suppression(class_dets, iou_threshold, class_dets.len());
        results.extend(kept);
    }

    results.sort_by(|a, b| b.score.total_cmp(&a.score));
    results
}
