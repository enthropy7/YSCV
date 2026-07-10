use rustc_hash::FxHashMap;
use std::path::Path;

use yscv_detect::{BoundingBox, Detection};

use crate::{DetectionDatasetFrame, EvalError, LabeledBox};

use super::helpers::{parse_image_id_manifest, read_optional_text_file};

pub(crate) fn load_kitti_label_dirs(
    manifest_text: &str,
    ground_truth_labels_dir: &Path,
    prediction_labels_dir: &Path,
) -> Result<Vec<DetectionDatasetFrame>, EvalError> {
    let image_ids = parse_image_id_manifest(manifest_text, "kitti")?;
    let mut frames = Vec::with_capacity(image_ids.len());
    let mut class_map = FxHashMap::default();

    for image_id in image_ids {
        let gt_path = kitti_label_path(ground_truth_labels_dir, &image_id);
        let pred_path = kitti_label_path(prediction_labels_dir, &image_id);

        let ground_truth = match read_optional_text_file(&gt_path)? {
            Some(text) => parse_kitti_ground_truth_labels(
                &text,
                &gt_path.display().to_string(),
                &mut class_map,
            )?,
            None => Vec::new(),
        };
        let predictions = match read_optional_text_file(&pred_path)? {
            Some(text) => {
                parse_kitti_predictions(&text, &pred_path.display().to_string(), &mut class_map)?
            }
            None => Vec::new(),
        };

        frames.push(DetectionDatasetFrame {
            ground_truth,
            predictions,
        });
    }

    Ok(frames)
}

fn kitti_label_path(labels_dir: &Path, image_id: &str) -> std::path::PathBuf {
    labels_dir.join(format!("{image_id}.txt"))
}

fn parse_kitti_ground_truth_labels(
    text: &str,
    source: &str,
    class_map: &mut FxHashMap<String, usize>,
) -> Result<Vec<LabeledBox>, EvalError> {
    let mut boxes = Vec::new();
    for (line_idx, raw_line) in text.lines().enumerate() {
        let line_no = line_idx + 1;
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.len() < 8 {
            return Err(EvalError::InvalidDatasetFormat {
                format: "kitti",
                message: format!(
                    "invalid ground-truth label `{source}` line {line_no}: expected at least 8 fields"
                ),
            });
        }
        if is_kitti_dont_care(tokens[0]) {
            continue;
        }

        let class_id = resolve_kitti_class_id(tokens[0], class_map);
        let bbox = parse_kitti_bbox_tokens(&tokens[4..8], source, line_no)?;
        boxes.push(LabeledBox { bbox, class_id });
    }
    Ok(boxes)
}

fn parse_kitti_predictions(
    text: &str,
    source: &str,
    class_map: &mut FxHashMap<String, usize>,
) -> Result<Vec<Detection>, EvalError> {
    let mut detections = Vec::new();
    for (line_idx, raw_line) in text.lines().enumerate() {
        let line_no = line_idx + 1;
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.len() < 8 {
            return Err(EvalError::InvalidDatasetFormat {
                format: "kitti",
                message: format!(
                    "invalid prediction label `{source}` line {line_no}: expected at least 8 fields"
                ),
            });
        }
        if is_kitti_dont_care(tokens[0]) {
            continue;
        }

        let class_id = resolve_kitti_class_id(tokens[0], class_map);
        let bbox = parse_kitti_bbox_tokens(&tokens[4..8], source, line_no)?;
        let score = match tokens.get(15).copied() {
            Some(raw) => parse_kitti_score(raw, source, line_no)?,
            None => 1.0,
        };
        detections.push(Detection {
            bbox,
            score,
            class_id,
        });
    }
    Ok(detections)
}

fn parse_kitti_bbox_tokens(
    tokens: &[&str],
    source: &str,
    line_no: usize,
) -> Result<BoundingBox, EvalError> {
    let x1 = parse_kitti_f32(tokens[0], "bbox_left", source, line_no)?;
    let y1 = parse_kitti_f32(tokens[1], "bbox_top", source, line_no)?;
    let x2 = parse_kitti_f32(tokens[2], "bbox_right", source, line_no)?;
    let y2 = parse_kitti_f32(tokens[3], "bbox_bottom", source, line_no)?;
    if x2 <= x1 || y2 <= y1 {
        return Err(EvalError::InvalidDatasetFormat {
            format: "kitti",
            message: format!(
                "invalid label `{source}` line {line_no}: expected bbox_right>bbox_left and bbox_bottom>bbox_top"
            ),
        });
    }
    Ok(BoundingBox { x1, y1, x2, y2 })
}

fn parse_kitti_f32(
    token: &str,
    field: &'static str,
    source: &str,
    line_no: usize,
) -> Result<f32, EvalError> {
    let value = token
        .parse::<f32>()
        .map_err(|err| EvalError::InvalidDatasetFormat {
            format: "kitti",
            message: format!(
                "invalid label `{source}` line {line_no} `{field}` value `{token}`: {err}"
            ),
        })?;
    if !value.is_finite() {
        return Err(EvalError::InvalidDatasetFormat {
            format: "kitti",
            message: format!("invalid label `{source}` line {line_no}: `{field}` must be finite"),
        });
    }
    Ok(value)
}

fn parse_kitti_score(token: &str, source: &str, line_no: usize) -> Result<f32, EvalError> {
    let score = parse_kitti_f32(token, "score", source, line_no)?;
    if score < 0.0 {
        return Err(EvalError::InvalidDatasetFormat {
            format: "kitti",
            message: format!("invalid label `{source}` line {line_no}: `score` must be >= 0"),
        });
    }
    Ok(score)
}

fn resolve_kitti_class_id(class_name: &str, class_map: &mut FxHashMap<String, usize>) -> usize {
    if let Some(&class_id) = class_map.get(class_name) {
        return class_id;
    }
    let class_id = class_map.len();
    class_map.insert(class_name.to_string(), class_id);
    class_id
}

fn is_kitti_dont_care(class_name: &str) -> bool {
    class_name.eq_ignore_ascii_case("DontCare")
}
