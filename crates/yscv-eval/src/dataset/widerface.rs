use rustc_hash::FxHashSet;
use rustc_hash::{FxBuildHasher, FxHashMap};

use yscv_detect::{BoundingBox, CLASS_ID_FACE, Detection};

use crate::{DetectionDatasetFrame, EvalError, LabeledBox};

#[derive(Debug, Clone, PartialEq)]
struct WiderFaceGroundTruthFrame {
    image_id: String,
    ground_truth: Vec<LabeledBox>,
}

#[derive(Debug, Clone, PartialEq)]
struct WiderFacePredictionFrame {
    image_id: String,
    predictions: Vec<Detection>,
}

pub(crate) fn parse_and_build_widerface(
    ground_truth_text: &str,
    predictions_text: &str,
) -> Result<Vec<DetectionDatasetFrame>, EvalError> {
    let ground_truth_frames = parse_widerface_ground_truth(ground_truth_text)?;
    let prediction_frames = parse_widerface_predictions(predictions_text)?;

    let mut predictions_by_image =
        FxHashMap::with_capacity_and_hasher(prediction_frames.len(), FxBuildHasher);
    for frame in prediction_frames {
        if predictions_by_image
            .insert(frame.image_id.clone(), frame.predictions)
            .is_some()
        {
            return Err(EvalError::InvalidDatasetFormat {
                format: "widerface",
                message: format!("duplicate prediction image id `{}`", frame.image_id),
            });
        }
    }

    let mut frames = Vec::with_capacity(ground_truth_frames.len());
    for frame in ground_truth_frames {
        let predictions = predictions_by_image
            .remove(&frame.image_id)
            .unwrap_or_default();
        frames.push(DetectionDatasetFrame {
            ground_truth: frame.ground_truth,
            predictions,
        });
    }

    if let Some((image_id, _)) = predictions_by_image.into_iter().next() {
        return Err(EvalError::InvalidDatasetFormat {
            format: "widerface",
            message: format!("predictions reference unknown image id `{image_id}`"),
        });
    }

    Ok(frames)
}

fn parse_widerface_ground_truth(text: &str) -> Result<Vec<WiderFaceGroundTruthFrame>, EvalError> {
    let lines = widerface_significant_lines(text);
    let mut cursor = 0usize;
    let mut frames = Vec::new();
    let mut seen_ids = FxHashSet::default();

    while cursor < lines.len() {
        let (image_line_no, image_id_raw) = lines[cursor];
        cursor += 1;
        let image_id = image_id_raw.to_string();
        if !seen_ids.insert(image_id.clone()) {
            return Err(EvalError::InvalidDatasetFormat {
                format: "widerface",
                message: format!(
                    "duplicate ground-truth image id `{image_id}` at line {image_line_no}"
                ),
            });
        }

        let Some((count_line_no, count_raw)) = lines.get(cursor).copied() else {
            return Err(EvalError::InvalidDatasetFormat {
                format: "widerface",
                message: format!(
                    "missing ground-truth box count for image `{image_id}` after line {image_line_no}"
                ),
            });
        };
        cursor += 1;
        let box_count = parse_widerface_box_count(count_raw, count_line_no, "ground-truth")?;
        let mut ground_truth = Vec::with_capacity(box_count);

        for box_idx in 0..box_count {
            let Some((line_no, row)) = lines.get(cursor).copied() else {
                return Err(EvalError::InvalidDatasetFormat {
                    format: "widerface",
                    message: format!(
                        "missing ground-truth box row {}/{} for image `{image_id}`",
                        box_idx + 1,
                        box_count
                    ),
                });
            };
            cursor += 1;
            let tokens = row.split_whitespace().collect::<Vec<_>>();
            if tokens.len() < 4 {
                return Err(EvalError::InvalidDatasetFormat {
                    format: "widerface",
                    message: format!(
                        "invalid ground-truth row for image `{image_id}` at line {line_no}: expected at least 4 fields `<x> <y> <w> <h>`"
                    ),
                });
            }

            let x1 = parse_widerface_f32(tokens[0], "x", "ground-truth", line_no)?;
            let y1 = parse_widerface_f32(tokens[1], "y", "ground-truth", line_no)?;
            let width = parse_widerface_f32(tokens[2], "w", "ground-truth", line_no)?;
            let height = parse_widerface_f32(tokens[3], "h", "ground-truth", line_no)?;
            if width <= 0.0 || height <= 0.0 {
                return Err(EvalError::InvalidDatasetFormat {
                    format: "widerface",
                    message: format!(
                        "invalid ground-truth row for image `{image_id}` at line {line_no}: `w` and `h` must be > 0"
                    ),
                });
            }
            if let Some(raw_invalid) = tokens.get(7).copied() {
                let invalid = parse_widerface_i32(raw_invalid, "invalid", "ground-truth", line_no)?;
                if invalid != 0 {
                    continue;
                }
            }

            let x2 = x1 + width;
            let y2 = y1 + height;
            if !x2.is_finite() || !y2.is_finite() {
                return Err(EvalError::InvalidDatasetFormat {
                    format: "widerface",
                    message: format!(
                        "invalid ground-truth row for image `{image_id}` at line {line_no}: bbox corner overflow"
                    ),
                });
            }
            ground_truth.push(LabeledBox {
                bbox: BoundingBox { x1, y1, x2, y2 },
                class_id: CLASS_ID_FACE,
            });
        }

        frames.push(WiderFaceGroundTruthFrame {
            image_id,
            ground_truth,
        });
    }

    Ok(frames)
}

fn parse_widerface_predictions(text: &str) -> Result<Vec<WiderFacePredictionFrame>, EvalError> {
    let lines = widerface_significant_lines(text);
    let mut cursor = 0usize;
    let mut frames = Vec::new();
    let mut seen_ids = FxHashSet::default();

    while cursor < lines.len() {
        let (image_line_no, image_id_raw) = lines[cursor];
        cursor += 1;
        let image_id = image_id_raw.to_string();
        if !seen_ids.insert(image_id.clone()) {
            return Err(EvalError::InvalidDatasetFormat {
                format: "widerface",
                message: format!(
                    "duplicate prediction image id `{image_id}` at line {image_line_no}"
                ),
            });
        }

        let Some((count_line_no, count_raw)) = lines.get(cursor).copied() else {
            return Err(EvalError::InvalidDatasetFormat {
                format: "widerface",
                message: format!(
                    "missing prediction box count for image `{image_id}` after line {image_line_no}"
                ),
            });
        };
        cursor += 1;
        let box_count = parse_widerface_box_count(count_raw, count_line_no, "prediction")?;
        let mut predictions = Vec::with_capacity(box_count);

        for box_idx in 0..box_count {
            let Some((line_no, row)) = lines.get(cursor).copied() else {
                return Err(EvalError::InvalidDatasetFormat {
                    format: "widerface",
                    message: format!(
                        "missing prediction box row {}/{} for image `{image_id}`",
                        box_idx + 1,
                        box_count
                    ),
                });
            };
            cursor += 1;
            let tokens = row.split_whitespace().collect::<Vec<_>>();
            if tokens.len() < 5 {
                return Err(EvalError::InvalidDatasetFormat {
                    format: "widerface",
                    message: format!(
                        "invalid prediction row for image `{image_id}` at line {line_no}: expected at least 5 fields `<x> <y> <w> <h> <score>`"
                    ),
                });
            }

            let x1 = parse_widerface_f32(tokens[0], "x", "prediction", line_no)?;
            let y1 = parse_widerface_f32(tokens[1], "y", "prediction", line_no)?;
            let width = parse_widerface_f32(tokens[2], "w", "prediction", line_no)?;
            let height = parse_widerface_f32(tokens[3], "h", "prediction", line_no)?;
            if width <= 0.0 || height <= 0.0 {
                return Err(EvalError::InvalidDatasetFormat {
                    format: "widerface",
                    message: format!(
                        "invalid prediction row for image `{image_id}` at line {line_no}: `w` and `h` must be > 0"
                    ),
                });
            }
            let score =
                parse_widerface_non_negative_f32(tokens[4], "score", "prediction", line_no)?;

            let x2 = x1 + width;
            let y2 = y1 + height;
            if !x2.is_finite() || !y2.is_finite() {
                return Err(EvalError::InvalidDatasetFormat {
                    format: "widerface",
                    message: format!(
                        "invalid prediction row for image `{image_id}` at line {line_no}: bbox corner overflow"
                    ),
                });
            }
            predictions.push(Detection {
                bbox: BoundingBox { x1, y1, x2, y2 },
                score,
                class_id: CLASS_ID_FACE,
            });
        }

        frames.push(WiderFacePredictionFrame {
            image_id,
            predictions,
        });
    }

    Ok(frames)
}

fn widerface_significant_lines(text: &str) -> Vec<(usize, &str)> {
    text.lines()
        .enumerate()
        .filter_map(|(line_idx, raw_line)| {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                None
            } else {
                Some((line_idx + 1, line))
            }
        })
        .collect()
}

fn parse_widerface_box_count(
    token: &str,
    line_no: usize,
    source: &'static str,
) -> Result<usize, EvalError> {
    token
        .parse::<usize>()
        .map_err(|err| EvalError::InvalidDatasetFormat {
            format: "widerface",
            message: format!("invalid {source} header line {line_no} box count `{token}`: {err}"),
        })
}

fn parse_widerface_f32(
    token: &str,
    field: &'static str,
    source: &'static str,
    line_no: usize,
) -> Result<f32, EvalError> {
    let value = token
        .parse::<f32>()
        .map_err(|err| EvalError::InvalidDatasetFormat {
            format: "widerface",
            message: format!("invalid {source} line {line_no} `{field}` value `{token}`: {err}"),
        })?;
    if !value.is_finite() {
        return Err(EvalError::InvalidDatasetFormat {
            format: "widerface",
            message: format!("invalid {source} line {line_no}: `{field}` must be finite"),
        });
    }
    Ok(value)
}

fn parse_widerface_non_negative_f32(
    token: &str,
    field: &'static str,
    source: &'static str,
    line_no: usize,
) -> Result<f32, EvalError> {
    let value = parse_widerface_f32(token, field, source, line_no)?;
    if value < 0.0 {
        return Err(EvalError::InvalidDatasetFormat {
            format: "widerface",
            message: format!("invalid {source} line {line_no}: `{field}` must be >= 0"),
        });
    }
    Ok(value)
}

fn parse_widerface_i32(
    token: &str,
    field: &'static str,
    source: &'static str,
    line_no: usize,
) -> Result<i32, EvalError> {
    token
        .parse::<i32>()
        .map_err(|err| EvalError::InvalidDatasetFormat {
            format: "widerface",
            message: format!("invalid {source} line {line_no} `{field}` value `{token}`: {err}"),
        })
}
