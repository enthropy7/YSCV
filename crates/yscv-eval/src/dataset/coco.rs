use rustc_hash::{FxBuildHasher, FxHashMap};

use yscv_detect::{BoundingBox, Detection};

use crate::{DetectionDatasetFrame, EvalError, LabeledBox};

use super::types::{CocoGroundTruthWire, CocoPredictionWire};

pub(crate) fn build_detection_dataset_from_coco(
    ground_truth: CocoGroundTruthWire,
    predictions: Vec<CocoPredictionWire>,
) -> Result<Vec<DetectionDatasetFrame>, EvalError> {
    let mut frames = Vec::with_capacity(ground_truth.images.len());
    let mut image_index_by_id =
        FxHashMap::with_capacity_and_hasher(ground_truth.images.len(), FxBuildHasher);

    for (frame_idx, image) in ground_truth.images.into_iter().enumerate() {
        if image_index_by_id.insert(image.id, frame_idx).is_some() {
            return Err(EvalError::InvalidDatasetFormat {
                format: "coco",
                message: format!("duplicate image id {}", image.id),
            });
        }
        frames.push(DetectionDatasetFrame {
            ground_truth: Vec::new(),
            predictions: Vec::new(),
        });
    }

    for (annotation_idx, annotation) in ground_truth.annotations.into_iter().enumerate() {
        let Some(&frame_idx) = image_index_by_id.get(&annotation.image_id) else {
            return Err(EvalError::InvalidDatasetFormat {
                format: "coco",
                message: format!(
                    "annotation[{annotation_idx}] references unknown image_id {}",
                    annotation.image_id
                ),
            });
        };
        let class_id =
            coco_category_id_to_class_id(annotation.category_id, "annotation", annotation_idx)?;
        let bbox = coco_bbox_to_runtime(annotation.bbox, "annotation", annotation_idx)?;
        frames[frame_idx]
            .ground_truth
            .push(LabeledBox { bbox, class_id });
    }

    for (prediction_idx, prediction) in predictions.into_iter().enumerate() {
        let Some(&frame_idx) = image_index_by_id.get(&prediction.image_id) else {
            return Err(EvalError::InvalidDatasetFormat {
                format: "coco",
                message: format!(
                    "prediction[{prediction_idx}] references unknown image_id {}",
                    prediction.image_id
                ),
            });
        };
        if !prediction.score.is_finite() {
            return Err(EvalError::InvalidDatasetFormat {
                format: "coco",
                message: format!(
                    "prediction[{prediction_idx}] has non-finite score {}",
                    prediction.score
                ),
            });
        }
        let class_id =
            coco_category_id_to_class_id(prediction.category_id, "prediction", prediction_idx)?;
        let bbox = coco_bbox_to_runtime(prediction.bbox, "prediction", prediction_idx)?;
        frames[frame_idx].predictions.push(Detection {
            bbox,
            score: prediction.score,
            class_id,
        });
    }

    Ok(frames)
}

fn coco_category_id_to_class_id(
    category_id: u64,
    item_kind: &'static str,
    item_idx: usize,
) -> Result<usize, EvalError> {
    usize::try_from(category_id).map_err(|_| EvalError::InvalidDatasetFormat {
        format: "coco",
        message: format!(
            "{item_kind}[{item_idx}] category_id {category_id} exceeds usize::MAX on this platform"
        ),
    })
}

fn coco_bbox_to_runtime(
    bbox: [f32; 4],
    item_kind: &'static str,
    item_idx: usize,
) -> Result<BoundingBox, EvalError> {
    let [x, y, width, height] = bbox;
    if !x.is_finite() || !y.is_finite() || !width.is_finite() || !height.is_finite() {
        return Err(EvalError::InvalidDatasetFormat {
            format: "coco",
            message: format!("{item_kind}[{item_idx}] has non-finite bbox values"),
        });
    }
    if width < 0.0 || height < 0.0 {
        return Err(EvalError::InvalidDatasetFormat {
            format: "coco",
            message: format!(
                "{item_kind}[{item_idx}] has negative bbox size width={width}, height={height}"
            ),
        });
    }

    let x2 = x + width;
    let y2 = y + height;
    if !x2.is_finite() || !y2.is_finite() {
        return Err(EvalError::InvalidDatasetFormat {
            format: "coco",
            message: format!("{item_kind}[{item_idx}] bbox corner overflow"),
        });
    }
    Ok(BoundingBox {
        x1: x,
        y1: y,
        x2,
        y2,
    })
}
