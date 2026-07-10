use rustc_hash::FxHashMap;

use csv::StringRecord;
use yscv_detect::{BoundingBox, Detection};

use crate::{DetectionDatasetFrame, EvalError, LabeledBox};

use super::helpers::ensure_detection_frame_index;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpenImagesCsvKind {
    GroundTruth,
    Prediction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OpenImagesColumns {
    image_id: usize,
    label_name: usize,
    x_min: usize,
    x_max: usize,
    y_min: usize,
    y_max: usize,
    score_idx: Option<usize>,
}

impl OpenImagesColumns {
    fn from_headers(headers: &StringRecord, kind: OpenImagesCsvKind) -> Result<Self, EvalError> {
        let image_id = find_openimages_header(headers, "ImageID").ok_or_else(|| {
            EvalError::InvalidDatasetFormat {
                format: "openimages",
                message: "missing required header `ImageID`".to_string(),
            }
        })?;
        let label_name = find_openimages_header(headers, "LabelName").ok_or_else(|| {
            EvalError::InvalidDatasetFormat {
                format: "openimages",
                message: "missing required header `LabelName`".to_string(),
            }
        })?;
        let x_min = find_openimages_header(headers, "XMin").ok_or_else(|| {
            EvalError::InvalidDatasetFormat {
                format: "openimages",
                message: "missing required header `XMin`".to_string(),
            }
        })?;
        let x_max = find_openimages_header(headers, "XMax").ok_or_else(|| {
            EvalError::InvalidDatasetFormat {
                format: "openimages",
                message: "missing required header `XMax`".to_string(),
            }
        })?;
        let y_min = find_openimages_header(headers, "YMin").ok_or_else(|| {
            EvalError::InvalidDatasetFormat {
                format: "openimages",
                message: "missing required header `YMin`".to_string(),
            }
        })?;
        let y_max = find_openimages_header(headers, "YMax").ok_or_else(|| {
            EvalError::InvalidDatasetFormat {
                format: "openimages",
                message: "missing required header `YMax`".to_string(),
            }
        })?;

        let score_idx = match kind {
            OpenImagesCsvKind::GroundTruth => None,
            OpenImagesCsvKind::Prediction => find_openimages_header(headers, "Score")
                .or_else(|| find_openimages_header(headers, "Confidence"))
                .or_else(|| find_openimages_header(headers, "Conf"))
                .ok_or_else(|| EvalError::InvalidDatasetFormat {
                    format: "openimages",
                    message: "missing required prediction score header (`Score` or `Confidence`)"
                        .to_string(),
                })
                .map(Some)?,
        };

        Ok(Self {
            image_id,
            label_name,
            x_min,
            x_max,
            y_min,
            y_max,
            score_idx,
        })
    }
}

fn find_openimages_header(headers: &StringRecord, name: &str) -> Option<usize> {
    headers
        .iter()
        .position(|header| header.trim().eq_ignore_ascii_case(name))
}

#[derive(Debug, Clone, PartialEq)]
struct OpenImagesGroundTruthEntry {
    image_id: String,
    label_name: String,
    bbox: BoundingBox,
}

#[derive(Debug, Clone, PartialEq)]
struct OpenImagesPredictionEntry {
    image_id: String,
    label_name: String,
    bbox: BoundingBox,
    score: f32,
}

pub(crate) fn parse_and_build_openimages(
    ground_truth_csv: &str,
    predictions_csv: &str,
) -> Result<Vec<DetectionDatasetFrame>, EvalError> {
    let ground_truth = parse_openimages_ground_truth(ground_truth_csv)?;
    let predictions = parse_openimages_predictions(predictions_csv)?;

    let mut frames = Vec::new();
    let mut image_index_by_id = FxHashMap::default();
    let mut class_id_by_label = FxHashMap::default();

    for entry in ground_truth {
        let frame_idx = ensure_detection_frame_index(
            &mut frames,
            &mut image_index_by_id,
            entry.image_id.as_str(),
        );
        let class_id =
            resolve_openimages_class_id(entry.label_name.as_str(), &mut class_id_by_label);
        frames[frame_idx].ground_truth.push(LabeledBox {
            bbox: entry.bbox,
            class_id,
        });
    }

    for entry in predictions {
        let frame_idx = ensure_detection_frame_index(
            &mut frames,
            &mut image_index_by_id,
            entry.image_id.as_str(),
        );
        let class_id =
            resolve_openimages_class_id(entry.label_name.as_str(), &mut class_id_by_label);
        frames[frame_idx].predictions.push(Detection {
            bbox: entry.bbox,
            score: entry.score,
            class_id,
        });
    }

    Ok(frames)
}

fn resolve_openimages_class_id(
    label_name: &str,
    class_id_by_label: &mut FxHashMap<String, usize>,
) -> usize {
    if let Some(class_id) = class_id_by_label.get(label_name) {
        return *class_id;
    }
    let class_id = class_id_by_label.len();
    class_id_by_label.insert(label_name.to_string(), class_id);
    class_id
}

fn parse_openimages_ground_truth(
    csv_text: &str,
) -> Result<Vec<OpenImagesGroundTruthEntry>, EvalError> {
    let mut reader = csv::ReaderBuilder::new()
        .trim(csv::Trim::All)
        .from_reader(csv_text.as_bytes());
    let headers = reader
        .headers()
        .map_err(|err| EvalError::InvalidDatasetFormat {
            format: "openimages",
            message: format!("failed to read ground-truth CSV headers: {err}"),
        })?
        .clone();
    let columns = OpenImagesColumns::from_headers(&headers, OpenImagesCsvKind::GroundTruth)?;

    let mut rows = Vec::new();
    for (row_idx, record_result) in reader.records().enumerate() {
        let row_no = row_idx + 2;
        let record = record_result.map_err(|err| EvalError::InvalidDatasetFormat {
            format: "openimages",
            message: format!("failed to read ground-truth CSV row {row_no}: {err}"),
        })?;
        let image_id = parse_openimages_non_empty_field(
            &record,
            columns.image_id,
            "ImageID",
            row_no,
            "ground-truth",
        )?;
        let label_name = parse_openimages_non_empty_field(
            &record,
            columns.label_name,
            "LabelName",
            row_no,
            "ground-truth",
        )?;
        let bbox = parse_openimages_bbox(&record, &columns, row_no, "ground-truth")?;
        rows.push(OpenImagesGroundTruthEntry {
            image_id: image_id.to_string(),
            label_name: label_name.to_string(),
            bbox,
        });
    }
    Ok(rows)
}

fn parse_openimages_predictions(
    csv_text: &str,
) -> Result<Vec<OpenImagesPredictionEntry>, EvalError> {
    let mut reader = csv::ReaderBuilder::new()
        .trim(csv::Trim::All)
        .from_reader(csv_text.as_bytes());
    let headers = reader
        .headers()
        .map_err(|err| EvalError::InvalidDatasetFormat {
            format: "openimages",
            message: format!("failed to read prediction CSV headers: {err}"),
        })?
        .clone();
    let columns = OpenImagesColumns::from_headers(&headers, OpenImagesCsvKind::Prediction)?;

    let mut rows = Vec::new();
    for (row_idx, record_result) in reader.records().enumerate() {
        let row_no = row_idx + 2;
        let record = record_result.map_err(|err| EvalError::InvalidDatasetFormat {
            format: "openimages",
            message: format!("failed to read prediction CSV row {row_no}: {err}"),
        })?;
        let image_id = parse_openimages_non_empty_field(
            &record,
            columns.image_id,
            "ImageID",
            row_no,
            "prediction",
        )?;
        let label_name = parse_openimages_non_empty_field(
            &record,
            columns.label_name,
            "LabelName",
            row_no,
            "prediction",
        )?;
        let bbox = parse_openimages_bbox(&record, &columns, row_no, "prediction")?;
        let score_idx = columns
            .score_idx
            .ok_or_else(|| EvalError::InvalidDatasetFormat {
                format: "openimages",
                message: "missing score column index in prediction parser".to_string(),
            })?;
        let score_raw = parse_openimages_non_empty_field(
            &record,
            score_idx,
            "Score/Confidence",
            row_no,
            "prediction",
        )?;
        let score = parse_openimages_unit_f32(score_raw, "Score/Confidence", row_no, "prediction")?;

        rows.push(OpenImagesPredictionEntry {
            image_id: image_id.to_string(),
            label_name: label_name.to_string(),
            bbox,
            score,
        });
    }
    Ok(rows)
}

fn parse_openimages_non_empty_field<'a>(
    record: &'a StringRecord,
    index: usize,
    field: &'static str,
    row_no: usize,
    source: &'static str,
) -> Result<&'a str, EvalError> {
    let Some(raw) = record.get(index) else {
        return Err(EvalError::InvalidDatasetFormat {
            format: "openimages",
            message: format!("missing `{field}` in {source} CSV row {row_no}"),
        });
    };
    let value = raw.trim();
    if value.is_empty() {
        return Err(EvalError::InvalidDatasetFormat {
            format: "openimages",
            message: format!("empty `{field}` in {source} CSV row {row_no}"),
        });
    }
    Ok(value)
}

fn parse_openimages_bbox(
    record: &StringRecord,
    columns: &OpenImagesColumns,
    row_no: usize,
    source: &'static str,
) -> Result<BoundingBox, EvalError> {
    let x1_raw = parse_openimages_non_empty_field(record, columns.x_min, "XMin", row_no, source)?;
    let x2_raw = parse_openimages_non_empty_field(record, columns.x_max, "XMax", row_no, source)?;
    let y1_raw = parse_openimages_non_empty_field(record, columns.y_min, "YMin", row_no, source)?;
    let y2_raw = parse_openimages_non_empty_field(record, columns.y_max, "YMax", row_no, source)?;
    let x1 = parse_openimages_unit_f32(x1_raw, "XMin", row_no, source)?;
    let x2 = parse_openimages_unit_f32(x2_raw, "XMax", row_no, source)?;
    let y1 = parse_openimages_unit_f32(y1_raw, "YMin", row_no, source)?;
    let y2 = parse_openimages_unit_f32(y2_raw, "YMax", row_no, source)?;
    if x2 <= x1 || y2 <= y1 {
        return Err(EvalError::InvalidDatasetFormat {
            format: "openimages",
            message: format!("invalid {source} CSV row {row_no}: expected XMax>XMin and YMax>YMin"),
        });
    }
    Ok(BoundingBox { x1, y1, x2, y2 })
}

fn parse_openimages_unit_f32(
    token: &str,
    field: &'static str,
    row_no: usize,
    source: &'static str,
) -> Result<f32, EvalError> {
    let value = token
        .parse::<f32>()
        .map_err(|err| EvalError::InvalidDatasetFormat {
            format: "openimages",
            message: format!("invalid {source} CSV row {row_no} `{field}` value `{token}`: {err}"),
        })?;
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(EvalError::InvalidDatasetFormat {
            format: "openimages",
            message: format!(
                "invalid {source} CSV row {row_no}: `{field}` must be finite and in [0, 1]"
            ),
        });
    }
    Ok(value)
}
