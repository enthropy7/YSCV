use rustc_hash::FxHashMap;
use std::fs;
use std::io::ErrorKind;
use std::path::Path;

use crate::{DetectionDatasetFrame, EvalError};

pub(crate) fn read_optional_text_file(path: &Path) -> Result<Option<String>, EvalError> {
    match fs::read_to_string(path) {
        Ok(text) => Ok(Some(text)),
        Err(err) if err.kind() == ErrorKind::NotFound => Ok(None),
        Err(err) => Err(EvalError::DatasetIo {
            path: path.display().to_string(),
            message: err.to_string(),
        }),
    }
}

pub(crate) fn ensure_detection_frame_index(
    frames: &mut Vec<DetectionDatasetFrame>,
    image_index_by_id: &mut FxHashMap<String, usize>,
    image_id: &str,
) -> usize {
    if let Some(frame_idx) = image_index_by_id.get(image_id) {
        return *frame_idx;
    }
    let frame_idx = frames.len();
    image_index_by_id.insert(image_id.to_string(), frame_idx);
    frames.push(DetectionDatasetFrame {
        ground_truth: Vec::new(),
        predictions: Vec::new(),
    });
    frame_idx
}

pub(crate) fn ensure_frame_slot<T>(frames: &mut Vec<Vec<T>>, frame_idx: usize) {
    if frames.len() <= frame_idx {
        frames.resize_with(frame_idx + 1, Vec::new);
    }
}

pub(crate) fn parse_image_id_manifest(
    text: &str,
    format: &'static str,
) -> Result<Vec<String>, EvalError> {
    use rustc_hash::FxHashSet;
    let mut image_ids = Vec::new();
    let mut seen = FxHashSet::default();
    for (line_idx, raw_line) in text.lines().enumerate() {
        let line_no = line_idx + 1;
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.split_whitespace().count() != 1 {
            return Err(EvalError::InvalidDatasetFormat {
                format,
                message: format!("manifest line {line_no} must contain exactly one image id"),
            });
        }
        if !seen.insert(line.to_string()) {
            return Err(EvalError::InvalidDatasetFormat {
                format,
                message: format!("duplicate image id `{line}` in manifest at line {line_no}"),
            });
        }
        image_ids.push(line.to_string());
    }
    Ok(image_ids)
}
