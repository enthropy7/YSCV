use std::path::{Path, PathBuf};

use crate::ModelError;

use super::helpers::{
    adapter_sample_len, build_supervised_dataset_from_flat_values, load_dataset_text_file,
    validate_adapter_sample_shape, validate_csv_delimiter, validate_finite_values,
};
use super::image_folder::load_image_as_normalized_rgb_tensor;

/// Configuration for parsing/loading supervised image-manifest CSV datasets.
///
/// Expected row format:
/// `image_path,target_0,target_1,...`
#[derive(Debug, Clone, PartialEq)]
pub struct SupervisedImageManifestConfig {
    target_shape: Vec<usize>,
    output_height: usize,
    output_width: usize,
    pub(super) image_root: PathBuf,
    delimiter: char,
    has_header: bool,
}

impl SupervisedImageManifestConfig {
    pub fn new(
        target_shape: Vec<usize>,
        output_height: usize,
        output_width: usize,
    ) -> Result<Self, ModelError> {
        validate_adapter_sample_shape("target_shape", &target_shape)?;
        if output_height == 0 {
            return Err(ModelError::InvalidDatasetAdapterShape {
                field: "output_height",
                shape: vec![output_height],
                message: "output_height must be > 0".to_string(),
            });
        }
        if output_width == 0 {
            return Err(ModelError::InvalidDatasetAdapterShape {
                field: "output_width",
                shape: vec![output_width],
                message: "output_width must be > 0".to_string(),
            });
        }

        Ok(Self {
            target_shape,
            output_height,
            output_width,
            image_root: PathBuf::from("."),
            delimiter: ',',
            has_header: false,
        })
    }

    pub fn with_image_root<P: Into<PathBuf>>(mut self, image_root: P) -> Self {
        self.image_root = image_root.into();
        self
    }

    pub fn with_delimiter(mut self, delimiter: char) -> Result<Self, ModelError> {
        validate_csv_delimiter(delimiter)?;
        self.delimiter = delimiter;
        Ok(self)
    }

    pub const fn with_header(mut self, has_header: bool) -> Self {
        self.has_header = has_header;
        self
    }

    pub fn target_shape(&self) -> &[usize] {
        &self.target_shape
    }

    pub const fn output_height(&self) -> usize {
        self.output_height
    }

    pub const fn output_width(&self) -> usize {
        self.output_width
    }

    pub fn image_root(&self) -> &Path {
        &self.image_root
    }

    pub const fn delimiter(&self) -> char {
        self.delimiter
    }

    pub const fn has_header(&self) -> bool {
        self.has_header
    }
}

fn resolve_manifest_image_path(image_root: &Path, image_field: &str) -> PathBuf {
    let manifest_path = Path::new(image_field);
    if manifest_path.is_absolute() {
        manifest_path.to_path_buf()
    } else {
        image_root.join(manifest_path)
    }
}

/// Parses supervised training image-manifest CSV into a `SupervisedDataset`.
///
/// Manifest row format:
/// `image_path,target_0,target_1,...`
pub fn parse_supervised_image_manifest_csv(
    content: &str,
    config: &SupervisedImageManifestConfig,
) -> Result<super::types::SupervisedDataset, ModelError> {
    let target_row_len = adapter_sample_len("target_shape", config.target_shape())?;
    let expected_columns =
        target_row_len
            .checked_add(1)
            .ok_or_else(|| ModelError::InvalidDatasetAdapterShape {
                field: "manifest_columns",
                shape: vec![target_row_len],
                message: "column count overflow".to_string(),
            })?;

    let mut input_values = Vec::new();
    let mut target_values = Vec::new();
    let mut sample_count = 0usize;
    let mut header_skipped = false;
    let mut target_row = Vec::with_capacity(target_row_len);

    for (line_idx, raw_line) in content.lines().enumerate() {
        let line_number = line_idx + 1;
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if config.has_header() && !header_skipped {
            header_skipped = true;
            continue;
        }

        let columns = line
            .split(config.delimiter())
            .map(str::trim)
            .collect::<Vec<_>>();
        if columns.len() != expected_columns {
            return Err(ModelError::InvalidDatasetRecordColumns {
                line: line_number,
                expected: expected_columns,
                got: columns.len(),
            });
        }

        let image_field = columns[0];
        if image_field.is_empty() {
            return Err(ModelError::InvalidDatasetRecordPath {
                line: line_number,
                message: "image path is empty".to_string(),
            });
        }

        let image_path = resolve_manifest_image_path(config.image_root(), image_field);
        let image_tensor = load_image_as_normalized_rgb_tensor(
            &image_path,
            config.output_height(),
            config.output_width(),
        )?;
        input_values.extend_from_slice(image_tensor.data());

        target_row.clear();
        for (target_idx, target_str) in columns[1..].iter().enumerate() {
            let value = target_str
                .parse::<f32>()
                .map_err(|error| ModelError::DatasetCsvParse {
                    line: line_number,
                    column: target_idx + 2,
                    message: error.to_string(),
                })?;
            target_row.push(value);
        }
        validate_finite_values(line_number, "target", &target_row)?;
        target_values.extend_from_slice(&target_row);

        sample_count =
            sample_count
                .checked_add(1)
                .ok_or_else(|| ModelError::InvalidDatasetAdapterShape {
                    field: "sample_count",
                    shape: vec![sample_count],
                    message: "sample count overflow".to_string(),
                })?;
    }

    if sample_count == 0 {
        return Err(ModelError::EmptyDataset);
    }

    build_supervised_dataset_from_flat_values(
        &[config.output_height(), config.output_width(), 3],
        config.target_shape(),
        sample_count,
        input_values,
        target_values,
    )
}

/// Loads supervised training image-manifest CSV from file.
pub fn load_supervised_image_manifest_csv_file<P: AsRef<Path>>(
    path: P,
    config: &SupervisedImageManifestConfig,
) -> Result<super::types::SupervisedDataset, ModelError> {
    let path_ref = path.as_ref();
    let content = load_dataset_text_file(path_ref)?;

    let mut effective_config = config.clone();
    if !effective_config.image_root.is_absolute() {
        let manifest_dir = path_ref.parent().unwrap_or_else(|| Path::new("."));
        effective_config.image_root = manifest_dir.join(&effective_config.image_root);
    }

    parse_supervised_image_manifest_csv(&content, &effective_config)
}
