use std::path::Path;

use crate::ModelError;

use super::helpers::{
    adapter_sample_len, build_supervised_dataset_from_flat_values, load_dataset_text_file,
    validate_adapter_sample_shape, validate_csv_delimiter, validate_finite_values,
};

/// Configuration for parsing/loading supervised CSV datasets.
#[derive(Debug, Clone, PartialEq)]
pub struct SupervisedCsvConfig {
    input_shape: Vec<usize>,
    target_shape: Vec<usize>,
    delimiter: char,
    has_header: bool,
}

impl SupervisedCsvConfig {
    pub fn new(input_shape: Vec<usize>, target_shape: Vec<usize>) -> Result<Self, ModelError> {
        validate_adapter_sample_shape("input_shape", &input_shape)?;
        validate_adapter_sample_shape("target_shape", &target_shape)?;
        Ok(Self {
            input_shape,
            target_shape,
            delimiter: ',',
            has_header: false,
        })
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

    pub fn input_shape(&self) -> &[usize] {
        &self.input_shape
    }

    pub fn target_shape(&self) -> &[usize] {
        &self.target_shape
    }

    pub const fn delimiter(&self) -> char {
        self.delimiter
    }

    pub const fn has_header(&self) -> bool {
        self.has_header
    }
}

/// Parses supervised training samples from CSV text into a `SupervisedDataset`.
///
/// Each non-empty line is treated as one sample row with
/// `input_len + target_len` numeric columns.
/// Optional header row skipping is controlled by `config.has_header`.
pub fn parse_supervised_dataset_csv(
    content: &str,
    config: &SupervisedCsvConfig,
) -> Result<super::types::SupervisedDataset, ModelError> {
    let input_row_len = adapter_sample_len("input_shape", config.input_shape())?;
    let target_row_len = adapter_sample_len("target_shape", config.target_shape())?;
    let expected_columns = input_row_len.checked_add(target_row_len).ok_or_else(|| {
        ModelError::InvalidDatasetAdapterShape {
            field: "row_columns",
            shape: vec![input_row_len, target_row_len],
            message: "column count overflow".to_string(),
        }
    })?;

    let mut input_values = Vec::new();
    let mut target_values = Vec::new();
    let mut sample_count = 0usize;
    let mut header_skipped = false;
    let mut row_values = Vec::with_capacity(expected_columns);

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

        row_values.clear();
        for (column_idx, value_str) in columns.iter().enumerate() {
            let value = value_str
                .parse::<f32>()
                .map_err(|error| ModelError::DatasetCsvParse {
                    line: line_number,
                    column: column_idx + 1,
                    message: error.to_string(),
                })?;
            row_values.push(value);
        }

        let (input_row, target_row) = row_values.split_at(input_row_len);
        validate_finite_values(line_number, "input", input_row)?;
        validate_finite_values(line_number, "target", target_row)?;
        input_values.extend_from_slice(input_row);
        target_values.extend_from_slice(target_row);

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
        config.input_shape(),
        config.target_shape(),
        sample_count,
        input_values,
        target_values,
    )
}

/// Loads supervised training samples from a CSV file.
pub fn load_supervised_dataset_csv_file<P: AsRef<Path>>(
    path: P,
    config: &SupervisedCsvConfig,
) -> Result<super::types::SupervisedDataset, ModelError> {
    let content = load_dataset_text_file(path)?;
    parse_supervised_dataset_csv(&content, config)
}
