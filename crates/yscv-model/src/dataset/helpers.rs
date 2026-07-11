use std::collections::HashMap;
use std::fs;
use std::path::Path;
use yscv_tensor::{Tensor, TensorError};

use crate::ModelError;

use super::types::SupervisedDataset;

pub(super) fn build_supervised_dataset_from_flat_values(
    input_shape: &[usize],
    target_shape: &[usize],
    sample_count: usize,
    input_values: Vec<f32>,
    target_values: Vec<f32>,
) -> Result<SupervisedDataset, ModelError> {
    let mut full_input_shape = Vec::with_capacity(input_shape.len() + 1);
    full_input_shape.push(sample_count);
    full_input_shape.extend_from_slice(input_shape);

    let mut full_target_shape = Vec::with_capacity(target_shape.len() + 1);
    full_target_shape.push(sample_count);
    full_target_shape.extend_from_slice(target_shape);

    let inputs = Tensor::from_vec(full_input_shape, input_values)?;
    let targets = Tensor::from_vec(full_target_shape, target_values)?;
    SupervisedDataset::new(inputs, targets)
}

pub(super) fn load_dataset_text_file<P: AsRef<Path>>(path: P) -> Result<String, ModelError> {
    let path_ref = path.as_ref();
    fs::read_to_string(path_ref).map_err(|error| ModelError::DatasetLoadIo {
        path: path_ref.display().to_string(),
        message: error.to_string(),
    })
}

pub(super) fn validate_adapter_sample_shape(
    field: &'static str,
    shape: &[usize],
) -> Result<(), ModelError> {
    if let Some((index, _)) = shape.iter().enumerate().find(|(_, dim)| **dim == 0) {
        return Err(ModelError::InvalidDatasetAdapterShape {
            field,
            shape: shape.to_vec(),
            message: format!("dimension at index {index} must be > 0"),
        });
    }
    adapter_sample_len(field, shape).map(|_| ())
}

pub(super) fn adapter_sample_len(
    field: &'static str,
    shape: &[usize],
) -> Result<usize, ModelError> {
    shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| ModelError::InvalidDatasetAdapterShape {
                field,
                shape: shape.to_vec(),
                message: "shape product overflow".to_string(),
            })
    })
}

pub(super) fn validate_csv_delimiter(delimiter: char) -> Result<(), ModelError> {
    if delimiter.is_control() {
        return Err(ModelError::InvalidCsvDelimiter { delimiter });
    }
    Ok(())
}

pub(super) fn validate_split_ratios(
    train_ratio: f32,
    validation_ratio: f32,
) -> Result<(), ModelError> {
    if !train_ratio.is_finite()
        || !validation_ratio.is_finite()
        || !(0.0..=1.0).contains(&train_ratio)
        || !(0.0..=1.0).contains(&validation_ratio)
        || train_ratio + validation_ratio > 1.0
    {
        return Err(ModelError::InvalidSplitRatios {
            train_ratio,
            validation_ratio,
        });
    }
    Ok(())
}

pub(super) fn validate_finite_values(
    line: usize,
    field: &'static str,
    values: &[f32],
) -> Result<(), ModelError> {
    for (index, value) in values.iter().enumerate() {
        if value.is_nan() {
            return Err(ModelError::InvalidDatasetRecordValue {
                line,
                field,
                index,
                reason: "NaN is not allowed",
            });
        }
        if value.is_infinite() {
            return Err(ModelError::InvalidDatasetRecordValue {
                line,
                field,
                index,
                reason: "infinite value is not allowed",
            });
        }
    }
    Ok(())
}

pub(super) fn gather_rows(tensor: &Tensor, indices: &[usize]) -> Result<Tensor, ModelError> {
    if tensor.rank() == 0 {
        return Err(ModelError::InvalidDatasetRank {
            inputs_rank: tensor.rank(),
            targets_rank: tensor.rank(),
        });
    }

    let rows = tensor.shape()[0];
    for index in indices {
        if *index >= rows {
            return Err(ModelError::DatasetShapeMismatch {
                inputs: tensor.shape().to_vec(),
                targets: vec![*index, rows],
            });
        }
    }

    let row_width = tensor.shape()[1..]
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| {
            ModelError::Tensor(TensorError::SizeOverflow {
                shape: tensor.shape().to_vec(),
            })
        })?;
    let out_len = indices.len().checked_mul(row_width).ok_or_else(|| {
        ModelError::Tensor(TensorError::SizeOverflow {
            shape: vec![indices.len(), row_width],
        })
    })?;
    let mut out = Vec::with_capacity(out_len);

    for index in indices {
        let data_start = index.checked_mul(row_width).ok_or_else(|| {
            ModelError::Tensor(TensorError::SizeOverflow {
                shape: tensor.shape().to_vec(),
            })
        })?;
        let data_end = data_start.checked_add(row_width).ok_or_else(|| {
            ModelError::Tensor(TensorError::SizeOverflow {
                shape: tensor.shape().to_vec(),
            })
        })?;
        out.extend_from_slice(&tensor.data()[data_start..data_end]);
    }

    let mut new_shape = tensor.shape().to_vec();
    new_shape[0] = indices.len();
    Tensor::from_vec(new_shape, out).map_err(Into::into)
}

pub(super) fn parse_scalar_class_label(
    sample_index: usize,
    value: f32,
) -> Result<usize, ModelError> {
    if !value.is_finite() {
        return Err(ModelError::InvalidClassSamplingTargetValue {
            index: sample_index,
            value,
            reason: "class label must be finite",
        });
    }
    if value < 0.0 {
        return Err(ModelError::InvalidClassSamplingTargetValue {
            index: sample_index,
            value,
            reason: "class label must be >= 0",
        });
    }

    let rounded = value.round();
    if (value - rounded).abs() > 1e-6 {
        return Err(ModelError::InvalidClassSamplingTargetValue {
            index: sample_index,
            value,
            reason: "class label must be an integer value",
        });
    }
    if rounded > usize::MAX as f32 {
        return Err(ModelError::InvalidClassSamplingTargetValue {
            index: sample_index,
            value,
            reason: "class label is out of usize range",
        });
    }

    Ok(rounded as usize)
}

pub(super) fn parse_one_hot_class_label(
    sample_index: usize,
    values: &[f32],
) -> Result<usize, ModelError> {
    const ONE_HOT_VALUE_TOLERANCE: f32 = 1e-5;

    let mut active_class = None;
    for (class_index, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(ModelError::InvalidClassSamplingTargetValue {
                index: sample_index,
                value,
                reason: "one-hot value must be finite",
            });
        }
        if !(0.0..=1.0).contains(&value) {
            return Err(ModelError::InvalidClassSamplingTargetValue {
                index: sample_index,
                value,
                reason: "one-hot value must be in [0, 1]",
            });
        }

        let near_zero = value.abs() <= ONE_HOT_VALUE_TOLERANCE;
        let near_one = (1.0 - value).abs() <= ONE_HOT_VALUE_TOLERANCE;
        if !near_zero && !near_one {
            return Err(ModelError::InvalidClassSamplingTargetValue {
                index: sample_index,
                value,
                reason: "one-hot value must be close to 0 or 1",
            });
        }

        if near_one {
            if active_class.is_some() {
                return Err(ModelError::InvalidClassSamplingTargetValue {
                    index: sample_index,
                    value,
                    reason: "one-hot row must contain exactly one active class",
                });
            }
            active_class = Some(class_index);
        }
    }

    active_class.ok_or(ModelError::InvalidClassSamplingTargetValue {
        index: sample_index,
        value: 0.0,
        reason: "one-hot row must contain exactly one active class",
    })
}

pub(super) fn class_labels_from_targets(targets: &Tensor) -> Result<Vec<usize>, ModelError> {
    if targets.rank() == 0 {
        return Err(ModelError::InvalidDatasetRank {
            inputs_rank: targets.rank(),
            targets_rank: targets.rank(),
        });
    }
    if targets.rank() != 2 || targets.shape()[1] == 0 {
        return Err(ModelError::InvalidClassSamplingTargetShape {
            got: targets.shape().to_vec(),
        });
    }

    let sample_count = targets.shape()[0];
    let target_width = targets.shape()[1];
    let mut class_ids = Vec::with_capacity(sample_count);
    for sample_index in 0..sample_count {
        let row_start = sample_index.checked_mul(target_width).ok_or_else(|| {
            ModelError::Tensor(TensorError::SizeOverflow {
                shape: targets.shape().to_vec(),
            })
        })?;
        let row_end = row_start.checked_add(target_width).ok_or_else(|| {
            ModelError::Tensor(TensorError::SizeOverflow {
                shape: targets.shape().to_vec(),
            })
        })?;
        let row = &targets.data()[row_start..row_end];
        let class_id = if target_width == 1 {
            parse_scalar_class_label(sample_index, row[0])?
        } else {
            parse_one_hot_class_label(sample_index, row)?
        };
        class_ids.push(class_id);
    }
    Ok(class_ids)
}

pub(super) fn class_balanced_sampling_weights(targets: &Tensor) -> Result<Vec<f32>, ModelError> {
    let class_ids = class_labels_from_targets(targets)?;
    if class_ids.is_empty() {
        return Ok(Vec::new());
    }

    let mut class_counts = HashMap::<usize, usize>::default();
    for class_id in &class_ids {
        let next_count = class_counts
            .get(class_id)
            .copied()
            .unwrap_or(0)
            .saturating_add(1);
        class_counts.insert(*class_id, next_count);
    }

    let mut weights = Vec::with_capacity(class_ids.len());
    for class_id in class_ids {
        let class_count = class_counts
            .get(&class_id)
            .copied()
            .ok_or(ModelError::InvalidSamplingDistribution)?;
        weights.push(1.0 / class_count as f32);
    }
    Ok(weights)
}

pub(super) fn shuffle_indices(indices: &mut [usize], seed: u64) {
    let mut rng = LcgRng::new(seed);
    let mut index = indices.len();
    while index > 1 {
        index -= 1;
        let swap_idx = rng.next_usize(index + 1);
        indices.swap(index, swap_idx);
    }
}

pub(super) fn should_apply_probability(probability: f32, rng: &mut LcgRng) -> bool {
    if probability <= 0.0 {
        return false;
    }
    if probability >= 1.0 {
        return true;
    }
    rng.next_unit_f64() < probability as f64
}

#[derive(Debug, Clone, Copy)]
pub(super) struct LcgRng {
    state: u64,
}

impl LcgRng {
    const MULTIPLIER: u64 = 6364136223846793005;
    const INCREMENT: u64 = 1;

    pub(super) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub(super) fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(Self::MULTIPLIER)
            .wrapping_add(Self::INCREMENT);
        self.state
    }

    pub(super) fn next_unit_f64(&mut self) -> f64 {
        self.next_u64() as f64 / (u64::MAX as f64 + 1.0)
    }

    pub(super) fn next_unit_open_f64(&mut self) -> f64 {
        (self.next_u64() as f64 + 1.0) / (u64::MAX as f64 + 2.0)
    }

    pub(super) fn next_usize(&mut self, upper_exclusive: usize) -> usize {
        if upper_exclusive == 0 {
            return 0;
        }
        (self.next_u64() as usize) % upper_exclusive
    }
}
