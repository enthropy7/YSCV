use yscv_tensor::{Tensor, TensorError};

use crate::{ImageAugmentationOp, ImageAugmentationPipeline, ModelError};

use super::helpers::{
    LcgRng, class_balanced_sampling_weights, should_apply_probability, shuffle_indices,
};
use super::types::{BatchIterOptions, SamplingPolicy, SupervisedDataset};

/// Controls per-batch sample/label interpolation for regularized training.
#[derive(Debug, Clone, PartialEq)]
pub struct MixUpConfig {
    probability: f32,
    lambda_min: f32,
}

impl Default for MixUpConfig {
    fn default() -> Self {
        Self {
            probability: 1.0,
            lambda_min: 0.0,
        }
    }
}

impl MixUpConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_probability(mut self, probability: f32) -> Result<Self, ModelError> {
        validate_mixup_probability(probability)?;
        self.probability = probability;
        Ok(self)
    }

    pub fn with_lambda_min(mut self, lambda_min: f32) -> Result<Self, ModelError> {
        validate_mixup_lambda_min(lambda_min)?;
        self.lambda_min = lambda_min;
        Ok(self)
    }

    pub const fn probability(&self) -> f32 {
        self.probability
    }

    pub const fn lambda_min(&self) -> f32 {
        self.lambda_min
    }
}

/// Controls per-batch region replacement interpolation for image tensors.
#[derive(Debug, Clone, PartialEq)]
pub struct CutMixConfig {
    probability: f32,
    min_patch_fraction: f32,
    max_patch_fraction: f32,
}

impl Default for CutMixConfig {
    fn default() -> Self {
        Self {
            probability: 1.0,
            min_patch_fraction: 0.1,
            max_patch_fraction: 0.5,
        }
    }
}

impl CutMixConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_probability(mut self, probability: f32) -> Result<Self, ModelError> {
        validate_cutmix_probability(probability)?;
        self.probability = probability;
        Ok(self)
    }

    pub fn with_min_patch_fraction(mut self, min_patch_fraction: f32) -> Result<Self, ModelError> {
        validate_cutmix_patch_fraction("min_patch_fraction", min_patch_fraction)?;
        self.min_patch_fraction = min_patch_fraction;
        if self.min_patch_fraction > self.max_patch_fraction {
            return Err(ModelError::InvalidCutMixArgument {
                field: "min_patch_fraction",
                value: self.min_patch_fraction,
                message: format!(
                    "min_patch_fraction must be <= max_patch_fraction ({})",
                    self.max_patch_fraction
                ),
            });
        }
        Ok(self)
    }

    pub fn with_max_patch_fraction(mut self, max_patch_fraction: f32) -> Result<Self, ModelError> {
        validate_cutmix_patch_fraction("max_patch_fraction", max_patch_fraction)?;
        self.max_patch_fraction = max_patch_fraction;
        if self.min_patch_fraction > self.max_patch_fraction {
            return Err(ModelError::InvalidCutMixArgument {
                field: "max_patch_fraction",
                value: self.max_patch_fraction,
                message: format!(
                    "max_patch_fraction must be >= min_patch_fraction ({})",
                    self.min_patch_fraction
                ),
            });
        }
        Ok(self)
    }

    pub const fn probability(&self) -> f32 {
        self.probability
    }

    pub const fn min_patch_fraction(&self) -> f32 {
        self.min_patch_fraction
    }

    pub const fn max_patch_fraction(&self) -> f32 {
        self.max_patch_fraction
    }
}

pub(super) struct MixUpBatch {
    pub(super) inputs: Tensor,
    pub(super) targets: Tensor,
}

pub(super) fn validate_augmentation_compatibility(
    inputs: &Tensor,
    pipeline: &ImageAugmentationPipeline,
) -> Result<(), ModelError> {
    if inputs.rank() != 4 {
        return Err(ModelError::InvalidAugmentationInputShape {
            got: inputs.shape().to_vec(),
        });
    }
    let channels = inputs.shape()[3];
    for op in pipeline.ops() {
        if let ImageAugmentationOp::ChannelNormalize { mean, std: _ } = op
            && mean.len() != channels
        {
            return Err(ModelError::InvalidAugmentationArgument {
                operation: "channel_normalize",
                message: format!(
                    "channel count mismatch: dataset_channels={channels}, mean/std_len={}",
                    mean.len()
                ),
            });
        }
    }
    Ok(())
}

pub(super) fn validate_mixup_config(config: &MixUpConfig) -> Result<(), ModelError> {
    validate_mixup_probability(config.probability())?;
    validate_mixup_lambda_min(config.lambda_min())?;
    Ok(())
}

pub(super) fn validate_cutmix_config(config: &CutMixConfig) -> Result<(), ModelError> {
    validate_cutmix_probability(config.probability())?;
    validate_cutmix_patch_fraction("min_patch_fraction", config.min_patch_fraction())?;
    validate_cutmix_patch_fraction("max_patch_fraction", config.max_patch_fraction())?;
    if config.min_patch_fraction() > config.max_patch_fraction() {
        return Err(ModelError::InvalidCutMixArgument {
            field: "min_patch_fraction",
            value: config.min_patch_fraction(),
            message: format!(
                "min_patch_fraction must be <= max_patch_fraction ({})",
                config.max_patch_fraction()
            ),
        });
    }
    Ok(())
}

fn validate_mixup_probability(probability: f32) -> Result<(), ModelError> {
    if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
        return Err(ModelError::InvalidMixupArgument {
            field: "probability",
            value: probability,
            message: "probability must be finite and in [0, 1]".to_string(),
        });
    }
    Ok(())
}

fn validate_cutmix_probability(probability: f32) -> Result<(), ModelError> {
    if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
        return Err(ModelError::InvalidCutMixArgument {
            field: "probability",
            value: probability,
            message: "probability must be finite and in [0, 1]".to_string(),
        });
    }
    Ok(())
}

fn validate_cutmix_patch_fraction(field: &'static str, value: f32) -> Result<(), ModelError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(ModelError::InvalidCutMixArgument {
            field,
            value,
            message: format!("{field} must be finite and in [0, 1]"),
        });
    }
    Ok(())
}

pub(super) fn validate_cutmix_compatibility(inputs: &Tensor) -> Result<(), ModelError> {
    if inputs.rank() != 4 {
        return Err(ModelError::InvalidCutMixInputShape {
            got: inputs.shape().to_vec(),
        });
    }
    Ok(())
}

fn validate_mixup_lambda_min(lambda_min: f32) -> Result<(), ModelError> {
    if !lambda_min.is_finite() || !(0.0..=0.5).contains(&lambda_min) {
        return Err(ModelError::InvalidMixupArgument {
            field: "lambda_min",
            value: lambda_min,
            message: "lambda_min must be finite and in [0, 0.5]".to_string(),
        });
    }
    Ok(())
}

pub(super) fn apply_mixup_batch(
    inputs: &Tensor,
    targets: &Tensor,
    config: &MixUpConfig,
    seed: u64,
) -> Result<MixUpBatch, ModelError> {
    validate_mixup_config(config)?;
    if inputs.rank() == 0 || targets.rank() == 0 {
        return Err(ModelError::InvalidDatasetRank {
            inputs_rank: inputs.rank(),
            targets_rank: targets.rank(),
        });
    }
    let batch_size = inputs.shape()[0];
    if batch_size != targets.shape()[0] {
        return Err(ModelError::DatasetShapeMismatch {
            inputs: inputs.shape().to_vec(),
            targets: targets.shape().to_vec(),
        });
    }
    if batch_size < 2 {
        return Ok(MixUpBatch {
            inputs: inputs.clone(),
            targets: targets.clone(),
        });
    }

    let mut rng = LcgRng::new(seed);
    if !should_apply_probability(config.probability(), &mut rng) {
        return Ok(MixUpBatch {
            inputs: inputs.clone(),
            targets: targets.clone(),
        });
    }

    let lambda =
        config.lambda_min() + rng.next_unit_f64() as f32 * (1.0 - 2.0 * config.lambda_min());
    let partner_indices = build_partner_indices(batch_size, seed ^ 0xA5A5_A5A5_5A5A_5A5A);

    Ok(MixUpBatch {
        inputs: blend_rows(inputs, &partner_indices, lambda)?,
        targets: blend_rows(targets, &partner_indices, lambda)?,
    })
}

pub(super) fn apply_cutmix_batch(
    inputs: &Tensor,
    targets: &Tensor,
    config: &CutMixConfig,
    seed: u64,
) -> Result<MixUpBatch, ModelError> {
    validate_cutmix_config(config)?;
    validate_cutmix_compatibility(inputs)?;
    if targets.rank() == 0 {
        return Err(ModelError::InvalidDatasetRank {
            inputs_rank: inputs.rank(),
            targets_rank: targets.rank(),
        });
    }
    let batch_size = inputs.shape()[0];
    if batch_size != targets.shape()[0] {
        return Err(ModelError::DatasetShapeMismatch {
            inputs: inputs.shape().to_vec(),
            targets: targets.shape().to_vec(),
        });
    }
    if batch_size < 2 {
        return Ok(MixUpBatch {
            inputs: inputs.clone(),
            targets: targets.clone(),
        });
    }

    let mut rng = LcgRng::new(seed);
    if !should_apply_probability(config.probability(), &mut rng) {
        return Ok(MixUpBatch {
            inputs: inputs.clone(),
            targets: targets.clone(),
        });
    }

    let height = inputs.shape()[1];
    let width = inputs.shape()[2];
    let channels = inputs.shape()[3];
    if height == 0 || width == 0 || channels == 0 {
        return Ok(MixUpBatch {
            inputs: inputs.clone(),
            targets: targets.clone(),
        });
    }

    let input_row_width = height
        .checked_mul(width)
        .and_then(|value| value.checked_mul(channels))
        .ok_or_else(|| {
            ModelError::Tensor(TensorError::SizeOverflow {
                shape: inputs.shape().to_vec(),
            })
        })?;
    let target_row_width = targets.shape()[1..]
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| {
            ModelError::Tensor(TensorError::SizeOverflow {
                shape: targets.shape().to_vec(),
            })
        })?;

    let mut mixed_inputs = inputs.data().to_vec();
    let mut mixed_targets = targets.data().to_vec();
    let partner_indices = build_partner_indices(batch_size, seed ^ 0x5A5A_A5A5_DEAD_BEEF);
    let total_pixels = (height * width) as f32;

    for (row_index, partner_index) in partner_indices.iter().enumerate() {
        let patch_fraction = sample_cutmix_patch_fraction(config, &mut rng);
        let patch_height = ((height as f32 * patch_fraction).floor() as usize)
            .max(1)
            .min(height);
        let patch_width = ((width as f32 * patch_fraction).floor() as usize)
            .max(1)
            .min(width);
        let top = rng.next_usize(height - patch_height + 1);
        let left = rng.next_usize(width - patch_width + 1);

        let row_start = row_index.checked_mul(input_row_width).ok_or_else(|| {
            ModelError::Tensor(TensorError::SizeOverflow {
                shape: inputs.shape().to_vec(),
            })
        })?;
        let partner_start = partner_index.checked_mul(input_row_width).ok_or_else(|| {
            ModelError::Tensor(TensorError::SizeOverflow {
                shape: inputs.shape().to_vec(),
            })
        })?;

        for y in 0..patch_height {
            for x in 0..patch_width {
                let pixel_offset = ((top + y) * width + (left + x)) * channels;
                let dst = row_start + pixel_offset;
                let src = partner_start + pixel_offset;
                mixed_inputs[dst..(dst + channels)]
                    .copy_from_slice(&inputs.data()[src..(src + channels)]);
            }
        }

        let replaced_ratio = (patch_height * patch_width) as f32 / total_pixels;
        let lambda = 1.0 - replaced_ratio;
        let target_row_start = row_index.checked_mul(target_row_width).ok_or_else(|| {
            ModelError::Tensor(TensorError::SizeOverflow {
                shape: targets.shape().to_vec(),
            })
        })?;
        let partner_target_start =
            partner_index.checked_mul(target_row_width).ok_or_else(|| {
                ModelError::Tensor(TensorError::SizeOverflow {
                    shape: targets.shape().to_vec(),
                })
            })?;
        for offset in 0..target_row_width {
            mixed_targets[target_row_start + offset] = lambda
                * targets.data()[target_row_start + offset]
                + (1.0 - lambda) * targets.data()[partner_target_start + offset];
        }
    }

    Ok(MixUpBatch {
        inputs: Tensor::from_vec(inputs.shape().to_vec(), mixed_inputs)?,
        targets: Tensor::from_vec(targets.shape().to_vec(), mixed_targets)?,
    })
}

fn sample_cutmix_patch_fraction(config: &CutMixConfig, rng: &mut LcgRng) -> f32 {
    if (config.max_patch_fraction() - config.min_patch_fraction()).abs() <= f32::EPSILON {
        return config.min_patch_fraction();
    }
    config.min_patch_fraction()
        + rng.next_unit_f64() as f32 * (config.max_patch_fraction() - config.min_patch_fraction())
}

fn blend_rows(
    tensor: &Tensor,
    partner_indices: &[usize],
    lambda: f32,
) -> Result<Tensor, ModelError> {
    if tensor.rank() == 0 {
        return Err(ModelError::InvalidDatasetRank {
            inputs_rank: tensor.rank(),
            targets_rank: tensor.rank(),
        });
    }
    let batch_size = tensor.shape()[0];
    if partner_indices.len() != batch_size {
        return Err(ModelError::DatasetShapeMismatch {
            inputs: tensor.shape().to_vec(),
            targets: vec![partner_indices.len()],
        });
    }
    let row_width = tensor.shape()[1..]
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| {
            ModelError::Tensor(TensorError::SizeOverflow {
                shape: tensor.shape().to_vec(),
            })
        })?;

    let mut out = vec![0.0f32; tensor.len()];
    let left_weight = lambda;
    let right_weight = 1.0 - lambda;
    for (row_index, partner_index) in partner_indices.iter().enumerate() {
        if *partner_index >= batch_size {
            return Err(ModelError::DatasetShapeMismatch {
                inputs: tensor.shape().to_vec(),
                targets: vec![*partner_index, batch_size],
            });
        }
        let row_start = row_index.checked_mul(row_width).ok_or_else(|| {
            ModelError::Tensor(TensorError::SizeOverflow {
                shape: tensor.shape().to_vec(),
            })
        })?;
        let partner_start = partner_index.checked_mul(row_width).ok_or_else(|| {
            ModelError::Tensor(TensorError::SizeOverflow {
                shape: tensor.shape().to_vec(),
            })
        })?;
        for offset in 0..row_width {
            let dst = row_start + offset;
            out[dst] = left_weight * tensor.data()[row_start + offset]
                + right_weight * tensor.data()[partner_start + offset];
        }
    }
    Tensor::from_vec(tensor.shape().to_vec(), out).map_err(Into::into)
}

fn build_partner_indices(batch_size: usize, seed: u64) -> Vec<usize> {
    let mut partner_indices = (0..batch_size).collect::<Vec<_>>();
    shuffle_indices(&mut partner_indices, seed);
    if partner_indices
        .iter()
        .enumerate()
        .all(|(index, partner)| index == *partner)
    {
        partner_indices.rotate_left(1);
    }
    partner_indices
}

pub(super) fn build_sample_order(
    dataset: &SupervisedDataset,
    options: &BatchIterOptions,
) -> Result<Vec<usize>, ModelError> {
    if let Some(policy) = options.sampling.as_ref() {
        return build_sample_order_from_policy(dataset, policy);
    }

    let mut order = (0..dataset.len()).collect::<Vec<_>>();
    if options.shuffle {
        shuffle_indices(&mut order, options.shuffle_seed);
    }
    Ok(order)
}

fn build_sample_order_from_policy(
    dataset: &SupervisedDataset,
    policy: &SamplingPolicy,
) -> Result<Vec<usize>, ModelError> {
    let dataset_len = dataset.len();
    match policy {
        SamplingPolicy::Sequential => Ok((0..dataset_len).collect()),
        SamplingPolicy::Shuffled { seed } => {
            let mut order = (0..dataset_len).collect::<Vec<_>>();
            shuffle_indices(&mut order, *seed);
            Ok(order)
        }
        SamplingPolicy::BalancedByClass {
            seed,
            with_replacement,
        } => {
            let weights = class_balanced_sampling_weights(dataset.targets())?;
            if *with_replacement {
                sample_weighted_with_replacement(&weights, dataset_len, *seed)
            } else {
                sample_weighted_without_replacement(&weights, *seed)
            }
        }
        SamplingPolicy::Weighted {
            weights,
            seed,
            with_replacement,
        } => {
            validate_sampling_weights(weights, dataset_len)?;
            if *with_replacement {
                sample_weighted_with_replacement(weights, dataset_len, *seed)
            } else {
                sample_weighted_without_replacement(weights, *seed)
            }
        }
    }
}

fn validate_sampling_weights(weights: &[f32], dataset_len: usize) -> Result<(), ModelError> {
    if weights.len() != dataset_len {
        return Err(ModelError::InvalidSamplingWeightsLength {
            expected: dataset_len,
            got: weights.len(),
        });
    }
    let mut positive = false;
    for (index, weight) in weights.iter().enumerate() {
        if !weight.is_finite() || *weight < 0.0 {
            return Err(ModelError::InvalidSamplingWeight {
                index,
                value: *weight,
            });
        }
        if *weight > 0.0 {
            positive = true;
        }
    }
    if !positive && dataset_len > 0 {
        return Err(ModelError::InvalidSamplingDistribution);
    }
    Ok(())
}

fn sample_weighted_with_replacement(
    weights: &[f32],
    draw_count: usize,
    seed: u64,
) -> Result<Vec<usize>, ModelError> {
    if draw_count == 0 {
        return Ok(Vec::new());
    }

    let mut cumulative = Vec::with_capacity(weights.len());
    let mut total = 0.0f64;
    for weight in weights {
        total += *weight as f64;
        cumulative.push(total);
    }
    if total <= 0.0 {
        return Err(ModelError::InvalidSamplingDistribution);
    }

    let mut rng = LcgRng::new(seed);
    let mut out = Vec::with_capacity(draw_count);
    for _ in 0..draw_count {
        let draw = rng.next_unit_f64() * total;
        let mut sampled = cumulative.partition_point(|prefix| *prefix <= draw);
        if sampled >= weights.len() {
            sampled = weights.len() - 1;
        }
        out.push(sampled);
    }
    Ok(out)
}

fn sample_weighted_without_replacement(
    weights: &[f32],
    seed: u64,
) -> Result<Vec<usize>, ModelError> {
    if weights.is_empty() {
        return Ok(Vec::new());
    }

    let mut rng = LcgRng::new(seed);
    let mut keyed = Vec::with_capacity(weights.len());
    for (index, weight) in weights.iter().enumerate() {
        let key = if *weight == 0.0 {
            0.0
        } else {
            let u = rng.next_unit_open_f64();
            u.powf(1.0 / *weight as f64)
        };
        keyed.push((index, key));
    }

    keyed.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    Ok(keyed.into_iter().map(|(index, _)| index).collect())
}
