use rustc_hash::FxHashMap;
use yscv_tensor::Tensor;

use crate::{ImageAugmentationPipeline, ModelError};

use super::helpers::{
    class_labels_from_targets, gather_rows, shuffle_indices, validate_split_ratios,
};
use super::iter::{
    CutMixConfig, MixUpConfig, apply_cutmix_batch, apply_mixup_batch, build_sample_order,
    validate_augmentation_compatibility, validate_cutmix_compatibility, validate_cutmix_config,
    validate_mixup_config,
};

/// Supervised dataset with aligned input/target sample axis at position 0.
#[derive(Debug, Clone, PartialEq)]
pub struct SupervisedDataset {
    inputs: Tensor,
    targets: Tensor,
}

impl SupervisedDataset {
    pub fn new(inputs: Tensor, targets: Tensor) -> Result<Self, ModelError> {
        if inputs.rank() == 0 || targets.rank() == 0 {
            return Err(ModelError::InvalidDatasetRank {
                inputs_rank: inputs.rank(),
                targets_rank: targets.rank(),
            });
        }
        if inputs.shape()[0] != targets.shape()[0] {
            return Err(ModelError::DatasetShapeMismatch {
                inputs: inputs.shape().to_vec(),
                targets: targets.shape().to_vec(),
            });
        }
        Ok(Self { inputs, targets })
    }

    pub fn len(&self) -> usize {
        self.inputs.shape()[0]
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn inputs(&self) -> &Tensor {
        &self.inputs
    }

    pub fn targets(&self) -> &Tensor {
        &self.targets
    }

    pub fn batches(&self, batch_size: usize) -> Result<MiniBatchIter<'_>, ModelError> {
        self.batches_with_options(batch_size, BatchIterOptions::default())
    }

    pub fn batches_with_options(
        &self,
        batch_size: usize,
        options: BatchIterOptions,
    ) -> Result<MiniBatchIter<'_>, ModelError> {
        if batch_size == 0 {
            return Err(ModelError::InvalidBatchSize { batch_size });
        }
        if let Some(pipeline) = options.augmentation.as_ref() {
            validate_augmentation_compatibility(self.inputs(), pipeline)?;
        }
        if let Some(mixup) = options.mixup.as_ref() {
            validate_mixup_config(mixup)?;
        }
        if let Some(cutmix) = options.cutmix.as_ref() {
            validate_cutmix_config(cutmix)?;
            validate_cutmix_compatibility(self.inputs())?;
        }

        let order = build_sample_order(self, &options)?;

        Ok(MiniBatchIter {
            dataset: self,
            batch_size,
            cursor: 0,
            order,
            drop_last: options.drop_last,
            augmentation: options.augmentation,
            augmentation_seed: options.augmentation_seed,
            mixup: options.mixup,
            mixup_seed: options.mixup_seed,
            cutmix: options.cutmix,
            cutmix_seed: options.cutmix_seed,
            emitted_batches: 0,
        })
    }

    pub fn split_by_counts(
        &self,
        train_count: usize,
        validation_count: usize,
        shuffle: bool,
        seed: u64,
    ) -> Result<DatasetSplit, ModelError> {
        if train_count
            .checked_add(validation_count)
            .is_none_or(|sum| sum > self.len())
        {
            return Err(ModelError::InvalidSplitCounts {
                train_count,
                validation_count,
                dataset_len: self.len(),
            });
        }

        let mut order = (0..self.len()).collect::<Vec<_>>();
        if shuffle {
            shuffle_indices(&mut order, seed);
        }

        let train = self.subset_by_indices(&order[..train_count])?;
        let validation_end = train_count + validation_count;
        let validation = self.subset_by_indices(&order[train_count..validation_end])?;
        let test = self.subset_by_indices(&order[validation_end..])?;
        Ok(DatasetSplit {
            train,
            validation,
            test,
        })
    }

    pub fn split_by_ratio(
        &self,
        train_ratio: f32,
        validation_ratio: f32,
        shuffle: bool,
        seed: u64,
    ) -> Result<DatasetSplit, ModelError> {
        validate_split_ratios(train_ratio, validation_ratio)?;

        let len = self.len();
        let train_count = ((len as f64) * train_ratio as f64).floor() as usize;
        let remaining = len.saturating_sub(train_count);
        let validation_count =
            (((len as f64) * validation_ratio as f64).floor() as usize).min(remaining);
        self.split_by_counts(train_count, validation_count, shuffle, seed)
    }

    pub fn split_by_class_ratio(
        &self,
        train_ratio: f32,
        validation_ratio: f32,
        shuffle: bool,
        seed: u64,
    ) -> Result<DatasetSplit, ModelError> {
        validate_split_ratios(train_ratio, validation_ratio)?;

        let class_labels = class_labels_from_targets(self.targets())?;
        let mut indices_by_class = FxHashMap::<usize, Vec<usize>>::default();
        for (sample_index, class_id) in class_labels.into_iter().enumerate() {
            indices_by_class
                .entry(class_id)
                .or_default()
                .push(sample_index);
        }

        let mut class_order = indices_by_class.keys().copied().collect::<Vec<_>>();
        class_order.sort_unstable();

        let mut train_indices = Vec::new();
        let mut validation_indices = Vec::new();
        let mut test_indices = Vec::new();

        for class_id in class_order {
            let mut class_indices = indices_by_class
                .remove(&class_id)
                .ok_or(ModelError::InvalidSamplingDistribution)?;
            if shuffle {
                let class_seed = seed ^ (class_id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                shuffle_indices(&mut class_indices, class_seed);
            }

            let class_len = class_indices.len();
            let class_train_count = ((class_len as f64) * train_ratio as f64).floor() as usize;
            let class_remaining = class_len.saturating_sub(class_train_count);
            let class_validation_count = (((class_len as f64) * validation_ratio as f64).floor()
                as usize)
                .min(class_remaining);

            let validation_start = class_train_count;
            let validation_end = validation_start + class_validation_count;
            train_indices.extend_from_slice(&class_indices[..class_train_count]);
            validation_indices.extend_from_slice(&class_indices[validation_start..validation_end]);
            test_indices.extend_from_slice(&class_indices[validation_end..]);
        }

        if shuffle {
            shuffle_indices(&mut train_indices, seed ^ 0xA55A_5AA5_1234_5678);
            shuffle_indices(&mut validation_indices, seed ^ 0xBEE5_F00D_89AB_CDEF);
            shuffle_indices(&mut test_indices, seed ^ 0xDEAD_BEEF_CAFE_BABE);
        }

        let train = self.subset_by_indices(&train_indices)?;
        let validation = self.subset_by_indices(&validation_indices)?;
        let test = self.subset_by_indices(&test_indices)?;
        Ok(DatasetSplit {
            train,
            validation,
            test,
        })
    }

    fn subset_by_indices(&self, indices: &[usize]) -> Result<SupervisedDataset, ModelError> {
        let inputs = gather_rows(self.inputs(), indices)?;
        let targets = gather_rows(self.targets(), indices)?;
        SupervisedDataset::new(inputs, targets)
    }
}

/// One deterministic batch from dataset iterator.
#[derive(Debug, Clone, PartialEq)]
pub struct Batch {
    pub inputs: Tensor,
    pub targets: Tensor,
}

/// Sample-order policy used by `BatchIterOptions`.
#[derive(Debug, Clone, PartialEq)]
pub enum SamplingPolicy {
    Sequential,
    Shuffled {
        seed: u64,
    },
    BalancedByClass {
        seed: u64,
        with_replacement: bool,
    },
    Weighted {
        weights: Vec<f32>,
        seed: u64,
        with_replacement: bool,
    },
}

/// Controls mini-batch order, truncation behavior, and optional per-batch regularization.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BatchIterOptions {
    pub shuffle: bool,
    pub shuffle_seed: u64,
    pub drop_last: bool,
    pub augmentation: Option<ImageAugmentationPipeline>,
    pub augmentation_seed: u64,
    pub mixup: Option<MixUpConfig>,
    pub mixup_seed: u64,
    pub cutmix: Option<CutMixConfig>,
    pub cutmix_seed: u64,
    pub sampling: Option<SamplingPolicy>,
}

/// Deterministic dataset split produced by `split_by_counts` / `split_by_ratio`.
#[derive(Debug, Clone, PartialEq)]
pub struct DatasetSplit {
    pub train: SupervisedDataset,
    pub validation: SupervisedDataset,
    pub test: SupervisedDataset,
}

/// Deterministic sequential mini-batch iterator.
#[derive(Debug)]
pub struct MiniBatchIter<'a> {
    pub(super) dataset: &'a SupervisedDataset,
    pub(super) batch_size: usize,
    pub(super) cursor: usize,
    pub(super) order: Vec<usize>,
    pub(super) drop_last: bool,
    pub(super) augmentation: Option<ImageAugmentationPipeline>,
    pub(super) augmentation_seed: u64,
    pub(super) mixup: Option<MixUpConfig>,
    pub(super) mixup_seed: u64,
    pub(super) cutmix: Option<CutMixConfig>,
    pub(super) cutmix_seed: u64,
    pub(super) emitted_batches: usize,
}

impl Iterator for MiniBatchIter<'_> {
    type Item = Batch;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor >= self.order.len() {
            return None;
        }
        let start = self.cursor;
        let end = (self.cursor + self.batch_size).min(self.order.len());
        if self.drop_last && (end - start) < self.batch_size {
            self.cursor = self.order.len();
            return None;
        }
        self.cursor = end;

        let batch_indices = &self.order[start..end];
        let mut inputs = gather_rows(&self.dataset.inputs, batch_indices).ok()?;
        let mut targets = gather_rows(&self.dataset.targets, batch_indices).ok()?;
        if let Some(pipeline) = self.augmentation.as_ref() {
            let seed = self.augmentation_seed
                ^ (self.emitted_batches as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            inputs = pipeline.apply_nhwc(&inputs, seed).ok()?;
        }
        if let Some(mixup) = self.mixup.as_ref() {
            let seed =
                self.mixup_seed ^ (self.emitted_batches as u64).wrapping_mul(0xD134_2543_DE82_EF95);
            let mixed = apply_mixup_batch(&inputs, &targets, mixup, seed).ok()?;
            inputs = mixed.inputs;
            targets = mixed.targets;
        }
        if let Some(cutmix) = self.cutmix.as_ref() {
            let seed = self.cutmix_seed
                ^ (self.emitted_batches as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
            let mixed = apply_cutmix_batch(&inputs, &targets, cutmix, seed).ok()?;
            inputs = mixed.inputs;
            targets = mixed.targets;
        }
        self.emitted_batches = self.emitted_batches.saturating_add(1);
        Some(Batch { inputs, targets })
    }
}
