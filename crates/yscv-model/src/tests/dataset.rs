use yscv_tensor::Tensor;

use crate::{
    BatchIterOptions, CutMixConfig, ImageAugmentationOp, ImageAugmentationPipeline,
    ImageFolderTargetMode, MixUpConfig, ModelError, SamplingPolicy, SupervisedCsvConfig,
    SupervisedDataset, SupervisedImageFolderConfig, SupervisedImageManifestConfig,
    SupervisedJsonlConfig, load_supervised_dataset_csv_file, load_supervised_dataset_jsonl_file,
    load_supervised_image_folder_dataset, load_supervised_image_folder_dataset_with_classes,
    load_supervised_image_manifest_csv_file, parse_supervised_dataset_csv,
    parse_supervised_dataset_jsonl, parse_supervised_image_manifest_csv,
};

use super::{
    assert_slice_approx_eq, unique_temp_path, unique_temp_path_with_extension, write_solid_rgb_png,
    write_test_rgb_png,
};

#[test]
fn dataset_batches_are_deterministic() {
    let dataset = SupervisedDataset::new(
        Tensor::from_vec(vec![5, 1], vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap(),
        Tensor::from_vec(vec![5, 1], vec![10.0, 20.0, 30.0, 40.0, 50.0]).unwrap(),
    )
    .unwrap();

    let batches = dataset.batches(2).unwrap().collect::<Vec<_>>();
    assert_eq!(batches.len(), 3);
    assert_eq!(batches[0].inputs.data(), &[1.0, 2.0]);
    assert_eq!(batches[1].inputs.data(), &[3.0, 4.0]);
    assert_eq!(batches[2].inputs.data(), &[5.0]);
}

#[test]
fn dataset_batches_with_shuffle_are_seed_deterministic() {
    let dataset = SupervisedDataset::new(
        Tensor::from_vec(vec![6, 1], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap(),
        Tensor::from_vec(vec![6, 1], vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0]).unwrap(),
    )
    .unwrap();
    let options = BatchIterOptions {
        shuffle: true,
        shuffle_seed: 42,
        drop_last: false,
        augmentation: None,
        augmentation_seed: 0,
        mixup: None,
        mixup_seed: 0,
        cutmix: None,
        cutmix_seed: 0,
        sampling: None,
    };

    let a = dataset
        .batches_with_options(2, options.clone())
        .unwrap()
        .flat_map(|batch| batch.inputs.data().to_vec())
        .collect::<Vec<_>>();
    let b = dataset
        .batches_with_options(2, options)
        .unwrap()
        .flat_map(|batch| batch.inputs.data().to_vec())
        .collect::<Vec<_>>();
    let c = dataset
        .batches_with_options(
            2,
            BatchIterOptions {
                shuffle: true,
                shuffle_seed: 7,
                drop_last: false,
                augmentation: None,
                augmentation_seed: 0,
                mixup: None,
                mixup_seed: 0,
                cutmix: None,
                cutmix_seed: 0,
                sampling: None,
            },
        )
        .unwrap()
        .flat_map(|batch| batch.inputs.data().to_vec())
        .collect::<Vec<_>>();

    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn dataset_batches_with_drop_last_skips_partial_tail() {
    let dataset = SupervisedDataset::new(
        Tensor::from_vec(vec![5, 1], vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap(),
        Tensor::from_vec(vec![5, 1], vec![10.0, 11.0, 12.0, 13.0, 14.0]).unwrap(),
    )
    .unwrap();
    let batches = dataset
        .batches_with_options(
            2,
            BatchIterOptions {
                drop_last: true,
                ..BatchIterOptions::default()
            },
        )
        .unwrap()
        .collect::<Vec<_>>();
    assert_eq!(batches.len(), 2);
    assert_eq!(batches[0].inputs.data(), &[0.0, 1.0]);
    assert_eq!(batches[1].inputs.data(), &[2.0, 3.0]);
}

#[test]
fn dataset_batches_with_sampling_policy_shuffled_is_seed_deterministic() {
    let dataset = SupervisedDataset::new(
        Tensor::from_vec(vec![6, 1], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap(),
        Tensor::from_vec(vec![6, 1], vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0]).unwrap(),
    )
    .unwrap();

    let options = BatchIterOptions {
        shuffle: false,
        shuffle_seed: 0,
        drop_last: false,
        augmentation: None,
        augmentation_seed: 0,
        mixup: None,
        mixup_seed: 0,
        cutmix: None,
        cutmix_seed: 0,
        sampling: Some(SamplingPolicy::Shuffled { seed: 123 }),
    };
    let a = dataset
        .batches_with_options(2, options.clone())
        .unwrap()
        .flat_map(|batch| batch.inputs.data().to_vec())
        .collect::<Vec<_>>();
    let b = dataset
        .batches_with_options(2, options)
        .unwrap()
        .flat_map(|batch| batch.inputs.data().to_vec())
        .collect::<Vec<_>>();
    assert_eq!(a, b);
}

#[test]
fn dataset_batches_with_weighted_sampling_replacement_supports_hot_class() {
    let dataset = SupervisedDataset::new(
        Tensor::from_vec(vec![4, 1], vec![0.0, 1.0, 2.0, 3.0]).unwrap(),
        Tensor::from_vec(vec![4, 1], vec![10.0, 11.0, 12.0, 13.0]).unwrap(),
    )
    .unwrap();

    let batches = dataset
        .batches_with_options(
            2,
            BatchIterOptions {
                sampling: Some(SamplingPolicy::Weighted {
                    weights: vec![1.0, 0.0, 0.0, 0.0],
                    seed: 9,
                    with_replacement: true,
                }),
                ..BatchIterOptions::default()
            },
        )
        .unwrap()
        .collect::<Vec<_>>();
    let all_inputs = batches
        .iter()
        .flat_map(|batch| batch.inputs.data().iter().copied())
        .collect::<Vec<_>>();
    assert_eq!(all_inputs, vec![0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn dataset_batches_with_weighted_sampling_without_replacement_is_deterministic() {
    let dataset = SupervisedDataset::new(
        Tensor::from_vec(vec![4, 1], vec![0.0, 1.0, 2.0, 3.0]).unwrap(),
        Tensor::from_vec(vec![4, 1], vec![10.0, 11.0, 12.0, 13.0]).unwrap(),
    )
    .unwrap();
    let options = BatchIterOptions {
        sampling: Some(SamplingPolicy::Weighted {
            weights: vec![1.0, 0.0, 0.0, 0.0],
            seed: 5,
            with_replacement: false,
        }),
        ..BatchIterOptions::default()
    };

    let a = dataset
        .batches_with_options(2, options.clone())
        .unwrap()
        .flat_map(|batch| batch.inputs.data().to_vec())
        .collect::<Vec<_>>();
    let b = dataset
        .batches_with_options(2, options)
        .unwrap()
        .flat_map(|batch| batch.inputs.data().to_vec())
        .collect::<Vec<_>>();
    assert_eq!(a, b);
    assert_eq!(a.len(), 4);
    assert_eq!(a[0], 0.0);
}

#[test]
fn dataset_batches_with_weighted_sampling_reject_invalid_weights() {
    let dataset = SupervisedDataset::new(
        Tensor::from_vec(vec![3, 1], vec![0.0, 1.0, 2.0]).unwrap(),
        Tensor::from_vec(vec![3, 1], vec![10.0, 11.0, 12.0]).unwrap(),
    )
    .unwrap();

    let err = dataset
        .batches_with_options(
            2,
            BatchIterOptions {
                sampling: Some(SamplingPolicy::Weighted {
                    weights: vec![1.0, 2.0],
                    seed: 0,
                    with_replacement: true,
                }),
                ..BatchIterOptions::default()
            },
        )
        .unwrap_err();
    assert_eq!(
        err,
        ModelError::InvalidSamplingWeightsLength {
            expected: 3,
            got: 2
        }
    );

    let err = dataset
        .batches_with_options(
            2,
            BatchIterOptions {
                sampling: Some(SamplingPolicy::Weighted {
                    weights: vec![1.0, -1.0, 0.0],
                    seed: 0,
                    with_replacement: true,
                }),
                ..BatchIterOptions::default()
            },
        )
        .unwrap_err();
    assert_eq!(
        err,
        ModelError::InvalidSamplingWeight {
            index: 1,
            value: -1.0
        }
    );

    let err = dataset
        .batches_with_options(
            2,
            BatchIterOptions {
                sampling: Some(SamplingPolicy::Weighted {
                    weights: vec![0.0, 0.0, 0.0],
                    seed: 0,
                    with_replacement: true,
                }),
                ..BatchIterOptions::default()
            },
        )
        .unwrap_err();
    assert_eq!(err, ModelError::InvalidSamplingDistribution);
}

#[test]
fn dataset_batches_with_balanced_class_sampling_replacement_is_seed_deterministic() {
    let inputs = (0..20).map(|value| value as f32).collect::<Vec<_>>();
    let mut labels = vec![0.0f32; 20];
    labels[19] = 1.0;
    let dataset = SupervisedDataset::new(
        Tensor::from_vec(vec![20, 1], inputs).unwrap(),
        Tensor::from_vec(vec![20, 1], labels).unwrap(),
    )
    .unwrap();

    let options = BatchIterOptions {
        sampling: Some(SamplingPolicy::BalancedByClass {
            seed: 123,
            with_replacement: true,
        }),
        ..BatchIterOptions::default()
    };
    let a = dataset
        .batches_with_options(4, options.clone())
        .unwrap()
        .flat_map(|batch| batch.inputs.data().to_vec())
        .collect::<Vec<_>>();
    let b = dataset
        .batches_with_options(4, options)
        .unwrap()
        .flat_map(|batch| batch.inputs.data().to_vec())
        .collect::<Vec<_>>();
    assert_eq!(a, b);
    let minority_count = a
        .iter()
        .filter(|index| (**index - 19.0).abs() < 1e-6)
        .count();
    assert!(minority_count >= 2);
}

#[test]
fn dataset_batches_with_balanced_class_sampling_without_replacement_returns_permutation() {
    let dataset = SupervisedDataset::new(
        Tensor::from_vec(vec![6, 1], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap(),
        Tensor::from_vec(vec![6, 1], vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).unwrap(),
    )
    .unwrap();

    let options = BatchIterOptions {
        sampling: Some(SamplingPolicy::BalancedByClass {
            seed: 77,
            with_replacement: false,
        }),
        ..BatchIterOptions::default()
    };
    let a = dataset
        .batches_with_options(2, options.clone())
        .unwrap()
        .flat_map(|batch| batch.inputs.data().to_vec())
        .collect::<Vec<_>>();
    let b = dataset
        .batches_with_options(2, options)
        .unwrap()
        .flat_map(|batch| batch.inputs.data().to_vec())
        .collect::<Vec<_>>();
    assert_eq!(a, b);

    let mut sorted = a.clone();
    sorted.sort_by(f32::total_cmp);
    assert_eq!(sorted, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn dataset_batches_with_balanced_class_sampling_supports_one_hot_targets() {
    let dataset = SupervisedDataset::new(
        Tensor::from_vec(vec![6, 1], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap(),
        Tensor::from_vec(
            vec![6, 2],
            vec![
                1.0, 0.0, //
                1.0, 0.0, //
                1.0, 0.0, //
                1.0, 0.0, //
                1.0, 0.0, //
                0.0, 1.0, //
            ],
        )
        .unwrap(),
    )
    .unwrap();

    let options = BatchIterOptions {
        sampling: Some(SamplingPolicy::BalancedByClass {
            seed: 2026,
            with_replacement: true,
        }),
        ..BatchIterOptions::default()
    };

    let a = dataset
        .batches_with_options(2, options.clone())
        .unwrap()
        .flat_map(|batch| batch.inputs.data().to_vec())
        .collect::<Vec<_>>();
    let b = dataset
        .batches_with_options(2, options)
        .unwrap()
        .flat_map(|batch| batch.inputs.data().to_vec())
        .collect::<Vec<_>>();
    assert_eq!(a, b);

    let minority_count = a
        .iter()
        .filter(|index| (**index - 5.0).abs() < 1e-6)
        .count();
    assert!(minority_count >= 2);
}

#[test]
fn dataset_batches_with_balanced_class_sampling_reject_invalid_one_hot_targets() {
    let dataset = SupervisedDataset::new(
        Tensor::from_vec(vec![3, 1], vec![0.0, 1.0, 2.0]).unwrap(),
        Tensor::from_vec(vec![3, 2], vec![1.0, 0.0, 0.4, 0.6, 0.0, 1.0]).unwrap(),
    )
    .unwrap();

    let err = dataset
        .batches_with_options(
            2,
            BatchIterOptions {
                sampling: Some(SamplingPolicy::BalancedByClass {
                    seed: 0,
                    with_replacement: true,
                }),
                ..BatchIterOptions::default()
            },
        )
        .unwrap_err();
    assert_eq!(
        err,
        ModelError::InvalidClassSamplingTargetValue {
            index: 1,
            value: 0.4,
            reason: "one-hot value must be close to 0 or 1",
        }
    );
}

#[test]
fn dataset_batches_with_balanced_class_sampling_reject_invalid_class_label() {
    let dataset = SupervisedDataset::new(
        Tensor::from_vec(vec![3, 1], vec![0.0, 1.0, 2.0]).unwrap(),
        Tensor::from_vec(vec![3, 1], vec![0.0, 1.5, 1.0]).unwrap(),
    )
    .unwrap();

    let err = dataset
        .batches_with_options(
            2,
            BatchIterOptions {
                sampling: Some(SamplingPolicy::BalancedByClass {
                    seed: 0,
                    with_replacement: false,
                }),
                ..BatchIterOptions::default()
            },
        )
        .unwrap_err();
    assert_eq!(
        err,
        ModelError::InvalidClassSamplingTargetValue {
            index: 1,
            value: 1.5,
            reason: "class label must be an integer value",
        }
    );
}

#[test]
fn mixup_config_rejects_invalid_arguments() {
    let lambda_err = MixUpConfig::new().with_lambda_min(0.6).unwrap_err();
    assert_eq!(
        lambda_err,
        ModelError::InvalidMixupArgument {
            field: "lambda_min",
            value: 0.6,
            message: "lambda_min must be finite and in [0, 0.5]".to_string()
        }
    );

    let probability_err = MixUpConfig::new().with_probability(1.2).unwrap_err();
    assert_eq!(
        probability_err,
        ModelError::InvalidMixupArgument {
            field: "probability",
            value: 1.2,
            message: "probability must be finite and in [0, 1]".to_string()
        }
    );
}

#[test]
fn dataset_batches_with_mixup_are_seed_deterministic() {
    let dataset = SupervisedDataset::new(
        Tensor::from_vec(vec![4, 1], vec![0.0, 1.0, 2.0, 3.0]).unwrap(),
        Tensor::from_vec(vec![4, 1], vec![10.0, 20.0, 30.0, 40.0]).unwrap(),
    )
    .unwrap();
    let mixup = MixUpConfig::new().with_lambda_min(0.2).unwrap();

    let options = BatchIterOptions {
        mixup: Some(mixup.clone()),
        mixup_seed: 17,
        ..BatchIterOptions::default()
    };
    let a = dataset
        .batches_with_options(4, options.clone())
        .unwrap()
        .next()
        .unwrap();
    let b = dataset
        .batches_with_options(4, options)
        .unwrap()
        .next()
        .unwrap();
    let c = dataset
        .batches_with_options(
            4,
            BatchIterOptions {
                mixup: Some(mixup),
                mixup_seed: 18,
                ..BatchIterOptions::default()
            },
        )
        .unwrap()
        .next()
        .unwrap();

    assert_eq!(a.inputs, b.inputs);
    assert_eq!(a.targets, b.targets);
    assert_ne!(a.inputs, c.inputs);
    assert_ne!(a.targets, c.targets);
    assert_eq!(a.inputs.shape(), &[4, 1]);
    assert_eq!(a.targets.shape(), &[4, 1]);
}

#[test]
fn dataset_batches_with_mixup_probability_zero_leaves_batch_unchanged() {
    let dataset = SupervisedDataset::new(
        Tensor::from_vec(vec![4, 1], vec![0.0, 1.0, 2.0, 3.0]).unwrap(),
        Tensor::from_vec(vec![4, 1], vec![10.0, 20.0, 30.0, 40.0]).unwrap(),
    )
    .unwrap();
    let mixup = MixUpConfig::new().with_probability(0.0).unwrap();

    let batch = dataset
        .batches_with_options(
            4,
            BatchIterOptions {
                mixup: Some(mixup),
                mixup_seed: 5,
                ..BatchIterOptions::default()
            },
        )
        .unwrap()
        .next()
        .unwrap();

    assert_eq!(batch.inputs.data(), &[0.0, 1.0, 2.0, 3.0]);
    assert_eq!(batch.targets.data(), &[10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn cutmix_config_rejects_invalid_arguments() {
    let probability_err = CutMixConfig::new().with_probability(-0.1).unwrap_err();
    assert_eq!(
        probability_err,
        ModelError::InvalidCutMixArgument {
            field: "probability",
            value: -0.1,
            message: "probability must be finite and in [0, 1]".to_string()
        }
    );

    let min_err = CutMixConfig::new()
        .with_min_patch_fraction(1.1)
        .unwrap_err();
    assert_eq!(
        min_err,
        ModelError::InvalidCutMixArgument {
            field: "min_patch_fraction",
            value: 1.1,
            message: "min_patch_fraction must be finite and in [0, 1]".to_string()
        }
    );

    let max_err = CutMixConfig::new()
        .with_max_patch_fraction(0.2)
        .unwrap()
        .with_min_patch_fraction(0.8)
        .unwrap_err();
    assert_eq!(
        max_err,
        ModelError::InvalidCutMixArgument {
            field: "min_patch_fraction",
            value: 0.8,
            message: "min_patch_fraction must be <= max_patch_fraction (0.2)".to_string()
        }
    );
}

#[test]
fn dataset_batches_with_cutmix_full_patch_is_seed_deterministic() {
    let dataset = SupervisedDataset::new(
        Tensor::from_vec(vec![4, 2, 2, 1], (0..16).map(|v| v as f32).collect()).unwrap(),
        Tensor::from_vec(vec![4, 1], vec![10.0, 20.0, 30.0, 40.0]).unwrap(),
    )
    .unwrap();
    let cutmix = CutMixConfig::new()
        .with_max_patch_fraction(1.0)
        .unwrap()
        .with_min_patch_fraction(1.0)
        .unwrap();

    let a = dataset
        .batches_with_options(
            4,
            BatchIterOptions {
                cutmix: Some(cutmix.clone()),
                cutmix_seed: 9,
                ..BatchIterOptions::default()
            },
        )
        .unwrap()
        .next()
        .unwrap();
    let b = dataset
        .batches_with_options(
            4,
            BatchIterOptions {
                cutmix: Some(cutmix.clone()),
                cutmix_seed: 9,
                ..BatchIterOptions::default()
            },
        )
        .unwrap()
        .next()
        .unwrap();
    let c = dataset
        .batches_with_options(
            4,
            BatchIterOptions {
                cutmix: Some(cutmix),
                cutmix_seed: 10,
                ..BatchIterOptions::default()
            },
        )
        .unwrap()
        .next()
        .unwrap();

    assert_eq!(a.inputs, b.inputs);
    assert_eq!(a.targets, b.targets);
    assert_ne!(a.inputs, c.inputs);
    assert_ne!(a.targets, c.targets);
    assert_ne!(a.inputs.data(), dataset.inputs().data());
    assert_ne!(a.targets.data(), dataset.targets().data());
}

#[test]
fn dataset_batches_with_cutmix_probability_zero_leaves_batch_unchanged() {
    let dataset = SupervisedDataset::new(
        Tensor::from_vec(vec![4, 2, 2, 1], (0..16).map(|v| v as f32).collect()).unwrap(),
        Tensor::from_vec(vec![4, 1], vec![10.0, 20.0, 30.0, 40.0]).unwrap(),
    )
    .unwrap();
    let cutmix = CutMixConfig::new().with_probability(0.0).unwrap();

    let batch = dataset
        .batches_with_options(
            4,
            BatchIterOptions {
                cutmix: Some(cutmix),
                cutmix_seed: 2,
                ..BatchIterOptions::default()
            },
        )
        .unwrap()
        .next()
        .unwrap();

    assert_eq!(batch.inputs.data(), dataset.inputs().data());
    assert_eq!(batch.targets.data(), dataset.targets().data());
}

#[test]
fn dataset_batches_with_cutmix_reject_non_nhwc_inputs() {
    let dataset = SupervisedDataset::new(
        Tensor::from_vec(vec![4, 2], vec![0.0; 8]).unwrap(),
        Tensor::from_vec(vec![4, 1], vec![0.0; 4]).unwrap(),
    )
    .unwrap();
    let err = dataset
        .batches_with_options(
            4,
            BatchIterOptions {
                cutmix: Some(CutMixConfig::new()),
                cutmix_seed: 1,
                ..BatchIterOptions::default()
            },
        )
        .unwrap_err();
    assert_eq!(err, ModelError::InvalidCutMixInputShape { got: vec![4, 2] });
}

#[test]
fn dataset_batches_with_augmentation_apply_pipeline() {
    let dataset = SupervisedDataset::new(
        Tensor::from_vec(vec![1, 1, 2, 1], vec![0.0, 1.0]).unwrap(),
        Tensor::from_vec(vec![1, 1], vec![5.0]).unwrap(),
    )
    .unwrap();
    let pipeline = ImageAugmentationPipeline::new(vec![ImageAugmentationOp::HorizontalFlip {
        probability: 1.0,
    }])
    .unwrap();
    let options = BatchIterOptions {
        augmentation: Some(pipeline),
        ..BatchIterOptions::default()
    };
    let batches = dataset
        .batches_with_options(1, options)
        .unwrap()
        .collect::<Vec<_>>();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].inputs.shape(), &[1, 1, 2, 1]);
    assert_eq!(batches[0].inputs.data(), &[1.0, 0.0]);
    assert_eq!(batches[0].targets.data(), &[5.0]);
}

#[test]
fn dataset_split_by_ratio_produces_expected_lengths() {
    let dataset = SupervisedDataset::new(
        Tensor::from_vec(vec![10, 1], (0..10).map(|v| v as f32).collect()).unwrap(),
        Tensor::from_vec(vec![10, 1], (100..110).map(|v| v as f32).collect()).unwrap(),
    )
    .unwrap();

    let split = dataset.split_by_ratio(0.6, 0.2, false, 0).unwrap();
    assert_eq!(split.train.len(), 6);
    assert_eq!(split.validation.len(), 2);
    assert_eq!(split.test.len(), 2);
    assert_eq!(split.train.inputs().data(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(split.validation.inputs().data(), &[6.0, 7.0]);
    assert_eq!(split.test.inputs().data(), &[8.0, 9.0]);
}

#[test]
fn dataset_split_by_counts_rejects_invalid_counts() {
    let dataset = SupervisedDataset::new(
        Tensor::from_vec(vec![3, 1], vec![1.0, 2.0, 3.0]).unwrap(),
        Tensor::from_vec(vec![3, 1], vec![4.0, 5.0, 6.0]).unwrap(),
    )
    .unwrap();

    let err = dataset.split_by_counts(2, 2, false, 0).unwrap_err();
    assert_eq!(
        err,
        ModelError::InvalidSplitCounts {
            train_count: 2,
            validation_count: 2,
            dataset_len: 3
        }
    );
}

#[test]
fn dataset_split_by_class_ratio_preserves_class_distribution_for_scalar_targets() {
    let dataset = SupervisedDataset::new(
        Tensor::from_vec(vec![10, 1], (0..10).map(|v| v as f32).collect()).unwrap(),
        Tensor::from_vec(
            vec![10, 1],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        )
        .unwrap(),
    )
    .unwrap();

    let split = dataset.split_by_class_ratio(0.5, 0.25, false, 123).unwrap();
    assert_eq!(split.train.len(), 5);
    assert_eq!(split.validation.len(), 2);
    assert_eq!(split.test.len(), 3);

    let train_targets = split.train.targets().data();
    let validation_targets = split.validation.targets().data();
    let test_targets = split.test.targets().data();

    let train_class_0 = train_targets.iter().filter(|value| **value == 0.0).count();
    let train_class_1 = train_targets.iter().filter(|value| **value == 1.0).count();
    let validation_class_0 = validation_targets
        .iter()
        .filter(|value| **value == 0.0)
        .count();
    let validation_class_1 = validation_targets
        .iter()
        .filter(|value| **value == 1.0)
        .count();
    let test_class_0 = test_targets.iter().filter(|value| **value == 0.0).count();
    let test_class_1 = test_targets.iter().filter(|value| **value == 1.0).count();

    assert_eq!(train_class_0, 4);
    assert_eq!(train_class_1, 1);
    assert_eq!(validation_class_0, 2);
    assert_eq!(validation_class_1, 0);
    assert_eq!(test_class_0, 2);
    assert_eq!(test_class_1, 1);
}

#[test]
fn dataset_split_by_class_ratio_supports_one_hot_targets() {
    let dataset = SupervisedDataset::new(
        Tensor::from_vec(vec![6, 1], (0..6).map(|v| v as f32).collect()).unwrap(),
        Tensor::from_vec(
            vec![6, 2],
            vec![
                1.0, 0.0, //
                1.0, 0.0, //
                1.0, 0.0, //
                1.0, 0.0, //
                0.0, 1.0, //
                0.0, 1.0, //
            ],
        )
        .unwrap(),
    )
    .unwrap();

    let split = dataset.split_by_class_ratio(0.5, 0.0, false, 7).unwrap();
    assert_eq!(split.train.len(), 3);
    assert_eq!(split.validation.len(), 0);
    assert_eq!(split.test.len(), 3);

    let train_targets = split
        .train
        .targets()
        .data()
        .as_chunks::<2>()
        .0
        .iter()
        .collect::<Vec<_>>();
    let test_targets = split
        .test
        .targets()
        .data()
        .as_chunks::<2>()
        .0
        .iter()
        .collect::<Vec<_>>();
    let train_class_1 = train_targets
        .iter()
        .filter(|row| row[0] == 0.0 && row[1] == 1.0)
        .count();
    let test_class_1 = test_targets
        .iter()
        .filter(|row| row[0] == 0.0 && row[1] == 1.0)
        .count();
    assert_eq!(train_class_1, 1);
    assert_eq!(test_class_1, 1);
}

#[test]
fn dataset_split_by_class_ratio_rejects_invalid_one_hot_targets() {
    let dataset = SupervisedDataset::new(
        Tensor::from_vec(vec![3, 1], vec![0.0, 1.0, 2.0]).unwrap(),
        Tensor::from_vec(vec![3, 2], vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0]).unwrap(),
    )
    .unwrap();

    let err = dataset
        .split_by_class_ratio(0.6, 0.2, true, 999)
        .unwrap_err();
    assert_eq!(
        err,
        ModelError::InvalidClassSamplingTargetValue {
            index: 1,
            value: 1.0,
            reason: "one-hot row must contain exactly one active class",
        }
    );
}

#[test]
fn dataset_rejects_invalid_shapes() {
    let err = SupervisedDataset::new(
        Tensor::from_vec(vec![2, 1], vec![1.0, 2.0]).unwrap(),
        Tensor::from_vec(vec![3, 1], vec![1.0, 2.0, 3.0]).unwrap(),
    )
    .unwrap_err();
    assert_eq!(
        err,
        ModelError::DatasetShapeMismatch {
            inputs: vec![2, 1],
            targets: vec![3, 1]
        }
    );
}

#[test]
fn dataset_batches_with_augmentation_reject_non_nhwc_inputs() {
    let dataset = SupervisedDataset::new(
        Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        Tensor::from_vec(vec![2, 1], vec![0.0, 1.0]).unwrap(),
    )
    .unwrap();
    let pipeline = ImageAugmentationPipeline::new(vec![ImageAugmentationOp::HorizontalFlip {
        probability: 1.0,
    }])
    .unwrap();
    let err = dataset
        .batches_with_options(
            1,
            BatchIterOptions {
                augmentation: Some(pipeline),
                ..BatchIterOptions::default()
            },
        )
        .unwrap_err();
    assert_eq!(
        err,
        ModelError::InvalidAugmentationInputShape { got: vec![2, 2] }
    );
}

// ── JSONL dataset tests ─────────────────────────────────────────────

#[test]
fn parse_supervised_dataset_jsonl_builds_expected_dataset() {
    let config = SupervisedJsonlConfig::new(vec![2], vec![1]).unwrap();
    let content = r#"
# comment
{"input":[1.0,2.0],"target":[0.0]}
{"inputs":[3.0,4.0],"targets":[1.0]}
"#;

    let dataset = parse_supervised_dataset_jsonl(content, &config).unwrap();
    assert_eq!(dataset.inputs().shape(), &[2, 2]);
    assert_eq!(dataset.targets().shape(), &[2, 1]);
    assert_eq!(dataset.inputs().data(), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(dataset.targets().data(), &[0.0, 1.0]);
}

#[test]
fn parse_supervised_dataset_jsonl_supports_field_aliases() {
    let config = SupervisedJsonlConfig::new(vec![2], vec![1]).unwrap();
    let content = r#"
{"inputs":[1.0,2.0],"label":[0.0]}
{"features":[3.0,4.0],"labels":[1.0]}
"#;

    let dataset = parse_supervised_dataset_jsonl(content, &config).unwrap();
    assert_eq!(dataset.inputs().shape(), &[2, 2]);
    assert_eq!(dataset.targets().shape(), &[2, 1]);
    assert_eq!(dataset.inputs().data(), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(dataset.targets().data(), &[0.0, 1.0]);
}

#[test]
fn parse_supervised_dataset_jsonl_supports_scalar_target_for_single_value_shape() {
    let config = SupervisedJsonlConfig::new(vec![2], vec![1]).unwrap();
    let content = r#"
{"features":[1.0,2.0],"label":0.0}
{"input":[3.0,4.0],"target":1.0}
"#;

    let dataset = parse_supervised_dataset_jsonl(content, &config).unwrap();
    assert_eq!(dataset.inputs().shape(), &[2, 2]);
    assert_eq!(dataset.targets().shape(), &[2, 1]);
    assert_eq!(dataset.inputs().data(), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(dataset.targets().data(), &[0.0, 1.0]);
}

#[test]
fn parse_supervised_dataset_jsonl_rejects_scalar_target_for_multi_value_shape() {
    let config = SupervisedJsonlConfig::new(vec![1], vec![2]).unwrap();
    let content = r#"{"input":[1.0],"target":0.0}"#;

    let err = parse_supervised_dataset_jsonl(content, &config).unwrap_err();
    assert_eq!(
        err,
        ModelError::InvalidDatasetRecordLength {
            line: 1,
            field: "target",
            expected: 2,
            got: 1,
        }
    );
}

#[test]
fn parse_supervised_dataset_jsonl_rejects_invalid_json() {
    let config = SupervisedJsonlConfig::new(vec![1], vec![1]).unwrap();
    let content = "{\"input\":[1.0],\"target\":[0.0]}\n{bad-json}";
    let err = parse_supervised_dataset_jsonl(content, &config).unwrap_err();
    assert!(matches!(
        err,
        ModelError::DatasetJsonlParse {
            line: 2,
            message: _
        }
    ));
}

#[test]
fn parse_supervised_dataset_jsonl_rejects_invalid_record_length() {
    let config = SupervisedJsonlConfig::new(vec![2], vec![1]).unwrap();
    let content = "{\"input\":[1.0],\"target\":[0.0]}";
    let err = parse_supervised_dataset_jsonl(content, &config).unwrap_err();
    assert_eq!(
        err,
        ModelError::InvalidDatasetRecordLength {
            line: 1,
            field: "input",
            expected: 2,
            got: 1
        }
    );
}

#[test]
fn parse_supervised_dataset_jsonl_rejects_infinite_values() {
    let config = SupervisedJsonlConfig::new(vec![1], vec![1]).unwrap();
    let content = "{\"input\":[1e39],\"target\":[0.0]}";
    let err = parse_supervised_dataset_jsonl(content, &config).unwrap_err();
    assert_eq!(
        err,
        ModelError::InvalidDatasetRecordValue {
            line: 1,
            field: "input",
            index: 0,
            reason: "infinite value is not allowed"
        }
    );
}

#[test]
fn load_supervised_dataset_jsonl_file_reads_from_disk() {
    let config = SupervisedJsonlConfig::new(vec![1], vec![1]).unwrap();
    let path = unique_temp_path("yscv-model-jsonl-load");
    std::fs::write(&path, "{\"input\":[2.0],\"target\":[3.0]}\n").unwrap();

    let loaded = load_supervised_dataset_jsonl_file(&path, &config).unwrap();
    assert_eq!(loaded.inputs().shape(), &[1, 1]);
    assert_eq!(loaded.targets().shape(), &[1, 1]);
    assert_eq!(loaded.inputs().data(), &[2.0]);
    assert_eq!(loaded.targets().data(), &[3.0]);

    let _ = std::fs::remove_file(path);
}

#[test]
fn load_supervised_dataset_jsonl_file_reports_missing_file() {
    let config = SupervisedJsonlConfig::new(vec![1], vec![1]).unwrap();
    let path = unique_temp_path("yscv-model-jsonl-missing");
    let err = load_supervised_dataset_jsonl_file(&path, &config).unwrap_err();
    assert!(matches!(
        err,
        ModelError::DatasetLoadIo {
            path: _,
            message: _
        }
    ));
}

#[test]
fn supervised_jsonl_config_rejects_zero_dimensions() {
    let err = SupervisedJsonlConfig::new(vec![0, 2], vec![1]).unwrap_err();
    assert_eq!(
        err,
        ModelError::InvalidDatasetAdapterShape {
            field: "input_shape",
            shape: vec![0, 2],
            message: "dimension at index 0 must be > 0".to_string()
        }
    );
}

// ── CSV dataset tests ───────────────────────────────────────────────

#[test]
fn parse_supervised_dataset_csv_builds_expected_dataset() {
    let config = SupervisedCsvConfig::new(vec![2], vec![1])
        .unwrap()
        .with_header(true);
    let content = r#"
# comment
f0,f1,target
1.0,2.0,0.0
3.0,4.0,1.0
"#;

    let dataset = parse_supervised_dataset_csv(content, &config).unwrap();
    assert_eq!(dataset.inputs().shape(), &[2, 2]);
    assert_eq!(dataset.targets().shape(), &[2, 1]);
    assert_eq!(dataset.inputs().data(), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(dataset.targets().data(), &[0.0, 1.0]);
}

#[test]
fn parse_supervised_dataset_csv_supports_custom_delimiter() {
    let config = SupervisedCsvConfig::new(vec![1], vec![1])
        .unwrap()
        .with_delimiter(';')
        .unwrap();
    let content = "2.0;3.0";
    let dataset = parse_supervised_dataset_csv(content, &config).unwrap();
    assert_eq!(dataset.inputs().data(), &[2.0]);
    assert_eq!(dataset.targets().data(), &[3.0]);
}

#[test]
fn parse_supervised_dataset_csv_rejects_invalid_column_count() {
    let config = SupervisedCsvConfig::new(vec![2], vec![1]).unwrap();
    let err = parse_supervised_dataset_csv("1.0,2.0", &config).unwrap_err();
    assert_eq!(
        err,
        ModelError::InvalidDatasetRecordColumns {
            line: 1,
            expected: 3,
            got: 2
        }
    );
}

#[test]
fn parse_supervised_dataset_csv_rejects_invalid_numeric_value() {
    let config = SupervisedCsvConfig::new(vec![1], vec![1]).unwrap();
    let err = parse_supervised_dataset_csv("abc,1.0", &config).unwrap_err();
    assert!(matches!(
        err,
        ModelError::DatasetCsvParse {
            line: 1,
            column: 1,
            message: _
        }
    ));
}

#[test]
fn load_supervised_dataset_csv_file_reads_from_disk() {
    let config = SupervisedCsvConfig::new(vec![1], vec![1]).unwrap();
    let path = unique_temp_path("yscv-model-csv-load");
    std::fs::write(&path, "2.0,3.0\n").unwrap();

    let loaded = load_supervised_dataset_csv_file(&path, &config).unwrap();
    assert_eq!(loaded.inputs().shape(), &[1, 1]);
    assert_eq!(loaded.targets().shape(), &[1, 1]);
    assert_eq!(loaded.inputs().data(), &[2.0]);
    assert_eq!(loaded.targets().data(), &[3.0]);

    let _ = std::fs::remove_file(path);
}

#[test]
fn load_supervised_dataset_csv_file_reports_missing_file() {
    let config = SupervisedCsvConfig::new(vec![1], vec![1]).unwrap();
    let path = unique_temp_path("yscv-model-csv-missing");
    let err = load_supervised_dataset_csv_file(&path, &config).unwrap_err();
    assert!(matches!(
        err,
        ModelError::DatasetLoadIo {
            path: _,
            message: _
        }
    ));
}

#[test]
fn supervised_csv_config_rejects_control_delimiter() {
    let err = SupervisedCsvConfig::new(vec![1], vec![1])
        .unwrap()
        .with_delimiter('\n')
        .unwrap_err();
    assert_eq!(err, ModelError::InvalidCsvDelimiter { delimiter: '\n' });
}

// ── Image manifest/folder dataset tests ─────────────────────────────

#[test]
fn parse_supervised_image_manifest_csv_builds_expected_dataset() {
    let temp_root = unique_temp_path_with_extension("yscv-model-image-manifest-parse", "d");
    std::fs::create_dir_all(&temp_root).unwrap();
    let image_path = temp_root.join("sample.png");
    write_test_rgb_png(&image_path);

    let config = SupervisedImageManifestConfig::new(vec![1], 2, 2)
        .unwrap()
        .with_image_root(&temp_root)
        .with_header(true);
    let content = "image,target\nsample.png,1.0\n";

    let dataset = parse_supervised_image_manifest_csv(content, &config).unwrap();
    assert_eq!(dataset.inputs().shape(), &[1, 2, 2, 3]);
    assert_eq!(dataset.targets().shape(), &[1, 1]);
    assert_slice_approx_eq(
        &dataset.inputs().data()[..6],
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        1e-6,
    );
    assert_eq!(dataset.targets().data(), &[1.0]);

    let _ = std::fs::remove_file(&image_path);
    let _ = std::fs::remove_dir(&temp_root);
}

#[test]
fn parse_supervised_image_manifest_csv_resizes_to_configured_shape() {
    let temp_root = unique_temp_path_with_extension("yscv-model-image-manifest-resize", "d");
    std::fs::create_dir_all(&temp_root).unwrap();
    let image_path = temp_root.join("sample.png");
    write_test_rgb_png(&image_path);

    let config = SupervisedImageManifestConfig::new(vec![1], 1, 1)
        .unwrap()
        .with_image_root(&temp_root);
    let content = "sample.png,0.0\n";
    let dataset = parse_supervised_image_manifest_csv(content, &config).unwrap();
    assert_eq!(dataset.inputs().shape(), &[1, 1, 1, 3]);
    assert_slice_approx_eq(dataset.inputs().data(), &[1.0, 0.0, 0.0], 1e-6);

    let _ = std::fs::remove_file(&image_path);
    let _ = std::fs::remove_dir(&temp_root);
}

#[test]
fn parse_supervised_image_manifest_csv_rejects_empty_image_path() {
    let config = SupervisedImageManifestConfig::new(vec![1], 2, 2).unwrap();
    let err = parse_supervised_image_manifest_csv(",1.0", &config).unwrap_err();
    assert_eq!(
        err,
        ModelError::InvalidDatasetRecordPath {
            line: 1,
            message: "image path is empty".to_string()
        }
    );
}

#[test]
fn parse_supervised_image_manifest_csv_reports_decode_errors() {
    let config = SupervisedImageManifestConfig::new(vec![1], 2, 2)
        .unwrap()
        .with_image_root(std::env::temp_dir());
    let err = parse_supervised_image_manifest_csv("does-not-exist.png,0.0", &config).unwrap_err();
    assert!(matches!(
        err,
        ModelError::DatasetImageDecode {
            path: _,
            message: _
        }
    ));
}

#[test]
fn load_supervised_image_manifest_csv_file_resolves_manifest_relative_root() {
    let temp_root = unique_temp_path_with_extension("yscv-model-image-manifest-load", "d");
    let images_dir = temp_root.join("images");
    std::fs::create_dir_all(&images_dir).unwrap();
    let image_path = images_dir.join("sample.png");
    write_test_rgb_png(&image_path);

    let manifest_path = temp_root.join("manifest.csv");
    std::fs::write(&manifest_path, "sample.png,1.0\n").unwrap();

    let config = SupervisedImageManifestConfig::new(vec![1], 2, 2)
        .unwrap()
        .with_image_root("images");
    let dataset = load_supervised_image_manifest_csv_file(&manifest_path, &config).unwrap();
    assert_eq!(dataset.inputs().shape(), &[1, 2, 2, 3]);
    assert_eq!(dataset.targets().shape(), &[1, 1]);
    assert_eq!(dataset.targets().data(), &[1.0]);

    let _ = std::fs::remove_file(&manifest_path);
    let _ = std::fs::remove_file(&image_path);
    let _ = std::fs::remove_dir(&images_dir);
    let _ = std::fs::remove_dir(&temp_root);
}

#[test]
fn load_supervised_image_folder_dataset_builds_expected_dataset() {
    let temp_root = unique_temp_path_with_extension("yscv-model-image-folder-load", "d");
    let cats_dir = temp_root.join("cats");
    let dogs_dir = temp_root.join("dogs");
    std::fs::create_dir_all(&cats_dir).unwrap();
    std::fs::create_dir_all(&dogs_dir).unwrap();

    let cat_image = cats_dir.join("a.png");
    let dog_image = dogs_dir.join("b.png");
    write_solid_rgb_png(&cat_image, [255, 0, 0]);
    write_solid_rgb_png(&dog_image, [0, 255, 0]);
    std::fs::write(cats_dir.join("README.txt"), "ignore me").unwrap();

    let config = SupervisedImageFolderConfig::new(2, 2).unwrap();
    let dataset = load_supervised_image_folder_dataset(&temp_root, &config).unwrap();

    assert_eq!(dataset.inputs().shape(), &[2, 2, 2, 3]);
    assert_eq!(dataset.targets().shape(), &[2, 1]);
    assert_eq!(dataset.targets().data(), &[0.0, 1.0]);
    assert_slice_approx_eq(&dataset.inputs().data()[0..3], &[1.0, 0.0, 0.0], 1e-6);
    assert_slice_approx_eq(&dataset.inputs().data()[12..15], &[0.0, 1.0, 0.0], 1e-6);

    let _ = std::fs::remove_file(cats_dir.join("README.txt"));
    let _ = std::fs::remove_file(&cat_image);
    let _ = std::fs::remove_file(&dog_image);
    let _ = std::fs::remove_dir(&cats_dir);
    let _ = std::fs::remove_dir(&dogs_dir);
    let _ = std::fs::remove_dir(&temp_root);
}

#[test]
fn load_supervised_image_folder_dataset_supports_one_hot_targets() {
    let temp_root = unique_temp_path_with_extension("yscv-model-image-folder-onehot", "d");
    let cats_dir = temp_root.join("cats");
    let dogs_dir = temp_root.join("dogs");
    std::fs::create_dir_all(&cats_dir).unwrap();
    std::fs::create_dir_all(&dogs_dir).unwrap();

    let cat_image = cats_dir.join("a.png");
    let dog_image = dogs_dir.join("b.png");
    write_solid_rgb_png(&cat_image, [255, 0, 0]);
    write_solid_rgb_png(&dog_image, [0, 255, 0]);

    let config = SupervisedImageFolderConfig::new(2, 2)
        .unwrap()
        .with_target_mode(ImageFolderTargetMode::OneHot);
    let dataset = load_supervised_image_folder_dataset(&temp_root, &config).unwrap();

    assert_eq!(dataset.targets().shape(), &[2, 2]);
    assert_eq!(dataset.targets().data(), &[1.0, 0.0, 0.0, 1.0]);

    let _ = std::fs::remove_file(&cat_image);
    let _ = std::fs::remove_file(&dog_image);
    let _ = std::fs::remove_dir(&cats_dir);
    let _ = std::fs::remove_dir(&dogs_dir);
    let _ = std::fs::remove_dir(&temp_root);
}

#[test]
fn load_supervised_image_folder_dataset_with_classes_returns_stable_mapping() {
    let temp_root = unique_temp_path_with_extension("yscv-model-image-folder-classes", "d");
    let cats_dir = temp_root.join("cats");
    let dogs_dir = temp_root.join("dogs");
    std::fs::create_dir_all(&cats_dir).unwrap();
    std::fs::create_dir_all(&dogs_dir).unwrap();

    let cat_image = cats_dir.join("a.png");
    let dog_image = dogs_dir.join("b.png");
    write_solid_rgb_png(&cat_image, [255, 0, 0]);
    write_solid_rgb_png(&dog_image, [0, 255, 0]);

    let config = SupervisedImageFolderConfig::new(2, 2).unwrap();
    let loaded = load_supervised_image_folder_dataset_with_classes(&temp_root, &config).unwrap();

    assert_eq!(
        loaded.class_names,
        vec!["cats".to_string(), "dogs".to_string()]
    );
    assert_eq!(loaded.dataset.targets().shape(), &[2, 1]);
    assert_eq!(loaded.dataset.targets().data(), &[0.0, 1.0]);

    let _ = std::fs::remove_file(&cat_image);
    let _ = std::fs::remove_file(&dog_image);
    let _ = std::fs::remove_dir(&cats_dir);
    let _ = std::fs::remove_dir(&dogs_dir);
    let _ = std::fs::remove_dir(&temp_root);
}

#[test]
fn load_supervised_image_folder_dataset_supports_bmp_extension() {
    let temp_root = unique_temp_path_with_extension("yscv-model-image-folder-bmp", "d");
    let class_dir = temp_root.join("class0");
    std::fs::create_dir_all(&class_dir).unwrap();

    let bmp_image = class_dir.join("a.bmp");
    std::fs::write(&bmp_image, b"not-a-valid-bmp").unwrap();

    let config = SupervisedImageFolderConfig::new(2, 2).unwrap();
    let err = load_supervised_image_folder_dataset(&temp_root, &config).unwrap_err();
    assert!(matches!(
        err,
        ModelError::DatasetImageDecode {
            path: _,
            message: _
        }
    ));

    let _ = std::fs::remove_file(&bmp_image);
    let _ = std::fs::remove_dir(&class_dir);
    let _ = std::fs::remove_dir(&temp_root);
}

#[test]
fn supervised_image_folder_config_rejects_invalid_allowed_extensions() {
    let empty_err = SupervisedImageFolderConfig::new(2, 2)
        .unwrap()
        .with_allowed_extensions(Vec::new())
        .unwrap_err();
    assert_eq!(
        empty_err,
        ModelError::InvalidImageFolderExtension {
            extension: "<list>".to_string(),
            message: "extension list must be non-empty".to_string(),
        }
    );

    let dotted_err = SupervisedImageFolderConfig::new(2, 2)
        .unwrap()
        .with_allowed_extensions(vec![".png".to_string()])
        .unwrap_err();
    assert_eq!(
        dotted_err,
        ModelError::InvalidImageFolderExtension {
            extension: ".png".to_string(),
            message: "extension must not start with '.'".to_string(),
        }
    );
}

#[test]
fn load_supervised_image_folder_dataset_with_allowed_extensions_filters_files() {
    let temp_root = unique_temp_path_with_extension("yscv-model-image-folder-extensions", "d");
    let class_dir = temp_root.join("class0");
    std::fs::create_dir_all(&class_dir).unwrap();

    let png_image = class_dir.join("a.png");
    let bmp_image = class_dir.join("b.bmp");
    write_solid_rgb_png(&png_image, [255, 0, 0]);
    std::fs::write(&bmp_image, b"not-a-valid-bmp").unwrap();

    let config = SupervisedImageFolderConfig::new(2, 2)
        .unwrap()
        .with_allowed_extensions(vec!["png".to_string()])
        .unwrap();
    let dataset = load_supervised_image_folder_dataset(&temp_root, &config).unwrap();
    assert_eq!(dataset.inputs().shape(), &[1, 2, 2, 3]);
    assert_eq!(dataset.targets().shape(), &[1, 1]);
    assert_eq!(dataset.targets().data(), &[0.0]);

    let _ = std::fs::remove_file(&png_image);
    let _ = std::fs::remove_file(&bmp_image);
    let _ = std::fs::remove_dir(&class_dir);
    let _ = std::fs::remove_dir(&temp_root);
}

#[test]
fn load_supervised_image_folder_dataset_rejects_empty_dataset() {
    let temp_root = unique_temp_path_with_extension("yscv-model-image-folder-empty", "d");
    let class_dir = temp_root.join("class0");
    std::fs::create_dir_all(&class_dir).unwrap();
    std::fs::write(class_dir.join("notes.txt"), "not an image").unwrap();

    let config = SupervisedImageFolderConfig::new(2, 2).unwrap();
    let err = load_supervised_image_folder_dataset(&temp_root, &config).unwrap_err();
    assert_eq!(err, ModelError::EmptyDataset);

    let _ = std::fs::remove_file(class_dir.join("notes.txt"));
    let _ = std::fs::remove_dir(&class_dir);
    let _ = std::fs::remove_dir(&temp_root);
}
