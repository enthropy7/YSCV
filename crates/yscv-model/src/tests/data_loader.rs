use yscv_tensor::Tensor;

use crate::{
    DataLoader, DataLoaderConfig, RandomSampler, SequentialSampler, WeightedRandomSampler,
};

/// Helper: create `n` sample tensors of shape `sample_shape`, each filled with `index as f32`.
fn make_samples(n: usize, sample_shape: &[usize]) -> Vec<Tensor> {
    (0..n)
        .map(|i| {
            let len: usize = sample_shape.iter().product();
            Tensor::from_vec(sample_shape.to_vec(), vec![i as f32; len]).unwrap()
        })
        .collect()
}

#[test]
fn test_data_loader_creation() {
    let inputs = make_samples(100, &[3, 4]);
    let targets = make_samples(100, &[1]);
    let config = DataLoaderConfig {
        batch_size: 10,
        num_workers: 2,
        prefetch_factor: 2,
        drop_last: false,
        shuffle: false,
    };
    let loader = DataLoader::new(inputs, targets, config).unwrap();
    assert_eq!(loader.len(), 10);
    assert_eq!(loader.sample_count(), 100);
    assert!(!loader.is_empty());
}

#[test]
fn test_data_loader_single_worker() {
    let inputs = make_samples(20, &[4]);
    let targets = make_samples(20, &[1]);
    let config = DataLoaderConfig {
        batch_size: 5,
        num_workers: 1,
        prefetch_factor: 2,
        drop_last: false,
        shuffle: false,
    };
    let loader = DataLoader::new(inputs, targets, config).unwrap();
    assert_eq!(loader.len(), 4);

    let batches: Vec<_> = loader.iter().collect();
    assert_eq!(batches.len(), 4);
    for batch_result in &batches {
        assert!(batch_result.is_ok());
    }
}

#[test]
fn test_data_loader_multi_worker() {
    let n = 40;
    let inputs = make_samples(n, &[3]);
    let targets = make_samples(n, &[1]);
    let config = DataLoaderConfig {
        batch_size: 5,
        num_workers: 4,
        prefetch_factor: 2,
        drop_last: false,
        shuffle: false,
    };
    let loader = DataLoader::new(inputs, targets, config).unwrap();
    assert_eq!(loader.len(), 8);

    let batches: Vec<_> = loader.iter().map(|r| r.unwrap()).collect();
    assert_eq!(batches.len(), 8);

    // Verify all samples are covered (each sample has a unique fill value).
    let mut seen = std::collections::HashSet::new();
    for batch in &batches {
        let batch_size = batch.inputs.shape()[0];
        let sample_len = 3;
        for s in 0..batch_size {
            let value = batch.inputs.data()[s * sample_len] as usize;
            seen.insert(value);
        }
    }
    assert_eq!(seen.len(), n);
}

#[test]
fn test_data_loader_drop_last() {
    let inputs = make_samples(15, &[2]);
    let targets = make_samples(15, &[1]);
    let config = DataLoaderConfig {
        batch_size: 10,
        num_workers: 1,
        prefetch_factor: 1,
        drop_last: true,
        shuffle: false,
    };
    let loader = DataLoader::new(inputs, targets, config).unwrap();
    assert_eq!(loader.len(), 1);

    let batches: Vec<_> = loader.iter().map(|r| r.unwrap()).collect();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].inputs.shape()[0], 10);
}

#[test]
fn test_data_loader_keep_last() {
    let inputs = make_samples(15, &[2]);
    let targets = make_samples(15, &[1]);
    let config = DataLoaderConfig {
        batch_size: 10,
        num_workers: 1,
        prefetch_factor: 1,
        drop_last: false,
        shuffle: false,
    };
    let loader = DataLoader::new(inputs, targets, config).unwrap();
    assert_eq!(loader.len(), 2);

    let batches: Vec<_> = loader.iter().map(|r| r.unwrap()).collect();
    assert_eq!(batches.len(), 2);
    assert_eq!(
        batches[0].inputs.shape()[0] + batches[1].inputs.shape()[0],
        15
    );
}

#[test]
fn test_data_loader_batch_shapes() {
    let inputs = make_samples(12, &[3, 4]);
    let targets = make_samples(12, &[2]);
    let config = DataLoaderConfig {
        batch_size: 4,
        num_workers: 2,
        prefetch_factor: 1,
        drop_last: false,
        shuffle: false,
    };
    let loader = DataLoader::new(inputs, targets, config).unwrap();

    for batch_result in loader.iter() {
        let batch = batch_result.unwrap();
        assert_eq!(batch.inputs.shape().len(), 3); // [batch, 3, 4]
        assert_eq!(batch.inputs.shape()[1], 3);
        assert_eq!(batch.inputs.shape()[2], 4);
        assert_eq!(batch.targets.shape().len(), 2); // [batch, 2]
        assert_eq!(batch.targets.shape()[1], 2);
    }
}

#[test]
fn test_data_loader_shuffle() {
    let n = 50;
    let inputs = make_samples(n, &[1]);
    let targets = make_samples(n, &[1]);
    let config = DataLoaderConfig {
        batch_size: 5,
        num_workers: 1,
        prefetch_factor: 1,
        drop_last: false,
        shuffle: true,
    };
    let loader = DataLoader::new(inputs, targets, config).unwrap();

    // Collect the ordering of two epochs.
    let epoch1: Vec<f32> = loader
        .iter()
        .flat_map(|r| {
            let batch = r.unwrap();
            let batch_size = batch.inputs.shape()[0];
            (0..batch_size)
                .map(|i| batch.inputs.data()[i])
                .collect::<Vec<_>>()
        })
        .collect();

    let epoch2: Vec<f32> = loader
        .iter()
        .flat_map(|r| {
            let batch = r.unwrap();
            let batch_size = batch.inputs.shape()[0];
            (0..batch_size)
                .map(|i| batch.inputs.data()[i])
                .collect::<Vec<_>>()
        })
        .collect();

    assert_eq!(epoch1.len(), n);
    assert_eq!(epoch2.len(), n);
    // With high probability these should differ.
    assert_ne!(epoch1, epoch2, "two shuffled epochs should differ");
}

#[test]
fn test_data_loader_empty() {
    let inputs: Vec<Tensor> = Vec::new();
    let targets: Vec<Tensor> = Vec::new();
    let config = DataLoaderConfig {
        batch_size: 10,
        num_workers: 2,
        prefetch_factor: 2,
        drop_last: false,
        shuffle: false,
    };
    let loader = DataLoader::new(inputs, targets, config).unwrap();
    assert_eq!(loader.len(), 0);
    assert!(loader.is_empty());

    let batches: Vec<_> = loader.iter().collect();
    assert_eq!(batches.len(), 0);
}

#[test]
fn test_data_loader_mismatched_counts() {
    let inputs = make_samples(10, &[3]);
    let targets = make_samples(5, &[1]);
    let config = DataLoaderConfig {
        batch_size: 5,
        num_workers: 1,
        prefetch_factor: 1,
        drop_last: false,
        shuffle: false,
    };
    assert!(DataLoader::new(inputs, targets, config).is_err());
}

#[test]
fn test_data_loader_batch_size_larger_than_dataset() {
    let inputs = make_samples(3, &[2]);
    let targets = make_samples(3, &[1]);
    let config = DataLoaderConfig {
        batch_size: 10,
        num_workers: 1,
        prefetch_factor: 1,
        drop_last: false,
        shuffle: false,
    };
    let loader = DataLoader::new(inputs, targets, config).unwrap();
    assert_eq!(loader.len(), 1);

    let batches: Vec<_> = loader.iter().map(|r| r.unwrap()).collect();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].inputs.shape()[0], 3);
}

#[test]
fn test_data_loader_batch_size_larger_than_dataset_drop_last() {
    let inputs = make_samples(3, &[2]);
    let targets = make_samples(3, &[1]);
    let config = DataLoaderConfig {
        batch_size: 10,
        num_workers: 1,
        prefetch_factor: 1,
        drop_last: true,
        shuffle: false,
    };
    let loader = DataLoader::new(inputs, targets, config).unwrap();
    assert_eq!(loader.len(), 0);

    let batches: Vec<_> = loader.iter().collect();
    assert_eq!(batches.len(), 0);
}

#[test]
fn test_sequential_sampler() {
    let sampler = SequentialSampler::new(10);
    let indices = sampler.indices();
    assert_eq!(indices, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
}

#[test]
fn test_random_sampler() {
    let sampler = RandomSampler::new(10, 42);
    let indices = sampler.indices();
    assert_eq!(indices.len(), 10);
    // All indices present (it's a permutation)
    let mut sorted = indices.clone();
    sorted.sort();
    assert_eq!(sorted, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    // Different seed gives different order
    let sampler2 = RandomSampler::new(10, 99);
    let indices2 = sampler2.indices();
    assert_ne!(indices, indices2);
}

#[test]
fn test_weighted_random_sampler() {
    // Weight all on index 0
    let sampler = WeightedRandomSampler::new(vec![100.0, 0.0, 0.0, 0.0], 20, 42).unwrap();
    let indices = sampler.indices();
    assert_eq!(indices.len(), 20);
    // All samples should be index 0 since others have weight 0
    for &i in &indices {
        assert_eq!(i, 0);
    }
}

#[test]
fn test_weighted_random_sampler_balanced() {
    // Equal weights should give roughly equal distribution
    let n = 1000;
    let sampler = WeightedRandomSampler::new(vec![1.0, 1.0, 1.0, 1.0], n, 123).unwrap();
    let indices = sampler.indices();
    assert_eq!(indices.len(), n);
    let mut counts = [0usize; 4];
    for &i in &indices {
        counts[i] += 1;
    }
    // Each class should get roughly n/4 = 250 samples, allow ±100
    for &c in &counts {
        assert!(c > 150, "count too low: {c}");
        assert!(c < 350, "count too high: {c}");
    }
}
