use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc;
use std::thread;

use yscv_tensor::Tensor;

use crate::ModelError;

/// Configuration for the parallel data loader.
#[derive(Debug, Clone)]
pub struct DataLoaderConfig {
    /// Number of samples per batch.
    pub batch_size: usize,
    /// Number of worker threads for prefetching.
    pub num_workers: usize,
    /// Number of batches each worker prefetches before blocking.
    pub prefetch_factor: usize,
    /// Whether to drop the last incomplete batch.
    pub drop_last: bool,
    /// Whether to shuffle samples each epoch.
    pub shuffle: bool,
}

impl Default for DataLoaderConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            num_workers: 1,
            prefetch_factor: 2,
            drop_last: false,
            shuffle: false,
        }
    }
}

/// A batch of samples produced by the data loader.
#[derive(Debug, Clone, PartialEq)]
pub struct DataLoaderBatch {
    /// Stacked input tensors with shape `[batch_size, ...]`.
    pub inputs: Tensor,
    /// Stacked target tensors with shape `[batch_size, ...]`.
    pub targets: Tensor,
}

/// Parallel data loader that prefetches batches using worker threads.
pub struct DataLoader {
    config: DataLoaderConfig,
    inputs: Vec<Tensor>,
    targets: Vec<Tensor>,
    epoch_counter: std::cell::Cell<u64>,
}

impl DataLoader {
    /// Creates a new data loader from individual sample tensors and configuration.
    ///
    /// All input tensors must have the same shape, and all target tensors must
    /// have the same shape. The number of inputs must equal the number of targets.
    pub fn new(
        inputs: Vec<Tensor>,
        targets: Vec<Tensor>,
        config: DataLoaderConfig,
    ) -> Result<Self, ModelError> {
        if inputs.len() != targets.len() {
            return Err(ModelError::DatasetShapeMismatch {
                inputs: vec![inputs.len()],
                targets: vec![targets.len()],
            });
        }
        if config.batch_size == 0 {
            return Err(ModelError::InvalidBatchSize {
                batch_size: config.batch_size,
            });
        }
        if config.num_workers == 0 {
            return Err(ModelError::InvalidBatchSize {
                batch_size: config.num_workers,
            });
        }
        // Validate that all inputs share the same shape.
        if let Some(first) = inputs.first() {
            let expected = first.shape();
            for t in inputs.iter().skip(1) {
                if t.shape() != expected {
                    return Err(ModelError::InvalidParameterShape {
                        parameter: "data_loader_input",
                        expected: expected.to_vec(),
                        got: t.shape().to_vec(),
                    });
                }
            }
        }
        // Validate that all targets share the same shape.
        if let Some(first) = targets.first() {
            let expected = first.shape();
            for t in targets.iter().skip(1) {
                if t.shape() != expected {
                    return Err(ModelError::InvalidParameterShape {
                        parameter: "data_loader_target",
                        expected: expected.to_vec(),
                        got: t.shape().to_vec(),
                    });
                }
            }
        }
        Ok(Self {
            config,
            inputs,
            targets,
            epoch_counter: std::cell::Cell::new(0),
        })
    }

    /// Returns the number of batches per epoch.
    pub fn len(&self) -> usize {
        let n = self.inputs.len();
        if n == 0 || self.config.batch_size == 0 {
            return 0;
        }
        if self.config.drop_last {
            n / self.config.batch_size
        } else {
            n.div_ceil(self.config.batch_size)
        }
    }

    /// Returns `true` if the loader has no batches.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the underlying configuration.
    pub fn config(&self) -> &DataLoaderConfig {
        &self.config
    }

    /// Returns the total number of samples.
    pub fn sample_count(&self) -> usize {
        self.inputs.len()
    }

    /// Creates an iterator that spawns worker threads and prefetches batches.
    ///
    /// Each call increments the internal epoch counter, producing a different
    /// shuffle order when `config.shuffle` is `true`.
    pub fn iter(&self) -> DataLoaderIter {
        let epoch = self.epoch_counter.get();
        self.epoch_counter.set(epoch.wrapping_add(1));

        let num_samples = self.inputs.len();
        let batch_size = self.config.batch_size;

        // Build sample ordering.
        let mut indices: Vec<usize> = (0..num_samples).collect();
        if self.config.shuffle {
            lcg_shuffle(&mut indices, epoch);
        }

        // Build batch index ranges.
        let mut batch_ranges: Vec<(usize, usize)> = Vec::new();
        let mut start = 0;
        while start < num_samples {
            let end = (start + batch_size).min(num_samples);
            let is_full = (end - start) == batch_size;
            if is_full || !self.config.drop_last {
                batch_ranges.push((start, end));
            }
            start = end;
        }

        let total_batches = batch_ranges.len();

        if total_batches == 0 {
            // No work to do; return an iterator that immediately finishes.
            let (_tx, rx) = mpsc::sync_channel::<Result<DataLoaderBatch, String>>(0);
            return DataLoaderIter {
                receiver: rx,
                _workers: Vec::new(),
                remaining: 0,
            };
        }

        let channel_capacity = self
            .config
            .num_workers
            .saturating_mul(self.config.prefetch_factor)
            .max(1);
        let (tx, rx) = mpsc::sync_channel::<Result<DataLoaderBatch, String>>(channel_capacity);

        // Share data with workers via Arc.
        // NOTE: cloning Vec<Tensor> here is unavoidable without changing the
        // DataLoader field types to Arc<Vec<Tensor>>, which would alter the
        // public API (new(), etc.).  The clone cost is amortised over the epoch
        // and dominates only for very small datasets.
        let shared_inputs = Arc::new(self.inputs.clone());
        let shared_targets = Arc::new(self.targets.clone());
        let shared_indices = Arc::new(indices);

        let num_workers = self.config.num_workers.min(total_batches);
        let mut workers = Vec::with_capacity(num_workers);

        for worker_id in 0..num_workers {
            // Each worker handles batches: worker_id, worker_id + N, worker_id + 2N, ...
            let worker_batch_indices: Vec<usize> =
                (worker_id..total_batches).step_by(num_workers).collect();
            let worker_ranges: Vec<(usize, usize)> = worker_batch_indices
                .iter()
                .map(|&bi| batch_ranges[bi])
                .collect();

            let tx = tx.clone();
            let inputs = Arc::clone(&shared_inputs);
            let targets = Arc::clone(&shared_targets);
            let sample_indices = Arc::clone(&shared_indices);

            let handle = thread::spawn(move || {
                for (range_start, range_end) in worker_ranges {
                    let batch_indices: Vec<usize> = (range_start..range_end)
                        .map(|i| sample_indices[i])
                        .collect();

                    let result = build_batch(&inputs, &targets, &batch_indices);
                    let send_result = match result {
                        Ok(batch) => tx.send(Ok(batch)),
                        Err(e) => tx.send(Err(e.to_string())),
                    };
                    if send_result.is_err() {
                        // Receiver dropped; stop producing.
                        break;
                    }
                }
            });
            workers.push(handle);
        }

        // Drop the original sender so the channel closes when all workers finish.
        drop(tx);

        DataLoaderIter {
            receiver: rx,
            _workers: workers,
            remaining: total_batches,
        }
    }
}

/// Iterator over batches produced by worker threads.
pub struct DataLoaderIter {
    receiver: mpsc::Receiver<Result<DataLoaderBatch, String>>,
    _workers: Vec<thread::JoinHandle<()>>,
    remaining: usize,
}

impl Iterator for DataLoaderIter {
    type Item = Result<DataLoaderBatch, ModelError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        match self.receiver.recv() {
            Ok(Ok(batch)) => {
                self.remaining -= 1;
                Some(Ok(batch))
            }
            Ok(Err(msg)) => {
                self.remaining -= 1;
                Some(Err(ModelError::DatasetLoadIo {
                    path: String::new(),
                    message: msg,
                }))
            }
            Err(_) => {
                // Channel closed unexpectedly.
                self.remaining = 0;
                None
            }
        }
    }
}

/// Stack individual sample tensors into a single batch tensor.
///
/// Given tensors each with shape `[d0, d1, ...]`, produces a tensor with
/// shape `[batch_size, d0, d1, ...]`.
fn stack_tensors(tensors: &[&Tensor]) -> Result<Tensor, ModelError> {
    if tensors.is_empty() {
        return Err(ModelError::EmptyDataset);
    }
    let sample_shape = tensors[0].shape();
    let sample_len = tensors[0].len();

    let batch_size = tensors.len();
    let mut batch_shape = Vec::with_capacity(sample_shape.len() + 1);
    batch_shape.push(batch_size);
    batch_shape.extend_from_slice(sample_shape);

    let total_len = batch_size * sample_len;
    let mut data = Vec::with_capacity(total_len);
    for tensor in tensors {
        data.extend_from_slice(tensor.data());
    }

    Tensor::from_vec(batch_shape, data).map_err(ModelError::from)
}

/// Build a single batch from the given sample indices.
fn build_batch(
    inputs: &[Tensor],
    targets: &[Tensor],
    indices: &[usize],
) -> Result<DataLoaderBatch, ModelError> {
    let input_refs: Vec<&Tensor> = indices.iter().map(|&i| &inputs[i]).collect();
    let target_refs: Vec<&Tensor> = indices.iter().map(|&i| &targets[i]).collect();

    let stacked_inputs = stack_tensors(&input_refs)?;
    let stacked_targets = stack_tensors(&target_refs)?;

    Ok(DataLoaderBatch {
        inputs: stacked_inputs,
        targets: stacked_targets,
    })
}

/// Simple LCG-based Fisher-Yates shuffle, deterministic for a given seed.
fn lcg_shuffle(indices: &mut [usize], seed: u64) {
    let mut state = seed ^ 0x6C62_272E_07BB_0142;
    let mut index = indices.len();
    while index > 1 {
        index -= 1;
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let swap_idx = ((state >> 33) as usize) % (index + 1);
        indices.swap(index, swap_idx);
    }
}

// ---------------------------------------------------------------------------
// Samplers
// ---------------------------------------------------------------------------

/// A sampler that yields indices in sequential order.
#[derive(Debug, Clone)]
pub struct SequentialSampler {
    len: usize,
}

impl SequentialSampler {
    pub fn new(len: usize) -> Self {
        Self { len }
    }

    /// Returns indices `[0, 1, 2, ..., len-1]`.
    pub fn indices(&self) -> Vec<usize> {
        (0..self.len).collect()
    }
}

/// A sampler that yields indices in a random (deterministic) order.
#[derive(Debug, Clone)]
pub struct RandomSampler {
    len: usize,
    seed: u64,
}

impl RandomSampler {
    pub fn new(len: usize, seed: u64) -> Self {
        Self { len, seed }
    }

    /// Returns a shuffled permutation of `[0, len)` using the given seed.
    pub fn indices(&self) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..self.len).collect();
        lcg_shuffle(&mut idx, self.seed);
        idx
    }
}

/// Weighted random sampler: draws `num_samples` indices with probability proportional to weights.
///
/// Useful for imbalanced datasets where minority classes should be oversampled.
#[derive(Debug, Clone)]
pub struct WeightedRandomSampler {
    weights: Vec<f64>,
    num_samples: usize,
    seed: u64,
}

impl WeightedRandomSampler {
    /// Creates a new weighted sampler.
    ///
    /// `weights`: per-sample weight (higher = more likely to be sampled).
    /// `num_samples`: how many indices to draw per epoch.
    /// `seed`: deterministic random seed.
    pub fn new(weights: Vec<f64>, num_samples: usize, seed: u64) -> Result<Self, ModelError> {
        if weights.is_empty() {
            return Err(ModelError::EmptyDataset);
        }
        Ok(Self {
            weights,
            num_samples,
            seed,
        })
    }

    /// Draw `num_samples` indices with replacement, proportional to weights.
    pub fn indices(&self) -> Vec<usize> {
        let total: f64 = self.weights.iter().sum();
        if total <= 0.0 {
            return (0..self.num_samples)
                .map(|i| i % self.weights.len())
                .collect();
        }

        // Build CDF
        let mut cdf = Vec::with_capacity(self.weights.len());
        let mut acc = 0.0;
        for &w in &self.weights {
            acc += w / total;
            cdf.push(acc);
        }

        let mut state = self.seed ^ 0x5DEE_CE66_D1A4_F87D;
        let mut result = Vec::with_capacity(self.num_samples);
        for _ in 0..self.num_samples {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let u = (state >> 11) as f64 / (1u64 << 53) as f64; // uniform [0, 1)
            // Binary search in CDF
            let idx = match cdf
                .binary_search_by(|v| v.partial_cmp(&u).unwrap_or(std::cmp::Ordering::Equal))
            {
                Ok(i) => i,
                Err(i) => i.min(self.weights.len() - 1),
            };
            result.push(idx);
        }
        result
    }

    /// Number of samples drawn per epoch.
    pub fn num_samples(&self) -> usize {
        self.num_samples
    }
}

// ---------------------------------------------------------------------------
// Streaming Data Loader
// ---------------------------------------------------------------------------

/// A data loader that lazily reads batches from disk, using a background thread
/// to prefetch the next batch while the current batch is being processed.
///
/// The loader scans a directory for numbered batch files (`batch_0000.bin`,
/// `batch_0001.bin`, etc.). Each file stores a pair of tensors (inputs + targets)
/// in a simple binary format:
///
/// ```text
/// [input_ndims: u32] [input_shape...] [input_data: f32...]
/// [target_ndims: u32] [target_shape...] [target_data: f32...]
/// ```
pub struct StreamingDataLoader {
    path: PathBuf,
    batch_size: usize,
    file_paths: Vec<PathBuf>,
    current_index: usize,
    prefetch_rx: Option<mpsc::Receiver<Result<(Tensor, Tensor), ModelError>>>,
    _prefetch_handle: Option<thread::JoinHandle<()>>,
}

impl StreamingDataLoader {
    /// Create a new streaming data loader that reads batch files from `path`.
    ///
    /// The directory is scanned for files matching `batch_NNNN.bin`. If no
    /// batch files are found, an empty loader is returned.
    pub fn new(path: impl Into<PathBuf>, batch_size: usize) -> Result<Self, ModelError> {
        let path = path.into();
        if batch_size == 0 {
            return Err(ModelError::InvalidBatchSize { batch_size });
        }
        let file_paths = Self::scan_batch_files(&path);
        let mut loader = Self {
            path,
            batch_size,
            file_paths,
            current_index: 0,
            prefetch_rx: None,
            _prefetch_handle: None,
        };
        loader.start_prefetch();
        Ok(loader)
    }

    /// Returns the next batch of (inputs, targets), or `None` if all batches
    /// have been consumed for this epoch.
    pub fn next_batch(&mut self) -> Option<(Tensor, Tensor)> {
        if self.current_index >= self.file_paths.len() {
            return None;
        }

        // Receive the prefetched batch.
        let result = if let Some(rx) = self.prefetch_rx.take() {
            match rx.recv() {
                Ok(Ok(batch)) => Some(batch),
                Ok(Err(_)) => None,
                Err(_) => None,
            }
        } else {
            // Fallback: load synchronously.
            Self::load_batch_file(&self.file_paths[self.current_index]).ok()
        };

        self.current_index += 1;

        // Start prefetching the next batch.
        self.start_prefetch();

        result
    }

    /// Reset the loader to the beginning so it can be iterated again.
    pub fn reset(&mut self) {
        // Drop any in-flight prefetch.
        self.prefetch_rx = None;
        self._prefetch_handle = None;
        self.current_index = 0;
        self.start_prefetch();
    }

    /// Returns the total number of batch files available.
    pub fn len(&self) -> usize {
        self.file_paths.len()
    }

    /// Returns `true` if there are no batch files.
    pub fn is_empty(&self) -> bool {
        self.file_paths.is_empty()
    }

    /// Returns the configured batch size.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Returns the directory path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    // -- internal helpers ---------------------------------------------------

    fn start_prefetch(&mut self) {
        if self.current_index >= self.file_paths.len() {
            return;
        }
        let file_path = self.file_paths[self.current_index].clone();
        let (tx, rx) = mpsc::sync_channel(1);
        let handle = thread::spawn(move || {
            let result = Self::load_batch_file(&file_path);
            let _ = tx.send(result);
        });
        self.prefetch_rx = Some(rx);
        self._prefetch_handle = Some(handle);
    }

    /// Scan directory for `batch_NNNN.bin` files, sorted by name.
    fn scan_batch_files(dir: &Path) -> Vec<PathBuf> {
        let read_dir = match std::fs::read_dir(dir) {
            Ok(rd) => rd,
            Err(_) => return Vec::new(),
        };
        let mut paths: Vec<PathBuf> = read_dir
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|p| {
                if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                    name.starts_with("batch_") && name.ends_with(".bin")
                } else {
                    false
                }
            })
            .collect();
        paths.sort();
        paths
    }

    /// Load a single batch file in the simple binary tensor-pair format.
    fn load_batch_file(path: &Path) -> Result<(Tensor, Tensor), ModelError> {
        let data = std::fs::read(path).map_err(|e| ModelError::DatasetLoadIo {
            path: path.display().to_string(),
            message: e.to_string(),
        })?;
        let input = Self::read_tensor_from_bytes(&data, 0)?;
        let offset = Self::tensor_byte_size(&input) + 4 + input.shape().len() * 4;
        let target = Self::read_tensor_from_bytes(&data, offset)?;
        Ok((input, target))
    }

    /// Write a tensor pair to the simple binary format.
    pub fn write_batch_file(
        path: &Path,
        inputs: &Tensor,
        targets: &Tensor,
    ) -> Result<(), ModelError> {
        let mut buf = Vec::new();
        Self::write_tensor_to_bytes(&mut buf, inputs);
        Self::write_tensor_to_bytes(&mut buf, targets);
        std::fs::write(path, &buf).map_err(|e| ModelError::DatasetLoadIo {
            path: path.display().to_string(),
            message: e.to_string(),
        })
    }

    fn write_tensor_to_bytes(buf: &mut Vec<u8>, tensor: &Tensor) {
        let ndims = tensor.shape().len() as u32;
        buf.extend_from_slice(&ndims.to_le_bytes());
        for &d in tensor.shape() {
            buf.extend_from_slice(&(d as u32).to_le_bytes());
        }
        for &v in tensor.data() {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }

    fn tensor_byte_size(tensor: &Tensor) -> usize {
        tensor.data().len() * 4
    }

    fn read_tensor_from_bytes(data: &[u8], offset: usize) -> Result<Tensor, ModelError> {
        if offset + 4 > data.len() {
            return Err(ModelError::DatasetLoadIo {
                path: String::new(),
                message: "unexpected end of batch file (ndims)".to_string(),
            });
        }
        let ndims = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        let shape_start = offset + 4;
        let shape_end = shape_start + ndims * 4;
        if shape_end > data.len() {
            return Err(ModelError::DatasetLoadIo {
                path: String::new(),
                message: "unexpected end of batch file (shape)".to_string(),
            });
        }
        let mut shape = Vec::with_capacity(ndims);
        for i in 0..ndims {
            let s = shape_start + i * 4;
            shape.push(
                u32::from_le_bytes([data[s], data[s + 1], data[s + 2], data[s + 3]]) as usize,
            );
        }
        let num_elements: usize = shape.iter().product();
        let data_start = shape_end;
        let data_end = data_start + num_elements * 4;
        if data_end > data.len() {
            return Err(ModelError::DatasetLoadIo {
                path: String::new(),
                message: "unexpected end of batch file (data)".to_string(),
            });
        }
        let mut values = Vec::with_capacity(num_elements);
        for i in 0..num_elements {
            let s = data_start + i * 4;
            values.push(f32::from_le_bytes([
                data[s],
                data[s + 1],
                data[s + 2],
                data[s + 3],
            ]));
        }
        Tensor::from_vec(shape, values).map_err(ModelError::from)
    }
}
