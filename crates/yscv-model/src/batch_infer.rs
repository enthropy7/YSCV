use yscv_tensor::Tensor;

use crate::ModelError;

/// Dynamic batching configuration for inference.
#[derive(Debug, Clone)]
pub struct DynamicBatchConfig {
    /// Maximum batch size to accumulate before dispatching.
    pub max_batch_size: usize,
    /// Pad incomplete batches with zeros to enable fixed-size dispatch.
    pub pad_incomplete: bool,
}

impl Default for DynamicBatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            pad_incomplete: false,
        }
    }
}

/// Splits a large input into batches, runs inference, and reassembles.
///
/// `input` is `[N, ...]`, `infer_fn` processes `[B, ...]` and returns `[B, ...]`.
pub fn batched_inference<F>(
    input: &Tensor,
    config: &DynamicBatchConfig,
    infer_fn: F,
) -> Result<Tensor, ModelError>
where
    F: Fn(&Tensor) -> Result<Tensor, ModelError>,
{
    let shape = input.shape();
    if shape.is_empty() {
        return Err(ModelError::InvalidFlattenShape {
            got: shape.to_vec(),
        });
    }
    let n = shape[0];
    let sample_size: usize = shape[1..].iter().product();
    let data = input.data();
    let bs = config.max_batch_size;

    let mut all_outputs: Vec<f32> = Vec::new();
    let mut out_sample_shape: Option<Vec<usize>> = None;

    let mut offset = 0;
    while offset < n {
        let batch_n = (n - offset).min(bs);
        let start = offset * sample_size;
        let end = (offset + batch_n) * sample_size;
        let mut batch_data = data[start..end].to_vec();

        let actual_n = if config.pad_incomplete && batch_n < bs {
            let pad = (bs - batch_n) * sample_size;
            batch_data.extend(std::iter::repeat_n(0.0f32, pad));
            bs
        } else {
            batch_n
        };

        let mut batch_shape = shape.to_vec();
        batch_shape[0] = actual_n;
        let batch_tensor = Tensor::from_vec(batch_shape, batch_data)?;

        let result = infer_fn(&batch_tensor)?;
        let result_shape = result.shape();

        if out_sample_shape.is_none() {
            out_sample_shape = Some(result_shape[1..].to_vec());
        }

        let out_sample_size: usize = result_shape[1..].iter().product();
        let useful_data = &result.data()[..batch_n * out_sample_size];
        all_outputs.extend_from_slice(useful_data);

        offset += batch_n;
    }

    let sample_shape = out_sample_shape.unwrap_or_default();
    let mut final_shape = vec![n];
    final_shape.extend_from_slice(&sample_shape);
    Tensor::from_vec(final_shape, all_outputs).map_err(Into::into)
}

/// Collects individual samples into batches for efficient processing.
pub struct BatchCollector {
    samples: Vec<Vec<f32>>,
    sample_shape: Vec<usize>,
    max_batch: usize,
}

impl BatchCollector {
    pub const fn new(sample_shape: Vec<usize>, max_batch: usize) -> Self {
        Self {
            samples: Vec::new(),
            sample_shape,
            max_batch,
        }
    }

    /// Adds a sample `[...]` (must match `sample_shape`).
    pub fn push(&mut self, sample: &Tensor) -> Result<(), ModelError> {
        if sample.shape() != self.sample_shape {
            return Err(ModelError::InvalidParameterShape {
                parameter: "sample",
                expected: self.sample_shape.clone(),
                got: sample.shape().to_vec(),
            });
        }
        self.samples.push(sample.data().to_vec());
        Ok(())
    }

    /// Returns true if the collector has enough samples for a full batch.
    pub const fn is_ready(&self) -> bool {
        self.samples.len() >= self.max_batch
    }

    /// Flushes collected samples as a batched tensor `[N, ...]`.
    pub fn flush(&mut self) -> Result<Option<Tensor>, ModelError> {
        if self.samples.is_empty() {
            return Ok(None);
        }
        let n = self.samples.len().min(self.max_batch);
        let batch: Vec<f32> = self.samples.drain(..n).flatten().collect();
        let mut shape = vec![n];
        shape.extend_from_slice(&self.sample_shape);
        let t = Tensor::from_vec(shape, batch)?;
        Ok(Some(t))
    }

    /// Number of pending samples.
    pub const fn pending(&self) -> usize {
        self.samples.len()
    }
}
