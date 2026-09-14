use yscv_tensor::Tensor;

use crate::ModelError;

/// Quantized tensor representation: INT8 values + per-tensor scale + zero-point.
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    pub data: Vec<i8>,
    pub shape: Vec<usize>,
    pub scale: f32,
    pub zero_point: i8,
}

/// Quantization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantMode {
    /// Symmetric: zero_point = 0, range maps to [-127, 127].
    Symmetric,
    /// Asymmetric: full [-128, 127] range with dynamic zero_point.
    Asymmetric,
}

impl QuantizedTensor {
    /// Quantize an f32 tensor to INT8.
    pub fn from_tensor(tensor: &Tensor, mode: QuantMode) -> Self {
        let data = tensor.data();
        let shape = tensor.shape().to_vec();

        match mode {
            QuantMode::Symmetric => {
                let max_abs = data
                    .iter()
                    .map(|v| v.abs())
                    .fold(0.0f32, f32::max)
                    .max(1e-8);
                let scale = max_abs / 127.0;
                let quantized: Vec<i8> = data
                    .iter()
                    .map(|&v| (v / scale).round().clamp(-127.0, 127.0) as i8)
                    .collect();
                Self {
                    data: quantized,
                    shape,
                    scale,
                    zero_point: 0,
                }
            }
            QuantMode::Asymmetric => {
                let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);
                let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let range = (max_val - min_val).max(1e-8);
                let scale = range / 255.0;
                let zp = (-128.0 - min_val / scale).round().clamp(-128.0, 127.0) as i8;
                let quantized: Vec<i8> = data
                    .iter()
                    .map(|&v| (v / scale + zp as f32).round().clamp(-128.0, 127.0) as i8)
                    .collect();
                Self {
                    data: quantized,
                    shape,
                    scale,
                    zero_point: zp,
                }
            }
        }
    }

    /// Dequantize back to f32 tensor.
    pub fn to_tensor(&self) -> Result<Tensor, ModelError> {
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&q| (q as f32 - self.zero_point as f32) * self.scale)
            .collect();
        Tensor::from_vec(self.shape.clone(), data).map_err(Into::into)
    }

    /// Number of elements.
    pub const fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether empty.
    pub const fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Compression ratio vs f32 (4x for INT8).
    pub const fn compression_ratio(&self) -> f32 {
        4.0
    }

    /// Total bytes of quantized data (not including metadata).
    pub const fn byte_size(&self) -> usize {
        self.data.len()
    }
}

/// Quantized matmul: dequantize -> f32 matmul -> re-quantize.
///
/// This is a naive implementation; a production path would use integer GEMM.
/// Integer-accumulating quantized matmul: `C = A @ B` in INT8 with INT32 accumulation.
///
/// Avoids dequantizing to f32 — computes directly in integer domain:
/// `C_f32[i,j] = scale_a * scale_b * sum_k((A_i8[i,k] - zp_a) * (B_i8[k,j] - zp_b))`
///
/// Then re-quantizes the result.
pub fn quantized_matmul(
    lhs: &QuantizedTensor,
    rhs: &QuantizedTensor,
    mode: QuantMode,
) -> Result<QuantizedTensor, ModelError> {
    if lhs.shape.len() != 2 || rhs.shape.len() != 2 {
        // Fallback to dequant path for non-2D
        let a = lhs.to_tensor()?;
        let b = rhs.to_tensor()?;
        let c = yscv_kernels::matmul_2d(&a, &b)?;
        return Ok(QuantizedTensor::from_tensor(&c, mode));
    }

    let m = lhs.shape[0];
    let k = lhs.shape[1];
    let n = rhs.shape[1];
    if rhs.shape[0] != k {
        let a = lhs.to_tensor()?;
        let b = rhs.to_tensor()?;
        let c = yscv_kernels::matmul_2d(&a, &b)?;
        return Ok(QuantizedTensor::from_tensor(&c, mode));
    }

    let zp_a = lhs.zero_point as i32;
    let zp_b = rhs.zero_point as i32;
    let combined_scale = lhs.scale * rhs.scale;

    // Integer GEMM with i32 accumulation
    let mut c_f32 = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0i32;
            for kk in 0..k {
                let a_val = lhs.data[i * k + kk] as i32 - zp_a;
                let b_val = rhs.data[kk * n + j] as i32 - zp_b;
                acc += a_val * b_val;
            }
            c_f32[i * n + j] = acc as f32 * combined_scale;
        }
    }

    let result = Tensor::from_vec(vec![m, n], c_f32)?;
    Ok(QuantizedTensor::from_tensor(&result, mode))
}

/// Quantize all weight tensors in a model checkpoint for storage/inference.
///
/// Returns `(quantized_weights, original_shapes)` for each weight tensor.
pub fn quantize_weights(weights: &[Tensor], mode: QuantMode) -> Vec<QuantizedTensor> {
    weights
        .iter()
        .map(|w| QuantizedTensor::from_tensor(w, mode))
        .collect()
}

/// Dequantize a set of quantized weights back to f32 tensors.
pub fn dequantize_weights(quantized: &[QuantizedTensor]) -> Result<Vec<Tensor>, ModelError> {
    quantized.iter().map(|q| q.to_tensor()).collect()
}

/// Per-channel symmetric quantization for conv weights `[KH, KW, C_in, C_out]`.
///
/// Each output channel gets its own scale factor for better accuracy.
/// Per-channel quantization result.
pub struct PerChannelQuantResult {
    pub data: Vec<i8>,
    pub scales: Vec<f32>,
    pub shape: Vec<usize>,
}

pub fn quantize_per_channel(
    tensor: &Tensor,
    channel_axis: usize,
) -> Result<PerChannelQuantResult, ModelError> {
    let shape = tensor.shape();
    let data = tensor.data();
    let num_channels = shape[channel_axis];
    let total = data.len();
    let channel_stride: usize = shape[channel_axis + 1..].iter().product();
    let _outer_stride: usize = shape[channel_axis..].iter().product();

    let mut scales = vec![0.0f32; num_channels];
    let mut quantized = vec![0i8; total];

    // Compute per-channel max abs
    for (i, &v) in data.iter().enumerate() {
        let ch = (i / channel_stride) % num_channels;
        scales[ch] = scales[ch].max(v.abs());
    }
    for s in &mut scales {
        *s = (*s).max(1e-8) / 127.0;
    }

    for (i, &v) in data.iter().enumerate() {
        let ch = (i / channel_stride) % num_channels;
        quantized[i] = (v / scales[ch]).round().clamp(-127.0, 127.0) as i8;
    }

    Ok(PerChannelQuantResult {
        data: quantized,
        scales,
        shape: shape.to_vec(),
    })
}

// ---------------------------------------------------------------------------
// Weight Pruning
// ---------------------------------------------------------------------------

/// Result of magnitude-based weight pruning.
#[derive(Debug, Clone)]
pub struct PrunedTensor {
    /// Binary mask: 1.0 = keep, 0.0 = pruned.
    pub mask: Tensor,
    /// Original weights with pruned values zeroed out.
    pub pruned_weights: Tensor,
    /// Fraction of weights set to zero (0.0–1.0).
    pub sparsity: f32,
}

/// Prune weights by magnitude: zero out the smallest `sparsity` fraction.
///
/// For example, `sparsity = 0.5` removes the 50% of weights with smallest
/// absolute value. Returns the pruned weights and binary mask.
pub fn prune_magnitude(weights: &Tensor, sparsity: f32) -> Result<PrunedTensor, ModelError> {
    if !(0.0..=1.0).contains(&sparsity) {
        return Err(ModelError::InvalidDropoutRate { rate: sparsity });
    }
    let data = weights.data();
    let n = data.len();
    if n == 0 || sparsity == 0.0 {
        let mask = Tensor::from_vec(weights.shape().to_vec(), vec![1.0f32; n])?;
        return Ok(PrunedTensor {
            mask,
            pruned_weights: weights.clone(),
            sparsity: 0.0,
        });
    }

    // Find threshold: sort absolute values, pick the sparsity-percentile value
    let mut abs_vals: Vec<f32> = data.iter().map(|v| v.abs()).collect();
    abs_vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let cutoff_idx = ((n as f32 * sparsity) as usize).min(n - 1);
    let threshold = abs_vals[cutoff_idx];

    let mut mask_data = Vec::with_capacity(n);
    let mut pruned_data = Vec::with_capacity(n);
    let mut pruned_count = 0usize;
    for &v in data {
        if v.abs() <= threshold {
            mask_data.push(0.0f32);
            pruned_data.push(0.0f32);
            pruned_count += 1;
        } else {
            mask_data.push(1.0f32);
            pruned_data.push(v);
        }
    }

    let actual_sparsity = pruned_count as f32 / n as f32;
    let mask = Tensor::from_vec(weights.shape().to_vec(), mask_data)?;
    let pruned_weights = Tensor::from_vec(weights.shape().to_vec(), pruned_data)?;

    Ok(PrunedTensor {
        mask,
        pruned_weights,
        sparsity: actual_sparsity,
    })
}

/// Apply a binary mask to weights (element-wise multiply).
pub fn apply_pruning_mask(weights: &Tensor, mask: &Tensor) -> Result<Tensor, ModelError> {
    weights.mul(mask).map_err(ModelError::Tensor)
}
