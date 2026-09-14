use std::sync::atomic::{AtomicU64, Ordering};

use super::error::ModelError;
use yscv_tensor::Tensor;

/// Trait for deterministic tensor transforms (preprocessing).
pub trait Transform: Send + Sync {
    fn apply(&self, input: &Tensor) -> Result<Tensor, ModelError>;
}

/// Chains multiple transforms sequentially.
pub struct Compose {
    transforms: Vec<Box<dyn Transform>>,
}

impl Default for Compose {
    fn default() -> Self {
        Self::new()
    }
}

impl Compose {
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
        }
    }

    pub fn add<T: Transform + 'static>(mut self, t: T) -> Self {
        self.transforms.push(Box::new(t));
        self
    }
}

impl Transform for Compose {
    fn apply(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        let mut current = input.clone();
        for t in &self.transforms {
            current = t.apply(&current)?;
        }
        Ok(current)
    }
}

/// Normalize channels: `(x - mean) / std`
pub struct Normalize {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

impl Normalize {
    pub const fn new(mean: Vec<f32>, std: Vec<f32>) -> Self {
        Self { mean, std }
    }
}

impl Transform for Normalize {
    fn apply(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        // Apply per-channel normalization on last dim
        let data = input.data();
        let c = self.mean.len();
        let mut out = data.to_vec();
        for (i, val) in out.iter_mut().enumerate() {
            let ch = i % c;
            *val = (*val - self.mean[ch]) / self.std[ch];
        }
        Ok(Tensor::from_vec(input.shape().to_vec(), out)?)
    }
}

/// Scale f32 values by a constant factor.
pub struct ScaleValues {
    pub factor: f32,
}

impl ScaleValues {
    pub const fn new(factor: f32) -> Self {
        Self { factor }
    }
}

impl Transform for ScaleValues {
    fn apply(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        Ok(input.scale(self.factor))
    }
}

/// Permute dimensions.
pub struct PermuteDims {
    pub order: Vec<usize>,
}

impl PermuteDims {
    pub const fn new(order: Vec<usize>) -> Self {
        Self { order }
    }
}

impl Transform for PermuteDims {
    fn apply(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        Ok(input.permute(&self.order)?)
    }
}

/// Resize image tensor to target height and width using bilinear interpolation.
/// Input shape: `[H, W, C]`.
pub struct Resize {
    pub height: usize,
    pub width: usize,
}

impl Resize {
    pub const fn new(height: usize, width: usize) -> Self {
        Self { height, width }
    }
}

impl Transform for Resize {
    fn apply(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(ModelError::InvalidInputShape {
                expected_features: 3,
                got: shape.to_vec(),
            });
        }
        let (in_h, in_w, c) = (shape[0], shape[1], shape[2]);
        let data = input.data();
        let out_h = self.height;
        let out_w = self.width;
        let mut out = vec![0.0f32; out_h * out_w * c];

        for oh in 0..out_h {
            for ow in 0..out_w {
                // Map output pixel center to input coordinates
                let sy = if out_h > 1 {
                    oh as f32 * (in_h as f32 - 1.0) / (out_h as f32 - 1.0)
                } else {
                    (in_h as f32 - 1.0) / 2.0
                };
                let sx = if out_w > 1 {
                    ow as f32 * (in_w as f32 - 1.0) / (out_w as f32 - 1.0)
                } else {
                    (in_w as f32 - 1.0) / 2.0
                };

                let y0 = sy.floor() as usize;
                let x0 = sx.floor() as usize;
                let y1 = (y0 + 1).min(in_h - 1);
                let x1 = (x0 + 1).min(in_w - 1);
                let fy = sy - sy.floor();
                let fx = sx - sx.floor();

                for ch in 0..c {
                    let v00 = data[(y0 * in_w + x0) * c + ch];
                    let v01 = data[(y0 * in_w + x1) * c + ch];
                    let v10 = data[(y1 * in_w + x0) * c + ch];
                    let v11 = data[(y1 * in_w + x1) * c + ch];
                    let val = v00 * (1.0 - fy) * (1.0 - fx)
                        + v01 * (1.0 - fy) * fx
                        + v10 * fy * (1.0 - fx)
                        + v11 * fy * fx;
                    out[(oh * out_w + ow) * c + ch] = val;
                }
            }
        }
        Ok(Tensor::from_vec(vec![out_h, out_w, c], out)?)
    }
}

/// Crop the center region of an image tensor.
/// Input shape: `[H, W, C]`. Output shape: `[size, size, C]`.
pub struct CenterCrop {
    pub size: usize,
}

impl CenterCrop {
    pub const fn new(size: usize) -> Self {
        Self { size }
    }
}

impl Transform for CenterCrop {
    fn apply(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(ModelError::InvalidInputShape {
                expected_features: 3,
                got: shape.to_vec(),
            });
        }
        let (h, w) = (shape[0], shape[1]);
        let start_h = (h.saturating_sub(self.size)) / 2;
        let start_w = (w.saturating_sub(self.size)) / 2;
        let cropped = input.narrow(0, start_h, self.size)?;
        let cropped = cropped.narrow(1, start_w, self.size)?;
        Ok(cropped)
    }
}

/// Randomly flip horizontally with probability `p`.
/// Uses xorshift64 PRNG seeded at construction.
/// Input shape: `[H, W, C]`.
pub struct RandomHorizontalFlip {
    p: f32,
    seed: AtomicU64,
}

impl RandomHorizontalFlip {
    pub const fn new(p: f32, seed: u64) -> Self {
        Self {
            p,
            seed: AtomicU64::new(seed),
        }
    }

    fn next_rand(&self) -> f32 {
        let mut s = self.seed.load(Ordering::Relaxed);
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        self.seed.store(s, Ordering::Relaxed);
        // Map to [0, 1)
        (s as u32 as f32) / (u32::MAX as f32)
    }
}

impl Transform for RandomHorizontalFlip {
    fn apply(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(ModelError::InvalidInputShape {
                expected_features: 3,
                got: shape.to_vec(),
            });
        }
        if self.next_rand() >= self.p {
            return Ok(input.clone());
        }
        let (h, w, c) = (shape[0], shape[1], shape[2]);
        let data = input.data();
        let mut out = vec![0.0f32; h * w * c];
        for row in 0..h {
            for col in 0..w {
                let src = (row * w + (w - 1 - col)) * c;
                let dst = (row * w + col) * c;
                out[dst..dst + c].copy_from_slice(&data[src..src + c]);
            }
        }
        Ok(Tensor::from_vec(shape.to_vec(), out)?)
    }
}

/// Apply Gaussian blur to an image tensor.
/// Input shape: `[H, W, C]`.
pub struct GaussianBlur {
    pub kernel_size: usize,
    pub sigma: f32,
}

impl GaussianBlur {
    pub const fn new(kernel_size: usize, sigma: f32) -> Self {
        Self { kernel_size, sigma }
    }

    fn build_kernel(&self) -> Vec<f32> {
        let ks = self.kernel_size;
        let half = ks as f32 / 2.0;
        let mut kernel = vec![0.0f32; ks * ks];
        let mut sum = 0.0f32;
        for ky in 0..ks {
            for kx in 0..ks {
                let dy = ky as f32 - half + 0.5;
                let dx = kx as f32 - half + 0.5;
                let val = (-(dy * dy + dx * dx) / (2.0 * self.sigma * self.sigma)).exp();
                kernel[ky * ks + kx] = val;
                sum += val;
            }
        }
        for v in kernel.iter_mut() {
            *v /= sum;
        }
        kernel
    }
}

impl Transform for GaussianBlur {
    fn apply(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(ModelError::InvalidInputShape {
                expected_features: 3,
                got: shape.to_vec(),
            });
        }
        let (h, w, c) = (shape[0], shape[1], shape[2]);
        let ks = self.kernel_size;
        let pad = ks / 2;
        let kernel = self.build_kernel();
        let data = input.data();
        let mut out = vec![0.0f32; h * w * c];

        for row in 0..h {
            for col in 0..w {
                for ch in 0..c {
                    let mut acc = 0.0f32;
                    for ky in 0..ks {
                        for kx in 0..ks {
                            let sy = row as isize + ky as isize - pad as isize;
                            let sx = col as isize + kx as isize - pad as isize;
                            // Clamp to border
                            let sy = sy.max(0).min(h as isize - 1) as usize;
                            let sx = sx.max(0).min(w as isize - 1) as usize;
                            acc += data[(sy * w + sx) * c + ch] * kernel[ky * ks + kx];
                        }
                    }
                    out[(row * w + col) * c + ch] = acc;
                }
            }
        }
        Ok(Tensor::from_vec(shape.to_vec(), out)?)
    }
}
