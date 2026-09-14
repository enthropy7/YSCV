//! Weight initialization strategies (Kaiming, Xavier, orthogonal).
//!
//! These match PyTorch's `torch.nn.init` functions and produce deterministic
//! results given the same seed.

use yscv_tensor::Tensor;

use crate::ModelError;

/// Simple xorshift64 PRNG (deterministic, no external deps).
struct Rng(u64);

impl Rng {
    const fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0xDEAD_BEEF } else { seed })
    }

    const fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform in [0, 1).
    fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform in [lo, hi).
    fn uniform_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.uniform()
    }

    /// Standard normal via Box-Muller.
    fn normal(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-10);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

/// Kaiming (He) uniform initialization.
///
/// Fills with values from U(-bound, bound) where bound = sqrt(6 / fan_in).
pub fn kaiming_uniform(shape: Vec<usize>, fan_in: usize, seed: u64) -> Result<Tensor, ModelError> {
    let bound = (6.0 / fan_in as f32).sqrt();
    let n: usize = shape.iter().product();
    let mut rng = Rng::new(seed);
    let data: Vec<f32> = (0..n).map(|_| rng.uniform_range(-bound, bound)).collect();
    Ok(Tensor::from_vec(shape, data)?)
}

/// Kaiming (He) normal initialization.
///
/// Fills with values from N(0, std) where std = sqrt(2 / fan_in).
pub fn kaiming_normal(shape: Vec<usize>, fan_in: usize, seed: u64) -> Result<Tensor, ModelError> {
    let std = (2.0 / fan_in as f32).sqrt();
    let n: usize = shape.iter().product();
    let mut rng = Rng::new(seed);
    let data: Vec<f32> = (0..n).map(|_| rng.normal() * std).collect();
    Ok(Tensor::from_vec(shape, data)?)
}

/// Xavier (Glorot) uniform initialization.
///
/// Fills with values from U(-bound, bound) where bound = sqrt(6 / (fan_in + fan_out)).
pub fn xavier_uniform(
    shape: Vec<usize>,
    fan_in: usize,
    fan_out: usize,
    seed: u64,
) -> Result<Tensor, ModelError> {
    let bound = (6.0 / (fan_in + fan_out) as f32).sqrt();
    let n: usize = shape.iter().product();
    let mut rng = Rng::new(seed);
    let data: Vec<f32> = (0..n).map(|_| rng.uniform_range(-bound, bound)).collect();
    Ok(Tensor::from_vec(shape, data)?)
}

/// Xavier (Glorot) normal initialization.
///
/// Fills with values from N(0, std) where std = sqrt(2 / (fan_in + fan_out)).
pub fn xavier_normal(
    shape: Vec<usize>,
    fan_in: usize,
    fan_out: usize,
    seed: u64,
) -> Result<Tensor, ModelError> {
    let std = (2.0 / (fan_in + fan_out) as f32).sqrt();
    let n: usize = shape.iter().product();
    let mut rng = Rng::new(seed);
    let data: Vec<f32> = (0..n).map(|_| rng.normal() * std).collect();
    Ok(Tensor::from_vec(shape, data)?)
}

/// Orthogonal initialization via QR decomposition (simplified Gram-Schmidt).
///
/// Creates a matrix of shape `[rows, cols]` with orthonormal rows (or columns).
pub fn orthogonal(rows: usize, cols: usize, seed: u64) -> Result<Tensor, ModelError> {
    let n = rows.max(cols);
    let mut rng = Rng::new(seed);

    // Generate random matrix
    let mut mat: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..n).map(|_| rng.normal()).collect())
        .collect();

    // Modified Gram-Schmidt QR
    for i in 0..n {
        // Normalize column i
        let norm: f32 = (0..n).map(|r| mat[r][i] * mat[r][i]).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for r in 0..n {
                mat[r][i] /= norm;
            }
        }
        // Orthogonalize remaining columns
        for j in (i + 1)..n {
            let dot: f32 = (0..n).map(|r| mat[r][i] * mat[r][j]).sum();
            for r in 0..n {
                mat[r][j] -= dot * mat[r][i];
            }
        }
    }

    // Extract [rows, cols] submatrix from Q
    let mut data = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            data.push(mat[r][c]);
        }
    }
    Ok(Tensor::from_vec(vec![rows, cols], data)?)
}

/// Fill a tensor with a constant value.
pub fn constant(shape: Vec<usize>, value: f32) -> Result<Tensor, ModelError> {
    Ok(Tensor::filled(shape, value)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kaiming_uniform_shape_and_bounds() {
        let t = kaiming_uniform(vec![64, 32], 32, 42).unwrap();
        assert_eq!(t.shape(), &[64, 32]);
        let bound = (6.0f32 / 32.0).sqrt();
        for &v in t.data() {
            assert!(v >= -bound && v <= bound, "value {v} out of bounds");
        }
    }

    #[test]
    fn kaiming_normal_shape() {
        let t = kaiming_normal(vec![128, 64], 64, 42).unwrap();
        assert_eq!(t.shape(), &[128, 64]);
        // Check mean is roughly 0
        let mean: f32 = t.data().iter().sum::<f32>() / t.data().len() as f32;
        assert!(mean.abs() < 0.1, "mean {mean} too far from 0");
    }

    #[test]
    fn xavier_uniform_shape_and_bounds() {
        let t = xavier_uniform(vec![100, 50], 100, 50, 42).unwrap();
        assert_eq!(t.shape(), &[100, 50]);
        let bound = (6.0f32 / 150.0).sqrt();
        for &v in t.data() {
            assert!(v >= -bound && v <= bound);
        }
    }

    #[test]
    fn xavier_normal_shape() {
        let t = xavier_normal(vec![100, 50], 100, 50, 42).unwrap();
        assert_eq!(t.shape(), &[100, 50]);
        let mean: f32 = t.data().iter().sum::<f32>() / t.data().len() as f32;
        assert!(mean.abs() < 0.1);
    }

    #[test]
    fn orthogonal_produces_orthonormal_columns() {
        let t = orthogonal(4, 4, 42).unwrap();
        let d = t.data();
        // Check column 0 has unit norm
        let norm: f32 = (0..4).map(|r| d[r * 4] * d[r * 4]).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "col 0 norm = {norm}");
        // Check col 0 dot col 1 ≈ 0
        let dot: f32 = (0..4).map(|r| d[r * 4] * d[r * 4 + 1]).sum();
        assert!(dot.abs() < 1e-4, "dot = {dot}");
    }

    #[test]
    fn constant_fills_with_value() {
        let t = constant(vec![2, 3], 7.0).unwrap();
        assert_eq!(t.data(), &[7.0; 6]);
    }
}
