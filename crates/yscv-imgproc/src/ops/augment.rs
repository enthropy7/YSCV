//! Image augmentation transforms (random crop, flip, rotation, color jitter, etc.).

use yscv_tensor::Tensor;

use super::super::ImgProcError;
use super::super::shape::hwc_shape;

/// Simple deterministic PRNG (xorshift64).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0xDEAD_BEEF } else { seed })
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    fn uniform_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.uniform()
    }

    fn normal(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-10);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

/// Randomly crop a `[H, W, C]` image to `(out_h, out_w)`.
pub fn random_crop(
    image: &Tensor,
    out_h: usize,
    out_w: usize,
    seed: u64,
) -> Result<Tensor, ImgProcError> {
    let (h, w, c) = hwc_shape(image)?;
    if out_h > h || out_w > w {
        return Err(ImgProcError::InvalidSize {
            height: out_h,
            width: out_w,
        });
    }
    let mut rng = Rng::new(seed);
    let y0 = (rng.uniform() * (h - out_h + 1) as f32) as usize;
    let x0 = (rng.uniform() * (w - out_w + 1) as f32) as usize;

    let data = image.data();
    let mut out = vec![0.0f32; out_h * out_w * c];
    for y in 0..out_h {
        let src_start = ((y0 + y) * w + x0) * c;
        let dst_start = (y * out_w) * c;
        out[dst_start..dst_start + out_w * c]
            .copy_from_slice(&data[src_start..src_start + out_w * c]);
    }
    Ok(Tensor::from_vec(vec![out_h, out_w, c], out)?)
}

/// Randomly flip horizontally with probability `p`.
pub fn random_horizontal_flip(image: &Tensor, p: f32, seed: u64) -> Result<Tensor, ImgProcError> {
    let mut rng = Rng::new(seed);
    if rng.uniform() >= p {
        return Ok(image.clone());
    }
    let (h, w, c) = hwc_shape(image)?;
    let data = image.data();
    let mut out = vec![0.0f32; h * w * c];
    for y in 0..h {
        for x in 0..w {
            let src = (y * w + (w - 1 - x)) * c;
            let dst = (y * w + x) * c;
            out[dst..dst + c].copy_from_slice(&data[src..src + c]);
        }
    }
    Ok(Tensor::from_vec(vec![h, w, c], out)?)
}

/// Randomly flip vertically with probability `p`.
pub fn random_vertical_flip(image: &Tensor, p: f32, seed: u64) -> Result<Tensor, ImgProcError> {
    let mut rng = Rng::new(seed);
    if rng.uniform() >= p {
        return Ok(image.clone());
    }
    let (h, w, c) = hwc_shape(image)?;
    let data = image.data();
    let mut out = vec![0.0f32; h * w * c];
    for y in 0..h {
        let src_row = (h - 1 - y) * w * c;
        let dst_row = y * w * c;
        out[dst_row..dst_row + w * c].copy_from_slice(&data[src_row..src_row + w * c]);
    }
    Ok(Tensor::from_vec(vec![h, w, c], out)?)
}

/// Rotate image by a random angle in `[-max_degrees, max_degrees]`.
///
/// Uses bilinear interpolation. Pixels outside the image are filled with 0.
pub fn random_rotation(
    image: &Tensor,
    max_degrees: f32,
    seed: u64,
) -> Result<Tensor, ImgProcError> {
    let (h, w, c) = hwc_shape(image)?;
    let mut rng = Rng::new(seed);
    let angle_deg = rng.uniform_range(-max_degrees, max_degrees);
    let angle = angle_deg * std::f32::consts::PI / 180.0;
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;

    let data = image.data();
    let mut out = vec![0.0f32; h * w * c];

    for y in 0..h {
        for x in 0..w {
            // Inverse rotation to find source pixel
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let src_x = cos_a * dx + sin_a * dy + cx;
            let src_y = -sin_a * dx + cos_a * dy + cy;

            if src_x >= 0.0 && src_x < (w - 1) as f32 && src_y >= 0.0 && src_y < (h - 1) as f32 {
                let x0 = src_x.floor() as usize;
                let y0 = src_y.floor() as usize;
                let x1 = x0 + 1;
                let y1 = y0 + 1;
                let fx = src_x - x0 as f32;
                let fy = src_y - y0 as f32;

                for ch in 0..c {
                    let v00 = data[(y0 * w + x0) * c + ch];
                    let v10 = data[(y0 * w + x1) * c + ch];
                    let v01 = data[(y1 * w + x0) * c + ch];
                    let v11 = data[(y1 * w + x1) * c + ch];
                    out[(y * w + x) * c + ch] = v00 * (1.0 - fx) * (1.0 - fy)
                        + v10 * fx * (1.0 - fy)
                        + v01 * (1.0 - fx) * fy
                        + v11 * fx * fy;
                }
            }
        }
    }
    Ok(Tensor::from_vec(vec![h, w, c], out)?)
}

/// Random erasing (cutout): randomly erase a rectangular region with probability `p`.
///
/// The erased region is filled with `fill_value` (typically 0.0).
pub fn random_erasing(
    image: &Tensor,
    p: f32,
    scale_min: f32,
    scale_max: f32,
    ratio_min: f32,
    ratio_max: f32,
    fill_value: f32,
    seed: u64,
) -> Result<Tensor, ImgProcError> {
    let (h, w, c) = hwc_shape(image)?;
    let mut rng = Rng::new(seed);
    if rng.uniform() >= p {
        return Ok(image.clone());
    }

    let area = (h * w) as f32;
    let target_area = area * rng.uniform_range(scale_min, scale_max);
    let ratio = rng.uniform_range(ratio_min, ratio_max);
    let eh = (target_area * ratio).sqrt() as usize;
    let ew = (target_area / ratio).sqrt() as usize;
    let eh = eh.min(h);
    let ew = ew.min(w);

    let y0 = (rng.uniform() * (h - eh + 1) as f32) as usize;
    let x0 = (rng.uniform() * (w - ew + 1) as f32) as usize;

    let mut out = image.data().to_vec();
    for y in y0..y0 + eh {
        for x in x0..x0 + ew {
            let base = (y * w + x) * c;
            for ch in 0..c {
                out[base + ch] = fill_value;
            }
        }
    }
    Ok(Tensor::from_vec(vec![h, w, c], out)?)
}

/// Apply random color jitter: brightness, contrast, saturation, hue adjustments.
///
/// Each factor is randomized in `[1-amount, 1+amount]`.
pub fn color_jitter(
    image: &Tensor,
    brightness: f32,
    contrast: f32,
    saturation: f32,
    hue: f32,
    seed: u64,
) -> Result<Tensor, ImgProcError> {
    let (h, w, c) = hwc_shape(image)?;
    if c != 3 {
        return Err(ImgProcError::InvalidChannelCount {
            expected: 3,
            got: c,
        });
    }
    let mut rng = Rng::new(seed);
    let data = image.data();
    let mut out = data.to_vec();

    // Brightness
    if brightness > 0.0 {
        let factor = rng
            .uniform_range(1.0 - brightness, 1.0 + brightness)
            .max(0.0);
        for v in out.iter_mut() {
            *v *= factor;
        }
    }

    // Contrast
    if contrast > 0.0 {
        let factor = rng.uniform_range(1.0 - contrast, 1.0 + contrast).max(0.0);
        let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
        for v in out.iter_mut() {
            *v = (*v - mean) * factor + mean;
        }
    }

    // Saturation
    if saturation > 0.0 {
        let factor = rng
            .uniform_range(1.0 - saturation, 1.0 + saturation)
            .max(0.0);
        for i in 0..(h * w) {
            let base = i * 3;
            let gray = 0.299 * out[base] + 0.587 * out[base + 1] + 0.114 * out[base + 2];
            for ch in 0..3 {
                out[base + ch] = gray + (out[base + ch] - gray) * factor;
            }
        }
    }

    // Hue (simple rotation in RGB space approximation)
    if hue > 0.0 {
        let angle = rng.uniform_range(-hue, hue) * std::f32::consts::PI;
        let cos_h = angle.cos();
        let sin_h = angle.sin();
        let sqrt3 = 3.0f32.sqrt();
        for i in 0..(h * w) {
            let base = i * 3;
            let r = out[base];
            let g = out[base + 1];
            let b = out[base + 2];
            out[base] =
                (r + g + b) / 3.0 + (2.0 * r - g - b) / 3.0 * cos_h + (g - b) / sqrt3 * sin_h;
            out[base + 1] = (r + g + b) / 3.0 - (2.0 * r - g - b) / 6.0 * cos_h
                + (2.0 * b - 2.0 * g + 2.0 * r - g - b) / (2.0 * sqrt3) * sin_h;
            // simplified: recompute b from the constraint r+g+b is preserved
            out[base + 2] = r + g + b - out[base] - out[base + 1];
        }
    }

    // Clamp to [0, 1]
    for v in out.iter_mut() {
        *v = v.clamp(0.0, 1.0);
    }

    Ok(Tensor::from_vec(vec![h, w, c], out)?)
}

/// Add Gaussian noise to an image.
///
/// Each pixel channel is offset by `N(0, sigma)` sampled independently.
/// Output is clamped to `[0, 1]`.
pub fn gaussian_noise(image: &Tensor, sigma: f32, seed: u64) -> Result<Tensor, ImgProcError> {
    let (h, w, c) = hwc_shape(image)?;
    let data = image.data();
    let total = h * w * c;
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(total);
    for i in 0..total {
        let noisy = data[i] + sigma * rng.normal();
        out.push(noisy.clamp(0.0, 1.0));
    }
    Ok(Tensor::from_vec(vec![h, w, c], out)?)
}

/// Elastic distortion transform.
///
/// Generates random displacement fields and applies Gaussian smoothing,
/// then remaps pixels with bilinear interpolation.
pub fn elastic_transform(
    image: &Tensor,
    alpha: f32,
    sigma: f32,
    seed: u64,
) -> Result<Tensor, ImgProcError> {
    let (h, w, c) = hwc_shape(image)?;
    let mut rng = Rng::new(seed);

    // Generate random displacement fields [-1, 1]
    let n = h * w;
    let mut dx: Vec<f32> = (0..n).map(|_| rng.uniform_range(-1.0, 1.0)).collect();
    let mut dy: Vec<f32> = (0..n).map(|_| rng.uniform_range(-1.0, 1.0)).collect();

    // Gaussian blur the displacement fields (simple box blur approximation)
    let kernel_size = (sigma * 3.0) as usize | 1; // ensure odd
    let half_k = kernel_size / 2;
    for _ in 0..2 {
        // 2 passes of box blur ≈ gaussian
        let dx_copy = dx.clone();
        let dy_copy = dy.clone();
        for y in 0..h {
            for x in 0..w {
                let mut sx = 0.0f32;
                let mut sy = 0.0f32;
                let mut count = 0.0f32;
                for ky in y.saturating_sub(half_k)..=(y + half_k).min(h - 1) {
                    for kx in x.saturating_sub(half_k)..=(x + half_k).min(w - 1) {
                        sx += dx_copy[ky * w + kx];
                        sy += dy_copy[ky * w + kx];
                        count += 1.0;
                    }
                }
                dx[y * w + x] = sx / count;
                dy[y * w + x] = sy / count;
            }
        }
    }

    // Scale by alpha
    for v in dx.iter_mut() {
        *v *= alpha;
    }
    for v in dy.iter_mut() {
        *v *= alpha;
    }

    // Remap with bilinear interpolation
    let data = image.data();
    let mut out = vec![0.0f32; h * w * c];
    for y in 0..h {
        for x in 0..w {
            let src_x = x as f32 + dx[y * w + x];
            let src_y = y as f32 + dy[y * w + x];

            if src_x >= 0.0 && src_x < (w - 1) as f32 && src_y >= 0.0 && src_y < (h - 1) as f32 {
                let x0 = src_x.floor() as usize;
                let y0 = src_y.floor() as usize;
                let x1 = x0 + 1;
                let y1 = y0 + 1;
                let fx = src_x - x0 as f32;
                let fy = src_y - y0 as f32;
                for ch in 0..c {
                    let v00 = data[(y0 * w + x0) * c + ch];
                    let v10 = data[(y0 * w + x1) * c + ch];
                    let v01 = data[(y1 * w + x0) * c + ch];
                    let v11 = data[(y1 * w + x1) * c + ch];
                    out[(y * w + x) * c + ch] = v00 * (1.0 - fx) * (1.0 - fy)
                        + v10 * fx * (1.0 - fy)
                        + v01 * (1.0 - fx) * fy
                        + v11 * fx * fy;
                }
            }
        }
    }
    Ok(Tensor::from_vec(vec![h, w, c], out)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_image(h: usize, w: usize) -> Tensor {
        let data: Vec<f32> = (0..h * w * 3)
            .map(|i| (i as f32) / (h * w * 3) as f32)
            .collect();
        Tensor::from_vec(vec![h, w, 3], data).unwrap()
    }

    #[test]
    fn test_random_crop_shape() {
        let img = test_image(32, 32);
        let cropped = random_crop(&img, 16, 16, 42).unwrap();
        assert_eq!(cropped.shape(), &[16, 16, 3]);
    }

    #[test]
    fn test_random_horizontal_flip() {
        let img = Tensor::from_vec(vec![1, 3, 1], vec![1.0, 2.0, 3.0]).unwrap();
        let flipped = random_horizontal_flip(&img, 1.0, 42).unwrap();
        assert_eq!(flipped.data(), &[3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_random_vertical_flip() {
        let img = Tensor::from_vec(vec![3, 1, 1], vec![1.0, 2.0, 3.0]).unwrap();
        let flipped = random_vertical_flip(&img, 1.0, 42).unwrap();
        assert_eq!(flipped.data(), &[3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_random_rotation_preserves_shape() {
        let img = test_image(20, 20);
        let rotated = random_rotation(&img, 30.0, 42).unwrap();
        assert_eq!(rotated.shape(), &[20, 20, 3]);
    }

    #[test]
    fn test_random_erasing_modifies_pixels() {
        let img = Tensor::from_vec(vec![10, 10, 3], vec![1.0f32; 300]).unwrap();
        let erased = random_erasing(&img, 1.0, 0.1, 0.3, 0.5, 2.0, 0.0, 42).unwrap();
        let zeros = erased.data().iter().filter(|&&v| v == 0.0).count();
        assert!(zeros > 0, "expected some erased pixels");
    }

    #[test]
    fn test_color_jitter_preserves_shape() {
        let img = test_image(8, 8);
        let jittered = color_jitter(&img, 0.2, 0.2, 0.2, 0.0, 42).unwrap();
        assert_eq!(jittered.shape(), &[8, 8, 3]);
        // Values should be in [0, 1]
        for &v in jittered.data() {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn test_elastic_transform_preserves_shape() {
        let img = test_image(16, 16);
        let out = elastic_transform(&img, 10.0, 3.0, 42).unwrap();
        assert_eq!(out.shape(), &[16, 16, 3]);
    }
}
