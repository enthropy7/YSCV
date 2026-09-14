//! u8 image processing operations.
//!
//! These operate directly on `&[u8]` pixel data in HWC layout, giving ~4x throughput
//! over f32 paths (16 u8 per NEON register vs 4 f32).
#![allow(unsafe_code)]

// WHY 4096: below 4096 pixels, thread dispatch overhead (~1-3us) exceeds the compute saved by parallelism.
pub(crate) const RAYON_THRESHOLD: usize = 4096; // min pixels before we go parallel

/// Low-overhead parallel-for.
///
/// On macOS uses rayon (which meshes well with GCD), on other platforms uses
/// `std::thread::scope` for ~1us dispatch latency (vs rayon's 3-5us).
pub(crate) mod gcd {
    /// Execute `f(0), f(1), ..., f(n-1)` in parallel.
    #[inline]
    #[cfg(target_os = "macos")]
    pub fn parallel_for<F: Fn(usize) + Sync + Send>(n: usize, f: F) {
        if n <= 1 {
            for i in 0..n {
                f(i);
            }
            return;
        }
        use rayon::prelude::*;
        (0..n).into_par_iter().for_each(f);
    }

    /// Execute `f(0), f(1), ..., f(n-1)` in parallel using `std::thread::scope`.
    ///
    /// Lower dispatch overhead than rayon (~1us vs 3-5us) which matters for
    /// small-to-medium workloads common in image processing.
    #[inline]
    #[cfg(not(target_os = "macos"))]
    pub fn parallel_for<F: Fn(usize) + Sync>(n: usize, f: F) {
        let cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        if cpus <= 1 || n <= 1 {
            for i in 0..n {
                f(i);
            }
            return;
        }
        let threads = cpus.min(n);
        let chunk = n.div_ceil(threads);
        std::thread::scope(|s| {
            for t in 0..threads {
                let start = t * chunk;
                let end = (start + chunk).min(n);
                if start >= end {
                    continue;
                }
                let f = &f;
                s.spawn(move || {
                    for i in start..end {
                        f(i);
                    }
                });
            }
        });
    }
}

/// Simple u8 image wrapper (HWC layout, row-major).
#[derive(Clone, Debug)]
pub struct ImageU8 {
    data: Vec<u8>,
    height: usize,
    width: usize,
    channels: usize,
}

impl ImageU8 {
    /// Creates a new image from raw bytes. Returns `None` if length doesn't match.
    pub fn new(data: Vec<u8>, height: usize, width: usize, channels: usize) -> Option<Self> {
        if data.len() != height * width * channels {
            return None;
        }
        Some(Self {
            data,
            height,
            width,
            channels,
        })
    }

    /// Creates a zero-filled image.
    pub fn zeros(height: usize, width: usize, channels: usize) -> Self {
        Self {
            data: vec![0u8; height * width * channels],
            height,
            width,
            channels,
        }
    }

    pub fn data(&self) -> &[u8] {
        &self.data
    }
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }
    pub const fn height(&self) -> usize {
        self.height
    }
    pub const fn width(&self) -> usize {
        self.width
    }
    pub const fn channels(&self) -> usize {
        self.channels
    }
    pub const fn len(&self) -> usize {
        self.data.len()
    }
    pub const fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Convert from f32 Tensor `[H,W,C]` to u8 image (clamp to `[0,255]`).
    pub fn from_tensor(tensor: &yscv_tensor::Tensor) -> Option<Self> {
        let shape = tensor.shape();
        if shape.len() != 3 {
            return None;
        }
        let (h, w, c) = (shape[0], shape[1], shape[2]);
        let data: Vec<u8> = tensor
            .data()
            .iter()
            .map(|&v| (v * 255.0).clamp(0.0, 255.0) as u8)
            .collect();
        Some(Self {
            data,
            height: h,
            width: w,
            channels: c,
        })
    }

    /// Convert to f32 Tensor `[H,W,C]` with values in `[0,1]`.
    pub fn to_tensor(&self) -> yscv_tensor::Tensor {
        let data: Vec<f32> = self.data.iter().map(|&v| v as f32 / 255.0).collect();
        // Shape always matches data length for a valid ImageU8, so this cannot fail.
        yscv_tensor::Tensor::from_vec(vec![self.height, self.width, self.channels], data)
            .expect("ImageU8::to_tensor: shape mismatch (bug)")
    }
}

// ============================================================================
// ImageF32 — zero-overhead f32 image wrapper (HWC layout, row-major)
// ============================================================================

/// Simple f32 image wrapper (HWC layout, row-major). Zero overhead.
#[derive(Clone, Debug)]
pub struct ImageF32 {
    data: Vec<f32>,
    height: usize,
    width: usize,
    channels: usize,
}

impl ImageF32 {
    /// Creates a new image from raw f32 data. Returns `None` if length doesn't match.
    pub fn new(data: Vec<f32>, height: usize, width: usize, channels: usize) -> Option<Self> {
        if data.len() != height * width * channels {
            return None;
        }
        Some(Self {
            data,
            height,
            width,
            channels,
        })
    }

    /// Creates a zero-filled f32 image.
    pub fn zeros(height: usize, width: usize, channels: usize) -> Self {
        Self {
            data: vec![0.0f32; height * width * channels],
            height,
            width,
            channels,
        }
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }
    pub const fn height(&self) -> usize {
        self.height
    }
    pub const fn width(&self) -> usize {
        self.width
    }
    pub const fn channels(&self) -> usize {
        self.channels
    }
    pub const fn len(&self) -> usize {
        self.data.len()
    }
    pub const fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Convert to Tensor `[H,W,C]`.
    pub fn to_tensor(&self) -> yscv_tensor::Tensor {
        yscv_tensor::Tensor::from_vec(
            vec![self.height, self.width, self.channels],
            self.data.clone(),
        )
        .expect("ImageF32::to_tensor: shape mismatch")
    }

    /// Convert from Tensor `[H,W,C]`.
    pub fn from_tensor(tensor: &yscv_tensor::Tensor) -> Option<Self> {
        let shape = tensor.shape();
        if shape.len() != 3 {
            return None;
        }
        Self::new(tensor.data().to_vec(), shape[0], shape[1], shape[2])
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::super::f32_ops::*;
    use super::super::u8_canny::*;
    use super::super::u8_features::*;
    use super::super::u8_filters::*;
    use super::super::u8_resize::*;
    use super::*;

    #[test]
    fn test_grayscale_u8() {
        let mut data = vec![0u8; 4 * 4 * 3];
        data[0] = 255;
        data[1] = 255;
        data[2] = 255;
        let img = ImageU8::new(data, 4, 4, 3).unwrap();
        let gray = grayscale_u8(&img).unwrap();
        assert_eq!(gray.channels(), 1);
        assert_eq!(gray.height(), 4);
        assert_eq!(gray.width(), 4);
        assert!(gray.data()[0] >= 250, "got {}", gray.data()[0]);
        assert_eq!(gray.data()[1], 0);
    }

    #[test]
    fn test_dilate_u8() {
        let mut data = vec![0u8; 8 * 8];
        data[4 * 8 + 4] = 200;
        let img = ImageU8::new(data, 8, 8, 1).unwrap();
        let dilated = dilate_3x3_u8(&img).unwrap();
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let y = (4 + dy) as usize;
                let x = (4 + dx) as usize;
                assert_eq!(dilated.data()[y * 8 + x], 200, "at ({},{})", y, x);
            }
        }
    }

    #[test]
    fn test_erode_u8() {
        let data = vec![200u8; 8 * 8];
        let img = ImageU8::new(data, 8, 8, 1).unwrap();
        let eroded = erode_3x3_u8(&img).unwrap();
        assert_eq!(eroded.data()[4 * 8 + 4], 200);
    }

    #[test]
    fn test_gaussian_blur_u8() {
        let mut data = vec![128u8; 16 * 16];
        data[8 * 16 + 8] = 255;
        let img = ImageU8::new(data, 16, 16, 1).unwrap();
        let blurred = gaussian_blur_3x3_u8(&img).unwrap();
        assert_eq!(blurred.channels(), 1);
        let center = blurred.data()[8 * 16 + 8];
        assert!(center > 128 && center < 255, "got {}", center);
    }

    #[test]
    fn test_box_blur_u8() {
        let data = vec![100u8; 16 * 16];
        let img = ImageU8::new(data, 16, 16, 1).unwrap();
        let blurred = box_blur_3x3_u8(&img).unwrap();
        let center = blurred.data()[8 * 16 + 8];
        assert!((99..=101).contains(&center), "got {}", center);
    }

    #[test]
    fn test_sobel_u8() {
        let mut data = vec![0u8; 16 * 16];
        for y in 0..16 {
            for x in 8..16 {
                data[y * 16 + x] = 255;
            }
        }
        let img = ImageU8::new(data, 16, 16, 1).unwrap();
        let edges = sobel_3x3_magnitude_u8(&img).unwrap();
        let edge_val = edges.data()[8 * 16 + 8];
        assert!(edge_val > 100, "edge value = {}", edge_val);
        assert_eq!(edges.data()[8 * 16 + 2], 0);
    }

    #[test]
    fn test_image_u8_tensor_roundtrip() {
        let data = vec![128u8; 4 * 4 * 3];
        let img = ImageU8::new(data, 4, 4, 3).unwrap();
        let tensor = img.to_tensor();
        assert_eq!(tensor.shape(), &[4, 4, 3]);
        let back = ImageU8::from_tensor(&tensor).unwrap();
        for (a, b) in img.data().iter().zip(back.data().iter()) {
            assert!((*a as i16 - *b as i16).unsigned_abs() <= 1);
        }
    }

    #[test]
    fn test_median_blur_u8() {
        // Uniform image should stay the same
        let data = vec![100u8; 16 * 16];
        let img = ImageU8::new(data, 16, 16, 1).unwrap();
        let blurred = median_blur_3x3_u8(&img).unwrap();
        let center = blurred.data()[8 * 16 + 8];
        assert_eq!(center, 100);

        // Salt & pepper: single outlier should be removed by median
        let mut data2 = vec![128u8; 16 * 16];
        data2[8 * 16 + 8] = 255; // outlier
        let img2 = ImageU8::new(data2, 16, 16, 1).unwrap();
        let blurred2 = median_blur_3x3_u8(&img2).unwrap();
        // Median of 8x128 + 1x255 = 128
        assert_eq!(blurred2.data()[8 * 16 + 8], 128);
    }

    #[test]
    fn test_canny_u8() {
        // Vertical edge
        let mut data = vec![0u8; 32 * 32];
        for y in 0..32 {
            for x in 16..32 {
                data[y * 32 + x] = 255;
            }
        }
        let img = ImageU8::new(data, 32, 32, 1).unwrap();
        let edges = canny_u8(&img, 30, 100).unwrap();
        assert_eq!(edges.channels(), 1);
        // Should have some edge pixels
        let edge_count: usize = edges.data().iter().filter(|&&v| v == 255).count();
        assert!(edge_count > 5, "too few edges: {}", edge_count);
        // Interior of uniform region should be 0
        assert_eq!(edges.data()[16 * 32 + 2], 0);
    }

    #[test]
    fn test_resize_bilinear_u8() {
        // Simple downscale
        let data = vec![128u8; 16 * 16];
        let img = ImageU8::new(data, 16, 16, 1).unwrap();
        let resized = resize_bilinear_u8(&img, 8, 8).unwrap();
        assert_eq!(resized.height(), 8);
        assert_eq!(resized.width(), 8);
        assert_eq!(resized.channels(), 1);
        // All pixels should be 128 for uniform input
        for &v in resized.data() {
            assert_eq!(v, 128);
        }

        // Upscale RGB
        let data3 = vec![100u8; 4 * 4 * 3];
        let img3 = ImageU8::new(data3, 4, 4, 3).unwrap();
        let up = resize_bilinear_u8(&img3, 8, 8).unwrap();
        assert_eq!(up.height(), 8);
        assert_eq!(up.width(), 8);
        assert_eq!(up.channels(), 3);
        for &v in up.data() {
            assert_eq!(v, 100);
        }
    }

    #[test]
    fn test_resize_nearest_u8() {
        // Uniform downscale
        let data = vec![128u8; 16 * 16];
        let img = ImageU8::new(data, 16, 16, 1).unwrap();
        let resized = resize_nearest_u8(&img, 8, 8).unwrap();
        assert_eq!(resized.height(), 8);
        assert_eq!(resized.width(), 8);
        assert_eq!(resized.channels(), 1);
        for &v in resized.data() {
            assert_eq!(v, 128);
        }

        // Upscale: each pixel should be replicated
        let data4: Vec<u8> = (0..16).collect();
        let img4 = ImageU8::new(data4, 4, 4, 1).unwrap();
        let up = resize_nearest_u8(&img4, 8, 8).unwrap();
        assert_eq!(up.height(), 8);
        assert_eq!(up.width(), 8);
        // Top-left 2x2 block should all be pixel (0,0) = 0
        assert_eq!(up.data()[0], 0);
        assert_eq!(up.data()[1], 0);
        assert_eq!(up.data()[8], 0);
        assert_eq!(up.data()[9], 0);

        // RGB uniform
        let data3 = vec![100u8; 4 * 4 * 3];
        let img3 = ImageU8::new(data3, 4, 4, 3).unwrap();
        let up3 = resize_nearest_u8(&img3, 8, 8).unwrap();
        assert_eq!(up3.height(), 8);
        assert_eq!(up3.width(), 8);
        assert_eq!(up3.channels(), 3);
        for &v in up3.data() {
            assert_eq!(v, 100);
        }

        // Zero target returns None
        assert!(resize_nearest_u8(&img, 0, 8).is_none());
        assert!(resize_nearest_u8(&img, 8, 0).is_none());
    }

    // ========================================================================
    // ImageF32 and f32 ops tests
    // ========================================================================

    #[test]
    fn test_image_f32_new() {
        let img = ImageF32::new(vec![1.0; 12], 2, 2, 3).unwrap();
        assert_eq!(img.height(), 2);
        assert_eq!(img.width(), 2);
        assert_eq!(img.channels(), 3);
        assert_eq!(img.len(), 12);
        assert!(!img.is_empty());
        // Mismatched length returns None
        assert!(ImageF32::new(vec![1.0; 10], 2, 2, 3).is_none());
    }

    #[test]
    fn test_image_f32_tensor_roundtrip() {
        let data: Vec<f32> = (0..48).map(|i| i as f32 / 48.0).collect();
        let img = ImageF32::new(data.clone(), 4, 4, 3).unwrap();
        let tensor = img.to_tensor();
        assert_eq!(tensor.shape(), &[4, 4, 3]);
        let back = ImageF32::from_tensor(&tensor).unwrap();
        assert_eq!(back.data(), img.data());
    }

    #[test]
    fn test_grayscale_f32_known_values() {
        // Pure white: 0.299*1 + 0.587*1 + 0.114*1 = 1.0
        let data = vec![1.0f32; 4 * 4 * 3];
        let img = ImageF32::new(data, 4, 4, 3).unwrap();
        let gray = grayscale_f32(&img).unwrap();
        assert_eq!(gray.channels(), 1);
        assert_eq!(gray.height(), 4);
        assert_eq!(gray.width(), 4);
        for &v in gray.data() {
            assert!((v - 1.0).abs() < 0.01, "white pixel gray = {}", v);
        }

        // Pure red: 0.299*1 + 0.587*0 + 0.114*0 = 0.299
        let mut red_data = vec![0.0f32; 4 * 4 * 3];
        for i in 0..16 {
            red_data[i * 3] = 1.0;
        }
        let red_img = ImageF32::new(red_data, 4, 4, 3).unwrap();
        let red_gray = grayscale_f32(&red_img).unwrap();
        for &v in red_gray.data() {
            assert!((v - 0.299).abs() < 0.01, "red pixel gray = {}", v);
        }

        // Wrong channel count returns None
        let gray_img = ImageF32::zeros(4, 4, 1);
        assert!(grayscale_f32(&gray_img).is_none());
    }

    #[test]
    fn test_gaussian_blur_f32_uniform() {
        // Uniform image should be unchanged by gaussian blur.
        // Use a larger image to exercise both SIMD and scalar paths.
        let data = vec![0.5f32; 32 * 32];
        let img = ImageF32::new(data, 32, 32, 1).unwrap();
        let blurred = gaussian_blur_3x3_f32(&img).unwrap();
        // Check interior pixels (borders may differ due to clamping)
        for y in 1..31 {
            for x in 1..31 {
                let v = blurred.data()[y * 32 + x];
                assert!(
                    (v - 0.5).abs() < 1e-4,
                    "uniform gaussian at ({},{}) got {}",
                    x,
                    y,
                    v
                );
            }
        }
    }

    #[test]
    fn test_gaussian_blur_f32_smoothing() {
        let mut data = vec![0.5f32; 16 * 16];
        data[8 * 16 + 8] = 1.0; // spike
        let img = ImageF32::new(data, 16, 16, 1).unwrap();
        let blurred = gaussian_blur_3x3_f32(&img).unwrap();
        let center = blurred.data()[8 * 16 + 8];
        assert!(center > 0.5 && center < 1.0, "gaussian center = {}", center);
    }

    #[test]
    fn test_box_blur_f32_uniform() {
        let data = vec![0.75f32; 32 * 32];
        let img = ImageF32::new(data, 32, 32, 1).unwrap();
        let blurred = box_blur_3x3_f32(&img).unwrap();
        // Check interior pixels (borders may differ due to clamping)
        for y in 1..31 {
            for x in 1..31 {
                let v = blurred.data()[y * 32 + x];
                assert!(
                    (v - 0.75).abs() < 1e-4,
                    "uniform box at ({},{}) got {}",
                    x,
                    y,
                    v
                );
            }
        }
    }

    #[test]
    fn test_dilate_f32_known_pattern() {
        // Single bright pixel should spread to 3x3
        let mut data = vec![0.0f32; 8 * 8];
        data[4 * 8 + 4] = 0.9;
        let img = ImageF32::new(data, 8, 8, 1).unwrap();
        let dilated = dilate_3x3_f32(&img).unwrap();
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let y = (4 + dy) as usize;
                let x = (4 + dx) as usize;
                assert!(
                    (dilated.data()[y * 8 + x] - 0.9).abs() < 1e-6,
                    "dilate at ({},{}) = {}",
                    y,
                    x,
                    dilated.data()[y * 8 + x]
                );
            }
        }
        // Far corner should still be 0
        assert!((dilated.data()[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_sobel_f32_flat_is_zero() {
        // Flat image: all gradients should be zero
        let data = vec![0.5f32; 16 * 16];
        let img = ImageF32::new(data, 16, 16, 1).unwrap();
        let edges = sobel_3x3_f32(&img).unwrap();
        // Interior pixels should be zero (borders are always zero)
        for y in 1..15 {
            for x in 1..15 {
                assert!(
                    (edges.data()[y * 16 + x]).abs() < 1e-5,
                    "sobel flat at ({},{}) = {}",
                    y,
                    x,
                    edges.data()[y * 16 + x]
                );
            }
        }
    }

    #[test]
    fn test_sobel_f32_edge_detection() {
        // Vertical step edge at x=8
        let mut data = vec![0.0f32; 16 * 16];
        for y in 0..16 {
            for x in 8..16 {
                data[y * 16 + x] = 1.0;
            }
        }
        let img = ImageF32::new(data, 16, 16, 1).unwrap();
        let edges = sobel_3x3_f32(&img).unwrap();
        // Edge magnitude at (8,8) should be nonzero
        let edge_val = edges.data()[8 * 16 + 8];
        assert!(edge_val > 0.5, "sobel edge = {}", edge_val);
        // Interior uniform region should be ~0
        assert!(edges.data()[8 * 16 + 2].abs() < 1e-5);
    }

    #[test]
    fn test_threshold_binary_f32() {
        let data = vec![0.0, 0.3, 0.5, 0.7, 0.8, 1.0, 0.1, 0.6, 0.9];
        let img = ImageF32::new(data, 3, 3, 1).unwrap();
        let result = threshold_binary_f32(&img, 0.5, 1.0).unwrap();
        let expected = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0];
        for (i, (&got, &exp)) in result.data().iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "threshold at {} = {}, expected {}",
                i,
                got,
                exp
            );
        }
    }

    // ====================================================================
    // FAST-9 tests
    // ====================================================================

    #[test]
    fn test_fast9_detect_u8_empty() {
        // Uniform image: no corners
        let img = ImageU8::zeros(32, 32, 1);
        let corners = fast9_detect_u8(&img, 20, false);
        assert!(
            corners.is_empty(),
            "uniform image should have 0 corners, got {}",
            corners.len()
        );
    }

    #[test]
    fn test_fast9_detect_u8_bright_spot() {
        let mut data = vec![0u8; 32 * 32];
        data[16 * 32 + 16] = 255;
        let img = ImageU8::new(data, 32, 32, 1).unwrap();
        let corners = fast9_detect_u8(&img, 20, false);
        assert!(
            !corners.is_empty(),
            "bright spot should produce at least one corner"
        );
    }

    #[test]
    fn test_fast9_detect_u8_too_small() {
        let img = ImageU8::zeros(5, 5, 1);
        let corners = fast9_detect_u8(&img, 20, false);
        assert!(corners.is_empty());
    }

    #[test]
    fn test_fast9_detect_u8_wrong_channels() {
        let img = ImageU8::zeros(32, 32, 3);
        let corners = fast9_detect_u8(&img, 20, false);
        assert!(corners.is_empty());
    }

    #[test]
    fn test_fast9_detect_u8_nms_reduces() {
        let mut data = vec![50u8; 64 * 64];
        for i in 0..5 {
            let y = 10 + i * 10;
            let x = 10 + i * 10;
            data[y * 64 + x] = 255;
        }
        let img = ImageU8::new(data, 64, 64, 1).unwrap();
        let no_nms = fast9_detect_u8(&img, 20, false);
        let with_nms = fast9_detect_u8(&img, 20, true);
        assert!(
            with_nms.len() <= no_nms.len(),
            "NMS should not increase corners: {} > {}",
            with_nms.len(),
            no_nms.len()
        );
    }

    // ====================================================================
    // Distance transform tests
    // ====================================================================

    #[test]
    fn test_distance_transform_u8_all_nonzero() {
        let data = vec![255u8; 16 * 16];
        let img = ImageU8::new(data, 16, 16, 1).unwrap();
        let dt = distance_transform_u8(&img);
        assert_eq!(dt.len(), 256);
        for &d in &dt {
            assert_eq!(d, 0);
        }
    }

    #[test]
    fn test_distance_transform_u8_single_source() {
        let mut data = vec![0u8; 16 * 16];
        data[8 * 16 + 8] = 255;
        let img = ImageU8::new(data, 16, 16, 1).unwrap();
        let dt = distance_transform_u8(&img);
        assert_eq!(dt[8 * 16 + 8], 0);
        assert_eq!(dt[8 * 16 + 9], 1);
        assert_eq!(dt[7 * 16 + 8], 1);
        assert_eq!(dt[7 * 16 + 7], 2);
        assert_eq!(dt[0], 16);
    }

    #[test]
    fn test_distance_transform_u8_wrong_channels() {
        let img = ImageU8::zeros(16, 16, 3);
        let dt = distance_transform_u8(&img);
        assert!(dt.is_empty());
    }

    #[test]
    fn test_distance_transform_u8_row_edge() {
        let mut data = vec![0u8; 8 * 8];
        data[4 * 8] = 255;
        let img = ImageU8::new(data, 8, 8, 1).unwrap();
        let dt = distance_transform_u8(&img);
        assert_eq!(dt[4 * 8], 0);
        assert_eq!(dt[4 * 8 + 1], 1);
        assert_eq!(dt[4 * 8 + 2], 2);
    }

    // ====================================================================
    // Warp perspective tests
    // ====================================================================

    #[test]
    fn test_warp_perspective_u8_identity() {
        let mut data = vec![0u8; 16 * 16];
        data[8 * 16 + 8] = 200;
        let img = ImageU8::new(data.clone(), 16, 16, 1).unwrap();
        let identity: [f64; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let result = warp_perspective_u8(&img, &identity, 16, 16);
        assert_eq!(result.height(), 16);
        assert_eq!(result.width(), 16);
        assert_eq!(result.channels(), 1);
        assert_eq!(result.data()[8 * 16 + 8], 200);
    }

    #[test]
    fn test_warp_perspective_u8_translation() {
        let mut data = vec![0u8; 16 * 16];
        data[8 * 16 + 8] = 200;
        let img = ImageU8::new(data, 16, 16, 1).unwrap();
        let h_translate: [f64; 9] = [1.0, 0.0, 2.0, 0.0, 1.0, 3.0, 0.0, 0.0, 1.0];
        let result = warp_perspective_u8(&img, &h_translate, 16, 16);
        assert_eq!(result.data()[5 * 16 + 6], 200);
    }

    #[test]
    fn test_warp_perspective_u8_out_of_bounds() {
        let data = vec![128u8; 8 * 8];
        let img = ImageU8::new(data, 8, 8, 1).unwrap();
        let h_big: [f64; 9] = [1.0, 0.0, 1000.0, 0.0, 1.0, 1000.0, 0.0, 0.0, 1.0];
        let result = warp_perspective_u8(&img, &h_big, 8, 8);
        for &v in result.data() {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn test_warp_perspective_u8_different_output_size() {
        let data = vec![100u8; 8 * 8];
        let img = ImageU8::new(data, 8, 8, 1).unwrap();
        let identity: [f64; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let result = warp_perspective_u8(&img, &identity, 4, 4);
        assert_eq!(result.height(), 4);
        assert_eq!(result.width(), 4);
        assert_eq!(result.data()[2 * 4 + 2], 100);
    }

    #[test]
    fn test_bilateral_filter_u8_flat_image() {
        let data = vec![128u8; 32 * 32];
        let result = bilateral_filter_u8(&data, 32, 32, 1, 5, 75.0, 75.0);
        assert_eq!(result.len(), 32 * 32);
        for &v in &result {
            assert_eq!(v, 128, "flat image should be preserved");
        }
    }

    #[test]
    fn test_bilateral_filter_u8_preserves_edges() {
        let mut data = vec![0u8; 64 * 64];
        for y in 0..64 {
            for x in 0..64 {
                data[y * 64 + x] = if x < 32 { 50 } else { 200 };
            }
        }
        let result = bilateral_filter_u8(&data, 64, 64, 1, 5, 20.0, 75.0);
        assert!(
            (result[32 * 64 + 5] as i16 - 50).abs() <= 2,
            "left interior preserved"
        );
        assert!(
            (result[32 * 64 + 58] as i16 - 200).abs() <= 2,
            "right interior preserved"
        );
    }

    #[test]
    fn test_bilateral_filter_u8_dimensions() {
        let data = vec![100u8; 480 * 640];
        let result = bilateral_filter_u8(&data, 640, 480, 1, 5, 75.0, 75.0);
        assert_eq!(result.len(), 480 * 640);
    }
}
