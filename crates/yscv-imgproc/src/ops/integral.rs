use yscv_tensor::Tensor;

use super::super::ImgProcError;
use super::super::shape::hwc_shape;

/// Summed-area table with `f64` accumulation and a zero border row and column.
///
/// Built once from a per-pixel value, it answers the sum over any axis-aligned window in four
/// loads. Accumulating in `f64` keeps sums of squares and products exact enough for windowed
/// variance and correlation on megapixel images, where an `f32` table loses the low digits;
/// [`integral_image`](super::integral_image) is the fast `f32` variant for plain box filters.
#[derive(Debug, Clone, PartialEq)]
pub struct IntegralImage {
    width: usize,
    height: usize,
    table: Vec<f64>,
}

impl IntegralImage {
    /// Builds the table from `value(x, y)` evaluated once per pixel, so a derived quantity
    /// (a square, a product, an indicator) never has to be materialised as an image.
    pub fn from_fn(width: usize, height: usize, value: impl Fn(usize, usize) -> f64) -> Self {
        let stride = width + 1;
        let mut table = vec![0.0f64; stride * (height + 1)];
        for y in 0..height {
            let mut row = 0.0f64;
            for x in 0..width {
                row += value(x, y);
                table[(y + 1) * stride + x + 1] = table[y * stride + x + 1] + row;
            }
        }
        Self {
            width,
            height,
            table,
        }
    }

    /// Builds the table from a single-channel `[H, W, 1]` image.
    pub fn from_tensor(input: &Tensor) -> Result<Self, ImgProcError> {
        let (h, w, c) = hwc_shape(input)?;
        if c != 1 {
            return Err(ImgProcError::InvalidChannelCount {
                expected: 1,
                got: c,
            });
        }
        let data = input.data();
        Ok(Self::from_fn(w, h, |x, y| f64::from(data[y * w + x])))
    }

    pub const fn width(&self) -> usize {
        self.width
    }

    pub const fn height(&self) -> usize {
        self.height
    }

    /// Sum over the half-open rectangle `[x0, x1) × [y0, y1)`, clipped to the image.
    pub fn rect_sum(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> f64 {
        let (x0, y0) = (x0.min(self.width), y0.min(self.height));
        let (x1, y1) = (x1.min(self.width).max(x0), y1.min(self.height).max(y0));
        let s = self.width + 1;
        self.table[y1 * s + x1] - self.table[y0 * s + x1] - self.table[y1 * s + x0]
            + self.table[y0 * s + x0]
    }

    /// Sum over the `(2r + 1)²` window centred on pixel `(x, y)`, clipped to the image.
    pub fn window_sum(&self, x: usize, y: usize, r: usize) -> f64 {
        self.rect_sum(
            x.saturating_sub(r),
            y.saturating_sub(r),
            x + r + 1,
            y + r + 1,
        )
    }

    /// Number of pixels the clipped window around `(x, y)` covers, so that
    /// `window_sum / window_area` is a mean that stays correct at the borders.
    pub fn window_area(&self, x: usize, y: usize, r: usize) -> usize {
        let (x0, y0) = (
            x.saturating_sub(r).min(self.width),
            y.saturating_sub(r).min(self.height),
        );
        let (x1, y1) = ((x + r + 1).min(self.width), (y + r + 1).min(self.height));
        (x1 - x0) * (y1 - y0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_are_clipped_at_the_borders() {
        let it = IntegralImage::from_fn(4, 3, |_, _| 1.0);
        assert_eq!(it.window_sum(0, 0, 1), 4.0);
        assert_eq!(it.window_area(0, 0, 1), 4);
        assert_eq!(it.window_sum(2, 1, 1), 9.0);
        assert_eq!(it.window_sum(3, 2, 5), 12.0);
        assert_eq!(it.window_area(3, 2, 5), 12);
        assert_eq!(it.rect_sum(1, 1, 9, 9), 6.0);
        assert_eq!(
            it.rect_sum(3, 2, 1, 1),
            0.0,
            "an inverted rectangle is empty"
        );
    }

    #[test]
    fn rect_sums_match_a_direct_sum_of_squares() {
        let (w, h) = (37, 23);
        let img: Vec<f32> = (0..w * h)
            .map(|i| ((i * 7919) % 1000) as f32 / 3.0)
            .collect();
        let img = &img;
        let it = IntegralImage::from_fn(w, h, |x, y| f64::from(img[y * w + x]).powi(2));
        for &(x0, y0, x1, y1) in &[
            (0, 0, w, h),
            (5, 3, 20, 17),
            (36, 22, 37, 23),
            (10, 10, 10, 20),
        ] {
            let direct: f64 = (y0..y1)
                .flat_map(|y| (x0..x1).map(move |x| f64::from(img[y * w + x]).powi(2)))
                .sum();
            assert!((it.rect_sum(x0, y0, x1, y1) - direct).abs() < 1e-6 * direct.max(1.0));
        }
    }

    #[test]
    fn from_tensor_wants_one_channel() {
        let rgb = Tensor::from_vec(vec![2, 2, 3], vec![0.0; 12]).unwrap();
        assert!(IntegralImage::from_tensor(&rgb).is_err());
        let gray = Tensor::from_vec(vec![2, 2, 1], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let it = IntegralImage::from_tensor(&gray).unwrap();
        assert_eq!((it.width(), it.height()), (2, 2));
        assert_eq!(it.rect_sum(0, 0, 2, 2), 10.0);
        assert_eq!(it.rect_sum(1, 0, 2, 2), 6.0);
    }
}
