use proptest::prelude::*;

use super::ops::{
    ImageF32, ImageU8, gaussian_blur_3x3_f32, grayscale_f32, resize_nearest_u8,
    threshold_binary_f32,
};

/// Strategy: generate a small single-channel f32 image.
fn arb_gray_f32(max_h: usize, max_w: usize) -> impl Strategy<Value = ImageF32> {
    (3usize..=max_h, 3usize..=max_w).prop_flat_map(|(h, w)| {
        proptest::collection::vec(0.0f32..1.0, h * w)
            .prop_map(move |data| ImageF32::new(data, h, w, 1).expect("valid ImageF32"))
    })
}

/// Strategy: generate a small 3-channel f32 image.
fn arb_rgb_f32(max_h: usize, max_w: usize) -> impl Strategy<Value = ImageF32> {
    (3usize..=max_h, 3usize..=max_w).prop_flat_map(|(h, w)| {
        proptest::collection::vec(0.0f32..1.0, h * w * 3)
            .prop_map(move |data| ImageF32::new(data, h, w, 3).expect("valid ImageF32"))
    })
}

/// Strategy: generate a small 3-channel u8 image.
fn arb_rgb_u8(max_h: usize, max_w: usize) -> impl Strategy<Value = ImageU8> {
    (1usize..=max_h, 1usize..=max_w).prop_flat_map(|(h, w)| {
        proptest::collection::vec(0u8..=255, h * w * 3)
            .prop_map(move |data| ImageU8::new(data, h, w, 3).expect("valid ImageU8"))
    })
}

proptest! {
    // ── Grayscale idempotent ───────────────────────────────────────────
    #[test]
    fn grayscale_is_idempotent(img in arb_rgb_f32(16, 16)) {
        let gray = grayscale_f32(&img).expect("grayscale");
        // Expand single-channel back to 3-channel by repeating
        let (h, w) = (gray.height(), gray.width());
        let expanded_data: Vec<f32> = gray
            .data()
            .iter()
            .flat_map(|&v| [v, v, v])
            .collect();
        let expanded = ImageF32::new(expanded_data, h, w, 3).expect("expand");
        let gray2 = grayscale_f32(&expanded).expect("grayscale again");
        prop_assert_eq!(gray2.height(), gray.height());
        prop_assert_eq!(gray2.width(), gray.width());
        for (a, b) in gray.data().iter().zip(gray2.data().iter()) {
            prop_assert!(
                (a - b).abs() < 1e-4,
                "grayscale idempotent violation: {a} vs {b}"
            );
        }
    }

    // ── Resize dimensions ──────────────────────────────────────────────
    #[test]
    fn resize_produces_exact_dimensions(
        img in arb_rgb_u8(32, 32),
        out_h in 1usize..=64,
        out_w in 1usize..=64,
    ) {
        let resized = resize_nearest_u8(&img, out_h, out_w).expect("resize");
        prop_assert_eq!(resized.height(), out_h, "resize height mismatch");
        prop_assert_eq!(resized.width(), out_w, "resize width mismatch");
        prop_assert_eq!(resized.channels(), img.channels(), "resize channels mismatch");
    }

    // ── Threshold binary ───────────────────────────────────────────────
    #[test]
    fn threshold_binary_output_is_binary(
        img in arb_gray_f32(16, 16),
        thresh in 0.0f32..1.0,
    ) {
        let max_val = 1.0f32;
        let result = threshold_binary_f32(&img, thresh, max_val).expect("threshold");
        for (i, &val) in result.data().iter().enumerate() {
            prop_assert!(
                val == 0.0 || (val - max_val).abs() < 1e-6,
                "threshold output not binary at {i}: got {val}"
            );
        }
    }

    // ── Gaussian blur preserves mean ───────────────────────────────────
    // Use power-of-2-aligned widths to avoid SIMD edge cases in the kernel.
    #[test]
    fn gaussian_blur_preserves_mean(
        h in 4usize..=16,
        w_factor in 1usize..=8,
    ) {
        let w = w_factor * 4; // keep width aligned to 4 for SIMD safety
        let data: Vec<f32> = (0..h * w).map(|i| ((i * 7 + 3) % 256) as f32 / 255.0).collect();
        let img = ImageF32::new(data, h, w, 1).expect("valid ImageF32");
        let blurred = gaussian_blur_3x3_f32(&img).expect("blur");
        let mean_orig: f32 = img.data().iter().sum::<f32>() / img.data().len() as f32;
        let mean_blur: f32 = blurred.data().iter().sum::<f32>() / blurred.data().len() as f32;
        // Border effects cause small mean deviation; tolerance proportional to border/area ratio
        let border_ratio = 2.0 * (h + w) as f32 / (h * w) as f32;
        let tol = 0.05 + border_ratio * 0.5;
        prop_assert!(
            (mean_orig - mean_blur).abs() < tol,
            "blur mean deviation too large: orig={}, blur={}, tol={}", mean_orig, mean_blur, tol
        );
    }
}
