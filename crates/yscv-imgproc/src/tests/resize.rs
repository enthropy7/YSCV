use yscv_tensor::Tensor;

use super::super::{resize_bilinear, resize_nearest};

#[test]
fn resize_nearest_scales_2x2_to_4x4() {
    let input = Tensor::from_vec(
        vec![2, 2, 1],
        vec![
            1.0, 2.0, //
            3.0, 4.0,
        ],
    )
    .unwrap();
    let out = resize_nearest(&input, 4, 4).unwrap();
    assert_eq!(out.shape(), &[4, 4, 1]);
    assert_eq!(
        out.data(),
        &[
            1.0, 1.0, 2.0, 2.0, //
            1.0, 1.0, 2.0, 2.0, //
            3.0, 3.0, 4.0, 4.0, //
            3.0, 3.0, 4.0, 4.0,
        ]
    );
}

#[test]
fn resize_bilinear_scales_2x2_to_4x4() {
    let img = Tensor::from_vec(vec![2, 2, 1], vec![0.0, 1.0, 0.0, 1.0]).unwrap();
    let resized = resize_bilinear(&img, 4, 4).unwrap();
    assert_eq!(resized.shape(), &[4, 4, 1]);
    assert!((resized.data()[0] - 0.0).abs() < 1e-5);
    assert!((resized.data()[3] - 1.0).abs() < 1e-5);
}

#[test]
fn resize_bilinear_rejects_zero_dimensions() {
    let img = Tensor::filled(vec![2, 2, 1], 0.5).unwrap();
    assert!(resize_bilinear(&img, 0, 4).is_err());
}

// ── Half-pixel (cv2 / ONNX) sampling ────────────────────────────────

use super::super::resize_bilinear_half_pixel;

#[test]
fn resize_half_pixel_matches_cv2_2x_upscale() {
    // cv2.resize([[0, 1], [2, 3]], (4, 4), INTER_LINEAR) reference values:
    // src = (dst + 0.5) / 2 − 0.5 → per-axis samples at −0.25, 0.25, 0.75, 1.25
    // clamped to [0, 1] ⇒ weights 0, 0.25, 0.75, 1.
    let img = Tensor::from_vec(vec![2, 2, 1], vec![0.0, 1.0, 2.0, 3.0]).unwrap();
    let out = resize_bilinear_half_pixel(&img, 4, 4).unwrap();
    let d = out.data();
    let expected = [
        0.0, 0.25, 0.75, 1.0, //
        0.5, 0.75, 1.25, 1.5, //
        1.5, 1.75, 2.25, 2.5, //
        2.0, 2.25, 2.75, 3.0,
    ];
    for (i, (&got, &want)) in d.iter().zip(&expected).enumerate() {
        assert!((got - want).abs() < 1e-6, "px {i}: got {got}, want {want}");
    }
}

#[test]
fn resize_half_pixel_downscale_averages_centers() {
    // 4→2 per axis: src = (dst + 0.5) * 2 − 0.5 → samples at 0.5 and 2.5,
    // i.e. the average of adjacent pixel pairs.
    let img = Tensor::from_vec(vec![1, 4, 1], vec![0.0, 10.0, 20.0, 30.0]).unwrap();
    let out = resize_bilinear_half_pixel(&img, 1, 2).unwrap();
    assert!((out.data()[0] - 5.0).abs() < 1e-6);
    assert!((out.data()[1] - 25.0).abs() < 1e-6);
}

#[test]
fn resize_half_pixel_identity_is_exact() {
    let img = Tensor::from_vec(vec![3, 3, 1], (0..9).map(|v| v as f32).collect()).unwrap();
    let out = resize_bilinear_half_pixel(&img, 3, 3).unwrap();
    assert_eq!(out.data(), img.data());
}
