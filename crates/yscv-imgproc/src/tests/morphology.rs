use yscv_tensor::Tensor;

use super::super::{
    closing_3x3, dilate, dilate_3x3, erode, erode_3x3, morph_blackhat, morph_gradient_3x3,
    morph_tophat, opening_3x3, remove_small_objects, skeletonize,
};

#[test]
fn dilate_3x3_expands_local_peak() {
    let input = Tensor::from_vec(
        vec![3, 3, 1],
        vec![
            0.0, 0.0, 0.0, //
            0.0, 5.0, 0.0, //
            0.0, 0.0, 0.0,
        ],
    )
    .unwrap();

    let out = dilate_3x3(&input).unwrap();
    assert_eq!(out.shape(), &[3, 3, 1]);
    assert_eq!(
        out.data(),
        &[
            5.0, 5.0, 5.0, //
            5.0, 5.0, 5.0, //
            5.0, 5.0, 5.0,
        ]
    );
}

#[test]
fn erode_3x3_shrinks_local_peak() {
    let input = Tensor::from_vec(
        vec![3, 3, 1],
        vec![
            1.0, 1.0, 1.0, //
            1.0, 5.0, 1.0, //
            1.0, 1.0, 1.0,
        ],
    )
    .unwrap();

    let out = erode_3x3(&input).unwrap();
    assert_eq!(out.shape(), &[3, 3, 1]);
    assert_eq!(
        out.data(),
        &[
            1.0, 1.0, 1.0, //
            1.0, 1.0, 1.0, //
            1.0, 1.0, 1.0,
        ]
    );
}

#[test]
fn opening_3x3_removes_isolated_bright_noise() {
    let input = Tensor::from_vec(
        vec![3, 3, 1],
        vec![
            0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0,
        ],
    )
    .unwrap();

    let out = opening_3x3(&input).unwrap();
    assert_eq!(out.shape(), &[3, 3, 1]);
    assert_eq!(out.data(), &[0.0; 9]);
}

#[test]
fn closing_3x3_fills_isolated_dark_hole() {
    let input = Tensor::from_vec(
        vec![3, 3, 1],
        vec![
            1.0, 1.0, 1.0, //
            1.0, 0.0, 1.0, //
            1.0, 1.0, 1.0,
        ],
    )
    .unwrap();

    let out = closing_3x3(&input).unwrap();
    assert_eq!(out.shape(), &[3, 3, 1]);
    assert_eq!(out.data(), &[1.0; 9]);
}

#[test]
fn morph_gradient_3x3_matches_dilate_minus_erode() {
    let input = Tensor::from_vec(
        vec![3, 3, 1],
        vec![
            0.0, 0.0, 0.0, //
            0.0, 5.0, 0.0, //
            0.0, 0.0, 0.0,
        ],
    )
    .unwrap();

    let gradient = morph_gradient_3x3(&input).unwrap();
    assert_eq!(gradient.shape(), &[3, 3, 1]);
    assert_eq!(
        gradient.data(),
        &[
            5.0, 5.0, 5.0, //
            5.0, 5.0, 5.0, //
            5.0, 5.0, 5.0,
        ]
    );
}

#[test]
fn dilate_with_cross_kernel() {
    let mut data = vec![0.0f32; 5 * 5];
    data[2 * 5 + 2] = 1.0; // center pixel
    let img = Tensor::from_vec(vec![5, 5, 1], data).unwrap();
    // Cross kernel 3x3
    let kernel = Tensor::from_vec(
        vec![3, 3, 1],
        vec![0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0],
    )
    .unwrap();
    let out = dilate(&img, &kernel).unwrap();
    // Center and 4 neighbors should be 1.0
    assert_eq!(out.data()[12], 1.0); // (2,2)
    assert_eq!(out.data()[7], 1.0); // (1,2)
    assert_eq!(out.data()[17], 1.0); // (3,2)
    assert_eq!(out.data()[11], 1.0); // (2,1)
    assert_eq!(out.data()[13], 1.0); // (2,3)
    // Diagonals should remain 0
    assert_eq!(out.data()[6], 0.0); // (1,1)
}

#[test]
fn erode_with_full_kernel() {
    let mut data = vec![1.0f32; 5 * 5];
    data[0] = 0.0; // top-left corner
    let img = Tensor::from_vec(vec![5, 5, 1], data).unwrap();
    let kernel = Tensor::from_vec(vec![3, 3, 1], vec![1.0; 9]).unwrap();
    let out = erode(&img, &kernel).unwrap();
    // Pixels near the zero corner should be eroded to 0
    assert_eq!(out.data()[0], 0.0);
    assert_eq!(out.data()[1], 0.0);
    assert_eq!(out.data()[5], 0.0);
    // Far pixels should remain 1
    assert_eq!(out.data()[4 * 5 + 4], 1.0);
}

#[test]
fn morph_tophat_uniform_is_zero() {
    // Uniform image: opening == input, so tophat = input - opening = 0
    let img = Tensor::from_vec(vec![5, 5, 1], vec![0.7; 25]).unwrap();
    let out = morph_tophat(&img).unwrap();
    for &v in out.data() {
        assert!(v.abs() < 1e-6, "tophat of uniform should be ~0, got {}", v);
    }
}

#[test]
fn morph_blackhat_uniform_is_zero() {
    // Uniform image: closing == input, so blackhat = closing - input = 0
    let img = Tensor::from_vec(vec![5, 5, 1], vec![0.3; 25]).unwrap();
    let out = morph_blackhat(&img).unwrap();
    for &v in out.data() {
        assert!(
            v.abs() < 1e-6,
            "blackhat of uniform should be ~0, got {}",
            v
        );
    }
}

#[test]
fn skeletonize_thick_horizontal_bar() {
    // 20x10 image with a thick horizontal bar (rows 3..7, all columns)
    let h = 10;
    let w = 20;
    let mut data = vec![0.0f32; h * w];
    for y in 3..7 {
        for x in 0..w {
            data[y * w + x] = 1.0;
        }
    }
    let img = Tensor::from_vec(vec![h, w, 1], data).unwrap();
    let skel = skeletonize(&img).unwrap();
    let skel_data = skel.data();

    // The skeleton should be thinner than the original bar
    let skel_fg: usize = skel_data.iter().filter(|&&v| v > 0.5).count();
    let orig_fg = 4 * w; // 80 pixels
    assert!(
        skel_fg < orig_fg,
        "skeleton ({}) should have fewer fg pixels than original ({})",
        skel_fg,
        orig_fg
    );
    // Should still have some foreground (the thin line)
    assert!(skel_fg > 0, "skeleton should not be empty");

    // The skeleton should be at most ~1 pixel thick in the vertical direction
    // Check that for interior columns, at most 1-2 rows are foreground
    for x in 2..w - 2 {
        let fg_in_col: usize = (0..h).filter(|&y| skel_data[y * w + x] > 0.5).count();
        assert!(
            fg_in_col <= 2,
            "column {} has {} fg pixels, expected thin line",
            x,
            fg_in_col
        );
    }
}

#[test]
fn remove_small_objects_removes_small_blobs() {
    // 10x10 image with one large blob (16 pixels) and one small blob (2 pixels)
    let mut data = vec![0.0f32; 10 * 10];
    // Large blob: 4x4 block at (1,1)
    for y in 1..5 {
        for x in 1..5 {
            data[y * 10 + x] = 1.0;
        }
    }
    // Small blob: 2 pixels at (8,8) and (8,9)
    data[8 * 10 + 8] = 1.0;
    data[8 * 10 + 9] = 1.0;

    let img = Tensor::from_vec(vec![10, 10, 1], data).unwrap();
    let out = remove_small_objects(&img, 5).unwrap();
    let out_data = out.data();

    // Large blob should remain
    assert_eq!(out_data[10 + 1], 1.0, "large blob should remain");
    assert_eq!(out_data[4 * 10 + 4], 1.0, "large blob should remain");

    // Small blob should be removed
    assert_eq!(out_data[8 * 10 + 8], 0.0, "small blob should be removed");
    assert_eq!(out_data[8 * 10 + 9], 0.0, "small blob should be removed");
}

/// Test dilate_3x3 with RGB (3-channel) data to exercise multi-channel SIMD path.
#[test]
fn dilate_3x3_rgb_multichannel() {
    // 8x8 RGB image: all zeros except one bright pixel at (4,4)
    let (h, w, c) = (8, 8, 3);
    let mut data = vec![0.0f32; h * w * c];
    // Set pixel (4,4) to (0.9, 0.5, 0.3)
    let idx = (4 * w + 4) * c;
    data[idx] = 0.9;
    data[idx + 1] = 0.5;
    data[idx + 2] = 0.3;

    let input = Tensor::from_vec(vec![h, w, c], data).unwrap();
    let out = dilate_3x3(&input).unwrap();
    let od = out.data();

    // Dilated pixel should spread to 3x3 neighborhood
    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            let y = (4 + dy) as usize;
            let x = (4 + dx) as usize;
            let i = (y * w + x) * c;
            assert!(
                (od[i] - 0.9).abs() < 1e-6,
                "dilate RGB R at ({},{}) = {}, expected 0.9",
                y,
                x,
                od[i]
            );
            assert!(
                (od[i + 1] - 0.5).abs() < 1e-6,
                "dilate RGB G at ({},{}) = {}, expected 0.5",
                y,
                x,
                od[i + 1]
            );
            assert!(
                (od[i + 2] - 0.3).abs() < 1e-6,
                "dilate RGB B at ({},{}) = {}, expected 0.3",
                y,
                x,
                od[i + 2]
            );
        }
    }
    // Far corner should still be 0
    assert!((od[0]).abs() < 1e-6);
    assert!((od[1]).abs() < 1e-6);
    assert!((od[2]).abs() < 1e-6);
}

/// Test erode_3x3 with RGB (3-channel) data to exercise multi-channel SIMD path.
#[test]
fn erode_3x3_rgb_multichannel() {
    // 8x8 RGB image: all (0.9, 0.5, 0.3) except one dark pixel at (4,4) = (0.1, 0.2, 0.05)
    let (h, w, c) = (8, 8, 3);
    let mut data = vec![0.0f32; h * w * c];
    for px in 0..(h * w) {
        data[px * c] = 0.9;
        data[px * c + 1] = 0.5;
        data[px * c + 2] = 0.3;
    }
    let idx = (4 * w + 4) * c;
    data[idx] = 0.1;
    data[idx + 1] = 0.2;
    data[idx + 2] = 0.05;

    let input = Tensor::from_vec(vec![h, w, c], data).unwrap();
    let out = erode_3x3(&input).unwrap();
    let od = out.data();

    // Eroded: dark pixel should spread to 3x3 neighborhood
    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            let y = (4 + dy) as usize;
            let x = (4 + dx) as usize;
            let i = (y * w + x) * c;
            assert!(
                (od[i] - 0.1).abs() < 1e-6,
                "erode RGB R at ({},{}) = {}, expected 0.1",
                y,
                x,
                od[i]
            );
            assert!(
                (od[i + 1] - 0.2).abs() < 1e-6,
                "erode RGB G at ({},{}) = {}, expected 0.2",
                y,
                x,
                od[i + 1]
            );
            assert!(
                (od[i + 2] - 0.05).abs() < 1e-6,
                "erode RGB B at ({},{}) = {}, expected 0.05",
                y,
                x,
                od[i + 2]
            );
        }
    }
    // Far corner should still be original values
    assert!((od[0] - 0.9).abs() < 1e-6);
    assert!((od[1] - 0.5).abs() < 1e-6);
    assert!((od[2] - 0.3).abs() < 1e-6);
}

/// Test dilate_3x3 with larger RGB image to exercise GCD/rayon parallel + SIMD path.
#[test]
fn dilate_3x3_rgb_large() {
    let (h, w, c) = (64, 80, 3);
    let mut data = vec![0.0f32; h * w * c];
    // Set a bright pixel at (32, 40)
    let idx = (32 * w + 40) * c;
    data[idx] = 0.8;
    data[idx + 1] = 0.6;
    data[idx + 2] = 0.4;

    let input = Tensor::from_vec(vec![h, w, c], data.clone()).unwrap();
    let out = dilate_3x3(&input).unwrap();
    let od = out.data();

    // Check 3x3 neighborhood
    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            let y = (32 + dy) as usize;
            let x = (40 + dx) as usize;
            let i = (y * w + x) * c;
            assert!(
                (od[i] - 0.8).abs() < 1e-6,
                "large dilate RGB R at ({},{}) = {}, expected 0.8",
                y,
                x,
                od[i]
            );
        }
    }
    // Pixel far away should be 0
    assert!((od[0]).abs() < 1e-6);
}

#[test]
fn box_morphology_matches_generic_ones_kernel() {
    use super::super::{dilate, dilate_box, erode, erode_box};
    // псевдослучайное изображение 17x13 — сверяем с generic-ядром единиц
    let (h, w) = (17usize, 13usize);
    let mut state = 0x1234_5678_u64;
    let mut data = Vec::with_capacity(h * w);
    for _ in 0..h * w {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        data.push(((state >> 33) % 1000) as f32 / 100.0);
    }
    let img = Tensor::from_vec(vec![h, w, 1], data).unwrap();
    for ksize in [1usize, 3, 5, 7] {
        let kernel = Tensor::from_vec(vec![ksize, ksize, 1], vec![1.0f32; ksize * ksize]).unwrap();
        let d_ref = dilate(&img, &kernel).unwrap();
        let d_box = dilate_box(&img, ksize).unwrap();
        assert_eq!(d_ref.data(), d_box.data(), "dilate ksize={ksize}");
        let e_ref = erode(&img, &kernel).unwrap();
        let e_box = erode_box(&img, ksize).unwrap();
        assert_eq!(e_ref.data(), e_box.data(), "erode ksize={ksize}");
    }
}

#[test]
fn box_morphology_rejects_even_kernel() {
    use super::super::dilate_box;
    let img = Tensor::from_vec(vec![4, 4, 1], vec![0.0f32; 16]).unwrap();
    assert!(dilate_box(&img, 4).is_err());
    assert!(dilate_box(&img, 0).is_err());
}

#[test]
fn binary_box_matches_generic_ones_kernel() {
    use super::super::{dilate, dilate_binary_box, erode, erode_binary_box};
    let (h, w) = (19usize, 11usize);
    let mut state = 0xDEAD_BEEF_u64;
    let mut data = Vec::with_capacity(h * w);
    for _ in 0..h * w {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        data.push(if (state >> 33) % 3 == 0 { 1.0f32 } else { 0.0 });
    }
    let img = Tensor::from_vec(vec![h, w, 1], data).unwrap();
    for ksize in [1usize, 3, 5, 9] {
        let kernel = Tensor::from_vec(vec![ksize, ksize, 1], vec![1.0f32; ksize * ksize]).unwrap();
        assert_eq!(
            dilate(&img, &kernel).unwrap().data(),
            dilate_binary_box(&img, ksize).unwrap().data(),
            "dilate ksize={ksize}"
        );
        assert_eq!(
            erode(&img, &kernel).unwrap().data(),
            erode_binary_box(&img, ksize).unwrap().data(),
            "erode ksize={ksize}"
        );
    }
}
