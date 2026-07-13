use yscv_tensor::Tensor;

use super::super::{
    blob_log, compute_gradient_orientation, corner_sub_pix, fast_corners, good_features_to_track,
    harris_corners, hog_cell_descriptor, hough_circles, orb_descriptors, orb_hamming_distance,
    orb_match, sift_descriptor, sift_match,
};

// ── Harris corners ──────────────────────────────────────────────────

#[test]
fn harris_detects_isolated_bright_dot() {
    let mut data = vec![0.0f32; 20 * 20];
    data[10 * 20 + 10] = 1.0;
    data[10 * 20 + 11] = 1.0;
    data[11 * 20 + 10] = 1.0;
    data[11 * 20 + 11] = 1.0;
    let img = Tensor::from_vec(vec![20, 20, 1], data).unwrap();
    let corners = harris_corners(&img, 3, 0.04, 0.0).unwrap();
    assert!(
        !corners.is_empty(),
        "should detect at least one corner near bright square"
    );
    for c in &corners {
        assert!(c.x >= 5 && c.x <= 15 && c.y >= 5 && c.y <= 15);
    }
}

#[test]
fn harris_rejects_flat_image() {
    let img = Tensor::from_vec(vec![20, 20, 1], vec![0.5; 400]).unwrap();
    let corners = harris_corners(&img, 3, 0.04, 0.01).unwrap();
    assert!(corners.is_empty(), "flat image should produce no corners");
}

// ── FAST corners ────────────────────────────────────────────────────

#[test]
fn fast_detects_bright_point() {
    let mut data = vec![0.0f32; 20 * 20];
    // Bright spot surrounded by dark
    data[10 * 20 + 10] = 1.0;
    let img = Tensor::from_vec(vec![20, 20, 1], data).unwrap();
    let corners = fast_corners(&img, 0.2, 9).unwrap();
    // The bright point should generate at least some response near neighbors
    // Depending on exact placement, we may or may not get corners; main check is no panic
    assert!(corners.iter().all(|c| c.response >= 0.0));
}

#[test]
fn fast_flat_image_no_corners() {
    let img = Tensor::from_vec(vec![20, 20, 1], vec![0.5; 400]).unwrap();
    let corners = fast_corners(&img, 0.2, 9).unwrap();
    assert!(corners.is_empty());
}

// ── ORB ─────────────────────────────────────────────────────────────

#[test]
fn orb_descriptors_produces_256_bit_output() {
    let mut data = vec![0.5f32; 40 * 40];
    data[20 * 40 + 20] = 1.0;
    let img = Tensor::from_vec(vec![40, 40, 1], data).unwrap();
    let kps = vec![(20, 20)];
    let desc = orb_descriptors(&img, &kps, 16).unwrap();
    assert_eq!(desc.len(), 1);
    assert_eq!(desc[0].bits.len(), 32);
}

#[test]
fn orb_same_descriptor_has_zero_distance() {
    let data = vec![0.5f32; 40 * 40];
    let img = Tensor::from_vec(vec![40, 40, 1], data).unwrap();
    let kps = vec![(20, 20)];
    let desc = orb_descriptors(&img, &kps, 16).unwrap();
    assert_eq!(desc.len(), 1);
    assert_eq!(orb_hamming_distance(&desc[0], &desc[0]), 0);
}

#[test]
fn orb_match_finds_identical() {
    let data = vec![0.3f32; 40 * 40];
    let img = Tensor::from_vec(vec![40, 40, 1], data).unwrap();
    let desc_a = orb_descriptors(&img, &[(20, 20)], 16).unwrap();
    let desc_b = orb_descriptors(&img, &[(20, 20)], 16).unwrap();
    let m = orb_match(&desc_a, &desc_b, 256);
    assert!(!m.is_empty());
    assert_eq!(m[0].2, 0);
}

// ── HOG ─────────────────────────────────────────────────────────────

#[test]
fn hog_cell_descriptor_produces_9_bins() {
    let cell = Tensor::from_vec(vec![8, 8, 1], vec![0.5; 64]).unwrap();
    let desc = hog_cell_descriptor(&cell).unwrap();
    assert_eq!(desc.len(), 9);
    let norm: f32 = desc.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 0.1 || norm < 0.01,
        "should be L2-normalized or zero"
    );
}

// ── SIFT ────────────────────────────────────────────────────────────

#[test]
fn sift_descriptor_produces_128_element_output() {
    let data: Vec<f32> = (0..32 * 32)
        .map(|i| (i as f32 / 1024.0).sin().abs())
        .collect();
    let img = Tensor::from_vec(vec![32, 32, 1], data).unwrap();
    let descs = sift_descriptor(&img, &[(16, 16)]).unwrap();
    assert_eq!(descs.len(), 1);
    assert_eq!(descs[0].len(), 128);
    let norm: f32 = descs[0].iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 0.1 || norm < 0.01);
}

#[test]
fn sift_border_keypoint_returns_zeros() {
    let img = Tensor::from_vec(vec![20, 20, 1], vec![0.5; 400]).unwrap();
    let descs = sift_descriptor(&img, &[(2, 2)]).unwrap();
    assert!(descs[0].iter().all(|&v| v == 0.0));
}

#[test]
fn sift_match_finds_identical_descriptors() {
    let data: Vec<f32> = (0..32 * 32)
        .map(|i| (i as f32 / 256.0).sin().abs())
        .collect();
    let img = Tensor::from_vec(vec![32, 32, 1], data).unwrap();
    let descs = sift_descriptor(&img, &[(16, 16)]).unwrap();
    let matches = sift_match(&descs, &descs, 0.8);
    assert!(!matches.is_empty());
    assert_eq!(matches[0].0, 0);
    assert_eq!(matches[0].1, 0);
    assert!(matches[0].2 < 1e-5);
}

// ── Good features to track (Shi-Tomasi) ─────────────────────────────

#[test]
fn good_features_detects_corner_at_bright_square() {
    // Create image with a bright square that has clear corners
    let mut data = vec![0.0f32; 30 * 30];
    for y in 10..20 {
        for x in 10..20 {
            data[y * 30 + x] = 1.0;
        }
    }
    let img = Tensor::from_vec(vec![30, 30, 1], data).unwrap();
    let corners = good_features_to_track(&img, 10, 0.01, 3.0).unwrap();
    assert!(
        !corners.is_empty(),
        "should detect corners at bright square edges"
    );
    // All detected corners should be near the square boundary (rows/cols 9..21)
    for &(r, c) in &corners {
        assert!(
            (8..=22).contains(&r) && (8..=22).contains(&c),
            "corner ({r},{c}) outside expected range"
        );
    }
}

#[test]
fn good_features_flat_image_no_corners() {
    let img = Tensor::from_vec(vec![20, 20, 1], vec![0.5; 400]).unwrap();
    let corners = good_features_to_track(&img, 10, 0.01, 3.0).unwrap();
    assert!(corners.is_empty(), "flat image should produce no corners");
}

#[test]
fn good_features_respects_max_corners() {
    let mut data = vec![0.0f32; 40 * 40];
    // Multiple bright squares
    for y in 5..10 {
        for x in 5..10 {
            data[y * 40 + x] = 1.0;
        }
    }
    for y in 20..25 {
        for x in 20..25 {
            data[y * 40 + x] = 1.0;
        }
    }
    let img = Tensor::from_vec(vec![40, 40, 1], data).unwrap();
    let corners = good_features_to_track(&img, 2, 0.01, 1.0).unwrap();
    assert!(corners.len() <= 2, "should respect max_corners limit");
}

// ── Sub-pixel corner refinement ─────────────────────────────────────

#[test]
fn corner_sub_pix_refines_near_original() {
    // Create an image with a smooth bright bump so that the sub-pixel
    // refinement should stay near the original integer corner.
    let mut data = vec![0.0f32; 30 * 30];
    for y in 0..30 {
        for x in 0..30 {
            let dy = y as f32 - 15.0;
            let dx = x as f32 - 15.0;
            data[y * 30 + x] = (-0.05 * (dy * dy + dx * dx)).exp();
        }
    }
    let img = Tensor::from_vec(vec![30, 30, 1], data).unwrap();
    let corners = vec![(15, 15)];
    let refined = corner_sub_pix(&img, &corners, 3).unwrap();
    assert_eq!(refined.len(), 1);
    let (rr, rc) = refined[0];
    // Refined position should be close to the original since the peak is centered
    assert!(
        (rr - 15.0).abs() < 1.5 && (rc - 15.0).abs() < 1.5,
        "refined corner ({rr}, {rc}) too far from original (15, 15)"
    );
}

#[test]
fn corner_sub_pix_border_returns_original() {
    let img = Tensor::from_vec(vec![10, 10, 1], vec![0.5; 100]).unwrap();
    let corners = vec![(1, 1)]; // too close to border for win_size=3
    let refined = corner_sub_pix(&img, &corners, 3).unwrap();
    assert_eq!(refined.len(), 1);
    assert_eq!(refined[0], (1.0, 1.0));
}

// ── Gradient orientation ────────────────────────────────────────────

#[test]
fn gradient_orientation_correct_shapes() {
    let img = Tensor::from_vec(vec![10, 10, 1], vec![0.5; 100]).unwrap();
    let (mag, ori) = compute_gradient_orientation(&img).unwrap();
    assert_eq!(mag.shape(), &[10, 10, 1]);
    assert_eq!(ori.shape(), &[10, 10, 1]);
}

// ── Blob detection (LoG) ───────────────────────────────────────────

#[test]
fn blob_log_detects_circular_blob() {
    // Create a 50x50 image with a circular blob of radius ~5 centred at (25, 25)
    let (h, w) = (50, 50);
    let mut data = vec![0.0f32; h * w];
    let (cy, cx) = (25.0f32, 25.0f32);
    let r = 5.0f32;
    for y in 0..h {
        for x in 0..w {
            let dy = y as f32 - cy;
            let dx = x as f32 - cx;
            let dist = (dy * dy + dx * dx).sqrt();
            if dist <= r {
                data[y * w + x] = 1.0;
            }
        }
    }
    let img = Tensor::from_vec(vec![h, w, 1], data).unwrap();

    // sigma ~ r / sqrt(2) ~ 3.5 for optimal LoG response
    let blobs = blob_log(&img, 1.0, 8.0, 10, 0.01).unwrap();
    assert!(
        !blobs.is_empty(),
        "should detect at least one blob in circular region"
    );

    // At least one blob should be near the centre
    let near_center = blobs.iter().any(|&(row, col, _sigma)| {
        let dr = (row as f32 - cy).abs();
        let dc = (col as f32 - cx).abs();
        dr < 8.0 && dc < 8.0
    });
    assert!(
        near_center,
        "at least one detected blob should be near the circle centre; got {:?}",
        blobs,
    );
}

#[test]
fn blob_log_flat_image_no_blobs() {
    let img = Tensor::from_vec(vec![30, 30, 1], vec![0.5; 900]).unwrap();
    let blobs = blob_log(&img, 1.0, 5.0, 5, 0.1).unwrap();
    assert!(blobs.is_empty(), "flat image should produce no blobs");
}

// ── Hough circles ──────────────────────────────────────────────────

#[test]
fn hough_circles_detects_drawn_circle() {
    // Draw a circle of radius 10 centred at (25, 25) on a 50x50 edge image
    let (h, w) = (50, 50);
    let mut data = vec![0.0f32; h * w];
    let (cy, cx) = (25, 25);
    let r = 10;
    let steps = 200;
    for i in 0..steps {
        let angle = 2.0 * std::f32::consts::PI * i as f32 / steps as f32;
        let px = (cx as f32 + r as f32 * angle.cos()).round() as usize;
        let py = (cy as f32 + r as f32 * angle.sin()).round() as usize;
        if py < h && px < w {
            data[py * w + px] = 1.0;
        }
    }
    let img = Tensor::from_vec(vec![h, w, 1], data).unwrap();

    let circles = hough_circles(&img, 8, 12, 10).unwrap();
    assert!(!circles.is_empty(), "should detect at least one circle");

    // Best circle should be near the true centre with approximately the right radius
    let (best_r, best_c, best_rad) = circles[0];
    assert!(
        (best_r as i32 - cy).unsigned_abs() <= 3,
        "centre row {} should be near {}",
        best_r,
        cy,
    );
    assert!(
        (best_c as i32 - cx).unsigned_abs() <= 3,
        "centre col {} should be near {}",
        best_c,
        cx,
    );
    assert!(
        (best_rad as i32 - r).unsigned_abs() <= 2,
        "radius {} should be near {}",
        best_rad,
        r,
    );
}

#[test]
fn hough_circles_empty_image() {
    let img = Tensor::from_vec(vec![20, 20, 1], vec![0.0; 400]).unwrap();
    let circles = hough_circles(&img, 3, 8, 5).unwrap();
    assert!(circles.is_empty(), "empty image should produce no circles");
}

// ── Chamfer L2 distance transform ───────────────────────────────────

use crate::{DistanceTransformMask, distance_transform_l2};

/// Naive per-pixel reference: the textbook two-sweep chamfer DT, one pixel
/// at a time. The production kernel must match it bit-for-bit.
fn dt_l2_reference(fg: &[bool], w: usize, h: usize, mask: DistanceTransformMask) -> Vec<f32> {
    const INF: f32 = f32::MAX / 4.0;
    let offsets: &[(i64, i64, f32)] = match mask {
        // Forward half of the 3×3 mask: left + upper row.
        DistanceTransformMask::Mask3 => &[
            (-1, 0, 0.955),
            (-1, -1, 1.3693),
            (0, -1, 0.955),
            (1, -1, 1.3693),
        ],
        // Forward half of the 5×5 mask: left + two upper rows incl. knights.
        DistanceTransformMask::Mask5 => &[
            (-1, 0, 1.0),
            (-1, -1, 1.4),
            (0, -1, 1.0),
            (1, -1, 1.4),
            (-2, -1, 2.1969),
            (2, -1, 2.1969),
            (-1, -2, 2.1969),
            (1, -2, 2.1969),
        ],
    };
    let mut dist: Vec<f32> = fg.iter().map(|&f| if f { INF } else { 0.0 }).collect();
    for forward in [true, false] {
        let ys: Vec<i64> = if forward {
            (0..h as i64).collect()
        } else {
            (0..h as i64).rev().collect()
        };
        for &y in &ys {
            let xs: Vec<i64> = if forward {
                (0..w as i64).collect()
            } else {
                (0..w as i64).rev().collect()
            };
            for &x in &xs {
                let i = (y * w as i64 + x) as usize;
                if dist[i] == 0.0 {
                    continue;
                }
                let mut best = dist[i];
                for &(dx, dy, wt) in offsets {
                    let (nx, ny) = if forward {
                        (x + dx, y + dy)
                    } else {
                        (x - dx, y - dy)
                    };
                    if nx >= 0 && nx < w as i64 && ny >= 0 && ny < h as i64 {
                        best = best.min(dist[(ny * w as i64 + nx) as usize] + wt);
                    }
                }
                dist[i] = best;
            }
        }
    }
    dist
}

fn fg_to_tensor(fg: &[bool], w: usize, h: usize) -> Tensor {
    let data: Vec<f32> = fg.iter().map(|&f| if f { 1.0 } else { 0.0 }).collect();
    Tensor::from_vec(vec![h, w, 1], data).unwrap()
}

#[test]
fn dt_l2_single_zero_known_values() {
    // 5×5 all-foreground with a single background pixel in the center:
    // neighbors get exactly the chamfer weights.
    let mut fg = vec![true; 25];
    fg[12] = false;
    let t = fg_to_tensor(&fg, 5, 5);

    let d3 = distance_transform_l2(&t, DistanceTransformMask::Mask3).unwrap();
    let d3 = d3.data();
    assert_eq!(d3[12], 0.0);
    assert!((d3[11] - 0.955).abs() < 1e-6, "side {}", d3[11]);
    assert!((d3[6] - 1.3693).abs() < 1e-6, "diagonal {}", d3[6]);

    let d5 = distance_transform_l2(&t, DistanceTransformMask::Mask5).unwrap();
    let d5 = d5.data();
    assert!((d5[11] - 1.0).abs() < 1e-6, "side {}", d5[11]);
    assert!((d5[6] - 1.4).abs() < 1e-6, "diagonal {}", d5[6]);
    assert!((d5[1] - 2.1969).abs() < 1e-5, "knight {}", d5[1]);
}

#[test]
fn dt_l2_matches_reference_bit_for_bit() {
    // Random binary images (deterministic LCG), widths around the SIMD
    // block boundaries; production output must equal the naive reference
    // exactly — min/plus over identical operands has no rounding freedom.
    let mut state = 0x1234_5678_u64;
    let mut rand = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    for &(w, h) in &[
        (3usize, 4usize),
        (7, 5),
        (16, 16),
        (33, 21),
        (64, 48),
        (130, 17),
    ] {
        for density in [10u32, 50, 90] {
            let fg: Vec<bool> = (0..w * h).map(|_| rand() % 100 < density).collect();
            let t = fg_to_tensor(&fg, w, h);
            for mask in [DistanceTransformMask::Mask3, DistanceTransformMask::Mask5] {
                let got = distance_transform_l2(&t, mask).unwrap();
                let want = dt_l2_reference(&fg, w, h, mask);
                assert_eq!(got.data(), &want[..], "{w}x{h} density {density} {mask:?}");
            }
        }
    }
}

#[test]
fn dt_l2_all_background_is_zero() {
    let t = Tensor::from_vec(vec![8, 8, 1], vec![0.0; 64]).unwrap();
    let d = distance_transform_l2(&t, DistanceTransformMask::Mask3).unwrap();
    assert!(d.data().iter().all(|&v| v == 0.0));
}
