use yscv_tensor::Tensor;

use super::super::{
    arc_length, bounding_rect, connected_components_4, connected_components_with_stats,
    contour_area, find_contours, hu_moments, region_props,
};

#[test]
fn connected_components_4_counts_blobs() {
    #[rustfmt::skip]
    let data = vec![
        1.0, 0.0, 1.0,
        0.0, 0.0, 0.0,
        1.0, 0.0, 1.0,
    ];
    let img = Tensor::from_vec(vec![3, 3, 1], data).unwrap();
    let (labels, count) = connected_components_4(&img).unwrap();
    assert_eq!(labels.shape(), &[3, 3, 1]);
    assert_eq!(count, 4, "four isolated corners");
}

#[test]
fn connected_components_4_merges_h_connected() {
    #[rustfmt::skip]
    let data = vec![
        1.0, 1.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 1.0, 1.0,
    ];
    let img = Tensor::from_vec(vec![3, 3, 1], data).unwrap();
    let (labels, count) = connected_components_4(&img).unwrap();
    assert_eq!(count, 2);
    assert_eq!(labels.data()[0], labels.data()[1], "top-left connected");
}

#[test]
fn connected_components_4_empty_image() {
    let img = Tensor::zeros(vec![3, 3, 1]).unwrap();
    let (_, count) = connected_components_4(&img).unwrap();
    assert_eq!(count, 0);
}

#[test]
fn find_contours_detects_rectangle() {
    let mut data = vec![0.0f32; 10 * 10];
    for y in 3..7 {
        for x in 3..7 {
            data[y * 10 + x] = 1.0;
        }
    }
    let img = Tensor::from_vec(vec![10, 10, 1], data).unwrap();
    let contours = find_contours(&img).unwrap();
    assert!(!contours.is_empty(), "should detect rectangle contour");
    // The contour should be around the perimeter of the rectangle
    let total_points: usize = contours.iter().map(|c| c.points.len()).sum();
    assert!(total_points >= 4, "contour should have at least 4 points");
}

#[test]
fn find_contours_empty_image() {
    let img = Tensor::from_vec(vec![5, 5, 1], vec![0.0; 25]).unwrap();
    let contours = find_contours(&img).unwrap();
    assert!(contours.is_empty());
}

#[test]
fn convex_hull_triangle() {
    let pts = vec![(0.0, 0.0), (1.0, 0.0), (0.5, 1.0), (0.5, 0.3)];
    let hull = super::super::convex_hull(&pts);
    assert_eq!(hull.len(), 3);
}

#[test]
fn min_area_rect_returns_some() {
    let pts = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
    let rect = super::super::min_area_rect(&pts);
    assert!(rect.is_some());
    let (cx, cy, w, h, _angle) = rect.unwrap();
    assert!((cx - 0.5).abs() < 0.2);
    assert!((cy - 0.5).abs() < 0.2);
    assert!((w - 1.0).abs() < 0.2 || (h - 1.0).abs() < 0.2);
}

#[test]
fn fit_ellipse_circle_points() {
    let mut pts = Vec::new();
    for i in 0..20 {
        let angle = (i as f32) * std::f32::consts::TAU / 20.0;
        pts.push((5.0 + 2.0 * angle.cos(), 5.0 + 2.0 * angle.sin()));
    }
    let result = super::super::fit_ellipse(&pts);
    assert!(result.is_some());
    let (cx, cy, a, b, _angle) = result.unwrap();
    assert!((cx - 5.0).abs() < 0.5);
    assert!((cy - 5.0).abs() < 0.5);
    assert!((a - b).abs() < 0.5);
}

#[test]
fn approx_poly_dp_simplifies() {
    let contour: Vec<(f32, f32)> =
        vec![(0.0, 0.0), (0.5, 0.001), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
    let simplified = super::super::approx_poly_dp(&contour, 0.01);
    assert!(simplified.len() <= contour.len());
    assert!(simplified.len() >= 2);
}

#[test]
fn contour_area_square() {
    let square = vec![(0, 0), (10, 0), (10, 10), (0, 10)];
    let area = contour_area(&square);
    assert!((area - 100.0).abs() < 1e-9, "expected 100, got {}", area);
}

#[test]
fn arc_length_square_closed() {
    let square = vec![(0, 0), (10, 0), (10, 10), (0, 10)];
    let length = arc_length(&square, true);
    assert!((length - 40.0).abs() < 1e-9, "expected 40, got {}", length);
}

#[test]
fn arc_length_square_open() {
    let square = vec![(0, 0), (10, 0), (10, 10), (0, 10)];
    let length = arc_length(&square, false);
    assert!((length - 30.0).abs() < 1e-9, "expected 30, got {}", length);
}

#[test]
fn bounding_rect_square() {
    let square = vec![(0, 0), (10, 0), (10, 10), (0, 10)];
    let (x, y, w, h) = bounding_rect(&square);
    assert_eq!((x, y, w, h), (0, 0, 10, 10));
}

#[test]
fn bounding_rect_offset() {
    let pts = vec![(5, 3), (15, 3), (15, 8), (5, 8)];
    let (x, y, w, h) = bounding_rect(&pts);
    assert_eq!((x, y, w, h), (5, 3, 10, 5));
}

#[test]
fn connected_components_two_blobs() {
    // 10x10 image with two separate 3x3 squares
    let mut data = vec![0.0f32; 10 * 10];
    // Square 1 at (1,1)-(3,3)
    for y in 1..=3 {
        for x in 1..=3 {
            data[y * 10 + x] = 1.0;
        }
    }
    // Square 2 at (6,6)-(8,8)
    for y in 6..=8 {
        for x in 6..=8 {
            data[y * 10 + x] = 1.0;
        }
    }
    let img = Tensor::from_vec(vec![10, 10, 1], data).unwrap();
    let (labels, stats) = connected_components_with_stats(&img).unwrap();
    assert_eq!(labels.shape(), &[10, 10, 1]);
    assert_eq!(stats.len(), 2, "expected 2 components");
    assert_eq!(stats[0].area, 9, "first square area");
    assert_eq!(stats[1].area, 9, "second square area");
}

#[test]
fn connected_components_single_blob() {
    // 5x5 image with one 3x3 blob in center
    let mut data = vec![0.0f32; 5 * 5];
    for y in 1..=3 {
        for x in 1..=3 {
            data[y * 5 + x] = 1.0;
        }
    }
    let img = Tensor::from_vec(vec![5, 5, 1], data).unwrap();
    let (_labels, stats) = connected_components_with_stats(&img).unwrap();
    assert_eq!(stats.len(), 1, "expected 1 component");
    assert_eq!(stats[0].area, 9);
    assert_eq!(stats[0].bbox, (1, 1, 3, 3));
}

#[test]
fn connected_components_empty() {
    let img = Tensor::zeros(vec![5, 5, 1]).unwrap();
    let (_labels, stats) = connected_components_with_stats(&img).unwrap();
    assert_eq!(stats.len(), 0, "expected 0 components");
}

#[test]
fn region_props_correct_area_and_bbox() {
    // Create a 10x10 image with a known 4x3 rectangle at (2,3)-(5,5)
    let mut data = vec![0.0f32; 10 * 10];
    for y in 3..=5 {
        for x in 2..=5 {
            data[y * 10 + x] = 1.0;
        }
    }
    let img = Tensor::from_vec(vec![10, 10, 1], data).unwrap();
    let (labels, _stats) = connected_components_with_stats(&img).unwrap();

    let props = region_props(&labels).unwrap();
    assert_eq!(props.len(), 1);
    let p = &props[0];
    assert_eq!(p.area, 12); // 4 wide x 3 tall
    assert_eq!(p.bbox, (2, 3, 4, 3)); // (x, y, w, h)
    // Centroid should be at center of the rectangle
    assert!((p.centroid.0 - 3.5).abs() < 0.01, "cx={}", p.centroid.0);
    assert!((p.centroid.1 - 4.0).abs() < 0.01, "cy={}", p.centroid.1);
    // Perimeter = boundary pixels. Interior pixels (3,4) and (4,4) have
    // same-label neighbors on all 4 sides, so perimeter = 12 - 2 = 10.
    assert_eq!(p.perimeter, 10.0);
}

#[test]
fn hu_moments_seven_elements() {
    // Create a simple 5x5 image with a bright square in the center
    let mut data = vec![0.0f32; 5 * 5];
    for y in 1..=3 {
        for x in 1..=3 {
            data[y * 5 + x] = 1.0;
        }
    }
    let img = Tensor::from_vec(vec![5, 5, 1], data).unwrap();
    let hu = hu_moments(&img).unwrap();

    // Should return exactly 7 elements
    assert_eq!(hu.len(), 7);

    // h1 should be positive (sum of normalised 2nd-order central moments)
    assert!(hu[0] > 0.0, "h1 should be positive, got {}", hu[0]);

    // For a symmetric shape, h2 should be close to zero (square is symmetric)
    assert!(
        hu[1].abs() < 1e-6,
        "h2 for symmetric square should be ~0, got {}",
        hu[1]
    );
}

#[test]
fn connected_components_8_joins_diagonal_bridge() {
    use super::super::{connected_components_with_stats, connected_components_with_stats_8};
    // Две "капли", соединённые только диагональю: 8-связность видит один
    // компонент, 4-связность — два.
    let mut data = vec![0.0f32; 25];
    data[0] = 1.0; // (0,0)
    data[6] = 1.0; // (1,1) — диагональный мост
    data[12] = 1.0; // (2,2)
    let img = Tensor::from_vec(vec![5, 5, 1], data).unwrap();

    let (_, stats4) = connected_components_with_stats(&img).unwrap();
    let (labels8, stats8) = connected_components_with_stats_8(&img).unwrap();
    assert_eq!(stats4.len(), 3);
    assert_eq!(stats8.len(), 1);
    assert_eq!(stats8[0].area, 3);
    assert_eq!(stats8[0].bbox, (0, 0, 3, 3));
    // все три пикселя получили метку 1
    let ld = labels8.data();
    assert_eq!(ld[0], 1.0);
    assert_eq!(ld[6], 1.0);
    assert_eq!(ld[12], 1.0);
}

#[test]
fn min_enclosing_circle_exact_cases() {
    use super::super::min_enclosing_circle;
    // один пункт — нулевой радиус
    let ((cx, cy), r) = min_enclosing_circle(&[(2.0, 3.0)]).unwrap();
    assert!((cx - 2.0).abs() < 1e-6 && (cy - 3.0).abs() < 1e-6 && r < 1e-6);

    // две точки — диаметр
    let ((cx, cy), r) = min_enclosing_circle(&[(0.0, 0.0), (4.0, 0.0)]).unwrap();
    assert!((cx - 2.0).abs() < 1e-5 && cy.abs() < 1e-5);
    assert!((r - 2.0).abs() < 1e-5);

    // квадрат — окружность через противоположные углы
    let ((cx, cy), r) =
        min_enclosing_circle(&[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]).unwrap();
    assert!((cx - 1.0).abs() < 1e-5 && (cy - 1.0).abs() < 1e-5);
    assert!((r - std::f32::consts::SQRT_2).abs() < 1e-5);

    // точка внутри не влияет
    let ((_, _), r2) =
        min_enclosing_circle(&[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (1.0, 1.0)])
            .unwrap();
    assert!((r2 - std::f32::consts::SQRT_2).abs() < 1e-5);

    assert!(min_enclosing_circle(&[]).is_none());
}

// ── Least-squares circle fit (Kåsa) ─────────────────────────────────

#[test]
fn fit_circle_recovers_exact_circle() {
    use super::super::fit_circle;
    let pts: Vec<(f32, f32)> = (0..16)
        .map(|i| {
            let a = i as f32 / 16.0 * std::f32::consts::TAU;
            (3.0 + 5.0 * a.cos(), -2.0 + 5.0 * a.sin())
        })
        .collect();
    let (cx, cy, r) = fit_circle(&pts).unwrap();
    assert!((cx - 3.0).abs() < 1e-4, "cx = {cx}");
    assert!((cy + 2.0).abs() < 1e-4, "cy = {cy}");
    assert!((r - 5.0).abs() < 1e-4, "r = {r}");
}

#[test]
fn fit_circle_partial_arc_and_degenerate_cases() {
    use super::super::fit_circle;
    // Half arc still pins the circle.
    let pts: Vec<(f32, f32)> = (0..12)
        .map(|i| {
            let a = i as f32 / 12.0 * std::f32::consts::PI;
            (10.0 * a.cos(), 10.0 * a.sin())
        })
        .collect();
    let (cx, cy, r) = fit_circle(&pts).unwrap();
    assert!(cx.abs() < 1e-3 && cy.abs() < 1e-3 && (r - 10.0).abs() < 1e-3);

    // Degenerate inputs refuse instead of inventing a circle.
    assert!(fit_circle(&[(0.0, 0.0), (1.0, 1.0)]).is_none());
    assert!(fit_circle(&[(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]).is_none());
}
