/// Camera intrinsic parameters including radial and tangential distortion coefficients.
#[derive(Debug, Clone, PartialEq)]
pub struct CameraIntrinsics {
    /// Focal length in x (pixels).
    pub fx: f32,
    /// Focal length in y (pixels).
    pub fy: f32,
    /// Principal point x (pixels).
    pub cx: f32,
    /// Principal point y (pixels).
    pub cy: f32,
    /// First radial distortion coefficient.
    pub k1: f32,
    /// Second radial distortion coefficient.
    pub k2: f32,
    /// First tangential distortion coefficient.
    pub p1: f32,
    /// Second tangential distortion coefficient.
    pub p2: f32,
}

impl CameraIntrinsics {
    /// Create intrinsics with zero distortion.
    pub const fn new(fx: f32, fy: f32, cx: f32, cy: f32) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }
}

/// Remove lens distortion from a set of 2-D pixel coordinates.
///
/// Each point is first normalised to camera coordinates, the Brown–Conrady
/// distortion model is applied in reverse (one Newton-style iteration is
/// typically sufficient for small distortion), and the result is projected
/// back to pixel coordinates.
pub fn undistort_points(points: &[(f32, f32)], intrinsics: &CameraIntrinsics) -> Vec<(f32, f32)> {
    let CameraIntrinsics {
        fx,
        fy,
        cx,
        cy,
        k1,
        k2,
        p1,
        p2,
    } = *intrinsics;

    points
        .iter()
        .map(|&(px, py)| {
            // Normalise to camera coords.
            let x = (px - cx) / fx;
            let y = (py - cy) / fy;

            // Iterative undistortion (5 iterations).
            let mut xd = x;
            let mut yd = y;
            for _ in 0..5 {
                let r2 = xd * xd + yd * yd;
                let radial = 1.0 + k1 * r2 + k2 * r2 * r2;
                let dx = 2.0 * p1 * xd * yd + p2 * (r2 + 2.0 * xd * xd);
                let dy = p1 * (r2 + 2.0 * yd * yd) + 2.0 * p2 * xd * yd;
                xd = (x - dx) / radial;
                yd = (y - dy) / radial;
            }

            (xd * fx + cx, yd * fy + cy)
        })
        .collect()
}

/// Project 3-D world points to 2-D pixel coordinates using a pinhole camera model
/// (no extrinsic rotation/translation — assumes camera-frame coordinates).
pub fn project_points(
    points_3d: &[(f32, f32, f32)],
    intrinsics: &CameraIntrinsics,
) -> Vec<(f32, f32)> {
    let CameraIntrinsics {
        fx,
        fy,
        cx,
        cy,
        k1,
        k2,
        p1,
        p2,
    } = *intrinsics;

    points_3d
        .iter()
        .map(|&(x, y, z)| {
            let xn = x / z;
            let yn = y / z;

            let r2 = xn * xn + yn * yn;
            let radial = 1.0 + k1 * r2 + k2 * r2 * r2;
            let xd = xn * radial + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn);
            let yd = yn * radial + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn;

            (fx * xd + cx, fy * yd + cy)
        })
        .collect()
}

/// Triangulate 3-D points from corresponding 2-D observations in a rectified
/// stereo pair.
///
/// Uses the disparity `d = pts_a.x - pts_b.x` together with the known
/// `baseline` (distance between cameras) and shared `focal` length to recover
/// depth:
///
/// ```text
/// Z = baseline * focal / disparity
/// X = (u - cx) * Z / focal   (cx assumed 0 here)
/// Y = (v - cy) * Z / focal   (cy assumed 0 here)
/// ```
///
/// Points with zero or negative disparity are placed at the origin `(0, 0, 0)`.
pub fn triangulate_points(
    pts_a: &[(f32, f32)],
    pts_b: &[(f32, f32)],
    baseline: f32,
    focal: f32,
) -> Vec<(f32, f32, f32)> {
    pts_a
        .iter()
        .zip(pts_b.iter())
        .map(|(&(ax, ay), &(bx, _by))| {
            let d = ax - bx;
            if d <= 0.0 {
                return (0.0, 0.0, 0.0);
            }
            let z = baseline * focal / d;
            let x = ax * z / focal;
            let y = ay * z / focal;
            (x, y, z)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_project_points_no_distortion() {
        let intr = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0);
        let pts = vec![(0.0f32, 0.0, 1.0)];
        let proj = project_points(&pts, &intr);
        assert!(approx_eq(proj[0].0, 320.0, 1e-5));
        assert!(approx_eq(proj[0].1, 240.0, 1e-5));
    }

    #[test]
    fn test_project_and_undistort_roundtrip_no_distortion() {
        let intr = CameraIntrinsics::new(600.0, 600.0, 300.0, 200.0);
        let pts_3d = vec![(1.0f32, -0.5, 3.0), (-2.0, 1.0, 5.0)];
        let projected = project_points(&pts_3d, &intr);
        // With zero distortion, undistort should be identity.
        let undistorted = undistort_points(&projected, &intr);
        for (p, u) in projected.iter().zip(undistorted.iter()) {
            assert!(approx_eq(p.0, u.0, 1e-4));
            assert!(approx_eq(p.1, u.1, 1e-4));
        }
    }

    #[test]
    fn test_undistort_with_distortion() {
        let intr = CameraIntrinsics {
            fx: 500.0,
            fy: 500.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.1,
            k2: 0.01,
            p1: 0.0,
            p2: 0.0,
        };
        // A point at the principal point should be unchanged regardless of distortion.
        let pts = vec![(320.0f32, 240.0)];
        let result = undistort_points(&pts, &intr);
        assert!(approx_eq(result[0].0, 320.0, 1e-4));
        assert!(approx_eq(result[0].1, 240.0, 1e-4));
    }

    #[test]
    fn test_triangulate_known_depth() {
        // baseline=0.1m, focal=500px, disparity=50px → Z = 0.1*500/50 = 1.0
        let pts_a = vec![(250.0f32, 100.0)];
        let pts_b = vec![(200.0f32, 100.0)];
        let result = triangulate_points(&pts_a, &pts_b, 0.1, 500.0);
        assert!(approx_eq(result[0].2, 1.0, 1e-5));
        // X = 250 * 1.0 / 500 = 0.5
        assert!(approx_eq(result[0].0, 0.5, 1e-5));
    }

    #[test]
    fn test_triangulate_zero_disparity() {
        let pts_a = vec![(100.0f32, 100.0)];
        let pts_b = vec![(100.0f32, 100.0)];
        let result = triangulate_points(&pts_a, &pts_b, 0.1, 500.0);
        assert!(approx_eq(result[0].0, 0.0, 1e-5));
        assert!(approx_eq(result[0].1, 0.0, 1e-5));
        assert!(approx_eq(result[0].2, 0.0, 1e-5));
    }
}
