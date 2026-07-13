use crate::KalmanFilter;
use yscv_detect::BoundingBox;

#[test]
fn kalman_initial_bbox_round_trip() {
    let bbox = BoundingBox {
        x1: 10.0,
        y1: 20.0,
        x2: 30.0,
        y2: 50.0,
    };
    let kf = KalmanFilter::new(bbox);
    let out = kf.bbox();
    assert!((out.x1 - 10.0).abs() < 1e-3);
    assert!((out.y1 - 20.0).abs() < 1e-3);
    assert!((out.x2 - 30.0).abs() < 1e-3);
    assert!((out.y2 - 50.0).abs() < 1e-3);
}

#[test]
fn kalman_predict_moves_state() {
    let bbox = BoundingBox {
        x1: 0.0,
        y1: 0.0,
        x2: 10.0,
        y2: 10.0,
    };
    let mut kf = KalmanFilter::new(bbox);
    kf.predict();
    let out = kf.bbox();
    assert!((out.x1 - 0.0).abs() < 1.0);
    assert!((out.y1 - 0.0).abs() < 1.0);
}

#[test]
fn kalman_update_converges() {
    let bbox = BoundingBox {
        x1: 0.0,
        y1: 0.0,
        x2: 10.0,
        y2: 10.0,
    };
    let mut kf = KalmanFilter::new(bbox);
    for _ in 0..10 {
        kf.predict();
        kf.update([15.0, 15.0, 10.0, 10.0]);
    }
    let out = kf.bbox();
    let cx = (out.x1 + out.x2) * 0.5;
    let cy = (out.y1 + out.y2) * 0.5;
    assert!((cx - 15.0).abs() < 1.0);
    assert!((cy - 15.0).abs() < 1.0);
}

#[test]
fn kalman_predicted_bbox_no_mutation() {
    let bbox = BoundingBox {
        x1: 0.0,
        y1: 0.0,
        x2: 10.0,
        y2: 10.0,
    };
    let kf = KalmanFilter::new(bbox);
    let pred = kf.predicted_bbox();
    let current = kf.bbox();
    assert!((pred.x1 - current.x1).abs() < 1e-3);
}

// ── LinearKalman / ConstantVelocity2d ───────────────────────────────

use crate::{ConstantVelocity2d, LinearKalman};

#[test]
fn linear_kalman_converges_on_static_target_1d() {
    // 1-state random walk observed directly.
    let mut kf: LinearKalman<1, 1> = LinearKalman::new([0.0], [[100.0]]);
    let f = [[1.0]];
    let q = [[0.01]];
    let h = [[1.0]];
    let r = [[1.0]];
    for _ in 0..50 {
        kf.predict(&f, &q);
        assert!(kf.update(&[5.0], &h, &r));
    }
    assert!((kf.x[0] - 5.0).abs() < 0.1);
}

#[test]
fn linear_kalman_singular_innovation_is_a_noop() {
    let mut kf: LinearKalman<2, 2> = LinearKalman::new([1.0, 2.0], [[0.0; 2]; 2]);
    let before = kf.x;
    // P = 0 and R = 0 ⇒ S = 0 (singular): update must refuse and not touch x.
    let updated = kf.update(&[9.0, 9.0], &[[1.0, 0.0], [0.0, 1.0]], &[[0.0; 2]; 2]);
    assert!(!updated);
    assert_eq!(kf.x, before);
}

#[test]
fn linear_kalman_predict_grows_uncertainty() {
    let mut kf: LinearKalman<2, 1> = LinearKalman::new([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]);
    let f = [[1.0, 1.0], [0.0, 1.0]];
    let q = [[0.1, 0.0], [0.0, 0.1]];
    kf.predict(&f, &q);
    // var(x) = p00 + 2 p01 + p11 + q00 = 1 + 0 + 1 + 0.1
    assert!((kf.p[0][0] - 2.1).abs() < 1e-6);
}

#[test]
fn constant_velocity_tracks_moving_point() {
    // Point moving at (100, -50) units/s, measured every 0.1 s.
    let mut kf = ConstantVelocity2d::new(50.0, 1.0, 0.0, 0.0);
    for step in 1..=30 {
        let t = step as f32 * 0.1;
        kf.predict(0.1);
        kf.update(100.0 * t, -50.0 * t);
    }
    let (x, y) = kf.position();
    let (vx, vy) = kf.velocity();
    assert!((x - 300.0).abs() < 2.0, "x = {x}");
    assert!((y + 150.0).abs() < 2.0, "y = {y}");
    assert!((vx - 100.0).abs() < 5.0, "vx = {vx}");
    assert!((vy + 50.0).abs() < 5.0, "vy = {vy}");
}

#[test]
fn constant_velocity_handles_irregular_dt() {
    // Same trajectory sampled at alternating 0.05/0.15 s intervals: the
    // dt-aware model must keep predicting on the line between updates.
    let mut kf = ConstantVelocity2d::new(50.0, 1.0, 0.0, 0.0);
    let mut t = 0.0f32;
    for step in 0..40 {
        let dt = if step % 2 == 0 { 0.05 } else { 0.15 };
        t += dt;
        kf.predict(dt);
        kf.update(10.0 * t, 0.0);
    }
    kf.predict(0.5); // coast half a second with no measurement
    let (x, _) = kf.position();
    assert!((x - 10.0 * (t + 0.5)).abs() < 1.0, "coasted x = {x}");
}
