//! Linear Kalman filters.
//!
//! [`LinearKalman`] is the generic small-dimension core: state size `NX` and
//! measurement size `NZ` are const generics, matrices are nested fixed-size
//! arrays. At these sizes (NX ≤ 8 in practice) the monomorphized loops fully
//! unroll and auto-vectorize; explicit SIMD would add `unsafe` for no
//! measurable gain — a whole predict/update cycle is tens of nanoseconds.
//!
//! [`ConstantVelocity2d`] tracks a point in the plane (state `[x, y, vx, vy]`)
//! with a white-acceleration process model and per-step `dt`, so irregular or
//! dropped frames are handled correctly.
//!
//! [`KalmanFilter`] is the bounding-box filter used by the trackers
//! (state `[cx, cy, w, h, vx, vy, vw, vh]`, dt = 1); it is built on the same
//! generic core.

use yscv_detect::BoundingBox;

// ── Generic core ────────────────────────────────────────────────────

/// Generic linear Kalman filter with `NX` state and `NZ` measurement
/// dimensions.
///
/// The caller supplies the model matrices per call (`F`/`Q` for predict,
/// `H`/`R` for update), which keeps the filter free of assumptions about the
/// motion model and lets `dt`-dependent models rebuild them cheaply.
#[derive(Debug, Clone)]
pub struct LinearKalman<const NX: usize, const NZ: usize> {
    /// State estimate.
    pub x: [f32; NX],
    /// Error covariance.
    pub p: [[f32; NX]; NX],
}

impl<const NX: usize, const NZ: usize> LinearKalman<NX, NZ> {
    /// Create a filter from an initial state and covariance.
    pub fn new(x: [f32; NX], p: [[f32; NX]; NX]) -> Self {
        Self { x, p }
    }

    /// Time update: `x ← F x`, `P ← F P Fᵀ + Q`.
    pub fn predict(&mut self, f: &[[f32; NX]; NX], q: &[[f32; NX]; NX]) {
        let mut nx = [0.0f32; NX];
        for (out, f_row) in nx.iter_mut().zip(f) {
            *out = dot(f_row, &self.x);
        }
        self.x = nx;

        let fp = mat_mul(f, &self.p);
        // (F P Fᵀ)[i][j] = (FP)_i · F_j  — Fᵀ never materialized.
        let mut p = *q;
        for (p_row, fp_row) in p.iter_mut().zip(&fp) {
            for (pv, f_row) in p_row.iter_mut().zip(f) {
                *pv += dot(fp_row, f_row);
            }
        }
        self.p = p;
    }

    /// Measurement update with measurement `z`, model `H` and noise `R`.
    ///
    /// Returns `false` (leaving the state untouched) when the innovation
    /// covariance `S = H P Hᵀ + R` is singular.
    pub fn update(&mut self, z: &[f32; NZ], h: &[[f32; NX]; NZ], r: &[[f32; NZ]; NZ]) -> bool {
        // Innovation y = z − H x.
        let mut y = [0.0f32; NZ];
        for ((yv, h_row), &zv) in y.iter_mut().zip(h).zip(z) {
            *yv = zv - dot(h_row, &self.x);
        }

        // P Hᵀ (NX×NZ): rows of P dotted with rows of H.
        let mut pht = [[0.0f32; NZ]; NX];
        for (pht_row, p_row) in pht.iter_mut().zip(&self.p) {
            for (v, h_row) in pht_row.iter_mut().zip(h) {
                *v = dot(p_row, h_row);
            }
        }

        // S = H (P Hᵀ) + R.
        let mut s = mat_mul(h, &pht);
        for (s_row, r_row) in s.iter_mut().zip(r) {
            for (sv, &rv) in s_row.iter_mut().zip(r_row) {
                *sv += rv;
            }
        }
        let Some(s_inv) = invert(&s) else {
            return false;
        };

        // Gain K = P Hᵀ S⁻¹ (NX×NZ).
        let k = mat_mul(&pht, &s_inv);

        // x ← x + K y.
        for (xv, k_row) in self.x.iter_mut().zip(&k) {
            *xv += dot(k_row, &y);
        }

        // P ← (I − K H) P  ⇒  P −= (K H) P.
        let kh = mat_mul(&k, h);
        let khp = mat_mul(&kh, &self.p);
        for (p_row, khp_row) in self.p.iter_mut().zip(&khp) {
            for (pv, &dv) in p_row.iter_mut().zip(khp_row) {
                *pv -= dv;
            }
        }
        true
    }
}

#[inline]
fn dot<const N: usize>(a: &[f32; N], b: &[f32; N]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `A (RA×RB) · B (RB×RC)` with an i-k-j loop order: the inner loop is a
/// contiguous axpy over the output row, which auto-vectorizes.
fn mat_mul<const RA: usize, const RB: usize, const RC: usize>(
    a: &[[f32; RB]; RA],
    b: &[[f32; RC]; RB],
) -> [[f32; RC]; RA] {
    let mut out = [[0.0f32; RC]; RA];
    for (out_row, a_row) in out.iter_mut().zip(a) {
        for (&av, b_row) in a_row.iter().zip(b) {
            for (ov, &bv) in out_row.iter_mut().zip(b_row) {
                *ov += av * bv;
            }
        }
    }
    out
}

/// Gauss-Jordan inverse with partial pivoting; `None` for singular input.
fn invert<const N: usize>(m: &[[f32; N]; N]) -> Option<[[f32; N]; N]> {
    let mut a = *m;
    let mut inv = [[0.0f32; N]; N];
    for (i, row) in inv.iter_mut().enumerate() {
        row[i] = 1.0;
    }

    for col in 0..N {
        let pivot = (col..N).max_by(|&r1, &r2| a[r1][col].abs().total_cmp(&a[r2][col].abs()))?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        inv.swap(col, pivot);

        let inv_p = 1.0 / a[col][col];
        for v in &mut a[col] {
            *v *= inv_p;
        }
        for v in &mut inv[col] {
            *v *= inv_p;
        }

        let a_piv = a[col];
        let inv_piv = inv[col];
        for row in 0..N {
            if row == col {
                continue;
            }
            let f = a[row][col];
            if f == 0.0 {
                continue;
            }
            for (v, &pv) in a[row].iter_mut().zip(&a_piv) {
                *v -= f * pv;
            }
            for (v, &pv) in inv[row].iter_mut().zip(&inv_piv) {
                *v -= f * pv;
            }
        }
    }
    Some(inv)
}

// ── Constant-velocity point filter ──────────────────────────────────

/// Constant-velocity Kalman filter for a point in the plane.
///
/// State is `[x, y, vx, vy]`, the measurement is `[x, y]`. The process noise
/// is the standard continuous white-acceleration model with standard
/// deviation `process_noise` (units of acceleration), integrated over the
/// per-step `dt` passed to [`predict`](Self::predict) — irregular frame
/// intervals and dropped frames are therefore handled exactly.
#[derive(Debug, Clone)]
pub struct ConstantVelocity2d {
    q: f32,
    r: f32,
    /// Underlying generic filter; `x = [x, y, vx, vy]`.
    pub filter: LinearKalman<4, 2>,
}

impl ConstantVelocity2d {
    /// Create a filter at position `(x, y)` with zero initial velocity.
    ///
    /// `process_noise` is the white-acceleration σ, `measurement_noise` the
    /// position measurement σ (same length unit as `x`/`y`).
    pub fn new(process_noise: f32, measurement_noise: f32, x: f32, y: f32) -> Self {
        let r2 = measurement_noise * measurement_noise;
        let mut p = [[0.0f32; 4]; 4];
        p[0][0] = r2;
        p[1][1] = r2;
        p[2][2] = 1e4;
        p[3][3] = 1e4;
        Self {
            q: process_noise,
            r: measurement_noise,
            filter: LinearKalman::new([x, y, 0.0, 0.0], p),
        }
    }

    /// Advance the state by `dt`.
    pub fn predict(&mut self, dt: f32) {
        let f = [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let q2 = self.q * self.q;
        let d3 = dt * dt * dt / 3.0;
        let d2 = dt * dt / 2.0;
        let q = [
            [q2 * d3, 0.0, q2 * d2, 0.0],
            [0.0, q2 * d3, 0.0, q2 * d2],
            [q2 * d2, 0.0, q2 * dt, 0.0],
            [0.0, q2 * d2, 0.0, q2 * dt],
        ];
        self.filter.predict(&f, &q);
    }

    /// Fold in a position measurement. No-op on a singular innovation
    /// covariance (returns `false`).
    pub fn update(&mut self, x: f32, y: f32) -> bool {
        let h = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]];
        let r2 = self.r * self.r;
        let r = [[r2, 0.0], [0.0, r2]];
        self.filter.update(&[x, y], &h, &r)
    }

    /// Current position estimate.
    pub fn position(&self) -> (f32, f32) {
        (self.filter.x[0], self.filter.x[1])
    }

    /// Current velocity estimate.
    pub fn velocity(&self) -> (f32, f32) {
        (self.filter.x[2], self.filter.x[3])
    }
}

// ── Bounding-box filter (used by the trackers) ──────────────────────

/// Dimension of the state vector.
const STATE_DIM: usize = 8;
/// Dimension of the measurement vector.
const MEAS_DIM: usize = 4;

/// A 2-D Kalman filter for bounding-box tracking.
///
/// State vector: `[cx, cy, w, h, vx, vy, vw, vh]` — center position, size,
/// and their velocities. The measurement is `[cx, cy, w, h]`; the time step
/// is one frame (`dt = 1`).
#[derive(Debug, Clone)]
pub struct KalmanFilter {
    core: LinearKalman<STATE_DIM, MEAS_DIM>,
    q: [[f32; STATE_DIM]; STATE_DIM],
    r: [[f32; MEAS_DIM]; MEAS_DIM],
}

impl KalmanFilter {
    /// Create a Kalman filter initialized from a bounding box.
    pub fn new(bbox: BoundingBox) -> Self {
        let cx = (bbox.x1 + bbox.x2) * 0.5;
        let cy = (bbox.y1 + bbox.y2) * 0.5;

        let mut x = [0.0f32; STATE_DIM];
        x[0] = cx;
        x[1] = cy;
        x[2] = bbox.width();
        x[3] = bbox.height();
        // velocities start at zero

        // Initial covariance: large uncertainty on velocities.
        let mut p = [[0.0f32; STATE_DIM]; STATE_DIM];
        let mut q = [[0.0f32; STATE_DIM]; STATE_DIM];
        for i in 0..4 {
            p[i][i] = 10.0;
            q[i][i] = 1.0;
            p[i + 4][i + 4] = 100.0;
            q[i + 4][i + 4] = 0.01;
        }
        let mut r = [[0.0f32; MEAS_DIM]; MEAS_DIM];
        for (i, row) in r.iter_mut().enumerate() {
            row[i] = 1.0;
        }

        Self {
            core: LinearKalman::new(x, p),
            q,
            r,
        }
    }

    /// Predict the next state (one time step, dt = 1).
    pub fn predict(&mut self) {
        // F = identity + position-velocity coupling at dt = 1.
        let mut f = [[0.0f32; STATE_DIM]; STATE_DIM];
        for (i, row) in f.iter_mut().enumerate() {
            row[i] = 1.0;
            if i < 4 {
                row[i + 4] = 1.0;
            }
        }
        self.core.predict(&f, &self.q);
    }

    /// Update the filter with a measurement `[cx, cy, w, h]`.
    pub fn update(&mut self, measurement: [f32; MEAS_DIM]) {
        // H selects the first four state components.
        let mut h = [[0.0f32; STATE_DIM]; MEAS_DIM];
        for (i, row) in h.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        // A singular S leaves the state as-is.
        let _ = self.core.update(&measurement, &h, &self.r);
    }

    /// Get current state as bounding box.
    pub fn bbox(&self) -> BoundingBox {
        let x = &self.core.x;
        let w = x[2].max(1e-3);
        let h = x[3].max(1e-3);
        BoundingBox {
            x1: x[0] - w * 0.5,
            y1: x[1] - h * 0.5,
            x2: x[0] + w * 0.5,
            y2: x[1] + h * 0.5,
        }
    }

    /// Get predicted bbox without mutating state.
    pub fn predicted_bbox(&self) -> BoundingBox {
        let x = &self.core.x;
        let cx = x[0] + x[4];
        let cy = x[1] + x[5];
        let w = (x[2] + x[6]).max(1e-3);
        let h = (x[3] + x[7]).max(1e-3);
        BoundingBox {
            x1: cx - w * 0.5,
            y1: cy - h * 0.5,
            x2: cx + w * 0.5,
            y2: cy + h * 0.5,
        }
    }
}
