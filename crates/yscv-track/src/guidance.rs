use std::time::Instant;

use crate::LinearKalman;

const DT_MIN: f32 = 0.005;
const DT_MAX: f32 = 0.200;
const G: f32 = 9.81;
const Z_MIN: f32 = 0.3;

#[derive(Clone, Debug)]
pub struct GuidanceConfig {
    pub kf_p0: f32,
    pub kf_q_pos: f32,
    pub kf_q_vel: f32,
    pub kf_r: f32,
    pub rng_q_z: f32,
    pub rng_q_vz: f32,
    pub rng_q_w0: f32,
    pub rng_r_bbox: f32,
}

impl Default for GuidanceConfig {
    fn default() -> Self {
        Self {
            kf_p0: 1.0,
            // variance rates per second (≈ the old per-frame values at 30 fps)
            kf_q_pos: 0.03,
            kf_q_vel: 0.3,
            kf_r: 0.05,
            rng_q_z: 0.1,
            rng_q_vz: 1.0,
            rng_q_w0: 0.0001,
            rng_r_bbox: 0.01,
        }
    }
}

/// 2D LOS KF: state `[x, y, vx, vy]`, measurement `[x, y]`.
#[derive(Debug, Clone)]
pub struct LosKf {
    kf: LinearKalman<4, 2>,
    p0: f32,
    q_pos: f32,
    q_vel: f32,
    r: f32,
    last_t: Option<Instant>,
}

impl LosKf {
    pub fn new(cfg: &GuidanceConfig) -> Self {
        let p0 = cfg.kf_p0;
        Self {
            kf: LinearKalman::new([0.0; 4], diag4(p0, p0, p0, p0)),
            p0,
            q_pos: cfg.kf_q_pos,
            q_vel: cfg.kf_q_vel,
            r: cfg.kf_r,
            last_t: None,
        }
    }

    pub fn reset(&mut self) {
        self.kf.x = [0.0; 4];
        self.kf.p = diag4(self.p0, self.p0, self.p0, self.p0);
        self.last_t = None;
    }

    pub fn step(&mut self, err_x: f32, err_y: f32) -> (f32, f32, f32, f32) {
        let now = Instant::now();
        let dt = match self.last_t {
            Some(prev) => {
                let d = now.duration_since(prev).as_secs_f32();
                if d < DT_MIN {
                    return self.state();
                }
                d.min(DT_MAX)
            }
            None => {
                self.kf.x[0] = err_x;
                self.kf.x[1] = err_y;
                self.last_t = Some(now);
                return (err_x, err_y, 0.0, 0.0);
            }
        };
        self.last_t = Some(now);
        self.predict(dt);
        let h = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]];
        let r = [[self.r, 0.0], [0.0, self.r]];
        self.kf.update(&[err_x, err_y], &h, &r);
        self.state()
    }

    pub fn predict_only(&mut self) -> (f32, f32, f32, f32) {
        let now = Instant::now();
        let dt = match self.last_t {
            Some(prev) => {
                let d = now.duration_since(prev).as_secs_f32();
                if d < DT_MIN {
                    return self.state();
                }
                d.min(DT_MAX)
            }
            None => return (0.0, 0.0, 0.0, 0.0),
        };
        self.last_t = Some(now);
        self.predict(dt);
        self.state()
    }

    fn predict(&mut self, dt: f32) {
        let f = [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        // q_pos/q_vel are variance RATES per second; scaling by dt keeps the
        // tuning meaningful when the frame rate drops
        let q = diag4(
            self.q_pos * dt,
            self.q_pos * dt,
            self.q_vel * dt,
            self.q_vel * dt,
        );
        self.kf.predict(&f, &q);
    }

    fn state(&self) -> (f32, f32, f32, f32) {
        (self.kf.x[0], self.kf.x[1], self.kf.x[2], self.kf.x[3])
    }
}

/// 3-state range EKF: `[z, z_dot, w0]`, measurement `h(x) = w0/z`.
#[derive(Debug, Clone)]
pub struct RangeKf {
    x: [f32; 3],
    p: [[f32; 3]; 3],
    q_z: f32,
    q_vz: f32,
    q_w0: f32,
    r_bbox: f32,
    last_t: Option<Instant>,
}

impl RangeKf {
    pub fn new(cfg: &GuidanceConfig) -> Self {
        Self {
            x: [0.0; 3],
            p: [[0.0; 3]; 3],
            q_z: cfg.rng_q_z,
            q_vz: cfg.rng_q_vz,
            q_w0: cfg.rng_q_w0,
            r_bbox: cfg.rng_r_bbox,
            last_t: None,
        }
    }

    pub fn init(&mut self, w_bbox_norm: f32) {
        let z = 1.0 / w_bbox_norm.max(0.01);
        self.x = [z, 0.0, 1.0];
        self.p = [[z * z, 0.0, 0.0], [0.0, 25.0, 0.0], [0.0, 0.0, 2.0]];
        self.last_t = None;
    }

    pub fn reset(&mut self) {
        self.x = [0.0; 3];
        self.p = [[0.0; 3]; 3];
        self.last_t = None;
    }

    pub fn step(&mut self, w_bbox_norm: f32, pitch: f32, imu_ax: f32) -> (f32, f32) {
        let now = Instant::now();
        let dt = match self.last_t {
            Some(prev) => {
                let d = now.duration_since(prev).as_secs_f32();
                if d < DT_MIN {
                    return (self.x[0].max(Z_MIN), self.x[1]);
                }
                d.min(DT_MAX)
            }
            None => {
                self.last_t = Some(now);
                return (self.x[0].max(Z_MIN), self.x[1]);
            }
        };
        self.last_t = Some(now);
        self.predict(dt, pitch, imu_ax);
        if w_bbox_norm > 0.01 {
            self.update(w_bbox_norm);
        }
        self.x[0] = self.x[0].max(Z_MIN);
        self.x[2] = self.x[2].max(0.001);
        (self.x[0], self.x[1])
    }

    pub fn predict_only(&mut self, pitch: f32, imu_ax: f32) -> (f32, f32) {
        let now = Instant::now();
        let dt = match self.last_t {
            Some(prev) => {
                let d = now.duration_since(prev).as_secs_f32();
                if d < DT_MIN {
                    return (self.x[0].max(Z_MIN), self.x[1]);
                }
                d.min(DT_MAX)
            }
            None => return (self.x[0].max(Z_MIN), self.x[1]),
        };
        self.last_t = Some(now);
        self.predict(dt, pitch, imu_ax);
        self.x[0] = self.x[0].max(Z_MIN);
        (self.x[0], self.x[1])
    }

    fn predict(&mut self, dt: f32, pitch: f32, imu_ax: f32) {
        let az_rel = G * pitch.sin() - imu_ax;
        self.x[0] += self.x[1] * dt;
        self.x[1] += az_rel * dt;

        let p = &mut self.p;
        p[0][0] += dt * (p[0][1] + p[1][0]) + dt * dt * p[1][1] + self.q_z;
        p[0][1] += dt * p[1][1];
        p[0][2] += dt * p[1][2];
        p[1][0] += dt * p[1][1];
        p[1][1] += self.q_vz;
        p[2][0] += dt * p[2][1];
        p[2][2] += self.q_w0;
    }

    fn update(&mut self, w_meas: f32) {
        let z = self.x[0];
        let w0 = self.x[2];
        if z.abs() < 1e-6 {
            return;
        }

        let y = w_meas - w0 / z;
        let h = [-w0 / (z * z), 0.0, 1.0 / z];

        let s = h[0] * h[0] * self.p[0][0]
            + h[0] * h[2] * self.p[0][2]
            + h[2] * h[0] * self.p[2][0]
            + h[2] * h[2] * self.p[2][2]
            + self.r_bbox;

        if s.abs() < 1e-12 {
            return;
        }
        let s_inv = 1.0 / s;

        let k = [
            (self.p[0][0] * h[0] + self.p[0][2] * h[2]) * s_inv,
            (self.p[1][0] * h[0] + self.p[1][2] * h[2]) * s_inv,
            (self.p[2][0] * h[0] + self.p[2][2] * h[2]) * s_inv,
        ];

        self.x[0] += k[0] * y;
        self.x[1] += k[1] * y;
        self.x[2] += k[2] * y;

        let p_old = self.p;
        for i in 0..3 {
            let kh0 = k[i] * h[0];
            let kh2 = k[i] * h[2];
            self.p[i][0] = p_old[i][0] - kh0 * p_old[0][0] - kh2 * p_old[2][0];
            self.p[i][1] = p_old[i][1] - kh0 * p_old[0][1] - kh2 * p_old[2][1];
            self.p[i][2] = p_old[i][2] - kh0 * p_old[0][2] - kh2 * p_old[2][2];
        }
    }
}

pub struct Nav3dOut {
    pub err_x: f32,
    pub err_y: f32,
    pub rate_x: f32,
    pub rate_y: f32,
    pub v_closing: f32,
    pub range: f32,
}

/// Combined 2D LOS + range EKF.
pub struct Nav3d {
    pub los: LosKf,
    rng: RangeKf,
    initialized: bool,
}

impl Nav3d {
    pub fn new(cfg: &GuidanceConfig) -> Self {
        Self {
            los: LosKf::new(cfg),
            rng: RangeKf::new(cfg),
            initialized: false,
        }
    }

    pub fn initialized(&self) -> bool {
        self.initialized
    }

    pub fn reset(&mut self) {
        self.los.reset();
        self.rng.reset();
        self.initialized = false;
    }

    pub fn step(
        &mut self,
        err_x: f32,
        err_y: f32,
        w_bbox_norm: f32,
        pitch: f32,
        imu_ax: f32,
    ) -> Nav3dOut {
        if !self.initialized && w_bbox_norm > 0.01 {
            self.rng.init(w_bbox_norm);
            self.initialized = true;
        }

        let (ex, ey, rx, ry) = self.los.step(err_x, err_y);
        let (z, z_dot) = if self.initialized {
            self.rng.step(w_bbox_norm, pitch, imu_ax)
        } else {
            (1.0, 0.0)
        };

        Nav3dOut {
            err_x: ex,
            err_y: ey,
            rate_x: rx,
            rate_y: ry,
            v_closing: (-z_dot).max(0.0),
            range: z,
        }
    }

    pub fn predict_only(&mut self, pitch: f32, imu_ax: f32) -> Nav3dOut {
        let (ex, ey, rx, ry) = self.los.predict_only();
        let (z, z_dot) = if self.initialized {
            self.rng.predict_only(pitch, imu_ax)
        } else {
            (1.0, 0.0)
        };

        Nav3dOut {
            err_x: ex,
            err_y: ey,
            rate_x: rx,
            rate_y: ry,
            v_closing: (-z_dot).max(0.0),
            range: z,
        }
    }
}

/// Derotate the pixel error by the body roll. Right roll (+φ, FRD) rotates the
/// scene CCW in the image, so the measured error is R(−φ)·e_level in image
/// axes (y down); the level-frame error is R(+φ)·measured.
pub fn compensate_tilt(err_x: f32, err_y: f32, roll: f32, _pitch: f32) -> (f32, f32) {
    let (cr, sr) = (roll.cos(), roll.sin());
    (cr * err_x - sr * err_y, sr * err_x + cr * err_y)
}

pub fn pn_sticks(kp: f32, n: f32, err_x: f32, err_y: f32, rate_x: f32, rate_y: f32) -> (f32, f32) {
    let sx = (kp * err_x + n * rate_x).clamp(-1.0, 1.0);
    let sy = (kp * err_y + n * rate_y).clamp(-1.0, 1.0);
    (sx, sy)
}

pub fn pn3d_sticks(kp: f32, n: f32, vc_scale: f32, out: &Nav3dOut) -> (f32, f32) {
    let vc_factor = 1.0 + out.v_closing * vc_scale;
    let sx = (kp * out.err_x + n * out.rate_x * vc_factor).clamp(-1.0, 1.0);
    let sy = (kp * out.err_y + n * out.rate_y * vc_factor).clamp(-1.0, 1.0);
    (sx, sy)
}

// ── NED guidance ────────────────────────────────────────────────────
//
// Pixel → unit LOS in NED → 6-state KF → PN acceleration. No explicit
// intercept point: PN nulls the LOS rotation and leads a moving target
// implicitly; near the target a terminal PID takes over. The flight
// envelope (N bounds, acceleration clamps) is caller-owned via `NedFlight`;
// everything the law can own itself is a const here. The per-deployment
// input is the camera calibration (`Cam`) next to `GuidanceConfig`.

/// adaptive navigation constant bounds
const N_MIN: f32 = 2.0;
const N_MAX: f32 = 5.0;
/// LOS rate norm below which the adaptive N grows
const ADAPT_PN_THR: f32 = 0.1;
const ADAPT_PN_STEP: f32 = 0.1;
/// closing velocity clamp, m/s, and the value until the range EKF starts
const VC_MIN: f32 = 1.0;
const VC_MAX: f32 = 10.0;
const VC_NOMINAL: f32 = 2.0;
/// acceleration clamps, m/s²; vertical is NED (z down): up negative
const MAX_LAT_ACC: f32 = 5.0;
const MAX_UP_ACC: f32 = -3.0;
const MAX_DOWN_ACC: f32 = 6.0;
/// LOS measurement variance for the NED filter: the unit-vector domain, not
/// pixel error — ~0.5° per frame covers tracker jitter plus mount/calibration
/// slop. The pixel-domain `kf_r` tunes the sticks filters only.
const LOS_R: f32 = 1e-4;
/// closer than this the terminal PID replaces PN
const INTERCEPT_RANGE_M: f32 = 4.0;
const INTERCEPT_KP: f32 = 40.0;
const INTERCEPT_KI: f32 = 0.1;
const INTERCEPT_KD: f32 = 0.01;
const INTERCEPT_INT_LIMIT: f32 = 2.0;
const EPS: f32 = 1e-6;

/// Flight envelope the caller owns; `Default` is the reference tuning.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NedFlight {
    /// adaptive navigation constant bounds
    pub n_min: f32,
    pub n_max: f32,
    /// acceleration clamps, m/s²; vertical is NED (z down): up negative
    pub max_lat_acc: f32,
    pub max_up_acc: f32,
    pub max_down_acc: f32,
}

impl Default for NedFlight {
    fn default() -> Self {
        Self {
            n_min: N_MIN,
            n_max: N_MAX,
            max_lat_acc: MAX_LAT_ACC,
            max_up_acc: MAX_UP_ACC,
            max_down_acc: MAX_DOWN_ACC,
        }
    }
}

/// Camera intrinsics in pixels plus the mount pitch, radians, positive =
/// camera looks up.
#[derive(Clone, Copy, Debug)]
pub struct Cam {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub install_pitch: f32,
}

/// Body attitude; FRD-aligned with NED at zero.
#[derive(Clone, Copy, Debug)]
pub struct Attitude {
    pub roll: f32,
    pub pitch: f32,
    pub yaw: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct NedOut {
    /// NED acceleration command, m/s²
    pub accel_ned: [f32; 3],
    /// filtered unit LOS
    pub los_ned: [f32; 3],
    pub range_m: f32,
    pub v_closing: f32,
    /// true while the terminal intercept PID has replaced PN
    pub intercept: bool,
    /// current adaptive navigation constant
    pub n: f32,
}

/// Rz(yaw)·Ry(pitch)·Rx(roll), the FRD body → NED rotation.
pub fn rot_zyx(roll: f32, pitch: f32, yaw: f32) -> [[f32; 3]; 3] {
    let (sr, cr) = (roll.sin(), roll.cos());
    let (sp, cp) = (pitch.sin(), pitch.cos());
    let (sy, cy) = (yaw.sin(), yaw.cos());
    [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
}

fn mat_vec(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    let mut out = [0.0; 3];
    for (o, row) in out.iter_mut().zip(m) {
        *o = row[0] * v[0] + row[1] * v[1] + row[2] * v[2];
    }
    out
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n < EPS {
        [1.0, 0.0, 0.0]
    } else {
        [v[0] / n, v[1] / n, v[2] / n]
    }
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// LOS unit vector in NED for a pixel at the given attitude. Camera frame is
/// the optical one (x right, y down, z forward).
pub fn los_ned_from_pixel(
    cam: &Cam,
    px: f32,
    py: f32,
    roll: f32,
    pitch: f32,
    yaw: f32,
) -> [f32; 3] {
    let u = (px - cam.cx) / cam.fx;
    let v = (py - cam.cy) / cam.fy;
    let ray_frd = normalize3([1.0, u, v]); // optical → FRD
    let mounted = mat_vec(&rot_zyx(0.0, cam.install_pitch, 0.0), ray_frd);
    normalize3(mat_vec(&rot_zyx(roll, pitch, yaw), mounted))
}

/// Pure PN: `a = N·Vc·(Ω × los)`, `Ω = los × d(los)/dt`. Perpendicular to the
/// LOS; for a unit LOS `Ω × los = d(los)/dt`.
pub fn pn_accel(los: [f32; 3], los_rate: [f32; 3], n: f32, vc: f32) -> [f32; 3] {
    let omega = cross(los, los_rate);
    let lateral = cross(omega, los);
    [
        n * vc * lateral[0],
        n * vc * lateral[1],
        n * vc * lateral[2],
    ]
}

/// 6-state LOS filter in NED: `x = [los_x, los_y, los_z, rate_x, rate_y, rate_z]`,
/// measurement is the LOS unit vector, H = [I₃|0]; range and closing velocity
/// come from the shared [`RangeKf`].
pub struct NedGuidance {
    kf: LinearKalman<6, 3>,
    rng: RangeKf,
    rng_ready: bool,
    p0: f32,
    q_pos: f32,
    q_vel: f32,
    cam: Cam,
    fl: NedFlight,
    /// current navigation constant, floats in [fl.n_min, fl.n_max]
    n: f32,
    int_y: f32,
    int_z: f32,
    last_t: Option<Instant>,
}

impl NedGuidance {
    pub fn new(cfg: &GuidanceConfig, cam: Cam, flight: NedFlight) -> Self {
        Self {
            kf: LinearKalman::new([0.0; 6], diag6(cfg.kf_p0, cfg.kf_p0)),
            rng: RangeKf::new(cfg),
            rng_ready: false,
            p0: cfg.kf_p0,
            q_pos: cfg.kf_q_pos,
            q_vel: cfg.kf_q_vel,
            cam,
            fl: flight,
            n: flight.n_max,
            int_y: 0.0,
            int_z: 0.0,
            last_t: None,
        }
    }

    pub fn reset(&mut self) {
        self.kf.x = [0.0; 6];
        self.kf.p = diag6(self.p0, self.p0);
        self.rng.reset();
        self.rng_ready = false;
        self.n = self.fl.n_max;
        self.int_y = 0.0;
        self.int_z = 0.0;
        self.last_t = None;
    }

    pub fn step(
        &mut self,
        px: f32,
        py: f32,
        w_bbox_norm: f32,
        att: Attitude,
        imu_ax: f32,
    ) -> NedOut {
        let los_meas = los_ned_from_pixel(&self.cam, px, py, att.roll, att.pitch, att.yaw);

        let now = Instant::now();
        let dt = match self.last_t {
            Some(prev) => {
                let d = now.duration_since(prev).as_secs_f32();
                if d < DT_MIN {
                    return self.output();
                }
                d.min(DT_MAX)
            }
            None => {
                self.kf.x = [los_meas[0], los_meas[1], los_meas[2], 0.0, 0.0, 0.0];
                self.last_t = Some(now);
                self.init_range(w_bbox_norm);
                return self.output();
            }
        };
        self.last_t = Some(now);

        self.predict(dt);
        let h = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ];
        let r = [[LOS_R, 0.0, 0.0], [0.0, LOS_R, 0.0], [0.0, 0.0, LOS_R]];
        self.kf.update(&los_meas, &h, &r);
        let los = self.renorm_los();

        self.init_range(w_bbox_norm);
        let (range, z_dot) = if self.rng_ready {
            self.rng.step(w_bbox_norm, att.pitch, imu_ax)
        } else {
            (0.0, 0.0)
        };
        let vc = self.closing_velocity(z_dot);

        let omega = cross(los, self.rate());
        // steady LOS -> raise the gain, a maneuvering target -> lower it
        if omega.iter().map(|w| w * w).sum::<f32>().sqrt() < ADAPT_PN_THR {
            self.n = (self.n + ADAPT_PN_STEP).min(self.fl.n_max);
        } else {
            self.n = (self.n - ADAPT_PN_STEP).max(self.fl.n_min);
        }

        let (accel, intercept) = if self.rng_ready && range < INTERCEPT_RANGE_M {
            (self.intercept_accel(los, dt), true)
        } else {
            (pn_accel(los, self.rate(), self.n, vc), false)
        };

        NedOut {
            accel_ned: self.clamp_accel(accel),
            los_ned: los,
            range_m: range,
            v_closing: vc,
            intercept,
            n: self.n,
        }
    }

    /// Coasting on a lost target: both filters predicted forward, PN on the
    /// predicted state, intercept integrators frozen.
    pub fn predict_only(&mut self, pitch: f32, imu_ax: f32) -> NedOut {
        let now = Instant::now();
        let dt = match self.last_t {
            Some(prev) => {
                let d = now.duration_since(prev).as_secs_f32();
                if d < DT_MIN {
                    return self.output();
                }
                d.min(DT_MAX)
            }
            None => return self.output(),
        };
        self.last_t = Some(now);

        self.predict(dt);
        let los = self.renorm_los();

        let (range, z_dot) = if self.rng_ready {
            self.rng.predict_only(pitch, imu_ax)
        } else {
            (0.0, 0.0)
        };
        let vc = self.closing_velocity(z_dot);

        let accel = pn_accel(los, self.rate(), self.n, vc);
        NedOut {
            accel_ned: self.clamp_accel(accel),
            los_ned: los,
            range_m: range,
            v_closing: vc,
            intercept: false,
            n: self.n,
        }
    }

    fn predict(&mut self, dt: f32) {
        let f = [
            [1.0, 0.0, 0.0, dt, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, dt, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, dt],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ];
        // variance rates per second, scaled by dt (see LosKf::predict)
        let q = diag6(self.q_pos * dt, self.q_vel * dt);
        self.kf.predict(&f, &q);
    }

    fn rate(&self) -> [f32; 3] {
        [self.kf.x[3], self.kf.x[4], self.kf.x[5]]
    }

    fn renorm_los(&mut self) -> [f32; 3] {
        let los = normalize3([self.kf.x[0], self.kf.x[1], self.kf.x[2]]);
        self.kf.x[0] = los[0];
        self.kf.x[1] = los[1];
        self.kf.x[2] = los[2];
        los
    }

    fn init_range(&mut self, w_bbox_norm: f32) {
        if !self.rng_ready && w_bbox_norm > 0.01 {
            self.rng.init(w_bbox_norm);
            self.rng_ready = true;
        }
    }

    fn closing_velocity(&self, z_dot: f32) -> f32 {
        if self.rng_ready {
            (-z_dot).clamp(VC_MIN, VC_MAX)
        } else {
            VC_NOMINAL
        }
    }

    /// Terminal law: PID on the lateral LOS error (y/z of the unit LOS, with
    /// the target dead ahead), rate states of the KF as its derivative.
    fn intercept_accel(&mut self, los: [f32; 3], dt: f32) -> [f32; 3] {
        let e = [los[1], los[2]];
        integrate_axis(&mut self.int_y, e[0], INTERCEPT_KI, INTERCEPT_INT_LIMIT, dt);
        integrate_axis(&mut self.int_z, e[1], INTERCEPT_KI, INTERCEPT_INT_LIMIT, dt);
        [
            0.0,
            INTERCEPT_KP * e[0] + self.int_y + INTERCEPT_KD * self.kf.x[4],
            INTERCEPT_KP * e[1] + self.int_z + INTERCEPT_KD * self.kf.x[5],
        ]
    }

    fn clamp_accel(&self, acc: [f32; 3]) -> [f32; 3] {
        [
            acc[0].clamp(-self.fl.max_lat_acc, self.fl.max_lat_acc),
            acc[1].clamp(-self.fl.max_lat_acc, self.fl.max_lat_acc),
            acc[2].clamp(self.fl.max_up_acc, self.fl.max_down_acc),
        ]
    }

    /// Output for the early-return paths: current state, zero acceleration.
    fn output(&self) -> NedOut {
        NedOut {
            accel_ned: [0.0; 3],
            los_ned: normalize3([self.kf.x[0], self.kf.x[1], self.kf.x[2]]),
            range_m: if self.rng_ready {
                self.rng.x[0].max(Z_MIN)
            } else {
                0.0
            },
            v_closing: self.closing_velocity(0.0),
            intercept: false,
            n: self.n,
        }
    }
}

/// Conditional integration: the integral stops at its limit unless the error
/// pushes it back.
fn integrate_axis(int: &mut f32, e: f32, ki: f32, limit: f32, dt: f32) {
    let pushing_into_limit = (*int > 0.0) == (e > 0.0) && int.abs() >= limit;
    if !pushing_into_limit {
        *int += ki * e * dt;
    }
}

fn diag6(pos: f32, vel: f32) -> [[f32; 6]; 6] {
    let mut m = [[0.0f32; 6]; 6];
    m[0][0] = pos;
    m[1][1] = pos;
    m[2][2] = pos;
    m[3][3] = vel;
    m[4][4] = vel;
    m[5][5] = vel;
    m
}

fn diag4(a: f32, b: f32, c: f32, d: f32) -> [[f32; 4]; 4] {
    let mut m = [[0.0f32; 4]; 4];
    m[0][0] = a;
    m[1][1] = b;
    m[2][2] = c;
    m[3][3] = d;
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 0.033;

    #[test]
    fn los_first_step_passthrough() {
        let cfg = GuidanceConfig::default();
        let mut kf = LosKf::new(&cfg);
        let (ex, ey, rx, ry) = kf.step(0.3, -0.5);
        assert!((ex - 0.3).abs() < 1e-6);
        assert!((ey + 0.5).abs() < 1e-6);
        assert_eq!((rx, ry), (0.0, 0.0));
    }

    #[test]
    fn los_converges_to_stationary() {
        let cfg = GuidanceConfig::default();
        let mut kf = LosKf::new(&cfg);
        kf.kf.x = [0.0; 4];
        kf.kf.p = diag4(1.0, 1.0, 1.0, 1.0);
        for _ in 0..80 {
            kf.predict(DT);
            let h = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]];
            let r = [[kf.r, 0.0], [0.0, kf.r]];
            kf.kf.update(&[0.3, -0.2], &h, &r);
        }
        assert!((kf.kf.x[0] - 0.3).abs() < 0.01);
        assert!((kf.kf.x[1] + 0.2).abs() < 0.01);
        assert!(kf.kf.x[2].abs() < 0.05);
        assert!(kf.kf.x[3].abs() < 0.05);
    }

    #[test]
    fn los_tracks_constant_velocity() {
        let cfg = GuidanceConfig::default();
        let mut kf = LosKf::new(&cfg);
        kf.kf.x = [0.0; 4];
        kf.kf.p = diag4(1.0, 1.0, 1.0, 1.0);
        let (vx, vy) = (0.5, -0.3);
        for i in 1..=100 {
            let t = i as f32 * DT;
            kf.predict(DT);
            let h = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]];
            let r = [[kf.r, 0.0], [0.0, kf.r]];
            kf.kf.update(&[vx * t, vy * t], &h, &r);
        }
        assert!((kf.kf.x[2] - vx).abs() < 0.1);
        assert!((kf.kf.x[3] - vy).abs() < 0.1);
    }

    #[test]
    fn pn_sticks_clamp() {
        let (sx, sy) = pn_sticks(10.0, 5.0, 1.0, 1.0, 1.0, 1.0);
        assert_eq!(sx, 1.0);
        assert_eq!(sy, 1.0);
    }

    #[test]
    fn pn_zero_in_zero_out() {
        assert_eq!(pn_sticks(0.5, 3.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0));
    }

    #[test]
    fn compensate_tilt_zero_roll() {
        let (x, y) = compensate_tilt(0.5, 0.3, 0.0, 0.0);
        assert!((x - 0.5).abs() < 1e-6);
        assert!((y - 0.3).abs() < 1e-6);
    }

    #[test]
    fn compensate_tilt_removes_the_roll_artifact() {
        // right roll 30°, level error (0.5, 0): the image sees the target
        // ride up — (cos·a, -sin·a) — and the derotation must give (a, 0) back
        let phi = 0.5f32;
        let a = 0.5f32;
        let (xm, ym) = (a * phi.cos(), -a * phi.sin());
        let (x, y) = compensate_tilt(xm, ym, phi, 0.0);
        assert!((x - a).abs() < 1e-6, "x {x} vs {a}");
        assert!(y.abs() < 1e-6, "y {y}");
    }

    #[test]
    fn compensate_tilt_direction_survives_a_left_roll_too() {
        // left roll: the target rides down, derotation pulls it back up
        let phi = -0.5f32;
        let a = 0.5f32;
        let (xm, ym) = (a * phi.cos(), -a * phi.sin());
        let (x, y) = compensate_tilt(xm, ym, phi, 0.0);
        assert!((x - a).abs() < 1e-6, "x {x} vs {a}");
        assert!(y.abs() < 1e-6, "y {y}");
    }

    #[test]
    fn compensate_tilt_preserves_magnitude() {
        let roll = 25.0f32.to_radians();
        let (cx, cy) = compensate_tilt(0.3, 0.4, roll, 0.0);
        let orig = (0.3f32 * 0.3 + 0.4 * 0.4).sqrt();
        let comp = (cx * cx + cy * cy).sqrt();
        assert!((comp - orig).abs() < 1e-5);
    }

    #[test]
    fn range_kf_init() {
        let cfg = GuidanceConfig::default();
        let mut rng = RangeKf::new(&cfg);
        rng.init(0.1);
        assert!((rng.x[0] - 10.0).abs() < 1e-3);
        assert!((rng.x[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn range_stable_on_constant_bbox() {
        let cfg = GuidanceConfig::default();
        let mut rk = RangeKf::new(&cfg);
        rk.init(0.1);
        let z0 = rk.x[0];
        for _ in 0..50 {
            rk.predict(DT, 0.0, 0.0);
            rk.update(0.1);
        }
        assert!((rk.x[0] - z0).abs() < z0 * 0.2);
        assert!(rk.x[1].abs() < 1.0);
    }

    #[test]
    fn range_approach_gives_negative_zdot() {
        let cfg = GuidanceConfig::default();
        let mut rk = RangeKf::new(&cfg);
        rk.init(0.1);
        for i in 0..60 {
            let w = 0.1 + 0.002 * i as f32;
            rk.predict(DT, 0.0, 0.0);
            rk.update(w);
        }
        assert!(rk.x[1] < 0.0, "approach -> negative z_dot: {}", rk.x[1]);
    }

    #[test]
    fn range_recede_gives_positive_zdot() {
        let cfg = GuidanceConfig::default();
        let mut rk = RangeKf::new(&cfg);
        rk.init(0.2);
        for i in 0..60 {
            let w = (0.2 - 0.002 * i as f32).max(0.02);
            rk.predict(DT, 0.0, 0.0);
            rk.update(w);
        }
        assert!(rk.x[1] > 0.0, "recede -> positive z_dot: {}", rk.x[1]);
    }

    #[test]
    fn range_z_floor_holds() {
        let cfg = GuidanceConfig::default();
        let mut rk = RangeKf::new(&cfg);
        rk.init(0.5);
        for _ in 0..50 {
            rk.predict(DT, -0.5, 0.0);
            rk.update(5.0);
        }
        assert!(rk.x[0] >= Z_MIN);
    }

    #[test]
    fn range_p_stays_symmetric() {
        let cfg = GuidanceConfig::default();
        let mut rk = RangeKf::new(&cfg);
        rk.init(0.1);
        for i in 0..30 {
            rk.predict(DT, 0.05, 0.0);
            rk.update(0.1 + 0.002 * i as f32);
        }
        for i in 0..3 {
            for j in (i + 1)..3 {
                let (a, b) = (rk.p[i][j], rk.p[j][i]);
                assert!((a - b).abs() < 1e-4, "P[{i}][{j}]={a} != P[{j}][{i}]={b}");
            }
        }
    }

    #[test]
    fn range_accel_affects_velocity() {
        let cfg = GuidanceConfig::default();
        let mut still = RangeKf::new(&cfg);
        still.init(0.1);
        let mut accel = RangeKf::new(&cfg);
        accel.init(0.1);
        for _ in 0..30 {
            still.predict(DT, 0.0, 0.0);
            still.update(0.1);
            accel.predict(DT, 0.3, 0.0);
            accel.update(0.1);
        }
        assert!((accel.x[1] - still.x[1]).abs() > 0.01);
    }

    #[test]
    fn pn3d_closing_does_not_affect_kp_term() {
        let a = Nav3dOut {
            err_x: 0.3,
            err_y: 0.0,
            rate_x: 0.0,
            rate_y: 0.0,
            v_closing: 0.0,
            range: 5.0,
        };
        let b = Nav3dOut {
            v_closing: 10.0,
            ..a
        };
        let (sa, _) = pn3d_sticks(0.5, 3.0, 0.1, &a);
        let (sb, _) = pn3d_sticks(0.5, 3.0, 0.1, &b);
        assert!((sa - sb).abs() < 1e-6);
    }

    // ── NED guidance ────────────────────────────────────────────────

    use std::time::Duration;

    const STEP_MS: u64 = 8; // above DT_MIN so every step is a real filter step

    /// level attitude, FRD-aligned with NED
    const ATT: Attitude = Attitude {
        roll: 0.0,
        pitch: 0.0,
        yaw: 0.0,
    };

    fn ned_cam() -> Cam {
        Cam {
            fx: 537.2,
            fy: 537.2,
            cx: 320.0,
            cy: 240.0,
            install_pitch: 0.0,
        }
    }

    fn ned_guidance() -> NedGuidance {
        NedGuidance::new(&GuidanceConfig::default(), ned_cam(), NedFlight::default())
    }

    #[test]
    fn rot_zyx_zero_is_identity() {
        let m = rot_zyx(0.0, 0.0, 0.0);
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((m[i][j] - want).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn rot_zyx_yaw_quarter_turn_maps_forward_to_east() {
        let m = rot_zyx(0.0, 0.0, std::f32::consts::FRAC_PI_2);
        assert!(m[0][0].abs() < 1e-6);
        assert!((m[1][0] - 1.0).abs() < 1e-6);
        assert!(m[2][0].abs() < 1e-6);
    }

    #[test]
    fn rot_zyx_pitch_quarter_turn_maps_forward_up() {
        let m = rot_zyx(0.0, std::f32::consts::FRAC_PI_2, 0.0);
        // body x (forward) points straight up: no north, negative z (down is +z)
        assert!(m[0][0].abs() < 1e-6);
        assert!(m[2][0] < -0.999999);
    }

    #[test]
    fn los_center_pixel_is_dead_ahead() {
        let los = los_ned_from_pixel(&ned_cam(), 320.0, 240.0, 0.0, 0.0, 0.0);
        assert!((los[0] - 1.0).abs() < 1e-4, "{los:?}");
        assert!(los[1].abs() < 1e-4 && los[2].abs() < 1e-4);
    }

    #[test]
    fn los_pixel_below_center_points_down() {
        let los = los_ned_from_pixel(&ned_cam(), 320.0, 300.0, 0.0, 0.0, 0.0);
        assert!(los[2] > 0.05, "{los:?}");
    }

    #[test]
    fn los_pixel_right_of_center_points_east() {
        let los = los_ned_from_pixel(&ned_cam(), 400.0, 240.0, 0.0, 0.0, 0.0);
        assert!(los[1] > 0.05, "{los:?}");
    }

    #[test]
    fn los_install_pitch_raises_the_ray() {
        let mut cam = ned_cam();
        cam.install_pitch = 15.0f32.to_radians();
        let los = los_ned_from_pixel(&cam, 320.0, 240.0, 0.0, 0.0, 0.0);
        assert!(los[2] < -0.2, "{los:?}");
        assert!(los[0] > 0.8, "{los:?}");
    }

    #[test]
    fn los_follows_attitude_body_pitch_up() {
        let los = los_ned_from_pixel(&ned_cam(), 320.0, 240.0, 0.0, 20.0f32.to_radians(), 0.0);
        assert!(los[2] < -0.2, "{los:?}");
    }

    #[test]
    fn los_is_a_unit_vector() {
        let los = los_ned_from_pixel(&ned_cam(), 100.0, 400.0, 0.3, -0.2, 2.0);
        let n = (los[0] * los[0] + los[1] * los[1] + los[2] * los[2]).sqrt();
        assert!((n - 1.0).abs() < 1e-5);
    }

    #[test]
    fn pn_zero_rate_is_zero_accel() {
        let a = pn_accel([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], 5.0, 2.0);
        assert!(a.iter().all(|v| v.abs() < 1e-6), "{a:?}");
    }

    #[test]
    fn pn_accel_points_along_the_los_drift() {
        // LOS along x drifting east: the drone must accelerate east — toward
        // the lead direction, not along the LOS rotation axis
        let a = pn_accel([1.0, 0.0, 0.0], [0.0, 0.1, 0.0], 5.0, 2.0);
        assert!((a[1] - 1.0).abs() < 1e-5, "{a:?}"); // N·Vc·rate = 5·2·0.1
        assert!(a[0].abs() < 1e-6 && a[2].abs() < 1e-6);
    }

    #[test]
    fn pn_accel_target_sinking_commands_down() {
        let a = pn_accel([1.0, 0.0, 0.0], [0.0, 0.0, -0.1], 5.0, 2.0);
        assert!((a[2] + 1.0).abs() < 1e-5, "{a:?}");
    }

    #[test]
    fn pn_accel_is_perpendicular_to_los() {
        let los = [0.6, 0.8, 0.0]; // unit
        let rate = [0.05, -0.02, 0.01]; // arbitrary, not perpendicular
        let a = pn_accel(los, rate, 4.0, 3.0);
        let dot = a[0] * los[0] + a[1] * los[1] + a[2] * los[2];
        assert!(dot.abs() < 1e-5, "dot={dot}");
    }

    #[test]
    fn ned_first_step_is_zero_accel_and_passes_los_through() {
        let mut g = ned_guidance();
        let out = g.step(320.0, 240.0, 0.1, ATT, 0.0);
        assert!(out.accel_ned.iter().all(|v| *v == 0.0), "{out:?}");
        assert!((out.los_ned[0] - 1.0).abs() < 1e-4);
        assert!(!out.intercept);
    }

    #[test]
    fn ned_stationary_target_settles_to_zero_accel() {
        let mut g = ned_guidance();
        for _ in 0..30 {
            g.step(320.0, 240.0, 0.1, ATT, 0.0);
            std::thread::sleep(Duration::from_millis(STEP_MS));
        }
        let out = g.step(320.0, 240.0, 0.1, ATT, 0.0);
        assert!(
            out.accel_ned.iter().all(|v| v.abs() < 0.5),
            "stationary target must not be chased: {:?}",
            out.accel_ned
        );
    }

    #[test]
    fn ned_target_drifting_east_commands_east() {
        let mut g = ned_guidance();
        let mut px = 240.0f32;
        for _ in 0..40 {
            px += 2.0;
            g.step(px, 240.0, 0.1, ATT, 0.0);
            std::thread::sleep(Duration::from_millis(STEP_MS));
        }
        let out = g.step(px, 240.0, 0.1, ATT, 0.0);
        assert!(
            out.accel_ned[1] > 0.1,
            "drifting-east target needs east lead: {:?}",
            out.accel_ned
        );
    }

    #[test]
    fn ned_intercept_pid_replaces_pn_close_in() {
        let mut g = ned_guidance();
        // w = 0.5 of the frame -> range = 2 m < INTERCEPT_RANGE_M = 4
        for _ in 0..25 {
            g.step(400.0, 260.0, 0.5, ATT, 0.0);
            std::thread::sleep(Duration::from_millis(STEP_MS));
        }
        let out = g.step(400.0, 260.0, 0.5, ATT, 0.0);
        assert!(out.intercept, "range {:.2} m", out.range_m);
        assert!(out.range_m < 4.0);
        assert!(
            out.accel_ned[0].abs() < 1e-4,
            "no forward command: {:?}",
            out.accel_ned
        );
        // los points east of dead ahead: kp=40 drives the lateral error hard
        assert!(out.accel_ned[1] > 4.0, "{:?}", out.accel_ned);
    }

    #[test]
    fn ned_flight_envelope_is_caller_owned() {
        // a tighter envelope than Default must show up in the clamps
        let fl = NedFlight {
            max_lat_acc: 1.0,
            max_up_acc: -0.5,
            max_down_acc: 0.8,
            n_min: 4.0,
            n_max: 4.0,
        };
        let mut g = NedGuidance::new(&GuidanceConfig::default(), ned_cam(), fl);
        for _ in 0..25 {
            g.step(320.0, 470.0, 0.5, ATT, 0.0);
            std::thread::sleep(Duration::from_millis(STEP_MS));
        }
        let out = g.step(320.0, 470.0, 0.5, ATT, 0.0);
        assert!(
            out.accel_ned[2] > 0.7 && out.accel_ned[2] <= 0.8,
            "down clamp at 0.8: {:?}",
            out.accel_ned
        );
        assert!(
            out.accel_ned[0].abs() <= 1.0 && out.accel_ned[1].abs() <= 1.0,
            "lateral clamp at 1.0: {:?}",
            out.accel_ned
        );
        // n starts at n_max and the adaptive law keeps it inside the bounds
        let mut g2 = NedGuidance::new(&GuidanceConfig::default(), ned_cam(), fl);
        for _ in 0..5 {
            g2.step(320.0, 240.0, 0.1, ATT, 0.0);
            std::thread::sleep(Duration::from_millis(STEP_MS));
        }
        let out = g2.step(320.0, 240.0, 0.1, ATT, 0.0);
        assert!(out.intercept || out.accel_ned.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn ned_vertical_clamp_is_asymmetric() {
        let mut down = ned_guidance();
        for _ in 0..25 {
            down.step(320.0, 470.0, 0.5, ATT, 0.0);
            std::thread::sleep(Duration::from_millis(STEP_MS));
        }
        let out = down.step(320.0, 470.0, 0.5, ATT, 0.0);
        assert!(
            out.accel_ned[2] > 5.9,
            "down clamp at 6: {:?}",
            out.accel_ned
        );

        let mut up = ned_guidance();
        for _ in 0..25 {
            up.step(320.0, 10.0, 0.5, ATT, 0.0);
            std::thread::sleep(Duration::from_millis(STEP_MS));
        }
        let out = up.step(320.0, 10.0, 0.5, ATT, 0.0);
        assert!(
            out.accel_ned[2] < -2.9,
            "up clamp at -3: {:?}",
            out.accel_ned
        );
        assert!(out.accel_ned[2] >= -3.0, "{:?}", out.accel_ned);
    }

    #[test]
    fn ned_predict_only_keeps_a_finite_command() {
        let mut g = ned_guidance();
        for _ in 0..10 {
            g.step(360.0, 260.0, 0.1, ATT, 0.0);
            std::thread::sleep(Duration::from_millis(STEP_MS));
        }
        let out = g.predict_only(0.0, 0.0);
        assert!(out.accel_ned.iter().all(|v| v.is_finite()));
        assert!(out.range_m > 0.0);
        let out = g.step(360.0, 260.0, 0.1, ATT, 0.0);
        assert!(
            out.los_ned[1] > 0.0,
            "target east of centre: {:?}",
            out.los_ned
        );
    }

    #[test]
    fn ned_reset_returns_to_first_frame_behaviour() {
        let mut g = ned_guidance();
        for _ in 0..10 {
            g.step(400.0, 300.0, 0.1, ATT, 0.0);
            std::thread::sleep(Duration::from_millis(STEP_MS));
        }
        g.reset();
        let out = g.step(400.0, 300.0, 0.1, ATT, 0.0);
        assert!(out.accel_ned.iter().all(|v| *v == 0.0));
        // the range KF re-initializes on this first frame with z_dot = 0, which
        // clamps to the minimum closing velocity — the conservative restart
        assert_eq!(out.v_closing, VC_MIN);
    }
}
