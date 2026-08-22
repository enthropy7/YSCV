//! HOG gradients, orientation binning and Colour-Name area pooling.
//!
//! The three kernels a correlation-filter tracker spends its feature time in.
//! None is specific to one tracker: the gradient is a 1x3 central difference
//! with reflect-101 borders, the orientation pass is soft binning over an atan2
//! approximation, and the Colour-Name pass is a 4x4 area pool feeding a
//! caller-supplied lookup table.

use std::f32::consts::PI;
use yscv_tensor::Tensor;

/// Colour-Name pooling cell: 4 lets the NEON path read a block row with one
/// `vld3q`.
pub const CELL: usize = 4;
/// Orientation bins.
pub const NBINS: usize = 9;

/// Horizontal/vertical gradient by 1x3 central difference with reflect-101
/// borders -- cv2.Sobel(ksize=1), the operator the Python reference uses. (yscv's
/// 3x3 Sobel is available too, but it smooths across the gradient and so shifts
/// orientations; matching the reference operator keeps the port faithful.)
pub fn central_diff(g: &[f32], h: usize, w: usize) -> (Vec<f32>, Vec<f32>) {
    #[cfg(any(target_arch = "aarch64", all(target_arch = "arm", feature = "neon-v7")))]
    if h >= 2 && w >= 2 {
        // SAFETY: NEON is baseline on aarch64; bit-exact vs scalar (same subs).
        return unsafe { central_diff_neon(g, h, w) };
    }
    central_diff_scalar(g, h, w)
}

fn central_diff_scalar(g: &[f32], h: usize, w: usize) -> (Vec<f32>, Vec<f32>) {
    let mut gx = vec![0f32; h * w];
    let mut gy = vec![0f32; h * w];
    for y in 0..h {
        let ym1 = if y == 0 { 1 } else { y - 1 };
        let yp1 = if y + 1 == h { h - 2 } else { y + 1 };
        for x in 0..w {
            let xm1 = if x == 0 { 1 } else { x - 1 };
            let xp1 = if x + 1 == w { w - 2 } else { x + 1 };
            gx[y * w + x] = g[y * w + xp1] - g[y * w + xm1];
            gy[y * w + x] = g[yp1 * w + x] - g[ym1 * w + x];
        }
    }
    (gx, gy)
}

/// NEON central difference: `gy` is a whole-row vector subtract of the y±1 rows;
/// `gx` vectorises the interior `x` (border columns are 0 under reflect-101).
/// Bit-exact vs scalar (same `a - b` per element).
#[cfg(any(target_arch = "aarch64", all(target_arch = "arm", feature = "neon-v7")))]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn central_diff_neon(g: &[f32], h: usize, w: usize) -> (Vec<f32>, Vec<f32>) {
    #[cfg(target_arch = "aarch64")]
    use std::arch::aarch64::*;
    #[cfg(target_arch = "arm")]
    use std::arch::arm::*;
    let mut gx = vec![0f32; h * w];
    let mut gy = vec![0f32; h * w];
    for y in 0..h {
        let ym1 = if y == 0 { 1 } else { y - 1 };
        let yp1 = if y + 1 == h { h - 2 } else { y + 1 };
        let (row, rm, rp) = (y * w, ym1 * w, yp1 * w);
        // gy: whole row = g[rp+x] - g[rm+x]
        let mut x = 0;
        while x + 4 <= w {
            let a = vld1q_f32(g.as_ptr().add(rp + x));
            let b = vld1q_f32(g.as_ptr().add(rm + x));
            vst1q_f32(gy.as_mut_ptr().add(row + x), vsubq_f32(a, b));
            x += 4;
        }
        while x < w {
            *gy.get_unchecked_mut(row + x) = g[rp + x] - g[rm + x];
            x += 1;
        }
        // gx interior x in [1, w-2]: g[row+x+1] - g[row+x-1]; borders stay 0
        let mut x = 1;
        while x + 4 <= w - 1 {
            let a = vld1q_f32(g.as_ptr().add(row + x + 1));
            let b = vld1q_f32(g.as_ptr().add(row + x - 1));
            vst1q_f32(gx.as_mut_ptr().add(row + x), vsubq_f32(a, b));
            x += 4;
        }
        while x < w - 1 {
            *gx.get_unchecked_mut(row + x) = g[row + x + 1] - g[row + x - 1];
            x += 1;
        }
    }
    (gx, gy)
}

/// Fast atan2 approximation (max err ~1.5e-3 rad) returning a value in (-PI, PI].
/// The tracker is validated at DTB70-AO level, which is robust to this vs libm
/// atan2, and *every* arch path uses this same polynomial — so the x86 eval
/// exercises the exact numerics the NEON path will produce.
// The coefficients are OpenCV's, kept at their published digits so the
// polynomial can be checked against the reference by eye.
#[allow(clippy::excessive_precision)]
#[inline(always)]
fn fast_atan2(y: f32, x: f32) -> f32 {
    let ax = x.abs();
    let ay = y.abs();
    let mx = ax.max(ay);
    let mn = ax.min(ay);
    let a = if mx > 0.0 { mn / mx } else { 0.0 };
    let s = a * a;
    let mut r = (((-0.046_496_474_9 * s + 0.159_314_22) * s - 0.327_622_764) * s) * a + a;
    if ay > ax {
        r = std::f32::consts::FRAC_PI_2 - r;
    }
    if x < 0.0 {
        r = PI - r;
    }
    if y < 0.0 {
        r = -r;
    }
    r
}

/// Per-pixel orientation binning: magnitude, unsigned-orientation soft bin `b0`
/// and the two magnitude weights `(c0 -> b0, c1 -> (b0+1)%NBINS)`.
#[inline(always)]
fn orient_px(dx: f32, dy: f32) -> (u8, f32, f32) {
    let mag = (dx * dx + dy * dy).sqrt();
    let mut ang = fast_atan2(dy, dx);
    if ang < 0.0 {
        ang += PI; // unsigned orientation [0, PI)
    }
    // scale to bins; clamp just below NBINS so floor stays in 0..NBINS-1
    let b = (ang * (NBINS as f32 / PI)).min(NBINS as f32 - 1e-4);
    let bf = b.floor();
    let w1 = b - bf;
    (bf as u8, mag * (1.0 - w1), mag * w1)
}

/// Orientation pass over the `mh x mw` region of `gx/gy` (indexed by full width
/// `w`). Dispatches to NEON on aarch64 (baseline there), scalar elsewhere.
#[inline]
pub fn orient_pass(
    gx: &[f32],
    gy: &[f32],
    w: usize,
    mh: usize,
    mw: usize,
) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    #[cfg(any(target_arch = "aarch64", all(target_arch = "arm", feature = "neon-v7")))]
    {
        // SAFETY: NEON is a baseline feature of every aarch64 target.
        return unsafe { orient_pass_neon(gx, gy, w, mh, mw) };
    }
    #[allow(unreachable_code)]
    orient_pass_scalar(gx, gy, w, mh, mw)
}

fn orient_pass_scalar(
    gx: &[f32],
    gy: &[f32],
    w: usize,
    mh: usize,
    mw: usize,
) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let mut b0 = vec![0u8; mh * mw];
    let mut c0 = vec![0f32; mh * mw];
    let mut c1 = vec![0f32; mh * mw];
    for y in 0..mh {
        let (row, orow) = (y * w, y * mw);
        for x in 0..mw {
            let (bi, a0, a1) = orient_px(gx[row + x], gy[row + x]);
            let k = orow + x;
            b0[k] = bi;
            c0[k] = a0;
            c1[k] = a1;
        }
    }
    (b0, c0, c1)
}

/// NEON orientation pass: sqrt magnitude and the fast-atan2 polynomial evaluated
/// four pixels at a time (the folds are `vbslq` selects); the `< 4` tail per row
/// falls back to the shared scalar `orient_px` so the numerics are identical.
#[cfg(any(target_arch = "aarch64", all(target_arch = "arm", feature = "neon-v7")))]
#[target_feature(enable = "neon")]
unsafe fn orient_pass_neon(
    gx: &[f32],
    gy: &[f32],
    w: usize,
    mh: usize,
    mw: usize,
) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    #[cfg(target_arch = "aarch64")]
    use std::arch::aarch64::*;
    #[cfg(target_arch = "arm")]
    use std::arch::arm::*;
    let mut b0 = vec![0u8; mh * mw];
    let mut c0v = vec![0f32; mh * mw];
    let mut c1v = vec![0f32; mh * mw];

    let vpi = vdupq_n_f32(PI);
    let vhalfpi = vdupq_n_f32(std::f32::consts::FRAC_PI_2);
    let vscale = vdupq_n_f32(NBINS as f32 / PI);
    let vclamp = vdupq_n_f32(NBINS as f32 - 1e-4);
    let (ca, cb, cc) = (
        vdupq_n_f32(-0.046_496_474_9),
        vdupq_n_f32(0.159_314_22),
        vdupq_n_f32(-0.327_622_764),
    );
    let vzero = vdupq_n_f32(0.0);
    let vone = vdupq_n_f32(1.0);

    for y in 0..mh {
        let (row, orow) = (y * w, y * mw);
        let mut x = 0;
        while x + 4 <= mw {
            let gxv = vld1q_f32(gx.as_ptr().add(row + x));
            let gyv = vld1q_f32(gy.as_ptr().add(row + x));
            let mag =
                super::neon_compat::sqrtq_f32(vaddq_f32(vmulq_f32(gxv, gxv), vmulq_f32(gyv, gyv)));

            let ax = vabsq_f32(gxv);
            let ay = vabsq_f32(gyv);
            let mx = vmaxq_f32(ax, ay);
            let mn = vminq_f32(ax, ay);
            let a = vbslq_f32(
                vcgtq_f32(mx, vzero),
                super::neon_compat::divq_f32(mn, mx),
                vzero,
            );
            let s = vmulq_f32(a, a);
            // (((ca*s + cb)*s + cc)*s)*a + a
            let mut r = vaddq_f32(vmulq_f32(ca, s), cb);
            r = vaddq_f32(vmulq_f32(r, s), cc);
            r = vmulq_f32(vmulq_f32(r, s), a);
            r = vaddq_f32(r, a);
            r = vbslq_f32(vcgtq_f32(ay, ax), vsubq_f32(vhalfpi, r), r); // ay>ax
            r = vbslq_f32(vcltq_f32(gxv, vzero), vsubq_f32(vpi, r), r); // x<0
            r = vbslq_f32(vcltq_f32(gyv, vzero), vsubq_f32(vzero, r), r); // y<0
            r = vbslq_f32(vcltq_f32(r, vzero), vaddq_f32(r, vpi), r); // fold [0,PI)

            let b = vminq_f32(vmulq_f32(r, vscale), vclamp);
            let bi = vcvtq_s32_f32(b); // trunc == floor for b >= 0
            let w1 = vsubq_f32(b, vcvtq_f32_s32(bi));
            let c0 = vmulq_f32(mag, vsubq_f32(vone, w1));
            let c1 = vmulq_f32(mag, w1);

            vst1q_f32(c0v.as_mut_ptr().add(orow + x), c0);
            vst1q_f32(c1v.as_mut_ptr().add(orow + x), c1);
            let mut bins = [0i32; 4];
            vst1q_s32(bins.as_mut_ptr(), bi);
            b0[orow + x] = bins[0] as u8;
            b0[orow + x + 1] = bins[1] as u8;
            b0[orow + x + 2] = bins[2] as u8;
            b0[orow + x + 3] = bins[3] as u8;
            x += 4;
        }
        while x < mw {
            let (bi, a0, a1) = orient_px(gx[row + x], gy[row + x]);
            b0[orow + x] = bi;
            c0v[orow + x] = a0;
            c1v[orow + x] = a1;
            x += 1;
        }
    }
    (b0, c0v, c1v)
}

/// Colour-Name channels: `fh x fw x 11`. Each 4x4 RGB block is area-averaged
/// (matching cv2 INTER_AREA on an exact integer downscale), then a single lookup
/// into the learned table per cell -- the area-pooled CN of the Python reference.
/// `color` is `H x W x 3` f32 with 0..255 values.
pub fn cn(color: &Tensor, table: &[f32]) -> Tensor {
    // CELL == 4 lets the NEON path use a single vld3q per block row.
    #[cfg(any(target_arch = "aarch64", all(target_arch = "arm", feature = "neon-v7")))]
    if CELL == 4 {
        // SAFETY: NEON is baseline on aarch64. The 4x4 block sum uses a tree
        // order vs the scalar row-major one, but the result feeds a >>3 quantiser
        // (bins of width 8 in round(s/16) space), so a ~1e-5 sum difference never
        // crosses a bin boundary -> identical index/output (see cn_selfcheck).
        return unsafe { cn_neon(color, table) };
    }
    cn_scalar(color, table)
}

fn cn_scalar(color: &Tensor, table: &[f32]) -> Tensor {
    let sh = color.shape();
    let (h, w) = (sh[0], sh[1]);
    let (fh, fw) = (h / CELL, w / CELL);
    let data = color.data();
    let tbl = table;
    let n = (CELL * CELL) as f32;
    let mut out = vec![0f32; fh * fw * 11];
    for cy in 0..fh {
        for cx in 0..fw {
            let (mut sr, mut sg, mut sb) = (0f32, 0f32, 0f32);
            for dy in 0..CELL {
                let row = (cy * CELL + dy) * w + cx * CELL;
                for dx in 0..CELL {
                    let p = (row + dx) * 3;
                    sr += data[p];
                    sg += data[p + 1];
                    sb += data[p + 2];
                }
            }
            let q = |s: f32| ((s / n).round().clamp(0.0, 255.0) as i32) >> 3;
            let idx = (q(sr) + 32 * q(sg) + 1024 * q(sb)) as usize;
            let base = (cy * fw + cx) * 11;
            out[base..base + 11].copy_from_slice(&tbl[idx * 11..idx * 11 + 11]);
        }
    }
    Tensor::from_vec(vec![fh, fw, 11], out).expect("cn tensor")
}

/// NEON Colour-Name: the 4x4 RGB block sum uses `vld3q_f32` (deinterleaves 4 RGB
/// pixels per row) accumulated over the 4 rows, then a horizontal add; the
/// quantise + table lookup stay scalar (a gather).
#[cfg(any(target_arch = "aarch64", all(target_arch = "arm", feature = "neon-v7")))]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn cn_neon(color: &Tensor, table: &[f32]) -> Tensor {
    #[cfg(target_arch = "aarch64")]
    use std::arch::aarch64::*;
    #[cfg(target_arch = "arm")]
    use std::arch::arm::*;
    let sh = color.shape();
    let (h, w) = (sh[0], sh[1]);
    let (fh, fw) = (h / CELL, w / CELL);
    let data = color.data();
    let tbl = table;
    let n = (CELL * CELL) as f32;
    let mut out = vec![0f32; fh * fw * 11];
    for cy in 0..fh {
        for cx in 0..fw {
            let (mut sr, mut sg, mut sb) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0), vdupq_n_f32(0.0));
            for dy in 0..CELL {
                let base = ((cy * CELL + dy) * w + cx * CELL) * 3;
                let v = vld3q_f32(data.as_ptr().add(base)); // 4 RGB pixels
                sr = vaddq_f32(sr, v.0);
                sg = vaddq_f32(sg, v.1);
                sb = vaddq_f32(sb, v.2);
            }
            let q = |s: f32| ((s / n).round().clamp(0.0, 255.0) as i32) >> 3;
            let idx = (q(super::neon_compat::addvq_f32(sr))
                + 32 * q(super::neon_compat::addvq_f32(sg))
                + 1024 * q(super::neon_compat::addvq_f32(sb))) as usize;
            let base = (cy * fw + cx) * 11;
            out[base..base + 11].copy_from_slice(&tbl[idx * 11..idx * 11 + 11]);
        }
    }
    Tensor::from_vec(vec![fh, fw, 11], out).expect("cn tensor")
}

/// Debug: compare the NEON orientation pass against the scalar one on synthetic
/// gradients (includes a non-multiple-of-4 width to exercise the tail). Returns
/// `(max |c0/c1| diff, bin mismatches, total pixels)`.
#[cfg(any(target_arch = "aarch64", all(target_arch = "arm", feature = "neon-v7")))]
pub fn orient_selfcheck() -> (f32, usize, usize) {
    let (h, w) = (10usize, 41usize); // 41 -> a 1-pixel tail per row
    let (mut gx, mut gy) = (vec![0f32; h * w], vec![0f32; h * w]);
    let mut st = 12345u32;
    let mut rng = || {
        st = st.wrapping_mul(1664525).wrapping_add(1013904223);
        (st >> 8) as f32 / 16_777_216.0 * 2.0 - 1.0
    };
    for i in 0..h * w {
        gx[i] = rng() * 10.0;
        gy[i] = rng() * 10.0;
    }
    let (sb, sc0, sc1) = orient_pass_scalar(&gx, &gy, w, h, w);
    let (nb, nc0, nc1) = unsafe { orient_pass_neon(&gx, &gy, w, h, w) };
    let (mut binmis, mut maxc) = (0usize, 0f32);
    for i in 0..h * w {
        if sb[i] != nb[i] {
            binmis += 1;
        }
        maxc = maxc
            .max((sc0[i] - nc0[i]).abs())
            .max((sc1[i] - nc1[i]).abs());
    }
    (maxc, binmis, h * w)
}

/// Debug: NEON vs scalar `central_diff` and `cn` on synthetic data. Returns
/// `(max grad diff, max cn diff)` — both expected 0 (bit-exact / quantiser-stable).
#[cfg(any(target_arch = "aarch64", all(target_arch = "arm", feature = "neon-v7")))]
pub fn grad_cn_selfcheck(table: &[f32]) -> (f32, f32) {
    let (h, w) = (48usize, 52usize);
    let mut st = 2246u32;
    let mut rng = || {
        st = st.wrapping_mul(1664525).wrapping_add(1013904223);
        (st >> 8) as f32 / 16_777_216.0
    };
    let gray: Vec<f32> = (0..h * w).map(|_| rng()).collect();
    let color: Vec<f32> = (0..h * w * 3).map(|_| rng() * 255.0).collect();
    let (sgx, sgy) = central_diff_scalar(&gray, h, w);
    let (ngx, ngy) = unsafe { central_diff_neon(&gray, h, w) };
    let mut gmax = 0f32;
    for i in 0..h * w {
        gmax = gmax
            .max((sgx[i] - ngx[i]).abs())
            .max((sgy[i] - ngy[i]).abs());
    }
    let ct = Tensor::from_vec(vec![h, w, 3], color).unwrap();
    let (cs, cnn) = (cn_scalar(&ct, table), unsafe { cn_neon(&ct, table) });
    let cmax = cs
        .data()
        .iter()
        .zip(cnn.data())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    (gmax, cmax)
}
