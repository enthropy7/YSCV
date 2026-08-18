//! Fused crop + bilinear resize.
//!
//! Samples a `tpl_h x tpl_w x ch` template directly from a source image for the
//! window of size `win_w x win_h` centred at `(cx, cy)`, in a single bilinear
//! pass — equivalent to OpenCV `getRectSubPix` (subpixel crop, border replicate)
//! followed by `resize` (INTER_LINEAR, half-pixel), but it only ever touches
//! template-many output pixels instead of materialising the full window first.
//! This is the hot primitive for correlation-filter trackers (crop a padded
//! search window around the target and normalise it to a fixed model size).
//!
//! Multi-arch, all bit-identical to the scalar reference (same non-fused
//! `w00*p00 + w01*p01 + w10*p10 + w11*p11` order, verified by the parity test):
//!   - NEON / SSE4.1: per output pixel, bilinear across the `ch <= 4` lanes.
//!   - AVX2 / AVX512F: across 8 / 16 output pixels, hardware gather for the four
//!     taps (`ch` in {1, 3}).

use super::super::ImgProcError;
use super::super::shape::hwc_shape;
use yscv_tensor::Tensor;

/// Fused crop + bilinear resize on a raw HWC f32 buffer (zero-copy: the caller's
/// frame is borrowed, never wrapped/copied). Returns the `tpl_h*tpl_w*ch`
/// template, row-major. `cx,cy` and the window are in source pixel coordinates.
pub fn crop_resize_bilinear_raw(
    src: &[f32],
    h: usize,
    w: usize,
    ch: usize,
    cx: f32,
    cy: f32,
    win_w: usize,
    win_h: usize,
    tpl_w: usize,
    tpl_h: usize,
) -> Vec<f32> {
    #[cfg(target_arch = "aarch64")]
    if !cfg!(miri) && (1..=4).contains(&ch) && yscv_cpu::host_cpu().features.neon {
        // SAFETY: guarded by runtime NEON detection; bit-exact vs scalar.
        return unsafe { crop_resize_neon(src, h, w, ch, cx, cy, win_w, win_h, tpl_w, tpl_h) };
    }
    #[cfg(target_arch = "x86_64")]
    if !cfg!(miri) {
        // SAFETY: each path is guarded by runtime feature detection; all are
        // bit-exact vs scalar (parity test). Gather-based (AVX2/AVX512) handle
        // ch in {1,3}; the SSE channel-lane path handles ch <= 4.
        let f = &yscv_cpu::host_cpu().features;
        if (ch == 1 || ch == 3) && f.avx512f {
            return unsafe {
                crop_resize_avx512(src, h, w, ch, cx, cy, win_w, win_h, tpl_w, tpl_h)
            };
        }
        if (ch == 1 || ch == 3) && f.avx2 {
            return unsafe { crop_resize_avx2(src, h, w, ch, cx, cy, win_w, win_h, tpl_w, tpl_h) };
        }
        if (1..=4).contains(&ch) && f.sse41 {
            return unsafe { crop_resize_sse(src, h, w, ch, cx, cy, win_w, win_h, tpl_w, tpl_h) };
        }
    }
    crop_resize_scalar(src, h, w, ch, cx, cy, win_w, win_h, tpl_w, tpl_h)
}

/// Tensor wrapper around [`crop_resize_bilinear_raw`]. `input` is HWC f32; the
/// result is `tpl_h x tpl_w x ch`.
pub fn crop_resize_bilinear(
    input: &Tensor,
    cx: f32,
    cy: f32,
    win_w: usize,
    win_h: usize,
    tpl_h: usize,
    tpl_w: usize,
) -> Result<Tensor, ImgProcError> {
    let (h, w, ch) = hwc_shape(input)?;
    if tpl_h == 0 || tpl_w == 0 || win_w == 0 || win_h == 0 {
        return Err(ImgProcError::InvalidOutputDimensions {
            out_h: tpl_h,
            out_w: tpl_w,
        });
    }
    let out = crop_resize_bilinear_raw(input.data(), h, w, ch, cx, cy, win_w, win_h, tpl_w, tpl_h);
    Tensor::from_vec(vec![tpl_h, tpl_w, ch], out).map_err(Into::into)
}

/// Fused crop + bilinear resize with a **constant border**: bilinear taps whose
/// source pixel lies outside `[0,w) x [0,h)` contribute `border[c]` instead of
/// the replicated edge pixel (the [`crop_resize_bilinear_raw`] behaviour). This
/// is OpenCV `copyMakeBorder(BORDER_CONSTANT)` + `resize`, fused — the crop for
/// context-padded trackers (e.g. FEAR) whose window runs far off the frame and
/// pads with the image mean. Same coordinate convention as
/// [`crop_resize_bilinear_raw`], so a region `[x0,y0,rw,rh]` maps to
/// `cx = x0 + (rw-1)/2`, `cy = y0 + (rh-1)/2`, `win = (rw,rh)`. `border` must
/// have at least `ch` elements.
#[allow(clippy::too_many_arguments)]
pub fn crop_resize_bilinear_border_raw(
    src: &[f32],
    h: usize,
    w: usize,
    ch: usize,
    cx: f32,
    cy: f32,
    win_w: usize,
    win_h: usize,
    tpl_w: usize,
    tpl_h: usize,
    border: &[f32],
) -> Vec<f32> {
    #[cfg(target_arch = "aarch64")]
    if !cfg!(miri) && (1..=4).contains(&ch) && yscv_cpu::host_cpu().features.neon {
        // SAFETY: guarded by runtime NEON detection; bit-exact vs scalar.
        return unsafe {
            crop_resize_border_neon(src, h, w, ch, cx, cy, win_w, win_h, tpl_w, tpl_h, border)
        };
    }
    #[cfg(target_arch = "x86_64")]
    if !cfg!(miri) {
        // SAFETY: each path is guarded by runtime feature detection; all are
        // bit-exact vs scalar (parity test). Gather paths (AVX2/AVX512) handle
        // ch in {1,3}; the SSE channel-lane path handles ch <= 4.
        let f = &yscv_cpu::host_cpu().features;
        if (ch == 1 || ch == 3) && f.avx512f {
            return unsafe {
                crop_resize_border_avx512(src, h, w, ch, cx, cy, win_w, win_h, tpl_w, tpl_h, border)
            };
        }
        if (ch == 1 || ch == 3) && f.avx2 {
            return unsafe {
                crop_resize_border_avx2(src, h, w, ch, cx, cy, win_w, win_h, tpl_w, tpl_h, border)
            };
        }
        if (1..=4).contains(&ch) && f.sse41 {
            return unsafe {
                crop_resize_border_sse(src, h, w, ch, cx, cy, win_w, win_h, tpl_w, tpl_h, border)
            };
        }
    }
    crop_resize_border_scalar(src, h, w, ch, cx, cy, win_w, win_h, tpl_w, tpl_h, border)
}

/// Tensor wrapper around [`crop_resize_bilinear_border_raw`]. `input` is HWC f32;
/// the result is `tpl_h x tpl_w x ch`. `border` must have at least `ch` elements.
#[allow(clippy::too_many_arguments)]
pub fn crop_resize_bilinear_border(
    input: &Tensor,
    cx: f32,
    cy: f32,
    win_w: usize,
    win_h: usize,
    tpl_h: usize,
    tpl_w: usize,
    border: &[f32],
) -> Result<Tensor, ImgProcError> {
    let (h, w, ch) = hwc_shape(input)?;
    if tpl_h == 0 || tpl_w == 0 || win_w == 0 || win_h == 0 {
        return Err(ImgProcError::InvalidOutputDimensions {
            out_h: tpl_h,
            out_w: tpl_w,
        });
    }
    if border.len() < ch {
        return Err(ImgProcError::InvalidOutputDimensions {
            out_h: border.len(),
            out_w: ch,
        });
    }
    let out = crop_resize_bilinear_border_raw(
        input.data(),
        h,
        w,
        ch,
        cx,
        cy,
        win_w,
        win_h,
        tpl_w,
        tpl_h,
        border,
    );
    Tensor::from_vec(vec![tpl_h, tpl_w, ch], out).map_err(Into::into)
}

// ---- coordinate mapping (shared): cv2 half-pixel + getRectSubPix window offset.
#[inline(always)]
fn map_origin(cx: f32, cy: f32, win_w: usize, win_h: usize) -> (f32, f32) {
    (
        cx - (win_w as f32 - 1.0) * 0.5,
        cy - (win_h as f32 - 1.0) * 0.5,
    )
}

// ===========================================================================
// Scalar reference
// ===========================================================================
fn crop_resize_scalar(
    src: &[f32],
    h: usize,
    w: usize,
    ch: usize,
    cx: f32,
    cy: f32,
    win_w: usize,
    win_h: usize,
    tpl_w: usize,
    tpl_h: usize,
) -> Vec<f32> {
    let (ox, oy) = map_origin(cx, cy, win_w, win_h);
    let sx = win_w as f32 / tpl_w as f32;
    let sy = win_h as f32 / tpl_h as f32;
    let (wi, hi) = (w as isize, h as isize);
    let mut out = vec![0f32; tpl_h * tpl_w * ch];
    for j in 0..tpl_h {
        let fy = oy + (j as f32 + 0.5) * sy - 0.5;
        let y0f = fy.floor();
        let ay = fy - y0f;
        let y0 = (y0f as isize).clamp(0, hi - 1) as usize;
        let y1 = (y0f as isize + 1).clamp(0, hi - 1) as usize;
        for i in 0..tpl_w {
            let fx = ox + (i as f32 + 0.5) * sx - 0.5;
            let x0f = fx.floor();
            let ax = fx - x0f;
            let x0 = (x0f as isize).clamp(0, wi - 1) as usize;
            let x1 = (x0f as isize + 1).clamp(0, wi - 1) as usize;
            let (w00, w01) = ((1.0 - ax) * (1.0 - ay), ax * (1.0 - ay));
            let (w10, w11) = ((1.0 - ax) * ay, ax * ay);
            let o00 = (y0 * w + x0) * ch;
            let o01 = (y0 * w + x1) * ch;
            let o10 = (y1 * w + x0) * ch;
            let o11 = (y1 * w + x1) * ch;
            let base = (j * tpl_w + i) * ch;
            for c in 0..ch {
                out[base + c] = w00 * src[o00 + c]
                    + w01 * src[o01 + c]
                    + w10 * src[o10 + c]
                    + w11 * src[o11 + c];
            }
        }
    }
    out
}

// ===========================================================================
// NEON: per output pixel, bilinear across the ch<=4 channel lanes of one q-reg.
// ===========================================================================
/// The four bilinear taps of one output pixel when none of them is clamped or
/// off-image: the two x-taps are then adjacent pixels, so each row pair is one
/// unaligned 128-bit load instead of `2*ch` scalar loads and lane inserts, and
/// the weights arrive as lanes of one vector so the multiplies need no
/// general-register broadcast (an A53 stalls on those).
///
/// Multiply/add order is `w00*p00 + w01*p01 + w10*p10 + w11*p11`, non-fused,
/// matching the scalar reference bit for bit.
///
/// # Safety
/// `o00`/`o10` must admit a 4-element read at `o + ch`, and `base` a 4-element
/// write.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn tap4_neon(
    src: &[f32],
    out: &mut [f32],
    o00: usize,
    o10: usize,
    ch: usize,
    base: usize,
    axv: std::arch::aarch64::float32x4_t,
    ayv: std::arch::aarch64::float32x4_t,
) {
    use std::arch::aarch64::*;
    let sp = src.as_ptr();
    let wv = vmulq_f32(axv, ayv);
    let p00 = vld1q_f32(sp.add(o00));
    let p01 = vld1q_f32(sp.add(o00 + ch));
    let p10 = vld1q_f32(sp.add(o10));
    let p11 = vld1q_f32(sp.add(o10 + ch));
    let mut acc = vmulq_laneq_f32(p00, wv, 0);
    acc = vaddq_f32(acc, vmulq_laneq_f32(p01, wv, 1));
    acc = vaddq_f32(acc, vmulq_laneq_f32(p10, wv, 2));
    acc = vaddq_f32(acc, vmulq_laneq_f32(p11, wv, 3));
    // Lanes past `ch` belong to the next output pixel and are rewritten when it
    // is stored; the tail slack in `out` covers the last one.
    vst1q_f32(out.as_mut_ptr().add(base), acc);
}

/// `[1-a, a, 1-a, a]`, the x half of the tap weights.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn frac_x_neon(a: f32) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    let d = vdupq_n_f32(a);
    vzip1q_f32(vsubq_f32(vdupq_n_f32(1.0), d), d)
}

/// `[1-a, 1-a, a, a]`, the y half of the tap weights.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn frac_y_neon(a: f32) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    let d = vdupq_n_f32(a);
    vcombine_f32(
        vget_low_f32(vsubq_f32(vdupq_n_f32(1.0), d)),
        vget_low_f32(d),
    )
}

/// The four taps of four consecutive output pixels, when the whole group is
/// clear of the image edges. Returns false if it is not, leaving the pixels to
/// the caller's edge path.
///
/// Four at a time because the per-pixel `int -> float -> floor -> int` address
/// chain is ~30 cycles of latency on an A53 and a one-pixel loop has nothing to
/// overlap it with; here the four tap chains are independent.
///
/// # Safety
/// Caller must be on a NEON target, and `out` must have four elements of slack.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]
unsafe fn group4_neon(
    src: &[f32],
    out: &mut [f32],
    i: usize,
    ox: f32,
    sx: f32,
    ch: usize,
    w: usize,
    wi: isize,
    y0: usize,
    y1: usize,
    base: usize,
    ayv: std::arch::aarch64::float32x4_t,
) -> bool {
    use std::arch::aarch64::*;
    let ramp = vld1q_s32([0i32, 1, 2, 3].as_ptr());
    let fi = vcvtq_f32_s32(vaddq_s32(vdupq_n_s32(i as i32), ramp));
    let half = vdupq_n_f32(0.5);
    // Same operation order as the scalar path, so the fractions are the same bits.
    let fx = vsubq_f32(
        vaddq_f32(vdupq_n_f32(ox), vmulq_n_f32(vaddq_f32(fi, half), sx)),
        half,
    );
    let fl = vrndmq_f32(fx);
    let mut ax = [0f32; 4];
    let mut ix = [0i32; 4];
    vst1q_f32(ax.as_mut_ptr(), vsubq_f32(fx, fl));
    vst1q_s32(ix.as_mut_ptr(), vcvtq_s32_f32(fl));
    // sx > 0, so the group is ordered: the ends bound it.
    if (ix[0] as isize) < 0 || (ix[3] as isize) + 1 >= wi {
        return false;
    }
    if (y1 * w + ix[3] as usize) * ch + ch + 4 > src.len() {
        return false;
    }
    for k in 0..4 {
        let x = ix[k] as usize;
        tap4_neon(
            src,
            out,
            (y0 * w + x) * ch,
            (y1 * w + x) * ch,
            ch,
            base + k * ch,
            frac_x_neon(ax[k]),
            ayv,
        );
    }
    true
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn crop_resize_neon(
    src: &[f32],
    h: usize,
    w: usize,
    ch: usize,
    cx: f32,
    cy: f32,
    win_w: usize,
    win_h: usize,
    tpl_w: usize,
    tpl_h: usize,
) -> Vec<f32> {
    use std::arch::aarch64::*;

    // load ch (<=4) contiguous floats at element offset `o` into lanes [0,ch)
    #[inline(always)]
    unsafe fn loadn(src: &[f32], o: usize, ch: usize) -> float32x4_t {
        let mut v = vdupq_n_f32(0.0);
        v = vsetq_lane_f32(*src.get_unchecked(o), v, 0);
        if ch > 1 {
            v = vsetq_lane_f32(*src.get_unchecked(o + 1), v, 1);
        }
        if ch > 2 {
            v = vsetq_lane_f32(*src.get_unchecked(o + 2), v, 2);
        }
        if ch > 3 {
            v = vsetq_lane_f32(*src.get_unchecked(o + 3), v, 3);
        }
        v
    }

    let (ox, oy) = map_origin(cx, cy, win_w, win_h);
    let sx = win_w as f32 / tpl_w as f32;
    let sy = win_h as f32 / tpl_h as f32;
    let (wi, hi) = (w as isize, h as isize);
    let n = tpl_h * tpl_w * ch;
    // Slack for the 4-lane store of the last pixel; see `tap4_neon`.
    let mut out = vec![0f32; n + 4];
    for j in 0..tpl_h {
        let fy = oy + (j as f32 + 0.5) * sy - 0.5;
        let y0f = fy.floor();
        let ay = fy - y0f;
        let iy0 = y0f as isize;
        let y0 = iy0.clamp(0, hi - 1) as usize;
        let y1 = (iy0 + 1).clamp(0, hi - 1) as usize;
        let rows_free = iy0 >= 0 && iy0 + 1 < hi;
        let ayv = frac_y_neon(ay);
        let mut i = 0;
        while i < tpl_w {
            let at = (j * tpl_w + i) * ch;
            if rows_free
                && i + 4 <= tpl_w
                && group4_neon(src, &mut out, i, ox, sx, ch, w, wi, y0, y1, at, ayv)
            {
                i += 4;
                continue;
            }
            let fx = ox + (i as f32 + 0.5) * sx - 0.5;
            let x0f = fx.floor();
            let ax = fx - x0f;
            let ix0 = x0f as isize;
            let x0 = ix0.clamp(0, wi - 1) as usize;
            let x1 = (ix0 + 1).clamp(0, wi - 1) as usize;
            let (w00, w01) = ((1.0 - ax) * (1.0 - ay), ax * (1.0 - ay));
            let (w10, w11) = ((1.0 - ax) * ay, ax * ay);
            let p00 = loadn(src, (y0 * w + x0) * ch, ch);
            let p01 = loadn(src, (y0 * w + x1) * ch, ch);
            let p10 = loadn(src, (y1 * w + x0) * ch, ch);
            let p11 = loadn(src, (y1 * w + x1) * ch, ch);
            let mut acc = vmulq_n_f32(p00, w00);
            acc = vaddq_f32(acc, vmulq_n_f32(p01, w01));
            acc = vaddq_f32(acc, vmulq_n_f32(p10, w10));
            acc = vaddq_f32(acc, vmulq_n_f32(p11, w11));
            let base = (j * tpl_w + i) * ch;
            *out.get_unchecked_mut(base) = vgetq_lane_f32(acc, 0);
            if ch > 1 {
                *out.get_unchecked_mut(base + 1) = vgetq_lane_f32(acc, 1);
            }
            if ch > 2 {
                *out.get_unchecked_mut(base + 2) = vgetq_lane_f32(acc, 2);
            }
            if ch > 3 {
                *out.get_unchecked_mut(base + 3) = vgetq_lane_f32(acc, 3);
            }
            i += 1;
        }
    }
    out.truncate(n);
    out
}

// ===========================================================================
// AVX2: across 8 output pixels per row, hardware gather for the four taps.
// ch == 1 (contiguous store) and ch == 3 (strided store) — the tracker cases.
// ===========================================================================
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn crop_resize_avx2(
    src: &[f32],
    h: usize,
    w: usize,
    ch: usize,
    cx: f32,
    cy: f32,
    win_w: usize,
    win_h: usize,
    tpl_w: usize,
    tpl_h: usize,
) -> Vec<f32> {
    use std::arch::x86_64::*;

    let (ox, oy) = map_origin(cx, cy, win_w, win_h);
    let sx = win_w as f32 / tpl_w as f32;
    let sy = win_h as f32 / tpl_h as f32;
    let mut out = vec![0f32; tpl_h * tpl_w * ch];

    let one = _mm256_set1_ps(1.0);
    let ramp = _mm256_set_ps(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
    let wmax = _mm256_set1_epi32(w as i32 - 1);
    let hmax = _mm256_set1_epi32(h as i32 - 1);
    let zero_i = _mm256_setzero_si256();
    let ch_i = _mm256_set1_epi32(ch as i32);
    let src_ptr = src.as_ptr();

    for j in 0..tpl_h {
        let fy = oy + (j as f32 + 0.5) * sy - 0.5;
        let y0f = fy.floor();
        let ay = _mm256_set1_ps(fy - y0f);
        let ay1 = _mm256_sub_ps(one, ay);
        let y0 = (y0f as isize).clamp(0, h as isize - 1) as i32;
        let y1 = (y0f as isize + 1).clamp(0, h as isize - 1) as i32;
        let r0 = _mm256_set1_epi32(y0 * w as i32);
        let r1 = _mm256_set1_epi32(y1 * w as i32);

        let mut i = 0usize;
        while i + 8 <= tpl_w {
            // fx = (ox + (i + lane + 0.5) * sx) - 0.5  — same association as the
            // scalar path (FP is non-associative; matching it keeps bit-exactness,
            // which the chaos-sensitive tracker relies on).
            let idx = _mm256_add_ps(_mm256_set1_ps(i as f32), ramp);
            let t = _mm256_mul_ps(_mm256_add_ps(idx, _mm256_set1_ps(0.5)), _mm256_set1_ps(sx));
            let fx = _mm256_sub_ps(_mm256_add_ps(_mm256_set1_ps(ox), t), _mm256_set1_ps(0.5));
            let x0f = _mm256_floor_ps(fx);
            let ax = _mm256_sub_ps(fx, x0f);
            let ax1 = _mm256_sub_ps(one, ax);
            // clamp x0,x1 to [0,w-1]
            let x0i = _mm256_min_epi32(_mm256_max_epi32(_mm256_cvttps_epi32(x0f), zero_i), wmax);
            let x1i = _mm256_min_epi32(
                _mm256_max_epi32(
                    _mm256_add_epi32(_mm256_cvttps_epi32(x0f), _mm256_set1_epi32(1)),
                    zero_i,
                ),
                wmax,
            );
            let _ = hmax; // y already clamped scalar
            // weights
            let w00 = _mm256_mul_ps(ax1, ay1);
            let w01 = _mm256_mul_ps(ax, ay1);
            let w10 = _mm256_mul_ps(ax1, ay);
            let w11 = _mm256_mul_ps(ax, ay);
            // pixel base indices (element offsets) for the 4 taps, times ch
            let b00 = _mm256_mullo_epi32(_mm256_add_epi32(r0, x0i), ch_i);
            let b01 = _mm256_mullo_epi32(_mm256_add_epi32(r0, x1i), ch_i);
            let b10 = _mm256_mullo_epi32(_mm256_add_epi32(r1, x0i), ch_i);
            let b11 = _mm256_mullo_epi32(_mm256_add_epi32(r1, x1i), ch_i);
            for c in 0..ch as i32 {
                let cc = _mm256_set1_epi32(c);
                let g00 = _mm256_i32gather_ps::<4>(src_ptr, _mm256_add_epi32(b00, cc));
                let g01 = _mm256_i32gather_ps::<4>(src_ptr, _mm256_add_epi32(b01, cc));
                let g10 = _mm256_i32gather_ps::<4>(src_ptr, _mm256_add_epi32(b10, cc));
                let g11 = _mm256_i32gather_ps::<4>(src_ptr, _mm256_add_epi32(b11, cc));
                // ((w00*g00 + w01*g01) + w10*g10) + w11*g11  (no fma: bit-exact)
                let mut acc = _mm256_mul_ps(w00, g00);
                acc = _mm256_add_ps(acc, _mm256_mul_ps(w01, g01));
                acc = _mm256_add_ps(acc, _mm256_mul_ps(w10, g10));
                acc = _mm256_add_ps(acc, _mm256_mul_ps(w11, g11));
                if ch == 1 {
                    _mm256_storeu_ps(out.as_mut_ptr().add(j * tpl_w + i), acc);
                } else {
                    let mut tmp = [0f32; 8];
                    _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
                    for lane in 0..8 {
                        *out.get_unchecked_mut((j * tpl_w + i + lane) * ch + c as usize) =
                            tmp[lane];
                    }
                }
            }
            i += 8;
        }
        // scalar tail for the remaining < 8 columns of this row
        for ii in i..tpl_w {
            let fx = ox + (ii as f32 + 0.5) * sx - 0.5;
            let x0f = fx.floor();
            let ax = fx - x0f;
            let x0 = (x0f as isize).clamp(0, w as isize - 1) as usize;
            let x1 = (x0f as isize + 1).clamp(0, w as isize - 1) as usize;
            let ayf = fy - y0f;
            let (cw00, cw01) = ((1.0 - ax) * (1.0 - ayf), ax * (1.0 - ayf));
            let (cw10, cw11) = ((1.0 - ax) * ayf, ax * ayf);
            let o00 = (y0 as usize * w + x0) * ch;
            let o01 = (y0 as usize * w + x1) * ch;
            let o10 = (y1 as usize * w + x0) * ch;
            let o11 = (y1 as usize * w + x1) * ch;
            let base = (j * tpl_w + ii) * ch;
            for c in 0..ch {
                *out.get_unchecked_mut(base + c) = cw00 * src[o00 + c]
                    + cw01 * src[o01 + c]
                    + cw10 * src[o10 + c]
                    + cw11 * src[o11 + c];
            }
        }
    }
    out
}

// ===========================================================================
// SSE4.1: per output pixel, bilinear across the ch<=4 channel lanes of one xmm.
// (SSE has no gather, so this mirrors the NEON channel-lane structure.)
// ===========================================================================
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn crop_resize_sse(
    src: &[f32],
    h: usize,
    w: usize,
    ch: usize,
    cx: f32,
    cy: f32,
    win_w: usize,
    win_h: usize,
    tpl_w: usize,
    tpl_h: usize,
) -> Vec<f32> {
    use std::arch::x86_64::*;
    #[inline(always)]
    unsafe fn loadn(src: &[f32], o: usize, ch: usize) -> __m128 {
        let mut a = [0f32; 4];
        a[0] = *src.get_unchecked(o);
        if ch > 1 {
            a[1] = *src.get_unchecked(o + 1);
        }
        if ch > 2 {
            a[2] = *src.get_unchecked(o + 2);
        }
        if ch > 3 {
            a[3] = *src.get_unchecked(o + 3);
        }
        _mm_loadu_ps(a.as_ptr())
    }

    let (ox, oy) = map_origin(cx, cy, win_w, win_h);
    let sx = win_w as f32 / tpl_w as f32;
    let sy = win_h as f32 / tpl_h as f32;
    let (wi, hi) = (w as isize, h as isize);
    let mut out = vec![0f32; tpl_h * tpl_w * ch];
    for j in 0..tpl_h {
        let fy = oy + (j as f32 + 0.5) * sy - 0.5;
        let y0f = fy.floor();
        let ay = fy - y0f;
        let y0 = (y0f as isize).clamp(0, hi - 1) as usize;
        let y1 = (y0f as isize + 1).clamp(0, hi - 1) as usize;
        for i in 0..tpl_w {
            let fx = ox + (i as f32 + 0.5) * sx - 0.5;
            let x0f = fx.floor();
            let ax = fx - x0f;
            let x0 = (x0f as isize).clamp(0, wi - 1) as usize;
            let x1 = (x0f as isize + 1).clamp(0, wi - 1) as usize;
            let (w00, w01) = ((1.0 - ax) * (1.0 - ay), ax * (1.0 - ay));
            let (w10, w11) = ((1.0 - ax) * ay, ax * ay);
            let p00 = loadn(src, (y0 * w + x0) * ch, ch);
            let p01 = loadn(src, (y0 * w + x1) * ch, ch);
            let p10 = loadn(src, (y1 * w + x0) * ch, ch);
            let p11 = loadn(src, (y1 * w + x1) * ch, ch);
            let mut acc = _mm_mul_ps(p00, _mm_set1_ps(w00));
            acc = _mm_add_ps(acc, _mm_mul_ps(p01, _mm_set1_ps(w01)));
            acc = _mm_add_ps(acc, _mm_mul_ps(p10, _mm_set1_ps(w10)));
            acc = _mm_add_ps(acc, _mm_mul_ps(p11, _mm_set1_ps(w11)));
            let mut r = [0f32; 4];
            _mm_storeu_ps(r.as_mut_ptr(), acc);
            let base = (j * tpl_w + i) * ch;
            for c in 0..ch {
                *out.get_unchecked_mut(base + c) = r[c];
            }
        }
    }
    out
}

// ===========================================================================
// AVX512F: across 16 output pixels per row, hardware gather for the 4 taps.
// ===========================================================================
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn crop_resize_avx512(
    src: &[f32],
    h: usize,
    w: usize,
    ch: usize,
    cx: f32,
    cy: f32,
    win_w: usize,
    win_h: usize,
    tpl_w: usize,
    tpl_h: usize,
) -> Vec<f32> {
    use std::arch::x86_64::*;

    let (ox, oy) = map_origin(cx, cy, win_w, win_h);
    let sx = win_w as f32 / tpl_w as f32;
    let sy = win_h as f32 / tpl_h as f32;
    let mut out = vec![0f32; tpl_h * tpl_w * ch];

    let one = _mm512_set1_ps(1.0);
    let half = _mm512_set1_ps(0.5);
    let ramp = _mm512_set_ps(
        15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0,
    );
    let wmax = _mm512_set1_epi32(w as i32 - 1);
    let zero_i = _mm512_setzero_si512();
    let one_i = _mm512_set1_epi32(1);
    let ch_i = _mm512_set1_epi32(ch as i32);
    let src_ptr = src.as_ptr();

    for j in 0..tpl_h {
        let fy = oy + (j as f32 + 0.5) * sy - 0.5;
        let y0f = fy.floor();
        let ay = _mm512_set1_ps(fy - y0f);
        let ay1 = _mm512_sub_ps(one, ay);
        let y0 = (y0f as isize).clamp(0, h as isize - 1) as i32;
        let y1 = (y0f as isize + 1).clamp(0, h as isize - 1) as i32;
        let r0 = _mm512_set1_epi32(y0 * w as i32);
        let r1 = _mm512_set1_epi32(y1 * w as i32);

        let mut i = 0usize;
        while i + 16 <= tpl_w {
            let idx = _mm512_add_ps(_mm512_set1_ps(i as f32), ramp);
            let t = _mm512_mul_ps(_mm512_add_ps(idx, half), _mm512_set1_ps(sx));
            let fx = _mm512_sub_ps(_mm512_add_ps(_mm512_set1_ps(ox), t), half);
            let x0f = _mm512_roundscale_ps::<0x01>(fx); // 0x01 = round toward -inf (floor)
            let ax = _mm512_sub_ps(fx, x0f);
            let ax1 = _mm512_sub_ps(one, ax);
            let x0t = _mm512_cvttps_epi32(x0f);
            let x0i = _mm512_min_epi32(_mm512_max_epi32(x0t, zero_i), wmax);
            let x1i =
                _mm512_min_epi32(_mm512_max_epi32(_mm512_add_epi32(x0t, one_i), zero_i), wmax);
            let w00 = _mm512_mul_ps(ax1, ay1);
            let w01 = _mm512_mul_ps(ax, ay1);
            let w10 = _mm512_mul_ps(ax1, ay);
            let w11 = _mm512_mul_ps(ax, ay);
            let b00 = _mm512_mullo_epi32(_mm512_add_epi32(r0, x0i), ch_i);
            let b01 = _mm512_mullo_epi32(_mm512_add_epi32(r0, x1i), ch_i);
            let b10 = _mm512_mullo_epi32(_mm512_add_epi32(r1, x0i), ch_i);
            let b11 = _mm512_mullo_epi32(_mm512_add_epi32(r1, x1i), ch_i);
            for c in 0..ch as i32 {
                let cc = _mm512_set1_epi32(c);
                let g00 = _mm512_i32gather_ps::<4>(_mm512_add_epi32(b00, cc), src_ptr);
                let g01 = _mm512_i32gather_ps::<4>(_mm512_add_epi32(b01, cc), src_ptr);
                let g10 = _mm512_i32gather_ps::<4>(_mm512_add_epi32(b10, cc), src_ptr);
                let g11 = _mm512_i32gather_ps::<4>(_mm512_add_epi32(b11, cc), src_ptr);
                let mut acc = _mm512_mul_ps(w00, g00);
                acc = _mm512_add_ps(acc, _mm512_mul_ps(w01, g01));
                acc = _mm512_add_ps(acc, _mm512_mul_ps(w10, g10));
                acc = _mm512_add_ps(acc, _mm512_mul_ps(w11, g11));
                if ch == 1 {
                    _mm512_storeu_ps(out.as_mut_ptr().add(j * tpl_w + i), acc);
                } else {
                    let mut tmp = [0f32; 16];
                    _mm512_storeu_ps(tmp.as_mut_ptr(), acc);
                    for lane in 0..16 {
                        *out.get_unchecked_mut((j * tpl_w + i + lane) * ch + c as usize) =
                            tmp[lane];
                    }
                }
            }
            i += 16;
        }
        for ii in i..tpl_w {
            let fx = ox + (ii as f32 + 0.5) * sx - 0.5;
            let x0f = fx.floor();
            let ax = fx - x0f;
            let x0 = (x0f as isize).clamp(0, w as isize - 1) as usize;
            let x1 = (x0f as isize + 1).clamp(0, w as isize - 1) as usize;
            let ayf = fy - y0f;
            let (cw00, cw01) = ((1.0 - ax) * (1.0 - ayf), ax * (1.0 - ayf));
            let (cw10, cw11) = ((1.0 - ax) * ayf, ax * ayf);
            let o00 = (y0 as usize * w + x0) * ch;
            let o01 = (y0 as usize * w + x1) * ch;
            let o10 = (y1 as usize * w + x0) * ch;
            let o11 = (y1 as usize * w + x1) * ch;
            let base = (j * tpl_w + ii) * ch;
            for c in 0..ch {
                *out.get_unchecked_mut(base + c) = cw00 * src[o00 + c]
                    + cw01 * src[o01 + c]
                    + cw10 * src[o10 + c]
                    + cw11 * src[o11 + c];
            }
        }
    }
    out
}

// ===========================================================================
// Constant-border variants. Identical structure to the replicate paths above,
// except a bilinear tap whose *source* index is out of `[0,w)x[0,h)` contributes
// `border[c]` instead of the clamped edge pixel. Indices are still clamped for
// the memory access (gather/load); the out-of-bounds value is then masked out.
// ===========================================================================

#[allow(clippy::too_many_arguments)]
fn crop_resize_border_scalar(
    src: &[f32],
    h: usize,
    w: usize,
    ch: usize,
    cx: f32,
    cy: f32,
    win_w: usize,
    win_h: usize,
    tpl_w: usize,
    tpl_h: usize,
    border: &[f32],
) -> Vec<f32> {
    let (ox, oy) = map_origin(cx, cy, win_w, win_h);
    let sx = win_w as f32 / tpl_w as f32;
    let sy = win_h as f32 / tpl_h as f32;
    let (wi, hi) = (w as isize, h as isize);
    let mut out = vec![0f32; tpl_h * tpl_w * ch];
    for j in 0..tpl_h {
        let fy = oy + (j as f32 + 0.5) * sy - 0.5;
        let y0f = fy.floor();
        let ay = fy - y0f;
        let iy0 = y0f as isize;
        let (y0ib, y1ib) = ((0..hi).contains(&iy0), (0..hi).contains(&(iy0 + 1)));
        let y0 = iy0.clamp(0, hi - 1) as usize;
        let y1 = (iy0 + 1).clamp(0, hi - 1) as usize;
        for i in 0..tpl_w {
            let fx = ox + (i as f32 + 0.5) * sx - 0.5;
            let x0f = fx.floor();
            let ax = fx - x0f;
            let ix0 = x0f as isize;
            let (x0ib, x1ib) = ((0..wi).contains(&ix0), (0..wi).contains(&(ix0 + 1)));
            let x0 = ix0.clamp(0, wi - 1) as usize;
            let x1 = (ix0 + 1).clamp(0, wi - 1) as usize;
            let (w00, w01) = ((1.0 - ax) * (1.0 - ay), ax * (1.0 - ay));
            let (w10, w11) = ((1.0 - ax) * ay, ax * ay);
            let o00 = (y0 * w + x0) * ch;
            let o01 = (y0 * w + x1) * ch;
            let o10 = (y1 * w + x0) * ch;
            let o11 = (y1 * w + x1) * ch;
            let (m00, m01) = (x0ib && y0ib, x1ib && y0ib);
            let (m10, m11) = (x0ib && y1ib, x1ib && y1ib);
            let base = (j * tpl_w + i) * ch;
            for c in 0..ch {
                let p00 = if m00 { src[o00 + c] } else { border[c] };
                let p01 = if m01 { src[o01 + c] } else { border[c] };
                let p10 = if m10 { src[o10 + c] } else { border[c] };
                let p11 = if m11 { src[o11 + c] } else { border[c] };
                out[base + c] = w00 * p00 + w01 * p01 + w10 * p10 + w11 * p11;
            }
        }
    }
    out
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]
unsafe fn crop_resize_border_neon(
    src: &[f32],
    h: usize,
    w: usize,
    ch: usize,
    cx: f32,
    cy: f32,
    win_w: usize,
    win_h: usize,
    tpl_w: usize,
    tpl_h: usize,
    border: &[f32],
) -> Vec<f32> {
    use std::arch::aarch64::*;

    #[inline(always)]
    unsafe fn loadn(src: &[f32], o: usize, ch: usize) -> float32x4_t {
        let mut v = vdupq_n_f32(0.0);
        v = vsetq_lane_f32(*src.get_unchecked(o), v, 0);
        if ch > 1 {
            v = vsetq_lane_f32(*src.get_unchecked(o + 1), v, 1);
        }
        if ch > 2 {
            v = vsetq_lane_f32(*src.get_unchecked(o + 2), v, 2);
        }
        if ch > 3 {
            v = vsetq_lane_f32(*src.get_unchecked(o + 3), v, 3);
        }
        v
    }

    let (ox, oy) = map_origin(cx, cy, win_w, win_h);
    let sx = win_w as f32 / tpl_w as f32;
    let sy = win_h as f32 / tpl_h as f32;
    let (wi, hi) = (w as isize, h as isize);
    let bv = loadn(border, 0, ch);
    let n = tpl_h * tpl_w * ch;
    // Slack for the 4-lane store of the last pixel; see `tap4_neon`.
    let mut out = vec![0f32; n + 4];
    for j in 0..tpl_h {
        let fy = oy + (j as f32 + 0.5) * sy - 0.5;
        let y0f = fy.floor();
        let ay = fy - y0f;
        let iy0 = y0f as isize;
        let (y0ib, y1ib) = ((0..hi).contains(&iy0), (0..hi).contains(&(iy0 + 1)));
        let y0 = iy0.clamp(0, hi - 1) as usize;
        let y1 = (iy0 + 1).clamp(0, hi - 1) as usize;
        let rows_free = y0ib && y1ib;
        let ayv = frac_y_neon(ay);
        let mut i = 0;
        while i < tpl_w {
            let at = (j * tpl_w + i) * ch;
            if rows_free
                && i + 4 <= tpl_w
                && group4_neon(src, &mut out, i, ox, sx, ch, w, wi, y0, y1, at, ayv)
            {
                i += 4;
                continue;
            }
            let fx = ox + (i as f32 + 0.5) * sx - 0.5;
            let x0f = fx.floor();
            let ax = fx - x0f;
            let ix0 = x0f as isize;
            let (x0ib, x1ib) = ((0..wi).contains(&ix0), (0..wi).contains(&(ix0 + 1)));
            let x0 = ix0.clamp(0, wi - 1) as usize;
            let x1 = (ix0 + 1).clamp(0, wi - 1) as usize;
            let (w00, w01) = ((1.0 - ax) * (1.0 - ay), ax * (1.0 - ay));
            let (w10, w11) = ((1.0 - ax) * ay, ax * ay);
            let sel = |ib: bool, o: usize| if ib { loadn(src, o, ch) } else { bv };
            let p00 = sel(x0ib && y0ib, (y0 * w + x0) * ch);
            let p01 = sel(x1ib && y0ib, (y0 * w + x1) * ch);
            let p10 = sel(x0ib && y1ib, (y1 * w + x0) * ch);
            let p11 = sel(x1ib && y1ib, (y1 * w + x1) * ch);
            let mut acc = vmulq_n_f32(p00, w00);
            acc = vaddq_f32(acc, vmulq_n_f32(p01, w01));
            acc = vaddq_f32(acc, vmulq_n_f32(p10, w10));
            acc = vaddq_f32(acc, vmulq_n_f32(p11, w11));
            let base = (j * tpl_w + i) * ch;
            *out.get_unchecked_mut(base) = vgetq_lane_f32(acc, 0);
            if ch > 1 {
                *out.get_unchecked_mut(base + 1) = vgetq_lane_f32(acc, 1);
            }
            if ch > 2 {
                *out.get_unchecked_mut(base + 2) = vgetq_lane_f32(acc, 2);
            }
            if ch > 3 {
                *out.get_unchecked_mut(base + 3) = vgetq_lane_f32(acc, 3);
            }
            i += 1;
        }
    }
    out.truncate(n);
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]
unsafe fn crop_resize_border_sse(
    src: &[f32],
    h: usize,
    w: usize,
    ch: usize,
    cx: f32,
    cy: f32,
    win_w: usize,
    win_h: usize,
    tpl_w: usize,
    tpl_h: usize,
    border: &[f32],
) -> Vec<f32> {
    use std::arch::x86_64::*;
    #[inline(always)]
    unsafe fn loadn(src: &[f32], o: usize, ch: usize) -> __m128 {
        let mut a = [0f32; 4];
        a[0] = *src.get_unchecked(o);
        if ch > 1 {
            a[1] = *src.get_unchecked(o + 1);
        }
        if ch > 2 {
            a[2] = *src.get_unchecked(o + 2);
        }
        if ch > 3 {
            a[3] = *src.get_unchecked(o + 3);
        }
        _mm_loadu_ps(a.as_ptr())
    }

    let (ox, oy) = map_origin(cx, cy, win_w, win_h);
    let sx = win_w as f32 / tpl_w as f32;
    let sy = win_h as f32 / tpl_h as f32;
    let (wi, hi) = (w as isize, h as isize);
    let bv = loadn(border, 0, ch);
    let mut out = vec![0f32; tpl_h * tpl_w * ch];
    for j in 0..tpl_h {
        let fy = oy + (j as f32 + 0.5) * sy - 0.5;
        let y0f = fy.floor();
        let ay = fy - y0f;
        let iy0 = y0f as isize;
        let (y0ib, y1ib) = ((0..hi).contains(&iy0), (0..hi).contains(&(iy0 + 1)));
        let y0 = iy0.clamp(0, hi - 1) as usize;
        let y1 = (iy0 + 1).clamp(0, hi - 1) as usize;
        for i in 0..tpl_w {
            let fx = ox + (i as f32 + 0.5) * sx - 0.5;
            let x0f = fx.floor();
            let ax = fx - x0f;
            let ix0 = x0f as isize;
            let (x0ib, x1ib) = ((0..wi).contains(&ix0), (0..wi).contains(&(ix0 + 1)));
            let x0 = ix0.clamp(0, wi - 1) as usize;
            let x1 = (ix0 + 1).clamp(0, wi - 1) as usize;
            let (w00, w01) = ((1.0 - ax) * (1.0 - ay), ax * (1.0 - ay));
            let (w10, w11) = ((1.0 - ax) * ay, ax * ay);
            let sel = |ib: bool, o: usize| if ib { loadn(src, o, ch) } else { bv };
            let p00 = sel(x0ib && y0ib, (y0 * w + x0) * ch);
            let p01 = sel(x1ib && y0ib, (y0 * w + x1) * ch);
            let p10 = sel(x0ib && y1ib, (y1 * w + x0) * ch);
            let p11 = sel(x1ib && y1ib, (y1 * w + x1) * ch);
            let mut acc = _mm_mul_ps(p00, _mm_set1_ps(w00));
            acc = _mm_add_ps(acc, _mm_mul_ps(p01, _mm_set1_ps(w01)));
            acc = _mm_add_ps(acc, _mm_mul_ps(p10, _mm_set1_ps(w10)));
            acc = _mm_add_ps(acc, _mm_mul_ps(p11, _mm_set1_ps(w11)));
            let mut r = [0f32; 4];
            _mm_storeu_ps(r.as_mut_ptr(), acc);
            let base = (j * tpl_w + i) * ch;
            for c in 0..ch {
                *out.get_unchecked_mut(base + c) = r[c];
            }
        }
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]
unsafe fn crop_resize_border_avx2(
    src: &[f32],
    h: usize,
    w: usize,
    ch: usize,
    cx: f32,
    cy: f32,
    win_w: usize,
    win_h: usize,
    tpl_w: usize,
    tpl_h: usize,
    border: &[f32],
) -> Vec<f32> {
    use std::arch::x86_64::*;

    let (ox, oy) = map_origin(cx, cy, win_w, win_h);
    let sx = win_w as f32 / tpl_w as f32;
    let sy = win_h as f32 / tpl_h as f32;
    let (wi, hi) = (w as isize, h as isize);
    let mut out = vec![0f32; tpl_h * tpl_w * ch];

    let one = _mm256_set1_ps(1.0);
    let ramp = _mm256_set_ps(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
    let wmax = _mm256_set1_epi32(w as i32 - 1);
    let zero_i = _mm256_setzero_si256();
    let neg1_i = _mm256_set1_epi32(-1);
    let wcount = _mm256_set1_epi32(w as i32);
    let ch_i = _mm256_set1_epi32(ch as i32);
    let src_ptr = src.as_ptr();

    for j in 0..tpl_h {
        let fy = oy + (j as f32 + 0.5) * sy - 0.5;
        let y0f = fy.floor();
        let ay = _mm256_set1_ps(fy - y0f);
        let ay1 = _mm256_sub_ps(one, ay);
        let iy0 = y0f as isize;
        let (y0ib, y1ib) = ((0..hi).contains(&iy0), (0..hi).contains(&(iy0 + 1)));
        let y0 = iy0.clamp(0, hi - 1) as i32;
        let y1 = (iy0 + 1).clamp(0, hi - 1) as i32;
        let r0 = _mm256_set1_epi32(y0 * w as i32);
        let r1 = _mm256_set1_epi32(y1 * w as i32);
        let ym0 = if y0ib { neg1_i } else { zero_i };
        let ym1 = if y1ib { neg1_i } else { zero_i };

        let mut i = 0usize;
        while i + 8 <= tpl_w {
            let idx = _mm256_add_ps(_mm256_set1_ps(i as f32), ramp);
            let t = _mm256_mul_ps(_mm256_add_ps(idx, _mm256_set1_ps(0.5)), _mm256_set1_ps(sx));
            let fx = _mm256_sub_ps(_mm256_add_ps(_mm256_set1_ps(ox), t), _mm256_set1_ps(0.5));
            let x0f = _mm256_floor_ps(fx);
            let ax = _mm256_sub_ps(fx, x0f);
            let ax1 = _mm256_sub_ps(one, ax);
            let x0t = _mm256_cvttps_epi32(x0f);
            let x1t = _mm256_add_epi32(x0t, _mm256_set1_epi32(1));
            // per-lane in-bounds: 0 <= idx <= w-1  ==  (idx > -1) & (w > idx)
            let x0v = _mm256_and_si256(
                _mm256_cmpgt_epi32(x0t, neg1_i),
                _mm256_cmpgt_epi32(wcount, x0t),
            );
            let x1v = _mm256_and_si256(
                _mm256_cmpgt_epi32(x1t, neg1_i),
                _mm256_cmpgt_epi32(wcount, x1t),
            );
            let x0i = _mm256_min_epi32(_mm256_max_epi32(x0t, zero_i), wmax);
            let x1i = _mm256_min_epi32(_mm256_max_epi32(x1t, zero_i), wmax);
            let m00 = _mm256_castsi256_ps(_mm256_and_si256(x0v, ym0));
            let m01 = _mm256_castsi256_ps(_mm256_and_si256(x1v, ym0));
            let m10 = _mm256_castsi256_ps(_mm256_and_si256(x0v, ym1));
            let m11 = _mm256_castsi256_ps(_mm256_and_si256(x1v, ym1));
            let w00 = _mm256_mul_ps(ax1, ay1);
            let w01 = _mm256_mul_ps(ax, ay1);
            let w10 = _mm256_mul_ps(ax1, ay);
            let w11 = _mm256_mul_ps(ax, ay);
            let b00 = _mm256_mullo_epi32(_mm256_add_epi32(r0, x0i), ch_i);
            let b01 = _mm256_mullo_epi32(_mm256_add_epi32(r0, x1i), ch_i);
            let b10 = _mm256_mullo_epi32(_mm256_add_epi32(r1, x0i), ch_i);
            let b11 = _mm256_mullo_epi32(_mm256_add_epi32(r1, x1i), ch_i);
            for c in 0..ch as i32 {
                let cc = _mm256_set1_epi32(c);
                let bc = _mm256_set1_ps(*border.get_unchecked(c as usize));
                let g00 = _mm256_blendv_ps(
                    bc,
                    _mm256_i32gather_ps::<4>(src_ptr, _mm256_add_epi32(b00, cc)),
                    m00,
                );
                let g01 = _mm256_blendv_ps(
                    bc,
                    _mm256_i32gather_ps::<4>(src_ptr, _mm256_add_epi32(b01, cc)),
                    m01,
                );
                let g10 = _mm256_blendv_ps(
                    bc,
                    _mm256_i32gather_ps::<4>(src_ptr, _mm256_add_epi32(b10, cc)),
                    m10,
                );
                let g11 = _mm256_blendv_ps(
                    bc,
                    _mm256_i32gather_ps::<4>(src_ptr, _mm256_add_epi32(b11, cc)),
                    m11,
                );
                let mut acc = _mm256_mul_ps(w00, g00);
                acc = _mm256_add_ps(acc, _mm256_mul_ps(w01, g01));
                acc = _mm256_add_ps(acc, _mm256_mul_ps(w10, g10));
                acc = _mm256_add_ps(acc, _mm256_mul_ps(w11, g11));
                if ch == 1 {
                    _mm256_storeu_ps(out.as_mut_ptr().add(j * tpl_w + i), acc);
                } else {
                    let mut tmp = [0f32; 8];
                    _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
                    for lane in 0..8 {
                        *out.get_unchecked_mut((j * tpl_w + i + lane) * ch + c as usize) =
                            tmp[lane];
                    }
                }
            }
            i += 8;
        }
        for ii in i..tpl_w {
            let fx = ox + (ii as f32 + 0.5) * sx - 0.5;
            let x0f = fx.floor();
            let ax = fx - x0f;
            let ix0 = x0f as isize;
            let (x0ib, x1ib) = ((0..wi).contains(&ix0), (0..wi).contains(&(ix0 + 1)));
            let x0 = ix0.clamp(0, wi - 1) as usize;
            let x1 = (ix0 + 1).clamp(0, wi - 1) as usize;
            let ayf = fy - y0f;
            let (cw00, cw01) = ((1.0 - ax) * (1.0 - ayf), ax * (1.0 - ayf));
            let (cw10, cw11) = ((1.0 - ax) * ayf, ax * ayf);
            let o00 = (y0 as usize * w + x0) * ch;
            let o01 = (y0 as usize * w + x1) * ch;
            let o10 = (y1 as usize * w + x0) * ch;
            let o11 = (y1 as usize * w + x1) * ch;
            let base = (j * tpl_w + ii) * ch;
            for c in 0..ch {
                let p00 = if x0ib && y0ib {
                    src[o00 + c]
                } else {
                    border[c]
                };
                let p01 = if x1ib && y0ib {
                    src[o01 + c]
                } else {
                    border[c]
                };
                let p10 = if x0ib && y1ib {
                    src[o10 + c]
                } else {
                    border[c]
                };
                let p11 = if x1ib && y1ib {
                    src[o11 + c]
                } else {
                    border[c]
                };
                *out.get_unchecked_mut(base + c) =
                    cw00 * p00 + cw01 * p01 + cw10 * p10 + cw11 * p11;
            }
        }
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]
unsafe fn crop_resize_border_avx512(
    src: &[f32],
    h: usize,
    w: usize,
    ch: usize,
    cx: f32,
    cy: f32,
    win_w: usize,
    win_h: usize,
    tpl_w: usize,
    tpl_h: usize,
    border: &[f32],
) -> Vec<f32> {
    use std::arch::x86_64::*;

    let (ox, oy) = map_origin(cx, cy, win_w, win_h);
    let sx = win_w as f32 / tpl_w as f32;
    let sy = win_h as f32 / tpl_h as f32;
    let (wi, hi) = (w as isize, h as isize);
    let mut out = vec![0f32; tpl_h * tpl_w * ch];

    let one = _mm512_set1_ps(1.0);
    let half = _mm512_set1_ps(0.5);
    let ramp = _mm512_set_ps(
        15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0,
    );
    let wmax = _mm512_set1_epi32(w as i32 - 1);
    let zero_i = _mm512_setzero_si512();
    let one_i = _mm512_set1_epi32(1);
    let ch_i = _mm512_set1_epi32(ch as i32);
    let src_ptr = src.as_ptr();

    for j in 0..tpl_h {
        let fy = oy + (j as f32 + 0.5) * sy - 0.5;
        let y0f = fy.floor();
        let ay = _mm512_set1_ps(fy - y0f);
        let ay1 = _mm512_sub_ps(one, ay);
        let iy0 = y0f as isize;
        let (y0ib, y1ib) = ((0..hi).contains(&iy0), (0..hi).contains(&(iy0 + 1)));
        let y0 = iy0.clamp(0, hi - 1) as i32;
        let y1 = (iy0 + 1).clamp(0, hi - 1) as i32;
        let r0 = _mm512_set1_epi32(y0 * w as i32);
        let r1 = _mm512_set1_epi32(y1 * w as i32);
        let ym0: __mmask16 = if y0ib { 0xFFFF } else { 0 };
        let ym1: __mmask16 = if y1ib { 0xFFFF } else { 0 };

        let mut i = 0usize;
        while i + 16 <= tpl_w {
            let idx = _mm512_add_ps(_mm512_set1_ps(i as f32), ramp);
            let t = _mm512_mul_ps(_mm512_add_ps(idx, half), _mm512_set1_ps(sx));
            let fx = _mm512_sub_ps(_mm512_add_ps(_mm512_set1_ps(ox), t), half);
            let x0f = _mm512_roundscale_ps::<0x01>(fx);
            let ax = _mm512_sub_ps(fx, x0f);
            let ax1 = _mm512_sub_ps(one, ax);
            let x0t = _mm512_cvttps_epi32(x0f);
            let x1t = _mm512_add_epi32(x0t, one_i);
            let x0v = _kand_mask16(
                _mm512_cmpge_epi32_mask(x0t, zero_i),
                _mm512_cmple_epi32_mask(x0t, wmax),
            );
            let x1v = _kand_mask16(
                _mm512_cmpge_epi32_mask(x1t, zero_i),
                _mm512_cmple_epi32_mask(x1t, wmax),
            );
            let x0i = _mm512_min_epi32(_mm512_max_epi32(x0t, zero_i), wmax);
            let x1i = _mm512_min_epi32(_mm512_max_epi32(x1t, zero_i), wmax);
            let m00 = _kand_mask16(x0v, ym0);
            let m01 = _kand_mask16(x1v, ym0);
            let m10 = _kand_mask16(x0v, ym1);
            let m11 = _kand_mask16(x1v, ym1);
            let w00 = _mm512_mul_ps(ax1, ay1);
            let w01 = _mm512_mul_ps(ax, ay1);
            let w10 = _mm512_mul_ps(ax1, ay);
            let w11 = _mm512_mul_ps(ax, ay);
            let b00 = _mm512_mullo_epi32(_mm512_add_epi32(r0, x0i), ch_i);
            let b01 = _mm512_mullo_epi32(_mm512_add_epi32(r0, x1i), ch_i);
            let b10 = _mm512_mullo_epi32(_mm512_add_epi32(r1, x0i), ch_i);
            let b11 = _mm512_mullo_epi32(_mm512_add_epi32(r1, x1i), ch_i);
            for c in 0..ch as i32 {
                let cc = _mm512_set1_epi32(c);
                let bc = _mm512_set1_ps(*border.get_unchecked(c as usize));
                let g00 = _mm512_mask_blend_ps(
                    m00,
                    bc,
                    _mm512_i32gather_ps::<4>(_mm512_add_epi32(b00, cc), src_ptr),
                );
                let g01 = _mm512_mask_blend_ps(
                    m01,
                    bc,
                    _mm512_i32gather_ps::<4>(_mm512_add_epi32(b01, cc), src_ptr),
                );
                let g10 = _mm512_mask_blend_ps(
                    m10,
                    bc,
                    _mm512_i32gather_ps::<4>(_mm512_add_epi32(b10, cc), src_ptr),
                );
                let g11 = _mm512_mask_blend_ps(
                    m11,
                    bc,
                    _mm512_i32gather_ps::<4>(_mm512_add_epi32(b11, cc), src_ptr),
                );
                let mut acc = _mm512_mul_ps(w00, g00);
                acc = _mm512_add_ps(acc, _mm512_mul_ps(w01, g01));
                acc = _mm512_add_ps(acc, _mm512_mul_ps(w10, g10));
                acc = _mm512_add_ps(acc, _mm512_mul_ps(w11, g11));
                if ch == 1 {
                    _mm512_storeu_ps(out.as_mut_ptr().add(j * tpl_w + i), acc);
                } else {
                    let mut tmp = [0f32; 16];
                    _mm512_storeu_ps(tmp.as_mut_ptr(), acc);
                    for lane in 0..16 {
                        *out.get_unchecked_mut((j * tpl_w + i + lane) * ch + c as usize) =
                            tmp[lane];
                    }
                }
            }
            i += 16;
        }
        for ii in i..tpl_w {
            let fx = ox + (ii as f32 + 0.5) * sx - 0.5;
            let x0f = fx.floor();
            let ax = fx - x0f;
            let ix0 = x0f as isize;
            let (x0ib, x1ib) = ((0..wi).contains(&ix0), (0..wi).contains(&(ix0 + 1)));
            let x0 = ix0.clamp(0, wi - 1) as usize;
            let x1 = (ix0 + 1).clamp(0, wi - 1) as usize;
            let ayf = fy - y0f;
            let (cw00, cw01) = ((1.0 - ax) * (1.0 - ayf), ax * (1.0 - ayf));
            let (cw10, cw11) = ((1.0 - ax) * ayf, ax * ayf);
            let o00 = (y0 as usize * w + x0) * ch;
            let o01 = (y0 as usize * w + x1) * ch;
            let o10 = (y1 as usize * w + x0) * ch;
            let o11 = (y1 as usize * w + x1) * ch;
            let base = (j * tpl_w + ii) * ch;
            for c in 0..ch {
                let p00 = if x0ib && y0ib {
                    src[o00 + c]
                } else {
                    border[c]
                };
                let p01 = if x1ib && y0ib {
                    src[o01 + c]
                } else {
                    border[c]
                };
                let p10 = if x0ib && y1ib {
                    src[o10 + c]
                } else {
                    border[c]
                };
                let p11 = if x1ib && y1ib {
                    src[o11 + c]
                } else {
                    border[c]
                };
                *out.get_unchecked_mut(base + c) =
                    cw00 * p00 + cw01 * p01 + cw10 * p10 + cw11 * p11;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth(h: usize, w: usize, ch: usize) -> Vec<f32> {
        let mut s = 7u32;
        (0..h * w * ch)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 8) as f32 / 16_777_216.0 * 255.0
            })
            .collect()
    }

    /// The constant-border variant: every SIMD path bit-identical to the scalar
    /// border reference, on windows that run off the image edges so the border
    /// is actually taken (mask + blend / per-tap select), plus a fully-off case.
    #[test]
    fn simd_border_matches_scalar() {
        let (h, w) = (200usize, 240usize);
        // (win_w, win_h, tpl_w, tpl_h, cx, cy): corners, big context, fully off.
        let cases = [
            (300usize, 300usize, 128usize, 128usize, 18.0f32, 12.0f32),
            (400, 260, 160, 136, 120.0, 100.0),
            (64, 64, 40, 40, -220.0, -200.0), // entirely off -> all border
            (150, 150, 48, 48, 236.0, 196.0), // off the bottom-right corner
        ];
        let bitexact = |want: &[f32], got: &[f32], label: &str| {
            let mx = want
                .iter()
                .zip(got)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            assert_eq!(mx, 0.0, "{label} max diff {mx}");
        };
        for &ch in &[1usize, 3] {
            let src = synth(h, w, ch);
            let border: Vec<f32> = (0..ch).map(|c| 30.0 + 50.0 * c as f32).collect();
            for &(ww, wh, tw, th, cx, cy) in &cases {
                let want =
                    crop_resize_border_scalar(&src, h, w, ch, cx, cy, ww, wh, tw, th, &border);
                let lbl = format!("ch={ch} win={ww}x{wh} tpl={tw}x{th} c=({cx},{cy})");
                bitexact(
                    &want,
                    &crop_resize_bilinear_border_raw(
                        &src, h, w, ch, cx, cy, ww, wh, tw, th, &border,
                    ),
                    &lbl,
                );
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    if (ch == 1 || ch == 3) && yscv_cpu::host_cpu().features.avx512f {
                        bitexact(
                            &want,
                            &crop_resize_border_avx512(
                                &src, h, w, ch, cx, cy, ww, wh, tw, th, &border,
                            ),
                            &lbl,
                        );
                    }
                    if (ch == 1 || ch == 3) && yscv_cpu::host_cpu().features.avx2 {
                        bitexact(
                            &want,
                            &crop_resize_border_avx2(
                                &src, h, w, ch, cx, cy, ww, wh, tw, th, &border,
                            ),
                            &lbl,
                        );
                    }
                    if yscv_cpu::host_cpu().features.sse41 {
                        bitexact(
                            &want,
                            &crop_resize_border_sse(
                                &src, h, w, ch, cx, cy, ww, wh, tw, th, &border,
                            ),
                            &lbl,
                        );
                    }
                }
                #[cfg(target_arch = "aarch64")]
                unsafe {
                    bitexact(
                        &want,
                        &crop_resize_border_neon(&src, h, w, ch, cx, cy, ww, wh, tw, th, &border),
                        &lbl,
                    );
                }
            }
        }
    }

    /// Every dispatched SIMD path must be **bit-identical** to the scalar
    /// reference (a 1-ULP drift bifurcates chaos-sensitive trackers). Several
    /// window/template ratios, including a large down-sampling one like a real
    /// search window, to exercise coordinate association and the 8-lane tail.
    #[test]
    fn simd_matches_scalar() {
        let (h, w) = (720, 1280);
        let cases = [
            (55usize, 47usize, 23usize, 19usize), // small up/near
            (350, 350, 160, 136),                 // search window -> capped model
            (140, 140, 33, 33),                   // scale-filter-ish
            (37, 41, 40, 40),                     // up-sampling
        ];
        let bitexact = |want: &[f32], got: &[f32], label: &str| {
            let mx = want
                .iter()
                .zip(got)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            assert_eq!(mx, 0.0, "{label} max diff {mx}");
        };
        for &ch in &[1usize, 3] {
            let src = synth(h, w, ch);
            for &(ww, wh, tw, th) in &cases {
                let (cx, cy) = (623.4f32, 408.6f32);
                let want = crop_resize_scalar(&src, h, w, ch, cx, cy, ww, wh, tw, th);
                let lbl = format!("ch={ch} win={ww}x{wh} tpl={tw}x{th}");
                // the dispatched entry point (whatever the host picks)
                bitexact(
                    &want,
                    &crop_resize_bilinear_raw(&src, h, w, ch, cx, cy, ww, wh, tw, th),
                    &lbl,
                );
                // and each individual SIMD path the host supports
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    if (ch == 1 || ch == 3) && yscv_cpu::host_cpu().features.avx512f {
                        bitexact(
                            &want,
                            &crop_resize_avx512(&src, h, w, ch, cx, cy, ww, wh, tw, th),
                            &lbl,
                        );
                    }
                    if (ch == 1 || ch == 3) && yscv_cpu::host_cpu().features.avx2 {
                        bitexact(
                            &want,
                            &crop_resize_avx2(&src, h, w, ch, cx, cy, ww, wh, tw, th),
                            &lbl,
                        );
                    }
                    if yscv_cpu::host_cpu().features.sse41 {
                        bitexact(
                            &want,
                            &crop_resize_sse(&src, h, w, ch, cx, cy, ww, wh, tw, th),
                            &lbl,
                        );
                    }
                }
                #[cfg(target_arch = "aarch64")]
                unsafe {
                    bitexact(
                        &want,
                        &crop_resize_neon(&src, h, w, ch, cx, cy, ww, wh, tw, th),
                        &lbl,
                    );
                }
            }
        }
    }
}
