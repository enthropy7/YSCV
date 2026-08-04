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
        return unsafe {
            crop_resize_neon(src, h, w, ch, cx, cy, win_w, win_h, tpl_w, tpl_h)
        };
    }
    #[cfg(target_arch = "x86_64")]
    if !cfg!(miri) {
        // SAFETY: each path is guarded by runtime feature detection; all are
        // bit-exact vs scalar (parity test). Gather-based (AVX2/AVX512) handle
        // ch in {1,3}; the SSE channel-lane path handles ch <= 4.
        let f = &yscv_cpu::host_cpu().features;
        if (ch == 1 || ch == 3) && f.avx512f {
            return unsafe { crop_resize_avx512(src, h, w, ch, cx, cy, win_w, win_h, tpl_w, tpl_h) };
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
        return Err(ImgProcError::InvalidOutputDimensions { out_h: tpl_h, out_w: tpl_w });
    }
    let out = crop_resize_bilinear_raw(input.data(), h, w, ch, cx, cy, win_w, win_h, tpl_w, tpl_h);
    Tensor::from_vec(vec![tpl_h, tpl_w, ch], out).map_err(Into::into)
}

// ---- coordinate mapping (shared): cv2 half-pixel + getRectSubPix window offset.
#[inline(always)]
fn map_origin(cx: f32, cy: f32, win_w: usize, win_h: usize) -> (f32, f32) {
    (cx - (win_w as f32 - 1.0) * 0.5, cy - (win_h as f32 - 1.0) * 0.5)
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
        }
    }
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
                _mm256_max_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(x0f), _mm256_set1_epi32(1)), zero_i),
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
                    _mm256_storeu_ps(out.as_mut_ptr().add((j * tpl_w + i) + 0), acc);
                } else {
                    let mut tmp = [0f32; 8];
                    _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
                    for lane in 0..8 {
                        *out.get_unchecked_mut((j * tpl_w + i + lane) * ch + c as usize) = tmp[lane];
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
            let x1i = _mm512_min_epi32(_mm512_max_epi32(_mm512_add_epi32(x0t, one_i), zero_i), wmax);
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
                        *out.get_unchecked_mut((j * tpl_w + i + lane) * ch + c as usize) = tmp[lane];
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
            let mx = want.iter().zip(got).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
            assert_eq!(mx, 0.0, "{label} max diff {mx}");
        };
        for &ch in &[1usize, 3] {
            let src = synth(h, w, ch);
            for &(ww, wh, tw, th) in &cases {
                let (cx, cy) = (623.4f32, 408.6f32);
                let want = crop_resize_scalar(&src, h, w, ch, cx, cy, ww, wh, tw, th);
                let lbl = format!("ch={ch} win={ww}x{wh} tpl={tw}x{th}");
                // the dispatched entry point (whatever the host picks)
                bitexact(&want, &crop_resize_bilinear_raw(&src, h, w, ch, cx, cy, ww, wh, tw, th), &lbl);
                // and each individual SIMD path the host supports
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    if (ch == 1 || ch == 3) && std::is_x86_feature_detected!("avx512f") {
                        bitexact(&want, &crop_resize_avx512(&src, h, w, ch, cx, cy, ww, wh, tw, th), &lbl);
                    }
                    if (ch == 1 || ch == 3) && std::is_x86_feature_detected!("avx2") {
                        bitexact(&want, &crop_resize_avx2(&src, h, w, ch, cx, cy, ww, wh, tw, th), &lbl);
                    }
                    if std::is_x86_feature_detected!("sse4.1") {
                        bitexact(&want, &crop_resize_sse(&src, h, w, ch, cx, cy, ww, wh, tw, th), &lbl);
                    }
                }
                #[cfg(target_arch = "aarch64")]
                unsafe {
                    bitexact(&want, &crop_resize_neon(&src, h, w, ch, cx, cy, ww, wh, tw, th), &lbl);
                }
            }
        }
    }
}
