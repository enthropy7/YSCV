//! H.264 P-slice motion compensation: motion vector parsing, prediction,
//! reference frame buffering, and inter-frame block copy.

use crate::{BitstreamReader, VideoError};

// ---------------------------------------------------------------------------
// Motion vector
// ---------------------------------------------------------------------------

/// Motion vector for a macroblock partition.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MotionVector {
    pub dx: i16,
    pub dy: i16,
    pub ref_idx: usize,
}

/// Parse motion vector difference from bitstream (Exp-Golomb coded).
pub fn parse_mvd(reader: &mut BitstreamReader) -> Result<(i16, i16), VideoError> {
    let mvd_x = reader.read_se()?;
    let mvd_y = reader.read_se()?;
    Ok((mvd_x as i16, mvd_y as i16))
}

/// Predict motion vector from neighboring blocks (median prediction).
pub fn predict_mv(left: MotionVector, top: MotionVector, top_right: MotionVector) -> MotionVector {
    MotionVector {
        dx: median_of_three(left.dx, top.dx, top_right.dx),
        dy: median_of_three(left.dy, top.dy, top_right.dy),
        ref_idx: 0,
    }
}

fn median_of_three(a: i16, b: i16, c: i16) -> i16 {
    let mut arr = [a, b, c];
    arr.sort();
    arr[1]
}

// ---------------------------------------------------------------------------
// Motion compensation
// ---------------------------------------------------------------------------

/// Apply motion compensation: copy a 16x16 block from the reference frame with
/// the given motion vector offset. Pixel coordinates that fall outside the
/// reference frame are clamped to the nearest edge sample. `ref_stride` /
/// `out_stride` are row pitches in pixels (equal to the widths for contiguous
/// planes).
#[allow(clippy::too_many_arguments)]
pub fn motion_compensate_16x16(
    reference: &[u8],
    ref_width: usize,
    ref_height: usize,
    ref_stride: usize,
    channels: usize,
    mv: MotionVector,
    mb_x: usize,
    mb_y: usize,
    output: &mut [u8],
    out_stride: usize,
) {
    let src_x = (mb_x * 16) as i32 + mv.dx as i32;
    let src_y = (mb_y * 16) as i32 + mv.dy as i32;
    let ref_w = ref_width as i32;
    let ref_h = ref_height as i32;

    // Fast path for single-channel (luma) when entire block is in-bounds
    if channels == 1 && src_x >= 0 && src_x + 16 <= ref_w && src_y >= 0 && src_y + 16 <= ref_h {
        let sx = src_x as usize;
        let sy = src_y as usize;
        for row in 0..16 {
            let dst_start = (mb_y * 16 + row) * out_stride + mb_x * 16;
            let src_start = (sy + row) * ref_stride + sx;
            if dst_start + 16 <= output.len() && src_start + 16 <= reference.len() {
                output[dst_start..dst_start + 16]
                    .copy_from_slice(&reference[src_start..src_start + 16]);
            }
        }
        return;
    }

    // Slow path: per-pixel with clamping (handles edge cases + multi-channel)
    for row in 0..16 {
        let sy = (src_y + row as i32).clamp(0, ref_h - 1) as usize;
        let dst_y = mb_y * 16 + row;
        for col in 0..16 {
            let sx = (src_x + col as i32).clamp(0, ref_w - 1) as usize;
            let dst_x = mb_x * 16 + col;
            for c in 0..channels {
                let dst_idx = (dst_y * out_stride + dst_x) * channels + c;
                let src_idx = (sy * ref_stride + sx) * channels + c;
                if dst_idx < output.len() && src_idx < reference.len() {
                    output[dst_idx] = reference[src_idx];
                }
            }
        }
    }
}

/// Generic block-size motion compensation for sub-partitions (16x8, 8x16, 8x8, etc.).
#[allow(clippy::too_many_arguments)]
pub fn motion_compensate_block(
    reference: &[u8],
    ref_width: usize,
    ref_height: usize,
    ref_stride: usize,
    mv: MotionVector,
    block_x: usize,
    block_y: usize,
    block_w: usize,
    block_h: usize,
    output: &mut [u8],
    out_stride: usize,
) {
    let src_x = block_x as i32 + mv.dx as i32;
    let src_y = block_y as i32 + mv.dy as i32;
    let ref_w = ref_width as i32;
    let ref_h = ref_height as i32;

    // Fast path: entire block in-bounds
    if src_x >= 0
        && src_x + block_w as i32 <= ref_w
        && src_y >= 0
        && src_y + block_h as i32 <= ref_h
    {
        let sx = src_x as usize;
        let sy = src_y as usize;
        for row in 0..block_h {
            let dst_start = (block_y + row) * out_stride + block_x;
            let src_start = (sy + row) * ref_stride + sx;
            if dst_start + block_w <= output.len() && src_start + block_w <= reference.len() {
                output[dst_start..dst_start + block_w]
                    .copy_from_slice(&reference[src_start..src_start + block_w]);
            }
        }
        return;
    }

    // Slow path: per-pixel with clamping
    for row in 0..block_h {
        let sy = (src_y + row as i32).clamp(0, ref_h - 1) as usize;
        for col in 0..block_w {
            let sx = (src_x + col as i32).clamp(0, ref_w - 1) as usize;
            let dst_idx = (block_y + row) * out_stride + block_x + col;
            let src_idx = sy * ref_stride + sx;
            if dst_idx < output.len() && src_idx < reference.len() {
                output[dst_idx] = reference[src_idx];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Spec-compliant sub-pel motion compensation (clauses 8.4.2.2.1 / 8.4.2.2.2)
//
// The quarter-pel interpolation is separable, so each partition is predicted
// by running one or two half-pel primitives directly over the padded
// reference plane and (for quarter positions) averaging the results. The
// padding ring replicates the frame edges, so every hot inner loop is free of
// per-sample edge clamping and the kernels vectorise (NEON / SSE2 / AVX2,
// with a scalar fallback).
// ---------------------------------------------------------------------------

/// Padding (in samples) around every decoded plane. The ring holds the
/// replicated frame edges, so motion compensation reads the reference in
/// place for every vector: the six-tap apron, the SIMD load tails and
/// out-of-frame motion all land inside the buffer (vectors reaching past the
/// ring clamp in `mc_luma` / `mc_chroma` without changing the prediction).
pub(crate) const PLANE_PAD: usize = 32;

/// Row pitch, index of sample (0, 0) and total length of a padded plane.
#[inline]
pub(crate) fn padded_plane_geometry(w: usize, h: usize) -> (usize, usize, usize) {
    let stride = w + 2 * PLANE_PAD;
    (
        stride,
        PLANE_PAD * stride + PLANE_PAD,
        stride * (h + 2 * PLANE_PAD),
    )
}

/// Replicates the frame edges of a completed reference picture into the
/// padding ring (the clause 8.4.2.2.1 edge clamp, materialised): left/right
/// columns first, then whole padded rows top/bottom so the corner regions
/// replicate the corner samples.
pub(crate) fn replicate_plane_edges(plane: &mut [u8], w: usize, h: usize) {
    let (stride, origin, _) = padded_plane_geometry(w, h);
    for r in 0..h {
        let row = origin - PLANE_PAD + r * stride;
        let left = plane[row + PLANE_PAD];
        plane[row..row + PLANE_PAD].fill(left);
        let right = plane[row + PLANE_PAD + w - 1];
        plane[row + PLANE_PAD + w..row + stride].fill(right);
    }
    let (top_pad, body) = plane.split_at_mut(PLANE_PAD * stride);
    for row in top_pad.chunks_exact_mut(stride) {
        row.copy_from_slice(&body[..stride]);
    }
    let (body, bottom_pad) = plane.split_at_mut((PLANE_PAD + h) * stride);
    let last = &body[(PLANE_PAD + h - 1) * stride..];
    for row in bottom_pad.chunks_exact_mut(stride) {
        row.copy_from_slice(last);
    }
}

/// Copies `bh` rows of `N` bytes between strided planes: the constant width
/// compiles to inline vector moves instead of per-row memcpy calls.
#[inline]
fn copy_rows<const N: usize>(
    src: &[u8],
    src_off: usize,
    src_stride: usize,
    bh: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
) {
    for r in 0..bh {
        let s = src_off + r * src_stride;
        let d = dst_off + r * dst_stride;
        if let (Ok(sv), Ok(dv)) = (
            <&[u8; N]>::try_from(&src[s..s + N]),
            <&mut [u8; N]>::try_from(&mut dst[d..d + N]),
        ) {
            *dv = *sv;
        }
    }
}

/// Copies a `bw`x`bh` block of rows between strided planes. All H.264
/// partition widths are 4/8/16, which take the constant-size row paths.
fn copy_block(
    src: &[u8],
    src_off: usize,
    src_stride: usize,
    bw: usize,
    bh: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
) {
    match bw {
        4 => copy_rows::<4>(src, src_off, src_stride, bh, dst, dst_off, dst_stride),
        8 => copy_rows::<8>(src, src_off, src_stride, bh, dst, dst_off, dst_stride),
        16 => copy_rows::<16>(src, src_off, src_stride, bh, dst, dst_off, dst_stride),
        _ => {
            for r in 0..bh {
                let s = src_off + r * src_stride;
                let d = dst_off + r * dst_stride;
                dst[d..d + bw].copy_from_slice(&src[s..s + bw]);
            }
        }
    }
}

/// One half-pel filter primitive. `dy`/`dx` shift the kernel origin by one
/// integer sample for the y+1 / x+1 operands of Table 8-12 (s = b at y+1,
/// m = h at x+1).
#[derive(Clone, Copy)]
enum LumaFilter {
    /// Horizontal half-pel b: six-tap across the row, Clip1((t + 16) >> 5).
    H { dy: usize },
    /// Vertical half-pel h: six-tap down the column, Clip1((t + 16) >> 5).
    V { dx: usize },
    /// Centre half-pel j: vertical six-tap over the *unclipped* horizontal
    /// intermediates, Clip1((t + 512) >> 10).
    C,
}

/// One interpolation operand of the quarter-pel decomposition: either integer
/// samples G (optionally shifted by one) or a half-pel filter output.
#[derive(Clone, Copy)]
enum LumaOp {
    Copy { dx: usize, dy: usize },
    Filter(LumaFilter),
}

/// The one or two operands whose rounded average forms the prediction for
/// fractional position (`xf`, `yf`) per clause 8.4.2.2.1.
fn luma_ops(xf: i32, yf: i32) -> (LumaOp, Option<LumaOp>) {
    use LumaFilter::{C, H, V};
    use LumaOp::{Copy, Filter};
    let g = Copy { dx: 0, dy: 0 };
    let b = Filter(H { dy: 0 });
    let hh = Filter(V { dx: 0 });
    let m = Filter(V { dx: 1 });
    let s = Filter(H { dy: 1 });
    let j = Filter(C);
    match (xf, yf) {
        (0, 0) => (g, None),
        (1, 0) => (g, Some(b)),
        (2, 0) => (b, None),
        (3, 0) => (Copy { dx: 1, dy: 0 }, Some(b)),
        (0, 1) => (g, Some(hh)),
        (1, 1) => (b, Some(hh)),
        (2, 1) => (b, Some(j)),
        (3, 1) => (b, Some(m)),
        (0, 2) => (hh, None),
        (1, 2) => (hh, Some(j)),
        (2, 2) => (j, None),
        (3, 2) => (j, Some(m)),
        (0, 3) => (Copy { dx: 0, dy: 1 }, Some(hh)),
        (1, 3) => (hh, Some(s)),
        (2, 3) => (j, Some(s)),
        _ => (m, Some(s)), // (3, 3)
    }
}

/// Runs one interpolation operand over the source tile `win` (row pitch
/// `stride`) whose (2, 2) origin is the block's integer position — either the
/// edge-padded copy or a window straight into the reference plane.
#[allow(unsafe_code)]
#[allow(clippy::too_many_arguments)]
fn run_luma_op(
    op: LumaOp,
    win: &[u8],
    stride: usize,
    bw: usize,
    bh: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
) {
    let filter = match op {
        LumaOp::Copy { dx, dy } => {
            // Integer samples: straight row copies out of the window.
            copy_block(
                win,
                (2 + dy) * stride + 2 + dx,
                stride,
                bw,
                bh,
                dst,
                dst_off,
                dst_stride,
            );
            return;
        }
        LumaOp::Filter(f) => f,
    };

    #[cfg(target_arch = "aarch64")]
    if bw >= 8 && yscv_cpu::host_cpu().features.neon {
        // SAFETY: NEON detected at runtime; window and destination bounds hold
        // by construction (see the kernel-side debug asserts).
        unsafe {
            match filter {
                LumaFilter::H { dy } => {
                    hpel_h_neon(win, stride, 2 + dy, bw, bh, dst, dst_off, dst_stride)
                }
                LumaFilter::V { dx } => {
                    hpel_v_neon(win, stride, 2 + dx, bw, bh, dst, dst_off, dst_stride)
                }
                LumaFilter::C => hpel_c_neon(win, stride, bw, bh, dst, dst_off, dst_stride),
            }
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        // A full 16-wide block fills one ymm of u16 lanes; 8-wide blocks map
        // natively onto the SSE2 xmm kernels.
        if bw == 16 && yscv_cpu::host_cpu().features.avx2 {
            // SAFETY: AVX2 detected at runtime; bounds as above.
            unsafe {
                match filter {
                    LumaFilter::H { dy } => {
                        hpel_h_avx2(win, stride, 2 + dy, bh, dst, dst_off, dst_stride)
                    }
                    LumaFilter::V { dx } => {
                        hpel_v_avx2(win, stride, 2 + dx, bh, dst, dst_off, dst_stride)
                    }
                    LumaFilter::C => hpel_c_avx2(win, stride, bh, dst, dst_off, dst_stride),
                }
            }
            return;
        }
        if bw >= 8 && yscv_cpu::host_cpu().features.sse2 {
            // SAFETY: SSE2 detected at runtime; bounds as above.
            unsafe {
                match filter {
                    LumaFilter::H { dy } => {
                        hpel_h_sse2(win, stride, 2 + dy, bw, bh, dst, dst_off, dst_stride)
                    }
                    LumaFilter::V { dx } => {
                        hpel_v_sse2(win, stride, 2 + dx, bw, bh, dst, dst_off, dst_stride)
                    }
                    LumaFilter::C => hpel_c_sse2(win, stride, bw, bh, dst, dst_off, dst_stride),
                }
            }
            return;
        }
    }

    match filter {
        LumaFilter::H { dy } => {
            hpel_h_scalar(win, stride, 2 + dy, bw, bh, dst, dst_off, dst_stride)
        }
        LumaFilter::V { dx } => {
            hpel_v_scalar(win, stride, 2 + dx, bw, bh, dst, dst_off, dst_stride)
        }
        LumaFilter::C => hpel_c_scalar(win, stride, bw, bh, dst, dst_off, dst_stride),
    }
}

/// Rounded average of two prediction buffers (quarter-pel positions):
/// `dst = (a + b + 1) >> 1`.
#[allow(unsafe_code)]
fn avg_block(
    a: &[u8],
    b: &[u8],
    bw: usize,
    bh: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
) {
    #[cfg(target_arch = "aarch64")]
    if bw >= 8 && yscv_cpu::host_cpu().features.neon {
        // SAFETY: NEON detected at runtime; `a`/`b` hold bw*bh samples and the
        // destination block lies inside `dst`.
        unsafe {
            avg_block_neon(a, b, bw, bh, dst, dst_off, dst_stride);
        }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if bw >= 8 && yscv_cpu::host_cpu().features.sse2 {
        // SAFETY: SSE2 detected at runtime; bounds as above.
        unsafe {
            avg_block_sse2(a, b, bw, bh, dst, dst_off, dst_stride);
        }
        return;
    }
    for r in 0..bh {
        let d = dst_off + r * dst_stride;
        for c in 0..bw {
            let i = r * bw + c;
            dst[d + c] = ((a[i] as u16 + b[i] as u16 + 1) >> 1) as u8;
        }
    }
}

/// Horizontal six-tap (1,-5,20,20,-5,1) half-pel: `Clip1((t + 16) >> 5)`.
fn hpel_h_scalar(
    win: &[u8],
    stride: usize,
    oy: usize,
    bw: usize,
    bh: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
) {
    for r in 0..bh {
        let base = (oy + r) * stride + 2;
        let d = dst_off + r * dst_stride;
        for c in 0..bw {
            let i = base + c;
            let t = win[i - 2] as i32 - 5 * win[i - 1] as i32
                + 20 * win[i] as i32
                + 20 * win[i + 1] as i32
                - 5 * win[i + 2] as i32
                + win[i + 3] as i32;
            dst[d + c] = ((t + 16) >> 5).clamp(0, 255) as u8;
        }
    }
}

/// Vertical six-tap half-pel: `Clip1((t + 16) >> 5)`.
fn hpel_v_scalar(
    win: &[u8],
    stride: usize,
    ox: usize,
    bw: usize,
    bh: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
) {
    let s_ = stride;
    for r in 0..bh {
        let base = (2 + r) * s_ + ox;
        let d = dst_off + r * dst_stride;
        for c in 0..bw {
            let i = base + c;
            let t = win[i - 2 * s_] as i32 - 5 * win[i - s_] as i32
                + 20 * win[i] as i32
                + 20 * win[i + s_] as i32
                - 5 * win[i + 2 * s_] as i32
                + win[i + 3 * s_] as i32;
            dst[d + c] = ((t + 16) >> 5).clamp(0, 255) as u8;
        }
    }
}

/// Centre half-pel j: horizontal taps into an i16 scratch, then a vertical
/// six-tap over the intermediates, `Clip1((t + 512) >> 10)`.
fn hpel_c_scalar(
    win: &[u8],
    stride: usize,
    bw: usize,
    bh: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
) {
    // Pass 1: unclipped horizontal taps for the bh+5 rows the vertical filter
    // reads (range ±10710 fits i16).
    let mut htmp = [0i16; 16 * 21];
    for k in 0..bh + 5 {
        let base = k * stride + 2;
        for c in 0..bw {
            let i = base + c;
            let t = win[i - 2] as i32 - 5 * win[i - 1] as i32
                + 20 * win[i] as i32
                + 20 * win[i + 1] as i32
                - 5 * win[i + 2] as i32
                + win[i + 3] as i32;
            htmp[k * 16 + c] = t as i16;
        }
    }
    // Pass 2: vertical six-tap over the intermediates in i32.
    for r in 0..bh {
        let d = dst_off + r * dst_stride;
        for c in 0..bw {
            let i = r * 16 + c;
            let t = htmp[i] as i32 - 5 * htmp[i + 16] as i32
                + 20 * htmp[i + 32] as i32
                + 20 * htmp[i + 48] as i32
                - 5 * htmp[i + 64] as i32
                + htmp[i + 80] as i32;
            dst[d + c] = ((t + 512) >> 10).clamp(0, 255) as u8;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn hpel_h_neon(
    win: &[u8],
    stride: usize,
    oy: usize,
    bw: usize,
    bh: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
) {
    use std::arch::aarch64::*;
    debug_assert!(dst_off + (bh - 1) * dst_stride + bw <= dst.len());
    for r in 0..bh {
        // Column c of the 16-byte load is sample x-2 for output column c.
        let src = win.as_ptr().add((oy + r) * stride);
        let d = dst.as_mut_ptr().add(dst_off + r * dst_stride);
        for c in (0..bw).step_by(8) {
            let v = vld1q_u8(src.add(c));
            let s0 = vget_low_u8(v);
            let s1 = vget_low_u8(vextq_u8::<1>(v, v));
            let s2 = vget_low_u8(vextq_u8::<2>(v, v));
            let s3 = vget_low_u8(vextq_u8::<3>(v, v));
            let s4 = vget_low_u8(vextq_u8::<4>(v, v));
            let s5 = vget_low_u8(vextq_u8::<5>(v, v));
            // t = (s0+s5) + 20*(s2+s3) - 5*(s1+s4); |t| ≤ 10710 fits i16.
            let a05 = vreinterpretq_s16_u16(vaddl_u8(s0, s5));
            let a14 = vreinterpretq_s16_u16(vaddl_u8(s1, s4));
            let a23 = vreinterpretq_s16_u16(vaddl_u8(s2, s3));
            let t = vmlsq_n_s16(vmlaq_n_s16(a05, a23, 20), a14, 5);
            // Rounding shift by 5 with unsigned saturation = Clip1((t+16)>>5).
            vst1_u8(d.add(c), vqrshrun_n_s16::<5>(t));
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn hpel_v_neon(
    win: &[u8],
    stride: usize,
    ox: usize,
    bw: usize,
    bh: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
) {
    use std::arch::aarch64::*;
    let s_ = stride;
    debug_assert!(dst_off + (bh - 1) * dst_stride + bw <= dst.len());
    for r in 0..bh {
        let base = win.as_ptr().add((2 + r) * s_ + ox);
        let d = dst.as_mut_ptr().add(dst_off + r * dst_stride);
        for c in (0..bw).step_by(8) {
            let p = base.add(c);
            let r0 = vld1_u8(p.sub(2 * s_));
            let r1 = vld1_u8(p.sub(s_));
            let r2 = vld1_u8(p);
            let r3 = vld1_u8(p.add(s_));
            let r4 = vld1_u8(p.add(2 * s_));
            let r5 = vld1_u8(p.add(3 * s_));
            let a05 = vreinterpretq_s16_u16(vaddl_u8(r0, r5));
            let a14 = vreinterpretq_s16_u16(vaddl_u8(r1, r4));
            let a23 = vreinterpretq_s16_u16(vaddl_u8(r2, r3));
            let t = vmlsq_n_s16(vmlaq_n_s16(a05, a23, 20), a14, 5);
            vst1_u8(d.add(c), vqrshrun_n_s16::<5>(t));
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn hpel_c_neon(
    win: &[u8],
    stride: usize,
    bw: usize,
    bh: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
) {
    use std::arch::aarch64::*;
    debug_assert!(dst_off + (bh - 1) * dst_stride + bw <= dst.len());
    // Pass 1: unclipped horizontal taps (i16) for the bh+5 rows pass 2 reads.
    let mut htmp = [0i16; 16 * 21];
    for k in 0..bh + 5 {
        let src = win.as_ptr().add(k * stride);
        let t_row = htmp.as_mut_ptr().add(k * 16);
        for c in (0..bw).step_by(8) {
            let v = vld1q_u8(src.add(c));
            let s0 = vget_low_u8(v);
            let s1 = vget_low_u8(vextq_u8::<1>(v, v));
            let s2 = vget_low_u8(vextq_u8::<2>(v, v));
            let s3 = vget_low_u8(vextq_u8::<3>(v, v));
            let s4 = vget_low_u8(vextq_u8::<4>(v, v));
            let s5 = vget_low_u8(vextq_u8::<5>(v, v));
            let a05 = vreinterpretq_s16_u16(vaddl_u8(s0, s5));
            let a14 = vreinterpretq_s16_u16(vaddl_u8(s1, s4));
            let a23 = vreinterpretq_s16_u16(vaddl_u8(s2, s3));
            vst1q_s16(t_row.add(c), vmlsq_n_s16(vmlaq_n_s16(a05, a23, 20), a14, 5));
        }
    }
    // Pass 2: vertical six-tap over the intermediates, widened to i32.
    for r in 0..bh {
        let d = dst.as_mut_ptr().add(dst_off + r * dst_stride);
        for c in (0..bw).step_by(8) {
            let p = htmp.as_ptr().add(r * 16 + c);
            let r0 = vld1q_s16(p);
            let r1 = vld1q_s16(p.add(16));
            let r2 = vld1q_s16(p.add(32));
            let r3 = vld1q_s16(p.add(48));
            let r4 = vld1q_s16(p.add(64));
            let r5 = vld1q_s16(p.add(80));
            let lo = {
                let a05 = vaddl_s16(vget_low_s16(r0), vget_low_s16(r5));
                let a14 = vaddl_s16(vget_low_s16(r1), vget_low_s16(r4));
                let a23 = vaddl_s16(vget_low_s16(r2), vget_low_s16(r3));
                vmlsq_n_s32(vmlaq_n_s32(a05, a23, 20), a14, 5)
            };
            let hi = {
                let a05 = vaddl_s16(vget_high_s16(r0), vget_high_s16(r5));
                let a14 = vaddl_s16(vget_high_s16(r1), vget_high_s16(r4));
                let a23 = vaddl_s16(vget_high_s16(r2), vget_high_s16(r3));
                vmlsq_n_s32(vmlaq_n_s32(a05, a23, 20), a14, 5)
            };
            // Rounding shift by 10, narrow s32→s16 (no overflow: |t| < 2^19),
            // then unsigned-saturating narrow = Clip1((t+512)>>10).
            let n = vcombine_s16(vqrshrn_n_s32::<10>(lo), vqrshrn_n_s32::<10>(hi));
            vst1_u8(d.add(c), vqmovun_s16(n));
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn avg_block_neon(
    a: &[u8],
    b: &[u8],
    bw: usize,
    bh: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
) {
    use std::arch::aarch64::*;
    debug_assert!(bw * bh <= a.len() && bw * bh <= b.len());
    debug_assert!(dst_off + (bh - 1) * dst_stride + bw <= dst.len());
    for r in 0..bh {
        let pa = a.as_ptr().add(r * bw);
        let pb = b.as_ptr().add(r * bw);
        let d = dst.as_mut_ptr().add(dst_off + r * dst_stride);
        let mut c = 0usize;
        while c + 16 <= bw {
            vst1q_u8(
                d.add(c),
                vrhaddq_u8(vld1q_u8(pa.add(c)), vld1q_u8(pb.add(c))),
            );
            c += 16;
        }
        while c + 8 <= bw {
            vst1_u8(d.add(c), vrhadd_u8(vld1_u8(pa.add(c)), vld1_u8(pb.add(c))));
            c += 8;
        }
    }
}

/// Sign-extends the low / high four i16 lanes to i32 (SSE2 has no `cvtepi16`).
#[cfg(target_arch = "x86_64")]
#[inline]
#[allow(unsafe_code)]
unsafe fn sse2_widen_s16(
    x: std::arch::x86_64::__m128i,
) -> (std::arch::x86_64::__m128i, std::arch::x86_64::__m128i) {
    use std::arch::x86_64::*;
    // SAFETY: SSE2 is part of the x86_64 baseline.
    unsafe {
        let zero = _mm_setzero_si128();
        let lo = _mm_srai_epi32(_mm_unpacklo_epi16(zero, x), 16);
        let hi = _mm_srai_epi32(_mm_unpackhi_epi16(zero, x), 16);
        (lo, hi)
    }
}

/// `(s0+s5) + 20*(s2+s3) - 5*(s1+s4)` over i16 lanes (fits i16: |t| ≤ 10710).
#[cfg(target_arch = "x86_64")]
#[inline]
#[allow(unsafe_code)]
unsafe fn sse2_tap6_epi16(s: [std::arch::x86_64::__m128i; 6]) -> std::arch::x86_64::__m128i {
    use std::arch::x86_64::*;
    // SAFETY: SSE2 is part of the x86_64 baseline.
    unsafe {
        let a05 = _mm_add_epi16(s[0], s[5]);
        let a14 = _mm_add_epi16(s[1], s[4]);
        let a23 = _mm_add_epi16(s[2], s[3]);
        _mm_add_epi16(
            a05,
            _mm_sub_epi16(
                _mm_mullo_epi16(a23, _mm_set1_epi16(20)),
                _mm_mullo_epi16(a14, _mm_set1_epi16(5)),
            ),
        )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn hpel_h_sse2(
    win: &[u8],
    stride: usize,
    oy: usize,
    bw: usize,
    bh: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
) {
    use std::arch::x86_64::*;
    debug_assert!(dst_off + (bh - 1) * dst_stride + bw <= dst.len());
    let zero = _mm_setzero_si128();
    for r in 0..bh {
        let src = win.as_ptr().add((oy + r) * stride);
        let d = dst.as_mut_ptr().add(dst_off + r * dst_stride);
        for c in (0..bw).step_by(8) {
            let v = _mm_loadu_si128(src.add(c) as *const __m128i);
            let s = [
                _mm_unpacklo_epi8(v, zero),
                _mm_unpacklo_epi8(_mm_srli_si128(v, 1), zero),
                _mm_unpacklo_epi8(_mm_srli_si128(v, 2), zero),
                _mm_unpacklo_epi8(_mm_srli_si128(v, 3), zero),
                _mm_unpacklo_epi8(_mm_srli_si128(v, 4), zero),
                _mm_unpacklo_epi8(_mm_srli_si128(v, 5), zero),
            ];
            let t = _mm_srai_epi16(_mm_add_epi16(sse2_tap6_epi16(s), _mm_set1_epi16(16)), 5);
            _mm_storel_epi64(d.add(c) as *mut __m128i, _mm_packus_epi16(t, t));
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn hpel_v_sse2(
    win: &[u8],
    stride: usize,
    ox: usize,
    bw: usize,
    bh: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
) {
    use std::arch::x86_64::*;
    let s_ = stride;
    debug_assert!(dst_off + (bh - 1) * dst_stride + bw <= dst.len());
    let zero = _mm_setzero_si128();
    for r in 0..bh {
        let base = win.as_ptr().add((2 + r) * s_ + ox);
        let d = dst.as_mut_ptr().add(dst_off + r * dst_stride);
        for c in (0..bw).step_by(8) {
            let p = base.add(c);
            let s = [
                _mm_unpacklo_epi8(_mm_loadl_epi64(p.sub(2 * s_) as *const __m128i), zero),
                _mm_unpacklo_epi8(_mm_loadl_epi64(p.sub(s_) as *const __m128i), zero),
                _mm_unpacklo_epi8(_mm_loadl_epi64(p as *const __m128i), zero),
                _mm_unpacklo_epi8(_mm_loadl_epi64(p.add(s_) as *const __m128i), zero),
                _mm_unpacklo_epi8(_mm_loadl_epi64(p.add(2 * s_) as *const __m128i), zero),
                _mm_unpacklo_epi8(_mm_loadl_epi64(p.add(3 * s_) as *const __m128i), zero),
            ];
            let t = _mm_srai_epi16(_mm_add_epi16(sse2_tap6_epi16(s), _mm_set1_epi16(16)), 5);
            _mm_storel_epi64(d.add(c) as *mut __m128i, _mm_packus_epi16(t, t));
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn hpel_c_sse2(
    win: &[u8],
    stride: usize,
    bw: usize,
    bh: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
) {
    use std::arch::x86_64::*;
    debug_assert!(dst_off + (bh - 1) * dst_stride + bw <= dst.len());
    let zero = _mm_setzero_si128();
    let mut htmp = [0i16; 16 * 21];
    for k in 0..bh + 5 {
        let src = win.as_ptr().add(k * stride);
        let t_row = htmp.as_mut_ptr().add(k * 16);
        for c in (0..bw).step_by(8) {
            let v = _mm_loadu_si128(src.add(c) as *const __m128i);
            let s = [
                _mm_unpacklo_epi8(v, zero),
                _mm_unpacklo_epi8(_mm_srli_si128(v, 1), zero),
                _mm_unpacklo_epi8(_mm_srli_si128(v, 2), zero),
                _mm_unpacklo_epi8(_mm_srli_si128(v, 3), zero),
                _mm_unpacklo_epi8(_mm_srli_si128(v, 4), zero),
                _mm_unpacklo_epi8(_mm_srli_si128(v, 5), zero),
            ];
            _mm_storeu_si128(t_row.add(c) as *mut __m128i, sse2_tap6_epi16(s));
        }
    }
    let c512 = _mm_set1_epi32(512);
    for r in 0..bh {
        let d = dst.as_mut_ptr().add(dst_off + r * dst_stride);
        for c in (0..bw).step_by(8) {
            let p = htmp.as_ptr().add(r * 16 + c);
            let rows = [
                _mm_loadu_si128(p as *const __m128i),
                _mm_loadu_si128(p.add(16) as *const __m128i),
                _mm_loadu_si128(p.add(32) as *const __m128i),
                _mm_loadu_si128(p.add(48) as *const __m128i),
                _mm_loadu_si128(p.add(64) as *const __m128i),
                _mm_loadu_si128(p.add(80) as *const __m128i),
            ];
            let mut lo_hi = [zero; 2];
            for (half, out) in lo_hi.iter_mut().enumerate() {
                let pick = |x| {
                    let (l, h) = sse2_widen_s16(x);
                    if half == 0 { l } else { h }
                };
                let a05 = _mm_add_epi32(pick(rows[0]), pick(rows[5]));
                let a14 = _mm_add_epi32(pick(rows[1]), pick(rows[4]));
                let a23 = _mm_add_epi32(pick(rows[2]), pick(rows[3]));
                // 20*x = (x<<4)+(x<<2); 5*x = (x<<2)+x (SSE2 has no mullo_epi32).
                let m20 = _mm_add_epi32(_mm_slli_epi32(a23, 4), _mm_slli_epi32(a23, 2));
                let m5 = _mm_add_epi32(_mm_slli_epi32(a14, 2), a14);
                let t = _mm_add_epi32(a05, _mm_sub_epi32(m20, m5));
                *out = _mm_srai_epi32(_mm_add_epi32(t, c512), 10);
            }
            let n = _mm_packs_epi32(lo_hi[0], lo_hi[1]);
            _mm_storel_epi64(d.add(c) as *mut __m128i, _mm_packus_epi16(n, n));
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn avg_block_sse2(
    a: &[u8],
    b: &[u8],
    bw: usize,
    bh: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
) {
    use std::arch::x86_64::*;
    debug_assert!(bw * bh <= a.len() && bw * bh <= b.len());
    debug_assert!(dst_off + (bh - 1) * dst_stride + bw <= dst.len());
    for r in 0..bh {
        let pa = a.as_ptr().add(r * bw);
        let pb = b.as_ptr().add(r * bw);
        let d = dst.as_mut_ptr().add(dst_off + r * dst_stride);
        let mut c = 0usize;
        while c + 16 <= bw {
            let v = _mm_avg_epu8(
                _mm_loadu_si128(pa.add(c) as *const __m128i),
                _mm_loadu_si128(pb.add(c) as *const __m128i),
            );
            _mm_storeu_si128(d.add(c) as *mut __m128i, v);
            c += 16;
        }
        while c + 8 <= bw {
            let v = _mm_avg_epu8(
                _mm_loadl_epi64(pa.add(c) as *const __m128i),
                _mm_loadl_epi64(pb.add(c) as *const __m128i),
            );
            _mm_storel_epi64(d.add(c) as *mut __m128i, v);
            c += 8;
        }
    }
}

/// Reorders `_mm256_packus_epi16(t, t)` output (per-128-lane packing) into 16
/// contiguous bytes in the low xmm.
#[cfg(target_arch = "x86_64")]
#[inline]
#[allow(unsafe_code)]
unsafe fn pack16_avx2(t: std::arch::x86_64::__m256i) -> std::arch::x86_64::__m128i {
    use std::arch::x86_64::*;
    // SAFETY: caller runs under the AVX2 target feature.
    unsafe {
        _mm256_castsi256_si128(_mm256_permute4x64_epi64(
            _mm256_packus_epi16(t, t),
            0b00_00_10_00,
        ))
    }
}

/// `(s0+s5) + 20*(s2+s3) - 5*(s1+s4)` over 16 u16 lanes from six shifted
/// 16-byte loads at `src` (the six-tap kernel, unclipped, fits i16).
#[cfg(target_arch = "x86_64")]
#[inline]
#[allow(unsafe_code)]
unsafe fn tap6_row_avx2(src: *const u8) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    // SAFETY: caller runs under the AVX2 target feature and guarantees
    // `src..src+21+15` stays inside the padded window row (stride 32).
    unsafe {
        let s = |n: usize| _mm256_cvtepu8_epi16(_mm_loadu_si128(src.add(n) as *const __m128i));
        let a05 = _mm256_add_epi16(s(0), s(5));
        let a14 = _mm256_add_epi16(s(1), s(4));
        let a23 = _mm256_add_epi16(s(2), s(3));
        _mm256_add_epi16(
            a05,
            _mm256_sub_epi16(
                _mm256_mullo_epi16(a23, _mm256_set1_epi16(20)),
                _mm256_mullo_epi16(a14, _mm256_set1_epi16(5)),
            ),
        )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn hpel_h_avx2(
    win: &[u8],
    stride: usize,
    oy: usize,
    bh: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
) {
    use std::arch::x86_64::*;
    debug_assert!(dst_off + (bh - 1) * dst_stride + 16 <= dst.len());
    for r in 0..bh {
        let t = tap6_row_avx2(win.as_ptr().add((oy + r) * stride));
        let t = _mm256_srai_epi16(_mm256_add_epi16(t, _mm256_set1_epi16(16)), 5);
        _mm_storeu_si128(
            dst.as_mut_ptr().add(dst_off + r * dst_stride) as *mut __m128i,
            pack16_avx2(t),
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn hpel_v_avx2(
    win: &[u8],
    stride: usize,
    ox: usize,
    bh: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
) {
    use std::arch::x86_64::*;
    let s_ = stride;
    debug_assert!(dst_off + (bh - 1) * dst_stride + 16 <= dst.len());
    for r in 0..bh {
        let p = win.as_ptr().add((2 + r) * s_ + ox);
        // SAFETY (closure): rows 0..bh+4 of the padded window are in bounds.
        let row = |off: isize| unsafe {
            _mm256_cvtepu8_epi16(_mm_loadu_si128(p.offset(off) as *const __m128i))
        };
        let a05 = _mm256_add_epi16(row(-2 * s_ as isize), row(3 * s_ as isize));
        let a14 = _mm256_add_epi16(row(-(s_ as isize)), row(2 * s_ as isize));
        let a23 = _mm256_add_epi16(row(0), row(s_ as isize));
        let t = _mm256_add_epi16(
            a05,
            _mm256_sub_epi16(
                _mm256_mullo_epi16(a23, _mm256_set1_epi16(20)),
                _mm256_mullo_epi16(a14, _mm256_set1_epi16(5)),
            ),
        );
        let t = _mm256_srai_epi16(_mm256_add_epi16(t, _mm256_set1_epi16(16)), 5);
        _mm_storeu_si128(
            dst.as_mut_ptr().add(dst_off + r * dst_stride) as *mut __m128i,
            pack16_avx2(t),
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn hpel_c_avx2(
    win: &[u8],
    stride: usize,
    bh: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
) {
    use std::arch::x86_64::*;
    debug_assert!(dst_off + (bh - 1) * dst_stride + 16 <= dst.len());
    // Pass 1: unclipped horizontal taps, one ymm (16 i16) per window row.
    let mut htmp = [0i16; 16 * 21];
    for k in 0..bh + 5 {
        let t = tap6_row_avx2(win.as_ptr().add(k * stride));
        _mm256_storeu_si256(htmp.as_mut_ptr().add(k * 16) as *mut __m256i, t);
    }
    // Pass 2: vertical six-tap in i32 (two ymm halves of 8 lanes each).
    let c512 = _mm256_set1_epi32(512);
    for r in 0..bh {
        let mut halves = [_mm256_setzero_si256(); 2];
        for (half, out) in halves.iter_mut().enumerate() {
            let p = htmp.as_ptr().add(r * 16 + half * 8);
            // SAFETY (closure): htmp rows r..r+5 (≤ bh+4) are in bounds.
            let row = |k: usize| unsafe {
                _mm256_cvtepi16_epi32(_mm_loadu_si128(p.add(k * 16) as *const __m128i))
            };
            let a05 = _mm256_add_epi32(row(0), row(5));
            let a14 = _mm256_add_epi32(row(1), row(4));
            let a23 = _mm256_add_epi32(row(2), row(3));
            let t = _mm256_add_epi32(
                a05,
                _mm256_sub_epi32(
                    _mm256_mullo_epi32(a23, _mm256_set1_epi32(20)),
                    _mm256_mullo_epi32(a14, _mm256_set1_epi32(5)),
                ),
            );
            *out = _mm256_srai_epi32(_mm256_add_epi32(t, c512), 10);
        }
        // i32×8 halves → i16×16 (per-lane pack + reorder) → u8×16.
        let n = _mm256_permute4x64_epi64(_mm256_packs_epi32(halves[0], halves[1]), 0b11_01_10_00);
        _mm_storeu_si128(
            dst.as_mut_ptr().add(dst_off + r * dst_stride) as *mut __m128i,
            pack16_avx2(n),
        );
    }
}

/// Spec-compliant luma motion compensation for a `bw`x`bh` block at output
/// position (dst_x, dst_y). The motion vector is in quarter-pel units and the
/// prediction is written into `output` (clause 8.4.2.2.1).
///
/// `reference` is the full padded plane (`padded_plane_geometry` layout with
/// replicated edges); `output` is an origin-based view of the destination
/// plane with row pitch `out_stride`.
#[allow(clippy::too_many_arguments)]
pub fn mc_luma(
    reference: &[u8],
    ref_origin: usize,
    ref_stride: usize,
    ref_width: usize,
    ref_height: usize,
    mvx: i32,
    mvy: i32,
    dst_x: usize,
    dst_y: usize,
    bw: usize,
    bh: usize,
    output: &mut [u8],
    out_stride: usize,
) {
    // The block variant samples the reference at (dst_x, dst_y) and writes to
    // index 0 of the buffer it is given, so a subslice at the destination offset
    // reproduces the in-place write exactly.
    let dst_off = dst_y * out_stride + dst_x;
    mc_luma_block(
        reference,
        ref_origin,
        ref_stride,
        ref_width,
        ref_height,
        mvx,
        mvy,
        dst_x,
        dst_y,
        bw,
        bh,
        &mut output[dst_off..],
        out_stride,
    );
}

/// Motion-compensates one luma block into a compact `out` buffer whose top-left
/// sample is at index 0 (row pitch `out_stride`), while sampling the reference
/// at the frame position (`sample_x`, `sample_y`). Used for B-slice
/// bi-prediction, where each list's prediction is formed separately and then
/// combined (clause 8.4.2.3).
#[allow(clippy::too_many_arguments)]
pub fn mc_luma_block(
    reference: &[u8],
    ref_origin: usize,
    ref_stride: usize,
    ref_width: usize,
    ref_height: usize,
    mvx: i32,
    mvy: i32,
    sample_x: usize,
    sample_y: usize,
    bw: usize,
    bh: usize,
    out: &mut [u8],
    out_stride: usize,
) {
    // Two's-complement `& 3` / arithmetic `>> 2` give the correct floor split for
    // negative quarter-pel vectors (e.g. -1 → int -1, frac 3).
    let xf = mvx & 3;
    let yf = mvy & 3;
    // Once every tap column (row) of the block reads the same replicated edge
    // sample, moving the block further out cannot change the prediction — the
    // taps span ix0-2 ..= ix0+bw+2, so any base beyond these bounds is
    // equivalent to the bound itself and the padding ring covers every MV.
    let ix0 = (sample_x as i32 + (mvx >> 2)).clamp(-(bw as i32) - 2, ref_width as i32 + 1);
    let iy0 = (sample_y as i32 + (mvy >> 2)).clamp(-(bh as i32) - 2, ref_height as i32 + 1);

    // Kernel source tile whose (2, 2) origin is the block's integer position.
    let src_off =
        (ref_origin as i64 + (iy0 as i64 - 2) * ref_stride as i64 + ix0 as i64 - 2) as usize;
    // Deepest access: tile row bh+4, then a 16-byte SIMD row load tail.
    debug_assert!(src_off + (bh + 4) * ref_stride + bw + 8 <= reference.len());
    let src = &reference[src_off..];

    // Integer-pel fast path: plain row copies out of the padded plane.
    if xf == 0 && yf == 0 {
        copy_block(
            src,
            2 * ref_stride + 2,
            ref_stride,
            bw,
            bh,
            out,
            0,
            out_stride,
        );
        return;
    }

    match luma_ops(xf, yf) {
        (op, None) => run_luma_op(op, src, ref_stride, bw, bh, out, 0, out_stride),
        (op1, Some(op2)) => {
            let mut t1 = [0u8; 256];
            let mut t2 = [0u8; 256];
            run_luma_op(op1, src, ref_stride, bw, bh, &mut t1, 0, bw);
            run_luma_op(op2, src, ref_stride, bw, bh, &mut t2, 0, bw);
            avg_block(&t1, &t2, bw, bh, out, 0, out_stride);
        }
    }
}

/// Bilinear chroma tap weights for an eighth-pel fraction: A=(8-xf)(8-yf),
/// B=xf(8-yf), C=(8-xf)yf, D=xf*yf (all ≤ 64).
#[inline]
fn chroma_weights(xf: i32, yf: i32) -> [u8; 4] {
    [
        ((8 - xf) * (8 - yf)) as u8,
        (xf * (8 - yf)) as u8,
        ((8 - xf) * yf) as u8,
        (xf * yf) as u8,
    ]
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn mc_chroma_row8_neon(
    win: &[u8],
    stride: usize,
    w: [u8; 4],
    bh: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
) {
    use std::arch::aarch64::*;
    debug_assert!(dst_off + (bh - 1) * dst_stride + 8 <= dst.len());
    let (wa, wb, wc, wd) = (
        vdup_n_u8(w[0]),
        vdup_n_u8(w[1]),
        vdup_n_u8(w[2]),
        vdup_n_u8(w[3]),
    );
    for r in 0..bh {
        let p = win.as_ptr().add(r * stride);
        let v0 = vld1q_u8(p);
        let v1 = vld1q_u8(p.add(stride));
        let a = vget_low_u8(v0);
        let b = vget_low_u8(vextq_u8::<1>(v0, v0));
        let c = vget_low_u8(v1);
        let d = vget_low_u8(vextq_u8::<1>(v1, v1));
        // Weights sum to 64, so the u16 accumulator peaks at 16320 + 32.
        let mut acc = vmull_u8(a, wa);
        acc = vmlal_u8(acc, b, wb);
        acc = vmlal_u8(acc, c, wc);
        acc = vmlal_u8(acc, d, wd);
        vst1_u8(
            dst.as_mut_ptr().add(dst_off + r * dst_stride),
            vrshrn_n_u16::<6>(acc),
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn mc_chroma_row8_sse2(
    win: &[u8],
    stride: usize,
    w: [u8; 4],
    bh: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
) {
    use std::arch::x86_64::*;
    debug_assert!(dst_off + (bh - 1) * dst_stride + 8 <= dst.len());
    let zero = _mm_setzero_si128();
    let wa = _mm_set1_epi16(w[0] as i16);
    let wb = _mm_set1_epi16(w[1] as i16);
    let wc = _mm_set1_epi16(w[2] as i16);
    let wd = _mm_set1_epi16(w[3] as i16);
    let c32 = _mm_set1_epi16(32);
    for r in 0..bh {
        let p = win.as_ptr().add(r * stride);
        let v0 = _mm_loadu_si128(p as *const __m128i);
        let v1 = _mm_loadu_si128(p.add(stride) as *const __m128i);
        let a = _mm_unpacklo_epi8(v0, zero);
        let b = _mm_unpacklo_epi8(_mm_srli_si128(v0, 1), zero);
        let c = _mm_unpacklo_epi8(v1, zero);
        let d = _mm_unpacklo_epi8(_mm_srli_si128(v1, 1), zero);
        let acc = _mm_add_epi16(
            _mm_add_epi16(_mm_mullo_epi16(a, wa), _mm_mullo_epi16(b, wb)),
            _mm_add_epi16(_mm_mullo_epi16(c, wc), _mm_mullo_epi16(d, wd)),
        );
        // Max 16320 + 32 stays positive in i16, so a logical shift is exact.
        let t = _mm_srli_epi16(_mm_add_epi16(acc, c32), 6);
        _mm_storel_epi64(
            dst.as_mut_ptr().add(dst_off + r * dst_stride) as *mut __m128i,
            _mm_packus_epi16(t, t),
        );
    }
}

/// NEON bilinear kernel for 4-wide chroma blocks (the chroma of 8x8/8x16/16x8
/// luma partitions): full 8-lane math per row, low four lanes stored. AVX2
/// adds nothing here — four output samples fill half an xmm of u16 lanes.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn mc_chroma_row4_neon(
    win: &[u8],
    stride: usize,
    w: [u8; 4],
    bh: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
) {
    use std::arch::aarch64::*;
    debug_assert!(dst_off + (bh - 1) * dst_stride + 4 <= dst.len());
    let (wa, wb, wc, wd) = (
        vdup_n_u8(w[0]),
        vdup_n_u8(w[1]),
        vdup_n_u8(w[2]),
        vdup_n_u8(w[3]),
    );
    for r in 0..bh {
        let p = win.as_ptr().add(r * stride);
        let v0 = vld1_u8(p);
        let v1 = vld1_u8(p.add(stride));
        let mut acc = vmull_u8(v0, wa);
        acc = vmlal_u8(acc, vext_u8::<1>(v0, v0), wb);
        acc = vmlal_u8(acc, v1, wc);
        acc = vmlal_u8(acc, vext_u8::<1>(v1, v1), wd);
        let n = vrshrn_n_u16::<6>(acc);
        let out = vget_lane_u32::<0>(vreinterpret_u32_u8(n));
        (dst.as_mut_ptr().add(dst_off + r * dst_stride) as *mut u32).write_unaligned(out);
    }
}

/// SSE2 analogue of [`mc_chroma_row4_neon`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn mc_chroma_row4_sse2(
    win: &[u8],
    stride: usize,
    w: [u8; 4],
    bh: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
) {
    use std::arch::x86_64::*;
    debug_assert!(dst_off + (bh - 1) * dst_stride + 4 <= dst.len());
    let zero = _mm_setzero_si128();
    let wa = _mm_set1_epi16(w[0] as i16);
    let wb = _mm_set1_epi16(w[1] as i16);
    let wc = _mm_set1_epi16(w[2] as i16);
    let wd = _mm_set1_epi16(w[3] as i16);
    let c32 = _mm_set1_epi16(32);
    for r in 0..bh {
        let p = win.as_ptr().add(r * stride);
        let v0 = _mm_loadl_epi64(p as *const __m128i);
        let v1 = _mm_loadl_epi64(p.add(stride) as *const __m128i);
        let a = _mm_unpacklo_epi8(v0, zero);
        let b = _mm_unpacklo_epi8(_mm_srli_si128(v0, 1), zero);
        let c = _mm_unpacklo_epi8(v1, zero);
        let d = _mm_unpacklo_epi8(_mm_srli_si128(v1, 1), zero);
        let acc = _mm_add_epi16(
            _mm_add_epi16(_mm_mullo_epi16(a, wa), _mm_mullo_epi16(b, wb)),
            _mm_add_epi16(_mm_mullo_epi16(c, wc), _mm_mullo_epi16(d, wd)),
        );
        let t = _mm_srli_epi16(_mm_add_epi16(acc, c32), 6);
        let out = _mm_cvtsi128_si32(_mm_packus_epi16(t, t));
        (dst.as_mut_ptr().add(dst_off + r * dst_stride) as *mut i32).write_unaligned(out);
    }
}

/// Spec-compliant chroma motion compensation (4:2:0) for a `bw`x`bh` block
/// (clause 8.4.2.2.2). The luma-domain quarter-pel MV maps to eighth-pel in the
/// half-resolution chroma plane; samples use bilinear interpolation.
///
/// Plane layout as in [`mc_luma`]: `reference` is the full padded chroma
/// plane, `output` an origin-based view with row pitch `out_stride`.
#[allow(clippy::too_many_arguments)]
pub fn mc_chroma(
    reference: &[u8],
    ref_origin: usize,
    ref_stride: usize,
    ref_width: usize,
    ref_height: usize,
    mvx: i32,
    mvy: i32,
    dst_x: usize,
    dst_y: usize,
    bw: usize,
    bh: usize,
    output: &mut [u8],
    out_stride: usize,
) {
    let dst_off = dst_y * out_stride + dst_x;
    mc_chroma_block(
        reference,
        ref_origin,
        ref_stride,
        ref_width,
        ref_height,
        mvx,
        mvy,
        dst_x,
        dst_y,
        bw,
        bh,
        &mut output[dst_off..],
        out_stride,
    );
}

/// Motion-compensates one chroma block into a compact `out` buffer whose
/// top-left sample is at index 0, sampling the reference at frame position
/// (`sample_x`, `sample_y`). Chroma sibling of [`mc_luma_block`] for B-slice
/// bi-prediction.
#[allow(clippy::too_many_arguments)]
#[allow(unsafe_code)]
pub fn mc_chroma_block(
    reference: &[u8],
    ref_origin: usize,
    ref_stride: usize,
    ref_width: usize,
    ref_height: usize,
    mvx: i32,
    mvy: i32,
    sample_x: usize,
    sample_y: usize,
    bw: usize,
    bh: usize,
    out: &mut [u8],
    out_stride: usize,
) {
    let output = out;
    let dst_off = 0usize;
    // Chroma MV is the luma MV; in the half-res plane it is eighth-pel.
    let xf = mvx & 7;
    let yf = mvy & 7;
    // The bilinear taps span ix0 ..= ix0+bw; as in `mc_luma`, a base beyond
    // the all-replicated bounds is equivalent to the bound itself.
    let ix0 = (sample_x as i32 + (mvx >> 3)).clamp(-(bw as i32) - 1, ref_width as i32 - 1);
    let iy0 = (sample_y as i32 + (mvy >> 3)).clamp(-(bh as i32) - 1, ref_height as i32 - 1);

    let src_off = (ref_origin as i64 + iy0 as i64 * ref_stride as i64 + ix0 as i64) as usize;
    // Deepest access: tap row bh, then a 16-byte SIMD row load.
    debug_assert!(src_off + bh * ref_stride + 16 <= reference.len());
    let src = &reference[src_off..];

    if xf == 0 && yf == 0 {
        copy_block(src, 0, ref_stride, bw, bh, output, dst_off, out_stride);
        return;
    }

    let w = chroma_weights(xf, yf);

    #[cfg(target_arch = "aarch64")]
    if bw == 8 && yscv_cpu::host_cpu().features.neon {
        // SAFETY: NEON detected at runtime; the padding ring allows full
        // 16-byte loads and the destination block lies inside `output`.
        unsafe {
            mc_chroma_row8_neon(src, ref_stride, w, bh, output, dst_off, out_stride);
        }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if bw == 8 && yscv_cpu::host_cpu().features.sse2 {
        // SAFETY: SSE2 detected at runtime; bounds as above.
        unsafe {
            mc_chroma_row8_sse2(src, ref_stride, w, bh, output, dst_off, out_stride);
        }
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if bw == 4 && yscv_cpu::host_cpu().features.neon {
        // SAFETY: NEON detected at runtime; the 8-byte tap loads stay inside
        // the padding ring and exactly four bytes are stored per row.
        unsafe {
            mc_chroma_row4_neon(src, ref_stride, w, bh, output, dst_off, out_stride);
        }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if bw == 4 && yscv_cpu::host_cpu().features.sse2 {
        // SAFETY: SSE2 detected at runtime; bounds as above.
        unsafe {
            mc_chroma_row4_sse2(src, ref_stride, w, bh, output, dst_off, out_stride);
        }
        return;
    }

    let (wa, wb, wc, wd) = (w[0] as u32, w[1] as u32, w[2] as u32, w[3] as u32);
    for r in 0..bh {
        let row0 = r * ref_stride;
        let row1 = row0 + ref_stride;
        let d = dst_off + r * out_stride;
        for c in 0..bw {
            let v = wa * src[row0 + c] as u32
                + wb * src[row0 + c + 1] as u32
                + wc * src[row1 + c] as u32
                + wd * src[row1 + c + 1] as u32;
            output[d + c] = ((v + 32) >> 6) as u8;
        }
    }
}

// ---------------------------------------------------------------------------
// Weighted prediction (H.264 7.4.3.2)
// ---------------------------------------------------------------------------

/// Apply uni-directional weighted prediction to a 16x16 block in-place.
///
/// For each sample: `out = clip((log2_denom == 0 ? 0 : (1 << (log2_denom-1))) + weight * pred + (offset << log2_denom)) >> log2_denom`
/// Simplified: `out = clip(((weight * pred + round) >> log2_denom) + offset)`
#[allow(clippy::too_many_arguments)]
pub fn apply_weighted_pred(
    plane: &mut [u8],
    stride: usize,
    bx: usize,
    by: usize,
    block_w: usize,
    block_h: usize,
    weight: i32,
    offset: i32,
    log2_denom: u32,
) {
    let round = if log2_denom > 0 {
        1i32 << (log2_denom - 1)
    } else {
        0
    };
    for row in 0..block_h {
        for col in 0..block_w {
            let idx = (by + row) * stride + bx + col;
            if idx < plane.len() {
                let pred = plane[idx] as i32;
                let val = ((weight * pred + round) >> log2_denom) + offset;
                plane[idx] = val.clamp(0, 255) as u8;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// P-slice macroblock decoder
// ---------------------------------------------------------------------------

/// Decode a P-slice macroblock: parse mb_type, motion vectors, and apply
/// motion compensation from the reference frame.
///
/// Returns the decoded motion vector so the caller can store it for
/// neighboring-block prediction of subsequent macroblocks.
#[allow(clippy::too_many_arguments)]
pub fn decode_p_macroblock(
    reader: &mut BitstreamReader,
    reference_frame: &[u8],
    ref_width: usize,
    ref_height: usize,
    mb_x: usize,
    mb_y: usize,
    neighbor_mvs: &[MotionVector],
    output: &mut [u8],
    out_width: usize,
) -> Result<MotionVector, VideoError> {
    // 1. Parse mb_type (P_L0_16x16 = 0 for simplest case)
    let _mb_type = reader.read_ue()?;

    // 2. Parse motion vector difference
    let (mvd_x, mvd_y) = parse_mvd(reader)?;

    // 3. Predict MV from neighbors (left, top, top-right)
    let predicted = predict_mv(
        neighbor_mvs.first().copied().unwrap_or_default(),
        neighbor_mvs.get(1).copied().unwrap_or_default(),
        neighbor_mvs.get(2).copied().unwrap_or_default(),
    );

    // 4. Final MV = predicted + difference
    let mv = MotionVector {
        dx: predicted.dx + mvd_x,
        dy: predicted.dy + mvd_y,
        ref_idx: 0,
    };

    // 5. Motion compensate (integer-pel for P_L0_16x16)
    motion_compensate_16x16(
        reference_frame,
        ref_width,
        ref_height,
        ref_width,
        3,
        mv,
        mb_x,
        mb_y,
        output,
        out_width,
    );

    Ok(mv)
}

// ---------------------------------------------------------------------------
// Reference frame buffer
// ---------------------------------------------------------------------------

/// Simple reference frame buffer for P-slice decoding.
///
/// Maintains a bounded FIFO of recent reconstructed frames so that P-slices
/// can reference them for motion compensation.
pub struct ReferenceFrameBuffer {
    frames: Vec<Vec<u8>>,
    max_refs: usize,
}

impl ReferenceFrameBuffer {
    /// Creates a new buffer that keeps at most `max_refs` reference frames.
    pub fn new(max_refs: usize) -> Self {
        Self {
            frames: Vec::new(),
            max_refs,
        }
    }

    /// Pushes a reconstructed frame into the buffer, evicting the oldest if
    /// the capacity is exceeded.
    pub fn push(&mut self, frame: Vec<u8>) {
        if self.frames.len() >= self.max_refs {
            self.frames.remove(0);
        }
        self.frames.push(frame);
    }

    /// Returns the reference frame at `idx` (0 = oldest retained frame).
    pub fn get(&self, idx: usize) -> Option<&[u8]> {
        self.frames.get(idx).map(|v| v.as_slice())
    }

    /// Returns the most recently pushed reference frame.
    pub fn latest(&self) -> Option<&[u8]> {
        self.frames.last().map(|v| v.as_slice())
    }

    /// Returns the number of reference frames currently stored.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Returns `true` if the buffer contains no reference frames.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Test helpers (same convention as h264_decoder.rs) -------------------

    fn push_exp_golomb(bits: &mut Vec<u8>, value: u32) {
        if value == 0 {
            bits.push(1);
            return;
        }
        let code = value + 1;
        let bit_len = 32 - code.leading_zeros();
        let leading_zeros = bit_len - 1;
        for _ in 0..leading_zeros {
            bits.push(0);
        }
        for i in (0..bit_len).rev() {
            bits.push(((code >> i) & 1) as u8);
        }
    }

    fn push_signed_exp_golomb(bits: &mut Vec<u8>, value: i32) {
        let code = if value > 0 {
            (2 * value - 1) as u32
        } else if value < 0 {
            (2 * (-value)) as u32
        } else {
            0
        };
        push_exp_golomb(bits, code);
    }

    fn bits_to_bytes(bits: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for chunk in bits.chunks(8) {
            let mut byte = 0u8;
            for (i, &bit) in chunk.iter().enumerate() {
                byte |= bit << (7 - i);
            }
            bytes.push(byte);
        }
        bytes
    }

    // -- 1. Median prediction -----------------------------------------------

    #[test]
    fn motion_vector_median_prediction() {
        let left = MotionVector {
            dx: 2,
            dy: -4,
            ref_idx: 0,
        };
        let top = MotionVector {
            dx: 6,
            dy: 1,
            ref_idx: 0,
        };
        let top_right = MotionVector {
            dx: -3,
            dy: 8,
            ref_idx: 0,
        };

        let pred = predict_mv(left, top, top_right);
        // median(2, 6, -3) = 2; median(-4, 1, 8) = 1
        assert_eq!(pred.dx, 2);
        assert_eq!(pred.dy, 1);

        // All-zero neighbors
        let zero = MotionVector::default();
        let pred_zero = predict_mv(zero, zero, zero);
        assert_eq!(pred_zero.dx, 0);
        assert_eq!(pred_zero.dy, 0);

        // Two equal, one different
        let a = MotionVector {
            dx: 5,
            dy: 5,
            ref_idx: 0,
        };
        let b = MotionVector {
            dx: 5,
            dy: 5,
            ref_idx: 0,
        };
        let c = MotionVector {
            dx: -10,
            dy: 20,
            ref_idx: 0,
        };
        let pred2 = predict_mv(a, b, c);
        assert_eq!(pred2.dx, 5);
        assert_eq!(pred2.dy, 5);
    }

    // -- 2. Motion compensation block copy ----------------------------------

    #[test]
    fn motion_compensate_copies_block() {
        // 32x32 reference, 1 channel, filled with row index as pixel value
        let ref_w = 32;
        let ref_h = 32;
        let channels = 1;
        let mut reference = vec![0u8; ref_w * ref_h * channels];
        for row in 0..ref_h {
            for col in 0..ref_w {
                reference[row * ref_w + col] = row as u8;
            }
        }

        // MB at (0, 0) with mv=(0, 0) should copy top-left 16x16
        let mut output = vec![0u8; ref_w * ref_h * channels];
        let mv = MotionVector {
            dx: 0,
            dy: 0,
            ref_idx: 0,
        };
        motion_compensate_16x16(
            &reference,
            ref_w,
            ref_h,
            ref_w,
            channels,
            mv,
            0,
            0,
            &mut output,
            ref_w,
        );

        for row in 0..16 {
            for col in 0..16 {
                assert_eq!(
                    output[row * ref_w + col],
                    row as u8,
                    "mismatch at ({row}, {col})"
                );
            }
        }

        // MB at (0, 0) with mv=(4, 2) should copy from (4, 2)
        let mut output2 = vec![0u8; ref_w * ref_h * channels];
        let mv2 = MotionVector {
            dx: 4,
            dy: 2,
            ref_idx: 0,
        };
        motion_compensate_16x16(
            &reference,
            ref_w,
            ref_h,
            ref_w,
            channels,
            mv2,
            0,
            0,
            &mut output2,
            ref_w,
        );

        for row in 0..16 {
            let expected_src_y = (row as i32 + 2).clamp(0, ref_h as i32 - 1) as u8;
            for col in 0..16 {
                assert_eq!(
                    output2[row * ref_w + col],
                    expected_src_y,
                    "offset mismatch at ({row}, {col})"
                );
            }
        }
    }

    // -- 3. Reference frame buffer FIFO -------------------------------------

    #[test]
    fn reference_frame_buffer_fifo() {
        let mut buf = ReferenceFrameBuffer::new(3);
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
        assert!(buf.latest().is_none());

        buf.push(vec![1, 2, 3]);
        buf.push(vec![4, 5, 6]);
        buf.push(vec![7, 8, 9]);
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.get(0), Some([1u8, 2, 3].as_slice()));
        assert_eq!(buf.get(1), Some([4u8, 5, 6].as_slice()));
        assert_eq!(buf.get(2), Some([7u8, 8, 9].as_slice()));
        assert_eq!(buf.latest(), Some([7u8, 8, 9].as_slice()));

        // Pushing a 4th frame evicts the oldest
        buf.push(vec![10, 11, 12]);
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.get(0), Some([4u8, 5, 6].as_slice()));
        assert_eq!(buf.latest(), Some([10u8, 11, 12].as_slice()));
        assert!(buf.get(3).is_none());
    }

    // -- 4. parse_mvd roundtrip ---------------------------------------------

    #[test]
    fn parse_mvd_roundtrip() {
        // Encode se(3) and se(-5) into a bitstream, then parse them back.
        let mut bits = Vec::new();
        push_signed_exp_golomb(&mut bits, 3);
        push_signed_exp_golomb(&mut bits, -5);
        // Pad to byte boundary
        while bits.len() % 8 != 0 {
            bits.push(0);
        }
        let bytes = bits_to_bytes(&bits);

        let mut reader = BitstreamReader::new(&bytes);
        let (mvd_x, mvd_y) = parse_mvd(&mut reader).unwrap();
        assert_eq!(mvd_x, 3);
        assert_eq!(mvd_y, -5);

        // Zero MVD
        let mut bits2 = Vec::new();
        push_signed_exp_golomb(&mut bits2, 0);
        push_signed_exp_golomb(&mut bits2, 0);
        while bits2.len() % 8 != 0 {
            bits2.push(0);
        }
        let bytes2 = bits_to_bytes(&bits2);

        let mut reader2 = BitstreamReader::new(&bytes2);
        let (mvd_x2, mvd_y2) = parse_mvd(&mut reader2).unwrap();
        assert_eq!(mvd_x2, 0);
        assert_eq!(mvd_y2, 0);
    }
}
