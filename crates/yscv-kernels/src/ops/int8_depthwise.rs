//! INT8 depthwise Conv2D kernels for quantized tracker chains.
//!
//! Layout: NHWC input, KHWC depthwise weights with depth_multiplier=1,
//! int32 output in NHWC. The kernel intentionally stops at accumulation:
//! bias/requant/residual/activation epilogues are chain-level policy and
//! are fused by the caller that owns the quantization scales.

#![allow(unsafe_code, unsafe_op_in_unsafe_fn)]

use rayon::ThreadPool;
use rayon::prelude::*;

/// Parameters for the INT8 3x3 depthwise kernels (shapes, strides, zero-points, scales).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Depthwise3x3I8Params {
    pub batch: usize,
    pub in_h: usize,
    pub in_w: usize,
    pub channels: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub pad_top: usize,
    pub pad_left: usize,
    pub out_h: usize,
    pub out_w: usize,
}

impl Depthwise3x3I8Params {
    #[inline]
    pub fn input_len(self) -> usize {
        DepthwiseI8Params::from(self).input_len()
    }

    #[inline]
    pub fn output_len(self) -> usize {
        DepthwiseI8Params::from(self).output_len()
    }

    #[inline]
    pub fn weight_len(self) -> usize {
        DepthwiseI8Params::from(self).weight_len()
    }
}

/// Largest square kernel the SIMD NHWC depthwise path keeps: beyond 49 taps it
/// falls back to the scalar kernel, so routing there buys nothing.
pub const DEPTHWISE_I8_MAX_KERNEL: usize = 7;

/// Parameters for the generic INT8 depthwise kernels (kernel size, shapes, quant params).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DepthwiseI8Params {
    pub batch: usize,
    pub in_h: usize,
    pub in_w: usize,
    pub channels: usize,
    pub kernel: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub pad_top: usize,
    pub pad_left: usize,
    pub out_h: usize,
    pub out_w: usize,
}

impl From<Depthwise3x3I8Params> for DepthwiseI8Params {
    #[inline]
    fn from(p: Depthwise3x3I8Params) -> Self {
        Self {
            batch: p.batch,
            in_h: p.in_h,
            in_w: p.in_w,
            channels: p.channels,
            kernel: 3,
            stride_h: p.stride_h,
            stride_w: p.stride_w,
            pad_top: p.pad_top,
            pad_left: p.pad_left,
            out_h: p.out_h,
            out_w: p.out_w,
        }
    }
}

impl DepthwiseI8Params {
    #[inline]
    fn input_len(self) -> usize {
        self.batch * self.in_h * self.in_w * self.channels
    }

    #[inline]
    fn output_len(self) -> usize {
        self.batch * self.out_h * self.out_w * self.channels
    }

    #[inline]
    fn weight_len(self) -> usize {
        self.kernel * self.kernel * self.channels
    }

    #[inline]
    fn input_offset(self, n: usize, y: usize, x: usize, c: usize) -> usize {
        ((n * self.in_h + y) * self.in_w + x) * self.channels + c
    }

    #[inline]
    fn weight_offset(self, ky: usize, kx: usize, c: usize) -> usize {
        (ky * self.kernel + kx) * self.channels + c
    }

    #[inline]
    fn valid_input_y(self, oh: usize, ky: usize) -> Option<usize> {
        let y = oh * self.stride_h + ky;
        if y >= self.pad_top && y < self.pad_top + self.in_h {
            Some(y - self.pad_top)
        } else {
            None
        }
    }

    #[inline]
    fn valid_input_x(self, ow: usize, kx: usize) -> Option<usize> {
        let x = ow * self.stride_w + kx;
        if x >= self.pad_left && x < self.pad_left + self.in_w {
            Some(x - self.pad_left)
        } else {
            None
        }
    }
}

#[inline]
fn validate_depthwise(input: &[i8], weight: &[i8], p: DepthwiseI8Params, out: &[i32]) {
    debug_assert_eq!(input.len(), p.input_len());
    debug_assert_eq!(weight.len(), p.weight_len());
    debug_assert_eq!(out.len(), p.output_len());
}

/// Scalar reference INT8 NHWC depthwise convolution accumulating into i32.
pub fn depthwise_i8_i32_nhwc_scalar(
    input: &[i8],
    weight: &[i8],
    p: DepthwiseI8Params,
    out: &mut [i32],
) {
    validate_depthwise(input, weight, p, out);
    depthwise_i8_i32_nhwc_scalar_range(input, weight, p, out, 0);
}

fn depthwise_i8_i32_nhwc_scalar_range(
    input: &[i8],
    weight: &[i8],
    p: DepthwiseI8Params,
    out: &mut [i32],
    pixel_start: usize,
) {
    let total_pixels = p.batch * p.out_h * p.out_w;
    let pixel_end = pixel_start + out.len() / p.channels;
    for n in 0..p.batch {
        for oh in 0..p.out_h {
            for ow in 0..p.out_w {
                let pixel = (n * p.out_h + oh) * p.out_w + ow;
                if pixel < pixel_start || pixel >= pixel_end || pixel >= total_pixels {
                    continue;
                }
                let out_base = (pixel - pixel_start) * p.channels;
                scalar_pixel_tail(
                    input,
                    weight,
                    p,
                    Pixel {
                        n,
                        oh,
                        ow,
                        out_base,
                    },
                    0,
                    p.channels,
                    out,
                );
            }
        }
    }
}

/// Depthwise int8 convolution for NCHW activations with KHWC-packed
/// depthwise weights. Output is NCHW. This is intentionally scalar: it
/// targets large stride-2 tracker layers where avoiding NCHW→NHWC layout
/// materialisation beats the existing NHWC SIMD path.
pub fn depthwise_i8_i32_nchw_khwc_scalar(
    input: &[i8],
    weight: &[i8],
    p: DepthwiseI8Params,
    out: &mut [i32],
) {
    validate_depthwise(input, weight, p, out);
    for n in 0..p.batch {
        for c in 0..p.channels {
            for oh in 0..p.out_h {
                for ow in 0..p.out_w {
                    let mut acc = 0_i32;
                    for ky in 0..p.kernel {
                        let Some(iy) = p.valid_input_y(oh, ky) else {
                            continue;
                        };
                        for kx in 0..p.kernel {
                            let Some(ix) = p.valid_input_x(ow, kx) else {
                                continue;
                            };
                            let x_idx = ((n * p.channels + c) * p.in_h + iy) * p.in_w + ix;
                            acc +=
                                (input[x_idx] as i32) * (weight[p.weight_offset(ky, kx, c)] as i32);
                        }
                    }
                    let dst = ((n * p.channels + c) * p.out_h + oh) * p.out_w + ow;
                    out[dst] = acc;
                }
            }
        }
    }
}

/// INT8 depthwise dispatch for NCHW activations with KHWC-packed weights.
pub fn depthwise_i8_i32_nchw_khwc_dispatch(
    input: &[i8],
    weight: &[i8],
    p: DepthwiseI8Params,
    out: &mut [i32],
) {
    depthwise_i8_i32_nchw_khwc_scalar(input, weight, p, out);
}

/// Scalar INT8 NHWC 3x3 depthwise convolution accumulating into i32.
pub fn depthwise3x3_i8_i32_nhwc_scalar(
    input: &[i8],
    weight: &[i8],
    p: Depthwise3x3I8Params,
    out: &mut [i32],
) {
    depthwise_i8_i32_nhwc_scalar(input, weight, p.into(), out);
}

#[derive(Clone, Copy)]
struct Pixel {
    n: usize,
    oh: usize,
    ow: usize,
    out_base: usize,
}

fn scalar_pixel_tail(
    input: &[i8],
    weight: &[i8],
    p: DepthwiseI8Params,
    px: Pixel,
    c_start: usize,
    c_end: usize,
    out: &mut [i32],
) {
    for c in c_start..c_end {
        let mut acc = 0_i32;
        for ky in 0..p.kernel {
            let Some(iy) = p.valid_input_y(px.oh, ky) else {
                continue;
            };
            for kx in 0..p.kernel {
                let Some(ix) = p.valid_input_x(px.ow, kx) else {
                    continue;
                };
                acc += (input[p.input_offset(px.n, iy, ix, c)] as i32)
                    * (weight[p.weight_offset(ky, kx, c)] as i32);
            }
        }
        out[px.out_base + c] = acc;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn depthwise3x3_i8_i32_nhwc_avx2_range(
    input: &[i8],
    weight: &[i8],
    p: DepthwiseI8Params,
    out: &mut [i32],
    pixel_start: usize,
) {
    use std::arch::x86_64::*;
    let c8 = p.channels & !7;
    let total_pixels = p.batch * p.out_h * p.out_w;
    let pixel_end = pixel_start + out.len() / p.channels;
    for n in 0..p.batch {
        for oh in 0..p.out_h {
            for ow in 0..p.out_w {
                let pixel = (n * p.out_h + oh) * p.out_w + ow;
                if pixel < pixel_start || pixel >= pixel_end || pixel >= total_pixels {
                    continue;
                }
                let out_base = (pixel - pixel_start) * p.channels;
                for c in (0..c8).step_by(8) {
                    let mut acc = _mm256_setzero_si256();
                    for ky in 0..p.kernel {
                        let Some(iy) = p.valid_input_y(oh, ky) else {
                            continue;
                        };
                        for kx in 0..p.kernel {
                            let Some(ix) = p.valid_input_x(ow, kx) else {
                                continue;
                            };
                            let x_ptr = input.as_ptr().add(p.input_offset(n, iy, ix, c));
                            let w_ptr = weight.as_ptr().add(p.weight_offset(ky, kx, c));
                            let xv8 = _mm_loadl_epi64(x_ptr as *const __m128i);
                            let wv8 = _mm_loadl_epi64(w_ptr as *const __m128i);
                            let x16 = _mm256_cvtepi8_epi16(xv8);
                            let w16 = _mm256_cvtepi8_epi16(wv8);
                            let prod16 = _mm256_mullo_epi16(x16, w16);
                            let prod32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(prod16));
                            acc = _mm256_add_epi32(acc, prod32);
                        }
                    }
                    _mm256_storeu_si256(out.as_mut_ptr().add(out_base + c) as *mut __m256i, acc);
                }
                scalar_pixel_tail(
                    input,
                    weight,
                    p,
                    Pixel {
                        n,
                        oh,
                        ow,
                        out_base,
                    },
                    c8,
                    p.channels,
                    out,
                );
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn depthwise3x3_i8_i32_nhwc_avx512_range(
    input: &[i8],
    weight: &[i8],
    p: DepthwiseI8Params,
    out: &mut [i32],
    pixel_start: usize,
) {
    use std::arch::x86_64::*;
    let c16 = p.channels & !15;
    let total_pixels = p.batch * p.out_h * p.out_w;
    let pixel_end = pixel_start + out.len() / p.channels;
    for n in 0..p.batch {
        for oh in 0..p.out_h {
            for ow in 0..p.out_w {
                let pixel = (n * p.out_h + oh) * p.out_w + ow;
                if pixel < pixel_start || pixel >= pixel_end || pixel >= total_pixels {
                    continue;
                }
                let out_base = (pixel - pixel_start) * p.channels;
                for c in (0..c16).step_by(16) {
                    let mut acc = _mm512_setzero_si512();
                    for ky in 0..p.kernel {
                        let Some(iy) = p.valid_input_y(oh, ky) else {
                            continue;
                        };
                        for kx in 0..p.kernel {
                            let Some(ix) = p.valid_input_x(ow, kx) else {
                                continue;
                            };
                            let x = _mm512_cvtepi8_epi32(_mm_loadu_si128(
                                input.as_ptr().add(p.input_offset(n, iy, ix, c)) as *const __m128i,
                            ));
                            let wv = _mm512_cvtepi8_epi32(_mm_loadu_si128(
                                weight.as_ptr().add(p.weight_offset(ky, kx, c)) as *const __m128i,
                            ));
                            acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(x, wv));
                        }
                    }
                    _mm512_storeu_si512(out.as_mut_ptr().add(out_base + c) as *mut __m512i, acc);
                }
                scalar_pixel_tail(
                    input,
                    weight,
                    p,
                    Pixel {
                        n,
                        oh,
                        ow,
                        out_base,
                    },
                    c16,
                    p.channels,
                    out,
                );
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn depthwise3x3_i8_i32_nhwc_neon_range(
    input: &[i8],
    weight: &[i8],
    p: DepthwiseI8Params,
    out: &mut [i32],
    pixel_start: usize,
) {
    use std::arch::aarch64::*;
    // Kernels larger than 7x7 (49 taps) don't fit the stack tap table below;
    // depthwise here is 3x3/5x5, but stay correct for anything else.
    if p.kernel * p.kernel > 49 {
        depthwise_i8_i32_nhwc_scalar_range(input, weight, p, out, pixel_start);
        return;
    }
    let c8 = p.channels & !7;
    let chan = p.channels;
    let total_pixels = p.batch * p.out_h * p.out_w;
    let pixel_end = pixel_start + out.len() / chan;
    let inp = input.as_ptr();
    let wp = weight.as_ptr();
    // No weight == i8::MIN (-128) — always true for symmetric per-channel
    // quantization ([-127,127]) — bounds two taps' products within i16
    // (max |2·128·127| = 32512 < 32767), so the 16-wide path can sum a tap PAIR
    // in i16 with vmlal_s8, cutting the 2-cycle widen (vaddw) count per MAC.
    // Two independent pairs are interleaved so their vmull->vmlal->vaddw chains
    // overlap (a lone vmlal chain serialises and loses on the in-order A53).
    let weight_pairs_safe = !weight.contains(&i8::MIN);
    // Relative tap offsets for a fully in-bounds (interior) pixel: the input
    // offset of tap (ky,kx) is `base_in + rel[t].0` where `base_in` is the
    // window's top-left offset, and `w_base = rel[t].1` is pixel-independent.
    // Interior pixels resolve their taps with one base + adds instead of the
    // per-tap padding checks and index multiplies — the win on the low-channel
    // large-map convs (Conv_2: 16ch/128×128) where the resolve isn't amortised
    // over many channel blocks.
    let mut rel: [(usize, usize); 49] = [(0, 0); 49];
    {
        let mut t = 0;
        for ky in 0..p.kernel {
            for kx in 0..p.kernel {
                rel[t] = ((ky * p.in_w + kx) * chan, (ky * p.kernel + kx) * chan);
                t += 1;
            }
        }
    }
    let ntaps_full = p.kernel * p.kernel;
    for n in 0..p.batch {
        for oh in 0..p.out_h {
            let iy0 = oh * p.stride_h;
            let row_interior = iy0 >= p.pad_top && iy0 + p.kernel <= p.pad_top + p.in_h;
            for ow in 0..p.out_w {
                let pixel = (n * p.out_h + oh) * p.out_w + ow;
                if pixel < pixel_start || pixel >= pixel_end || pixel >= total_pixels {
                    continue;
                }
                let out_base = (pixel - pixel_start) * chan;
                let mut taps: [(usize, usize); 49] = [(0, 0); 49];
                let ntaps;
                let ix0 = ow * p.stride_w;
                if row_interior && ix0 >= p.pad_left && ix0 + p.kernel <= p.pad_left + p.in_w {
                    // Interior: taps = base_in + rel[t] (no padding checks).
                    let base_in =
                        ((n * p.in_h + (iy0 - p.pad_top)) * p.in_w + (ix0 - p.pad_left)) * chan;
                    for t in 0..ntaps_full {
                        taps[t] = (base_in + rel[t].0, rel[t].1);
                    }
                    ntaps = ntaps_full;
                } else {
                    // Border: resolve the in-bounds taps with per-tap checks.
                    let mut nt = 0usize;
                    for ky in 0..p.kernel {
                        let Some(iy) = p.valid_input_y(oh, ky) else {
                            continue;
                        };
                        for kx in 0..p.kernel {
                            let Some(ix) = p.valid_input_x(ow, kx) else {
                                continue;
                            };
                            let in_base = ((n * p.in_h + iy) * p.in_w + ix) * chan;
                            let w_base = (ky * p.kernel + kx) * chan;
                            taps[nt] = (in_base, w_base);
                            nt += 1;
                        }
                    }
                    ntaps = nt;
                }
                let op = out.as_mut_ptr();
                let mut c = 0;
                // 16 channels/iter: one int8x16 load each for x and w amortises
                // the tap-loop bookkeeping over twice the work (halves the
                // c-block trip count on the 288/240-channel 5x5 convs).
                while c + 16 <= chan {
                    let mut a0 = vdupq_n_s32(0);
                    let mut a1 = vdupq_n_s32(0);
                    let mut a2 = vdupq_n_s32(0);
                    let mut a3 = vdupq_n_s32(0);
                    let mut t = 0;
                    if weight_pairs_safe {
                        // 4 taps / iter as two interleaved i16-accumulated pairs.
                        while t + 4 <= ntaps {
                            let (ia0, wa0) = taps[t];
                            let (ia1, wa1) = taps[t + 1];
                            let (ib0, wb0) = taps[t + 2];
                            let (ib1, wb1) = taps[t + 3];
                            let xa0 = vld1q_s8(inp.add(ia0 + c));
                            let wva0 = vld1q_s8(wp.add(wa0 + c));
                            let xa1 = vld1q_s8(inp.add(ia1 + c));
                            let wva1 = vld1q_s8(wp.add(wa1 + c));
                            let xb0 = vld1q_s8(inp.add(ib0 + c));
                            let wvb0 = vld1q_s8(wp.add(wb0 + c));
                            let xb1 = vld1q_s8(inp.add(ib1 + c));
                            let wvb1 = vld1q_s8(wp.add(wb1 + c));
                            let plo_a = vmlal_s8(
                                vmull_s8(vget_low_s8(xa0), vget_low_s8(wva0)),
                                vget_low_s8(xa1),
                                vget_low_s8(wva1),
                            );
                            let phi_a = vmlal_s8(
                                vmull_s8(vget_high_s8(xa0), vget_high_s8(wva0)),
                                vget_high_s8(xa1),
                                vget_high_s8(wva1),
                            );
                            let plo_b = vmlal_s8(
                                vmull_s8(vget_low_s8(xb0), vget_low_s8(wvb0)),
                                vget_low_s8(xb1),
                                vget_low_s8(wvb1),
                            );
                            let phi_b = vmlal_s8(
                                vmull_s8(vget_high_s8(xb0), vget_high_s8(wvb0)),
                                vget_high_s8(xb1),
                                vget_high_s8(wvb1),
                            );
                            a0 = vaddw_s16(vaddw_s16(a0, vget_low_s16(plo_a)), vget_low_s16(plo_b));
                            a1 = vaddw_s16(
                                vaddw_s16(a1, vget_high_s16(plo_a)),
                                vget_high_s16(plo_b),
                            );
                            a2 = vaddw_s16(vaddw_s16(a2, vget_low_s16(phi_a)), vget_low_s16(phi_b));
                            a3 = vaddw_s16(
                                vaddw_s16(a3, vget_high_s16(phi_a)),
                                vget_high_s16(phi_b),
                            );
                            t += 4;
                        }
                    }
                    while t < ntaps {
                        let (in_base, w_base) = taps[t];
                        let xv = vld1q_s8(inp.add(in_base + c));
                        let wv = vld1q_s8(wp.add(w_base + c));
                        let plo = vmull_s8(vget_low_s8(xv), vget_low_s8(wv));
                        let phi = vmull_s8(vget_high_s8(xv), vget_high_s8(wv));
                        a0 = vaddw_s16(a0, vget_low_s16(plo));
                        a1 = vaddw_s16(a1, vget_high_s16(plo));
                        a2 = vaddw_s16(a2, vget_low_s16(phi));
                        a3 = vaddw_s16(a3, vget_high_s16(phi));
                        t += 1;
                    }
                    vst1q_s32(op.add(out_base + c), a0);
                    vst1q_s32(op.add(out_base + c + 4), a1);
                    vst1q_s32(op.add(out_base + c + 8), a2);
                    vst1q_s32(op.add(out_base + c + 12), a3);
                    c += 16;
                }
                while c + 8 <= chan {
                    let mut acc_lo = vdupq_n_s32(0);
                    let mut acc_hi = vdupq_n_s32(0);
                    for &(in_base, w_base) in &taps[..ntaps] {
                        let xv = vld1_s8(inp.add(in_base + c));
                        let wv = vld1_s8(wp.add(w_base + c));
                        let prod = vmull_s8(xv, wv);
                        // vaddw folds the i16->i32 widen and the add into one
                        // instruction per half (was vmovl + vaddq).
                        acc_lo = vaddw_s16(acc_lo, vget_low_s16(prod));
                        acc_hi = vaddw_s16(acc_hi, vget_high_s16(prod));
                    }
                    vst1q_s32(op.add(out_base + c), acc_lo);
                    vst1q_s32(op.add(out_base + c + 4), acc_hi);
                    c += 8;
                }
                if c8 < chan {
                    scalar_pixel_tail(
                        input,
                        weight,
                        p,
                        Pixel {
                            n,
                            oh,
                            ow,
                            out_base,
                        },
                        c8,
                        chan,
                        out,
                    );
                }
            }
        }
    }
}

/// Runtime-dispatched INT8 NHWC depthwise convolution (AVX-512 / AVX2 / NEON / scalar).
pub fn depthwise_i8_i32_nhwc_dispatch(
    input: &[i8],
    weight: &[i8],
    p: DepthwiseI8Params,
    out: &mut [i32],
) {
    depthwise_i8_i32_nhwc_dispatch_with_pool(input, weight, p, out, None);
}

/// INT8 NHWC depthwise dispatch parallelised over a caller-supplied thread pool.
pub fn depthwise_i8_i32_nhwc_dispatch_with_pool(
    input: &[i8],
    weight: &[i8],
    p: DepthwiseI8Params,
    out: &mut [i32],
    thread_pool: Option<&ThreadPool>,
) {
    validate_depthwise(input, weight, p, out);

    let pixels = p.batch * p.out_h * p.out_w;
    let nthreads = thread_pool
        .map(|pool| pool.current_num_threads().max(1))
        .unwrap_or_else(|| rayon::current_num_threads().max(1));
    if !cfg!(miri) && pixels >= nthreads * 8 && nthreads > 1 {
        let pixels_per_chunk = pixels.div_ceil(nthreads * 2).max(8);
        let elems_per_chunk = pixels_per_chunk * p.channels;
        let mut work = || {
            out.par_chunks_mut(elems_per_chunk)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let pixel_start = chunk_idx * pixels_per_chunk;
                    depthwise_i8_i32_nhwc_dispatch_range(input, weight, p, chunk, pixel_start);
                });
        };
        if let Some(pool) = thread_pool {
            pool.install(work);
        } else {
            work();
        }
        return;
    }

    depthwise_i8_i32_nhwc_dispatch_range(input, weight, p, out, 0);
}

fn depthwise_i8_i32_nhwc_dispatch_range(
    input: &[i8],
    weight: &[i8],
    p: DepthwiseI8Params,
    out: &mut [i32],
    pixel_start: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        let features = crate::host_cpu().features;
        if features.x86_avx512_bw() && p.channels >= 16 {
            unsafe { depthwise3x3_i8_i32_nhwc_avx512_range(input, weight, p, out, pixel_start) };
            return;
        }
        if features.avx2 && p.channels >= 8 {
            unsafe { depthwise3x3_i8_i32_nhwc_avx2_range(input, weight, p, out, pixel_start) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if crate::host_cpu().features.neon && p.channels >= 8 {
            unsafe { depthwise3x3_i8_i32_nhwc_neon_range(input, weight, p, out, pixel_start) };
            return;
        }
    }
    depthwise_i8_i32_nhwc_scalar_range(input, weight, p, out, pixel_start);
}

/// Runtime-dispatched INT8 NHWC 3x3 depthwise convolution.
pub fn depthwise3x3_i8_i32_nhwc_dispatch(
    input: &[i8],
    weight: &[i8],
    p: Depthwise3x3I8Params,
    out: &mut [i32],
) {
    depthwise_i8_i32_nhwc_dispatch(input, weight, p.into(), out);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pseudo_i8(seed: u64, n: usize) -> Vec<i8> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as i64 % 256 - 128) as i8
            })
            .collect()
    }

    fn check_shape(
        batch: usize,
        h: usize,
        w: usize,
        c: usize,
        sh: usize,
        sw: usize,
        pt: usize,
        pl: usize,
    ) {
        let p = Depthwise3x3I8Params {
            batch,
            in_h: h,
            in_w: w,
            channels: c,
            stride_h: sh,
            stride_w: sw,
            pad_top: pt,
            pad_left: pl,
            out_h: (h + pt + pt - 3) / sh + 1,
            out_w: (w + pl + pl - 3) / sw + 1,
        };
        let input = pseudo_i8(0xA, p.input_len());
        let weight = pseudo_i8(0xB, p.weight_len());
        let mut expected = vec![0_i32; p.output_len()];
        let mut got = vec![0_i32; p.output_len()];
        depthwise3x3_i8_i32_nhwc_scalar(&input, &weight, p, &mut expected);
        depthwise3x3_i8_i32_nhwc_dispatch(&input, &weight, p, &mut got);
        assert_eq!(
            got, expected,
            "b={batch} h={h} w={w} c={c} s={sh}x{sw} p={pt}x{pl}"
        );
    }

    #[test]
    fn dispatch_matches_scalar_tracker_shapes() {
        for &(batch, h, w, c, sh, sw, pt, pl) in &[
            (1, 128, 128, 16, 1, 1, 1, 1),
            (1, 64, 64, 96, 1, 1, 1, 1),
            (1, 32, 32, 192, 1, 1, 1, 1),
            (1, 16, 16, 672, 1, 1, 1, 1),
            (1, 17, 19, 15, 1, 1, 1, 1),
            (2, 15, 13, 9, 2, 2, 1, 1),
        ] {
            check_shape(batch, h, w, c, sh, sw, pt, pl);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x86_simd_paths_match_scalar_when_available() {
        let p = Depthwise3x3I8Params {
            batch: 1,
            in_h: 16,
            in_w: 16,
            channels: 32,
            stride_h: 1,
            stride_w: 1,
            pad_top: 1,
            pad_left: 1,
            out_h: 16,
            out_w: 16,
        };
        let input = pseudo_i8(0xC, p.input_len());
        let weight = pseudo_i8(0xD, p.weight_len());
        let mut expected = vec![0_i32; p.output_len()];
        depthwise3x3_i8_i32_nhwc_scalar(&input, &weight, p, &mut expected);
        let features = crate::host_cpu().features;
        if features.avx2 {
            let mut got = vec![0_i32; expected.len()];
            unsafe { depthwise3x3_i8_i32_nhwc_avx2_range(&input, &weight, p.into(), &mut got, 0) };
            assert_eq!(got, expected);
        }
        if features.x86_avx512_bw() {
            let mut got = vec![0_i32; expected.len()];
            unsafe {
                depthwise3x3_i8_i32_nhwc_avx512_range(&input, &weight, p.into(), &mut got, 0)
            };
            assert_eq!(got, expected);
        }
    }

    #[test]
    fn generic_dispatch_matches_scalar_5x5_tracker_shapes() {
        for &(h, w, c) in &[(32, 32, 192), (16, 16, 672), (17, 19, 15)] {
            let p = DepthwiseI8Params {
                batch: 1,
                in_h: h,
                in_w: w,
                channels: c,
                kernel: 5,
                stride_h: 1,
                stride_w: 1,
                pad_top: 2,
                pad_left: 2,
                out_h: h,
                out_w: w,
            };
            let input = pseudo_i8(0xE, p.input_len());
            let weight = pseudo_i8(0xF, p.weight_len());
            let mut expected = vec![0_i32; p.output_len()];
            let mut got = vec![0_i32; p.output_len()];
            depthwise_i8_i32_nhwc_scalar(&input, &weight, p, &mut expected);
            depthwise_i8_i32_nhwc_dispatch(&input, &weight, p, &mut got);
            assert_eq!(got, expected, "5x5 h={h} w={w} c={c}");
        }
    }

    #[test]
    fn nchw_khwc_matches_nhwc_reference_tracker_shapes() {
        for &(h, w, c, kernel, stride, pad) in &[
            (128, 128, 96, 3, 2, 1),
            (64, 64, 144, 3, 2, 1),
            (32, 32, 192, 5, 1, 2),
            (17, 19, 15, 3, 2, 1),
        ] {
            let p = DepthwiseI8Params {
                batch: 1,
                in_h: h,
                in_w: w,
                channels: c,
                kernel,
                stride_h: stride,
                stride_w: stride,
                pad_top: pad,
                pad_left: pad,
                out_h: (h + pad + pad - kernel) / stride + 1,
                out_w: (w + pad + pad - kernel) / stride + 1,
            };
            let input_nhwc = pseudo_i8(0x11, p.input_len());
            let weight = pseudo_i8(0x12, p.weight_len());
            let mut input_nchw = vec![0_i8; p.input_len()];
            for y in 0..h {
                for x in 0..w {
                    let src_base = (y * w + x) * c;
                    for ch in 0..c {
                        input_nchw[(ch * h + y) * w + x] = input_nhwc[src_base + ch];
                    }
                }
            }

            let mut expected_nhwc = vec![0_i32; p.output_len()];
            depthwise_i8_i32_nhwc_scalar(&input_nhwc, &weight, p, &mut expected_nhwc);

            let mut got_nchw = vec![0_i32; p.output_len()];
            depthwise_i8_i32_nchw_khwc_dispatch(&input_nchw, &weight, p, &mut got_nchw);

            for oh in 0..p.out_h {
                for ow in 0..p.out_w {
                    let src_base = (oh * p.out_w + ow) * c;
                    for ch in 0..c {
                        let dst = (ch * p.out_h + oh) * p.out_w + ow;
                        assert_eq!(
                            got_nchw[dst],
                            expected_nhwc[src_base + ch],
                            "h={h} w={w} c={c} k={kernel} s={stride} ch={ch} oh={oh} ow={ow}"
                        );
                    }
                }
            }
        }
    }
}
