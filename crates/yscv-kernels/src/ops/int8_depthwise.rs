//! INT8 depthwise Conv2D kernels for quantized tracker chains.
//!
//! Layout: NHWC input, KHWC depthwise weights with depth_multiplier=1,
//! int32 output in NHWC. The kernel intentionally stops at accumulation:
//! bias/requant/residual/activation epilogues are chain-level policy and
//! are fused by the caller that owns the quantization scales.

#![allow(unsafe_code, unsafe_op_in_unsafe_fn)]

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
    fn output_offset(self, n: usize, y: usize, x: usize) -> usize {
        ((n * self.out_h + y) * self.out_w + x) * self.channels
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

pub fn depthwise_i8_i32_nhwc_scalar(
    input: &[i8],
    weight: &[i8],
    p: DepthwiseI8Params,
    out: &mut [i32],
) {
    validate_depthwise(input, weight, p, out);
    for n in 0..p.batch {
        for oh in 0..p.out_h {
            for ow in 0..p.out_w {
                let out_base = p.output_offset(n, oh, ow);
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

pub fn depthwise_i8_i32_nchw_khwc_dispatch(
    input: &[i8],
    weight: &[i8],
    p: DepthwiseI8Params,
    out: &mut [i32],
) {
    depthwise_i8_i32_nchw_khwc_scalar(input, weight, p, out);
}

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
unsafe fn depthwise3x3_i8_i32_nhwc_avx2(
    input: &[i8],
    weight: &[i8],
    p: DepthwiseI8Params,
    out: &mut [i32],
) {
    use std::arch::x86_64::*;
    let c8 = p.channels & !7;
    for n in 0..p.batch {
        for oh in 0..p.out_h {
            for ow in 0..p.out_w {
                let out_base = p.output_offset(n, oh, ow);
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
unsafe fn depthwise3x3_i8_i32_nhwc_avx512(
    input: &[i8],
    weight: &[i8],
    p: DepthwiseI8Params,
    out: &mut [i32],
) {
    use std::arch::x86_64::*;
    let c16 = p.channels & !15;
    for n in 0..p.batch {
        for oh in 0..p.out_h {
            for ow in 0..p.out_w {
                let out_base = p.output_offset(n, oh, ow);
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
unsafe fn depthwise3x3_i8_i32_nhwc_neon(
    input: &[i8],
    weight: &[i8],
    p: DepthwiseI8Params,
    out: &mut [i32],
) {
    use std::arch::aarch64::*;
    let c8 = p.channels & !7;
    for n in 0..p.batch {
        for oh in 0..p.out_h {
            for ow in 0..p.out_w {
                let out_base = p.output_offset(n, oh, ow);
                for c in (0..c8).step_by(8) {
                    let mut acc_lo = vdupq_n_s32(0);
                    let mut acc_hi = vdupq_n_s32(0);
                    for ky in 0..p.kernel {
                        let Some(iy) = p.valid_input_y(oh, ky) else {
                            continue;
                        };
                        for kx in 0..p.kernel {
                            let Some(ix) = p.valid_input_x(ow, kx) else {
                                continue;
                            };
                            let xv = vld1_s8(input.as_ptr().add(p.input_offset(n, iy, ix, c)));
                            let wv = vld1_s8(weight.as_ptr().add(p.weight_offset(ky, kx, c)));
                            let prod = vmull_s8(xv, wv);
                            acc_lo = vaddq_s32(acc_lo, vmovl_s16(vget_low_s16(prod)));
                            acc_hi = vaddq_s32(acc_hi, vmovl_s16(vget_high_s16(prod)));
                        }
                    }
                    vst1q_s32(out.as_mut_ptr().add(out_base + c), acc_lo);
                    vst1q_s32(out.as_mut_ptr().add(out_base + c + 4), acc_hi);
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

pub fn depthwise_i8_i32_nhwc_dispatch(
    input: &[i8],
    weight: &[i8],
    p: DepthwiseI8Params,
    out: &mut [i32],
) {
    validate_depthwise(input, weight, p, out);

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && p.channels >= 16
        {
            unsafe { depthwise3x3_i8_i32_nhwc_avx512(input, weight, p, out) };
            return;
        }
        if std::is_x86_feature_detected!("avx2") && p.channels >= 8 {
            unsafe { depthwise3x3_i8_i32_nhwc_avx2(input, weight, p, out) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") && p.channels >= 8 {
            unsafe { depthwise3x3_i8_i32_nhwc_neon(input, weight, p, out) };
            return;
        }
    }
    depthwise_i8_i32_nhwc_scalar(input, weight, p, out);
}

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
        if std::is_x86_feature_detected!("avx2") {
            let mut got = vec![0_i32; expected.len()];
            unsafe { depthwise3x3_i8_i32_nhwc_avx2(&input, &weight, p.into(), &mut got) };
            assert_eq!(got, expected);
        }
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
            let mut got = vec![0_i32; expected.len()];
            unsafe { depthwise3x3_i8_i32_nhwc_avx512(&input, &weight, p.into(), &mut got) };
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
