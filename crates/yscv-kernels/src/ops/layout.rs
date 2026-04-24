//! Layout conversion kernels for the NCHWc transformer pass.
//!
//! These four routines reorder a contiguous f32 tensor between the three
//! layouts that `yscv_tensor::Layout` tracks:
//!   - `NCHW`  — `[N, C, H, W]`
//!   - `NHWC`  — `[N, H, W, C]`
//!   - `NCHWc` — `[N, C/block, H, W, block]`
//!
//! NHWC ↔ NCHWc is a contiguous memcpy of `block`-element chunks — in NHWC
//! the innermost C stride is 1, and NCHWc's innermost lane is also C in
//! units of `block`, so each `block`-chunk copy is dense→dense. When
//! `C % block != 0`, the trailing partial chunk is zero-padded.
//!
//! NCHW ↔ NCHWc involves a strided gather/scatter: NCHW has stride `H*W`
//! between channels, while NCHWc needs consecutive `block` channels
//! packed contiguously in the innermost dim. Done as a tiled transpose.
//!
//! All four functions preserve contiguous storage and tag the output
//! `Tensor` with its new `Layout`. No unsafe — the copies are expressed
//! as slice operations and iterator-driven loops so the compiler can
//! auto-vectorize the dense cases.
//!
//! Cross-architecture: no intrinsics here, so it's the same code on
//! x86_64, aarch64, and scalar fallback targets. The plan's projected
//! SIMD win (`_mm256_loadu_ps` / `ld1` for the block-sized chunks) is
//! already what LLVM generates for the slice-copy hot path; explicit
//! intrinsics would only help for the strided NCHW↔NCHWc transpose,
//! which runs at graph boundaries (cold path by construction).

use yscv_tensor::{AlignedVec, Layout, Tensor};

use crate::core::error::KernelError;

/// Returns `(N, C, H, W)` from a 4-D NCHW or NHWC shape, or an error.
#[inline]
fn nchw_dims(shape: &[usize]) -> Result<(usize, usize, usize, usize), KernelError> {
    match shape {
        [n, c, h, w] => Ok((*n, *c, *h, *w)),
        _ => Err(KernelError::LayoutConversion(format!(
            "expected 4-D shape [N,C,H,W], got {shape:?}"
        ))),
    }
}

#[inline]
fn nhwc_dims(shape: &[usize]) -> Result<(usize, usize, usize, usize), KernelError> {
    match shape {
        [n, h, w, c] => Ok((*n, *h, *w, *c)),
        _ => Err(KernelError::LayoutConversion(format!(
            "expected 4-D shape [N,H,W,C], got {shape:?}"
        ))),
    }
}

#[inline]
fn nchwc_dims(shape: &[usize]) -> Result<(usize, usize, usize, usize, usize), KernelError> {
    match shape {
        [n, co, h, w, b] => Ok((*n, *co, *h, *w, *b)),
        _ => Err(KernelError::LayoutConversion(format!(
            "expected 5-D NCHWc shape [N,C/b,H,W,b], got {shape:?}"
        ))),
    }
}

/// Number of channel blocks needed for `channels` with chunk size `block`,
/// rounded up (the trailing partial block is zero-padded on convert).
#[inline]
fn channel_blocks(channels: usize, block: usize) -> usize {
    channels.div_ceil(block)
}

/// NHWC → NCHWc. Reorders `[N, H, W, C]` into `[N, C/block, H, W, block]`,
/// padding C with zeros if `C % block != 0`. Output tensor carries the
/// `Layout::NCHWc { block }` tag.
pub fn nhwc_to_nchwc(input: &Tensor, block: usize) -> Result<Tensor, KernelError> {
    if block == 0 || !block.is_power_of_two() || block > u8::MAX as usize {
        return Err(KernelError::LayoutConversion(format!(
            "block must be power-of-two in 1..=255, got {block}"
        )));
    }
    let (n, h, w, c) = nhwc_dims(input.shape())?;
    let co = channel_blocks(c, block);
    let src = input.try_data().map_err(KernelError::from)?;

    let out_len = n * co * h * w * block;
    let mut out = AlignedVec::<f32>::calloc(out_len);
    let dst = out.as_mut_slice();

    let nchwc_row_stride = block; // innermost b
    let nchwc_hw_stride = w * block;
    let nchwc_co_stride = h * w * block;
    let nchwc_n_stride = co * h * w * block;
    let nhwc_w_stride = c;
    let nhwc_h_stride = w * c;
    let nhwc_n_stride = h * w * c;

    for ni in 0..n {
        for hi in 0..h {
            for wi in 0..w {
                let src_base = ni * nhwc_n_stride + hi * nhwc_h_stride + wi * nhwc_w_stride;
                for coi in 0..co {
                    let c_start = coi * block;
                    let c_take = block.min(c - c_start);
                    let src_off = src_base + c_start;
                    let dst_off = ni * nchwc_n_stride
                        + coi * nchwc_co_stride
                        + hi * nchwc_hw_stride
                        + wi * nchwc_row_stride;
                    dst[dst_off..dst_off + c_take].copy_from_slice(&src[src_off..src_off + c_take]);
                    // Trailing lanes are already zero (calloc).
                }
            }
        }
    }

    let shape = vec![n, co, h, w, block];
    let out_tensor = Tensor::from_aligned(shape, out).map_err(KernelError::from)?;
    Ok(out_tensor.with_layout(Layout::NCHWc { block: block as u8 }))
}

/// NCHWc → NHWC. Reorders `[N, C/block, H, W, block]` into `[N, H, W, C]`
/// where `C = input's effective channel count` (passed explicitly since
/// the padded tail is stripped). Output tensor carries `Layout::NHWC`.
pub fn nchwc_to_nhwc(input: &Tensor, channels: usize) -> Result<Tensor, KernelError> {
    let (n, co, h, w, block) = nchwc_dims(input.shape())?;
    if channels > co * block {
        return Err(KernelError::LayoutConversion(format!(
            "channels {channels} exceeds NCHWc capacity {co}*{block}"
        )));
    }
    if channels == 0 {
        return Err(KernelError::LayoutConversion(
            "channels must be > 0".to_string(),
        ));
    }
    let src = input.try_data().map_err(KernelError::from)?;

    let out_len = n * h * w * channels;
    let mut out = AlignedVec::<f32>::uninitialized(out_len);
    let dst = out.as_mut_slice();

    let nchwc_row_stride = block;
    let nchwc_hw_stride = w * block;
    let nchwc_co_stride = h * w * block;
    let nchwc_n_stride = co * h * w * block;
    let nhwc_w_stride = channels;
    let nhwc_h_stride = w * channels;
    let nhwc_n_stride = h * w * channels;

    for ni in 0..n {
        for hi in 0..h {
            for wi in 0..w {
                let dst_base = ni * nhwc_n_stride + hi * nhwc_h_stride + wi * nhwc_w_stride;
                for coi in 0..co {
                    let c_start = coi * block;
                    if c_start >= channels {
                        break;
                    }
                    let c_take = block.min(channels - c_start);
                    let src_off = ni * nchwc_n_stride
                        + coi * nchwc_co_stride
                        + hi * nchwc_hw_stride
                        + wi * nchwc_row_stride;
                    let dst_off = dst_base + c_start;
                    dst[dst_off..dst_off + c_take].copy_from_slice(&src[src_off..src_off + c_take]);
                }
            }
        }
    }

    let shape = vec![n, h, w, channels];
    let out_tensor = Tensor::from_aligned(shape, out).map_err(KernelError::from)?;
    Ok(out_tensor.with_layout(Layout::NHWC))
}

/// NCHW → NCHWc. Reorders `[N, C, H, W]` into `[N, C/block, H, W, block]`,
/// padding with zeros if `C % block != 0`. Output carries `Layout::NCHWc`.
///
/// Strided gather: for each (n, c_outer, h, w), reads `block` values from
/// NCHW that are `H*W` apart. Done inner-loop-over-block so the innermost
/// memory write is contiguous.
pub fn nchw_to_nchwc(input: &Tensor, block: usize) -> Result<Tensor, KernelError> {
    if block == 0 || !block.is_power_of_two() || block > u8::MAX as usize {
        return Err(KernelError::LayoutConversion(format!(
            "block must be power-of-two in 1..=255, got {block}"
        )));
    }
    let (n, c, h, w) = nchw_dims(input.shape())?;
    let co = channel_blocks(c, block);
    let src = input.try_data().map_err(KernelError::from)?;

    let out_len = n * co * h * w * block;
    let mut out = AlignedVec::<f32>::calloc(out_len);
    let dst = out.as_mut_slice();

    let nchw_c_stride = h * w;
    let nchw_n_stride = c * h * w;
    let nchwc_row_stride = block;
    let nchwc_hw_stride = w * block;
    let nchwc_co_stride = h * w * block;
    let nchwc_n_stride = co * h * w * block;

    for ni in 0..n {
        for coi in 0..co {
            let c_start = coi * block;
            let c_take = block.min(c - c_start);
            for hi in 0..h {
                for wi in 0..w {
                    let dst_off = ni * nchwc_n_stride
                        + coi * nchwc_co_stride
                        + hi * nchwc_hw_stride
                        + wi * nchwc_row_stride;
                    let src_hw = ni * nchw_n_stride + hi * w + wi;
                    let dst_chunk = &mut dst[dst_off..dst_off + c_take];
                    for k in 0..c_take {
                        let ci = c_start + k;
                        dst_chunk[k] = src[src_hw + ci * nchw_c_stride];
                    }
                    // Trailing lanes stay zero (calloc).
                }
            }
        }
    }

    let shape = vec![n, co, h, w, block];
    let out_tensor = Tensor::from_aligned(shape, out).map_err(KernelError::from)?;
    Ok(out_tensor.with_layout(Layout::NCHWc { block: block as u8 }))
}

/// NCHWc → NCHW. Reorders `[N, C/block, H, W, block]` into `[N, C, H, W]`
/// using the `channels` the caller wants (trailing zero-padded lanes
/// in the source are stripped). Output carries `Layout::NCHW`.
pub fn nchwc_to_nchw(input: &Tensor, channels: usize) -> Result<Tensor, KernelError> {
    let (n, co, h, w, block) = nchwc_dims(input.shape())?;
    if channels > co * block {
        return Err(KernelError::LayoutConversion(format!(
            "channels {channels} exceeds NCHWc capacity {co}*{block}"
        )));
    }
    if channels == 0 {
        return Err(KernelError::LayoutConversion(
            "channels must be > 0".to_string(),
        ));
    }
    let src = input.try_data().map_err(KernelError::from)?;

    let out_len = n * channels * h * w;
    let mut out = AlignedVec::<f32>::uninitialized(out_len);
    let dst = out.as_mut_slice();

    let nchw_c_stride = h * w;
    let nchw_n_stride = channels * h * w;
    let nchwc_row_stride = block;
    let nchwc_hw_stride = w * block;
    let nchwc_co_stride = h * w * block;
    let nchwc_n_stride = co * h * w * block;

    for ni in 0..n {
        for coi in 0..co {
            let c_start = coi * block;
            if c_start >= channels {
                break;
            }
            let c_take = block.min(channels - c_start);
            for hi in 0..h {
                for wi in 0..w {
                    let src_off = ni * nchwc_n_stride
                        + coi * nchwc_co_stride
                        + hi * nchwc_hw_stride
                        + wi * nchwc_row_stride;
                    let src_chunk = &src[src_off..src_off + c_take];
                    let dst_hw = ni * nchw_n_stride + hi * w + wi;
                    for k in 0..c_take {
                        let ci = c_start + k;
                        dst[dst_hw + ci * nchw_c_stride] = src_chunk[k];
                    }
                }
            }
        }
    }

    let shape = vec![n, channels, h, w];
    let out_tensor = Tensor::from_aligned(shape, out).map_err(KernelError::from)?;
    Ok(out_tensor.with_layout(Layout::NCHW))
}

/// Step S.3: specialized NCHW→NHWC conversion with AVX2 8×8 block
/// transpose for the (c, hw) → (hw, c) inner matrix. The generic
/// `Tensor::permute(&[0, 2, 3, 1])` walks one float at a time in a
/// 32×32 tile (scalar inner), burning ~58 µs @ 6T in tracker on two
/// [1, 320, 16, 16] DW Conv inputs.
///
/// Fast path fires when `c % 8 == 0 && (h*w) % 8 == 0`. 24 SIMD ops per
/// 8×8 block (8 loads + 8 stores + 8 transpose instrs) vs 128 scalar
/// loads/stores — ~5× fewer memory ops and fully vectorized.
///
/// Tail case (c%8 != 0 or hw%8 != 0) falls back to the generic
/// `Tensor::permute`. First-layer Conv [1, 3, ...] hits this tail.
///
/// Caller should feature-gate on `is_x86_feature_detected!("avx")`.
pub fn nchw_to_nhwc_fast(input: &Tensor) -> Result<Tensor, KernelError> {
    let shape = input.shape();
    let (n, c, h, w) = nchw_dims(shape)?;
    let hw = h * w;
    let out_count = n
        .checked_mul(c)
        .and_then(|v| v.checked_mul(hw))
        .ok_or_else(|| {
            KernelError::LayoutConversion(format!("nchw→nhwc overflow for {shape:?}"))
        })?;

    // Fast path requires both dims multiple of 8 AND x86 AVX runtime.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let fast_path_ok = c % 8 == 0 && hw % 8 == 0 && std::is_x86_feature_detected!("avx");
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let fast_path_ok = false;

    if !fast_path_ok {
        // Fall back to the generic tiled scalar transpose in yscv_tensor.
        // `permute` preserves the source's Layout tag (defaults to NCHW),
        // but semantically the caller asked for NHWC — retag explicitly
        // so downstream ops see the right layout. Phase 0.4 fix: without
        // this retag, `nchw_to_nhwc_fast_matches_generic_permute_aligned`
        // failed on aarch64 (the AVX fast path never fires, so this
        // branch is the only one that runs there).
        return input
            .permute(&[0, 2, 3, 1])
            .map(|t| t.with_layout(Layout::NHWC))
            .map_err(KernelError::from);
    }

    #[allow(unused_mut)]
    let mut out = AlignedVec::<f32>::uninitialized(out_count);

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[allow(unsafe_code)]
    {
        let src = input.try_data().map_err(KernelError::from)?;
        let dst = out.as_mut_slice();
        // SAFETY: AVX runtime feature checked above. Slice bounds
        // verified: src = n*c*hw, dst = n*hw*c (same total floats).
        // Block offsets stay within each n-batch's c*hw region.
        unsafe {
            nchw_to_nhwc_inner_avx(src, dst, n, c, h, w);
        }
    }

    let out_shape = vec![n, h, w, c];
    let out_tensor = Tensor::from_aligned(out_shape, out).map_err(KernelError::from)?;
    Ok(out_tensor.with_layout(Layout::NHWC))
}

/// 8×8 block transpose. Transposes a contiguous c×hw matrix in src
/// (row stride `hw`) into an hw×c matrix in dst (row stride `c`) for
/// each of the `n` batches. Callers: c%8==0, hw%8==0, AVX available.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
#[allow(unsafe_code, clippy::too_many_arguments)]
unsafe fn nchw_to_nhwc_inner_avx(
    src: &[f32],
    dst: &mut [f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
) {
    unsafe {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::{
            _mm256_loadu_ps, _mm256_permute2f128_ps, _mm256_shuffle_ps, _mm256_storeu_ps,
            _mm256_unpackhi_ps, _mm256_unpacklo_ps,
        };
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::{
            _mm256_loadu_ps, _mm256_permute2f128_ps, _mm256_shuffle_ps, _mm256_storeu_ps,
            _mm256_unpackhi_ps, _mm256_unpacklo_ps,
        };

        let hw = h * w;
        let src_ptr = src.as_ptr();
        let dst_ptr = dst.as_mut_ptr();

        for batch in 0..n {
            let s_base = src_ptr.add(batch * c * hw);
            let d_base = dst_ptr.add(batch * hw * c);
            // Block over c (step 8 rows), inner block over hw (step 8 cols).
            let mut ci = 0usize;
            while ci + 8 <= c {
                let mut hi = 0usize;
                while hi + 8 <= hw {
                    // Load 8 rows of 8 floats each (c rows ci..ci+8 at hw
                    // columns hi..hi+8).
                    let r0 = _mm256_loadu_ps(s_base.add(ci * hw + hi));
                    let r1 = _mm256_loadu_ps(s_base.add((ci + 1) * hw + hi));
                    let r2 = _mm256_loadu_ps(s_base.add((ci + 2) * hw + hi));
                    let r3 = _mm256_loadu_ps(s_base.add((ci + 3) * hw + hi));
                    let r4 = _mm256_loadu_ps(s_base.add((ci + 4) * hw + hi));
                    let r5 = _mm256_loadu_ps(s_base.add((ci + 5) * hw + hi));
                    let r6 = _mm256_loadu_ps(s_base.add((ci + 6) * hw + hi));
                    let r7 = _mm256_loadu_ps(s_base.add((ci + 7) * hw + hi));

                    // Stage 1: unpack pairs of rows.
                    let t0 = _mm256_unpacklo_ps(r0, r1);
                    let t1 = _mm256_unpackhi_ps(r0, r1);
                    let t2 = _mm256_unpacklo_ps(r2, r3);
                    let t3 = _mm256_unpackhi_ps(r2, r3);
                    let t4 = _mm256_unpacklo_ps(r4, r5);
                    let t5 = _mm256_unpackhi_ps(r4, r5);
                    let t6 = _mm256_unpacklo_ps(r6, r7);
                    let t7 = _mm256_unpackhi_ps(r6, r7);

                    // Stage 2: shuffle 4-element groups (0x44 = [1,0,1,0], 0xee = [3,2,3,2]).
                    let u0 = _mm256_shuffle_ps::<0x44>(t0, t2);
                    let u1 = _mm256_shuffle_ps::<0xee>(t0, t2);
                    let u2 = _mm256_shuffle_ps::<0x44>(t1, t3);
                    let u3 = _mm256_shuffle_ps::<0xee>(t1, t3);
                    let u4 = _mm256_shuffle_ps::<0x44>(t4, t6);
                    let u5 = _mm256_shuffle_ps::<0xee>(t4, t6);
                    let u6 = _mm256_shuffle_ps::<0x44>(t5, t7);
                    let u7 = _mm256_shuffle_ps::<0xee>(t5, t7);

                    // Stage 3: permute 128-bit lanes (0x20 = lo|lo, 0x31 = hi|hi).
                    let col0 = _mm256_permute2f128_ps::<0x20>(u0, u4);
                    let col1 = _mm256_permute2f128_ps::<0x20>(u1, u5);
                    let col2 = _mm256_permute2f128_ps::<0x20>(u2, u6);
                    let col3 = _mm256_permute2f128_ps::<0x20>(u3, u7);
                    let col4 = _mm256_permute2f128_ps::<0x31>(u0, u4);
                    let col5 = _mm256_permute2f128_ps::<0x31>(u1, u5);
                    let col6 = _mm256_permute2f128_ps::<0x31>(u2, u6);
                    let col7 = _mm256_permute2f128_ps::<0x31>(u3, u7);

                    // Store 8 rows of 8 floats in NHWC dst (hw rows hi..hi+8
                    // at c columns ci..ci+8). Row stride is c in dst.
                    _mm256_storeu_ps(d_base.add(hi * c + ci), col0);
                    _mm256_storeu_ps(d_base.add((hi + 1) * c + ci), col1);
                    _mm256_storeu_ps(d_base.add((hi + 2) * c + ci), col2);
                    _mm256_storeu_ps(d_base.add((hi + 3) * c + ci), col3);
                    _mm256_storeu_ps(d_base.add((hi + 4) * c + ci), col4);
                    _mm256_storeu_ps(d_base.add((hi + 5) * c + ci), col5);
                    _mm256_storeu_ps(d_base.add((hi + 6) * c + ci), col6);
                    _mm256_storeu_ps(d_base.add((hi + 7) * c + ci), col7);
                    hi += 8;
                }
                ci += 8;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nhwc(n: usize, h: usize, w: usize, c: usize) -> Tensor {
        let data: Vec<f32> = (0..n * h * w * c).map(|i| i as f32).collect();
        Tensor::from_vec(vec![n, h, w, c], data)
            .unwrap()
            .with_layout(Layout::NHWC)
    }

    fn nchw(n: usize, c: usize, h: usize, w: usize) -> Tensor {
        let data: Vec<f32> = (0..n * c * h * w).map(|i| i as f32).collect();
        Tensor::from_vec(vec![n, c, h, w], data)
            .unwrap()
            .with_layout(Layout::NCHW)
    }

    #[test]
    fn nhwc_to_nchwc_round_trip_c_divisible_by_block() {
        let t = nhwc(2, 3, 4, 8);
        let blocked = nhwc_to_nchwc(&t, 8).unwrap();
        assert_eq!(blocked.shape(), &[2, 1, 3, 4, 8]);
        assert_eq!(blocked.layout(), Layout::NCHWc { block: 8 });
        let back = nchwc_to_nhwc(&blocked, 8).unwrap();
        assert_eq!(back.shape(), t.shape());
        assert_eq!(back.layout(), Layout::NHWC);
        assert_eq!(back.data(), t.data());
    }

    #[test]
    fn nhwc_to_nchwc_round_trip_c_needs_padding() {
        let t = nhwc(1, 2, 2, 17);
        let blocked = nhwc_to_nchwc(&t, 8).unwrap();
        assert_eq!(blocked.shape(), &[1, 3, 2, 2, 8]);
        let back = nchwc_to_nhwc(&blocked, 17).unwrap();
        assert_eq!(back.shape(), t.shape());
        assert_eq!(back.data(), t.data());
    }

    #[test]
    fn nhwc_to_nchwc_pads_tail_with_zeros() {
        let t = nhwc(1, 1, 1, 5);
        let blocked = nhwc_to_nchwc(&t, 8).unwrap();
        let data = blocked.data();
        assert_eq!(data[..5], [0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&data[5..], &[0.0; 3]);
    }

    #[test]
    fn nchw_to_nchwc_round_trip() {
        let t = nchw(1, 8, 2, 3);
        let blocked = nchw_to_nchwc(&t, 8).unwrap();
        assert_eq!(blocked.shape(), &[1, 1, 2, 3, 8]);
        assert_eq!(blocked.layout(), Layout::NCHWc { block: 8 });
        let back = nchwc_to_nchw(&blocked, 8).unwrap();
        assert_eq!(back.shape(), t.shape());
        assert_eq!(back.layout(), Layout::NCHW);
        assert_eq!(back.data(), t.data());
    }

    #[test]
    fn nchw_to_nchwc_round_trip_with_padding() {
        let t = nchw(1, 10, 2, 2);
        let blocked = nchw_to_nchwc(&t, 8).unwrap();
        assert_eq!(blocked.shape(), &[1, 2, 2, 2, 8]);
        let back = nchwc_to_nchw(&blocked, 10).unwrap();
        assert_eq!(back.shape(), t.shape());
        assert_eq!(back.data(), t.data());
    }

    #[test]
    fn nchw_to_nchwc_places_channels_in_block_lanes() {
        // C=2 block=2: values (c=0,h=0,w=0) and (c=1,h=0,w=0) should
        // land in lanes 0 and 1 of NCHWc[0,0,0,0,:].
        let t = nchw(1, 2, 1, 1);
        let blocked = nchw_to_nchwc(&t, 2).unwrap();
        assert_eq!(blocked.shape(), &[1, 1, 1, 1, 2]);
        assert_eq!(blocked.data(), &[0.0, 1.0]);
    }

    #[test]
    fn nhwc_to_nchwc_block_1_is_identity_up_to_shape() {
        let t = nhwc(1, 2, 2, 4);
        let blocked = nhwc_to_nchwc(&t, 1).unwrap();
        assert_eq!(blocked.shape(), &[1, 4, 2, 2, 1]);
        assert_eq!(blocked.layout(), Layout::NCHWc { block: 1 });
        // Block=1 is equivalent to NCHW with a trailing singleton dim.
    }

    #[test]
    fn nchwc_block_must_be_power_of_two() {
        let t = nhwc(1, 1, 1, 6);
        assert!(nhwc_to_nchwc(&t, 3).is_err());
        assert!(nchw_to_nchwc(&t, 3).is_err());
    }

    #[test]
    fn conversions_reject_wrong_rank() {
        let t = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert!(nhwc_to_nchwc(&t, 8).is_err());
        assert!(nchw_to_nchwc(&t, 8).is_err());
    }

    #[test]
    fn block_16_avx512_variant_round_trips() {
        let t = nhwc(2, 3, 3, 32);
        let blocked = nhwc_to_nchwc(&t, 16).unwrap();
        assert_eq!(blocked.shape(), &[2, 2, 3, 3, 16]);
        assert_eq!(blocked.layout(), Layout::NCHWc { block: 16 });
        let back = nchwc_to_nhwc(&blocked, 32).unwrap();
        assert_eq!(back.data(), t.data());
    }

    // Helper: build NCHW tensor with a ramp so every (n,c,h,w) lane
    // has a unique value — transpose bugs surface as data mismatches.
    fn nchw_ramp(n: usize, c: usize, h: usize, w: usize) -> Tensor {
        let mut data = vec![0.0f32; n * c * h * w];
        for (i, v) in data.iter_mut().enumerate() {
            *v = (i as f32) * 0.013 + 0.25;
        }
        Tensor::from_vec(vec![n, c, h, w], data)
            .unwrap()
            .with_layout(Layout::NCHW)
    }

    #[test]
    fn nchw_to_nhwc_fast_matches_generic_permute_aligned() {
        // Tracker DW Conv input shape — primary fast path target.
        let t = nchw_ramp(1, 320, 16, 16);
        let fast = nchw_to_nhwc_fast(&t).unwrap();
        let slow = t.permute(&[0, 2, 3, 1]).unwrap();
        assert_eq!(fast.shape(), &[1, 16, 16, 320]);
        assert_eq!(fast.layout(), Layout::NHWC);
        assert_eq!(fast.data(), slow.data(), "nchw_to_nhwc_fast mismatch");
    }

    #[test]
    fn nchw_to_nhwc_fast_multi_batch() {
        let t = nchw_ramp(2, 32, 4, 4);
        let fast = nchw_to_nhwc_fast(&t).unwrap();
        let slow = t.permute(&[0, 2, 3, 1]).unwrap();
        assert_eq!(fast.data(), slow.data());
    }

    #[test]
    fn nchw_to_nhwc_fast_falls_back_when_c_not_8aligned() {
        // c=3 — first-layer RGB Conv input. Fast path must reject.
        let t = nchw_ramp(1, 3, 8, 8);
        let fast = nchw_to_nhwc_fast(&t).unwrap();
        let slow = t.permute(&[0, 2, 3, 1]).unwrap();
        assert_eq!(fast.shape(), &[1, 8, 8, 3]);
        assert_eq!(fast.data(), slow.data());
    }

    #[test]
    fn nchw_to_nhwc_fast_falls_back_when_hw_not_8aligned() {
        // hw = 7*7 = 49 — not multiple of 8.
        let t = nchw_ramp(1, 16, 7, 7);
        let fast = nchw_to_nhwc_fast(&t).unwrap();
        let slow = t.permute(&[0, 2, 3, 1]).unwrap();
        assert_eq!(fast.data(), slow.data());
    }

    #[test]
    fn nchw_to_nhwc_fast_large() {
        // 1×96×32×32 = 98304 floats — exercise many blocks.
        let t = nchw_ramp(1, 96, 32, 32);
        let fast = nchw_to_nhwc_fast(&t).unwrap();
        let slow = t.permute(&[0, 2, 3, 1]).unwrap();
        assert_eq!(fast.data(), slow.data());
    }

    #[test]
    fn nchw_to_nhwc_fast_tracker_320_16_16() {
        // Exactly the hot tracker shape.
        let t = nchw_ramp(1, 320, 16, 16);
        let fast = nchw_to_nhwc_fast(&t).unwrap();
        let slow = t.permute(&[0, 2, 3, 1]).unwrap();
        assert_eq!(fast.data(), slow.data());
    }
}
