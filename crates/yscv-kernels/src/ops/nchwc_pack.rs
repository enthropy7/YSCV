//! K.1: NCHWc weight pre-packing for session-level zero-copy reuse.
//!
//! Pointwise 1×1 Conv weights reuse the existing `PackedB` format —
//! the NCHWc PW GEMM (K.2) reads the weight as a [IC, OC] matrix which
//! is exactly the shape `pack_b_for_session` already handles.
//!
//! Depthwise 3×3 Conv weights need a new format: [C/block, 3, 3, block]
//! — one contiguous block of `block` channels at each of the 9 kernel
//! positions. `PackedNChwBc` stores this and is handed to the NCHWc DW
//! kernel (K.3) via `Arc<PackedNChwBc>`.
//!
//! Layout conversion (NHWC ↔ NCHWc) for activations lives in `layout.rs`;
//! only weight-specific packing is here.

use std::sync::Arc;

use yscv_tensor::AlignedVec;

/// Pre-packed depthwise 3×3 kernel in NCHWc-blocked layout.
///
/// Source weight shape (NHWC convention, depth-multiplier=1):
/// - NHWC KHWC: `[KH=3, KW=3, C, 1]` → stored as `weight[ky][kx][c]`
///
/// Packed layout: `[C_blocks, KH, KW, block]` where
/// `C_blocks = ceil(C / block)`, with zero-padded tail block lanes.
///
/// The DW inner kernel for block cb reads:
/// `data[cb * KH * KW * block .. (cb+1) * KH * KW * block]`
/// with `kernel_c_stride = block` and `kernel_row_stride = KW * block`.
pub struct PackedNChwBc {
    /// Packed data in [C_blocks, KH, KW, block] order (f32).
    data: AlignedVec<f32>,
    /// Actual channel count (before padding to block multiple).
    pub channels: usize,
    /// Kernel spatial extent.
    pub kh: usize,
    pub kw: usize,
    /// Channel block size (16 for AVX-512, 8 for AVX2/NEON).
    pub block: usize,
    /// Number of complete+partial channel blocks = ceil(channels / block).
    pub c_blocks: usize,
}

impl std::fmt::Debug for PackedNChwBc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PackedNChwBc {{ channels: {}, kh: {}, kw: {}, block: {}, c_blocks: {} }}",
            self.channels, self.kh, self.kw, self.block, self.c_blocks
        )
    }
}

impl PackedNChwBc {
    /// Pointer to the start of channel block `cb`'s kernel weights.
    ///
    /// The returned slice covers `KH * KW * block` floats in
    /// row-major [KH, KW, block] order (kernel_c_stride = block,
    /// kernel_row_stride = KW * block).
    #[inline]
    pub fn channel_block(&self, cb: usize) -> &[f32] {
        let step = self.kh * self.kw * self.block;
        let off = cb * step;
        &self.data.as_slice()[off..off + step]
    }

    /// Kernel-column stride in floats (distance from `weight[ky][kx][b]` to
    /// `weight[ky][kx+1][b]`) = block.
    #[inline]
    pub fn kernel_c_stride(&self) -> usize {
        self.block
    }

    /// Kernel-row stride in floats (distance from `weight[ky][kx][b]` to
    /// `weight[ky+1][kx][b]`) = KW * block.
    #[inline]
    pub fn kernel_row_stride(&self) -> usize {
        self.kw * self.block
    }

    /// Whether dimensions match (guard against stale prepack).
    #[inline]
    pub fn matches(&self, channels: usize, kh: usize, kw: usize) -> bool {
        self.channels == channels && self.kh == kh && self.kw == kw
    }

    /// Raw packed weight data.  Layout: `[c_blocks * kh * kw * block]`.
    /// Offset of channel block `cb`: `cb * kh * kw * block`.
    #[inline]
    pub fn raw_data(&self) -> &[f32] {
        self.data.as_slice()
    }
}

/// Pre-pack a depthwise Conv weight into NCHWc-blocked layout.
///
/// `weight_nhwc` is `[KH, KW, C, depth_mult]` in NHWC convention. Only
/// `depth_mult = 1` is supported (standard depthwise).
///
/// `block` must be 8 or 16. Returns `Arc<PackedNChwBc>` suitable for passing
/// to the NCHWc DW kernel at inference time.
pub fn pack_dw_nchwc_for_session(
    weight_nhwc: &[f32],
    c: usize,
    kh: usize,
    kw: usize,
    block: usize,
) -> Arc<PackedNChwBc> {
    debug_assert!(block == 8 || block == 16, "block must be 8 or 16");
    debug_assert_eq!(
        weight_nhwc.len(),
        kh * kw * c,
        "weight_nhwc len mismatch (depth_mult=1 expected)"
    );

    let c_blocks = c.div_ceil(block);
    let packed_len = c_blocks * kh * kw * block;
    let mut data = AlignedVec::<f32>::calloc(packed_len);
    let dst = data.as_mut_slice();

    // Source layout: weight_nhwc[ky * kw * c + kx * c + ci] (row-major, last dim = channel).
    // Dest layout: dst[cb * kh * kw * block + ky * kw * block + kx * block + bi].
    for cb in 0..c_blocks {
        let c_start = cb * block;
        for ky in 0..kh {
            for kx in 0..kw {
                let src_base = ky * kw * c + kx * c + c_start;
                let dst_base = cb * kh * kw * block + ky * kw * block + kx * block;
                let lanes = block.min(c.saturating_sub(c_start));
                dst[dst_base..dst_base + lanes]
                    .copy_from_slice(&weight_nhwc[src_base..src_base + lanes]);
                // Tail lanes remain zero (calloc).
            }
        }
    }

    Arc::new(PackedNChwBc {
        data,
        channels: c,
        kh,
        kw,
        block,
        c_blocks,
    })
}

/// Runtime block-size selection: 16 on x86_64 with AVX-512F, 8 otherwise.
pub fn runtime_nchwc_block() -> usize {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if crate::host_cpu().features.avx512f {
        return 16;
    }
    8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nhwc_dw_weight(kh: usize, kw: usize, c: usize) -> Vec<f32> {
        (0..kh * kw * c).map(|i| i as f32).collect()
    }

    #[test]
    fn pack_dw_round_trip_3x3_c16_block8() {
        let c = 16;
        let w = nhwc_dw_weight(3, 3, c);
        let packed = pack_dw_nchwc_for_session(&w, c, 3, 3, 8);
        assert_eq!(packed.channels, 16);
        assert_eq!(packed.kh, 3);
        assert_eq!(packed.kw, 3);
        assert_eq!(packed.block, 8);
        assert_eq!(packed.c_blocks, 2);

        // For cb=0, ky=0, kx=0: should be weight[0,0,0..8].
        let cb0 = packed.channel_block(0);
        assert_eq!(&cb0[0..8], &w[0..8]);
        // For cb=1, ky=0, kx=0: weight[0,0,8..16].
        let cb1 = packed.channel_block(1);
        assert_eq!(&cb1[0..8], &w[8..16]);
    }

    #[test]
    fn pack_dw_round_trip_3x3_c16_block16() {
        let c = 16;
        let w = nhwc_dw_weight(3, 3, c);
        let packed = pack_dw_nchwc_for_session(&w, c, 3, 3, 16);
        assert_eq!(packed.c_blocks, 1);
        let cb0 = packed.channel_block(0);
        // cb=0, ky=0, kx=0: first 16 channels = w[0..16]
        assert_eq!(&cb0[0..16], &w[0..16]);
        // ky=0, kx=1: w[16..32]
        assert_eq!(&cb0[16..32], &w[c..c + 16]);
    }

    #[test]
    fn pack_dw_tail_padding_zeros() {
        let c = 20; // not divisible by 16
        let w = nhwc_dw_weight(3, 3, c);
        let packed = pack_dw_nchwc_for_session(&w, c, 3, 3, 16);
        assert_eq!(packed.c_blocks, 2);
        let cb1 = packed.channel_block(1); // channels 16..19, padded to 16
        // ky=0, kx=0: packed cb1[0..4] = w[16..20] = [16.0, 17.0, 18.0, 19.0]
        assert_eq!(&cb1[0..4], &w[16..20]);
        // tail lanes 4..16 must be zero
        assert!(cb1[4..16].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn pack_dw_kernel_strides() {
        let c = 16;
        let w = nhwc_dw_weight(3, 3, c);
        let packed = pack_dw_nchwc_for_session(&w, c, 3, 3, 16);
        assert_eq!(packed.kernel_c_stride(), 16);
        assert_eq!(packed.kernel_row_stride(), 3 * 16);
    }

    #[test]
    fn pack_dw_matches_guard() {
        let packed = pack_dw_nchwc_for_session(&nhwc_dw_weight(3, 3, 32), 32, 3, 3, 16);
        assert!(packed.matches(32, 3, 3));
        assert!(!packed.matches(16, 3, 3));
        assert!(!packed.matches(32, 5, 5));
    }

    #[test]
    fn pack_dw_full_spatial_correctness() {
        // Verify every (cb, ky, kx, bi) value matches the source.
        let c = 48;
        let w = nhwc_dw_weight(3, 3, c);
        let block = 16;
        let packed = pack_dw_nchwc_for_session(&w, c, 3, 3, block);
        for cb in 0..packed.c_blocks {
            let packed_block = packed.channel_block(cb);
            for ky in 0..3usize {
                for kx in 0..3usize {
                    for bi in 0..block {
                        let ci = cb * block + bi;
                        let dst_idx = ky * 3 * block + kx * block + bi;
                        let src_idx = ky * 3 * c + kx * c + ci;
                        let expected = if ci < c { w[src_idx] } else { 0.0 };
                        assert_eq!(
                            packed_block[dst_idx], expected,
                            "mismatch at cb={cb} ky={ky} kx={kx} bi={bi}"
                        );
                    }
                }
            }
        }
    }
}
