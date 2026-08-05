//! Physical layout of a convolution weight.
//!
//! `load_onnx_model` pre-permutes Conv weights so the NHWC kernels can read
//! them directly, and records which permutation it applied in three side sets
//! on `OnnxModel` (`khwc_weights`, `dw_khwc_weights`, `group_khwc_weights`).
//! Any pass that rewrites a weight has to know which one, because the output
//! channel lives at a different axis in each.
//!
//! That is a layer-3 concern leaking into layer 1 — the permutation exists to
//! suit a kernel, and choosing it belongs with kernel selection, not with
//! loading. Until it moves there, the IR carries the tag so the passes have one
//! place to ask instead of three duplicated branch chains. **This module is
//! expected to be deleted** once the permute moves into plan construction and
//! passes see logical OIHW throughout.

/// How a convolution weight is physically laid out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum WeightLayout {
    /// ONNX-native `[O, I/group, KH, KW]`. Also the layout of any weight the
    /// loader chose not to permute.
    #[default]
    Oihw,
    /// group=1, permuted to `[KH, KW, I, O]`.
    Khwc,
    /// Depthwise, permuted to `[KH, KW, C, depth_multiplier]`.
    DepthwiseKhwc,
    /// Grouped, permuted to `[O, KH, KW, I/group]`.
    GroupKhwc,
}

impl WeightLayout {
    /// Number of output channels, given the weight's shape.
    ///
    /// Returns `None` for a shape too short to interpret, which is malformed
    /// input rather than something a pass should guess at.
    pub(crate) fn out_channels(self, shape: &[usize]) -> Option<usize> {
        if shape.len() < 4 {
            return None;
        }
        Some(match self {
            WeightLayout::Oihw | WeightLayout::GroupKhwc => shape[0],
            WeightLayout::Khwc => shape[3],
            // `[KH, KW, C, dm]`: one output channel per (C, dm) pair, but the
            // per-channel parameters BatchNormalization folds in are indexed by
            // C, so C is what callers scale by.
            WeightLayout::DepthwiseKhwc => shape[2],
        })
    }

    /// Output channel owning the element at `flat_index` in the weight buffer.
    ///
    /// This is the stride the folding passes need: absorbing a per-channel
    /// scale means multiplying every element by its own channel's factor, and
    /// which elements those are depends entirely on the layout.
    pub(crate) fn channel_of(
        self,
        flat_index: usize,
        shape: &[usize],
        out_channels: usize,
    ) -> usize {
        debug_assert!(out_channels > 0, "out_channels must be non-zero");
        match self {
            // Output channel is the fastest-varying axis.
            WeightLayout::Khwc => flat_index % out_channels,
            // `[KH, KW, C, dm]` — dm elements per channel, contiguous.
            WeightLayout::DepthwiseKhwc => {
                let dm = shape.get(3).copied().unwrap_or(1).max(1);
                (flat_index / dm) % out_channels
            }
            // Output channel is the slowest-varying axis, so each owns one
            // contiguous block.
            WeightLayout::Oihw | WeightLayout::GroupKhwc => {
                let per_channel = shape.iter().product::<usize>() / out_channels;
                flat_index.checked_div(per_channel).unwrap_or(0)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn out_channels_reads_the_right_axis() {
        assert_eq!(WeightLayout::Oihw.out_channels(&[8, 3, 3, 3]), Some(8));
        assert_eq!(WeightLayout::Khwc.out_channels(&[3, 3, 3, 8]), Some(8));
        assert_eq!(
            WeightLayout::DepthwiseKhwc.out_channels(&[3, 3, 8, 1]),
            Some(8)
        );
        assert_eq!(WeightLayout::GroupKhwc.out_channels(&[8, 3, 3, 2]), Some(8));
    }

    #[test]
    fn out_channels_declines_on_a_short_shape() {
        assert_eq!(WeightLayout::Oihw.out_channels(&[8, 3, 3]), None);
    }

    /// Every element must map to exactly one channel, and each channel must own
    /// its share — otherwise a folded scale is applied to the wrong weights.
    #[test]
    fn channel_of_partitions_the_buffer_evenly() {
        let cases: &[(WeightLayout, &[usize])] = &[
            (WeightLayout::Oihw, &[4, 2, 3, 3]),
            (WeightLayout::Khwc, &[3, 3, 2, 4]),
            (WeightLayout::DepthwiseKhwc, &[3, 3, 4, 2]),
            (WeightLayout::GroupKhwc, &[4, 3, 3, 2]),
        ];
        for &(layout, shape) in cases {
            let out_channels = layout.out_channels(shape).expect("shape is rank 4");
            let total: usize = shape.iter().product();
            let mut counts = vec![0usize; out_channels];
            for i in 0..total {
                counts[layout.channel_of(i, shape, out_channels)] += 1;
            }
            assert!(
                counts.iter().all(|&c| c == total / out_channels),
                "{layout:?} on {shape:?} split unevenly: {counts:?}"
            );
        }
    }

    /// The depthwise layout groups `dm` consecutive elements per channel.
    #[test]
    fn depthwise_channel_of_respects_depth_multiplier() {
        let shape = [1, 1, 3, 2];
        let layout = WeightLayout::DepthwiseKhwc;
        let channels: Vec<usize> = (0..6).map(|i| layout.channel_of(i, &shape, 3)).collect();
        assert_eq!(channels, vec![0, 0, 1, 1, 2, 2]);
    }
}
