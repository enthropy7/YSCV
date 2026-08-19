//! Kernel selection resolved at plan time.
//!
//! A single ONNX `Conv` resolves to one of several `yscv-kernels` entry points
//! depending on group, kernel size, padding, weight layout and whether a
//! load-time prepack exists. That choice used to be made inside the dispatch on
//! every inference, and was only *observable* afterwards, through a thread-local
//! cell the profiler read back.
//!
//! Every input to it is fixed once the plan is built. Resolving it here makes
//! the plan say which kernel each Conv runs before anything runs, which is what
//! a cost model would need to compare alternatives, and what lets a new kernel
//! be introduced by adding a case to one table rather than a branch to a hot
//! function.

use rustc_hash::FxHashMap;
use yscv_tensor::Tensor;

use super::ConvParams;
use crate::loader::OnnxNode;
use crate::runner::conv_kernel::ConvKernel;

/// Everything the entry-point choice depends on, in the form both callers can
/// produce: the plan from `ConvParams` plus the initializer's shape, the
/// dispatch from the scalars it has already resolved.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ConvShape {
    pub group: usize,
    pub has_padding: bool,
    pub stride_h: usize,
    pub stride_w: usize,
    pub pad_top: usize,
    pub pad_left: usize,
    pub pad_bottom: usize,
    pub pad_right: usize,
    pub out_channels: usize,
    pub in_per_group: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    /// A load-time prepacked B exists for this weight.
    pub has_prepack: bool,
}

/// The `yscv-kernels` entry point each Conv node will dispatch to, indexed by
/// node index. `None` for nodes that are not Convs, or whose weight is not a
/// constant the plan can inspect.
///
/// Reads the weight as ONNX-native `[O, I/G, KH, KW]`. Prepacking runs after
/// this and keeps its permuted copies to itself, so there is no layout to
/// dispatch on — which is also why the kernel a Conv gets does not depend on
/// whether its weight was prepacked.
///
/// Does not cover the Apple BNNS path: that one is chosen before the input is
/// converted to NHWC, from whether the input happens to already be NCHW, so it
/// stays a runtime pre-check in `exec_conv_with_params`.
pub(crate) fn resolve_conv_kernels(
    nodes: &[OnnxNode],
    conv_params: &[Option<ConvParams>],
    initializers: &FxHashMap<String, Tensor>,
    prepacked: &FxHashMap<String, std::sync::Arc<yscv_kernels::PackedB>>,
) -> Vec<Option<ConvKernel>> {
    nodes
        .iter()
        .enumerate()
        .map(|(idx, node)| {
            let cp = conv_params.get(idx).and_then(|o| o.as_ref())?;
            let w_name = node.inputs.get(1)?;
            let shape = initializers.get(w_name)?.shape();
            if shape.len() != 4 {
                return None;
            }
            Some(select_conv_kernel(&ConvShape {
                group: cp.group,
                has_padding: cp.has_padding,
                stride_h: cp.stride_h,
                stride_w: cp.stride_w,
                pad_top: cp.pad_top,
                pad_left: cp.pad_left,
                pad_bottom: cp.pad_bottom,
                pad_right: cp.pad_right,
                out_channels: shape[0],
                in_per_group: shape[1],
                kernel_h: shape[2],
                kernel_w: shape[3],
                has_prepack: prepacked.contains_key(w_name),
            }))
        })
        .collect()
}

/// The dispatch decision itself, over values both the plan and the runtime can
/// produce.
///
/// The single place a Conv entry point's *condition* lives. Adding a kernel
/// means a variant on [`ConvKernel`], an arm here, and the call in
/// `conv_compute_nhwc` — not a predicate duplicated between plan and dispatch
/// that can drift apart.
pub(crate) fn select_conv_kernel(s: &ConvShape) -> ConvKernel {
    if s.group == 1 {
        // aarch64 indirect 3×3. `YSCV_INDIRECT_MAX_COUT` defaults to 0, which
        // makes the ceiling unsatisfiable and the path off by default.
        #[cfg(target_arch = "aarch64")]
        {
            // With group == 1 the weight's I/G *is* the input channel count, so
            // the first-layer RGB case is knowable without shape inference.
            let is_first_layer_3ch = s.in_per_group == 3 && s.stride_h == 2 && s.stride_w == 2;
            if s.kernel_h == 3
                && s.kernel_w == 3
                && !cfg!(miri)
                && !is_first_layer_3ch
                && s.out_channels <= crate::runner::conv::tuning::indirect_max_cout()
            {
                return ConvKernel::IndirectNhwc3x3;
            }
        }
        return if s.has_padding {
            ConvKernel::NhwcPadded
        } else if s.has_prepack {
            ConvKernel::NhwcGemmPrepacked
        } else {
            ConvKernel::NhwcGemm
        };
    }

    // Depthwise. The dispatch writes this as `group == out_channels && group ==
    // input_channels`; for any well-formed Conv the input has `group *
    // in_per_group` channels, so `in_per_group == 1` says the same thing
    // without needing the activation shape.
    if s.group == s.out_channels && s.in_per_group == 1 {
        let depth_mult = s.out_channels / s.group;
        if s.kernel_h == 3
            && s.kernel_w == 3
            && s.stride_h == 1
            && s.stride_w == 1
            && s.pad_top == 1
            && s.pad_left == 1
            && s.pad_bottom == 1
            && s.pad_right == 1
            && depth_mult == 1
            && s.group.is_multiple_of(8)
            && crate::runner::conv::tuning::nchwc_depthwise_enabled()
        {
            return ConvKernel::DepthwiseNchwc3x3;
        }
        return if s.has_padding {
            ConvKernel::DepthwiseNhwcPadded
        } else {
            ConvKernel::DepthwiseNhwc
        };
    }

    ConvKernel::Grouped
}
