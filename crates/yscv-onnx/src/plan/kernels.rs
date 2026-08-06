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

use rustc_hash::{FxHashMap, FxHashSet};
use yscv_tensor::Tensor;

use super::ConvParams;
use crate::loader::OnnxNode;
use crate::runner::conv_kernel::ConvKernel;

/// Weight geometry as the conv dispatch reads it, accounting for the layouts
/// the loader pre-permutes weights into.
struct WeightGeometry {
    out_channels: usize,
    in_per_group: usize,
    kernel_h: usize,
    kernel_w: usize,
    /// Depthwise weight already in `[KH, KW, C, dm]`.
    dw_khwc: bool,
}

fn weight_geometry(
    name: &str,
    shape: &[usize],
    group: usize,
    khwc: &FxHashSet<String>,
    dw_khwc: &FxHashSet<String>,
    group_khwc: &FxHashSet<String>,
) -> Option<WeightGeometry> {
    if shape.len() != 4 {
        return None;
    }
    let is_dw_khwc = group > 1 && dw_khwc.contains(name);
    let is_group_khwc = group > 1 && group_khwc.contains(name);
    let (out_channels, in_per_group, kernel_h, kernel_w) = if khwc.contains(name) {
        (shape[3], shape[2], shape[0], shape[1])
    } else if is_dw_khwc {
        (shape[2].saturating_mul(shape[3]), 1, shape[0], shape[1])
    } else if is_group_khwc {
        (shape[0], shape[3], shape[1], shape[2])
    } else {
        (shape[0], shape[1], shape[2], shape[3])
    };
    Some(WeightGeometry {
        out_channels,
        in_per_group,
        kernel_h,
        kernel_w,
        dw_khwc: is_dw_khwc,
    })
}

/// The `yscv-kernels` entry point each Conv node will dispatch to, indexed by
/// node index. `None` for nodes that are not Convs, or whose weight is not a
/// constant the plan can inspect.
///
/// Does not cover the Apple BNNS path: that one is chosen before the input is
/// converted to NHWC, from whether the input happens to already be NCHW, so it
/// stays a runtime pre-check in `exec_conv_with_params`.
pub(crate) fn resolve_conv_kernels(
    nodes: &[OnnxNode],
    conv_params: &[Option<ConvParams>],
    initializers: &FxHashMap<String, Tensor>,
    khwc: &FxHashSet<String>,
    dw_khwc: &FxHashSet<String>,
    group_khwc: &FxHashSet<String>,
    prepacked: &FxHashMap<String, std::sync::Arc<yscv_kernels::PackedB>>,
) -> Vec<Option<ConvKernel>> {
    nodes
        .iter()
        .enumerate()
        .map(|(idx, node)| {
            let cp = conv_params.get(idx).and_then(|o| o.as_ref())?;
            let w_name = node.inputs.get(1)?;
            let shape = initializers.get(w_name)?.shape();
            let g = weight_geometry(w_name, shape, cp.group, khwc, dw_khwc, group_khwc)?;
            Some(select(cp, &g, prepacked.contains_key(w_name)))
        })
        .collect()
}

/// The dispatch decision itself, over values the plan already holds.
///
/// Mirrors `conv_compute_nhwc`'s branch structure exactly; the runner asserts
/// the two agree on every Conv it executes in debug builds.
fn select(cp: &ConvParams, g: &WeightGeometry, has_prepack: bool) -> ConvKernel {
    let group = cp.group;

    if group == 1 {
        // aarch64 indirect 3×3. `YSCV_INDIRECT_MAX_COUT` defaults to 0, which
        // makes the ceiling unsatisfiable and the path off — it is read here
        // for the same reason it is read there, so the two stay in step.
        #[cfg(target_arch = "aarch64")]
        {
            // With group == 1 the weight's I/G *is* the input channel count, so
            // the first-layer RGB case is knowable without shape inference.
            let is_first_layer_3ch = g.in_per_group == 3 && cp.stride_h == 2 && cp.stride_w == 2;
            if g.kernel_h == 3
                && g.kernel_w == 3
                && !cfg!(miri)
                && !is_first_layer_3ch
                && g.out_channels <= crate::runner::conv::tuning::indirect_max_cout()
            {
                return ConvKernel::IndirectNhwc3x3;
            }
        }
        return if cp.has_padding {
            ConvKernel::NhwcPadded
        } else if has_prepack {
            ConvKernel::NhwcGemmPrepacked
        } else {
            ConvKernel::NhwcGemm
        };
    }

    // Depthwise. The runtime writes this as `group == out_channels && group ==
    // input_channels`; for any well-formed Conv the input has `group *
    // in_per_group` channels, so `in_per_group == 1` says the same thing
    // without needing the activation shape.
    if group == g.out_channels && g.in_per_group == 1 {
        let depth_mult = g.out_channels / group;
        if g.kernel_h == 3
            && g.kernel_w == 3
            && cp.stride_h == 1
            && cp.stride_w == 1
            && cp.pad_top == 1
            && cp.pad_left == 1
            && cp.pad_bottom == 1
            && cp.pad_right == 1
            && depth_mult == 1
            && group.is_multiple_of(8)
            && crate::runner::conv::tuning::nchwc_depthwise_enabled()
        {
            return ConvKernel::DepthwiseNchwc3x3;
        }
        let _ = g.dw_khwc;
        return if cp.has_padding {
            ConvKernel::DepthwiseNhwcPadded
        } else {
            ConvKernel::DepthwiseNhwc
        };
    }

    ConvKernel::Grouped
}
