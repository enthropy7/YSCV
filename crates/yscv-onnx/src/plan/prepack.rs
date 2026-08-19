//! Phase 3: convolution weights, laid out for the kernel that reads them.
//!
//! The NHWC kernels want their weights channel-last, so a Conv weight gets
//! permuted once at load time rather than on every inference. Which permutation
//! depends on which kernel the Conv dispatches to, so the choice belongs here,
//! beside kernel selection.
//!
//! It used to happen in `load_onnx_model`, which is what made the tracker model
//! fail to load. The permuted bytes replaced the initializer, so a name-keyed
//! side table was the only record of what a Conv weight now meant — and a pass
//! that renames a weight (constant-folding an `Identity` produces an alias)
//! dropped the record, leaving a `[KH, KW, C, 1]` tensor to be read as
//! `[O, I, KH, KW]`. Every pass that rewrote a weight had to consult that table
//! first, and the IR grew a `WeightLayout` tag to give them one place to ask.
//!
//! Keeping the permuted copy here instead of writing it back means
//! `initializers` is ONNX-native OIHW throughout: passes read what the model
//! said, no tag needed, and a renamed weight is re-examined from scratch the
//! next time the plan is built. The cost is holding both copies of each conv
//! weight for the model's lifetime — `Tensor` is `Arc`-backed, so this is one
//! extra buffer per weight at load, not per inference.

use rustc_hash::FxHashMap;
use yscv_tensor::Tensor;

use crate::attr::Attr;
use crate::loader::{OnnxAttribute, OnnxNode};
use crate::plan::NodeKind;
use crate::runner::conv_kernel::ConvKernel;

/// How a convolution weight is physically laid out for its kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum ConvWeightLayout {
    /// ONNX-native `[O, I/group, KH, KW]`. Also the layout of any weight the
    /// plan chose not to permute, which the runtime then repacks per inference.
    #[default]
    Oihw,
    /// group=1, permuted to `[KH, KW, I, O]`.
    Khwc,
    /// Depthwise with depth multiplier 1, packed to `[KH, KW, C, 1]`.
    DepthwiseKhwc,
    /// Grouped, packed to `[O, KH, KW, I/group]`.
    GroupKhwc,
}

/// The permuted weights and what each one was permuted into, by slot id.
///
/// Slot id rather than name deliberately: the name-keyed version of this table
/// is the bug described in the module docs. Ids come from the same sweep that
/// resolves node inputs, so a weight and the Conv reading it cannot disagree.
pub(super) struct PrepackedConvWeights {
    /// Permuted tensor per slot; `None` where the weight is untouched.
    pub(super) tensors: Vec<Option<Tensor>>,
    /// Layout per slot. `Oihw` for everything that is not a permuted Conv
    /// weight, including non-weights.
    pub(super) layouts: Vec<ConvWeightLayout>,
}

impl PrepackedConvWeights {
    /// The tensor a Conv reading `name` will actually see: the permuted copy if
    /// there is one, otherwise the initializer.
    ///
    /// Plan phases that inspect a weight's *bytes or shape as the kernel sees
    /// them* — weight packing, above all — must go through this. Phases
    /// reasoning about the model as written want the initializer directly.
    pub(super) fn weight_for<'a>(
        &'a self,
        name: &str,
        name_to_id: &FxHashMap<String, usize>,
        initializers: &'a FxHashMap<String, Tensor>,
    ) -> Option<&'a Tensor> {
        name_to_id
            .get(name)
            .and_then(|&id| self.tensors.get(id))
            .and_then(Option::as_ref)
            .or_else(|| initializers.get(name))
    }

    pub(super) fn layout_of(
        &self,
        name: &str,
        name_to_id: &FxHashMap<String, usize>,
    ) -> ConvWeightLayout {
        name_to_id
            .get(name)
            .and_then(|&id| self.layouts.get(id))
            .copied()
            .unwrap_or_default()
    }
}

/// Depthwise and grouped weights keep their ONNX-native layout on accelerator
/// builds: those backends' CPU fallback and dispatch paths read OIHW, and the
/// accelerator packs its own. Keeping one loader able to feed both is why this
/// is a build-time choice rather than a per-kernel one — the remaining reason
/// [`ConvWeightLayout`] has to be recorded at all, instead of being a pure
/// function of the selected kernel.
const PREPACK_GROUPED: bool = !cfg!(any(feature = "metal-backend", feature = "gpu"));

/// The layout a kernel reads its weight in.
///
/// Total, and the reason layout does not have to be selected separately from
/// the kernel: every entry point falls into one of three families, and each
/// family reads exactly one layout.
fn layout_read_by(kernel: ConvKernel) -> Option<ConvWeightLayout> {
    Some(match kernel {
        #[cfg(target_arch = "aarch64")]
        ConvKernel::IndirectNhwc3x3 => ConvWeightLayout::Khwc,
        ConvKernel::NhwcPadded | ConvKernel::NhwcGemmPrepacked | ConvKernel::NhwcGemm => {
            ConvWeightLayout::Khwc
        }
        ConvKernel::DepthwiseNchwc3x3
        | ConvKernel::DepthwiseNhwcPadded
        | ConvKernel::DepthwiseNhwc => ConvWeightLayout::DepthwiseKhwc,
        ConvKernel::Grouped => ConvWeightLayout::GroupKhwc,
        // Chosen at dispatch from whether the input happens to be NCHW, not
        // resolved here, so it never appears in the plan's kernel table.
        #[cfg(target_os = "macos")]
        ConvKernel::BnnsNchw => return None,
    })
}

/// Holds the prepacked layout and the resolved kernel together.
///
/// The two are decided independently — the layout from the weight's shape and
/// group attribute, the kernel from [`super::ConvParams`] — but they answer the
/// same question, and a Conv handed a weight in a layout its kernel does not
/// read is the whole failure mode this module exists to prevent. Checking it
/// where both are in scope is what keeps the two from drifting apart as kernels
/// are added.
///
/// One-directional: `Oihw` pairs with any kernel, because it also means "no
/// prepack happened" — the weight was not constant, not rank 4, or this is an
/// accelerator build. Those Convs repack at run time and are correct either
/// way. What must never happen is a *non*-`Oihw` layout under a kernel that
/// reads a different one.
pub(super) fn debug_assert_layouts_match_kernels(
    nodes: &[OnnxNode],
    conv_kernels: &[Option<ConvKernel>],
    layout_of: impl Fn(&str) -> ConvWeightLayout,
) {
    if !cfg!(debug_assertions) {
        return;
    }
    for (idx, node) in nodes.iter().enumerate() {
        let Some(Some(kernel)) = conv_kernels.get(idx).copied() else {
            continue;
        };
        let Some(weight_name) = node.inputs.get(1) else {
            continue;
        };
        let layout = layout_of(weight_name);
        if layout == ConvWeightLayout::Oihw {
            continue;
        }
        assert_eq!(
            Some(layout),
            layout_read_by(kernel),
            "node {idx} ({}) has its weight `{weight_name}` prepacked as \
             {layout:?}, but the plan dispatches it to {kernel:?}, which reads \
             {:?}",
            node.name,
            layout_read_by(kernel),
        );
    }
}

/// Permutes every constant Conv weight into the layout its kernel reads.
///
/// Weights that cannot be permuted — wrong rank, or a permutation the tensor
/// layer rejects — are left alone and reported as [`ConvWeightLayout::Oihw`].
/// That is a slower path rather than a wrong one: the runtime repacks per
/// inference for exactly this case (see `repack_depthwise_kernel_once`), so
/// declining here cannot produce a misread weight.
///
/// Takes [`NodeKind`] rather than testing the op-type string, because by the
/// time the plan is built `fuse_conv_relu` has renamed some Convs to
/// `Conv_Relu`. Matching `"Conv"` here silently dropped the prepack for every
/// one of them — invisible while the permute lived in the loader, which ran
/// before the rename.
pub(super) fn prepack_conv_weights(
    nodes: &[OnnxNode],
    node_kinds: &[NodeKind],
    initializers: &FxHashMap<String, Tensor>,
    name_to_id: &FxHashMap<String, usize>,
) -> PrepackedConvWeights {
    let mut out = PrepackedConvWeights {
        tensors: vec![None; name_to_id.len()],
        layouts: vec![ConvWeightLayout::Oihw; name_to_id.len()],
    };

    for (idx, node) in nodes.iter().enumerate() {
        let is_conv = matches!(
            node_kinds.get(idx),
            Some(
                NodeKind::Conv | NodeKind::ConvRelu | NodeKind::ConvSilu | NodeKind::ConvHardSwish
            )
        );
        if !is_conv || node.inputs.len() < 2 {
            continue;
        }
        let weight_name = &node.inputs[1];
        let Some(&id) = name_to_id.get(weight_name.as_str()) else {
            continue;
        };
        // A weight two Convs share is permuted once, on first sight. The two
        // agree on the layout because it follows from the weight's own shape
        // and the group attribute, which they must also agree on to be reading
        // the same tensor.
        if out.tensors[id].is_some() {
            continue;
        }
        let Some(weight) = initializers.get(weight_name) else {
            continue;
        };
        let group = match node.attributes.get(&Attr::Group) {
            Some(OnnxAttribute::Int(v)) => (*v).max(1) as usize,
            _ => 1,
        };
        let Some((layout, permuted)) = permute_for_kernel(weight, group) else {
            continue;
        };
        out.tensors[id] = Some(permuted);
        out.layouts[id] = layout;
    }

    out
}

/// The layout this weight's kernel wants, and the weight in it.
///
/// The three cases correspond exactly to the three families
/// `plan::kernels::select_conv_kernel` dispatches to — group-1, depthwise, and
/// general grouped — which is why layout does not need to be selected
/// separately from the kernel. `debug_assert_layout_matches_kernel` in
/// `plan::build` holds the two together.
fn permute_for_kernel(weight: &Tensor, group: usize) -> Option<(ConvWeightLayout, Tensor)> {
    let shape = weight.shape();
    if shape.len() != 4 {
        return None;
    }

    if group == 1 {
        // `[O, I, KH, KW]` -> `[KH, KW, I, O]`.
        return weight
            .permute(&[2, 3, 1, 0])
            .ok()
            .map(|t| (ConvWeightLayout::Khwc, t));
    }
    if !PREPACK_GROUPED {
        return None;
    }

    let (o_ch, i_per_g, kh, kw) = (shape[0], shape[1], shape[2], shape[3]);
    let data = weight.data();

    // Depthwise, `[C, 1, KH, KW]` -> `[KH, KW, C, 1]`. The CPU depthwise fast
    // path handles depth multiplier 1 only; anything else is a grouped conv as
    // far as the kernels are concerned.
    if i_per_g == 1 && o_ch == group {
        let mut packed = vec![0.0f32; kh * kw * group];
        for oc in 0..o_ch {
            for ki in 0..kh {
                for kj in 0..kw {
                    packed[(ki * kw + kj) * group + oc] = data[(oc * kh + ki) * kw + kj];
                }
            }
        }
        return Tensor::from_vec(vec![kh, kw, group, 1], packed)
            .ok()
            .map(|t| (ConvWeightLayout::DepthwiseKhwc, t));
    }

    // Grouped, `[O, I/G, KH, KW]` -> `[O, KH, KW, I/G]`.
    let mut packed = vec![0.0f32; o_ch * kh * kw * i_per_g];
    for oc in 0..o_ch {
        for ki in 0..kh {
            for kj in 0..kw {
                for ci in 0..i_per_g {
                    let src = ((oc * i_per_g + ci) * kh + ki) * kw + kj;
                    packed[((oc * kh + ki) * kw + kj) * i_per_g + ci] = data[src];
                }
            }
        }
    }
    Tensor::from_vec(vec![o_ch, kh, kw, i_per_g], packed)
        .ok()
        .map(|t| (ConvWeightLayout::GroupKhwc, t))
}
