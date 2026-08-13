//! Plan construction: classifies each node ([`NodeKind`]), selects fusions
//! ([`NodeAction`]), resolves convolution parameters and prepacks weights.
//!
//! Slot assignment, conv-parameter resolution, layout handoff and kernel
//! selection have moved into their own modules alongside this one. What remains
//! here is the fusion scan — subgraph patterns to [`NodeAction`]s — plus weight
//! prepacking, which is the next thing to lift out.
//!
//! Every matcher below works by dataflow: it walks from a value to the node
//! that reads it, through the `consumers` index built once at the top. None of
//! them depends on where the schedule happened to put a node.

use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};
use yscv_tensor::Tensor;

use crate::attr::Attr;
use crate::loader::{OnnxAttribute, OnnxNode};

use super::*;

pub(crate) fn build_runtime_index(
    inputs: &[String],
    outputs: &[String],
    initializers: &FxHashMap<String, Tensor>,
    nodes: &[OnnxNode],
    khwc_weights: &FxHashSet<String>,
    dw_khwc_weights: &FxHashSet<String>,
    group_khwc_weights: &FxHashSet<String>,
) -> RuntimeModelIndex {
    let SlotIndex {
        name_to_id,
        khwc_weight_ids,
        dw_khwc_weight_ids,
        group_khwc_weight_ids,
        use_counts,
        use_counts_by_id,
        node_kinds,
        node_branches,
        node_input_ids,
        node_output_ids,
    } = assign_slots(
        inputs,
        outputs,
        initializers,
        nodes,
        khwc_weights,
        dw_khwc_weights,
        group_khwc_weights,
    );

    let conv_params = resolve_conv_params(
        nodes,
        &node_kinds,
        initializers,
        khwc_weights,
        dw_khwc_weights,
        group_khwc_weights,
    );

    /// Returns `true` when the Transpose node's `perm` attribute swaps
    /// only the last two axes of a rank-3 tensor (i.e. `[0, 2, 1]`).
    /// Matches the pattern emitted by PyTorch's `.transpose(-2, -1)` on
    /// 3-D tensors — the pattern ORT folds into its
    /// `MatmulTransposeFusion` contrib op.
    fn transpose_perm_is_swap_last_two(node: &OnnxNode) -> bool {
        let perm = match node.attributes.get(&Attr::Perm) {
            Some(OnnxAttribute::Ints(p)) => p,
            _ => return false,
        };
        matches!(perm.as_slice(), [0, 2, 1])
    }

    fn init_scalar(initializers: &FxHashMap<String, Tensor>, name: &str) -> Option<f32> {
        initializers
            .get(name)
            .and_then(|t| t.data().first())
            .copied()
    }

    fn matching_zero_qparams(
        dequant: &OnnxNode,
        quant: &OnnxNode,
        initializers: &FxHashMap<String, Tensor>,
    ) -> bool {
        if dequant.inputs.len() < 3 || quant.inputs.len() < 3 {
            return false;
        }
        let Some(dq_scale) = init_scalar(initializers, &dequant.inputs[1]) else {
            return false;
        };
        let Some(q_scale) = init_scalar(initializers, &quant.inputs[1]) else {
            return false;
        };
        let Some(dq_zp) = init_scalar(initializers, &dequant.inputs[2]) else {
            return false;
        };
        let Some(q_zp) = init_scalar(initializers, &quant.inputs[2]) else {
            return false;
        };
        dq_scale.to_bits() == q_scale.to_bits()
            && dq_zp.to_bits() == 0.0_f32.to_bits()
            && q_zp.to_bits() == 0.0_f32.to_bits()
    }

    /// A DQ->[Relu]->Q boundary is foldable into one i8->i8 rescale pass when
    /// both the DequantizeLinear (activation input) and the QuantizeLinear
    /// (output) are per-tensor — i.e. scalar scale AND zero-point. Per-channel
    /// weight DQ (scale length > 1) fails `init_scalar` and is correctly
    /// excluded. Unlike `matching_zero_qparams` the scales/zps may differ; the
    /// runner does the real rescale.
    fn per_tensor_qdq_boundary(
        dequant: &OnnxNode,
        quant: &OnnxNode,
        initializers: &FxHashMap<String, Tensor>,
    ) -> bool {
        if dequant.inputs.len() < 3 || quant.inputs.len() < 3 {
            return false;
        }
        init_scalar(initializers, &dequant.inputs[1]).is_some()
            && init_scalar(initializers, &dequant.inputs[2]).is_some()
            && init_scalar(initializers, &quant.inputs[1]).is_some()
            && init_scalar(initializers, &quant.inputs[2]).is_some()
    }

    /// Classify a `QLinearConv` by weight initializer shape into the chain
    /// roles we currently fuse. Returns `Some("pw")` for 1×1 group-1 PW,
    /// `Some("dw")` for 3×3/5×5 depthwise (group=c_out=c_in*group), and
    /// `None` otherwise. Mirrors `bench_tracker::qlinear_conv_kind` so the
    /// load-time detector and the static counter agree node-for-node.
    fn qlc_kind(node: &OnnxNode, initializers: &FxHashMap<String, Tensor>) -> Option<&'static str> {
        if node.op_type != "QLinearConv" {
            return None;
        }
        let w_name = node.inputs.get(3)?;
        let weight = initializers.get(w_name)?;
        let shape = weight.shape();
        if shape.len() != 4 {
            return None;
        }
        // The fused INT8 kernels requantize with a single scalar composite scale
        // per convolution, so they are only correct for per-tensor weight
        // quantization. Per-channel weights (w_scale length > 1) must take the
        // per-op path, which applies the composite per output channel. Decline
        // the fusion here — the one choke point every INT8 chain gates on.
        let per_tensor_weight = node
            .inputs
            .get(4)
            .and_then(|s| initializers.get(s))
            .is_some_and(|t| t.data().len() == 1);
        if !per_tensor_weight {
            return None;
        }
        let group = match node.attributes.get(&Attr::Group) {
            Some(OnnxAttribute::Int(v)) => *v,
            _ => 1,
        };
        let dilations = match node.attributes.get(&Attr::Dilations) {
            Some(OnnxAttribute::Ints(v)) => v.clone(),
            _ => vec![1, 1],
        };
        if dilations != [1, 1] {
            return None;
        }
        if group == 1 && shape[2] == 1 && shape[3] == 1 {
            return Some("pw");
        }
        if group > 1
            && group as usize == shape[0]
            && shape[1] == 1
            && shape[2] == shape[3]
            && (shape[2] == 3 || shape[2] == 5)
        {
            return Some("dw");
        }
        None
    }

    /// Symmetric pad + supported stride for the 3×3/5×5 INT8 fused DW.
    /// The kernel asserts `pad = (kh - 1) / 2` and `stride ∈ {1, 2}`; if
    /// the QLinearConv carries different params we leave it to the per-op
    /// path.
    fn dw_geom_supported(node: &OnnxNode, kh: usize) -> bool {
        let pads = match node.attributes.get(&Attr::Pads) {
            Some(OnnxAttribute::Ints(v)) => v.clone(),
            _ => vec![0, 0, 0, 0],
        };
        let strides = match node.attributes.get(&Attr::Strides) {
            Some(OnnxAttribute::Ints(v)) => v.clone(),
            _ => vec![1, 1],
        };
        if pads.len() != 4 || strides.len() != 2 {
            return false;
        }
        let want_pad = ((kh - 1) / 2) as i64;
        if !pads.iter().all(|&p| p == want_pad) {
            return false;
        }
        if strides[0] != strides[1] {
            return false;
        }
        matches!(strides[0], 1 | 2)
    }

    /// All QLinearConv zero-points used by the symmetric chain are 0.
    /// QLinearConv input layout: `x, x_scale, x_zp, w, w_scale, w_zp,
    /// y_scale, y_zp, [B]`. PW additionally requires `y_zp == 0` because
    /// the QDQ boundary fold downstream expects it; DW's `y_zp` is the
    /// chain output zero-point and may be non-zero.
    fn qlc_zps_match_chain(
        node: &OnnxNode,
        initializers: &FxHashMap<String, Tensor>,
        require_y_zp_zero: bool,
    ) -> bool {
        let x_zp = init_scalar(initializers, &node.inputs[2]);
        let w_zp = init_scalar(initializers, &node.inputs[5]);
        let y_zp = init_scalar(initializers, &node.inputs[7]);
        let zero = 0.0_f32.to_bits();
        let xz = x_zp.is_some_and(|v| v.to_bits() == zero);
        let wz = w_zp.is_some_and(|v| v.to_bits() == zero);
        if require_y_zp_zero {
            xz && wz && y_zp.is_some_and(|v| v.to_bits() == zero)
        } else {
            xz && wz
        }
    }

    // Map tensor name → producing node index. Used by the
    // `FusedTransposeMatMul` detection below to walk from a MatMul
    // left-input back to its Transpose producer in O(1).
    let producers: FxHashMap<String, usize> = {
        let mut m: FxHashMap<String, usize> =
            FxHashMap::with_capacity_and_hasher(nodes.len(), FxBuildHasher);
        for (idx, node) in nodes.iter().enumerate() {
            for out in &node.outputs {
                if !out.is_empty() {
                    m.insert(out.clone(), idx);
                }
            }
        }
        m
    };

    // Map tensor name → the nodes that read it, one entry per use, in node
    // order.
    //
    // The fusion matchers below used to find a producer's consumer by looking
    // at the next non-skipped node — which is adjacency, not dataflow. They
    // therefore depended on `ReorderForFusion` having put the chain together,
    // and silently declined to fuse whenever an unrelated node was scheduled
    // between two links. This is the index that lets them ask the graph
    // instead.
    let consumers: FxHashMap<&str, Vec<usize>> = {
        let mut m: FxHashMap<&str, Vec<usize>> =
            FxHashMap::with_capacity_and_hasher(nodes.len(), FxBuildHasher);
        for (idx, node) in nodes.iter().enumerate() {
            for inp in &node.inputs {
                if !inp.is_empty() {
                    m.entry(inp.as_str()).or_default().push(idx);
                }
            }
        }
        m
    };

    // The reader of a value used exactly once. Almost every fusion site wants
    // this: they all require the intermediate to have a single reader before
    // absorbing it, which makes that reader unique and unambiguous. One use and
    // one reader are the same condition here because `consumers` records a node
    // once per use, so a node reading the same value twice lands twice.
    let sole_consumer = |name: &str| -> Option<usize> {
        match consumers.get(name).map(Vec::as_slice) {
            Some([only]) => Some(*only),
            _ => None,
        }
    };

    // The single reader of `name` whose op is `op` and which reads `name` as
    // its first input; `None` if there is no such reader or more than one.
    //
    // For the one matcher that walks *through* a fan-out: `QuantizedForkPair`
    // exists precisely because the value it crosses has a side consumer, so
    // `sole_consumer` cannot answer there.
    let unique_consumer_of_op = |name: &str, op: &str| -> Option<usize> {
        let mut found = None;
        for &j in consumers.get(name)?.iter() {
            if nodes[j].op_type == op && nodes[j].inputs.first().map(String::as_str) == Some(name) {
                if found.is_some() {
                    return None;
                }
                found = Some(j);
            }
        }
        found
    };

    // Is `name` computed by the time plan position `pos` executes?
    //
    // Matching by dataflow lets a fusion absorb a consumer that sits well after
    // the producer, and the fused action runs at the *producer's* position. So
    // any input that consumer reads besides the value being fused across has to
    // already exist there. Weights and graph inputs always do — they have no
    // producing node — but a second activation need not: in a ResNet block both
    // the main-path Conv and the shortcut Conv feed one Add, each is that Add's
    // only reader through its own output, and fusing the *earlier* one would
    // schedule the Add before the other branch had run.
    //
    // A producer at a lower node index always executes first, including when it
    // was itself absorbed, because a fusion is anchored at the first node of its
    // chain and only ever absorbs nodes after it.
    let available_at =
        |name: &str, pos: usize| -> bool { producers.get(name).is_none_or(|&p| p < pos) };

    // Build execution plan — pre-compiled dispatch table.
    let mut execution_plan = Vec::with_capacity(nodes.len());
    let mut plan_skip = vec![false; nodes.len()];
    for (i, (kind, cp)) in node_kinds.iter().zip(conv_params.iter()).enumerate() {
        if plan_skip[i] {
            execution_plan.push(NodeAction::Skip);
            continue;
        }
        // Try to fuse `QLinearConv(pw) -> DQ -> [Relu] -> Q -> QLinearConv(dw)`
        // into a single `QuantizedPwDw` action. Gated on:
        //   * pw and dw are the supported PW / DW kinds (shape-only check);
        //   * pw_x_zp = pw_w_zp = pw_y_zp = 0 (last forced by the matching
        //     boundary fold);
        //   * dw_x_zp = dw_w_zp = 0 (dw_y_zp may be non-zero — chain output);
        //   * matching_zero_qparams holds at the boundary;
        //   * dw geometry supported by the kernel (pad = (kh-1)/2,
        //     stride ∈ {1, 2});
        //   * single-use intermediates, no model output along the chain;
        //   * DW (KHWC i8) prepack fires unconditionally for any depthwise
        //     shape; PW (VNNI 4×16 / transposed-B) is force-prepacked for this
        //     action via `chain_pw_weights` regardless of the `c_out % 16`
        //     gate, so the chain's `prepacked_i8_b` / `prepacked_i8_depthwise`
        //     lookups can't miss at dispatch.
        if nodes[i].op_type == "QLinearConv"
            && qlc_kind(&nodes[i], initializers) == Some("pw")
            && qlc_zps_match_chain(&nodes[i], initializers, true)
        {
            let pw_out = nodes[i].outputs.first().map(String::as_str).unwrap_or("");
            let pw_w_name = nodes[i].inputs.get(3).map(String::as_str).unwrap_or("");
            if !pw_out.is_empty()
                && !pw_w_name.is_empty()
                && !outputs.iter().any(|o| o == pw_out)
                && let Some(dq_idx) = sole_consumer(pw_out)
                && nodes[dq_idx].op_type == "DequantizeLinear"
                && !plan_skip[dq_idx]
                && nodes[dq_idx].inputs.first().map(String::as_str) == Some(pw_out)
            {
                let dq_out = nodes[dq_idx]
                    .outputs
                    .first()
                    .map(String::as_str)
                    .unwrap_or("");
                let mut relu_idx = None;
                let mut q_input = dq_out;
                let mut q_idx = sole_consumer(dq_out);
                if let Some(ri) = q_idx
                    && nodes[ri].op_type == "Relu"
                    && !plan_skip[ri]
                    && nodes[ri].inputs.first().map(String::as_str) == Some(dq_out)
                    && !outputs.iter().any(|o| o == dq_out)
                {
                    let relu_out = nodes[ri].outputs.first().map(String::as_str).unwrap_or("");
                    if !relu_out.is_empty() && !outputs.iter().any(|o| o == relu_out) {
                        relu_idx = Some(ri);
                        q_input = relu_out;
                        q_idx = sole_consumer(relu_out);
                    }
                }
                if !dq_out.is_empty()
                    && !q_input.is_empty()
                    && let Some(q_idx) = q_idx
                    && nodes[q_idx].op_type == "QuantizeLinear"
                    && !plan_skip[q_idx]
                    && nodes[q_idx].inputs.first().map(String::as_str) == Some(q_input)
                    && matching_zero_qparams(&nodes[dq_idx], &nodes[q_idx], initializers)
                {
                    let q_out = nodes[q_idx]
                        .outputs
                        .first()
                        .map(String::as_str)
                        .unwrap_or("");
                    if !q_out.is_empty()
                        && !outputs.iter().any(|o| o == q_out)
                        && let Some(dw_idx) = sole_consumer(q_out)
                        && nodes[dw_idx].op_type == "QLinearConv"
                        && !plan_skip[dw_idx]
                        && nodes[dw_idx].inputs.first().map(String::as_str) == Some(q_out)
                        && qlc_kind(&nodes[dw_idx], initializers) == Some("dw")
                        && qlc_zps_match_chain(&nodes[dw_idx], initializers, false)
                    {
                        let dw_w_shape = nodes[dw_idx]
                            .inputs
                            .get(3)
                            .and_then(|name| initializers.get(name))
                            .map(|t| t.shape().to_vec())
                            .unwrap_or_default();
                        let kh = dw_w_shape.get(2).copied().unwrap_or(0);
                        if dw_geom_supported(&nodes[dw_idx], kh) {
                            let has_relu = relu_idx.is_some();
                            execution_plan.push(NodeAction::QuantizedPwDw {
                                pw_idx: i,
                                dq_idx,
                                relu_idx,
                                q_idx,
                                dw_idx,
                                has_relu,
                            });
                            plan_skip[dq_idx] = true;
                            if let Some(ri) = relu_idx {
                                plan_skip[ri] = true;
                            }
                            plan_skip[q_idx] = true;
                            plan_skip[dw_idx] = true;
                            continue;
                        }
                    }
                }
            }
        }
        // Try to fuse `QLinearConv(dw) -> DQ -> [Relu] -> Q -> QLinearConv(pw)`
        // into a single `QuantizedDwPw` action. Mirror of the PW->DW
        // detector above. Gates:
        //   * dw and pw are the supported DW / PW kinds (shape-only check);
        //   * dw_x_zp = dw_w_zp = dw_y_zp = 0 (last forced by the boundary
        //     fold below);
        //   * pw_x_zp = pw_w_zp = 0 (pw_y_zp may be non-zero — chain output);
        //   * matching_zero_qparams holds at the QDQ boundary;
        //   * dw geometry supported by the kernel (pad = (kh-1)/2,
        //     stride ∈ {1, 2});
        //   * single-use intermediates, no model output along the chain;
        //   * load-time prepacking for both DW (KHWC i8) and PW (VNNI 4×16)
        //     fires under the same shape gates we check here.
        if nodes[i].op_type == "QLinearConv"
            && qlc_kind(&nodes[i], initializers) == Some("dw")
            && qlc_zps_match_chain(&nodes[i], initializers, true)
        {
            let dw_w_shape = nodes[i]
                .inputs
                .get(3)
                .and_then(|name| initializers.get(name))
                .map(|t| t.shape().to_vec())
                .unwrap_or_default();
            let kh = dw_w_shape.get(2).copied().unwrap_or(0);
            if dw_geom_supported(&nodes[i], kh) {
                let dw_out = nodes[i].outputs.first().map(String::as_str).unwrap_or("");
                if !dw_out.is_empty()
                    && !outputs.iter().any(|o| o == dw_out)
                    && let Some(dq_idx) = sole_consumer(dw_out)
                    && nodes[dq_idx].op_type == "DequantizeLinear"
                    && !plan_skip[dq_idx]
                    && nodes[dq_idx].inputs.first().map(String::as_str) == Some(dw_out)
                {
                    let dq_out = nodes[dq_idx]
                        .outputs
                        .first()
                        .map(String::as_str)
                        .unwrap_or("");
                    let mut relu_idx = None;
                    let mut q_input = dq_out;
                    let mut q_idx = sole_consumer(dq_out);
                    if let Some(ri) = q_idx
                        && nodes[ri].op_type == "Relu"
                        && !plan_skip[ri]
                        && nodes[ri].inputs.first().map(String::as_str) == Some(dq_out)
                        && !outputs.iter().any(|o| o == dq_out)
                    {
                        let relu_out = nodes[ri].outputs.first().map(String::as_str).unwrap_or("");
                        if !relu_out.is_empty() && !outputs.iter().any(|o| o == relu_out) {
                            relu_idx = Some(ri);
                            q_input = relu_out;
                            q_idx = sole_consumer(relu_out);
                        }
                    }
                    if !dq_out.is_empty()
                        && !q_input.is_empty()
                        && let Some(q_idx) = q_idx
                        && nodes[q_idx].op_type == "QuantizeLinear"
                        && !plan_skip[q_idx]
                        && nodes[q_idx].inputs.first().map(String::as_str) == Some(q_input)
                        && matching_zero_qparams(&nodes[dq_idx], &nodes[q_idx], initializers)
                    {
                        let q_out = nodes[q_idx]
                            .outputs
                            .first()
                            .map(String::as_str)
                            .unwrap_or("");
                        if !q_out.is_empty()
                            && !outputs.iter().any(|o| o == q_out)
                            && let Some(pw_idx) = sole_consumer(q_out)
                            && nodes[pw_idx].op_type == "QLinearConv"
                            && !plan_skip[pw_idx]
                            && nodes[pw_idx].inputs.first().map(String::as_str) == Some(q_out)
                            && qlc_kind(&nodes[pw_idx], initializers) == Some("pw")
                            && qlc_zps_match_chain(&nodes[pw_idx], initializers, false)
                        {
                            let has_relu = relu_idx.is_some();
                            execution_plan.push(NodeAction::QuantizedDwPw {
                                dw_idx: i,
                                dq_idx,
                                relu_idx,
                                q_idx,
                                pw_idx,
                                has_relu,
                            });
                            plan_skip[dq_idx] = true;
                            if let Some(ri) = relu_idx {
                                plan_skip[ri] = true;
                            }
                            plan_skip[q_idx] = true;
                            plan_skip[pw_idx] = true;
                            continue;
                        }
                    }
                }
            }
        }
        // Forked quant pair: the first QLinearConv's dequantized output has
        // a side consumer (usually a residual Add), so the stricter fused
        // kernel above cannot consume the DQ/Q boundary exclusively. Keep
        // the same graph values but schedule the pair as one quant action.
        if nodes[i].op_type == "QLinearConv"
            && let Some(first_kind) = qlc_kind(&nodes[i], initializers)
        {
            let first_out = nodes[i].outputs.first().map(String::as_str).unwrap_or("");
            if !first_out.is_empty()
                && !outputs.iter().any(|o| o == first_out)
                && let Some(dq_idx) = sole_consumer(first_out)
                && nodes[dq_idx].op_type == "DequantizeLinear"
                && !plan_skip[dq_idx]
                && nodes[dq_idx].inputs.first().map(String::as_str) == Some(first_out)
            {
                let dq_out = nodes[dq_idx]
                    .outputs
                    .first()
                    .map(String::as_str)
                    .unwrap_or("");
                let mut relu_idx = None;
                let mut q_input = dq_out;
                // Absorbing the Relu means the fused kernel writes the Relu's
                // output as the side value and never materializes `dq_out`, so
                // the Relu has to be its only reader — and `dq_out` must not be
                // a graph output, which no reader appears in. The positional
                // form checked neither, and left anything expecting `dq_out`
                // looking for a tensor nothing produced.
                if let Some(ri) = sole_consumer(dq_out)
                    && !outputs.iter().any(|o| o == dq_out)
                    && nodes[ri].op_type == "Relu"
                    && !plan_skip[ri]
                    && nodes[ri].inputs.first().map(String::as_str) == Some(dq_out)
                {
                    let relu_out = nodes[ri].outputs.first().map(String::as_str).unwrap_or("");
                    if !relu_out.is_empty() {
                        relu_idx = Some(ri);
                        q_input = relu_out;
                    }
                }
                // The fork itself: `q_input` is read by the QuantizeLinear that
                // continues the INT8 chain *and* by something else, which is
                // why the stricter `QuantizedPwDw` above declined it.
                if !q_input.is_empty()
                    && use_counts.get(q_input).copied().unwrap_or(0) > 1
                    && let Some(q_idx) = unique_consumer_of_op(q_input, "QuantizeLinear")
                    && !plan_skip[q_idx]
                {
                    let q_out = nodes[q_idx]
                        .outputs
                        .first()
                        .map(String::as_str)
                        .unwrap_or("");
                    let want_second = if first_kind == "pw" { "dw" } else { "pw" };
                    if !q_out.is_empty()
                        && !outputs.iter().any(|o| o == q_out)
                        && let Some(second_idx) = sole_consumer(q_out)
                        && nodes[second_idx].op_type == "QLinearConv"
                        && !plan_skip[second_idx]
                        && nodes[second_idx].inputs.first().map(String::as_str) == Some(q_out)
                        && qlc_kind(&nodes[second_idx], initializers) == Some(want_second)
                    {
                        execution_plan.push(NodeAction::QuantizedForkPair {
                            first_idx: i,
                            dq_idx,
                            relu_idx,
                            q_idx,
                            second_idx,
                            first_kind: u8::from(first_kind == "dw"),
                            has_relu: relu_idx.is_some(),
                        });
                        plan_skip[dq_idx] = true;
                        if let Some(ri) = relu_idx {
                            plan_skip[ri] = true;
                        }
                        plan_skip[q_idx] = true;
                        plan_skip[second_idx] = true;
                        continue;
                    }
                }
            }
        }
        // Residual suffix: QLinearConv output is dequantized, passed through
        // Relu, then a fp32 pointwise Conv + Add, and finally quantized back
        // for the next INT8 chain. This action keeps the suffix in one plan
        // slot and accounts for it as a quant-chain execution.
        if nodes[i].op_type == "QLinearConv"
            && let Some(kind) = qlc_kind(&nodes[i], initializers)
        {
            // Every value inside the chain is consumed by the next link and by
            // nothing else, because `exec_quantized_dw_residual` writes only the
            // final quantized output — it never materializes the intermediates.
            // Walking the chain through `sole_consumer` states that directly:
            // each step both finds the next node and proves the value crossing
            // into it has exactly one reader.
            let link = |from: usize| -> Option<usize> {
                let out = nodes[from].outputs.first().map(String::as_str)?;
                if out.is_empty() || outputs.iter().any(|o| o == out) {
                    return None;
                }
                sole_consumer(out)
            };
            let qconv_out = nodes[i].outputs.first().map(String::as_str).unwrap_or("");
            if !qconv_out.is_empty()
                && !outputs.iter().any(|o| o == qconv_out)
                && let Some(dq_idx) = sole_consumer(qconv_out)
                && nodes[dq_idx].op_type == "DequantizeLinear"
                && nodes[dq_idx].inputs.first().map(String::as_str) == Some(qconv_out)
                && let Some(relu_idx) = link(dq_idx)
                && nodes[relu_idx].op_type == "Relu"
                && let Some(conv_idx) = link(relu_idx)
                && nodes[conv_idx].op_type == "Conv"
                && let Some(add_idx) = link(conv_idx)
                && nodes[add_idx].op_type == "Add"
                && let Some(q_idx) = link(add_idx)
                && nodes[q_idx].op_type == "QuantizeLinear"
                && !plan_skip[dq_idx]
                && !plan_skip[relu_idx]
                && !plan_skip[conv_idx]
                && !plan_skip[add_idx]
                && !plan_skip[q_idx]
                // The Add's residual input comes from outside the chain, and
                // the whole chain executes at `i`, so it has to exist by then.
                && nodes[add_idx]
                    .inputs
                    .iter()
                    .all(|inp| Some(inp) == nodes[conv_idx].outputs.first() || available_at(inp, i))
                && nodes[relu_idx].inputs.first() == nodes[dq_idx].outputs.first()
                && nodes[conv_idx].inputs.first() == nodes[relu_idx].outputs.first()
                && nodes[q_idx].inputs.first() == nodes[add_idx].outputs.first()
            {
                execution_plan.push(NodeAction::QuantizedResidualChain {
                    qconv_idx: i,
                    dq_idx,
                    relu_idx,
                    conv_idx,
                    add_idx,
                    q_idx,
                    qconv_kind: u8::from(kind == "dw"),
                });
                plan_skip[dq_idx] = true;
                plan_skip[relu_idx] = true;
                plan_skip[conv_idx] = true;
                plan_skip[add_idx] = true;
                plan_skip[q_idx] = true;
                continue;
            }
        }
        if nodes[i].op_type == "QLinearConv"
            && let Some(kind) = qlc_kind(&nodes[i], initializers)
        {
            let qconv_out = nodes[i].outputs.first().map(String::as_str).unwrap_or("");
            if !qconv_out.is_empty()
                && !outputs.iter().any(|o| o == qconv_out)
                && let Some(dq_idx) = sole_consumer(qconv_out)
                && nodes[dq_idx].op_type == "DequantizeLinear"
                && !plan_skip[dq_idx]
                && nodes[dq_idx].inputs.first().map(String::as_str) == Some(qconv_out)
                && nodes[dq_idx]
                    .outputs
                    .first()
                    .is_some_and(|out| !outputs.iter().any(|model_out| model_out == out))
            {
                execution_plan.push(NodeAction::QuantizedConvDq {
                    qconv_idx: i,
                    dq_idx,
                    qconv_kind: u8::from(kind == "dw"),
                });
                plan_skip[dq_idx] = true;
                continue;
            }
        }
        if nodes[i].op_type == "DequantizeLinear" {
            let dequant_out = nodes[i].outputs.first().map(String::as_str).unwrap_or("");
            let dequant_input = nodes[i].inputs.first().map(String::as_str).unwrap_or("");
            if !dequant_out.is_empty()
                && !dequant_input.is_empty()
                && use_counts.get(dequant_input).copied().unwrap_or(0) == 1
                && !outputs.iter().any(|o| o == dequant_out)
            {
                let mut relu_idx = None;
                let mut quant_idx = sole_consumer(dequant_out);
                if let Some(ri) = quant_idx
                    && !plan_skip[ri]
                    && node_kinds[ri] == NodeKind::Relu
                    && nodes[ri].inputs.len() == 1
                    && nodes[ri].inputs[0] == dequant_out
                    && !outputs.iter().any(|o| o == &nodes[ri].outputs[0])
                {
                    relu_idx = Some(ri);
                    quant_idx = sole_consumer(&nodes[ri].outputs[0]);
                }

                if let Some(quant_idx) = quant_idx
                    && let quant_node = &nodes[quant_idx]
                    && !plan_skip[quant_idx]
                    && quant_node.op_type == "QuantizeLinear"
                    && quant_node.inputs.len() >= 3
                    && quant_node.outputs.len() == 1
                    && ((relu_idx.is_none() && quant_node.inputs[0] == dequant_out)
                        || relu_idx
                            .map(|ri| quant_node.inputs[0] == nodes[ri].outputs[0])
                            .unwrap_or(false))
                    && per_tensor_qdq_boundary(&nodes[i], quant_node, initializers)
                {
                    execution_plan.push(NodeAction::QuantizedQdq {
                        dequant_idx: i,
                        relu_idx,
                        hardswish: None,
                        quant_idx,
                    });
                    if let Some(ri) = relu_idx {
                        plan_skip[ri] = true;
                    }
                    plan_skip[quant_idx] = true;
                    continue;
                }

                // HardSwish fold: `DQ -> HardSigmoid -> Mul(dq, hs) -> Q`
                // (MobileNetV3's `x * HardSigmoid(x)`). The DQ output fans out to
                // exactly the HardSigmoid and the Mul; fold all four into one
                // i8->i8 pass. Distinct from a Squeeze-Excite Mul, whose second
                // operand is a Reshape gate, not `HardSigmoid(dq)`.
                if let Some(cons) = consumers.get(dequant_out)
                    && cons.len() == 2
                {
                    let hs_idx = cons
                        .iter()
                        .copied()
                        .find(|&c| nodes[c].op_type == "HardSigmoid");
                    let mul_idx = cons.iter().copied().find(|&c| nodes[c].op_type == "Mul");
                    if let (Some(hs_idx), Some(mul_idx)) = (hs_idx, mul_idx)
                        && !plan_skip[hs_idx]
                        && !plan_skip[mul_idx]
                        && nodes[hs_idx].inputs.len() == 1
                        && nodes[hs_idx].inputs[0] == dequant_out
                        && nodes[hs_idx].outputs.len() == 1
                        && !outputs.iter().any(|o| o == &nodes[hs_idx].outputs[0])
                    {
                        let hs_out = nodes[hs_idx].outputs[0].as_str();
                        let mul = &nodes[mul_idx];
                        let mul_ok = mul.inputs.len() == 2
                            && ((mul.inputs[0] == dequant_out && mul.inputs[1] == hs_out)
                                || (mul.inputs[0] == hs_out && mul.inputs[1] == dequant_out));
                        if mul_ok
                            && sole_consumer(hs_out) == Some(mul_idx)
                            && mul.outputs.len() == 1
                            && !outputs.iter().any(|o| o == &mul.outputs[0])
                            && let Some(q_idx) = sole_consumer(&mul.outputs[0])
                            && !plan_skip[q_idx]
                            && nodes[q_idx].op_type == "QuantizeLinear"
                            && nodes[q_idx].inputs.len() >= 3
                            && nodes[q_idx].outputs.len() == 1
                            && per_tensor_qdq_boundary(&nodes[i], &nodes[q_idx], initializers)
                        {
                            execution_plan.push(NodeAction::QuantizedQdq {
                                dequant_idx: i,
                                relu_idx: None,
                                hardswish: Some((hs_idx, mul_idx)),
                                quant_idx: q_idx,
                            });
                            plan_skip[hs_idx] = true;
                            plan_skip[mul_idx] = true;
                            plan_skip[q_idx] = true;
                            continue;
                        }
                    }
                }
            }
        }
        match kind {
            NodeKind::Conv | NodeKind::ConvRelu | NodeKind::ConvSilu => {
                let activation = match kind {
                    NodeKind::ConvRelu => 1,
                    NodeKind::ConvSilu => 2,
                    _ => 0,
                };

                // Try DW+PW fusion. Backs off when PW has a downstream
                // Add — the stronger `Conv_Add_fused` op saves an
                // entire output memory pass via
                // `conv2d_nhwc_pointwise_with_residual_relu`, worth
                // more than the DW+PW dispatch savings on tracker.
                let mut fused = false;
                if let Some(cp) = cp
                    && cp.is_depthwise
                {
                    // Look ahead for pointwise consuming our output
                    let dw_out = &nodes[i].outputs[0];
                    let dw_uses = use_counts.get(dw_out).copied().unwrap_or(0);
                    if dw_uses == 1
                        && let Some(j) = sole_consumer(dw_out)
                        && !plan_skip[j]
                    {
                        {
                            let nk = node_kinds[j];
                            if matches!(
                                nk,
                                NodeKind::Conv | NodeKind::ConvRelu | NodeKind::ConvSilu
                            ) && let Some(ncp) = &conv_params[j]
                                && ncp.is_pointwise
                                && !ncp.has_padding
                                && nodes[j].inputs.first().map(|s| s.as_str())
                                    == Some(dw_out.as_str())
                            {
                                // Skip DW+PW when PW would instead form ConvAdd.
                                // `pointwise_nx16_direct` in ConvAdd is faster than
                                // band-streaming for N%16==0 shapes (e.g. N=112).
                                let pw_kind_plain = matches!(nk, NodeKind::Conv);
                                let pw_out = &nodes[j].outputs[0];
                                let pw_out_uses = use_counts.get(pw_out).copied().unwrap_or(0);
                                // Asks the same question the `ConvAdd` matcher
                                // below answers, so it must ask it the same way
                                // — through the consumer index, not at j + 1.
                                let pw_has_convadd = pw_kind_plain
                                    && pw_out_uses == 1
                                    && sole_consumer(pw_out).is_some_and(|a| {
                                        node_kinds[a] == NodeKind::Add
                                            && !plan_skip[a]
                                            && nodes[a].inputs.len() == 2
                                            && (nodes[a].inputs[0] == *pw_out
                                                || nodes[a].inputs[1] == *pw_out)
                                    });
                                if !pw_has_convadd {
                                    let pw_act = match nk {
                                        NodeKind::ConvRelu => 1,
                                        NodeKind::ConvSilu => 2,
                                        _ => 0,
                                    };
                                    execution_plan.push(NodeAction::FusedDwPw {
                                        dw_idx: i,
                                        pw_idx: j,
                                        dw_activation: activation,
                                        pw_activation: pw_act,
                                    });
                                    plan_skip[j] = true;
                                    fused = true;
                                }
                            }
                        }
                    }
                }
                // Try PW+DW fusion (current is PW expansion feeding into DW).
                // Mirrors the DW+PW block above but swapped: when the
                // current node is a non-DW pointwise 1×1 Conv whose output
                // is consumed exclusively by an immediately-following
                // depthwise Conv, fuse them. This targets the
                // MobileNetV2 `PW_expand → DW` opening that the
                // residual-suffix `Conv_Add_fused` leaves alone. Skips
                // when PW's activation is SiLU (not a typical
                // MobileNet pattern and the fused exec only supports
                // None/Relu epilogues for now).
                if !fused
                    && let Some(cp) = cp
                    && cp.is_pointwise
                    && !cp.has_padding
                    && activation != 2
                {
                    let pw_out = &nodes[i].outputs[0];
                    let pw_uses = use_counts.get(pw_out).copied().unwrap_or(0);
                    if pw_uses == 1
                        && let Some(j) = sole_consumer(pw_out)
                        && !plan_skip[j]
                    {
                        {
                            let nk = node_kinds[j];
                            if matches!(
                                nk,
                                NodeKind::Conv | NodeKind::ConvRelu | NodeKind::ConvSilu
                            ) && let Some(ncp) = &conv_params[j]
                                && ncp.is_depthwise
                                && nodes[j].inputs.first().map(|s| s.as_str())
                                    == Some(pw_out.as_str())
                            {
                                let dw_act = match nk {
                                    NodeKind::ConvRelu => 1u8,
                                    NodeKind::ConvSilu => 2u8,
                                    _ => 0u8,
                                };
                                execution_plan.push(NodeAction::FusedPwDw {
                                    pw_idx: i,
                                    dw_idx: j,
                                    pw_activation: activation,
                                    dw_activation: dw_act,
                                });
                                plan_skip[j] = true;
                                fused = true;
                            }
                        }
                    }
                }
                if !fused {
                    // Try Conv → Add (residual), optionally followed by Relu.
                    let conv_out = &nodes[i].outputs[0];
                    let conv_out_uses = use_counts.get(conv_out).copied().unwrap_or(0);
                    let mut conv_add_emitted = false;
                    if conv_out_uses == 1
                        && activation == 0
                        && let Some(add_idx) = sole_consumer(conv_out)
                        && let Some(add_node) = nodes.get(add_idx)
                        && node_kinds[add_idx] == NodeKind::Add
                        && add_node.inputs.len() == 2
                        && !plan_skip[add_idx]
                        && (add_node.inputs[0] == *conv_out || add_node.inputs[1] == *conv_out)
                        // The residual input has to already exist at `i`: the
                        // fused action runs there, not at `add_idx`.
                        && available_at(
                            &add_node.inputs[usize::from(add_node.inputs[0] == *conv_out)],
                            i,
                        )
                    {
                        let skip_input_idx = u8::from(add_node.inputs[0] == *conv_out);
                        let add_out = &add_node.outputs[0];
                        let (post_activation, relu_idx_field) = match sole_consumer(add_out)
                            .and_then(|r| nodes.get(r).map(|n| (r, n)))
                        {
                            Some((relu_idx, relu_node))
                                if node_kinds[relu_idx] == NodeKind::Relu
                                    && !plan_skip[relu_idx]
                                    && relu_node.inputs.len() == 1
                                    && relu_node.inputs[0] == *add_out =>
                            {
                                (1u8, relu_idx as u32)
                            }
                            _ => (0u8, 0u32),
                        };
                        execution_plan.push(NodeAction::ConvAdd {
                            conv_idx: i,
                            add_idx,
                            skip_input_idx,
                            post_activation,
                            relu_idx: relu_idx_field,
                        });
                        plan_skip[add_idx] = true;
                        if post_activation == 1 {
                            plan_skip[relu_idx_field as usize] = true;
                        }
                        conv_add_emitted = true;
                    }
                    if !conv_add_emitted {
                        execution_plan.push(NodeAction::Conv {
                            node_idx: i,
                            activation,
                        });
                    }
                }
            }
            NodeKind::MatMul => {
                // Try Transpose+MatMul fusion: when the MatMul's left
                // input (index 0) is the output of a `Transpose` node
                // whose `perm` swaps the last two axes of a rank-3
                // tensor (i.e. `[0,2,1]`) AND every consumer of that
                // Transpose is a MatMul that can absorb it, we elide
                // the Transpose entirely and dispatch a `transA=1`
                // GEMM. Mirrors ORT's `MatmulTransposeFusion`. Weaker
                // fusion (Transpose has other consumers) still pays
                // the materialization cost once, so we don't fuse.
                let left_input = nodes[i].inputs.first().map(|s| s.as_str()).unwrap_or("");
                // "every consumer ... can absorb it" is load-bearing, not just a
                // profitability rule. The fused action reads the value *before*
                // the Transpose; when some consumer cannot absorb it the
                // Transpose still runs, and running it consumes that
                // pre-transpose value if this was its last use — leaving the
                // fused action reading a tensor that is gone.
                let absorbable = |value: &str| {
                    !value.is_empty()
                        && !outputs.iter().any(|o| o == value)
                        && consumers.get(value).is_some_and(|readers| {
                            readers.iter().all(|&r| {
                                node_kinds[r] == NodeKind::MatMul
                                    && nodes[r].inputs.first().map(String::as_str) == Some(value)
                            })
                        })
                };
                let mut emitted = false;
                if !left_input.is_empty()
                    && let Some(&t_idx) = producers.get(left_input)
                    && t_idx < nodes.len()
                    && node_kinds[t_idx] == NodeKind::Transpose
                    && !plan_skip[t_idx]
                    && transpose_perm_is_swap_last_two(&nodes[t_idx])
                    && absorbable(left_input)
                {
                    execution_plan.push(NodeAction::FusedTransposeMatMul {
                        transpose_idx: t_idx,
                        matmul_idx: i,
                        cleanup_transpose: false,
                    });
                    emitted = true;
                }
                if !emitted {
                    execution_plan.push(NodeAction::Generic {
                        node_idx: i,
                        kind: *kind,
                    });
                }
            }
            _ => {
                execution_plan.push(NodeAction::Generic {
                    node_idx: i,
                    kind: *kind,
                });
            }
        }
    }

    // The loop above pushes exactly one action per node, so plan position and
    // node index are the same thing. Everything downstream relies on it — the
    // transpose post-pass below assigns through `execution_plan[node_idx]`, and
    // a short plan silently stops executing the tail of the graph rather than
    // failing. Worth stating, because an early `break` in the fusion scan broke
    // it once already.
    debug_assert_eq!(
        execution_plan.len(),
        nodes.len(),
        "execution plan must hold one action per node"
    );

    // Post-pass: elide Transpose nodes whose every consumer is a
    // `FusedTransposeMatMul` that absorbed them. Counts the number of
    // fused actions pointing at each transpose and compares to the
    // transpose output's total graph-use count (input edges + model
    // output membership). When every consumer was absorbed, the
    // original Transpose does no useful work and becomes `Skip`.
    let model_outputs: FxHashSet<&str> = outputs.iter().map(|s| s.as_str()).collect();
    let mut fused_refs: Vec<usize> = vec![0; nodes.len()];
    // Plan position of the last `FusedTransposeMatMul` referencing each
    // transpose idx. Used to mark exactly one variant as the cleanup
    // owner so the pre-transpose tensor stays in `env` until every
    // consumer has read it.
    let mut last_fused_pos: FxHashMap<usize, usize> = FxHashMap::default();
    for (pos, action) in execution_plan.iter().enumerate() {
        if let NodeAction::FusedTransposeMatMul { transpose_idx, .. } = action {
            fused_refs[*transpose_idx] += 1;
            last_fused_pos.insert(*transpose_idx, pos);
        }
    }
    for (&t_idx, &pos) in &last_fused_pos {
        if let NodeAction::FusedTransposeMatMul {
            transpose_idx,
            cleanup_transpose,
            ..
        } = &mut execution_plan[pos]
        {
            debug_assert_eq!(*transpose_idx, t_idx);
            *cleanup_transpose = true;
        }
    }
    for t_idx in 0..nodes.len() {
        if fused_refs[t_idx] == 0 {
            continue;
        }
        let t_out = match nodes[t_idx].outputs.first() {
            Some(s) if !s.is_empty() => s,
            _ => continue,
        };
        let edge_uses = use_counts.get(t_out).copied().unwrap_or(0);
        let is_model_output = model_outputs.contains(t_out.as_str());
        let consumer_total = edge_uses + usize::from(is_model_output);
        if fused_refs[t_idx] >= consumer_total && consumer_total > 0 {
            execution_plan[t_idx] = NodeAction::Skip;
        }
    }

    // build the set of `Reshape` output tensor names whose
    // single consumer is a `Transpose(perm=[0,2,1])` that got absorbed
    // into a `FusedTransposeMatMul`. The runtime fast path checks this
    // set before deciding whether to skip the NHWC→NCHW permute for a
    // Reshape input. Only NHWC-passthrough-safe Reshapes get the
    // optimisation; others continue paying the legacy `ensure_nchw`.
    let mut reshape_nhwc_passthrough_safe: FxHashSet<String> = FxHashSet::default();
    for action in &execution_plan {
        let NodeAction::FusedTransposeMatMul { transpose_idx, .. } = action else {
            continue;
        };
        // The Transpose's input is the candidate producer.
        let Some(t_in) = nodes[*transpose_idx].inputs.first() else {
            continue;
        };
        if t_in.is_empty() {
            continue;
        }
        let Some(&prod_idx) = producers.get(t_in.as_str()) else {
            continue;
        };
        if prod_idx >= nodes.len() {
            continue;
        }
        // Producer must be a Reshape node.
        if node_kinds.get(prod_idx).copied() != Some(NodeKind::Reshape) {
            continue;
        }
        // Reshape's output must have exactly ONE graph consumer (the
        // Transpose). If the same tensor is read by other ops, those
        // ops won't honour the NHWC tag and would see garbage.
        let edge_uses = use_counts.get(t_in.as_str()).copied().unwrap_or(0);
        let is_model_output = model_outputs.contains(t_in.as_str());
        if edge_uses + usize::from(is_model_output) != 1 {
            continue;
        }
        reshape_nhwc_passthrough_safe.insert(t_in.clone());
    }

    // Load-time weight pre-packing. For every pointwise Conv (KH=KW=1,
    // group=1) whose weight is already laid out KHWC, pre-pack the B-matrix
    // in blocked-GEMM format and cache it by weight-tensor name. The execution
    // plan looks it up per call and hands the shared `Arc<PackedB>` to the
    // GEMM layer, skipping the runtime fingerprint cache and `pack_b_panel`.
    //
    // We can't prepack non-KHWC weights here because the runtime path re-
    // permutes them to KHWC on first use (which would make the prepack stale).
    // For the models we care about, `_with_khwc_once` has already normalized
    // all pointwise Conv weights at model-load, so this check is typically
    // true. Non-pointwise Convs go through 3×3 direct / im2col paths that
    // don't consume packed B — prepack isn't useful there.
    let mut prepacked_weights: FxHashMap<String, std::sync::Arc<yscv_kernels::PackedB>> =
        FxHashMap::default();
    for (i, node) in nodes.iter().enumerate() {
        let Some(cp) = conv_params[i].as_ref() else {
            continue;
        };
        if !cp.is_pointwise {
            continue;
        }
        let Some(w_name) = node.inputs.get(1) else {
            continue;
        };
        if !initializers.contains_key(w_name) || !khwc_weights.contains(w_name) {
            continue;
        }
        if prepacked_weights.contains_key(w_name) {
            continue;
        }
        let Some(weight) = initializers.get(w_name) else {
            continue;
        };
        // KHWC pointwise weight shape: [KH=1, KW=1, IC, OC]. k = IC, n = OC.
        let shape = weight.shape();
        if shape.len() != 4 || shape[0] != 1 || shape[1] != 1 {
            continue;
        }
        let k = shape[2];
        let n = shape[3];
        let packed = yscv_kernels::pack_b_for_session(weight.data(), k, n);
        prepacked_weights.insert(w_name.clone(), packed);
    }
    for node in nodes {
        if node.op_type != "MatMul" {
            continue;
        }
        let Some(w_name) = node.inputs.get(1) else {
            continue;
        };
        if prepacked_weights.contains_key(w_name) {
            continue;
        }
        let Some(weight) = initializers.get(w_name) else {
            continue;
        };
        let shape = weight.shape();
        if shape.len() != 2 {
            continue;
        }
        let k = shape[0];
        let n = shape[1];
        // Prepacking every tiny MatMul would bloat model load for little gain.
        // LLM projection weights are MB-class and run once per layer/token, so
        // they easily amortize this load-time pack and must not repack in
        // every decode step.
        if k.saturating_mul(n) < 256 * 1024 {
            continue;
        }
        let packed = yscv_kernels::pack_b_for_session(weight.data(), k, n);
        prepacked_weights.insert(w_name.clone(), packed);
    }
    let mut prepacked_weights_by_id: Vec<Option<std::sync::Arc<yscv_kernels::PackedB>>> =
        vec![None; name_to_id.len()];
    for (name, packed) in &prepacked_weights {
        if let Some(&id) = name_to_id.get(name) {
            prepacked_weights_by_id[id] = Some(packed.clone());
        }
    }

    fn attr_int(node: &OnnxNode, name: Attr, default: i64) -> i64 {
        match node.attributes.get(&name) {
            Some(OnnxAttribute::Int(v)) => *v,
            _ => default,
        }
    }

    fn attr_ints(node: &OnnxNode, name: Attr, default: &[i64]) -> Vec<i64> {
        match node.attributes.get(&name) {
            Some(OnnxAttribute::Ints(v)) => v.clone(),
            _ => default.to_vec(),
        }
    }

    fn tensor_data_as_i8(t: &Tensor) -> Vec<i8> {
        t.data().iter().map(|&v| v.round() as i8).collect()
    }

    fn should_prepack_i8_b(k: usize, n: usize) -> bool {
        // Load-time packing now carries the AVX-512 VNNI 4x16 layout in
        // addition to transposed-B, so tracker pointwise Conv (small K/N,
        // huge M) can skip per-inference RHS packing. Keep the previous
        // large MatMul gate for non-conv heads.
        (k >= 4 && n.is_multiple_of(16)) || (k >= 512 && n >= 1024)
    }

    let mut prepacked_i8_weights: FxHashMap<String, std::sync::Arc<yscv_kernels::PackedI8B>> =
        FxHashMap::default();
    let mut prepacked_i8_depthwise: FxHashMap<String, std::sync::Arc<Vec<i8>>> =
        FxHashMap::default();
    // Closing-pair `QuantizedDwPw` chains always need the PW weight
    // prepacked because the kernel reads `env.prepacked_i8_b` directly
    // and there is no per-iteration packing fallback. The default
    // `should_prepack_i8_b` predicate skips PWs whose `c_out` is not a
    // multiple of 16 (typical bottleneck/head widths like 24, 12, 4),
    // so collect those PW weight names from the execution plan and
    // force-prepack them below regardless of the gate. The transposed-B
    // fallback inside `pack_i8_b_for_matmul` handles non-multiples of
    // 16 — the VNNI 4×16 path is just unavailable; the kernel's
    // `int8_matmul_prepacked_dispatch` picks the next-best variant.
    let chain_pw_weights: rustc_hash::FxHashSet<String> = execution_plan
        .iter()
        .filter_map(|action| match action {
            // Opening-pair PW->DW: `exec_quantized_pw_dw` reads the packed PW
            // RHS directly with no per-call packing fallback, so its PW weight
            // must be force-prepacked even when `c_out` is not a multiple of 16
            // (expansion widths like 72/88/120 are common in MobileNetV3). This
            // arm was previously missing, so those chains panicked at dispatch
            // with "PW weight not prepacked (loader gate broken)".
            NodeAction::QuantizedPwDw { pw_idx, .. } => nodes[*pw_idx].inputs.get(3).cloned(),
            NodeAction::QuantizedDwPw { pw_idx, .. } => nodes[*pw_idx].inputs.get(3).cloned(),
            NodeAction::QuantizedForkPair {
                first_idx,
                first_kind,
                ..
            } if *first_kind == 0 => nodes[*first_idx].inputs.get(3).cloned(),
            NodeAction::QuantizedConvDq {
                qconv_idx,
                qconv_kind,
                ..
            } if *qconv_kind == 0 => nodes[*qconv_idx].inputs.get(3).cloned(),
            _ => None,
        })
        .collect();
    for node in nodes {
        match node.op_type.as_str() {
            "QLinearMatMul" => {
                let Some(w_name) = node.inputs.get(3) else {
                    continue;
                };
                if prepacked_i8_weights.contains_key(w_name) {
                    continue;
                }
                let Some(weight) = initializers.get(w_name) else {
                    continue;
                };
                let shape = weight.shape();
                if shape.len() == 2 && should_prepack_i8_b(shape[0], shape[1]) {
                    let (k, n) = (shape[0], shape[1]);
                    let data = tensor_data_as_i8(weight);
                    let packed = yscv_kernels::pack_i8_b_for_matmul(&data, k, n);
                    prepacked_i8_weights.insert(w_name.clone(), std::sync::Arc::new(packed));
                }
            }
            "MatMulInteger" => {
                let Some(w_name) = node.inputs.get(1) else {
                    continue;
                };
                if prepacked_i8_weights.contains_key(w_name) {
                    continue;
                }
                let Some(weight) = initializers.get(w_name) else {
                    continue;
                };
                let shape = weight.shape();
                if shape.len() == 2 && should_prepack_i8_b(shape[0], shape[1]) {
                    let (k, n) = (shape[0], shape[1]);
                    let data = tensor_data_as_i8(weight);
                    let packed = yscv_kernels::pack_i8_b_for_matmul(&data, k, n);
                    prepacked_i8_weights.insert(w_name.clone(), std::sync::Arc::new(packed));
                }
            }
            "QLinearConv" | "ConvInteger" => {
                let w_input_idx = if node.op_type == "QLinearConv" { 3 } else { 1 };
                let Some(w_name) = node.inputs.get(w_input_idx) else {
                    continue;
                };
                if prepacked_i8_weights.contains_key(w_name) {
                    continue;
                }
                let group = attr_int(node, Attr::Group, 1);
                let dilations = attr_ints(node, Attr::Dilations, &[1, 1]);
                let Some(weight) = initializers.get(w_name) else {
                    continue;
                };
                let shape = weight.shape();
                if shape.len() != 4 {
                    continue;
                }
                let (c_out, c_in, kh, kw) = (shape[0], shape[1], shape[2], shape[3]);
                if group > 1
                    && group as usize == c_out
                    && c_in == 1
                    && kh == kw
                    && (kh == 3 || kh == 5)
                    && dilations == [1, 1]
                {
                    let mut khwc = vec![0_i8; kh * kw * c_out];
                    let w_data = weight.data();
                    for c in 0..c_out {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let src = ((c * c_in) * kh + ky) * kw + kx;
                                let dst = (ky * kw + kx) * c_out + c;
                                khwc[dst] = w_data[src].round() as i8;
                            }
                        }
                    }
                    prepacked_i8_depthwise.insert(w_name.clone(), std::sync::Arc::new(khwc));
                    continue;
                }
                if group != 1 || dilations != [1, 1] {
                    continue;
                }
                let k_dim = c_in * kh * kw;
                let force_for_chain = chain_pw_weights.contains(w_name);
                if !should_prepack_i8_b(k_dim, c_out) && !force_for_chain {
                    continue;
                }
                let w_data = weight.data();
                let mut b = vec![0_i8; k_dim * c_out];
                for o in 0..c_out {
                    for ci in 0..c_in {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let src = ((o * c_in + ci) * kh + ky) * kw + kx;
                                let dst_k = (ci * kh + ky) * kw + kx;
                                b[dst_k * c_out + o] = w_data[src].round() as i8;
                            }
                        }
                    }
                }
                let packed = yscv_kernels::pack_i8_b_for_matmul(&b, k_dim, c_out);
                prepacked_i8_weights.insert(w_name.clone(), std::sync::Arc::new(packed));
            }
            _ => {}
        }
    }

    // NCHWc PW conv weight prepack (K.4). Pack 1×1 pointwise weights stored in
    // KHWC format [1, 1, Cin, Cexp] as PackedB [Cin, Cexp] for the fused
    // PW→DW streaming kernel.  These are small (Cin*Cexp << 256K) so they
    // are not covered by the existing MatMul prepack threshold loop above.
    for (i, node) in nodes.iter().enumerate() {
        let Some(cp) = conv_params[i].as_ref() else {
            continue;
        };
        if !cp.is_pointwise || cp.group != 1 {
            continue;
        }
        let Some(w_name) = node.inputs.get(1) else {
            continue;
        };
        if prepacked_weights.contains_key(w_name) {
            continue;
        }
        if !khwc_weights.contains(w_name) {
            continue;
        }
        let Some(weight) = initializers.get(w_name) else {
            continue;
        };
        let shape = weight.shape();
        // KHWC 1×1 pointwise: [1, 1, Cin, Cexp].
        if shape.len() != 4 || shape[0] != 1 || shape[1] != 1 {
            continue;
        }
        let (cin, cexp) = (shape[2], shape[3]);
        let packed = yscv_kernels::pack_b_for_session(weight.data(), cin, cexp);
        prepacked_weights.insert(w_name.clone(), packed);
    }
    // Rebuild prepacked_weights_by_id after PW conv packs are added.
    prepacked_weights_by_id = vec![None; name_to_id.len()];
    for (name, packed) in &prepacked_weights {
        if let Some(&id) = name_to_id.get(name) {
            prepacked_weights_by_id[id] = Some(packed.clone());
        }
    }

    // ── FusedPwDwPwReduce: streaming PW_expand → DW 3×3 → PW_reduce ──
    // Scan the execution plan one more time. For each FusedPwDw action,
    // check if the next non-skipped node is a 1×1 PW Conv consuming the
    // DW output, with c_in matching DW's c_out and no Add/Relu (Conv_Add
    // residual blocks keep the existing ConvAdd path which is faster on
    // those shapes). If so, rewrite the action into FusedPwDwPwReduce and
    // prepack the PW reduce weight + bias for runtime.
    //
    // Kill switch: `YSCV_FUSED_PW_DW_PW_REDUCE_OFF=1` keeps everything as
    // FusedPwDw and lets the PW reduce stay a separate Conv action.
    let mut prepacked_fused_pw_dw_pw_reduce: FxHashMap<
        usize,
        std::sync::Arc<FusedPwDwPwReduceWeights>,
    > = FxHashMap::default();
    let fusion_off = std::env::var_os("YSCV_FUSED_PW_DW_PW_REDUCE_OFF").is_some();
    let fusion_debug = std::env::var_os("YSCV_FUSED_PW_DW_PW_REDUCE_DEBUG").is_some();
    if !fusion_off {
        // Walk the execution plan; replace FusedPwDw actions in-place when
        // the next live node is a fusable PW reduce.
        let mut new_plan: Vec<NodeAction> = Vec::with_capacity(execution_plan.len());
        let mut skip_pw_reduce_actions: FxHashSet<usize> = FxHashSet::default();
        let mut fused_count = 0usize;
        let mut pwdw_total = 0usize;
        // Residual Adds the `ConvAdd` matcher already claimed, keyed by the Conv
        // they fused into.
        //
        // When the PW reduce this merge absorbs is that Conv, the merge must
        // take over the whole `ConvAdd` — the Add included — because absorbing
        // the Conv alone deletes the `ConvAdd` action in the `retain` below and
        // the Add's own slot is already `Skip`. Reusing what `ConvAdd` resolved,
        // rather than re-deriving it, is also what keeps the two matchers from
        // disagreeing about which Add belongs to which Conv.
        let conv_add_residuals: FxHashMap<usize, FusedPwDwPwReduceResidual> = execution_plan
            .iter()
            .filter_map(|a| match a {
                NodeAction::ConvAdd {
                    conv_idx,
                    add_idx,
                    skip_input_idx,
                    post_activation,
                    relu_idx,
                } => Some((
                    *conv_idx,
                    FusedPwDwPwReduceResidual {
                        add_idx: *add_idx,
                        residual_skip_input: *skip_input_idx,
                        post_activation: *post_activation,
                        relu_idx: *relu_idx,
                    },
                )),
                _ => None,
            })
            .collect();
        for action in execution_plan.iter() {
            if let NodeAction::FusedPwDw {
                pw_idx,
                dw_idx,
                pw_activation,
                dw_activation,
            } = action
            {
                pwdw_total += 1;
                // Activations: kernel supports None (0) / Relu (1) only.
                if *pw_activation > 1 || *dw_activation > 1 {
                    new_plan.push(action.clone());
                    continue;
                }
                let dw_out = &nodes[*dw_idx].outputs[0];
                let dw_uses = use_counts.get(dw_out).copied().unwrap_or(0);
                if dw_uses != 1 {
                    new_plan.push(action.clone());
                    continue;
                }
                // Streaming kernel supports 3×3 DW always; 5×5 DW when
                // c_exp ≤ 256 (microbench shows regression for wider c_exp,
                // where the existing oc_tiled path's output-channel tiling
                // wins on cache behaviour).
                let dw_shape = nodes[*dw_idx]
                    .inputs
                    .get(1)
                    .and_then(|n| initializers.get(n))
                    .map(|t| t.shape().to_vec());
                // c_exp lives in s[2] for KHWC depthwise [kH, kW, c_exp, 1].
                // 5×5 gated at c_exp ≤ 256: wider c_exp uses the existing
                // `fused_pw_expand_dw_5x5_oc_tiled` path which beats streaming.
                let dw_kernel_size: u8 = match dw_shape.as_deref() {
                    Some(s) if s.len() == 4 && s[0] == 3 && s[1] == 3 => 3,
                    Some(s) if s.len() == 4 && s[0] == 5 && s[1] == 5 && s[2] <= 256 => 5,
                    _ => 0,
                };
                if dw_kernel_size == 0 {
                    new_plan.push(action.clone());
                    continue;
                }
                // The DW output's reader. `dw_uses == 1` above makes it unique,
                // so it is a lookup — this used to take the first non-skipped
                // node after the DW, which is adjacency and lost the fusion
                // whenever the schedule put anything between the two.
                //
                // Declining when that reader is already claimed matches what
                // the scan did: it walked past skipped nodes and then found
                // nothing reading `dw_out`.
                let pw_reduce_node_idx = sole_consumer(dw_out).filter(|&j| {
                    !plan_skip[j] && nodes[j].inputs.first().map(|s| s.as_str()) == Some(dw_out)
                });
                let Some(pw_reduce_idx) = pw_reduce_node_idx else {
                    new_plan.push(action.clone());
                    continue;
                };
                let pw_kind = node_kinds[pw_reduce_idx];
                if !matches!(pw_kind, NodeKind::Conv | NodeKind::ConvRelu) {
                    new_plan.push(action.clone());
                    continue;
                }
                let Some(pw_cp) = &conv_params[pw_reduce_idx] else {
                    new_plan.push(action.clone());
                    continue;
                };
                if !pw_cp.is_pointwise || pw_cp.has_padding {
                    new_plan.push(action.clone());
                    continue;
                }
                // Detect optional residual Add (inverted bottleneck skip).
                // Folded inline by the streaming kernel via the `residual`
                // arg — no per-pixel ConvAdd needed.
                let pw_out = &nodes[pw_reduce_idx].outputs[0];
                let pw_out_uses = use_counts.get(pw_out).copied().unwrap_or(0);
                // Note: don't check plan_skip on Add/Relu — those will be
                // set because the existing ConvAdd fusion has already
                // absorbed them. Our pass subsumes that ConvAdd entirely.
                let residual_meta: Option<FusedPwDwPwReduceResidual> =
                    match conv_add_residuals.get(&pw_reduce_idx) {
                        // Already resolved by `ConvAdd`, by dataflow. Take that
                        // rather than re-deriving it, so the two matchers cannot
                        // disagree about which Add belongs to this Conv.
                        Some(r) => Some(*r),
                        // No `ConvAdd` here — it declines a PW reduce carrying a
                        // fused activation, which this kernel can still absorb. Find
                        // the Add the same way `ConvAdd` would have.
                        None if pw_out_uses == 1 => sole_consumer(pw_out)
                            .filter(|&add_idx| {
                                node_kinds[add_idx] == NodeKind::Add
                                    && nodes[add_idx].inputs.len() == 2
                                    && (nodes[add_idx].inputs[0] == *pw_out
                                        || nodes[add_idx].inputs[1] == *pw_out)
                            })
                            .map(|add_idx| {
                                let add_node = &nodes[add_idx];
                                let add_out = &add_node.outputs[0];
                                let (post_activation, relu_idx) = match sole_consumer(add_out) {
                                    Some(ri)
                                        if node_kinds[ri] == NodeKind::Relu
                                            && nodes[ri].inputs.len() == 1
                                            && nodes[ri].inputs[0] == *add_out =>
                                    {
                                        (1u8, ri as u32)
                                    }
                                    _ => (0u8, 0u32),
                                };
                                FusedPwDwPwReduceResidual {
                                    add_idx,
                                    residual_skip_input: u8::from(add_node.inputs[0] == *pw_out),
                                    post_activation,
                                    relu_idx,
                                }
                            }),
                        None => None,
                    };

                // The merged action runs at the PW expand, which is earlier than
                // either the PW reduce or the Add. So a residual operand that
                // was available where `ConvAdd` fused it need not be available
                // here — `ConvAdd`'s own `available_at` check was made against a
                // later position and does not carry over.
                let residual_meta = residual_meta.filter(|r| {
                    available_at(
                        &nodes[r.add_idx].inputs[usize::from(r.residual_skip_input)],
                        *pw_idx,
                    )
                });

                // Absorbing the PW reduce deletes any `ConvAdd` built on it, so
                // failing to also absorb that Add would drop the addition from
                // the plan entirely — silently, since the Add's own slot is
                // already `Skip`. Decline the merge instead.
                if conv_add_residuals.contains_key(&pw_reduce_idx) && residual_meta.is_none() {
                    new_plan.push(action.clone());
                    continue;
                }
                // Resolve PW reduce weight from initializers.
                let Some(w_name) = nodes[pw_reduce_idx].inputs.get(1) else {
                    new_plan.push(action.clone());
                    continue;
                };
                let Some(weight) = initializers.get(w_name) else {
                    new_plan.push(action.clone());
                    continue;
                };
                let w_shape = weight.shape();
                // KHWC 1×1 weight shape: [1, 1, c_exp, c_out].
                if w_shape.len() != 4 || w_shape[0] != 1 || w_shape[1] != 1 {
                    new_plan.push(action.clone());
                    continue;
                }
                let c_exp = w_shape[2];
                let c_out = w_shape[3];
                // DW's c_exp must match PW reduce c_in.
                let dw_weight_name = nodes[*dw_idx].inputs.get(1);
                let dw_c_exp = dw_weight_name
                    .and_then(|n| initializers.get(n))
                    .map(|t| {
                        let s = t.shape();
                        // KHWC depthwise weight: [kH, kW, c_exp, 1].
                        if s.len() == 4 { s[2] } else { 0 }
                    })
                    .unwrap_or(0);
                if dw_c_exp != c_exp {
                    new_plan.push(action.clone());
                    continue;
                }
                // Build the packed weight: KHWC `[1,1,c_exp,c_out]` is
                // c_exp-major (HWIO order). pack_pw_reduce_weight_for_fusion
                // expects c_out-major `[c_out, c_exp]`, so transpose first.
                let mut khwc_cout_major: Vec<f32> = vec![0.0; c_out * c_exp];
                let w_data = weight.data();
                for cx in 0..c_exp {
                    for oc in 0..c_out {
                        khwc_cout_major[oc * c_exp + cx] = w_data[cx * c_out + oc];
                    }
                }
                let c_out_padded = c_out.div_ceil(16) * 16;
                let weight_packed = yscv_kernels::pack_pw_reduce_weight_for_fusion(
                    &khwc_cout_major,
                    c_out,
                    c_exp,
                    c_out_padded,
                );
                let bias_padded = nodes[pw_reduce_idx]
                    .inputs
                    .get(2)
                    .and_then(|n| initializers.get(n))
                    .and_then(|t| {
                        let bias_data = t.data();
                        if bias_data.len() == c_out {
                            yscv_kernels::pack_pw_reduce_bias_for_fusion(
                                Some(bias_data),
                                c_out,
                                c_out_padded,
                            )
                        } else {
                            None
                        }
                    });
                let pw_reduce_activation: u8 = match pw_kind {
                    NodeKind::ConvRelu => 1,
                    _ => 0,
                };
                prepacked_fused_pw_dw_pw_reduce.insert(
                    pw_reduce_idx,
                    std::sync::Arc::new(FusedPwDwPwReduceWeights {
                        weight_packed,
                        bias_padded,
                        c_out,
                        c_out_padded,
                        c_exp,
                    }),
                );
                skip_pw_reduce_actions.insert(pw_reduce_idx);
                if let Some(r) = &residual_meta {
                    skip_pw_reduce_actions.insert(r.add_idx);
                    if r.post_activation == 1 {
                        skip_pw_reduce_actions.insert(r.relu_idx as usize);
                    }
                }
                fused_count += 1;
                if fusion_debug {
                    let has_res = residual_meta.is_some();
                    eprintln!(
                        "FusedPwDwPwReduce: pw_expand_idx={} dw_idx={} pw_reduce_idx={} c_exp={} c_out={} c_out_padded={} residual={} ({}/{})",
                        *pw_idx,
                        *dw_idx,
                        pw_reduce_idx,
                        c_exp,
                        c_out,
                        c_out_padded,
                        has_res,
                        nodes[*pw_idx].name,
                        nodes[pw_reduce_idx].name,
                    );
                }
                new_plan.push(NodeAction::FusedPwDwPwReduce {
                    pw_expand_idx: *pw_idx,
                    dw_idx: *dw_idx,
                    pw_reduce_idx,
                    pw_expand_activation: *pw_activation,
                    dw_activation: *dw_activation,
                    pw_reduce_activation,
                    dw_kernel_size,
                    residual: residual_meta,
                });
            } else {
                new_plan.push(action.clone());
            }
        }
        // Now drop any action whose target node was just absorbed into
        // a FusedPwDwPwReduce — standalone Conv, ConvAdd, generic Add/Relu.
        new_plan.retain(|act| match act {
            NodeAction::Conv { node_idx, .. } => !skip_pw_reduce_actions.contains(node_idx),
            NodeAction::Generic { node_idx, .. } => !skip_pw_reduce_actions.contains(node_idx),
            NodeAction::ConvAdd {
                conv_idx,
                add_idx,
                relu_idx,
                post_activation,
                ..
            } => {
                let conv_absorbed = skip_pw_reduce_actions.contains(conv_idx);
                let add_absorbed = skip_pw_reduce_actions.contains(add_idx);
                let relu_absorbed =
                    *post_activation == 1 && skip_pw_reduce_actions.contains(&(*relu_idx as usize));
                // Dropping this action is only safe because absorbing the Conv
                // means the merge also took the Add — it declines otherwise.
                // Were that not so, the addition would vanish from the plan
                // with nothing reporting it: the Add's own slot is `Skip`.
                debug_assert!(
                    !conv_absorbed || add_absorbed,
                    "FusedPwDwPwReduce absorbed the Conv of a ConvAdd without its Add"
                );
                !(conv_absorbed || add_absorbed || relu_absorbed)
            }
            _ => true,
        });
        execution_plan = new_plan;
        if fusion_debug {
            eprintln!(
                "FusedPwDwPwReduce summary: fused={}/{} FusedPwDw actions",
                fused_count, pwdw_total
            );
        }
    }

    // Last, because it reads the plan after the `FusedPwDwPwReduce` merge has
    // rewritten and dropped actions.
    let nchwc_handoff = resolve_nchwc_handoff(&execution_plan, nodes, initializers);
    let conv_kernels = resolve_conv_kernels(
        nodes,
        &conv_params,
        initializers,
        khwc_weights,
        dw_khwc_weights,
        group_khwc_weights,
        &prepacked_weights,
    );

    RuntimeModelIndex {
        name_to_id,
        khwc_weight_ids,
        dw_khwc_weight_ids,
        group_khwc_weight_ids,
        use_counts,
        use_counts_by_id,
        node_kinds,
        node_branches,
        node_input_ids,
        node_output_ids,
        conv_params,
        execution_plan,
        prepacked_weights,
        prepacked_weights_by_id,
        prepacked_i8_weights,
        prepacked_i8_depthwise,
        prepacked_fused_pw_dw_pw_reduce,
        reshape_nhwc_passthrough_safe,
        nchwc_handoff,
        conv_kernels,
    }
}
