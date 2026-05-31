//! build_runtime_index: the load-time graph optimizer that classifies
//! each node (NodeKind), plans fusions (NodeAction), and prepacks weights.

use super::*;

pub(super) fn build_runtime_index(
    inputs: &[String],
    outputs: &[String],
    initializers: &HashMap<String, Tensor>,
    nodes: &[OnnxNode],
    khwc_weights: &HashSet<String>,
    dw_khwc_weights: &HashSet<String>,
    group_khwc_weights: &HashSet<String>,
) -> RuntimeModelIndex {
    let mut names: HashSet<&str> = HashSet::new();
    for name in inputs {
        names.insert(name.as_str());
    }
    for name in outputs {
        names.insert(name.as_str());
    }
    for name in initializers.keys() {
        names.insert(name.as_str());
    }
    for node in nodes {
        for name in &node.inputs {
            names.insert(name.as_str());
        }
        for name in &node.outputs {
            names.insert(name.as_str());
        }
    }
    let name_to_id: HashMap<String, usize> = names
        .into_iter()
        .enumerate()
        .map(|(id, name)| (name.to_string(), id))
        .collect();

    let khwc_weight_ids: HashSet<usize> = khwc_weights
        .iter()
        .filter_map(|name| name_to_id.get(name.as_str()).copied())
        .collect();
    let dw_khwc_weight_ids: HashSet<usize> = dw_khwc_weights
        .iter()
        .filter_map(|name| name_to_id.get(name.as_str()).copied())
        .collect();
    let group_khwc_weight_ids: HashSet<usize> = group_khwc_weights
        .iter()
        .filter_map(|name| name_to_id.get(name.as_str()).copied())
        .collect();

    let mut use_counts: HashMap<String, usize> = HashMap::new();
    for node in nodes {
        for inp in &node.inputs {
            if !inp.is_empty() {
                *use_counts.entry(inp.clone()).or_insert(0) += 1;
            }
        }
    }
    let mut use_counts_by_id = vec![0usize; name_to_id.len()];
    for (name, count) in &use_counts {
        if let Some(&id) = name_to_id.get(name) {
            use_counts_by_id[id] = *count;
        }
    }
    let node_kinds: Vec<NodeKind> = nodes
        .iter()
        .map(|node| NodeKind::from_op_type(&node.op_type))
        .collect();

    // Tower-parallel branch classification. For a siamese graph we want two
    // input-rooted subgraphs to run concurrently, then a merge tail. Nodes
    // are tagged 0 = reachable from first dynamic input only, 1 = second only,
    // 2 = shared/merge. If either branch ends up too small, we clear the
    // vector to signal "no parallel split".
    let node_branches: Vec<u8> = {
        let dyn_inputs: Vec<&str> = inputs
            .iter()
            .map(|s| s.as_str())
            .filter(|s| !initializers.contains_key(*s))
            .collect();
        if dyn_inputs.len() >= 2 {
            let mut tensor_branch: HashMap<&str, u8> = HashMap::new();
            tensor_branch.insert(dyn_inputs[0], 0);
            tensor_branch.insert(dyn_inputs[1], 1);
            let mut branches = Vec::with_capacity(nodes.len());
            for node in nodes {
                let mut seen = 0u8; // bitmask: bit 0 = branch 0, bit 1 = branch 1
                for inp in &node.inputs {
                    if inp.is_empty() || initializers.contains_key(inp.as_str()) {
                        continue;
                    }
                    match tensor_branch.get(inp.as_str()) {
                        Some(&0) => seen |= 1,
                        Some(&1) => seen |= 2,
                        Some(&2) => seen |= 3,
                        _ => {}
                    }
                }
                let branch = match seen {
                    0 => 2, // constant-fed node treated as merge-safe
                    1 => 0,
                    2 => 1,
                    _ => 2,
                };
                for out in &node.outputs {
                    tensor_branch.insert(out.as_str(), branch);
                }
                branches.push(branch);
            }
            let b0 = branches.iter().filter(|&&b| b == 0).count();
            let b1 = branches.iter().filter(|&&b| b == 1).count();
            // Require both branches to carry meaningful work, otherwise the
            // parallel split's overhead (env fork, rayon::join) dominates.
            if b0 >= 10 && b1 >= 10 {
                branches
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        }
    };

    // Pre-resolve input names to slot IDs for O(1) hot-path lookups.
    let node_input_ids: Vec<Vec<Option<usize>>> = nodes
        .iter()
        .map(|node| {
            node.inputs
                .iter()
                .map(|name| {
                    if name.is_empty() {
                        None
                    } else {
                        name_to_id.get(name).copied()
                    }
                })
                .collect()
        })
        .collect();

    // pre-resolve output names to slot IDs. Used by
    // `env.insert_by_id` on the hot path to skip the HashMap lookup
    // inside `resolve_id`. `node_input_ids` was already cached; this
    // extends the same optimisation to output slots.
    let node_output_ids: Vec<Vec<Option<usize>>> = nodes
        .iter()
        .map(|node| {
            node.outputs
                .iter()
                .map(|name| {
                    if name.is_empty() {
                        None
                    } else {
                        name_to_id.get(name).copied()
                    }
                })
                .collect()
        })
        .collect();

    // Pre-parse Conv attributes to avoid HashMap lookups in hot path.
    let conv_params: Vec<Option<ConvParams>> = nodes
        .iter()
        .zip(node_kinds.iter())
        .map(|(node, kind)| {
            if !matches!(
                kind,
                NodeKind::Conv | NodeKind::ConvRelu | NodeKind::ConvSilu
            ) {
                return None;
            }
            let strides = node
                .attributes
                .get("strides")
                .and_then(|a| {
                    if let OnnxAttribute::Ints(v) = a {
                        Some(v.clone())
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| vec![1, 1]);
            let pads = node
                .attributes
                .get("pads")
                .and_then(|a| {
                    if let OnnxAttribute::Ints(v) = a {
                        Some(v.clone())
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| vec![0, 0, 0, 0]);
            let group = node
                .attributes
                .get("group")
                .and_then(|a| {
                    if let OnnxAttribute::Int(v) = a {
                        Some(*v as usize)
                    } else {
                        None
                    }
                })
                .unwrap_or(1);
            let (pt, pl, pb, pr) = (
                pads[0] as usize,
                pads[1] as usize,
                pads.get(2).copied().unwrap_or(0) as usize,
                pads.get(3).copied().unwrap_or(0) as usize,
            );
            // Determine depthwise/pointwise from weight shape. Weights
            // may already be permuted to KHWC `[KH, KW, I, O]` by the
            // load-time normalization above (`khwc_weights` pass) for
            // group==1 Conv. Check both layouts and infer which applies.
            //
            // Must dispatch by layout: KHWC-permuted weights are [KH, KW, I, O]
            // (shape[2]=I, shape[3]=O), not OIHW. Reading shape[2]/shape[3] as
            // kernel dims on a KHWC 1×1 weight misclassifies it as non-pointwise
            // and the Conv_Add fast path never fires.
            let weight_name = node.inputs.get(1).map(|s| s.as_str()).unwrap_or("");
            let weight_shape = initializers
                .get(weight_name)
                .map(|t| t.shape().to_vec())
                .unwrap_or_default();
            let weight_is_khwc = khwc_weights.contains(weight_name);
            let weight_is_dw_khwc = dw_khwc_weights.contains(weight_name);
            let weight_is_group_khwc = group_khwc_weights.contains(weight_name);
            // the loader permutes three KHWC
            // variants. DW-permuted `[KH, KW, C, dm]` and grouped
            // `[O, KH, KW, I/G]` previously fell through to the OIHW
            // branch and produced garbage, wrongly setting
            // `is_depthwise = false` for every tracker DW conv. That
            // blocked `FusedDwPw` detection. With the pure-compute
            // `conv_compute_nhwc` split the fused path now keeps the
            // DW intermediate as a local `Tensor` (no env traffic),
            // so enabling this detection no longer regresses tracker.
            let (o_ch, kh_w, kw_w) = if weight_shape.len() == 4 {
                if weight_is_dw_khwc {
                    // Depthwise KHWC: `[KH, KW, C, depth_multiplier]`.
                    let dm = weight_shape[3];
                    (weight_shape[2] * dm, weight_shape[0], weight_shape[1])
                } else if weight_is_group_khwc {
                    // Grouped KHWC: `[O, KH, KW, I/G]`.
                    (weight_shape[0], weight_shape[1], weight_shape[2])
                } else if weight_is_khwc {
                    // Regular KHWC: `[KH, KW, I, O]`.
                    (weight_shape[3], weight_shape[0], weight_shape[1])
                } else {
                    // Plain OIHW: `[O, I, KH, KW]`.
                    (weight_shape[0], weight_shape[2], weight_shape[3])
                }
            } else {
                (0, 0, 0)
            };
            let is_depthwise = group > 1 && group == o_ch;
            let is_pointwise = kh_w == 1 && kw_w == 1 && group == 1;

            Some(ConvParams {
                stride_h: strides[0] as usize,
                stride_w: strides.get(1).copied().unwrap_or(1) as usize,
                pad_top: pt,
                pad_left: pl,
                pad_bottom: pb,
                pad_right: pr,
                group,
                has_padding: pt > 0 || pl > 0 || pb > 0 || pr > 0,
                is_depthwise,
                is_pointwise,
            })
        })
        .collect();

    /// Returns `true` when the Transpose node's `perm` attribute swaps
    /// only the last two axes of a rank-3 tensor (i.e. `[0, 2, 1]`).
    /// Matches the pattern emitted by PyTorch's `.transpose(-2, -1)` on
    /// 3-D tensors — the pattern ORT folds into its
    /// `MatmulTransposeFusion` contrib op.
    fn transpose_perm_is_swap_last_two(node: &OnnxNode) -> bool {
        let perm = match node.attributes.get("perm") {
            Some(OnnxAttribute::Ints(p)) => p,
            _ => return false,
        };
        matches!(perm.as_slice(), [0, 2, 1])
    }

    fn init_scalar(initializers: &HashMap<String, Tensor>, name: &str) -> Option<f32> {
        initializers
            .get(name)
            .and_then(|t| t.data().first())
            .copied()
    }

    fn matching_zero_qparams(
        dequant: &OnnxNode,
        quant: &OnnxNode,
        initializers: &HashMap<String, Tensor>,
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

    /// Classify a `QLinearConv` by weight initializer shape into the chain
    /// roles we currently fuse. Returns `Some("pw")` for 1×1 group-1 PW,
    /// `Some("dw")` for 3×3/5×5 depthwise (group=c_out=c_in*group), and
    /// `None` otherwise. Mirrors `bench_tracker::qlinear_conv_kind` so the
    /// load-time detector and the static counter agree node-for-node.
    fn qlc_kind(node: &OnnxNode, initializers: &HashMap<String, Tensor>) -> Option<&'static str> {
        if node.op_type != "QLinearConv" {
            return None;
        }
        let w_name = node.inputs.get(3)?;
        let weight = initializers.get(w_name)?;
        let shape = weight.shape();
        if shape.len() != 4 {
            return None;
        }
        let group = match node.attributes.get("group") {
            Some(OnnxAttribute::Int(v)) => *v,
            _ => 1,
        };
        let dilations = match node.attributes.get("dilations") {
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
        let pads = match node.attributes.get("pads") {
            Some(OnnxAttribute::Ints(v)) => v.clone(),
            _ => vec![0, 0, 0, 0],
        };
        let strides = match node.attributes.get("strides") {
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
        initializers: &HashMap<String, Tensor>,
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
    let producers: HashMap<String, usize> = {
        let mut m: HashMap<String, usize> = HashMap::with_capacity(nodes.len());
        for (idx, node) in nodes.iter().enumerate() {
            for out in &node.outputs {
                if !out.is_empty() {
                    m.insert(out.clone(), idx);
                }
            }
        }
        m
    };

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
        //   * load-time prepacking for both PW (VNNI 4×16) and DW (KHWC i8)
        //     is fired by the existing prepack pass under the same shape
        //     gates we check here, so by the time the chain dispatches the
        //     `prepacked_i8_b` / `prepacked_i8_depthwise` lookups can't miss.
        if nodes[i].op_type == "QLinearConv"
            && qlc_kind(&nodes[i], initializers) == Some("pw")
            && qlc_zps_match_chain(&nodes[i], initializers, true)
        {
            let pw_out = nodes[i].outputs.first().map(String::as_str).unwrap_or("");
            let pw_w_name = nodes[i].inputs.get(3).map(String::as_str).unwrap_or("");
            let dq_idx = i + 1;
            if !pw_out.is_empty()
                && !pw_w_name.is_empty()
                && use_counts.get(pw_out).copied().unwrap_or(0) == 1
                && !outputs.iter().any(|o| o == pw_out)
                && nodes
                    .get(dq_idx)
                    .is_some_and(|n| n.op_type == "DequantizeLinear")
                && !plan_skip[dq_idx]
                && nodes[dq_idx].inputs.first().map(String::as_str) == Some(pw_out)
            {
                let dq_out = nodes[dq_idx]
                    .outputs
                    .first()
                    .map(String::as_str)
                    .unwrap_or("");
                let mut relu_idx = None;
                let mut q_idx = dq_idx + 1;
                let mut q_input = dq_out;
                if let Some(relu_node) = nodes.get(q_idx)
                    && relu_node.op_type == "Relu"
                    && !plan_skip[q_idx]
                    && relu_node.inputs.first().map(String::as_str) == Some(dq_out)
                    && use_counts.get(dq_out).copied().unwrap_or(0) == 1
                    && !outputs.iter().any(|o| o == dq_out)
                {
                    let relu_out = relu_node.outputs.first().map(String::as_str).unwrap_or("");
                    if !relu_out.is_empty() && !outputs.iter().any(|o| o == relu_out) {
                        relu_idx = Some(q_idx);
                        q_input = relu_out;
                        q_idx += 1;
                    }
                }
                if !dq_out.is_empty()
                    && !q_input.is_empty()
                    && nodes
                        .get(q_idx)
                        .is_some_and(|n| n.op_type == "QuantizeLinear")
                    && !plan_skip[q_idx]
                    && nodes[q_idx].inputs.first().map(String::as_str) == Some(q_input)
                    && use_counts.get(q_input).copied().unwrap_or(0) == 1
                    && matching_zero_qparams(&nodes[dq_idx], &nodes[q_idx], initializers)
                {
                    let q_out = nodes[q_idx]
                        .outputs
                        .first()
                        .map(String::as_str)
                        .unwrap_or("");
                    let dw_idx = q_idx + 1;
                    if !q_out.is_empty()
                        && !outputs.iter().any(|o| o == q_out)
                        && use_counts.get(q_out).copied().unwrap_or(0) == 1
                        && nodes
                            .get(dw_idx)
                            .is_some_and(|n| n.op_type == "QLinearConv")
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
                let dq_idx = i + 1;
                if !dw_out.is_empty()
                    && use_counts.get(dw_out).copied().unwrap_or(0) == 1
                    && !outputs.iter().any(|o| o == dw_out)
                    && nodes
                        .get(dq_idx)
                        .is_some_and(|n| n.op_type == "DequantizeLinear")
                    && !plan_skip[dq_idx]
                    && nodes[dq_idx].inputs.first().map(String::as_str) == Some(dw_out)
                {
                    let dq_out = nodes[dq_idx]
                        .outputs
                        .first()
                        .map(String::as_str)
                        .unwrap_or("");
                    let mut relu_idx = None;
                    let mut q_idx = dq_idx + 1;
                    let mut q_input = dq_out;
                    if let Some(relu_node) = nodes.get(q_idx)
                        && relu_node.op_type == "Relu"
                        && !plan_skip[q_idx]
                        && relu_node.inputs.first().map(String::as_str) == Some(dq_out)
                        && use_counts.get(dq_out).copied().unwrap_or(0) == 1
                        && !outputs.iter().any(|o| o == dq_out)
                    {
                        let relu_out = relu_node.outputs.first().map(String::as_str).unwrap_or("");
                        if !relu_out.is_empty() && !outputs.iter().any(|o| o == relu_out) {
                            relu_idx = Some(q_idx);
                            q_input = relu_out;
                            q_idx += 1;
                        }
                    }
                    if !dq_out.is_empty()
                        && !q_input.is_empty()
                        && nodes
                            .get(q_idx)
                            .is_some_and(|n| n.op_type == "QuantizeLinear")
                        && !plan_skip[q_idx]
                        && nodes[q_idx].inputs.first().map(String::as_str) == Some(q_input)
                        && use_counts.get(q_input).copied().unwrap_or(0) == 1
                        && matching_zero_qparams(&nodes[dq_idx], &nodes[q_idx], initializers)
                    {
                        let q_out = nodes[q_idx]
                            .outputs
                            .first()
                            .map(String::as_str)
                            .unwrap_or("");
                        let pw_idx = q_idx + 1;
                        if !q_out.is_empty()
                            && !outputs.iter().any(|o| o == q_out)
                            && use_counts.get(q_out).copied().unwrap_or(0) == 1
                            && nodes
                                .get(pw_idx)
                                .is_some_and(|n| n.op_type == "QLinearConv")
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
            let dq_idx = i + 1;
            if !first_out.is_empty()
                && use_counts.get(first_out).copied().unwrap_or(0) == 1
                && !outputs.iter().any(|o| o == first_out)
                && nodes
                    .get(dq_idx)
                    .is_some_and(|n| n.op_type == "DequantizeLinear")
                && !plan_skip[dq_idx]
                && nodes[dq_idx].inputs.first().map(String::as_str) == Some(first_out)
            {
                let dq_out = nodes[dq_idx]
                    .outputs
                    .first()
                    .map(String::as_str)
                    .unwrap_or("");
                let mut relu_idx = None;
                let mut q_idx = dq_idx + 1;
                let mut q_input = dq_out;
                if let Some(relu_node) = nodes.get(q_idx)
                    && relu_node.op_type == "Relu"
                    && !plan_skip[q_idx]
                    && relu_node.inputs.first().map(String::as_str) == Some(dq_out)
                {
                    let relu_out = relu_node.outputs.first().map(String::as_str).unwrap_or("");
                    if !relu_out.is_empty() {
                        relu_idx = Some(q_idx);
                        q_input = relu_out;
                        q_idx += 1;
                    }
                }
                if !q_input.is_empty()
                    && nodes
                        .get(q_idx)
                        .is_some_and(|n| n.op_type == "QuantizeLinear")
                    && !plan_skip[q_idx]
                    && nodes[q_idx].inputs.first().map(String::as_str) == Some(q_input)
                    && use_counts.get(q_input).copied().unwrap_or(0) > 1
                {
                    let q_out = nodes[q_idx]
                        .outputs
                        .first()
                        .map(String::as_str)
                        .unwrap_or("");
                    let second_idx = q_idx + 1;
                    let want_second = if first_kind == "pw" { "dw" } else { "pw" };
                    if !q_out.is_empty()
                        && !outputs.iter().any(|o| o == q_out)
                        && use_counts.get(q_out).copied().unwrap_or(0) == 1
                        && nodes
                            .get(second_idx)
                            .is_some_and(|n| n.op_type == "QLinearConv")
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
            let qconv_out = nodes[i].outputs.first().map(String::as_str).unwrap_or("");
            let dq_idx = i + 1;
            let relu_idx = i + 2;
            let conv_idx = i + 3;
            let add_idx = i + 4;
            let q_idx = i + 5;
            if !qconv_out.is_empty()
                && use_counts.get(qconv_out).copied().unwrap_or(0) == 1
                && !outputs.iter().any(|o| o == qconv_out)
                && nodes
                    .get(dq_idx)
                    .is_some_and(|n| n.op_type == "DequantizeLinear")
                && nodes.get(relu_idx).is_some_and(|n| n.op_type == "Relu")
                && nodes.get(conv_idx).is_some_and(|n| n.op_type == "Conv")
                && nodes.get(add_idx).is_some_and(|n| n.op_type == "Add")
                && nodes
                    .get(q_idx)
                    .is_some_and(|n| n.op_type == "QuantizeLinear")
                && !plan_skip[dq_idx]
                && !plan_skip[relu_idx]
                && !plan_skip[conv_idx]
                && !plan_skip[add_idx]
                && !plan_skip[q_idx]
                && nodes[dq_idx].inputs.first().map(String::as_str) == Some(qconv_out)
                && nodes[relu_idx].inputs.first() == nodes[dq_idx].outputs.first()
                && nodes[conv_idx].inputs.first() == nodes[relu_idx].outputs.first()
                && nodes[add_idx]
                    .inputs
                    .iter()
                    .any(|input| Some(input) == nodes[conv_idx].outputs.first())
                && nodes[q_idx].inputs.first() == nodes[add_idx].outputs.first()
                && use_counts
                    .get(
                        nodes[dq_idx]
                            .outputs
                            .first()
                            .map(String::as_str)
                            .unwrap_or(""),
                    )
                    .copied()
                    .unwrap_or(0)
                    == 1
                && use_counts
                    .get(
                        nodes[relu_idx]
                            .outputs
                            .first()
                            .map(String::as_str)
                            .unwrap_or(""),
                    )
                    .copied()
                    .unwrap_or(0)
                    == 1
                && use_counts
                    .get(
                        nodes[conv_idx]
                            .outputs
                            .first()
                            .map(String::as_str)
                            .unwrap_or(""),
                    )
                    .copied()
                    .unwrap_or(0)
                    == 1
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
            let dq_idx = i + 1;
            if !qconv_out.is_empty()
                && use_counts.get(qconv_out).copied().unwrap_or(0) == 1
                && !outputs.iter().any(|o| o == qconv_out)
                && nodes
                    .get(dq_idx)
                    .is_some_and(|n| n.op_type == "DequantizeLinear")
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
                let mut quant_idx = i + 1;
                if let Some(relu_node) = nodes.get(i + 1)
                    && !plan_skip[i + 1]
                    && node_kinds[i + 1] == NodeKind::Relu
                    && relu_node.inputs.len() == 1
                    && relu_node.inputs[0] == dequant_out
                    && use_counts.get(dequant_out).copied().unwrap_or(0) == 1
                    && !outputs.iter().any(|o| o == &relu_node.outputs[0])
                {
                    relu_idx = Some(i + 1);
                    quant_idx = i + 2;
                }

                if let Some(quant_node) = nodes.get(quant_idx)
                    && !plan_skip[quant_idx]
                    && quant_node.op_type == "QuantizeLinear"
                    && quant_node.inputs.len() >= 3
                    && quant_node.outputs.len() == 1
                    && ((relu_idx.is_none() && quant_node.inputs[0] == dequant_out)
                        || relu_idx
                            .map(|ri| quant_node.inputs[0] == nodes[ri].outputs[0])
                            .unwrap_or(false))
                    && use_counts.get(&quant_node.inputs[0]).copied().unwrap_or(0) == 1
                    && matching_zero_qparams(&nodes[i], quant_node, initializers)
                {
                    execution_plan.push(NodeAction::QuantizedQdq {
                        dequant_idx: i,
                        relu_idx,
                        quant_idx,
                    });
                    if let Some(ri) = relu_idx {
                        plan_skip[ri] = true;
                    }
                    plan_skip[quant_idx] = true;
                    continue;
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
                    if dw_uses == 1 {
                        for j in (i + 1)..nodes.len() {
                            if plan_skip[j] {
                                continue;
                            }
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
                                let pw_has_convadd = pw_kind_plain
                                    && pw_out_uses == 1
                                    && nodes.get(j + 1).is_some_and(|n| {
                                        node_kinds[j + 1] == NodeKind::Add
                                            && n.inputs.len() == 2
                                            && (n.inputs[0] == *pw_out || n.inputs[1] == *pw_out)
                                    });
                                if pw_has_convadd {
                                    break;
                                }
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
                            break; // only check next non-skipped
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
                    if pw_uses == 1 {
                        for j in (i + 1)..nodes.len() {
                            if plan_skip[j] {
                                continue;
                            }
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
                            break; // only check next non-skipped
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
                        && let Some(add_node) = nodes.get(i + 1)
                        && node_kinds[i + 1] == NodeKind::Add
                        && add_node.inputs.len() == 2
                        && !plan_skip[i + 1]
                        && (add_node.inputs[0] == *conv_out || add_node.inputs[1] == *conv_out)
                    {
                        let skip_input_idx: u8 = if add_node.inputs[0] == *conv_out {
                            1
                        } else {
                            0
                        };
                        let add_out = &add_node.outputs[0];
                        let (post_activation, relu_idx_field) =
                            if let Some(relu_node) = nodes.get(i + 2) {
                                if node_kinds[i + 2] == NodeKind::Relu
                                    && !plan_skip[i + 2]
                                    && relu_node.inputs.len() == 1
                                    && relu_node.inputs[0] == *add_out
                                    && use_counts.get(add_out).copied().unwrap_or(0) == 1
                                {
                                    (1u8, (i + 2) as u32)
                                } else {
                                    (0u8, 0u32)
                                }
                            } else {
                                (0u8, 0u32)
                            };
                        execution_plan.push(NodeAction::ConvAdd {
                            conv_idx: i,
                            add_idx: i + 1,
                            skip_input_idx,
                            post_activation,
                            relu_idx: relu_idx_field,
                        });
                        plan_skip[i + 1] = true;
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
                let mut emitted = false;
                if !left_input.is_empty()
                    && let Some(&t_idx) = producers.get(left_input)
                    && t_idx < nodes.len()
                    && node_kinds[t_idx] == NodeKind::Transpose
                    && !plan_skip[t_idx]
                    && transpose_perm_is_swap_last_two(&nodes[t_idx])
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

    // Post-pass: elide Transpose nodes whose every consumer is a
    // `FusedTransposeMatMul` that absorbed them. Counts the number of
    // fused actions pointing at each transpose and compares to the
    // transpose output's total graph-use count (input edges + model
    // output membership). When every consumer was absorbed, the
    // original Transpose does no useful work and becomes `Skip`.
    let model_outputs: HashSet<&str> = outputs.iter().map(|s| s.as_str()).collect();
    let mut fused_refs: Vec<usize> = vec![0; nodes.len()];
    // Plan position of the last `FusedTransposeMatMul` referencing each
    // transpose idx. Used to mark exactly one variant as the cleanup
    // owner so the pre-transpose tensor stays in `env` until every
    // consumer has read it.
    let mut last_fused_pos: HashMap<usize, usize> = HashMap::new();
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
    let mut reshape_nhwc_passthrough_safe: HashSet<String> = HashSet::new();
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
    let mut prepacked_weights: HashMap<String, std::sync::Arc<yscv_kernels::PackedB>> =
        HashMap::new();
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

    fn attr_int(node: &OnnxNode, name: &str, default: i64) -> i64 {
        match node.attributes.get(name) {
            Some(OnnxAttribute::Int(v)) => *v,
            _ => default,
        }
    }

    fn attr_ints(node: &OnnxNode, name: &str, default: &[i64]) -> Vec<i64> {
        match node.attributes.get(name) {
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

    let mut prepacked_i8_weights: HashMap<String, std::sync::Arc<yscv_kernels::PackedI8B>> =
        HashMap::new();
    let mut prepacked_i8_depthwise: HashMap<String, std::sync::Arc<Vec<i8>>> = HashMap::new();
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
    let chain_pw_weights: std::collections::HashSet<String> = execution_plan
        .iter()
        .filter_map(|action| match action {
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
                let group = attr_int(node, "group", 1);
                let dilations = attr_ints(node, "dilations", &[1, 1]);
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
    let mut prepacked_fused_pw_dw_pw_reduce: HashMap<
        usize,
        std::sync::Arc<FusedPwDwPwReduceWeights>,
    > = HashMap::new();
    let fusion_off = std::env::var_os("YSCV_FUSED_PW_DW_PW_REDUCE_OFF").is_some()
        || (cfg!(target_arch = "aarch64")
            && std::env::var_os("YSCV_FUSED_PW_DW_PW_REDUCE_ON").is_none());
    let fusion_debug = std::env::var_os("YSCV_FUSED_PW_DW_PW_REDUCE_DEBUG").is_some();
    if !fusion_off {
        // Walk the execution plan; replace FusedPwDw actions in-place when
        // the next live node is a fusable PW reduce.
        let mut new_plan: Vec<NodeAction> = Vec::with_capacity(execution_plan.len());
        let mut skip_pw_reduce_actions: HashSet<usize> = HashSet::new();
        let mut fused_count = 0usize;
        let mut pwdw_total = 0usize;
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
                // Find the immediate consumer of the DW output by scanning
                // forward past nodes already marked as skipped.
                let mut pw_reduce_node_idx: Option<usize> = None;
                for j in (*dw_idx + 1)..nodes.len() {
                    if plan_skip[j] {
                        continue;
                    }
                    if nodes[j].inputs.first().map(|s| s.as_str()) == Some(dw_out.as_str()) {
                        pw_reduce_node_idx = Some(j);
                    }
                    break;
                }
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
                let residual_meta: Option<FusedPwDwPwReduceResidual> = if pw_out_uses == 1
                    && let Some(add_node) = nodes.get(pw_reduce_idx + 1)
                    && node_kinds[pw_reduce_idx + 1] == NodeKind::Add
                    && add_node.inputs.len() == 2
                    && (add_node.inputs[0] == *pw_out || add_node.inputs[1] == *pw_out)
                {
                    let residual_skip_input: u8 = if add_node.inputs[0] == *pw_out { 1 } else { 0 };
                    let add_out = &add_node.outputs[0];
                    let (post_activation, relu_idx) = if let Some(relu_node) =
                        nodes.get(pw_reduce_idx + 2)
                        && node_kinds[pw_reduce_idx + 2] == NodeKind::Relu
                        && relu_node.inputs.len() == 1
                        && relu_node.inputs[0] == *add_out
                        && use_counts.get(add_out).copied().unwrap_or(0) == 1
                    {
                        (1u8, (pw_reduce_idx + 2) as u32)
                    } else {
                        (0u8, 0u32)
                    };
                    Some(FusedPwDwPwReduceResidual {
                        add_idx: pw_reduce_idx + 1,
                        residual_skip_input,
                        post_activation,
                        relu_idx,
                    })
                } else {
                    None
                };
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
    }
}
