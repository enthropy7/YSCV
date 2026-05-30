//! Top-level inference drivers: the JIT (timed-loop) and sequential
//! single-pass executors that walk the optimized plan via execute_plan_branch.

use super::*;

/// JIT execution path: pre-compiled dispatch table, no per-node matching.
/// Skips NodeKind matching, layout checks are minimal, Conv params pre-resolved.
pub(crate) fn run_onnx_model_jit(
    model: &OnnxModel,
    mut env: TensorEnv<'_, '_>,
) -> Result<HashMap<String, Tensor>, OnnxError> {
    let branches = &model.runtime_index.node_branches;
    let use_counts_by_id = &model.runtime_index.use_counts_by_id;
    let output_id_mask = build_output_id_mask(model, &env, use_counts_by_id.len());

    let do_profile = std::env::var("YSCV_PROFILE").is_ok();
    let mut conv_ns: u64 = 0;
    let mut other_ns: u64 = 0;
    let mut conv_count: u32 = 0;
    let mut other_count: u32 = 0;

    // Tower-parallel: if the graph splits into two input-rooted subgraphs,
    // run them concurrently, then merge back for the shared tail. Each branch
    // gets its own env fork so concurrent inserts don't race.
    //
    // Keep explicit env control for A/B:
    //   - `YSCV_NO_TOWER_PARALLEL=1`   force OFF
    //   - `YSCV_FORCE_TOWER_PARALLEL=1` force ON
    //   - `YSCV_TOWER_MIN_THREADS=<N>`  default gate override
    let thread_count = rayon::current_num_threads();
    let no_tower_parallel = std::env::var_os("YSCV_NO_TOWER_PARALLEL").is_some();
    let force_tower_parallel = matches!(
        std::env::var_os("YSCV_FORCE_TOWER_PARALLEL").as_deref(),
        Some(v) if v == "1"
    );
    let default_tower_min_threads = 2usize;
    let tower_min_threads = std::env::var("YSCV_TOWER_MIN_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(default_tower_min_threads);
    let use_tower_parallel = !branches.is_empty()
        && (force_tower_parallel || (!no_tower_parallel && thread_count >= tower_min_threads));
    if use_tower_parallel {
        let branches_ref = branches.as_slice();
        let mut env0 = env.fork();
        let mut env1 = env.fork();
        let mut remaining0 = use_counts_by_id.clone();
        let mut remaining1 = use_counts_by_id.clone();

        // A″.5: route the tower-parallel fork through the installed
        // `ParallelScope` instead of calling `rayon::join` directly.
        // Under `YSCV_POOL=yscv` this goes through `YscvPool::join_dyn`
        // (no rayon touched). Under `YSCV_POOL=rayon` (default) the
        // rayon-backed ParallelScope ends up calling `rayon::join`
        // internally — same runtime behaviour.
        let mut r0: Result<(), OnnxError> = Ok(());
        let mut r1: Result<(), OnnxError> = Ok(());
        yscv_kernels::with_scope(|scope| {
            let mut a = || {
                let mut c_ns = 0u64;
                let mut o_ns = 0u64;
                let mut c_n = 0u32;
                let mut o_n = 0u32;
                r0 = execute_plan_branch(
                    model,
                    &mut env0,
                    &mut remaining0,
                    &output_id_mask,
                    |nidx| branches_ref.get(nidx).copied() == Some(0),
                    &mut c_ns,
                    &mut o_ns,
                    &mut c_n,
                    &mut o_n,
                    do_profile,
                );
            };
            let mut b = || {
                let mut c_ns = 0u64;
                let mut o_ns = 0u64;
                let mut c_n = 0u32;
                let mut o_n = 0u32;
                r1 = execute_plan_branch(
                    model,
                    &mut env1,
                    &mut remaining1,
                    &output_id_mask,
                    |nidx| branches_ref.get(nidx).copied() == Some(1),
                    &mut c_ns,
                    &mut o_ns,
                    &mut c_n,
                    &mut o_n,
                    do_profile,
                );
            };
            if let Some(scope) = scope {
                scope.join_dyn(&mut a, &mut b);
            } else {
                // Fallback: no scope installed (test harness / benches).
                rayon::join(a, b);
            }
        });
        r0?;
        r1?;

        env.merge_from(env0);
        env.merge_from(env1);

        // Merge-branch (id 2) runs on the reunited env.
        let mut remaining: Vec<usize> = use_counts_by_id.clone();
        execute_plan_branch(
            model,
            &mut env,
            &mut remaining,
            &output_id_mask,
            |nidx| {
                branches_ref.get(nidx).copied() != Some(0)
                    && branches_ref.get(nidx).copied() != Some(1)
            },
            &mut conv_ns,
            &mut other_ns,
            &mut conv_count,
            &mut other_count,
            do_profile,
        )?;
    } else {
        let mut remaining_uses: Vec<usize> = use_counts_by_id.clone();
        execute_plan_branch(
            model,
            &mut env,
            &mut remaining_uses,
            &output_id_mask,
            |_| true,
            &mut conv_ns,
            &mut other_ns,
            &mut conv_count,
            &mut other_count,
            do_profile,
        )?;
    }

    if do_profile {
        eprintln!(
            "\n[JIT profile] Conv: {:.1}ms ({} ops, {:.0}µs/op) | Other: {:.1}ms ({} ops, {:.0}µs/op) | Total: {:.1}ms",
            conv_ns as f64 / 1e6,
            conv_count,
            if conv_count > 0 {
                conv_ns as f64 / conv_count as f64 / 1e3
            } else {
                0.0
            },
            other_ns as f64 / 1e6,
            other_count,
            if other_count > 0 {
                other_ns as f64 / other_count as f64 / 1e3
            } else {
                0.0
            },
            (conv_ns + other_ns) as f64 / 1e6,
        );
    }

    // Ensure outputs in NCHW
    for name in &model.outputs {
        env.materialize_quant_i8_raw(name)?;
        ensure_nchw(&mut env, name)?;
    }
    let mut result = HashMap::with_capacity(model.outputs.len());
    for name in &model.outputs {
        if let Some(t) = env.remove(name) {
            result.insert(name.clone(), t);
        } else if let Some(t) = env.get(name) {
            result.insert(name.clone(), t.clone());
        }
    }
    Ok(result)
}

pub(crate) fn run_onnx_model_sequential(
    model: &OnnxModel,
    mut env: TensorEnv<'_, '_>,
) -> Result<HashMap<String, Tensor>, OnnxError> {
    // --- Operator fusion: scan for fusible patterns ---
    // Build a set of node indices that should be skipped because they were
    // fused into the preceding node.  We also create synthetic "fused" nodes
    // that carry a combined op_type (e.g. "Conv_Relu").
    let nodes = &model.nodes;
    let node_kinds = &model.runtime_index.node_kinds;
    let mut skip = vec![false; nodes.len()];

    // Build reference counts: how many nodes consume each tensor as input.
    // Used by SiLU fusions to decide in-place vs allocating path.
    let use_counts = &model.runtime_index.use_counts;
    let use_counts_by_id = &model.runtime_index.use_counts_by_id;

    // Mutable remaining-use counters for early tensor deallocation.
    // When a tensor's remaining uses reach zero, free it to reduce working set.
    let mut remaining_uses: Vec<usize> = use_counts_by_id.clone();
    let output_id_mask = build_output_id_mask(model, &env, use_counts_by_id.len());

    for (i, node) in nodes.iter().enumerate() {
        if skip[i] {
            continue;
        }
        let kind = node_kind(node_kinds, nodes, i);

        // --- Conv → BatchNorm → Relu 3-node fusion ---
        if kind == NodeKind::Conv
            && let Some(next) = nodes.get(i + 1)
            && node_kind(node_kinds, nodes, i + 1) == NodeKind::BatchNormalization
            && !next.inputs.is_empty()
            && next.inputs[0] == node.outputs[0]
            && let Some((relu_idx, identity_idxs)) =
                find_relu_after_identity_chain(nodes, node_kinds, i + 2, &next.outputs[0])
        {
            execute_node_with_layout_kind(node, &mut env, kind)?;
            execute_node_with_layout_kind(next, &mut env, node_kind(node_kinds, nodes, i + 1))?;
            if let Some(tensor) = env.get_mut(&next.outputs[0]) {
                relu_inplace(tensor);
            }
            let source = &next.outputs[0];
            for &id_idx in &identity_idxs {
                env.alias(&nodes[id_idx].outputs[0], source);
            }
            env.alias(&nodes[relu_idx].outputs[0], source);
            if i + 1 < skip.len() {
                skip[i + 1] = true;
            }
            mark_skip_indices(&mut skip, &identity_idxs);
            mark_skip_indices(&mut skip, &[relu_idx]);
            continue;
        }

        // --- Conv + Relu fusion ---
        if kind == NodeKind::Conv
            && let Some((relu_idx, identity_idxs)) =
                find_relu_after_identity_chain(nodes, node_kinds, i + 1, &node.outputs[0])
        {
            exec_conv(node, &mut env, yscv_kernels::Activation::Relu)?;
            let source = &node.outputs[0];
            for &id_idx in &identity_idxs {
                env.alias(&nodes[id_idx].outputs[0], source);
            }
            env.alias(&nodes[relu_idx].outputs[0], source);
            mark_skip_indices(&mut skip, &identity_idxs);
            mark_skip_indices(&mut skip, &[relu_idx]);
            continue;
        }

        // --- Conv + SiLU fusion (Conv → Sigmoid → Mul) ---
        // Detect Sigmoid at i+1 and Mul at i+2 that form SiLU on Conv output.
        if kind == NodeKind::Conv {
            let conv_out = &node.outputs[0];
            // Look for Sigmoid(conv_out) → Mul(conv_out, sigmoid_out) pattern
            let mut silu_mul_idx = None;
            for sig_offset in 1..=2 {
                if let Some(sig) = nodes.get(i + sig_offset)
                    && node_kind(node_kinds, nodes, i + sig_offset) == NodeKind::Sigmoid
                    && sig.inputs.len() == 1
                    && sig.inputs[0] == *conv_out
                {
                    let sig_out = &sig.outputs[0];
                    for mul_offset in (sig_offset + 1)..=(sig_offset + 2) {
                        if let Some(mul) = nodes.get(i + mul_offset)
                            && node_kind(node_kinds, nodes, i + mul_offset) == NodeKind::Mul
                            && mul.inputs.len() == 2
                            && ((mul.inputs[0] == *sig_out && mul.inputs[1] == *conv_out)
                                || (mul.inputs[1] == *sig_out && mul.inputs[0] == *conv_out))
                        {
                            silu_mul_idx = Some((sig_offset, mul_offset, mul.outputs[0].clone()));
                            break;
                        }
                    }
                    if silu_mul_idx.is_some() {
                        break;
                    }
                }
            }
            if let Some((sig_off, mul_off, mul_out)) = silu_mul_idx {
                let conv_out_uses = tensor_use_count(&env, use_counts_by_id, use_counts, conv_out);
                if conv_out_uses <= 2 {
                    // Fuse SiLU into Conv GEMM tiles (applied cache-hot after bias).
                    exec_conv(node, &mut env, yscv_kernels::Activation::Silu)?;
                    env.alias(&mul_out, conv_out);
                } else {
                    // Other consumers need raw conv_out — can't fuse.
                    execute_node_with_layout_kind(node, &mut env, kind)?;
                    if let Some(tensor) = env.get(conv_out) {
                        let result = yscv_kernels::silu(tensor);
                        env.insert(mul_out.clone(), result);
                    }
                }
                let is_nhwc = env.is_nhwc(conv_out);
                if is_nhwc {
                    env.mark_nhwc(&mul_out);
                }
                // Execute any intermediate nodes between Conv and Sigmoid,
                // then mark them as done so the main loop doesn't re-execute them.
                for mid in 1..sig_off {
                    if i + mid < skip.len() && !skip[i + mid] {
                        execute_node_with_layout_kind(
                            &nodes[i + mid],
                            &mut env,
                            node_kind(node_kinds, nodes, i + mid),
                        )?;
                        skip[i + mid] = true;
                    }
                }
                if i + sig_off < skip.len() {
                    skip[i + sig_off] = true;
                }
                if i + mul_off < skip.len() {
                    skip[i + mul_off] = true;
                }
                continue;
            }
        }

        // --- Conv + Add (residual connection) fusion ---
        // Pattern: Conv → Add(conv_out, skip_connection), optionally → Relu
        // Reuses conv_out buffer for the result, avoiding allocation.
        if kind == NodeKind::Conv
            && let Some(next) = nodes.get(i + 1)
            && node_kind(node_kinds, nodes, i + 1) == NodeKind::Add
            && next.inputs.len() == 2
            && (next.inputs[0] == node.outputs[0] || next.inputs[1] == node.outputs[0])
        {
            let conv_out = &node.outputs[0];
            let skip_idx = if &next.inputs[0] == conv_out { 1 } else { 0 };
            let skip_name = &next.inputs[skip_idx];
            let conv_out_uses = tensor_use_count(&env, use_counts_by_id, use_counts, conv_out);

            // Only fuse if conv_out has 2 uses (Add is its only other consumer besides the
            // initializer lookup that may happen). If it has more uses, we need to keep it
            // for other consumers.
            if conv_out_uses <= 2 {
                execute_node_with_layout_kind(node, &mut env, kind)?;

                // Capture NHWC flag before remove (remove clears it).
                let is_nhwc = env.is_nhwc(conv_out);

                // Add skip_connection in-place to conv_out
                if let Some(mut conv_tensor) = env.remove(conv_out) {
                    if let Some(skip_tensor) = env.get(skip_name) {
                        yscv_kernels::add_inplace(&mut conv_tensor, skip_tensor);
                        let add_out = &next.outputs[0];

                        // Check if Relu follows Add
                        if let Some((relu_idx, identity_idxs)) =
                            find_relu_after_identity_chain(nodes, node_kinds, i + 2, add_out)
                        {
                            relu_inplace(&mut conv_tensor);
                            env.insert(add_out.clone(), conv_tensor);
                            if is_nhwc {
                                env.mark_nhwc(add_out);
                            }
                            let source = add_out;
                            for &id_idx in &identity_idxs {
                                env.alias(&nodes[id_idx].outputs[0], source);
                            }
                            env.alias(&nodes[relu_idx].outputs[0], source);
                            mark_skip_indices(&mut skip, &identity_idxs);
                            mark_skip_indices(&mut skip, &[relu_idx]);
                        } else {
                            env.insert(add_out.clone(), conv_tensor);
                            if is_nhwc {
                                env.mark_nhwc(add_out);
                            }
                        }

                        if i + 1 < skip.len() {
                            skip[i + 1] = true;
                        }
                        continue;
                    } else {
                        env.insert(conv_out.clone(), conv_tensor);
                    }
                }
            }
        }

        // --- BatchNormalization + Relu fusion ---
        if kind == NodeKind::BatchNormalization
            && let Some((relu_idx, identity_idxs)) =
                find_relu_after_identity_chain(nodes, node_kinds, i + 1, &node.outputs[0])
        {
            execute_node_with_layout_kind(node, &mut env, kind)?;
            if let Some(tensor) = env.get_mut(&node.outputs[0]) {
                relu_inplace(tensor);
            }
            let source = &node.outputs[0];
            for &id_idx in &identity_idxs {
                env.alias(&nodes[id_idx].outputs[0], source);
            }
            env.alias(&nodes[relu_idx].outputs[0], source);
            mark_skip_indices(&mut skip, &identity_idxs);
            mark_skip_indices(&mut skip, &[relu_idx]);
            continue;
        }

        // --- Gemm + Relu fusion ---
        if kind == NodeKind::Gemm
            && let Some((relu_idx, identity_idxs)) =
                find_relu_after_identity_chain(nodes, node_kinds, i + 1, &node.outputs[0])
        {
            execute_node_with_layout_kind(node, &mut env, kind)?;
            if let Some(tensor) = env.get_mut(&node.outputs[0]) {
                relu_inplace(tensor);
            }
            let source = &node.outputs[0];
            for &id_idx in &identity_idxs {
                env.alias(&nodes[id_idx].outputs[0], source);
            }
            env.alias(&nodes[relu_idx].outputs[0], source);
            mark_skip_indices(&mut skip, &identity_idxs);
            mark_skip_indices(&mut skip, &[relu_idx]);
            continue;
        }

        // --- Add + Relu fusion (with in-place Add when possible) ---
        if kind == NodeKind::Add
            && node.inputs.len() == 2
            && let Some((relu_idx, identity_idxs)) =
                find_relu_after_identity_chain(nodes, node_kinds, i + 1, &node.outputs[0])
        {
            let a_nhwc = env.is_nhwc(&node.inputs[0]);
            let b_nhwc = env.is_nhwc(&node.inputs[1]);
            let same_shape_nhwc = a_nhwc == b_nhwc
                && match (env.get(&node.inputs[0]), env.get(&node.inputs[1])) {
                    (Some(a), Some(b)) => a.shape() == b.shape(),
                    _ => false,
                };
            let mut did_inplace = false;
            if same_shape_nhwc {
                let a_uses = tensor_use_count(&env, use_counts_by_id, use_counts, &node.inputs[0]);
                let b_uses = tensor_use_count(&env, use_counts_by_id, use_counts, &node.inputs[1]);
                if a_uses <= 1 || b_uses <= 1 {
                    let (consume_idx, other_idx) = if a_uses <= 1 { (0, 1) } else { (1, 0) };
                    if let Some(mut target) = env.remove(&node.inputs[consume_idx]) {
                        if let Some(other) = env.get(&node.inputs[other_idx]) {
                            yscv_kernels::add_relu_inplace(&mut target, other);
                            env.insert(node.outputs[0].clone(), target);
                            if a_nhwc {
                                env.mark_nhwc(&node.outputs[0]);
                            }
                            did_inplace = true;
                        } else {
                            env.insert(node.inputs[consume_idx].clone(), target);
                        }
                    }
                }
            }
            if !did_inplace {
                execute_node_with_layout_kind(node, &mut env, kind)?;
                if let Some(tensor) = env.get_mut(&node.outputs[0]) {
                    relu_inplace(tensor);
                }
            }
            let source = &node.outputs[0];
            for &id_idx in &identity_idxs {
                env.alias(&nodes[id_idx].outputs[0], source);
            }
            env.alias(&nodes[relu_idx].outputs[0], source);
            mark_skip_indices(&mut skip, &identity_idxs);
            mark_skip_indices(&mut skip, &[relu_idx]);
            continue;
        }

        // --- MatMul + Add fusion (Gemm-like) ---
        if kind == NodeKind::MatMul
            && let Some(next) = nodes.get(i + 1)
            && node_kind(node_kinds, nodes, i + 1) == NodeKind::Add
            && next.inputs.len() == 2
            && (next.inputs[0] == node.outputs[0] || next.inputs[1] == node.outputs[0])
        {
            execute_node_with_layout_kind(node, &mut env, kind)?;
            execute_node_with_layout_kind(next, &mut env, node_kind(node_kinds, nodes, i + 1))?;
            if i + 1 < skip.len() {
                skip[i + 1] = true;
            }
            continue;
        }

        // --- Sigmoid + Mul → SiLU fusion ---
        // SiLU(x) = x * sigmoid(x).  Pattern: Sigmoid(x)->y, Mul(x,y)->z
        // Single-pass SIMD kernel avoids separate sigmoid allocation + Mul dispatch.
        if kind == NodeKind::Sigmoid && node.inputs.len() == 1 {
            let sig_in = &node.inputs[0];
            let sig_out = &node.outputs[0];
            // Look ahead up to 3 positions for a matching Mul (SiLU pattern).
            let mut found_silu = false;
            for look in 1..=3 {
                if let Some(next) = nodes.get(i + look)
                    && node_kind(node_kinds, nodes, i + look) == NodeKind::Mul
                    && next.inputs.len() == 2
                {
                    let is_silu = (next.inputs[0] == *sig_out && next.inputs[1] == *sig_in)
                        || (next.inputs[1] == *sig_out && next.inputs[0] == *sig_in);
                    if is_silu {
                        let is_nhwc = env.is_nhwc(sig_in);
                        let mul_out = &next.outputs[0];
                        // sig_in is used by Sigmoid + Mul = 2 fused consumers.
                        // Only remove if no other node needs it.
                        let sig_in_uses =
                            tensor_use_count(&env, use_counts_by_id, use_counts, sig_in);
                        if sig_in_uses <= 2 {
                            if let Some(mut tensor) = env.remove(sig_in) {
                                yscv_kernels::silu_inplace(&mut tensor);
                                env.insert(mul_out.clone(), tensor);
                            }
                        } else if let Some(x_tensor) = env.get(sig_in) {
                            let result_tensor = yscv_kernels::silu(x_tensor);
                            env.insert(mul_out.clone(), result_tensor);
                        }
                        if is_nhwc {
                            env.mark_nhwc(mul_out);
                        }
                        // Execute any intermediate nodes, then mark them done
                        // so the main loop doesn't re-execute them.
                        for mid in 1..look {
                            if i + mid < skip.len() && !skip[i + mid] {
                                execute_node_with_layout_kind(
                                    &nodes[i + mid],
                                    &mut env,
                                    node_kind(node_kinds, nodes, i + mid),
                                )?;
                                skip[i + mid] = true;
                            }
                        }
                        if i + look < skip.len() {
                            skip[i + look] = true;
                        }
                        found_silu = true;
                        break;
                    }
                }
            }
            if found_silu {
                continue;
            }
        }

        // --- In-place Add: reuse buffer when one input is last-use ---
        if kind == NodeKind::Add && node.inputs.len() == 2 {
            let a_nhwc = env.is_nhwc(&node.inputs[0]);
            let b_nhwc = env.is_nhwc(&node.inputs[1]);
            if a_nhwc == b_nhwc {
                let same_shape = match (env.get(&node.inputs[0]), env.get(&node.inputs[1])) {
                    (Some(a), Some(b)) => a.shape() == b.shape(),
                    _ => false,
                };
                if same_shape {
                    let a_uses =
                        tensor_use_count(&env, use_counts_by_id, use_counts, &node.inputs[0]);
                    let b_uses =
                        tensor_use_count(&env, use_counts_by_id, use_counts, &node.inputs[1]);
                    if a_uses <= 1 || b_uses <= 1 {
                        let (consume_idx, other_idx) = if a_uses <= 1 { (0, 1) } else { (1, 0) };
                        if let Some(mut target) = env.remove(&node.inputs[consume_idx]) {
                            if let Some(other) = env.get(&node.inputs[other_idx]) {
                                yscv_kernels::add_inplace(&mut target, other);
                                env.insert(node.outputs[0].clone(), target);
                                if a_nhwc {
                                    env.mark_nhwc(&node.outputs[0]);
                                }
                                continue;
                            }
                            env.insert(node.inputs[consume_idx].clone(), target);
                        }
                    }
                }
            }
        }

        // Zero-copy Reshape: avoid data clone when the data input has only
        // one consumer (this Reshape node).
        if kind == NodeKind::Reshape {
            // NHWC-passthrough fast path. When the input is NHWC
            // physical rank-4 `[N,H,W,C]` and the model reshapes to a rank-3
            // `[N, C, H*W]` (merge-spatial), the data layout is already in
            // `[N, H*W, C]` order — which is exactly what a downstream
            // `Transpose(perm=[0,2,1])+MatMul` (FusedTransposeMatMul) wants
            // as the post-transpose A operand. Skip the `ensure_nchw` permute
            // and propagate the NHWC tag.
            // Kill switch: `YSCV_RESHAPE_NHWC_PASSTHROUGH_OFF=1`.
            if !reshape_nhwc_passthrough_disabled()
                && try_reshape_nhwc_passthrough(node, &mut env, use_counts)?
            {
                continue;
            }
            for name in &node.inputs {
                if !name.is_empty() {
                    ensure_nchw(&mut env, name)?;
                }
            }
            exec_reshape_zerocopy(node, &mut env, use_counts)?;
            continue;
        }

        // Fast path for Conv: use pre-computed params to skip attr HashMap lookups
        if matches!(
            kind,
            NodeKind::Conv | NodeKind::ConvRelu | NodeKind::ConvSilu
        ) {
            let cp = model
                .runtime_index
                .conv_params
                .get(i)
                .and_then(|o| o.as_ref());
            let activation = match kind {
                NodeKind::ConvRelu => yscv_kernels::Activation::Relu,
                NodeKind::ConvSilu => yscv_kernels::Activation::Silu,
                _ => yscv_kernels::Activation::None,
            };

            // --- DW+PW fusion: detect depthwise Conv followed by pointwise 1x1 ---
            if cp.is_some_and(|p| p.is_depthwise) {
                // Look ahead for pointwise 1x1 Conv consuming our output exclusively
                let dw_out = &node.outputs[0];
                let dw_uses = use_counts.get(dw_out).copied().unwrap_or(0);
                if dw_uses == 1
                    && let Some(next_idx) = (i + 1..nodes.len()).find(|&j| !skip[j])
                {
                    let next_cp = model
                        .runtime_index
                        .conv_params
                        .get(next_idx)
                        .and_then(|o| o.as_ref());
                    let next = &nodes[next_idx];
                    let next_kind = node_kind(node_kinds, nodes, next_idx);
                    if next_cp.is_some_and(|p| p.is_pointwise && !p.has_padding)
                        && next.inputs.first().map(|s| s.as_str()) == Some(dw_out.as_str())
                        && matches!(
                            next_kind,
                            NodeKind::Conv | NodeKind::ConvRelu | NodeKind::ConvSilu
                        )
                    {
                        let pw_activation = match next_kind {
                            NodeKind::ConvRelu => yscv_kernels::Activation::Relu,
                            NodeKind::ConvSilu => yscv_kernels::Activation::Silu,
                            _ => yscv_kernels::Activation::None,
                        };
                        let dw_input_ids_slice: &[Option<usize>] = model
                            .runtime_index
                            .node_input_ids
                            .get(i)
                            .map(|v| v.as_slice())
                            .unwrap_or(&[]);
                        exec_fused_dw_pw(
                            node,
                            next,
                            &mut env,
                            activation,
                            pw_activation,
                            cp,
                            next_cp,
                            dw_input_ids_slice,
                            &mut remaining_uses,
                            &output_id_mask,
                            // Legacy single-pass dispatch path (used by
                            // calibration/quantize flows). M3 enclave
                            // lookahead is wired only in the plan-based
                            // `execute_plan_branch`; here we always
                            // convert back to NHWC at the boundary.
                            false,
                        )?;
                        // Decrement PW inputs so DW output also gets
                        // freed. Mirrors the post-action cleanup below,
                        // but tailored to the fused pair since the
                        // outer `continue` below would otherwise skip
                        // it.
                        let pw_pre_ids = model
                            .runtime_index
                            .node_input_ids
                            .get(next_idx)
                            .map(|v| v.as_slice())
                            .unwrap_or(&[]);
                        for (inp_idx, inp) in next.inputs.iter().enumerate() {
                            if inp.is_empty() {
                                continue;
                            }
                            let id = pw_pre_ids
                                .get(inp_idx)
                                .and_then(|opt| *opt)
                                .or_else(|| env.resolve_id(inp));
                            if let Some(id) = id
                                && id < remaining_uses.len()
                            {
                                remaining_uses[id] = remaining_uses[id].saturating_sub(1);
                                if remaining_uses[id] == 0 && !output_id_mask[id] {
                                    env.remove_by_id(id);
                                }
                            }
                        }
                        skip[next_idx] = true;
                        continue;
                    }
                }
            }

            let prepacked = prepacked_for_conv_node(model, i);
            exec_conv_with_params(node, &mut env, activation, cp, prepacked)?;
            env.mark_nhwc(&node.outputs[0]);
        } else {
            execute_node_with_layout_kind(node, &mut env, kind)?;
        }

        // --- Early deallocation: free tensors whose last consumer was this node ---
        let input_ids = &model.runtime_index.node_input_ids;
        let pre_ids = if i < input_ids.len() {
            &input_ids[i]
        } else {
            &[][..]
        };
        for (inp_idx, inp) in node.inputs.iter().enumerate() {
            if inp.is_empty() {
                continue;
            }
            // Use pre-resolved ID (O(1)) when available, fallback to HashMap.
            let id = pre_ids
                .get(inp_idx)
                .and_then(|opt| *opt)
                .or_else(|| env.resolve_id(inp));
            if let Some(id) = id
                && id < remaining_uses.len()
            {
                remaining_uses[id] = remaining_uses[id].saturating_sub(1);
                if remaining_uses[id] == 0 && !output_id_mask[id] {
                    env.remove_by_id(id);
                }
            }
        }
    }

    // Optional per-op trace for debugging inference divergence.
    if std::env::var("CPU_TRACE").is_ok() {
        for node in nodes {
            for out_name in &node.outputs {
                if let Some(t) = env.get(out_name) {
                    let d = t.data();
                    if d.is_empty() {
                        continue;
                    }
                    let max = d.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let min = d.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                    let mean = d.iter().sum::<f32>() / d.len() as f32;
                    let nhwc = if env.is_nhwc(out_name) { " [NHWC]" } else { "" };
                    eprintln!(
                        "[{:>20}] {:60} shape={:?} min={:>10.4} max={:>10.4} mean={:>10.4}{}",
                        node.op_type,
                        out_name,
                        t.shape(),
                        min,
                        max,
                        mean,
                        nhwc,
                    );
                }
            }
        }
    }

    // Ensure all outputs are in NCHW (ONNX standard layout)
    for name in &model.outputs {
        env.materialize_quant_i8_raw(name)?;
        ensure_nchw(&mut env, name)?;
    }

    let mut result = HashMap::with_capacity(model.outputs.len());
    for name in &model.outputs {
        if let Some(t) = env.remove(name) {
            result.insert(name.clone(), t);
        } else if let Some(t) = env.get(name) {
            result.insert(name.clone(), t.clone());
        } else {
            eprintln!("warning: ONNX output '{}' not found in environment", name);
        }
    }
    Ok(result)
}
