//! Branch executor: runs one (possibly tower-parallel) slice of the
//! optimized node plan, threading the TensorEnv through fused dispatch.

use super::*;

/// Run a subset of the JIT execution plan, filtered by a predicate on the
/// node's branch assignment. Shared body for both the single-branch path and
/// the tower-parallel wrapper.
/// The entry point the plan resolved for this Conv node, if it resolved one.
#[inline]
fn planned_conv_kernel(
    model: &OnnxModel,
    node_idx: usize,
) -> Option<crate::runner::conv_kernel::ConvKernel> {
    model
        .runtime_index
        .conv_kernels
        .get(node_idx)
        .copied()
        .flatten()
}

pub(crate) fn execute_plan_branch(
    model: &OnnxModel,
    env: &mut TensorEnv<'_, '_>,
    remaining_uses: &mut [usize],
    output_id_mask: &[bool],
    reshape_shapes: Option<&FxHashMap<usize, Vec<usize>>>,
    mut accept: impl FnMut(usize) -> bool,
    conv_ns: &mut u64,
    other_ns: &mut u64,
    conv_count: &mut u32,
    other_count: &mut u32,
    do_profile: bool,
) -> Result<(), OnnxError> {
    use crate::plan::NodeAction;

    let nodes = &model.nodes;
    let plan = &model.runtime_index.execution_plan;
    let runner_profile_enabled = runner_profile_active();
    let timing_enabled = do_profile || runner_profile_enabled;

    for (action_idx, action) in plan.iter().enumerate() {
        // Lookup the representative node index to check branch filter.
        let rep_idx = match action {
            NodeAction::Skip => continue,
            NodeAction::Conv { node_idx, .. } | NodeAction::Generic { node_idx, .. } => *node_idx,
            NodeAction::FusedDwPw { dw_idx, .. } => *dw_idx,
            NodeAction::FusedPwDw { pw_idx, .. } => *pw_idx,
            NodeAction::FusedPwDwPwReduce { pw_expand_idx, .. } => *pw_expand_idx,
            NodeAction::FusedTransposeMatMul { matmul_idx, .. } => *matmul_idx,
            NodeAction::QuantizedQdq { dequant_idx, .. } => *dequant_idx,
            NodeAction::FusedHardSwishQuant { hs_idx, .. } => *hs_idx,
            NodeAction::FusedSeMulQuant { mul_idx, .. } => *mul_idx,
            NodeAction::QuantizedPwDw { pw_idx, .. } => *pw_idx,
            NodeAction::QuantizedDwPw { dw_idx, .. } => *dw_idx,
            NodeAction::QuantizedForkPair { first_idx, .. } => *first_idx,
            NodeAction::QuantizedResidualChain { qconv_idx, .. } => *qconv_idx,
            NodeAction::QuantizedConvDq { qconv_idx, .. } => *qconv_idx,
            NodeAction::ConvAdd { conv_idx, .. } => *conv_idx,
        };
        if !accept(rep_idx) {
            continue;
        }

        // Whether this action leaves its output in blocked NCHWc16 for the next
        // one to read directly. Resolved at plan time — the walk that used to
        // answer it here read nothing that changes between inferences.
        let nchwc_handoff = model
            .runtime_index
            .nchwc_handoff
            .get(action_idx)
            .copied()
            .unwrap_or(false);

        let t0 = timing_enabled.then(std::time::Instant::now);

        match action {
            NodeAction::Skip => continue,

            NodeAction::Conv {
                node_idx,
                activation,
            } => {
                let node = &nodes[*node_idx];
                let act = match activation {
                    1 => yscv_kernels::Activation::Relu,
                    2 => yscv_kernels::Activation::Silu,
                    _ => yscv_kernels::Activation::None,
                };
                let cp = model
                    .runtime_index
                    .conv_params
                    .get(*node_idx)
                    .and_then(|o| o.as_ref());
                let prepacked = prepacked_for_conv_node(model, *node_idx);
                // Plain Conv stays NHWC. The only NCHWc-eligible backbone Conv
                // (`xif1_0/dw`, c=16) would force the downstream `xif1_0/pwl`
                // Conv_Add into NCHWc, where `conv2d_nchwc_pointwise` at a
                // single c-block (c=16) loses to the NHWC fast path. Higher-c
                // backbone convs already go through FusedPwDwPwReduce, which
                // IS profitable in NCHWc.
                let planned = planned_conv_kernel(model, *node_idx);
                exec_conv_with_params(node, env, act, cp, prepacked, planned)?;
                // activation == 3: fused Conv + HardSwish. The conv ran with no
                // epilogue activation; apply HardSwish in place on the still-warm
                // output (no separate node, no second buffer).
                if *activation == 3
                    && let Some(mut t) = env.remove(&node.outputs[0])
                {
                    yscv_kernels::hardswish_slice_inplace(t.data_mut());
                    env.insert(node.outputs[0].clone(), t);
                }
                if let Some(oid) = model
                    .runtime_index
                    .node_output_ids
                    .get(*node_idx)
                    .and_then(|v| v.first())
                    .and_then(|o| *o)
                {
                    env.mark_nhwc_by_id(oid);
                } else {
                    env.mark_nhwc(&node.outputs[0]);
                }
            }

            NodeAction::FusedDwPw {
                dw_idx,
                pw_idx,
                dw_activation,
                pw_activation,
            } => {
                let dw_node = &nodes[*dw_idx];
                let pw_node = &nodes[*pw_idx];
                let dw_act = match dw_activation {
                    1 => yscv_kernels::Activation::Relu,
                    2 => yscv_kernels::Activation::Silu,
                    _ => yscv_kernels::Activation::None,
                };
                let pw_act = match pw_activation {
                    1 => yscv_kernels::Activation::Relu,
                    2 => yscv_kernels::Activation::Silu,
                    _ => yscv_kernels::Activation::None,
                };
                let dw_cp = model
                    .runtime_index
                    .conv_params
                    .get(*dw_idx)
                    .and_then(|o| o.as_ref());
                let pw_cp = model
                    .runtime_index
                    .conv_params
                    .get(*pw_idx)
                    .and_then(|o| o.as_ref());
                let dw_input_ids_slice: &[Option<usize>] = model
                    .runtime_index
                    .node_input_ids
                    .get(*dw_idx)
                    .map(|v| v.as_slice())
                    .unwrap_or(&[]);
                exec_fused_dw_pw(
                    dw_node,
                    pw_node,
                    env,
                    dw_act,
                    pw_act,
                    dw_cp,
                    pw_cp,
                    dw_input_ids_slice,
                    remaining_uses,
                    output_id_mask,
                    nchwc_handoff,
                )?;
            }

            NodeAction::FusedPwDw {
                pw_idx,
                dw_idx,
                pw_activation,
                dw_activation,
            } => {
                let pw_node = &nodes[*pw_idx];
                let dw_node = &nodes[*dw_idx];
                let pw_act = match pw_activation {
                    1 => yscv_kernels::Activation::Relu,
                    2 => yscv_kernels::Activation::Silu,
                    _ => yscv_kernels::Activation::None,
                };
                let dw_act = match dw_activation {
                    1 => yscv_kernels::Activation::Relu,
                    2 => yscv_kernels::Activation::Silu,
                    _ => yscv_kernels::Activation::None,
                };
                let pw_cp = model
                    .runtime_index
                    .conv_params
                    .get(*pw_idx)
                    .and_then(|o| o.as_ref());
                let dw_cp = model
                    .runtime_index
                    .conv_params
                    .get(*dw_idx)
                    .and_then(|o| o.as_ref());
                let pw_input_ids_slice: &[Option<usize>] = model
                    .runtime_index
                    .node_input_ids
                    .get(*pw_idx)
                    .map(|v| v.as_slice())
                    .unwrap_or(&[]);
                exec_fused_pw_dw(
                    pw_node,
                    dw_node,
                    env,
                    pw_act,
                    dw_act,
                    pw_cp,
                    dw_cp,
                    pw_input_ids_slice,
                    remaining_uses,
                    output_id_mask,
                    nchwc_handoff,
                )?;
            }

            NodeAction::FusedPwDwPwReduce {
                pw_expand_idx,
                dw_idx,
                pw_reduce_idx,
                pw_expand_activation,
                dw_activation,
                pw_reduce_activation,
                dw_kernel_size,
                residual,
            } => {
                let pw_expand_node = &nodes[*pw_expand_idx];
                let dw_node = &nodes[*dw_idx];
                let pw_reduce_node = &nodes[*pw_reduce_idx];
                let pw_expand_act = match pw_expand_activation {
                    1 => yscv_kernels::Activation::Relu,
                    _ => yscv_kernels::Activation::None,
                };
                let dw_act = match dw_activation {
                    1 => yscv_kernels::Activation::Relu,
                    _ => yscv_kernels::Activation::None,
                };
                let pw_reduce_act = match pw_reduce_activation {
                    1 => yscv_kernels::Activation::Relu,
                    _ => yscv_kernels::Activation::None,
                };
                let pw_expand_cp = model
                    .runtime_index
                    .conv_params
                    .get(*pw_expand_idx)
                    .and_then(|o| o.as_ref());
                let dw_cp = model
                    .runtime_index
                    .conv_params
                    .get(*dw_idx)
                    .and_then(|o| o.as_ref());
                let pw_expand_input_ids: &[Option<usize>] = model
                    .runtime_index
                    .node_input_ids
                    .get(*pw_expand_idx)
                    .map(|v| v.as_slice())
                    .unwrap_or(&[]);
                let prepacked = model
                    .runtime_index
                    .prepacked_fused_pw_dw_pw_reduce
                    .get(pw_reduce_idx)
                    .ok_or_else(|| OnnxError::DecodeFailed {
                        message: format!(
                            "FusedPwDwPwReduce: missing prepacked weight for pw_reduce_idx={}",
                            pw_reduce_idx
                        ),
                    })?;
                let residual_meta_tuple: Option<(&OnnxNode, u8, u8, Option<&OnnxNode>)> =
                    residual.as_ref().map(|r| {
                        let add_node = &nodes[r.add_idx];
                        let relu_node = if r.post_activation == 1 {
                            Some(&nodes[r.relu_idx as usize])
                        } else {
                            None
                        };
                        (
                            add_node,
                            r.residual_skip_input,
                            r.post_activation,
                            relu_node,
                        )
                    });
                exec_fused_pw_dw_pw_reduce(
                    pw_expand_node,
                    dw_node,
                    pw_reduce_node,
                    env,
                    pw_expand_act,
                    dw_act,
                    pw_reduce_act,
                    pw_expand_cp,
                    dw_cp,
                    pw_expand_input_ids,
                    prepacked.as_ref(),
                    residual_meta_tuple,
                    *dw_kernel_size,
                    remaining_uses,
                    output_id_mask,
                    nchwc_handoff,
                )?;
            }

            NodeAction::ConvAdd {
                conv_idx,
                add_idx,
                skip_input_idx,
                post_activation,
                relu_idx,
            } => {
                let conv_node = &nodes[*conv_idx];
                let add_node = &nodes[*add_idx];
                let cp = model
                    .runtime_index
                    .conv_params
                    .get(*conv_idx)
                    .and_then(|o| o.as_ref());
                let conv_out = &conv_node.outputs[0];
                let skip_name = &add_node.inputs[*skip_input_idx as usize];

                // Fast path — pointwise Conv + residual Add + optional Relu
                // fused in one GEMM pass. Writes `out = conv_acc + bias +
                // residual + activation` inline, avoiding the 2-pass
                // `add_relu_inplace` which doubles output-side memory
                // traffic (tracker Conv_Add is ~1.2ms @ 6T = 18% of total,
                // mostly on high-k shapes). Step S.2: removed the former
                // `k_small` gate — Phase 1.2 added residual support to
                // blocked GEMM 4×24/4×16 microkernels, so ALL pointwise
                // Conv+Add now fuses (matmul dispatches row_gemm for k<32,
                // blocked for k≥32, both residual-aware).
                //
                // `blocked_residual_has_unsupported_tail(n)` at matmul
                // dispatch auto-routes to row_gemm for shapes whose jr
                // tail would hit 4×8 / scalar (no residual there yet).
                let fused_pointwise = cp
                    .map(|p| {
                        p.is_pointwise
                            && !p.has_padding
                            && p.stride_h == 1
                            && p.stride_w == 1
                            && p.group == 1
                    })
                    .unwrap_or(false);
                let activation_for_fused = if *post_activation == 1 {
                    yscv_kernels::Activation::Relu
                } else {
                    yscv_kernels::Activation::None
                };
                // ── Whole-graph-NCHWc fast path (YSCV_BACKBONE_NCHWC=1) ──
                // Run Conv1×1 + residual Add (+optional Relu) as a single
                // NCHWc pointwise-with-residual kernel. When the input and
                // residual were left NCHWc by upstream chained ops, this is
                // a zero-conversion link in the xif4 ladder.
                let backbone_nchwc = {
                    use std::sync::OnceLock;
                    static C: OnceLock<bool> = OnceLock::new();
                    *C.get_or_init(|| std::env::var_os("YSCV_BACKBONE_NCHWC").is_some())
                };
                // ConvAdd chain lookahead: leave NCHWc when next op consumes
                // this Conv_Add's output (after optional Relu absorption).
                let convadd_output_name: &str = if *post_activation == 1 {
                    &nodes[*relu_idx as usize].outputs[0]
                } else {
                    &add_node.outputs[0]
                };
                let convadd_leave_nchwc = if backbone_nchwc {
                    let mut found = false;
                    let mut probe = action_idx + 1;
                    while let Some(next) = plan.get(probe) {
                        match next {
                            NodeAction::Skip => {
                                probe += 1;
                                continue;
                            }
                            NodeAction::FusedPwDwPwReduce {
                                pw_expand_idx: next_idx,
                                ..
                            }
                            | NodeAction::FusedPwDw {
                                pw_idx: next_idx, ..
                            }
                            | NodeAction::ConvAdd {
                                conv_idx: next_idx, ..
                            } => {
                                let next_node = &nodes[*next_idx];
                                if !next_node.inputs.is_empty()
                                    && next_node.inputs[0] == *convadd_output_name
                                {
                                    found = true;
                                }
                                break;
                            }
                            _ => break,
                        }
                    }
                    found
                } else {
                    false
                };
                let (fused_result, fused_is_nchwc): (Option<Tensor>, bool) = if backbone_nchwc
                    && fused_pointwise
                {
                    // Only fire NCHWc Conv_Add when the upstream already left
                    // NCHWc — otherwise the existing fast NHWC path wins (we
                    // would have to convert input + residual + output, which
                    // is pure overhead). The skip can be converted cheaply (or
                    // chains too).
                    let in_is_nchwc = env.nchwc_block(&conv_node.inputs[0]) == Some(16);
                    let skip_is_nchwc = env.nchwc_block(skip_name) == Some(16);
                    let skip_is_nhwc = env.is_nhwc(skip_name);
                    let layouts_ok = in_is_nchwc && (skip_is_nchwc || skip_is_nhwc);
                    let w_shape_ok = env
                        .get(&conv_node.inputs[1])
                        .map(|w| {
                            w.rank() == 4
                                && w.shape()[0] == 1
                                && w.shape()[1] == 1
                                && w.shape()[3].is_multiple_of(16)
                        })
                        .unwrap_or(false);
                    if layouts_ok && w_shape_ok {
                        let cfg = yscv_kernels::ParallelElementwiseConfig::default();
                        let w_tensor = env.get(&conv_node.inputs[1]).cloned();
                        let bias_tensor = conv_node.inputs.get(2).and_then(|n| {
                            if n.is_empty() {
                                None
                            } else {
                                env.get(n).cloned()
                            }
                        });
                        let input_src = env.get(&conv_node.inputs[0]).cloned();
                        let skip_src = env.get(skip_name).cloned();
                        let try_nchwc = |i_src: &Tensor,
                                         s_src: &Tensor,
                                         w: &Tensor|
                         -> Option<Tensor> {
                            let in_owned;
                            let in_t: &Tensor = if in_is_nchwc {
                                i_src
                            } else {
                                in_owned = yscv_kernels::nhwc_to_nchwc(i_src, 16).ok()?;
                                &in_owned
                            };
                            let s_owned;
                            let s_t: &Tensor = if skip_is_nchwc {
                                s_src
                            } else {
                                s_owned = yscv_kernels::nhwc_to_nchwc(s_src, 16).ok()?;
                                &s_owned
                            };
                            let c_in_actual = w.shape()[2];
                            yscv_kernels::conv2d_nchwc_pointwise_with_residual_activation_prepacked(
                                in_t,
                                w,
                                bias_tensor.as_ref(),
                                s_t,
                                c_in_actual,
                                activation_for_fused,
                                cfg,
                                None,
                                None,
                            )
                            .ok()
                        };
                        let result = match (input_src, skip_src, w_tensor) {
                            (Some(i), Some(s), Some(w)) => try_nchwc(&i, &s, &w),
                            _ => None,
                        };
                        if let Some(r) = result {
                            if convadd_leave_nchwc {
                                (Some(r), true)
                            } else {
                                let c_out = r.shape().get(1).copied().unwrap_or(0)
                                    * r.shape().get(4).copied().unwrap_or(0);
                                // c_out may include padding; trim via the kernel's actual_in_channels.
                                let actual_c = env
                                    .get(&conv_node.inputs[1])
                                    .map(|w| w.shape()[3])
                                    .unwrap_or(c_out);
                                let out_nhwc = yscv_kernels::nchwc_to_nhwc(&r, actual_c).ok();
                                (out_nhwc, false)
                            }
                        } else {
                            (None, false)
                        }
                    } else {
                        (None, false)
                    }
                } else {
                    (None, false)
                };
                let fused_result: Option<Tensor> = if fused_result.is_some() {
                    fused_result
                } else if fused_pointwise {
                    // Scoped block so all `env.get` immutable borrows drop
                    // before the mutable `env.insert` below.
                    let input_ok = env.is_nhwc(&conv_node.inputs[0]);
                    if input_ok {
                        let input_tensor = env.get(&conv_node.inputs[0]);
                        let w_tensor = env.get(&conv_node.inputs[1]);
                        let skip_tensor = env.get(skip_name);
                        let bias_tensor = conv_node
                            .inputs
                            .get(2)
                            .and_then(|n| if n.is_empty() { None } else { env.get(n) });
                        match (input_tensor, w_tensor, skip_tensor) {
                            (Some(i), Some(w), Some(s))
                                if i.rank() == 4
                                    && w.rank() == 4
                                    && w.shape()[0] == 1
                                    && w.shape()[1] == 1 =>
                            {
                                let prepacked = prepacked_for_conv_node(model, *conv_idx);
                                yscv_kernels::conv2d_nhwc_pointwise_with_residual_relu(
                                    i,
                                    w,
                                    bias_tensor,
                                    s,
                                    activation_for_fused,
                                    None,
                                    prepacked,
                                )
                                .ok()
                            }
                            _ => None,
                        }
                    } else {
                        None
                    }
                } else {
                    None
                };

                if let Some(fused_out) = fused_result {
                    let add_out = &add_node.outputs[0];
                    // cached add_out slot ID avoids a FxHashMap
                    // lookup inside insert + mark_nhwc. On tracker this fires
                    // 24× per inference (one per residual block).
                    let add_out_id = model
                        .runtime_index
                        .node_output_ids
                        .get(*add_idx)
                        .and_then(|v| v.first())
                        .and_then(|o| *o);
                    if let Some(oid) = add_out_id {
                        env.insert_by_id(oid, fused_out);
                        if fused_is_nchwc {
                            env.mark_nchwc(add_out, 16);
                        } else {
                            env.mark_nhwc_by_id(oid);
                        }
                    } else {
                        env.insert(add_out.clone(), fused_out);
                        if fused_is_nchwc {
                            env.mark_nchwc(add_out, 16);
                        } else {
                            env.mark_nhwc(add_out);
                        }
                    }
                    if *post_activation == 1 {
                        let relu_out = &nodes[*relu_idx as usize].outputs[0];
                        env.alias(relu_out, add_out);
                    }
                    if do_profile {
                        let elapsed = t0
                            .as_ref()
                            .map(|start| start.elapsed().as_nanos() as u64)
                            .unwrap_or(0);
                        *conv_ns += elapsed;
                        *conv_count += 1;
                    }
                    if runner_profile_enabled {
                        let elapsed = t0
                            .as_ref()
                            .map(|start| start.elapsed().as_nanos() as u64)
                            .unwrap_or(0);
                        let in_sh = env
                            .get(&conv_node.inputs[0])
                            .map(|t| t.shape().to_vec())
                            .unwrap_or_default();
                        let out_sh = env
                            .get(&add_node.outputs[0])
                            .map(|t| t.shape().to_vec())
                            .unwrap_or_default();
                        let op_label = if *post_activation == 1 {
                            "Conv_Add_Relu_fused"
                        } else {
                            "Conv_Add_fused"
                        };
                        runner_profile_record(&conv_node.name, op_label, elapsed, in_sh, out_sh);
                    }
                    // Early-dealloc input refs the same way the generic
                    // branch does at function scope.
                    let covered = &[*conv_idx, *add_idx][..];
                    let input_ids = &model.runtime_index.node_input_ids;
                    for &nidx in covered {
                        let n = &nodes[nidx];
                        let pre_ids = if nidx < input_ids.len() {
                            &input_ids[nidx]
                        } else {
                            &[][..]
                        };
                        for (inp_idx, inp) in n.inputs.iter().enumerate() {
                            if inp.is_empty() {
                                continue;
                            }
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
                    continue;
                }

                let prepacked = prepacked_for_conv_node(model, *conv_idx);
                exec_conv_with_params(
                    conv_node,
                    env,
                    yscv_kernels::Activation::None,
                    cp,
                    prepacked,
                    planned_conv_kernel(model, *conv_idx),
                )?;
                // cached slot IDs for the fallback path too.
                let conv_out_id = model
                    .runtime_index
                    .node_output_ids
                    .get(*conv_idx)
                    .and_then(|v| v.first())
                    .and_then(|o| *o);
                let add_out_id = model
                    .runtime_index
                    .node_output_ids
                    .get(*add_idx)
                    .and_then(|v| v.first())
                    .and_then(|o| *o);
                if let Some(oid) = conv_out_id {
                    env.mark_nhwc_by_id(oid);
                } else {
                    env.mark_nhwc(conv_out);
                }
                if let Some(mut conv_tensor) = env.remove(conv_out) {
                    if let Some(skip_tensor) = env.get(skip_name) {
                        if *post_activation == 1 {
                            yscv_kernels::add_relu_inplace(&mut conv_tensor, skip_tensor);
                        } else {
                            yscv_kernels::add_inplace(&mut conv_tensor, skip_tensor);
                        }
                        let add_out = &add_node.outputs[0];
                        if let Some(oid) = add_out_id {
                            env.insert_by_id(oid, conv_tensor);
                            env.mark_nhwc_by_id(oid);
                        } else {
                            env.insert(add_out.clone(), conv_tensor);
                            env.mark_nhwc(add_out);
                        }
                        if *post_activation == 1 {
                            let relu_out = &nodes[*relu_idx as usize].outputs[0];
                            env.alias(relu_out, add_out);
                        }
                    } else {
                        env.insert(conv_out.clone(), conv_tensor);
                        execute_node_with_layout_kind(
                            add_node,
                            env,
                            node_kind(&model.runtime_index.node_kinds, nodes, *add_idx),
                        )?;
                    }
                }
            }

            NodeAction::FusedTransposeMatMul {
                transpose_idx,
                matmul_idx,
                ..
            } => {
                // Transpose node is elided — read the pre-transpose
                // source (its input[0]) and feed it to the MatMul via
                // `matmul_2d_slices_trans_a` (BLAS `CblasTrans` under
                // the hood, else scratch-buffer fallback in the kernel).
                let transpose_node = &nodes[*transpose_idx];
                let matmul_node = &nodes[*matmul_idx];
                exec_fused_transpose_matmul(transpose_node, matmul_node, env)?;
            }

            NodeAction::QuantizedQdq {
                dequant_idx,
                relu_idx,
                hardswish,
                quant_idx,
            } => {
                let dequant_node = &nodes[*dequant_idx];
                let quant_node = &nodes[*quant_idx];
                if !quant_int8_fast_enabled() {
                    execute_node_with_layout_kind(dequant_node, env, NodeKind::Other)?;
                    if let Some(ri) = relu_idx {
                        execute_node_with_layout_kind(&nodes[*ri], env, NodeKind::Relu)?;
                    }
                    if let Some((hs_idx, mul_idx)) = hardswish {
                        execute_node_with_layout_kind(&nodes[*hs_idx], env, NodeKind::Other)?;
                        execute_node_with_layout_kind(&nodes[*mul_idx], env, NodeKind::Mul)?;
                    }
                    execute_node_with_layout_kind(quant_node, env, NodeKind::Other)?;
                    continue;
                }
                note_quant_qdq_boundary();
                let input_name =
                    dequant_node
                        .inputs
                        .first()
                        .ok_or_else(|| OnnxError::DecodeFailed {
                            message: format!("{}: missing quantized input", dequant_node.name),
                        })?;
                // The folded activation as real-value bounds: Relu is
                // `(0, inf)`, a Clip its own min/max, nothing `(-inf, inf)`.
                let clamp = relu_idx.map_or((f32::NEG_INFINITY, f32::INFINITY), |ri| {
                    let act = &nodes[ri];
                    if act.op_type != "Clip" {
                        return (0.0, f32::INFINITY);
                    }
                    let bound = |idx: usize, dflt: f32| {
                        act.inputs
                            .get(idx)
                            .filter(|n| !n.is_empty())
                            .and_then(|n| env.get(n))
                            .and_then(|t| t.data().first().copied())
                            .unwrap_or(dflt)
                    };
                    (bound(1, f32::NEG_INFINITY), bound(2, f32::INFINITY))
                });
                let relu = clamp == (0.0, f32::INFINITY);
                // Rescale params: DQ (input) scale/zp and Q (output) scale/zp.
                let scalar = |node: &OnnxNode, idx: usize| -> Option<f32> {
                    node.inputs
                        .get(idx)
                        .and_then(|name| env.get(name))
                        .and_then(|t| t.data().first().copied())
                };
                let dq_scale = scalar(dequant_node, 1);
                let dq_zp = scalar(dequant_node, 2).unwrap_or(0.0);
                let q_scale = scalar(quant_node, 1);
                let q_zp = scalar(quant_node, 2).unwrap_or(0.0);
                // HardSwish `(alpha, beta)` from the folded HardSigmoid node.
                let hs_params = hardswish.map(|(hs_idx, _)| {
                    let hs = &nodes[hs_idx];
                    (
                        get_attr_float(hs, Attr::Alpha).unwrap_or(0.2),
                        get_attr_float(hs, Attr::Beta).unwrap_or(0.5),
                    )
                });
                // Fast identity path (Relu/none only): same scale + symmetric
                // zero-points, so the rescale is a no-op relabel.
                // Same scale and zero-point on both sides: the rescale is a
                // relabel, and the activation becomes a clamp on the stored
                // values. That is exact, not an approximation — for an
                // unclamped value the kernel would compute `round(q*s/s) = q`,
                // and for a clamped one `round(bound/s)`, so clamping the
                // stored value at that bound gives the same answer.
                let identity = hs_params.is_none()
                    && matches!((dq_scale, q_scale), (Some(a), Some(b)) if a.to_bits() == b.to_bits())
                    && dq_zp == q_zp;
                let bounds_i8 = |scale: f32| {
                    let at = |v: f32| {
                        let q = if v.is_infinite() {
                            v
                        } else {
                            (v / scale).round_ties_even() + q_zp
                        };
                        q.clamp(-128.0, 127.0)
                    };
                    (at(clamp.0), at(clamp.1))
                };
                if let Some(mut qt) = env.take_quant_i8(input_name) {
                    if let (Some((alpha, beta)), Some(sin), Some(sout)) =
                        (hs_params, dq_scale, q_scale)
                    {
                        // Fused DQ -> HardSwish -> Q, one i8->i8 pass.
                        let mut rescaled = vec![0_i8; qt.data.len()];
                        yscv_kernels::requant_i8_dq_hardswish_q_dispatch(
                            &qt.data,
                            sin,
                            dq_zp,
                            sout,
                            q_zp,
                            alpha,
                            beta,
                            &mut rescaled,
                        );
                        qt.data = rescaled;
                        qt.scale = sout;
                        qt.zero_point = q_zp;
                    } else if identity {
                        if clamp.0.is_finite() || clamp.1.is_finite() {
                            let (lo, hi) = bounds_i8(qt.scale);
                            let (lo, hi) = (lo as i8, hi as i8);
                            for v in &mut qt.data {
                                *v = (*v).clamp(lo, hi);
                            }
                        }
                        qt.scale = q_scale.unwrap_or(qt.scale);
                        qt.zero_point = q_zp;
                    } else if let (Some(sin), Some(sout)) = (dq_scale, q_scale) {
                        // Fused DQ -> [Relu] -> Q, one i8->i8 pass, no f32 buffer.
                        let mut rescaled = vec![0_i8; qt.data.len()];
                        yscv_kernels::requant_i8_dq_relu_q_dispatch(
                            &qt.data,
                            sin,
                            dq_zp,
                            sout,
                            q_zp,
                            clamp,
                            &mut rescaled,
                        );
                        qt.data = rescaled;
                        qt.scale = sout;
                        qt.zero_point = q_zp;
                    } else if relu {
                        // Missing scalar params (shouldn't happen under the
                        // loader gate) — fall back to relu-only relabel.
                        for v in &mut qt.data {
                            *v = (*v).max(0);
                        }
                    }
                    env.insert_quant_i8(quant_node.outputs[0].clone(), qt);
                    continue;
                }
                // f32-encoded i8 activation (conv fallback output): same rescale,
                // producing an f32-encoded i8 tensor to match the per-op Q.
                let mut tensor = env
                    .remove(input_name)
                    .or_else(|| env.get(input_name).cloned())
                    .ok_or_else(|| OnnxError::MissingInput {
                        node: dequant_node.name.clone(),
                        input: input_name.clone(),
                    })?;
                if let (Some((alpha, beta)), Some(sin), Some(sout)) = (hs_params, dq_scale, q_scale)
                {
                    let out: Vec<f32> = tensor
                        .data()
                        .iter()
                        .map(|&v| {
                            let f = (v - dq_zp) * sin;
                            let hs = (alpha * f + beta).clamp(0.0, 1.0);
                            ((f * hs) / sout + q_zp)
                                .round_ties_even()
                                .clamp(-128.0, 127.0)
                        })
                        .collect();
                    tensor = Tensor::from_vec(tensor.shape().to_vec(), out).map_err(|e| {
                        OnnxError::DecodeFailed {
                            message: e.to_string(),
                        }
                    })?;
                } else if identity {
                    if clamp.0.is_finite() || clamp.1.is_finite() {
                        let (lo, hi) = bounds_i8(dq_scale.unwrap_or(1.0));
                        for v in tensor.data_mut() {
                            *v = v.clamp(lo, hi);
                        }
                    }
                } else if let (Some(sin), Some(sout)) = (dq_scale, q_scale) {
                    let out: Vec<f32> = tensor
                        .data()
                        .iter()
                        .map(|&v| {
                            let f = ((v - dq_zp) * sin).clamp(clamp.0, clamp.1);
                            (f / sout + q_zp).round_ties_even().clamp(-128.0, 127.0)
                        })
                        .collect();
                    tensor = Tensor::from_vec(tensor.shape().to_vec(), out).map_err(|e| {
                        OnnxError::DecodeFailed {
                            message: e.to_string(),
                        }
                    })?;
                } else if relu {
                    relu_inplace(&mut tensor);
                }
                env.insert(quant_node.outputs[0].clone(), tensor);
            }

            NodeAction::FusedHardSwishQuant {
                hs_idx,
                mul_idx,
                quant_idx,
            } => {
                let hs_node = &nodes[*hs_idx];
                let mul_node = &nodes[*mul_idx];
                let quant_node = &nodes[*quant_idx];
                if !quant_int8_fast_enabled() {
                    execute_node_with_layout_kind(hs_node, env, NodeKind::Other)?;
                    execute_node_with_layout_kind(mul_node, env, NodeKind::Mul)?;
                    execute_node_with_layout_kind(quant_node, env, NodeKind::Other)?;
                    continue;
                }
                note_quant_qdq_boundary();
                let alpha = get_attr_float(hs_node, Attr::Alpha).unwrap_or(0.2);
                let beta = get_attr_float(hs_node, Attr::Beta).unwrap_or(0.5);
                let scalar = |node: &OnnxNode, idx: usize| -> Option<f32> {
                    node.inputs
                        .get(idx)
                        .and_then(|name| env.get(name))
                        .and_then(|t| t.data().first().copied())
                };
                let y_scale = scalar(quant_node, 1);
                let y_zp = scalar(quant_node, 2).unwrap_or(0.0);
                let x_name = hs_node
                    .inputs
                    .first()
                    .ok_or_else(|| OnnxError::DecodeFailed {
                        message: format!("{}: HardSigmoid missing input", hs_node.name),
                    })?;
                if let (Some(ys), Some(x)) = (y_scale, env.get(x_name)) {
                    // One f32->i8 pass: quantize(x * hardsigmoid(x)).
                    let shape = x.shape().to_vec();
                    let mut out = vec![0_i8; x.data().len()];
                    yscv_kernels::hardswish_quantize_f32_to_i8_dispatch(
                        x.data(),
                        alpha,
                        beta,
                        ys,
                        y_zp,
                        &mut out,
                    );
                    let nhwc = env.is_nhwc(x_name);
                    env.insert_quant_i8(
                        quant_node.outputs[0].clone(),
                        QuantTensor {
                            data: out,
                            shape,
                            scale: ys,
                            zero_point: y_zp,
                            nhwc,
                        },
                    );
                } else {
                    // Missing scale (shouldn't happen under the plan gate) — run
                    // the three ops via the standard per-op path.
                    execute_node_with_layout_kind(hs_node, env, NodeKind::Other)?;
                    execute_node_with_layout_kind(mul_node, env, NodeKind::Mul)?;
                    execute_node_with_layout_kind(quant_node, env, NodeKind::Other)?;
                }
            }

            NodeAction::FusedSeMulQuant {
                mul_idx,
                quant_idx,
                feat_operand,
            } => {
                let mul_node = &nodes[*mul_idx];
                let quant_node = &nodes[*quant_idx];
                let feat_name = mul_node.inputs[*feat_operand as usize].clone();
                let gate_name = mul_node.inputs[1 - *feat_operand as usize].clone();
                let scalar = |node: &OnnxNode, idx: usize| -> Option<f32> {
                    node.inputs
                        .get(idx)
                        .and_then(|name| env.get(name))
                        .and_then(|t| t.data().first().copied())
                };
                let y_scale = scalar(quant_node, 1);
                let y_zp = scalar(quant_node, 2).unwrap_or(0.0);
                // Fold `Mul(feat, gate[N,C,1,1]) -> Quantize` into one per-channel
                // scaled f32->i8 pass. Bit-identical: `feat*gate` is the same f32
                // product the Mul computes, then `(v/scale+zp).round_ties_even()`
                // is the same quantise formula. Keeps the SE output as true int8.
                // Fast path only when the feature is NHWC (channel-contiguous):
                // then the SE gate broadcasts along the contiguous axis and the
                // fused kernel vectorises over C. NCHW would need a strided gate
                // broadcast and is rare here, so fall back to the two ops.
                let nhwc = env.is_nhwc(&feat_name);
                let fused_ok = quant_int8_fast_enabled()
                    && nhwc
                    && y_scale.is_some()
                    && env.get(&feat_name).is_some()
                    && env.get(&gate_name).is_some();
                if fused_ok {
                    let ys = y_scale.unwrap();
                    let feat = env.get(&feat_name).unwrap();
                    let gate = env.get(&gate_name).unwrap();
                    let fsh = feat.shape().to_vec();
                    let (n, c) = if fsh.len() == 4 {
                        (fsh[0], fsh[3])
                    } else {
                        (1, *fsh.last().unwrap_or(&1))
                    };
                    let mut out = vec![0_i8; feat.data().len()];
                    yscv_kernels::se_mul_quantize_nhwc_dispatch(
                        feat.data(),
                        gate.data(),
                        n.max(1),
                        c,
                        ys,
                        y_zp,
                        &mut out,
                    );
                    env.insert_quant_i8(
                        quant_node.outputs[0].clone(),
                        QuantTensor {
                            data: out,
                            shape: fsh,
                            scale: ys,
                            zero_point: y_zp,
                            nhwc,
                        },
                    );
                } else {
                    execute_node_with_layout_kind(mul_node, env, NodeKind::Mul)?;
                    execute_node_with_layout_kind(quant_node, env, NodeKind::Other)?;
                }
            }

            NodeAction::QuantizedPwDw {
                pw_idx,
                dq_idx,
                relu_idx,
                q_idx,
                dw_idx,
                has_relu,
            } => {
                if !quant_int8_fast_enabled() {
                    // Disabled via env: run the underlying nodes via the
                    // standard per-op path. Keeps `YSCV_QUANT_INT8_FAST=0`
                    // a true bitwise reference for the fused chain.
                    let pw_node = &nodes[*pw_idx];
                    execute_node_with_layout_kind(pw_node, env, NodeKind::Other)?;
                    let dq_node = &nodes[*dq_idx];
                    execute_node_with_layout_kind(dq_node, env, NodeKind::Other)?;
                    if let Some(ri) = relu_idx {
                        execute_node_with_layout_kind(&nodes[*ri], env, NodeKind::Relu)?;
                    }
                    let q_node = &nodes[*q_idx];
                    execute_node_with_layout_kind(q_node, env, NodeKind::Other)?;
                    let dw_node = &nodes[*dw_idx];
                    execute_node_with_layout_kind(dw_node, env, NodeKind::Other)?;
                    continue;
                }
                let pw_node = &nodes[*pw_idx];
                let dw_node = &nodes[*dw_idx];
                exec_quantized_pw_dw(pw_node, dw_node, env, *has_relu)?;
            }

            NodeAction::QuantizedDwPw {
                dw_idx,
                dq_idx,
                relu_idx,
                q_idx,
                pw_idx,
                has_relu,
            } => {
                if !quant_int8_fast_enabled() {
                    let dw_node = &nodes[*dw_idx];
                    execute_node_with_layout_kind(dw_node, env, NodeKind::Other)?;
                    let dq_node = &nodes[*dq_idx];
                    execute_node_with_layout_kind(dq_node, env, NodeKind::Other)?;
                    if let Some(ri) = relu_idx {
                        execute_node_with_layout_kind(&nodes[*ri], env, NodeKind::Relu)?;
                    }
                    let q_node = &nodes[*q_idx];
                    execute_node_with_layout_kind(q_node, env, NodeKind::Other)?;
                    let pw_node = &nodes[*pw_idx];
                    execute_node_with_layout_kind(pw_node, env, NodeKind::Other)?;
                    continue;
                }
                let dw_node = &nodes[*dw_idx];
                let pw_node = &nodes[*pw_idx];
                exec_quantized_dw_pw(dw_node, pw_node, env, *has_relu)?;
            }

            NodeAction::QuantizedForkPair {
                first_idx,
                dq_idx,
                relu_idx,
                q_idx,
                second_idx,
                first_kind,
                has_relu,
            } => {
                if *first_kind == 0 {
                    let side_node = relu_idx.map(|ri| &nodes[ri]).unwrap_or(&nodes[*dq_idx]);
                    exec_quantized_pw_dw_fork(
                        &nodes[*first_idx],
                        side_node,
                        &nodes[*second_idx],
                        env,
                        *has_relu,
                    )?;
                } else {
                    execute_node_with_layout_kind(&nodes[*first_idx], env, NodeKind::Other)?;
                    execute_node_with_layout_kind(&nodes[*dq_idx], env, NodeKind::Other)?;
                    if let Some(ri) = relu_idx {
                        execute_node_with_layout_kind(&nodes[*ri], env, NodeKind::Relu)?;
                    }
                    execute_node_with_layout_kind(&nodes[*q_idx], env, NodeKind::Other)?;
                    execute_node_with_layout_kind(&nodes[*second_idx], env, NodeKind::Other)?;
                }
            }

            NodeAction::QuantizedResidualChain {
                qconv_idx,
                dq_idx,
                relu_idx,
                conv_idx,
                add_idx,
                q_idx,
                qconv_kind,
            } => {
                if *qconv_kind == 1 {
                    exec_quantized_dw_residual(
                        &nodes[*qconv_idx],
                        &nodes[*conv_idx],
                        &nodes[*add_idx],
                        &nodes[*q_idx],
                        env,
                    )?;
                } else {
                    execute_node_with_layout_kind(&nodes[*qconv_idx], env, NodeKind::Other)?;
                    execute_node_with_layout_kind(&nodes[*dq_idx], env, NodeKind::Other)?;
                    execute_node_with_layout_kind(&nodes[*relu_idx], env, NodeKind::Relu)?;
                    execute_node_with_layout_kind(&nodes[*conv_idx], env, NodeKind::Conv)?;
                    execute_node_with_layout_kind(&nodes[*add_idx], env, NodeKind::Add)?;
                    execute_node_with_layout_kind(&nodes[*q_idx], env, NodeKind::Other)?;
                }
            }

            NodeAction::QuantizedConvDq {
                qconv_idx,
                dq_idx,
                qconv_kind,
            } => {
                if *qconv_kind == 0 {
                    exec_quantized_pw_dq(&nodes[*qconv_idx], &nodes[*dq_idx], env)?;
                } else {
                    execute_node_with_layout_kind(&nodes[*qconv_idx], env, NodeKind::Other)?;
                    execute_node_with_layout_kind(&nodes[*dq_idx], env, NodeKind::Other)?;
                }
            }

            NodeAction::Generic { node_idx, kind } => {
                let node = &nodes[*node_idx];
                if *kind == NodeKind::Reshape
                    && !env.is_nhwc(&node.inputs[0])
                    && let Some(shape) = reshape_shapes.and_then(|shapes| shapes.get(node_idx))
                {
                    exec_reshape_known(node, env, shape)?;
                } else {
                    execute_node_with_layout_kind(node, env, *kind)?;
                }
            }
        }

        if do_profile {
            let elapsed = t0
                .as_ref()
                .map(|start| start.elapsed().as_nanos() as u64)
                .unwrap_or(0);
            match action {
                NodeAction::Conv { .. }
                | NodeAction::FusedDwPw { .. }
                | NodeAction::FusedPwDw { .. }
                | NodeAction::FusedPwDwPwReduce { .. }
                | NodeAction::QuantizedPwDw { .. }
                | NodeAction::QuantizedDwPw { .. }
                | NodeAction::QuantizedForkPair { .. }
                | NodeAction::QuantizedResidualChain { .. }
                | NodeAction::QuantizedConvDq { .. }
                | NodeAction::ConvAdd { .. } => {
                    *conv_ns += elapsed;
                    *conv_count += 1;
                }
                NodeAction::Generic { .. } => {
                    *other_ns += elapsed;
                    *other_count += 1;
                }
                _ => {}
            }
        }

        // YSCV_RUNNER_PROFILE=path — per-node aggregated timing for the
        // fused path. Skips the measurement entirely when env was unset.
        if runner_profile_enabled {
            let elapsed = t0
                .as_ref()
                .map(|start| start.elapsed().as_nanos() as u64)
                .unwrap_or(0);
            let (name, op, in_shape, out_shape) = match action {
                NodeAction::Skip => continue,
                NodeAction::Conv {
                    node_idx,
                    activation,
                } => {
                    let n = &nodes[*node_idx];
                    let op_label = match activation {
                        1 => "Conv_Relu",
                        2 => "Conv_Silu",
                        _ => "Conv",
                    };
                    let in_sh = n
                        .inputs
                        .first()
                        .and_then(|nm| env.get(nm))
                        .map(|t| t.shape().to_vec())
                        .unwrap_or_default();
                    let out_sh = n
                        .outputs
                        .first()
                        .and_then(|nm| env.get(nm))
                        .map(|t| t.shape().to_vec())
                        .unwrap_or_default();
                    (n.name.clone(), op_label.to_string(), in_sh, out_sh)
                }
                NodeAction::FusedDwPw { dw_idx, pw_idx, .. } => {
                    let dw = &nodes[*dw_idx];
                    let pw = &nodes[*pw_idx];
                    let in_sh = dw
                        .inputs
                        .first()
                        .and_then(|nm| env.get(nm))
                        .map(|t| t.shape().to_vec())
                        .unwrap_or_default();
                    let out_sh = pw
                        .outputs
                        .first()
                        .and_then(|nm| env.get(nm))
                        .map(|t| t.shape().to_vec())
                        .unwrap_or_default();
                    (
                        format!("{}+{}", dw.name, pw.name),
                        "FusedDwPw".to_string(),
                        in_sh,
                        out_sh,
                    )
                }
                NodeAction::FusedPwDw { pw_idx, dw_idx, .. } => {
                    let pw = &nodes[*pw_idx];
                    let dw = &nodes[*dw_idx];
                    let in_sh = pw
                        .inputs
                        .first()
                        .and_then(|nm| env.get(nm))
                        .map(|t| t.shape().to_vec())
                        .unwrap_or_default();
                    let out_sh = dw
                        .outputs
                        .first()
                        .and_then(|nm| env.get(nm))
                        .map(|t| t.shape().to_vec())
                        .unwrap_or_default();
                    (
                        format!("{}+{}", pw.name, dw.name),
                        "FusedPwDw".to_string(),
                        in_sh,
                        out_sh,
                    )
                }
                NodeAction::FusedPwDwPwReduce {
                    pw_expand_idx,
                    dw_idx,
                    pw_reduce_idx,
                    ..
                } => {
                    let pw_e = &nodes[*pw_expand_idx];
                    let dw = &nodes[*dw_idx];
                    let pw_r = &nodes[*pw_reduce_idx];
                    let in_sh = pw_e
                        .inputs
                        .first()
                        .and_then(|nm| env.get(nm))
                        .map(|t| t.shape().to_vec())
                        .unwrap_or_default();
                    let out_sh = pw_r
                        .outputs
                        .first()
                        .and_then(|nm| env.get(nm))
                        .map(|t| t.shape().to_vec())
                        .unwrap_or_default();
                    (
                        format!("{}+{}+{}", pw_e.name, dw.name, pw_r.name),
                        "FusedPwDwPwReduce".to_string(),
                        in_sh,
                        out_sh,
                    )
                }
                NodeAction::ConvAdd {
                    conv_idx,
                    add_idx,
                    post_activation,
                    ..
                } => {
                    let c = &nodes[*conv_idx];
                    let a = &nodes[*add_idx];
                    let op_label = if *post_activation == 1 {
                        "Conv_Add_Relu"
                    } else {
                        "Conv_Add"
                    };
                    let in_sh = c
                        .inputs
                        .first()
                        .and_then(|nm| env.get(nm))
                        .map(|t| t.shape().to_vec())
                        .unwrap_or_default();
                    let out_sh = a
                        .outputs
                        .first()
                        .and_then(|nm| env.get(nm))
                        .map(|t| t.shape().to_vec())
                        .unwrap_or_default();
                    (c.name.clone(), op_label.to_string(), in_sh, out_sh)
                }
                NodeAction::FusedTransposeMatMul {
                    transpose_idx,
                    matmul_idx,
                    ..
                } => {
                    let t = &nodes[*transpose_idx];
                    let m = &nodes[*matmul_idx];
                    let in_sh = t
                        .inputs
                        .first()
                        .and_then(|nm| env.get(nm))
                        .map(|v| v.shape().to_vec())
                        .unwrap_or_default();
                    let out_sh = m
                        .outputs
                        .first()
                        .and_then(|nm| env.get(nm))
                        .map(|v| v.shape().to_vec())
                        .unwrap_or_default();
                    (
                        format!("{}+{}", t.name, m.name),
                        "FusedTransposeMatMul".to_string(),
                        in_sh,
                        out_sh,
                    )
                }
                NodeAction::QuantizedQdq {
                    dequant_idx,
                    relu_idx,
                    hardswish,
                    quant_idx,
                } => {
                    let d = &nodes[*dequant_idx];
                    let q = &nodes[*quant_idx];
                    let sh = q
                        .outputs
                        .first()
                        .and_then(|nm| env.get(nm))
                        .map(|t| t.shape().to_vec())
                        .unwrap_or_default();
                    let op = if hardswish.is_some() {
                        "QuantizedHardSwish"
                    } else if relu_idx.is_some() {
                        "QuantizedRelu"
                    } else {
                        "QuantizedQdq"
                    };
                    (d.name.clone(), op.to_string(), sh.clone(), sh)
                }
                NodeAction::FusedHardSwishQuant {
                    hs_idx, quant_idx, ..
                } => {
                    let hs = &nodes[*hs_idx];
                    let q = &nodes[*quant_idx];
                    let sh = q
                        .outputs
                        .first()
                        .and_then(|nm| env.get(nm))
                        .map(|t| t.shape().to_vec())
                        .unwrap_or_default();
                    (
                        hs.name.clone(),
                        "FusedHardSwishQuant".to_string(),
                        sh.clone(),
                        sh,
                    )
                }
                NodeAction::FusedSeMulQuant {
                    mul_idx, quant_idx, ..
                } => {
                    let m = &nodes[*mul_idx];
                    let q = &nodes[*quant_idx];
                    let sh = q
                        .outputs
                        .first()
                        .and_then(|nm| env.get(nm))
                        .map(|t| t.shape().to_vec())
                        .unwrap_or_default();
                    (
                        m.name.clone(),
                        "FusedSeMulQuant".to_string(),
                        sh.clone(),
                        sh,
                    )
                }
                NodeAction::QuantizedPwDw {
                    pw_idx,
                    dw_idx,
                    has_relu,
                    ..
                } => {
                    let pw = &nodes[*pw_idx];
                    let dw = &nodes[*dw_idx];
                    let in_sh = pw
                        .inputs
                        .first()
                        .and_then(|nm| env.get(nm))
                        .map(|t| t.shape().to_vec())
                        .unwrap_or_default();
                    let out_sh = dw
                        .outputs
                        .first()
                        .and_then(|nm| env.get(nm))
                        .map(|t| t.shape().to_vec())
                        .unwrap_or_default();
                    let op = if *has_relu {
                        "QuantizedPwReluDw"
                    } else {
                        "QuantizedPwDw"
                    };
                    (
                        format!("{}+{}", pw.name, dw.name),
                        op.to_string(),
                        in_sh,
                        out_sh,
                    )
                }
                NodeAction::QuantizedDwPw {
                    dw_idx,
                    pw_idx,
                    has_relu,
                    ..
                } => {
                    let dw = &nodes[*dw_idx];
                    let pw = &nodes[*pw_idx];
                    let in_sh = dw
                        .inputs
                        .first()
                        .and_then(|nm| env.get(nm))
                        .map(|t| t.shape().to_vec())
                        .unwrap_or_default();
                    let out_sh = pw
                        .outputs
                        .first()
                        .and_then(|nm| env.get(nm))
                        .map(|t| t.shape().to_vec())
                        .unwrap_or_default();
                    let op = if *has_relu {
                        "QuantizedDwReluPw"
                    } else {
                        "QuantizedDwPw"
                    };
                    (
                        format!("{}+{}", dw.name, pw.name),
                        op.to_string(),
                        in_sh,
                        out_sh,
                    )
                }
                NodeAction::QuantizedForkPair {
                    first_idx,
                    second_idx,
                    first_kind,
                    has_relu,
                    ..
                } => {
                    let first = &nodes[*first_idx];
                    let second = &nodes[*second_idx];
                    let in_sh = first
                        .inputs
                        .first()
                        .and_then(|nm| env.get(nm))
                        .map(|t| t.shape().to_vec())
                        .unwrap_or_default();
                    let out_sh = second
                        .outputs
                        .first()
                        .and_then(|nm| env.get(nm))
                        .map(|t| t.shape().to_vec())
                        .unwrap_or_default();
                    let prefix = if *first_kind == 1 {
                        "QuantizedForkDw"
                    } else {
                        "QuantizedForkPw"
                    };
                    let op = if *has_relu {
                        format!("{prefix}Relu")
                    } else {
                        prefix.to_string()
                    };
                    (format!("{}+{}", first.name, second.name), op, in_sh, out_sh)
                }
                NodeAction::QuantizedResidualChain {
                    qconv_idx,
                    conv_idx,
                    add_idx,
                    qconv_kind,
                    ..
                } => {
                    let qconv = &nodes[*qconv_idx];
                    let conv = &nodes[*conv_idx];
                    let add = &nodes[*add_idx];
                    let in_sh = qconv
                        .inputs
                        .first()
                        .and_then(|nm| env.get(nm))
                        .map(|t| t.shape().to_vec())
                        .unwrap_or_default();
                    let out_sh = add
                        .outputs
                        .first()
                        .and_then(|nm| env.get(nm))
                        .map(|t| t.shape().to_vec())
                        .unwrap_or_default();
                    let op = if *qconv_kind == 1 {
                        "QuantizedDwResidual"
                    } else {
                        "QuantizedPwResidual"
                    };
                    (
                        format!("{}+{}", qconv.name, conv.name),
                        op.to_string(),
                        in_sh,
                        out_sh,
                    )
                }
                NodeAction::QuantizedConvDq {
                    qconv_idx,
                    qconv_kind,
                    ..
                } => {
                    let qconv = &nodes[*qconv_idx];
                    let in_sh = qconv
                        .inputs
                        .first()
                        .and_then(|nm| env.get(nm))
                        .map(|t| t.shape().to_vec())
                        .unwrap_or_default();
                    let out_sh = qconv
                        .outputs
                        .first()
                        .and_then(|nm| env.get(nm))
                        .map(|t| t.shape().to_vec())
                        .unwrap_or_default();
                    let op = if *qconv_kind == 1 {
                        "QuantizedDwDq"
                    } else {
                        "QuantizedPwDq"
                    };
                    (qconv.name.clone(), op.to_string(), in_sh, out_sh)
                }
                NodeAction::Generic { node_idx, .. } => {
                    let n = &nodes[*node_idx];
                    let in_sh = n
                        .inputs
                        .first()
                        .and_then(|nm| env.get(nm))
                        .map(|t| t.shape().to_vec())
                        .unwrap_or_default();
                    let out_sh = n
                        .outputs
                        .first()
                        .and_then(|nm| env.get(nm))
                        .map(|t| t.shape().to_vec())
                        .unwrap_or_default();
                    (n.name.clone(), n.op_type.clone(), in_sh, out_sh)
                }
            };
            // Unnamed nodes would otherwise all merge under the empty key.
            let key = if name.is_empty() {
                format!("{op}@{action_idx}")
            } else {
                name
            };
            runner_profile_record(&key, &op, elapsed, in_shape, out_shape);
        }

        // Early deallocation: static slice match covers fixed-arity variants.
        let covered_dyn: Vec<usize>;
        let covered_nodes: &[usize] = match action {
            NodeAction::Conv { node_idx, .. } | NodeAction::Generic { node_idx, .. } => {
                std::slice::from_ref(node_idx)
            }
            // `exec_fused_dw_pw` does its own early
            // cleanup of DW's inputs between DW and PW exec calls — the
            // outer loop must only handle PW's inputs here, otherwise
            // DW's inputs get double-decremented and the `saturating_sub`
            // hides the resulting off-by-one.
            NodeAction::FusedDwPw { pw_idx, .. } => std::slice::from_ref(pw_idx),
            // Mirror of FusedDwPw: `exec_fused_pw_dw` cleans up PW's
            // inputs between PW and DW, so the outer loop here only
            // touches DW's inputs (which includes the locally-owned PW
            // output that was never inserted into env — harmless since
            // `resolve_id` returns None for that unresolved name, skipping
            // the decrement).
            NodeAction::FusedPwDw { dw_idx, .. } => std::slice::from_ref(dw_idx),
            // Streaming variant: `exec_fused_pw_dw_pw_reduce` already
            // decrements PW expand's inputs internally (same as FusedPwDw).
            // The outer loop covers DW's inputs (PW-expand intermediate is
            // unresolved — harmless), PW reduce's inputs (DW intermediate
            // unresolved — harmless), and any absorbed Add/Relu nodes
            // (their inputs include the residual side-branch, which IS
            // resolved in env and must be decremented).
            NodeAction::FusedPwDwPwReduce {
                dw_idx,
                pw_reduce_idx,
                residual,
                ..
            } => {
                let mut v = vec![*dw_idx, *pw_reduce_idx];
                if let Some(r) = residual {
                    v.push(r.add_idx);
                    if r.post_activation == 1 {
                        v.push(r.relu_idx as usize);
                    }
                }
                covered_dyn = v;
                &covered_dyn[..]
            }
            // `FusedTransposeMatMul` cleanup: the Transpose node was
            // elided from the plan, but its input tensor still lives
            // in `env`. Only the variant flagged with `cleanup_transpose`
            // covers the transpose's inputs — otherwise a transpose
            // feeding N MatMuls would get its input decremented N
            // times against an original use-count of 1, evicting the
            // pre-transpose tensor before every consumer has read it.
            NodeAction::FusedTransposeMatMul {
                transpose_idx,
                matmul_idx,
                cleanup_transpose,
            } => {
                if *cleanup_transpose {
                    &[*transpose_idx, *matmul_idx][..]
                } else {
                    std::slice::from_ref(matmul_idx)
                }
            }
            NodeAction::ConvAdd {
                conv_idx, add_idx, ..
            } => &[*conv_idx, *add_idx][..],
            NodeAction::QuantizedQdq {
                dequant_idx,
                relu_idx,
                hardswish,
                quant_idx,
            } => match (relu_idx, hardswish) {
                (_, Some((hs_idx, mul_idx))) => {
                    covered_dyn = vec![*dequant_idx, *hs_idx, *mul_idx, *quant_idx];
                    &covered_dyn[..]
                }
                (Some(ri), None) => &[*dequant_idx, *ri, *quant_idx][..],
                (None, None) => &[*dequant_idx, *quant_idx][..],
            },
            NodeAction::FusedHardSwishQuant {
                hs_idx,
                mul_idx,
                quant_idx,
            } => &[*hs_idx, *mul_idx, *quant_idx][..],
            NodeAction::FusedSeMulQuant {
                mul_idx, quant_idx, ..
            } => &[*mul_idx, *quant_idx][..],
            // Fused INT8 chain: the action wraps PW + DQ + (Relu) + Q + DW.
            // `exec_quantized_pw_dw` consumes PW's inputs internally
            // (specifically: `take_quant_i8` on PW.inputs[0] when the
            // refcount is 1), so the outer cleanup must cover the rest
            // (PW's other inputs — scales/zps/weights/bias — plus DQ, Q,
            // optional Relu, and DW). Mirrors how `FusedPwDw` only
            // cleans DW's inputs because the kernel handles PW's
            // intra-chain output directly.
            NodeAction::QuantizedPwDw {
                pw_idx,
                dq_idx,
                relu_idx,
                q_idx,
                dw_idx,
                ..
            } => {
                covered_dyn = match relu_idx {
                    Some(ri) => vec![*pw_idx, *dq_idx, *ri, *q_idx, *dw_idx],
                    None => vec![*pw_idx, *dq_idx, *q_idx, *dw_idx],
                };
                &covered_dyn[..]
            }
            // Mirror of QuantizedPwDw: `exec_quantized_dw_pw` consumes
            // DW's first input internally via `take_quant_i8`; the outer
            // cleanup covers DW's other inputs (scales/zps/weights/bias),
            // DQ, optional Relu, Q, and PW.
            NodeAction::QuantizedDwPw {
                dw_idx,
                dq_idx,
                relu_idx,
                q_idx,
                pw_idx,
                ..
            } => {
                covered_dyn = match relu_idx {
                    Some(ri) => vec![*dw_idx, *dq_idx, *ri, *q_idx, *pw_idx],
                    None => vec![*dw_idx, *dq_idx, *q_idx, *pw_idx],
                };
                &covered_dyn[..]
            }
            NodeAction::QuantizedForkPair {
                first_idx,
                dq_idx,
                relu_idx,
                q_idx,
                second_idx,
                ..
            } => {
                covered_dyn = match relu_idx {
                    Some(ri) => vec![*first_idx, *dq_idx, *ri, *q_idx, *second_idx],
                    None => vec![*first_idx, *dq_idx, *q_idx, *second_idx],
                };
                &covered_dyn[..]
            }
            NodeAction::QuantizedResidualChain {
                qconv_idx,
                dq_idx,
                relu_idx,
                conv_idx,
                add_idx,
                q_idx,
                ..
            } => {
                covered_dyn = vec![*qconv_idx, *dq_idx, *relu_idx, *conv_idx, *add_idx, *q_idx];
                &covered_dyn[..]
            }
            NodeAction::QuantizedConvDq {
                qconv_idx, dq_idx, ..
            } => &[*qconv_idx, *dq_idx][..],
            NodeAction::Skip => continue,
        };
        let input_ids = &model.runtime_index.node_input_ids;
        for &nidx in covered_nodes {
            let node = &nodes[nidx];
            let pre_ids = if nidx < input_ids.len() {
                &input_ids[nidx]
            } else {
                &[][..]
            };
            for (inp_idx, inp) in node.inputs.iter().enumerate() {
                if inp.is_empty() {
                    continue;
                }
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
    }
    Ok(())
}
