//! Branch executor: runs one (possibly tower-parallel) slice of the
//! optimized node plan, threading the TensorEnv through fused dispatch.

use super::*;

/// Run a subset of the JIT execution plan, filtered by a predicate on the
/// node's branch assignment. Shared body for both the single-branch path and
/// the tower-parallel wrapper.
pub(crate) fn execute_plan_branch(
    model: &OnnxModel,
    env: &mut TensorEnv<'_, '_>,
    remaining_uses: &mut [usize],
    output_id_mask: &[bool],
    mut accept: impl FnMut(usize) -> bool,
    conv_ns: &mut u64,
    other_ns: &mut u64,
    conv_count: &mut u32,
    other_count: &mut u32,
    do_profile: bool,
) -> Result<(), OnnxError> {
    use crate::loader::NodeAction;

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
                exec_conv_with_params(node, env, act, cp, prepacked)?;
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
                // M3 enclave lookahead: if the next non-Skip plan action
                // is another FusedDwPw whose DW input is this PW output,
                // the chain can skip the round-trip nchwc→nhwc +
                // nhwc→nchwc. Caller (exec_fused_dw_pw AVX-512 path)
                // honours `leave_nchwc_output` and marks the env-slot
                // via `mark_nchwc`. Absorbed Relu nodes become
                // `NodeAction::Skip` in the plan — walk past them so
                // genuine FusedDwPw chains (e.g. cls_dw/enc.0 →
                // cls_tower.0 connected by an absorbed Relu) are found.
                let leave_nchwc_output = {
                    let mut found = false;
                    let mut probe = action_idx + 1;
                    while let Some(next) = plan.get(probe) {
                        match next {
                            NodeAction::Skip => {
                                probe += 1;
                                continue;
                            }
                            NodeAction::FusedDwPw {
                                dw_idx: next_dw_idx,
                                pw_idx: next_pw_idx,
                                ..
                            } => {
                                let next_dw = &nodes[*next_dw_idx];
                                let next_pw = &nodes[*next_pw_idx];
                                let next_dw_ws = next_dw
                                    .inputs
                                    .get(1)
                                    .and_then(|n| env.get(n))
                                    .map(|t| t.shape().to_vec())
                                    .unwrap_or_default();
                                let next_pw_ws = next_pw
                                    .inputs
                                    .get(1)
                                    .and_then(|n| env.get(n))
                                    .map(|t| t.shape().to_vec())
                                    .unwrap_or_default();
                                // Same gate conditions as the AVX-512 path
                                // inside exec_fused_dw_pw: must be 3×3 DW
                                // SAME-pad, 1×1 PW, both c dims multiple
                                // of 16. Without this, the next op fails
                                // the gate and falls into NHWC path with
                                // NCHWc input → crash.
                                let next_gate_ok = next_dw.inputs.len() >= 2
                                    && next_pw.inputs.len() >= 2
                                    && next_dw_ws.len() == 4
                                    && next_pw_ws.len() == 4
                                    && next_dw_ws[0] == 3
                                    && next_dw_ws[1] == 3
                                    && next_dw_ws[3] == 1
                                    && next_pw_ws[0] == 1
                                    && next_pw_ws[1] == 1
                                    && next_pw_ws[2] == next_dw_ws[2]
                                    && next_dw_ws[2].is_multiple_of(16)
                                    && next_pw_ws[3].is_multiple_of(16);
                                if !next_dw.inputs.is_empty()
                                    && next_dw.inputs[0] == pw_node.outputs[0]
                                    && next_gate_ok
                                {
                                    found = true;
                                }
                                break;
                            }
                            _ => break,
                        }
                    }
                    found
                };
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
                    leave_nchwc_output,
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
                // NCHWc chain lookahead: leave NCHWc when the next op is
                // another FusedPwDwPwReduce/FusedPwDw whose input is this
                // FusedPwDw's DW output (`dw_node.outputs[0]`).
                let this_output_name = &dw_node.outputs[0];
                let leave_nchwc_output = {
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
                                    && next_node.inputs[0] == *this_output_name
                                {
                                    found = true;
                                }
                                break;
                            }
                            _ => break,
                        }
                    }
                    found
                };
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
                    leave_nchwc_output,
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
                // Backbone NCHWc chaining lookahead: leave NCHWc output when
                // the next plan action is another FusedPwDwPwReduce whose
                // first input is this node's final output (accounting for
                // absorbed Add+Relu).
                let this_output_name: &str = match &residual_meta_tuple {
                    Some((_, _, 1, Some(relu_node))) => &relu_node.outputs[0],
                    Some((add_node, _, _, _)) => &add_node.outputs[0],
                    None => &pw_reduce_node.outputs[0],
                };
                let leave_nchwc_output = {
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
                                    && next_node.inputs[0] == *this_output_name
                                {
                                    found = true;
                                }
                                break;
                            }
                            _ => break,
                        }
                    }
                    found
                };
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
                    leave_nchwc_output,
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
                    // cached add_out slot ID avoids a HashMap
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
                quant_idx,
            } => {
                let dequant_node = &nodes[*dequant_idx];
                let quant_node = &nodes[*quant_idx];
                if !quant_int8_fast_enabled() {
                    execute_node_with_layout_kind(dequant_node, env, NodeKind::Other)?;
                    if let Some(ri) = relu_idx {
                        execute_node_with_layout_kind(&nodes[*ri], env, NodeKind::Relu)?;
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
                if let Some(mut qt) = env.take_quant_i8(input_name) {
                    if relu_idx.is_some() {
                        for v in &mut qt.data {
                            *v = (*v).max(0);
                        }
                    }
                    qt.scale = quant_node
                        .inputs
                        .get(1)
                        .and_then(|name| env.get(name))
                        .and_then(|t| t.data().first().copied())
                        .unwrap_or(qt.scale);
                    qt.zero_point = quant_node
                        .inputs
                        .get(2)
                        .and_then(|name| env.get(name))
                        .and_then(|t| t.data().first().copied())
                        .unwrap_or(qt.zero_point);
                    env.insert_quant_i8(quant_node.outputs[0].clone(), qt);
                    continue;
                }
                let mut tensor = env
                    .remove(input_name)
                    .or_else(|| env.get(input_name).cloned())
                    .ok_or_else(|| OnnxError::MissingInput {
                        node: dequant_node.name.clone(),
                        input: input_name.clone(),
                    })?;
                if relu_idx.is_some() {
                    relu_inplace(&mut tensor);
                }
                env.insert(quant_node.outputs[0].clone(), tensor);
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
                execute_node_with_layout_kind(node, env, *kind)?;
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
                    let op = if relu_idx.is_some() {
                        "QuantizedRelu"
                    } else {
                        "QuantizedQdq"
                    };
                    (d.name.clone(), op.to_string(), sh.clone(), sh)
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
            runner_profile_record(&name, &op, elapsed, in_shape, out_shape);
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
                quant_idx,
            } => match relu_idx {
                Some(ri) => &[*dequant_idx, *ri, *quant_idx][..],
                None => &[*dequant_idx, *quant_idx][..],
            },
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
