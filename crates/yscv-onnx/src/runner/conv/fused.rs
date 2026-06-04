//! fp32 fused convolution paths: PW→DW, DW→PW, and the streaming
//! PW-expand→DW→PW-reduce MBConv block (incl. the NCHWc-chained variants).

use super::super::*;
use super::*;

/// ONNX Conv: NHWC-aware. Skips layout conversion if input is already NHWC.
/// Output is left in NHWC to avoid redundant conversions in spatial chains.
/// `activation` is applied fused into GEMM tiles (cache-hot) when using BLAS padded path.
/// Fused DW+PW execution: depthwise Conv + pointwise 1x1 Conv back-to-back.
/// Executes DW first, then PW immediately while DW output is hot in L1 cache.
/// Skips intermediate tensor early-deallocation check to keep it cached.
/// True-fuse DW → PW by keeping the DW output as a local `Tensor`,
/// never touching the env HashMap for the intermediate. Chains two
/// `conv_compute_nhwc` calls and only binds the final PW result to env.
/// Saves one `env.insert` + one `env.get` + one `env.remove` + the
/// associated HashMap lookups per fused pair vs the previous "call
/// exec_conv_with_params twice" implementation.
///
/// Also does early cleanup of DW's input refs between DW and PW so
/// the outer dispatch loop's cleanup pass must *not* re-walk them
/// (it would double-decrement `remaining_uses`). Caller's
/// `covered_nodes` for `NodeAction::FusedDwPw` therefore contains
/// only `pw_idx`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn exec_fused_dw_pw(
    dw_node: &OnnxNode,
    pw_node: &OnnxNode,
    env: &mut TensorEnv,
    dw_activation: yscv_kernels::Activation,
    pw_activation: yscv_kernels::Activation,
    dw_params: Option<&crate::loader::ConvParams>,
    pw_params: Option<&crate::loader::ConvParams>,
    dw_input_ids: &[Option<usize>],
    remaining_uses: &mut [usize],
    output_id_mask: &[bool],
    leave_nchwc_output: bool,
) -> Result<(), OnnxError> {
    // ---- DW compute ----
    let dw_input_is_nhwc = env.is_nhwc(&dw_node.inputs[0]);
    let (dw_sh, dw_sw, dw_group, dw_pt, dw_pl, dw_pb, dw_pr, dw_has_pad) =
        resolve_conv_params(dw_node, dw_params);

    // Borrow DW tensors from env. `Tensor` is `Arc<_>` so cheap clones
    // decouple the borrow lifetime from env; this matters because the
    // early-cleanup loop below mutates env while `dw_output` is still
    // live. Cloning the Arcs is a few ns of refcount bumps.
    let dw_input_src = get_tensor(env, &dw_node.name, &dw_node.inputs[0])?.clone();
    let dw_weight = get_tensor(env, &dw_node.name, &dw_node.inputs[1])?.clone();
    let dw_bias = if dw_node.inputs.len() > 2 && !dw_node.inputs[2].is_empty() {
        Some(get_tensor(env, &dw_node.name, &dw_node.inputs[2])?.clone())
    } else {
        None
    };
    // M3 enclave: if the env-tracked input is NCHWc(16) (from an upstream
    // FusedDwPw_AVX512 in the same chain), it's a 5-D tensor and the
    // standard NHWC materialization (`nchw_to_nhwc`) would error on rank
    // mismatch. Skip the NHWC dance in that case; the AVX-512 NCHWc-pair
    // branch below handles the tensor directly.
    let input_already_nchwc16 = env.nchwc_block(&dw_node.inputs[0]) == Some(16);
    let dw_input_nhwc_owned;
    let dw_input_nhwc: &Tensor = if input_already_nchwc16 {
        // The branch below uses `dw_input_src` directly; this binding is
        // only kept for code-path uniformity (the NHWC streaming path
        // never fires when nchwc16 is set because the AVX-512 gate runs
        // first).
        &dw_input_src
    } else if dw_input_is_nhwc {
        &dw_input_src
    } else {
        dw_input_nhwc_owned = nchw_to_nhwc(&dw_input_src)?;
        &dw_input_nhwc_owned
    };
    let (pw_sh, pw_sw, pw_group, pw_pt, pw_pl, pw_pb, pw_pr, pw_has_pad) =
        resolve_conv_params(pw_node, pw_params);

    // Streaming DW->PW fast path. Reuses the per-row fused kernel from
    // yscv-kernels and avoids materializing the full DW intermediate in
    // memory. Default ON at all thread counts: the kernel now parallelises
    // internally via par_chunks_mut_dispatch, so each thread streams its own
    // row-chunk through a per-thread L2-resident DW scratch. Kill switch:
    // `YSCV_FUSED_DW_PW_STREAM_OFF=1`.
    //
    // Padded-DW streaming is opt-in (`YSCV_FUSED_DW_PW_STREAM_PADDED=1`):
    // on low-end ARM boards this path can regress vs the existing
    // non-stream fused route, so keep it disabled by default until
    // we land a tuned padded streaming schedule.
    let stream_disabled = {
        use std::sync::OnceLock;
        static CACHED: OnceLock<bool> = OnceLock::new();
        *CACHED.get_or_init(|| {
            std::env::var_os("YSCV_FUSED_DW_PW_STREAM_OFF").is_some()
                || (cfg!(target_arch = "aarch64")
                    && std::env::var_os("YSCV_FUSED_DW_PW_STREAM_ON").is_none())
        })
    };
    let stream_padded_enabled = {
        use std::sync::OnceLock;
        static CACHED: OnceLock<bool> = OnceLock::new();
        *CACHED.get_or_init(|| std::env::var_os("YSCV_FUSED_DW_PW_STREAM_PADDED").is_some())
    };

    let pw_weight_stream = get_tensor(env, &pw_node.name, &pw_node.inputs[1])?.clone();
    let pw_bias_stream = if pw_node.inputs.len() > 2 && !pw_node.inputs[2].is_empty() {
        Some(get_tensor(env, &pw_node.name, &pw_node.inputs[2])?.clone())
    } else {
        None
    };
    let dw_ws = dw_weight.shape();
    let pw_ws = pw_weight_stream.shape();
    let dw_streamable = env.is_dw_khwc_weight(&dw_node.inputs[1])
        && dw_ws.len() == 4
        && dw_ws[3] == 1
        && (!dw_has_pad || stream_padded_enabled);
    let pw_streamable = env.is_khwc_weight(&pw_node.inputs[1])
        && pw_ws.len() == 4
        && pw_ws[0] == 1
        && pw_ws[1] == 1
        && pw_group == 1
        && pw_sh == 1
        && pw_sw == 1
        && !pw_has_pad
        && pw_ws[2] == dw_ws[2];
    let nchwc_pair_enabled = {
        use std::sync::OnceLock;
        static CACHED: OnceLock<bool> = OnceLock::new();
        *CACHED.get_or_init(|| std::env::var_os("YSCV_FUSED_DW_PW_NCHWC").is_some())
    };
    // M3 NCHWc-pair AVX-512 gate. Decoupled from `dw_streamable` because
    // the rewritten `conv2d_nchwc_dw3x3_s1_same_pad` handles SAME-pad
    // natively via border-aware loads. PW pad/stride still constrained
    // (we lack a strided NCHWc PW kernel).
    let nchwc_pair_avx512_enabled = {
        use std::sync::OnceLock;
        static CACHED: OnceLock<bool> = OnceLock::new();
        *CACHED.get_or_init(|| std::env::var_os("YSCV_FUSED_DW_PW_NCHWC_AVX512").is_some())
    };
    // Head FusedDwPw NCHWc under `YSCV_BACKBONE_NCHWC`, chain-only gated.
    // For a tower block the NCHWc path (convert-in + fast NCHWc DW + direct
    // AVX-512 PW + convert-out) beats NHWC. Pred-heads (cls_pred c_out=1,
    // bbox_pred c_out=4) stay NHWC: `conv2d_nchwc_pointwise` pads c_out to the
    // 16-lane block, so a 256→1 PW computes 16× the FLOPs of the NHWC fast
    // path. So gate on c_out ≥ 16 (multiple of 16) AND chain-eligibility.
    let backbone_nchwc = {
        use std::sync::OnceLock;
        static CACHED: OnceLock<bool> = OnceLock::new();
        *CACHED.get_or_init(|| std::env::var_os("YSCV_BACKBONE_NCHWC").is_some())
    };
    let pw_cout = if pw_ws.len() == 4 { pw_ws[3] } else { 0 };
    let input_already_nchwc16_pre = env.nchwc_block(&dw_node.inputs[0]) == Some(16);
    // Chain-only firing: input already NCHWc (no input convert) OR leaving
    // NCHWc forward (no output convert). Per-op NCHWc for an ISOLATED head
    // loses — the cold-cache NCHWc PW + per-op conversion/alloc costs outweigh
    // the DW saving — so only fire inside an NCHWc chain.
    let chain_entry = input_already_nchwc16_pre || leave_nchwc_output;
    let dwpw_smart_enabled =
        backbone_nchwc && chain_entry && pw_cout >= 16 && pw_cout.is_multiple_of(16);
    #[cfg(target_arch = "x86_64")]
    let has_avx512f = std::is_x86_feature_detected!("avx512f");
    #[cfg(not(target_arch = "x86_64"))]
    let has_avx512f = false;
    if (nchwc_pair_avx512_enabled || dwpw_smart_enabled)
        && has_avx512f
        && !crate::quantize::calibrate::calibration_active()
        && env.is_dw_khwc_weight(&dw_node.inputs[1])
        && env.is_khwc_weight(&pw_node.inputs[1])
        && dw_ws.len() == 4
        && dw_ws[3] == 1
        && pw_ws.len() == 4
        && pw_ws[0] == 1
        && pw_ws[1] == 1
        && pw_group == 1
        && pw_sh == 1
        && pw_sw == 1
        && !pw_has_pad
        && pw_ws[2] == dw_ws[2]
        && dw_ws[0] == 3
        && dw_ws[1] == 3
        && dw_sh == 1
        && dw_sw == 1
        && dw_pt == 1
        && dw_pl == 1
        && dw_pb == 1
        && dw_pr == 1
        && dw_ws[2].is_multiple_of(16)
        && pw_ws[3].is_multiple_of(16)
    {
        // M3 enclave: if the input is already NCHWc(16) from an upstream
        // FusedDwPw_AVX512 in the same chain, use it directly. Otherwise
        // convert from NHWC.
        let input_already_nchwc16 = env.nchwc_block(&dw_node.inputs[0]) == Some(16);
        let input_nchwc_owned;
        let input_nchwc: &yscv_tensor::Tensor = if input_already_nchwc16 {
            &dw_input_src
        } else {
            input_nchwc_owned = yscv_kernels::nhwc_to_nchwc(dw_input_nhwc, 16).map_err(|e| {
                OnnxError::DecodeFailed {
                    message: e.to_string(),
                }
            })?;
            &input_nchwc_owned
        };
        let dw_nchwc = yscv_kernels::conv2d_nchwc_dw3x3_s1_same_pad(
            input_nchwc,
            &dw_weight,
            dw_bias.as_ref(),
            dw_activation,
            dw_ws[2],
            yscv_kernels::ParallelElementwiseConfig::default(),
            None,
            None,
        )
        .map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
        let pw_nchwc = yscv_kernels::conv2d_nchwc_pointwise_with_activation_prepacked(
            &dw_nchwc,
            &pw_weight_stream,
            pw_bias_stream.as_ref(),
            dw_ws[2],
            pw_activation,
            yscv_kernels::ParallelElementwiseConfig::default(),
            None,
            None,
        )
        .map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
        let pw_out_channels = pw_ws[3];
        drop(dw_input_src);
        drop(dw_weight);
        drop(dw_bias);
        drop(pw_weight_stream);
        drop(pw_bias_stream);
        for (inp_idx, inp) in dw_node.inputs.iter().enumerate() {
            if inp.is_empty() {
                continue;
            }
            let id = dw_input_ids
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
        if leave_nchwc_output {
            // Skip nchwc→nhwc; the next FusedDwPw_AVX512 consumes it directly.
            env.insert(pw_node.outputs[0].clone(), pw_nchwc);
            env.mark_nchwc(&pw_node.outputs[0], 16);
        } else {
            let pw_output =
                yscv_kernels::nchwc_to_nhwc(&pw_nchwc, pw_out_channels).map_err(|e| {
                    OnnxError::DecodeFailed {
                        message: e.to_string(),
                    }
                })?;
            env.insert(pw_node.outputs[0].clone(), pw_output);
            env.mark_nhwc(&pw_node.outputs[0]);
        }
        return Ok(());
    }
    if nchwc_pair_enabled
        && !crate::quantize::calibrate::calibration_active()
        && dw_streamable
        && pw_streamable
        && dw_ws[0] == 3
        && dw_ws[1] == 3
        && dw_sh == 1
        && dw_sw == 1
        && dw_pt == 1
        && dw_pl == 1
        && dw_pb == 1
        && dw_pr == 1
        && dw_ws[2].is_multiple_of(8)
    {
        let input_nchwc =
            yscv_kernels::nhwc_to_nchwc(dw_input_nhwc, 8).map_err(|e| OnnxError::DecodeFailed {
                message: e.to_string(),
            })?;
        let dw_nchwc = yscv_kernels::conv2d_nchwc_dw3x3_s1_same_pad(
            &input_nchwc,
            &dw_weight,
            dw_bias.as_ref(),
            dw_activation,
            dw_ws[2],
            yscv_kernels::ParallelElementwiseConfig::default(),
            None,
            None,
        )
        .map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
        let pw_nchwc = yscv_kernels::conv2d_nchwc_pointwise_with_activation_prepacked(
            &dw_nchwc,
            &pw_weight_stream,
            pw_bias_stream.as_ref(),
            dw_ws[2],
            pw_activation,
            yscv_kernels::ParallelElementwiseConfig::default(),
            None,
            None,
        )
        .map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
        let pw_output = yscv_kernels::nchwc_to_nhwc(&pw_nchwc, pw_ws[3]).map_err(|e| {
            OnnxError::DecodeFailed {
                message: e.to_string(),
            }
        })?;

        drop(dw_input_src);
        drop(dw_weight);
        drop(dw_bias);
        drop(pw_weight_stream);
        drop(pw_bias_stream);

        for (inp_idx, inp) in dw_node.inputs.iter().enumerate() {
            if inp.is_empty() {
                continue;
            }
            let id = dw_input_ids
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

        env.insert(pw_node.outputs[0].clone(), pw_output);
        env.mark_nhwc(&pw_node.outputs[0]);
        return Ok(());
    }
    if !stream_disabled
        && !crate::quantize::calibrate::calibration_active()
        && dw_streamable
        && pw_streamable
    {
        let pw_output = yscv_kernels::fused_dw_pw_nhwc_streaming(
            dw_input_nhwc,
            &dw_weight,
            dw_bias.as_ref(),
            &pw_weight_stream,
            pw_bias_stream.as_ref(),
            None, // no residual for plain FusedDwPw
            dw_sh,
            dw_sw,
            dw_pt,
            dw_pl,
            dw_pb,
            dw_pr,
            dw_activation,
            pw_activation,
        )
        .map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;

        drop(dw_input_src);
        drop(dw_weight);
        drop(dw_bias);
        drop(pw_weight_stream);
        drop(pw_bias_stream);

        // Early cleanup mirrors the unfused path.
        for (inp_idx, inp) in dw_node.inputs.iter().enumerate() {
            if inp.is_empty() {
                continue;
            }
            let id = dw_input_ids
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

        env.insert(pw_node.outputs[0].clone(), pw_output);
        env.mark_nhwc(&pw_node.outputs[0]);
        return Ok(());
    }

    drop(pw_weight_stream);
    drop(pw_bias_stream);

    let dw_output = conv_compute_nhwc(
        dw_node,
        dw_input_nhwc,
        &dw_weight,
        dw_bias.as_ref(),
        env,
        None,
        dw_activation,
        dw_sh,
        dw_sw,
        dw_group,
        dw_pt,
        dw_pl,
        dw_pb,
        dw_pr,
        dw_has_pad,
    )?;
    // `dw_output` is a local owned `Tensor` — NOT inserted into env.
    // That's the whole point of true fusion: zero env-HashMap traffic
    // for the intermediate.

    // Drop borrowed tensors so env can be mutated below.
    drop(dw_input_src);
    drop(dw_weight);
    drop(dw_bias);

    // ---- Early DW-input cleanup ----
    // Matches what the outer dispatch loop does between separate
    // NodeAction::Conv iterations.
    for (inp_idx, inp) in dw_node.inputs.iter().enumerate() {
        if inp.is_empty() {
            continue;
        }
        let id = dw_input_ids
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

    // ---- PW compute ----
    // PW reads DW output directly (not from env) — the whole savings.
    // We still need PW's weight + bias from env.
    let pw_weight = get_tensor(env, &pw_node.name, &pw_node.inputs[1])?.clone();
    let pw_bias = if pw_node.inputs.len() > 2 && !pw_node.inputs[2].is_empty() {
        Some(get_tensor(env, &pw_node.name, &pw_node.inputs[2])?.clone())
    } else {
        None
    };

    let pw_output = conv_compute_nhwc(
        pw_node,
        &dw_output,
        &pw_weight,
        pw_bias.as_ref(),
        env,
        None,
        pw_activation,
        pw_sh,
        pw_sw,
        pw_group,
        pw_pt,
        pw_pl,
        pw_pb,
        pw_pr,
        pw_has_pad,
    )?;

    crate::quantize::calibrate::record_activation(&dw_node.outputs[0], &dw_output);

    drop(pw_weight);
    drop(pw_bias);
    drop(dw_output); // explicit: intermediate memory returns to allocator

    // ---- Bind PW output to env ----
    env.insert(pw_node.outputs[0].clone(), pw_output);
    env.mark_nhwc(&pw_node.outputs[0]);
    Ok(())
}

/// Mirror of `exec_fused_dw_pw` for the inverted-bottleneck opening:
/// chains `PW_expand → DW` through a local intermediate `Tensor` so the
/// PW output never touches the env HashMap. Same early DW-input
/// cleanup (here DW's only input is the PW output, which is owned
/// locally and dropped after DW compute, so the explicit early-cleanup
/// loop touches only the PW's inputs to free the original activation
/// as soon as PW is done with it).
#[allow(clippy::too_many_arguments)]
pub(crate) fn exec_fused_pw_dw(
    pw_node: &OnnxNode,
    dw_node: &OnnxNode,
    env: &mut TensorEnv,
    pw_activation: yscv_kernels::Activation,
    dw_activation: yscv_kernels::Activation,
    pw_params: Option<&crate::loader::ConvParams>,
    dw_params: Option<&crate::loader::ConvParams>,
    pw_input_ids: &[Option<usize>],
    remaining_uses: &mut [usize],
    output_id_mask: &[bool],
    leave_nchwc_output: bool,
) -> Result<(), OnnxError> {
    // ---- PW compute ----
    let pw_input_is_nhwc = env.is_nhwc(&pw_node.inputs[0]);
    let pw_input_is_nchwc16 = !pw_input_is_nhwc && env.nchwc_block(&pw_node.inputs[0]) == Some(16);
    let (pw_sh, pw_sw, pw_group, pw_pt, pw_pl, pw_pb, pw_pr, pw_has_pad) =
        resolve_conv_params(pw_node, pw_params);
    let (
        dw_sh_early,
        dw_sw_early,
        _dw_group_early,
        dw_pt_early,
        dw_pl_early,
        dw_pb_early,
        dw_pr_early,
        _dw_has_pad_early,
    ) = resolve_conv_params(dw_node, dw_params);

    let pw_input_src = get_tensor(env, &pw_node.name, &pw_node.inputs[0])?.clone();
    let pw_weight = get_tensor(env, &pw_node.name, &pw_node.inputs[1])?.clone();
    let pw_bias = if pw_node.inputs.len() > 2 && !pw_node.inputs[2].is_empty() {
        Some(get_tensor(env, &pw_node.name, &pw_node.inputs[2])?.clone())
    } else {
        None
    };
    let pw_input_nhwc_owned;
    let pw_input_nhwc: &Tensor = if pw_input_is_nhwc {
        &pw_input_src
    } else if pw_input_is_nchwc16 {
        // Upstream left NCHWc(16) via whole-graph-NCHWc chaining; convert to
        // NHWC so the streaming fallback works. The NCHWc fast path below uses
        // `pw_input_src` directly via `env.nchwc_block`.
        let c_in_actual = pw_weight.shape()[2];
        pw_input_nhwc_owned = yscv_kernels::nchwc_to_nhwc(&pw_input_src, c_in_actual)?;
        &pw_input_nhwc_owned
    } else {
        pw_input_nhwc_owned = nchw_to_nhwc(&pw_input_src)?;
        &pw_input_nhwc_owned
    };

    // Streaming-fused fast path. When the pair matches the kernel's
    // tight shape contract — PW is a plain 1×1 (stride 1, no pad,
    // group 1), DW is 3×3 SAME-pad (symmetric pad 1, stride 1 or 2,
    // depth_multiplier 1), both activations in {None, Relu} — run the
    // row-buffered streaming kernel that skips the full PW intermediate
    // tensor (~6 MB at tracker's `/xif2_0` shape).
    //
    // Default ON. Output is bitwise-close to the unfused chain (1-ULP
    // FP-ordering drift). Kill switch `YSCV_FUSED_PW_DW_STREAM_OFF=1` forces
    // the fallback compute chain for A/B measurement.
    let stream_disabled = {
        use std::sync::OnceLock;
        static CACHED: OnceLock<bool> = OnceLock::new();
        *CACHED.get_or_init(|| {
            std::env::var_os("YSCV_FUSED_PW_DW_STREAM_OFF").is_some()
                || (cfg!(target_arch = "aarch64")
                    && std::env::var_os("YSCV_FUSED_PW_DW_STREAM_ON").is_none())
        })
    };
    let stream_5x5_disabled = {
        use std::sync::OnceLock;
        static CACHED: OnceLock<bool> = OnceLock::new();
        *CACHED.get_or_init(|| std::env::var_os("YSCV_FUSED_PW_DW_5X5_OFF").is_some())
    };
    let pw_activation_streamable = matches!(
        pw_activation,
        yscv_kernels::Activation::None | yscv_kernels::Activation::Relu
    );
    let dw_activation_streamable = matches!(
        dw_activation,
        yscv_kernels::Activation::None | yscv_kernels::Activation::Relu
    );
    let dw_weight_tensor = get_tensor(env, &dw_node.name, &dw_node.inputs[1])?.clone();
    let pw_is_trivial_1x1 = {
        let ws = pw_weight.shape();
        env.is_khwc_weight(&pw_node.inputs[1])
            && ws.len() == 4
            && ws[0] == 1
            && ws[1] == 1
            && pw_group == 1
            && pw_sh == 1
            && pw_sw == 1
            && !pw_has_pad
    };
    let dw_is_3x3_stride_1_or_2 = {
        let dws = dw_weight_tensor.shape();
        env.is_dw_khwc_weight(&dw_node.inputs[1])
            && dws.len() == 4
            && dws[0] == 3
            && dws[1] == 3
            && dws[3] == 1
            && (dw_sh_early == 1 || dw_sh_early == 2)
            && dw_sh_early == dw_sw_early
            && dw_pt_early == 1
            && dw_pl_early == 1
            && dw_pb_early == 1
            && dw_pr_early == 1
    };
    let dw_is_5x5_stride_1_or_2 = {
        let dws = dw_weight_tensor.shape();
        env.is_dw_khwc_weight(&dw_node.inputs[1])
            && dws.len() == 4
            && dws[0] == 5
            && dws[1] == 5
            && dws[3] == 1
            && (dw_sh_early == 1 || dw_sh_early == 2)
            && dw_sh_early == dw_sw_early
            && dw_pt_early == 2
            && dw_pl_early == 2
            && dw_pb_early == 2
            && dw_pr_early == 2
    };
    // Streaming row-buffer requires enough rows per thread to amortise the
    // ring-buffer startup cost and keep the DW 3-row window hot in L2.
    // Threshold: keep streaming only when (a) spatial is large (>1024 pixels,
    // where the ring saves meaningful L3 traffic) OR (b) each thread gets ≥4
    // rows (ring stays L2-resident; below this, flat GEMM's position-level
    // parallelism wins even when ring fits).
    //
    // At 6T:
    //   xif4 16×16 → 16/6=2 rows/thread, spatial=256 ≤ 1024: flat GEMM
    //   xif3 32×32 → 32/6=5 rows/thread, spatial=1024 ≤ 1024: streaming
    //                (5 rows/thread sufficient; 72 KB ring stays L2-resident)
    //   xif2 64×64 → spatial=4096 > 1024: streaming
    // At 1T:
    //   xif3/xif2: streaming wins (PW weight ≤ L2, avoids large intermediate)
    // Both 3×3 (pad=1) and 5×5 (pad=2) satisfy out = (in-1)/stride+1.
    let stream_spatial_ok = {
        let s = pw_input_nhwc.shape();
        let out_h_est = s[1].saturating_sub(1) / dw_sh_early + 1;
        let out_w_est = s[2].saturating_sub(1) / dw_sw_early + 1;
        let nthreads = rayon::current_num_threads().max(1);
        out_h_est * out_w_est > 1024 || out_h_est / nthreads >= 4
    };
    // ── Whole-graph-NCHWc experiment (YSCV_BACKBONE_NCHWC=1) ──
    // Run PW-expand → DW as separate NCHWc plane kernels. Same trade-off and
    // chaining contract as the FusedPwDwPwReduce path; the next op (typically
    // Conv_Add_fused) consumes the DW output and continues the chain when its
    // own NCHWc path lands.
    let backbone_nchwc = {
        use std::sync::OnceLock;
        static C: OnceLock<bool> = OnceLock::new();
        *C.get_or_init(|| std::env::var_os("YSCV_BACKBONE_NCHWC").is_some())
    };
    let pw_ws_chk = pw_weight.shape();
    let c_exp_chk = if pw_ws_chk.len() == 4 {
        pw_ws_chk[3]
    } else {
        0
    };
    // Per-op NCHWc with cold in+out conversions is ~always a loss vs the
    // strong NHWC streaming kernel. Fire only inside an NCHWc chain: either
    // (a) the upstream op already produced NCHWc input (no input conversion)
    // or (b) we leave NCHWc forward (no output conversion).
    let chain_eligible = pw_input_is_nchwc16 || leave_nchwc_output;
    if backbone_nchwc
        && chain_eligible
        && !crate::quantize::calibrate::calibration_active()
        && pw_is_trivial_1x1
        && dw_is_3x3_stride_1_or_2
        && dw_sh_early == 1
        && pw_activation_streamable
        && dw_activation_streamable
        && c_exp_chk.is_multiple_of(16)
        && pw_input_nhwc.shape()[0] == 1
    {
        let in_shape = pw_input_nhwc.shape();
        let c_in_actual = in_shape[3];
        let c_exp = pw_ws_chk[3];
        let dw_bias_tensor = if dw_node.inputs.len() > 2 && !dw_node.inputs[2].is_empty() {
            Some(get_tensor(env, &dw_node.name, &dw_node.inputs[2])?.clone())
        } else {
            None
        };
        if let Some(dw_out_tensor) = try_fused_pw_dw_nchwc(
            pw_input_nhwc,
            &pw_input_src,
            &pw_node.inputs[0],
            &pw_weight,
            pw_bias.as_ref(),
            &dw_weight_tensor,
            dw_bias_tensor.as_ref(),
            env,
            c_in_actual,
            c_exp,
            pw_activation,
            dw_activation,
            leave_nchwc_output,
        ) {
            drop(pw_input_src);
            drop(pw_weight);
            drop(pw_bias);
            drop(dw_weight_tensor);
            drop(dw_bias_tensor);
            // Decrement PW inputs (parity with streaming-path cleanup below).
            for (inp_idx, inp) in pw_node.inputs.iter().enumerate() {
                if inp.is_empty() {
                    continue;
                }
                let id = pw_input_ids
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
            env.insert(dw_node.outputs[0].clone(), dw_out_tensor);
            if leave_nchwc_output {
                env.mark_nchwc(&dw_node.outputs[0], 16);
            } else {
                env.mark_nhwc(&dw_node.outputs[0]);
            }
            return Ok(());
        }
        // NCHWc bailed: fall through to streaming/fallback below.
    }

    if !stream_disabled
        && !crate::quantize::calibrate::calibration_active()
        && pw_is_trivial_1x1
        && (dw_is_3x3_stride_1_or_2 || dw_is_5x5_stride_1_or_2)
        && (!dw_is_5x5_stride_1_or_2 || !stream_5x5_disabled)
        && pw_activation_streamable
        && dw_activation_streamable
        && stream_spatial_ok
    {
        let in_shape = pw_input_nhwc.shape();
        let (batch, in_h, in_w, c_in) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
        let pw_ws = pw_weight.shape();
        let c_exp = pw_ws[3];
        let (dw_k, dw_pad) = if dw_is_5x5_stride_1_or_2 {
            (5usize, 2usize)
        } else {
            (3usize, 1usize)
        };
        let dw_bias_tensor = if dw_node.inputs.len() > 2 && !dw_node.inputs[2].is_empty() {
            Some(get_tensor(env, &dw_node.name, &dw_node.inputs[2])?.clone())
        } else {
            None
        };
        let out_h = (in_h + 2 * dw_pad - dw_k) / dw_sh_early + 1;
        let out_w = (in_w + 2 * dw_pad - dw_k) / dw_sw_early + 1;
        let out_len = batch * out_h * out_w * c_exp;
        #[allow(unsafe_code)]
        let mut out = yscv_tensor::AlignedVec::<f32>::uninitialized(out_len);
        if dw_k == 5 {
            yscv_kernels::fused_pw_expand_dw_5x5(yscv_kernels::FusedPwDw5x5 {
                input: pw_input_nhwc.data(),
                pw_weight: pw_weight.data(),
                pw_bias: pw_bias.as_ref().map(|b| b.data()),
                dw_weight: dw_weight_tensor.data(),
                dw_bias: dw_bias_tensor.as_ref().map(|b| b.data()),
                output: out.as_mut_slice(),
                batch,
                in_h,
                in_w,
                c_in,
                c_exp,
                stride: dw_sh_early,
                pw_activation,
                dw_activation,
                thread_pool: None, // rayon ambient scope picks up current pool
            });
        } else {
            yscv_kernels::fused_pw_expand_dw_3x3(
                pw_input_nhwc.data(),
                pw_weight.data(),
                pw_bias.as_ref().map(|b| b.data()),
                dw_weight_tensor.data(),
                dw_bias_tensor.as_ref().map(|b| b.data()),
                out.as_mut_slice(),
                batch,
                in_h,
                in_w,
                c_in,
                c_exp,
                dw_sh_early,
                pw_activation,
                dw_activation,
                None, // rayon ambient scope picks up current pool
            );
        }
        drop(pw_input_src);
        drop(pw_weight);
        drop(pw_bias);
        drop(dw_weight_tensor);
        drop(dw_bias_tensor);

        // Early cleanup: PW inputs done.
        for (inp_idx, inp) in pw_node.inputs.iter().enumerate() {
            if inp.is_empty() {
                continue;
            }
            let id = pw_input_ids
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

        let dw_out_tensor =
            Tensor::from_aligned(vec![batch, out_h, out_w, c_exp], out).map_err(|e| {
                OnnxError::DecodeFailed {
                    message: e.to_string(),
                }
            })?;
        env.insert(dw_node.outputs[0].clone(), dw_out_tensor);
        env.mark_nhwc(&dw_node.outputs[0]);
        return Ok(());
    }

    // Not streamable — fall through to the compute/bind chaining.
    drop(dw_weight_tensor);
    let pw_output = conv_compute_nhwc(
        pw_node,
        pw_input_nhwc,
        &pw_weight,
        pw_bias.as_ref(),
        env,
        None,
        pw_activation,
        pw_sh,
        pw_sw,
        pw_group,
        pw_pt,
        pw_pl,
        pw_pb,
        pw_pr,
        pw_has_pad,
    )?;

    drop(pw_input_src);
    drop(pw_weight);
    drop(pw_bias);

    // Early cleanup: PW's inputs are done. Without this, the original
    // activation tensor stays alive across the DW call and adds L1/L2
    // pressure. Matches the `exec_fused_dw_pw` pattern.
    for (inp_idx, inp) in pw_node.inputs.iter().enumerate() {
        if inp.is_empty() {
            continue;
        }
        let id = pw_input_ids
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

    // ---- DW compute ----
    // DW reads `pw_output` (local Tensor) directly.
    let dw_weight = get_tensor(env, &dw_node.name, &dw_node.inputs[1])?.clone();
    let dw_bias = if dw_node.inputs.len() > 2 && !dw_node.inputs[2].is_empty() {
        Some(get_tensor(env, &dw_node.name, &dw_node.inputs[2])?.clone())
    } else {
        None
    };
    let (dw_sh, dw_sw, dw_group, dw_pt, dw_pl, dw_pb, dw_pr, dw_has_pad) =
        resolve_conv_params(dw_node, dw_params);

    let dw_output = conv_compute_nhwc(
        dw_node,
        &pw_output,
        &dw_weight,
        dw_bias.as_ref(),
        env,
        None,
        dw_activation,
        dw_sh,
        dw_sw,
        dw_group,
        dw_pt,
        dw_pl,
        dw_pb,
        dw_pr,
        dw_has_pad,
    )?;

    crate::quantize::calibrate::record_activation(&pw_node.outputs[0], &pw_output);

    drop(dw_weight);
    drop(dw_bias);
    drop(pw_output);

    env.insert(dw_node.outputs[0].clone(), dw_output);
    env.mark_nhwc(&dw_node.outputs[0]);
    Ok(())
}

/// Whole-graph-NCHWc experiment: run a PW-expand → DW3×3(s1) → PW-reduce MBConv
/// block as three NCHWc plane kernels. Returns `Some(tensor)` on success (the
/// tensor is NCHWc when `leave_nchwc` else NHWC) or `None` to fall back to the
/// NHWC streaming kernel.
///
/// Chaining: input is consumed from `env` directly as NCHWc when
/// `env.nchwc_block` reports it (no NHWC→NCHWc conversion); output is left
/// NCHWc when `leave_nchwc` is true (next op will consume it). This is the key
/// step that turns the per-op layout switch into a zero-conversion region.
#[allow(clippy::too_many_arguments)]
fn try_fused_pw_dw_pw_reduce_nchwc(
    pw_input_nhwc: &Tensor,
    pw_input_src: &Tensor,
    pw_input_name: &str,
    pw_expand_weight: &Tensor,
    pw_expand_bias: Option<&Tensor>,
    dw_weight_tensor: &Tensor,
    dw_bias_tensor: Option<&Tensor>,
    pw_reduce_node: &OnnxNode,
    env: &TensorEnv,
    residual_slice: Option<&[f32]>,
    residual_nchwc: Option<&Tensor>,
    c_in: usize,
    c_exp: usize,
    c_out: usize,
    out_h: usize,
    out_w: usize,
    pw_expand_activation: yscv_kernels::Activation,
    dw_activation: yscv_kernels::Activation,
    pw_reduce_activation: yscv_kernels::Activation,
    leave_nchwc: bool,
) -> Option<Tensor> {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::sync::Arc;
    // Lazy NCHWc DW-weight pack, cached by weight pointer (packed once, reused
    // across all inferences — the bench timing loop never re-packs).
    thread_local! {
        static DW_PACK: RefCell<HashMap<usize, Arc<yscv_kernels::PackedNChwBc>>> =
            RefCell::new(HashMap::new());
    }
    let cfg = yscv_kernels::ParallelElementwiseConfig::default();
    let reduce_w = env.get(&pw_reduce_node.inputs[1])?.clone();
    let reduce_bias = if pw_reduce_node.inputs.len() > 2 && !pw_reduce_node.inputs[2].is_empty() {
        env.get(&pw_reduce_node.inputs[2]).cloned()
    } else {
        None
    };

    let dw_key = dw_weight_tensor.data().as_ptr() as usize;
    let packed_dw = DW_PACK.with(|c| {
        c.borrow_mut()
            .entry(dw_key)
            .or_insert_with(|| {
                yscv_kernels::pack_dw_nchwc_for_session(dw_weight_tensor.data(), c_exp, 3, 3, 16)
            })
            .clone()
    });

    // INPUT CHAINING: when the upstream op left this tensor as NCHWc(16),
    // consume it directly — no NHWC→NCHWc conversion.
    let input_nchwc_owned;
    let input_nchwc: &Tensor = if env.nchwc_block(pw_input_name) == Some(16) {
        pw_input_src
    } else {
        input_nchwc_owned = yscv_kernels::nhwc_to_nchwc(pw_input_nhwc, 16).ok()?;
        &input_nchwc_owned
    };
    let expanded = yscv_kernels::conv2d_nchwc_pointwise_with_activation_prepacked(
        input_nchwc,
        pw_expand_weight,
        pw_expand_bias,
        c_in,
        pw_expand_activation,
        cfg,
        None,
        None,
    )
    .ok()?;
    let dw_out = yscv_kernels::conv2d_nchwc_dw3x3_s1_same_pad(
        &expanded,
        dw_weight_tensor,
        dw_bias_tensor,
        dw_activation,
        c_exp,
        cfg,
        None,
        Some(&packed_dw),
    )
    .ok()?;
    let res_nchwc_owned;
    let reduced = if let Some(res_t) = residual_nchwc {
        // Residual is already NCHWc(16) upstream — use directly, zero conversion.
        yscv_kernels::conv2d_nchwc_pointwise_with_residual_activation_prepacked(
            &dw_out,
            &reduce_w,
            reduce_bias.as_ref(),
            res_t,
            c_exp,
            pw_reduce_activation,
            cfg,
            None,
            None,
        )
    } else if let Some(res) = residual_slice {
        // Residual is NHWC `[out_h*out_w*c_out]`; convert to NCHWc once.
        let res_t = Tensor::from_vec(vec![1, out_h, out_w, c_out], res.to_vec()).ok()?;
        res_nchwc_owned = yscv_kernels::nhwc_to_nchwc(&res_t, 16).ok()?;
        yscv_kernels::conv2d_nchwc_pointwise_with_residual_activation_prepacked(
            &dw_out,
            &reduce_w,
            reduce_bias.as_ref(),
            &res_nchwc_owned,
            c_exp,
            pw_reduce_activation,
            cfg,
            None,
            None,
        )
    } else {
        yscv_kernels::conv2d_nchwc_pointwise_with_activation_prepacked(
            &dw_out,
            &reduce_w,
            reduce_bias.as_ref(),
            c_exp,
            pw_reduce_activation,
            cfg,
            None,
            None,
        )
    }
    .ok()?;
    // OUTPUT CHAINING: leave NCHWc when the next op will consume it; else
    // convert back to NHWC for the rest of the graph.
    if leave_nchwc {
        Some(reduced)
    } else {
        yscv_kernels::nchwc_to_nhwc(&reduced, c_out).ok()
    }
}

/// Whole-graph-NCHWc helper for the `FusedPwDw` (PW-expand → DW3×3, no reduce)
/// pattern. Returns `Some(tensor)` on success (NCHWc when `leave_nchwc`, else
/// NHWC) or `None` to fall back to NHWC streaming. Mirrors
/// [`try_fused_pw_dw_pw_reduce_nchwc`] but without the PW-reduce step — the
/// downstream Conv_Add_fused consumes the DW output as the new PW-reduce input.
#[allow(clippy::too_many_arguments)]
fn try_fused_pw_dw_nchwc(
    pw_input_nhwc: &Tensor,
    pw_input_src: &Tensor,
    pw_input_name: &str,
    pw_weight: &Tensor,
    pw_bias: Option<&Tensor>,
    dw_weight_tensor: &Tensor,
    dw_bias_tensor: Option<&Tensor>,
    env: &TensorEnv,
    c_in: usize,
    c_exp: usize,
    pw_activation: yscv_kernels::Activation,
    dw_activation: yscv_kernels::Activation,
    leave_nchwc: bool,
) -> Option<Tensor> {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::sync::Arc;
    thread_local! {
        static DW_PACK: RefCell<HashMap<usize, Arc<yscv_kernels::PackedNChwBc>>> =
            RefCell::new(HashMap::new());
    }
    let cfg = yscv_kernels::ParallelElementwiseConfig::default();
    let dw_key = dw_weight_tensor.data().as_ptr() as usize;
    let packed_dw = DW_PACK.with(|c| {
        c.borrow_mut()
            .entry(dw_key)
            .or_insert_with(|| {
                yscv_kernels::pack_dw_nchwc_for_session(dw_weight_tensor.data(), c_exp, 3, 3, 16)
            })
            .clone()
    });

    let input_nchwc_owned;
    let input_nchwc: &Tensor = if env.nchwc_block(pw_input_name) == Some(16) {
        pw_input_src
    } else {
        input_nchwc_owned = yscv_kernels::nhwc_to_nchwc(pw_input_nhwc, 16).ok()?;
        &input_nchwc_owned
    };
    let expanded = yscv_kernels::conv2d_nchwc_pointwise_with_activation_prepacked(
        input_nchwc,
        pw_weight,
        pw_bias,
        c_in,
        pw_activation,
        cfg,
        None,
        None,
    )
    .ok()?;
    let dw_out = yscv_kernels::conv2d_nchwc_dw3x3_s1_same_pad(
        &expanded,
        dw_weight_tensor,
        dw_bias_tensor,
        dw_activation,
        c_exp,
        cfg,
        None,
        Some(&packed_dw),
    )
    .ok()?;
    if leave_nchwc {
        Some(dw_out)
    } else {
        yscv_kernels::nchwc_to_nhwc(&dw_out, c_exp).ok()
    }
}

/// Streaming MobileNet-V2 inverted bottleneck:
/// `PW_expand → DW 3×3 → PW_reduce` executed as one kernel call. The c_exp
/// intermediate stays L1-resident (one DW output row at a time, ≤ 24 KB).
/// Falls through to the chained PW+DW + Conv path when the input shape /
/// weight layout doesn't match the kernel's contract.
#[allow(clippy::too_many_arguments)]
pub(crate) fn exec_fused_pw_dw_pw_reduce(
    pw_expand_node: &OnnxNode,
    dw_node: &OnnxNode,
    pw_reduce_node: &OnnxNode,
    env: &mut TensorEnv,
    pw_expand_activation: yscv_kernels::Activation,
    dw_activation: yscv_kernels::Activation,
    pw_reduce_activation: yscv_kernels::Activation,
    pw_expand_params: Option<&crate::loader::ConvParams>,
    dw_params: Option<&crate::loader::ConvParams>,
    pw_expand_input_ids: &[Option<usize>],
    prepacked_pw_reduce: &crate::loader::FusedPwDwPwReduceWeights,
    residual_meta: Option<(&OnnxNode, u8, u8, Option<&OnnxNode>)>,
    dw_kernel_size: u8,
    remaining_uses: &mut [usize],
    output_id_mask: &[bool],
    leave_nchwc_output: bool,
) -> Result<(), OnnxError> {
    let pw_input_is_nhwc = env.is_nhwc(&pw_expand_node.inputs[0]);
    let pw_input_is_nchwc16 =
        !pw_input_is_nhwc && env.nchwc_block(&pw_expand_node.inputs[0]) == Some(16);
    let (pw_sh, pw_sw, pw_group, _pw_pt, _pw_pl, _pw_pb, _pw_pr, pw_has_pad) =
        resolve_conv_params(pw_expand_node, pw_expand_params);
    let (dw_sh, dw_sw, _dw_group, dw_pt, dw_pl, dw_pb, dw_pr, _dw_has_pad) =
        resolve_conv_params(dw_node, dw_params);

    let pw_input_src = get_tensor(env, &pw_expand_node.name, &pw_expand_node.inputs[0])?.clone();
    let pw_expand_weight =
        get_tensor(env, &pw_expand_node.name, &pw_expand_node.inputs[1])?.clone();
    let pw_expand_bias = if pw_expand_node.inputs.len() > 2 && !pw_expand_node.inputs[2].is_empty()
    {
        Some(get_tensor(env, &pw_expand_node.name, &pw_expand_node.inputs[2])?.clone())
    } else {
        None
    };
    let pw_input_nhwc_owned;
    let pw_input_nhwc: &Tensor = if pw_input_is_nhwc {
        &pw_input_src
    } else if pw_input_is_nchwc16 {
        // Upstream left NCHWc(16) via whole-graph-NCHWc chaining. Convert
        // here so the streaming fallback can run if the NCHWc helper bails.
        // The NCHWc helper itself uses `pw_input_src` directly via
        // `env.nchwc_block`, so the convert is wasted on chain-hit paths —
        // a tried-and-reverted lazy variant introduced p50 jitter without a
        // net min improvement, so keep eager for predictable latency.
        let c_in_actual = pw_expand_weight.shape()[2];
        pw_input_nhwc_owned = yscv_kernels::nchwc_to_nhwc(&pw_input_src, c_in_actual)?;
        &pw_input_nhwc_owned
    } else {
        pw_input_nhwc_owned = nchw_to_nhwc(&pw_input_src)?;
        &pw_input_nhwc_owned
    };

    let dw_weight_tensor = get_tensor(env, &dw_node.name, &dw_node.inputs[1])?.clone();
    let dw_bias_tensor = if dw_node.inputs.len() > 2 && !dw_node.inputs[2].is_empty() {
        Some(get_tensor(env, &dw_node.name, &dw_node.inputs[2])?.clone())
    } else {
        None
    };

    // Full-block PW-expand → DW → PW-reduce fusion keeps the expanded
    // intermediate in cache instead of spilling it to DRAM between ops — a win
    // on both x86 and the in-order aarch64 cores. (The plainer DW→PW / PW→DW
    // streaming fusions stay aarch64-gated; they regress the small-L2 A53.)
    let stream_disabled = std::env::var_os("YSCV_FUSED_PW_DW_PW_REDUCE_OFF").is_some();
    let pw_expand_ok = env.is_khwc_weight(&pw_expand_node.inputs[1])
        && pw_expand_weight.shape().len() == 4
        && pw_expand_weight.shape()[0] == 1
        && pw_expand_weight.shape()[1] == 1
        && pw_group == 1
        && pw_sh == 1
        && pw_sw == 1
        && !pw_has_pad;
    let dw_ws = dw_weight_tensor.shape();
    let dw_ok = env.is_dw_khwc_weight(&dw_node.inputs[1])
        && dw_ws.len() == 4
        && dw_ws[3] == 1
        && (dw_sh == 1 || dw_sh == 2)
        && dw_sh == dw_sw
        && match dw_kernel_size {
            3 => {
                dw_ws[0] == 3
                    && dw_ws[1] == 3
                    && dw_pt == 1
                    && dw_pl == 1
                    && dw_pb == 1
                    && dw_pr == 1
            }
            5 => {
                dw_ws[0] == 5
                    && dw_ws[1] == 5
                    && dw_pt == 2
                    && dw_pl == 2
                    && dw_pb == 2
                    && dw_pr == 2
            }
            _ => false,
        };
    let acts_ok = matches!(
        pw_expand_activation,
        yscv_kernels::Activation::None | yscv_kernels::Activation::Relu
    ) && matches!(
        dw_activation,
        yscv_kernels::Activation::None | yscv_kernels::Activation::Relu
    ) && matches!(
        pw_reduce_activation,
        yscv_kernels::Activation::None | yscv_kernels::Activation::Relu
    );
    if stream_disabled || !pw_expand_ok || !dw_ok || !acts_ok {
        // Fallback: degrade to the existing FusedPwDw path then run the
        // PW reduce as a normal Conv (+optional Add+Relu when residual).
        let _ = dw_kernel_size; // fallback handles both via FusedPwDw inside exec_fused_pw_dw
        return exec_fused_pw_dw_then_reduce_fallback(
            pw_expand_node,
            dw_node,
            pw_reduce_node,
            env,
            pw_expand_activation,
            dw_activation,
            pw_reduce_activation,
            pw_expand_params,
            dw_params,
            pw_expand_input_ids,
            residual_meta,
            remaining_uses,
            output_id_mask,
        );
    }

    let in_shape = pw_input_nhwc.shape();
    let (batch, in_h, in_w, c_in) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
    let c_exp = pw_expand_weight.shape()[3];
    debug_assert_eq!(c_exp, prepacked_pw_reduce.c_exp);
    let c_out = prepacked_pw_reduce.c_out;
    let c_out_padded = prepacked_pw_reduce.c_out_padded;
    let (dw_pad, dw_k) = if dw_kernel_size == 3 {
        (1usize, 3usize)
    } else {
        (2, 5)
    };
    let out_h = (in_h + 2 * dw_pad - dw_k) / dw_sh + 1;
    let out_w = (in_w + 2 * dw_pad - dw_k) / dw_sw + 1;
    let out_len = batch * out_h * out_w * c_out;

    // Resolve residual tensor (if Add fusion): the streaming kernel folds
    // the Add inline via the `residual` arg, eliminating the separate
    // ConvAdd dispatch. The optional post-Relu is applied via the kernel's
    // `pw_reduce_activation` parameter — but only when both PW reduce's
    // *own* activation is None (the normal residual block case) and there
    // is a Relu after the Add. We re-resolve that here from residual_meta.
    let residual_tensor: Option<Tensor> = if let Some((add_node, skip_input, _, _)) = residual_meta
    {
        let skip_name = &add_node.inputs[skip_input as usize];
        Some(get_tensor(env, &add_node.name, skip_name)?.clone())
    } else {
        None
    };
    // Residual handling: prefer the upstream layout to avoid round-trip
    // conversions. The NCHWc helper consumes `residual_nchwc_ref` directly
    // when available (zero conversion). The NHWC streaming fallback path
    // needs an NHWC `&[f32]`; we build it eagerly only when the upstream is
    // already NHWC (free). For NCHWc/NCHW residuals we defer the conversion
    // until the fallback actually runs, avoiding a wasted convert on the
    // common chained path.
    let res_name_opt = residual_meta.as_ref().map(|m| &m.0.inputs[m.1 as usize]);
    let res_is_nhwc = res_name_opt.is_some_and(|n| env.is_nhwc(n));
    let res_is_nchwc16 =
        res_name_opt.is_some_and(|n| !env.is_nhwc(n) && env.nchwc_block(n) == Some(16));
    let residual_nchwc_ref: Option<&Tensor> = if res_is_nchwc16 {
        residual_tensor.as_ref()
    } else {
        None
    };
    let residual_slice_eager: Option<&[f32]> = if res_is_nhwc {
        residual_tensor.as_ref().map(|t| t.data())
    } else {
        None
    };
    // When there's a residual Add with a trailing Relu, the *effective*
    // pw_reduce_activation is Relu (the streaming kernel applies it to
    // `acc + residual + bias`). Otherwise use the bare pw_reduce_activation.
    let effective_pw_reduce_act = match residual_meta {
        Some((_, _, 1, _)) => yscv_kernels::Activation::Relu,
        _ => pw_reduce_activation,
    };

    // ── Whole-graph-NCHWc experiment (YSCV_BACKBONE_NCHWC=1) ──
    // Run PW-expand → DW → PW-reduce as separate NCHWc plane kernels (ORT-style)
    // instead of the NHWC streaming fusion. Plane NCHWc DW reuses input rows
    // across output rows (~3× faster than the streaming row DW). Trades the
    // no-materialization benefit (intermediates L2-resident at tracker spatial
    // sizes) for fast DW. Chaining via `leave_nchwc_output` removes per-op
    // conversions for back-to-back NCHWc nodes.
    let backbone_nchwc = {
        use std::sync::OnceLock;
        static C: OnceLock<bool> = OnceLock::new();
        *C.get_or_init(|| std::env::var_os("YSCV_BACKBONE_NCHWC").is_some())
    };
    // The NCHWc helper accepts residual in either NCHWc (preferred, zero
    // conversion) or NHWC form. If the residual exists in NCHW (rare in this
    // path — only at graph boundaries), force the NHWC streaming fallback.
    let residual_layout_supported_by_nchwc =
        residual_tensor.is_none() || res_is_nhwc || res_is_nchwc16;
    let nchwc_eligible = backbone_nchwc
        && dw_kernel_size == 3
        && dw_sh == 1
        && c_exp.is_multiple_of(16)
        && batch == 1
        && residual_layout_supported_by_nchwc;
    let nchwc_out: Option<Tensor> = if nchwc_eligible {
        try_fused_pw_dw_pw_reduce_nchwc(
            pw_input_nhwc,
            &pw_input_src,
            &pw_expand_node.inputs[0],
            &pw_expand_weight,
            pw_expand_bias.as_ref(),
            &dw_weight_tensor,
            dw_bias_tensor.as_ref(),
            pw_reduce_node,
            env,
            residual_slice_eager,
            residual_nchwc_ref,
            c_in,
            c_exp,
            c_out,
            out_h,
            out_w,
            pw_expand_activation,
            dw_activation,
            effective_pw_reduce_act,
            leave_nchwc_output,
        )
    } else {
        None
    };
    // Build the NHWC residual slice that the streaming fallback needs. Only
    // reached when the NCHWc path was ineligible or returned None — keeps the
    // chained NCHWc-residual path conversion-free.
    let residual_nhwc_owned;
    let residual_slice: Option<&[f32]> = if nchwc_out.is_some() {
        None
    } else if let Some(slice) = residual_slice_eager {
        Some(slice)
    } else if let Some(res_t) = residual_tensor.as_ref() {
        let converted = if res_is_nchwc16 {
            yscv_kernels::nchwc_to_nhwc(res_t, c_out)?
        } else {
            nchw_to_nhwc(res_t)?
        };
        residual_nhwc_owned = converted;
        Some(residual_nhwc_owned.data())
    } else {
        None
    };
    let (pw_reduce_out, output_is_nchwc) = if let Some(t) = nchwc_out {
        (t, leave_nchwc_output)
    } else {
        #[allow(unsafe_code)]
        let mut out = yscv_tensor::AlignedVec::<f32>::uninitialized(out_len);
        let kernel_args = yscv_kernels::FusedPwDwPwReduce {
            input: pw_input_nhwc.data(),
            pw_expand_weight: pw_expand_weight.data(),
            pw_expand_bias: pw_expand_bias.as_ref().map(|b| b.data()),
            dw_weight: dw_weight_tensor.data(),
            dw_bias: dw_bias_tensor.as_ref().map(|b| b.data()),
            pw_reduce_weight_packed: prepacked_pw_reduce.weight_packed.as_slice(),
            pw_reduce_bias: prepacked_pw_reduce
                .bias_padded
                .as_ref()
                .map(|b| b.as_slice()),
            residual: residual_slice,
            output: out.as_mut_slice(),
            batch,
            in_h,
            in_w,
            c_in,
            c_exp,
            c_out,
            c_out_padded,
            stride: dw_sh,
            pw_expand_activation,
            dw_activation,
            pw_reduce_activation: effective_pw_reduce_act,
            thread_pool: None,
        };
        // Experimental NCHWc-streaming variant: drop-in for the 3×3 stride-1
        // case (no residual), gated by `YSCV_NCHWC_STREAM=1`, default OFF.
        // Same NHWC input/output as the existing kernel; the internal c_exp
        // ring is held in NCHWc-blocked layout `[3, c_blocks, w, 16]`.
        // Microbench-faster but tracker-neutral, so kept opt-in.
        #[cfg(target_arch = "x86_64")]
        let nchwc_stream_enabled = {
            use std::sync::OnceLock;
            static C: OnceLock<bool> = OnceLock::new();
            *C.get_or_init(|| std::env::var_os("YSCV_NCHWC_STREAM").is_some())
        };
        #[cfg(target_arch = "x86_64")]
        let use_nchwc_stream = nchwc_stream_enabled
            && dw_kernel_size == 3
            && dw_sh == 1
            && c_exp.is_multiple_of(16)
            && batch == 1
            && residual_slice.is_none()
            && std::is_x86_feature_detected!("avx512f")
            && matches!(
                pw_expand_activation,
                yscv_kernels::Activation::None | yscv_kernels::Activation::Relu
            )
            && matches!(
                dw_activation,
                yscv_kernels::Activation::None | yscv_kernels::Activation::Relu
            )
            && matches!(
                effective_pw_reduce_act,
                yscv_kernels::Activation::None | yscv_kernels::Activation::Relu
            );
        #[cfg(not(target_arch = "x86_64"))]
        let use_nchwc_stream = false;
        if use_nchwc_stream {
            // x86_64-only NCHWc streaming path. `use_nchwc_stream` is a literal
            // `false` on other arches (set above), so this branch is dead there
            // — but the body still has to compile, and the NCHWc-streaming
            // kernel + its DW-pack helper are x86_64-only exports. Gate the
            // whole body; the non-x86 fall-through is unreachable.
            #[cfg(target_arch = "x86_64")]
            {
                // Lazy DW-weight pack into NCHWc-blocked layout.
                use std::cell::RefCell;
                use std::collections::HashMap;
                use std::sync::Arc;
                thread_local! {
                    static DW_BLOCK_PACK: RefCell<HashMap<usize, Arc<Vec<f32>>>> =
                        RefCell::new(HashMap::new());
                }
                let dw_key = dw_weight_tensor.data().as_ptr() as usize;
                let dw_packed = DW_BLOCK_PACK.with(|c| {
                    c.borrow_mut()
                        .entry(dw_key)
                        .or_insert_with(|| {
                            Arc::new(yscv_kernels::pack_dw_weight_nchwc_blocked(
                                dw_weight_tensor.data(),
                                c_exp,
                            ))
                        })
                        .clone()
                });
                yscv_kernels::fused_pw_expand_dw_pw_reduce_3x3_nchwc_streaming(
                    pw_input_nhwc.data(),
                    pw_expand_weight.data(),
                    pw_expand_bias.as_ref().map(|b| b.data()),
                    dw_packed.as_slice(),
                    dw_bias_tensor.as_ref().map(|b| b.data()),
                    prepacked_pw_reduce.weight_packed.as_slice(),
                    prepacked_pw_reduce
                        .bias_padded
                        .as_ref()
                        .map(|b| b.as_slice()),
                    out.as_mut_slice(),
                    in_h,
                    in_w,
                    c_in,
                    c_exp,
                    c_out,
                    c_out_padded,
                    matches!(pw_expand_activation, yscv_kernels::Activation::Relu),
                    matches!(dw_activation, yscv_kernels::Activation::Relu),
                    matches!(effective_pw_reduce_act, yscv_kernels::Activation::Relu),
                );
            }
            #[cfg(not(target_arch = "x86_64"))]
            unreachable!("use_nchwc_stream is always false on non-x86_64");
        } else if dw_kernel_size == 3 {
            yscv_kernels::fused_pw_expand_dw_pw_reduce_3x3(kernel_args);
        } else {
            yscv_kernels::fused_pw_expand_dw_pw_reduce_5x5(kernel_args);
        }
        let t = Tensor::from_aligned(vec![batch, out_h, out_w, c_out], out).map_err(|e| {
            OnnxError::DecodeFailed {
                message: e.to_string(),
            }
        })?;
        (t, false)
    };

    drop(pw_input_src);
    drop(pw_expand_weight);
    drop(pw_expand_bias);
    drop(dw_weight_tensor);
    drop(dw_bias_tensor);
    drop(residual_tensor);

    // Decrement PW expand inputs (input activation + weight + optional bias).
    for (inp_idx, inp) in pw_expand_node.inputs.iter().enumerate() {
        if inp.is_empty() {
            continue;
        }
        let id = pw_expand_input_ids
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

    // Insert under the LAST consumer's output name: if residual fold ran,
    // the streaming kernel produced `add` (or `relu(add)`) output, so the
    // tensor key is that node's output name, not PW reduce's.
    let final_out_name = match residual_meta {
        Some((_, _, 1, Some(relu_node))) => relu_node.outputs[0].clone(),
        Some((add_node, _, _, _)) => add_node.outputs[0].clone(),
        None => pw_reduce_node.outputs[0].clone(),
    };
    env.insert(final_out_name.clone(), pw_reduce_out);
    if output_is_nchwc {
        env.mark_nchwc(&final_out_name, 16);
    } else {
        env.mark_nhwc(&final_out_name);
    }
    Ok(())
}

/// Fallback path when the streaming kernel can't run (shape contract miss /
/// kill switch ON / non-streamable activation). Executes the original
/// FusedPwDw + standalone Conv1x1 (+ optional Add + Relu) sequence.
#[allow(clippy::too_many_arguments)]
pub(crate) fn exec_fused_pw_dw_then_reduce_fallback(
    pw_expand_node: &OnnxNode,
    dw_node: &OnnxNode,
    pw_reduce_node: &OnnxNode,
    env: &mut TensorEnv,
    pw_expand_activation: yscv_kernels::Activation,
    dw_activation: yscv_kernels::Activation,
    pw_reduce_activation: yscv_kernels::Activation,
    pw_expand_params: Option<&crate::loader::ConvParams>,
    dw_params: Option<&crate::loader::ConvParams>,
    pw_expand_input_ids: &[Option<usize>],
    residual_meta: Option<(&OnnxNode, u8, u8, Option<&OnnxNode>)>,
    remaining_uses: &mut [usize],
    output_id_mask: &[bool],
) -> Result<(), OnnxError> {
    exec_fused_pw_dw(
        pw_expand_node,
        dw_node,
        env,
        pw_expand_activation,
        dw_activation,
        pw_expand_params,
        dw_params,
        pw_expand_input_ids,
        remaining_uses,
        output_id_mask,
        false,
    )?;
    exec_conv(pw_reduce_node, env, pw_reduce_activation)?;
    // Replay the Add (+ optional Relu) that the fusion had absorbed.
    if let Some((add_node, _, post_act, relu_opt)) = residual_meta {
        crate::runner::elementwise::exec_add(add_node, env)?;
        if post_act == 1
            && let Some(relu_node) = relu_opt
        {
            crate::runner::elementwise::exec_relu(relu_node, env)?;
        }
    }
    Ok(())
}
