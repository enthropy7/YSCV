use super::*;
use std::sync::OnceLock;

type DotFn = fn(&[f32], &[f32]) -> f32;

#[inline]
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

#[inline]
fn dot_dispatch() -> DotFn {
    static DOT_FN: OnceLock<DotFn> = OnceLock::new();
    *DOT_FN.get_or_init(|| {
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return dot_neon;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma") {
                return dot_avx;
            }
            if std::is_x86_feature_detected!("sse") {
                return dot_sse;
            }
        }
        dot_scalar
    })
}

#[inline]
fn apply_conv_activation(
    tensor: &mut Tensor,
    activation: yscv_kernels::Activation,
    activation_fused: bool,
) {
    if activation_fused {
        return;
    }
    match activation {
        yscv_kernels::Activation::Silu => yscv_kernels::silu_inplace(tensor),
        yscv_kernels::Activation::Relu => yscv_kernels::relu_inplace(tensor),
        yscv_kernels::Activation::None => {}
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn dot_neon_impl(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = a.len();
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut i = 0usize;
    while i + 8 <= n {
        let va0 = vld1q_f32(ap.add(i));
        let vb0 = vld1q_f32(bp.add(i));
        acc0 = vfmaq_f32(acc0, va0, vb0);
        let va1 = vld1q_f32(ap.add(i + 4));
        let vb1 = vld1q_f32(bp.add(i + 4));
        acc1 = vfmaq_f32(acc1, va1, vb1);
        i += 8;
    }
    while i + 4 <= n {
        let va = vld1q_f32(ap.add(i));
        let vb = vld1q_f32(bp.add(i));
        acc0 = vfmaq_f32(acc0, va, vb);
        i += 4;
    }
    acc0 = vaddq_f32(acc0, acc1);
    let mut sum = vaddvq_f32(acc0);
    while i < n {
        sum += *ap.add(i) * *bp.add(i);
        i += 1;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    #[allow(unsafe_code)]
    unsafe {
        dot_neon_impl(a, b)
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx", enable = "fma")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn dot_avx_impl(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let n = a.len();
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 16 <= n {
        let va0 = _mm256_loadu_ps(ap.add(i));
        let vb0 = _mm256_loadu_ps(bp.add(i));
        acc0 = _mm256_fmadd_ps(va0, vb0, acc0);
        let va1 = _mm256_loadu_ps(ap.add(i + 8));
        let vb1 = _mm256_loadu_ps(bp.add(i + 8));
        acc1 = _mm256_fmadd_ps(va1, vb1, acc1);
        i += 16;
    }
    while i + 8 <= n {
        let va = _mm256_loadu_ps(ap.add(i));
        let vb = _mm256_loadu_ps(bp.add(i));
        acc0 = _mm256_fmadd_ps(va, vb, acc0);
        i += 8;
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    let hi = _mm256_extractf128_ps(acc0, 1);
    let lo = _mm256_castps256_ps128(acc0);
    let sum128 = _mm_add_ps(lo, hi);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    let mut sum = _mm_cvtss_f32(sum32);
    while i < n {
        sum += *ap.add(i) * *bp.add(i);
        i += 1;
    }
    sum
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn dot_avx(a: &[f32], b: &[f32]) -> f32 {
    #[allow(unsafe_code)]
    unsafe {
        dot_avx_impl(a, b)
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn dot_sse_impl(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let n = a.len();
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let mut acc0 = _mm_setzero_ps();
    let mut acc1 = _mm_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= n {
        let va0 = _mm_loadu_ps(ap.add(i));
        let vb0 = _mm_loadu_ps(bp.add(i));
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(va0, vb0));
        let va1 = _mm_loadu_ps(ap.add(i + 4));
        let vb1 = _mm_loadu_ps(bp.add(i + 4));
        acc1 = _mm_add_ps(acc1, _mm_mul_ps(va1, vb1));
        i += 8;
    }
    while i + 4 <= n {
        let va = _mm_loadu_ps(ap.add(i));
        let vb = _mm_loadu_ps(bp.add(i));
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(va, vb));
        i += 4;
    }
    acc0 = _mm_add_ps(acc0, acc1);
    let shuf = _mm_movehl_ps(acc0, acc0);
    let sum64 = _mm_add_ps(acc0, shuf);
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    let mut sum = _mm_cvtss_f32(sum32);
    while i < n {
        sum += *ap.add(i) * *bp.add(i);
        i += 1;
    }
    sum
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn dot_sse(a: &[f32], b: &[f32]) -> f32 {
    #[allow(unsafe_code)]
    unsafe {
        dot_sse_impl(a, b)
    }
}

fn repack_depthwise_kernel_once(
    weight: &Tensor,
    o_ch: usize,
    i_per_g: usize,
    kh: usize,
    kw: usize,
    channels: usize,
    depth_mult: usize,
) -> Result<Tensor, OnnxError> {
    let w_data = weight.data();
    let mut dw_data = vec![0.0f32; kh * kw * channels * depth_mult];
    for oc in 0..o_ch {
        let g = oc / depth_mult;
        let dm = oc % depth_mult;
        for ki in 0..kh {
            for kj in 0..kw {
                let src = ((oc * i_per_g) * kh + ki) * kw + kj;
                let dst = ((ki * kw + kj) * channels + g) * depth_mult + dm;
                dw_data[dst] = w_data[src];
            }
        }
    }
    Tensor::from_vec(vec![kh, kw, channels, depth_mult], dw_data).map_err(|e| {
        OnnxError::DecodeFailed {
            message: e.to_string(),
        }
    })
}

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
pub(super) fn exec_fused_dw_pw(
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
    let dw_input_nhwc_owned;
    let dw_input_nhwc: &Tensor = if dw_input_is_nhwc {
        &dw_input_src
    } else {
        dw_input_nhwc_owned = nchw_to_nhwc(&dw_input_src)?;
        &dw_input_nhwc_owned
    };
    let (pw_sh, pw_sw, pw_group, pw_pt, pw_pl, pw_pb, pw_pr, pw_has_pad) =
        resolve_conv_params(pw_node, pw_params);

    // Streaming DW->PW fast path. Reuses the per-row fused kernel from
    // yscv-kernels and avoids materializing the full DW intermediate in
    // memory. Default ON for 1-thread inference; multi-thread can opt in
    // via `YSCV_FUSED_DW_PW_STREAM_MT=1`. Kill switch:
    // `YSCV_FUSED_DW_PW_STREAM_OFF=1`.
    //
    // Padded-DW streaming is opt-in (`YSCV_FUSED_DW_PW_STREAM_PADDED=1`):
    // on low-end ARM boards this path can regress vs the existing
    // non-stream fused route, so keep it disabled by default until
    // we land a tuned padded streaming schedule.
    let stream_disabled = {
        use std::sync::OnceLock;
        static CACHED: OnceLock<bool> = OnceLock::new();
        *CACHED.get_or_init(|| std::env::var_os("YSCV_FUSED_DW_PW_STREAM_OFF").is_some())
    };
    let stream_mt_enabled = {
        use std::sync::OnceLock;
        static CACHED: OnceLock<bool> = OnceLock::new();
        *CACHED.get_or_init(|| std::env::var_os("YSCV_FUSED_DW_PW_STREAM_MT").is_some())
    };
    let stream_padded_enabled = {
        use std::sync::OnceLock;
        static CACHED: OnceLock<bool> = OnceLock::new();
        *CACHED.get_or_init(|| std::env::var_os("YSCV_FUSED_DW_PW_STREAM_PADDED").is_some())
    };
    let stream_allowed_threads = rayon::current_num_threads() <= 1 || stream_mt_enabled;

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
    if !stream_disabled && stream_allowed_threads && dw_streamable && pw_streamable {
        let pw_output = yscv_kernels::fused_dw_pw_nhwc_streaming(
            dw_input_nhwc,
            &dw_weight,
            dw_bias.as_ref(),
            &pw_weight_stream,
            pw_bias_stream.as_ref(),
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
pub(super) fn exec_fused_pw_dw(
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
) -> Result<(), OnnxError> {
    // ---- PW compute ----
    let pw_input_is_nhwc = env.is_nhwc(&pw_node.inputs[0]);
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
    // Ships **default ON** as of R7's AVX-512 register-blocked variant:
    // tracker 8-run × 1000-iter A/B shows −126 µs p50 / −160 µs min
    // (bitwise-close, 1-ULP FP-ordering drift). Kill switch
    // `YSCV_FUSED_PW_DW_STREAM_OFF=1` forces the fallback compute
    // chain for A/B measurement.
    let stream_disabled = {
        use std::sync::OnceLock;
        static CACHED: OnceLock<bool> = OnceLock::new();
        *CACHED.get_or_init(|| std::env::var_os("YSCV_FUSED_PW_DW_STREAM_OFF").is_some())
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
    if !stream_disabled
        && pw_is_trivial_1x1
        && dw_is_3x3_stride_1_or_2
        && pw_activation_streamable
        && dw_activation_streamable
    {
        let in_shape = pw_input_nhwc.shape();
        let (batch, in_h, in_w, c_in) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
        let pw_ws = pw_weight.shape();
        let c_exp = pw_ws[3];
        let dw_bias_tensor = if dw_node.inputs.len() > 2 && !dw_node.inputs[2].is_empty() {
            Some(get_tensor(env, &dw_node.name, &dw_node.inputs[2])?.clone())
        } else {
            None
        };
        let out_h = (in_h + 2 - 3) / dw_sh_early + 1;
        let out_w = (in_w + 2 - 3) / dw_sw_early + 1;
        let out_len = batch * out_h * out_w * c_exp;
        #[allow(unsafe_code)]
        let mut out = yscv_tensor::AlignedVec::<f32>::uninitialized(out_len);
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

    drop(dw_weight);
    drop(dw_bias);
    drop(pw_output);

    env.insert(dw_node.outputs[0].clone(), dw_output);
    env.mark_nhwc(&dw_node.outputs[0]);
    Ok(())
}

pub(super) fn exec_conv(
    node: &OnnxNode,
    env: &mut TensorEnv,
    activation: yscv_kernels::Activation,
) -> Result<(), OnnxError> {
    exec_conv_with_params(node, env, activation, None, None)
}

/// Resolves `(stride, group, pads, has_padding)` from either precomputed
/// `ConvParams` or the ONNX attribute HashMap. Shared between the thin
/// env-binding wrapper `exec_conv_with_params` and the fused-pair path
/// `exec_fused_dw_pw`.
#[inline]
fn resolve_conv_params(
    node: &OnnxNode,
    precomputed: Option<&crate::loader::ConvParams>,
) -> (usize, usize, usize, usize, usize, usize, usize, bool) {
    if let Some(p) = precomputed {
        (
            p.stride_h,
            p.stride_w,
            p.group,
            p.pad_top,
            p.pad_left,
            p.pad_bottom,
            p.pad_right,
            p.has_padding,
        )
    } else {
        let strides = get_attr_ints(node, "strides").unwrap_or_else(|| vec![1, 1]);
        let pads = get_attr_ints(node, "pads").unwrap_or_else(|| vec![0, 0, 0, 0]);
        let group = get_attr_int(node, "group").unwrap_or(1) as usize;
        let (pt, pl) = (pads[0] as usize, pads[1] as usize);
        let (pb, pr) = (
            pads.get(2).copied().unwrap_or(0) as usize,
            pads.get(3).copied().unwrap_or(0) as usize,
        );
        let sh = strides[0] as usize;
        let sw = strides.get(1).copied().unwrap_or(1) as usize;
        let has_padding = pads.iter().any(|&p| p > 0);
        (sh, sw, group, pt, pl, pb, pr, has_padding)
    }
}

/// Conv with optional pre-computed params (skips HashMap attr lookups).
///
/// Thin wrapper around [`conv_compute_nhwc`]: resolves input/weight/bias
/// from `env`, converts input to NHWC if needed, calls the pure-compute
/// path, then binds the result back into `env`. The split lets the
/// DW+PW fused path (`exec_fused_dw_pw`) chain two conv computes with
/// the DW intermediate kept as a local `Tensor` — never touching the
/// env HashMap.
pub(super) fn exec_conv_with_params(
    node: &OnnxNode,
    env: &mut TensorEnv,
    activation: yscv_kernels::Activation,
    precomputed: Option<&crate::loader::ConvParams>,
    prepacked_weight: Option<&yscv_kernels::PackedB>,
) -> Result<(), OnnxError> {
    let input_is_nhwc = env.is_nhwc(&node.inputs[0]);

    // BNNS NCHW fast path: when input is already NCHW, use Apple Accelerate
    // directly without any layout conversion. Opt-in via YSCV_BNNS=1.
    #[cfg(all(target_os = "macos", feature = "blas"))]
    if !input_is_nhwc
        && std::env::var("YSCV_BNNS").is_ok()
        && let Some(result) = exec_conv_bnns_nchw(node, env, activation)?
    {
        env.insert(node.outputs[0].clone(), result);
        // Do NOT mark_nhwc — output stays NCHW
        return Ok(());
    }

    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let weight = get_tensor(env, &node.name, &node.inputs[1])?;
    let bias = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
        Some(get_tensor(env, &node.name, &node.inputs[2])?)
    } else {
        None
    };

    let (sh, sw, group, pt, pl, pb, pr, has_padding) = resolve_conv_params(node, precomputed);

    // Skip NCHW→NHWC if input is already NHWC
    let input_nhwc_owned;
    let input_nhwc: &Tensor = if input_is_nhwc {
        input
    } else {
        input_nhwc_owned = nchw_to_nhwc(input)?;
        &input_nhwc_owned
    };

    let output = conv_compute_nhwc(
        node,
        input_nhwc,
        weight,
        bias,
        env,
        prepacked_weight,
        activation,
        sh,
        sw,
        group,
        pt,
        pl,
        pb,
        pr,
        has_padding,
    )?;
    env.insert(node.outputs[0].clone(), output);
    env.mark_nhwc(&node.outputs[0]);
    Ok(())
}

/// Pure-compute variant of the Conv dispatch. Takes an already-NHWC
/// input tensor, the raw weight and bias tensors, plus all resolved
/// shape/stride/pad params, and returns the NHWC output tensor. Does
/// **not** mutate `env` — the caller is responsible for binding the
/// result. `env` is read only for weight-layout flags (KHWC variants)
/// and the prepacked-B lookup.
///
/// Mirrors the three-branch dispatch (group == 1 → NHWC Conv, group
/// == C → depthwise, general grouped) of the original inline body
/// with every `env.insert` / `env.mark_nhwc` replaced by a returned
/// tensor.
#[allow(clippy::too_many_arguments)]
fn conv_compute_nhwc(
    node: &OnnxNode,
    input_nhwc: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    env: &TensorEnv,
    prepacked_weight: Option<&yscv_kernels::PackedB>,
    activation: yscv_kernels::Activation,
    sh: usize,
    sw: usize,
    group: usize,
    pt: usize,
    pl: usize,
    pb: usize,
    pr: usize,
    has_padding: bool,
) -> Result<Tensor, OnnxError> {
    // Weight: ONNX [O, I/group, KH, KW]; pre-permuted group=1 is [KH, KW, I, O].
    let w_shape = weight.shape();
    let is_dw_khwc = group > 1 && env.is_dw_khwc_weight(&node.inputs[1]);
    let is_group_khwc = group > 1 && env.is_group_khwc_weight(&node.inputs[1]);
    let (o_ch, i_per_g, kh, kw) = if env.is_khwc_weight(&node.inputs[1]) {
        (w_shape[3], w_shape[2], w_shape[0], w_shape[1])
    } else if is_dw_khwc {
        (
            w_shape[2].saturating_mul(w_shape[3]),
            1,
            w_shape[0],
            w_shape[1],
        )
    } else if is_group_khwc {
        (w_shape[0], w_shape[3], w_shape[1], w_shape[2])
    } else {
        (w_shape[0], w_shape[1], w_shape[2], w_shape[3])
    };

    if group == 1 {
        // Use pre-permuted weight if available (OIHW→KHWC done once upfront).
        let w_nhwc_owned;
        let w_nhwc: &Tensor = if env.is_khwc_weight(&node.inputs[1]) {
            weight
        } else {
            w_nhwc_owned = oihw_to_khwc_cout(weight)?;
            &w_nhwc_owned
        };
        // Phase 3.I: aarch64 3×3 non-DW via indirect convolution — avoids
        // the separate pad_nhwc allocation AND the im2col step by walking
        // output positions with inline kernel-tap padding checks. Benefits
        // RAM-bandwidth-constrained ARM SoCs (RK3588, Graviton). Only 3×3
        // since the current indirect implementation assumes KH=KW with
        // simple strided indexing.
        #[cfg(target_arch = "aarch64")]
        {
            if kh == 3 && kw == 3 && group == 1 && !cfg!(miri) {
                let t = yscv_kernels::conv2d_nhwc_indirect_padded(
                    input_nhwc, w_nhwc, bias, sh, sw, pt, pl, pb, pr, activation,
                )
                .map_err(|e| OnnxError::DecodeFailed {
                    message: e.to_string(),
                })?;
                return Ok(t);
            }
        }

        // Look up a load-time pre-packed B for this weight, if any. Only
        // pointwise Convs with KHWC layout get a prepack (see build_runtime_index).
        let prepacked = prepacked_weight.or_else(|| env.prepacked_b(&node.inputs[1]));

        // BLAS padded path fuses activation inside GEMM tiles (cache-hot).
        // All other paths apply activation afterward.
        let (mut out_nhwc, activation_fused) = if has_padding {
            #[cfg(feature = "blas")]
            {
                let t = yscv_kernels::conv2d_nhwc_padded(
                    input_nhwc, w_nhwc, bias, sh, sw, pt, pl, pb, pr, activation,
                )
                .map_err(|e| OnnxError::DecodeFailed {
                    message: e.to_string(),
                })?;
                (t, true)
            }
            #[cfg(not(feature = "blas"))]
            {
                let padded = pad_nhwc(input_nhwc, pt, pl, pb, pr)?;
                let t = yscv_kernels::conv2d_nhwc_with_activation_prepacked_default(
                    &padded, w_nhwc, bias, sh, sw, activation, prepacked,
                )
                .map_err(|e| OnnxError::DecodeFailed {
                    message: e.to_string(),
                })?;
                (t, true)
            }
        } else {
            let t = yscv_kernels::conv2d_nhwc_with_activation_prepacked_default(
                input_nhwc, w_nhwc, bias, sh, sw, activation, prepacked,
            )
            .map_err(|e| OnnxError::DecodeFailed {
                message: e.to_string(),
            })?;
            (t, true)
        };
        apply_conv_activation(&mut out_nhwc, activation, activation_fused);
        Ok(out_nhwc)
    } else if group == o_ch && group == input_nhwc.shape()[3] {
        let c = group;
        let depth_mult = o_ch / c;

        // Step E2 opt-in: native NCHWc DW 3×3 stride-1 SAME-pad kernel.
        // Gated on `YSCV_NCHWC_DW=1` — default OFF because the
        // micro-bench shows the current intrinsics path loses to the
        // existing mature NHWC im2col+SIMD+MT path (see
        // `project_step_e2_dw3x3_landed.md` memory). Kept wired so a
        // future inline-asm tuning pass can flip the default.
        if kh == 3
            && kw == 3
            && sh == 1
            && sw == 1
            && pt == 1
            && pl == 1
            && pb == 1
            && pr == 1
            && depth_mult == 1
            && c.is_multiple_of(8)
            && std::env::var("YSCV_NCHWC_DW").is_ok()
        {
            let dw_kernel_owned;
            let dw_kernel: &Tensor = if is_dw_khwc {
                weight
            } else {
                dw_kernel_owned =
                    repack_depthwise_kernel_once(weight, o_ch, i_per_g, kh, kw, c, depth_mult)?;
                &dw_kernel_owned
            };
            let input_nchwc = yscv_kernels::nhwc_to_nchwc(input_nhwc, 8).map_err(|e| {
                OnnxError::DecodeFailed {
                    message: e.to_string(),
                }
            })?;
            let out_nchwc = yscv_kernels::conv2d_nchwc_dw3x3_s1_same_pad(
                &input_nchwc,
                dw_kernel,
                bias,
                activation,
                c,
                yscv_kernels::ParallelElementwiseConfig::default(),
                None,
            )
            .map_err(|e| OnnxError::DecodeFailed {
                message: e.to_string(),
            })?;
            let mut out_nhwc = yscv_kernels::nchwc_to_nhwc(&out_nchwc, c).map_err(|e| {
                OnnxError::DecodeFailed {
                    message: e.to_string(),
                }
            })?;
            apply_conv_activation(&mut out_nhwc, activation, true);
            return Ok(out_nhwc);
        }

        let activation_fused = true;
        let mut out_nhwc = if has_padding {
            if is_dw_khwc {
                yscv_kernels::depthwise_conv2d_nhwc_padded_with_activation(
                    input_nhwc, weight, bias, sh, sw, pt, pl, pb, pr, activation,
                )
                .map_err(|e| OnnxError::DecodeFailed {
                    message: e.to_string(),
                })
            } else {
                let dw_kernel =
                    repack_depthwise_kernel_once(weight, o_ch, i_per_g, kh, kw, c, depth_mult)?;
                yscv_kernels::depthwise_conv2d_nhwc_padded_with_activation(
                    input_nhwc, &dw_kernel, bias, sh, sw, pt, pl, pb, pr, activation,
                )
                .map_err(|e| OnnxError::DecodeFailed {
                    message: e.to_string(),
                })
            }
        } else if is_dw_khwc {
            yscv_kernels::depthwise_conv2d_nhwc_with_activation(
                input_nhwc, weight, bias, sh, sw, activation,
            )
            .map_err(|e| OnnxError::DecodeFailed {
                message: e.to_string(),
            })
        } else {
            let dw_kernel =
                repack_depthwise_kernel_once(weight, o_ch, i_per_g, kh, kw, c, depth_mult)?;
            yscv_kernels::depthwise_conv2d_nhwc_with_activation(
                input_nhwc, &dw_kernel, bias, sh, sw, activation,
            )
            .map_err(|e| OnnxError::DecodeFailed {
                message: e.to_string(),
            })
        }?;
        apply_conv_activation(&mut out_nhwc, activation, activation_fused);
        Ok(out_nhwc)
    } else {
        // Grouped convolution with virtual padding (no explicit padded tensor).
        let in_shape = input_nhwc.shape();
        let (n, ih, iw, total_ic) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
        if total_ic % group != 0 || o_ch % group != 0 {
            return Err(OnnxError::DecodeFailed {
                message: format!(
                    "Grouped Conv channel mismatch: IC={total_ic}, OC={o_ch}, group={group}"
                ),
            });
        }
        let ic_per_group = total_ic / group;
        let oc_per_group = o_ch / group;
        if i_per_g != ic_per_group {
            return Err(OnnxError::DecodeFailed {
                message: format!(
                    "Grouped Conv weight/input mismatch: weight I/G={i_per_g}, input I/G={ic_per_group}"
                ),
            });
        }
        let padded_h = ih + pt + pb;
        let padded_w = iw + pl + pr;
        if kh > padded_h || kw > padded_w {
            return Err(OnnxError::DecodeFailed {
                message: format!(
                    "Grouped Conv kernel larger than padded input: input=({ih},{iw}), pads=({pt},{pl},{pb},{pr}), kernel=({kh},{kw})"
                ),
            });
        }
        let oh = (padded_h - kh) / sh + 1;
        let ow = (padded_w - kw) / sw + 1;
        let mut out_data = vec![0.0f32; n * oh * ow * o_ch];

        let in_data = input_nhwc.data();
        let w_data = weight.data();

        let w_khwc_stride = kh * kw * ic_per_group;
        let w_reordered: std::borrow::Cow<'_, [f32]> = if is_group_khwc {
            // Already pre-packed at model-load time: [O, KH, KW, I/G].
            std::borrow::Cow::Borrowed(w_data)
        } else {
            // Fallback for non-prepacked models: OIHW -> [O, KH, KW, I/G].
            let mut reordered = vec![0.0f32; o_ch * w_khwc_stride];
            for oc in 0..o_ch {
                for ki in 0..kh {
                    for kj in 0..kw {
                        let dst_base = oc * w_khwc_stride + (ki * kw + kj) * ic_per_group;
                        for ci in 0..ic_per_group {
                            reordered[dst_base + ci] =
                                w_data[((oc * ic_per_group + ci) * kh + ki) * kw + kj];
                        }
                    }
                }
            }
            std::borrow::Cow::Owned(reordered)
        };

        let bias_data: &[f32] = match &bias {
            Some(b) => b.data(),
            None => &[],
        };
        let dot = dot_dispatch();
        let relu_fused = activation == yscv_kernels::Activation::Relu;

        for batch in 0..n {
            for g in 0..group {
                let ic_start = g * ic_per_group;
                let oc_start = g * oc_per_group;
                for orow in 0..oh {
                    for ocol in 0..ow {
                        let out_base = ((batch * oh + orow) * ow + ocol) * o_ch + oc_start;
                        for oc in 0..oc_per_group {
                            let abs_oc = oc_start + oc;
                            let mut val = if !bias_data.is_empty() {
                                bias_data[abs_oc]
                            } else {
                                0.0
                            };
                            let w_oc_base = abs_oc * w_khwc_stride;
                            if has_padding {
                                for ki in 0..kh {
                                    let ir_raw = orow * sh + ki;
                                    if ir_raw < pt || ir_raw >= pt + ih {
                                        continue;
                                    }
                                    let ir = ir_raw - pt;
                                    for kj in 0..kw {
                                        let ic_raw = ocol * sw + kj;
                                        if ic_raw < pl || ic_raw >= pl + iw {
                                            continue;
                                        }
                                        let ic_pos = ic_raw - pl;
                                        let in_base =
                                            ((batch * ih + ir) * iw + ic_pos) * total_ic + ic_start;
                                        let w_base = w_oc_base + (ki * kw + kj) * ic_per_group;
                                        let in_slice = &in_data[in_base..in_base + ic_per_group];
                                        let w_slice = &w_reordered[w_base..w_base + ic_per_group];
                                        val += dot(in_slice, w_slice);
                                    }
                                }
                            } else {
                                for ki in 0..kh {
                                    let ir = orow * sh + ki;
                                    for kj in 0..kw {
                                        let ic_pos = ocol * sw + kj;
                                        let in_base =
                                            ((batch * ih + ir) * iw + ic_pos) * total_ic + ic_start;
                                        let w_base = w_oc_base + (ki * kw + kj) * ic_per_group;
                                        let in_slice = &in_data[in_base..in_base + ic_per_group];
                                        let w_slice = &w_reordered[w_base..w_base + ic_per_group];
                                        val += dot(in_slice, w_slice);
                                    }
                                }
                            }
                            out_data[out_base + oc] = if relu_fused { val.max(0.0) } else { val };
                        }
                    }
                }
            }
        }
        let mut out_nhwc = Tensor::from_vec(vec![n, oh, ow, o_ch], out_data).map_err(|e| {
            OnnxError::DecodeFailed {
                message: e.to_string(),
            }
        })?;
        apply_conv_activation(&mut out_nhwc, activation, relu_fused);
        Ok(out_nhwc)
    }
}

pub(super) fn exec_conv_transpose(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input_is_nhwc = env.is_nhwc(&node.inputs[0]);
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let weight = get_tensor(env, &node.name, &node.inputs[1])?;
    let bias = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
        Some(get_tensor(env, &node.name, &node.inputs[2])?)
    } else {
        None
    };

    let strides = get_attr_ints(node, "strides").unwrap_or_else(|| vec![1, 1]);
    let sh = strides[0] as usize;
    let sw = strides[1] as usize;

    let input_nhwc = if input_is_nhwc {
        input.clone()
    } else {
        nchw_to_nhwc(input)?
    };
    // ONNX ConvTranspose weight: [C_in, C_out, KH, KW] → [KH, KW, C_in, C_out]
    let w_t = weight
        .permute(&[2, 3, 0, 1])
        .map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;

    let out_nhwc =
        yscv_kernels::transpose_conv2d_nhwc(&input_nhwc, &w_t, bias, sh, sw).map_err(|e| {
            OnnxError::DecodeFailed {
                message: e.to_string(),
            }
        })?;
    env.insert(node.outputs[0].clone(), out_nhwc);
    env.mark_nhwc(&node.outputs[0]);
    Ok(())
}

pub(super) fn exec_qlinear_conv(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let x = get_tensor(env, &node.name, &node.inputs[0])?;
    let x_scale = get_tensor(env, &node.name, &node.inputs[1])?.data()[0];
    let x_zp = get_tensor(env, &node.name, &node.inputs[2])?.data()[0];
    let w = get_tensor(env, &node.name, &node.inputs[3])?;
    let w_scale = get_tensor(env, &node.name, &node.inputs[4])?.data()[0];
    let w_zp = get_tensor(env, &node.name, &node.inputs[5])?.data()[0];
    let y_scale = get_tensor(env, &node.name, &node.inputs[6])?.data()[0];
    let y_zp = get_tensor(env, &node.name, &node.inputs[7])?.data()[0];
    let bias = if node.inputs.len() > 8 && !node.inputs[8].is_empty() {
        Some(get_tensor(env, &node.name, &node.inputs[8])?.clone())
    } else {
        None
    };

    let deq_x: Vec<f32> = x.data().iter().map(|&v| (v - x_zp) * x_scale).collect();
    let deq_w: Vec<f32> = w.data().iter().map(|&v| (v - w_zp) * w_scale).collect();

    let deq_x_t =
        Tensor::from_vec(x.shape().to_vec(), deq_x).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
    let deq_w_t =
        Tensor::from_vec(w.shape().to_vec(), deq_w).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;

    let float_node = OnnxNode {
        name: node.name.clone(),
        op_type: "Conv".to_string(),
        inputs: vec!["__qx".into(), "__qw".into(), "__qb".into()],
        outputs: vec!["__qconv_out".into()],
        attributes: node.attributes.clone(),
    };
    env.insert("__qx".into(), deq_x_t);
    env.insert("__qw".into(), deq_w_t);
    if let Some(b) = bias {
        env.insert("__qb".into(), b);
    }
    exec_conv(&float_node, env, yscv_kernels::Activation::None)?;
    let float_out = env
        .remove("__qconv_out")
        .ok_or_else(|| OnnxError::MissingInput {
            node: node.name.clone(),
            input: "__qconv_out".into(),
        })?;
    env.remove("__qx");
    env.remove("__qw");
    env.remove("__qb");

    let quant: Vec<f32> = float_out
        .data()
        .iter()
        .map(|&v| (v / y_scale + y_zp).round().clamp(-128.0, 127.0))
        .collect();
    let out = Tensor::from_vec(float_out.shape().to_vec(), quant).map_err(|e| {
        OnnxError::DecodeFailed {
            message: e.to_string(),
        }
    })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_conv_integer(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let x = get_tensor(env, &node.name, &node.inputs[0])?;
    let w = get_tensor(env, &node.name, &node.inputs[1])?;
    let x_zp = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
        get_tensor(env, &node.name, &node.inputs[2])?.data()[0]
    } else {
        0.0
    };
    let w_zp = if node.inputs.len() > 3 && !node.inputs[3].is_empty() {
        get_tensor(env, &node.name, &node.inputs[3])?.data()[0]
    } else {
        0.0
    };

    let deq_x: Vec<f32> = x.data().iter().map(|&v| v - x_zp).collect();
    let deq_w: Vec<f32> = w.data().iter().map(|&v| v - w_zp).collect();

    let t_x = Tensor::from_vec(x.shape().to_vec(), deq_x).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    let t_w = Tensor::from_vec(w.shape().to_vec(), deq_w).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;

    let conv_node = OnnxNode {
        name: node.name.clone(),
        op_type: "Conv".to_string(),
        inputs: vec!["__ci_x".into(), "__ci_w".into(), "".into()],
        outputs: vec!["__ci_out".into()],
        attributes: node.attributes.clone(),
    };
    env.insert("__ci_x".into(), t_x);
    env.insert("__ci_w".into(), t_w);
    exec_conv(&conv_node, env, yscv_kernels::Activation::None)?;
    let out = env
        .remove("__ci_out")
        .ok_or_else(|| OnnxError::MissingInput {
            node: node.name.clone(),
            input: "__ci_out".into(),
        })?;
    env.remove("__ci_x");
    env.remove("__ci_w");

    let rounded: Vec<f32> = out.data().iter().map(|&v| v.round()).collect();
    let result =
        Tensor::from_vec(out.shape().to_vec(), rounded).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
    env.insert(node.outputs[0].clone(), result);
    Ok(())
}

/// ONNX DeformConv: deformable convolution with learned offsets.
///
/// Inputs: X (NCHW), offset, W (OIHW), [bias]
/// Attributes: strides, pads, group (only group=1 supported currently)
///
/// Converts NCHW inputs to NHWC, permutes weight from OIHW to [KH,KW,C_in,C_out],
/// and delegates to the `deformable_conv2d_nhwc` kernel.
pub(super) fn exec_deform_conv(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input_is_nhwc = env.is_nhwc(&node.inputs[0]);

    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let offset = get_tensor(env, &node.name, &node.inputs[1])?;
    let weight = get_tensor(env, &node.name, &node.inputs[2])?;
    let bias = if node.inputs.len() > 3 && !node.inputs[3].is_empty() {
        Some(get_tensor(env, &node.name, &node.inputs[3])?)
    } else {
        None
    };

    let strides = get_attr_ints(node, "strides").unwrap_or_else(|| vec![1, 1]);
    let pads = get_attr_ints(node, "pads").unwrap_or_else(|| vec![0, 0, 0, 0]);

    let stride = strides[0] as usize;
    // Symmetric padding only — use first pad value
    let padding = pads[0] as usize;

    // Convert input to NHWC if needed
    let input_nhwc = if input_is_nhwc {
        input.clone()
    } else {
        nchw_to_nhwc(input)?
    };

    // Convert offset from NCHW [N, kH*kW*2, out_H, out_W] to NHWC [N, out_H, out_W, kH*kW*2]
    let offset_nhwc = if offset.rank() == 4 {
        offset
            .permute(&[0, 2, 3, 1])
            .map_err(|e| OnnxError::DecodeFailed {
                message: e.to_string(),
            })?
    } else {
        offset.clone()
    };

    // Weight: ONNX [O, I, KH, KW] → kernel expects [KH, KW, I, O]
    let w_nhwc = weight
        .permute(&[2, 3, 1, 0])
        .map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;

    let out_nhwc = yscv_kernels::deformable_conv2d_nhwc(
        &input_nhwc,
        &w_nhwc,
        &offset_nhwc,
        bias,
        stride,
        padding,
    )
    .map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;

    env.insert(node.outputs[0].clone(), out_nhwc);
    env.mark_nhwc(&node.outputs[0]);
    Ok(())
}

// ── layout conversion helpers ──────────────────────────────────────

pub(super) fn nchw_to_nhwc(input: &Tensor) -> Result<Tensor, OnnxError> {
    // Step S.3: specialized AVX 8×8 block transpose when c%8==0 and
    // hw%8==0. Hot path fires on tracker DW Conv [1, 320, 16, 16]
    // twice per frame. Falls back to generic `Tensor::permute` for
    // non-aligned shapes (first-layer RGB c=3).
    // Measured tracker A/B (4 runs × 2 modes, min-of-3):
    //   6T p50 median −99 µs (−2.4%), 6T min −51 µs
    //   1T p50 median −33 µs (−0.3%), no regression
    yscv_kernels::nchw_to_nhwc_fast(input).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })
}

/// Convert ONNX Conv weight [O, I, KH, KW] to yscv [KH, KW, I, O]
pub(super) fn oihw_to_khwc_cout(weight: &Tensor) -> Result<Tensor, OnnxError> {
    weight
        .permute(&[2, 3, 1, 0])
        .map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })
}

/// Zero-pad an NHWC tensor on H/W dimensions.
pub(super) fn pad_nhwc(
    input: &Tensor,
    top: usize,
    left: usize,
    bottom: usize,
    right: usize,
) -> Result<Tensor, OnnxError> {
    pad_nhwc_val(input, top, left, bottom, right, 0.0)
}

pub(super) fn pad_nhwc_val(
    input: &Tensor,
    top: usize,
    left: usize,
    bottom: usize,
    right: usize,
    val: f32,
) -> Result<Tensor, OnnxError> {
    let shape = input.shape();
    let (n, h, w, c) = (shape[0], shape[1], shape[2], shape[3]);
    let oh = h + top + bottom;
    let ow = w + left + right;
    let mut out = vec![val; n * oh * ow * c];
    let in_data = input.data();
    let row_bytes = w * c;
    for batch in 0..n {
        for row in 0..h {
            let src_start = (batch * h + row) * w * c;
            let dst_start = ((batch * oh + row + top) * ow + left) * c;
            out[dst_start..dst_start + row_bytes]
                .copy_from_slice(&in_data[src_start..src_start + row_bytes]);
        }
    }
    Tensor::from_vec(vec![n, oh, ow, c], out).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })
}

// ── BNNS NCHW fast path ────────────────────────────────────────────

/// Try to execute conv via Apple BNNS on NCHW data.
/// Returns `Ok(Some(tensor))` on success, `Ok(None)` if BNNS can't handle this op.
#[cfg(all(target_os = "macos", feature = "blas"))]
fn exec_conv_bnns_nchw(
    node: &OnnxNode,
    env: &mut TensorEnv,
    activation: yscv_kernels::Activation,
) -> Result<Option<Tensor>, OnnxError> {
    use yscv_kernels::bnns_conv::{BnnsActivation, BnnsConvParams, conv2d_nchw_bnns};

    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let weight = get_tensor(env, &node.name, &node.inputs[1])?;
    let bias = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
        Some(get_tensor(env, &node.name, &node.inputs[2])?)
    } else {
        None
    };

    let strides = get_attr_ints(node, "strides").unwrap_or_else(|| vec![1, 1]);
    let pads = get_attr_ints(node, "pads").unwrap_or_else(|| vec![0, 0, 0, 0]);
    let group = get_attr_int(node, "group").unwrap_or(1) as usize;
    let sh = strides[0] as usize;
    let sw = strides[1] as usize;
    let (pt, pl, pb, pr) = (
        pads[0] as usize,
        pads[1] as usize,
        pads[2] as usize,
        pads[3] as usize,
    );

    // Weight must be OIHW for BNNS. group=1 weights are pre-permuted to KHWC —
    // reverse them back. Depthwise/grouped weights are already OIHW.
    let w_oihw_owned;
    let w_oihw: &Tensor = if env.is_khwc_weight(&node.inputs[1]) {
        // KHWC [KH, KW, I, O] → OIHW [O, I, KH, KW]
        w_oihw_owned = weight
            .permute(&[3, 2, 0, 1])
            .map_err(|e| OnnxError::DecodeFailed {
                message: e.to_string(),
            })?;
        &w_oihw_owned
    } else {
        weight
    };

    let in_shape = input.shape();
    if in_shape.len() != 4 {
        return Ok(None);
    }
    let (batch, in_c, in_h, in_w) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);

    let w_shape = w_oihw.shape();
    let (out_c, _ic_per_g, kh, kw) = (w_shape[0], w_shape[1], w_shape[2], w_shape[3]);

    let out_h = (in_h + pt + pb - kh) / sh + 1;
    let out_w = (in_w + pl + pr - kw) / sw + 1;

    let bnns_act = match activation {
        yscv_kernels::Activation::Silu => BnnsActivation::Silu,
        yscv_kernels::Activation::Relu => BnnsActivation::Relu,
        yscv_kernels::Activation::None => BnnsActivation::None,
    };

    let params = BnnsConvParams {
        batch,
        in_c,
        in_h,
        in_w,
        out_c,
        out_h,
        out_w,
        kh,
        kw,
        stride_h: sh,
        stride_w: sw,
        pad_top: pt,
        pad_left: pl,
        pad_bottom: pb,
        pad_right: pr,
        groups: group,
        activation: bnns_act,
    };

    Ok(conv2d_nchw_bnns(input, w_oihw, bias, &params))
}
