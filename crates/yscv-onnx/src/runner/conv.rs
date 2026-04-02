use super::*;

/// ONNX Conv: NHWC-aware. Skips layout conversion if input is already NHWC.
/// Output is left in NHWC to avoid redundant conversions in spatial chains.
/// `activation` is applied fused into GEMM tiles (cache-hot) when using BLAS padded path.
pub(super) fn exec_conv(
    node: &OnnxNode,
    env: &mut TensorEnv,
    activation: yscv_kernels::Activation,
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

    let strides = get_attr_ints(node, "strides").unwrap_or_else(|| vec![1, 1]);
    let pads = get_attr_ints(node, "pads").unwrap_or_else(|| vec![0, 0, 0, 0]);
    let group = get_attr_int(node, "group").unwrap_or(1) as usize;

    let sh = strides[0] as usize;
    let sw = strides[1] as usize;

    // Skip NCHW→NHWC if input is already NHWC
    let input_nhwc_owned;
    let input_nhwc: &Tensor = if input_is_nhwc {
        input
    } else {
        input_nhwc_owned = nchw_to_nhwc(input)?;
        &input_nhwc_owned
    };

    // Weight: ONNX [O, I/group, KH, KW]; pre-permuted group=1 is [KH, KW, I, O].
    let w_shape = weight.shape();
    let (o_ch, i_per_g, kh, kw) = if env.is_khwc_weight(&node.inputs[1]) {
        (w_shape[3], w_shape[2], w_shape[0], w_shape[1])
    } else {
        (w_shape[0], w_shape[1], w_shape[2], w_shape[3])
    };

    let has_padding = pads.iter().any(|&p| p > 0);
    let (pt, pl, pb, pr) = (
        pads[0] as usize,
        pads[1] as usize,
        pads[2] as usize,
        pads[3] as usize,
    );

    if group == 1 {
        // Use pre-permuted weight if available (OIHW→KHWC done once upfront).
        let w_nhwc_owned;
        let w_nhwc: &Tensor = if env.is_khwc_weight(&node.inputs[1]) {
            weight
        } else {
            w_nhwc_owned = oihw_to_khwc_cout(weight)?;
            &w_nhwc_owned
        };
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
                (t, true) // activation applied in-tile
            }
            #[cfg(not(feature = "blas"))]
            {
                let padded = pad_nhwc(input_nhwc, pt, pl, pb, pr)?;
                let t = conv2d_nhwc(&padded, w_nhwc, bias, sh, sw).map_err(|e| {
                    OnnxError::DecodeFailed {
                        message: e.to_string(),
                    }
                })?;
                (t, false)
            }
        } else {
            let t = conv2d_nhwc(input_nhwc, w_nhwc, bias, sh, sw).map_err(|e| {
                OnnxError::DecodeFailed {
                    message: e.to_string(),
                }
            })?;
            (t, false)
        };
        if !activation_fused && activation == yscv_kernels::Activation::Silu {
            yscv_kernels::silu_inplace(&mut out_nhwc);
        }
        env.insert(node.outputs[0].clone(), out_nhwc);
        env.mark_nhwc(&node.outputs[0]);
    } else if group as usize == o_ch
        && group as usize
            == if input_is_nhwc {
                input.shape()[3]
            } else {
                input.shape()[1]
            }
    {
        // Depthwise — pad input if needed
        let padded_owned;
        let input_padded: &Tensor = if has_padding {
            padded_owned = pad_nhwc(input_nhwc, pt, pl, pb, pr)?;
            &padded_owned
        } else {
            input_nhwc
        };
        let c = group;
        let depth_mult = o_ch / c;
        let mut dw_data = vec![0.0f32; kh * kw * c * depth_mult];
        let w_data = weight.data();
        for oc in 0..o_ch {
            let g = oc / depth_mult;
            let dm = oc % depth_mult;
            for ki in 0..kh {
                for kj in 0..kw {
                    let src = ((oc * i_per_g) * kh + ki) * kw + kj;
                    let dst = ((ki * kw + kj) * c + g) * depth_mult + dm;
                    dw_data[dst] = w_data[src];
                }
            }
        }
        let dw_kernel = Tensor::from_vec(vec![kh, kw, c, depth_mult], dw_data).map_err(|e| {
            OnnxError::DecodeFailed {
                message: e.to_string(),
            }
        })?;
        let mut out_nhwc =
            yscv_kernels::depthwise_conv2d_nhwc(input_padded, &dw_kernel, bias, sh, sw).map_err(
                |e| OnnxError::DecodeFailed {
                    message: e.to_string(),
                },
            )?;
        if activation == yscv_kernels::Activation::Silu {
            yscv_kernels::silu_inplace(&mut out_nhwc);
        }
        env.insert(node.outputs[0].clone(), out_nhwc);
        env.mark_nhwc(&node.outputs[0]);
    } else {
        // Grouped convolution — pad input if needed
        let padded_owned;
        let input_padded: &Tensor = if has_padding {
            padded_owned = pad_nhwc(input_nhwc, pt, pl, pb, pr)?;
            &padded_owned
        } else {
            input_nhwc
        };
        let in_shape = input_padded.shape();
        let (n, ih, iw, total_ic) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
        let ic_per_group = total_ic / group;
        let oc_per_group = o_ch / group;
        let oh = (ih - kh) / sh + 1;
        let ow = (iw - kw) / sw + 1;
        let mut out_data = vec![0.0f32; n * oh * ow * o_ch];

        let in_data = input_padded.data();
        let w_data = weight.data();

        // Pre-permute weights from OIHW to [oc, kh, kw, ic_per_group] so the
        // inner ci dimension is contiguous, enabling a slice dot-product.
        let w_khwc_stride = kh * kw * ic_per_group;
        let mut w_reordered = vec![0.0f32; o_ch * w_khwc_stride];
        for oc in 0..o_ch {
            for ki in 0..kh {
                for kj in 0..kw {
                    let dst_base = oc * w_khwc_stride + (ki * kw + kj) * ic_per_group;
                    for ci in 0..ic_per_group {
                        w_reordered[dst_base + ci] =
                            w_data[((oc * ic_per_group + ci) * kh + ki) * kw + kj];
                    }
                }
            }
        }

        let bias_data: &[f32] = match &bias {
            Some(b) => b.data(),
            None => &[],
        };

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
                            for ki in 0..kh {
                                let ir = orow * sh + ki;
                                for kj in 0..kw {
                                    let ic_pos = ocol * sw + kj;
                                    let in_base =
                                        ((batch * ih + ir) * iw + ic_pos) * total_ic + ic_start;
                                    let w_base = w_oc_base + (ki * kw + kj) * ic_per_group;
                                    let in_slice = &in_data[in_base..in_base + ic_per_group];
                                    let w_slice = &w_reordered[w_base..w_base + ic_per_group];
                                    for ci in 0..ic_per_group {
                                        val += in_slice[ci] * w_slice[ci];
                                    }
                                }
                            }
                            out_data[out_base + oc] = val;
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
        if activation == yscv_kernels::Activation::Silu {
            yscv_kernels::silu_inplace(&mut out_nhwc);
        }
        env.insert(node.outputs[0].clone(), out_nhwc);
        env.mark_nhwc(&node.outputs[0]);
    }

    Ok(())
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

// ── layout conversion helpers ──────────────────────────────────────

pub(super) fn nchw_to_nhwc(input: &Tensor) -> Result<Tensor, OnnxError> {
    input
        .permute(&[0, 2, 3, 1])
        .map_err(|e| OnnxError::DecodeFailed {
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
