//! Quantized convolution ops: QLinearConv, ConvInteger, and the fused
//! int8 PW/DW / DW-residual quantized paths.

use super::super::*;
use super::*;

pub(crate) fn exec_qlinear_conv(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let x_scale = get_tensor(env, &node.name, &node.inputs[1])?.data()[0];
    let x_zp = get_tensor(env, &node.name, &node.inputs[2])?.data()[0];
    let w = get_tensor(env, &node.name, &node.inputs[3])?.clone();
    let w_scale = get_tensor(env, &node.name, &node.inputs[4])?.data()[0];
    let w_zp = get_tensor(env, &node.name, &node.inputs[5])?.data()[0];
    let y_scale = get_tensor(env, &node.name, &node.inputs[6])?.data()[0];
    let y_zp = get_tensor(env, &node.name, &node.inputs[7])?.data()[0];
    let bias = if node.inputs.len() > 8 && !node.inputs[8].is_empty() {
        Some(get_tensor(env, &node.name, &node.inputs[8])?.clone())
    } else {
        None
    };
    let x_quant = if crate::runner::quant_int8_fast_enabled() {
        if env.static_use_count(&node.inputs[0]) <= 1 {
            env.take_quant_i8(&node.inputs[0])
        } else {
            env.get_quant_i8(&node.inputs[0]).cloned()
        }
    } else {
        None
    };
    let x_tensor = if x_quant.is_none() {
        Some(get_tensor(env, &node.name, &node.inputs[0])?.clone())
    } else {
        None
    };
    let x_shape: &[usize] = x_quant
        .as_ref()
        .map(|q| q.shape.as_slice())
        .or_else(|| x_tensor.as_ref().map(|t| t.shape()))
        .ok_or_else(|| OnnxError::MissingInput {
            node: node.name.clone(),
            input: node.inputs[0].clone(),
        })?;
    let x_q_data = x_quant.as_ref().map(|q| q.data.as_slice());
    let x_t_data = x_tensor.as_ref().map(|t| t.data());

    // Symmetric int8 fast path: NCHW input, OIHW weight, group=1, no
    // dilation, both zero-points 0. im2col + integer GEMM + composite
    // requantize. Loader's KHWC permute fires only on `Conv` op_type,
    // so QLinearConv weights stay OIHW here.
    if x_shape.len() == 4 && w.shape().len() == 4 {
        let group = crate::runner::get_attr_int(node, "group").unwrap_or(1);
        let dilations =
            crate::runner::get_attr_ints(node, "dilations").unwrap_or_else(|| vec![1, 1]);
        if x_zp == 0.0 && w_zp == 0.0 && group == 1 && dilations == [1, 1] {
            let pads =
                crate::runner::get_attr_ints(node, "pads").unwrap_or_else(|| vec![0, 0, 0, 0]);
            let strides =
                crate::runner::get_attr_ints(node, "strides").unwrap_or_else(|| vec![1, 1]);
            let xs = x_shape;
            let ws = w.shape();
            let (n_n, c_in, ih, iw) = (xs[0], xs[1], xs[2], xs[3]);
            let (c_out, _, kh, kw) = (ws[0], ws[1], ws[2], ws[3]);
            if ws[1] == c_in && pads.len() == 4 && strides.len() == 2 {
                let (pt, pl, pb, pr) = (
                    pads[0] as usize,
                    pads[1] as usize,
                    pads[2] as usize,
                    pads[3] as usize,
                );
                let (sh, sw) = (strides[0] as usize, strides[1] as usize);
                let oh = (ih + pt + pb - kh) / sh + 1;
                let ow = (iw + pl + pr - kw) / sw + 1;
                let m = n_n * oh * ow;
                let k_dim = c_in * kh * kw;

                // im2col NCHW → [M, K] i8.
                let mut x_im2col = env.take_i8_scratch_a(m * k_dim);
                for ni in 0..n_n {
                    for oh_i in 0..oh {
                        for ow_i in 0..ow {
                            let row = (ni * oh + oh_i) * ow + ow_i;
                            for ci in 0..c_in {
                                for ky in 0..kh {
                                    for kx in 0..kw {
                                        let ih_i = oh_i * sh + ky;
                                        let iw_i = ow_i * sw + kx;
                                        let col = (ci * kh + ky) * kw + kx;
                                        let idx = row * k_dim + col;
                                        if ih_i >= pt
                                            && ih_i < pt + ih
                                            && iw_i >= pl
                                            && iw_i < pl + iw
                                        {
                                            let h = ih_i - pt;
                                            let v = iw_i - pl;
                                            let src = ((ni * c_in + ci) * ih + h) * iw + v;
                                            x_im2col[idx] = x_q_data
                                                .map(|d| d[src])
                                                .unwrap_or_else(|| x_t_data.unwrap()[src] as i8);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Integer GEMM.
                let mut acc = env.take_i32_scratch(m * c_out);
                if crate::runner::should_use_prepacked_i8_b(m, k_dim, c_out)
                    && let Some(packed) = env.prepacked_i8_b(&node.inputs[3])
                {
                    if packed.k() == k_dim && packed.n() == c_out {
                        yscv_kernels::int8_matmul_prepacked_dispatch(
                            &x_im2col, packed, m, &mut acc,
                        );
                    } else {
                        return Err(OnnxError::DecodeFailed {
                            message: format!(
                                "QLinearConv {}: prepacked weight shape mismatch",
                                node.name
                            ),
                        });
                    }
                } else {
                    // Reshape weight OIHW → [K=C*KH*KW, N=O].
                    let w_data = w.data();
                    let mut w_packed: Vec<i8> = vec![0; k_dim * c_out];
                    for o in 0..c_out {
                        for ci in 0..c_in {
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let src = ((o * c_in + ci) * kh + ky) * kw + kx;
                                    let dst_k = (ci * kh + ky) * kw + kx;
                                    w_packed[dst_k * c_out + o] = w_data[src] as i8;
                                }
                            }
                        }
                    }
                    yscv_kernels::int8_matmul_dispatch(
                        &x_im2col, &w_packed, m, k_dim, c_out, &mut acc,
                    );
                }

                // Add per-channel bias (i32 stored as f32) if present.
                if let Some(b) = &bias {
                    let bdata = b.data();
                    for row in 0..m {
                        for o in 0..c_out {
                            acc[row * c_out + o] += bdata[o] as i32;
                        }
                    }
                }

                // Composite-scale requantize + clamp; reshape [M, O] → NCHW.
                let composite = (x_scale * w_scale) / y_scale;
                let mut out = vec![0_i8; n_n * c_out * oh * ow];
                for ni in 0..n_n {
                    for o in 0..c_out {
                        for oh_i in 0..oh {
                            for ow_i in 0..ow {
                                let row = (ni * oh + oh_i) * ow + ow_i;
                                let v = (acc[row * c_out + o] as f32) * composite + y_zp;
                                let dst = ((ni * c_out + o) * oh + oh_i) * ow + ow_i;
                                out[dst] = v.round().clamp(-128.0, 127.0) as i8;
                            }
                        }
                    }
                }
                env.put_i32_scratch(acc);
                env.put_i8_scratch_a(x_im2col);
                env.insert_quant_i8(
                    node.outputs[0].clone(),
                    QuantTensor {
                        data: out,
                        shape: vec![n_n, c_out, oh, ow],
                        scale: y_scale,
                        zero_point: y_zp,
                        nhwc: false,
                    },
                );
                crate::runner::note_qlinear_conv_fast();
                return Ok(());
            }
        }
        if group >= 1 && dilations == [1, 1] {
            let pads =
                crate::runner::get_attr_ints(node, "pads").unwrap_or_else(|| vec![0, 0, 0, 0]);
            let strides =
                crate::runner::get_attr_ints(node, "strides").unwrap_or_else(|| vec![1, 1]);
            let xs = x_shape;
            let ws = w.shape();
            let (c_out, c_per_g, kh, kw) = (ws[0], ws[1], ws[2], ws[3]);
            let group_usize = group as usize;
            let expected_c = c_per_g * group_usize;
            let input_is_nhwc = xs[3] == expected_c && xs[1] != expected_c;
            let (n_n, c_in, ih, iw) = if input_is_nhwc {
                (xs[0], xs[3], xs[1], xs[2])
            } else {
                (xs[0], xs[1], xs[2], xs[3])
            };
            if pads.len() == 4
                && strides.len() == 2
                && group_usize > 0
                && c_in == expected_c
                && c_out % group_usize == 0
            {
                let (pt, pl, pb, pr) = (
                    pads[0] as usize,
                    pads[1] as usize,
                    pads[2] as usize,
                    pads[3] as usize,
                );
                let (sh, sw) = (strides[0] as usize, strides[1] as usize);
                let oh = (ih + pt + pb - kh) / sh + 1;
                let ow = (iw + pl + pr - kw) / sw + 1;
                if x_zp == 0.0
                    && w_zp == 0.0
                    && c_per_g == 1
                    && c_out == group_usize
                    && c_in == group_usize
                    && kh == kw
                    && (kh == 3 || kh == 5)
                    && !input_is_nhwc
                    && sh == 2
                    && sw == 2
                    && ih > 64
                    && let Some(dw_weight) = env.prepacked_i8_depthwise(&node.inputs[3])
                {
                    let p = yscv_kernels::DepthwiseI8Params {
                        batch: n_n,
                        in_h: ih,
                        in_w: iw,
                        channels: c_in,
                        kernel: kh,
                        stride_h: sh,
                        stride_w: sw,
                        pad_top: pt,
                        pad_left: pl,
                        out_h: oh,
                        out_w: ow,
                    };
                    let mut x_nchw = env.take_i8_scratch_a(n_n * c_in * ih * iw);
                    if let Some(data) = x_q_data {
                        x_nchw.copy_from_slice(data);
                    } else {
                        for (dst, &v) in x_nchw.iter_mut().zip(x_t_data.unwrap()) {
                            *dst = v as i8;
                        }
                    }
                    let mut acc = env.take_i32_scratch(n_n * c_out * oh * ow);
                    yscv_kernels::depthwise_i8_i32_nchw_khwc_dispatch(
                        &x_nchw,
                        dw_weight.as_slice(),
                        p,
                        &mut acc,
                    );

                    let bias_data = bias.as_ref().map(|b| b.data());
                    let composite = (x_scale * w_scale) / y_scale;
                    let mut out = vec![0_i8; n_n * c_out * oh * ow];
                    for ni in 0..n_n {
                        for c in 0..c_out {
                            let bias_i32 = bias_data.map(|b| b[c] as i32).unwrap_or(0);
                            for oh_i in 0..oh {
                                for ow_i in 0..ow {
                                    let idx = ((ni * c_out + c) * oh + oh_i) * ow + ow_i;
                                    out[idx] = (((acc[idx] + bias_i32) as f32) * composite + y_zp)
                                        .round()
                                        .clamp(-128.0, 127.0)
                                        as i8;
                                }
                            }
                        }
                    }
                    env.put_i32_scratch(acc);
                    env.put_i8_scratch_a(x_nchw);
                    env.insert_quant_i8(
                        node.outputs[0].clone(),
                        QuantTensor {
                            data: out,
                            shape: vec![n_n, c_out, oh, ow],
                            scale: y_scale,
                            zero_point: y_zp,
                            nhwc: false,
                        },
                    );
                    crate::runner::note_qlinear_conv_fast();
                    return Ok(());
                }
                if x_zp == 0.0
                    && w_zp == 0.0
                    && c_per_g == 1
                    && c_out == group_usize
                    && c_in == group_usize
                    && kh == kw
                    && (kh == 3 || kh == 5)
                    && ((sh == 1 && sw == 1) || (sh == 2 && sw == 2 && ih <= 64))
                    && let Some(dw_weight) = env.prepacked_i8_depthwise(&node.inputs[3])
                {
                    let p = yscv_kernels::DepthwiseI8Params {
                        batch: n_n,
                        in_h: ih,
                        in_w: iw,
                        channels: c_in,
                        kernel: kh,
                        stride_h: sh,
                        stride_w: sw,
                        pad_top: pt,
                        pad_left: pl,
                        out_h: oh,
                        out_w: ow,
                    };
                    let mut x_nhwc = env.take_i8_scratch_a(n_n * ih * iw * c_in);
                    if input_is_nhwc {
                        if let Some(data) = x_q_data {
                            x_nhwc.copy_from_slice(data);
                        } else {
                            for (dst, &v) in x_nhwc.iter_mut().zip(x_t_data.unwrap()) {
                                *dst = v as i8;
                            }
                        }
                    } else {
                        for ni in 0..n_n {
                            for h in 0..ih {
                                for v in 0..iw {
                                    let dst_base = ((ni * ih + h) * iw + v) * c_in;
                                    for c in 0..c_in {
                                        let src = ((ni * c_in + c) * ih + h) * iw + v;
                                        x_nhwc[dst_base + c] = x_q_data
                                            .map(|d| d[src])
                                            .unwrap_or_else(|| x_t_data.unwrap()[src] as i8);
                                    }
                                }
                            }
                        }
                    }
                    let mut acc = env.take_i32_scratch(n_n * oh * ow * c_out);
                    yscv_kernels::depthwise_i8_i32_nhwc_dispatch(
                        &x_nhwc,
                        dw_weight.as_slice(),
                        p,
                        &mut acc,
                    );

                    let bias_data = bias.as_ref().map(|b| b.data());
                    let composite = (x_scale * w_scale) / y_scale;
                    let mut out = vec![0_i8; n_n * c_out * oh * ow];
                    for ni in 0..n_n {
                        for oh_i in 0..oh {
                            for ow_i in 0..ow {
                                let acc_base = ((ni * oh + oh_i) * ow + ow_i) * c_out;
                                for c in 0..c_out {
                                    let biased = acc[acc_base + c]
                                        + bias_data.map(|b| b[c] as i32).unwrap_or(0);
                                    let dst = ((ni * c_out + c) * oh + oh_i) * ow + ow_i;
                                    out[dst] = ((biased as f32) * composite + y_zp)
                                        .round()
                                        .clamp(-128.0, 127.0)
                                        as i8;
                                }
                            }
                        }
                    }
                    env.put_i32_scratch(acc);
                    env.put_i8_scratch_a(x_nhwc);
                    env.insert_quant_i8(
                        node.outputs[0].clone(),
                        QuantTensor {
                            data: out,
                            shape: vec![n_n, c_out, oh, ow],
                            scale: y_scale,
                            zero_point: y_zp,
                            nhwc: false,
                        },
                    );
                    crate::runner::note_qlinear_conv_fast();
                    return Ok(());
                }

                let out_per_g = c_out / group_usize;
                let w_data = w.data();
                let bias_data = bias.as_ref().map(|b| b.data());
                let composite = (x_scale * w_scale) / y_scale;
                let mut out = vec![0_i8; n_n * c_out * oh * ow];
                for ni in 0..n_n {
                    for o in 0..c_out {
                        let g = o / out_per_g;
                        let c_base = g * c_per_g;
                        for oh_i in 0..oh {
                            for ow_i in 0..ow {
                                let mut acc = bias_data.map(|b| b[o] as i32).unwrap_or(0);
                                for ci in 0..c_per_g {
                                    let src_c = c_base + ci;
                                    for ky in 0..kh {
                                        for kx in 0..kw {
                                            let ih_i = oh_i * sh + ky;
                                            let iw_i = ow_i * sw + kx;
                                            if ih_i >= pt
                                                && ih_i < pt + ih
                                                && iw_i >= pl
                                                && iw_i < pl + iw
                                            {
                                                let h = ih_i - pt;
                                                let v = iw_i - pl;
                                                let x_idx = if input_is_nhwc {
                                                    ((ni * ih + h) * iw + v) * c_in + src_c
                                                } else {
                                                    ((ni * c_in + src_c) * ih + h) * iw + v
                                                };
                                                let w_idx =
                                                    ((o * c_per_g + ci) * kh + ky) * kw + kx;
                                                let xv = x_q_data
                                                    .map(|d| d[x_idx] as f32)
                                                    .unwrap_or_else(|| x_t_data.unwrap()[x_idx]);
                                                let xv = (xv - x_zp).round() as i32;
                                                let wv = (w_data[w_idx] - w_zp).round() as i32;
                                                acc += xv * wv;
                                            }
                                        }
                                    }
                                }
                                let dst = ((ni * c_out + o) * oh + oh_i) * ow + ow_i;
                                out[dst] = ((acc as f32) * composite + y_zp)
                                    .round()
                                    .clamp(-128.0, 127.0)
                                    as i8;
                            }
                        }
                    }
                }
                env.insert_quant_i8(
                    node.outputs[0].clone(),
                    QuantTensor {
                        data: out,
                        shape: vec![n_n, c_out, oh, ow],
                        scale: y_scale,
                        zero_point: y_zp,
                        nhwc: false,
                    },
                );
                crate::runner::note_qlinear_conv_fast();
                return Ok(());
            }
        }
    }

    crate::runner::note_qlinear_conv_fallback();
    let deq_x: Vec<f32> = if let Some(data) = x_q_data {
        data.iter()
            .map(|&v| ((v as f32) - x_zp) * x_scale)
            .collect()
    } else {
        x_t_data
            .unwrap()
            .iter()
            .map(|&v| (v - x_zp) * x_scale)
            .collect()
    };
    let deq_w: Vec<f32> = w.data().iter().map(|&v| (v - w_zp) * w_scale).collect();

    let deq_x_t =
        Tensor::from_vec(x_shape.to_vec(), deq_x).map_err(|e| OnnxError::DecodeFailed {
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

/// Dispatch the fused INT8 PW->DW chain for the action emitted by the
/// loader's `QuantizedPwDw` detector. Resolves composite scales, biases
/// and pre-packed weights from `env`, then hands off to the kernel.
///
/// Caller contract (loader-enforced gates):
/// * `pw` is a 1×1 group-1 `QLinearConv` with `x_zp = w_zp = y_zp = 0`;
/// * `dw` is a 3×3/5×5 depthwise `QLinearConv` (group=`c_exp`) with
///   `x_zp = w_zp = 0`, dilations [1,1], symmetric pad and stride
///   1 or 2;
/// * pre-packed PW (VNNI 4×16) and DW (KHWC i8) weights are present
///   in `env`.
///
/// Output is inserted as a `QuantTensor` keyed by `dw.outputs[0]`,
/// shape `[N, c_exp, out_h, out_w]`, scale = `dw.y_scale`,
/// zero-point = `dw.y_zp`. Bitwise-identical to running the underlying
/// PW + QuantizedQdq + DW actions in sequence.
pub(crate) fn exec_quantized_pw_dw(
    pw: &OnnxNode,
    dw: &OnnxNode,
    env: &mut TensorEnv,
    has_relu: bool,
) -> Result<(), OnnxError> {
    let pw_x_scale = get_tensor(env, &pw.name, &pw.inputs[1])?.data()[0];
    let pw_w_scale = get_tensor(env, &pw.name, &pw.inputs[4])?.data()[0];
    let pw_y_scale = get_tensor(env, &pw.name, &pw.inputs[6])?.data()[0];
    let dw_x_scale = get_tensor(env, &dw.name, &dw.inputs[1])?.data()[0];
    let dw_w_scale = get_tensor(env, &dw.name, &dw.inputs[4])?.data()[0];
    let dw_y_scale = get_tensor(env, &dw.name, &dw.inputs[6])?.data()[0];
    let dw_y_zp = get_tensor(env, &dw.name, &dw.inputs[7])?.data()[0];

    let pw_w = get_tensor(env, &pw.name, &pw.inputs[3])?;
    let pw_w_shape = pw_w.shape().to_vec();
    if pw_w_shape.len() != 4 {
        return Err(OnnxError::DecodeFailed {
            message: format!("QuantizedPwDw {}: PW weight rank must be 4", pw.name),
        });
    }
    let (c_exp_pw, c_in_pw, _, _) = (pw_w_shape[0], pw_w_shape[1], pw_w_shape[2], pw_w_shape[3]);

    let dw_w = get_tensor(env, &dw.name, &dw.inputs[3])?;
    let dw_w_shape = dw_w.shape().to_vec();
    if dw_w_shape.len() != 4 {
        return Err(OnnxError::DecodeFailed {
            message: format!("QuantizedPwDw {}: DW weight rank must be 4", dw.name),
        });
    }
    let (c_exp_dw, _, kh, kw) = (dw_w_shape[0], dw_w_shape[1], dw_w_shape[2], dw_w_shape[3]);
    if c_exp_pw != c_exp_dw || kh != kw {
        return Err(OnnxError::DecodeFailed {
            message: format!(
                "QuantizedPwDw {}+{}: c_exp mismatch ({} vs {}) or kh != kw",
                pw.name, dw.name, c_exp_pw, c_exp_dw
            ),
        });
    }

    let strides = crate::runner::get_attr_ints(dw, "strides").unwrap_or_else(|| vec![1, 1]);
    let pads = crate::runner::get_attr_ints(dw, "pads").unwrap_or_else(|| vec![0, 0, 0, 0]);
    if strides.len() != 2 || pads.len() != 4 {
        return Err(OnnxError::DecodeFailed {
            message: format!("QuantizedPwDw {}: unexpected DW strides/pads", dw.name),
        });
    }
    let stride = strides[0] as usize;
    let pad = pads[0] as usize;

    // PW bias is loaded as f32 but represents i32 values per ONNX QLinearConv.
    let pw_bias = if pw.inputs.len() > 8 && !pw.inputs[8].is_empty() {
        let t = get_tensor(env, &pw.name, &pw.inputs[8])?;
        Some(t.data().iter().map(|&v| v as i32).collect::<Vec<i32>>())
    } else {
        None
    };
    let dw_bias = if dw.inputs.len() > 8 && !dw.inputs[8].is_empty() {
        let t = get_tensor(env, &dw.name, &dw.inputs[8])?;
        Some(t.data().iter().map(|&v| v as i32).collect::<Vec<i32>>())
    } else {
        None
    };

    // Resolve PW input. Two valid producers per the existing
    // `exec_qlinear_conv` contract:
    //   * a previous QLinearConv (or `QuantizedQdq` fold) that wrote an
    //     i8 `QuantTensor` — cheap, no per-call cast;
    //   * a `QuantizeLinear` that stored f32 values representing i8
    //     (the special-name path bypasses this; the general path stores
    //     f32) — fall back to `get_tensor` and round/cast.
    let x_quant = if env.static_use_count(&pw.inputs[0]) <= 1 {
        env.take_quant_i8(&pw.inputs[0])
    } else {
        env.get_quant_i8(&pw.inputs[0]).cloned()
    };
    let (x_data, x_shape): (Vec<i8>, Vec<usize>) = match x_quant {
        Some(qt) => (qt.data, qt.shape),
        None => {
            let t = get_tensor(env, &pw.name, &pw.inputs[0])?;
            let shape = t.shape().to_vec();
            let data: Vec<i8> = t.data().iter().map(|&v| v as i8).collect();
            (data, shape)
        }
    };
    if x_shape.len() != 4 {
        return Err(OnnxError::DecodeFailed {
            message: format!(
                "QuantizedPwDw {}: input rank must be 4, got {:?}",
                pw.name, x_shape
            ),
        });
    }
    let (n_n, c_in, ih, iw) = (x_shape[0], x_shape[1], x_shape[2], x_shape[3]);
    if c_in != c_in_pw {
        return Err(OnnxError::DecodeFailed {
            message: format!(
                "QuantizedPwDw {}: input c={} != PW weight c_in={}",
                pw.name, c_in, c_in_pw
            ),
        });
    }
    let out_h = (ih + 2 * pad - kh) / stride + 1;
    let out_w = (iw + 2 * pad - kw) / stride + 1;

    // NCHW -> NHWC i8 transpose at chain entry. The kernel reads the
    // pre-packed PW RHS so it expects contiguous c_in per pixel.
    let mut input_nhwc = vec![0_i8; n_n * ih * iw * c_in];
    nchw_i8_to_nhwc(&x_data, &mut input_nhwc, n_n, c_in, ih, iw);

    let pw_packed = env
        .prepacked_i8_b(&pw.inputs[3])
        .ok_or_else(|| OnnxError::DecodeFailed {
            message: format!(
                "QuantizedPwDw {}: PW weight not prepacked (loader gate broken)",
                pw.name
            ),
        })?;
    if pw_packed.k() != c_in_pw || pw_packed.n() != c_exp_pw {
        return Err(OnnxError::DecodeFailed {
            message: format!(
                "QuantizedPwDw {}: PW prepacked shape {}x{} mismatch with weight",
                pw.name,
                pw_packed.k(),
                pw_packed.n()
            ),
        });
    }
    let dw_weight =
        env.prepacked_i8_depthwise(&dw.inputs[3])
            .ok_or_else(|| OnnxError::DecodeFailed {
                message: format!(
                    "QuantizedPwDw {}: DW weight not prepacked (loader gate broken)",
                    dw.name
                ),
            })?;

    let params = yscv_kernels::Int8FusedPwDwParams {
        batch: n_n,
        in_h: ih,
        in_w: iw,
        c_in: c_in_pw,
        c_exp: c_exp_pw,
        kh,
        stride,
        pad,
        out_h,
        out_w,
        pw_relu: has_relu,
        pw_composite: (pw_x_scale * pw_w_scale) / pw_y_scale,
        dw_composite: (dw_x_scale * dw_w_scale) / dw_y_scale,
        dw_y_zp,
    };
    let mut output_nchw = vec![0_i8; params.output_len()];
    yscv_kernels::int8_fused_pw_dw_dispatch(
        &input_nhwc,
        pw_packed,
        pw_bias.as_deref(),
        dw_weight.as_ref(),
        dw_bias.as_deref(),
        params,
        &mut output_nchw,
        None,
    );

    env.insert_quant_i8(
        dw.outputs[0].clone(),
        QuantTensor {
            data: output_nchw,
            shape: vec![n_n, c_exp_pw, out_h, out_w],
            scale: dw_y_scale,
            zero_point: dw_y_zp,
            nhwc: false,
        },
    );
    crate::runner::note_quant_chain_executed();
    Ok(())
}

pub(crate) fn exec_quantized_pw_dw_fork(
    pw: &OnnxNode,
    dq: &OnnxNode,
    dw: &OnnxNode,
    env: &mut TensorEnv,
    has_relu: bool,
) -> Result<(), OnnxError> {
    let pw_x_scale = get_tensor(env, &pw.name, &pw.inputs[1])?.data()[0];
    let pw_w_scale = get_tensor(env, &pw.name, &pw.inputs[4])?.data()[0];
    let pw_y_scale = get_tensor(env, &pw.name, &pw.inputs[6])?.data()[0];
    let pw_y_zp = get_tensor(env, &pw.name, &pw.inputs[7])?.data()[0];
    let dw_x_scale = get_tensor(env, &dw.name, &dw.inputs[1])?.data()[0];
    let dw_w_scale = get_tensor(env, &dw.name, &dw.inputs[4])?.data()[0];
    let dw_y_scale = get_tensor(env, &dw.name, &dw.inputs[6])?.data()[0];
    let dw_y_zp = get_tensor(env, &dw.name, &dw.inputs[7])?.data()[0];

    let pw_w = get_tensor(env, &pw.name, &pw.inputs[3])?;
    let pw_w_shape = pw_w.shape().to_vec();
    if pw_w_shape.len() != 4 {
        return Err(OnnxError::DecodeFailed {
            message: format!("QuantizedPwDwFork {}: PW weight rank must be 4", pw.name),
        });
    }
    let (c_exp_pw, c_in_pw, _, _) = (pw_w_shape[0], pw_w_shape[1], pw_w_shape[2], pw_w_shape[3]);

    let dw_w = get_tensor(env, &dw.name, &dw.inputs[3])?;
    let dw_w_shape = dw_w.shape().to_vec();
    if dw_w_shape.len() != 4 {
        return Err(OnnxError::DecodeFailed {
            message: format!("QuantizedPwDwFork {}: DW weight rank must be 4", dw.name),
        });
    }
    let (c_exp_dw, _, kh, kw) = (dw_w_shape[0], dw_w_shape[1], dw_w_shape[2], dw_w_shape[3]);
    if c_exp_pw != c_exp_dw || kh != kw {
        return Err(OnnxError::DecodeFailed {
            message: format!(
                "QuantizedPwDwFork {}+{}: c_exp mismatch ({} vs {}) or kh != kw",
                pw.name, dw.name, c_exp_pw, c_exp_dw
            ),
        });
    }

    let strides = crate::runner::get_attr_ints(dw, "strides").unwrap_or_else(|| vec![1, 1]);
    let pads = crate::runner::get_attr_ints(dw, "pads").unwrap_or_else(|| vec![0, 0, 0, 0]);
    if strides.len() != 2 || pads.len() != 4 {
        return Err(OnnxError::DecodeFailed {
            message: format!("QuantizedPwDwFork {}: unexpected DW strides/pads", dw.name),
        });
    }
    let stride = strides[0] as usize;
    let pad = pads[0] as usize;

    let pw_bias = if pw.inputs.len() > 8 && !pw.inputs[8].is_empty() {
        let t = get_tensor(env, &pw.name, &pw.inputs[8])?;
        Some(t.data().iter().map(|&v| v as i32).collect::<Vec<i32>>())
    } else {
        None
    };
    let dw_bias = if dw.inputs.len() > 8 && !dw.inputs[8].is_empty() {
        let t = get_tensor(env, &dw.name, &dw.inputs[8])?;
        Some(t.data().iter().map(|&v| v as i32).collect::<Vec<i32>>())
    } else {
        None
    };

    let x_quant = if env.static_use_count(&pw.inputs[0]) <= 1 {
        env.take_quant_i8(&pw.inputs[0])
    } else {
        env.get_quant_i8(&pw.inputs[0]).cloned()
    };
    let (x_data, x_shape): (Vec<i8>, Vec<usize>) = match x_quant {
        Some(qt) => (qt.data, qt.shape),
        None => {
            let t = get_tensor(env, &pw.name, &pw.inputs[0])?;
            let shape = t.shape().to_vec();
            let data: Vec<i8> = t.data().iter().map(|&v| v as i8).collect();
            (data, shape)
        }
    };
    if x_shape.len() != 4 {
        return Err(OnnxError::DecodeFailed {
            message: format!(
                "QuantizedPwDwFork {}: input rank must be 4, got {:?}",
                pw.name, x_shape
            ),
        });
    }
    let (n_n, c_in, ih, iw) = (x_shape[0], x_shape[1], x_shape[2], x_shape[3]);
    if c_in != c_in_pw {
        return Err(OnnxError::DecodeFailed {
            message: format!(
                "QuantizedPwDwFork {}: input c={} != PW weight c_in={}",
                pw.name, c_in, c_in_pw
            ),
        });
    }
    let out_h = (ih + 2 * pad - kh) / stride + 1;
    let out_w = (iw + 2 * pad - kw) / stride + 1;

    let mut input_nhwc = vec![0_i8; n_n * ih * iw * c_in];
    nchw_i8_to_nhwc(&x_data, &mut input_nhwc, n_n, c_in, ih, iw);

    let pw_packed = env
        .prepacked_i8_b(&pw.inputs[3])
        .ok_or_else(|| OnnxError::DecodeFailed {
            message: format!("QuantizedPwDwFork {}: PW weight not prepacked", pw.name),
        })?;
    let dw_weight =
        env.prepacked_i8_depthwise(&dw.inputs[3])
            .ok_or_else(|| OnnxError::DecodeFailed {
                message: format!("QuantizedPwDwFork {}: DW weight not prepacked", dw.name),
            })?;

    let params = yscv_kernels::Int8FusedPwDwParams {
        batch: n_n,
        in_h: ih,
        in_w: iw,
        c_in: c_in_pw,
        c_exp: c_exp_pw,
        kh,
        stride,
        pad,
        out_h,
        out_w,
        pw_relu: has_relu,
        pw_composite: (pw_x_scale * pw_w_scale) / pw_y_scale,
        dw_composite: (dw_x_scale * dw_w_scale) / dw_y_scale,
        dw_y_zp,
    };
    let mut side_nchw = vec![0_f32; n_n * c_exp_pw * ih * iw];
    let mut output_nchw = vec![0_i8; params.output_len()];
    yscv_kernels::int8_fused_pw_dw_with_pw_side_dispatch(
        &input_nhwc,
        pw_packed,
        pw_bias.as_deref(),
        dw_weight.as_ref(),
        dw_bias.as_deref(),
        params,
        pw_y_scale,
        pw_y_zp,
        &mut side_nchw,
        &mut output_nchw,
        None,
    );

    let side = Tensor::from_vec(vec![n_n, c_exp_pw, ih, iw], side_nchw).map_err(|e| {
        OnnxError::DecodeFailed {
            message: e.to_string(),
        }
    })?;
    let side_name = dq
        .outputs
        .first()
        .cloned()
        .ok_or_else(|| OnnxError::DecodeFailed {
            message: format!("QuantizedPwDwFork {}: DQ has no output", dq.name),
        })?;
    env.insert(side_name, side);
    env.insert_quant_i8(
        dw.outputs[0].clone(),
        QuantTensor {
            data: output_nchw,
            shape: vec![n_n, c_exp_pw, out_h, out_w],
            scale: dw_y_scale,
            zero_point: dw_y_zp,
            nhwc: false,
        },
    );
    crate::runner::note_quant_chain_executed();
    Ok(())
}

pub(crate) fn exec_quantized_dw_residual(
    dw: &OnnxNode,
    conv: &OnnxNode,
    add: &OnnxNode,
    q: &OnnxNode,
    env: &mut TensorEnv,
) -> Result<(), OnnxError> {
    let dw_x_scale = get_tensor(env, &dw.name, &dw.inputs[1])?.data()[0];
    let dw_w_scale = get_tensor(env, &dw.name, &dw.inputs[4])?.data()[0];
    let dw_y_scale = get_tensor(env, &dw.name, &dw.inputs[6])?.data()[0];
    let dw_y_zp = get_tensor(env, &dw.name, &dw.inputs[7])?.data()[0];
    let q_y_scale = get_tensor(env, &q.name, &q.inputs[1])?.data()[0];
    let q_y_zp = get_tensor(env, &q.name, &q.inputs[2])?.data()[0];

    let dw_w = get_tensor(env, &dw.name, &dw.inputs[3])?;
    let dw_w_shape = dw_w.shape().to_vec();
    if dw_w_shape.len() != 4 {
        return Err(OnnxError::DecodeFailed {
            message: format!("QuantizedDwResidual {}: DW weight rank must be 4", dw.name),
        });
    }
    let (c_in_dw, _, kh, kw) = (dw_w_shape[0], dw_w_shape[1], dw_w_shape[2], dw_w_shape[3]);
    if kh != kw {
        return Err(OnnxError::DecodeFailed {
            message: format!("QuantizedDwResidual {}: kh != kw", dw.name),
        });
    }

    let conv_w = get_tensor(env, &conv.name, &conv.inputs[1])?;
    let conv_w_shape = conv_w.shape().to_vec();
    let conv_w_data = conv_w.data().to_vec();
    if conv_w_shape.len() != 4 || conv_w_shape[0] != 1 || conv_w_shape[1] != 1 {
        return Err(OnnxError::DecodeFailed {
            message: format!(
                "QuantizedDwResidual {}: expected KHWC 1x1 Conv weight, got {:?}",
                conv.name, conv_w_shape
            ),
        });
    }
    let (c_in_pw, c_out_pw) = (conv_w_shape[2], conv_w_shape[3]);
    if c_in_pw != c_in_dw {
        return Err(OnnxError::DecodeFailed {
            message: format!(
                "QuantizedDwResidual {}+{}: c mismatch {} vs {}",
                dw.name, conv.name, c_in_dw, c_in_pw
            ),
        });
    }

    let strides = crate::runner::get_attr_ints(dw, "strides").unwrap_or_else(|| vec![1, 1]);
    let pads = crate::runner::get_attr_ints(dw, "pads").unwrap_or_else(|| vec![0, 0, 0, 0]);
    if strides.len() != 2 || pads.len() != 4 {
        return Err(OnnxError::DecodeFailed {
            message: format!(
                "QuantizedDwResidual {}: unexpected DW strides/pads",
                dw.name
            ),
        });
    }
    let stride = strides[0] as usize;
    let pad = pads[0] as usize;

    let dw_bias = if dw.inputs.len() > 8 && !dw.inputs[8].is_empty() {
        let t = get_tensor(env, &dw.name, &dw.inputs[8])?;
        Some(t.data().iter().map(|&v| v as i32).collect::<Vec<i32>>())
    } else {
        None
    };
    let conv_bias = if conv.inputs.len() > 2 && !conv.inputs[2].is_empty() {
        Some(
            get_tensor(env, &conv.name, &conv.inputs[2])?
                .data()
                .to_vec(),
        )
    } else {
        None
    };

    let x_quant = if env.static_use_count(&dw.inputs[0]) <= 1 {
        env.take_quant_i8(&dw.inputs[0])
    } else {
        env.get_quant_i8(&dw.inputs[0]).cloned()
    };
    let (x_data, x_shape): (Vec<i8>, Vec<usize>) = match x_quant {
        Some(qt) => (qt.data, qt.shape),
        None => {
            let t = get_tensor(env, &dw.name, &dw.inputs[0])?;
            let shape = t.shape().to_vec();
            let data: Vec<i8> = t.data().iter().map(|&v| v as i8).collect();
            (data, shape)
        }
    };
    if x_shape.len() != 4 {
        return Err(OnnxError::DecodeFailed {
            message: format!(
                "QuantizedDwResidual {}: input rank must be 4, got {:?}",
                dw.name, x_shape
            ),
        });
    }
    let (n_n, c_in, ih, iw) = (x_shape[0], x_shape[1], x_shape[2], x_shape[3]);
    if c_in != c_in_dw {
        return Err(OnnxError::DecodeFailed {
            message: format!(
                "QuantizedDwResidual {}: input c={} != DW c={}",
                dw.name, c_in, c_in_dw
            ),
        });
    }
    let out_h = (ih + 2 * pad - kh) / stride + 1;
    let out_w = (iw + 2 * pad - kw) / stride + 1;
    let m = n_n * out_h * out_w;

    let mut input_nhwc = vec![0_i8; n_n * ih * iw * c_in];
    nchw_i8_to_nhwc(&x_data, &mut input_nhwc, n_n, c_in, ih, iw);

    let dw_weight =
        env.prepacked_i8_depthwise(&dw.inputs[3])
            .ok_or_else(|| OnnxError::DecodeFailed {
                message: format!("QuantizedDwResidual {}: DW weight not prepacked", dw.name),
            })?;
    let dw_params = yscv_kernels::DepthwiseI8Params {
        batch: n_n,
        in_h: ih,
        in_w: iw,
        channels: c_in_dw,
        kernel: kh,
        stride_h: stride,
        stride_w: stride,
        pad_top: pad,
        pad_left: pad,
        out_h,
        out_w,
    };
    let mut dw_acc = vec![0_i32; m * c_in_dw];
    yscv_kernels::depthwise_i8_i32_nhwc_dispatch_with_pool(
        &input_nhwc,
        dw_weight.as_ref(),
        dw_params,
        &mut dw_acc,
        None,
    );
    if let Some(bias) = dw_bias.as_deref() {
        dw_acc.par_chunks_mut(c_in_dw).for_each(|row| {
            for (v, b) in row.iter_mut().zip(bias.iter()) {
                *v += *b;
            }
        });
    }
    let dw_composite = (dw_x_scale * dw_w_scale) / dw_y_scale;
    let mut dw_i8 = vec![0_i8; m * c_in_dw];
    dw_i8
        .par_chunks_mut(c_in_dw)
        .enumerate()
        .for_each(|(row, dst)| {
            let src = &dw_acc[row * c_in_dw..(row + 1) * c_in_dw];
            yscv_kernels::requant_i32_row_to_i8_dispatch(
                src,
                None,
                dw_composite,
                dw_y_zp,
                true,
                dst,
                c_in_dw,
            );
        });
    let mut pw_input = vec![0_f32; m * c_in_dw];
    pw_input
        .par_iter_mut()
        .zip(dw_i8.par_iter())
        .for_each(|(dst, &qv)| {
            *dst = (qv as f32 - dw_y_zp) * dw_y_scale;
        });

    let conv_out_name = conv
        .outputs
        .first()
        .ok_or_else(|| OnnxError::DecodeFailed {
            message: format!("QuantizedDwResidual {}: Conv has no output", conv.name),
        })?;
    let residual_name = add
        .inputs
        .iter()
        .find(|name| *name != conv_out_name)
        .ok_or_else(|| OnnxError::DecodeFailed {
            message: format!("QuantizedDwResidual {}: Add residual missing", add.name),
        })?;
    let residual = get_tensor(env, &add.name, residual_name)?;
    let residual_shape = residual.shape().to_vec();
    let mut residual_nhwc;
    let residual_ptr = if env.is_nhwc(residual_name) {
        residual_nhwc = residual.data().to_vec();
        residual_nhwc.as_ptr()
    } else if residual_shape == [n_n, c_out_pw, out_h, out_w] {
        residual_nhwc = vec![0_f32; m * c_out_pw];
        let out_spatial = out_h * out_w;
        if !cfg!(miri) && residual_nhwc.len() >= 16_384 && rayon::current_num_threads() > 1 {
            residual_nhwc
                .par_chunks_mut(c_out_pw)
                .enumerate()
                .for_each(|(pixel, dst)| {
                    let n = pixel / out_spatial;
                    let rem = pixel % out_spatial;
                    let y = rem / out_w;
                    let x = rem % out_w;
                    for (c, d) in dst.iter_mut().enumerate() {
                        *d = residual.data()[((n * c_out_pw + c) * out_h + y) * out_w + x];
                    }
                });
        } else {
            for n in 0..n_n {
                for c in 0..c_out_pw {
                    for y in 0..out_h {
                        for x in 0..out_w {
                            let src = ((n * c_out_pw + c) * out_h + y) * out_w + x;
                            let dst = ((n * out_h + y) * out_w + x) * c_out_pw + c;
                            residual_nhwc[dst] = residual.data()[src];
                        }
                    }
                }
            }
        }
        residual_nhwc.as_ptr()
    } else {
        return Err(OnnxError::DecodeFailed {
            message: format!(
                "QuantizedDwResidual {}: residual shape mismatch {:?}",
                add.name, residual_shape
            ),
        });
    };

    let mut conv_add_nhwc = vec![0_f32; m * c_out_pw];
    let epilogue = yscv_kernels::GemmEpilogue {
        bias: conv_bias.as_ref().map(|b| b.as_ptr()),
        activation: yscv_kernels::Activation::None,
        residual: Some(residual_ptr),
    };
    let config = yscv_kernels::ParallelMatmulConfig {
        min_parallel_shared_dim: 1,
        min_parallel_output_elements: 4096,
    };
    yscv_kernels::matmul_2d_slices_fused_maybe_packed(
        &pw_input,
        m,
        c_in_dw,
        &conv_w_data,
        c_out_pw,
        &mut conv_add_nhwc,
        env.prepacked_b(&conv.inputs[1]),
        epilogue,
        config,
        None,
    );

    let mut q_nhwc = vec![0_i8; m * c_out_pw];
    let nthreads = rayon::current_num_threads().max(1);
    if !cfg!(miri) && q_nhwc.len() >= 65_536 && nthreads > 1 {
        let chunk = q_nhwc.len().div_ceil(nthreads * 2).max(16_384);
        q_nhwc
            .par_chunks_mut(chunk)
            .zip(conv_add_nhwc.par_chunks(chunk))
            .for_each(|(dst, src)| {
                yscv_kernels::quantize_linear_f32_to_i8_dispatch(src, q_y_scale, q_y_zp, dst);
            });
    } else {
        yscv_kernels::quantize_linear_f32_to_i8_dispatch(
            &conv_add_nhwc,
            q_y_scale,
            q_y_zp,
            &mut q_nhwc,
        );
    }
    let mut q_nchw = vec![0_i8; m * c_out_pw];
    nhwc_i8_to_nchw(&q_nhwc, &mut q_nchw, n_n, c_out_pw, out_h, out_w);
    env.insert_quant_i8(
        q.outputs[0].clone(),
        QuantTensor {
            data: q_nchw,
            shape: vec![n_n, c_out_pw, out_h, out_w],
            scale: q_y_scale,
            zero_point: q_y_zp,
            nhwc: false,
        },
    );
    crate::runner::note_quant_chain_executed();
    Ok(())
}

pub(crate) fn exec_quantized_pw_dq(
    qconv: &OnnxNode,
    dq: &OnnxNode,
    env: &mut TensorEnv,
) -> Result<(), OnnxError> {
    let x_scale = get_tensor(env, &qconv.name, &qconv.inputs[1])?.data()[0];
    let w_scale = get_tensor(env, &qconv.name, &qconv.inputs[4])?.data()[0];
    let y_scale = get_tensor(env, &qconv.name, &qconv.inputs[6])?.data()[0];
    let y_zp = get_tensor(env, &qconv.name, &qconv.inputs[7])?.data()[0];

    let w = get_tensor(env, &qconv.name, &qconv.inputs[3])?;
    let w_shape = w.shape().to_vec();
    if w_shape.len() != 4 || w_shape[2] != 1 || w_shape[3] != 1 {
        return Err(OnnxError::DecodeFailed {
            message: format!(
                "QuantizedPwDq {}: expected OIHW 1x1 weight, got {:?}",
                qconv.name, w_shape
            ),
        });
    }
    let (c_out, c_in_w) = (w_shape[0], w_shape[1]);
    let bias = if qconv.inputs.len() > 8 && !qconv.inputs[8].is_empty() {
        let t = get_tensor(env, &qconv.name, &qconv.inputs[8])?;
        Some(t.data().iter().map(|&v| v as i32).collect::<Vec<i32>>())
    } else {
        None
    };

    let x_quant = if env.static_use_count(&qconv.inputs[0]) <= 1 {
        env.take_quant_i8(&qconv.inputs[0])
    } else {
        env.get_quant_i8(&qconv.inputs[0]).cloned()
    };
    let (x_data, x_shape): (Vec<i8>, Vec<usize>) = match x_quant {
        Some(qt) => (qt.data, qt.shape),
        None => {
            let t = get_tensor(env, &qconv.name, &qconv.inputs[0])?;
            let shape = t.shape().to_vec();
            let data: Vec<i8> = t.data().iter().map(|&v| v as i8).collect();
            (data, shape)
        }
    };
    if x_shape.len() != 4 {
        return Err(OnnxError::DecodeFailed {
            message: format!(
                "QuantizedPwDq {}: input rank must be 4, got {:?}",
                qconv.name, x_shape
            ),
        });
    }
    let (n_n, c_in, h, w_in) = (x_shape[0], x_shape[1], x_shape[2], x_shape[3]);
    if c_in != c_in_w {
        return Err(OnnxError::DecodeFailed {
            message: format!(
                "QuantizedPwDq {}: input c={} != weight c={}",
                qconv.name, c_in, c_in_w
            ),
        });
    }

    let mut input_nhwc = vec![0_i8; n_n * h * w_in * c_in];
    nchw_i8_to_nhwc(&x_data, &mut input_nhwc, n_n, c_in, h, w_in);

    let packed = env
        .prepacked_i8_b(&qconv.inputs[3])
        .ok_or_else(|| OnnxError::DecodeFailed {
            message: format!("QuantizedPwDq {}: PW weight not prepacked", qconv.name),
        })?;
    let m = n_n * h * w_in;
    let mut acc = vec![0_i32; m * c_out];
    yscv_kernels::int8_matmul_prepacked_dispatch(&input_nhwc, packed, m, &mut acc);
    let mut q_i8 = vec![0_i8; m * c_out];
    let composite = (x_scale * w_scale) / y_scale;
    let nthreads = rayon::current_num_threads().max(1);
    if !cfg!(miri) && q_i8.len() >= 16_384 && nthreads > 1 {
        q_i8.par_chunks_mut(c_out)
            .enumerate()
            .take(m)
            .for_each(|(row, dst)| {
                let src = &acc[row * c_out..(row + 1) * c_out];
                yscv_kernels::requant_i32_row_to_i8_dispatch(
                    src,
                    bias.as_deref(),
                    composite,
                    y_zp,
                    false,
                    dst,
                    c_out,
                );
            });
    } else {
        for row in 0..m {
            let src = &acc[row * c_out..(row + 1) * c_out];
            let dst = &mut q_i8[row * c_out..(row + 1) * c_out];
            yscv_kernels::requant_i32_row_to_i8_dispatch(
                src,
                bias.as_deref(),
                composite,
                y_zp,
                false,
                dst,
                c_out,
            );
        }
    }
    let mut out_nchw = vec![0_f32; m * c_out];
    let plane = h * w_in;
    if !cfg!(miri) && out_nchw.len() >= 16_384 && nthreads > 1 {
        out_nchw
            .par_chunks_mut(plane)
            .enumerate()
            .for_each(|(plane_idx, dst_plane)| {
                let n = plane_idx / c_out;
                let c = plane_idx % c_out;
                for y in 0..h {
                    for x in 0..w_in {
                        let src = ((n * h + y) * w_in + x) * c_out + c;
                        dst_plane[y * w_in + x] = (q_i8[src] as f32 - y_zp) * y_scale;
                    }
                }
            });
    } else {
        for n in 0..n_n {
            for c in 0..c_out {
                for y in 0..h {
                    for x in 0..w_in {
                        let src = ((n * h + y) * w_in + x) * c_out + c;
                        let dst = ((n * c_out + c) * h + y) * w_in + x;
                        out_nchw[dst] = (q_i8[src] as f32 - y_zp) * y_scale;
                    }
                }
            }
        }
    }
    let out_name = dq
        .outputs
        .first()
        .cloned()
        .ok_or_else(|| OnnxError::DecodeFailed {
            message: format!("QuantizedPwDq {}: DQ has no output", dq.name),
        })?;
    let tensor = Tensor::from_vec(vec![n_n, c_out, h, w_in], out_nchw).map_err(|e| {
        OnnxError::DecodeFailed {
            message: e.to_string(),
        }
    })?;
    env.insert(out_name, tensor);
    crate::runner::note_quant_chain_executed();
    Ok(())
}

/// Dispatch the fused INT8 DW->PW chain for the action emitted by the
/// loader's `QuantizedDwPw` detector. Mirror of [`exec_quantized_pw_dw`]
/// for the closing pair of an inverted bottleneck.
///
/// Caller contract (loader-enforced gates):
/// * `dw` is a 3×3/5×5 depthwise `QLinearConv` with
///   `x_zp = w_zp = y_zp = 0`, dilations [1,1], symmetric pad and
///   stride 1 or 2;
/// * `pw` is a 1×1 group-1 `QLinearConv` with `x_zp = w_zp = 0`
///   (`y_zp` is the chain output and may be non-zero);
/// * pre-packed DW (KHWC i8) and PW (VNNI 4×16) weights are present
///   in `env`.
///
/// Output is inserted as a `QuantTensor` keyed by `pw.outputs[0]`,
/// shape `[N, c_out, out_h, out_w]`, scale = `pw.y_scale`,
/// zero-point = `pw.y_zp`. Bitwise-identical to running DW + QuantizedQdq
/// + PW in sequence.
pub(crate) fn exec_quantized_dw_pw(
    dw: &OnnxNode,
    pw: &OnnxNode,
    env: &mut TensorEnv,
    has_relu: bool,
) -> Result<(), OnnxError> {
    let dw_x_scale = get_tensor(env, &dw.name, &dw.inputs[1])?.data()[0];
    let dw_w_scale = get_tensor(env, &dw.name, &dw.inputs[4])?.data()[0];
    let dw_y_scale = get_tensor(env, &dw.name, &dw.inputs[6])?.data()[0];
    let pw_x_scale = get_tensor(env, &pw.name, &pw.inputs[1])?.data()[0];
    let pw_w_scale = get_tensor(env, &pw.name, &pw.inputs[4])?.data()[0];
    let pw_y_scale = get_tensor(env, &pw.name, &pw.inputs[6])?.data()[0];
    let pw_y_zp = get_tensor(env, &pw.name, &pw.inputs[7])?.data()[0];

    let dw_w = get_tensor(env, &dw.name, &dw.inputs[3])?;
    let dw_w_shape = dw_w.shape().to_vec();
    if dw_w_shape.len() != 4 {
        return Err(OnnxError::DecodeFailed {
            message: format!("QuantizedDwPw {}: DW weight rank must be 4", dw.name),
        });
    }
    let (c_in_dw, _, kh, kw) = (dw_w_shape[0], dw_w_shape[1], dw_w_shape[2], dw_w_shape[3]);
    if kh != kw {
        return Err(OnnxError::DecodeFailed {
            message: format!("QuantizedDwPw {}: kh != kw", dw.name),
        });
    }

    let pw_w = get_tensor(env, &pw.name, &pw.inputs[3])?;
    let pw_w_shape = pw_w.shape().to_vec();
    if pw_w_shape.len() != 4 {
        return Err(OnnxError::DecodeFailed {
            message: format!("QuantizedDwPw {}: PW weight rank must be 4", pw.name),
        });
    }
    let (c_out_pw, c_in_pw, _, _) = (pw_w_shape[0], pw_w_shape[1], pw_w_shape[2], pw_w_shape[3]);
    if c_in_pw != c_in_dw {
        return Err(OnnxError::DecodeFailed {
            message: format!(
                "QuantizedDwPw {}+{}: c_in mismatch (DW={}, PW K={})",
                dw.name, pw.name, c_in_dw, c_in_pw
            ),
        });
    }

    let strides = crate::runner::get_attr_ints(dw, "strides").unwrap_or_else(|| vec![1, 1]);
    let pads = crate::runner::get_attr_ints(dw, "pads").unwrap_or_else(|| vec![0, 0, 0, 0]);
    if strides.len() != 2 || pads.len() != 4 {
        return Err(OnnxError::DecodeFailed {
            message: format!("QuantizedDwPw {}: unexpected DW strides/pads", dw.name),
        });
    }
    let stride = strides[0] as usize;
    let pad = pads[0] as usize;

    // Biases are stored as f32 with integral values per ONNX spec.
    let dw_bias = if dw.inputs.len() > 8 && !dw.inputs[8].is_empty() {
        let t = get_tensor(env, &dw.name, &dw.inputs[8])?;
        Some(t.data().iter().map(|&v| v as i32).collect::<Vec<i32>>())
    } else {
        None
    };
    let pw_bias = if pw.inputs.len() > 8 && !pw.inputs[8].is_empty() {
        let t = get_tensor(env, &pw.name, &pw.inputs[8])?;
        Some(t.data().iter().map(|&v| v as i32).collect::<Vec<i32>>())
    } else {
        None
    };

    // Resolve DW input. Two valid producers (mirrors `exec_quantized_pw_dw`):
    //   * a previous QLinearConv / chain action that wrote an i8
    //     `QuantTensor` (cheap path);
    //   * a `QuantizeLinear` (or fp32 producer) that stored f32
    //     representing i8 — fall back to `get_tensor` and round/cast.
    let x_quant = if env.static_use_count(&dw.inputs[0]) <= 1 {
        env.take_quant_i8(&dw.inputs[0])
    } else {
        env.get_quant_i8(&dw.inputs[0]).cloned()
    };
    let (x_data, x_shape): (Vec<i8>, Vec<usize>) = match x_quant {
        Some(qt) => (qt.data, qt.shape),
        None => {
            let t = get_tensor(env, &dw.name, &dw.inputs[0])?;
            let shape = t.shape().to_vec();
            let data: Vec<i8> = t.data().iter().map(|&v| v as i8).collect();
            (data, shape)
        }
    };
    if x_shape.len() != 4 {
        return Err(OnnxError::DecodeFailed {
            message: format!(
                "QuantizedDwPw {}: input rank must be 4, got {:?}",
                dw.name, x_shape
            ),
        });
    }
    let (n_n, c_in, ih, iw) = (x_shape[0], x_shape[1], x_shape[2], x_shape[3]);
    if c_in != c_in_dw {
        return Err(OnnxError::DecodeFailed {
            message: format!(
                "QuantizedDwPw {}: input c={} != DW c_in={}",
                dw.name, c_in, c_in_dw
            ),
        });
    }
    let out_h = (ih + 2 * pad - kh) / stride + 1;
    let out_w = (iw + 2 * pad - kw) / stride + 1;

    // NCHW -> NHWC i8 transpose at chain entry. The kernel addresses
    // input pixels as contiguous `c_in` runs.
    let mut input_nhwc = vec![0_i8; n_n * ih * iw * c_in];
    for n in 0..n_n {
        for c in 0..c_in {
            for y in 0..ih {
                for x in 0..iw {
                    let src = ((n * c_in + c) * ih + y) * iw + x;
                    let dst = ((n * ih + y) * iw + x) * c_in + c;
                    input_nhwc[dst] = x_data[src];
                }
            }
        }
    }

    let dw_weight =
        env.prepacked_i8_depthwise(&dw.inputs[3])
            .ok_or_else(|| OnnxError::DecodeFailed {
                message: format!(
                    "QuantizedDwPw {}: DW weight not prepacked (loader gate broken)",
                    dw.name
                ),
            })?;
    let pw_packed = env
        .prepacked_i8_b(&pw.inputs[3])
        .ok_or_else(|| OnnxError::DecodeFailed {
            message: format!(
                "QuantizedDwPw {}: PW weight not prepacked (loader gate broken)",
                pw.name
            ),
        })?;
    if pw_packed.k() != c_in_dw || pw_packed.n() != c_out_pw {
        return Err(OnnxError::DecodeFailed {
            message: format!(
                "QuantizedDwPw {}: PW prepacked shape {}x{} mismatch with weight",
                pw.name,
                pw_packed.k(),
                pw_packed.n()
            ),
        });
    }

    let params = yscv_kernels::Int8FusedDwPwParams {
        batch: n_n,
        in_h: ih,
        in_w: iw,
        c_in: c_in_dw,
        c_out: c_out_pw,
        kh,
        stride,
        pad,
        out_h,
        out_w,
        dw_relu: has_relu,
        dw_composite: (dw_x_scale * dw_w_scale) / dw_y_scale,
        pw_composite: (pw_x_scale * pw_w_scale) / pw_y_scale,
        pw_y_zp,
    };
    let mut output_nchw = vec![0_i8; params.output_len()];
    yscv_kernels::int8_fused_dw_pw_dispatch(
        &input_nhwc,
        dw_weight.as_ref(),
        dw_bias.as_deref(),
        pw_packed,
        pw_bias.as_deref(),
        params,
        &mut output_nchw,
        None,
    );

    env.insert_quant_i8(
        pw.outputs[0].clone(),
        QuantTensor {
            data: output_nchw,
            shape: vec![n_n, c_out_pw, out_h, out_w],
            scale: pw_y_scale,
            zero_point: pw_y_zp,
            nhwc: false,
        },
    );
    crate::runner::note_quant_chain_executed();
    Ok(())
}

pub(crate) fn exec_conv_integer(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
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

    // Symmetric int8 fast path: NCHW input + OIHW weight + group=1 +
    // dilations=[1,1] + zero-points 0 → integer im2col + GEMM, no
    // requantize (ConvInteger output is raw int32). Same gate / layout
    // checks as `exec_qlinear_conv`.
    if x_zp == 0.0 && w_zp == 0.0 && x.shape().len() == 4 && w.shape().len() == 4 {
        let group = crate::runner::get_attr_int(node, "group").unwrap_or(1);
        let dilations =
            crate::runner::get_attr_ints(node, "dilations").unwrap_or_else(|| vec![1, 1]);
        if group == 1 && dilations == [1, 1] {
            let pads =
                crate::runner::get_attr_ints(node, "pads").unwrap_or_else(|| vec![0, 0, 0, 0]);
            let strides =
                crate::runner::get_attr_ints(node, "strides").unwrap_or_else(|| vec![1, 1]);
            let xs = x.shape();
            let ws = w.shape();
            let (n_n, c_in, ih, iw) = (xs[0], xs[1], xs[2], xs[3]);
            let (c_out, _, kh, kw) = (ws[0], ws[1], ws[2], ws[3]);
            if ws[1] == c_in && pads.len() == 4 && strides.len() == 2 {
                let (pt, pl, pb, pr) = (
                    pads[0] as usize,
                    pads[1] as usize,
                    pads[2] as usize,
                    pads[3] as usize,
                );
                let (sh, sw) = (strides[0] as usize, strides[1] as usize);
                let oh = (ih + pt + pb - kh) / sh + 1;
                let ow = (iw + pl + pr - kw) / sw + 1;
                let m = n_n * oh * ow;
                let k_dim = c_in * kh * kw;

                let x_data = x.data();
                let mut x_im2col: Vec<i8> = vec![0; m * k_dim];
                for ni in 0..n_n {
                    for oh_i in 0..oh {
                        for ow_i in 0..ow {
                            let row = (ni * oh + oh_i) * ow + ow_i;
                            for ci in 0..c_in {
                                for ky in 0..kh {
                                    for kx in 0..kw {
                                        let ih_i = oh_i * sh + ky;
                                        let iw_i = ow_i * sw + kx;
                                        let col = (ci * kh + ky) * kw + kx;
                                        let idx = row * k_dim + col;
                                        if ih_i >= pt
                                            && ih_i < pt + ih
                                            && iw_i >= pl
                                            && iw_i < pl + iw
                                        {
                                            let h = ih_i - pt;
                                            let v = iw_i - pl;
                                            let src = ((ni * c_in + ci) * ih + h) * iw + v;
                                            x_im2col[idx] = x_data[src] as i8;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                let mut acc = vec![0_i32; m * c_out];
                if crate::runner::should_use_prepacked_i8_b(m, k_dim, c_out)
                    && let Some(packed) = env.prepacked_i8_b(&node.inputs[1])
                {
                    if packed.k() == k_dim && packed.n() == c_out {
                        yscv_kernels::int8_matmul_prepacked_dispatch(
                            &x_im2col, packed, m, &mut acc,
                        );
                    } else {
                        return Err(OnnxError::DecodeFailed {
                            message: format!(
                                "ConvInteger {}: prepacked weight shape mismatch",
                                node.name
                            ),
                        });
                    }
                } else {
                    let w_data = w.data();
                    let mut w_packed: Vec<i8> = vec![0; k_dim * c_out];
                    for o in 0..c_out {
                        for ci in 0..c_in {
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let src = ((o * c_in + ci) * kh + ky) * kw + kx;
                                    let dst_k = (ci * kh + ky) * kw + kx;
                                    w_packed[dst_k * c_out + o] = w_data[src] as i8;
                                }
                            }
                        }
                    }
                    yscv_kernels::int8_matmul_dispatch(
                        &x_im2col, &w_packed, m, k_dim, c_out, &mut acc,
                    );
                }
                // Reshape [M, O] → NCHW [N, O, OH, OW] without requantize
                // (ConvInteger emits raw i32).
                let mut out = vec![0.0_f32; n_n * c_out * oh * ow];
                for ni in 0..n_n {
                    for o in 0..c_out {
                        for oh_i in 0..oh {
                            for ow_i in 0..ow {
                                let row = (ni * oh + oh_i) * ow + ow_i;
                                let dst = ((ni * c_out + o) * oh + oh_i) * ow + ow_i;
                                out[dst] = acc[row * c_out + o] as f32;
                            }
                        }
                    }
                }
                let out_t = Tensor::from_vec(vec![n_n, c_out, oh, ow], out).map_err(|e| {
                    OnnxError::DecodeFailed {
                        message: e.to_string(),
                    }
                })?;
                env.insert(node.outputs[0].clone(), out_t);
                return Ok(());
            }
        }
    }

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
