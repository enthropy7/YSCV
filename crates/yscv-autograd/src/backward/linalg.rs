use yscv_tensor::Tensor;

use super::error::AutogradError;
use super::graph::Graph;
use super::node::NodeId;

/// MatMul2D backward.
pub(crate) fn matmul2d_backward(
    graph: &mut Graph,
    upstream: Tensor,
    left: NodeId,
    right: NodeId,
) -> Result<(), AutogradError> {
    let (left_grad, right_grad) = {
        let lv = &graph.nodes[left.0].value;
        let rv = &graph.nodes[right.0].value;
        let rt = graph.dispatch_transpose_2d(rv)?;
        let lt = graph.dispatch_transpose_2d(lv)?;
        (
            graph.dispatch_matmul_2d(&upstream, &rt)?,
            graph.dispatch_matmul_2d(&lt, &upstream)?,
        )
    };
    graph.accumulate_grad(left, left_grad)?;
    graph.accumulate_grad(right, right_grad)?;
    Ok(())
}

/// Transpose2D backward.
pub(crate) fn transpose2d_backward(
    graph: &mut Graph,
    upstream: Tensor,
    input: NodeId,
) -> Result<(), AutogradError> {
    let input_grad = graph.dispatch_transpose_2d(&upstream)?;
    graph.accumulate_grad(input, input_grad)?;
    Ok(())
}

/// Conv2d NHWC backward: computes grad_input, grad_weight, grad_bias.
pub(crate) fn conv2d_nhwc_backward(
    graph: &mut Graph,
    upstream: &Tensor,
    input_id: NodeId,
    weight_id: NodeId,
    bias_id: Option<NodeId>,
    stride_h: usize,
    stride_w: usize,
) -> Result<(), AutogradError> {
    // Try BackwardOps trait first for GPU-accelerated conv2d backward.
    // Compute GPU results first (immutable borrow), then accumulate (mutable borrow).
    {
        let gpu_result = if let Some(ref backend) = graph.backend {
            let iv = &graph.nodes[input_id.0].value;
            let wv = &graph.nodes[weight_id.0].value;
            let w_shape = wv.shape().to_vec();
            let c_out = *w_shape.last().unwrap_or(&0);
            let needs_weight = graph.nodes[weight_id.0].requires_grad;
            let needs_bias = bias_id
                .map(|b| graph.nodes[b.0].requires_grad)
                .unwrap_or(false);

            let gw = if needs_weight {
                match backend.conv2d_weight_backward(upstream, iv, &w_shape, stride_h, stride_w) {
                    Ok(t) => Some(t),
                    Err(_e) => {
                        #[cfg(debug_assertions)]
                        eprintln!("[autograd] conv2d_weight_backward GPU fallback: {_e}");
                        None
                    }
                }
            } else {
                Some(Tensor::zeros(vec![1])?) // placeholder, won't be used
            };

            let gb = if gw.is_some() && needs_bias {
                match backend.conv2d_bias_backward(upstream, c_out) {
                    Ok(t) => Some(t),
                    Err(_e) => {
                        #[cfg(debug_assertions)]
                        eprintln!("[autograd] conv2d_bias_backward GPU fallback: {_e}");
                        None
                    }
                }
            } else if gw.is_some() {
                Some(Tensor::zeros(vec![1])?) // placeholder, won't be used
            } else {
                None
            };

            match (gw, gb) {
                (Some(w), Some(b)) if needs_weight || needs_bias => {
                    Some((w, b, needs_weight, needs_bias))
                }
                _ => None,
            }
        } else {
            None
        };

        if let Some((gw, gb, needs_weight, needs_bias)) = gpu_result {
            if needs_weight {
                graph.accumulate_grad(weight_id, gw)?;
            }
            if needs_bias && let Some(b_id) = bias_id {
                graph.accumulate_grad(b_id, gb)?;
            }
            // GPU handled weight+bias; still need CPU for input grad.
            // Skip the weight/bias parts of CPU fallback below.
            let iv = &graph.nodes[input_id.0].value;
            let wv = &graph.nodes[weight_id.0].value;
            if graph.nodes[input_id.0].requires_grad {
                let in_shape = iv.shape();
                let w_shape = wv.shape();
                if in_shape.len() >= 4 && w_shape.len() >= 4 {
                    let (n, ih, iw, c_in) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
                    let (kh, kw, _, c_out) = (w_shape[0], w_shape[1], w_shape[2], w_shape[3]);
                    let (oh, ow) = (upstream.shape()[1], upstream.shape()[2]);
                    let up_data = upstream.data();
                    let w_data = wv.data();
                    let mut gi = vec![0.0f32; n * ih * iw * c_in];
                    for batch in 0..n {
                        for or in 0..oh {
                            for oc_col in 0..ow {
                                for ki in 0..kh {
                                    for kj in 0..kw {
                                        let ir = or * stride_h + ki;
                                        let ic = oc_col * stride_w + kj;
                                        for co in 0..c_out {
                                            let up_v = up_data
                                                [((batch * oh + or) * ow + oc_col) * c_out + co];
                                            for ci in 0..c_in {
                                                gi[((batch * ih + ir) * iw + ic) * c_in + ci] +=
                                                    up_v * w_data
                                                        [((ki * kw + kj) * c_in + ci) * c_out + co];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    let gi_tensor = Tensor::from_vec(in_shape.to_vec(), gi)?;
                    graph.accumulate_grad(input_id, gi_tensor)?;
                }
            }
            return Ok(());
        }
    }
    let up_data = upstream.data();
    let (grad_weight, grad_input, grad_bias) = {
        let iv = &graph.nodes[input_id.0].value;
        let wv = &graph.nodes[weight_id.0].value;
        let in_shape = iv.shape();
        let w_shape = wv.shape();
        if in_shape.len() < 4 || w_shape.len() < 4 {
            return Ok(());
        }
        let (n, ih, iw, c_in) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
        let (kh, kw, _, c_out) = (w_shape[0], w_shape[1], w_shape[2], w_shape[3]);
        let up_shape = upstream.shape();
        if up_shape.len() < 3 {
            return Ok(());
        }
        let (oh, ow) = (up_shape[1], up_shape[2]);
        let in_data = iv.data();
        let w_data = wv.data();

        let gw = if graph.nodes[weight_id.0].requires_grad {
            let mut gw = vec![0.0f32; kh * kw * c_in * c_out];
            for batch in 0..n {
                for or in 0..oh {
                    for oc in 0..ow {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let ir = or * stride_h + ki;
                                let ic = oc * stride_w + kj;
                                for ci in 0..c_in {
                                    let in_idx = ((batch * ih + ir) * iw + ic) * c_in + ci;
                                    let in_v = in_data[in_idx];
                                    for co in 0..c_out {
                                        let up_idx = ((batch * oh + or) * ow + oc) * c_out + co;
                                        let gw_idx = ((ki * kw + kj) * c_in + ci) * c_out + co;
                                        gw[gw_idx] += in_v * up_data[up_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Some(Tensor::from_vec(vec![kh, kw, c_in, c_out], gw)?)
        } else {
            None
        };

        let gi = if graph.nodes[input_id.0].requires_grad {
            let mut gi = vec![0.0f32; n * ih * iw * c_in];
            for batch in 0..n {
                for or in 0..oh {
                    for oc_col in 0..ow {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let ir = or * stride_h + ki;
                                let ic = oc_col * stride_w + kj;
                                for ci in 0..c_in {
                                    let gi_idx = ((batch * ih + ir) * iw + ic) * c_in + ci;
                                    for co in 0..c_out {
                                        let up_idx = ((batch * oh + or) * ow + oc_col) * c_out + co;
                                        let w_idx = ((ki * kw + kj) * c_in + ci) * c_out + co;
                                        gi[gi_idx] += w_data[w_idx] * up_data[up_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Some(Tensor::from_vec(vec![n, ih, iw, c_in], gi)?)
        } else {
            None
        };

        let gb = if let Some(b_id) = bias_id
            && graph.nodes[b_id.0].requires_grad
        {
            let mut gb = vec![0.0f32; c_out];
            up_data
                .iter()
                .enumerate()
                .for_each(|(i, &v)| gb[i % c_out] += v);
            Some(Tensor::from_vec(vec![c_out], gb)?)
        } else {
            None
        };

        (gw, gi, gb)
    };

    if let Some(gw) = grad_weight {
        graph.accumulate_grad(weight_id, gw)?;
    }
    if let Some(gi) = grad_input {
        graph.accumulate_grad(input_id, gi)?;
    }
    if let (Some(b_id), Some(gb)) = (bias_id, grad_bias) {
        graph.accumulate_grad(b_id, gb)?;
    }

    Ok(())
}

/// Depthwise Conv2d NHWC backward.
pub(crate) fn depthwise_conv2d_nhwc_backward(
    graph: &mut Graph,
    upstream: &Tensor,
    input_id: NodeId,
    weight_id: NodeId,
    bias_id: Option<NodeId>,
    stride_h: usize,
    stride_w: usize,
) -> Result<(), AutogradError> {
    let up_data = upstream.data();
    let (grad_weight, grad_input, grad_bias) = {
        let iv = &graph.nodes[input_id.0].value;
        let wv = &graph.nodes[weight_id.0].value;
        let in_shape = iv.shape();
        let w_shape = wv.shape();
        if in_shape.len() < 4 || w_shape.len() < 2 {
            return Ok(());
        }
        let (n, ih, iw, c) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
        let (kh, kw) = (w_shape[0], w_shape[1]);
        let up_shape = upstream.shape();
        if up_shape.len() < 3 {
            return Ok(());
        }
        let (oh, ow) = (up_shape[1], up_shape[2]);
        let in_data = iv.data();
        let w_data = wv.data();

        let gw = if graph.nodes[weight_id.0].requires_grad {
            let mut gw = vec![0.0f32; kh * kw * c];
            for batch in 0..n {
                for or in 0..oh {
                    for oc in 0..ow {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let ir = or * stride_h + ki;
                                let ic = oc * stride_w + kj;
                                for ch in 0..c {
                                    let in_idx = ((batch * ih + ir) * iw + ic) * c + ch;
                                    let up_idx = ((batch * oh + or) * ow + oc) * c + ch;
                                    let gw_idx = (ki * kw + kj) * c + ch;
                                    gw[gw_idx] += in_data[in_idx] * up_data[up_idx];
                                }
                            }
                        }
                    }
                }
            }
            Some(Tensor::from_vec(vec![kh, kw, c, 1], gw)?)
        } else {
            None
        };

        let gi = if graph.nodes[input_id.0].requires_grad {
            let mut gi = vec![0.0f32; n * ih * iw * c];
            for batch in 0..n {
                for or in 0..oh {
                    for oc in 0..ow {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let ir = or * stride_h + ki;
                                let ic = oc * stride_w + kj;
                                for ch in 0..c {
                                    let gi_idx = ((batch * ih + ir) * iw + ic) * c + ch;
                                    let up_idx = ((batch * oh + or) * ow + oc) * c + ch;
                                    let w_idx = (ki * kw + kj) * c + ch;
                                    gi[gi_idx] += w_data[w_idx] * up_data[up_idx];
                                }
                            }
                        }
                    }
                }
            }
            Some(Tensor::from_vec(vec![n, ih, iw, c], gi)?)
        } else {
            None
        };

        let gb = if let Some(b_id) = bias_id
            && graph.nodes[b_id.0].requires_grad
        {
            let mut gb = vec![0.0f32; c];
            up_data
                .iter()
                .enumerate()
                .for_each(|(i, &u)| gb[i % c] += u);
            Some(Tensor::from_vec(vec![c], gb)?)
        } else {
            None
        };

        (gw, gi, gb)
    };

    if let Some(gw) = grad_weight {
        graph.accumulate_grad(weight_id, gw)?;
    }
    if let Some(gi) = grad_input {
        graph.accumulate_grad(input_id, gi)?;
    }
    if let (Some(b_id), Some(gb)) = (bias_id, grad_bias) {
        graph.accumulate_grad(b_id, gb)?;
    }

    Ok(())
}

/// ConvTranspose2d NHWC backward.
pub(crate) fn conv_transpose2d_nhwc_backward(
    graph: &mut Graph,
    upstream: &Tensor,
    input_id: NodeId,
    weight_id: NodeId,
    bias_id: Option<NodeId>,
    stride_h: usize,
    stride_w: usize,
) -> Result<(), AutogradError> {
    let up_data = upstream.data();
    let (grad_weight, grad_input, grad_bias) = {
        let iv = &graph.nodes[input_id.0].value;
        let wv = &graph.nodes[weight_id.0].value;
        let in_shape = iv.shape();
        let w_shape = wv.shape();
        if in_shape.len() < 4 || w_shape.len() < 4 {
            return Ok(());
        }
        let (n, h, w_dim, c_in) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
        let (kh, kw, c_out, _) = (w_shape[0], w_shape[1], w_shape[2], w_shape[3]);
        let up_shape = upstream.shape();
        if up_shape.len() < 3 {
            return Ok(());
        }
        let (oh, ow) = (up_shape[1], up_shape[2]);
        let in_data = iv.data();
        let w_data = wv.data();

        let gw = if graph.nodes[weight_id.0].requires_grad {
            let mut gw = vec![0.0f32; kh * kw * c_out * c_in];
            for batch in 0..n {
                for ih in 0..h {
                    for iw in 0..w_dim {
                        for ic in 0..c_in {
                            let in_val = in_data[((batch * h + ih) * w_dim + iw) * c_in + ic];
                            for ki in 0..kh {
                                for kj in 0..kw {
                                    let out_r = ih * stride_h + ki;
                                    let out_c = iw * stride_w + kj;
                                    for oc in 0..c_out {
                                        let up_idx =
                                            ((batch * oh + out_r) * ow + out_c) * c_out + oc;
                                        let gw_idx = ((ki * kw + kj) * c_out + oc) * c_in + ic;
                                        gw[gw_idx] += in_val * up_data[up_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Some(Tensor::from_vec(vec![kh, kw, c_out, c_in], gw)?)
        } else {
            None
        };

        let gi = if graph.nodes[input_id.0].requires_grad {
            let mut gi = vec![0.0f32; n * h * w_dim * c_in];
            for batch in 0..n {
                for ih in 0..h {
                    for iw in 0..w_dim {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let out_r = ih * stride_h + ki;
                                let out_c = iw * stride_w + kj;
                                for ic in 0..c_in {
                                    let gi_idx = ((batch * h + ih) * w_dim + iw) * c_in + ic;
                                    for oc in 0..c_out {
                                        let up_idx =
                                            ((batch * oh + out_r) * ow + out_c) * c_out + oc;
                                        let w_idx = ((ki * kw + kj) * c_out + oc) * c_in + ic;
                                        gi[gi_idx] += w_data[w_idx] * up_data[up_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Some(Tensor::from_vec(vec![n, h, w_dim, c_in], gi)?)
        } else {
            None
        };

        let gb = if let Some(b_id) = bias_id
            && graph.nodes[b_id.0].requires_grad
        {
            let mut gb = vec![0.0f32; c_out];
            up_data
                .iter()
                .enumerate()
                .for_each(|(i, &v)| gb[i % c_out] += v);
            Some(Tensor::from_vec(vec![c_out], gb)?)
        } else {
            None
        };

        (gw, gi, gb)
    };

    if let Some(gw) = grad_weight {
        graph.accumulate_grad(weight_id, gw)?;
    }
    if let Some(gi) = grad_input {
        graph.accumulate_grad(input_id, gi)?;
    }
    if let (Some(b_id), Some(gb)) = (bias_id, grad_bias) {
        graph.accumulate_grad(b_id, gb)?;
    }

    Ok(())
}

/// Conv1d NLC backward.
pub(crate) fn conv1d_nlc_backward(
    graph: &mut Graph,
    upstream: &Tensor,
    input_id: NodeId,
    weight_id: NodeId,
    bias_id: Option<NodeId>,
    stride: usize,
) -> Result<(), AutogradError> {
    let up_data = upstream.data();
    let (grad_weight, grad_input, grad_bias) = {
        let iv = &graph.nodes[input_id.0].value;
        let wv = &graph.nodes[weight_id.0].value;
        let in_shape = iv.shape();
        let w_shape = wv.shape();
        let (n, length, c_in) = (in_shape[0], in_shape[1], in_shape[2]);
        let (kernel_size, _, c_out) = (w_shape[0], w_shape[1], w_shape[2]);
        let up_shape = upstream.shape();
        let out_len = up_shape[1];
        let in_data = iv.data();
        let w_data = wv.data();

        let gw = if graph.nodes[weight_id.0].requires_grad {
            let mut gw = vec![0.0f32; kernel_size * c_in * c_out];
            for batch in 0..n {
                for ol in 0..out_len {
                    let start = ol * stride;
                    for k in 0..kernel_size {
                        for ci in 0..c_in {
                            let in_idx = (batch * length + start + k) * c_in + ci;
                            let in_v = in_data[in_idx];
                            for co in 0..c_out {
                                let up_idx = (batch * out_len + ol) * c_out + co;
                                let gw_idx = (k * c_in + ci) * c_out + co;
                                gw[gw_idx] += in_v * up_data[up_idx];
                            }
                        }
                    }
                }
            }
            Some(Tensor::from_vec(vec![kernel_size, c_in, c_out], gw)?)
        } else {
            None
        };

        let gi = if graph.nodes[input_id.0].requires_grad {
            let mut gi = vec![0.0f32; n * length * c_in];
            for batch in 0..n {
                for ol in 0..out_len {
                    let start = ol * stride;
                    for k in 0..kernel_size {
                        for ci in 0..c_in {
                            let gi_idx = (batch * length + start + k) * c_in + ci;
                            for co in 0..c_out {
                                let up_idx = (batch * out_len + ol) * c_out + co;
                                let w_idx = (k * c_in + ci) * c_out + co;
                                gi[gi_idx] += w_data[w_idx] * up_data[up_idx];
                            }
                        }
                    }
                }
            }
            Some(Tensor::from_vec(vec![n, length, c_in], gi)?)
        } else {
            None
        };

        let gb = if let Some(b_id) = bias_id
            && graph.nodes[b_id.0].requires_grad
        {
            let mut gb = vec![0.0f32; c_out];
            up_data
                .iter()
                .enumerate()
                .for_each(|(i, &v)| gb[i % c_out] += v);
            Some(Tensor::from_vec(vec![c_out], gb)?)
        } else {
            None
        };

        (gw, gi, gb)
    };

    if let Some(gw) = grad_weight {
        graph.accumulate_grad(weight_id, gw)?;
    }
    if let Some(gi) = grad_input {
        graph.accumulate_grad(input_id, gi)?;
    }
    if let (Some(b_id), Some(gb)) = (bias_id, grad_bias) {
        graph.accumulate_grad(b_id, gb)?;
    }

    Ok(())
}

/// Conv3d NDHWC backward.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv3d_ndhwc_backward(
    graph: &mut Graph,
    upstream: &Tensor,
    input_id: NodeId,
    weight_id: NodeId,
    bias_id: Option<NodeId>,
    stride_d: usize,
    stride_h: usize,
    stride_w: usize,
) -> Result<(), AutogradError> {
    let up_data = upstream.data();
    let (grad_weight, grad_input, grad_bias) = {
        let iv = &graph.nodes[input_id.0].value;
        let wv = &graph.nodes[weight_id.0].value;
        let in_shape = iv.shape();
        let w_shape = wv.shape();
        let (n, in_d, in_h, in_w, c_in) = (
            in_shape[0],
            in_shape[1],
            in_shape[2],
            in_shape[3],
            in_shape[4],
        );
        let (kd, kh, kw, _, c_out) = (w_shape[0], w_shape[1], w_shape[2], w_shape[3], w_shape[4]);
        let up_shape = upstream.shape();
        let (od, oh, ow) = (up_shape[1], up_shape[2], up_shape[3]);
        let in_data = iv.data();
        let w_data = wv.data();

        // Precompute flattened (fd,fh,fw) -> (filter_flat_idx, id_offset, ih_offset, iw)
        // offsets so the inner loops over the 3D kernel become a single flat iteration.
        let filter_size = kd * kh * kw;
        let mut filter_offsets: Vec<(usize, usize, usize, usize)> = Vec::with_capacity(filter_size);
        for fd in 0..kd {
            for fh in 0..kh {
                for fw in 0..kw {
                    // flat index into kernel spatial dims: (fd*kh+fh)*kw+fw
                    let flat = (fd * kh + fh) * kw + fw;
                    filter_offsets.push((flat, fd, fh, fw));
                }
            }
        }

        #[allow(unsafe_code)]
        let gw = if graph.nodes[weight_id.0].requires_grad {
            let mut gw = vec![0.0f32; kd * kh * kw * c_in * c_out];
            for batch in 0..n {
                for o_d in 0..od {
                    for o_h in 0..oh {
                        for o_w in 0..ow {
                            let up_base = (((batch * od + o_d) * oh + o_h) * ow + o_w) * c_out;
                            // SAFETY: all indices are in-bounds by construction of
                            // output/input dimensions from valid convolution params.
                            unsafe {
                                let up_ptr = up_data.as_ptr().add(up_base);
                                for &(flat, fd, fh, fw) in &filter_offsets {
                                    let id = o_d * stride_d + fd;
                                    let ih = o_h * stride_h + fh;
                                    let iw = o_w * stride_w + fw;
                                    let in_base =
                                        (((batch * in_d + id) * in_h + ih) * in_w + iw) * c_in;
                                    let gw_spatial = flat * c_in * c_out;
                                    let in_ptr = in_data.as_ptr().add(in_base);
                                    let gw_ptr = gw.as_mut_ptr().add(gw_spatial);
                                    for ci in 0..c_in {
                                        let in_v = *in_ptr.add(ci);
                                        let gw_row = gw_ptr.add(ci * c_out);
                                        for co in 0..c_out {
                                            *gw_row.add(co) += in_v * *up_ptr.add(co);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Some(Tensor::from_vec(vec![kd, kh, kw, c_in, c_out], gw)?)
        } else {
            None
        };

        #[allow(unsafe_code)]
        let gi = if graph.nodes[input_id.0].requires_grad {
            let mut gi = vec![0.0f32; n * in_d * in_h * in_w * c_in];
            for batch in 0..n {
                for o_d in 0..od {
                    for o_h in 0..oh {
                        for o_w in 0..ow {
                            let up_base = (((batch * od + o_d) * oh + o_h) * ow + o_w) * c_out;
                            // SAFETY: same bounds guarantees as grad_weight above.
                            unsafe {
                                let up_ptr = up_data.as_ptr().add(up_base);
                                for &(flat, fd, fh, fw) in &filter_offsets {
                                    let id = o_d * stride_d + fd;
                                    let ih = o_h * stride_h + fh;
                                    let iw = o_w * stride_w + fw;
                                    let gi_base =
                                        (((batch * in_d + id) * in_h + ih) * in_w + iw) * c_in;
                                    let w_spatial = flat * c_in * c_out;
                                    let gi_ptr = gi.as_mut_ptr().add(gi_base);
                                    let w_ptr = w_data.as_ptr().add(w_spatial);
                                    for ci in 0..c_in {
                                        let w_row = w_ptr.add(ci * c_out);
                                        let mut acc = *gi_ptr.add(ci);
                                        for co in 0..c_out {
                                            acc += *w_row.add(co) * *up_ptr.add(co);
                                        }
                                        *gi_ptr.add(ci) = acc;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Some(Tensor::from_vec(vec![n, in_d, in_h, in_w, c_in], gi)?)
        } else {
            None
        };

        let gb = if let Some(b_id) = bias_id
            && graph.nodes[b_id.0].requires_grad
        {
            let mut gb = vec![0.0f32; c_out];
            up_data
                .iter()
                .enumerate()
                .for_each(|(i, &v)| gb[i % c_out] += v);
            Some(Tensor::from_vec(vec![c_out], gb)?)
        } else {
            None
        };

        (gw, gi, gb)
    };

    if let Some(gw) = grad_weight {
        graph.accumulate_grad(weight_id, gw)?;
    }
    if let Some(gi) = grad_input {
        graph.accumulate_grad(input_id, gi)?;
    }
    if let (Some(b_id), Some(gb)) = (bias_id, grad_bias) {
        graph.accumulate_grad(b_id, gb)?;
    }

    Ok(())
}

/// Deformable conv2d NHWC backward.
#[allow(clippy::too_many_arguments)]
pub(crate) fn deformable_conv2d_nhwc_backward(
    graph: &mut Graph,
    upstream: &Tensor,
    input_id: NodeId,
    weight_id: NodeId,
    offsets_id: NodeId,
    bias_id: Option<NodeId>,
    stride: usize,
    padding: usize,
) -> Result<(), AutogradError> {
    let up_data = upstream.data().to_vec();
    let up_shape = upstream.shape().to_vec();

    let (grad_input, grad_weight, grad_offsets, grad_bias) = {
        let iv = &graph.nodes[input_id.0].value;
        let wv = &graph.nodes[weight_id.0].value;
        let ov = &graph.nodes[offsets_id.0].value;
        let in_data = iv.data();
        let w_data = wv.data();
        let o_data = ov.data();
        let in_shape = iv.shape();
        let w_shape = wv.shape();

        let (batch, in_h, in_w, in_c) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
        let (kh, kw, _, out_c) = (w_shape[0], w_shape[1], w_shape[2], w_shape[3]);
        let (oh, ow) = (up_shape[1], up_shape[2]);

        let need_input_grad = graph.nodes[input_id.0].requires_grad;
        let need_weight_grad = graph.nodes[weight_id.0].requires_grad;
        let need_offset_grad = graph.nodes[offsets_id.0].requires_grad;
        let need_bias_grad = bias_id.is_some_and(|b| graph.nodes[b.0].requires_grad);

        let mut g_input = if need_input_grad {
            vec![0.0f32; iv.len()]
        } else {
            vec![]
        };
        let mut g_weight = if need_weight_grad {
            vec![0.0f32; wv.len()]
        } else {
            vec![]
        };
        let mut g_offsets = if need_offset_grad {
            vec![0.0f32; ov.len()]
        } else {
            vec![]
        };
        let mut g_bias = if need_bias_grad {
            vec![0.0f32; out_c]
        } else {
            vec![]
        };

        let expected_offset_last = kh * kw * 2;
        let in_wc = in_w * in_c;
        let in_hwc = in_h * in_wc;
        let offset_hwk = oh * ow * expected_offset_last;
        let offset_wk = ow * expected_offset_last;
        let out_wc = ow * out_c;
        let out_hwc = oh * out_wc;

        for n in 0..batch {
            let batch_input_base = n * in_hwc;
            let batch_offset_base = n * offset_hwk;
            let batch_output_base = n * out_hwc;

            for ohi in 0..oh {
                for owi in 0..ow {
                    let out_base = batch_output_base + ohi * out_wc + owi * out_c;
                    let off_base = batch_offset_base + ohi * offset_wk + owi * expected_offset_last;

                    for oc in 0..out_c {
                        let d_out = up_data[out_base + oc];
                        if need_bias_grad {
                            g_bias[oc] += d_out;
                        }

                        for ki in 0..kh {
                            for kj in 0..kw {
                                let offset_idx = (ki * kw + kj) * 2;
                                let dy = o_data[off_base + offset_idx];
                                let dx = o_data[off_base + offset_idx + 1];

                                let sampled_y = (ohi * stride + ki) as f32 - padding as f32 + dy;
                                let sampled_x = (owi * stride + kj) as f32 - padding as f32 + dx;

                                let kernel_base = (ki * kw + kj) * in_c * out_c;

                                for ic in 0..in_c {
                                    let w_idx = kernel_base + ic * out_c + oc;

                                    if need_weight_grad {
                                        let val = bilinear_sample_f32(
                                            in_data,
                                            batch_input_base,
                                            in_h,
                                            in_w,
                                            in_c,
                                            in_wc,
                                            sampled_y,
                                            sampled_x,
                                            ic,
                                        );
                                        g_weight[w_idx] += val * d_out;
                                    }

                                    if need_input_grad || need_offset_grad {
                                        let w_val = w_data[w_idx];

                                        if sampled_y >= -1.0
                                            && sampled_y <= in_h as f32
                                            && sampled_x >= -1.0
                                            && sampled_x <= in_w as f32
                                        {
                                            let y0 = sampled_y.floor() as i32;
                                            let x0 = sampled_x.floor() as i32;
                                            let y1 = y0 + 1;
                                            let x1 = x0 + 1;
                                            let ly = sampled_y - y0 as f32;
                                            let lx = sampled_x - x0 as f32;
                                            let hy = 1.0 - ly;
                                            let hx = 1.0 - lx;

                                            let contrib = d_out * w_val;

                                            if need_input_grad {
                                                let corners = [
                                                    (y0, x0, hy * hx),
                                                    (y0, x1, hy * lx),
                                                    (y1, x0, ly * hx),
                                                    (y1, x1, ly * lx),
                                                ];
                                                for (iy, ix, coeff) in corners {
                                                    if iy >= 0
                                                        && iy < in_h as i32
                                                        && ix >= 0
                                                        && ix < in_w as i32
                                                    {
                                                        g_input[batch_input_base
                                                            + (iy as usize) * in_wc
                                                            + (ix as usize) * in_c
                                                            + ic] += contrib * coeff;
                                                    }
                                                }
                                            }

                                            if need_offset_grad {
                                                let fetch = |iy: i32, ix: i32| -> f32 {
                                                    if iy >= 0
                                                        && iy < in_h as i32
                                                        && ix >= 0
                                                        && ix < in_w as i32
                                                    {
                                                        in_data[batch_input_base
                                                            + (iy as usize) * in_wc
                                                            + (ix as usize) * in_c
                                                            + ic]
                                                    } else {
                                                        0.0
                                                    }
                                                };
                                                let v00 = fetch(y0, x0);
                                                let v01 = fetch(y0, x1);
                                                let v10 = fetch(y1, x0);
                                                let v11 = fetch(y1, x1);

                                                let d_dy =
                                                    -hx * v00 - lx * v01 + hx * v10 + lx * v11;
                                                let d_dx =
                                                    -hy * v00 + hy * v01 - ly * v10 + ly * v11;

                                                g_offsets[off_base + offset_idx] +=
                                                    d_out * w_val * d_dy;
                                                g_offsets[off_base + offset_idx + 1] +=
                                                    d_out * w_val * d_dx;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        (
            if need_input_grad {
                Some(Tensor::from_vec(in_shape.to_vec(), g_input)?)
            } else {
                None
            },
            if need_weight_grad {
                Some(Tensor::from_vec(w_shape.to_vec(), g_weight)?)
            } else {
                None
            },
            if need_offset_grad {
                Some(Tensor::from_vec(ov.shape().to_vec(), g_offsets)?)
            } else {
                None
            },
            if need_bias_grad {
                Some(Tensor::from_vec(vec![out_c], g_bias)?)
            } else {
                None
            },
        )
    };

    if let Some(gi) = grad_input {
        graph.accumulate_grad(input_id, gi)?;
    }
    if let Some(gw) = grad_weight {
        graph.accumulate_grad(weight_id, gw)?;
    }
    if let Some(go) = grad_offsets {
        graph.accumulate_grad(offsets_id, go)?;
    }
    if let (Some(bias), Some(gb)) = (bias_id, grad_bias) {
        graph.accumulate_grad(bias, gb)?;
    }
    Ok(())
}

/// Bilinear interpolation sampling for deformable conv backward.
#[inline]
fn bilinear_sample_f32(
    data: &[f32],
    batch_base: usize,
    in_h: usize,
    in_w: usize,
    in_c: usize,
    in_wc: usize,
    y: f32,
    x: f32,
    channel: usize,
) -> f32 {
    if y < -1.0 || y > in_h as f32 || x < -1.0 || x > in_w as f32 {
        return 0.0;
    }
    let y0 = y.floor() as i32;
    let x0 = x.floor() as i32;
    let y1 = y0 + 1;
    let x1 = x0 + 1;
    let ly = y - y0 as f32;
    let lx = x - x0 as f32;
    let hy = 1.0 - ly;
    let hx = 1.0 - lx;

    let fetch = |iy: i32, ix: i32| -> f32 {
        if iy < 0 || iy >= in_h as i32 || ix < 0 || ix >= in_w as i32 {
            return 0.0;
        }
        data[batch_base + (iy as usize) * in_wc + (ix as usize) * in_c + channel]
    };

    hy * hx * fetch(y0, x0)
        + hy * lx * fetch(y0, x1)
        + ly * hx * fetch(y1, x0)
        + ly * lx * fetch(y1, x1)
}
