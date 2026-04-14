use yscv_tensor::Tensor;

use super::error::AutogradError;
use super::graph::Graph;
use super::node::{AuxData, NodeId};

/// MaxPool2d NHWC backward.
pub(crate) fn max_pool2d_nhwc_backward(
    graph: &mut Graph,
    upstream: Tensor,
    index: usize,
    input: NodeId,
) -> Result<(), AutogradError> {
    let indices = match &graph.nodes[index].aux {
        Some(AuxData::MaxPoolIndices(idx)) => idx.clone(),
        _ => {
            return Err(AutogradError::InvalidGradientShape {
                node: index,
                expected: vec![],
                got: vec![],
            });
        }
    };
    let input_shape = graph.node(input)?.value.shape().to_vec();
    let input_len = graph.node(input)?.value.len();
    let mut grad_data = vec![0.0f32; input_len];
    let up_data = upstream.data();
    for (out_idx, &in_idx) in indices.iter().enumerate() {
        grad_data[in_idx] += up_data[out_idx];
    }
    let input_grad = Tensor::from_vec(input_shape, grad_data)?;
    graph.accumulate_grad(input, input_grad)?;
    Ok(())
}

/// AvgPool2d NHWC backward.
pub(crate) fn avg_pool2d_nhwc_backward(
    graph: &mut Graph,
    upstream: &Tensor,
    input_id: NodeId,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
) -> Result<(), AutogradError> {
    let input_shape = graph.node(input_id)?.value.shape().to_vec();
    if input_shape.len() < 4 {
        return Ok(());
    }

    // Try BackwardOps for GPU-accelerated avg_pool backward
    if let Some(ref backend) = graph.backend {
        match backend.avg_pool2d_backward(upstream, &input_shape, kh, kw, sh, sw) {
            Ok(gi) => {
                graph.accumulate_grad(input_id, gi)?;
                return Ok(());
            }
            Err(_e) => {
                #[cfg(debug_assertions)]
                eprintln!("[autograd] avg_pool2d_backward GPU fallback: {_e}");
                // fall through to CPU
            }
        }
    }
    let (n, ih, iw, c) = (
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    );
    let up_shape = upstream.shape();
    if up_shape.len() < 3 {
        return Ok(());
    }
    let (oh, ow) = (up_shape[1], up_shape[2]);
    let up_data = upstream.data();
    let pool_size = (kh * kw) as f32;

    let mut gi = vec![0.0f32; n * ih * iw * c];
    for batch in 0..n {
        for or in 0..oh {
            for oc_col in 0..ow {
                for ch in 0..c {
                    let up_idx = ((batch * oh + or) * ow + oc_col) * c + ch;
                    let grad_val = up_data[up_idx] / pool_size;
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let ir = or * sh + ki;
                            let ic = oc_col * sw + kj;
                            let gi_idx = ((batch * ih + ir) * iw + ic) * c + ch;
                            gi[gi_idx] += grad_val;
                        }
                    }
                }
            }
        }
    }
    let grad_input = Tensor::from_vec(input_shape, gi)?;
    graph.accumulate_grad(input_id, grad_input)?;
    Ok(())
}

/// Adaptive average pool 2d NHWC backward.
pub(crate) fn adaptive_avg_pool2d_nhwc_backward(
    graph: &mut Graph,
    upstream: &Tensor,
    input_id: NodeId,
    out_h: usize,
    out_w: usize,
) -> Result<(), AutogradError> {
    let input_shape = graph.node(input_id)?.value.shape().to_vec();
    if input_shape.len() < 4 {
        return Ok(());
    }
    let (n, h, w, c) = (
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    );
    let up_data = upstream.data();

    let mut gi = vec![
        0.0f32;
        n.checked_mul(h)
            .unwrap_or(0)
            .checked_mul(w)
            .unwrap_or(0)
            .checked_mul(c)
            .unwrap_or(0)
    ];
    for b in 0..n {
        for oh_idx in 0..out_h {
            let h_start = if out_h == 0 { 0 } else { oh_idx * h / out_h };
            let h_end = ((oh_idx + 1) * h / out_h).max(h_start + 1);
            for ow_idx in 0..out_w {
                let w_start = ow_idx * w / out_w;
                let w_end = ((ow_idx + 1) * w / out_w).max(w_start + 1);
                let count = (h_end - h_start) * (w_end - w_start);
                for ch in 0..c {
                    let up_idx = ((b * out_h + oh_idx) * out_w + ow_idx) * c + ch;
                    let grad_val = up_data[up_idx] / count as f32;
                    for ih in h_start..h_end {
                        for iw in w_start..w_end {
                            let gi_idx = ((b * h + ih) * w + iw) * c + ch;
                            gi[gi_idx] += grad_val;
                        }
                    }
                }
            }
        }
    }
    let grad_input = Tensor::from_vec(input_shape, gi)?;
    graph.accumulate_grad(input_id, grad_input)?;
    Ok(())
}

/// Adaptive max pool 2d NHWC backward.
pub(crate) fn adaptive_max_pool2d_nhwc_backward(
    graph: &mut Graph,
    upstream: Tensor,
    index: usize,
    input: NodeId,
) -> Result<(), AutogradError> {
    let indices = match &graph.nodes[index].aux {
        Some(AuxData::MaxPoolIndices(idx)) => idx.clone(),
        _ => {
            return Err(AutogradError::InvalidGradientShape {
                node: index,
                expected: vec![],
                got: vec![],
            });
        }
    };
    let input_shape = graph.node(input)?.value.shape().to_vec();
    let input_len = graph.node(input)?.value.len();
    let mut grad_data = vec![0.0f32; input_len];
    let up_data = upstream.data();
    for (out_idx, &in_idx) in indices.iter().enumerate() {
        grad_data[in_idx] += up_data[out_idx];
    }
    let input_grad = Tensor::from_vec(input_shape, grad_data)?;
    graph.accumulate_grad(input, input_grad)?;
    Ok(())
}
