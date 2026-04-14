use yscv_tensor::Tensor;

use super::error::AutogradError;
use super::graph::Graph;
use super::node::NodeId;

/// Reshape backward.
pub(crate) fn reshape_backward(
    graph: &mut Graph,
    upstream: Tensor,
    input: NodeId,
) -> Result<(), AutogradError> {
    let orig_shape = graph.node(input)?.value.shape().to_vec();
    let input_grad = upstream.reshape(orig_shape)?;
    graph.accumulate_grad(input, input_grad)?;
    Ok(())
}

/// Flatten backward.
pub(crate) fn flatten_backward(
    graph: &mut Graph,
    upstream: Tensor,
    input: NodeId,
) -> Result<(), AutogradError> {
    let input_shape = graph.node(input)?.value.shape().to_vec();
    let input_grad = upstream.reshape(input_shape)?;
    graph.accumulate_grad(input, input_grad)?;
    Ok(())
}

/// Unsqueeze backward.
pub(crate) fn unsqueeze_backward(
    graph: &mut Graph,
    upstream: Tensor,
    input: NodeId,
    axis: u16,
) -> Result<(), AutogradError> {
    let input_grad = upstream.squeeze(axis as usize)?;
    graph.accumulate_grad(input, input_grad)?;
    Ok(())
}

/// Squeeze backward.
pub(crate) fn squeeze_backward(
    graph: &mut Graph,
    upstream: Tensor,
    input: NodeId,
    axis: u16,
) -> Result<(), AutogradError> {
    let input_grad = upstream.unsqueeze(axis as usize)?;
    graph.accumulate_grad(input, input_grad)?;
    Ok(())
}

/// Cat backward.
pub(crate) fn cat_backward(
    graph: &mut Graph,
    upstream: Tensor,
    inputs: &[NodeId],
    axis: u16,
) -> Result<(), AutogradError> {
    let ax = axis as usize;
    let mut offset = 0usize;
    for &inp in inputs {
        let dim = graph.node(inp)?.value.shape()[ax];
        let grad_slice = upstream.narrow(ax, offset, dim)?;
        graph.accumulate_grad(inp, grad_slice)?;
        offset += dim;
    }
    Ok(())
}

/// Select backward.
pub(crate) fn select_backward(
    graph: &mut Graph,
    upstream: Tensor,
    input: NodeId,
    axis: u16,
    index: u32,
) -> Result<(), AutogradError> {
    let input_shape = graph.node(input)?.value.shape().to_vec();
    let mut grad_data = vec![0.0f32; input_shape.iter().product()];
    let ax = axis as usize;
    let idx = index as usize;
    let outer: usize = input_shape[..ax].iter().product();
    let dim = input_shape[ax];
    let inner: usize = input_shape[ax + 1..].iter().product();
    let up = upstream.data();
    for o in 0..outer {
        for i in 0..inner {
            grad_data[(o * dim + idx) * inner + i] = up[o * inner + i];
        }
    }
    let input_grad = Tensor::from_vec(input_shape, grad_data)?;
    graph.accumulate_grad(input, input_grad)?;
    Ok(())
}

/// Narrow backward.
pub(crate) fn narrow_backward(
    graph: &mut Graph,
    upstream: Tensor,
    input: NodeId,
    axis: u16,
    start: u32,
    len: u32,
) -> Result<(), AutogradError> {
    let input_shape = graph.node(input)?.value.shape().to_vec();
    let ax = axis as usize;
    let s = start as usize;
    let l = len as usize;
    let mut grad_data = vec![0.0f32; input_shape.iter().product()];
    let outer: usize = input_shape[..ax].iter().product();
    let dim = input_shape[ax];
    let inner: usize = input_shape[ax + 1..].iter().product();
    let up = upstream.data();
    for o in 0..outer {
        for d in 0..l {
            for i in 0..inner {
                grad_data[(o * dim + s + d) * inner + i] = up[(o * l + d) * inner + i];
            }
        }
    }
    let input_grad = Tensor::from_vec(input_shape, grad_data)?;
    graph.accumulate_grad(input, input_grad)?;
    Ok(())
}

/// Gather backward.
pub(crate) fn gather_backward(
    graph: &mut Graph,
    upstream: Tensor,
    input: NodeId,
    axis: u16,
    index: NodeId,
) -> Result<(), AutogradError> {
    let ax = axis as usize;
    let input_shape = graph.node(input)?.value.shape().to_vec();
    let idx_tensor = &graph.nodes[index.0].value;
    let zeros = Tensor::zeros(input_shape)?;
    let input_grad = zeros.scatter_add(ax, idx_tensor, &upstream)?;
    graph.accumulate_grad(input, input_grad)?;
    Ok(())
}

/// ScatterAdd backward.
pub(crate) fn scatter_add_backward(
    graph: &mut Graph,
    upstream: Tensor,
    input: NodeId,
    axis: u16,
    index: NodeId,
    src: NodeId,
) -> Result<(), AutogradError> {
    let ax = axis as usize;
    if graph.nodes[input.0].requires_grad {
        graph.accumulate_grad(input, upstream.clone())?;
    }
    if graph.nodes[src.0].requires_grad {
        let idx_tensor = &graph.nodes[index.0].value;
        let grad_src = upstream.gather(ax, idx_tensor)?;
        graph.accumulate_grad(src, grad_src)?;
    }
    Ok(())
}

/// Pad backward.
#[allow(clippy::needless_range_loop)]
pub(crate) fn pad_backward(
    graph: &mut Graph,
    upstream: Tensor,
    input: NodeId,
    pad_before: &[u32],
) -> Result<(), AutogradError> {
    let input_shape = graph.node(input)?.value.shape().to_vec();
    let up = upstream.data();
    let up_shape = upstream.shape();
    let rank = input_shape.len();
    let total: usize = input_shape.iter().product();
    let mut grad_data = vec![0.0f32; total];

    let mut src_strides = vec![1usize; rank];
    let mut up_strides = vec![1usize; rank];
    for d in (0..rank - 1).rev() {
        src_strides[d] = src_strides[d + 1] * input_shape[d + 1];
        up_strides[d] = up_strides[d + 1] * up_shape[d + 1];
    }
    for flat in 0..total {
        let mut rem = flat;
        let mut up_flat = 0usize;
        for d in 0..rank {
            let coord = rem / src_strides[d];
            rem %= src_strides[d];
            up_flat += (coord + pad_before[d] as usize) * up_strides[d];
        }
        grad_data[flat] = up[up_flat];
    }
    let input_grad = Tensor::from_vec(input_shape, grad_data)?;
    graph.accumulate_grad(input, input_grad)?;
    Ok(())
}

/// Repeat backward.
#[allow(clippy::needless_range_loop)]
pub(crate) fn repeat_backward(
    graph: &mut Graph,
    upstream: Tensor,
    input: NodeId,
) -> Result<(), AutogradError> {
    let input_shape = graph.node(input)?.value.shape().to_vec();
    let up = upstream.data();
    let up_shape = upstream.shape();
    let rank = input_shape.len();
    let total: usize = input_shape.iter().product();
    let mut grad_data = vec![0.0f32; total];

    let total_up: usize = up_shape.iter().product();
    let mut up_strides = vec![1usize; rank];
    let mut src_strides = vec![1usize; rank];
    for d in (0..rank - 1).rev() {
        up_strides[d] = up_strides[d + 1] * up_shape[d + 1];
        src_strides[d] = src_strides[d + 1] * input_shape[d + 1];
    }
    for flat_up in 0..total_up {
        let mut rem = flat_up;
        let mut src_flat = 0usize;
        for d in 0..rank {
            let coord = rem / up_strides[d];
            rem %= up_strides[d];
            src_flat += (coord % input_shape[d]) * src_strides[d];
        }
        grad_data[src_flat] += up[flat_up];
    }
    let input_grad = Tensor::from_vec(input_shape, grad_data)?;
    graph.accumulate_grad(input, input_grad)?;
    Ok(())
}

/// Scatter backward.
pub(crate) fn scatter_backward(
    graph: &mut Graph,
    upstream: Tensor,
    input: NodeId,
    indices: NodeId,
    src: NodeId,
) -> Result<(), AutogradError> {
    let idx_data = graph.nodes[indices.0].value.data().to_vec();
    let input_shape = graph.nodes[input.0].value.shape().to_vec();
    let d = input_shape[1];
    if graph.nodes[input.0].requires_grad {
        let mut grad_input_data = upstream.data().to_vec();
        for &raw_idx in &idx_data {
            let row = raw_idx as usize;
            let offset = row * d;
            grad_input_data[offset..offset + d].fill(0.0);
        }
        let grad_input = Tensor::from_vec(input_shape, grad_input_data)?;
        graph.accumulate_grad(input, grad_input)?;
    }
    if graph.nodes[src.0].requires_grad {
        let src_shape = graph.nodes[src.0].value.shape().to_vec();
        let m = idx_data.len();
        let up_data = upstream.data();
        let mut grad_src_data = vec![0.0f32; m * d];
        for (i, &raw_idx) in idx_data.iter().enumerate() {
            let row = raw_idx as usize;
            let src_off = i * d;
            let up_off = row * d;
            grad_src_data[src_off..src_off + d].copy_from_slice(&up_data[up_off..up_off + d]);
        }
        let grad_src = Tensor::from_vec(src_shape, grad_src_data)?;
        graph.accumulate_grad(src, grad_src)?;
    }
    Ok(())
}

/// EmbeddingLookup backward.
pub(crate) fn embedding_lookup_backward(
    graph: &mut Graph,
    upstream: Tensor,
    weight: NodeId,
    indices: NodeId,
) -> Result<(), AutogradError> {
    if graph.nodes[weight.0].requires_grad {
        let weight_shape = graph.nodes[weight.0].value.shape().to_vec();
        let embed_dim = weight_shape[1];
        let num_embeddings = weight_shape[0];
        let idx_data = graph.nodes[indices.0].value.data();

        // Try BackwardOps for GPU-accelerated embedding backward
        if let Some(ref backend) = graph.backend {
            let indices_usize: Vec<usize> = idx_data.iter().map(|&v| v as usize).collect();
            match backend.embedding_backward(&upstream, &indices_usize, num_embeddings, embed_dim) {
                Ok(gw) => {
                    graph.accumulate_grad(weight, gw)?;
                    return Ok(());
                }
                Err(_e) => {
                    #[cfg(debug_assertions)]
                    eprintln!("[autograd] embedding_backward GPU fallback: {_e}");
                    // fall through to CPU
                }
            }
        }

        let up_data = upstream.data();
        let mut grad_weight_data = vec![0.0f32; weight_shape.iter().product::<usize>()];
        for (i, &raw_idx) in idx_data.iter().enumerate() {
            let row = raw_idx as usize;
            let src_off = i * embed_dim;
            let dst_off = row * embed_dim;
            grad_weight_data[dst_off..dst_off + embed_dim]
                .iter_mut()
                .zip(&up_data[src_off..src_off + embed_dim])
                .for_each(|(g, &u)| *g += u);
        }
        let grad_weight = Tensor::from_vec(weight_shape, grad_weight_data)?;
        graph.accumulate_grad(weight, grad_weight)?;
    }
    Ok(())
}

/// PixelShuffle backward.
pub(crate) fn pixel_shuffle_backward(
    graph: &mut Graph,
    upstream: &Tensor,
    input_id: NodeId,
    r: usize,
) -> Result<(), AutogradError> {
    if !graph.nodes[input_id.0].requires_grad {
        return Ok(());
    }
    let input_shape = graph.nodes[input_id.0].value.shape().to_vec();
    if input_shape.len() < 4 {
        return Ok(());
    }
    let (batch, h, w, c) = (
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    );
    let out_c = c / (r * r);
    let out_h = h * r;
    let out_w = w * r;
    let up_data = upstream.data();
    let mut grad_data = vec![0.0f32; batch * h * w * c];

    for b in 0..batch {
        for ih in 0..h {
            for iw in 0..w {
                for oc in 0..out_c {
                    for ry in 0..r {
                        for rx in 0..r {
                            let ic = oc * r * r + ry * r + rx;
                            let oh = ih * r + ry;
                            let ow = iw * r + rx;
                            grad_data[((b * h + ih) * w + iw) * c + ic] =
                                up_data[((b * out_h + oh) * out_w + ow) * out_c + oc];
                        }
                    }
                }
            }
        }
    }
    let input_grad = Tensor::from_vec(input_shape, grad_data)?;
    graph.accumulate_grad(input_id, input_grad)?;
    Ok(())
}

/// Nearest-neighbor upsample backward.
pub(crate) fn upsample_nearest_backward(
    graph: &mut Graph,
    upstream: &Tensor,
    input_id: NodeId,
    r: usize,
) -> Result<(), AutogradError> {
    if !graph.nodes[input_id.0].requires_grad {
        return Ok(());
    }
    let input_shape = graph.nodes[input_id.0].value.shape().to_vec();
    if input_shape.len() < 4 {
        return Ok(());
    }
    let (batch, h, w, c) = (
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    );
    let out_h = h * r;
    let out_w = w * r;
    let up_data = upstream.data();
    let mut grad_data = vec![0.0f32; batch * h * w * c];

    for b in 0..batch {
        for oh in 0..out_h {
            let ih = oh / r;
            for ow in 0..out_w {
                let iw = ow / r;
                let src = ((b * out_h + oh) * out_w + ow) * c;
                let dst = ((b * h + ih) * w + iw) * c;
                for ch in 0..c {
                    grad_data[dst + ch] += up_data[src + ch];
                }
            }
        }
    }
    let input_grad = Tensor::from_vec(input_shape, grad_data)?;
    graph.accumulate_grad(input_id, input_grad)?;
    Ok(())
}
