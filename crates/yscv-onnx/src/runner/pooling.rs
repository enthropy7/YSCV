use super::conv::{nchw_to_nhwc, pad_nhwc, pad_nhwc_val};
use super::*;

pub(super) fn exec_max_pool(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input_is_nhwc = env.is_nhwc(&node.inputs[0]);
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let kernel_shape = get_attr_ints(node, Attr::KernelShape).unwrap_or_else(|| vec![2, 2]);
    let strides = get_attr_ints(node, Attr::Strides).unwrap_or_else(|| vec![1, 1]);
    let pads = get_attr_ints(node, Attr::Pads).unwrap_or_else(|| vec![0, 0, 0, 0]);

    // NCHW fast path: use the optimized NCHW kernel (parallel + SIMD).
    if !input_is_nhwc && input.rank() == 4 {
        let result = yscv_kernels::max_pool2d_nchw(
            input,
            kernel_shape[0] as usize,
            kernel_shape[1] as usize,
            strides[0] as usize,
            strides[1] as usize,
            pads[0] as usize,
            pads[1] as usize,
            pads[2] as usize,
            pads[3] as usize,
        )
        .map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
        env.insert(node.outputs[0].clone(), result);
        return Ok(());
    }

    let input_nhwc_owned;
    let input_nhwc: &Tensor = if input_is_nhwc {
        input
    } else {
        input_nhwc_owned = nchw_to_nhwc(input)?;
        &input_nhwc_owned
    };
    let padded_owned;
    let input_padded: &Tensor = if pads.iter().any(|&p| p > 0) {
        padded_owned = pad_nhwc_val(
            input_nhwc,
            pads[0] as usize,
            pads[1] as usize,
            pads[2] as usize,
            pads[3] as usize,
            f32::NEG_INFINITY,
        )?;
        &padded_owned
    } else {
        input_nhwc
    };
    let out_nhwc = max_pool2d_nhwc(
        input_padded,
        kernel_shape[0] as usize,
        kernel_shape[1] as usize,
        strides[0] as usize,
        strides[1] as usize,
    )
    .map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    env.insert(node.outputs[0].clone(), out_nhwc);
    env.mark_nhwc(&node.outputs[0]);
    Ok(())
}

pub(super) fn exec_avg_pool(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input_is_nhwc = env.is_nhwc(&node.inputs[0]);
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let kernel_shape = get_attr_ints(node, Attr::KernelShape).unwrap_or_else(|| vec![2, 2]);
    let strides = get_attr_ints(node, Attr::Strides).unwrap_or_else(|| vec![1, 1]);
    let pads = get_attr_ints(node, Attr::Pads).unwrap_or_else(|| vec![0, 0, 0, 0]);

    // NCHW fast path: use the optimized NCHW kernel (parallel + SIMD).
    if !input_is_nhwc && input.rank() == 4 {
        let result = yscv_kernels::avg_pool2d_nchw(
            input,
            kernel_shape[0] as usize,
            kernel_shape[1] as usize,
            strides[0] as usize,
            strides[1] as usize,
            pads[0] as usize,
            pads[1] as usize,
            pads[2] as usize,
            pads[3] as usize,
        )
        .map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
        env.insert(node.outputs[0].clone(), result);
        return Ok(());
    }

    let input_nhwc_owned;
    let input_nhwc: &Tensor = if input_is_nhwc {
        input
    } else {
        input_nhwc_owned = nchw_to_nhwc(input)?;
        &input_nhwc_owned
    };
    let padded_owned;
    let input_padded: &Tensor = if pads.iter().any(|&p| p > 0) {
        padded_owned = pad_nhwc(
            input_nhwc,
            pads[0] as usize,
            pads[1] as usize,
            pads[2] as usize,
            pads[3] as usize,
        )?;
        &padded_owned
    } else {
        input_nhwc
    };
    let out_nhwc = avg_pool2d_nhwc(
        input_padded,
        kernel_shape[0] as usize,
        kernel_shape[1] as usize,
        strides[0] as usize,
        strides[1] as usize,
    )
    .map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    env.insert(node.outputs[0].clone(), out_nhwc);
    env.mark_nhwc(&node.outputs[0]);
    Ok(())
}

pub(super) fn exec_global_avg_pool(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input_is_nhwc = env.is_nhwc(&node.inputs[0]);
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let shape = input.shape();

    // NCHW fast path: global avg is simple sum/count per channel plane
    if !input_is_nhwc && shape.len() == 4 {
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let hw_f = (h * w) as f32;
        let data = input.data();
        let mut out = Vec::with_capacity(n * c);
        for b in 0..n {
            for ch in 0..c {
                let base = (b * c + ch) * h * w;
                let sum: f32 = data[base..base + h * w].iter().sum();
                out.push(sum / hw_f);
            }
        }
        let result =
            Tensor::from_vec(vec![n, c, 1, 1], out).map_err(|e| OnnxError::DecodeFailed {
                message: e.to_string(),
            })?;
        env.insert(node.outputs[0].clone(), result);
        // Output stays NCHW
        return Ok(());
    }

    let (h, w) = if input_is_nhwc {
        (shape[1], shape[2])
    } else {
        (shape[2], shape[3])
    };
    let input_nhwc_owned;
    let input_nhwc: &Tensor = if input_is_nhwc {
        input
    } else {
        input_nhwc_owned = nchw_to_nhwc(input)?;
        &input_nhwc_owned
    };
    let out_nhwc =
        avg_pool2d_nhwc(input_nhwc, h, w, 1, 1).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
    env.insert(node.outputs[0].clone(), out_nhwc);
    env.mark_nhwc(&node.outputs[0]);
    Ok(())
}
