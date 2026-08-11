use super::*;

pub(super) fn exec_relu(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let out = relu(input);
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_relu_inplace(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let output_name = &node.outputs[0];
    if let Some(tensor) = env.get(output_name).cloned() {
        let out = relu(&tensor);
        env.insert(output_name.clone(), out);
    }
    Ok(())
}

pub(super) fn exec_sigmoid(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let out = sigmoid(input);
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_add(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let a = get_tensor(env, &node.name, &node.inputs[0])?;
    let b = get_tensor(env, &node.name, &node.inputs[1])?;
    let out = kernel_add(a, b).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_sub(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let a = get_tensor(env, &node.name, &node.inputs[0])?;
    let b = get_tensor(env, &node.name, &node.inputs[1])?;
    let out = kernel_sub(a, b).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_mul(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let a = get_tensor(env, &node.name, &node.inputs[0])?;
    let b = get_tensor(env, &node.name, &node.inputs[1])?;
    let out = kernel_mul(a, b).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_div(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let a = get_tensor(env, &node.name, &node.inputs[0])?;
    let b = get_tensor(env, &node.name, &node.inputs[1])?;
    let out = a.div(b).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_pow(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let a = get_tensor(env, &node.name, &node.inputs[0])?;
    let b = get_tensor(env, &node.name, &node.inputs[1])?;
    let out = a.pow(b).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_sqrt(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let a = get_tensor(env, &node.name, &node.inputs[0])?;
    let out = a.sqrt();
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_exp(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let a = get_tensor(env, &node.name, &node.inputs[0])?;
    let out = a.exp();
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_log(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let a = get_tensor(env, &node.name, &node.inputs[0])?;
    let out = a.ln();
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_neg(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let a = get_tensor(env, &node.name, &node.inputs[0])?;
    let out = a.neg();
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_abs(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let a = get_tensor(env, &node.name, &node.inputs[0])?;
    let out = a.abs();
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_reciprocal(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let a = get_tensor(env, &node.name, &node.inputs[0])?;
    let out = a.reciprocal();
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_tanh(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let out = yscv_kernels::tanh_act(input);
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_hardswish(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let out = yscv_kernels::hardswish(input);
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_floor(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let a = get_tensor(env, &node.name, &node.inputs[0])?;
    let out = a.floor();
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_ceil(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let a = get_tensor(env, &node.name, &node.inputs[0])?;
    let out = a.ceil();
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_clip(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let min_val = if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
        get_tensor(env, &node.name, &node.inputs[1])?.data()[0]
    } else {
        f32::NEG_INFINITY
    };
    let max_val = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
        get_tensor(env, &node.name, &node.inputs[2])?.data()[0]
    } else {
        f32::INFINITY
    };
    let out = input.clamp(min_val, max_val);
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_leaky_relu(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let alpha = get_attr_float(node, Attr::Alpha).unwrap_or(0.01);
    let data: Vec<f32> = input
        .data()
        .iter()
        .map(|&v| if v >= 0.0 { v } else { alpha * v })
        .collect();
    let out =
        Tensor::from_vec(input.shape().to_vec(), data).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_elu(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let alpha = get_attr_float(node, Attr::Alpha).unwrap_or(1.0);
    let data: Vec<f32> = input
        .data()
        .iter()
        .map(|&v| if v >= 0.0 { v } else { alpha * (v.exp() - 1.0) })
        .collect();
    let out =
        Tensor::from_vec(input.shape().to_vec(), data).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_selu(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let alpha = get_attr_float(node, Attr::Alpha).unwrap_or(1.673_263_2);
    let gamma = get_attr_float(node, Attr::Gamma).unwrap_or(1.050_701);
    let data: Vec<f32> = input
        .data()
        .iter()
        .map(|&x| {
            if x > 0.0 {
                gamma * x
            } else {
                gamma * (alpha * x.exp() - alpha)
            }
        })
        .collect();
    let out =
        Tensor::from_vec(input.shape().to_vec(), data).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_celu(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let alpha = get_attr_float(node, Attr::Alpha).unwrap_or(1.0);
    let data: Vec<f32> = input
        .data()
        .iter()
        .map(|&x| x.max(0.0) + (alpha * ((x / alpha).exp() - 1.0)).min(0.0))
        .collect();
    let out =
        Tensor::from_vec(input.shape().to_vec(), data).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_thresholded_relu(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let alpha = get_attr_float(node, Attr::Alpha).unwrap_or(1.0);
    let data: Vec<f32> = input
        .data()
        .iter()
        .map(|&x| if x > alpha { x } else { 0.0 })
        .collect();
    let out =
        Tensor::from_vec(input.shape().to_vec(), data).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_hard_sigmoid(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let alpha = get_attr_float(node, Attr::Alpha).unwrap_or(0.2);
    let beta = get_attr_float(node, Attr::Beta).unwrap_or(0.5);
    let data: Vec<f32> = input
        .data()
        .iter()
        .map(|&x| (alpha * x + beta).clamp(0.0, 1.0))
        .collect();
    let out =
        Tensor::from_vec(input.shape().to_vec(), data).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_gelu(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let sqrt2_inv = 1.0 / std::f32::consts::SQRT_2;
    let data: Vec<f32> = input
        .data()
        .iter()
        .map(|&x| 0.5 * x * (1.0 + erf_approx(x * sqrt2_inv)))
        .collect();
    let out =
        Tensor::from_vec(input.shape().to_vec(), data).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_erf(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let data: Vec<f32> = input.data().iter().map(|&x| erf_approx(x)).collect();
    let out =
        Tensor::from_vec(input.shape().to_vec(), data).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

/// Abramowitz & Stegun approximation of erf(x), max error ~1.5e-7.
#[allow(clippy::excessive_precision)]
fn erf_approx(x: f32) -> f32 {
    let sign = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1_f32 * x);
    let poly = t
        * (0.254_829_6
            + t * (-0.284_496_74 + t * (1.421_413_7 + t * (-1.453_152 + t * 1.061_405_4))));
    sign * (1.0 - poly * (-x * x).exp())
}

/// Execute a Tensor-level unary op (SIMD-accelerated: sin, cos, exp, etc.)
pub(super) fn exec_tensor_op(
    node: &OnnxNode,
    env: &mut TensorEnv,
    f: fn(&Tensor) -> Tensor,
) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let out = f(input);
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

/// Execute a scalar unary op (no SIMD — fallback for trig functions etc.)
pub(super) fn exec_unary(
    node: &OnnxNode,
    env: &mut TensorEnv,
    f: fn(f32) -> f32,
) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let data: Vec<f32> = input.data().iter().map(|&v| f(v)).collect();
    let out =
        Tensor::from_vec(input.shape().to_vec(), data).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_mod(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let a = get_tensor(env, &node.name, &node.inputs[0])?;
    let b = get_tensor(env, &node.name, &node.inputs[1])?;
    let fmod = get_attr_int(node, Attr::Fmod).unwrap_or(0);
    let a_d = a.data();
    let b_d = b.data();
    let data: Vec<f32> = a_d
        .iter()
        .zip(b_d.iter())
        .map(|(&x, &y)| {
            if fmod != 0 {
                x % y
            } else {
                x - (x / y).floor() * y
            }
        })
        .collect();
    let out = Tensor::from_vec(a.shape().to_vec(), data).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_bitshift(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let a = get_tensor(env, &node.name, &node.inputs[0])?;
    let b = get_tensor(env, &node.name, &node.inputs[1])?;
    let direction = get_attr_string(node, Attr::Direction).unwrap_or_default();
    let data: Vec<f32> = a
        .data()
        .iter()
        .zip(b.data().iter())
        .map(|(&x, &y)| {
            let xi = x as u32;
            let yi = y as u32;
            let r = if direction == "RIGHT" {
                xi >> yi
            } else {
                xi << yi
            };
            r as f32
        })
        .collect();
    let out = Tensor::from_vec(a.shape().to_vec(), data).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}
