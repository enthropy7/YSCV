use super::*;

pub(super) fn exec_cmp(node: &OnnxNode, env: &mut TensorEnv, mode: u8) -> Result<(), OnnxError> {
    let a = get_tensor(env, &node.name, &node.inputs[0])?;
    let b = get_tensor(env, &node.name, &node.inputs[1])?;
    let out = match mode {
        0 => a.eq_tensor(b),
        1 => a.gt_tensor(b),
        2 => a.lt_tensor(b),
        3 => {
            let gt = a.gt_tensor(b).map_err(|e| OnnxError::DecodeFailed {
                message: e.to_string(),
            })?;
            let eq = a.eq_tensor(b).map_err(|e| OnnxError::DecodeFailed {
                message: e.to_string(),
            })?;
            let gd = gt.data();
            let ed = eq.data();
            let d: Vec<f32> = gd
                .iter()
                .zip(ed.iter())
                .map(|(&g, &e)| if g != 0.0 || e != 0.0 { 1.0 } else { 0.0 })
                .collect();
            Tensor::from_vec(a.shape().to_vec(), d)
        }
        4 => {
            let lt = a.lt_tensor(b).map_err(|e| OnnxError::DecodeFailed {
                message: e.to_string(),
            })?;
            let eq = a.eq_tensor(b).map_err(|e| OnnxError::DecodeFailed {
                message: e.to_string(),
            })?;
            let ld = lt.data();
            let ed = eq.data();
            let d: Vec<f32> = ld
                .iter()
                .zip(ed.iter())
                .map(|(&l, &e)| if l != 0.0 || e != 0.0 { 1.0 } else { 0.0 })
                .collect();
            Tensor::from_vec(a.shape().to_vec(), d)
        }
        _ => unreachable!(),
    };
    let out = out.map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_where(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let cond = get_tensor(env, &node.name, &node.inputs[0])?;
    let x = get_tensor(env, &node.name, &node.inputs[1])?;
    let y = get_tensor(env, &node.name, &node.inputs[2])?;
    let out = Tensor::where_select(cond, x, y).map_err(|e| OnnxError::DecodeFailed {
        message: format!("Where: {e}"),
    })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

/// `Trilu` — element-wise upper or lower triangular over the last two
/// dims. `upper` attribute (default 1) selects upper triangle; the
/// optional second input `k` (scalar i64) shifts the diagonal. Output
/// keeps shape; entries outside the selected triangle are zeroed.
///
/// Used by HuggingFace ONNX exports to build the causal-attention mask
/// in cached-decoder mode (Llama / Qwen / Phi).
pub(super) fn exec_trilu(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let upper = get_attr_int(node, "upper").unwrap_or(1) != 0;
    let k = if node.inputs.len() >= 2 {
        let kt = get_tensor(env, &node.name, &node.inputs[1])?;
        if kt.data().is_empty() {
            0
        } else {
            kt.data()[0] as i64
        }
    } else {
        0
    };

    let shape = input.shape().to_vec();
    if shape.len() < 2 {
        return Err(OnnxError::DecodeFailed {
            message: format!("Trilu requires rank ≥ 2 input, got {:?}", shape),
        });
    }
    let rows = shape[shape.len() - 2];
    let cols = shape[shape.len() - 1];
    let plane = rows * cols;
    let outer: usize = shape[..shape.len() - 2].iter().product();

    let src = input.data();
    let mut out = vec![0.0_f32; src.len()];
    for o in 0..outer {
        let base = o * plane;
        for i in 0..rows {
            for j in 0..cols {
                let keep = if upper {
                    (j as i64) >= (i as i64) + k
                } else {
                    (j as i64) <= (i as i64) + k
                };
                if keep {
                    out[base + i * cols + j] = src[base + i * cols + j];
                }
            }
        }
    }

    let result = Tensor::from_vec(shape, out).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    env.insert(node.outputs[0].clone(), result);
    Ok(())
}

pub(super) fn exec_min_max(
    node: &OnnxNode,
    env: &mut TensorEnv,
    is_max: bool,
) -> Result<(), OnnxError> {
    let mut result = get_tensor(env, &node.name, &node.inputs[0])?.clone();
    for inp in &node.inputs[1..] {
        let other = get_tensor(env, &node.name, inp)?;
        result = if is_max {
            result.maximum(other).map_err(|e| OnnxError::DecodeFailed {
                message: e.to_string(),
            })?
        } else {
            result.minimum(other).map_err(|e| OnnxError::DecodeFailed {
                message: e.to_string(),
            })?
        };
    }
    env.insert(node.outputs[0].clone(), result);
    Ok(())
}

pub(super) fn exec_not(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let data: Vec<f32> = input
        .data()
        .iter()
        .map(|&v| if v == 0.0 { 1.0 } else { 0.0 })
        .collect();
    let out =
        Tensor::from_vec(input.shape().to_vec(), data).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_logical_bin(
    node: &OnnxNode,
    env: &mut TensorEnv,
    mode: u8,
) -> Result<(), OnnxError> {
    let a = get_tensor(env, &node.name, &node.inputs[0])?;
    let b = get_tensor(env, &node.name, &node.inputs[1])?;
    let a_d = a.data();
    let b_d = b.data();
    if a_d.len() != b_d.len() {
        return Err(OnnxError::ShapeMismatch {
            detail: format!("logical op: left={}, right={}", a_d.len(), b_d.len()),
        });
    }
    let data: Vec<f32> = a_d
        .iter()
        .zip(b_d.iter())
        .map(|(&x, &y)| {
            let bx = x != 0.0;
            let by = y != 0.0;
            let r = match mode {
                0 => bx && by,
                1 => bx || by,
                _ => bx ^ by,
            };
            if r { 1.0 } else { 0.0 }
        })
        .collect();
    let out = Tensor::from_vec(a.shape().to_vec(), data).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}
