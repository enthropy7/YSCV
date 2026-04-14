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
    let cd = cond.data();
    let xd = x.data();
    let yd = y.data();
    if cd.len() != xd.len() || cd.len() != yd.len() {
        return Err(OnnxError::DecodeFailed {
            message: "Where: shape mismatch".into(),
        });
    }
    let data: Vec<f32> = cd
        .iter()
        .enumerate()
        .map(|(i, &c)| if c > 0.0 { xd[i] } else { yd[i] })
        .collect();
    let out =
        Tensor::from_vec(cond.shape().to_vec(), data).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
    env.insert(node.outputs[0].clone(), out);
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
