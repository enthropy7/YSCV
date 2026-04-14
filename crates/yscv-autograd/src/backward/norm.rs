use yscv_tensor::Tensor;

use super::error::AutogradError;
use super::graph::Graph;
use super::node::{AuxData, NodeId};

/// BatchNorm2d NHWC backward (inference-mode running-stats path).
#[allow(clippy::too_many_arguments)]
pub(crate) fn batch_norm2d_nhwc_backward(
    graph: &mut Graph,
    node_index: usize,
    upstream: &Tensor,
    input_id: NodeId,
    gamma_id: NodeId,
    beta_id: NodeId,
    running_var_id: NodeId,
    epsilon: f32,
) -> Result<(), AutogradError> {
    // Try BackwardOps for input gradient (the most expensive part)
    let mut gpu_handled_input = false;
    if let Some(ref backend) = graph.backend
        && graph.nodes[input_id.0].requires_grad
    {
        let gamma = &graph.nodes[gamma_id.0].value;
        let running_var = &graph.nodes[running_var_id.0].value;
        match backend.batch_norm2d_input_backward(upstream, gamma, running_var, epsilon) {
            Ok(gi) => {
                graph.accumulate_grad(input_id, gi)?;
                gpu_handled_input = true;
            }
            Err(_e) => {
                #[cfg(debug_assertions)]
                eprintln!("[autograd] batch_norm2d_input_backward GPU fallback: {_e}");
            }
        }
    }

    let up_data = upstream.data();
    let (grad_beta, grad_gamma, grad_input) = {
        let gamma_data = graph.nodes[gamma_id.0].value.data();
        let var_data = graph.nodes[running_var_id.0].value.data();
        let input_shape = graph.nodes[input_id.0].value.shape().to_vec();
        let c = *input_shape.last().unwrap_or(&0);
        if c == 0 {
            return Ok(());
        }

        let gb = if graph.nodes[beta_id.0].requires_grad {
            let mut gb = vec![0.0f32; c];
            up_data
                .iter()
                .enumerate()
                .for_each(|(i, &v)| gb[i % c] += v);
            Some(Tensor::from_vec(vec![c], gb)?)
        } else {
            None
        };

        let gg = if graph.nodes[gamma_id.0].requires_grad {
            let norm_data = match &graph.nodes[node_index].aux {
                Some(AuxData::BatchNormNormalized(t)) => t.data(),
                _ => {
                    return Err(AutogradError::InvalidGradientShape {
                        node: node_index,
                        expected: vec![],
                        got: vec![],
                    });
                }
            };
            let mut gg = vec![0.0f32; c];
            up_data
                .iter()
                .zip(norm_data.iter())
                .enumerate()
                .for_each(|(i, (&u, &n))| gg[i % c] += u * n);
            Some(Tensor::from_vec(vec![c], gg)?)
        } else {
            None
        };

        let gi = if graph.nodes[input_id.0].requires_grad && !gpu_handled_input {
            let gi: Vec<f32> = up_data
                .iter()
                .enumerate()
                .map(|(i, &u)| {
                    let ch = i % c;
                    u * gamma_data[ch] / (var_data[ch] + epsilon).sqrt()
                })
                .collect();
            Some(Tensor::from_vec(input_shape, gi)?)
        } else {
            None
        };

        (gb, gg, gi)
    };

    if let Some(gb) = grad_beta {
        graph.accumulate_grad(beta_id, gb)?;
    }
    if let Some(gg) = grad_gamma {
        graph.accumulate_grad(gamma_id, gg)?;
    }
    if let Some(gi) = grad_input {
        graph.accumulate_grad(input_id, gi)?;
    }

    Ok(())
}

/// LayerNorm backward.
#[allow(clippy::too_many_arguments)]
pub(crate) fn layer_norm_backward(
    graph: &mut Graph,
    node_index: usize,
    upstream: &Tensor,
    input_id: NodeId,
    gamma_id: NodeId,
    beta_id: NodeId,
    epsilon: f32,
) -> Result<(), AutogradError> {
    // Try BackwardOps for input gradient
    let mut gpu_handled_input = false;
    if let Some(ref backend) = graph.backend
        && graph.nodes[input_id.0].requires_grad
    {
        let iv = &graph.nodes[input_id.0].value;
        let gamma = &graph.nodes[gamma_id.0].value;
        match backend.layer_norm_input_backward(upstream, iv, gamma, epsilon) {
            Ok(gi) => {
                graph.accumulate_grad(input_id, gi)?;
                gpu_handled_input = true;
            }
            Err(_e) => {
                #[cfg(debug_assertions)]
                eprintln!("[autograd] layer_norm_input_backward GPU fallback: {_e}");
            }
        }
    }

    let up_data = upstream.data();
    let (grad_beta, grad_gamma, grad_input) = {
        let iv = &graph.nodes[input_id.0].value;
        let gamma_data = graph.nodes[gamma_id.0].value.data().to_vec();
        let input_shape = iv.shape().to_vec();
        let last_dim = *input_shape.last().unwrap_or(&0);
        if last_dim == 0 {
            return Ok(());
        }
        let num_groups = iv.len() / last_dim;
        let in_data = iv.data();

        let x_hat_data = match &graph.nodes[node_index].aux {
            Some(AuxData::NormNormalized(t)) => t.data().to_vec(),
            _ => {
                return Err(AutogradError::InvalidGradientShape {
                    node: node_index,
                    expected: vec![],
                    got: vec![],
                });
            }
        };

        let gb = if graph.nodes[beta_id.0].requires_grad {
            let mut gb = vec![0.0f32; last_dim];
            for g in 0..num_groups {
                let up_row = &up_data[g * last_dim..(g + 1) * last_dim];
                gb.iter_mut()
                    .zip(up_row.iter())
                    .for_each(|(acc, &u)| *acc += u);
            }
            Some(Tensor::from_vec(vec![last_dim], gb)?)
        } else {
            None
        };

        let gg = if graph.nodes[gamma_id.0].requires_grad {
            let mut gg = vec![0.0f32; last_dim];
            for g in 0..num_groups {
                let base = g * last_dim;
                let xh_row = &x_hat_data[base..base + last_dim];
                let up_row = &up_data[base..base + last_dim];
                gg.iter_mut()
                    .zip(xh_row.iter().zip(up_row.iter()))
                    .for_each(|(acc, (&x, &u))| *acc += x * u);
            }
            Some(Tensor::from_vec(vec![last_dim], gg)?)
        } else {
            None
        };

        let gi = if graph.nodes[input_id.0].requires_grad && !gpu_handled_input {
            let mut gi = vec![0.0f32; iv.len()];
            let n = last_dim as f32;
            for g in 0..num_groups {
                let base = g * last_dim;
                let slice = &in_data[base..base + last_dim];
                let mean = slice.iter().sum::<f32>() / n;
                let var = slice.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
                let inv_std = 1.0 / (var + epsilon).sqrt();

                let up_row = &up_data[base..base + last_dim];
                let dy_gamma: Vec<f32> = up_row
                    .iter()
                    .zip(gamma_data.iter())
                    .map(|(&u, &g)| u * g)
                    .collect();
                let mean_dy_gamma = dy_gamma.iter().sum::<f32>() / n;
                let xh_row = &x_hat_data[base..base + last_dim];
                let mean_xhat_dy_gamma: f32 = xh_row
                    .iter()
                    .zip(dy_gamma.iter())
                    .map(|(&x, &d)| x * d)
                    .sum::<f32>()
                    / n;

                let gi_row = &mut gi[base..base + last_dim];
                gi_row
                    .iter_mut()
                    .zip(dy_gamma.iter().zip(xh_row.iter()))
                    .for_each(|(g, (&dy, &xh))| {
                        *g = inv_std * (dy - mean_dy_gamma - xh * mean_xhat_dy_gamma);
                    });
            }
            Some(Tensor::from_vec(input_shape, gi)?)
        } else {
            None
        };

        (gb, gg, gi)
    };

    if let Some(gb) = grad_beta {
        graph.accumulate_grad(beta_id, gb)?;
    }
    if let Some(gg) = grad_gamma {
        graph.accumulate_grad(gamma_id, gg)?;
    }
    if let Some(gi) = grad_input {
        graph.accumulate_grad(input_id, gi)?;
    }

    Ok(())
}

/// GroupNorm backward (NHWC layout).
#[allow(clippy::too_many_arguments)]
pub(crate) fn group_norm_backward(
    graph: &mut Graph,
    node_index: usize,
    upstream: &Tensor,
    input_id: NodeId,
    gamma_id: NodeId,
    beta_id: NodeId,
    num_groups: usize,
    epsilon: f32,
) -> Result<(), AutogradError> {
    let up_data = upstream.data();
    let (grad_beta, grad_gamma, grad_input) = {
        let iv = &graph.nodes[input_id.0].value;
        let gamma_data = graph.nodes[gamma_id.0].value.data().to_vec();
        let input_shape = iv.shape().to_vec();
        if input_shape.len() < 4 {
            return Ok(());
        }
        let (n_batch, h, w, c) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        if num_groups == 0 {
            return Ok(());
        }
        let channels_per_group = c / num_groups;
        let spatial = h.checked_mul(w).unwrap_or(0);
        let group_size = spatial.checked_mul(channels_per_group).unwrap_or(0);
        if group_size == 0 {
            return Ok(());
        }
        let in_data = iv.data();

        let x_hat_data = match &graph.nodes[node_index].aux {
            Some(AuxData::NormNormalized(t)) => t.data().to_vec(),
            _ => {
                return Err(AutogradError::InvalidGradientShape {
                    node: node_index,
                    expected: vec![],
                    got: vec![],
                });
            }
        };

        let gb = if graph.nodes[beta_id.0].requires_grad {
            let mut gb = vec![0.0f32; c];
            up_data
                .iter()
                .enumerate()
                .for_each(|(i, &v)| gb[i % c] += v);
            Some(Tensor::from_vec(vec![c], gb)?)
        } else {
            None
        };

        let gg = if graph.nodes[gamma_id.0].requires_grad {
            let mut gg = vec![0.0f32; c];
            x_hat_data
                .iter()
                .zip(up_data.iter())
                .enumerate()
                .for_each(|(i, (&x, &u))| gg[i % c] += x * u);
            Some(Tensor::from_vec(vec![c], gg)?)
        } else {
            None
        };

        let gi = if graph.nodes[input_id.0].requires_grad {
            let mut gi = vec![0.0f32; iv.len()];
            let gs = group_size as f32;

            for ni in 0..n_batch {
                for gidx in 0..num_groups {
                    let c_start = gidx * channels_per_group;
                    let c_end = c_start + channels_per_group;

                    let mut sum = 0.0f32;
                    for hi in 0..h {
                        for wi in 0..w {
                            let base = ((ni * h + hi) * w + wi) * c;
                            for ci in c_start..c_end {
                                sum += in_data[base + ci];
                            }
                        }
                    }
                    let mean = sum / gs;
                    let mut var_sum = 0.0f32;
                    for hi in 0..h {
                        for wi in 0..w {
                            let base = ((ni * h + hi) * w + wi) * c;
                            for ci in c_start..c_end {
                                let d = in_data[base + ci] - mean;
                                var_sum += d * d;
                            }
                        }
                    }
                    let inv_std = 1.0 / (var_sum / gs + epsilon).sqrt();

                    let mut sum_dy_gamma = 0.0f32;
                    let mut sum_xhat_dy_gamma = 0.0f32;
                    for hi in 0..h {
                        for wi in 0..w {
                            let base = ((ni * h + hi) * w + wi) * c;
                            for ci in c_start..c_end {
                                let dy_g = up_data[base + ci] * gamma_data[ci];
                                sum_dy_gamma += dy_g;
                                sum_xhat_dy_gamma += x_hat_data[base + ci] * dy_g;
                            }
                        }
                    }
                    let mean_dy_gamma = sum_dy_gamma / gs;
                    let mean_xhat_dy_gamma = sum_xhat_dy_gamma / gs;

                    for hi in 0..h {
                        for wi in 0..w {
                            let base = ((ni * h + hi) * w + wi) * c;
                            for ci in c_start..c_end {
                                let dy_g = up_data[base + ci] * gamma_data[ci];
                                gi[base + ci] = inv_std
                                    * (dy_g
                                        - mean_dy_gamma
                                        - x_hat_data[base + ci] * mean_xhat_dy_gamma);
                            }
                        }
                    }
                }
            }
            Some(Tensor::from_vec(input_shape, gi)?)
        } else {
            None
        };

        (gb, gg, gi)
    };

    if let Some(gb) = grad_beta {
        graph.accumulate_grad(beta_id, gb)?;
    }
    if let Some(gg) = grad_gamma {
        graph.accumulate_grad(gamma_id, gg)?;
    }
    if let Some(gi) = grad_input {
        graph.accumulate_grad(input_id, gi)?;
    }

    Ok(())
}

/// Instance normalization NHWC backward.
#[allow(clippy::too_many_arguments)]
pub(crate) fn instance_norm_nhwc_backward(
    graph: &mut Graph,
    node_index: usize,
    upstream: &Tensor,
    input_id: NodeId,
    gamma_id: NodeId,
    beta_id: NodeId,
    epsilon: f32,
) -> Result<(), AutogradError> {
    let up_data = upstream.data();
    let (grad_beta, grad_gamma, grad_input) = {
        let iv = &graph.nodes[input_id.0].value;
        let gamma_data = graph.nodes[gamma_id.0].value.data().to_vec();
        let input_shape = iv.shape().to_vec();
        if input_shape.len() < 4 {
            return Ok(());
        }
        let (n, h, w, c) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        let spatial = h.checked_mul(w).unwrap_or(0);
        if spatial == 0 {
            return Ok(());
        }
        let in_data = iv.data();

        let x_hat_data = match &graph.nodes[node_index].aux {
            Some(AuxData::NormNormalized(t)) => t.data().to_vec(),
            _ => {
                return Err(AutogradError::InvalidGradientShape {
                    node: node_index,
                    expected: vec![],
                    got: vec![],
                });
            }
        };

        let gb = if graph.nodes[beta_id.0].requires_grad {
            let mut gb = vec![0.0f32; c];
            up_data
                .iter()
                .enumerate()
                .for_each(|(i, &v)| gb[i % c] += v);
            Some(Tensor::from_vec(vec![c], gb)?)
        } else {
            None
        };

        let gg = if graph.nodes[gamma_id.0].requires_grad {
            let mut gg = vec![0.0f32; c];
            x_hat_data
                .iter()
                .zip(up_data.iter())
                .enumerate()
                .for_each(|(i, (&x, &u))| gg[i % c] += x * u);
            Some(Tensor::from_vec(vec![c], gg)?)
        } else {
            None
        };

        let gi = if graph.nodes[input_id.0].requires_grad {
            let mut gi = vec![0.0f32; iv.len()];
            let sp = spatial as f32;

            for ni in 0..n {
                for ch in 0..c {
                    let mut sum = 0.0f32;
                    for s in 0..spatial {
                        let idx = (ni * h * w + s) * c + ch;
                        sum += in_data[idx];
                    }
                    let mean = sum / sp;
                    let mut var_sum = 0.0f32;
                    for s in 0..spatial {
                        let idx = (ni * h * w + s) * c + ch;
                        let d = in_data[idx] - mean;
                        var_sum += d * d;
                    }
                    let inv_std = 1.0 / (var_sum / sp + epsilon).sqrt();

                    let mut sum_dy_gamma = 0.0f32;
                    let mut sum_xhat_dy_gamma = 0.0f32;
                    for s in 0..spatial {
                        let idx = (ni * h * w + s) * c + ch;
                        let dy_g = up_data[idx] * gamma_data[ch];
                        sum_dy_gamma += dy_g;
                        sum_xhat_dy_gamma += x_hat_data[idx] * dy_g;
                    }
                    let mean_dy_gamma = sum_dy_gamma / sp;
                    let mean_xhat_dy_gamma = sum_xhat_dy_gamma / sp;

                    for s in 0..spatial {
                        let idx = (ni * h * w + s) * c + ch;
                        let dy_g = up_data[idx] * gamma_data[ch];
                        gi[idx] =
                            inv_std * (dy_g - mean_dy_gamma - x_hat_data[idx] * mean_xhat_dy_gamma);
                    }
                }
            }
            Some(Tensor::from_vec(input_shape, gi)?)
        } else {
            None
        };

        (gb, gg, gi)
    };

    if let Some(gb) = grad_beta {
        graph.accumulate_grad(beta_id, gb)?;
    }
    if let Some(gg) = grad_gamma {
        graph.accumulate_grad(gamma_id, gg)?;
    }
    if let Some(gi) = grad_input {
        graph.accumulate_grad(input_id, gi)?;
    }

    Ok(())
}
