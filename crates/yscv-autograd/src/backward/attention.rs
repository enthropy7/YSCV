use yscv_tensor::Tensor;

use super::error::AutogradError;
use super::graph::Graph;
use super::node::{AuxData, NodeId};

/// Scaled dot-product attention backward.
pub(crate) fn scaled_dot_product_attention_backward(
    graph: &mut Graph,
    upstream: &Tensor,
    index: usize,
    query_id: NodeId,
    key_id: NodeId,
    value_id: NodeId,
) -> Result<(), AutogradError> {
    let attn_weights = match &graph.nodes[index].aux {
        Some(AuxData::AttentionWeights(w)) => w.clone(),
        _ => {
            return Err(AutogradError::InvalidGradientShape {
                node: index,
                expected: vec![],
                got: vec![],
            });
        }
    };

    // Try BackwardOps for GPU-accelerated attention backward
    if let Some(ref backend) = graph.backend {
        let qv = &graph.nodes[query_id.0].value;
        let kv = &graph.nodes[key_id.0].value;
        let vv = &graph.nodes[value_id.0].value;
        match backend.attention_backward(upstream, qv, kv, vv, &attn_weights) {
            Ok((gq, gk, gv)) => {
                if graph.nodes[query_id.0].requires_grad {
                    graph.accumulate_grad(query_id, gq)?;
                }
                if graph.nodes[key_id.0].requires_grad {
                    graph.accumulate_grad(key_id, gk)?;
                }
                if graph.nodes[value_id.0].requires_grad {
                    graph.accumulate_grad(value_id, gv)?;
                }
                return Ok(());
            }
            Err(_e) => {
                #[cfg(debug_assertions)]
                eprintln!("[autograd] attention_backward GPU fallback: {_e}");
                // fall through to CPU
            }
        }
    }

    let qv = &graph.nodes[query_id.0].value;
    let kv = &graph.nodes[key_id.0].value;
    let vv = &graph.nodes[value_id.0].value;

    let d_k = qv.shape()[1];
    let seq_q = qv.shape()[0];
    let seq_k = kv.shape()[0];
    let d_v = vv.shape()[1];
    let scale = (d_k as f32).sqrt().recip();

    let q_data = qv.data();
    let k_data = kv.data();
    let v_data = vv.data();
    let a_data = attn_weights.data();
    let up_data = upstream.data();

    // dV = A^T @ dOut : [seq_k, d_v]
    let grad_v = if graph.nodes[value_id.0].requires_grad {
        let mut dv = vec![0.0f32; seq_k * d_v];
        for sk in 0..seq_k {
            for dvi in 0..d_v {
                let mut sum = 0.0f32;
                for sq in 0..seq_q {
                    sum += a_data[sq * seq_k + sk] * up_data[sq * d_v + dvi];
                }
                dv[sk * d_v + dvi] = sum;
            }
        }
        Some(Tensor::from_vec(vec![seq_k, d_v], dv)?)
    } else {
        None
    };

    // dA = dOut @ V^T : [seq_q, seq_k]
    let need_da = graph.nodes[query_id.0].requires_grad || graph.nodes[key_id.0].requires_grad;
    let (grad_q, grad_k) = if need_da {
        let mut da = vec![0.0f32; seq_q * seq_k];
        for sq in 0..seq_q {
            for sk in 0..seq_k {
                let mut sum = 0.0f32;
                for dvi in 0..d_v {
                    sum += up_data[sq * d_v + dvi] * v_data[sk * d_v + dvi];
                }
                da[sq * seq_k + sk] = sum;
            }
        }

        // Softmax backward: dS = A * (dA - (dA * A).sum(-1, keepdim))
        let mut ds = vec![0.0f32; seq_q * seq_k];
        for sq in 0..seq_q {
            let base = sq * seq_k;
            let mut dot = 0.0f32;
            for sk in 0..seq_k {
                dot += da[base + sk] * a_data[base + sk];
            }
            for sk in 0..seq_k {
                ds[base + sk] = a_data[base + sk] * (da[base + sk] - dot);
            }
        }

        // dQ = dS @ K * scale : [seq_q, d_k]
        let gq = if graph.nodes[query_id.0].requires_grad {
            let mut dq = vec![0.0f32; seq_q * d_k];
            for sq in 0..seq_q {
                for dk_i in 0..d_k {
                    let mut sum = 0.0f32;
                    for sk in 0..seq_k {
                        sum += ds[sq * seq_k + sk] * k_data[sk * d_k + dk_i];
                    }
                    dq[sq * d_k + dk_i] = sum * scale;
                }
            }
            Some(Tensor::from_vec(vec![seq_q, d_k], dq)?)
        } else {
            None
        };

        // dK = dS^T @ Q * scale : [seq_k, d_k]
        let gk = if graph.nodes[key_id.0].requires_grad {
            let mut dk_grad = vec![0.0f32; seq_k * d_k];
            for sk in 0..seq_k {
                for dk_i in 0..d_k {
                    let mut sum = 0.0f32;
                    for sq in 0..seq_q {
                        sum += ds[sq * seq_k + sk] * q_data[sq * d_k + dk_i];
                    }
                    dk_grad[sk * d_k + dk_i] = sum * scale;
                }
            }
            Some(Tensor::from_vec(vec![seq_k, d_k], dk_grad)?)
        } else {
            None
        };

        (gq, gk)
    } else {
        (None, None)
    };

    if let Some(gq) = grad_q {
        graph.accumulate_grad(query_id, gq)?;
    }
    if let Some(gk) = grad_k {
        graph.accumulate_grad(key_id, gk)?;
    }
    if let Some(gv) = grad_v {
        graph.accumulate_grad(value_id, gv)?;
    }

    Ok(())
}

/// PReLU backward.
pub(crate) fn prelu_backward(
    graph: &mut Graph,
    upstream: &Tensor,
    input_id: NodeId,
    alpha_id: NodeId,
) -> Result<(), AutogradError> {
    let up_data = upstream.data();
    let (grad_input, grad_alpha) = {
        let iv = &graph.nodes[input_id.0].value;
        let av = &graph.nodes[alpha_id.0].value;
        let in_data = iv.data();
        let alpha_data = av.data();
        let alpha_len = alpha_data.len();

        let gi = if graph.nodes[input_id.0].requires_grad {
            let gi: Vec<f32> = in_data
                .iter()
                .zip(up_data.iter())
                .enumerate()
                .map(|(i, (&x, &u))| {
                    let a = if alpha_len == 1 {
                        alpha_data[0]
                    } else {
                        alpha_data[i % alpha_len]
                    };
                    if x > 0.0 { u } else { u * a }
                })
                .collect();
            Some(Tensor::from_vec(iv.shape().to_vec(), gi)?)
        } else {
            None
        };

        let ga = if graph.nodes[alpha_id.0].requires_grad {
            let mut ga = vec![0.0f32; alpha_len];
            in_data
                .iter()
                .zip(up_data.iter())
                .enumerate()
                .for_each(|(i, (&x, &u))| {
                    if x <= 0.0 {
                        let ch = if alpha_len == 1 { 0 } else { i % alpha_len };
                        ga[ch] += u * x;
                    }
                });
            Some(Tensor::from_vec(av.shape().to_vec(), ga)?)
        } else {
            None
        };

        (gi, ga)
    };

    if let Some(gi) = grad_input {
        graph.accumulate_grad(input_id, gi)?;
    }
    if let Some(ga) = grad_alpha {
        graph.accumulate_grad(alpha_id, ga)?;
    }

    Ok(())
}
