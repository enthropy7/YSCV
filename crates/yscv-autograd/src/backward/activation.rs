use yscv_tensor::Tensor;

use super::error::AutogradError;
use super::graph::Graph;
use super::node::NodeId;

/// ReLU backward.
pub(crate) fn relu_backward(
    graph: &mut Graph,
    upstream: Tensor,
    _index: usize,
    input: NodeId,
) -> Result<(), AutogradError> {
    let input_grad = {
        let iv = &graph.nodes[input.0].value;
        if iv.shape() != upstream.shape() {
            return Err(AutogradError::InvalidGradientShape {
                node: input.0,
                expected: iv.shape().to_vec(),
                got: upstream.shape().to_vec(),
            });
        }
        if let Some(ref backend) = graph.backend {
            match backend.relu_backward(&upstream, iv) {
                Ok(t) => t,
                Err(_e) => {
                    #[cfg(debug_assertions)]
                    eprintln!("[autograd] relu_backward GPU fallback: {_e}");
                    let mut result = upstream;
                    relu_backward_slice(result.data_mut(), iv.data());
                    result
                }
            }
        } else {
            let mut result = upstream;
            relu_backward_slice(result.data_mut(), iv.data());
            result
        }
    };
    graph.accumulate_grad(input, input_grad)?;
    Ok(())
}

/// Exp backward: d(exp(x))/dx = exp(x).
pub(crate) fn exp_backward(
    graph: &mut Graph,
    upstream: Tensor,
    index: usize,
    input: NodeId,
) -> Result<(), AutogradError> {
    let input_grad = {
        let ov = &graph.nodes[index].value;
        if let Some(ref backend) = graph.backend {
            match backend.exp_backward(&upstream, ov) {
                Ok(t) => t,
                Err(_e) => {
                    #[cfg(debug_assertions)]
                    eprintln!("[autograd] exp_backward GPU fallback: {_e}");
                    graph.dispatch_mul(&upstream, ov)?
                }
            }
        } else {
            graph.dispatch_mul(&upstream, ov)?
        }
    };
    graph.accumulate_grad(input, input_grad)?;
    Ok(())
}

/// Log backward: d(ln(x))/dx = 1/x.
pub(crate) fn log_backward(
    graph: &mut Graph,
    upstream: Tensor,
    input: NodeId,
) -> Result<(), AutogradError> {
    let input_grad = {
        let iv = &graph.nodes[input.0].value;
        let inv_x = iv.reciprocal();
        graph.dispatch_mul(&upstream, &inv_x)?
    };
    graph.accumulate_grad(input, input_grad)?;
    Ok(())
}

/// Sqrt backward: d(sqrt(x))/dx = 0.5 / sqrt(x).
pub(crate) fn sqrt_backward(
    graph: &mut Graph,
    upstream: Tensor,
    index: usize,
    input: NodeId,
) -> Result<(), AutogradError> {
    let input_grad = {
        let ov = &graph.nodes[index].value;
        let half_inv = ov.reciprocal().scale(0.5);
        graph.dispatch_mul(&upstream, &half_inv)?
    };
    graph.accumulate_grad(input, input_grad)?;
    Ok(())
}

/// Sigmoid backward: d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x)).
pub(crate) fn sigmoid_backward(
    graph: &mut Graph,
    upstream: Tensor,
    index: usize,
    input: NodeId,
) -> Result<(), AutogradError> {
    let input_grad = {
        let ov = &graph.nodes[index].value;
        if let Some(ref backend) = graph.backend {
            match backend.sigmoid_backward(&upstream, ov) {
                Ok(t) => t,
                Err(_e) => {
                    #[cfg(debug_assertions)]
                    eprintln!("[autograd] sigmoid_backward GPU fallback: {_e}");
                    let one_minus = graph.dispatch_neg(ov).add_scalar(1.0);
                    let local = graph.dispatch_mul(ov, &one_minus)?;
                    graph.dispatch_mul(&upstream, &local)?
                }
            }
        } else {
            let one_minus = graph.dispatch_neg(ov).add_scalar(1.0);
            let local = graph.dispatch_mul(ov, &one_minus)?;
            graph.dispatch_mul(&upstream, &local)?
        }
    };
    graph.accumulate_grad(input, input_grad)?;
    Ok(())
}

/// Tanh backward: d(tanh(x))/dx = 1 - tanh(x)^2.
pub(crate) fn tanh_backward(
    graph: &mut Graph,
    upstream: Tensor,
    index: usize,
    input: NodeId,
) -> Result<(), AutogradError> {
    let input_grad = {
        let ov = &graph.nodes[index].value;
        if let Some(ref backend) = graph.backend {
            match backend.tanh_backward(&upstream, ov) {
                Ok(t) => t,
                Err(_e) => {
                    #[cfg(debug_assertions)]
                    eprintln!("[autograd] tanh_backward GPU fallback: {_e}");
                    let sq = graph.dispatch_mul(ov, ov)?;
                    let one_minus_sq = graph.dispatch_neg(&sq).add_scalar(1.0);
                    graph.dispatch_mul(&upstream, &one_minus_sq)?
                }
            }
        } else {
            let sq = graph.dispatch_mul(ov, ov)?;
            let one_minus_sq = graph.dispatch_neg(&sq).add_scalar(1.0);
            graph.dispatch_mul(&upstream, &one_minus_sq)?
        }
    };
    graph.accumulate_grad(input, input_grad)?;
    Ok(())
}

/// GELU backward.
pub(crate) fn gelu_backward(
    graph: &mut Graph,
    upstream: Tensor,
    input: NodeId,
) -> Result<(), AutogradError> {
    let input_grad = {
        let iv = &graph.nodes[input.0].value;
        let mut result = upstream;
        gelu_backward_slice(result.data_mut(), iv.data());
        result
    };
    graph.accumulate_grad(input, input_grad)?;
    Ok(())
}

/// SiLU backward.
pub(crate) fn silu_backward(
    graph: &mut Graph,
    upstream: Tensor,
    input: NodeId,
) -> Result<(), AutogradError> {
    let input_grad = {
        let iv = &graph.nodes[input.0].value;
        let mut result = upstream;
        silu_backward_slice(result.data_mut(), iv.data());
        result
    };
    graph.accumulate_grad(input, input_grad)?;
    Ok(())
}

/// Mish backward.
pub(crate) fn mish_backward(
    graph: &mut Graph,
    upstream: Tensor,
    input: NodeId,
) -> Result<(), AutogradError> {
    let input_grad = {
        let iv = &graph.nodes[input.0].value;
        let mut result = upstream;
        mish_backward_slice(result.data_mut(), iv.data());
        result
    };
    graph.accumulate_grad(input, input_grad)?;
    Ok(())
}

/// LeakyReLU backward.
pub(crate) fn leaky_relu_backward(
    graph: &mut Graph,
    upstream: Tensor,
    input: NodeId,
    negative_slope: u32,
) -> Result<(), AutogradError> {
    let slope = f32::from_bits(negative_slope);
    let input_grad = {
        let iv = &graph.nodes[input.0].value;
        let mut result = upstream;
        leaky_relu_backward_slice(result.data_mut(), iv.data(), slope);
        result
    };
    graph.accumulate_grad(input, input_grad)?;
    Ok(())
}

/// Softmax backward.
pub(crate) fn softmax_backward(
    graph: &mut Graph,
    upstream: Tensor,
    index: usize,
    input: NodeId,
) -> Result<(), AutogradError> {
    let input_grad = {
        let ov = &graph.nodes[index].value;
        let shape = ov.shape();
        let last = *shape.last().unwrap_or(&1);
        let outer = ov.len() / last;
        let sm = ov.data();
        let up = upstream.data();
        let mut grad = vec![0.0f32; ov.len()];

        for o in 0..outer {
            let base = o * last;
            let up_row = &up[base..base + last];
            let sm_row = &sm[base..base + last];
            let grad_row = &mut grad[base..base + last];
            let dot: f32 = up_row.iter().zip(sm_row.iter()).map(|(&u, &s)| u * s).sum();
            grad_row
                .iter_mut()
                .zip(sm_row.iter().zip(up_row.iter()))
                .for_each(|(g, (&s, &u))| *g = s * (u - dot));
        }
        Tensor::from_vec(shape.to_vec(), grad)?
    };
    graph.accumulate_grad(input, input_grad)?;
    Ok(())
}

/// LogSoftmax backward.
pub(crate) fn log_softmax_backward(
    graph: &mut Graph,
    upstream: Tensor,
    index: usize,
    input: NodeId,
) -> Result<(), AutogradError> {
    let input_grad = {
        let ov = &graph.nodes[index].value;
        let shape = ov.shape();
        let last = *shape.last().unwrap_or(&1);
        let outer = ov.len() / last;
        let ov_data = ov.data();
        let up = upstream.data();
        let mut grad = vec![0.0f32; ov.len()];

        for o in 0..outer {
            let base = o * last;
            let up_row = &up[base..base + last];
            let ov_row = &ov_data[base..base + last];
            let grad_row = &mut grad[base..base + last];
            let sum_up: f32 = up_row.iter().sum();
            grad_row
                .iter_mut()
                .zip(up_row.iter().zip(ov_row.iter()))
                .for_each(|(g, (&u, &v))| *g = u - v.exp() * sum_up);
        }
        Tensor::from_vec(shape.to_vec(), grad)?
    };
    graph.accumulate_grad(input, input_grad)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// SIMD-friendly backward helpers
// ---------------------------------------------------------------------------

#[inline(always)]
fn relu_backward_slice(grad: &mut [f32], input: &[f32]) {
    debug_assert_eq!(grad.len(), input.len());
    grad.iter_mut().zip(input.iter()).for_each(|(g, &x)| {
        if x <= 0.0 {
            *g = 0.0;
        }
    });
}

#[inline(always)]
fn gelu_backward_slice(grad: &mut [f32], input: &[f32]) {
    debug_assert_eq!(grad.len(), input.len());
    grad.iter_mut().zip(input.iter()).for_each(|(g, &x)| {
        let a = 1.702 * x;
        let ea = (-a).exp();
        let s = 1.0 / (1.0 + ea);
        *g *= s + x * 1.702 * s * (1.0 - s);
    });
}

#[inline(always)]
fn silu_backward_slice(grad: &mut [f32], input: &[f32]) {
    debug_assert_eq!(grad.len(), input.len());
    grad.iter_mut().zip(input.iter()).for_each(|(g, &x)| {
        let s = 1.0 / (1.0 + (-x).exp());
        *g *= s + x * s * (1.0 - s);
    });
}

#[inline(always)]
fn mish_backward_slice(grad: &mut [f32], input: &[f32]) {
    debug_assert_eq!(grad.len(), input.len());
    grad.iter_mut().zip(input.iter()).for_each(|(g, &x)| {
        let sp = (1.0 + x.exp()).ln();
        let tanh_sp = sp.tanh();
        let sech2_sp = 1.0 - tanh_sp * tanh_sp;
        let sig = 1.0 / (1.0 + (-x).exp());
        *g *= tanh_sp + x * sech2_sp * sig;
    });
}

#[inline(always)]
fn leaky_relu_backward_slice(grad: &mut [f32], input: &[f32], slope: f32) {
    debug_assert_eq!(grad.len(), input.len());
    grad.iter_mut().zip(input.iter()).for_each(|(g, &x)| {
        if x < 0.0 {
            *g *= slope;
        }
    });
}
