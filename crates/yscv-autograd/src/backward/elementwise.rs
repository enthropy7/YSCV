use yscv_tensor::Tensor;

use super::error::AutogradError;
use super::graph::Graph;
use super::node::NodeId;

use super::reduce_broadcast_gradient;

/// Add backward: gradient is passed through with broadcast reduction.
pub(crate) fn add_backward(
    graph: &mut Graph,
    upstream: Tensor,
    left: NodeId,
    right: NodeId,
) -> Result<(), AutogradError> {
    let (left_grad, right_grad) = {
        let ls = graph.nodes[left.0].value.shape().to_vec();
        let rs = graph.nodes[right.0].value.shape().to_vec();
        (
            reduce_broadcast_gradient(&upstream, &ls)?,
            reduce_broadcast_gradient(&upstream, &rs)?,
        )
    };
    graph.accumulate_grad(left, left_grad)?;
    graph.accumulate_grad(right, right_grad)?;
    Ok(())
}

/// Sub backward: left gets upstream, right gets negated upstream.
pub(crate) fn sub_backward(
    graph: &mut Graph,
    upstream: Tensor,
    left: NodeId,
    right: NodeId,
) -> Result<(), AutogradError> {
    let (left_grad, right_grad) = {
        let ls = graph.nodes[left.0].value.shape().to_vec();
        let rs = graph.nodes[right.0].value.shape().to_vec();
        let neg_up = graph.dispatch_neg(&upstream);
        (
            reduce_broadcast_gradient(&upstream, &ls)?,
            reduce_broadcast_gradient(&neg_up, &rs)?,
        )
    };
    graph.accumulate_grad(left, left_grad)?;
    graph.accumulate_grad(right, right_grad)?;
    Ok(())
}

/// Mul backward: cross-multiply with upstream.
pub(crate) fn mul_backward(
    graph: &mut Graph,
    upstream: Tensor,
    left: NodeId,
    right: NodeId,
) -> Result<(), AutogradError> {
    let (left_grad, right_grad) = {
        let lv = &graph.nodes[left.0].value;
        let rv = &graph.nodes[right.0].value;
        let ls = lv.shape().to_vec();
        let rs = rv.shape().to_vec();
        let ll = graph.dispatch_mul(&upstream, rv)?;
        let rl = graph.dispatch_mul(&upstream, lv)?;
        (
            reduce_broadcast_gradient(&ll, &ls)?,
            reduce_broadcast_gradient(&rl, &rs)?,
        )
    };
    graph.accumulate_grad(left, left_grad)?;
    graph.accumulate_grad(right, right_grad)?;
    Ok(())
}

/// Div backward: d(a/b)/da = 1/b, d(a/b)/db = -a/b^2.
pub(crate) fn div_backward(
    graph: &mut Graph,
    upstream: Tensor,
    left: NodeId,
    right: NodeId,
) -> Result<(), AutogradError> {
    let (left_grad, right_grad) = {
        let lv = &graph.nodes[left.0].value;
        let rv = &graph.nodes[right.0].value;
        let ls = lv.shape().to_vec();
        let rs = rv.shape().to_vec();
        let inv_b = rv.reciprocal();
        let ll = graph.dispatch_mul(&upstream, &inv_b)?;
        let lg = reduce_broadcast_gradient(&ll, &ls)?;
        let b_sq = graph.dispatch_mul(rv, rv)?;
        let neg_a = graph.dispatch_neg(lv);
        let neg_a_over_b_sq = neg_a.div(&b_sq)?;
        let rl = graph.dispatch_mul(&upstream, &neg_a_over_b_sq)?;
        let rg = reduce_broadcast_gradient(&rl, &rs)?;
        (lg, rg)
    };
    graph.accumulate_grad(left, left_grad)?;
    graph.accumulate_grad(right, right_grad)?;
    Ok(())
}

/// Neg backward.
pub(crate) fn neg_backward(
    graph: &mut Graph,
    upstream: Tensor,
    input: NodeId,
) -> Result<(), AutogradError> {
    let input_grad = graph.dispatch_neg(&upstream);
    graph.accumulate_grad(input, input_grad)?;
    Ok(())
}

/// Pow backward: d(b^e)/db = e * b^(e-1), d(b^e)/de = b^e * ln(b).
pub(crate) fn pow_backward(
    graph: &mut Graph,
    upstream: Tensor,
    index: usize,
    base: NodeId,
    exponent: NodeId,
) -> Result<(), AutogradError> {
    let (base_grad, exp_grad) = {
        let bv = &graph.nodes[base.0].value;
        let ev = &graph.nodes[exponent.0].value;
        let ov = &graph.nodes[index].value;
        let bs = bv.shape().to_vec();
        let es = ev.shape().to_vec();

        let e_minus_1 = ev.add_scalar(-1.0);
        let b_pow_em1 = bv.pow(&e_minus_1)?;
        let local_base = graph.dispatch_mul(ev, &b_pow_em1)?;
        let bgr = graph.dispatch_mul(&upstream, &local_base)?;
        let bg = reduce_broadcast_gradient(&bgr, &bs)?;

        let ln_b = bv.ln();
        let local_exp = graph.dispatch_mul(ov, &ln_b)?;
        let egr = graph.dispatch_mul(&upstream, &local_exp)?;
        let eg = reduce_broadcast_gradient(&egr, &es)?;
        (bg, eg)
    };
    graph.accumulate_grad(base, base_grad)?;
    graph.accumulate_grad(exponent, exp_grad)?;
    Ok(())
}

/// Abs backward: grad *= sign(x), with sign(0) = 0.
pub(crate) fn abs_backward(
    graph: &mut Graph,
    upstream: Tensor,
    input: NodeId,
) -> Result<(), AutogradError> {
    let input_grad = {
        let iv = &graph.nodes[input.0].value;
        let mut result = upstream;
        abs_backward_slice(result.data_mut(), iv.data());
        result
    };
    graph.accumulate_grad(input, input_grad)?;
    Ok(())
}

/// Clamp backward: zero out grad where input is outside [min_val, max_val].
pub(crate) fn clamp_backward(
    graph: &mut Graph,
    upstream: Tensor,
    input: NodeId,
    min_bits: u32,
    max_bits: u32,
) -> Result<(), AutogradError> {
    let min_val = f32::from_bits(min_bits);
    let max_val = f32::from_bits(max_bits);
    let input_grad = {
        let iv = &graph.nodes[input.0].value;
        let mut result = upstream;
        clamp_backward_slice(result.data_mut(), iv.data(), min_val, max_val);
        result
    };
    graph.accumulate_grad(input, input_grad)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// SIMD-friendly backward helpers
// ---------------------------------------------------------------------------

#[inline(always)]
fn abs_backward_slice(grad: &mut [f32], input: &[f32]) {
    debug_assert_eq!(grad.len(), input.len());
    grad.iter_mut().zip(input.iter()).for_each(|(g, &v)| {
        if v < 0.0 {
            *g = -*g;
        } else if v == 0.0 {
            *g = 0.0;
        }
    });
}

#[inline(always)]
fn clamp_backward_slice(grad: &mut [f32], input: &[f32], min_val: f32, max_val: f32) {
    debug_assert_eq!(grad.len(), input.len());
    grad.iter_mut().zip(input.iter()).for_each(|(g, &v)| {
        if v < min_val || v > max_val {
            *g = 0.0;
        }
    });
}
