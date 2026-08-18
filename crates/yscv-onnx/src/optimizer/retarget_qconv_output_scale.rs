use crate::loader::OnnxModel;

/// Points a `QLinearConv`'s output scale at the one the following rescale was
/// going to produce anyway.
///
/// A QOperator graph writes `QLinearConv -> DequantizeLinear -> [Relu | Clip]
/// -> QuantizeLinear`, where the conv rounds its accumulator to int8 at one
/// scale and the boundary immediately rounds it again to the next layer's.
/// Rounding twice to eight bits costs accuracy and a full pass over the tensor;
/// letting the conv round straight to the second scale costs neither. The
/// boundary is left in place — with both sides now on the same scale it is a
/// relabel plus the activation clamp, which the runner already recognises.
///
/// Only fires when the conv's output goes nowhere else, so nothing observes the
/// intermediate scale.
///
/// Returns the number of convolutions retargeted.
pub(super) fn run(model: &mut OnnxModel) -> usize {
    let consumers = |name: &str| -> usize {
        model
            .nodes
            .iter()
            .filter(|n| n.inputs.iter().any(|i| i == name))
            .count()
    };
    let sole_consumer = |name: &str| -> Option<usize> {
        let mut found = None;
        for (i, n) in model.nodes.iter().enumerate() {
            if n.inputs.iter().any(|x| x == name) {
                if found.is_some() {
                    return None;
                }
                found = Some(i);
            }
        }
        found
    };

    // (conv index, dq index, scale name, zero-point name)
    let mut rewrites: Vec<(usize, usize, String, String)> = Vec::new();
    for (ci, conv) in model.nodes.iter().enumerate() {
        if conv.op_type != "QLinearConv" || conv.inputs.len() < 8 || conv.outputs.len() != 1 {
            continue;
        }
        let out = &conv.outputs[0];
        if model.outputs.iter().any(|o| o == out) || consumers(out) != 1 {
            continue;
        }
        let Some(di) = sole_consumer(out) else {
            continue;
        };
        let dq = &model.nodes[di];
        if dq.op_type != "DequantizeLinear" || dq.inputs.len() < 3 || dq.outputs.len() != 1 {
            continue;
        }
        let Some(mut qi) = sole_consumer(&dq.outputs[0]) else {
            continue;
        };
        if matches!(model.nodes[qi].op_type.as_str(), "Relu" | "Clip") {
            let act = &model.nodes[qi];
            if act.outputs.len() != 1 || act.inputs.first() != Some(&dq.outputs[0]) {
                continue;
            }
            match sole_consumer(&act.outputs[0]) {
                Some(next) => qi = next,
                None => continue,
            }
        }
        let q = &model.nodes[qi];
        if q.op_type != "QuantizeLinear" || q.inputs.len() < 3 {
            continue;
        }
        // Same scale already: nothing to retarget.
        if conv.inputs[6] == q.inputs[1] && conv.inputs[7] == q.inputs[2] {
            continue;
        }
        rewrites.push((ci, di, q.inputs[1].clone(), q.inputs[2].clone()));
    }

    for (ci, di, scale, zp) in &rewrites {
        let conv = &mut model.nodes[*ci];
        conv.inputs[6] = scale.clone();
        conv.inputs[7] = zp.clone();
        let dq = &mut model.nodes[*di];
        dq.inputs[1] = scale.clone();
        dq.inputs[2] = zp.clone();
    }
    rewrites.len()
}
