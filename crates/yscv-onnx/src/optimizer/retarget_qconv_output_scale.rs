use crate::ir::{Graph, NodeId, ValueId};

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
pub(super) fn run(graph: &mut Graph) -> usize {
    let mut rewrites: Vec<(NodeId, NodeId, ValueId, ValueId)> = Vec::new();
    for ci in graph.node_ids().collect::<Vec<_>>() {
        let Some(conv) = graph.node(ci) else {
            continue;
        };
        if conv.op.as_str() != "QLinearConv" || conv.inputs.len() < 8 || conv.outputs.len() != 1 {
            continue;
        }
        let Some(di) = graph.sole_consumer(conv.outputs[0]).map(|u| u.node) else {
            continue;
        };
        let Some(dq) = graph.node(di) else {
            continue;
        };
        if dq.op.as_str() != "DequantizeLinear" || dq.inputs.len() < 3 || dq.outputs.len() != 1 {
            continue;
        }
        let Some(mut qi) = graph.sole_consumer(dq.outputs[0]).map(|u| u.node) else {
            continue;
        };
        let mut node = graph.node(qi);
        if node.is_some_and(|node| matches!(node.op.as_str(), "Relu" | "Clip")) {
            let Some(act) = node else {
                continue;
            };
            if act.outputs.len() != 1
                || act.inputs.first().copied().flatten() != Some(dq.outputs[0])
            {
                continue;
            }
            let Some(next) = graph.sole_consumer(act.outputs[0]).map(|u| u.node) else {
                continue;
            };
            qi = next;
            node = graph.node(qi);
        }
        let Some(q) = node else {
            continue;
        };
        if q.op.as_str() != "QuantizeLinear" || q.inputs.len() < 3 {
            continue;
        }
        let (Some(scale), Some(zp)) = (q.inputs[1], q.inputs[2]) else {
            continue;
        };
        if conv.inputs[6] == Some(scale) && conv.inputs[7] == Some(zp) {
            continue;
        }
        rewrites.push((ci, di, scale, zp));
    }

    for (ci, di, scale, zp) in &rewrites {
        graph.set_input(*ci, 6, *scale);
        graph.set_input(*ci, 7, *zp);
        graph.set_input(*di, 1, *scale);
        graph.set_input(*di, 2, *zp);
    }
    rewrites.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Node, Op};

    fn graph(activation: Option<&str>, fan_out: bool, conv_is_output: bool) -> Graph {
        let mut graph = Graph::new();
        let value = |graph: &mut Graph, name: &str| graph.value_by_name(name);
        let x = value(&mut graph, "x");
        let w = value(&mut graph, "w");
        let old_scale = value(&mut graph, "old_scale");
        let old_zp = value(&mut graph, "old_zp");
        let new_scale = value(&mut graph, "new_scale");
        let new_zp = value(&mut graph, "new_zp");
        let conv_out = value(&mut graph, "conv_out");
        let dq_out = value(&mut graph, "dq_out");
        let output = value(&mut graph, "output");
        graph.push_node(Node {
            op: Op::from_op_type("QLinearConv"),
            name: "conv".into(),
            inputs: vec![
                Some(x),
                Some(old_scale),
                Some(old_zp),
                Some(w),
                Some(old_scale),
                Some(old_zp),
                Some(old_scale),
                Some(old_zp),
            ],
            outputs: vec![conv_out],
            attributes: Default::default(),
        });
        graph.push_node(Node {
            op: Op::from_op_type("DequantizeLinear"),
            name: "dq".into(),
            inputs: vec![Some(conv_out), Some(old_scale), Some(old_zp)],
            outputs: vec![dq_out],
            attributes: Default::default(),
        });
        let mut q_input = dq_out;
        if let Some(op) = activation {
            let activation_out = value(&mut graph, "activation_out");
            graph.push_node(Node {
                op: Op::from_op_type(op),
                name: "activation".into(),
                inputs: vec![Some(dq_out)],
                outputs: vec![activation_out],
                attributes: Default::default(),
            });
            q_input = activation_out;
        }
        graph.push_node(Node {
            op: Op::from_op_type("QuantizeLinear"),
            name: "q".into(),
            inputs: vec![Some(q_input), Some(new_scale), Some(new_zp)],
            outputs: vec![output],
            attributes: Default::default(),
        });
        if fan_out {
            let side = value(&mut graph, "side");
            graph.push_node(Node {
                op: Op::from_op_type("Identity"),
                name: "side".into(),
                inputs: vec![Some(conv_out)],
                outputs: vec![side],
                attributes: Default::default(),
            });
        }
        graph.set_graph_outputs(if conv_is_output {
            vec![output, conv_out]
        } else {
            vec![output]
        });
        graph
    }

    fn input_name<'a>(graph: &'a Graph, node_name: &str, port: usize) -> &'a str {
        let id = graph
            .node_ids()
            .find(|id| graph.node(*id).is_some_and(|node| node.name == node_name))
            .expect("test node");
        let value = graph.node(id).expect("live test node").inputs[port].expect("test input");
        &graph.value(value).name
    }

    #[test]
    fn retargets_conv_and_dequantizer_to_next_scale() {
        let mut graph = graph(None, false, false);
        assert_eq!(run(&mut graph), 1);
        for node in ["conv", "dq"] {
            let scale_port = if node == "conv" { 6 } else { 1 };
            let zp_port = if node == "conv" { 7 } else { 2 };
            assert_eq!(input_name(&graph, node, scale_port), "new_scale");
            assert_eq!(input_name(&graph, node, zp_port), "new_zp");
        }
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn traverses_relu_and_clip() {
        for activation in ["Relu", "Clip"] {
            let mut graph = graph(Some(activation), false, false);
            assert_eq!(run(&mut graph), 1);
            assert_eq!(input_name(&graph, "conv", 6), "new_scale");
            assert_eq!(input_name(&graph, "dq", 2), "new_zp");
        }
    }

    #[test]
    fn rejects_conv_fan_out_and_graph_output() {
        for (fan_out, conv_is_output) in [(true, false), (false, true)] {
            let mut graph = graph(None, fan_out, conv_is_output);
            assert_eq!(run(&mut graph), 0);
            assert_eq!(input_name(&graph, "conv", 6), "old_scale");
            assert_eq!(input_name(&graph, "dq", 1), "old_scale");
        }
    }
}
