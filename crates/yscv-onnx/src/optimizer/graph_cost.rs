use crate::attr::Attr;
use crate::loader::{OnnxAttribute, OnnxModel, OnnxNode};
use crate::shape_infer::{Dim, ShapeInference, ShapeMap, TensorShape};
use std::fmt::Write as _;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeCost {
    pub index: usize,
    pub name: String,
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub output_shape: Option<TensorShape>,
    pub macs: u64,
    pub element_ops: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub score: u64,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphCost {
    pub node_count: usize,
    pub known_nodes: usize,
    pub unknown_nodes: usize,
    pub estimated_macs: u64,
    pub estimated_element_ops: u64,
    pub estimated_bytes_read: u64,
    pub estimated_bytes_written: u64,
    pub score: u64,
    pub nodes: Vec<NodeCost>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphCostDiff {
    pub before_score: u64,
    pub after_score: u64,
    pub delta_score: i128,
    pub before_nodes: usize,
    pub after_nodes: usize,
    pub delta_nodes: isize,
    pub before_macs: u64,
    pub after_macs: u64,
    pub delta_macs: i128,
}

pub fn graph_cost(model: &OnnxModel, shapes: &ShapeInference) -> GraphCost {
    let mut nodes = Vec::with_capacity(model.nodes.len());
    let mut known_nodes = 0usize;
    let mut unknown_nodes = 0usize;
    let mut estimated_macs = 0u64;
    let mut estimated_element_ops = 0u64;
    let mut estimated_bytes_read = 0u64;
    let mut estimated_bytes_written = 0u64;
    let diagnostic_by_index: rustc_hash::FxHashMap<usize, String> = shapes
        .diagnostics
        .iter()
        .map(|d| (d.node_index, d.error.to_string()))
        .collect();

    for (index, node) in model.nodes.iter().enumerate() {
        let output_shape = node
            .outputs
            .first()
            .and_then(|name| shapes.shapes.get(name))
            .cloned();
        let mut cost = cost_node(model, &shapes.shapes, node, output_shape.as_ref());
        cost.index = index;
        cost.name = node.name.clone();
        cost.op_type = node.op_type.clone();
        cost.inputs = node.inputs.clone();
        cost.outputs = node.outputs.clone();
        if cost.reason.is_none() {
            cost.reason = diagnostic_by_index.get(&index).cloned();
        }
        if cost.reason.is_some() {
            unknown_nodes += 1;
        } else {
            known_nodes += 1;
        }
        estimated_macs = estimated_macs.saturating_add(cost.macs);
        estimated_element_ops = estimated_element_ops.saturating_add(cost.element_ops);
        estimated_bytes_read = estimated_bytes_read.saturating_add(cost.bytes_read);
        estimated_bytes_written = estimated_bytes_written.saturating_add(cost.bytes_written);
        nodes.push(cost);
    }

    let score = estimated_macs
        .saturating_add(estimated_element_ops)
        .saturating_add(estimated_bytes_read.saturating_add(estimated_bytes_written) / 4);

    GraphCost {
        node_count: model.nodes.len(),
        known_nodes,
        unknown_nodes,
        estimated_macs,
        estimated_element_ops,
        estimated_bytes_read,
        estimated_bytes_written,
        score,
        nodes,
    }
}

pub fn graph_cost_diff(before: &GraphCost, after: &GraphCost) -> GraphCostDiff {
    GraphCostDiff {
        before_score: before.score,
        after_score: after.score,
        delta_score: after.score as i128 - before.score as i128,
        before_nodes: before.node_count,
        after_nodes: after.node_count,
        delta_nodes: after.node_count as isize - before.node_count as isize,
        before_macs: before.estimated_macs,
        after_macs: after.estimated_macs,
        delta_macs: after.estimated_macs as i128 - before.estimated_macs as i128,
    }
}

/// Formats a deterministic text snapshot of graph-cost estimates.
///
/// Intended for benchmarking and review workflows where the output is saved
/// and diffed across commits to check whether optimizer changes make the graph
/// lighter. The per-node section is sorted by a stable textual key rather than
/// execution order so incidental reordering produces less diff noise.
pub fn graph_cost_report(cost: &GraphCost) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "graph_cost v1");
    let _ = writeln!(out, "summary.node_count={}", cost.node_count);
    let _ = writeln!(out, "summary.known_nodes={}", cost.known_nodes);
    let _ = writeln!(out, "summary.unknown_nodes={}", cost.unknown_nodes);
    let _ = writeln!(out, "summary.estimated_macs={}", cost.estimated_macs);
    let _ = writeln!(
        out,
        "summary.estimated_element_ops={}",
        cost.estimated_element_ops
    );
    let _ = writeln!(
        out,
        "summary.estimated_bytes_read={}",
        cost.estimated_bytes_read
    );
    let _ = writeln!(
        out,
        "summary.estimated_bytes_written={}",
        cost.estimated_bytes_written
    );
    let _ = writeln!(out, "summary.score={}", cost.score);
    out.push('\n');
    out.push_str("nodes\n");

    let mut nodes: Vec<&NodeCost> = cost.nodes.iter().collect();
    nodes.sort_by(|a, b| {
        stable_node_key(a)
            .cmp(&stable_node_key(b))
            .then(a.index.cmp(&b.index))
    });

    for node in nodes {
        let _ = writeln!(out, "- key={}", stable_node_key(node));
        let _ = writeln!(out, "  index={}", node.index);
        let _ = writeln!(out, "  op_type={}", node.op_type);
        let _ = writeln!(out, "  name={}", printable_field(&node.name));
        let _ = writeln!(out, "  inputs={}", join_fields(&node.inputs));
        let _ = writeln!(out, "  outputs={}", join_fields(&node.outputs));
        let _ = writeln!(
            out,
            "  output_shape={}",
            format_output_shape(node.output_shape.as_ref())
        );
        let _ = writeln!(out, "  macs={}", node.macs);
        let _ = writeln!(out, "  element_ops={}", node.element_ops);
        let _ = writeln!(out, "  bytes_read={}", node.bytes_read);
        let _ = writeln!(out, "  bytes_written={}", node.bytes_written);
        let _ = writeln!(out, "  score={}", node.score);
        let _ = writeln!(out, "  reason={}", format_reason(node.reason.as_deref()));
    }

    out
}

fn cost_node(
    model: &OnnxModel,
    shapes: &ShapeMap,
    node: &OnnxNode,
    output_shape: Option<&TensorShape>,
) -> NodeCost {
    let mut cost = NodeCost {
        index: 0,
        name: String::new(),
        op_type: String::new(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        output_shape: output_shape.cloned(),
        macs: 0,
        element_ops: 0,
        bytes_read: 0,
        bytes_written: 0,
        score: 0,
        reason: None,
    };

    match node.op_type.as_str() {
        "Conv" | "Conv_Relu" | "Conv_SiLU" => cost_conv(model, node, output_shape, &mut cost),
        "MatMul" => cost_matmul(model, shapes, node, output_shape, &mut cost),
        "Gemm" => cost_gemm(model, node, output_shape, &mut cost),
        "Relu"
        | "Clip"
        | "Sigmoid"
        | "Tanh"
        | "Add"
        | "Sub"
        | "Mul"
        | "Div"
        | "Pow"
        | "BatchNormalization"
        | "BatchNormalization_Relu" => {
            let weight = match node.op_type.as_str() {
                "Sigmoid" | "Tanh" => 8,
                "BatchNormalization" | "BatchNormalization_Relu" => 4,
                _ => 1,
            };
            cost_elementwise(output_shape, weight, &mut cost);
        }
        "Transpose" => cost_materialized_copy(output_shape, &mut cost),
        "Reshape" | "Flatten" | "Squeeze" | "Unsqueeze" | "Identity" | "Dropout" | "Constant" => {
            cost_metadata_only(output_shape, &mut cost);
        }
        "Concat" => cost_concat(output_shape, &mut cost),
        "MaxPool" | "AveragePool" => cost_pool(node, output_shape, &mut cost),
        "GlobalAveragePool" => cost_global_pool(shapes, node, output_shape, &mut cost),
        _ => cost.reason = Some("unsupported cost rule".to_string()),
    }

    cost.score = cost
        .macs
        .saturating_add(cost.element_ops)
        .saturating_add(cost.bytes_read.saturating_add(cost.bytes_written) / 4);
    cost
}

fn cost_conv(
    model: &OnnxModel,
    node: &OnnxNode,
    output_shape: Option<&TensorShape>,
    cost: &mut NodeCost,
) {
    let Some(out) = output_shape else {
        cost.reason = Some("missing Conv output shape".to_string());
        return;
    };
    let Some(weight_name) = node.inputs.get(1) else {
        cost.reason = Some("Conv missing weight".to_string());
        return;
    };
    let Some(weight) = model.initializers.get(weight_name) else {
        cost.reason = Some("Conv weight is not constant".to_string());
        return;
    };
    let Some(out_numel) = out.num_elements() else {
        cost.reason = Some("Conv output has unknown dimensions".to_string());
        return;
    };
    let w_shape = weight.shape();
    if w_shape.len() != 4 {
        cost.reason = Some("Conv weight is not rank 4".to_string());
        return;
    }
    let group = int_attr(node, Attr::Group).unwrap_or(1).max(1) as usize;
    // ONNX-native `[O, I/G, KH, KW]`; see `plan::prepack`.
    let (ic_per_group, kh, kw) = (w_shape[1], w_shape[2], w_shape[3]);
    let kernel_work = if group == w_shape[0] && ic_per_group == 1 {
        kh.saturating_mul(kw)
    } else {
        ic_per_group.saturating_mul(kh).saturating_mul(kw)
    };
    cost.macs = out_numel.saturating_mul(kernel_work as u64);
    if matches!(node.op_type.as_str(), "Conv_Relu" | "Conv_SiLU") {
        cost.element_ops = out_numel;
    }
    cost.bytes_written = out_numel.saturating_mul(4);
    cost.bytes_read = cost
        .bytes_written
        .saturating_add((weight.data().len() as u64).saturating_mul(4));
}

fn cost_matmul(
    model: &OnnxModel,
    shapes: &ShapeMap,
    node: &OnnxNode,
    output_shape: Option<&TensorShape>,
    cost: &mut NodeCost,
) {
    let (Some(a), Some(b)) = (node.inputs.first(), node.inputs.get(1)) else {
        cost.reason = Some("MatMul missing inputs".to_string());
        return;
    };
    let Some(out) = output_shape else {
        cost.reason = Some("missing MatMul output shape".to_string());
        return;
    };
    let Some(out_numel) = out.num_elements() else {
        cost.reason = Some("MatMul output has unknown dimensions".to_string());
        return;
    };
    let Some(k) = matmul_reduction_dim(model, shapes, a, b) else {
        cost.reason = Some("MatMul reduction dimension unknown".to_string());
        return;
    };
    cost.macs = out_numel.saturating_mul(k as u64);
    cost.bytes_written = out_numel.saturating_mul(4);
    cost.bytes_read = cost.bytes_written;
    for input in [&a, &b] {
        if let Some(t) = model.initializers.get(*input) {
            cost.bytes_read = cost
                .bytes_read
                .saturating_add((t.data().len() as u64).saturating_mul(4));
        }
    }
}

fn cost_gemm(
    model: &OnnxModel,
    node: &OnnxNode,
    output_shape: Option<&TensorShape>,
    cost: &mut NodeCost,
) {
    let Some(out) = output_shape else {
        cost.reason = Some("missing Gemm output shape".to_string());
        return;
    };
    let Some(out_numel) = out.num_elements() else {
        cost.reason = Some("Gemm output has unknown dimensions".to_string());
        return;
    };
    let Some(b_name) = node.inputs.get(1) else {
        cost.reason = Some("Gemm missing B input".to_string());
        return;
    };
    let Some(b) = model.initializers.get(b_name) else {
        cost.reason = Some("Gemm B shape unknown".to_string());
        return;
    };
    if b.shape().len() != 2 {
        cost.reason = Some("Gemm B is not rank 2".to_string());
        return;
    }
    let trans_b = int_attr(node, Attr::TransB).unwrap_or(0) != 0;
    let k = if trans_b { b.shape()[1] } else { b.shape()[0] };
    cost.macs = out_numel.saturating_mul(k as u64);
    cost.bytes_written = out_numel.saturating_mul(4);
    cost.bytes_read = cost
        .bytes_written
        .saturating_add((b.data().len() as u64).saturating_mul(4));
}

fn cost_elementwise(output_shape: Option<&TensorShape>, weight: u64, cost: &mut NodeCost) {
    let Some(out_numel) = output_shape.and_then(TensorShape::num_elements) else {
        cost.reason = Some("elementwise output shape unknown".to_string());
        return;
    };
    cost.element_ops = out_numel.saturating_mul(weight);
    cost.bytes_read = out_numel.saturating_mul(4);
    cost.bytes_written = out_numel.saturating_mul(4);
}

fn cost_materialized_copy(output_shape: Option<&TensorShape>, cost: &mut NodeCost) {
    let Some(out_numel) = output_shape.and_then(TensorShape::num_elements) else {
        cost.reason = Some("copy output shape unknown".to_string());
        return;
    };
    cost.bytes_read = out_numel.saturating_mul(4);
    cost.bytes_written = out_numel.saturating_mul(4);
}

fn cost_metadata_only(output_shape: Option<&TensorShape>, cost: &mut NodeCost) {
    if output_shape.is_none() {
        cost.reason = Some("metadata op output shape unknown".to_string());
    }
}

fn cost_concat(output_shape: Option<&TensorShape>, cost: &mut NodeCost) {
    cost_materialized_copy(output_shape, cost);
}

fn cost_pool(node: &OnnxNode, output_shape: Option<&TensorShape>, cost: &mut NodeCost) {
    let Some(out_numel) = output_shape.and_then(TensorShape::num_elements) else {
        cost.reason = Some("pool output shape unknown".to_string());
        return;
    };
    let kernel = ints_attr(node, Attr::KernelShape)
        .and_then(|v| v.first().zip(v.get(1)).map(|(h, w)| h.max(&1) * w.max(&1)))
        .unwrap_or(1) as u64;
    cost.element_ops = out_numel.saturating_mul(kernel);
    cost.bytes_read = out_numel.saturating_mul(kernel).saturating_mul(4);
    cost.bytes_written = out_numel.saturating_mul(4);
}

fn cost_global_pool(
    shapes: &ShapeMap,
    node: &OnnxNode,
    output_shape: Option<&TensorShape>,
    cost: &mut NodeCost,
) {
    let Some(out_numel) = output_shape.and_then(TensorShape::num_elements) else {
        cost.reason = Some("global pool output shape unknown".to_string());
        return;
    };
    // The pooled input is an activation, so its spatial extent comes from shape
    // inference, not the initializer table — every element of every input pixel
    // is read once to form the average.
    let input_spatial = node
        .inputs
        .first()
        .and_then(|name| shapes.get(name))
        .and_then(|s| s.dim(2).zip(s.dim(3)).map(|(h, w)| h.saturating_mul(w)))
        .unwrap_or(1) as u64;
    cost.element_ops = out_numel.saturating_mul(input_spatial);
    cost.bytes_read = cost.element_ops.saturating_mul(4);
    cost.bytes_written = out_numel.saturating_mul(4);
}

/// Resolves the MatMul contraction dimension `K`. Prefers exact initializer
/// shapes (`A[..,K]` last dim, `B[K,..]` first dim), then falls back to inferred
/// activation shapes so activation×activation MatMuls — attention scores and the
/// like — are costed instead of written off as unknown.
fn matmul_reduction_dim(model: &OnnxModel, shapes: &ShapeMap, a: &str, b: &str) -> Option<usize> {
    if let Some(k) = model
        .initializers
        .get(a)
        .and_then(|t| t.shape().last().copied())
    {
        return Some(k);
    }
    if let Some(k) = model
        .initializers
        .get(b)
        .and_then(|t| t.shape().first().copied())
    {
        return Some(k);
    }
    if let Some(Dim::Known(k)) = shapes.get(a).and_then(|s| s.dims.last().copied()) {
        return Some(k);
    }
    shapes
        .get(b)
        .and_then(|s| match s.dims.get(s.rank().checked_sub(2)?)? {
            Dim::Known(k) => Some(*k),
            Dim::Unknown => None,
        })
}

fn int_attr(node: &OnnxNode, name: Attr) -> Option<i64> {
    if let Some(OnnxAttribute::Int(v)) = node.attributes.get(&name) {
        Some(*v)
    } else {
        None
    }
}

fn stable_node_key(node: &NodeCost) -> String {
    format!(
        "{}|{}|{}|{}",
        node.op_type,
        printable_field(&node.name),
        join_fields(&node.outputs),
        join_fields(&node.inputs)
    )
}

fn printable_field(s: &str) -> &str {
    if s.is_empty() { "-" } else { s }
}

fn join_fields(fields: &[String]) -> String {
    if fields.is_empty() {
        "-".to_string()
    } else {
        fields
            .iter()
            .map(|s| printable_field(s))
            .collect::<Vec<_>>()
            .join(",")
    }
}

fn format_output_shape(shape: Option<&TensorShape>) -> String {
    let Some(shape) = shape else {
        return "?".to_string();
    };
    let mut out = String::from("[");
    for (i, dim) in shape.dims.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        match dim {
            Dim::Known(v) => {
                let _ = write!(out, "{v}");
            }
            Dim::Unknown => out.push('?'),
        }
    }
    out.push(']');
    out
}

fn format_reason(reason: Option<&str>) -> &str {
    reason.unwrap_or("-")
}

fn ints_attr(node: &OnnxNode, name: Attr) -> Option<Vec<i64>> {
    if let Some(OnnxAttribute::Ints(v)) = node.attributes.get(&name) {
        Some(v.clone())
    } else {
        None
    }
}
