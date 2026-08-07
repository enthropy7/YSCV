use crate::error::OnnxError;
use crate::ir::{Changed, Graph, NodeId, Op, Pass, ValueId, WeightLayout};
use crate::plan::NodeKind;
use crate::runner::{ConstEvalTables, TensorEnv, execute_node_kind};

/// Evaluates nodes whose inputs are all load-time constants and replaces them
/// with the result.
///
/// # Why this executes operators
///
/// Folding by execution reuses the runtime's operator implementations. The
/// alternative — a separate const-eval interpreter over some whitelist — is a
/// second implementation of the same operators that then has to be kept in
/// lockstep with `runner/dispatch.rs`, which is the class of duplication this
/// whole IR migration exists to remove. ONNX Runtime folds by execution for the
/// same reason.
///
/// What was wrong before was the machinery around each evaluation, not the
/// evaluation. The string version built a throwaway `OnnxModel` per foldable
/// node, called `rebuild_runtime_index()` on it — the entire plan builder:
/// fusion scan, slot assignment, weight prepacking — and then `run_onnx_model`,
/// which sets up thread pools, output masks and tower-parallel probing. All to
/// run one operator. Here a node is dispatched directly onto a bare
/// [`TensorEnv`], with no model, index or plan involved.
///
/// # Ordering
///
/// Folding a node makes its output constant, which can only make *later* nodes
/// foldable. `ReorderForFusion` leaves `node_ids()` in topological order, so one
/// forward sweep reaches the same fixed point the old `loop { find(..) }`
/// reached by rescanning from index 0 every iteration.
pub(crate) struct FoldConstants;

/// Operators that must not be evaluated at load time.
///
/// Convolutions are excluded because they are expensive and their weights may
/// be physically pre-permuted. The `Random*` family is excluded because folding
/// would freeze one draw into the model, turning a per-inference distribution
/// into a constant.
///
/// This is a denylist, so a newly added operator is foldable by default. That
/// is deliberate — an allowlist would fold strictly less than the previous
/// implementation and silently regress models relying on the wider reach — and
/// [`RESULT_GROWTH_LIMIT`] bounds what a misjudged operator can cost.
fn is_denied(op: &Op) -> bool {
    matches!(op, Op::Conv | Op::ConvTranspose | Op::DeformConv)
        || op.as_str().starts_with("Random")
        || op.as_str() == "Multinomial"
}

/// How much larger than its inputs a folded result may be.
///
/// `Expand` and `Tile` over a constant can inflate a small initializer without
/// bound, and the result is baked into the model rather than computed on demand.
const RESULT_GROWTH_LIMIT: usize = 4;

/// Results at or below this many elements are always allowed, so that genuinely
/// small expansions — a scalar broadcast to a bias vector — still fold.
const RESULT_SIZE_FLOOR: usize = 1 << 16;

impl Pass for FoldConstants {
    fn name(&self) -> &'static str {
        "fold_constants"
    }

    fn run(&self, graph: &mut Graph) -> Result<Changed, OnnxError> {
        let tables = ConstEvalTables::default();
        let mut changed = false;

        for node_id in graph.node_ids().collect::<Vec<_>>() {
            let Some(inputs) = foldable_inputs(graph, node_id) else {
                continue;
            };
            let Some(folded) = evaluate(graph, node_id, &inputs, &tables) else {
                // Not foldable in practice — an operator the runner declines,
                // or a result too large to be worth baking in. Skipping is not
                // an error; the node simply stays in the graph. The string
                // version aborted the whole pass here, so one awkward operator
                // stopped every later fold too.
                continue;
            };

            let Some(node) = graph.node(node_id) else {
                continue;
            };
            let output = node.outputs[0];
            // An `Identity` folded away leaves its operand's bytes under a new
            // name. The layout tag has to travel with them, or the consumer
            // reads a permuted weight as ONNX-native OIHW.
            let carried = (node.op == Op::Identity)
                .then(|| node.inputs.first().copied().flatten())
                .flatten()
                .map(|src| graph.weight_layout(src))
                .filter(|l| *l != WeightLayout::Oihw);
            graph.set_constant(output, folded);
            if let Some(layout) = carried {
                graph.set_weight_layout(output, layout);
            }
            graph.remove_node(node_id);
            changed = true;
        }

        Ok(changed)
    }
}

/// The node's operands, if every one of them is a constant this pass may read.
fn foldable_inputs(graph: &Graph, node_id: NodeId) -> Option<Vec<(String, ValueId)>> {
    let node = graph.node(node_id)?;
    if node.inputs.is_empty() || node.outputs.len() != 1 || is_denied(&node.op) {
        return None;
    }
    // Only the sole output may be consumed; a graph output has to keep its
    // producer so the name survives lowering.
    if graph.is_graph_output(node.outputs[0]) {
        return None;
    }

    let mut operands = Vec::with_capacity(node.inputs.len());
    for slot in &node.inputs {
        // An omitted optional input stays omitted.
        let Some(value) = slot else { continue };
        graph.constant(*value)?;
        // A pre-permuted weight's bytes no longer match the graph's logical
        // view of it, so evaluating an operator against one would compute
        // against the wrong layout. `Identity` is the exception: it copies
        // bytes without interpreting them, and refusing it is not the safe
        // choice — the node then survives to run time, where it hands its
        // consumer a permuted tensor carrying no layout tag, and the Conv
        // reading it parses `[KH, KW, C, 1]` as `[O, I, KH, KW]`.
        if graph.weight_layout(*value) != WeightLayout::Oihw && node.op != Op::Identity {
            return None;
        }
        operands.push((graph.value(*value).name.clone(), *value));
    }
    Some(operands)
}

/// Runs the operator on a bare environment and returns its output.
///
/// `None` means "not foldable" rather than "broken": the runner may not
/// implement this operator, or the result may be too large to bake in.
fn evaluate(
    graph: &Graph,
    node_id: NodeId,
    inputs: &[(String, ValueId)],
    tables: &ConstEvalTables,
) -> Option<yscv_tensor::Tensor> {
    let node = graph.node(node_id)?;
    let onnx_node = graph.to_onnx_node(node_id)?;
    let output_name = graph.value(node.outputs[0]).name.clone();

    let mut input_elements = 0usize;
    let mut env = TensorEnv::for_const_eval(tables);
    for (name, value) in inputs {
        let tensor = graph.constant(*value)?;
        input_elements = input_elements.saturating_add(tensor.len());
        // `Tensor` is `Arc`-backed, so this is a refcount bump, not a copy.
        env.insert(name.clone(), tensor.clone());
    }

    let kind = NodeKind::from_op_type(&onnx_node.op_type);
    execute_node_kind(&onnx_node, &mut env, kind).ok()?;
    let result = env.remove(&output_name)?;

    let limit = input_elements
        .saturating_mul(RESULT_GROWTH_LIMIT)
        .max(RESULT_SIZE_FLOOR);
    if result.len() > limit {
        return None;
    }
    Some(result)
}
