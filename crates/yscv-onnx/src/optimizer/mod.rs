mod analyze_nchwc;
mod eliminate_dead_code;
mod eliminate_squeeze_unsqueeze_pairs;
mod fold_constants;
mod fold_conv_bn;
mod fold_conv_const_binary;
mod fuse_activation;
mod graph_cost;
mod graph_stats;
mod remove_dropout_nodes;
mod reorder_nodes_for_fusion;
mod rewrite_convtranspose_dts;
mod strip_qdq_within_fusion_chains;

use crate::{
    error::OnnxError,
    ir::{Pass, PassManager},
    loader::OnnxModel,
    optimizer::{
        eliminate_dead_code::EliminateDeadCode, remove_dropout_nodes::RemoveDropout,
        reorder_nodes_for_fusion::ReorderForFusion,
    },
};

pub use analyze_nchwc::analyze_nchwc;
/// Re-exported so tests can drive these passes on their own, without the
/// string-based pipeline reordering the graph first.
pub(crate) use eliminate_squeeze_unsqueeze_pairs::EliminateSqueezeUnsqueezePairs;
pub(crate) use fold_constants::FoldConstants;
pub(crate) use fold_conv_bn::FoldConvBatchNorm;
pub(crate) use fold_conv_const_binary::FoldConvConstBinary;
pub(crate) use fuse_activation::FuseActivation;
pub use graph_cost::{
    GraphCost, GraphCostDiff, NodeCost, graph_cost, graph_cost_diff, graph_cost_report,
};
pub use graph_stats::{GraphStats, graph_stats};
pub(crate) use rewrite_convtranspose_dts::RewriteConvTransposeToDepthToSpace;
pub use strip_qdq_within_fusion_chains::strip_qdq_within_fusion_chains;

/// Runs the passes that have moved onto the def-use IR.
///
/// The migration is incremental, so the two representations coexist: the
/// string-based passes run first, then the model is lowered once and the IR
/// pipeline runs to a fixpoint. Each pass ported in a later change moves from
/// the list above into this pipeline.
///
/// A pass failure leaves the model exactly as it was rather than applying a
/// half-transformed graph. `optimize_onnx_graph` returns `()` and its callers
/// treat optimization as best-effort, so the failure is reported on stderr.
/// Once a genuinely fallible pass lands — constant folding executes the graph,
/// so it can — this should become a real error return.
fn run_ir_pipeline(model: &mut OnnxModel) -> Result<(), OnnxError> {
    let mut graph = model.to_ir();
    PassManager::new(pipeline()).run(&mut graph)?;
    model.apply_ir(&graph);
    Ok(())
}

/// The pass pipeline, in order.
///
/// The driver sweeps this to a fixed point, so the order is a starting point
/// rather than a correctness requirement — every pass matches through the
/// def-use index and none depends on node adjacency. Two orderings still earn
/// their place by saving a sweep:
///
/// - `FoldConvBatchNorm` before the Conv-Mul / Conv-Add folds, which then only
///   see the stray scale and bias BatchNormalization did not already absorb.
/// - The folds before `FuseActivation`, which matches a plain `Conv` and would
///   miss one already retagged as `Conv_Relu`.
///
/// `ReorderForFusion` is last because it is not for these passes at all: the
/// layer-3 plan builder still matches `nodes[i + 1]`, so the order it leaves
/// behind is what reaches `build_runtime_index`.
fn pipeline() -> Vec<Box<dyn Pass>> {
    vec![
        Box::new(RemoveDropout) as Box<dyn Pass>,
        Box::new(EliminateSqueezeUnsqueezePairs),
        Box::new(RewriteConvTransposeToDepthToSpace),
        Box::new(FoldConvBatchNorm),
        Box::new(FoldConvConstBinary::mul()),
        Box::new(FoldConvConstBinary::add()),
        Box::new(FoldConstants),
        Box::new(FuseActivation::conv_relu()),
        Box::new(FuseActivation::bn_relu()),
        Box::new(EliminateDeadCode),
        Box::new(ReorderForFusion),
    ]
}

/// Optimizes an ONNX model graph in-place for inference.
///
/// Applies load-time passes modeled after ORT's Level-1 optimizer.
///
/// The migration to the def-use IR runs in three phases, because the passes
/// still on the string representation match positionally and so must sit
/// between an ordering repair and the fusions.
///
/// 1. [`ir_cleanup_passes`] — Dropout removal, Squeeze/Unsqueeze pair
///    elimination, dead code, then the topological reorder those positional
///    passes depend on.
/// 2. Still string-based, still matching `nodes[i + 1]`:
///    - ConvTranspose(k==s) rewrite to Conv1x1 + DepthToSpace (GEMM-backed
///      path, also unlocks backends without a ConvTranspose kernel)
///    - Conv-BatchNormalization folding (absorb BN γ/β/μ/σ into Conv weights)
///    - Conv-Mul(const) scale absorption (scalar/per-channel Mul into weights)
///    - Conv-Add(const) bias absorption (per-channel Add into Conv bias)
///    - Constant folding (execute nodes with all-initializer inputs at load)
/// 3. [`ir_fusion_passes`] — Conv+Relu / BN+Relu annotation fusion, then a
///    final dead-code sweep and reorder.
///
/// Order matters within phase 2: `fold_conv_bn` runs before Conv-Mul/Conv-Add
/// because BN usually absorbs into Conv already, so only stray scale/bias fall
/// through. The phase as a whole precedes the activation fusions, since
/// `fold_conv_bn` matches a plain `Conv` and would miss one already retagged as
/// `Conv_Relu`.
///
/// The runtime index is rebuilt once here, after every pass has run. Individual
/// passes must not rebuild it themselves — doing so re-runs plan construction
/// and weight prepacking once per pass.
pub fn optimize_onnx_graph(model: &mut OnnxModel) -> Result<(), OnnxError> {
    run_ir_pipeline(model)?;
    model.rebuild_runtime_index();

    if std::env::var("YSCV_NCHWC").as_deref() == Ok("on") {
        let stats = analyze_nchwc(model);
        eprintln!(
            "[yscv-onnx] NCHWc stats: capable={}/{} chains={} max_chain={} mean_chain={:.2}",
            stats.capable_nodes,
            stats.total_nodes,
            stats.chain_count,
            stats.max_chain_len,
            stats.mean_chain_len,
        );
        if !stats.op_types_capable.is_empty() {
            let top: Vec<String> = stats
                .op_types_capable
                .iter()
                .take(5)
                .map(|(op, n)| format!("{op}:{n}"))
                .collect();
            eprintln!("[yscv-onnx] NCHWc top ops: {}", top.join(" "));
        }
    }

    Ok(())
}
