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
mod retarget_qconv_output_scale;
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

/// Lowers the model to the def-use IR, runs the pass pipeline to a fixpoint,
/// and writes the result back.
///
/// A pass failure propagates rather than being logged and swallowed, and leaves
/// the model exactly as it was: `apply_ir` only runs once the whole pipeline has
/// succeeded, so a half-transformed graph never reaches the caller. Constant
/// folding is what made this genuinely fallible — it executes nodes.
///
/// Note the distinction that draws: a node the evaluator declines is *not
/// foldable* and is skipped silently. Only a structurally broken graph is an
/// error.
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
/// `ReorderForFusion` is last because it is not for these passes at all — nor,
/// any longer, for the plan builder, which matches by dataflow too. The order
/// it leaves behind is what reaches `build_runtime_index`, and what that order
/// now decides is peak memory rather than which fusions fire.
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
        Box::new(FuseActivation::conv_hardswish()),
        Box::new(FuseActivation::bn_relu()),
        Box::new(EliminateDeadCode),
        Box::new(ReorderForFusion),
    ]
}

/// Optimizes an ONNX model graph in-place for inference.
///
/// Applies load-time passes modeled after ORT's Level-1 optimizer.
///
/// Every pass runs over the def-use IR, from a single pipeline driven to a
/// bounded fixpoint. See the private `pipeline` function for the order and why
/// it is what it is.
///
/// What the passes do, in the order they are listed there: Dropout removal and
/// Squeeze/Unsqueeze pair elimination; the ConvTranspose(k==s) rewrite to
/// Conv1x1 + DepthToSpace (GEMM-backed, and it unlocks backends with no
/// ConvTranspose kernel); Conv-BatchNormalization folding; Conv-Mul and
/// Conv-Add constant absorption into weights and bias; constant folding;
/// Conv+Relu and BN+Relu annotation fusion; dead code; and the topological
/// reorder.
///
/// Matching is by def-use, so a pass finds its pattern wherever the operands
/// sit rather than needing them adjacent — which is what lets the pipeline run
/// to a fixpoint instead of depending on a hand-tuned order.
///
/// The runtime index is rebuilt once here, after every pass has run. Individual
/// passes must not rebuild it themselves — doing so re-runs plan construction
/// and weight prepacking once per pass.
pub fn optimize_onnx_graph(model: &mut OnnxModel) -> Result<(), OnnxError> {
    run_ir_pipeline(model)?;
    retarget_qconv_output_scale::run(model);
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
