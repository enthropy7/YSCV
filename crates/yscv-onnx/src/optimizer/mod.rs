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
    optimizer::{
        eliminate_dead_code::EliminateDeadCode, remove_dropout_nodes::RemoveDropout,
        reorder_nodes_for_fusion::ReorderForFusion,
    },
};

pub(crate) use analyze_nchwc::analyze_nchwc_nodes;
/// Re-exported so tests can drive these passes on their own, without the
/// string-based pipeline reordering the graph first.
pub(crate) use eliminate_squeeze_unsqueeze_pairs::EliminateSqueezeUnsqueezePairs;
pub(crate) use fold_constants::FoldConstants;
pub(crate) use fold_conv_bn::FoldConvBatchNorm;
pub(crate) use fold_conv_const_binary::FoldConvConstBinary;
pub(crate) use fuse_activation::FuseActivation;
#[cfg(test)]
pub(crate) use graph_cost::graph_cost_executable;
pub use graph_cost::{
    GraphCost, GraphCostDiff, NodeCost, graph_cost, graph_cost_diff, graph_cost_report,
};
pub use graph_stats::{GraphStats, graph_stats};
pub(crate) use rewrite_convtranspose_dts::RewriteConvTransposeToDepthToSpace;
pub use strip_qdq_within_fusion_chains::strip_qdq_within_fusion_chains;

/// Runs the pass pipeline to a fixpoint during executable construction.
///
/// A pass failure propagates rather than being logged and swallowed. Constant
/// folding is what made this genuinely fallible — it executes nodes. A node the
/// evaluator declines is *not foldable* and is skipped silently; only a
/// structurally broken graph is an error.
pub(crate) fn optimize_ir(graph: &mut crate::ir::Graph) -> Result<(), OnnxError> {
    PassManager::new(pipeline()).run(graph)?;
    retarget_qconv_output_scale::run(graph);
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
    let mut passes: Vec<Box<dyn Pass>> = vec![
        Box::new(RemoveDropout) as Box<dyn Pass>,
        Box::new(EliminateSqueezeUnsqueezePairs),
        Box::new(RewriteConvTransposeToDepthToSpace),
        Box::new(FoldConvBatchNorm),
        Box::new(FoldConvConstBinary::mul()),
        Box::new(FoldConvConstBinary::add()),
        Box::new(FoldConstants),
    ];
    // Conv+Relu fusion is a win where a fused Conv_Relu kernel exists (aarch64,
    // x86) but a measured regression on 32-bit ARM, which has none: the fused
    // node runs measurably slower than a plain Conv plus a separate vectorized
    // Relu. Skip it there; the HardSwish/BN fusions and the topological reorder
    // still help on armv7.
    #[cfg(not(target_arch = "arm"))]
    passes.push(Box::new(FuseActivation::conv_relu()));
    passes.push(Box::new(FuseActivation::conv_hardswish()));
    passes.push(Box::new(FuseActivation::bn_relu()));
    passes.push(Box::new(EliminateDeadCode));
    passes.push(Box::new(ReorderForFusion));
    passes
}
