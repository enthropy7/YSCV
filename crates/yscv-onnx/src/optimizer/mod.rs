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
pub use fold_constants::fold_constants;
pub(crate) use fold_conv_bn::FoldConvBatchNorm;
pub(crate) use fold_conv_const_binary::FoldConvConstBinary;
pub(crate) use fuse_activation::FuseActivation;
pub use graph_cost::{
    GraphCost, GraphCostDiff, NodeCost, graph_cost, graph_cost_diff, graph_cost_report,
};
pub use graph_stats::{GraphStats, graph_stats};
pub use rewrite_convtranspose_dts::rewrite_convtranspose_dts;
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
fn run_ir_pipeline(model: &mut OnnxModel, passes: Vec<Box<dyn Pass>>) {
    let mut graph = model.to_ir();
    match PassManager::new(passes).run(&mut graph) {
        Ok(()) => model.apply_ir(&graph),
        Err(e) => eprintln!("[yscv-onnx] IR pass pipeline failed, graph left unoptimized: {e}"),
    }
}

/// Cleanup and ordering, run before the passes that are still positional.
///
/// `ReorderForFusion` has to come last within the phase — sorting after the
/// removals avoids ordering nodes that then vanish — but the phase as a whole
/// has to precede the string-based passes below, which match on `nodes[i + 1]`
/// and need producer/consumer adjacency restored first.
fn ir_cleanup_passes() -> Vec<Box<dyn Pass>> {
    vec![
        Box::new(RemoveDropout) as Box<dyn Pass>,
        Box::new(EliminateSqueezeUnsqueezePairs),
        // Ahead of the Conv-Mul / Conv-Add folds below, which only need to
        // handle the stray scale and bias that BatchNormalization did not
        // already absorb.
        Box::new(FoldConvBatchNorm),
        Box::new(FoldConvConstBinary::mul()),
        Box::new(FoldConvConstBinary::add()),
        Box::new(EliminateDeadCode),
        Box::new(ReorderForFusion),
    ]
}

/// Annotation fusion, run after the folding passes have had their turn —
/// `fold_conv_bn` matches a plain `Conv` and would miss one already retagged
/// as `Conv_Relu`.
fn ir_fusion_passes() -> Vec<Box<dyn Pass>> {
    vec![
        Box::new(FuseActivation::conv_relu()) as Box<dyn Pass>,
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
pub fn optimize_onnx_graph(model: &mut OnnxModel) {
    run_ir_pipeline(model, ir_cleanup_passes());

    rewrite_convtranspose_dts(model);
    fold_constants::run(model);

    run_ir_pipeline(model, ir_fusion_passes());
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
}
