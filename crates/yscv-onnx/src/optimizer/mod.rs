mod analyze_nchwc;
mod eliminate_dead_code;
mod eliminate_squeeze_unsqueeze_pairs;
mod fold_constants;
mod fold_conv_add_const;
mod fold_conv_bn;
mod fold_conv_mul;
mod fuse_bn_relu;
mod fuse_conv_relu;
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
        reorder_nodes_for_fusion::reorder_nodes_for_fusion,
    },
};

pub use analyze_nchwc::analyze_nchwc;
/// Re-exported so tests can drive this pass on its own, without the
/// string-based pipeline reordering the graph first.
pub(crate) use eliminate_squeeze_unsqueeze_pairs::EliminateSqueezeUnsqueezePairs;
pub use fold_constants::fold_constants;
pub use fold_conv_bn::fold_conv_bn;
pub use fuse_bn_relu::fuse_bn_relu;
pub use fuse_conv_relu::fuse_conv_relu;
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
fn run_ir_pipeline(model: &mut OnnxModel) {
    let mut graph = model.to_ir();
    let manager = PassManager::new(vec![
        Box::new(RemoveDropout) as Box<dyn Pass>,
        Box::new(EliminateSqueezeUnsqueezePairs),
        Box::new(EliminateDeadCode),
    ]);
    match manager.run(&mut graph) {
        Ok(()) => model.apply_ir(&graph),
        Err(e) => eprintln!("[yscv-onnx] IR pass pipeline failed, graph left unoptimized: {e}"),
    }
}

/// Optimizes an ONNX model graph in-place for inference.
///
/// Applies load-time passes modeled after ORT's Level-1 optimizer.
///
/// Still matching on positional adjacency, and so dependent on
/// `reorder_nodes_for_fusion` running first:
/// - ConvTranspose(k==s) rewrite to Conv1x1 + DepthToSpace (GEMM-backed path,
///   also unlocks backends without a ConvTranspose kernel)
/// - Conv-BatchNormalization folding (absorb BN γ/β/μ/σ into Conv weights)
/// - Conv-Mul(const) scale absorption (absorb scalar/per-channel Mul into weights)
/// - Conv-Add(const) bias absorption (absorb per-channel Add into Conv bias)
/// - Constant folding (execute nodes with all-initializer inputs at load)
/// - Conv+Relu / BN+Relu fusion (annotation-only; kernel dispatches on op_type)
///
/// Ported to the def-use IR, where matching is order-independent — see
/// [`run_ir_pipeline`]:
/// - Dropout removal (inference-only, rewire consumers to the Dropout input)
/// - Squeeze/Unsqueeze pair elimination (drop inverse pairs left by PyTorch export)
/// - Dead code elimination
///
/// Order matters among the first group: `fold_conv_bn` runs before
/// Conv-Mul/Conv-Add because BN usually absorbs into Conv already, so only
/// stray scale/bias fall through. Constant folding runs before the activation
/// fusions because a folded constant can turn a data-dependent activation into
/// a plain one.
///
/// The runtime index is rebuilt once here, after every pass has run. Individual
/// passes must not rebuild it themselves — doing so re-runs plan construction
/// and weight prepacking once per pass.
pub fn optimize_onnx_graph(model: &mut OnnxModel) {
    reorder_nodes_for_fusion(model);
    rewrite_convtranspose_dts(model);
    fold_conv_bn::run(model);
    fold_conv_mul::run(model);
    fold_conv_add_const::run(model);
    fold_constants::run(model);
    fuse_conv_relu::run(model);
    fuse_bn_relu::run(model);
    run_ir_pipeline(model);
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
