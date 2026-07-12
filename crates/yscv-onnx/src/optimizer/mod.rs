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
    loader::OnnxModel,
    optimizer::{
        eliminate_dead_code::eliminate_dead_code,
        eliminate_squeeze_unsqueeze_pairs::eliminate_squeeze_unsqueeze_pairs,
        remove_dropout_nodes::remove_dropout_nodes,
        reorder_nodes_for_fusion::reorder_nodes_for_fusion,
    },
};

pub use analyze_nchwc::analyze_nchwc;
pub use fold_constants::fold_constants;
pub use fold_conv_add_const::fold_conv_add_const;
pub use fold_conv_bn::fold_conv_bn;
pub use fold_conv_mul::fold_conv_mul;
pub use fuse_bn_relu::fuse_bn_relu;
pub use fuse_conv_relu::fuse_conv_relu;
pub use graph_cost::{
    GraphCost, GraphCostDiff, NodeCost, graph_cost, graph_cost_diff, graph_cost_report,
};
pub use graph_stats::{GraphStats, graph_stats};
pub use rewrite_convtranspose_dts::rewrite_convtranspose_dts;
pub use strip_qdq_within_fusion_chains::strip_qdq_within_fusion_chains;

/// Optimizes an ONNX model graph in-place for inference.
///
/// Applies load-time passes modeled after ORT's Level-1 optimizer:
/// - Dropout removal (inference-only, rewire consumers to Dropout input)
/// - ConvTranspose(k==s) rewrite to Conv1x1 + DepthToSpace (GEMM-backed path,
///   also unlocks backends without a ConvTranspose kernel)
/// - Conv-BatchNormalization folding (absorb BN γ/β/μ/σ into Conv weights)
/// - Conv-Mul(const) scale absorption (absorb scalar/per-channel Mul into weights)
/// - Conv-Add(const) bias absorption (absorb per-channel Add into Conv bias)
/// - Constant folding (execute nodes with all-initializer inputs at load)
/// - Squeeze/Unsqueeze pair elimination (drop inverse pairs left by PyTorch export)
/// - Conv+Clip(0,max) fusion (ReLU6-style clamped activation)
/// - Conv+Relu / BN+Relu fusion (annotation-only; kernel dispatches on op_type)
/// - Dead code elimination (iterate to fixpoint)
///
/// Order matters: `fold_conv_bn` runs before Conv-Mul/Conv-Add because BN
/// usually absorbs into Conv already; only stray scale/bias left over fall
/// through. Constant folding runs before Relu/Clip fusions because folded
/// Relus may turn Clip-style patterns into plain activations.
pub fn optimize_onnx_graph(model: &mut OnnxModel) {
    reorder_nodes_for_fusion(model);
    remove_dropout_nodes(model);
    rewrite_convtranspose_dts(model);
    fold_conv_bn(model);
    fold_conv_mul(model);
    fold_conv_add_const(model);
    fold_constants(model);
    eliminate_squeeze_unsqueeze_pairs(model);
    fuse_conv_relu(model);
    fuse_bn_relu(model);
    eliminate_dead_code(model);
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
