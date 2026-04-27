//! Packed-INT4 weight quantization for the LLM decode hot path.
//!
//! Picks every MatMul / Gemm node whose second input is a 2-D
//! initializer, packs it as symmetric INT4 with per-group fp32 scales
//! (`pack_int4_symmetric_per_group`), and stores the result in
//! `OnnxModel::packed_int4_weights`. The original f32 initializer is
//! removed so the runtime MatMul / Gemm executors can detect the
//! packed weight via the side-table and route to
//! `packed_int4_gemv_dispatch` instead of f32 GEMM.
//!
//! Weight layout convention: most LLM linears store weights as
//! `[output_dim, input_dim]` (PyTorch `nn.Linear` style; transposed at
//! load time relative to MatMul's `K × N`). MatMul nodes consuming
//! that initializer therefore see `[input_dim, output_dim]` after
//! ONNX export, i.e. shape `[K, N]`. We pack along K (axis 1) — each
//! row of the packed weight corresponds to one output channel and
//! contains K nibbles, matching the format expected by the GEMV
//! kernel (`m_w = N`, `k = K`).

use yscv_kernels::pack_int4_symmetric_per_group;

use crate::error::OnnxError;
use crate::loader::{OnnxModel, OnnxNode};

/// Side-table entry: packed weight bytes + per-group fp32 scales
/// + the dimensions the GEMV kernel needs at runtime.
#[derive(Clone, Debug)]
pub struct PackedInt4Weight {
    /// `m_w * k / 2` bytes; two i4 nibbles per byte (low first).
    pub packed: Vec<u8>,
    /// `m_w * (k / group_size)` fp32 scales.
    pub scales: Vec<f32>,
    pub group_size: usize,
    /// Output channels (rows of the packed view).
    pub m_w: usize,
    /// Input channels.
    pub k: usize,
}

/// Pack symmetric INT4 weights for every eligible MatMul / Gemm node.
/// Returns the count of weights packed. `group_size` must be even and
/// divide each weight's K dimension; weights with mismatched shapes
/// are silently skipped.
///
/// MatMul weight is treated as `[K, N]` (post-transpose ONNX layout).
/// Gemm weight is `[N, K]` when `transB=1` (the common case for LLM
/// linears) — packed natively. Gemm with `transB=0` is `[K, N]` and
/// gets transposed to `[N, K]` before packing.
pub fn quantize_matmul_weights_int4_packed(
    model: &mut OnnxModel,
    group_size: usize,
) -> Result<usize, OnnxError> {
    if group_size == 0 || group_size & 1 != 0 {
        return Err(OnnxError::DecodeFailed {
            message: format!("group_size must be even and > 0, got {group_size}"),
        });
    }

    let candidates = collect_packing_candidates(&model.nodes);
    let mut packed_count = 0;

    for cand in candidates {
        let Some(weight) = model.initializers.get(&cand.weight_name).cloned() else {
            continue;
        };
        if weight.shape().len() != 2 {
            continue;
        }
        let s = weight.shape();
        // Determine (m_w = output dim, k = input dim) from the op-specific
        // layout. We always pack as M_w rows × K cols.
        let (m_w, k, source_data) = match cand.layout {
            WeightLayout::MatMulKN => {
                // shape = [K, N]; transpose to [N, K] before packing.
                let (k_dim, n_dim) = (s[0], s[1]);
                let mut transposed = vec![0.0_f32; n_dim * k_dim];
                let src = weight.data();
                for kk in 0..k_dim {
                    for nn in 0..n_dim {
                        transposed[nn * k_dim + kk] = src[kk * n_dim + nn];
                    }
                }
                (n_dim, k_dim, transposed)
            }
            WeightLayout::GemmNK => {
                // shape = [N, K]; pack as-is.
                (s[0], s[1], weight.data().to_vec())
            }
            WeightLayout::GemmKN => {
                let (k_dim, n_dim) = (s[0], s[1]);
                let mut transposed = vec![0.0_f32; n_dim * k_dim];
                let src = weight.data();
                for kk in 0..k_dim {
                    for nn in 0..n_dim {
                        transposed[nn * k_dim + kk] = src[kk * n_dim + nn];
                    }
                }
                (n_dim, k_dim, transposed)
            }
        };
        if k % group_size != 0 || k * m_w < 32 {
            continue;
        }

        let (packed, scales) = pack_int4_symmetric_per_group(&source_data, m_w, k, group_size);

        model.packed_int4_weights.insert(
            cand.weight_name.clone(),
            PackedInt4Weight {
                packed,
                scales,
                group_size,
                m_w,
                k,
            },
        );
        // Remove the original initializer so dispatch sees the side-
        // table miss-on-fp32 and uses the packed path.
        model.initializers.remove(&cand.weight_name);
        packed_count += 1;
    }

    if packed_count > 0 {
        model.rebuild_runtime_index();
    }
    Ok(packed_count)
}

#[derive(Clone, Copy)]
enum WeightLayout {
    /// MatMul weight: shape `[K, N]`.
    MatMulKN,
    /// Gemm with `transB=1`: shape `[N, K]`.
    GemmNK,
    /// Gemm with `transB=0`: shape `[K, N]`.
    GemmKN,
}

struct Candidate {
    weight_name: String,
    layout: WeightLayout,
}

fn collect_packing_candidates(nodes: &[OnnxNode]) -> Vec<Candidate> {
    let mut out = Vec::new();
    for node in nodes {
        match node.op_type.as_str() {
            "MatMul" if node.inputs.len() >= 2 => {
                out.push(Candidate {
                    weight_name: node.inputs[1].clone(),
                    layout: WeightLayout::MatMulKN,
                });
            }
            "Gemm" if node.inputs.len() >= 2 => {
                let trans_b = match node.attributes.get("transB") {
                    Some(crate::loader::OnnxAttribute::Int(v)) => *v,
                    _ => 0,
                };
                let layout = if trans_b == 1 {
                    WeightLayout::GemmNK
                } else {
                    WeightLayout::GemmKN
                };
                out.push(Candidate {
                    weight_name: node.inputs[1].clone(),
                    layout,
                });
            }
            _ => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::{OnnxAttribute, OnnxNode};
    use std::collections::HashMap;
    use yscv_tensor::Tensor;

    fn synth_model(weight_shape: Vec<usize>, weight_data: Vec<f32>, op: &str) -> OnnxModel {
        let mut initializers = HashMap::new();
        initializers.insert(
            "w".to_string(),
            Tensor::from_vec(weight_shape, weight_data).unwrap(),
        );
        let mut attrs = HashMap::new();
        if op == "Gemm" {
            attrs.insert("transB".to_string(), OnnxAttribute::Int(1));
        }
        OnnxModel {
            ir_version: 7,
            opset_version: 13,
            producer_name: "test".to_string(),
            graph_name: "g".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            initializers,
            nodes: vec![OnnxNode {
                op_type: op.to_string(),
                name: "n0".to_string(),
                inputs: vec!["x".to_string(), "w".to_string()],
                outputs: vec!["y".to_string()],
                attributes: attrs,
            }],
            khwc_weights: Default::default(),
            dw_khwc_weights: Default::default(),
            group_khwc_weights: Default::default(),
            packed_int4_weights: Default::default(),
            runtime_index: Default::default(),
        }
    }

    #[test]
    fn rejects_odd_group_size() {
        let mut model = synth_model(vec![32, 4], vec![0.1; 128], "MatMul");
        assert!(quantize_matmul_weights_int4_packed(&mut model, 7).is_err());
    }

    #[test]
    fn matmul_kn_weight_gets_packed_with_n_rows() {
        // shape [K=32, N=4] in MatMul → packed view [N=4, K=32].
        let weights: Vec<f32> = (0..128).map(|v| (v as f32 - 64.0) * 0.05).collect();
        let mut model = synth_model(vec![32, 4], weights, "MatMul");
        let n = quantize_matmul_weights_int4_packed(&mut model, 16).unwrap();
        assert_eq!(n, 1);
        assert!(!model.initializers.contains_key("w"));
        let pw = &model.packed_int4_weights["w"];
        assert_eq!(pw.m_w, 4);
        assert_eq!(pw.k, 32);
        assert_eq!(pw.group_size, 16);
        assert_eq!(pw.packed.len(), 4 * 32 / 2);
        assert_eq!(pw.scales.len(), 4 * (32 / 16));
    }

    #[test]
    fn gemm_transb_one_packs_native_layout() {
        // [N=4, K=32] (transB=1) packs without transpose.
        let weights: Vec<f32> = (0..128).map(|v| (v as f32 - 64.0) * 0.05).collect();
        let mut model = synth_model(vec![4, 32], weights, "Gemm");
        let n = quantize_matmul_weights_int4_packed(&mut model, 16).unwrap();
        assert_eq!(n, 1);
        let pw = &model.packed_int4_weights["w"];
        assert_eq!((pw.m_w, pw.k), (4, 32));
    }

    #[test]
    fn small_weights_are_skipped() {
        // 16 elements is below the threshold (k * m_w < 32).
        let weights: Vec<f32> = vec![0.1; 16];
        let mut model = synth_model(vec![16, 1], weights, "MatMul");
        let n = quantize_matmul_weights_int4_packed(&mut model, 16).unwrap();
        assert_eq!(n, 0);
        assert!(model.initializers.contains_key("w"));
    }

    #[test]
    fn k_not_multiple_of_group_size_skips() {
        // k=18, group_size=16: 18 % 16 != 0 — skipped silently.
        let weights: Vec<f32> = vec![0.1; 18 * 4];
        let mut model = synth_model(vec![18, 4], weights, "MatMul");
        let n = quantize_matmul_weights_int4_packed(&mut model, 16).unwrap();
        assert_eq!(n, 0);
        assert!(model.initializers.contains_key("w"));
    }
}
