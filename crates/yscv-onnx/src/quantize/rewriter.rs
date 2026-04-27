//! Rewrite an fp32 ONNX model into a QDQ-format quantized model.
//!
//! Given a calibrated `OnnxModel` (fp32 weights, fp32 activations) plus a
//! [`HashMap<String, MinMax>`] of per-tensor activation statistics
//! collected via [`super::calibrate::CalibrationCollector`], inserts
//! `QuantizeLinear` / `DequantizeLinear` (QDQ) pairs around every
//! Conv / MatMul / Gemm node and quantizes their weight initializers
//! per-channel symmetric int8.
//!
//! Output is a standard QDQ-format ONNX graph that runs through the
//! existing `exec_quantize_linear` / `exec_dequantize_linear` /
//! `exec_qlinear_*` ops. Today those still dispatch to fp32 fallback
//! arithmetic, so the rewritten graph executes at the same wall-clock
//! cost as the fp32 original; the win shows up once integer GEMM
//! lands on the QLinear path.
//!
//! Scheme choice: **symmetric int8 [-128, 127] for both weights and
//! activations**. Asymmetric uint8 [0, 255] for activations is the more
//! common practice, but our existing `QuantizeLinear` op clamps to int8
//! storage; sticking to symmetric int8 lets the rewriter target real ops
//! today without expanding the runner. Asymmetric uint8 is a follow-up
//! once the runner gains uint8 support.
//!
//! Activations without calibration stats are left fp32 (no QDQ pair
//! inserted); weights are always quantized regardless of stats. This
//! handles the "weight-only" PTQ mode for users with no calibration
//! dataset.

use std::collections::HashMap;

use yscv_tensor::Tensor;

use super::calibrate::MinMax;
use super::derive::{QuantTarget, derive_per_channel_symmetric, derive_symmetric};
use crate::error::OnnxError;
use crate::loader::{OnnxAttribute, OnnxModel, OnnxNode};

/// Ops whose first weight input is a 2-D-or-higher learnable tensor we
/// want to quantize per-channel along axis 0.
const QUANTIZABLE_OP_TYPES: &[&str] = &["Conv", "MatMul", "Gemm"];

/// Insert QDQ pairs around the inputs of every Conv / MatMul / Gemm node
/// in `model`, and quantize their weight initializers per-channel int8
/// symmetric. `activation_stats` should be the map returned by
/// [`super::CalibrationCollector::snapshot`] after running the model on
/// a calibration dataset; tensors absent from the map are left fp32.
///
/// The rewrite is idempotent on a `model` whose nodes already have
/// QDQ inputs: such nodes simply pass through (their input names point
/// to existing DequantizeLinear outputs, not raw activations, and the
/// stats lookup misses).
pub fn rewrite_to_qdq(
    model: &mut OnnxModel,
    activation_stats: &HashMap<String, MinMax>,
) -> Result<(), OnnxError> {
    let mut new_initializers: HashMap<String, Tensor> = HashMap::new();
    let mut node_prefix: Vec<OnnxNode> = Vec::new();
    let mut input_renames: HashMap<(usize, usize), String> = HashMap::new();
    // map (node_idx, input_idx) -> new input name

    for (node_idx, node) in model.nodes.iter().enumerate() {
        if !QUANTIZABLE_OP_TYPES.contains(&node.op_type.as_str()) {
            continue;
        }
        if node.inputs.len() < 2 {
            continue; // malformed; skip
        }

        // ---- weight (input #1) ----
        let weight_name = &node.inputs[1];
        if let Some(stored) = model.initializers.get(weight_name).cloned()
            && stored.shape().len() >= 2
            && stored.len() > 16
            && !new_initializers.contains_key(&format!("{weight_name}_q"))
        {
            // Loader pre-permutes group=1 Conv weights OIHW → KHWC for the
            // hot path. PTQ derivation operates on OIHW (axis 0 = output
            // channel) and the rewritten dequant feeds raw `w_dq` to the
            // Conv (no KHWC flag attached), so we must produce OIHW
            // initializers. Un-permute when the weight is flagged.
            let weight = if model.khwc_weights.contains(weight_name) {
                stored
                    .permute(&[3, 2, 0, 1])
                    .map_err(|e| OnnxError::DecodeFailed {
                        message: format!("KHWC→OIHW permute failed for {weight_name}: {e}"),
                    })?
            } else {
                stored
            };
            quantize_weight_into(
                weight_name,
                &weight,
                &mut new_initializers,
                &mut node_prefix,
            )?;
            input_renames.insert((node_idx, 1), format!("{weight_name}_dq"));
        }

        // ---- activation (input #0) ----
        let act_name = &node.inputs[0];
        if let Some(stat) = activation_stats.get(act_name) {
            let qp = derive_symmetric(*stat, QuantTarget::Int8);
            // Skip if the derived params are degenerate (IDENTITY) — a
            // QDQ pair around a constant tensor is just noise.
            if qp.scale > 0.0 && qp.scale.is_finite() {
                let scale_init = format!("{act_name}__qact_scale");
                let zp_init = format!("{act_name}__qact_zp");
                let q_node_name = format!("{act_name}__qact_q");
                let dq_node_name = format!("{act_name}__qact_dq");
                let q_out = format!("{act_name}__qact_out");
                let dq_out = format!("{act_name}__qact_dqout");

                if !new_initializers.contains_key(&scale_init) {
                    new_initializers
                        .insert(scale_init.clone(), scalar_tensor(qp.scale, &scale_init)?);
                    new_initializers.insert(
                        zp_init.clone(),
                        scalar_tensor(qp.zero_point as f32, &zp_init)?,
                    );
                    node_prefix.push(OnnxNode {
                        op_type: "QuantizeLinear".to_string(),
                        name: q_node_name,
                        inputs: vec![act_name.clone(), scale_init.clone(), zp_init.clone()],
                        outputs: vec![q_out.clone()],
                        attributes: HashMap::new(),
                    });
                    node_prefix.push(OnnxNode {
                        op_type: "DequantizeLinear".to_string(),
                        name: dq_node_name,
                        inputs: vec![q_out, scale_init, zp_init],
                        outputs: vec![dq_out.clone()],
                        attributes: HashMap::new(),
                    });
                }
                input_renames.insert((node_idx, 0), dq_out);
            }
        }
    }

    if input_renames.is_empty() && node_prefix.is_empty() {
        return Ok(()); // nothing to rewrite
    }

    // Original weight initializers are kept rather than removed: the same
    // tensor may be referenced by a non-quantizable op elsewhere in the
    // graph, in which case dropping it would leave that op with a dangling
    // input. The runtime index simply ends up with one unused slot per
    // quantized weight.
    model.initializers.extend(new_initializers);

    // Rename node inputs.
    for ((node_idx, input_idx), new_name) in input_renames {
        if let Some(node) = model.nodes.get_mut(node_idx)
            && let Some(slot) = node.inputs.get_mut(input_idx)
        {
            *slot = new_name;
        }
    }

    // Prepend the inserted Q/DQ/dequantize-weight nodes. Order within
    // `node_prefix` was append-only and respects topological dependencies
    // (Q before DQ for activations; dequant nodes for weights are
    // standalone with no inter-prefix dependency).
    let mut combined = node_prefix;
    combined.append(&mut model.nodes);
    model.nodes = combined;
    model.rebuild_runtime_index();
    Ok(())
}

/// Quantize a single weight tensor per-channel int8 symmetric (axis 0)
/// and emit the matching DequantizeLinear node + scale/zp initializers
/// into the supplied collections.
fn quantize_weight_into(
    weight_name: &str,
    weight: &Tensor,
    new_initializers: &mut HashMap<String, Tensor>,
    node_prefix: &mut Vec<OnnxNode>,
) -> Result<(), OnnxError> {
    let shape = weight.shape();
    let data = weight.data();
    let channels = shape[0];
    let channel_size = data.len() / channels.max(1);

    // Per-channel MinMax.
    let mut per_channel = Vec::with_capacity(channels);
    for ch in 0..channels {
        let start = ch * channel_size;
        let end = start + channel_size;
        let mut mm = MinMax::default();
        mm.update(&data[start..end]);
        per_channel.push(mm);
    }
    let qparams = derive_per_channel_symmetric(&per_channel, QuantTarget::Int8);

    // Quantize values channel-by-channel.
    let mut quant_data = Vec::with_capacity(data.len());
    let mut scales = Vec::with_capacity(channels);
    let mut zps = Vec::with_capacity(channels);
    for (ch, qp) in qparams.iter().enumerate() {
        let inv_s = if qp.scale.abs() > f32::EPSILON {
            1.0 / qp.scale
        } else {
            0.0
        };
        let zp = qp.zero_point;
        for &v in &data[ch * channel_size..(ch + 1) * channel_size] {
            let q = ((v * inv_s).round() as i32 + zp).clamp(-128, 127);
            quant_data.push(q as f32);
        }
        scales.push(qp.scale);
        zps.push(qp.zero_point as f32);
    }

    let q_name = format!("{weight_name}_q");
    let scale_name = format!("{weight_name}_scale");
    let zp_name = format!("{weight_name}_zp");
    let dq_name = format!("{weight_name}_dq");
    let dq_node_name = format!("{weight_name}_dequant");

    new_initializers.insert(
        q_name.clone(),
        Tensor::from_vec(shape.to_vec(), quant_data).map_err(|e| OnnxError::DecodeFailed {
            message: format!("quantized tensor for {weight_name}: {e}"),
        })?,
    );
    new_initializers.insert(
        scale_name.clone(),
        Tensor::from_vec(vec![channels], scales).map_err(|e| OnnxError::DecodeFailed {
            message: format!("scale tensor for {weight_name}: {e}"),
        })?,
    );
    new_initializers.insert(
        zp_name.clone(),
        Tensor::from_vec(vec![channels], zps).map_err(|e| OnnxError::DecodeFailed {
            message: format!("zp tensor for {weight_name}: {e}"),
        })?,
    );

    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), OnnxAttribute::Int(0));
    node_prefix.push(OnnxNode {
        op_type: "DequantizeLinear".to_string(),
        name: dq_node_name,
        inputs: vec![q_name, scale_name, zp_name],
        outputs: vec![dq_name],
        attributes: attrs,
    });
    Ok(())
}

fn scalar_tensor(value: f32, ctx: &str) -> Result<Tensor, OnnxError> {
    Tensor::from_vec(vec![1], vec![value]).map_err(|e| OnnxError::DecodeFailed {
        message: format!("scalar tensor for {ctx}: {e}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_minimal_model() -> OnnxModel {
        // Single Conv with a 32-element weight and a 4-element bias.
        let weight: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let mut initializers = HashMap::new();
        initializers.insert(
            "w".to_string(),
            Tensor::from_vec(vec![2, 16], weight).unwrap(),
        );
        OnnxModel {
            ir_version: 7,
            opset_version: 13,
            producer_name: "test".to_string(),
            graph_name: "g".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            initializers,
            nodes: vec![OnnxNode {
                op_type: "Conv".to_string(),
                name: "conv0".to_string(),
                inputs: vec!["x".to_string(), "w".to_string()],
                outputs: vec!["y".to_string()],
                attributes: HashMap::new(),
            }],
            khwc_weights: Default::default(),
            dw_khwc_weights: Default::default(),
            group_khwc_weights: Default::default(),
            packed_int4_weights: Default::default(),
            runtime_index: Default::default(),
        }
    }

    #[test]
    fn weight_only_path_when_no_activation_stats() {
        let mut model = build_minimal_model();
        let stats = HashMap::new();
        rewrite_to_qdq(&mut model, &stats).unwrap();

        // Quantized weight + scale + zp present.
        assert!(model.initializers.contains_key("w_q"));
        assert!(model.initializers.contains_key("w_scale"));
        assert!(model.initializers.contains_key("w_zp"));

        // DequantizeLinear node for the weight inserted.
        let dq = model
            .nodes
            .iter()
            .find(|n| n.op_type == "DequantizeLinear" && n.outputs[0] == "w_dq")
            .expect("weight DequantizeLinear missing");
        assert_eq!(dq.inputs.len(), 3);
        // axis attribute = 0 (per-channel)
        match dq.attributes.get("axis") {
            Some(OnnxAttribute::Int(0)) => {}
            other => panic!("axis attr not Int(0): {other:?}"),
        }

        // Conv input #1 rewritten to point to the dequant output.
        let conv = model
            .nodes
            .iter()
            .find(|n| n.op_type == "Conv")
            .expect("Conv missing");
        assert_eq!(conv.inputs[1], "w_dq");
        // Activation input untouched (no stats).
        assert_eq!(conv.inputs[0], "x");
    }

    #[test]
    fn activation_qdq_inserted_when_stats_present() {
        let mut model = build_minimal_model();
        let mut stats = HashMap::new();
        stats.insert(
            "x".to_string(),
            MinMax {
                min: -2.0,
                max: 3.0,
                count: 100,
            },
        );
        rewrite_to_qdq(&mut model, &stats).unwrap();

        // QuantizeLinear + DequantizeLinear inserted on the activation.
        let q = model
            .nodes
            .iter()
            .find(|n| n.op_type == "QuantizeLinear")
            .expect("QuantizeLinear missing");
        assert_eq!(q.inputs[0], "x");
        let dq_for_act = model
            .nodes
            .iter()
            .find(|n| n.op_type == "DequantizeLinear" && n.inputs[0] == q.outputs[0])
            .expect("DequantizeLinear for activation missing");

        // Conv input #0 rewritten to point to the act-DQ output.
        let conv = model.nodes.iter().find(|n| n.op_type == "Conv").unwrap();
        assert_eq!(conv.inputs[0], dq_for_act.outputs[0]);
        assert_eq!(conv.inputs[1], "w_dq");
    }

    #[test]
    fn rewrite_skips_small_initializers() {
        let mut model = build_minimal_model();
        // shrink the weight below the 16-element threshold
        model.initializers.insert(
            "w".to_string(),
            Tensor::from_vec(vec![2, 4], vec![0.1; 8]).unwrap(),
        );
        rewrite_to_qdq(&mut model, &HashMap::new()).unwrap();
        assert!(!model.initializers.contains_key("w_q"));
        // Conv input #1 still points to the original weight name.
        assert_eq!(model.nodes[0].inputs[1], "w");
    }

    #[test]
    fn idempotent_on_empty_input_renames() {
        // No quantizable nodes -> no-op.
        let mut model = build_minimal_model();
        model.nodes[0].op_type = "Relu".to_string();
        rewrite_to_qdq(&mut model, &HashMap::new()).unwrap();
        // Weight still present, no extra inits.
        assert!(model.initializers.contains_key("w"));
        assert!(!model.initializers.contains_key("w_q"));
        assert_eq!(model.nodes.len(), 1);
    }
}
