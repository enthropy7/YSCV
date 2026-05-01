//! Rewrite an fp32 ONNX model into QDQ- or QLinear-format quantized models.
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

use std::collections::{HashMap, HashSet};

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
    // Weight-dequant nodes consume only constant initialisers and are
    // safe to prepend (their inputs are always available). Activation
    // Q/DQ nodes must run AFTER their producer node, so we collect
    // them per-consumer-index and splice them into the node list
    // immediately before each consumer.
    let mut weight_dequant_prefix: Vec<OnnxNode> = Vec::new();
    let mut act_qdq_inserts: HashMap<usize, Vec<OnnxNode>> = HashMap::new();
    let mut input_renames: HashMap<(usize, usize), String> = HashMap::new();
    let mut emitted_act_qdq: HashMap<String, String> = HashMap::new();
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
            let weight = if node.op_type == "Conv" {
                conv_weight_as_oihw(model, weight_name, &stored)?
            } else {
                stored
            };
            quantize_weight_into(
                weight_name,
                &weight,
                &mut new_initializers,
                &mut weight_dequant_prefix,
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
                // Reuse a prior Q/DQ pair for the same source activation
                // if we already emitted one — multiple consumers of the
                // same tensor share the rewrite.
                let dq_out = if let Some(existing) = emitted_act_qdq.get(act_name) {
                    existing.clone()
                } else {
                    let scale_init = format!("{act_name}__qact_scale");
                    let zp_init = format!("{act_name}__qact_zp");
                    let q_node_name = format!("{act_name}__qact_q");
                    let dq_node_name = format!("{act_name}__qact_dq");
                    let q_out = format!("{act_name}__qact_out");
                    let dq_out = format!("{act_name}__qact_dqout");

                    new_initializers
                        .insert(scale_init.clone(), scalar_tensor(qp.scale, &scale_init)?);
                    new_initializers.insert(
                        zp_init.clone(),
                        scalar_tensor(qp.zero_point as f32, &zp_init)?,
                    );
                    let nodes_for_consumer = vec![
                        OnnxNode {
                            op_type: "QuantizeLinear".to_string(),
                            name: q_node_name,
                            inputs: vec![act_name.clone(), scale_init.clone(), zp_init.clone()],
                            outputs: vec![q_out.clone()],
                            attributes: HashMap::new(),
                        },
                        OnnxNode {
                            op_type: "DequantizeLinear".to_string(),
                            name: dq_node_name,
                            inputs: vec![q_out, scale_init, zp_init],
                            outputs: vec![dq_out.clone()],
                            attributes: HashMap::new(),
                        },
                    ];
                    act_qdq_inserts
                        .entry(node_idx)
                        .or_default()
                        .extend(nodes_for_consumer);
                    emitted_act_qdq.insert(act_name.clone(), dq_out.clone());
                    dq_out
                };
                input_renames.insert((node_idx, 0), dq_out);
            }
        }
    }

    if input_renames.is_empty() && weight_dequant_prefix.is_empty() && act_qdq_inserts.is_empty() {
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

    // Build the new node list:
    //   1. Weight-dequant prefix (consumes only initializers; safe at start).
    //   2. Walk original nodes in order, splicing each consumer's
    //      activation Q/DQ pair in immediately before it. This keeps Q/DQ
    //      *after* the producer node that emits act_name.
    let mut combined: Vec<OnnxNode> =
        Vec::with_capacity(weight_dequant_prefix.len() + model.nodes.len() + act_qdq_inserts.len());
    combined.append(&mut weight_dequant_prefix);
    let original_nodes = std::mem::take(&mut model.nodes);
    for (idx, node) in original_nodes.into_iter().enumerate() {
        if let Some(qdq) = act_qdq_inserts.remove(&idx) {
            combined.extend(qdq);
        }
        combined.push(node);
    }
    model.nodes = combined;
    model.rebuild_runtime_index();
    Ok(())
}

/// Rewrite Conv / Conv_Relu / MatMul nodes to standard ONNX QLinearConv /
/// QLinearMatMul nodes, with QuantizeLinear on dynamic activations and
/// DequantizeLinear on the quantized result so the rest of the graph keeps
/// its original fp32 tensor names.
///
/// This is intentionally an export/interoperability format. The yscv runtime
/// fast path can fuse the original QDQ graph without giving up NHWC Conv
/// layout, while this QLinear form gives ORT and `onnx.checker` a conventional
/// integer graph to validate against. We use symmetric int8 zero-points so the
/// existing yscv QLinear kernels can run the subset they support.
pub fn rewrite_to_qlinear(
    model: &mut OnnxModel,
    activation_stats: &HashMap<String, MinMax>,
) -> Result<(), OnnxError> {
    let mut new_initializers: HashMap<String, Tensor> = HashMap::new();
    let mut new_nodes = Vec::with_capacity(model.nodes.len() * 3);
    let original_nodes = std::mem::take(&mut model.nodes);
    let mut rewritten = 0usize;

    for node in original_nodes {
        match node.op_type.as_str() {
            "Conv" | "Conv_Relu" => {
                if let Some(nodes) = rewrite_conv_node_to_qlinear(
                    model,
                    &node,
                    activation_stats,
                    &mut new_initializers,
                )? {
                    new_nodes.extend(nodes);
                    rewritten += 1;
                } else {
                    new_nodes.push(node);
                }
            }
            "MatMul" => {
                if let Some(nodes) = rewrite_matmul_node_to_qlinear(
                    model,
                    &node,
                    activation_stats,
                    &mut new_initializers,
                )? {
                    new_nodes.extend(nodes);
                    rewritten += 1;
                } else {
                    new_nodes.push(node);
                }
            }
            _ => new_nodes.push(node),
        }
    }

    if rewritten == 0 {
        model.nodes = new_nodes;
        return Ok(());
    }

    model.initializers.extend(new_initializers);
    model.nodes = new_nodes;
    model.rebuild_runtime_index();
    Ok(())
}

/// Fold constant weight DequantizeLinear nodes back into fp32 initializers.
///
/// This is the yscv-fast companion to QDQ export: standard QDQ keeps quantized
/// weights as `initializer_q -> DequantizeLinear -> Conv`, but that hides the
/// weight from the loader's Conv layout normalisation and prepack passes. For
/// yscv execution we can pre-evaluate those constant DQ nodes once and make the
/// DQ output itself an initializer, restoring the regular Conv hot path while
/// keeping activation QDQ nodes in the graph.
pub fn fold_constant_qdq_weights_for_yscv_fast(model: &mut OnnxModel) -> Result<usize, OnnxError> {
    let mut folded = 0usize;
    let mut remove = vec![false; model.nodes.len()];
    let mut new_initializers = HashMap::new();
    for (idx, node) in model.nodes.iter().enumerate() {
        if node.op_type != "DequantizeLinear" || node.inputs.len() < 2 || node.outputs.len() != 1 {
            continue;
        }
        let q_name = &node.inputs[0];
        let Some(q) = model.initializers.get(q_name) else {
            continue;
        };
        let scale_name = &node.inputs[1];
        let Some(scale) = model.initializers.get(scale_name) else {
            continue;
        };
        let zp = node
            .inputs
            .get(2)
            .and_then(|name| model.initializers.get(name))
            .map(|t| t.data().to_vec())
            .unwrap_or_else(|| vec![0.0; scale.len().max(1)]);
        let folded_tensor = dequantize_initializer(q, scale, &zp, node)?;
        new_initializers.insert(node.outputs[0].clone(), folded_tensor);
        remove[idx] = true;
        folded += 1;
    }

    if folded == 0 {
        return Ok(0);
    }
    model.initializers.extend(new_initializers);
    let old_nodes = std::mem::take(&mut model.nodes);
    model.nodes = old_nodes
        .into_iter()
        .enumerate()
        .filter_map(|(idx, node)| (!remove[idx]).then_some(node))
        .collect();
    model.rebuild_runtime_index();
    Ok(folded)
}

/// Remove initializers that are no longer referenced after QDQ cleanup.
///
/// yscv-fast QDQ folds constant weight dequantizers and may strip activation
/// Q/DQ pairs inside Conv chains. The original quantized tensors/scales then
/// become dead graph baggage; pruning them keeps exported models quiet in ORT
/// and avoids reload-time initializer indexing work.
pub fn prune_unused_initializers(model: &mut OnnxModel) -> usize {
    let mut used: HashSet<&str> = HashSet::new();
    for node in &model.nodes {
        for input in &node.inputs {
            used.insert(input.as_str());
        }
    }
    for input in &model.inputs {
        used.insert(input.as_str());
    }
    for output in &model.outputs {
        used.insert(output.as_str());
    }

    let before = model.initializers.len();
    model
        .initializers
        .retain(|name, _| used.contains(name.as_str()));
    let removed = before - model.initializers.len();
    if removed != 0 {
        model.rebuild_runtime_index();
    }
    removed
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

fn rewrite_conv_node_to_qlinear(
    model: &OnnxModel,
    node: &OnnxNode,
    activation_stats: &HashMap<String, MinMax>,
    new_initializers: &mut HashMap<String, Tensor>,
) -> Result<Option<Vec<OnnxNode>>, OnnxError> {
    if node.inputs.len() < 2 || node.outputs.len() != 1 {
        return Ok(None);
    }
    let x_name = &node.inputs[0];
    let w_name = &node.inputs[1];
    let y_name = &node.outputs[0];
    let Some(x_stat) = activation_stats.get(x_name).copied() else {
        return Ok(None);
    };
    let Some(y_stat) = activation_stats.get(y_name).copied() else {
        return Ok(None);
    };
    let Some(stored_weight) = model.initializers.get(w_name) else {
        return Ok(None);
    };
    let weight = conv_weight_as_oihw(model, w_name, stored_weight)?;
    if weight.rank() != 4 || weight.len() <= 16 {
        return Ok(None);
    }

    let x_qp = derive_symmetric(x_stat, QuantTarget::Int8);
    let y_qp = derive_symmetric(y_stat, QuantTarget::Int8);
    let w_qp = derive_tensor_symmetric(&weight);
    if !qparams_usable(x_qp.scale) || !qparams_usable(y_qp.scale) || !qparams_usable(w_qp.scale) {
        return Ok(None);
    }

    let x_scale = format!("{x_name}__qlinear_x_scale");
    let x_zp = format!("{x_name}__qlinear_x_zp");
    // QLinear export targets standard ONNX/ORT semantics. The loader may keep
    // Conv weights in an internal KHWC/DW-KHWC layout for yscv execution, but
    // `conv_weight_as_oihw` above converts the quantized initializer back to
    // OIHW. Graph activations therefore stay in the public ONNX NCHW layout;
    // inserting NHWC<->NCHW transposes here corrupts residual Add shapes.
    let needs_io_transpose = false;
    let node_key = if node.name.is_empty() {
        y_name
    } else {
        &node.name
    };
    let x_q_nhwc = format!("{node_key}__qlinear_x_q");
    let x_q = if needs_io_transpose {
        format!("{node_key}__qlinear_x_q_nchw")
    } else {
        x_q_nhwc.clone()
    };
    let w_q = format!("{w_name}__qlinear_w_q");
    let w_scale = format!("{w_name}__qlinear_w_scale");
    let w_zp = format!("{w_name}__qlinear_w_zp");
    let y_scale = format!("{y_name}__qlinear_y_scale");
    let y_zp = format!("{y_name}__qlinear_y_zp");
    let qconv_out = format!("{y_name}__qlinear_y_q");
    let dq_out_nchw = format!("{y_name}__qlinear_y_nchw");
    let fp32_conv_out = if node.op_type == "Conv_Relu" {
        format!("{y_name}__qlinear_pre_relu")
    } else {
        y_name.clone()
    };

    new_initializers.insert(x_scale.clone(), scalar_tensor(x_qp.scale, &x_scale)?);
    new_initializers.insert(x_zp.clone(), scalar_tensor(0.0, &x_zp)?);
    new_initializers.insert(w_scale.clone(), scalar_tensor(w_qp.scale, &w_scale)?);
    new_initializers.insert(w_zp.clone(), scalar_tensor(0.0, &w_zp)?);
    new_initializers.insert(y_scale.clone(), scalar_tensor(y_qp.scale, &y_scale)?);
    new_initializers.insert(y_zp.clone(), scalar_tensor(0.0, &y_zp)?);
    new_initializers.insert(
        w_q.clone(),
        quantize_tensor_symmetric(&weight, w_qp.scale, &w_q)?,
    );

    let mut inputs = vec![
        x_q.clone(),
        x_scale.clone(),
        x_zp.clone(),
        w_q,
        w_scale.clone(),
        w_zp.clone(),
        y_scale.clone(),
        y_zp.clone(),
    ];
    if let Some(bias_name) = node.inputs.get(2)
        && !bias_name.is_empty()
        && let Some(bias) = model.initializers.get(bias_name)
    {
        let b_name = format!("{bias_name}__qlinear_bias_i32");
        new_initializers.insert(
            b_name.clone(),
            quantize_bias_to_i32_storage(bias, x_qp.scale * w_qp.scale, &b_name)?,
        );
        inputs.push(b_name);
    }

    let mut out = Vec::new();
    out.push(OnnxNode {
        op_type: "QuantizeLinear".to_string(),
        name: format!("{}__qlinear_x_quant", node.name),
        inputs: vec![x_name.clone(), x_scale, x_zp],
        outputs: vec![x_q_nhwc.clone()],
        attributes: HashMap::new(),
    });
    if needs_io_transpose {
        out.push(OnnxNode {
            op_type: "Transpose".to_string(),
            name: format!("{}__qlinear_x_nhwc_to_nchw", node.name),
            inputs: vec![x_q_nhwc],
            outputs: vec![x_q.clone()],
            attributes: HashMap::from([(
                "perm".to_string(),
                OnnxAttribute::Ints(vec![0, 3, 1, 2]),
            )]),
        });
    }
    out.extend([
        OnnxNode {
            op_type: "QLinearConv".to_string(),
            name: format!("{}__qlinear", node.name),
            inputs,
            outputs: vec![qconv_out.clone()],
            attributes: node.attributes.clone(),
        },
        OnnxNode {
            op_type: "DequantizeLinear".to_string(),
            name: format!("{}__qlinear_y_dequant", node.name),
            inputs: vec![qconv_out, y_scale, y_zp],
            outputs: vec![if needs_io_transpose {
                dq_out_nchw.clone()
            } else {
                fp32_conv_out.clone()
            }],
            attributes: HashMap::new(),
        },
    ]);
    if needs_io_transpose {
        out.push(OnnxNode {
            op_type: "Transpose".to_string(),
            name: format!("{}__qlinear_y_nchw_to_nhwc", node.name),
            inputs: vec![dq_out_nchw],
            outputs: vec![fp32_conv_out.clone()],
            attributes: HashMap::from([(
                "perm".to_string(),
                OnnxAttribute::Ints(vec![0, 2, 3, 1]),
            )]),
        });
    }
    if node.op_type == "Conv_Relu" {
        out.push(OnnxNode {
            op_type: "Relu".to_string(),
            name: format!("{}__qlinear_relu", node.name),
            inputs: vec![fp32_conv_out],
            outputs: vec![y_name.clone()],
            attributes: HashMap::new(),
        });
    }
    Ok(Some(out))
}

fn rewrite_matmul_node_to_qlinear(
    model: &OnnxModel,
    node: &OnnxNode,
    activation_stats: &HashMap<String, MinMax>,
    new_initializers: &mut HashMap<String, Tensor>,
) -> Result<Option<Vec<OnnxNode>>, OnnxError> {
    if node.inputs.len() < 2 || node.outputs.len() != 1 {
        return Ok(None);
    }
    let a_name = &node.inputs[0];
    let b_name = &node.inputs[1];
    let y_name = &node.outputs[0];
    let Some(a_stat) = activation_stats.get(a_name).copied() else {
        return Ok(None);
    };
    let Some(y_stat) = activation_stats.get(y_name).copied() else {
        return Ok(None);
    };
    let Some(b) = model.initializers.get(b_name) else {
        return Ok(None);
    };
    if b.rank() != 2 || b.len() <= 16 {
        return Ok(None);
    }

    let a_qp = derive_symmetric(a_stat, QuantTarget::Int8);
    let y_qp = derive_symmetric(y_stat, QuantTarget::Int8);
    let b_qp = derive_tensor_symmetric(b);
    if !qparams_usable(a_qp.scale) || !qparams_usable(y_qp.scale) || !qparams_usable(b_qp.scale) {
        return Ok(None);
    }

    let a_scale = format!("{a_name}__qlinear_a_scale");
    let a_zp = format!("{a_name}__qlinear_a_zp");
    let node_key = if node.name.is_empty() {
        y_name
    } else {
        &node.name
    };
    let a_q = format!("{node_key}__qlinear_a_q");
    let b_q = format!("{b_name}__qlinear_b_q");
    let b_scale = format!("{b_name}__qlinear_b_scale");
    let b_zp = format!("{b_name}__qlinear_b_zp");
    let y_scale = format!("{y_name}__qlinear_y_scale");
    let y_zp = format!("{y_name}__qlinear_y_zp");
    let qmm_out = format!("{y_name}__qlinear_y_q");

    new_initializers.insert(a_scale.clone(), scalar_tensor(a_qp.scale, &a_scale)?);
    new_initializers.insert(a_zp.clone(), scalar_tensor(0.0, &a_zp)?);
    new_initializers.insert(b_scale.clone(), scalar_tensor(b_qp.scale, &b_scale)?);
    new_initializers.insert(b_zp.clone(), scalar_tensor(0.0, &b_zp)?);
    new_initializers.insert(y_scale.clone(), scalar_tensor(y_qp.scale, &y_scale)?);
    new_initializers.insert(y_zp.clone(), scalar_tensor(0.0, &y_zp)?);
    new_initializers.insert(b_q.clone(), quantize_tensor_symmetric(b, b_qp.scale, &b_q)?);

    Ok(Some(vec![
        OnnxNode {
            op_type: "QuantizeLinear".to_string(),
            name: format!("{}__qlinear_a_quant", node.name),
            inputs: vec![a_name.clone(), a_scale.clone(), a_zp.clone()],
            outputs: vec![a_q.clone()],
            attributes: HashMap::new(),
        },
        OnnxNode {
            op_type: "QLinearMatMul".to_string(),
            name: format!("{}__qlinear", node.name),
            inputs: vec![
                a_q,
                a_scale,
                a_zp,
                b_q,
                b_scale,
                b_zp,
                y_scale.clone(),
                y_zp.clone(),
            ],
            outputs: vec![qmm_out.clone()],
            attributes: HashMap::new(),
        },
        OnnxNode {
            op_type: "DequantizeLinear".to_string(),
            name: format!("{}__qlinear_y_dequant", node.name),
            inputs: vec![qmm_out, y_scale, y_zp],
            outputs: vec![y_name.clone()],
            attributes: HashMap::new(),
        },
    ]))
}

#[derive(Clone, Copy)]
struct ScalarQParams {
    scale: f32,
}

fn derive_tensor_symmetric(tensor: &Tensor) -> ScalarQParams {
    let mut mm = MinMax::default();
    mm.update(tensor.data());
    let qp = derive_symmetric(mm, QuantTarget::Int8);
    ScalarQParams { scale: qp.scale }
}

fn qparams_usable(scale: f32) -> bool {
    scale.is_finite() && scale > 0.0
}

fn quantize_tensor_symmetric(tensor: &Tensor, scale: f32, ctx: &str) -> Result<Tensor, OnnxError> {
    let inv_s = if scale.abs() > f32::EPSILON {
        1.0 / scale
    } else {
        0.0
    };
    let data: Vec<f32> = tensor
        .data()
        .iter()
        .map(|&v| (v * inv_s).round().clamp(-128.0, 127.0))
        .collect();
    Tensor::from_vec(tensor.shape().to_vec(), data).map_err(|e| OnnxError::DecodeFailed {
        message: format!("quantized tensor for {ctx}: {e}"),
    })
}

fn quantize_bias_to_i32_storage(bias: &Tensor, scale: f32, ctx: &str) -> Result<Tensor, OnnxError> {
    if !qparams_usable(scale) {
        return Err(OnnxError::DecodeFailed {
            message: format!("bias scale for {ctx} is not usable: {scale}"),
        });
    }
    let inv_s = 1.0 / scale;
    let data: Vec<f32> = bias
        .data()
        .iter()
        .map(|&v| (v * inv_s).round().clamp(i32::MIN as f32, i32::MAX as f32))
        .collect();
    Tensor::from_vec(bias.shape().to_vec(), data).map_err(|e| OnnxError::DecodeFailed {
        message: format!("quantized bias for {ctx}: {e}"),
    })
}

fn dequantize_initializer(
    q: &Tensor,
    scale: &Tensor,
    zp: &[f32],
    node: &OnnxNode,
) -> Result<Tensor, OnnxError> {
    let q_shape = q.shape();
    let q_data = q.data();
    let scale_data = scale.data();
    let out = if scale_data.len() == 1 {
        let s = scale_data[0];
        let z = zp.first().copied().unwrap_or(0.0);
        q_data.iter().map(|&v| (v - z) * s).collect()
    } else {
        let axis = match node.attributes.get("axis") {
            Some(OnnxAttribute::Int(axis)) => *axis,
            _ => 1,
        };
        let axis = if axis < 0 {
            (q_shape.len() as i64 + axis) as usize
        } else {
            axis as usize
        };
        if axis >= q_shape.len() || q_shape[axis] != scale_data.len() {
            return Err(OnnxError::DecodeFailed {
                message: format!(
                    "fold weight DQ `{}`: scale len {} does not match axis {axis} in shape {:?}",
                    node.name,
                    scale_data.len(),
                    q_shape
                ),
            });
        }
        let outer = q_shape[..axis].iter().product::<usize>();
        let inner = q_shape[axis + 1..].iter().product::<usize>();
        let mut out = vec![0.0; q_data.len()];
        for o in 0..outer {
            for c in 0..scale_data.len() {
                let s = scale_data[c];
                let z = zp.get(c).copied().unwrap_or(0.0);
                let base = (o * scale_data.len() + c) * inner;
                for i in 0..inner {
                    out[base + i] = (q_data[base + i] - z) * s;
                }
            }
        }
        out
    };
    Tensor::from_vec(q_shape.to_vec(), out).map_err(|e| OnnxError::DecodeFailed {
        message: format!("fold weight DQ `{}`: {e}", node.name),
    })
}

fn conv_weight_as_oihw(
    model: &OnnxModel,
    weight_name: &str,
    weight: &Tensor,
) -> Result<Tensor, OnnxError> {
    if model.khwc_weights.contains(weight_name) && weight.rank() == 4 {
        return weight
            .permute(&[3, 2, 0, 1])
            .map_err(|e| OnnxError::DecodeFailed {
                message: format!("KHWC->OIHW permute failed for {weight_name}: {e}"),
            });
    }
    if model.group_khwc_weights.contains(weight_name) && weight.rank() == 4 {
        return weight
            .permute(&[0, 3, 1, 2])
            .map_err(|e| OnnxError::DecodeFailed {
                message: format!("group KHWC->OIHW permute failed for {weight_name}: {e}"),
            });
    }
    if model.dw_khwc_weights.contains(weight_name) && weight.rank() == 4 {
        return depthwise_khwc_to_oihw(weight, weight_name);
    }
    Ok(weight.clone())
}

fn depthwise_khwc_to_oihw(weight: &Tensor, weight_name: &str) -> Result<Tensor, OnnxError> {
    let shape = weight.shape();
    if shape.len() != 4 {
        return Ok(weight.clone());
    }
    let (kh, kw, channels, depth_mult) = (shape[0], shape[1], shape[2], shape[3]);
    let out_channels = channels
        .checked_mul(depth_mult)
        .ok_or_else(|| OnnxError::DecodeFailed {
            message: format!("depthwise weight channel overflow for {weight_name}"),
        })?;
    let mut data = vec![0.0_f32; out_channels * kh * kw];
    let src = weight.data();
    for c in 0..channels {
        for dm in 0..depth_mult {
            let oc = c * depth_mult + dm;
            for ki in 0..kh {
                for kj in 0..kw {
                    let src_idx = ((ki * kw + kj) * channels + c) * depth_mult + dm;
                    let dst_idx = (oc * kh + ki) * kw + kj;
                    data[dst_idx] = src[src_idx];
                }
            }
        }
    }
    Tensor::from_vec(vec![out_channels, 1, kh, kw], data).map_err(|e| OnnxError::DecodeFailed {
        message: format!("depthwise OIHW tensor for {weight_name}: {e}"),
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
    fn qlinear_rewrite_emits_standard_qlinear_conv_shell() {
        let mut model = build_minimal_model();
        model.initializers.insert(
            "w".to_string(),
            Tensor::from_vec(vec![2, 2, 4, 4], vec![0.05; 64]).unwrap(),
        );
        let mut stats = HashMap::new();
        stats.insert(
            "x".to_string(),
            MinMax {
                min: -2.0,
                max: 2.0,
                count: 16,
            },
        );
        stats.insert(
            "y".to_string(),
            MinMax {
                min: -4.0,
                max: 4.0,
                count: 16,
            },
        );

        rewrite_to_qlinear(&mut model, &stats).unwrap();
        assert!(model.nodes.iter().any(|n| n.op_type == "QLinearConv"));
        assert!(model.nodes.iter().any(|n| n.op_type == "QuantizeLinear"));
        assert!(model.nodes.iter().any(|n| n.op_type == "DequantizeLinear"));
        assert!(model.initializers.contains_key("w__qlinear_w_q"));
        assert!(model.initializers.contains_key("w__qlinear_w_scale"));
    }

    #[test]
    fn qlinear_rewrite_keeps_public_nchw_activations_for_internal_khwc_weight() {
        let mut model = build_minimal_model();
        model.khwc_weights.insert("w".to_string());
        model.initializers.insert(
            "w".to_string(),
            Tensor::from_vec(vec![1, 1, 16, 2], vec![0.05; 32]).unwrap(),
        );
        let mut stats = HashMap::new();
        stats.insert(
            "x".to_string(),
            MinMax {
                min: -2.0,
                max: 2.0,
                count: 16,
            },
        );
        stats.insert(
            "y".to_string(),
            MinMax {
                min: -4.0,
                max: 4.0,
                count: 16,
            },
        );

        rewrite_to_qlinear(&mut model, &stats).unwrap();
        assert!(!model.nodes.iter().any(|n| n.op_type == "Transpose"));
        assert_eq!(model.initializers["w__qlinear_w_q"].shape(), &[2, 16, 1, 1]);
    }

    #[test]
    fn qlinear_rewrite_uses_unique_activation_quant_outputs_for_shared_input() {
        let mut model = build_minimal_model();
        model.outputs = vec!["y0".to_string(), "y1".to_string()];
        model.initializers.insert(
            "w".to_string(),
            Tensor::from_vec(vec![2, 2, 4, 4], vec![0.05; 64]).unwrap(),
        );
        model.initializers.insert(
            "w1".to_string(),
            Tensor::from_vec(vec![2, 2, 4, 4], vec![0.07; 64]).unwrap(),
        );
        model.nodes = vec![
            OnnxNode {
                op_type: "Conv".to_string(),
                name: "conv0".to_string(),
                inputs: vec!["x".to_string(), "w".to_string()],
                outputs: vec!["y0".to_string()],
                attributes: HashMap::new(),
            },
            OnnxNode {
                op_type: "Conv".to_string(),
                name: "conv1".to_string(),
                inputs: vec!["x".to_string(), "w1".to_string()],
                outputs: vec!["y1".to_string()],
                attributes: HashMap::new(),
            },
        ];
        let mut stats = HashMap::new();
        for name in ["x", "y0", "y1"] {
            stats.insert(
                name.to_string(),
                MinMax {
                    min: -2.0,
                    max: 2.0,
                    count: 16,
                },
            );
        }

        rewrite_to_qlinear(&mut model, &stats).unwrap();

        let mut outputs = HashSet::new();
        for node in &model.nodes {
            for out in &node.outputs {
                assert!(outputs.insert(out.clone()), "duplicate output name `{out}`");
            }
        }
        assert!(
            model
                .nodes
                .iter()
                .any(|n| n.outputs.iter().any(|o| o == "conv0__qlinear_x_q"))
        );
        assert!(
            model
                .nodes
                .iter()
                .any(|n| n.outputs.iter().any(|o| o == "conv1__qlinear_x_q"))
        );
    }

    #[test]
    fn qdq_rewrite_quantizes_depthwise_khwc_weight_as_oihw() {
        let mut model = build_minimal_model();
        model.dw_khwc_weights.insert("w".to_string());
        model.initializers.insert(
            "w".to_string(),
            Tensor::from_vec(vec![3, 3, 2, 1], (0..18).map(|i| i as f32 - 9.0).collect()).unwrap(),
        );
        model.nodes[0]
            .attributes
            .insert("group".to_string(), OnnxAttribute::Int(2));

        rewrite_to_qdq(&mut model, &HashMap::new()).unwrap();

        assert_eq!(
            model
                .nodes
                .iter()
                .find(|n| n.op_type == "Conv")
                .unwrap()
                .inputs[1],
            "w_dq"
        );
        assert_eq!(model.initializers["w_q"].shape(), &[2, 1, 3, 3]);
        assert_eq!(model.initializers["w_scale"].shape(), &[2]);
        let dq = model
            .nodes
            .iter()
            .find(|n| n.op_type == "DequantizeLinear" && n.outputs[0] == "w_dq")
            .expect("depthwise weight DequantizeLinear missing");
        match dq.attributes.get("axis") {
            Some(OnnxAttribute::Int(0)) => {}
            other => panic!("axis attr not Int(0): {other:?}"),
        }
    }

    #[test]
    fn fold_constant_qdq_weight_restores_initializer_input() {
        let mut model = build_minimal_model();
        rewrite_to_qdq(&mut model, &HashMap::new()).unwrap();
        assert!(
            model
                .nodes
                .iter()
                .any(|n| n.op_type == "DequantizeLinear" && n.outputs[0] == "w_dq")
        );

        let folded = fold_constant_qdq_weights_for_yscv_fast(&mut model).unwrap();
        assert_eq!(folded, 1);
        assert!(model.initializers.contains_key("w_dq"));
        assert!(
            !model
                .nodes
                .iter()
                .any(|n| n.op_type == "DequantizeLinear" && n.outputs[0] == "w_dq")
        );
        let conv = model
            .nodes
            .iter()
            .find(|n| n.op_type == "Conv")
            .expect("Conv missing");
        assert_eq!(conv.inputs[1], "w_dq");
    }

    #[test]
    fn prune_unused_initializers_removes_folded_qdq_baggage() {
        let mut model = build_minimal_model();
        rewrite_to_qdq(&mut model, &HashMap::new()).unwrap();
        fold_constant_qdq_weights_for_yscv_fast(&mut model).unwrap();

        let removed = prune_unused_initializers(&mut model);

        assert_eq!(removed, 4);
        assert!(model.initializers.contains_key("w_dq"));
        assert!(!model.initializers.contains_key("w"));
        assert!(!model.initializers.contains_key("w_q"));
        assert!(!model.initializers.contains_key("w_scale"));
        assert!(!model.initializers.contains_key("w_zp"));
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
