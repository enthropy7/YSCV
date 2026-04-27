//! Quantization utilities for ONNX models.
//!
//! - [`quantize_weights_int4`]: post-training INT4 weight quantization with
//!   per-channel scale and zero-point, inserting DequantizeLinear nodes
//!   into the graph.
//! - [`calibrate`]: activation-statistics collection for post-training
//!   quantization (PTQ). Install a `CalibrationCollector` before running
//!   inference to capture per-tensor min/max/count, then derive scales
//!   downstream.

pub mod calibrate;
pub mod derive;
pub mod packed_int4;
pub mod rewriter;

pub use calibrate::{CalibrationCollector, CalibrationScope, MinMax};
pub use derive::{
    QuantParams, QuantTarget, derive_asymmetric, derive_per_channel_symmetric, derive_symmetric,
    int4_symmetric_per_channel, int8_asymmetric_per_tensor, int8_symmetric_per_channel,
    int8_symmetric_per_tensor,
};
pub use packed_int4::{PackedInt4Weight, quantize_matmul_weights_int4_packed};
pub use rewriter::rewrite_to_qdq;

use std::collections::HashMap;

use yscv_tensor::Tensor;

use crate::error::OnnxError;
use crate::loader::{OnnxAttribute, OnnxModel, OnnxNode};

/// Quantize all weight tensors in an ONNX model to INT4 with per-channel
/// scale and zero-point.
///
/// For each weight initializer with more than one element:
///   1. Compute per-channel min/max along axis 0 (output channel dimension).
///   2. Derive scale and zero_point to map the range to [-8, 7].
///   3. Quantize the data to packed INT4 nibbles.
///   4. Replace the original initializer with the packed INT4 data (as its
///      unpacked integer values in f32) and insert a DequantizeLinear node.
///
/// The resulting model produces the same outputs as the original at reduced
/// precision. The packing happens in the initializers; during execution the
/// DequantizeLinear nodes unpack values back to f32.
pub fn quantize_weights_int4(model: &mut OnnxModel) -> Result<(), OnnxError> {
    // Collect names of weight initializers that are referenced as conv/matmul
    // weight inputs (skip biases, norms, etc. which are small).
    let weight_names: Vec<String> = model
        .initializers
        .keys()
        .filter(|name| {
            let t = &model.initializers[name.as_str()];
            // Only quantize tensors with sufficient elements and rank >= 2
            t.shape().len() >= 2 && t.len() > 16
        })
        .cloned()
        .collect();

    if weight_names.is_empty() {
        return Ok(());
    }

    let mut new_initializers: HashMap<String, Tensor> = HashMap::new();
    let mut dequant_nodes: Vec<(String, OnnxNode)> = Vec::new();

    for name in &weight_names {
        let tensor = match model.initializers.get(name) {
            Some(t) => t,
            None => continue,
        };
        let shape = tensor.shape();
        let data = tensor.data();

        // Per-channel quantization along axis 0.
        let channels = shape[0];
        let channel_size = data.len() / channels;

        let mut scales = Vec::with_capacity(channels);
        let mut zero_points = Vec::with_capacity(channels);
        let mut quantized_i8 = Vec::with_capacity(data.len());

        for ch in 0..channels {
            let start = ch * channel_size;
            let end = start + channel_size;
            let slice = &data[start..end];

            let (mut min_val, mut max_val) = slice
                .iter()
                .fold((f32::MAX, f32::MIN), |(mn, mx), &v| (mn.min(v), mx.max(v)));

            // Ensure range includes zero for symmetric quantization
            min_val = min_val.min(0.0);
            max_val = max_val.max(0.0);

            let (scale, zp) = if (max_val - min_val).abs() < f32::EPSILON {
                (1.0f32, 0i8)
            } else {
                // Map [min_val, max_val] to [-8, 7]
                let s = (max_val - min_val) / 15.0;
                let z = (-8.0 - min_val / s).round().clamp(-8.0, 7.0) as i8;
                (s, z)
            };

            scales.push(scale);
            zero_points.push(zp);

            let inv_s = if scale.abs() > f32::EPSILON {
                1.0 / scale
            } else {
                0.0
            };
            let zp_i32 = zp as i32;

            for &v in slice {
                let q = ((v * inv_s).round() as i32 + zp_i32).clamp(-8, 7) as i8;
                quantized_i8.push(q);
            }
        }

        // Store quantized data as unpacked integer values in f32 (for runtime
        // DequantizeLinear which works on f32 tensors).
        // Note: `pack_int4(&quantized_i8)` can be used for serialization to
        // ONNX files with packed INT4 format.
        let quant_data: Vec<f32> = quantized_i8.iter().map(|&v| v as f32).collect();
        let quant_tensor =
            Tensor::from_vec(shape.to_vec(), quant_data).map_err(|e| OnnxError::DecodeFailed {
                message: format!("failed to create quantized tensor for {name}: {e}"),
            })?;

        let quant_name = format!("{name}_int4");
        new_initializers.insert(quant_name.clone(), quant_tensor);

        // Create per-channel scale tensor
        let scale_name = format!("{name}_scale");
        let scale_tensor = Tensor::from_vec(vec![channels], scales.clone()).map_err(|e| {
            OnnxError::DecodeFailed {
                message: format!("failed to create scale tensor for {name}: {e}"),
            }
        })?;
        new_initializers.insert(scale_name.clone(), scale_tensor);

        // Create per-channel zero_point tensor
        let zp_name = format!("{name}_zero_point");
        let zp_data: Vec<f32> = zero_points.iter().map(|&v| v as f32).collect();
        let zp_tensor =
            Tensor::from_vec(vec![channels], zp_data).map_err(|e| OnnxError::DecodeFailed {
                message: format!("failed to create zero_point tensor for {name}: {e}"),
            })?;
        new_initializers.insert(zp_name.clone(), zp_tensor);

        // Create DequantizeLinear node: (quantized, scale, zp) -> original_name
        let dequant_node = OnnxNode {
            op_type: "DequantizeLinear".to_string(),
            name: format!("{name}_dequant"),
            inputs: vec![quant_name, scale_name, zp_name],
            outputs: vec![name.clone()],
            attributes: {
                let mut attrs = HashMap::new();
                // axis=0 for per-channel dequantization
                attrs.insert("axis".to_string(), OnnxAttribute::Int(0));
                attrs
            },
        };

        dequant_nodes.push((name.clone(), dequant_node));
    }

    // Apply changes: remove original initializers, add quantized ones
    for (orig_name, _) in &dequant_nodes {
        model.initializers.remove(orig_name);
    }
    model.initializers.extend(new_initializers);

    // Insert DequantizeLinear nodes before the first consumer of each weight.
    // For simplicity, prepend all dequant nodes at the beginning of the graph.
    let mut dequant_node_list: Vec<OnnxNode> = dequant_nodes.into_iter().map(|(_, n)| n).collect();
    dequant_node_list.append(&mut model.nodes);
    model.nodes = dequant_node_list;
    model.rebuild_runtime_index();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_weights_basic() {
        // Create a minimal model with one weight tensor (must be > 16 elements)
        let weight_data: Vec<f32> = (0..32).map(|x| (x as f32 - 16.0) * 0.1).collect();
        let weight = Tensor::from_vec(vec![2, 16], weight_data.clone()).expect("valid shape");

        let mut model = OnnxModel {
            ir_version: 7,
            opset_version: 13,
            producer_name: "test".to_string(),
            graph_name: "test_graph".to_string(),
            inputs: vec!["input".to_string()],
            outputs: vec!["output".to_string()],
            initializers: {
                let mut m = HashMap::new();
                m.insert("conv_weight".to_string(), weight);
                m
            },
            nodes: vec![OnnxNode {
                op_type: "Conv".to_string(),
                name: "conv0".to_string(),
                inputs: vec!["input".to_string(), "conv_weight".to_string()],
                outputs: vec!["output".to_string()],
                attributes: HashMap::new(),
            }],
            khwc_weights: Default::default(),
            dw_khwc_weights: Default::default(),
            group_khwc_weights: Default::default(),
            packed_int4_weights: Default::default(),
            runtime_index: Default::default(),
        };

        let result = quantize_weights_int4(&mut model);
        assert!(result.is_ok(), "quantize_weights_int4 failed: {:?}", result);

        // Original weight should be removed
        assert!(
            !model.initializers.contains_key("conv_weight"),
            "original weight should be removed"
        );

        // Quantized tensors should exist
        assert!(model.initializers.contains_key("conv_weight_int4"));
        assert!(model.initializers.contains_key("conv_weight_scale"));
        assert!(model.initializers.contains_key("conv_weight_zero_point"));

        // DequantizeLinear node should be inserted
        let dequant = model
            .nodes
            .iter()
            .find(|n| n.op_type == "DequantizeLinear")
            .expect("missing DequantizeLinear node");
        assert_eq!(dequant.outputs[0], "conv_weight");

        // Quantized data should have the same shape
        let quant = &model.initializers["conv_weight_int4"];
        assert_eq!(quant.shape(), &[2, 16]);

        // All quantized values should be in INT4 range
        for &v in quant.data() {
            assert!(
                (-8.0..=7.0).contains(&v),
                "quantized value {v} outside INT4 range"
            );
        }
    }

    #[test]
    fn quantize_weights_skips_small_tensors() {
        let bias = Tensor::from_vec(vec![4], vec![0.1, 0.2, 0.3, 0.4]).expect("valid");
        let mut model = OnnxModel {
            ir_version: 7,
            opset_version: 13,
            producer_name: "test".to_string(),
            graph_name: "test_graph".to_string(),
            inputs: vec![],
            outputs: vec![],
            initializers: {
                let mut m = HashMap::new();
                m.insert("bias".to_string(), bias);
                m
            },
            nodes: vec![],
            khwc_weights: Default::default(),
            dw_khwc_weights: Default::default(),
            group_khwc_weights: Default::default(),
            packed_int4_weights: Default::default(),
            runtime_index: Default::default(),
        };

        quantize_weights_int4(&mut model).expect("should succeed");
        // Bias should still be there unchanged (rank 1, too small)
        assert!(model.initializers.contains_key("bias"));
    }
}
