use yscv_autograd::Graph;
use yscv_onnx::{
    OnnxExportAttr, OnnxExportGraph, OnnxExportNode, OnnxExportValueInfo, export_onnx_model,
    export_onnx_model_to_file,
};
use yscv_tensor::Tensor;

use crate::{ModelError, ModelLayer, SequentialModel};

/// Exports a `SequentialModel` to an ONNX protobuf byte vector.
///
/// `input_shape` is the full shape including batch dimension, e.g. `[1, 28, 28, 1]` for NHWC.
/// Linear layer weights are read from the autograd `graph`.
pub fn export_sequential_to_onnx(
    model: &SequentialModel,
    graph: &Graph,
    input_shape: &[i64],
    producer_name: &str,
    model_name: &str,
) -> Result<Vec<u8>, ModelError> {
    let export_graph = build_onnx_graph(model, graph, input_shape)?;
    export_onnx_model(&export_graph, producer_name, model_name)
        .map_err(|e| ModelError::OnnxExport(e.to_string()))
}

/// Exports a `SequentialModel` to an ONNX file.
pub fn export_sequential_to_onnx_file(
    model: &SequentialModel,
    graph: &Graph,
    input_shape: &[i64],
    producer_name: &str,
    model_name: &str,
    path: &std::path::Path,
) -> Result<(), ModelError> {
    let export_graph = build_onnx_graph(model, graph, input_shape)?;
    export_onnx_model_to_file(&export_graph, producer_name, model_name, path)
        .map_err(|e| ModelError::OnnxExport(e.to_string()))
}

fn build_onnx_graph(
    model: &SequentialModel,
    graph: &Graph,
    input_shape: &[i64],
) -> Result<OnnxExportGraph, ModelError> {
    let mut nodes = Vec::new();
    let mut initializers: Vec<(String, Tensor)> = Vec::new();
    let mut current_name = "input".to_string();
    let mut node_counter = 0usize;

    for (idx, layer) in model.layers().iter().enumerate() {
        let out_name = format!("layer{idx}_out");
        match layer {
            ModelLayer::Linear(l) => {
                let weight_tensor = graph
                    .value(l.weight_node().expect("linear layer has weight node"))
                    .map_err(|e| ModelError::OnnxExport(format!("Linear weight: {e}")))?
                    .clone();
                let bias_tensor = graph
                    .value(l.bias_node().expect("linear layer has bias node"))
                    .map_err(|e| ModelError::OnnxExport(format!("Linear bias: {e}")))?
                    .clone();

                let w_name = format!("linear{idx}_weight");
                let b_name = format!("linear{idx}_bias");

                let transposed = weight_tensor
                    .transpose_2d()
                    .map_err(|e| ModelError::OnnxExport(format!("transpose weight: {e}")))?;
                initializers.push((w_name.clone(), transposed));
                initializers.push((b_name.clone(), bias_tensor));

                nodes.push(OnnxExportNode {
                    op_type: "Gemm".to_string(),
                    name: format!("node{node_counter}"),
                    inputs: vec![current_name.clone(), w_name, b_name],
                    outputs: vec![out_name.clone()],
                    attributes: vec![OnnxExportAttr::Int("transB".to_string(), 0)],
                });
                node_counter += 1;
            }
            ModelLayer::ReLU(_) => {
                nodes.push(OnnxExportNode {
                    op_type: "Relu".to_string(),
                    name: format!("node{node_counter}"),
                    inputs: vec![current_name.clone()],
                    outputs: vec![out_name.clone()],
                    attributes: vec![],
                });
                node_counter += 1;
            }
            ModelLayer::LeakyReLU(l) => {
                nodes.push(OnnxExportNode {
                    op_type: "LeakyRelu".to_string(),
                    name: format!("node{node_counter}"),
                    inputs: vec![current_name.clone()],
                    outputs: vec![out_name.clone()],
                    attributes: vec![OnnxExportAttr::Float(
                        "alpha".to_string(),
                        l.negative_slope(),
                    )],
                });
                node_counter += 1;
            }
            ModelLayer::Sigmoid(_) => {
                nodes.push(OnnxExportNode {
                    op_type: "Sigmoid".to_string(),
                    name: format!("node{node_counter}"),
                    inputs: vec![current_name.clone()],
                    outputs: vec![out_name.clone()],
                    attributes: vec![],
                });
                node_counter += 1;
            }
            ModelLayer::Tanh(_) => {
                nodes.push(OnnxExportNode {
                    op_type: "Tanh".to_string(),
                    name: format!("node{node_counter}"),
                    inputs: vec![current_name.clone()],
                    outputs: vec![out_name.clone()],
                    attributes: vec![],
                });
                node_counter += 1;
            }
            ModelLayer::Dropout(d) => {
                let ratio_name = format!("dropout{idx}_ratio");
                initializers.push((ratio_name.clone(), Tensor::scalar(d.rate())));
                nodes.push(OnnxExportNode {
                    op_type: "Dropout".to_string(),
                    name: format!("node{node_counter}"),
                    inputs: vec![current_name.clone(), ratio_name],
                    outputs: vec![out_name.clone()],
                    attributes: vec![],
                });
                node_counter += 1;
            }
            ModelLayer::Conv2d(l) => {
                let weight_nhwc = l.weight();
                let w_onnx = nhwc_weight_to_nchw(weight_nhwc)?;
                let w_name = format!("conv{idx}_weight");
                initializers.push((w_name.clone(), w_onnx));

                let mut inputs = vec![current_name.clone(), w_name];
                if let Some(bias) = l.bias() {
                    let b_name = format!("conv{idx}_bias");
                    initializers.push((b_name.clone(), bias.clone()));
                    inputs.push(b_name);
                }

                nodes.push(OnnxExportNode {
                    op_type: "Conv".to_string(),
                    name: format!("node{node_counter}"),
                    inputs,
                    outputs: vec![out_name.clone()],
                    attributes: vec![
                        OnnxExportAttr::Ints(
                            "kernel_shape".to_string(),
                            vec![l.kernel_h() as i64, l.kernel_w() as i64],
                        ),
                        OnnxExportAttr::Ints(
                            "strides".to_string(),
                            vec![l.stride_h() as i64, l.stride_w() as i64],
                        ),
                    ],
                });
                node_counter += 1;
            }
            ModelLayer::BatchNorm2d(l) => {
                let scale_name = format!("bn{idx}_scale");
                let bias_name = format!("bn{idx}_bias");
                let mean_name = format!("bn{idx}_mean");
                let var_name = format!("bn{idx}_var");
                initializers.push((scale_name.clone(), l.gamma().clone()));
                initializers.push((bias_name.clone(), l.beta().clone()));
                initializers.push((mean_name.clone(), l.running_mean().clone()));
                initializers.push((var_name.clone(), l.running_var().clone()));

                nodes.push(OnnxExportNode {
                    op_type: "BatchNormalization".to_string(),
                    name: format!("node{node_counter}"),
                    inputs: vec![
                        current_name.clone(),
                        scale_name,
                        bias_name,
                        mean_name,
                        var_name,
                    ],
                    outputs: vec![out_name.clone()],
                    attributes: vec![OnnxExportAttr::Float("epsilon".to_string(), l.epsilon())],
                });
                node_counter += 1;
            }
            ModelLayer::MaxPool2d(l) => {
                nodes.push(OnnxExportNode {
                    op_type: "MaxPool".to_string(),
                    name: format!("node{node_counter}"),
                    inputs: vec![current_name.clone()],
                    outputs: vec![out_name.clone()],
                    attributes: vec![
                        OnnxExportAttr::Ints(
                            "kernel_shape".to_string(),
                            vec![l.kernel_h() as i64, l.kernel_w() as i64],
                        ),
                        OnnxExportAttr::Ints(
                            "strides".to_string(),
                            vec![l.stride_h() as i64, l.stride_w() as i64],
                        ),
                    ],
                });
                node_counter += 1;
            }
            ModelLayer::AvgPool2d(l) => {
                nodes.push(OnnxExportNode {
                    op_type: "AveragePool".to_string(),
                    name: format!("node{node_counter}"),
                    inputs: vec![current_name.clone()],
                    outputs: vec![out_name.clone()],
                    attributes: vec![
                        OnnxExportAttr::Ints(
                            "kernel_shape".to_string(),
                            vec![l.kernel_h() as i64, l.kernel_w() as i64],
                        ),
                        OnnxExportAttr::Ints(
                            "strides".to_string(),
                            vec![l.stride_h() as i64, l.stride_w() as i64],
                        ),
                    ],
                });
                node_counter += 1;
            }
            ModelLayer::GlobalAvgPool2d(_) => {
                nodes.push(OnnxExportNode {
                    op_type: "GlobalAveragePool".to_string(),
                    name: format!("node{node_counter}"),
                    inputs: vec![current_name.clone()],
                    outputs: vec![out_name.clone()],
                    attributes: vec![],
                });
                node_counter += 1;
            }
            ModelLayer::Flatten(_) => {
                nodes.push(OnnxExportNode {
                    op_type: "Flatten".to_string(),
                    name: format!("node{node_counter}"),
                    inputs: vec![current_name.clone()],
                    outputs: vec![out_name.clone()],
                    attributes: vec![OnnxExportAttr::Int("axis".to_string(), 1)],
                });
                node_counter += 1;
            }
            ModelLayer::Softmax(_) => {
                nodes.push(OnnxExportNode {
                    op_type: "Softmax".to_string(),
                    name: format!("node{node_counter}"),
                    inputs: vec![current_name.clone()],
                    outputs: vec![out_name.clone()],
                    attributes: vec![OnnxExportAttr::Int("axis".to_string(), -1)],
                });
                node_counter += 1;
            }
            ModelLayer::DepthwiseConv2d(l) => {
                let w_nhwc = l.weight();
                // Depthwise weight [KH,KW,C,1] → ONNX group conv [C,1,KH,KW]
                let s = w_nhwc.shape();
                let (kh, kw, c) = (s[0], s[1], s[2]);
                let src = w_nhwc.data();
                let mut dst = vec![0.0f32; src.len()];
                for ch in 0..c {
                    for r in 0..kh {
                        for col in 0..kw {
                            let src_idx = (r * kw + col) * c + ch;
                            let dst_idx = (ch * kh + r) * kw + col;
                            dst[dst_idx] = src[src_idx];
                        }
                    }
                }
                let w_onnx = Tensor::from_vec(vec![c, 1, kh, kw], dst)
                    .map_err(|e| ModelError::OnnxExport(e.to_string()))?;
                let w_name = format!("dwconv{idx}_weight");
                initializers.push((w_name.clone(), w_onnx));

                let mut inputs = vec![current_name.clone(), w_name];
                if let Some(bias) = l.bias() {
                    let b_name = format!("dwconv{idx}_bias");
                    initializers.push((b_name.clone(), bias.clone()));
                    inputs.push(b_name);
                }

                nodes.push(OnnxExportNode {
                    op_type: "Conv".to_string(),
                    name: format!("node{node_counter}"),
                    inputs,
                    outputs: vec![out_name.clone()],
                    attributes: vec![
                        OnnxExportAttr::Ints(
                            "kernel_shape".to_string(),
                            vec![l.kernel_h() as i64, l.kernel_w() as i64],
                        ),
                        OnnxExportAttr::Ints(
                            "strides".to_string(),
                            vec![l.stride_h() as i64, l.stride_w() as i64],
                        ),
                        OnnxExportAttr::Int("group".to_string(), c as i64),
                    ],
                });
                node_counter += 1;
            }
            ModelLayer::SeparableConv2d(l) => {
                // Depthwise part
                let dw = l.depthwise();
                let s = dw.weight().shape();
                let (kh, kw, c) = (s[0], s[1], s[2]);
                let src = dw.weight().data();
                let mut dst = vec![0.0f32; src.len()];
                for ch in 0..c {
                    for r in 0..kh {
                        for col in 0..kw {
                            let src_idx = (r * kw + col) * c + ch;
                            let dst_idx = (ch * kh + r) * kw + col;
                            dst[dst_idx] = src[src_idx];
                        }
                    }
                }
                let dw_onnx = Tensor::from_vec(vec![c, 1, kh, kw], dst)
                    .map_err(|e| ModelError::OnnxExport(e.to_string()))?;
                let dw_name = format!("sepconv{idx}_dw_weight");
                initializers.push((dw_name.clone(), dw_onnx));

                let dw_out_name = format!("layer{idx}_dw_out");
                nodes.push(OnnxExportNode {
                    op_type: "Conv".to_string(),
                    name: format!("node{node_counter}"),
                    inputs: vec![current_name.clone(), dw_name],
                    outputs: vec![dw_out_name.clone()],
                    attributes: vec![
                        OnnxExportAttr::Ints(
                            "kernel_shape".to_string(),
                            vec![kh as i64, kw as i64],
                        ),
                        OnnxExportAttr::Ints(
                            "strides".to_string(),
                            vec![l.stride_h() as i64, l.stride_w() as i64],
                        ),
                        OnnxExportAttr::Int("group".to_string(), c as i64),
                    ],
                });
                node_counter += 1;

                // Pointwise part
                let pw = l.pointwise();
                let pw_onnx = nhwc_weight_to_nchw(pw.weight())?;
                let pw_name = format!("sepconv{idx}_pw_weight");
                initializers.push((pw_name.clone(), pw_onnx));
                let mut pw_inputs = vec![dw_out_name, pw_name];
                if let Some(bias) = pw.bias() {
                    let b_name = format!("sepconv{idx}_bias");
                    initializers.push((b_name.clone(), bias.clone()));
                    pw_inputs.push(b_name);
                }
                nodes.push(OnnxExportNode {
                    op_type: "Conv".to_string(),
                    name: format!("node{node_counter}"),
                    inputs: pw_inputs,
                    outputs: vec![out_name.clone()],
                    attributes: vec![
                        OnnxExportAttr::Ints("kernel_shape".to_string(), vec![1, 1]),
                        OnnxExportAttr::Ints("strides".to_string(), vec![1, 1]),
                    ],
                });
                node_counter += 1;
            }
            ModelLayer::ResidualBlock(r) => {
                // Export inner layers as Identity nodes, then add skip connection.
                let skip_name = current_name.clone();
                let mut inner_name = current_name.clone();
                for (sub_idx, _inner_layer) in r.layers().iter().enumerate() {
                    let inner_out = format!("layer{idx}_res{sub_idx}_out");
                    nodes.push(OnnxExportNode {
                        op_type: "Identity".to_string(),
                        name: format!("node{node_counter}"),
                        inputs: vec![inner_name.clone()],
                        outputs: vec![inner_out.clone()],
                        attributes: vec![],
                    });
                    node_counter += 1;
                    inner_name = inner_out;
                }
                // Add node for skip connection: output = inner_output + skip_input
                nodes.push(OnnxExportNode {
                    op_type: "Add".to_string(),
                    name: format!("node{node_counter}"),
                    inputs: vec![inner_name, skip_name],
                    outputs: vec![out_name.clone()],
                    attributes: vec![],
                });
                node_counter += 1;
            }
            ModelLayer::Embedding(_)
            | ModelLayer::LayerNorm(_)
            | ModelLayer::GroupNorm(_)
            | ModelLayer::LoraLinear(_)
            | ModelLayer::Conv1d(_)
            | ModelLayer::Conv3d(_)
            | ModelLayer::ConvTranspose2d(_)
            | ModelLayer::AdaptiveAvgPool2d(_)
            | ModelLayer::AdaptiveMaxPool2d(_)
            | ModelLayer::InstanceNorm(_)
            | ModelLayer::PixelShuffle(_)
            | ModelLayer::Upsample(_)
            | ModelLayer::GELU(_)
            | ModelLayer::SiLU(_)
            | ModelLayer::Mish(_)
            | ModelLayer::PReLU(_)
            | ModelLayer::Rnn(_)
            | ModelLayer::Lstm(_)
            | ModelLayer::Gru(_)
            | ModelLayer::MultiHeadAttention(_)
            | ModelLayer::TransformerEncoder(_)
            | ModelLayer::FeedForward(_)
            | ModelLayer::DeformableConv2d(_) => {
                nodes.push(OnnxExportNode {
                    op_type: "Identity".to_string(),
                    name: format!("node{node_counter}"),
                    inputs: vec![current_name.clone()],
                    outputs: vec![out_name.clone()],
                    attributes: vec![],
                });
                node_counter += 1;
            }
        }
        current_name = out_name;
    }

    let inputs = vec![OnnxExportValueInfo {
        name: "input".to_string(),
        shape: input_shape.to_vec(),
    }];
    let outputs = vec![OnnxExportValueInfo {
        name: current_name,
        shape: vec![],
    }];

    Ok(OnnxExportGraph {
        nodes,
        initializers,
        inputs,
        outputs,
        opset_version: 13,
        int64_initializers: Vec::new(),
    })
}

/// Transpose convolution weight from NHWC `[KH, KW, Cin, Cout]`
/// to ONNX NCHW `[Cout, Cin, KH, KW]`.
fn nhwc_weight_to_nchw(w: &Tensor) -> Result<Tensor, ModelError> {
    let s = w.shape();
    if s.len() != 4 {
        return Err(ModelError::OnnxExport(format!(
            "conv weight rank must be 4, got {}",
            s.len()
        )));
    }
    let (kh, kw, cin, cout) = (s[0], s[1], s[2], s[3]);
    let src = w.data();
    let mut dst = vec![0.0f32; src.len()];
    for oc in 0..cout {
        for ic in 0..cin {
            for r in 0..kh {
                for c in 0..kw {
                    let src_idx = ((r * kw + c) * cin + ic) * cout + oc;
                    let dst_idx = ((oc * cin + ic) * kh + r) * kw + c;
                    dst[dst_idx] = src[src_idx];
                }
            }
        }
    }
    Tensor::from_vec(vec![cout, cin, kh, kw], dst)
        .map_err(|e| ModelError::OnnxExport(e.to_string()))
}
