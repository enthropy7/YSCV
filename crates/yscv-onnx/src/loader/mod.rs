use rustc_hash::FxHashSet;

use prost::Message;
use rustc_hash::FxHashMap;
use yscv_tensor::Tensor;

use crate::attr::Attr;
use crate::error::OnnxError;
use crate::proto::onnx;

/// A named tensor extracted from an ONNX model initializer.
#[derive(Debug, Clone)]
pub struct OnnxTensor {
    pub name: String,
    pub tensor: Tensor,
}

/// An ONNX operator node with its type, inputs, outputs, and attributes.
#[derive(Debug, Clone)]
pub struct OnnxNode {
    pub op_type: String,
    pub name: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: FxHashMap<Attr, OnnxAttribute>,
}

/// Supported ONNX attribute value types.
#[derive(Debug, Clone)]
pub enum OnnxAttribute {
    Int(i64),
    Float(f32),
    String(String),
    Ints(Vec<i64>),
    Floats(Vec<f32>),
    Tensor(Tensor),
}

/// Parsed ONNX model containing graph topology and weight tensors.
#[derive(Debug, Clone)]
pub struct OnnxModel {
    pub ir_version: i64,
    pub opset_version: i64,
    pub producer_name: String,
    pub graph_name: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub initializers: FxHashMap<String, Tensor>,
    pub nodes: Vec<OnnxNode>,
    /// Conv weight names that were pre-permuted OIHW → KHWC at load time.
    pub(crate) khwc_weights: FxHashSet<String>,
    /// Depthwise conv weight names pre-permuted [O,1,KH,KW] → [KH,KW,C,dm] at load time.
    pub(crate) dw_khwc_weights: FxHashSet<String>,
    /// Grouped conv weight names pre-permuted [O,I/G,KH,KW] → [O,KH,KW,I/G] at load time.
    pub(crate) group_khwc_weights: FxHashSet<String>,
    /// MatMul/Gemm weights packed to INT4 with per-group fp32 scales for
    /// the LLM decode hot path. Keyed by the original initializer name;
    /// the original `initializers` entry is removed when a weight is
    /// packed so dispatch routes through `packed_int4_gemv_dispatch`.
    pub(crate) packed_int4_weights: FxHashMap<String, crate::quantize::PackedInt4Weight>,
    /// Precomputed runtime metadata for fast per-inference environment setup.
    pub(crate) runtime_index: RuntimeModelIndex,
}

impl OnnxModel {
    /// Returns the weight tensor for a given initializer name, if present.
    pub fn get_initializer(&self, name: &str) -> Option<&Tensor> {
        self.initializers.get(name)
    }

    /// Returns the number of operator nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Rebuilds runtime slot/id metadata after graph mutations.
    pub(crate) fn rebuild_runtime_index(&mut self) {
        self.runtime_index = build_runtime_index(
            &self.inputs,
            &self.outputs,
            &self.initializers,
            &self.nodes,
            &self.khwc_weights,
            &self.dw_khwc_weights,
            &self.group_khwc_weights,
        );
    }
}

/// Loads an ONNX model from raw protobuf bytes.
pub fn load_onnx_model(data: &[u8]) -> Result<OnnxModel, OnnxError> {
    let model_proto = onnx::ModelProto::decode(data).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;

    let graph = model_proto.graph.ok_or(OnnxError::MissingGraph)?;

    let opset_version = model_proto
        .opset_import
        .first()
        .and_then(|o| o.version)
        .unwrap_or(0);

    let inputs: Vec<String> = graph
        .input
        .iter()
        .map(|v| v.name.clone().unwrap_or_default())
        .collect();

    let outputs: Vec<String> = graph
        .output
        .iter()
        .map(|v| v.name.clone().unwrap_or_default())
        .collect();

    let mut initializers = FxHashMap::default();
    for init in &graph.initializer {
        let name = init.name.clone().unwrap_or_default();
        let tensor = convert_tensor_proto(init)?;
        initializers.insert(name, tensor);
    }

    let mut nodes = Vec::new();
    for node_proto in &graph.node {
        let mut attributes = FxHashMap::default();
        for attr in &node_proto.attribute {
            let attr_name = attr.name.clone().unwrap_or_default();
            let value = convert_attribute(attr);
            if let Some(v) = value {
                attributes.insert(Attr::from_name(&attr_name), v);
            }
        }
        nodes.push(OnnxNode {
            op_type: node_proto.op_type.clone().unwrap_or_default(),
            name: node_proto.name.clone().unwrap_or_default(),
            inputs: node_proto.input.clone(),
            outputs: node_proto.output.clone(),
            attributes,
        });
    }

    let matmul_rhs_inputs: FxHashSet<String> = nodes
        .iter()
        .filter(|node| node.op_type == "MatMul")
        .filter_map(|node| node.inputs.get(1).cloned())
        .collect();
    let graph_outputs: FxHashSet<String> = outputs.iter().cloned().collect();
    let mut folded_nodes = Vec::with_capacity(nodes.len());
    for node in nodes {
        let can_fold_const_transpose = node.op_type == "Transpose"
            && node.inputs.len() == 1
            && node.outputs.len() == 1
            && matmul_rhs_inputs.contains(&node.outputs[0])
            && !graph_outputs.contains(&node.outputs[0]);
        if can_fold_const_transpose && let Some(input) = initializers.get(&node.inputs[0]) {
            let axes: Vec<usize> = match node.attributes.get(&Attr::Perm) {
                Some(OnnxAttribute::Ints(v)) if v.len() == input.rank() => {
                    v.iter().map(|&x| x as usize).collect()
                }
                _ => (0..input.rank()).rev().collect(),
            };
            if axes.iter().all(|&axis| axis < input.rank())
                && let Ok(permuted) = input.permute(&axes)
            {
                initializers.insert(node.outputs[0].clone(), permuted);
                continue;
            }
        }
        folded_nodes.push(node);
    }
    let nodes = folded_nodes;

    // Pre-permute group=1 Conv weights OIHW → KHWC at load time
    // so we don't pay the ~11ms permutation cost on every inference call.
    let mut khwc_weights = FxHashSet::default();
    for node in &nodes {
        if node.op_type != "Conv" || node.inputs.len() < 2 {
            continue;
        }
        let weight_name = &node.inputs[1];
        if khwc_weights.contains(weight_name) {
            continue;
        }
        // Only pre-permute group=1 conv weights
        let group = node
            .attributes
            .get(&Attr::Group)
            .and_then(|a| match a {
                OnnxAttribute::Int(v) => Some(*v),
                _ => None,
            })
            .unwrap_or(1);
        if group != 1 {
            continue;
        }
        if let Some(w) = initializers.get(weight_name)
            && w.rank() == 4
            && let Ok(permuted) = w.permute(&[2, 3, 1, 0])
        {
            initializers.insert(weight_name.clone(), permuted);
            khwc_weights.insert(weight_name.clone());
        }
    }

    // Pre-pack depthwise dm=1 weights to [KH, KW, C, 1] on CPU-only builds.
    // This removes per-inference OIHW→depthwise repack work in the hot path.
    //
    // Skipped on Metal and wgpu GPU builds: those backends' CPU fallback +
    // accelerator dispatch paths read weights in the original ONNX OIHW
    // layout. Keeping the export layout there means the same loader can
    // feed both CPU and accelerator runners; the accelerator handles its
    // own pre-permute internally if any.
    #[cfg(not(any(feature = "metal-backend", feature = "gpu")))]
    let mut dw_khwc_weights = FxHashSet::default();
    #[cfg(any(feature = "metal-backend", feature = "gpu"))]
    let dw_khwc_weights = FxHashSet::default();
    #[cfg(not(any(feature = "metal-backend", feature = "gpu")))]
    for node in &nodes {
        if node.op_type != "Conv" || node.inputs.len() < 2 {
            continue;
        }
        let weight_name = &node.inputs[1];
        if dw_khwc_weights.contains(weight_name) {
            continue;
        }
        let group = node
            .attributes
            .get(&Attr::Group)
            .and_then(|a| match a {
                OnnxAttribute::Int(v) => Some(*v as usize),
                _ => None,
            })
            .unwrap_or(1);
        if group <= 1 {
            continue;
        }

        if let Some(w) = initializers.get(weight_name)
            && w.rank() == 4
        {
            let ws = w.shape();
            let (o_ch, i_per_g, kh, kw) = (ws[0], ws[1], ws[2], ws[3]);
            // CPU depthwise fast path currently handles dm=1 only.
            if i_per_g != 1 || o_ch != group {
                continue;
            }

            let w_data = w.data();
            let mut packed = vec![0.0f32; kh * kw * group];
            for oc in 0..o_ch {
                for ki in 0..kh {
                    for kj in 0..kw {
                        let src = ((oc * i_per_g) * kh + ki) * kw + kj;
                        let dst = (ki * kw + kj) * group + oc;
                        packed[dst] = w_data[src];
                    }
                }
            }

            let packed_t = Tensor::from_vec(vec![kh, kw, group, 1], packed).map_err(|e| {
                OnnxError::DecodeFailed {
                    message: e.to_string(),
                }
            })?;
            initializers.insert(weight_name.clone(), packed_t);
            dw_khwc_weights.insert(weight_name.clone());
        }
    }

    // Pre-pack grouped conv weights [O, I/G, KH, KW] -> [O, KH, KW, I/G] on
    // CPU-only builds. This removes per-inference OIHW reordering in grouped
    // fallback path.
    #[cfg(not(any(feature = "metal-backend", feature = "gpu")))]
    let mut group_khwc_weights = FxHashSet::default();
    #[cfg(any(feature = "metal-backend", feature = "gpu"))]
    let group_khwc_weights = FxHashSet::default();
    #[cfg(not(any(feature = "metal-backend", feature = "gpu")))]
    for node in &nodes {
        if node.op_type != "Conv" || node.inputs.len() < 2 {
            continue;
        }
        let weight_name = &node.inputs[1];
        if group_khwc_weights.contains(weight_name) || dw_khwc_weights.contains(weight_name) {
            continue;
        }
        let group = node
            .attributes
            .get(&Attr::Group)
            .and_then(|a| match a {
                OnnxAttribute::Int(v) => Some(*v as usize),
                _ => None,
            })
            .unwrap_or(1);
        if group <= 1 {
            continue;
        }

        if let Some(w) = initializers.get(weight_name)
            && w.rank() == 4
        {
            let ws = w.shape();
            let (o_ch, i_per_g, kh, kw) = (ws[0], ws[1], ws[2], ws[3]);
            // Depthwise dm=1 is handled by the dedicated prepack path above.
            if i_per_g == 1 && o_ch == group {
                continue;
            }

            let w_data = w.data();
            let mut packed = vec![0.0f32; o_ch * kh * kw * i_per_g];
            for oc in 0..o_ch {
                for ki in 0..kh {
                    for kj in 0..kw {
                        for ci in 0..i_per_g {
                            let src = ((oc * i_per_g + ci) * kh + ki) * kw + kj;
                            let dst = ((oc * kh + ki) * kw + kj) * i_per_g + ci;
                            packed[dst] = w_data[src];
                        }
                    }
                }
            }

            let packed_t = Tensor::from_vec(vec![o_ch, kh, kw, i_per_g], packed).map_err(|e| {
                OnnxError::DecodeFailed {
                    message: e.to_string(),
                }
            })?;
            initializers.insert(weight_name.clone(), packed_t);
            group_khwc_weights.insert(weight_name.clone());
        }
    }

    let runtime_index = build_runtime_index(
        &inputs,
        &outputs,
        &initializers,
        &nodes,
        &khwc_weights,
        &dw_khwc_weights,
        &group_khwc_weights,
    );

    Ok(OnnxModel {
        ir_version: model_proto.ir_version.unwrap_or(0),
        opset_version,
        producer_name: model_proto.producer_name.unwrap_or_default(),
        graph_name: graph.name.unwrap_or_default(),
        inputs,
        outputs,
        initializers,
        nodes,
        khwc_weights,
        dw_khwc_weights,
        group_khwc_weights,
        packed_int4_weights: FxHashMap::default(),
        runtime_index,
    })
}

/// Loads an ONNX model from a file path.
///
/// Accepts any path-like type (`&str`, `String`, `&Path`, `PathBuf`, etc.).
pub fn load_onnx_model_from_file(
    path: impl AsRef<std::path::Path>,
) -> Result<OnnxModel, OnnxError> {
    let path = path.as_ref();
    let data = std::fs::read(path).map_err(|e| OnnxError::Io {
        message: format!("{}: {e}", path.display()),
    })?;
    load_onnx_model(&data)
}

mod convert;
use convert::*;

use crate::plan::{RuntimeModelIndex, build_runtime_index};
