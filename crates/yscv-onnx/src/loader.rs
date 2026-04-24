use std::collections::{HashMap, HashSet};

use prost::Message;
use yscv_tensor::Tensor;

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
    pub attributes: HashMap<String, OnnxAttribute>,
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

/// Precomputed runtime metadata built once at model load time.
#[derive(Debug, Clone, Default)]
pub(crate) struct RuntimeModelIndex {
    /// Dense slot id per tensor name for TensorEnv hot-path lookups.
    pub(crate) name_to_id: HashMap<String, usize>,
    /// Slot ids for weights pre-permuted OIHW -> KHWC.
    pub(crate) khwc_weight_ids: HashSet<usize>,
    /// Slot ids for depthwise weights pre-permuted [O,1,KH,KW] -> [KH,KW,C,dm].
    pub(crate) dw_khwc_weight_ids: HashSet<usize>,
    /// Slot ids for grouped-conv weights pre-permuted [O,I/G,KH,KW] -> [O,KH,KW,I/G].
    pub(crate) group_khwc_weight_ids: HashSet<usize>,
    /// Number of graph uses per value name (input edge count).
    pub(crate) use_counts: HashMap<String, usize>,
    /// Dense use-count table indexed by runtime slot id.
    pub(crate) use_counts_by_id: Vec<usize>,
    /// Lightweight op tags for fast fusion pattern matching at runtime.
    pub(crate) node_kinds: Vec<NodeKind>,
    /// Per-node branch classification for tower-parallel execution of
    /// siamese-style graphs. 0 = first-input-only, 1 = second-input-only,
    /// 2 = merge/head. Empty when the graph has no parallelizable split
    /// (single-input models, or both branches share too many nodes).
    pub(crate) node_branches: Vec<u8>,
    /// Pre-resolved slot IDs for each node's inputs.
    pub(crate) node_input_ids: Vec<Vec<Option<usize>>>,
    /// Pre-resolved slot IDs for each node's outputs (Session 13 R3).
    pub(crate) node_output_ids: Vec<Vec<Option<usize>>>,
    /// Pre-parsed Conv parameters per node. Only populated for Conv nodes.
    pub(crate) conv_params: Vec<Option<ConvParams>>,
    /// Pre-compiled execution plan. Each entry maps to a node action.
    /// Built once at model load — eliminates per-inference dispatch overhead.
    pub(crate) execution_plan: Vec<NodeAction>,
    /// Pre-packed B-matrix (blocked-GEMM layout) per constant Conv/MatMul
    /// weight tensor, keyed by weight tensor name. Built once at model load
    /// via `yscv_kernels::pack_b_for_session`; shared `Arc` handed to every
    /// inference, skipping both the fingerprint cache lookup AND the pack
    /// itself on the hot path. Only populated for weights whose dispatch
    /// routes through blocked GEMM (pointwise Conv with KHWC layout, MatMul).
    pub(crate) prepacked_weights: HashMap<String, std::sync::Arc<yscv_kernels::PackedB>>,
    /// Same pre-packed weights, but indexed by dense runtime slot id.
    /// Lets hot paths bypass string hashing by using `node_input_ids`.
    pub(crate) prepacked_weights_by_id: Vec<Option<std::sync::Arc<yscv_kernels::PackedB>>>,
}

/// Pre-computed Conv parameters parsed once at model load.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ConvParams {
    pub stride_h: usize,
    pub stride_w: usize,
    pub pad_top: usize,
    pub pad_left: usize,
    pub pad_bottom: usize,
    pub pad_right: usize,
    pub group: usize,
    pub has_padding: bool,
    /// True if group == output_channels (depthwise convolution).
    pub is_depthwise: bool,
    /// True if kernel is 1×1 (pointwise convolution).
    pub is_pointwise: bool,
}

/// Pre-compiled execution action for JIT dispatch.
/// Built once at model load, eliminates per-inference NodeKind matching + layout checks.
#[derive(Debug, Clone)]
pub(crate) enum NodeAction {
    /// Conv with pre-resolved params + activation.
    Conv {
        node_idx: usize,
        activation: u8, // 0=None, 1=Relu, 2=Silu
    },
    /// Conv + Add (residual) in-place: reuse Conv's output buffer as the Add
    /// destination. Optionally applies Relu in-place after the Add.
    ConvAdd {
        conv_idx: usize,
        add_idx: usize,
        skip_input_idx: u8,
        post_activation: u8, // 0=none, 1=Relu
        relu_idx: u32,       // valid only when post_activation == 1
    },
    /// Fused DW+PW: execute both convolutions back-to-back.
    FusedDwPw {
        dw_idx: usize,
        pw_idx: usize,
        dw_activation: u8,
        pw_activation: u8,
    },
    /// Fused PW+DW: PW (expansion 1×1) feeds directly into DW, with
    /// the PW output kept as a local `Tensor` (never inserted into
    /// `env`). Mirror of `FusedDwPw` for the inverted-bottleneck
    /// opening — MobileNet-style blocks are `PW_expand → DW → PW_reduce`,
    /// where the (DW, PW_reduce) pair is typically fused into
    /// `FusedDwPw` for non-residual blocks and (PW_reduce, Add) into
    /// `Conv_Add_fused` for residual blocks; the remaining
    /// (PW_expand, DW) pair is what this variant targets.
    FusedPwDw {
        pw_idx: usize,
        dw_idx: usize,
        pw_activation: u8,
        dw_activation: u8,
    },
    /// MatMul where the left operand comes directly from a `Transpose`
    /// with `perm=[0, 2, 1]` (swap of the last two axes of a rank-3
    /// tensor). The Transpose node is elided at dispatch time: the
    /// MatMul reads the pre-transpose input via a `transA=1` kernel
    /// (`matmul_2d_slices_trans_a`), so no intermediate transposed
    /// tensor hits the env HashMap or memory. Mirrors ORT's
    /// `MatmulTransposeFusion` contrib op.
    ///
    /// `transpose_idx` is the Transpose node's index (for profile
    /// labels and potential reuse when the Transpose has multiple
    /// MatMul consumers). `matmul_idx` is the MatMul node.
    /// `cleanup_transpose` is set only on the LAST `FusedTransposeMatMul`
    /// that references this transpose — at cleanup time that one is the
    /// sole variant that decrements the transpose's input-refcount,
    /// matching the original graph's single Transpose-use of its input
    /// (e.g. `Reshape_output_0`). Earlier variants re-read the same
    /// pre-transpose tensor and must not evict it from `env`.
    FusedTransposeMatMul {
        transpose_idx: usize,
        matmul_idx: usize,
        cleanup_transpose: bool,
    },
    /// Generic op — falls through to full dispatch.
    Generic { node_idx: usize, kind: NodeKind },
    /// Skipped (fused into previous action).
    Skip,
    /// Step A: consecutive NCHWc-capable ops that should execute as a
    /// single layout-native chain. At runtime, the input is reordered
    /// NHWC→NCHWc once at chain entry; every inner op runs on NCHWc
    /// tensors via `*_nchwc` kernels; the last output is reordered
    /// back to NHWC. Enabled via `YSCV_NCHWC_CHAIN=1` env (default off
    /// until runner dispatch lands in A.2).
    ///
    /// `members` is the list of execution-plan sub-actions that form
    /// this chain. Each one must be a Conv/ConvAdd/FusedDwPw or other
    /// NCHWc-capable variant. The original actions are still in
    /// `execution_plan` but marked `Skip`; `NchwcChain` owns dispatch.
    NchwcChain {
        /// Sub-actions belonging to this chain. At most one NchwcChain
        /// covers any given action index.
        members: Vec<NchwcChainMember>,
        /// NHWC input tensor name (consumed once at chain entry).
        entry_input: String,
        /// NHWC output tensor name (produced once at chain exit).
        exit_output: String,
    },
}

/// Step A: member of an `NchwcChain`. Stored as an enum (not a bare
/// `NodeAction` variant reference) so the chain carries all the data
/// needed for NCHWc dispatch without round-trips through
/// `execution_plan` indexing.
#[derive(Debug, Clone)]
pub(crate) enum NchwcChainMember {
    /// A single Conv (possibly with fused activation). Corresponds to
    /// [`NodeAction::Conv`].
    Conv { node_idx: usize, activation: u8 },
}

/// Small runtime op classification used by the CPU runner's fusion scanner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NodeKind {
    Conv,
    ConvRelu,
    ConvSilu,
    BatchNormalization,
    Relu,
    Sigmoid,
    Mul,
    Gemm,
    Add,
    MatMul,
    Reshape,
    Constant,
    Concat,
    Transpose,
    Other,
}

impl NodeKind {
    #[inline]
    pub(crate) fn from_op_type(op_type: &str) -> Self {
        match op_type {
            "Conv" => Self::Conv,
            "Conv_Relu" => Self::ConvRelu,
            "Conv_SiLU" => Self::ConvSilu,
            "BatchNormalization" => Self::BatchNormalization,
            "Relu" => Self::Relu,
            "Sigmoid" => Self::Sigmoid,
            "Mul" => Self::Mul,
            "Gemm" => Self::Gemm,
            "Add" => Self::Add,
            "MatMul" => Self::MatMul,
            "Reshape" => Self::Reshape,
            "Constant" => Self::Constant,
            "Concat" => Self::Concat,
            "Transpose" => Self::Transpose,
            _ => Self::Other,
        }
    }
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
    pub initializers: HashMap<String, Tensor>,
    pub nodes: Vec<OnnxNode>,
    /// Conv weight names that were pre-permuted OIHW → KHWC at load time.
    pub(crate) khwc_weights: HashSet<String>,
    /// Depthwise conv weight names pre-permuted [O,1,KH,KW] → [KH,KW,C,dm] at load time.
    pub(crate) dw_khwc_weights: HashSet<String>,
    /// Grouped conv weight names pre-permuted [O,I/G,KH,KW] → [O,KH,KW,I/G] at load time.
    pub(crate) group_khwc_weights: HashSet<String>,
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

    let mut initializers = HashMap::new();
    for init in &graph.initializer {
        let name = init.name.clone().unwrap_or_default();
        let tensor = convert_tensor_proto(init)?;
        initializers.insert(name, tensor);
    }

    let mut nodes = Vec::new();
    for node_proto in &graph.node {
        let mut attributes = HashMap::new();
        for attr in &node_proto.attribute {
            let attr_name = attr.name.clone().unwrap_or_default();
            let value = convert_attribute(attr);
            if let Some(v) = value {
                attributes.insert(attr_name, v);
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

    // Pre-permute group=1 Conv weights OIHW → KHWC at load time
    // so we don't pay the ~11ms permutation cost on every inference call.
    let mut khwc_weights = HashSet::new();
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
            .get("group")
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
    // Keep OIHW on Metal builds because the Metal depthwise path expects the
    // original export layout.
    let mut dw_khwc_weights = HashSet::new();
    #[cfg(not(feature = "metal-backend"))]
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
            .get("group")
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
    let mut group_khwc_weights = HashSet::new();
    #[cfg(not(feature = "metal-backend"))]
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
            .get("group")
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

fn convert_tensor_proto(tp: &onnx::TensorProto) -> Result<Tensor, OnnxError> {
    let shape: Vec<usize> = tp.dims.iter().map(|&d| d as usize).collect();
    let expected_len: usize = if shape.is_empty() {
        1
    } else {
        shape.iter().product()
    };
    let data_type = tp.data_type.unwrap_or(0);

    let data = match data_type {
        // FLOAT = 1
        1 => {
            if !tp.float_data.is_empty() {
                tp.float_data.clone()
            } else if let Some(ref raw) = tp.raw_data {
                raw_bytes_to_f32(raw)
            } else {
                vec![0.0f32; expected_len]
            }
        }
        // DOUBLE = 11
        11 => {
            if !tp.double_data.is_empty() {
                tp.double_data.iter().map(|&d| d as f32).collect()
            } else if let Some(ref raw) = tp.raw_data {
                raw_bytes_to_f64_as_f32(raw)
            } else {
                vec![0.0f32; expected_len]
            }
        }
        // INT64 = 7
        7 => {
            if !tp.int64_data.is_empty() {
                tp.int64_data.iter().map(|&v| v as f32).collect()
            } else if let Some(ref raw) = tp.raw_data {
                raw_bytes_to_i64_as_f32(raw)
            } else {
                vec![0.0f32; expected_len]
            }
        }
        // INT32 = 6
        6 => {
            if !tp.int32_data.is_empty() {
                tp.int32_data.iter().map(|&v| v as f32).collect()
            } else if let Some(ref raw) = tp.raw_data {
                raw_bytes_to_i32_as_f32(raw)
            } else {
                vec![0.0f32; expected_len]
            }
        }
        other => {
            return Err(OnnxError::UnsupportedDataType { data_type: other });
        }
    };

    if data.len() != expected_len {
        return Err(OnnxError::InitializerShapeMismatch {
            name: tp.name.clone().unwrap_or_default(),
            expected: expected_len,
            got: data.len(),
        });
    }

    // Preserve 0-D scalar shapes: ONNX TensorProto with dims=[] is a 0-D
    // scalar, not a 1-D tensor.  Many graph patterns (Gather with scalar
    // indices → Unsqueeze → Concat for reshape targets) depend on correct
    // rank propagation.
    Tensor::from_vec(shape, data).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })
}

fn raw_bytes_to_f32(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn raw_bytes_to_f64_as_f32(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(8)
        .map(|c| {
            let v = f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
            v as f32
        })
        .collect()
}

fn raw_bytes_to_i64_as_f32(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(8)
        .map(|c| {
            let v = i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
            v as f32
        })
        .collect()
}

fn raw_bytes_to_i32_as_f32(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4)
        .map(|c| {
            let v = i32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            v as f32
        })
        .collect()
}

fn convert_attribute(attr: &onnx::AttributeProto) -> Option<OnnxAttribute> {
    // Some exporter/toolchain combinations omit `AttributeProto.type` for
    // Constant nodes and rely on the populated value field (`t`, `ints`, ...).
    // Infer the type from payload presence when the enum tag is missing.
    let attr_type = attr.r#type.unwrap_or_else(|| {
        if attr.t.is_some() {
            4
        } else if attr.f.is_some() {
            1
        } else if attr.i.is_some() {
            2
        } else if attr.s.is_some() {
            3
        } else if !attr.floats.is_empty() {
            6
        } else if !attr.ints.is_empty() {
            7
        } else {
            0
        }
    });
    match attr_type {
        1 => Some(OnnxAttribute::Float(attr.f.unwrap_or(0.0))),
        2 => Some(OnnxAttribute::Int(attr.i.unwrap_or(0))),
        3 => {
            let s = attr
                .s
                .as_deref()
                .map(|b| String::from_utf8_lossy(b).to_string())
                .unwrap_or_default();
            Some(OnnxAttribute::String(s))
        }
        // TENSOR — used by Constant nodes to embed full tensor values
        4 => {
            let tp = attr.t.as_ref()?;
            convert_tensor_proto(tp).ok().map(OnnxAttribute::Tensor)
        }
        6 => Some(OnnxAttribute::Floats(attr.floats.clone())),
        7 => Some(OnnxAttribute::Ints(attr.ints.clone())),
        _ => None,
    }
}

/// Step A.1: coalesce consecutive NCHWc-capable `NodeAction::Conv` entries
/// into `NodeAction::NchwcChain`. Runs after the main fusion loop builds
/// `execution_plan`. For safety, chains are detected only when every
/// intermediate tensor has `use_count == 1` (no branching out of the
/// chain — a branching tensor would need to stay NHWC for the other
/// consumer).
///
/// Current scope (A.1): only LINEAR Conv chains are detected. Conv_Add
/// (residual) complicates NCHWc (skip tensor must also be in NCHWc
/// layout) — deferred to later Step A sub-phase. FusedDwPw likewise
/// deferred. This conservative scope is safe to enable incrementally.
fn fuse_nchwc_chains(
    execution_plan: &mut [NodeAction],
    nodes: &[OnnxNode],
    use_counts: &HashMap<String, usize>,
) {
    let mut i = 0;
    while i < execution_plan.len() {
        let start_node_idx = match &execution_plan[i] {
            NodeAction::Conv { node_idx, .. } => *node_idx,
            _ => {
                i += 1;
                continue;
            }
        };

        // Collect a contiguous run of Conv actions where the previous
        // Conv's output feeds this Conv's first input AND the previous
        // Conv's output has use_count == 1 (no branching).
        let mut chain_members: Vec<NchwcChainMember> = Vec::new();
        let mut end = i;
        let mut prev_node_idx = start_node_idx;
        while let NodeAction::Conv {
            node_idx,
            activation,
        } = execution_plan[end]
        {
            // First member always joins.
            if chain_members.is_empty() {
                chain_members.push(NchwcChainMember::Conv {
                    node_idx,
                    activation,
                });
                end += 1;
                prev_node_idx = node_idx;
                if end >= execution_plan.len() {
                    break;
                }
                continue;
            }
            // Subsequent members: check prev Conv's output feeds this
            // Conv AND has use_count == 1.
            let prev_output = &nodes[prev_node_idx].outputs[0];
            let this_input = nodes[node_idx].inputs.first();
            let prev_use = use_counts.get(prev_output).copied().unwrap_or(0);
            if this_input.map(|s| s.as_str()) != Some(prev_output.as_str()) || prev_use != 1 {
                break;
            }
            chain_members.push(NchwcChainMember::Conv {
                node_idx,
                activation,
            });
            prev_node_idx = node_idx;
            end += 1;
            if end >= execution_plan.len() {
                break;
            }
        }

        // Only emit a chain of ≥2 members — single-Conv chains pay the
        // reorder cost for zero gain.
        if chain_members.len() >= 2 {
            let entry_node = chain_members
                .first()
                .map(|m| match m {
                    NchwcChainMember::Conv { node_idx, .. } => *node_idx,
                })
                .expect("non-empty");
            let exit_node = chain_members
                .last()
                .map(|m| match m {
                    NchwcChainMember::Conv { node_idx, .. } => *node_idx,
                })
                .expect("non-empty");
            let entry_input = nodes[entry_node].inputs[0].clone();
            let exit_output = nodes[exit_node].outputs[0].clone();
            // Replace first member's slot with the chain; mark the rest Skip.
            execution_plan[i] = NodeAction::NchwcChain {
                members: chain_members,
                entry_input,
                exit_output,
            };
            for j in (i + 1)..end {
                execution_plan[j] = NodeAction::Skip;
            }
            i = end;
        } else {
            i += 1;
        }
    }
}

fn build_runtime_index(
    inputs: &[String],
    outputs: &[String],
    initializers: &HashMap<String, Tensor>,
    nodes: &[OnnxNode],
    khwc_weights: &HashSet<String>,
    dw_khwc_weights: &HashSet<String>,
    group_khwc_weights: &HashSet<String>,
) -> RuntimeModelIndex {
    let mut names: HashSet<&str> = HashSet::new();
    for name in inputs {
        names.insert(name.as_str());
    }
    for name in outputs {
        names.insert(name.as_str());
    }
    for name in initializers.keys() {
        names.insert(name.as_str());
    }
    for node in nodes {
        for name in &node.inputs {
            names.insert(name.as_str());
        }
        for name in &node.outputs {
            names.insert(name.as_str());
        }
    }
    let name_to_id: HashMap<String, usize> = names
        .into_iter()
        .enumerate()
        .map(|(id, name)| (name.to_string(), id))
        .collect();

    let khwc_weight_ids: HashSet<usize> = khwc_weights
        .iter()
        .filter_map(|name| name_to_id.get(name.as_str()).copied())
        .collect();
    let dw_khwc_weight_ids: HashSet<usize> = dw_khwc_weights
        .iter()
        .filter_map(|name| name_to_id.get(name.as_str()).copied())
        .collect();
    let group_khwc_weight_ids: HashSet<usize> = group_khwc_weights
        .iter()
        .filter_map(|name| name_to_id.get(name.as_str()).copied())
        .collect();

    let mut use_counts: HashMap<String, usize> = HashMap::new();
    for node in nodes {
        for inp in &node.inputs {
            if !inp.is_empty() {
                *use_counts.entry(inp.clone()).or_insert(0) += 1;
            }
        }
    }
    let mut use_counts_by_id = vec![0usize; name_to_id.len()];
    for (name, count) in &use_counts {
        if let Some(&id) = name_to_id.get(name) {
            use_counts_by_id[id] = *count;
        }
    }
    let node_kinds: Vec<NodeKind> = nodes
        .iter()
        .map(|node| NodeKind::from_op_type(&node.op_type))
        .collect();

    // Tower-parallel branch classification. For a siamese graph we want two
    // input-rooted subgraphs to run concurrently, then a merge tail. Nodes
    // are tagged 0 = reachable from first dynamic input only, 1 = second only,
    // 2 = shared/merge. If either branch ends up too small, we clear the
    // vector to signal "no parallel split".
    let node_branches: Vec<u8> = {
        let dyn_inputs: Vec<&str> = inputs
            .iter()
            .map(|s| s.as_str())
            .filter(|s| !initializers.contains_key(*s))
            .collect();
        if dyn_inputs.len() >= 2 {
            let mut tensor_branch: HashMap<&str, u8> = HashMap::new();
            tensor_branch.insert(dyn_inputs[0], 0);
            tensor_branch.insert(dyn_inputs[1], 1);
            let mut branches = Vec::with_capacity(nodes.len());
            for node in nodes {
                let mut seen = 0u8; // bitmask: bit 0 = branch 0, bit 1 = branch 1
                for inp in &node.inputs {
                    if inp.is_empty() || initializers.contains_key(inp.as_str()) {
                        continue;
                    }
                    match tensor_branch.get(inp.as_str()) {
                        Some(&0) => seen |= 1,
                        Some(&1) => seen |= 2,
                        Some(&2) => seen |= 3,
                        _ => {}
                    }
                }
                let branch = match seen {
                    0 => 2, // constant-fed node treated as merge-safe
                    1 => 0,
                    2 => 1,
                    _ => 2,
                };
                for out in &node.outputs {
                    tensor_branch.insert(out.as_str(), branch);
                }
                branches.push(branch);
            }
            let b0 = branches.iter().filter(|&&b| b == 0).count();
            let b1 = branches.iter().filter(|&&b| b == 1).count();
            // Require both branches to carry meaningful work, otherwise the
            // parallel split's overhead (env fork, rayon::join) dominates.
            if b0 >= 10 && b1 >= 10 {
                branches
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        }
    };

    // Pre-resolve input names to slot IDs for O(1) hot-path lookups.
    let node_input_ids: Vec<Vec<Option<usize>>> = nodes
        .iter()
        .map(|node| {
            node.inputs
                .iter()
                .map(|name| {
                    if name.is_empty() {
                        None
                    } else {
                        name_to_id.get(name).copied()
                    }
                })
                .collect()
        })
        .collect();

    // Session 13 R3: pre-resolve output names to slot IDs. Used by
    // `env.insert_by_id` on the hot path to skip the HashMap lookup
    // inside `resolve_id`. `node_input_ids` was already cached; this
    // extends the same optimisation to output slots.
    let node_output_ids: Vec<Vec<Option<usize>>> = nodes
        .iter()
        .map(|node| {
            node.outputs
                .iter()
                .map(|name| {
                    if name.is_empty() {
                        None
                    } else {
                        name_to_id.get(name).copied()
                    }
                })
                .collect()
        })
        .collect();

    // Pre-parse Conv attributes to avoid HashMap lookups in hot path.
    let conv_params: Vec<Option<ConvParams>> = nodes
        .iter()
        .zip(node_kinds.iter())
        .map(|(node, kind)| {
            if !matches!(
                kind,
                NodeKind::Conv | NodeKind::ConvRelu | NodeKind::ConvSilu
            ) {
                return None;
            }
            let strides = node
                .attributes
                .get("strides")
                .and_then(|a| {
                    if let OnnxAttribute::Ints(v) = a {
                        Some(v.clone())
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| vec![1, 1]);
            let pads = node
                .attributes
                .get("pads")
                .and_then(|a| {
                    if let OnnxAttribute::Ints(v) = a {
                        Some(v.clone())
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| vec![0, 0, 0, 0]);
            let group = node
                .attributes
                .get("group")
                .and_then(|a| {
                    if let OnnxAttribute::Int(v) = a {
                        Some(*v as usize)
                    } else {
                        None
                    }
                })
                .unwrap_or(1);
            let (pt, pl, pb, pr) = (
                pads[0] as usize,
                pads[1] as usize,
                pads.get(2).copied().unwrap_or(0) as usize,
                pads.get(3).copied().unwrap_or(0) as usize,
            );
            // Determine depthwise/pointwise from weight shape. Weights
            // may already be permuted to KHWC `[KH, KW, I, O]` by the
            // load-time normalization above (`khwc_weights` pass) for
            // group==1 Conv. Check both layouts and infer which applies.
            //
            // Session 11 R1 bug fix: previously we indexed shape[2]/shape[3]
            // assuming OIHW layout, but KHWC-permuted weights have
            // [KH, KW, I, O] meaning shape[2]=I, shape[3]=O. For a 1×1
            // pointwise Conv with I=16 O=96, that read `kh_w=16 kw_w=96`
            // → `is_pointwise=false` → Conv_Add fast-path never fires
            // on ALL 24 tracker residuals. Fix: dispatch by layout.
            let weight_name = node.inputs.get(1).map(|s| s.as_str()).unwrap_or("");
            let weight_shape = initializers
                .get(weight_name)
                .map(|t| t.shape().to_vec())
                .unwrap_or_default();
            let weight_is_khwc = khwc_weights.contains(weight_name);
            let weight_is_dw_khwc = dw_khwc_weights.contains(weight_name);
            let weight_is_group_khwc = group_khwc_weights.contains(weight_name);
            // Session 15 R5 (true-fuse): the loader permutes three KHWC
            // variants. DW-permuted `[KH, KW, C, dm]` and grouped
            // `[O, KH, KW, I/G]` previously fell through to the OIHW
            // branch and produced garbage, wrongly setting
            // `is_depthwise = false` for every tracker DW conv. That
            // blocked `FusedDwPw` detection. With the pure-compute
            // `conv_compute_nhwc` split the fused path now keeps the
            // DW intermediate as a local `Tensor` (no env traffic),
            // so enabling this detection no longer regresses tracker.
            let (o_ch, kh_w, kw_w) = if weight_shape.len() == 4 {
                if weight_is_dw_khwc {
                    // Depthwise KHWC: `[KH, KW, C, depth_multiplier]`.
                    let dm = weight_shape[3];
                    (weight_shape[2] * dm, weight_shape[0], weight_shape[1])
                } else if weight_is_group_khwc {
                    // Grouped KHWC: `[O, KH, KW, I/G]`.
                    (weight_shape[0], weight_shape[1], weight_shape[2])
                } else if weight_is_khwc {
                    // Regular KHWC: `[KH, KW, I, O]`.
                    (weight_shape[3], weight_shape[0], weight_shape[1])
                } else {
                    // Plain OIHW: `[O, I, KH, KW]`.
                    (weight_shape[0], weight_shape[2], weight_shape[3])
                }
            } else {
                (0, 0, 0)
            };
            let is_depthwise = group > 1 && group == o_ch;
            let is_pointwise = kh_w == 1 && kw_w == 1 && group == 1;

            Some(ConvParams {
                stride_h: strides[0] as usize,
                stride_w: strides.get(1).copied().unwrap_or(1) as usize,
                pad_top: pt,
                pad_left: pl,
                pad_bottom: pb,
                pad_right: pr,
                group,
                has_padding: pt > 0 || pl > 0 || pb > 0 || pr > 0,
                is_depthwise,
                is_pointwise,
            })
        })
        .collect();

    /// Returns `true` when the Transpose node's `perm` attribute swaps
    /// only the last two axes of a rank-3 tensor (i.e. `[0, 2, 1]`).
    /// Matches the pattern emitted by PyTorch's `.transpose(-2, -1)` on
    /// 3-D tensors — the pattern ORT folds into its
    /// `MatmulTransposeFusion` contrib op.
    fn transpose_perm_is_swap_last_two(node: &OnnxNode) -> bool {
        let perm = match node.attributes.get("perm") {
            Some(OnnxAttribute::Ints(p)) => p,
            _ => return false,
        };
        matches!(perm.as_slice(), [0, 2, 1])
    }

    // Map tensor name → producing node index. Used by the
    // `FusedTransposeMatMul` detection below to walk from a MatMul
    // left-input back to its Transpose producer in O(1).
    let producers: HashMap<String, usize> = {
        let mut m: HashMap<String, usize> = HashMap::with_capacity(nodes.len());
        for (idx, node) in nodes.iter().enumerate() {
            for out in &node.outputs {
                if !out.is_empty() {
                    m.insert(out.clone(), idx);
                }
            }
        }
        m
    };

    // Build execution plan — pre-compiled dispatch table.
    let mut execution_plan = Vec::with_capacity(nodes.len());
    let mut plan_skip = vec![false; nodes.len()];
    for (i, (kind, cp)) in node_kinds.iter().zip(conv_params.iter()).enumerate() {
        if plan_skip[i] {
            execution_plan.push(NodeAction::Skip);
            continue;
        }
        match kind {
            NodeKind::Conv | NodeKind::ConvRelu | NodeKind::ConvSilu => {
                let activation = match kind {
                    NodeKind::ConvRelu => 1,
                    NodeKind::ConvSilu => 2,
                    _ => 0,
                };

                // Try DW+PW fusion. Backs off when PW has a downstream
                // Add — the stronger `Conv_Add_fused` op saves an
                // entire output memory pass via
                // `conv2d_nhwc_pointwise_with_residual_relu`, worth
                // more than the DW+PW dispatch savings on tracker.
                let mut fused = false;
                if let Some(cp) = cp
                    && cp.is_depthwise
                {
                    // Look ahead for pointwise consuming our output
                    let dw_out = &nodes[i].outputs[0];
                    let dw_uses = use_counts.get(dw_out).copied().unwrap_or(0);
                    if dw_uses == 1 {
                        for j in (i + 1)..nodes.len() {
                            if plan_skip[j] {
                                continue;
                            }
                            let nk = node_kinds[j];
                            if matches!(
                                nk,
                                NodeKind::Conv | NodeKind::ConvRelu | NodeKind::ConvSilu
                            ) && let Some(ncp) = &conv_params[j]
                                && ncp.is_pointwise
                                && !ncp.has_padding
                                && nodes[j].inputs.first().map(|s| s.as_str())
                                    == Some(dw_out.as_str())
                            {
                                // Skip DW+PW when PW would
                                // instead form ConvAdd.
                                let pw_kind_plain = matches!(nk, NodeKind::Conv);
                                let pw_out_uses = {
                                    let pw_out = &nodes[j].outputs[0];
                                    use_counts.get(pw_out).copied().unwrap_or(0)
                                };
                                let pw_has_convadd = pw_kind_plain
                                    && pw_out_uses == 1
                                    && nodes.get(j + 1).is_some_and(|n| {
                                        node_kinds[j + 1] == NodeKind::Add
                                            && n.inputs.len() == 2
                                            && (n.inputs[0] == nodes[j].outputs[0]
                                                || n.inputs[1] == nodes[j].outputs[0])
                                    });
                                if pw_has_convadd {
                                    break;
                                }
                                let pw_act = match nk {
                                    NodeKind::ConvRelu => 1,
                                    NodeKind::ConvSilu => 2,
                                    _ => 0,
                                };
                                execution_plan.push(NodeAction::FusedDwPw {
                                    dw_idx: i,
                                    pw_idx: j,
                                    dw_activation: activation,
                                    pw_activation: pw_act,
                                });
                                plan_skip[j] = true;
                                fused = true;
                            }
                            break; // only check next non-skipped
                        }
                    }
                }
                // Try PW+DW fusion (current is PW expansion feeding into DW).
                // Mirrors the DW+PW block above but swapped: when the
                // current node is a non-DW pointwise 1×1 Conv whose output
                // is consumed exclusively by an immediately-following
                // depthwise Conv, fuse them. This targets the
                // MobileNetV2 `PW_expand → DW` opening that the
                // residual-suffix `Conv_Add_fused` leaves alone. Skips
                // when PW's activation is SiLU (not a typical
                // MobileNet pattern and the fused exec only supports
                // None/Relu epilogues for now).
                if !fused
                    && let Some(cp) = cp
                    && cp.is_pointwise
                    && !cp.has_padding
                    && activation != 2
                {
                    let pw_out = &nodes[i].outputs[0];
                    let pw_uses = use_counts.get(pw_out).copied().unwrap_or(0);
                    if pw_uses == 1 {
                        for j in (i + 1)..nodes.len() {
                            if plan_skip[j] {
                                continue;
                            }
                            let nk = node_kinds[j];
                            if matches!(
                                nk,
                                NodeKind::Conv | NodeKind::ConvRelu | NodeKind::ConvSilu
                            ) && let Some(ncp) = &conv_params[j]
                                && ncp.is_depthwise
                                && nodes[j].inputs.first().map(|s| s.as_str())
                                    == Some(pw_out.as_str())
                            {
                                let dw_act = match nk {
                                    NodeKind::ConvRelu => 1u8,
                                    NodeKind::ConvSilu => 2u8,
                                    _ => 0u8,
                                };
                                execution_plan.push(NodeAction::FusedPwDw {
                                    pw_idx: i,
                                    dw_idx: j,
                                    pw_activation: activation,
                                    dw_activation: dw_act,
                                });
                                plan_skip[j] = true;
                                fused = true;
                            }
                            break; // only check next non-skipped
                        }
                    }
                }
                if !fused {
                    // Try Conv → Add (residual), optionally followed by Relu.
                    let conv_out = &nodes[i].outputs[0];
                    let conv_out_uses = use_counts.get(conv_out).copied().unwrap_or(0);
                    let mut conv_add_emitted = false;
                    if conv_out_uses == 1
                        && activation == 0
                        && let Some(add_node) = nodes.get(i + 1)
                        && node_kinds[i + 1] == NodeKind::Add
                        && add_node.inputs.len() == 2
                        && !plan_skip[i + 1]
                        && (add_node.inputs[0] == *conv_out || add_node.inputs[1] == *conv_out)
                    {
                        let skip_input_idx: u8 = if add_node.inputs[0] == *conv_out {
                            1
                        } else {
                            0
                        };
                        let add_out = &add_node.outputs[0];
                        let (post_activation, relu_idx_field) =
                            if let Some(relu_node) = nodes.get(i + 2) {
                                if node_kinds[i + 2] == NodeKind::Relu
                                    && !plan_skip[i + 2]
                                    && relu_node.inputs.len() == 1
                                    && relu_node.inputs[0] == *add_out
                                    && use_counts.get(add_out).copied().unwrap_or(0) == 1
                                {
                                    (1u8, (i + 2) as u32)
                                } else {
                                    (0u8, 0u32)
                                }
                            } else {
                                (0u8, 0u32)
                            };
                        execution_plan.push(NodeAction::ConvAdd {
                            conv_idx: i,
                            add_idx: i + 1,
                            skip_input_idx,
                            post_activation,
                            relu_idx: relu_idx_field,
                        });
                        plan_skip[i + 1] = true;
                        if post_activation == 1 {
                            plan_skip[relu_idx_field as usize] = true;
                        }
                        conv_add_emitted = true;
                    }
                    if !conv_add_emitted {
                        execution_plan.push(NodeAction::Conv {
                            node_idx: i,
                            activation,
                        });
                    }
                }
            }
            NodeKind::MatMul => {
                // Try Transpose+MatMul fusion: when the MatMul's left
                // input (index 0) is the output of a `Transpose` node
                // whose `perm` swaps the last two axes of a rank-3
                // tensor (i.e. `[0,2,1]`) AND every consumer of that
                // Transpose is a MatMul that can absorb it, we elide
                // the Transpose entirely and dispatch a `transA=1`
                // GEMM. Mirrors ORT's `MatmulTransposeFusion`. Weaker
                // fusion (Transpose has other consumers) still pays
                // the materialization cost once, so we don't fuse.
                let left_input = nodes[i].inputs.first().map(|s| s.as_str()).unwrap_or("");
                let mut emitted = false;
                if !left_input.is_empty()
                    && let Some(&t_idx) = producers.get(left_input)
                    && t_idx < nodes.len()
                    && node_kinds[t_idx] == NodeKind::Transpose
                    && !plan_skip[t_idx]
                    && transpose_perm_is_swap_last_two(&nodes[t_idx])
                {
                    execution_plan.push(NodeAction::FusedTransposeMatMul {
                        transpose_idx: t_idx,
                        matmul_idx: i,
                        cleanup_transpose: false,
                    });
                    emitted = true;
                }
                if !emitted {
                    execution_plan.push(NodeAction::Generic {
                        node_idx: i,
                        kind: *kind,
                    });
                }
            }
            _ => {
                execution_plan.push(NodeAction::Generic {
                    node_idx: i,
                    kind: *kind,
                });
            }
        }
    }

    // Post-pass: elide Transpose nodes whose every consumer is a
    // `FusedTransposeMatMul` that absorbed them. Counts the number of
    // fused actions pointing at each transpose and compares to the
    // transpose output's total graph-use count (input edges + model
    // output membership). When every consumer was absorbed, the
    // original Transpose does no useful work and becomes `Skip`.
    let model_outputs: HashSet<&str> = outputs.iter().map(|s| s.as_str()).collect();
    let mut fused_refs: Vec<usize> = vec![0; nodes.len()];
    // Plan position of the last `FusedTransposeMatMul` referencing each
    // transpose idx. Used to mark exactly one variant as the cleanup
    // owner so the pre-transpose tensor stays in `env` until every
    // consumer has read it.
    let mut last_fused_pos: HashMap<usize, usize> = HashMap::new();
    for (pos, action) in execution_plan.iter().enumerate() {
        if let NodeAction::FusedTransposeMatMul { transpose_idx, .. } = action {
            fused_refs[*transpose_idx] += 1;
            last_fused_pos.insert(*transpose_idx, pos);
        }
    }
    for (&t_idx, &pos) in &last_fused_pos {
        if let NodeAction::FusedTransposeMatMul {
            transpose_idx,
            cleanup_transpose,
            ..
        } = &mut execution_plan[pos]
        {
            debug_assert_eq!(*transpose_idx, t_idx);
            *cleanup_transpose = true;
        }
    }
    for t_idx in 0..nodes.len() {
        if fused_refs[t_idx] == 0 {
            continue;
        }
        let t_out = match nodes[t_idx].outputs.first() {
            Some(s) if !s.is_empty() => s,
            _ => continue,
        };
        let edge_uses = use_counts.get(t_out).copied().unwrap_or(0);
        let is_model_output = model_outputs.contains(t_out.as_str());
        let consumer_total = edge_uses + usize::from(is_model_output);
        if fused_refs[t_idx] >= consumer_total && consumer_total > 0 {
            execution_plan[t_idx] = NodeAction::Skip;
        }
    }

    // after Conv/Add/DwPw fusions have been applied, coalesces runs of
    // NCHWc-capable actions (currently linear Conv chains) into
    // `NodeAction::NchwcChain` entries. Original sub-actions become
    // `Skip`. Enabled only when `YSCV_NCHWC_CHAIN=1` — default off
    // until runner dispatch lands and chain dispatch is actually
    // faster than per-op.
    if std::env::var("YSCV_NCHWC_CHAIN").as_deref() == Ok("1") {
        fuse_nchwc_chains(&mut execution_plan, nodes, &use_counts);
    }

    // Load-time weight pre-packing. For every pointwise Conv (KH=KW=1,
    // group=1) whose weight is already laid out KHWC, pre-pack the B-matrix
    // in blocked-GEMM format and cache it by weight-tensor name. The execution
    // plan looks it up per call and hands the shared `Arc<PackedB>` to the
    // GEMM layer, skipping the runtime fingerprint cache and `pack_b_panel`.
    //
    // We can't prepack non-KHWC weights here because the runtime path re-
    // permutes them to KHWC on first use (which would make the prepack stale).
    // For the models we care about, `_with_khwc_once` has already normalized
    // all pointwise Conv weights at model-load, so this check is typically
    // true. Non-pointwise Convs go through 3×3 direct / im2col paths that
    // don't consume packed B — prepack isn't useful there.
    let mut prepacked_weights: HashMap<String, std::sync::Arc<yscv_kernels::PackedB>> =
        HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        let Some(cp) = conv_params[i].as_ref() else {
            continue;
        };
        if !cp.is_pointwise {
            continue;
        }
        let Some(w_name) = node.inputs.get(1) else {
            continue;
        };
        if !initializers.contains_key(w_name) || !khwc_weights.contains(w_name) {
            continue;
        }
        if prepacked_weights.contains_key(w_name) {
            continue;
        }
        let Some(weight) = initializers.get(w_name) else {
            continue;
        };
        // KHWC pointwise weight shape: [KH=1, KW=1, IC, OC]. k = IC, n = OC.
        let shape = weight.shape();
        if shape.len() != 4 || shape[0] != 1 || shape[1] != 1 {
            continue;
        }
        let k = shape[2];
        let n = shape[3];
        let packed = yscv_kernels::pack_b_for_session(weight.data(), k, n);
        prepacked_weights.insert(w_name.clone(), packed);
    }
    let mut prepacked_weights_by_id: Vec<Option<std::sync::Arc<yscv_kernels::PackedB>>> =
        vec![None; name_to_id.len()];
    for (name, packed) in &prepacked_weights {
        if let Some(&id) = name_to_id.get(name) {
            prepacked_weights_by_id[id] = Some(packed.clone());
        }
    }

    RuntimeModelIndex {
        name_to_id,
        khwc_weight_ids,
        dw_khwc_weight_ids,
        group_khwc_weight_ids,
        use_counts,
        use_counts_by_id,
        node_kinds,
        node_branches,
        node_input_ids,
        node_output_ids,
        conv_params,
        execution_plan,
        prepacked_weights,
        prepacked_weights_by_id,
    }
}

#[cfg(test)]
mod nchwc_chain_tests {
    use super::*;

    fn conv_node(name: &str, input: &str, output: &str) -> OnnxNode {
        OnnxNode {
            op_type: "Conv".into(),
            name: name.into(),
            inputs: vec![input.into(), format!("{name}_weight")],
            outputs: vec![output.into()],
            attributes: HashMap::new(),
        }
    }

    #[test]
    fn fuse_nchwc_chains_coalesces_linear_conv_run() {
        let nodes = vec![
            conv_node("c0", "input", "t1"),
            conv_node("c1", "t1", "t2"),
            conv_node("c2", "t2", "out"),
        ];
        let mut use_counts: HashMap<String, usize> = HashMap::new();
        use_counts.insert("input".into(), 1);
        use_counts.insert("t1".into(), 1);
        use_counts.insert("t2".into(), 1);
        let mut plan = vec![
            NodeAction::Conv {
                node_idx: 0,
                activation: 0,
            },
            NodeAction::Conv {
                node_idx: 1,
                activation: 1,
            },
            NodeAction::Conv {
                node_idx: 2,
                activation: 0,
            },
        ];
        fuse_nchwc_chains(&mut plan, &nodes, &use_counts);
        // First slot becomes the chain; remaining are Skip.
        match &plan[0] {
            NodeAction::NchwcChain {
                members,
                entry_input,
                exit_output,
            } => {
                assert_eq!(members.len(), 3);
                assert_eq!(entry_input, "input");
                assert_eq!(exit_output, "out");
            }
            other => panic!("expected NchwcChain, got {:?}", other),
        }
        assert!(matches!(plan[1], NodeAction::Skip));
        assert!(matches!(plan[2], NodeAction::Skip));
    }

    #[test]
    fn fuse_nchwc_chains_breaks_on_branch() {
        // Middle output used twice → chain must stop there.
        let nodes = vec![
            conv_node("c0", "input", "t1"),
            conv_node("c1", "t1", "t2"),
            conv_node("c2", "t2", "out"),
        ];
        let mut use_counts: HashMap<String, usize> = HashMap::new();
        use_counts.insert("input".into(), 1);
        use_counts.insert("t1".into(), 1);
        use_counts.insert("t2".into(), 2); // branches — chain cannot cross.
        let mut plan = vec![
            NodeAction::Conv {
                node_idx: 0,
                activation: 0,
            },
            NodeAction::Conv {
                node_idx: 1,
                activation: 0,
            },
            NodeAction::Conv {
                node_idx: 2,
                activation: 0,
            },
        ];
        fuse_nchwc_chains(&mut plan, &nodes, &use_counts);
        // Chain contains first two only; third remains as-is.
        match &plan[0] {
            NodeAction::NchwcChain { members, .. } => assert_eq!(members.len(), 2),
            other => panic!("expected NchwcChain, got {:?}", other),
        }
        assert!(matches!(plan[1], NodeAction::Skip));
        assert!(matches!(plan[2], NodeAction::Conv { node_idx: 2, .. }));
    }

    #[test]
    fn fuse_nchwc_chains_skips_singleton() {
        // Single Conv should not be wrapped (no gain, pays reorder cost).
        let nodes = vec![conv_node("c0", "input", "out")];
        let mut use_counts: HashMap<String, usize> = HashMap::new();
        use_counts.insert("input".into(), 1);
        let mut plan = vec![NodeAction::Conv {
            node_idx: 0,
            activation: 1,
        }];
        fuse_nchwc_chains(&mut plan, &nodes, &use_counts);
        assert!(matches!(plan[0], NodeAction::Conv { node_idx: 0, .. }));
    }
}
