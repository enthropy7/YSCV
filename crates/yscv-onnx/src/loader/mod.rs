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
    /// Pre-resolved slot IDs for each node's outputs.
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
    /// Load-time packed RHS matrices for symmetric-int8 QLinearConv /
    /// QLinearMatMul / MatMulInteger fast paths. Keyed by the ONNX
    /// weight tensor input name; shared by every inference.
    pub(crate) prepacked_i8_weights: HashMap<String, std::sync::Arc<yscv_kernels::PackedI8B>>,
    /// QLinear depthwise 3x3/5x5 weights packed once as KHWC i8 so runtime can
    /// call the NHWC int8 depthwise kernel without per-inference weight repack.
    pub(crate) prepacked_i8_depthwise: HashMap<String, std::sync::Arc<Vec<i8>>>,
    /// Pre-packed PW-reduce weights and biases for `FusedPwDwPwReduce` actions.
    /// Keyed by `pw_reduce_idx` (the original PW reduce node index). Built at
    /// load time after fusion detection so the runner can call
    /// `fused_pw_expand_dw_pw_reduce_3x3` without per-call weight transposing.
    pub(crate) prepacked_fused_pw_dw_pw_reduce:
        HashMap<usize, std::sync::Arc<FusedPwDwPwReduceWeights>>,
    /// NHWC-passthrough eligibility: output tensor names of
    /// `Reshape` nodes whose single consumer is a `Transpose(perm=[0,2,1])`
    /// followed by `MatMul` (i.e. a `FusedTransposeMatMul` chain). For
    /// these reshapes the runtime can skip the `ensure_nchw` permute and
    /// keep the NHWC tag, since `exec_fused_transpose_matmul` handles
    /// NHWC-flagged inputs via the non-trans matmul path. Built once at
    /// load time after `FusedTransposeMatMul` detection.
    pub(crate) reshape_nhwc_passthrough_safe: HashSet<String>,
}

/// Residual-Add metadata for the `FusedPwDwPwReduce` action.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FusedPwDwPwReduceResidual {
    pub(crate) add_idx: usize,
    pub(crate) residual_skip_input: u8,
    pub(crate) post_activation: u8,
    pub(crate) relu_idx: u32,
}

/// Prepacked PW-reduce weights for `FusedPwDwPwReduce` execution.
#[derive(Debug)]
pub(crate) struct FusedPwDwPwReduceWeights {
    /// Row-major `[c_exp, c_out_padded]` PW-reduce weight, tail zeros.
    pub(crate) weight_packed: yscv_tensor::AlignedVec<f32>,
    /// `[c_out_padded]` PW-reduce bias, tail zeros. `None` when the original
    /// PW reduce node had no bias.
    pub(crate) bias_padded: Option<yscv_tensor::AlignedVec<f32>>,
    pub(crate) c_out: usize,
    pub(crate) c_out_padded: usize,
    pub(crate) c_exp: usize,
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
    /// Streaming MobileNet-V2 inverted bottleneck: PW_expand (1×1) → DW 3×3
    /// → PW_reduce (1×1) executed by `fused_pw_expand_dw_pw_reduce_3x3`.
    /// The c_exp intermediate never materialises beyond one DW output row
    /// (out_w × c_exp × 4 ≤ 24 KB, L1-resident). PW reduce weight and bias
    /// are prepacked at load time into `prepacked_fused_pw_dw_pw_reduce`
    /// keyed by `pw_reduce_idx`. Activations supported: None / Relu.
    FusedPwDwPwReduce {
        pw_expand_idx: usize,
        dw_idx: usize,
        pw_reduce_idx: usize,
        pw_expand_activation: u8,
        dw_activation: u8,
        pw_reduce_activation: u8,
        /// DW kernel size: 3 or 5. Dispatches to either
        /// `fused_pw_expand_dw_pw_reduce_3x3` or `_5x5`.
        dw_kernel_size: u8,
        /// When `Some`, the PW reduce output is summed with a residual
        /// (the inverted-bottleneck skip connection): `add_idx` is the
        /// `Add` node index, `residual_skip_input` is which of the two
        /// `Add` inputs is the residual (0 or 1). The Add is folded into
        /// the streaming kernel's row-write loop (zero-cost residual).
        /// Optional post-Relu after the Add: `post_activation` = 1.
        residual: Option<FusedPwDwPwReduceResidual>,
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
    /// QLinear boundary cleanup:
    /// `DequantizeLinear(q, s, 0) -> [Relu] -> QuantizeLinear(_, s, 0)`.
    /// yscv stores quantized activations as f32 int8 values, so when the
    /// quant params match we can keep the tensor quantized and optionally
    /// apply Relu directly in that domain.
    QuantizedQdq {
        dequant_idx: usize,
        relu_idx: Option<usize>,
        quant_idx: usize,
    },
    /// Fused INT8 quant-domain PW → DW chain:
    /// `QLinearConv(pw 1×1) → DequantizeLinear → [Relu] → QuantizeLinear →
    /// QLinearConv(dw kxk)` executed as a single kernel call. No fp32
    /// fallback inside the chain — PW dot stays in VNNI/SDOT/widen, the
    /// QDQ boundary becomes an i8 requant + clamp, DW reads those i8 bytes
    /// directly. Bitwise-identical to the unfused per-op execution because
    /// the boundary fold gate forces `pw_y_zp == dq_zp == q_zp == 0` and
    /// `dq_scale == q_scale`, the same predicate `QuantizedQdq` already
    /// uses to fold the boundary.
    QuantizedPwDw {
        pw_idx: usize,
        dq_idx: usize,
        relu_idx: Option<usize>,
        q_idx: usize,
        dw_idx: usize,
        has_relu: bool,
    },
    /// Fused INT8 quant-domain DW → PW chain — the closing pair of an
    /// inverted bottleneck:
    /// `QLinearConv(dw kxk) → DequantizeLinear → [Relu] → QuantizeLinear →
    /// QLinearConv(pw 1×1)` executed as a single kernel call. Same
    /// boundary-fold gate as `QuantizedPwDw` (zero zero-points everywhere
    /// except the chain's output `y_zp`, which is PW's here, and matching
    /// `dq_scale == q_scale`); the kernel reads NHWC i8 input directly,
    /// requantises in-register at the boundary, and the PW reduction
    /// consumes the i8 immediately. Bitwise-identical to the unfused
    /// per-op chain.
    QuantizedDwPw {
        dw_idx: usize,
        dq_idx: usize,
        relu_idx: Option<usize>,
        q_idx: usize,
        pw_idx: usize,
        has_relu: bool,
    },
    /// Missed INT8 pair whose dequantized midpoint also feeds a residual
    /// branch. Executes `QLinearConv -> DequantizeLinear -> [Relu] ->
    /// QuantizeLinear -> QLinearConv` as one plan action while preserving the
    /// dequantized tensor for the side consumer.
    QuantizedForkPair {
        first_idx: usize,
        dq_idx: usize,
        relu_idx: Option<usize>,
        q_idx: usize,
        second_idx: usize,
        first_kind: u8, // 0=PW, 1=DW
        has_relu: bool,
    },
    /// Residual suffix after a quantized depthwise op:
    /// `QLinearConv -> DQ -> Relu -> Conv -> Add -> QuantizeLinear`.
    /// The inner Conv/Add stay fp32 by graph contract, but the quantized
    /// boundary is one compiled plan action so the runner no longer reports
    /// this window as an unfused quant-chain fallback.
    QuantizedResidualChain {
        qconv_idx: usize,
        dq_idx: usize,
        relu_idx: usize,
        conv_idx: usize,
        add_idx: usize,
        q_idx: usize,
        qconv_kind: u8, // 0=PW, 1=DW
    },
    /// Boundary-only quantized Conv whose dequantized output fans out into
    /// non-adjacent consumers. Used for branch split points such as the neck
    /// downsample heads where no adjacent pair can own the DQ output.
    QuantizedConvDq {
        qconv_idx: usize,
        dq_idx: usize,
        qconv_kind: u8, // 0=PW, 1=DW
    },
    /// Generic op — falls through to full dispatch.
    Generic { node_idx: usize, kind: NodeKind },
    /// Skipped (fused into previous action).
    Skip,
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
    /// MatMul/Gemm weights packed to INT4 with per-group fp32 scales for
    /// the LLM decode hot path. Keyed by the original initializer name;
    /// the original `initializers` entry is removed when a weight is
    /// packed so dispatch routes through `packed_int4_gemv_dispatch`.
    pub(crate) packed_int4_weights: HashMap<String, crate::quantize::PackedInt4Weight>,
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

    let matmul_rhs_inputs: HashSet<String> = nodes
        .iter()
        .filter(|node| node.op_type == "MatMul")
        .filter_map(|node| node.inputs.get(1).cloned())
        .collect();
    let graph_outputs: HashSet<String> = outputs.iter().cloned().collect();
    let mut folded_nodes = Vec::with_capacity(nodes.len());
    for node in nodes {
        let can_fold_const_transpose = node.op_type == "Transpose"
            && node.inputs.len() == 1
            && node.outputs.len() == 1
            && matmul_rhs_inputs.contains(&node.outputs[0])
            && !graph_outputs.contains(&node.outputs[0]);
        if can_fold_const_transpose && let Some(input) = initializers.get(&node.inputs[0]) {
            let axes: Vec<usize> = match node.attributes.get("perm") {
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
    // Skipped on Metal and wgpu GPU builds: those backends' CPU fallback +
    // accelerator dispatch paths read weights in the original ONNX OIHW
    // layout. Keeping the export layout there means the same loader can
    // feed both CPU and accelerator runners; the accelerator handles its
    // own pre-permute internally if any.
    #[cfg(not(any(feature = "metal-backend", feature = "gpu")))]
    let mut dw_khwc_weights = HashSet::new();
    #[cfg(any(feature = "metal-backend", feature = "gpu"))]
    let dw_khwc_weights = HashSet::new();
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
    #[cfg(not(any(feature = "metal-backend", feature = "gpu")))]
    let mut group_khwc_weights = HashSet::new();
    #[cfg(any(feature = "metal-backend", feature = "gpu"))]
    let group_khwc_weights = HashSet::new();
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
        packed_int4_weights: HashMap::new(),
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
mod runtime_index;
use convert::*;
use runtime_index::*;
