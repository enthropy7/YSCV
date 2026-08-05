use crate::attr::Attr;
use rustc_hash::FxHashMap;

use thiserror::Error;
use yscv_tensor::Tensor;

use crate::loader::{OnnxAttribute, OnnxModel, OnnxNode};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dim {
    Known(usize),
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorShape {
    pub dims: Vec<Dim>,
}

impl TensorShape {
    pub fn known(dims: Vec<usize>) -> Self {
        Self {
            dims: dims.into_iter().map(Dim::Known).collect(),
        }
    }

    pub fn unknown_rank(rank: usize) -> Self {
        Self {
            dims: vec![Dim::Unknown; rank],
        }
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn dim(&self, idx: usize) -> Option<usize> {
        match self.dims.get(idx) {
            Some(Dim::Known(v)) => Some(*v),
            _ => None,
        }
    }

    pub fn num_elements(&self) -> Option<u64> {
        let mut n = 1u64;
        for dim in &self.dims {
            match dim {
                Dim::Known(v) => n = n.checked_mul(*v as u64)?,
                Dim::Unknown => return None,
            }
        }
        Some(n)
    }

    pub fn as_known_dims(&self) -> Option<Vec<usize>> {
        self.dims
            .iter()
            .map(|d| match d {
                Dim::Known(v) => Some(*v),
                Dim::Unknown => None,
            })
            .collect()
    }
}

pub type ShapeMap = FxHashMap<String, TensorShape>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeDiagnostic {
    pub node_index: usize,
    pub node_name: String,
    pub op_type: String,
    pub error: ShapeError,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ShapeError {
    #[error("unsupported shape rule for op {op_type}")]
    UnsupportedOp { op_type: String },
    #[error("{op_type} missing required input {index}")]
    MissingInput { op_type: String, index: usize },
    #[error("missing input shape for value {name}")]
    MissingInputShape { name: String },
    #[error("missing initializer {name}")]
    MissingInitializer { name: String },
    #[error("{op_type} requires rank {expected}, got rank {actual}")]
    RankMismatch {
        op_type: String,
        expected: usize,
        actual: usize,
    },
    #[error("{op_type} requires rank at least {min}, got rank {actual}")]
    RankTooSmall {
        op_type: String,
        min: usize,
        actual: usize,
    },
    #[error("{op_type} supports {expected}")]
    UnsupportedRank {
        op_type: String,
        expected: &'static str,
    },
    #[error("incompatible broadcast dimensions {left} and {right}")]
    BroadcastIncompatible { left: usize, right: usize },
    #[error("axis {axis} out of range for rank {rank}")]
    AxisOutOfRange { axis: i64, rank: usize },
    #[error("{op_type} missing required attribute {name}")]
    MissingAttribute { op_type: String, name: &'static str },
    #[error("Constant without tensor value attribute")]
    ConstantWithoutTensor,
    #[error("{op_type} input {name} must be constant")]
    NonConstantInput { op_type: String, name: String },
    #[error("Reshape target contains unsupported dimension {dim}")]
    InvalidReshapeDim { dim: i64 },
    #[error("{op_type} input ranks differ")]
    InputRanksDiffer { op_type: String },
    #[error("Transpose perm rank {perm_rank} does not match input rank {input_rank}")]
    TransposePermRankMismatch { perm_rank: usize, input_rank: usize },
}

#[derive(Debug, Clone)]
pub struct ShapeInference {
    pub shapes: ShapeMap,
    pub diagnostics: Vec<ShapeDiagnostic>,
}

pub fn infer_shapes(model: &OnnxModel, input_shapes: &ShapeMap) -> ShapeInference {
    let mut shapes = input_shapes.clone();
    let mut diagnostics = Vec::new();

    for (name, tensor) in &model.initializers {
        shapes.insert(name.clone(), TensorShape::known(tensor.shape().to_vec()));
    }

    for (idx, node) in model.nodes.iter().enumerate() {
        match infer_node(model, &shapes, node) {
            Ok(outputs) => {
                for (name, shape) in node.outputs.iter().zip(outputs) {
                    if !name.is_empty() {
                        shapes.insert(name.clone(), shape);
                    }
                }
            }
            Err(error) => diagnostics.push(ShapeDiagnostic {
                node_index: idx,
                node_name: node.name.clone(),
                op_type: node.op_type.clone(),
                error,
            }),
        }
    }

    ShapeInference {
        shapes,
        diagnostics,
    }
}

pub fn infer_shapes_from_tensors(
    model: &OnnxModel,
    inputs: &FxHashMap<String, Tensor>,
) -> ShapeInference {
    let input_shapes: ShapeMap = inputs
        .iter()
        .map(|(name, tensor)| (name.clone(), TensorShape::known(tensor.shape().to_vec())))
        .collect();
    infer_shapes(model, &input_shapes)
}

fn infer_node(
    model: &OnnxModel,
    shapes: &ShapeMap,
    node: &OnnxNode,
) -> Result<Vec<TensorShape>, ShapeError> {
    match node.op_type.as_str() {
        "Constant" => infer_constant(node),
        "Identity" | "Dropout" => unary_same_shape(shapes, node),

        "Relu" | "Sigmoid" | "Tanh" | "Clip" | "BatchNormalization" | "BatchNormalization_Relu" => {
            unary_same_shape(shapes, node)
        }
        "Conv" | "Conv_Relu" | "Conv_SiLU" => infer_conv(model, shapes, node),
        "Add" | "Sub" | "Mul" | "Div" | "Pow" => infer_broadcast(shapes, node),
        "Concat" => infer_concat(shapes, node),
        "Transpose" => infer_transpose(shapes, node),
        "Reshape" => infer_reshape(model, shapes, node),
        "Flatten" => infer_flatten(shapes, node),
        "Squeeze" => infer_squeeze(model, shapes, node),
        "Unsqueeze" => infer_unsqueeze(model, shapes, node),
        "MatMul" => infer_matmul(shapes, node),
        "Gemm" => infer_gemm(model, shapes, node),
        "MaxPool" | "AveragePool" => infer_pool(shapes, node),
        "GlobalAveragePool" => infer_global_pool(shapes, node),
        _ => Err(ShapeError::UnsupportedOp {
            op_type: node.op_type.clone(),
        }),
    }
}

fn unary_same_shape(shapes: &ShapeMap, node: &OnnxNode) -> Result<Vec<TensorShape>, ShapeError> {
    let input = node
        .inputs
        .first()
        .ok_or_else(|| ShapeError::MissingInput {
            op_type: node.op_type.clone(),
            index: 0,
        })?;
    let shape = shapes
        .get(input)
        .cloned()
        .ok_or_else(|| ShapeError::MissingInputShape {
            name: input.clone(),
        })?;
    Ok(vec![shape; node.outputs.len().max(1)])
}

fn infer_constant(node: &OnnxNode) -> Result<Vec<TensorShape>, ShapeError> {
    match node.attributes.get(&Attr::Value) {
        Some(OnnxAttribute::Tensor(t)) => Ok(vec![TensorShape::known(t.shape().to_vec())]),
        _ => Err(ShapeError::ConstantWithoutTensor),
    }
}

fn infer_conv(
    model: &OnnxModel,
    shapes: &ShapeMap,
    node: &OnnxNode,
) -> Result<Vec<TensorShape>, ShapeError> {
    let input_name = node
        .inputs
        .first()
        .ok_or_else(|| ShapeError::MissingInput {
            op_type: node.op_type.clone(),
            index: 0,
        })?;
    let weight_name = node.inputs.get(1).ok_or_else(|| ShapeError::MissingInput {
        op_type: node.op_type.clone(),
        index: 1,
    })?;
    let input = shapes
        .get(input_name)
        .ok_or_else(|| ShapeError::MissingInputShape {
            name: input_name.clone(),
        })?;
    let weight =
        model
            .initializers
            .get(weight_name)
            .ok_or_else(|| ShapeError::MissingInitializer {
                name: weight_name.clone(),
            })?;
    let w_shape = weight.shape();
    if input.rank() != 4 || w_shape.len() != 4 {
        return Err(ShapeError::UnsupportedRank {
            op_type: node.op_type.clone(),
            expected: "rank-4 NCHW inputs and rank-4 weights",
        });
    }

    let n = input.dims[0];
    let ih = input.dims[2];
    let iw = input.dims[3];
    let oc = if model.khwc_weights.contains(weight_name) {
        w_shape[3]
    } else if model.dw_khwc_weights.contains(weight_name) {
        w_shape[2]
    } else {
        w_shape[0]
    };
    let (kh, kw) = if model.khwc_weights.contains(weight_name)
        || model.dw_khwc_weights.contains(weight_name)
    {
        (w_shape[0], w_shape[1])
    } else {
        (w_shape[2], w_shape[3])
    };

    let strides = ints_attr(node, Attr::Strides).unwrap_or_else(|| vec![1, 1]);
    let dilations = ints_attr(node, Attr::Dilations).unwrap_or_else(|| vec![1, 1]);
    let pads = ints_attr(node, Attr::Pads).unwrap_or_else(|| vec![0, 0, 0, 0]);
    let sh = strides.first().copied().unwrap_or(1).max(1) as usize;
    let sw = strides.get(1).copied().unwrap_or(1).max(1) as usize;
    let dh = dilations.first().copied().unwrap_or(1).max(1) as usize;
    let dw = dilations.get(1).copied().unwrap_or(1).max(1) as usize;
    let pt = pads.first().copied().unwrap_or(0).max(0) as usize;
    let pl = pads.get(1).copied().unwrap_or(0).max(0) as usize;
    let pb = pads.get(2).copied().unwrap_or(0).max(0) as usize;
    let pr = pads.get(3).copied().unwrap_or(0).max(0) as usize;

    let oh = conv_output_dim(ih, kh, sh, dh, pt, pb);
    let ow = conv_output_dim(iw, kw, sw, dw, pl, pr);
    Ok(vec![TensorShape {
        dims: vec![n, Dim::Known(oc), oh, ow],
    }])
}

fn conv_output_dim(
    input: Dim,
    kernel: usize,
    stride: usize,
    dilation: usize,
    pad_begin: usize,
    pad_end: usize,
) -> Dim {
    match input {
        Dim::Known(v) => {
            let effective = dilation
                .saturating_mul(kernel.saturating_sub(1))
                .saturating_add(1);
            let padded = v.saturating_add(pad_begin).saturating_add(pad_end);
            if padded < effective {
                Dim::Known(0)
            } else {
                Dim::Known((padded - effective) / stride + 1)
            }
        }
        Dim::Unknown => Dim::Unknown,
    }
}

fn infer_broadcast(shapes: &ShapeMap, node: &OnnxNode) -> Result<Vec<TensorShape>, ShapeError> {
    if node.inputs.len() < 2 {
        return Err(ShapeError::MissingInput {
            op_type: node.op_type.clone(),
            index: 1,
        });
    }
    let a = shapes
        .get(&node.inputs[0])
        .ok_or_else(|| ShapeError::MissingInputShape {
            name: node.inputs[0].clone(),
        })?;
    let b = shapes
        .get(&node.inputs[1])
        .ok_or_else(|| ShapeError::MissingInputShape {
            name: node.inputs[1].clone(),
        })?;
    Ok(vec![broadcast_shapes(a, b)?])
}

fn broadcast_shapes(a: &TensorShape, b: &TensorShape) -> Result<TensorShape, ShapeError> {
    let rank = a.rank().max(b.rank());
    let mut dims = Vec::with_capacity(rank);
    for i in 0..rank {
        let ad = dim_from_right(a, i);
        let bd = dim_from_right(b, i);
        let out = match (ad, bd) {
            (Dim::Known(1), d) | (d, Dim::Known(1)) => d,
            (Dim::Known(x), Dim::Known(y)) if x == y => Dim::Known(x),
            (Dim::Unknown, _) | (_, Dim::Unknown) => Dim::Unknown,
            (Dim::Known(x), Dim::Known(y)) => {
                return Err(ShapeError::BroadcastIncompatible { left: x, right: y });
            }
        };
        dims.push(out);
    }
    dims.reverse();
    Ok(TensorShape { dims })
}

fn dim_from_right(shape: &TensorShape, idx_from_right: usize) -> Dim {
    if idx_from_right >= shape.rank() {
        Dim::Known(1)
    } else {
        shape.dims[shape.rank() - 1 - idx_from_right]
    }
}

fn infer_concat(shapes: &ShapeMap, node: &OnnxNode) -> Result<Vec<TensorShape>, ShapeError> {
    let first_name = node
        .inputs
        .first()
        .ok_or_else(|| ShapeError::MissingInput {
            op_type: node.op_type.clone(),
            index: 0,
        })?;
    let first = shapes
        .get(first_name)
        .cloned()
        .ok_or_else(|| ShapeError::MissingInputShape {
            name: first_name.clone(),
        })?;
    let rank = first.rank();
    let axis = normalize_axis(int_attr(node, Attr::Axis).unwrap_or(0), rank)?;
    let mut out = first;
    let mut axis_sum = 0usize;
    let mut axis_known = true;
    for name in &node.inputs {
        let shape = shapes
            .get(name)
            .ok_or_else(|| ShapeError::MissingInputShape { name: name.clone() })?;
        if shape.rank() != rank {
            return Err(ShapeError::InputRanksDiffer {
                op_type: node.op_type.clone(),
            });
        }
        for d in 0..rank {
            if d == axis {
                if let Some(v) = shape.dim(d) {
                    axis_sum = axis_sum.saturating_add(v);
                } else {
                    axis_known = false;
                }
            } else if out.dims[d] != shape.dims[d] {
                out.dims[d] = Dim::Unknown;
            }
        }
    }
    out.dims[axis] = if axis_known {
        Dim::Known(axis_sum)
    } else {
        Dim::Unknown
    };
    Ok(vec![out])
}

fn infer_transpose(shapes: &ShapeMap, node: &OnnxNode) -> Result<Vec<TensorShape>, ShapeError> {
    let input_name = node
        .inputs
        .first()
        .ok_or_else(|| ShapeError::MissingInput {
            op_type: node.op_type.clone(),
            index: 0,
        })?;
    let input = shapes
        .get(input_name)
        .ok_or_else(|| ShapeError::MissingInputShape {
            name: input_name.clone(),
        })?;
    let perm =
        ints_attr(node, Attr::Perm).unwrap_or_else(|| (0..input.rank() as i64).rev().collect());
    if perm.len() != input.rank() {
        return Err(ShapeError::TransposePermRankMismatch {
            perm_rank: perm.len(),
            input_rank: input.rank(),
        });
    }
    let mut dims = Vec::with_capacity(input.rank());
    for p in perm {
        let idx = normalize_axis(p, input.rank())?;
        dims.push(input.dims[idx]);
    }
    Ok(vec![TensorShape { dims }])
}

fn infer_reshape(
    model: &OnnxModel,
    shapes: &ShapeMap,
    node: &OnnxNode,
) -> Result<Vec<TensorShape>, ShapeError> {
    let input_name = node
        .inputs
        .first()
        .ok_or_else(|| ShapeError::MissingInput {
            op_type: node.op_type.clone(),
            index: 0,
        })?;
    let input = shapes
        .get(input_name)
        .ok_or_else(|| ShapeError::MissingInputShape {
            name: input_name.clone(),
        })?;
    let shape_name = node.inputs.get(1).ok_or_else(|| ShapeError::MissingInput {
        op_type: node.op_type.clone(),
        index: 1,
    })?;
    let target =
        model
            .initializers
            .get(shape_name)
            .ok_or_else(|| ShapeError::NonConstantInput {
                op_type: node.op_type.clone(),
                name: shape_name.clone(),
            })?;
    let raw: Vec<i64> = target.data().iter().map(|v| *v as i64).collect();
    let input_numel = input.num_elements();
    let mut dims = Vec::with_capacity(raw.len());
    let mut known_product = 1u64;
    let mut minus_one = None;
    for (idx, &d) in raw.iter().enumerate() {
        if d == 0 {
            let copied = input.dims.get(idx).copied().unwrap_or(Dim::Unknown);
            if let Dim::Known(v) = copied {
                known_product = known_product.saturating_mul(v as u64);
            }
            dims.push(copied);
        } else if d == -1 {
            minus_one = Some(idx);
            dims.push(Dim::Unknown);
        } else if d > 0 {
            known_product = known_product.saturating_mul(d as u64);
            dims.push(Dim::Known(d as usize));
        } else {
            return Err(ShapeError::InvalidReshapeDim { dim: d });
        }
    }
    if let (Some(total), Some(idx)) = (input_numel, minus_one)
        && known_product != 0
        && total % known_product == 0
    {
        dims[idx] = Dim::Known((total / known_product) as usize);
    }
    Ok(vec![TensorShape { dims }])
}

fn infer_flatten(shapes: &ShapeMap, node: &OnnxNode) -> Result<Vec<TensorShape>, ShapeError> {
    let input_name = node
        .inputs
        .first()
        .ok_or_else(|| ShapeError::MissingInput {
            op_type: node.op_type.clone(),
            index: 0,
        })?;
    let input = shapes
        .get(input_name)
        .ok_or_else(|| ShapeError::MissingInputShape {
            name: input_name.clone(),
        })?;
    let axis = normalize_axis(int_attr(node, Attr::Axis).unwrap_or(1), input.rank())?;
    let left = product_dims(&input.dims[..axis]);
    let right = product_dims(&input.dims[axis..]);
    Ok(vec![TensorShape {
        dims: vec![left, right],
    }])
}

fn infer_squeeze(
    model: &OnnxModel,
    shapes: &ShapeMap,
    node: &OnnxNode,
) -> Result<Vec<TensorShape>, ShapeError> {
    let input_name = node
        .inputs
        .first()
        .ok_or_else(|| ShapeError::MissingInput {
            op_type: node.op_type.clone(),
            index: 0,
        })?;
    let input = shapes
        .get(input_name)
        .ok_or_else(|| ShapeError::MissingInputShape {
            name: input_name.clone(),
        })?;
    let axes = node_axes(model, node)?;
    let dims = if axes.is_empty() {
        input
            .dims
            .iter()
            .copied()
            .filter(|d| *d != Dim::Known(1))
            .collect()
    } else {
        let axes: Vec<usize> = axes
            .into_iter()
            .map(|a| normalize_axis(a, input.rank()))
            .collect::<Result<_, _>>()?;
        input
            .dims
            .iter()
            .enumerate()
            .filter_map(|(idx, d)| if axes.contains(&idx) { None } else { Some(*d) })
            .collect()
    };
    Ok(vec![TensorShape { dims }])
}

fn infer_unsqueeze(
    model: &OnnxModel,
    shapes: &ShapeMap,
    node: &OnnxNode,
) -> Result<Vec<TensorShape>, ShapeError> {
    let input_name = node
        .inputs
        .first()
        .ok_or_else(|| ShapeError::MissingInput {
            op_type: node.op_type.clone(),
            index: 0,
        })?;
    let input = shapes
        .get(input_name)
        .ok_or_else(|| ShapeError::MissingInputShape {
            name: input_name.clone(),
        })?;
    let raw_axes = node_axes(model, node)?;
    let out_rank = input.rank().saturating_add(raw_axes.len());
    let mut axes: Vec<usize> = raw_axes
        .into_iter()
        .map(|a| normalize_axis(a, out_rank))
        .collect::<Result<_, _>>()?;
    axes.sort_unstable();
    let mut input_idx = 0usize;
    let mut dims = Vec::with_capacity(out_rank);
    for out_idx in 0..out_rank {
        if axes.contains(&out_idx) {
            dims.push(Dim::Known(1));
        } else {
            dims.push(input.dims[input_idx]);
            input_idx += 1;
        }
    }
    Ok(vec![TensorShape { dims }])
}

fn infer_matmul(shapes: &ShapeMap, node: &OnnxNode) -> Result<Vec<TensorShape>, ShapeError> {
    if node.inputs.len() < 2 {
        return Err(ShapeError::MissingInput {
            op_type: node.op_type.clone(),
            index: 1,
        });
    }
    let a = shapes
        .get(&node.inputs[0])
        .ok_or_else(|| ShapeError::MissingInputShape {
            name: node.inputs[0].clone(),
        })?;
    let b = shapes
        .get(&node.inputs[1])
        .ok_or_else(|| ShapeError::MissingInputShape {
            name: node.inputs[1].clone(),
        })?;
    if a.rank() < 2 || b.rank() < 2 {
        return Err(ShapeError::RankTooSmall {
            op_type: node.op_type.clone(),
            min: 2,
            actual: a.rank().min(b.rank()),
        });
    }
    let a_batch = TensorShape {
        dims: a.dims[..a.rank() - 2].to_vec(),
    };
    let b_batch = TensorShape {
        dims: b.dims[..b.rank() - 2].to_vec(),
    };
    let batch = broadcast_shapes(&a_batch, &b_batch)?.dims;
    let mut dims = batch;
    dims.push(a.dims[a.rank() - 2]);
    dims.push(b.dims[b.rank() - 1]);
    Ok(vec![TensorShape { dims }])
}

fn infer_gemm(
    model: &OnnxModel,
    shapes: &ShapeMap,
    node: &OnnxNode,
) -> Result<Vec<TensorShape>, ShapeError> {
    if node.inputs.len() < 2 {
        return Err(ShapeError::MissingInput {
            op_type: node.op_type.clone(),
            index: 1,
        });
    }
    let a = shapes
        .get(&node.inputs[0])
        .ok_or_else(|| ShapeError::MissingInputShape {
            name: node.inputs[0].clone(),
        })?;
    let b = shapes
        .get(&node.inputs[1])
        .ok_or_else(|| ShapeError::MissingInputShape {
            name: node.inputs[1].clone(),
        })?;
    if a.rank() != 2 || b.rank() != 2 {
        return Err(ShapeError::UnsupportedRank {
            op_type: node.op_type.clone(),
            expected: "rank-2 inputs",
        });
    }
    let trans_a = int_attr(node, Attr::TransA).unwrap_or(0) != 0;
    let trans_b = int_attr(node, Attr::TransB).unwrap_or(0) != 0;
    let m = if trans_a { a.dims[1] } else { a.dims[0] };
    let n = if trans_b { b.dims[0] } else { b.dims[1] };
    let out = TensorShape { dims: vec![m, n] };
    if let Some(c_name) = node.inputs.get(2)
        && !c_name.is_empty()
        && let Some(c) = model.initializers.get(c_name)
    {
        let c_shape = TensorShape::known(c.shape().to_vec());
        let _ = broadcast_shapes(&out, &c_shape)?;
    }
    Ok(vec![out])
}

fn infer_pool(shapes: &ShapeMap, node: &OnnxNode) -> Result<Vec<TensorShape>, ShapeError> {
    let input_name = node
        .inputs
        .first()
        .ok_or_else(|| ShapeError::MissingInput {
            op_type: node.op_type.clone(),
            index: 0,
        })?;
    let input = shapes
        .get(input_name)
        .ok_or_else(|| ShapeError::MissingInputShape {
            name: input_name.clone(),
        })?;
    if input.rank() != 4 {
        return Err(ShapeError::UnsupportedRank {
            op_type: node.op_type.clone(),
            expected: "rank-4 NCHW inputs",
        });
    }
    let kernels = ints_attr(node, Attr::KernelShape).ok_or(ShapeError::MissingAttribute {
        op_type: node.op_type.clone(),
        name: "kernel_shape",
    })?;
    let strides = ints_attr(node, Attr::Strides).unwrap_or_else(|| vec![1, 1]);
    let pads = ints_attr(node, Attr::Pads).unwrap_or_else(|| vec![0, 0, 0, 0]);
    let kh = kernels.first().copied().unwrap_or(1).max(1) as usize;
    let kw = kernels.get(1).copied().unwrap_or(1).max(1) as usize;
    let sh = strides.first().copied().unwrap_or(1).max(1) as usize;
    let sw = strides.get(1).copied().unwrap_or(1).max(1) as usize;
    let pt = pads.first().copied().unwrap_or(0).max(0) as usize;
    let pl = pads.get(1).copied().unwrap_or(0).max(0) as usize;
    let pb = pads.get(2).copied().unwrap_or(0).max(0) as usize;
    let pr = pads.get(3).copied().unwrap_or(0).max(0) as usize;
    Ok(vec![TensorShape {
        dims: vec![
            input.dims[0],
            input.dims[1],
            conv_output_dim(input.dims[2], kh, sh, 1, pt, pb),
            conv_output_dim(input.dims[3], kw, sw, 1, pl, pr),
        ],
    }])
}

fn infer_global_pool(shapes: &ShapeMap, node: &OnnxNode) -> Result<Vec<TensorShape>, ShapeError> {
    let input_name = node
        .inputs
        .first()
        .ok_or_else(|| ShapeError::MissingInput {
            op_type: node.op_type.clone(),
            index: 0,
        })?;
    let input = shapes
        .get(input_name)
        .ok_or_else(|| ShapeError::MissingInputShape {
            name: input_name.clone(),
        })?;
    if input.rank() != 4 {
        return Err(ShapeError::UnsupportedRank {
            op_type: node.op_type.clone(),
            expected: "rank-4 NCHW inputs",
        });
    }
    Ok(vec![TensorShape {
        dims: vec![input.dims[0], input.dims[1], Dim::Known(1), Dim::Known(1)],
    }])
}

fn product_dims(dims: &[Dim]) -> Dim {
    let mut product = 1usize;
    for dim in dims {
        match dim {
            Dim::Known(v) => product = product.saturating_mul(*v),
            Dim::Unknown => return Dim::Unknown,
        }
    }
    Dim::Known(product)
}

fn node_axes(model: &OnnxModel, node: &OnnxNode) -> Result<Vec<i64>, ShapeError> {
    if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
        let t = model.initializers.get(&node.inputs[1]).ok_or_else(|| {
            ShapeError::NonConstantInput {
                op_type: node.op_type.clone(),
                name: node.inputs[1].clone(),
            }
        })?;
        return Ok(t.data().iter().map(|v| *v as i64).collect());
    }
    if let Some(OnnxAttribute::Ints(axes)) = node.attributes.get(&Attr::Axes) {
        return Ok(axes.clone());
    }
    Ok(Vec::new())
}

fn normalize_axis(axis: i64, rank: usize) -> Result<usize, ShapeError> {
    let normalized = if axis < 0 { rank as i64 + axis } else { axis };
    if normalized < 0 || normalized >= rank as i64 {
        return Err(ShapeError::AxisOutOfRange { axis, rank });
    }
    Ok(normalized as usize)
}

fn int_attr(node: &OnnxNode, name: Attr) -> Option<i64> {
    match node.attributes.get(&name) {
        Some(OnnxAttribute::Int(v)) => Some(*v),
        _ => None,
    }
}

fn ints_attr(node: &OnnxNode, name: Attr) -> Option<Vec<i64>> {
    match node.attributes.get(&name) {
        Some(OnnxAttribute::Ints(v)) => Some(v.clone()),
        _ => None,
    }
}
