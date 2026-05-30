//! Per-node op dispatch: attribute/tensor accessors and the
//! op-type match that routes each node to its kernel family.

use super::*;

#[inline]
pub(crate) fn get_tensor<'a>(
    env: &'a TensorEnv,
    node_name: &str,
    input_name: &str,
) -> Result<&'a Tensor, OnnxError> {
    env.get(input_name).ok_or_else(|| OnnxError::MissingInput {
        node: node_name.to_string(),
        input: input_name.to_string(),
    })
}

#[inline]
pub(crate) fn get_attr_ints(node: &OnnxNode, name: &str) -> Option<Vec<i64>> {
    match node.attributes.get(name) {
        Some(OnnxAttribute::Ints(v)) => Some(v.clone()),
        _ => None,
    }
}

#[inline]
pub(crate) fn get_attr_int(node: &OnnxNode, name: &str) -> Option<i64> {
    match node.attributes.get(name) {
        Some(OnnxAttribute::Int(v)) => Some(*v),
        _ => None,
    }
}

#[inline]
pub(crate) fn get_attr_float(node: &OnnxNode, name: &str) -> Option<f32> {
    match node.attributes.get(name) {
        Some(OnnxAttribute::Float(v)) => Some(*v),
        _ => None,
    }
}

#[inline]
pub(crate) fn get_attr_string(node: &OnnxNode, name: &str) -> Option<String> {
    match node.attributes.get(name) {
        Some(OnnxAttribute::String(v)) => Some(v.clone()),
        _ => None,
    }
}

/// Converts inputs in the environment to f32 before executing a node, then converts
/// outputs back to the original dtype. Ops that handle dtypes themselves (Cast, Shape,
/// Identity, quantization ops) are exempt.
pub(crate) fn execute_node_kind(
    node: &OnnxNode,
    env: &mut TensorEnv,
    kind: NodeKind,
) -> Result<(), OnnxError> {
    // Ops that should NOT have automatic dtype conversion
    let dtype_exempt = matches!(
        node.op_type.as_str(),
        "Cast"
            | "Shape"
            | "Identity"
            | "Constant"
            | "ConstantOfShape"
            | "QuantizeLinear"
            | "DequantizeLinear"
            | "DynamicQuantizeLinear"
            | "QLinearConv"
            | "QLinearMatMul"
            | "MatMulInteger"
            | "ConvInteger"
    );

    // Detect original dtype from first input (if any) and convert inputs to f32
    let orig_dtype = if !dtype_exempt && !node.inputs.is_empty() {
        let first_dt = node
            .inputs
            .iter()
            .filter_map(|name| env.get(name))
            .map(|t| t.dtype())
            .find(|&dt| dt != DType::F32);

        if let Some(dt) = first_dt {
            // Convert all non-f32 inputs to f32 in-place
            for input_name in &node.inputs {
                if let Some(tensor) = env.get(input_name)
                    && tensor.dtype() != DType::F32
                {
                    let converted = tensor.to_dtype(DType::F32);
                    env.insert(input_name.clone(), converted);
                }
            }
            Some(dt)
        } else {
            None
        }
    } else {
        None
    };

    // Execute the actual op
    execute_node_inner_kind(node, env, kind)?;

    // Convert outputs back to original dtype if needed
    if let Some(dt) = orig_dtype {
        for output_name in &node.outputs {
            if let Some(tensor) = env.get(output_name)
                && tensor.dtype() == DType::F32
            {
                let converted = tensor.to_dtype(dt);
                env.insert(output_name.clone(), converted);
            }
        }
    }

    Ok(())
}

#[inline]
fn execute_node_inner_kind(
    node: &OnnxNode,
    env: &mut TensorEnv,
    kind: NodeKind,
) -> Result<(), OnnxError> {
    execute_node_inner_kind_fast(node, env, kind, None)
}

#[inline]
fn execute_node_inner_kind_fast(
    node: &OnnxNode,
    env: &mut TensorEnv,
    kind: NodeKind,
    conv_params: Option<&crate::loader::ConvParams>,
) -> Result<(), OnnxError> {
    match kind {
        NodeKind::Conv => {
            exec_conv_with_params(node, env, yscv_kernels::Activation::None, conv_params, None)
        }
        NodeKind::ConvRelu => {
            exec_conv_with_params(node, env, yscv_kernels::Activation::Relu, conv_params, None)
        }
        NodeKind::ConvSilu => {
            exec_conv_with_params(node, env, yscv_kernels::Activation::Silu, conv_params, None)
        }
        NodeKind::Relu => exec_relu(node, env),
        NodeKind::BatchNormalization => exec_batch_norm(node, env),
        NodeKind::Gemm => exec_gemm(node, env),
        NodeKind::Add => exec_add(node, env),
        NodeKind::MatMul => exec_matmul(node, env),
        NodeKind::Mul => exec_mul(node, env),
        NodeKind::Sigmoid => exec_sigmoid(node, env),
        NodeKind::Reshape => exec_reshape(node, env),
        NodeKind::Constant => exec_constant(node, env),
        NodeKind::Concat => exec_concat(node, env),
        NodeKind::Transpose => exec_transpose(node, env),
        NodeKind::Other => execute_node_inner_slow(node, env),
    }
}

fn execute_node_inner_slow(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    match node.op_type.as_str() {
        "MaxPool" => exec_max_pool(node, env),
        "AveragePool" => exec_avg_pool(node, env),
        "GlobalAveragePool" => exec_global_avg_pool(node, env),
        "Flatten" => exec_flatten(node, env),
        "Sub" => exec_sub(node, env),
        "Softmax" => exec_softmax(node, env),
        "Transpose" => exec_transpose(node, env),
        "Concat" => exec_concat(node, env),
        "Unsqueeze" => exec_unsqueeze(node, env),
        "Squeeze" => exec_squeeze(node, env),
        "Clip" => exec_clip(node, env),
        "Shape" => exec_shape(node, env),
        "Gather" => exec_gather(node, env),
        "Constant" => exec_constant(node, env),
        "Dropout" => exec_dropout(node, env),
        "Pad" => exec_pad(node, env),
        "Pow" => exec_pow(node, env),
        "Sqrt" => exec_sqrt(node, env),
        "Exp" => exec_exp(node, env),
        "Log" => exec_log(node, env),
        "Neg" => exec_neg(node, env),
        "Abs" => exec_abs(node, env),
        "Reciprocal" => exec_reciprocal(node, env),
        "Tanh" => exec_tanh(node, env),
        "Floor" => exec_floor(node, env),
        "Ceil" => exec_ceil(node, env),
        "Equal" => exec_cmp(node, env, 0),
        "Greater" => exec_cmp(node, env, 1),
        "Less" => exec_cmp(node, env, 2),
        "Where" => exec_where(node, env),
        "Trilu" => exec_trilu(node, env),
        "ReduceMean" => exec_reduce_mean(node, env),
        "ReduceSum" => exec_reduce_sum(node, env),
        "Split" => exec_split(node, env),
        "Slice" => exec_slice(node, env),
        "Expand" => exec_expand(node, env),
        "Tile" => exec_tile(node, env),
        "Cast" => exec_cast(node, env),
        "Div" => exec_div(node, env),
        "Min" => exec_min_max(node, env, false),
        "Max" => exec_min_max(node, env, true),
        "ReduceMax" => exec_reduce_max(node, env),
        "ConvTranspose" => exec_conv_transpose(node, env),
        "DeformConv" => exec_deform_conv(node, env),
        "Resize" => exec_resize(node, env),
        "LeakyRelu" => exec_leaky_relu(node, env),
        "Elu" => exec_elu(node, env),
        "ReduceMin" => exec_reduce_min(node, env),
        "ReduceProd" => exec_reduce_prod(node, env),
        "Identity" => exec_identity(node, env),
        "QuantizeLinear" => exec_quantize_linear(node, env),
        "DequantizeLinear" => exec_dequantize_linear(node, env),
        "Gelu" => exec_gelu(node, env),
        "Erf" => exec_erf(node, env),
        "HardSigmoid" => exec_hard_sigmoid(node, env),
        "InstanceNormalization" => exec_instance_norm(node, env),
        "LpNormalization" => exec_lp_norm(node, env),
        "Upsample" => exec_resize(node, env),
        "Selu" => exec_selu(node, env),
        "Celu" => exec_celu(node, env),
        "ThresholdedRelu" => exec_thresholded_relu(node, env),
        "Hardmax" => exec_hardmax(node, env),
        "OneHot" => exec_onehot(node, env),
        "Range" => exec_range(node, env),
        "NonZero" => exec_nonzero(node, env),
        "LayerNormalization" => exec_layer_norm(node, env),
        "GatherElements" => exec_gather_elements(node, env),
        "ScatterElements" => exec_scatter_elements(node, env),
        "Einsum" => exec_einsum(node, env),
        "ReduceL2" => exec_reduce_l2(node, env),
        "ReduceL1" => exec_reduce_l1(node, env),
        "CumSum" => exec_cumsum(node, env),
        "ArgMax" => exec_argmax(node, env),
        "ArgMin" => exec_argmin(node, env),
        "TopK" => exec_topk(node, env),
        "ScatterND" => exec_scatter_nd(node, env),
        "GatherND" => exec_gather_nd(node, env),
        "DepthToSpace" => exec_depth_to_space(node, env),
        "SpaceToDepth" => exec_space_to_depth(node, env),
        "GridSample" => exec_grid_sample(node, env),
        "RoiAlign" => exec_roi_align(node, env),
        "Compress" => exec_compress(node, env),
        "QLinearConv" => exec_qlinear_conv(node, env),
        "QLinearMatMul" => exec_qlinear_matmul(node, env),
        "MatMulInteger" => exec_matmul_integer(node, env),
        "ConvInteger" => exec_conv_integer(node, env),
        "DynamicQuantizeLinear" => exec_dynamic_quantize_linear(node, env),
        "Not" => exec_not(node, env),
        "And" => exec_logical_bin(node, env, 0),
        "Or" => exec_logical_bin(node, env, 1),
        "Xor" => exec_logical_bin(node, env, 2),
        "Sin" => exec_tensor_op(node, env, |t| t.sin()),
        "Cos" => exec_tensor_op(node, env, |t| t.cos()),
        "Tan" => exec_unary(node, env, |v| v.tan()),
        "Asin" => exec_unary(node, env, |v| v.asin()),
        "Acos" => exec_unary(node, env, |v| v.acos()),
        "Atan" => exec_unary(node, env, |v| v.atan()),
        "Sinh" => exec_unary(node, env, |v| v.sinh()),
        "Cosh" => exec_unary(node, env, |v| v.cosh()),
        "Asinh" => exec_unary(node, env, |v| v.asinh()),
        "Acosh" => exec_unary(node, env, |v| v.acosh()),
        "Atanh" => exec_unary(node, env, |v| v.atanh()),
        "Round" => exec_unary(node, env, |v| v.round()),
        "Sign" => exec_unary(node, env, |v| v.signum()),
        "IsNaN" => exec_unary(node, env, |v| if v.is_nan() { 1.0 } else { 0.0 }),
        "IsInf" => exec_unary(node, env, |v| if v.is_infinite() { 1.0 } else { 0.0 }),
        "Mod" => exec_mod(node, env),
        "GreaterOrEqual" => exec_cmp(node, env, 3),
        "LessOrEqual" => exec_cmp(node, env, 4),
        "BitShift" => exec_bitshift(node, env),
        "Mean" => exec_variadic_mean(node, env),
        "Sum" => exec_variadic_sum(node, env),
        "ConstantOfShape" => exec_constant_of_shape(node, env),
        "LRN" => exec_lrn(node, env),
        "Softplus" => exec_unary(node, env, |v| (1.0 + v.exp()).ln()),
        "Softsign" => exec_unary(node, env, |v| v / (1.0 + v.abs())),
        "HardSwish" => exec_unary(node, env, |v| v * ((v + 3.0).clamp(0.0, 6.0) / 6.0)),
        "Mish" => exec_unary(node, env, |v| v * (1.0 + v.exp()).ln().tanh()),
        "NonMaxSuppression" => exec_nms(node, env),
        "GroupQueryAttention" => exec_grouped_query_attention(node, env),
        "Conv_Relu" => exec_conv(node, env, yscv_kernels::Activation::Relu),
        "BatchNormalization_Relu" => {
            exec_batch_norm(node, env)?;
            exec_relu_inplace(node, env)
        }
        other => Err(OnnxError::UnsupportedOpType {
            op_type: other.to_string(),
        }),
    }
}
