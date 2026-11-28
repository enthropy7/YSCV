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
pub(crate) fn get_attr_ints(node: &OnnxNode, name: Attr) -> Option<Vec<i64>> {
    match node.attributes.get(&name) {
        Some(OnnxAttribute::Ints(v)) => Some(v.clone()),
        _ => None,
    }
}

#[inline]
pub(crate) fn get_attr_int(node: &OnnxNode, name: Attr) -> Option<i64> {
    match node.attributes.get(&name) {
        Some(OnnxAttribute::Int(v)) => Some(*v),
        _ => None,
    }
}

#[inline]
pub(crate) fn get_attr_float(node: &OnnxNode, name: Attr) -> Option<f32> {
    match node.attributes.get(&name) {
        Some(OnnxAttribute::Float(v)) => Some(*v),
        _ => None,
    }
}

#[inline]
pub(crate) fn get_attr_string(node: &OnnxNode, name: Attr) -> Option<String> {
    match node.attributes.get(&name) {
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
        &node.op_type,
        crate::OnnxOp::Cast
            | crate::OnnxOp::Shape
            | crate::OnnxOp::Identity
            | crate::OnnxOp::Constant
            | crate::OnnxOp::ConstantOfShape
            | crate::OnnxOp::QuantizeLinear
            | crate::OnnxOp::DequantizeLinear
            | crate::OnnxOp::DynamicQuantizeLinear
            | crate::OnnxOp::QLinearConv
            | crate::OnnxOp::QLinearMatMul
            | crate::OnnxOp::MatMulInteger
            | crate::OnnxOp::ConvInteger
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
    conv_params: Option<&crate::plan::ConvParams>,
) -> Result<(), OnnxError> {
    match kind {
        NodeKind::Conv => exec_conv_with_params(
            node,
            env,
            yscv_kernels::Activation::None,
            conv_params,
            None,
            None,
        ),
        NodeKind::ConvRelu => exec_conv_with_params(
            node,
            env,
            yscv_kernels::Activation::Relu,
            conv_params,
            None,
            None,
        ),
        NodeKind::ConvSilu => exec_conv_with_params(
            node,
            env,
            yscv_kernels::Activation::Silu,
            conv_params,
            None,
            None,
        ),
        NodeKind::ConvHardSwish => {
            exec_conv_with_params(
                node,
                env,
                yscv_kernels::Activation::None,
                conv_params,
                None,
                None,
            )?;
            if let Some(mut t) = env.remove(&node.outputs[0]) {
                yscv_kernels::hardswish_slice_inplace(t.data_mut());
                env.insert(node.outputs[0].clone(), t);
            }
            Ok(())
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
    match &node.op_type {
        crate::OnnxOp::MaxPool => exec_max_pool(node, env),
        crate::OnnxOp::AveragePool => exec_avg_pool(node, env),
        crate::OnnxOp::GlobalAveragePool => exec_global_avg_pool(node, env),
        crate::OnnxOp::Flatten => exec_flatten(node, env),
        crate::OnnxOp::Sub => exec_sub(node, env),
        crate::OnnxOp::Softmax => exec_softmax(node, env),
        crate::OnnxOp::Transpose => exec_transpose(node, env),
        crate::OnnxOp::Concat => exec_concat(node, env),
        crate::OnnxOp::Unsqueeze => exec_unsqueeze(node, env),
        crate::OnnxOp::Squeeze => exec_squeeze(node, env),
        crate::OnnxOp::Clip => exec_clip(node, env),
        crate::OnnxOp::Shape => exec_shape(node, env),
        crate::OnnxOp::Gather => exec_gather(node, env),
        crate::OnnxOp::Constant => exec_constant(node, env),
        crate::OnnxOp::Dropout => exec_dropout(node, env),
        crate::OnnxOp::Pad => exec_pad(node, env),
        crate::OnnxOp::Pow => exec_pow(node, env),
        crate::OnnxOp::Sqrt => exec_sqrt(node, env),
        crate::OnnxOp::Exp => exec_exp(node, env),
        crate::OnnxOp::Log => exec_log(node, env),
        crate::OnnxOp::Neg => exec_neg(node, env),
        crate::OnnxOp::Abs => exec_abs(node, env),
        crate::OnnxOp::Reciprocal => exec_reciprocal(node, env),
        crate::OnnxOp::Tanh => exec_tanh(node, env),
        crate::OnnxOp::Floor => exec_floor(node, env),
        crate::OnnxOp::Ceil => exec_ceil(node, env),
        crate::OnnxOp::Equal => exec_cmp(node, env, 0),
        crate::OnnxOp::Greater => exec_cmp(node, env, 1),
        crate::OnnxOp::Less => exec_cmp(node, env, 2),
        crate::OnnxOp::Where => exec_where(node, env),
        crate::OnnxOp::Trilu => exec_trilu(node, env),
        crate::OnnxOp::ReduceMean => exec_reduce_mean(node, env),
        crate::OnnxOp::ReduceSum => exec_reduce_sum(node, env),
        crate::OnnxOp::Split => exec_split(node, env),
        crate::OnnxOp::Slice => exec_slice(node, env),
        crate::OnnxOp::Expand => exec_expand(node, env),
        crate::OnnxOp::Tile => exec_tile(node, env),
        crate::OnnxOp::Cast => exec_cast(node, env),
        crate::OnnxOp::Div => exec_div(node, env),
        crate::OnnxOp::Min => exec_min_max(node, env, false),
        crate::OnnxOp::Max => exec_min_max(node, env, true),
        crate::OnnxOp::ReduceMax => exec_reduce_max(node, env),
        crate::OnnxOp::ConvTranspose => exec_conv_transpose(node, env),
        crate::OnnxOp::DeformConv => exec_deform_conv(node, env),
        crate::OnnxOp::Resize => exec_resize(node, env),
        crate::OnnxOp::LeakyRelu => exec_leaky_relu(node, env),
        crate::OnnxOp::Elu => exec_elu(node, env),
        crate::OnnxOp::ReduceMin => exec_reduce_min(node, env),
        crate::OnnxOp::ReduceProd => exec_reduce_prod(node, env),
        crate::OnnxOp::Identity => exec_identity(node, env),
        crate::OnnxOp::QuantizeLinear => exec_quantize_linear(node, env),
        crate::OnnxOp::DequantizeLinear => exec_dequantize_linear(node, env),
        crate::OnnxOp::Gelu => exec_gelu(node, env),
        crate::OnnxOp::Erf => exec_erf(node, env),
        crate::OnnxOp::HardSigmoid => exec_hard_sigmoid(node, env),
        crate::OnnxOp::InstanceNormalization => exec_instance_norm(node, env),
        crate::OnnxOp::LpNormalization => exec_lp_norm(node, env),
        crate::OnnxOp::Upsample => exec_resize(node, env),
        crate::OnnxOp::Selu => exec_selu(node, env),
        crate::OnnxOp::Celu => exec_celu(node, env),
        crate::OnnxOp::ThresholdedRelu => exec_thresholded_relu(node, env),
        crate::OnnxOp::Hardmax => exec_hardmax(node, env),
        crate::OnnxOp::OneHot => exec_onehot(node, env),
        crate::OnnxOp::Range => exec_range(node, env),
        crate::OnnxOp::NonZero => exec_nonzero(node, env),
        crate::OnnxOp::LayerNormalization => exec_layer_norm(node, env),
        crate::OnnxOp::GatherElements => exec_gather_elements(node, env),
        crate::OnnxOp::ScatterElements => exec_scatter_elements(node, env),
        crate::OnnxOp::Einsum => exec_einsum(node, env),
        crate::OnnxOp::ReduceL2 => exec_reduce_l2(node, env),
        crate::OnnxOp::ReduceL1 => exec_reduce_l1(node, env),
        crate::OnnxOp::CumSum => exec_cumsum(node, env),
        crate::OnnxOp::ArgMax => exec_argmax(node, env),
        crate::OnnxOp::ArgMin => exec_argmin(node, env),
        crate::OnnxOp::TopK => exec_topk(node, env),
        crate::OnnxOp::ScatterND => exec_scatter_nd(node, env),
        crate::OnnxOp::GatherND => exec_gather_nd(node, env),
        crate::OnnxOp::DepthToSpace => exec_depth_to_space(node, env),
        crate::OnnxOp::SpaceToDepth => exec_space_to_depth(node, env),
        crate::OnnxOp::GridSample => exec_grid_sample(node, env),
        crate::OnnxOp::RoiAlign => exec_roi_align(node, env),
        crate::OnnxOp::Compress => exec_compress(node, env),
        crate::OnnxOp::QLinearConv => exec_qlinear_conv(node, env),
        crate::OnnxOp::QLinearMatMul => exec_qlinear_matmul(node, env),
        crate::OnnxOp::MatMulInteger => exec_matmul_integer(node, env),
        crate::OnnxOp::ConvInteger => exec_conv_integer(node, env),
        crate::OnnxOp::DynamicQuantizeLinear => exec_dynamic_quantize_linear(node, env),
        crate::OnnxOp::Not => exec_not(node, env),
        crate::OnnxOp::And => exec_logical_bin(node, env, 0),
        crate::OnnxOp::Or => exec_logical_bin(node, env, 1),
        crate::OnnxOp::Xor => exec_logical_bin(node, env, 2),
        crate::OnnxOp::Sin => exec_tensor_op(node, env, |t| t.sin()),
        crate::OnnxOp::Cos => exec_tensor_op(node, env, |t| t.cos()),
        crate::OnnxOp::Tan => exec_unary(node, env, |v| v.tan()),
        crate::OnnxOp::Asin => exec_unary(node, env, |v| v.asin()),
        crate::OnnxOp::Acos => exec_unary(node, env, |v| v.acos()),
        crate::OnnxOp::Atan => exec_unary(node, env, |v| v.atan()),
        crate::OnnxOp::Sinh => exec_unary(node, env, |v| v.sinh()),
        crate::OnnxOp::Cosh => exec_unary(node, env, |v| v.cosh()),
        crate::OnnxOp::Asinh => exec_unary(node, env, |v| v.asinh()),
        crate::OnnxOp::Acosh => exec_unary(node, env, |v| v.acosh()),
        crate::OnnxOp::Atanh => exec_unary(node, env, |v| v.atanh()),
        crate::OnnxOp::Round => exec_unary(node, env, |v| v.round()),
        crate::OnnxOp::Sign => exec_unary(node, env, |v| v.signum()),
        crate::OnnxOp::IsNaN => exec_unary(node, env, |v| if v.is_nan() { 1.0 } else { 0.0 }),
        crate::OnnxOp::IsInf => exec_unary(node, env, |v| if v.is_infinite() { 1.0 } else { 0.0 }),
        crate::OnnxOp::Mod => exec_mod(node, env),
        crate::OnnxOp::GreaterOrEqual => exec_cmp(node, env, 3),
        crate::OnnxOp::LessOrEqual => exec_cmp(node, env, 4),
        crate::OnnxOp::BitShift => exec_bitshift(node, env),
        crate::OnnxOp::Mean => exec_variadic_mean(node, env),
        crate::OnnxOp::Sum => exec_variadic_sum(node, env),
        crate::OnnxOp::ConstantOfShape => exec_constant_of_shape(node, env),
        crate::OnnxOp::LRN => exec_lrn(node, env),
        crate::OnnxOp::Softplus => exec_unary(node, env, |v| (1.0 + v.exp()).ln()),
        crate::OnnxOp::Softsign => exec_unary(node, env, |v| v / (1.0 + v.abs())),
        crate::OnnxOp::HardSwish => exec_hardswish(node, env),
        crate::OnnxOp::Mish => exec_unary(node, env, |v| v * (1.0 + v.exp()).ln().tanh()),
        crate::OnnxOp::NonMaxSuppression => exec_nms(node, env),
        crate::OnnxOp::GroupQueryAttention => exec_grouped_query_attention(node, env),
        crate::OnnxOp::ConvRelu => exec_conv(node, env, yscv_kernels::Activation::Relu),
        crate::OnnxOp::BatchNormRelu => {
            exec_batch_norm(node, env)?;
            exec_relu_inplace(node, env)
        }
        other => Err(OnnxError::UnsupportedOpType {
            op_type: other.to_string(),
        }),
    }
}
