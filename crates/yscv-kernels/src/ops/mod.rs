pub(crate) mod attention;
#[cfg(all(target_os = "macos", feature = "blas"))]
pub mod bnns_conv;
mod config;
mod conv;
mod deformable_conv;
mod elementwise;
mod embedding;
mod first_layer_3x3;
mod fused_pw_dw_3x3;
pub mod int4_matmul;
pub mod int8_depthwise;
pub mod int8_fused_dw_pw_3x3;
pub mod int8_fused_pw_dw_3x3;
pub mod int8_matmul;
mod int8_requant;
mod layout;
mod matmul;
mod nchwc_dw3x3;
mod nchwc_ops;
mod norm;
mod pool;
pub mod quantize;
pub mod rope;
pub mod simd;

// Public API re-exports
pub use config::{
    BinaryKind, DEFAULT_ELEMENTWISE_MIN_PARALLEL_ELEMENTS,
    DEFAULT_MATMUL_MIN_PARALLEL_OUTPUT_ELEMENTS, DEFAULT_MATMUL_MIN_PARALLEL_SHARED_DIM,
    ParallelElementwiseConfig, ParallelMatmulConfig,
};
pub use elementwise::{
    add_inplace, add_relu_inplace, add_with_config, exp, exp_with_config, gelu, mish,
    mul_with_config, relu, relu_inplace, relu_out, relu_with_config, sigmoid, sigmoid_with_config,
    silu, silu_inplace, sub_with_config, tanh_act, tanh_act_with_config,
};
#[cfg(target_arch = "aarch64")]
pub use matmul::hgemm_6x16_neon;
pub use matmul::{
    GemmEpilogue, PackedB, matmul_2d, matmul_2d_sequential, matmul_2d_slices,
    matmul_2d_slices_fused_maybe_packed, matmul_2d_slices_trans_a, matmul_2d_with_config,
    pack_b_for_session,
};
pub use simd::{
    add_reduce_dispatch, binary_same_shape_dispatch, exp_slice_dispatch, fma_slice_dispatch,
    matmul_row_dispatch, max_reduce_dispatch, relu_slice_dispatch, relu_to_slice_dispatch,
    sigmoid_slice_dispatch, sub_exp_slice_dispatch, tanh_slice_dispatch,
};

pub use config::{
    BatchNorm2dTensors, Conv2dSpec, DepthwiseConv2dSpec, GroupNorm2dTensors,
    LayerNormLastDimTensors, Pool2dSpec, RmsNormLastDimTensors, SeparableConv2dKernels,
    SeparableConv2dSpec,
};
pub use conv::Activation;
pub use conv::conv2d_nhwc_indirect_padded;
pub use conv::conv2d_nhwc_padded;
pub use conv::conv3d;
pub use conv::{
    conv2d_nchwc_pointwise_with_activation_prepacked, conv2d_nchwc_with_activation_prepacked,
    conv2d_nhwc_pointwise_with_residual_relu, conv2d_nhwc_with_activation_prepacked,
    conv2d_nhwc_with_activation_with_config_and_pool, conv2d_nhwc_with_config_and_pool,
    depthwise_conv2d_nhwc_padded_with_activation_with_config_and_pool,
    depthwise_conv2d_nhwc_padded_with_config_and_pool,
    depthwise_conv2d_nhwc_with_activation_with_config_and_pool,
    depthwise_conv2d_nhwc_with_config_and_pool, fused_dw_pw_nhwc_streaming,
    separable_conv2d_nhwc_with_config_and_pool,
};
pub use deformable_conv::deformable_conv2d_nhwc;
pub use elementwise::{
    add_with_config_and_pool, exp_with_config_and_pool, mul_with_config_and_pool,
    relu_with_config_and_pool, sigmoid_with_config_and_pool, sub_with_config_and_pool,
    tanh_act_with_config_and_pool,
};
pub use embedding::{dropout, embedding_lookup};
pub use fused_pw_dw_3x3::fused_pw_expand_dw_3x3;
pub use int8_depthwise::{
    Depthwise3x3I8Params, DepthwiseI8Params, depthwise_i8_i32_nchw_khwc_dispatch,
    depthwise_i8_i32_nchw_khwc_scalar, depthwise_i8_i32_nhwc_dispatch,
    depthwise_i8_i32_nhwc_scalar, depthwise3x3_i8_i32_nhwc_dispatch,
    depthwise3x3_i8_i32_nhwc_scalar,
};
pub use int8_fused_dw_pw_3x3::{Int8FusedDwPwParams, int8_fused_dw_pw_dispatch};
pub use int8_fused_pw_dw_3x3::{Int8FusedPwDwParams, int8_fused_pw_dw_dispatch};
pub use layout::{nchw_to_nchwc, nchw_to_nhwc_fast, nchwc_to_nchw, nchwc_to_nhwc, nhwc_to_nchwc};
pub use matmul::matmul_2d_with_config_and_pool;
pub use nchwc_dw3x3::conv2d_nchwc_dw3x3_s1_same_pad;
pub use nchwc_ops::{
    add_nchwc, avg_pool2d_nchwc, batch_norm2d_nchwc, max_pool2d_nchwc, relu_nchwc, sigmoid_nchwc,
    silu_nchwc,
};
pub use norm::{
    batch_norm2d_nhwc_with_config_and_pool, group_norm_nhwc_with_config_and_pool,
    layer_norm_last_dim_with_config_and_pool, log_softmax_last_dim_with_config_and_pool,
    logsumexp_last_dim_with_config_and_pool, rms_norm_last_dim_with_config_and_pool,
    softmax_last_dim_with_config_and_pool,
};
pub use pool::{
    avg_pool2d_nchw, avg_pool2d_nhwc_with_config_and_pool, max_pool2d_nchw,
    max_pool2d_nhwc_with_config_and_pool,
};
