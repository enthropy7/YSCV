pub(crate) mod attention;
#[cfg(all(target_os = "macos", feature = "blas"))]
pub mod bnns_conv;
mod config;
mod conv;
mod deformable_conv;
mod elementwise;
mod embedding;
mod matmul;
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
    add_with_config, exp, exp_with_config, gelu, mish, mul_with_config, relu, relu_inplace,
    relu_out, relu_with_config, sigmoid, sigmoid_with_config, silu, silu_inplace, sub_with_config,
    tanh_act, tanh_act_with_config,
};
pub use matmul::{matmul_2d, matmul_2d_sequential, matmul_2d_slices, matmul_2d_with_config};
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
#[cfg(feature = "blas")]
pub use conv::conv2d_nhwc_padded;
pub use conv::conv3d;
pub use conv::{
    conv2d_nhwc_with_config_and_pool, depthwise_conv2d_nhwc_with_config_and_pool,
    separable_conv2d_nhwc_with_config_and_pool,
};
pub use deformable_conv::deformable_conv2d_nhwc;
pub use elementwise::{
    add_with_config_and_pool, exp_with_config_and_pool, mul_with_config_and_pool,
    relu_with_config_and_pool, sigmoid_with_config_and_pool, sub_with_config_and_pool,
    tanh_act_with_config_and_pool,
};
pub use embedding::{dropout, embedding_lookup};
pub use matmul::matmul_2d_with_config_and_pool;
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
