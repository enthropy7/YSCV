//! Execution kernels and backend abstraction for yscv.
#![deny(unsafe_code)]

pub const CRATE_ID: &str = "yscv-kernels";

#[path = "backend.rs"]
mod backend;
#[path = "error.rs"]
mod error;
#[cfg(feature = "gpu")]
#[path = "gpu/mod.rs"]
mod gpu_backend;
#[cfg(feature = "gpu")]
#[path = "gpu_session.rs"]
mod gpu_session;
#[cfg(feature = "gpu")]
#[path = "multi_device.rs"]
mod multi_device;
#[path = "ops/mod.rs"]
mod ops;
#[path = "scope_ctx.rs"]
mod scope_ctx;

pub use scope_ctx::{ScopeGuard, install_scope, with_installed_session, with_scope};

pub use backend::conv2d_nhwc_padded;
pub use backend::conv2d_nhwc_with_activation_prepacked_default;
pub use backend::{
    Backend, BackwardOps, BatchNorm2dParams, CpuBackend, GroupNormNhwcParams,
    LayerNormLastDimParams, RmsNormLastDimParams, SeparableConv2dParams, ThreadedCpuBackend,
    ThreadedCpuBackendConfig, add, add_inplace, add_relu_inplace, add_with_config, avg_pool2d_nhwc,
    avg_pool2d_nhwc_with_config, batch_norm2d_nhwc, batch_norm2d_nhwc_with_config, conv2d_nhwc,
    conv2d_nhwc_with_activation, conv2d_nhwc_with_config, deformable_conv2d_nhwc,
    depthwise_conv2d_nhwc, depthwise_conv2d_nhwc_padded,
    depthwise_conv2d_nhwc_padded_with_activation, depthwise_conv2d_nhwc_padded_with_config,
    depthwise_conv2d_nhwc_with_activation, depthwise_conv2d_nhwc_with_config, dropout,
    embedding_lookup, exp, exp_with_config, flash_attention, gelu, group_norm_nhwc,
    group_norm_nhwc_with_config, layer_norm_last_dim, layer_norm_last_dim_with_config,
    log_softmax_last_dim, log_softmax_last_dim_with_config, logsumexp_last_dim,
    logsumexp_last_dim_with_config, matmul_2d, matmul_2d_sequential, matmul_2d_slices,
    matmul_2d_slices_trans_a, matmul_2d_with_config, matmul_2d_with_threads, max_pool2d_nhwc,
    max_pool2d_nhwc_with_config, mish, mul, mul_with_config, relu, relu_inplace, relu_with_config,
    rms_norm_last_dim, rms_norm_last_dim_with_config, scaled_dot_product_attention,
    separable_conv2d_nhwc, separable_conv2d_nhwc_with_config, sigmoid, sigmoid_with_config, silu,
    silu_inplace, softmax_last_dim, softmax_last_dim_with_config, sub, sub_with_config, tanh_act,
    tanh_act_with_config, transpose_conv2d_nhwc,
};
pub use error::KernelError;
#[cfg(feature = "gpu")]
pub use gpu_backend::{
    GpuBackend, GpuBuffer, RecordedOp, gpu_batch_norm, gpu_layer_norm, gpu_transpose,
};
#[cfg(feature = "gpu")]
pub use gpu_session::GpuSession;
#[cfg(feature = "gpu")]
pub use multi_device::{
    GpuApiBackend, GpuDeviceInfo, GpuDeviceType, MultiGpuBackend, SchedulingStrategy,
    enumerate_gpu_devices,
};
pub use ops::Activation;
#[cfg(all(target_os = "macos", feature = "blas"))]
pub use ops::bnns_conv;
pub use ops::conv2d_nhwc_indirect_padded;
pub use ops::fused_pw_expand_dw_3x3;
#[cfg(target_arch = "aarch64")]
pub use ops::hgemm_6x16_neon;
pub use ops::int4_matmul::{
    pack_int4_symmetric_per_group, packed_int4_gemm_dispatch, packed_int4_gemm_scalar,
    packed_int4_gemv_dispatch, packed_int4_gemv_scalar,
};
pub use ops::int8_matmul::{int8_matmul_dispatch, int8_matmul_scalar};
pub use ops::quantize::{dequantize_int4_to_f32, quantize_f32_to_int4};
pub use ops::rope::apply_rotary_embedding;
pub use ops::{
    BinaryKind, DEFAULT_ELEMENTWISE_MIN_PARALLEL_ELEMENTS,
    DEFAULT_MATMUL_MIN_PARALLEL_OUTPUT_ELEMENTS, DEFAULT_MATMUL_MIN_PARALLEL_SHARED_DIM,
    GemmEpilogue, PackedB, ParallelElementwiseConfig, ParallelMatmulConfig, add_nchwc,
    add_reduce_dispatch, add_with_config_and_pool, avg_pool2d_nchw, avg_pool2d_nchwc,
    avg_pool2d_nhwc_with_config_and_pool, batch_norm2d_nchwc,
    batch_norm2d_nhwc_with_config_and_pool, binary_same_shape_dispatch,
    conv2d_nchwc_dw3x3_s1_same_pad, conv2d_nchwc_pointwise_with_activation_prepacked,
    conv2d_nchwc_with_activation_prepacked, conv2d_nhwc_pointwise_with_residual_relu,
    conv2d_nhwc_with_activation_prepacked, conv2d_nhwc_with_activation_with_config_and_pool,
    conv2d_nhwc_with_config_and_pool, conv3d,
    depthwise_conv2d_nhwc_padded_with_activation_with_config_and_pool,
    depthwise_conv2d_nhwc_padded_with_config_and_pool,
    depthwise_conv2d_nhwc_with_activation_with_config_and_pool,
    depthwise_conv2d_nhwc_with_config_and_pool, exp_slice_dispatch, exp_with_config_and_pool,
    fma_slice_dispatch, fused_dw_pw_nhwc_streaming, group_norm_nhwc_with_config_and_pool,
    layer_norm_last_dim_with_config_and_pool, log_softmax_last_dim_with_config_and_pool,
    logsumexp_last_dim_with_config_and_pool, matmul_2d_slices_fused_maybe_packed,
    matmul_2d_with_config_and_pool, matmul_row_dispatch, max_pool2d_nchw, max_pool2d_nchwc,
    max_pool2d_nhwc_with_config_and_pool, max_reduce_dispatch, mul_with_config_and_pool,
    nchw_to_nchwc, nchw_to_nhwc_fast, nchwc_to_nchw, nchwc_to_nhwc, nhwc_to_nchwc,
    pack_b_for_session, relu_nchwc, relu_out, relu_slice_dispatch, relu_to_slice_dispatch,
    relu_with_config_and_pool, rms_norm_last_dim_with_config_and_pool,
    separable_conv2d_nhwc_with_config_and_pool, sigmoid_nchwc, sigmoid_slice_dispatch,
    sigmoid_with_config_and_pool, silu_nchwc, softmax_last_dim_with_config_and_pool,
    sub_exp_slice_dispatch, sub_with_config_and_pool, tanh_act_with_config_and_pool,
    tanh_slice_dispatch,
};

#[cfg(test)]
#[path = "proptest_tests.rs"]
mod proptest_tests;

#[path = "tests/mod.rs"]
#[cfg(test)]
mod tests;
