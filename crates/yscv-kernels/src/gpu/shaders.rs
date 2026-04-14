pub(crate) const MATMUL_WGSL: &str = include_str!("../shaders/matmul.wgsl");
pub(crate) const ELEMENTWISE_WGSL: &str = include_str!("../shaders/elementwise.wgsl");
pub(crate) const UNARY_WGSL: &str = include_str!("../shaders/unary.wgsl");
pub(crate) const SOFTMAX_WGSL: &str = include_str!("../shaders/softmax.wgsl");
pub(crate) const LOG_SOFTMAX_WGSL: &str = include_str!("../shaders/log_softmax.wgsl");
pub(crate) const LOGSUMEXP_WGSL: &str = include_str!("../shaders/logsumexp.wgsl");
pub(crate) const CONV2D_WGSL: &str = include_str!("../shaders/conv2d.wgsl");
pub(crate) const POOL2D_WGSL: &str = include_str!("../shaders/pool2d.wgsl");
pub(crate) const BATCH_NORM_WGSL: &str = include_str!("../shaders/batch_norm.wgsl");
pub(crate) const LAYER_NORM_WGSL: &str = include_str!("../shaders/layer_norm.wgsl");
pub(crate) const DEPTHWISE_CONV2D_WGSL: &str = include_str!("../shaders/depthwise_conv2d.wgsl");
pub(crate) const TRANSPOSE_CONV2D_WGSL: &str = include_str!("../shaders/transpose_conv2d.wgsl");
pub(crate) const TRANSPOSE_2D_WGSL: &str = include_str!("../shaders/transpose_2d.wgsl");
pub(crate) const GATHER_WGSL: &str = include_str!("../shaders/gather.wgsl");
pub(crate) const ATTENTION_WGSL: &str = include_str!("../shaders/attention.wgsl");
pub(crate) const GROUP_NORM_WGSL: &str = include_str!("../shaders/group_norm.wgsl");
pub(crate) const RMS_NORM_WGSL: &str = include_str!("../shaders/rms_norm.wgsl");
pub(crate) const BACKWARD_BINARY_WGSL: &str = include_str!("../shaders/backward_binary.wgsl");
pub(crate) const CONV2D_INPUT_GRAD_WGSL: &str = include_str!("../shaders/conv2d_input_grad.wgsl");
pub(crate) const REDUCE_SUM_BACKWARD_WGSL: &str =
    include_str!("../shaders/reduce_sum_backward.wgsl");
pub(crate) const CHANNEL_SPLIT_WGSL: &str = include_str!("../shaders/channel_split.wgsl");
pub(crate) const CHANNEL_CONCAT_WGSL: &str = include_str!("../shaders/channel_concat.wgsl");
pub(crate) const RESIZE_NEAREST_WGSL: &str = include_str!("../shaders/resize_nearest.wgsl");
pub(crate) const PERMUTE_NHWC_NCHW_WGSL: &str = include_str!("../shaders/permute_nhwc_nchw.wgsl");
pub(crate) const PERMUTE_ND_WGSL: &str = include_str!("../shaders/permute_nd.wgsl");
pub(crate) const GENERAL_CONCAT_WGSL: &str = include_str!("../shaders/general_concat.wgsl");
pub(crate) const GENERAL_SPLIT_WGSL: &str = include_str!("../shaders/general_split.wgsl");
pub(crate) const SLICE_ND_WGSL: &str = include_str!("../shaders/slice_nd.wgsl");
pub(crate) const BROADCAST_BINARY_WGSL: &str = include_str!("../shaders/broadcast_binary.wgsl");
pub(crate) const BIAS_ADD_WGSL: &str = include_str!("../shaders/bias_add.wgsl");
pub(crate) const BATCHED_MATMUL_WGSL: &str = include_str!("../shaders/batched_matmul.wgsl");
pub(crate) const CONV_GEMM_WGSL: &str = include_str!("../shaders/conv_gemm.wgsl");
pub(crate) const CONV_GEMM_F16_WGSL: &str = include_str!("../shaders/conv_gemm_f16.wgsl");
pub(crate) const CONV_GEMM_FAST_WGSL: &str = include_str!("../shaders/conv_gemm_fast.wgsl");
pub(crate) const CONV_GEMM_V2_WGSL: &str = include_str!("../shaders/conv_gemm_v2.wgsl");
pub(crate) const CONV_GEMM_FAST_N32_WGSL: &str = include_str!("../shaders/conv_gemm_fast_n32.wgsl");
pub(crate) const CONV_GEMM_F16_N32_WGSL: &str = include_str!("../shaders/conv_gemm_f16_n32.wgsl");
pub(crate) const CONV_GEMM_F16_IO_WGSL: &str = include_str!("../shaders/conv_gemm_f16_io.wgsl");
pub(crate) const CONV_GEMM_F16_IO_N32_WGSL: &str =
    include_str!("../shaders/conv_gemm_f16_io_n32.wgsl");
pub(crate) const ELEMENTWISE_F16_WGSL: &str = include_str!("../shaders/elementwise_f16.wgsl");
pub(crate) const UNARY_F16_WGSL: &str = include_str!("../shaders/unary_f16.wgsl");
pub(crate) const CHANNEL_CONCAT_F16_WGSL: &str = include_str!("../shaders/channel_concat_f16.wgsl");
pub(crate) const CHANNEL_SPLIT_F16_WGSL: &str = include_str!("../shaders/channel_split_f16.wgsl");
pub(crate) const RESIZE_NEAREST_F16_WGSL: &str = include_str!("../shaders/resize_nearest_f16.wgsl");
pub(crate) const PERMUTE_NHWC_NCHW_F16_WGSL: &str =
    include_str!("../shaders/permute_nhwc_nchw_f16.wgsl");
pub(crate) const CONVERT_F32_TO_F16_WGSL: &str = include_str!("../shaders/convert_f32_to_f16.wgsl");
pub(crate) const CONVERT_F16_TO_F32_WGSL: &str = include_str!("../shaders/convert_f16_to_f32.wgsl");
pub(crate) const WINOGRAD_INPUT_WGSL: &str =
    include_str!("../shaders/winograd_input_transform.wgsl");
pub(crate) const WINOGRAD_OUTPUT_WGSL: &str =
    include_str!("../shaders/winograd_output_transform.wgsl");
