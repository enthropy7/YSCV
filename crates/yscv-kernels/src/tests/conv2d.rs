use yscv_tensor::Tensor;

use crate::{
    KernelError, ParallelElementwiseConfig, SeparableConv2dParams, conv2d_nhwc,
    conv2d_nhwc_with_activation, conv2d_nhwc_with_config, depthwise_conv2d_nhwc,
    depthwise_conv2d_nhwc_padded, depthwise_conv2d_nhwc_padded_with_activation,
    depthwise_conv2d_nhwc_with_config, fused_dw_pw_nhwc_streaming, separable_conv2d_nhwc,
    separable_conv2d_nhwc_with_config,
};

use super::build_tensor;

// --- conv2d ---

#[test]
fn conv2d_nhwc_computes_expected_result() {
    let input = Tensor::from_vec(
        vec![1, 3, 3, 1],
        vec![
            1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            7.0, 8.0, 9.0,
        ],
    )
    .unwrap();
    let kernel = Tensor::from_vec(vec![2, 2, 1, 1], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let out = conv2d_nhwc(&input, &kernel, None, 1, 1).unwrap();

    assert_eq!(out.shape(), &[1, 2, 2, 1]);
    assert_eq!(out.data(), &[6.0, 8.0, 12.0, 14.0]);
}

#[test]
fn conv2d_nhwc_supports_bias() {
    let input = Tensor::from_vec(vec![1, 2, 2, 1], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let kernel = Tensor::from_vec(vec![1, 1, 1, 2], vec![2.0, -1.0]).unwrap();
    let bias = Tensor::from_vec(vec![2], vec![0.5, 1.0]).unwrap();
    let out = conv2d_nhwc(&input, &kernel, Some(&bias), 1, 1).unwrap();

    assert_eq!(out.shape(), &[1, 2, 2, 2]);
    assert_eq!(out.data(), &[2.5, 0.0, 4.5, -1.0, 6.5, -2.0, 8.5, -3.0]);
}

#[test]
fn conv2d_nhwc_rejects_invalid_rank() {
    let input = Tensor::from_vec(vec![3, 3, 1], vec![0.0; 9]).unwrap();
    let kernel = Tensor::from_vec(vec![2, 2, 1, 1], vec![0.0; 4]).unwrap();
    let err = conv2d_nhwc(&input, &kernel, None, 1, 1).unwrap_err();
    assert_eq!(
        err,
        KernelError::InvalidConvRank {
            input_rank: 3,
            kernel_rank: 4
        }
    );
}

#[test]
fn conv2d_nhwc_rejects_invalid_parameters() {
    let input = Tensor::from_vec(vec![1, 3, 3, 1], vec![0.0; 9]).unwrap();
    let kernel = Tensor::from_vec(vec![2, 2, 1, 1], vec![0.0; 4]).unwrap();
    let err = conv2d_nhwc(&input, &kernel, None, 0, 1).unwrap_err();
    assert_eq!(
        err,
        KernelError::InvalidConvParameters {
            kernel_h: 2,
            kernel_w: 2,
            stride_h: 0,
            stride_w: 1,
        }
    );
}

#[test]
fn conv2d_nhwc_rejects_channel_mismatch() {
    let input = Tensor::from_vec(vec![1, 3, 3, 2], vec![0.0; 18]).unwrap();
    let kernel = Tensor::from_vec(vec![2, 2, 1, 1], vec![0.0; 4]).unwrap();
    let err = conv2d_nhwc(&input, &kernel, None, 1, 1).unwrap_err();
    assert_eq!(
        err,
        KernelError::ConvChannelMismatch {
            input_channels: 2,
            kernel_in_channels: 1,
        }
    );
}

#[test]
fn conv2d_nhwc_rejects_kernel_larger_than_input() {
    let input = Tensor::from_vec(vec![1, 3, 3, 1], vec![0.0; 9]).unwrap();
    let kernel = Tensor::from_vec(vec![4, 3, 1, 1], vec![0.0; 12]).unwrap();
    let err = conv2d_nhwc(&input, &kernel, None, 1, 1).unwrap_err();
    assert_eq!(
        err,
        KernelError::ConvKernelLargerThanInput {
            input_h: 3,
            input_w: 3,
            kernel_h: 4,
            kernel_w: 3,
        }
    );
}

#[test]
fn conv2d_nhwc_rejects_bias_shape_mismatch() {
    let input = Tensor::from_vec(vec![1, 2, 2, 1], vec![0.0; 4]).unwrap();
    let kernel = Tensor::from_vec(vec![1, 1, 1, 2], vec![1.0, 1.0]).unwrap();
    let bias = Tensor::from_vec(vec![1], vec![0.0]).unwrap();
    let err = conv2d_nhwc(&input, &kernel, Some(&bias), 1, 1).unwrap_err();
    assert_eq!(
        err,
        KernelError::ConvBiasShapeMismatch {
            bias_shape: vec![1],
            out_channels: 2,
        }
    );
}

#[test]
fn conv2d_with_config_disabled_matches_default() {
    let input = build_tensor(&[2, 16, 16, 8], 0.22);
    let kernel = build_tensor(&[3, 3, 8, 12], 0.77);
    let bias = build_tensor(&[12], 0.31);
    let baseline = conv2d_nhwc(&input, &kernel, Some(&bias), 1, 1).unwrap();
    let disabled = conv2d_nhwc_with_config(
        &input,
        &kernel,
        Some(&bias),
        1,
        1,
        ParallelElementwiseConfig::disabled(),
    )
    .unwrap();
    assert_eq!(baseline, disabled);
}

// --- depthwise conv2d ---

#[test]
fn depthwise_conv2d_nhwc_computes_expected_result() {
    let input = Tensor::from_vec(
        vec![1, 3, 3, 2],
        vec![
            1.0, 10.0, 2.0, 20.0, 3.0, 30.0, //
            4.0, 40.0, 5.0, 50.0, 6.0, 60.0, //
            7.0, 70.0, 8.0, 80.0, 9.0, 90.0,
        ],
    )
    .unwrap();
    let kernel = Tensor::from_vec(
        vec![2, 2, 2, 1],
        vec![1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5],
    )
    .unwrap();
    let out = depthwise_conv2d_nhwc(&input, &kernel, None, 1, 1).unwrap();

    assert_eq!(out.shape(), &[1, 2, 2, 2]);
    assert_eq!(out.data(), &[6.0, 30.0, 8.0, 40.0, 12.0, 60.0, 14.0, 70.0]);
}

#[test]
fn depthwise_conv2d_nhwc_supports_depth_multiplier_and_bias() {
    let input = Tensor::from_vec(vec![1, 2, 2, 1], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let kernel = Tensor::from_vec(vec![1, 1, 1, 2], vec![2.0, -1.0]).unwrap();
    let bias = Tensor::from_vec(vec![2], vec![0.5, 1.0]).unwrap();
    let out = depthwise_conv2d_nhwc(&input, &kernel, Some(&bias), 1, 1).unwrap();

    assert_eq!(out.shape(), &[1, 2, 2, 2]);
    assert_eq!(out.data(), &[2.5, 0.0, 4.5, -1.0, 6.5, -2.0, 8.5, -3.0]);
}

#[test]
fn depthwise_conv2d_nhwc_rejects_invalid_rank() {
    let input = Tensor::from_vec(vec![3, 3, 1], vec![0.0; 9]).unwrap();
    let kernel = Tensor::from_vec(vec![2, 2, 1, 1], vec![0.0; 4]).unwrap();
    let err = depthwise_conv2d_nhwc(&input, &kernel, None, 1, 1).unwrap_err();
    assert_eq!(
        err,
        KernelError::InvalidDepthwiseConvRank {
            input_rank: 3,
            kernel_rank: 4,
        }
    );
}

#[test]
fn depthwise_conv2d_nhwc_rejects_invalid_parameters() {
    let input = Tensor::from_vec(vec![1, 3, 3, 1], vec![0.0; 9]).unwrap();
    let kernel = Tensor::from_vec(vec![2, 2, 1, 1], vec![0.0; 4]).unwrap();
    let err = depthwise_conv2d_nhwc(&input, &kernel, None, 1, 0).unwrap_err();
    assert_eq!(
        err,
        KernelError::InvalidDepthwiseConvParameters {
            kernel_h: 2,
            kernel_w: 2,
            stride_h: 1,
            stride_w: 0,
        }
    );
}

#[test]
fn depthwise_conv2d_nhwc_rejects_channel_mismatch() {
    let input = Tensor::from_vec(vec![1, 3, 3, 2], vec![0.0; 18]).unwrap();
    let kernel = Tensor::from_vec(vec![2, 2, 1, 1], vec![0.0; 4]).unwrap();
    let err = depthwise_conv2d_nhwc(&input, &kernel, None, 1, 1).unwrap_err();
    assert_eq!(
        err,
        KernelError::DepthwiseConvChannelMismatch {
            input_channels: 2,
            kernel_channels: 1,
        }
    );
}

#[test]
fn depthwise_conv2d_nhwc_rejects_kernel_larger_than_input() {
    let input = Tensor::from_vec(vec![1, 3, 3, 1], vec![0.0; 9]).unwrap();
    let kernel = Tensor::from_vec(vec![4, 3, 1, 1], vec![0.0; 12]).unwrap();
    let err = depthwise_conv2d_nhwc(&input, &kernel, None, 1, 1).unwrap_err();
    assert_eq!(
        err,
        KernelError::DepthwiseConvKernelLargerThanInput {
            input_h: 3,
            input_w: 3,
            kernel_h: 4,
            kernel_w: 3,
        }
    );
}

#[test]
fn depthwise_conv2d_nhwc_rejects_bias_shape_mismatch() {
    let input = Tensor::from_vec(vec![1, 2, 2, 1], vec![0.0; 4]).unwrap();
    let kernel = Tensor::from_vec(vec![1, 1, 1, 2], vec![1.0, 1.0]).unwrap();
    let bias = Tensor::from_vec(vec![1], vec![0.0]).unwrap();
    let err = depthwise_conv2d_nhwc(&input, &kernel, Some(&bias), 1, 1).unwrap_err();
    assert_eq!(
        err,
        KernelError::DepthwiseConvBiasShapeMismatch {
            bias_shape: vec![1],
            out_channels: 2,
        }
    );
}

#[test]
fn depthwise_conv2d_with_config_disabled_matches_default() {
    let input = build_tensor(&[2, 16, 16, 8], 0.17);
    let kernel = build_tensor(&[3, 3, 8, 2], 0.66);
    let bias = build_tensor(&[16], 0.29);
    let baseline = depthwise_conv2d_nhwc(&input, &kernel, Some(&bias), 1, 1).unwrap();
    let disabled = depthwise_conv2d_nhwc_with_config(
        &input,
        &kernel,
        Some(&bias),
        1,
        1,
        ParallelElementwiseConfig::disabled(),
    )
    .unwrap();
    assert_eq!(baseline, disabled);
}

#[test]
fn depthwise_conv2d_nhwc_padded_matches_explicit_padding_dm1() {
    let input = Tensor::from_vec(
        vec![1, 3, 3, 1],
        vec![
            1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            7.0, 8.0, 9.0,
        ],
    )
    .unwrap();
    let kernel = Tensor::from_vec(
        vec![3, 3, 1, 1],
        vec![
            1.0, 0.0, -1.0, //
            1.0, 0.0, -1.0, //
            1.0, 0.0, -1.0,
        ],
    )
    .unwrap();

    let out_virtual =
        depthwise_conv2d_nhwc_padded(&input, &kernel, None, 1, 1, 1, 1, 1, 1).unwrap();

    let padded = Tensor::from_vec(
        vec![1, 5, 5, 1],
        vec![
            0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 2.0, 3.0, 0.0, //
            0.0, 4.0, 5.0, 6.0, 0.0, //
            0.0, 7.0, 8.0, 9.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 0.0,
        ],
    )
    .unwrap();
    let out_explicit = depthwise_conv2d_nhwc(&padded, &kernel, None, 1, 1).unwrap();

    assert_eq!(out_virtual.shape(), &[1, 3, 3, 1]);
    assert_eq!(out_virtual, out_explicit);
}

#[test]
fn depthwise_conv2d_nhwc_padded_matches_explicit_padding_dm2_with_bias() {
    let input = Tensor::from_vec(vec![1, 2, 2, 1], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let kernel = Tensor::from_vec(
        vec![2, 2, 1, 2],
        vec![1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.0, 1.0],
    )
    .unwrap();
    let bias = Tensor::from_vec(vec![2], vec![0.1, -0.2]).unwrap();

    let out_virtual =
        depthwise_conv2d_nhwc_padded(&input, &kernel, Some(&bias), 1, 1, 1, 1, 0, 0).unwrap();

    let padded = Tensor::from_vec(
        vec![1, 3, 3, 1],
        vec![
            0.0, 0.0, 0.0, //
            0.0, 1.0, 2.0, //
            0.0, 3.0, 4.0,
        ],
    )
    .unwrap();
    let out_explicit = depthwise_conv2d_nhwc(&padded, &kernel, Some(&bias), 1, 1).unwrap();

    assert_eq!(out_virtual.shape(), &[1, 2, 2, 2]);
    assert_eq!(out_virtual, out_explicit);
}

// --- separable conv2d ---

#[test]
fn separable_conv2d_nhwc_computes_expected_result() {
    let input = Tensor::from_vec(
        vec![1, 3, 3, 1],
        vec![
            1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            7.0, 8.0, 9.0,
        ],
    )
    .unwrap();
    let depthwise_kernel = Tensor::from_vec(vec![2, 2, 1, 1], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let pointwise_kernel = Tensor::from_vec(vec![1, 1, 1, 2], vec![2.0, -1.0]).unwrap();
    let pointwise_bias = Tensor::from_vec(vec![2], vec![0.5, 1.0]).unwrap();

    let out = separable_conv2d_nhwc(
        &input,
        SeparableConv2dParams {
            depthwise_kernel: &depthwise_kernel,
            depthwise_bias: None,
            pointwise_kernel: &pointwise_kernel,
            pointwise_bias: Some(&pointwise_bias),
        },
        1,
        1,
    )
    .unwrap();

    assert_eq!(out.shape(), &[1, 2, 2, 2]);
    assert_eq!(
        out.data(),
        &[12.5, -5.0, 16.5, -7.0, 24.5, -11.0, 28.5, -13.0]
    );
}

#[test]
fn separable_conv2d_nhwc_rejects_non_pointwise_kernel_shape() {
    let input = Tensor::from_vec(vec![1, 3, 3, 1], vec![0.0; 9]).unwrap();
    let depthwise_kernel = Tensor::from_vec(vec![2, 2, 1, 1], vec![0.0; 4]).unwrap();
    let pointwise_kernel = Tensor::from_vec(vec![2, 1, 1, 1], vec![0.0; 2]).unwrap();
    let err = separable_conv2d_nhwc(
        &input,
        SeparableConv2dParams {
            depthwise_kernel: &depthwise_kernel,
            depthwise_bias: None,
            pointwise_kernel: &pointwise_kernel,
            pointwise_bias: None,
        },
        1,
        1,
    )
    .unwrap_err();

    assert_eq!(
        err,
        KernelError::InvalidSeparablePointwiseKernelShape {
            pointwise_shape: vec![2, 1, 1, 1],
        }
    );
}

#[test]
fn separable_conv2d_with_config_disabled_matches_default() {
    let input = build_tensor(&[2, 16, 16, 8], 0.15);
    let depthwise_kernel = build_tensor(&[3, 3, 8, 2], 0.33);
    let depthwise_bias = build_tensor(&[16], 0.51);
    let pointwise_kernel = build_tensor(&[1, 1, 16, 12], 0.79);
    let pointwise_bias = build_tensor(&[12], 0.21);
    let baseline = separable_conv2d_nhwc(
        &input,
        SeparableConv2dParams {
            depthwise_kernel: &depthwise_kernel,
            depthwise_bias: Some(&depthwise_bias),
            pointwise_kernel: &pointwise_kernel,
            pointwise_bias: Some(&pointwise_bias),
        },
        1,
        1,
    )
    .unwrap();
    let disabled = separable_conv2d_nhwc_with_config(
        &input,
        SeparableConv2dParams {
            depthwise_kernel: &depthwise_kernel,
            depthwise_bias: Some(&depthwise_bias),
            pointwise_kernel: &pointwise_kernel,
            pointwise_bias: Some(&pointwise_bias),
        },
        1,
        1,
        ParallelElementwiseConfig::disabled(),
    )
    .unwrap();
    assert_eq!(baseline, disabled);
}

#[test]
fn fused_dw_pw_streaming_padded_matches_reference_chain() {
    let input = build_tensor(&[1, 8, 8, 8], 0.19);
    let depthwise_kernel = build_tensor(&[3, 3, 8, 1], 0.37);
    let depthwise_bias = build_tensor(&[8], 0.11);
    let pointwise_kernel = build_tensor(&[1, 1, 8, 12], 0.47);
    let pointwise_bias = build_tensor(&[12], 0.23);

    let reference_dw = depthwise_conv2d_nhwc_padded_with_activation(
        &input,
        &depthwise_kernel,
        Some(&depthwise_bias),
        1,
        1,
        1,
        1,
        1,
        1,
        crate::Activation::Relu,
    )
    .unwrap();
    let reference = conv2d_nhwc_with_activation(
        &reference_dw,
        &pointwise_kernel,
        Some(&pointwise_bias),
        1,
        1,
        crate::Activation::Silu,
    )
    .unwrap();

    let streamed = fused_dw_pw_nhwc_streaming(
        &input,
        &depthwise_kernel,
        Some(&depthwise_bias),
        &pointwise_kernel,
        Some(&pointwise_bias),
        1,
        1,
        1,
        1,
        1,
        1,
        crate::Activation::Relu,
        crate::Activation::Silu,
    )
    .unwrap();

    assert_eq!(streamed.shape(), reference.shape());
    for (idx, (a, b)) in streamed
        .data()
        .iter()
        .zip(reference.data().iter())
        .enumerate()
    {
        let diff = (a - b).abs();
        assert!(
            diff <= 1e-4,
            "mismatch at idx={idx}: streamed={a} reference={b} diff={diff}"
        );
    }
}

// --- transpose conv2d ---

#[test]
fn transpose_conv2d_nhwc_identity_kernel() {
    // 1x1 identity convolution with stride 1 just copies + bias
    let input = Tensor::from_vec(vec![1, 2, 2, 1], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let kernel = Tensor::from_vec(vec![1, 1, 1, 1], vec![1.0]).unwrap();
    let bias = Tensor::from_vec(vec![1], vec![0.5]).unwrap();
    let out = crate::transpose_conv2d_nhwc(&input, &kernel, Some(&bias), 1, 1).unwrap();
    assert_eq!(out.shape(), &[1, 2, 2, 1]);
    assert_eq!(out.data(), &[1.5, 2.5, 3.5, 4.5]);
}

#[test]
fn transpose_conv2d_nhwc_stride2_upsample() {
    let input = Tensor::from_vec(vec![1, 2, 2, 1], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let kernel = Tensor::from_vec(vec![2, 2, 1, 1], vec![1.0, 0.0, 0.0, 0.0]).unwrap();
    let out = crate::transpose_conv2d_nhwc(&input, &kernel, None, 2, 2).unwrap();
    // output shape: (2-1)*2 + 2 = 4
    assert_eq!(out.shape(), &[1, 4, 4, 1]);
    // top-left of each 2x2 region gets the input value
    assert_eq!(out.data()[0], 1.0);
    assert_eq!(out.data()[2], 2.0); // (0,2)
}
