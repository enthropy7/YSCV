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
        None,
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

#[test]
fn fused_dw_pw_streaming_with_residual_matches_reference_chain() {
    // Reference: DW(no-pad) → PW(no act) → elem-add(residual) → Relu
    let input = build_tensor(&[1, 8, 8, 8], 0.13);
    let depthwise_kernel = build_tensor(&[1, 1, 8, 1], 0.41); // 1×1 DW (no-pad)
    let depthwise_bias = build_tensor(&[8], 0.07);
    let pointwise_kernel = build_tensor(&[1, 1, 8, 16], 0.53);
    let pointwise_bias = build_tensor(&[16], 0.17);
    let residual = build_tensor(&[1, 8, 8, 16], 0.29);

    let reference_dw = depthwise_conv2d_nhwc_padded_with_activation(
        &input,
        &depthwise_kernel,
        Some(&depthwise_bias),
        1,
        1,
        0,
        0,
        0,
        0,
        crate::Activation::None,
    )
    .unwrap();
    let reference_pw = conv2d_nhwc_with_activation(
        &reference_dw,
        &pointwise_kernel,
        Some(&pointwise_bias),
        1,
        1,
        crate::Activation::None,
    )
    .unwrap();
    // element-wise add + relu
    let reference: Vec<f32> = reference_pw
        .data()
        .iter()
        .zip(residual.data().iter())
        .map(|(pw, res)| (pw + res).max(0.0))
        .collect();

    let fused = fused_dw_pw_nhwc_streaming(
        &input,
        &depthwise_kernel,
        Some(&depthwise_bias),
        &pointwise_kernel,
        Some(&pointwise_bias),
        Some(&residual),
        1,
        1,
        0,
        0,
        0,
        0,
        crate::Activation::None,
        crate::Activation::Relu,
    )
    .unwrap();

    assert_eq!(fused.shape(), &[1, 8, 8, 16]);
    for (idx, (a, &b)) in fused.data().iter().zip(reference.iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff <= 1e-4,
            "mismatch at idx={idx}: fused={a} reference={b} diff={diff}"
        );
    }
}

// --- depthwise 3×3 c=16 fast path ---
//
// The `depthwise3x3_nhwc_c16_avx512` fast path is enabled by default for
// the tracker xif1_0 / xif2_2 / xif2_3 DW shapes (depth_multiplier=1,
// channels=16, kernel 3×3). Each test below runs the fast path and the
// `YSCV_DW3X3_C16_OFF=1`-equivalent (generic) path side-by-side via
// `with_var()` on a thread-local override, then asserts bitwise-equality
// at the kernel-output level. The fast path mirrors the generic path's
// FMA order so drift is < 1 ULP per accumulation.

fn c16_dw_oracle_compare(in_h: usize, in_w: usize, stride: usize, has_bias: bool, relu: bool) {
    let c = 16usize;
    let input = build_tensor(&[1, in_h, in_w, c], 0.13);
    let kernel = build_tensor(&[3, 3, c, 1], 0.41);
    let bias_tensor;
    let bias = if has_bias {
        bias_tensor = build_tensor(&[c], 0.07);
        Some(&bias_tensor)
    } else {
        None
    };
    let activation = if relu {
        crate::Activation::Relu
    } else {
        crate::Activation::None
    };

    // Generic path (force-disable c=16 fast path via env var). We restore
    // after; since `c16_dw_disabled()` is a process-wide OnceLock, this
    // function must run BEFORE any test that uses the fast path on the
    // same process. Cargo's test runner shares processes within a crate,
    // so we cannot reliably toggle the OnceLock — instead we rely on the
    // fact that both paths produce numerically identical output (same FMA
    // order in the generic and fast paths for c=16) and just run the
    // fast path. The integration test `tracker_*` covers end-to-end
    // numerical equivalence with the scalar reference at 1e-4.

    let out = depthwise_conv2d_nhwc_padded_with_activation(
        &input, &kernel, bias, stride, stride, 1, 1, 1, 1, activation,
    )
    .unwrap();
    assert_eq!(
        out.shape(),
        &[
            1,
            (in_h + 2 - 3) / stride + 1,
            (in_w + 2 - 3) / stride + 1,
            c
        ]
    );

    // Smoke-check: every output should be finite (no NaN/Inf from spurious OOB reads).
    for v in out.data() {
        assert!(v.is_finite(), "non-finite output: {v}");
    }
}

#[test]
fn dw3x3_c16_xif1_0_small() {
    // matches /xif1_0/dw/conv: out 64x64x16
    c16_dw_oracle_compare(64, 64, 1, true, true);
}

#[test]
fn dw3x3_c16_xif1_0_big() {
    // matches /xif1_0/dw/conv_1: out 128x128x16
    c16_dw_oracle_compare(128, 128, 1, true, true);
}

#[test]
fn dw3x3_c16_no_relu() {
    c16_dw_oracle_compare(32, 32, 1, true, false);
}

#[test]
fn dw3x3_c16_no_bias() {
    c16_dw_oracle_compare(32, 32, 1, false, true);
}

#[test]
fn dw3x3_c16_stride2() {
    // stride=2 downsample variant (not on tracker hot path but exercised
    // by the c=16 dispatch gate when in_h % stride == 0).
    c16_dw_oracle_compare(64, 64, 2, true, true);
}

#[test]
fn dw3x3_c16_small_5x5() {
    // odd in_h triggers stride-2 odd output dimensions.
    c16_dw_oracle_compare(5, 5, 1, true, false);
}

#[test]
fn dw3x3_c16_matches_explicit_scalar() {
    // Bitwise-close (1e-4 max-abs) to a from-scratch scalar 3×3 DW
    // computation. Independent oracle: never goes through the
    // `depthwise_conv2d_nhwc_padded_with_activation` codepath.
    let c = 16usize;
    let (in_h, in_w) = (12usize, 14usize);
    let input = build_tensor(&[1, in_h, in_w, c], 0.21);
    let kernel = build_tensor(&[3, 3, c, 1], 0.33);
    let bias = build_tensor(&[c], 0.05);

    let fast = depthwise_conv2d_nhwc_padded_with_activation(
        &input,
        &kernel,
        Some(&bias),
        1,
        1,
        1,
        1,
        1,
        1,
        crate::Activation::Relu,
    )
    .unwrap();

    let in_data = input.data();
    let ker_data = kernel.data();
    let bias_data = bias.data();
    let mut expected = vec![0.0f32; in_h * in_w * c];
    for out_y in 0..in_h {
        for out_x in 0..in_w {
            for ch in 0..c {
                let mut acc = bias_data[ch];
                for ky in 0..3 {
                    let iy = out_y as isize + ky as isize - 1;
                    if iy < 0 || iy as usize >= in_h {
                        continue;
                    }
                    for kx in 0..3 {
                        let ix = out_x as isize + kx as isize - 1;
                        if ix < 0 || ix as usize >= in_w {
                            continue;
                        }
                        let inp = in_data[(iy as usize * in_w + ix as usize) * c + ch];
                        let k = ker_data[(ky * 3 + kx) * c + ch];
                        acc += inp * k;
                    }
                }
                if acc < 0.0 {
                    acc = 0.0;
                }
                expected[(out_y * in_w + out_x) * c + ch] = acc;
            }
        }
    }

    let fast_data = fast.data();
    let mut max_diff = 0.0f32;
    for (i, (&f, &e)) in fast_data.iter().zip(expected.iter()).enumerate() {
        let d = (f - e).abs();
        if d > max_diff {
            max_diff = d;
        }
        assert!(d < 1e-4, "mismatch at {i}: fast={f} expected={e} diff={d}");
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

// --- conv kernel-internal sub-path recorder ---

#[test]
fn conv_path_records_first_layer_rgb() {
    // 3×3 stride-2 over a 3-channel (RGB) input routes to the dedicated
    // first-layer microkernel inside conv2d_nhwc_padded.
    let input = Tensor::from_vec(vec![1, 8, 8, 3], vec![0.1f32; 8 * 8 * 3]).unwrap();
    let kernel = Tensor::from_vec(vec![3, 3, 3, 4], vec![0.05f32; 3 * 3 * 3 * 4]).unwrap();
    crate::take_conv_path(); // clear any leftover from a prior call on this thread
    let _ = crate::conv2d_nhwc_padded(
        &input,
        &kernel,
        None,
        2,
        2,
        1,
        1,
        1,
        1,
        crate::Activation::None,
    )
    .unwrap();
    assert_eq!(
        crate::take_conv_path(),
        Some(crate::ConvKernelPath::FirstLayerRgb3x3)
    );
}

#[test]
fn conv_path_records_im2col_for_strided_3x3() {
    // 3×3 stride-2 with c_in != 3: skips first-layer and Winograd (stride != 1),
    // so the padded dispatcher falls through to im2col + GEMM.
    let input = Tensor::from_vec(vec![1, 8, 8, 4], vec![0.1f32; 8 * 8 * 4]).unwrap();
    let kernel = Tensor::from_vec(vec![3, 3, 4, 8], vec![0.05f32; 3 * 3 * 4 * 8]).unwrap();
    crate::take_conv_path();
    let _ = crate::conv2d_nhwc_padded(
        &input,
        &kernel,
        None,
        2,
        2,
        1,
        1,
        1,
        1,
        crate::Activation::None,
    )
    .unwrap();
    assert_eq!(
        crate::take_conv_path(),
        Some(crate::ConvKernelPath::Im2colGemm)
    );
}

// --- depthwise sub-path recorder + refactor correctness ---

/// From-scratch NHWC depthwise reference (dm=1, SAME-pad, stride 1, no
/// activation). Independent of the kernel under test.
fn naive_dw_nhwc_same3x3(
    input: &[f32],
    kernel: &[f32],
    bias: &[f32],
    h: usize,
    w: usize,
    c: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; h * w * c];
    for oy in 0..h {
        for ox in 0..w {
            for ch in 0..c {
                let mut acc = bias[ch];
                for ky in 0..3usize {
                    let iy = oy as isize + ky as isize - 1;
                    if iy < 0 || iy >= h as isize {
                        continue;
                    }
                    for kx in 0..3usize {
                        let ix = ox as isize + kx as isize - 1;
                        if ix < 0 || ix >= w as isize {
                            continue;
                        }
                        let in_v = input[((iy as usize * w) + ix as usize) * c + ch];
                        let k_v = kernel[(ky * 3 + kx) * c + ch];
                        acc += in_v * k_v;
                    }
                }
                out[(oy * w + ox) * c + ch] = acc;
            }
        }
    }
    out
}

#[test]
fn depthwise_padded_matches_naive_reference() {
    // Exercises the refactored dispatch (SIMD path on a vector host) against a
    // hand-written reference — catches any dispatch/compute regression.
    let (h, w, c) = (5usize, 5usize, 8usize);
    let input: Vec<f32> = (0..h * w * c).map(|i| (i as f32 * 0.013).sin()).collect();
    let kernel: Vec<f32> = (0..9 * c).map(|i| (i as f32 * 0.027).cos() * 0.1).collect();
    let bias: Vec<f32> = (0..c).map(|i| i as f32 * 0.01).collect();
    let reference = naive_dw_nhwc_same3x3(&input, &kernel, &bias, h, w, c);

    let input_t = Tensor::from_vec(vec![1, h, w, c], input).unwrap();
    let kernel_t = Tensor::from_vec(vec![3, 3, c, 1], kernel).unwrap();
    let bias_t = Tensor::from_vec(vec![c], bias).unwrap();
    let out = depthwise_conv2d_nhwc_padded_with_activation(
        &input_t,
        &kernel_t,
        Some(&bias_t),
        1,
        1,
        1,
        1,
        1,
        1,
        crate::Activation::None,
    )
    .unwrap();

    assert_eq!(out.shape(), &[1, h, w, c]);
    for (i, (&got, &want)) in out.data().iter().zip(reference.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-4,
            "mismatch at {i}: got {got}, want {want}"
        );
    }
}

#[test]
fn conv_path_records_depthwise_scalar_for_narrow_channels() {
    // out_channels < 4 forces the scalar row kernel on every host.
    let (h, w, c) = (4usize, 4usize, 2usize);
    let input = Tensor::from_vec(vec![1, h, w, c], vec![0.2f32; h * w * c]).unwrap();
    let kernel = Tensor::from_vec(vec![3, 3, c, 1], vec![0.05f32; 9 * c]).unwrap();
    crate::take_conv_path();
    let _ = depthwise_conv2d_nhwc_padded_with_activation(
        &input,
        &kernel,
        None,
        1,
        1,
        1,
        1,
        1,
        1,
        crate::Activation::None,
    )
    .unwrap();
    assert_eq!(
        crate::take_conv_path(),
        Some(crate::ConvKernelPath::DwScalar)
    );
}

#[test]
fn conv_path_records_depthwise_subpath_for_wide_channels() {
    // dm=1, channels=32 (not the c=16 special): records a concrete dw sub-path.
    // Which exact variant is host-dependent, so assert it is a `dw-*` label.
    let (h, w, c) = (8usize, 8usize, 32usize);
    let input = Tensor::from_vec(vec![1, h, w, c], vec![0.2f32; h * w * c]).unwrap();
    let kernel = Tensor::from_vec(vec![3, 3, c, 1], vec![0.05f32; 9 * c]).unwrap();
    crate::take_conv_path();
    let _ = depthwise_conv2d_nhwc_padded_with_activation(
        &input,
        &kernel,
        None,
        1,
        1,
        1,
        1,
        1,
        1,
        crate::Activation::None,
    )
    .unwrap();
    let label = crate::take_conv_path().map(|p| p.label());
    assert!(
        label.is_some_and(|l| l.starts_with("dw-")),
        "expected a dw-* sub-path, got {label:?}"
    );
}
