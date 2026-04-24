//! End-to-end tests for the NCHWc layout path.
//!
//! Validates that the B.3 NCHWc pointwise Conv produces bitwise-identical
//! output to the equivalent NHWC pointwise Conv, after round-tripping
//! through the B.2 layout converters. This is the "correct by composition"
//! contract the plan describes: pointwise Conv is mathematically isomorphic
//! to NHWC, so the NCHWc path must not change numerics.

use yscv_tensor::{Layout, Tensor};

use crate::core::ops::Conv2dSpec;
use crate::{
    Activation, ParallelElementwiseConfig, conv2d_nchwc_pointwise_with_activation_prepacked,
    conv2d_nchwc_with_activation_prepacked, conv2d_nhwc_with_activation_prepacked, nhwc_to_nchwc,
};

fn make_nhwc(n: usize, h: usize, w: usize, c: usize) -> Tensor {
    let data: Vec<f32> = (0..n * h * w * c)
        .map(|i| ((i as f32) * 0.013).sin())
        .collect();
    Tensor::from_vec(vec![n, h, w, c], data)
        .unwrap()
        .with_layout(Layout::NHWC)
}

fn make_pointwise_kernel(c_in: usize, c_out: usize) -> Tensor {
    let data: Vec<f32> = (0..c_in * c_out)
        .map(|i| ((i as f32) * 0.031).cos())
        .collect();
    Tensor::from_vec(vec![1, 1, c_in, c_out], data).unwrap()
}

fn make_bias(c_out: usize) -> Tensor {
    let data: Vec<f32> = (0..c_out).map(|i| 0.125 * (i as f32 - 4.0)).collect();
    Tensor::from_vec(vec![c_out], data).unwrap()
}

#[test]
fn nchwc_pointwise_matches_nhwc_c_divisible_by_block() {
    let n = 1;
    let h = 4;
    let w = 6;
    let c_in = 16;
    let c_out = 8;
    let block = 8;

    let nhwc_input = make_nhwc(n, h, w, c_in);
    let kernel = make_pointwise_kernel(c_in, c_out);
    let bias = make_bias(c_out);

    let spec = Conv2dSpec {
        stride_h: 1,
        stride_w: 1,
    };
    let nhwc_out = conv2d_nhwc_with_activation_prepacked(
        &nhwc_input,
        &kernel,
        Some(&bias),
        spec,
        Activation::Relu,
        ParallelElementwiseConfig::default(),
        None,
        None,
    )
    .unwrap();
    assert_eq!(nhwc_out.shape(), &[n, h, w, c_out]);

    let nchwc_input = nhwc_to_nchwc(&nhwc_input, block).unwrap();
    let nchwc_out = conv2d_nchwc_pointwise_with_activation_prepacked(
        &nchwc_input,
        &kernel,
        Some(&bias),
        c_in,
        Activation::Relu,
        ParallelElementwiseConfig::default(),
        None,
        None,
    )
    .unwrap();
    assert_eq!(nchwc_out.layout(), Layout::NCHWc { block: block as u8 });
    assert_eq!(nchwc_out.shape(), &[n, c_out / block, h, w, block]);

    // Round-trip NCHWc output back to NHWC and compare elementwise.
    let nchwc_to_nhwc_out = crate::nchwc_to_nhwc(&nchwc_out, c_out).unwrap();
    assert_eq!(nchwc_to_nhwc_out.shape(), nhwc_out.shape());
    super::assert_slice_close(nchwc_to_nhwc_out.data(), nhwc_out.data(), 1e-5);
}

#[test]
fn nchwc_pointwise_handles_channel_padding() {
    // C_in = 17, C_out = 13 — neither divides 8. Tests padded lanes
    // don't leak non-zero values into the GEMM result.
    let n = 1;
    let h = 3;
    let w = 5;
    let c_in = 17;
    let c_out = 13;
    let block = 8;

    let nhwc_input = make_nhwc(n, h, w, c_in);
    let kernel = make_pointwise_kernel(c_in, c_out);

    let spec = Conv2dSpec {
        stride_h: 1,
        stride_w: 1,
    };
    let nhwc_out = conv2d_nhwc_with_activation_prepacked(
        &nhwc_input,
        &kernel,
        None,
        spec,
        Activation::None,
        ParallelElementwiseConfig::default(),
        None,
        None,
    )
    .unwrap();

    let nchwc_input = nhwc_to_nchwc(&nhwc_input, block).unwrap();
    let nchwc_out = conv2d_nchwc_pointwise_with_activation_prepacked(
        &nchwc_input,
        &kernel,
        None,
        c_in,
        Activation::None,
        ParallelElementwiseConfig::default(),
        None,
        None,
    )
    .unwrap();
    assert_eq!(nchwc_out.shape(), &[n, c_out.div_ceil(block), h, w, block]);

    let nchwc_to_nhwc_out = crate::nchwc_to_nhwc(&nchwc_out, c_out).unwrap();
    assert_eq!(nchwc_to_nhwc_out.shape(), nhwc_out.shape());
    super::assert_slice_close(nchwc_to_nhwc_out.data(), nhwc_out.data(), 1e-5);
}

#[test]
fn nchwc_pointwise_with_silu_activation() {
    let n = 2;
    let h = 2;
    let w = 2;
    let c_in = 8;
    let c_out = 8;
    let block = 8;

    let nhwc_input = make_nhwc(n, h, w, c_in);
    let kernel = make_pointwise_kernel(c_in, c_out);

    let spec = Conv2dSpec {
        stride_h: 1,
        stride_w: 1,
    };
    let nhwc_out = conv2d_nhwc_with_activation_prepacked(
        &nhwc_input,
        &kernel,
        None,
        spec,
        Activation::Silu,
        ParallelElementwiseConfig::default(),
        None,
        None,
    )
    .unwrap();

    let nchwc_input = nhwc_to_nchwc(&nhwc_input, block).unwrap();
    let nchwc_out = conv2d_nchwc_pointwise_with_activation_prepacked(
        &nchwc_input,
        &kernel,
        None,
        c_in,
        Activation::Silu,
        ParallelElementwiseConfig::default(),
        None,
        None,
    )
    .unwrap();

    let nchwc_to_nhwc_out = crate::nchwc_to_nhwc(&nchwc_out, c_out).unwrap();
    super::assert_slice_close(nchwc_to_nhwc_out.data(), nhwc_out.data(), 1e-5);
}

#[test]
fn nchwc_pointwise_rejects_non_5d_input() {
    let bad = Tensor::from_vec(vec![1, 3, 4, 4], vec![0.0; 48])
        .unwrap()
        .with_layout(Layout::NHWC);
    let kernel = make_pointwise_kernel(3, 3);
    let err = conv2d_nchwc_pointwise_with_activation_prepacked(
        &bad,
        &kernel,
        None,
        3,
        Activation::None,
        ParallelElementwiseConfig::default(),
        None,
        None,
    );
    assert!(err.is_err());
}

#[test]
fn nchwc_3x3_conv_matches_nhwc() {
    let n = 1;
    let h = 6;
    let w = 6;
    let c_in = 8;
    let c_out = 16;
    let block = 8;
    let kh = 3;
    let kw = 3;

    let nhwc_input = make_nhwc(n, h, w, c_in);
    let kernel_data: Vec<f32> = (0..kh * kw * c_in * c_out)
        .map(|i| ((i as f32) * 0.019).sin())
        .collect();
    let kernel = Tensor::from_vec(vec![kh, kw, c_in, c_out], kernel_data).unwrap();
    let bias = make_bias(c_out);

    let spec = Conv2dSpec {
        stride_h: 1,
        stride_w: 1,
    };
    let nhwc_out = conv2d_nhwc_with_activation_prepacked(
        &nhwc_input,
        &kernel,
        Some(&bias),
        spec,
        Activation::Relu,
        ParallelElementwiseConfig::default(),
        None,
        None,
    )
    .unwrap();

    let nchwc_input = nhwc_to_nchwc(&nhwc_input, block).unwrap();
    let nchwc_out = conv2d_nchwc_with_activation_prepacked(
        &nchwc_input,
        &kernel,
        Some(&bias),
        c_in,
        spec,
        Activation::Relu,
        ParallelElementwiseConfig::default(),
        None,
        None,
    )
    .unwrap();
    assert_eq!(nchwc_out.layout(), Layout::NCHWc { block: block as u8 });

    let nchwc_to_nhwc_out = crate::nchwc_to_nhwc(&nchwc_out, c_out).unwrap();
    assert_eq!(nchwc_to_nhwc_out.shape(), nhwc_out.shape());
    super::assert_slice_close(nchwc_to_nhwc_out.data(), nhwc_out.data(), 1e-4);
}

#[test]
fn nchwc_3x3_stride_2_matches_nhwc() {
    let n = 1;
    let h = 8;
    let w = 8;
    let c_in = 8;
    let c_out = 8;
    let block = 8;
    let kh = 3;
    let kw = 3;

    let nhwc_input = make_nhwc(n, h, w, c_in);
    let kernel_data: Vec<f32> = (0..kh * kw * c_in * c_out)
        .map(|i| ((i as f32) * 0.023).cos())
        .collect();
    let kernel = Tensor::from_vec(vec![kh, kw, c_in, c_out], kernel_data).unwrap();

    let spec = Conv2dSpec {
        stride_h: 2,
        stride_w: 2,
    };
    let nhwc_out = conv2d_nhwc_with_activation_prepacked(
        &nhwc_input,
        &kernel,
        None,
        spec,
        Activation::None,
        ParallelElementwiseConfig::default(),
        None,
        None,
    )
    .unwrap();

    let nchwc_input = nhwc_to_nchwc(&nhwc_input, block).unwrap();
    let nchwc_out = conv2d_nchwc_with_activation_prepacked(
        &nchwc_input,
        &kernel,
        None,
        c_in,
        spec,
        Activation::None,
        ParallelElementwiseConfig::default(),
        None,
        None,
    )
    .unwrap();

    let nchwc_to_nhwc_out = crate::nchwc_to_nhwc(&nchwc_out, c_out).unwrap();
    assert_eq!(nchwc_to_nhwc_out.shape(), nhwc_out.shape());
    super::assert_slice_close(nchwc_to_nhwc_out.data(), nhwc_out.data(), 1e-4);
}

#[test]
fn nchwc_max_pool_matches_nhwc() {
    use crate::core::ops::Pool2dSpec;
    let n = 1;
    let h = 6;
    let w = 6;
    let c = 8;
    let block = 8;
    let nhwc_input = make_nhwc(n, h, w, c);
    let spec = Pool2dSpec {
        kernel_h: 2,
        kernel_w: 2,
        stride_h: 2,
        stride_w: 2,
    };
    let nhwc_out = crate::core::ops::max_pool2d_nhwc_with_config_and_pool(
        &nhwc_input,
        spec,
        ParallelElementwiseConfig::default(),
        None,
    )
    .unwrap();
    let nchwc_input = nhwc_to_nchwc(&nhwc_input, block).unwrap();
    let nchwc_out = crate::max_pool2d_nchwc(
        &nchwc_input,
        c,
        spec,
        ParallelElementwiseConfig::default(),
        None,
    )
    .unwrap();
    let back = crate::nchwc_to_nhwc(&nchwc_out, c).unwrap();
    super::assert_slice_close(back.data(), nhwc_out.data(), 1e-6);
}

#[test]
fn nchwc_relu_is_layout_preserving() {
    let n = 1;
    let h = 3;
    let w = 3;
    let c = 8;
    let block = 8;
    let mut data = vec![];
    for i in 0..n * h * w * c {
        data.push(if i % 3 == 0 { -1.0 } else { i as f32 * 0.1 });
    }
    let nhwc_input = Tensor::from_vec(vec![n, h, w, c], data)
        .unwrap()
        .with_layout(Layout::NHWC);
    let nchwc_input = nhwc_to_nchwc(&nhwc_input, block).unwrap();
    let relud =
        crate::relu_nchwc(&nchwc_input, ParallelElementwiseConfig::default(), None).unwrap();
    assert_eq!(relud.layout(), Layout::NCHWc { block: block as u8 });
    // Negative values clamp to 0.
    for v in relud.data() {
        assert!(*v >= 0.0);
    }
}

#[test]
fn nchwc_add_elementwise_preserves_layout() {
    let n = 1;
    let h = 2;
    let w = 2;
    let c = 16;
    let block = 8;
    let a_nhwc = make_nhwc(n, h, w, c);
    let b_nhwc = make_nhwc(n, h, w, c);
    let a = nhwc_to_nchwc(&a_nhwc, block).unwrap();
    let b = nhwc_to_nchwc(&b_nhwc, block).unwrap();
    let sum = crate::add_nchwc(&a, &b, ParallelElementwiseConfig::default(), None).unwrap();
    assert_eq!(sum.layout(), Layout::NCHWc { block: block as u8 });
    // Sum is 2× the original.
    let back = crate::nchwc_to_nhwc(&sum, c).unwrap();
    for (i, v) in back.data().iter().enumerate() {
        let expected = a_nhwc.data()[i] + b_nhwc.data()[i];
        assert!(
            (v - expected).abs() < 1e-5,
            "mismatch at {i}: {v} vs {expected}"
        );
    }
}

#[test]
fn nchwc_silu_is_layout_preserving() {
    let n = 1;
    let h = 2;
    let w = 2;
    let c = 8;
    let block = 8;
    let nhwc_input = make_nhwc(n, h, w, c);
    let nchwc_input = nhwc_to_nchwc(&nhwc_input, block).unwrap();
    let silu_out = crate::silu_nchwc(&nchwc_input).unwrap();
    assert_eq!(silu_out.layout(), Layout::NCHWc { block: block as u8 });
    assert_eq!(silu_out.shape(), nchwc_input.shape());
}

#[test]
fn nchwc_pointwise_rejects_wrong_kernel_rank() {
    let n = 1;
    let h = 2;
    let w = 2;
    let c_in = 8;
    let block = 8;
    let nhwc_input = make_nhwc(n, h, w, c_in);
    let nchwc_input = nhwc_to_nchwc(&nhwc_input, block).unwrap();

    let bad_kernel = Tensor::from_vec(vec![2, 2, c_in, 8], vec![0.0; 2 * 2 * c_in * 8]).unwrap();
    let err = conv2d_nchwc_pointwise_with_activation_prepacked(
        &nchwc_input,
        &bad_kernel,
        None,
        c_in,
        Activation::None,
        ParallelElementwiseConfig::default(),
        None,
        None,
    );
    assert!(err.is_err());
}
