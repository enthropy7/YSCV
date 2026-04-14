//! Regression tests for the runtime Conv → Sigmoid → Mul (SiLU) fusion path.
//!
//! These tests protect a previously-shipped bug where the depthwise and grouped
//! Conv branches in [`crate::runner::conv::exec_conv`] forgot to apply
//! `silu_inplace` after the convolution, while the regular `group == 1` branch
//! did. The bug surfaced as a YOLO11n model losing detections (34 → 9).
//!
//! Each test builds a `Conv → Sigmoid → Mul` graph that the runtime fusion
//! detector at `runner::mod` collapses into a single
//! `exec_conv(..., Activation::Silu)` call. The reference output is computed
//! by hand in NCHW (the same layout `run_onnx_model` returns at graph
//! boundaries), then asserted element-wise.

use super::*;

/// Apply SiLU `x * sigmoid(x)` element-wise to a flat NCHW buffer.
fn silu_in_place(data: &mut [f32]) {
    for v in data.iter_mut() {
        *v *= 1.0 / (1.0 + (-*v).exp());
    }
}

/// Reference NCHW grouped/depthwise/regular conv2d, valid padding, stride 1,
/// no bias. `weight` shape is `[oc, ic_per_group, kh, kw]`. Returns flat NCHW
/// data of shape `[1, oc, oh, ow]`.
fn reference_grouped_conv2d_nchw(
    input: &[f32],
    n: usize,
    ic: usize,
    ih: usize,
    iw: usize,
    weight: &[f32],
    oc: usize,
    kh: usize,
    kw: usize,
    group: usize,
) -> (Vec<f32>, usize, usize) {
    assert_eq!(n, 1, "reference helper only handles N=1");
    assert_eq!(ic % group, 0);
    assert_eq!(oc % group, 0);
    let ic_per_g = ic / group;
    let oc_per_g = oc / group;
    let oh = ih - kh + 1;
    let ow = iw - kw + 1;
    let mut out = vec![0.0f32; oc * oh * ow];
    for g in 0..group {
        for ocg in 0..oc_per_g {
            let oc_abs = g * oc_per_g + ocg;
            for orow in 0..oh {
                for ocol in 0..ow {
                    let mut acc = 0.0f32;
                    for icg in 0..ic_per_g {
                        let ic_abs = g * ic_per_g + icg;
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let in_r = orow + ki;
                                let in_c = ocol + kj;
                                let in_idx = ((ic_abs * ih) + in_r) * iw + in_c;
                                let w_idx = ((oc_abs * ic_per_g + icg) * kh + ki) * kw + kj;
                                acc += input[in_idx] * weight[w_idx];
                            }
                        }
                    }
                    let out_idx = (oc_abs * oh + orow) * ow + ocol;
                    out[out_idx] = acc;
                }
            }
        }
    }
    (out, oh, ow)
}

/// Build the Conv → Sigmoid → Mul graph for a given `group` and weight, run
/// it through `run_onnx_model`, and compare against the hand-rolled reference.
fn run_conv_silu_graph(
    input_shape: [usize; 4], // NCHW
    input_data: Vec<f32>,
    weight_shape: [i64; 4], // [O, I/group, KH, KW]
    weight_data: Vec<f32>,
    group: i64,
) -> (Vec<f32>, Vec<f32>) {
    let conv_w = onnx::TensorProto {
        name: Some("w".into()),
        dims: weight_shape.to_vec(),
        data_type: Some(1),
        float_data: weight_data.clone(),
        ..Default::default()
    };
    let conv_node = onnx::NodeProto {
        op_type: Some("Conv".into()),
        name: Some("conv".into()),
        input: vec!["x".into(), "w".into()],
        output: vec!["conv_out".into()],
        attribute: vec![
            make_ints_attr("kernel_shape", vec![weight_shape[2], weight_shape[3]]),
            make_ints_attr("strides", vec![1, 1]),
            make_int_attr("group", group),
        ],
        ..Default::default()
    };
    let sigmoid_node = onnx::NodeProto {
        op_type: Some("Sigmoid".into()),
        name: Some("sig".into()),
        input: vec!["conv_out".into()],
        output: vec!["sig_out".into()],
        ..Default::default()
    };
    // Mul takes (conv_out, sig_out) — the runtime fusion detector accepts
    // either operand order. We use (conv_out, sig_out) to exercise the
    // canonical SiLU shape `x * sigmoid(x)`.
    let mul_node = onnx::NodeProto {
        op_type: Some("Mul".into()),
        name: Some("mul".into()),
        input: vec!["conv_out".into(), "sig_out".into()],
        output: vec!["y".into()],
        ..Default::default()
    };

    let bytes = build_minimal_onnx_model(
        vec![conv_node, sigmoid_node, mul_node],
        vec![conv_w],
        vec!["x", "w"],
        vec!["y"],
    );
    let model = load_onnx_model(&bytes).unwrap();

    let input_tensor = Tensor::from_vec(input_shape.to_vec(), input_data.clone()).unwrap();
    let mut feed = HashMap::new();
    feed.insert("x".to_string(), input_tensor);
    let result = run_onnx_model(&model, feed).unwrap();
    let actual = result["y"].data().to_vec();

    // Reference: NCHW conv → SiLU.
    let (mut reference, oh, ow) = reference_grouped_conv2d_nchw(
        &input_data,
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
        &weight_data,
        weight_shape[0] as usize,
        weight_shape[2] as usize,
        weight_shape[3] as usize,
        group as usize,
    );
    silu_in_place(&mut reference);

    let expected_shape = [input_shape[0], weight_shape[0] as usize, oh, ow];
    assert_eq!(
        result["y"].shape(),
        &expected_shape,
        "graph output shape mismatch"
    );

    (actual, reference)
}

fn make_input_pattern(n: usize) -> Vec<f32> {
    // Centered, deterministic, exercises both signs so SiLU's negative-side
    // behaviour is observable.
    (0..n).map(|i| (i as f32 - n as f32 / 2.0) / 8.0).collect()
}

fn make_weight_pattern(n: usize) -> Vec<f32> {
    // Small, deterministic, mixes signs.
    (0..n)
        .map(|i| ((i as f32) * 0.071 - 0.31).sin() * 0.5)
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "length mismatch: {} vs {}",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff < tol,
            "element {i}: actual {a} vs expected {e}, diff {diff} >= tol {tol}"
        );
    }
}

#[test]
fn depthwise_conv_silu_fusion_matches_reference() {
    // Depthwise: group == C_in == C_out, weight shape [C, 1, KH, KW].
    // This routes through the depthwise branch in
    // `runner::conv::exec_conv` (the formerly-buggy arm at lines 152-154).
    let input_shape = [1usize, 4, 4, 4]; // NCHW
    let input_data = make_input_pattern(input_shape.iter().product());
    let weight_shape = [4i64, 1, 3, 3];
    let weight_data = make_weight_pattern(4 * 3 * 3);
    let group = 4;

    let (actual, expected) =
        run_conv_silu_graph(input_shape, input_data, weight_shape, weight_data, group);
    // Tolerance: 1e-4 absorbs FMA reordering between the runner's NHWC
    // depthwise/grouped/regular paths and the NCHW reference. The bug we
    // guard against (missing `silu_inplace`) shifts values by O(0.1)+, well
    // above this threshold, so the safety margin is ~1000×.
    assert_close(&actual, &expected, 1e-4);
}

#[test]
fn grouped_conv_silu_fusion_matches_reference() {
    // Grouped (non-depthwise): group=2, C_in=4, C_out=4, weight [4, 2, 3, 3].
    // This routes through the grouped branch in
    // `runner::conv::exec_conv` (the formerly-buggy arm at lines 238-239).
    let input_shape = [1usize, 4, 4, 4]; // NCHW
    let input_data = make_input_pattern(input_shape.iter().product());
    let weight_shape = [4i64, 2, 3, 3];
    let weight_data = make_weight_pattern(4 * 2 * 3 * 3);
    let group = 2;

    let (actual, expected) =
        run_conv_silu_graph(input_shape, input_data, weight_shape, weight_data, group);
    // Tolerance: 1e-4 absorbs FMA reordering between the runner's NHWC
    // depthwise/grouped/regular paths and the NCHW reference. The bug we
    // guard against (missing `silu_inplace`) shifts values by O(0.1)+, well
    // above this threshold, so the safety margin is ~1000×.
    assert_close(&actual, &expected, 1e-4);
}

#[test]
fn regular_conv_silu_fusion_matches_reference() {
    // group=1, weight [4, 4, 3, 3]. This is the control: the regular branch
    // in `runner::conv::exec_conv` (lines 105-107) was always correct, and
    // the test confirms the reference computation matches it. If all three
    // tests fail together the bug is in the reference, not in the runner.
    let input_shape = [1usize, 4, 4, 4]; // NCHW
    let input_data = make_input_pattern(input_shape.iter().product());
    let weight_shape = [4i64, 4, 3, 3];
    let weight_data = make_weight_pattern(4 * 4 * 3 * 3);
    let group = 1;

    let (actual, expected) =
        run_conv_silu_graph(input_shape, input_data, weight_shape, weight_data, group);
    // Tolerance: 1e-4 absorbs FMA reordering between the runner's NHWC
    // depthwise/grouped/regular paths and the NCHW reference. The bug we
    // guard against (missing `silu_inplace`) shifts values by O(0.1)+, well
    // above this threshold, so the safety margin is ~1000×.
    assert_close(&actual, &expected, 1e-4);
}
