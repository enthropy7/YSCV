use super::*;

#[test]
fn onnx_gather_axis0_2d_multi_index() {
    // input [3, 4], indices [2] picks rows 2 and 0 → output [2, 4].
    let input = Tensor::from_vec(
        vec![3, 4],
        vec![
            0.0, 1.0, 2.0, 3.0, // row 0
            4.0, 5.0, 6.0, 7.0, // row 1
            8.0, 9.0, 10.0, 11.0, // row 2
        ],
    )
    .unwrap();
    let indices = Tensor::from_vec(vec![2], vec![2.0, 0.0]).unwrap();
    let axis = onnx::AttributeProto {
        name: Some("axis".into()),
        r#type: Some(2),
        i: Some(0),
        ..Default::default()
    };
    let out = run_single_op(
        "Gather",
        vec![("data", input), ("indices", indices)],
        vec![],
        vec![axis],
        vec!["data", "indices"],
        "y",
    );
    assert_eq!(out.shape(), &[2, 4]);
    assert_eq!(out.data(), &[8.0, 9.0, 10.0, 11.0, 0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn onnx_gather_axis1_with_2d_indices() {
    // input [2, 3], indices [[2, 0], [1, 2]] → output [2, 2, 2].
    let input = Tensor::from_vec(vec![2, 3], vec![10.0, 11.0, 12.0, 20.0, 21.0, 22.0]).unwrap();
    let indices = Tensor::from_vec(vec![2, 2], vec![2.0, 0.0, 1.0, 2.0]).unwrap();
    let axis = onnx::AttributeProto {
        name: Some("axis".into()),
        r#type: Some(2),
        i: Some(1),
        ..Default::default()
    };
    let out = run_single_op(
        "Gather",
        vec![("data", input), ("indices", indices)],
        vec![],
        vec![axis],
        vec!["data", "indices"],
        "y",
    );
    assert_eq!(out.shape(), &[2, 2, 2]);
    // row 0: [12,10,11,12], row 1: [22,20,21,22].
    assert_eq!(
        out.data(),
        &[12.0, 10.0, 11.0, 12.0, 22.0, 20.0, 21.0, 22.0]
    );
}

#[test]
fn onnx_gather_negative_indices_and_axis() {
    // axis = -1 (last), negative index wraps.
    let input = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let indices = Tensor::from_vec(vec![2], vec![-1.0, 0.0]).unwrap();
    let axis = onnx::AttributeProto {
        name: Some("axis".into()),
        r#type: Some(2),
        i: Some(-1),
        ..Default::default()
    };
    let out = run_single_op(
        "Gather",
        vec![("data", input), ("indices", indices)],
        vec![],
        vec![axis],
        vec!["data", "indices"],
        "y",
    );
    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(out.data(), &[3.0, 1.0, 6.0, 4.0]);
}

#[test]
fn onnx_gather_embedding_lookup_negative_index() {
    // axis=0, rank-1 input — embedding-table fast path, with negative idx.
    let input = Tensor::from_vec(vec![5], vec![10.0, 20.0, 30.0, 40.0, 50.0]).unwrap();
    let indices = Tensor::from_vec(vec![3], vec![-1.0, 0.0, -3.0]).unwrap();
    let axis = onnx::AttributeProto {
        name: Some("axis".into()),
        r#type: Some(2),
        i: Some(0),
        ..Default::default()
    };
    let out = run_single_op(
        "Gather",
        vec![("data", input), ("indices", indices)],
        vec![],
        vec![axis],
        vec!["data", "indices"],
        "y",
    );
    assert_eq!(out.shape(), &[3]);
    assert_eq!(out.data(), &[50.0, 10.0, 30.0]);
}

#[test]
fn onnx_gather_elements_axis0() {
    let input = Tensor::from_vec(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let indices = Tensor::from_vec(vec![2, 2], vec![2.0, 0.0, 0.0, 1.0]).unwrap();
    let axis_attr = onnx::AttributeProto {
        name: Some("axis".into()),
        r#type: Some(2),
        i: Some(0),
        ..Default::default()
    };
    let out = run_single_op(
        "GatherElements",
        vec![("data", input), ("indices", indices)],
        vec![],
        vec![axis_attr],
        vec!["data", "indices"],
        "y",
    );
    assert_eq!(out.data(), &[5.0, 2.0, 1.0, 4.0]);
}

#[test]
fn onnx_scatter_elements_axis0() {
    let input = Tensor::from_vec(vec![3, 3], vec![0.0; 9]).unwrap();
    let indices = Tensor::from_vec(vec![2, 3], vec![1.0, 0.0, 2.0, 0.0, 2.0, 1.0]).unwrap();
    let updates = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let axis_attr = onnx::AttributeProto {
        name: Some("axis".into()),
        r#type: Some(2),
        i: Some(0),
        ..Default::default()
    };
    let out = run_single_op(
        "ScatterElements",
        vec![("data", input), ("indices", indices), ("updates", updates)],
        vec![],
        vec![axis_attr],
        vec!["data", "indices", "updates"],
        "y",
    );
    assert_eq!(out.data()[0], 4.0);
    assert_eq!(out.data()[1], 2.0);
}

#[test]
fn onnx_argmax_axis0() {
    let input = Tensor::from_vec(vec![3, 2], vec![1.0, 5.0, 3.0, 2.0, 4.0, 1.0]).unwrap();
    let axis_attr = onnx::AttributeProto {
        name: Some("axis".into()),
        r#type: Some(2),
        i: Some(0),
        ..Default::default()
    };
    let keepdims_attr = onnx::AttributeProto {
        name: Some("keepdims".into()),
        r#type: Some(2),
        i: Some(0),
        ..Default::default()
    };
    let out = run_single_op(
        "ArgMax",
        vec![("data", input)],
        vec![],
        vec![axis_attr, keepdims_attr],
        vec!["data"],
        "y",
    );
    assert_eq!(out.shape(), &[2]);
    assert_eq!(out.data()[0] as usize, 2);
    assert_eq!(out.data()[1] as usize, 0);
}

#[test]
fn onnx_topk_basic() {
    let input = Tensor::from_vec(vec![1, 5], vec![1.0, 4.0, 2.0, 5.0, 3.0]).unwrap();
    let k = Tensor::from_vec(vec![1], vec![3.0]).unwrap();
    let out = run_single_op(
        "TopK",
        vec![("data", input), ("k", k)],
        vec![],
        vec![],
        vec!["data", "k"],
        "values",
    );
    assert_eq!(out.shape(), &[1, 3]);
    assert_eq!(out.data()[0], 5.0);
    assert_eq!(out.data()[1], 4.0);
    assert_eq!(out.data()[2], 3.0);
}

#[test]
fn onnx_range() {
    let out = run_single_op(
        "Range",
        vec![
            ("start", Tensor::from_vec(vec![1], vec![0.0]).unwrap()),
            ("limit", Tensor::from_vec(vec![1], vec![5.0]).unwrap()),
            ("delta", Tensor::from_vec(vec![1], vec![1.0]).unwrap()),
        ],
        vec![],
        vec![],
        vec!["start", "limit", "delta"],
        "y",
    );
    assert_eq!(out.data().len(), 5);
    assert!((out.data()[0]).abs() < 1e-5);
    assert!((out.data()[4] - 4.0).abs() < 1e-5);
}

#[test]
fn onnx_onehot() {
    let indices = Tensor::from_vec(vec![3], vec![0.0, 1.0, 2.0]).unwrap();
    let depth = Tensor::from_vec(vec![1], vec![3.0]).unwrap();
    let values = Tensor::from_vec(vec![2], vec![0.0, 1.0]).unwrap();
    let out = run_single_op(
        "OneHot",
        vec![("i", indices), ("d", depth), ("v", values)],
        vec![],
        vec![],
        vec!["i", "d", "v"],
        "y",
    );
    assert_eq!(out.shape(), &[3, 3]);
    assert_eq!(out.data()[0], 1.0); // [0] -> [1,0,0]
    assert_eq!(out.data()[1], 0.0);
}

#[test]
fn onnx_cumsum_axis0() {
    let input = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let axis = Tensor::scalar(0.0);
    let out = run_single_op(
        "CumSum",
        vec![("data", input), ("axis", axis)],
        vec![],
        vec![],
        vec!["data", "axis"],
        "y",
    );
    assert_eq!(out.data(), &[1.0, 3.0, 6.0, 10.0]);
}

#[test]
fn onnx_dynamic_quantize_linear() {
    let input = Tensor::from_vec(vec![4], vec![0.0, 1.0, 2.0, 3.0]).unwrap();
    let out = run_single_op(
        "DynamicQuantizeLinear",
        vec![("x", input)],
        vec![],
        vec![],
        vec!["x"],
        "y",
    );
    assert_eq!(out.shape(), &[4]);
    for &v in out.data() {
        assert!((0.0..=255.0).contains(&v));
    }
}
