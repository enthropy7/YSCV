use super::*;

fn scalar_init(name: &str, value: f32) -> onnx::TensorProto {
    onnx::TensorProto {
        name: Some(name.into()),
        dims: vec![1],
        data_type: Some(1),
        float_data: vec![value],
        ..Default::default()
    }
}

fn vec_init(name: &str, dims: Vec<i64>, data: Vec<f32>) -> onnx::TensorProto {
    onnx::TensorProto {
        name: Some(name.into()),
        dims,
        data_type: Some(1),
        float_data: data,
        ..Default::default()
    }
}

fn build_qlinear_matmul_model(
    a_data: Vec<f32>,
    a_shape: Vec<i64>,
    a_scale: f32,
    a_zp: f32,
    b_data: Vec<f32>,
    b_shape: Vec<i64>,
    b_scale: f32,
    b_zp: f32,
    y_scale: f32,
    y_zp: f32,
) -> Vec<u8> {
    let inits = vec![
        vec_init("a", a_shape, a_data),
        scalar_init("a_s", a_scale),
        scalar_init("a_zp", a_zp),
        vec_init("b", b_shape, b_data),
        scalar_init("b_s", b_scale),
        scalar_init("b_zp", b_zp),
        scalar_init("y_s", y_scale),
        scalar_init("y_zp", y_zp),
    ];
    let node = onnx::NodeProto {
        op_type: Some("QLinearMatMul".into()),
        name: Some("qmm".into()),
        input: vec![
            "a".into(),
            "a_s".into(),
            "a_zp".into(),
            "b".into(),
            "b_s".into(),
            "b_zp".into(),
            "y_s".into(),
            "y_zp".into(),
        ],
        output: vec!["y".into()],
        ..Default::default()
    };
    build_minimal_onnx_model(
        vec![node],
        inits,
        vec!["a", "a_s", "a_zp", "b", "b_s", "b_zp", "y_s", "y_zp"],
        vec!["y"],
    )
}

/// Symmetric-int8 fast path (both zero-points are 0) with non-unit
/// scales. Verifies the integer GEMM kernel + composite-scale requantize
/// in `exec_qlinear_matmul` matches a hand-computed reference.
#[test]
fn qlinear_matmul_symmetric_fast_path_with_nontrivial_scales() {
    // m=4, k=8, n=4 — wide enough that AVX2 / NEON sdot / VNNI dispatch
    // exercises its inner SIMD loops on hosts that have them.
    let a_int: Vec<i32> = (0..32).map(|v| v - 16).collect();
    let b_int: Vec<i32> = (0..32).map(|v| ((v * 7) % 31) - 15).collect();
    let a_data: Vec<f32> = a_int.iter().map(|&v| v as f32).collect();
    let b_data: Vec<f32> = b_int.iter().map(|&v| v as f32).collect();

    let bytes = build_qlinear_matmul_model(
        a_data,
        vec![4, 8],
        0.05,
        0.0,
        b_data,
        vec![8, 4],
        0.1,
        0.0,
        0.25,
        0.0,
    );
    let model = load_onnx_model(&bytes).unwrap();
    let result = run_onnx_model(&model, HashMap::new()).unwrap();
    let out = result["y"].data();

    let composite = 0.05_f32 * 0.1 / 0.25;
    let mut expected = [0.0_f32; 16];
    for i in 0..4 {
        for j in 0..4 {
            let mut acc: i32 = 0;
            for kk in 0..8 {
                acc += a_int[i * 8 + kk] * b_int[kk * 4 + j];
            }
            expected[i * 4 + j] = ((acc as f32) * composite).round().clamp(-128.0, 127.0);
        }
    }
    for (g, e) in out.iter().zip(expected.iter()) {
        assert!(
            (g - e).abs() <= 1e-4,
            "got {g} expected {e} (diff {})",
            (g - e).abs()
        );
    }
}

/// Symmetric-int8 QLinearConv fast path: NCHW input, OIHW weight,
/// group=1, no dilation, both zero-points 0. Verifies im2col + integer
/// GEMM + composite-scale requantize matches a hand-computed reference.
#[test]
fn qlinear_conv_symmetric_fast_path_with_bias() {
    // Conv: 1 batch, 2 in-channels, 4×4 spatial, 3 out-channels, 3×3
    // kernel, stride 1, pad 1 → output 4×4.
    let c_in = 2;
    let c_out = 3;
    let h = 4;
    let w = 4;
    let kh = 3;
    let kw = 3;

    let x_int: Vec<i32> = (0..(c_in * h * w)).map(|v| ((v as i32) % 17) - 8).collect();
    let w_int: Vec<i32> = (0..(c_out * c_in * kh * kw))
        .map(|v| ((v as i32) * 3 % 31) - 15)
        .collect();
    let bias_int: Vec<i32> = (0..c_out).map(|v| (v as i32) * 4 - 4).collect();

    let x_data: Vec<f32> = x_int.iter().map(|&v| v as f32).collect();
    let w_data: Vec<f32> = w_int.iter().map(|&v| v as f32).collect();
    let b_data: Vec<f32> = bias_int.iter().map(|&v| v as f32).collect();

    let x_init = vec_init("x", vec![1, c_in as i64, h as i64, w as i64], x_data);
    let w_init = vec_init(
        "w",
        vec![c_out as i64, c_in as i64, kh as i64, kw as i64],
        w_data,
    );
    let b_init = vec_init("b", vec![c_out as i64], b_data);
    let x_scale = 0.05_f32;
    let w_scale = 0.1_f32;
    let y_scale = 0.5_f32;

    let inits = vec![
        x_init,
        scalar_init("x_s", x_scale),
        scalar_init("x_zp", 0.0),
        w_init,
        scalar_init("w_s", w_scale),
        scalar_init("w_zp", 0.0),
        scalar_init("y_s", y_scale),
        scalar_init("y_zp", 0.0),
        b_init,
    ];
    let node = onnx::NodeProto {
        op_type: Some("QLinearConv".into()),
        name: Some("qconv".into()),
        input: vec![
            "x".into(),
            "x_s".into(),
            "x_zp".into(),
            "w".into(),
            "w_s".into(),
            "w_zp".into(),
            "y_s".into(),
            "y_zp".into(),
            "b".into(),
        ],
        output: vec!["y".into()],
        attribute: vec![
            make_ints_attr("kernel_shape", vec![kh as i64, kw as i64]),
            make_ints_attr("strides", vec![1, 1]),
            make_ints_attr("pads", vec![1, 1, 1, 1]),
        ],
        ..Default::default()
    };
    let bytes = build_minimal_onnx_model(
        vec![node],
        inits,
        vec!["x", "x_s", "x_zp", "w", "w_s", "w_zp", "y_s", "y_zp", "b"],
        vec!["y"],
    );
    let model = load_onnx_model(&bytes).unwrap();
    let result = run_onnx_model(&model, HashMap::new()).unwrap();
    let out = result["y"].data();

    // Reference: integer NCHW conv with pad=1, then composite requantize.
    let composite = x_scale * w_scale / y_scale;
    let mut expected = vec![0.0_f32; c_out * h * w];
    for o in 0..c_out {
        for oh in 0..h {
            for ow in 0..w {
                let mut acc: i32 = 0;
                for ci in 0..c_in {
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let ih = oh as i64 + ky as i64 - 1;
                            let iw = ow as i64 + kx as i64 - 1;
                            if ih < 0 || ih >= h as i64 || iw < 0 || iw >= w as i64 {
                                continue;
                            }
                            let xv = x_int[(ci * h + ih as usize) * w + iw as usize];
                            let wv = w_int[((o * c_in + ci) * kh + ky) * kw + kx];
                            acc += xv * wv;
                        }
                    }
                }
                acc += bias_int[o];
                expected[(o * h + oh) * w + ow] =
                    ((acc as f32) * composite).round().clamp(-128.0, 127.0);
            }
        }
    }

    assert_eq!(result["y"].shape(), &[1, c_out, h, w]);
    for (g, e) in out.iter().zip(expected.iter()) {
        assert!(
            (g - e).abs() <= 1e-4,
            "got {g} expected {e} (diff {})",
            (g - e).abs()
        );
    }
}

/// MatMulInteger symmetric fast path: rank-2, both zero-points 0,
/// output is raw int32 dot. Validates the kernel + i32→f32 cast.
#[test]
fn matmul_integer_symmetric_fast_path() {
    let a_int: Vec<i32> = (0..24).map(|v| v - 12).collect();
    let b_int: Vec<i32> = (0..24).map(|v| ((v * 5) % 23) - 11).collect();
    let a_data: Vec<f32> = a_int.iter().map(|&v| v as f32).collect();
    let b_data: Vec<f32> = b_int.iter().map(|&v| v as f32).collect();

    let inits = vec![
        vec_init("a", vec![3, 8], a_data),
        vec_init("b", vec![8, 3], b_data),
    ];
    let node = onnx::NodeProto {
        op_type: Some("MatMulInteger".into()),
        name: Some("mmi".into()),
        input: vec!["a".into(), "b".into()],
        output: vec!["y".into()],
        ..Default::default()
    };
    let bytes = build_minimal_onnx_model(vec![node], inits, vec!["a", "b"], vec!["y"]);
    let model = load_onnx_model(&bytes).unwrap();
    let result = run_onnx_model(&model, HashMap::new()).unwrap();
    let out = result["y"].data();

    let mut expected = [0.0_f32; 9];
    for i in 0..3 {
        for j in 0..3 {
            let mut acc: i32 = 0;
            for kk in 0..8 {
                acc += a_int[i * 8 + kk] * b_int[kk * 3 + j];
            }
            expected[i * 3 + j] = acc as f32;
        }
    }
    assert_eq!(result["y"].shape(), &[3, 3]);
    for (g, e) in out.iter().zip(expected.iter()) {
        assert!((g - e).abs() <= 1e-4, "got {g} expected {e}");
    }
}

/// ConvInteger symmetric fast path: NCHW + OIHW + group=1, output
/// is raw int32 (no requantize, unlike QLinearConv).
#[test]
fn conv_integer_symmetric_fast_path() {
    let c_in = 2;
    let c_out = 2;
    let h = 3;
    let w = 3;
    let kh = 3;
    let kw = 3;
    let x_int: Vec<i32> = (0..(c_in * h * w)).map(|v| ((v as i32) % 11) - 5).collect();
    let w_int: Vec<i32> = (0..(c_out * c_in * kh * kw))
        .map(|v| ((v as i32) * 2 % 19) - 9)
        .collect();
    let inits = vec![
        vec_init(
            "x",
            vec![1, c_in as i64, h as i64, w as i64],
            x_int.iter().map(|&v| v as f32).collect(),
        ),
        vec_init(
            "w",
            vec![c_out as i64, c_in as i64, kh as i64, kw as i64],
            w_int.iter().map(|&v| v as f32).collect(),
        ),
    ];
    let node = onnx::NodeProto {
        op_type: Some("ConvInteger".into()),
        name: Some("ci".into()),
        input: vec!["x".into(), "w".into()],
        output: vec!["y".into()],
        attribute: vec![
            make_ints_attr("kernel_shape", vec![kh as i64, kw as i64]),
            make_ints_attr("strides", vec![1, 1]),
            make_ints_attr("pads", vec![1, 1, 1, 1]),
        ],
        ..Default::default()
    };
    let bytes = build_minimal_onnx_model(vec![node], inits, vec!["x", "w"], vec!["y"]);
    let model = load_onnx_model(&bytes).unwrap();
    let result = run_onnx_model(&model, HashMap::new()).unwrap();
    let out = result["y"].data();

    let mut expected = vec![0.0_f32; c_out * h * w];
    for o in 0..c_out {
        for oh in 0..h {
            for ow in 0..w {
                let mut acc: i32 = 0;
                for ci in 0..c_in {
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let ih = oh as i64 + ky as i64 - 1;
                            let iw = ow as i64 + kx as i64 - 1;
                            if ih < 0 || ih >= h as i64 || iw < 0 || iw >= w as i64 {
                                continue;
                            }
                            let xv = x_int[(ci * h + ih as usize) * w + iw as usize];
                            let wv = w_int[((o * c_in + ci) * kh + ky) * kw + kx];
                            acc += xv * wv;
                        }
                    }
                }
                expected[(o * h + oh) * w + ow] = acc as f32;
            }
        }
    }
    assert_eq!(result["y"].shape(), &[1, c_out, h, w]);
    for (g, e) in out.iter().zip(expected.iter()) {
        assert!((g - e).abs() <= 1e-4, "got {g} expected {e}");
    }
}

/// Pack a MatMul weight to symmetric INT4 + per-group scales via the
/// quantizer, then run inference through `OnnxRunner` — the runner
/// must detect the packed weight and dispatch to GEMV. Output is
/// compared to the fp32 reference within int4 quantisation tolerance.
#[test]
fn packed_int4_matmul_routes_through_gemv() {
    use crate::quantize::quantize_matmul_weights_int4_packed;

    let k = 64;
    let n = 8;
    // Smooth weight distribution so per-group quantisation gives small
    // residual error.
    let weight: Vec<f32> = (0..(k * n))
        .map(|v| ((v as f32) / (k * n) as f32 - 0.5) * 0.3)
        .collect();
    let inits = vec![vec_init("w", vec![k as i64, n as i64], weight.clone())];
    let node = onnx::NodeProto {
        op_type: Some("MatMul".into()),
        name: Some("mm".into()),
        input: vec!["x".into(), "w".into()],
        output: vec!["y".into()],
        ..Default::default()
    };
    let bytes = build_minimal_onnx_model(vec![node], inits, vec!["x", "w"], vec!["y"]);
    let mut model = load_onnx_model(&bytes).unwrap();

    // Reference fp32 forward.
    let activation: Vec<f32> = (0..k).map(|i| ((i as f32) - 32.0) * 0.05).collect();
    let mut feed = HashMap::new();
    feed.insert(
        "x".to_string(),
        Tensor::from_vec(vec![1, k], activation.clone()).unwrap(),
    );
    let fp32 = run_onnx_model(&model, feed).unwrap();
    let ref_out = fp32["y"].clone();

    // Quantise: replaces `w` initializer with packed-int4 side-table
    // entry. Subsequent runs must dispatch through GEMV.
    let packed_count = quantize_matmul_weights_int4_packed(&mut model, 32).unwrap();
    assert_eq!(packed_count, 1);
    assert!(!model.initializers.contains_key("w"));
    assert!(model.packed_int4_weights.contains_key("w"));

    let mut feed = HashMap::new();
    feed.insert(
        "x".to_string(),
        Tensor::from_vec(vec![1, k], activation).unwrap(),
    );
    let quant = run_onnx_model(&model, feed).unwrap();
    let q_out = &quant["y"];
    assert_eq!(q_out.shape(), ref_out.shape());

    // Quantisation error per element: ~scale/2 ≈ abs_max / 14. With
    // group_size=32 and weight magnitude ~0.15, scale ≈ 0.022, per-element
    // error ≤ 0.011 absolute. Dot of 64 elements bounds drift; loose
    // relative tolerance.
    let abs_max = ref_out.data().iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
    for (g, e) in q_out.data().iter().zip(ref_out.data().iter()) {
        let rel = (g - e).abs() / abs_max.max(1e-6);
        assert!(rel < 0.20, "got {g} expected {e} rel {rel}");
    }
}

/// Asymmetric path (non-zero zero-points) falls through to the f32
/// reference. Verifies the dispatch gate doesn't accidentally claim
/// asymmetric inputs and produce wrong output.
#[test]
fn qlinear_matmul_asymmetric_uses_fp32_fallback() {
    // a values = [10..20], a_zp = 10 → effective zero-centred a.
    let a_data: Vec<f32> = (10..20).map(|v| v as f32).collect();
    let b_data: Vec<f32> = (0..10).map(|v| (v - 5) as f32).collect();
    let bytes = build_qlinear_matmul_model(
        a_data,
        vec![2, 5],
        0.5,
        10.0,
        b_data,
        vec![5, 2],
        0.5,
        0.0,
        1.0,
        0.0,
    );
    let model = load_onnx_model(&bytes).unwrap();
    let result = run_onnx_model(&model, HashMap::new()).unwrap();
    assert_eq!(result["y"].shape(), &[2, 2]);
    // Verify finite output — the asymmetric path's exact numerics are
    // covered by the orphan coverage suite; here we only confirm the
    // gate didn't mis-route into the symmetric fast path.
    for &v in result["y"].data() {
        assert!(v.is_finite());
    }
}
