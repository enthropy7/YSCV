use super::*;

// Serialise every test in this module that runs an ONNX model. The
// runner increments process-global `QuantRuntimeStats` atomic counters
// and reads the `YSCV_QUANT_INT8_FAST` env var; the counter-asserting
// test (`qlinear_conv_symmetric_fast_path_with_bias`) and the env-var
// toggling test (`quantized_pw_dw_chain_bitwise_matches_unfused`)
// would otherwise race other concurrent model executions. The mutex
// is poison-tolerant so a panic in one test does not cascade.
static SHARED_STATE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[track_caller]
fn lock_shared_state() -> std::sync::MutexGuard<'static, ()> {
    SHARED_STATE_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

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

#[test]
fn qlinear_matching_dq_relu_q_runs_in_quant_domain() {
    let _guard = lock_shared_state();
    let inits = vec![
        vec_init(
            "xq",
            vec![1, 1, 2, 4],
            vec![-4.0, -1.0, 0.0, 3.0, 7.0, -8.0, 2.0, 1.0],
        ),
        scalar_init("s0", 0.25),
        scalar_init("zp0", 0.0),
        scalar_init("s1", 0.25),
        scalar_init("zp1", 0.0),
    ];
    let nodes = vec![
        onnx::NodeProto {
            op_type: Some("DequantizeLinear".into()),
            name: Some("dq".into()),
            input: vec!["xq".into(), "s0".into(), "zp0".into()],
            output: vec!["xf".into()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Relu".into()),
            name: Some("relu".into()),
            input: vec!["xf".into()],
            output: vec!["rf".into()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("QuantizeLinear".into()),
            name: Some("q".into()),
            input: vec!["rf".into(), "s1".into(), "zp1".into()],
            output: vec!["yq".into()],
            ..Default::default()
        },
    ];
    let bytes = build_minimal_onnx_model(
        nodes,
        inits,
        vec!["xq", "s0", "zp0", "s1", "zp1"],
        vec!["yq"],
    );
    let model = load_onnx_model(&bytes).unwrap();
    assert!(
        model.runtime_index.execution_plan.iter().any(|a| matches!(
            a,
            crate::loader::NodeAction::QuantizedQdq {
                relu_idx: Some(_),
                ..
            }
        )),
        "matching DQ->Relu->Q should be represented as a quant-domain runtime action"
    );

    let result = run_onnx_model(&model, HashMap::new()).unwrap();
    assert_eq!(
        result["yq"].data(),
        &[0.0, 0.0, 0.0, 3.0, 7.0, 0.0, 2.0, 1.0]
    );
}

/// Symmetric-int8 fast path (both zero-points are 0) with non-unit
/// scales. Verifies the integer GEMM kernel + composite-scale requantize
/// in `exec_qlinear_matmul` matches a hand-computed reference.
#[test]
fn qlinear_matmul_symmetric_fast_path_with_nontrivial_scales() {
    let _guard = lock_shared_state();
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
    let _guard = lock_shared_state();
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
    crate::runner::reset_quant_runtime_stats();
    let result = run_onnx_model(&model, HashMap::new()).unwrap();
    let stats = crate::runner::quant_runtime_stats();
    assert_eq!(stats.qlinear_conv_fast, 1);
    assert_eq!(stats.qlinear_conv_fallback, 0);
    assert_eq!(stats.quant_i8_stores, 1);
    assert_eq!(stats.quant_i8_materializations, 1);
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

/// Symmetric depthwise QLinearConv fast path: OIHW depthwise weight is packed
/// once to KHWC i8 and dispatched through the multi-arch int8 depthwise kernel.
#[test]
fn qlinear_conv_depthwise_symmetric_fast_path_with_bias() {
    let _guard = lock_shared_state();
    let channels = 5;
    for (kernel, stride, h, w) in [
        (3_usize, 1_usize, 5_usize, 4_usize),
        (5, 1, 5, 4),
        (3, 2, 7, 6),
    ] {
        check_qlinear_depthwise_with_bias(channels, h, w, kernel, stride);
    }
}

fn check_qlinear_depthwise_with_bias(
    channels: usize,
    h: usize,
    w: usize,
    kernel: usize,
    stride: usize,
) {
    let pad = kernel / 2;
    let out_h = (h + 2 * pad - kernel) / stride + 1;
    let out_w = (w + 2 * pad - kernel) / stride + 1;
    let x_int: Vec<i32> = (0..(channels * h * w))
        .map(|v| ((v as i32 * 7) % 29) - 14)
        .collect();
    let w_int: Vec<i32> = (0..(channels * kernel * kernel))
        .map(|v| ((v as i32 * 5) % 17) - 8)
        .collect();
    let bias_int: Vec<i32> = (0..channels).map(|v| (v as i32) - 2).collect();

    let x_data: Vec<f32> = x_int.iter().map(|&v| v as f32).collect();
    let w_data: Vec<f32> = w_int.iter().map(|&v| v as f32).collect();
    let b_data: Vec<f32> = bias_int.iter().map(|&v| v as f32).collect();
    let x_scale = 0.04_f32;
    let w_scale = 0.08_f32;
    let y_scale = 0.25_f32;

    let node = onnx::NodeProto {
        op_type: Some("QLinearConv".into()),
        name: Some("qdw".into()),
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
            make_ints_attr("kernel_shape", vec![kernel as i64, kernel as i64]),
            make_ints_attr("strides", vec![stride as i64, stride as i64]),
            make_ints_attr("pads", vec![pad as i64, pad as i64, pad as i64, pad as i64]),
            make_int_attr("group", channels as i64),
        ],
        ..Default::default()
    };
    let bytes = build_minimal_onnx_model(
        vec![node],
        vec![
            vec_init("x", vec![1, channels as i64, h as i64, w as i64], x_data),
            scalar_init("x_s", x_scale),
            scalar_init("x_zp", 0.0),
            vec_init(
                "w",
                vec![channels as i64, 1, kernel as i64, kernel as i64],
                w_data,
            ),
            scalar_init("w_s", w_scale),
            scalar_init("w_zp", 0.0),
            scalar_init("y_s", y_scale),
            scalar_init("y_zp", 0.0),
            vec_init("b", vec![channels as i64], b_data),
        ],
        vec!["x", "x_s", "x_zp", "w", "w_s", "w_zp", "y_s", "y_zp", "b"],
        vec!["y"],
    );
    let model = load_onnx_model(&bytes).unwrap();
    let result = run_onnx_model(&model, HashMap::new()).unwrap();
    let out = result["y"].data();

    let composite = x_scale * w_scale / y_scale;
    let mut expected = vec![0.0_f32; channels * out_h * out_w];
    for c in 0..channels {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let mut acc = bias_int[c];
                for ky in 0..kernel {
                    for kx in 0..kernel {
                        let ih = (oh * stride) as i64 + ky as i64 - pad as i64;
                        let iw = (ow * stride) as i64 + kx as i64 - pad as i64;
                        if ih < 0 || ih >= h as i64 || iw < 0 || iw >= w as i64 {
                            continue;
                        }
                        let xv = x_int[(c * h + ih as usize) * w + iw as usize];
                        let wv = w_int[(c * kernel + ky) * kernel + kx];
                        acc += xv * wv;
                    }
                }
                expected[(c * out_h + oh) * out_w + ow] =
                    ((acc as f32) * composite).round().clamp(-128.0, 127.0);
            }
        }
    }

    assert_eq!(result["y"].shape(), &[1, channels, out_h, out_w]);
    for (g, e) in out.iter().zip(expected.iter()) {
        assert!(
            (g - e).abs() <= 1e-4,
            "kernel={kernel} stride={stride} got {g} expected {e}"
        );
    }
}

/// MatMulInteger symmetric fast path: rank-2, both zero-points 0,
/// output is raw int32 dot. Validates the kernel + i32→f32 cast.
#[test]
fn matmul_integer_symmetric_fast_path() {
    let _guard = lock_shared_state();
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
    let _guard = lock_shared_state();
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
    let _guard = lock_shared_state();
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

/// End-to-end bitwise check for `NodeAction::QuantizedPwDw`: build a
/// synthetic `QuantizeLinear -> QLinearConv(pw 1×1) -> DequantizeLinear ->
/// [Relu] -> QuantizeLinear -> QLinearConv(dw kxk) -> DequantizeLinear`
/// graph with all-zero zero-points and matching boundary scale, run it
/// twice — once with the chain enabled (default) and once with
/// `YSCV_QUANT_INT8_FAST=0` forcing the unfused per-op path — and assert
/// the f32 outputs are bit-for-bit identical. The test runs three
/// shapes covering Relu / no-Relu, stride 1 / 2, and 3×3 / 5×5 DW so
/// the bitwise contract is exercised across every code branch the
/// chain action has.
#[test]
fn quantized_pw_dw_chain_bitwise_matches_unfused() {
    let _env_guard = lock_shared_state();
    fn run_case(
        c_in: usize,
        c_exp: usize,
        h: usize,
        w: usize,
        kh: usize,
        stride: usize,
        with_relu: bool,
        x_scale: f32,
        pw_w_scale: f32,
        boundary_scale: f32,
        dw_w_scale: f32,
        y_scale: f32,
    ) {
        let pad = (kh - 1) / 2;
        let pw_w_int: Vec<i32> = (0..(c_exp * c_in))
            .map(|v| ((v as i32 * 11) % 23) - 11)
            .collect();
        let dw_w_int: Vec<i32> = (0..(c_exp * kh * kh))
            .map(|v| ((v as i32 * 7) % 17) - 8)
            .collect();
        let pw_b_int: Vec<i32> = (0..c_exp).map(|v| ((v as i32 * 3) % 19) - 9).collect();
        let dw_b_int: Vec<i32> = (0..c_exp).map(|v| ((v as i32 * 5) % 13) - 6).collect();
        let x_int: Vec<i32> = (0..(c_in * h * w))
            .map(|v| ((v as i32 * 7) % 31) - 15)
            .collect();

        let pw_w_data: Vec<f32> = pw_w_int.iter().map(|&v| v as f32).collect();
        let dw_w_data: Vec<f32> = dw_w_int.iter().map(|&v| v as f32).collect();
        let pw_b_data: Vec<f32> = pw_b_int.iter().map(|&v| v as f32).collect();
        let dw_b_data: Vec<f32> = dw_b_int.iter().map(|&v| v as f32).collect();
        let x_data: Vec<f32> = x_int.iter().map(|&v| v as f32 * x_scale).collect();

        let mut nodes = Vec::new();
        nodes.push(onnx::NodeProto {
            op_type: Some("QuantizeLinear".into()),
            name: Some("q_in".into()),
            input: vec!["x".into(), "x_s".into(), "x_zp".into()],
            output: vec!["xq".into()],
            ..Default::default()
        });
        nodes.push(onnx::NodeProto {
            op_type: Some("QLinearConv".into()),
            name: Some("pw".into()),
            input: vec![
                "xq".into(),
                "x_s".into(),
                "x_zp".into(),
                "pw_w".into(),
                "pw_w_s".into(),
                "pw_w_zp".into(),
                "pw_y_s".into(),
                "pw_y_zp".into(),
                "pw_b".into(),
            ],
            output: vec!["pw_q".into()],
            attribute: vec![
                make_ints_attr("kernel_shape", vec![1, 1]),
                make_ints_attr("strides", vec![1, 1]),
                make_ints_attr("pads", vec![0, 0, 0, 0]),
            ],
            ..Default::default()
        });
        nodes.push(onnx::NodeProto {
            op_type: Some("DequantizeLinear".into()),
            name: Some("dq".into()),
            input: vec!["pw_q".into(), "pw_y_s".into(), "pw_y_zp".into()],
            output: vec!["pw_f".into()],
            ..Default::default()
        });
        let q_input = if with_relu {
            nodes.push(onnx::NodeProto {
                op_type: Some("Relu".into()),
                name: Some("relu".into()),
                input: vec!["pw_f".into()],
                output: vec!["pw_relu".into()],
                ..Default::default()
            });
            "pw_relu".to_string()
        } else {
            "pw_f".to_string()
        };
        nodes.push(onnx::NodeProto {
            op_type: Some("QuantizeLinear".into()),
            name: Some("q".into()),
            input: vec![q_input, "q_s".into(), "q_zp".into()],
            output: vec!["dw_x".into()],
            ..Default::default()
        });
        nodes.push(onnx::NodeProto {
            op_type: Some("QLinearConv".into()),
            name: Some("dw".into()),
            input: vec![
                "dw_x".into(),
                "q_s".into(),
                "q_zp".into(),
                "dw_w".into(),
                "dw_w_s".into(),
                "dw_w_zp".into(),
                "y_s".into(),
                "y_zp".into(),
                "dw_b".into(),
            ],
            output: vec!["dw_q".into()],
            attribute: vec![
                make_ints_attr("kernel_shape", vec![kh as i64, kh as i64]),
                make_ints_attr("strides", vec![stride as i64, stride as i64]),
                make_ints_attr("pads", vec![pad as i64, pad as i64, pad as i64, pad as i64]),
                make_int_attr("group", c_exp as i64),
            ],
            ..Default::default()
        });
        nodes.push(onnx::NodeProto {
            op_type: Some("DequantizeLinear".into()),
            name: Some("dq_out".into()),
            input: vec!["dw_q".into(), "y_s".into(), "y_zp".into()],
            output: vec!["y".into()],
            ..Default::default()
        });

        let inits = vec![
            scalar_init("x_s", x_scale),
            scalar_init("x_zp", 0.0),
            vec_init(
                "pw_w",
                vec![c_exp as i64, c_in as i64, 1, 1],
                pw_w_data.clone(),
            ),
            scalar_init("pw_w_s", pw_w_scale),
            scalar_init("pw_w_zp", 0.0),
            scalar_init("pw_y_s", boundary_scale),
            scalar_init("pw_y_zp", 0.0),
            vec_init("pw_b", vec![c_exp as i64], pw_b_data.clone()),
            scalar_init("q_s", boundary_scale),
            scalar_init("q_zp", 0.0),
            vec_init(
                "dw_w",
                vec![c_exp as i64, 1, kh as i64, kh as i64],
                dw_w_data.clone(),
            ),
            scalar_init("dw_w_s", dw_w_scale),
            scalar_init("dw_w_zp", 0.0),
            scalar_init("y_s", y_scale),
            scalar_init("y_zp", 0.0),
            vec_init("dw_b", vec![c_exp as i64], dw_b_data.clone()),
        ];
        let bytes = build_minimal_onnx_model(
            nodes,
            inits,
            vec![
                "x_s", "x_zp", "pw_w", "pw_w_s", "pw_w_zp", "pw_y_s", "pw_y_zp", "pw_b", "q_s",
                "q_zp", "dw_w", "dw_w_s", "dw_w_zp", "y_s", "y_zp", "dw_b",
            ],
            vec!["y"],
        );
        let model = load_onnx_model(&bytes).unwrap();
        assert!(
            model
                .runtime_index
                .execution_plan
                .iter()
                .any(|a| matches!(a, crate::loader::NodeAction::QuantizedPwDw { .. })),
            "loader should have emitted QuantizedPwDw for the synthetic chain"
        );

        let x_tensor = Tensor::from_vec(vec![1, c_in, h, w], x_data.clone()).unwrap();
        let mut feed = HashMap::new();
        feed.insert("x".to_string(), x_tensor);

        // Toggle the fast-path env var around two model runs. This
        // module's tests run with `--test-threads=1` in CI; the
        // fast-path env var is read each iteration of the runner loop
        // so toggling it between runs is observed.
        #[allow(unsafe_code)]
        unsafe {
            std::env::remove_var("YSCV_QUANT_INT8_FAST");
        }
        let fast = run_onnx_model(&model, feed.clone()).unwrap();
        #[allow(unsafe_code)]
        unsafe {
            std::env::set_var("YSCV_QUANT_INT8_FAST", "0");
        }
        let slow = run_onnx_model(&model, feed).unwrap();
        #[allow(unsafe_code)]
        unsafe {
            std::env::remove_var("YSCV_QUANT_INT8_FAST");
        }

        let fast_y = fast["y"].data();
        let slow_y = slow["y"].data();
        assert_eq!(
            fast_y.len(),
            slow_y.len(),
            "fast/slow output length differs for c_in={c_in} c_exp={c_exp} h={h} w={w} kh={kh} s={stride} relu={with_relu}"
        );
        for (i, (g, e)) in fast_y.iter().zip(slow_y.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                e.to_bits(),
                "bitwise mismatch at idx={i} fast={g} slow={e} for c_in={c_in} c_exp={c_exp} h={h} w={w} kh={kh} s={stride} relu={with_relu}"
            );
        }
    }

    // Three coverage shapes for the chain action:
    //   * 3×3 stride 1 with Relu
    //   * 3×3 stride 2 without Relu
    //   * 5×5 stride 1 with Relu
    // c_in=4 c_exp=16 satisfies the load-time prepack gate
    // (`c_in >= 4 && c_exp.is_multiple_of(16)`).
    run_case(4, 16, 8, 8, 3, 1, true, 0.04, 0.06, 0.10, 0.05, 0.20);
    run_case(4, 16, 8, 8, 3, 2, false, 0.05, 0.07, 0.12, 0.04, 0.22);
    run_case(4, 16, 12, 12, 5, 1, true, 0.03, 0.05, 0.09, 0.06, 0.18);
}

/// End-to-end bitwise check for `NodeAction::QuantizedDwPw`: build a
/// synthetic `QuantizeLinear -> QLinearConv(dw kxk) -> DequantizeLinear ->
/// [Relu] -> QuantizeLinear -> QLinearConv(pw 1×1) -> DequantizeLinear`
/// graph with all-zero zero-points and matching boundary scale, run it
/// twice — once with the chain enabled (default) and once with
/// `YSCV_QUANT_INT8_FAST=0` forcing the unfused per-op path — and assert
/// the f32 outputs are bit-for-bit identical. Mirror of the
/// `QuantizedPwDw` end-to-end test for the closing pair.
#[test]
fn quantized_dw_pw_chain_bitwise_matches_unfused() {
    let _env_guard = lock_shared_state();
    fn run_case(
        c_in: usize,
        c_out: usize,
        h: usize,
        w: usize,
        kh: usize,
        stride: usize,
        with_relu: bool,
        x_scale: f32,
        dw_w_scale: f32,
        boundary_scale: f32,
        pw_w_scale: f32,
        y_scale: f32,
    ) {
        let pad = (kh - 1) / 2;
        let dw_w_int: Vec<i32> = (0..(c_in * kh * kh))
            .map(|v| ((v as i32 * 7) % 17) - 8)
            .collect();
        let pw_w_int: Vec<i32> = (0..(c_out * c_in))
            .map(|v| ((v as i32 * 11) % 23) - 11)
            .collect();
        let dw_b_int: Vec<i32> = (0..c_in).map(|v| ((v as i32 * 5) % 13) - 6).collect();
        let pw_b_int: Vec<i32> = (0..c_out).map(|v| ((v as i32 * 3) % 19) - 9).collect();
        let x_int: Vec<i32> = (0..(c_in * h * w))
            .map(|v| ((v as i32 * 7) % 31) - 15)
            .collect();

        let dw_w_data: Vec<f32> = dw_w_int.iter().map(|&v| v as f32).collect();
        let pw_w_data: Vec<f32> = pw_w_int.iter().map(|&v| v as f32).collect();
        let dw_b_data: Vec<f32> = dw_b_int.iter().map(|&v| v as f32).collect();
        let pw_b_data: Vec<f32> = pw_b_int.iter().map(|&v| v as f32).collect();
        let x_data: Vec<f32> = x_int.iter().map(|&v| v as f32 * x_scale).collect();

        let mut nodes = Vec::new();
        nodes.push(onnx::NodeProto {
            op_type: Some("QuantizeLinear".into()),
            name: Some("q_in".into()),
            input: vec!["x".into(), "x_s".into(), "x_zp".into()],
            output: vec!["xq".into()],
            ..Default::default()
        });
        nodes.push(onnx::NodeProto {
            op_type: Some("QLinearConv".into()),
            name: Some("dw".into()),
            input: vec![
                "xq".into(),
                "x_s".into(),
                "x_zp".into(),
                "dw_w".into(),
                "dw_w_s".into(),
                "dw_w_zp".into(),
                "dw_y_s".into(),
                "dw_y_zp".into(),
                "dw_b".into(),
            ],
            output: vec!["dw_q".into()],
            attribute: vec![
                make_ints_attr("kernel_shape", vec![kh as i64, kh as i64]),
                make_ints_attr("strides", vec![stride as i64, stride as i64]),
                make_ints_attr("pads", vec![pad as i64, pad as i64, pad as i64, pad as i64]),
                make_int_attr("group", c_in as i64),
            ],
            ..Default::default()
        });
        nodes.push(onnx::NodeProto {
            op_type: Some("DequantizeLinear".into()),
            name: Some("dq".into()),
            input: vec!["dw_q".into(), "dw_y_s".into(), "dw_y_zp".into()],
            output: vec!["dw_f".into()],
            ..Default::default()
        });
        let q_input = if with_relu {
            nodes.push(onnx::NodeProto {
                op_type: Some("Relu".into()),
                name: Some("relu".into()),
                input: vec!["dw_f".into()],
                output: vec!["dw_relu".into()],
                ..Default::default()
            });
            "dw_relu".to_string()
        } else {
            "dw_f".to_string()
        };
        nodes.push(onnx::NodeProto {
            op_type: Some("QuantizeLinear".into()),
            name: Some("q".into()),
            input: vec![q_input, "q_s".into(), "q_zp".into()],
            output: vec!["pw_x".into()],
            ..Default::default()
        });
        nodes.push(onnx::NodeProto {
            op_type: Some("QLinearConv".into()),
            name: Some("pw".into()),
            input: vec![
                "pw_x".into(),
                "q_s".into(),
                "q_zp".into(),
                "pw_w".into(),
                "pw_w_s".into(),
                "pw_w_zp".into(),
                "y_s".into(),
                "y_zp".into(),
                "pw_b".into(),
            ],
            output: vec!["pw_q".into()],
            attribute: vec![
                make_ints_attr("kernel_shape", vec![1, 1]),
                make_ints_attr("strides", vec![1, 1]),
                make_ints_attr("pads", vec![0, 0, 0, 0]),
            ],
            ..Default::default()
        });
        nodes.push(onnx::NodeProto {
            op_type: Some("DequantizeLinear".into()),
            name: Some("dq_out".into()),
            input: vec!["pw_q".into(), "y_s".into(), "y_zp".into()],
            output: vec!["y".into()],
            ..Default::default()
        });

        let inits = vec![
            scalar_init("x_s", x_scale),
            scalar_init("x_zp", 0.0),
            vec_init(
                "dw_w",
                vec![c_in as i64, 1, kh as i64, kh as i64],
                dw_w_data.clone(),
            ),
            scalar_init("dw_w_s", dw_w_scale),
            scalar_init("dw_w_zp", 0.0),
            scalar_init("dw_y_s", boundary_scale),
            scalar_init("dw_y_zp", 0.0),
            vec_init("dw_b", vec![c_in as i64], dw_b_data.clone()),
            scalar_init("q_s", boundary_scale),
            scalar_init("q_zp", 0.0),
            vec_init(
                "pw_w",
                vec![c_out as i64, c_in as i64, 1, 1],
                pw_w_data.clone(),
            ),
            scalar_init("pw_w_s", pw_w_scale),
            scalar_init("pw_w_zp", 0.0),
            scalar_init("y_s", y_scale),
            scalar_init("y_zp", 0.0),
            vec_init("pw_b", vec![c_out as i64], pw_b_data.clone()),
        ];
        let bytes = build_minimal_onnx_model(
            nodes,
            inits,
            vec![
                "x_s", "x_zp", "dw_w", "dw_w_s", "dw_w_zp", "dw_y_s", "dw_y_zp", "dw_b", "q_s",
                "q_zp", "pw_w", "pw_w_s", "pw_w_zp", "y_s", "y_zp", "pw_b",
            ],
            vec!["y"],
        );
        let model = load_onnx_model(&bytes).unwrap();
        assert!(
            model
                .runtime_index
                .execution_plan
                .iter()
                .any(|a| matches!(a, crate::loader::NodeAction::QuantizedDwPw { .. })),
            "loader should have emitted QuantizedDwPw for the synthetic chain"
        );

        let x_tensor = Tensor::from_vec(vec![1, c_in, h, w], x_data.clone()).unwrap();
        let mut feed = HashMap::new();
        feed.insert("x".to_string(), x_tensor);

        #[allow(unsafe_code)]
        unsafe {
            std::env::remove_var("YSCV_QUANT_INT8_FAST");
        }
        let fast = run_onnx_model(&model, feed.clone()).unwrap();
        #[allow(unsafe_code)]
        unsafe {
            std::env::set_var("YSCV_QUANT_INT8_FAST", "0");
        }
        let slow = run_onnx_model(&model, feed).unwrap();
        #[allow(unsafe_code)]
        unsafe {
            std::env::remove_var("YSCV_QUANT_INT8_FAST");
        }

        let fast_y = fast["y"].data();
        let slow_y = slow["y"].data();
        assert_eq!(
            fast_y.len(),
            slow_y.len(),
            "fast/slow output length differs for c_in={c_in} c_out={c_out} h={h} w={w} kh={kh} s={stride} relu={with_relu}"
        );
        for (i, (g, e)) in fast_y.iter().zip(slow_y.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                e.to_bits(),
                "bitwise mismatch at idx={i} fast={g} slow={e} for c_in={c_in} c_out={c_out} h={h} w={w} kh={kh} s={stride} relu={with_relu}"
            );
        }
    }

    // Three coverage shapes for the closing-pair chain action:
    //   * 3×3 stride 1 with Relu (c_in=16 c_out=16, both pass prepack
    //     gates `c_in >= 4 && c_out.is_multiple_of(16)`)
    //   * 3×3 stride 2 without Relu
    //   * 5×5 stride 1 with Relu
    run_case(16, 16, 8, 8, 3, 1, true, 0.04, 0.05, 0.10, 0.06, 0.20);
    run_case(16, 16, 8, 8, 3, 2, false, 0.05, 0.04, 0.12, 0.07, 0.22);
    run_case(16, 16, 12, 12, 5, 1, true, 0.03, 0.06, 0.09, 0.05, 0.18);
}

/// Asymmetric path (non-zero zero-points) falls through to the f32
/// reference. Verifies the dispatch gate doesn't accidentally claim
/// asymmetric inputs and produce wrong output.
#[test]
fn qlinear_matmul_asymmetric_uses_fp32_fallback() {
    let _guard = lock_shared_state();
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
