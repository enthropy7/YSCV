//! Weight prepacking: the structures that break when a permuted weight is
//! tracked by name.
//!
//! Conv weights are permuted once, into whichever layout the kernel that reads
//! them wants. Two graph shapes make that hard, and neither appears in any of
//! the four benchmark models — which is why the tracker model was the first
//! thing to hit it, in production rather than in CI:
//!
//! - **A weight reached under two names.** Constant folding rewrites an
//!   `Identity` into an alias, so the same bytes arrive at a second Conv under a
//!   name the permute never registered.
//! - **A weight shared by two Convs.** One tensor, two readers, and any pass
//!   that rewrites weights in place has to decline.
//! - **A weight that is not a weight.** A Siamese head's cross-correlation is a
//!   Conv whose kernel *is* the template feature map — a dynamic tensor. There
//!   is nothing to prepack, so the Conv has to reach the runtime repack instead.
//!
//! Each fixture is checked against an equivalent graph that avoids the sharing,
//! so the oracle is the arithmetic rather than a hand-computed constant: reusing
//! a tensor, or feeding it in at run time, must not change what the graph
//! computes.

use rustc_hash::FxHashMap;
use yscv_tensor::Tensor;

use super::equivalence::{Lcg, Tolerance, assert_outputs_match};
use super::*;
use crate::optimizer::optimize_onnx_graph;
#[cfg(not(any(feature = "metal-backend", feature = "gpu")))]
use crate::runner::conv_kernel::ConvKernel;

/// Loads, optimizes and runs a model, returning its outputs.
///
/// The optimizer is what turns an `Identity` into an alias and what tries to
/// fold BatchNormalization into a shared weight, so these fixtures have to run
/// it — loading alone reaches neither hazard.
fn run_optimized(
    label: &str,
    bytes: &[u8],
    feed: &FxHashMap<String, Tensor>,
) -> FxHashMap<String, Tensor> {
    let mut model = load_onnx_model(bytes).unwrap_or_else(|e| panic!("{label}: load: {e}"));
    optimize_onnx_graph(&mut model).unwrap_or_else(|e| panic!("{label}: optimize: {e}"));
    run_onnx_model(&model, feed.clone()).unwrap_or_else(|e| panic!("{label}: run: {e}"))
}

fn conv(name: &str, inputs: Vec<&str>, output: &str, group: i64, k: i64) -> onnx::NodeProto {
    let mut attribute = vec![
        make_ints_attr("kernel_shape", vec![k, k]),
        make_int_attr("group", group),
    ];
    if k > 1 {
        attribute.push(make_ints_attr("pads", vec![1, 1, 1, 1]));
    }
    onnx::NodeProto {
        op_type: Some("Conv".into()),
        name: Some(name.into()),
        input: inputs.into_iter().map(String::from).collect(),
        output: vec![output.into()],
        attribute,
        ..Default::default()
    }
}

/// Weight initializer with deterministic, distinguishable values. A constant
/// fill hides index transpositions — every permutation of it is itself.
fn weight(name: &str, dims: Vec<i64>, seed: u64) -> onnx::TensorProto {
    let len: i64 = dims.iter().product();
    let mut rng = Lcg::new(seed);
    onnx::TensorProto {
        name: Some(name.into()),
        dims,
        data_type: Some(1),
        float_data: rng.vec(len as usize),
        ..Default::default()
    }
}

fn feed(shape: Vec<usize>) -> FxHashMap<String, Tensor> {
    let mut rng = Lcg::new(0x5EED_0001);
    let mut feed = FxHashMap::default();
    feed.insert("x".to_string(), rng.tensor(shape));
    feed
}

/// A depthwise weight read both directly and through an `Identity` computes the
/// same as one where the second reader has its own copy.
///
/// The shape that broke the tracker model. Constant folding removes the
/// `Identity` by pointing its consumer at the aliased tensor under a new name;
/// if the permuted layout is tracked per name, the second Conv gets
/// `[KH, KW, C, 1]` bytes with nothing saying so and reads them as
/// `[O, I, KH, KW]`.
#[test]
fn a_depthwise_weight_aliased_through_identity_computes_the_same() {
    let c = 16i64;
    let aliased = build_minimal_onnx_model(
        vec![
            conv("dw_direct", vec!["x", "w"], "h", c, 3),
            onnx::NodeProto {
                op_type: Some("Identity".into()),
                name: Some("alias".into()),
                input: vec!["w".into()],
                output: vec!["w_alias".into()],
                ..Default::default()
            },
            conv("dw_aliased", vec!["h", "w_alias"], "y", c, 3),
        ],
        vec![weight("w", vec![c, 1, 3, 3], 0xA11A)],
        vec!["x"],
        vec!["y"],
    );
    // Same arithmetic, no alias: the second Conv reads its own initializer
    // holding identical bytes.
    let copied = build_minimal_onnx_model(
        vec![
            conv("dw_direct", vec!["x", "w"], "h", c, 3),
            conv("dw_aliased", vec!["h", "w_copy"], "y", c, 3),
        ],
        vec![
            weight("w", vec![c, 1, 3, 3], 0xA11A),
            weight("w_copy", vec![c, 1, 3, 3], 0xA11A),
        ],
        vec!["x"],
        vec!["y"],
    );

    let feed = feed(vec![1, c as usize, 6, 6]);
    assert_outputs_match(
        "identity-aliased depthwise weight",
        &run_optimized("copied", &copied, &feed),
        &run_optimized("aliased", &aliased, &feed),
        Tolerance::Exact,
    );
}

/// One Conv weight feeding two branches computes the same as two copies of it.
///
/// Both branches carry a BatchNormalization, so this is also the first model in
/// the suite to exercise `fold_conv_bn`'s shared-weight guard
/// (`fold_conv_bn.rs:80`): folding rewrites the weight in place, so with two
/// Convs reading it the pass has to decline. On the copied graph it may fold
/// freely — the two must still agree.
#[test]
fn a_conv_weight_shared_by_two_branches_computes_the_same() {
    let (cin, cout) = (8i64, 8i64);
    let bn_params = |suffix: &str, seed: u64| {
        vec![
            weight(&format!("scale{suffix}"), vec![cout], seed),
            weight(&format!("bias{suffix}"), vec![cout], seed + 1),
            weight(&format!("mean{suffix}"), vec![cout], seed + 2),
            // Variance must be positive; a squared LCG draw is both.
            onnx::TensorProto {
                name: Some(format!("var{suffix}")),
                dims: vec![cout],
                data_type: Some(1),
                float_data: Lcg::new(seed + 3)
                    .vec(cout as usize)
                    .into_iter()
                    .map(|v| v * v + 0.5)
                    .collect(),
                ..Default::default()
            },
        ]
    };
    let bn = |name: &str, input: &str, output: &str, suffix: &str| onnx::NodeProto {
        op_type: Some("BatchNormalization".into()),
        name: Some(name.into()),
        input: vec![
            input.into(),
            format!("scale{suffix}"),
            format!("bias{suffix}"),
            format!("mean{suffix}"),
            format!("var{suffix}"),
        ],
        output: vec![output.into()],
        ..Default::default()
    };
    let add = onnx::NodeProto {
        op_type: Some("Add".into()),
        name: Some("merge".into()),
        input: vec!["a_bn".into(), "b_bn".into()],
        output: vec!["y".into()],
        ..Default::default()
    };

    let mut shared_inits = vec![weight("w", vec![cout, cin, 3, 3], 0x5A5A)];
    shared_inits.extend(bn_params("_a", 0x11));
    shared_inits.extend(bn_params("_b", 0x22));
    let shared = build_minimal_onnx_model(
        vec![
            conv("conv_a", vec!["x", "w"], "a", 1, 3),
            bn("bn_a", "a", "a_bn", "_a"),
            conv("conv_b", vec!["x", "w"], "b", 1, 3),
            bn("bn_b", "b", "b_bn", "_b"),
            add.clone(),
        ],
        shared_inits,
        vec!["x"],
        vec!["y"],
    );

    let mut copied_inits = vec![
        weight("w_a", vec![cout, cin, 3, 3], 0x5A5A),
        weight("w_b", vec![cout, cin, 3, 3], 0x5A5A),
    ];
    copied_inits.extend(bn_params("_a", 0x11));
    copied_inits.extend(bn_params("_b", 0x22));
    let copied = build_minimal_onnx_model(
        vec![
            conv("conv_a", vec!["x", "w_a"], "a", 1, 3),
            bn("bn_a", "a", "a_bn", "_a"),
            conv("conv_b", vec!["x", "w_b"], "b", 1, 3),
            bn("bn_b", "b", "b_bn", "_b"),
            add,
        ],
        copied_inits,
        vec!["x"],
        vec!["y"],
    );

    let feed = feed(vec![1, cin as usize, 6, 6]);
    // Not `Exact`: on the copied graph `fold_conv_bn` fires and folds the
    // scale into the weights, which re-associates the arithmetic. The point of
    // the fixture is that declining to fold computes the same answer, not the
    // same float ops.
    assert_outputs_match(
        "shared conv weight across two branches",
        &run_optimized("copied", &copied, &feed),
        &run_optimized("shared", &shared, &feed),
        Tolerance::Abs(1e-4),
    );
}

/// Two towers sharing every weight compute the same run in parallel as in
/// sequence.
///
/// The Siamese tracker shape, built rather than downloaded: a real one is
/// exported as separate backbone and head files, so the two-input graph that
/// makes both hazards reachable exists only in the host code that calls the
/// backbone twice, and no single file contains it.
///
/// Two things meet here that nothing else in the suite reaches. Every Conv
/// weight is read by both towers, at a dozen convs rather than one. And
/// `node_branches` is non-empty, so the runner forks the environment and runs
/// the towers concurrently — which means each fork carries the packed-weight
/// tables by reference, and a fork that lost them would read ONNX-native bytes
/// as channel-last.
#[cfg(not(any(feature = "metal-backend", feature = "gpu")))]
#[test]
fn siamese_towers_sharing_weights_compute_the_same_forked() {
    let _env = super::equivalence::lock_env();
    let bytes = siamese_model();

    let model = load_onnx_model(&bytes).expect("load");
    assert!(
        !model.runtime_index.node_branches.is_empty(),
        "fixture must split into two towers, or the fork never happens"
    );

    let mut rng = Lcg::new(0x51A_3E5E);
    let mut feed = FxHashMap::default();
    feed.insert("x0".to_string(), rng.tensor(vec![1, 8, 8, 8]));
    feed.insert("x1".to_string(), rng.tensor(vec![1, 8, 8, 8]));

    // SAFETY: `set_var`/`remove_var` are not thread-safe and `cargo test` runs
    // tests concurrently. The `EnvGuard` held for this whole test proves this
    // thread has exclusive use of the environment, and both variables are
    // cleared before it returns.
    let run = |var: &str| {
        #[allow(unsafe_code)]
        unsafe {
            std::env::set_var(var, "1");
        }
        let out = run_onnx_model(&model, feed.clone());
        #[allow(unsafe_code)]
        unsafe {
            std::env::remove_var(var);
        }
        out.unwrap_or_else(|e| panic!("{var}: run: {e}"))
    };

    // Forking must not change arithmetic: the towers touch disjoint
    // activations, and share only weights, which nobody writes.
    assert_outputs_match(
        "siamese towers, forked vs sequential",
        &run("YSCV_NO_TOWER_PARALLEL"),
        &run("YSCV_FORCE_TOWER_PARALLEL"),
        Tolerance::Exact,
    );
}

/// Two identical 12-node towers over separate inputs, sharing all six weights,
/// merged by an `Add`. Sized past the runner's 10-node-per-branch floor for
/// splitting, and covering all three packed layouts in each tower.
#[cfg(not(any(feature = "metal-backend", feature = "gpu")))]
fn siamese_model() -> Vec<u8> {
    let mut nodes = Vec::new();
    for t in 0..2 {
        // Same weight names in both towers — that is the point.
        let layers: [(&str, i64, i64); 6] = [
            ("w_stem", 1, 3),
            ("w_dw", 8, 3),
            ("w_pw", 1, 1),
            ("w_grp", 2, 3),
            ("w_pw2", 1, 1),
            ("w_head", 1, 3),
        ];
        let mut cur = format!("x{t}");
        for (i, (w, group, k)) in layers.iter().enumerate() {
            let c_out = format!("t{t}_c{i}");
            let r_out = format!("t{t}_r{i}");
            nodes.push(conv(
                &format!("t{t}_conv{i}"),
                vec![&cur, w],
                &c_out,
                *group,
                *k,
            ));
            nodes.push(onnx::NodeProto {
                op_type: Some("Relu".into()),
                name: Some(format!("t{t}_relu{i}")),
                input: vec![c_out],
                output: vec![r_out.clone()],
                ..Default::default()
            });
            cur = r_out;
        }
    }
    nodes.push(onnx::NodeProto {
        op_type: Some("Add".into()),
        name: Some("merge".into()),
        input: vec!["t0_r5".into(), "t1_r5".into()],
        output: vec!["y".into()],
        ..Default::default()
    });

    build_minimal_onnx_model(
        nodes,
        vec![
            weight("w_stem", vec![8, 8, 3, 3], 0x11),
            weight("w_dw", vec![8, 1, 3, 3], 0x22),
            weight("w_pw", vec![8, 8, 1, 1], 0x33),
            weight("w_grp", vec![8, 4, 3, 3], 0x44),
            weight("w_pw2", vec![8, 8, 1, 1], 0x55),
            weight("w_head", vec![8, 8, 3, 3], 0x66),
        ],
        vec!["x0", "x1"],
        vec!["y"],
    )
}

/// A Conv that an optimizer pass has renamed still gets its weight packed.
///
/// `fuse_conv_relu` rewrites `Conv` to `Conv_Relu`, and the plan is built after
/// it runs. Selecting weights to pack by the op-type string therefore skips
/// every fused Conv — on mobilenet-v3-small that was all nine squeeze-excite
/// `fc1` layers, which quietly fell back to repacking on every inference.
///
/// Invisible until the permute moved out of the loader, which ran before any
/// pass could rename anything. Asserted on the layout rather than on output
/// values because the fallback is slower, not wrong: nothing about the numbers
/// says the pack went missing.
#[cfg(not(any(feature = "metal-backend", feature = "gpu")))]
#[test]
fn a_conv_renamed_by_fusion_keeps_its_packed_weight() {
    let bytes = build_minimal_onnx_model(
        vec![
            conv("pw", vec!["x", "w"], "h", 1, 1),
            onnx::NodeProto {
                op_type: Some("Relu".into()),
                name: Some("act".into()),
                input: vec!["h".into()],
                output: vec!["y".into()],
                ..Default::default()
            },
        ],
        vec![weight("w", vec![8, 8, 1, 1], 0xFEED)],
        vec!["x"],
        vec!["y"],
    );

    let mut model = load_onnx_model(&bytes).expect("load");
    optimize_onnx_graph(&mut model).expect("optimize");

    assert!(
        model.nodes.iter().any(|n| n.op_type == "Conv_Relu"),
        "fixture must reach the rename, or it is not testing anything; got {:?}",
        model.nodes.iter().map(|n| &n.op_type).collect::<Vec<_>>()
    );

    let id = model.runtime_index.name_to_id["w"];
    assert_eq!(
        model.runtime_index.conv_weight_layouts[id],
        crate::plan::ConvWeightLayout::Khwc,
        "`w` is read by a fused pointwise Conv and must still be packed KHWC"
    );
    assert!(
        model.runtime_index.prepacked_weights.contains_key("w"),
        "and must still have its load-time packed B"
    );
}

/// A Conv whose kernel arrives at run time computes the same as one whose
/// kernel is baked in.
///
/// The cross-correlation in a Siamese tracker head: the "weight" is the
/// template feature map, so it is a graph input, and prepacking — which only
/// ever touches initializers — has nothing to act on. The Conv has to fall
/// through to the runtime repack and still be right.
///
/// Worth its own fixture because the constant and dynamic cases take different
/// code: the constant one reads a pre-permuted tensor and the dynamic one
/// permutes per inference, and the two are supposed to agree. Run for both
/// grouped and depthwise, which repack differently.
#[test]
fn a_conv_kernel_fed_at_runtime_computes_the_same() {
    for (label, group, cin) in [("grouped", 4i64, 16i64), ("depthwise", 16i64, 16i64)] {
        let ipg = cin / group;
        let w_dims = vec![cin, ipg, 3, 3];
        let w_tensor = {
            let len: usize = w_dims.iter().product::<i64>() as usize;
            Tensor::from_vec(
                w_dims.iter().map(|&d| d as usize).collect(),
                Lcg::new(0xC0FFEE).vec(len),
            )
            .expect("shape matches")
        };

        // `w` as a second graph input: nothing for the loader to prepack.
        let dynamic = build_minimal_onnx_model(
            vec![conv("xcorr", vec!["x", "w"], "y", group, 3)],
            vec![],
            vec!["x", "w"],
            vec!["y"],
        );
        // `w` as an initializer holding identical bytes: prepacked as usual.
        let constant = build_minimal_onnx_model(
            vec![conv("xcorr", vec!["x", "w"], "y", group, 3)],
            vec![weight("w", w_dims, 0xC0FFEE)],
            vec!["x"],
            vec!["y"],
        );

        let model = load_onnx_model(&dynamic).unwrap_or_else(|e| panic!("{label}: load: {e}"));
        assert!(
            model.runtime_index.conv_kernels[0].is_none(),
            "{label}: the plan resolved a kernel for a Conv whose weight it \
             cannot see; kernel selection reads the weight's shape, which is \
             only known at run time here"
        );

        let mut dynamic_feed = feed(vec![1, cin as usize, 6, 6]);
        let constant_feed = dynamic_feed.clone();
        dynamic_feed.insert("w".to_string(), w_tensor);

        assert_outputs_match(
            &format!("{label} Conv with a runtime kernel"),
            &run_optimized(label, &constant, &constant_feed),
            &run_optimized(label, &dynamic, &dynamic_feed),
            Tolerance::Exact,
        );
    }
}

/// A graph reaching all three weight layouts, one Conv each.
///
/// Coverage, not correctness: the three permutes take different branches and
/// are gated separately, so a change that keeps two of them working proves
/// little, and this is what stops the fixture below from quietly ceasing to
/// exercise one. Stated in terms of the resolved kernel because that is what a
/// layout exists to feed, and because unlike the name sets it survives the
/// permute moving into the plan.
#[cfg(not(any(feature = "metal-backend", feature = "gpu")))]
#[test]
fn all_three_weight_layouts_reach_their_kernel() {
    let bytes = three_layout_model();
    let model = load_onnx_model(&bytes).expect("load");

    let kernels: Vec<ConvKernel> = model
        .runtime_index
        .conv_kernels
        .iter()
        .flatten()
        .copied()
        .collect();

    assert!(
        kernels.iter().any(|k| matches!(
            k,
            ConvKernel::NhwcPadded | ConvKernel::NhwcGemm | ConvKernel::NhwcGemmPrepacked
        )),
        "no group-1 Conv reached a KHWC kernel; got {kernels:?}"
    );
    assert!(
        kernels.iter().any(|k| matches!(
            k,
            ConvKernel::DepthwiseNhwcPadded | ConvKernel::DepthwiseNhwc
        )),
        "no depthwise Conv reached a depthwise-KHWC kernel; got {kernels:?}"
    );
    assert!(
        kernels.iter().any(|k| matches!(k, ConvKernel::Grouped)),
        "no grouped Conv reached the grouped-KHWC kernel; got {kernels:?}"
    );
}

/// The same graph computes the same with the streaming inverted-bottleneck
/// fusion off, so the layouts feeding it are being read as what they are.
///
/// The middle three Convs have to actually fuse for this to test anything, and
/// they only do so once the depthwise weight is in `[KH, KW, C, 1]` — so the
/// plan assertion below is the one that notices a permute going missing.
#[cfg(not(any(feature = "metal-backend", feature = "gpu")))]
#[test]
fn all_three_weight_layouts_survive_the_fused_path() {
    let env = super::equivalence::lock_env();
    let bytes = three_layout_model();

    let model = load_onnx_model(&bytes).expect("load");
    assert!(
        model
            .runtime_index
            .execution_plan
            .iter()
            .any(|a| matches!(a, crate::plan::NodeAction::FusedPwDwPwReduce { .. })),
        "the inverted bottleneck did not fuse, so the comparison below would \
         not exercise a fused path; plan: {:?}",
        model.runtime_index.execution_plan
    );

    super::equivalence::assert_plan_fusion_preserves_numerics(
        &env,
        "three weight layouts through PW→DW→PW",
        &bytes,
        &feed(vec![1, 8, 8, 8]),
        Tolerance::Exact,
        "YSCV_FUSED_PW_DW_PW_REDUCE_OFF",
    );
}

/// `Conv(3×3) → PW expand → DW 3×3 → PW reduce → Conv(grouped)`.
///
/// The middle three are the inverted bottleneck that fuses; the outer two exist
/// to bring the grouped layout into the same graph.
#[cfg(not(any(feature = "metal-backend", feature = "gpu")))]
fn three_layout_model() -> Vec<u8> {
    let (cin, c, cexp) = (8i64, 16i64, 32i64);
    build_minimal_onnx_model(
        vec![
            conv("stem", vec!["x", "w_stem"], "h0", 1, 3),
            conv("pw_expand", vec!["h0", "w_expand"], "h1", 1, 1),
            conv("dw", vec!["h1", "w_dw"], "h2", cexp, 3),
            conv("pw_reduce", vec!["h2", "w_reduce"], "h3", 1, 1),
            conv("grouped", vec!["h3", "w_group"], "y", 4, 3),
        ],
        vec![
            weight("w_stem", vec![c, cin, 3, 3], 0x0001),
            weight("w_expand", vec![cexp, c, 1, 1], 0x0002),
            weight("w_dw", vec![cexp, 1, 3, 3], 0x0003),
            weight("w_reduce", vec![c, cexp, 1, 1], 0x0004),
            weight("w_group", vec![c, c / 4, 3, 3], 0x0005),
        ],
        vec!["x"],
        vec!["y"],
    )
}
