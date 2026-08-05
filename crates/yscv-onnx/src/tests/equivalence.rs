//! Numerical-equivalence harness for graph transformations.
//!
//! Every optimizer pass, IR lowering and plan-selection step is supposed to
//! leave the model's arithmetic alone. Structural assertions ("the graph now
//! has one fewer node") do not catch a transformation that fuses the right
//! nodes and computes the wrong thing, so each one also gets checked here: run
//! the graph before and after through the real runner on pseudo-random inputs
//! and compare outputs.
//!
//! Generalized from `rewrite_convtranspose_dts_is_numerically_identical`, which
//! established the pattern.

use rustc_hash::FxHashMap;
use yscv_tensor::Tensor;

use super::super::loader::{OnnxModel, load_onnx_model};
use super::super::runner::run_onnx_model;

/// Deterministic linear-congruential generator.
///
/// Test inputs need to be pseudo-random — zeros and ones hide sign errors,
/// index transpositions and channel mix-ups — but a failure has to reproduce,
/// so this is seeded rather than drawn from `rand`.
pub(in crate::tests) struct Lcg(u64);

impl Lcg {
    pub(in crate::tests) fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Next value in `[-1.0, 1.0)`, quantized to 1/1000.
    pub(in crate::tests) fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) % 2000) as f32 / 1000.0 - 1.0
    }

    pub(in crate::tests) fn vec(&mut self, len: usize) -> Vec<f32> {
        (0..len).map(|_| self.next_f32()).collect()
    }

    /// Pseudo-random tensor of the given shape.
    pub(in crate::tests) fn tensor(&mut self, shape: Vec<usize>) -> Tensor {
        let len = shape.iter().product();
        Tensor::from_vec(shape, self.vec(len)).expect("shape matches generated length")
    }
}

/// How closely the transformed graph has to match the original.
#[derive(Debug, Clone, Copy)]
pub(in crate::tests) enum Tolerance {
    /// Bitwise identical. Required of transformations that only reorganize
    /// execution — kernel selection, fusion, layout assignment — since they
    /// must not change arithmetic at all.
    Exact,
    /// Elementwise absolute difference below the bound. For rewrites that
    /// legitimately re-associate arithmetic, such as folding BatchNormalization
    /// into Conv weights.
    Abs(f32),
}

/// Runs `model_bytes` before and after `transform`, and asserts the outputs
/// match within `tolerance`.
///
/// Loads the model twice so the reference run cannot observe any mutation the
/// transform makes. Rebuilds the runtime index afterwards, because passes
/// deliberately leave that to their caller.
///
/// `label` names the transformation in failure messages.
pub(in crate::tests) fn assert_transform_preserves_numerics(
    label: &str,
    model_bytes: &[u8],
    feed: &FxHashMap<String, Tensor>,
    tolerance: Tolerance,
    transform: impl FnOnce(&mut OnnxModel),
) {
    let reference_model =
        load_onnx_model(model_bytes).unwrap_or_else(|e| panic!("{label}: reference load: {e}"));
    let reference = run_onnx_model(&reference_model, feed.clone())
        .unwrap_or_else(|e| panic!("{label}: reference run: {e}"));

    let mut transformed_model =
        load_onnx_model(model_bytes).unwrap_or_else(|e| panic!("{label}: transformed load: {e}"));
    transform(&mut transformed_model);
    transformed_model.rebuild_runtime_index();
    let transformed = run_onnx_model(&transformed_model, feed.clone())
        .unwrap_or_else(|e| panic!("{label}: transformed run: {e}"));

    assert_outputs_match(label, &reference, &transformed, tolerance);
}

/// Compares two output maps, reporting the first mismatch with enough context
/// to locate it.
pub(in crate::tests) fn assert_outputs_match(
    label: &str,
    reference: &FxHashMap<String, Tensor>,
    actual: &FxHashMap<String, Tensor>,
    tolerance: Tolerance,
) {
    let mut reference_names: Vec<&String> = reference.keys().collect();
    let mut actual_names: Vec<&String> = actual.keys().collect();
    reference_names.sort();
    actual_names.sort();
    assert_eq!(
        reference_names, actual_names,
        "{label}: output names diverged"
    );

    for name in reference_names {
        let expected = &reference[name];
        let got = &actual[name];
        assert_eq!(
            expected.shape(),
            got.shape(),
            "{label}: output '{name}' changed shape"
        );

        let expected_data = expected.data();
        let got_data = got.data();
        for (i, (&want, &have)) in expected_data.iter().zip(got_data).enumerate() {
            let ok = match tolerance {
                Tolerance::Exact => want.to_bits() == have.to_bits(),
                Tolerance::Abs(eps) => (want - have).abs() < eps,
            };
            assert!(
                ok,
                "{label}: output '{name}' differs at element {i}: \
                 expected {want}, got {have} (tolerance {tolerance:?})"
            );
        }
    }
}

/// Asserts a transformation is a structural no-op *and* a numerical no-op.
///
/// Used for passes that must decline to fire on a graph outside their safe
/// subset — the common failure there is firing anyway and silently corrupting
/// the model.
pub(in crate::tests) fn assert_transform_is_noop(
    label: &str,
    model_bytes: &[u8],
    feed: &FxHashMap<String, Tensor>,
    transform: impl FnOnce(&mut OnnxModel),
) {
    let before = load_onnx_model(model_bytes).unwrap_or_else(|e| panic!("{label}: load: {e}"));
    let before_ops: Vec<String> = before.nodes.iter().map(|n| n.op_type.clone()).collect();

    let mut after = load_onnx_model(model_bytes).unwrap_or_else(|e| panic!("{label}: load: {e}"));
    transform(&mut after);
    after.rebuild_runtime_index();
    let after_ops: Vec<String> = after.nodes.iter().map(|n| n.op_type.clone()).collect();

    assert_eq!(
        before_ops, after_ops,
        "{label}: expected the transform to decline, but the graph changed"
    );

    let reference = run_onnx_model(&before, feed.clone())
        .unwrap_or_else(|e| panic!("{label}: reference run: {e}"));
    let actual =
        run_onnx_model(&after, feed.clone()).unwrap_or_else(|e| panic!("{label}: run: {e}"));
    assert_outputs_match(label, &reference, &actual, Tolerance::Exact);
}

/// Self-tests. A harness that cannot fail proves nothing about the passes it
/// guards, so each assertion is exercised against a deliberately broken
/// transformation.
#[cfg(test)]
mod harness_self_tests {
    use super::*;
    use crate::tests::build_minimal_onnx_model;
    use crate::{loader::OnnxNode, proto::onnx};

    /// `Relu(x) -> Relu(y)`, so a transform can corrupt it in obvious ways.
    fn two_relu_model() -> Vec<u8> {
        let nodes = vec![
            onnx::NodeProto {
                op_type: Some("Relu".into()),
                name: Some("r0".into()),
                input: vec!["x".into()],
                output: vec!["mid".into()],
                ..Default::default()
            },
            onnx::NodeProto {
                op_type: Some("Relu".into()),
                name: Some("r1".into()),
                input: vec!["mid".into()],
                output: vec!["y".into()],
                ..Default::default()
            },
        ];
        build_minimal_onnx_model(nodes, vec![], vec!["x"], vec!["y"])
    }

    fn feed() -> FxHashMap<String, Tensor> {
        let mut rng = Lcg::new(0x1234_5678);
        let mut feed = FxHashMap::default();
        feed.insert("x".to_string(), rng.tensor(vec![1, 4, 2, 2]));
        feed
    }

    #[test]
    fn identity_transform_passes() {
        assert_transform_preserves_numerics(
            "identity",
            &two_relu_model(),
            &feed(),
            Tolerance::Exact,
            |_| {},
        );
    }

    /// Swapping a Relu for a Sigmoid keeps the graph's shape and node count but
    /// changes what it computes — exactly the class of error structural
    /// assertions miss.
    #[test]
    #[should_panic(expected = "differs at element")]
    fn value_corruption_is_caught() {
        assert_transform_preserves_numerics(
            "corrupt-op",
            &two_relu_model(),
            &feed(),
            Tolerance::Exact,
            |model| model.nodes[1].op_type = "Sigmoid".to_string(),
        );
    }

    /// A transform that changes output rank must be reported as a shape
    /// divergence rather than slipping through elementwise comparison.
    #[test]
    #[should_panic(expected = "changed shape")]
    fn shape_divergence_is_caught() {
        assert_transform_preserves_numerics(
            "corrupt-shape",
            &two_relu_model(),
            &feed(),
            Tolerance::Exact,
            |model| {
                model.nodes[1].op_type = "GlobalAveragePool".to_string();
            },
        );
    }

    /// `Tolerance::Abs` must still reject differences above its bound.
    #[test]
    #[should_panic(expected = "differs at element")]
    fn abs_tolerance_still_rejects_large_drift() {
        assert_transform_preserves_numerics(
            "corrupt-abs",
            &two_relu_model(),
            &feed(),
            Tolerance::Abs(1e-5),
            |model| model.nodes[1].op_type = "Sigmoid".to_string(),
        );
    }

    /// `assert_transform_is_noop` must reject a transform that does fire.
    #[test]
    #[should_panic(expected = "expected the transform to decline")]
    fn noop_assertion_catches_a_firing_transform() {
        assert_transform_is_noop("not-a-noop", &two_relu_model(), &feed(), |model| {
            model.nodes.push(OnnxNode {
                op_type: "Relu".to_string(),
                name: "extra".to_string(),
                inputs: vec!["y".to_string()],
                outputs: vec!["z".to_string()],
                attributes: Default::default(),
            });
        });
    }
}
