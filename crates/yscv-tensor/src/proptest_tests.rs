use proptest::prelude::*;

use super::Tensor;
use super::shape::broadcast_shape;

/// Strategy: generate a random shape with 1-4 dimensions, each dimension 1-8.
fn arb_shape() -> impl Strategy<Value = Vec<usize>> {
    proptest::collection::vec(1usize..=8, 1..=4)
}

/// Strategy: generate a tensor with a given shape and random f32 values in [-10, 10].
fn arb_tensor(shape: Vec<usize>) -> impl Strategy<Value = Tensor> {
    let len: usize = shape.iter().product();
    proptest::collection::vec(-10.0f32..10.0, len)
        .prop_map(move |data| Tensor::from_vec(shape.clone(), data).expect("valid tensor"))
}

/// Strategy: generate a pair of shapes that are broadcast-compatible.
fn arb_broadcastable_shapes() -> impl Strategy<Value = (Vec<usize>, Vec<usize>)> {
    // Generate a base shape, then derive two shapes that are broadcast-compatible
    // by optionally replacing dimensions with 1.
    arb_shape().prop_flat_map(|base| {
        let len = base.len();
        let base2 = base.clone();
        (
            proptest::collection::vec(proptest::bool::ANY, len).prop_map(move |mask| {
                base.iter()
                    .zip(mask.iter())
                    .map(|(&d, &use_one)| if use_one { 1 } else { d })
                    .collect::<Vec<_>>()
            }),
            proptest::collection::vec(proptest::bool::ANY, len).prop_map(move |mask| {
                base2
                    .iter()
                    .zip(mask.iter())
                    .map(|(&d, &use_one)| if use_one { 1 } else { d })
                    .collect::<Vec<_>>()
            }),
        )
    })
}

/// Strategy: generate two tensors with broadcast-compatible shapes.
fn arb_broadcastable_tensors() -> impl Strategy<Value = (Tensor, Tensor)> {
    arb_broadcastable_shapes().prop_flat_map(|(s1, s2)| (arb_tensor(s1), arb_tensor(s2)))
}

// Reference broadcast_shape implementation (NumPy rules) for cross-checking.
fn reference_broadcast(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let rank = a.len().max(b.len());
    let mut out = Vec::with_capacity(rank);
    for i in 0..rank {
        let da = if i < rank - a.len() {
            1
        } else {
            a[i + a.len() - rank]
        };
        let db = if i < rank - b.len() {
            1
        } else {
            b[i + b.len() - rank]
        };
        if da == db {
            out.push(da);
        } else if da == 1 {
            out.push(db);
        } else if db == 1 {
            out.push(da);
        } else {
            return None;
        }
    }
    Some(out)
}

/// Strategy: generate a pair of tensors with the same random shape.
fn arb_same_shape_tensors() -> impl Strategy<Value = (Tensor, Tensor)> {
    arb_shape().prop_flat_map(|s| {
        let s2 = s.clone();
        (arb_tensor(s), arb_tensor(s2))
    })
}

/// Strategy: generate three tensors with the same random shape.
fn arb_same_shape_triple() -> impl Strategy<Value = (Tensor, Tensor, Tensor)> {
    arb_shape().prop_flat_map(|s| {
        let s2 = s.clone();
        let s3 = s.clone();
        (arb_tensor(s), arb_tensor(s2), arb_tensor(s3))
    })
}

#[test]
fn add_preserves_max_rank_under_broadcast() {
    // Regression: the lastdim-SIMD fast path used to return the
    // big-tensor's shape verbatim, dropping rank when the small
    // side was a higher-rank all-1s "row vector". Numpy says the
    // broadcast result keeps max(rank(lhs), rank(rhs)). This bug
    // showed up end-to-end as "broadcast mismatch [2] vs [4]"
    // on cached-decoder Qwen because downstream Shape ops then
    // saw a rank-2 tensor where the model expected rank-4.
    let a = Tensor::from_vec(vec![1, 1, 1, 1], vec![5.0]).unwrap();
    let data: Vec<f32> = (0..43).map(|i| i as f32).collect();
    let b = Tensor::from_vec(vec![43, 1], data.clone()).unwrap();
    let c = a.add(&b).expect("broadcast add");
    assert_eq!(c.shape(), &[1, 1, 43, 1]);
    // Symmetric: high-rank-vector on the rhs.
    let d = b.add(&a).expect("broadcast add reverse");
    assert_eq!(d.shape(), &[1, 1, 43, 1]);
}

proptest! {
    #[test]
    fn broadcast_shape_matches_numpy_rules(
        a in arb_shape(),
        b in arb_shape(),
    ) {
        let ours = broadcast_shape(&a, &b);
        let reference = reference_broadcast(&a, &b);
        prop_assert_eq!(ours, reference, "broadcast_shape({:?}, {:?})", a, b);
    }

    #[test]
    fn add_is_commutative((a, b) in arb_broadcastable_tensors()) {
        let ab = a.add(&b).expect("broadcast-compatible add");
        let ba = b.add(&a).expect("broadcast-compatible add");
        prop_assert_eq!(ab.shape(), ba.shape());
        for (x, y) in ab.data().iter().zip(ba.data().iter()) {
            prop_assert!((x - y).abs() < 1e-5, "commutative violation: {x} vs {y}");
        }
    }

    #[test]
    fn scale_preserves_shape(shape in arb_shape(), factor in -100.0f32..100.0) {
        let len: usize = shape.iter().product();
        let data: Vec<f32> = (0..len).map(|i| (i as f32) * 0.1).collect();
        let t = Tensor::from_vec(shape.clone(), data).expect("valid tensor");
        let scaled = t.scale(factor);
        prop_assert_eq!(scaled.shape(), t.shape(), "scale changed shape");
    }

    // ── Mul commutativity ──────────────────────────────────────────────
    #[test]
    fn mul_is_commutative((a, b) in arb_broadcastable_tensors()) {
        let ab = a.mul(&b).expect("broadcast-compatible mul");
        let ba = b.mul(&a).expect("broadcast-compatible mul");
        prop_assert_eq!(ab.shape(), ba.shape());
        for (x, y) in ab.data().iter().zip(ba.data().iter()) {
            prop_assert!((x - y).abs() < 1e-5, "mul commutative violation: {x} vs {y}");
        }
    }

    // ── Add associativity ──────────────────────────────────────────────
    #[test]
    fn add_is_associative((a, b, c) in arb_same_shape_triple()) {
        let ab_c = a.add(&b).expect("add").add(&c).expect("add");
        let a_bc = a.add(&b.add(&c).expect("add")).expect("add");
        prop_assert_eq!(ab_c.shape(), a_bc.shape());
        for (x, y) in ab_c.data().iter().zip(a_bc.data().iter()) {
            prop_assert!(
                (x - y).abs() < 1e-3,
                "associativity violation: {x} vs {y}"
            );
        }
    }

    // ── Scale distributivity ───────────────────────────────────────────
    #[test]
    fn scale_distributes_over_add(
        (a, b) in arb_same_shape_tensors(),
        s in -10.0f32..10.0,
    ) {
        let lhs = a.add(&b).expect("add").scale(s);
        let rhs = a.scale(s).add(&b.scale(s)).expect("add");
        prop_assert_eq!(lhs.shape(), rhs.shape());
        for (x, y) in lhs.data().iter().zip(rhs.data().iter()) {
            prop_assert!(
                (x - y).abs() < 1e-2,
                "scale distributivity violation: {x} vs {y}"
            );
        }
    }

    // ── Reshape round-trip ─────────────────────────────────────────────
    #[test]
    fn reshape_roundtrip(shape in arb_shape()) {
        let len: usize = shape.iter().product();
        if len == 0 {
            return Ok(());
        }
        let data: Vec<f32> = (0..len).map(|i| i as f32).collect();
        let t = Tensor::from_vec(shape.clone(), data).expect("valid tensor");
        let flat = t.reshape(vec![len]).expect("reshape to flat");
        let restored = flat.reshape(shape.clone()).expect("reshape back");
        prop_assert_eq!(restored.shape(), t.shape());
        for (x, y) in restored.data().iter().zip(t.data().iter()) {
            prop_assert_eq!(x, y, "reshape roundtrip data mismatch");
        }
    }

    // ── Sum invariant ──────────────────────────────────────────────────
    #[test]
    fn sum_of_add_equals_sum_plus_sum((a, b) in arb_same_shape_tensors()) {
        let sum_ab = a.add(&b).expect("add").sum();
        let sum_a_plus_sum_b = a.sum() + b.sum();
        prop_assert!(
            (sum_ab - sum_a_plus_sum_b).abs() < 1e-1 + sum_ab.abs() * 1e-4,
            "sum invariant violated: sum(a+b)={sum_ab}, sum(a)+sum(b)={sum_a_plus_sum_b}"
        );
    }

    // ── DType F16 round-trip ───────────────────────────────────────────
    #[test]
    fn f16_roundtrip(shape in arb_shape()) {
        let len: usize = shape.iter().product();
        if len == 0 {
            return Ok(());
        }
        // Use values in [-1, 1] to stay well within f16 representable range
        let data: Vec<f32> = (0..len).map(|i| ((i % 200) as f32 - 100.0) / 100.0).collect();
        let t = Tensor::from_vec(shape, data).expect("valid tensor");
        let roundtrip = t
            .to_dtype(super::DType::F16)
            .to_dtype(super::DType::F32);
        prop_assert_eq!(roundtrip.shape(), t.shape());
        for (x, y) in roundtrip.data().iter().zip(t.data().iter()) {
            // f16 has ~3 decimal digits of precision; tolerance ~1e-3
            prop_assert!(
                (x - y).abs() < 2e-3,
                "f16 roundtrip error: original={y}, recovered={x}"
            );
        }
    }

    // ── (m) Clamp bounds ──────────────────────────────────────────────
    #[test]
    fn clamp_bounds(
        data in proptest::collection::vec(-100.0f32..100.0, 16),
        lo in -50.0f32..0.0,
        hi in 0.0f32..50.0,
    ) {
        let t = Tensor::from_vec(vec![4, 4], data).expect("valid tensor");
        let clamped = t.clamp(lo, hi);

        prop_assert_eq!(clamped.shape(), &[4, 4], "clamp changed shape");

        for (i, &val) in clamped.data().iter().enumerate() {
            prop_assert!(
                val >= lo - 1e-6 && val <= hi + 1e-6,
                "clamp violation at {i}: val={val}, expected [{lo}, {hi}]"
            );
        }
    }
}
