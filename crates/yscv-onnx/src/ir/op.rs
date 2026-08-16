//! Operator identity for the optimization IR.
//!
//! Passes match on operator type constantly, and `String` comparison in an
//! O(N) scan is the wrong primitive for that. [`Op`] interns the operators the
//! optimizer actually reasons about into a `Copy` enum; everything else — the
//! long tail of the ~122 runtime operators that no pass inspects — stays a
//! string in [`Op::Other`].
//!
//! This is deliberately *not* a mirror of the runtime dispatch table in
//! `runner/dispatch.rs`. Duplicating 122 arms here would create a second table
//! to keep in lockstep for no benefit: the IR only needs to distinguish the ops
//! that passes branch on.

/// Operator type of an IR node.
///
/// `Other` carries any operator the optimizer has no opinion about, so lowering
/// back to `OnnxNode` is lossless for every graph the loader can produce.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum Op {
    Add,
    AveragePool,
    BatchNormalization,
    Clip,
    Concat,
    Constant,
    Conv,
    ConvTranspose,
    DeformConv,
    DepthToSpace,
    DequantizeLinear,
    Div,
    Dropout,
    Flatten,
    Gather,
    Gemm,
    GlobalAveragePool,
    Identity,
    MatMul,
    MaxPool,
    Mul,
    Pow,
    QuantizeLinear,
    Relu,
    Reshape,
    Sigmoid,
    Squeeze,
    Sub,
    Tanh,
    Transpose,
    Unsqueeze,
    /// Any operator the optimizer does not special-case.
    Other(Box<str>),
    // --- Fused operators ---
    /// Fused `BatchNormalization` + `Relu`, emitted by `fuse_bn_relu`.
    BatchNormRelu,
    /// Fused `Conv` + `Relu`, emitted by `fuse_conv_relu`.
    ConvRelu,
    /// Fused `Conv` + `Sigmoid` + `Mul`. Recognized downstream but currently
    /// produced by no pass.
    ConvSilu,
    /// Fused `Conv` + `HardSwish`, emitted by `fuse_conv_hardswish`.
    ConvHardSwish,
}

/// Pairs each interned variant with its exact ONNX (or mangled-fusion) op-type
/// string, so `from_str` and `as_str` cannot drift apart.
macro_rules! op_table {
    ($($variant:ident => $name:literal),+ $(,)?) => {
        impl Op {
            /// Interns an op-type string. Unrecognized types become
            /// [`Op::Other`], never an error — the IR must round-trip any graph
            /// the loader accepts.
            pub(crate) fn from_op_type(op_type: &str) -> Self {
                match op_type {
                    $($name => Op::$variant,)+
                    other => Op::Other(other.into()),
                }
            }

            /// The op-type string this operator lowers back to.
            pub(crate) fn as_str(&self) -> &str {
                match self {
                    $(Op::$variant => $name,)+
                    Op::Other(name) => name,
                }
            }
        }

        #[cfg(test)]
        const ALL_INTERNED: &[(Op, &str)] = &[$((Op::$variant, $name)),+];
    };
}

op_table! {
    Add => "Add",
    AveragePool => "AveragePool",
    BatchNormalization => "BatchNormalization",
    BatchNormRelu => "BatchNormalization_Relu",
    Clip => "Clip",
    Concat => "Concat",
    Constant => "Constant",
    Conv => "Conv",
    ConvRelu => "Conv_Relu",
    ConvSilu => "Conv_SiLU",
    ConvHardSwish => "Conv_HardSwish",
    ConvTranspose => "ConvTranspose",
    DeformConv => "DeformConv",
    DepthToSpace => "DepthToSpace",
    DequantizeLinear => "DequantizeLinear",
    Div => "Div",
    Dropout => "Dropout",
    Flatten => "Flatten",
    Gather => "Gather",
    Gemm => "Gemm",
    GlobalAveragePool => "GlobalAveragePool",
    Identity => "Identity",
    MatMul => "MatMul",
    MaxPool => "MaxPool",
    Mul => "Mul",
    Pow => "Pow",
    QuantizeLinear => "QuantizeLinear",
    Relu => "Relu",
    Reshape => "Reshape",
    Sigmoid => "Sigmoid",
    Squeeze => "Squeeze",
    Sub => "Sub",
    Tanh => "Tanh",
    Transpose => "Transpose",
    Unsqueeze => "Unsqueeze",
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every interned variant must survive a string round-trip exactly, or
    /// lowering back to ONNX would silently rename operators.
    #[test]
    fn interned_ops_round_trip() {
        for (op, name) in ALL_INTERNED {
            assert_eq!(op.as_str(), *name, "as_str drifted for {op:?}");
            assert_eq!(
                Op::from_op_type(name),
                *op,
                "from_op_type drifted for {name}"
            );
        }
    }

    /// The table must not map two variants onto the same string, which would
    /// make `from_op_type` unable to recover one of them.
    #[test]
    fn interned_names_are_unique() {
        let mut names: Vec<&str> = ALL_INTERNED.iter().map(|(_, n)| *n).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(before, names.len(), "duplicate op-type string in the table");
    }

    #[test]
    fn unknown_ops_round_trip_through_other() {
        let op = Op::from_op_type("ScatterElements");
        assert_eq!(op, Op::Other("ScatterElements".into()));
        assert_eq!(op.as_str(), "ScatterElements");
    }

    /// Empty op-type is malformed input, not a reason to lose data.
    #[test]
    fn empty_op_type_round_trips() {
        assert_eq!(Op::from_op_type("").as_str(), "");
    }
}
