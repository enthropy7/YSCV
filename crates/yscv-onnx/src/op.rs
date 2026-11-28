//! ONNX and internal fused operator identities.
//!
//! Known operators are enum variants so graph construction and dispatch do not
//! rely on allocated strings. Unknown operators remain lossless in
//! [`OnnxOp::Other`] for source inspection and export.

macro_rules! onnx_op_table {
    ($($variant:ident => $name:literal),+ $(,)?) => {
        /// An ONNX operator or an internal fused operator produced by IR passes.
        #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub enum OnnxOp {
            $($variant),+,
            /// An operator unknown to this runtime, preserved verbatim.
            Other(Box<str>),
        }

        impl OnnxOp {
            /// Converts an ONNX operator name into its typed identity.
            pub fn from_op_type(op_type: &str) -> Self {
                match op_type {
                    $($name => Self::$variant,)+
                    other => Self::Other(other.into()),
                }
            }

            /// The exact ONNX or internal fused operator name.
            pub fn as_str(&self) -> &str {
                match self {
                    $(Self::$variant => $name,)+
                    Self::Other(name) => name,
                }
            }
        }

        #[cfg(test)]
        const ALL_INTERNED: &[(OnnxOp, &str)] = &[$((OnnxOp::$variant, $name)),+];
    };
}

onnx_op_table! {
    Abs => "Abs",
    Acos => "Acos",
    Acosh => "Acosh",
    Add => "Add",
    And => "And",
    ArgMax => "ArgMax",
    ArgMin => "ArgMin",
    Asin => "Asin",
    Asinh => "Asinh",
    Atan => "Atan",
    Atanh => "Atanh",
    AveragePool => "AveragePool",
    BatchNormalization => "BatchNormalization",
    BatchNormRelu => "BatchNormalization_Relu",
    BitShift => "BitShift",
    Cast => "Cast",
    Ceil => "Ceil",
    Celu => "Celu",
    Clip => "Clip",
    Compress => "Compress",
    Concat => "Concat",
    Constant => "Constant",
    ConstantOfShape => "ConstantOfShape",
    Conv => "Conv",
    ConvHardSwish => "Conv_HardSwish",
    ConvInteger => "ConvInteger",
    ConvRelu => "Conv_Relu",
    ConvSilu => "Conv_SiLU",
    ConvTranspose => "ConvTranspose",
    Cos => "Cos",
    Cosh => "Cosh",
    CumSum => "CumSum",
    DeformConv => "DeformConv",
    DepthToSpace => "DepthToSpace",
    DequantizeLinear => "DequantizeLinear",
    Div => "Div",
    Dropout => "Dropout",
    DynamicQuantizeLinear => "DynamicQuantizeLinear",
    Einsum => "Einsum",
    Elu => "Elu",
    Equal => "Equal",
    Erf => "Erf",
    Exp => "Exp",
    Expand => "Expand",
    Flatten => "Flatten",
    Floor => "Floor",
    Gather => "Gather",
    GatherElements => "GatherElements",
    GatherND => "GatherND",
    Gelu => "Gelu",
    Gemm => "Gemm",
    GlobalAveragePool => "GlobalAveragePool",
    Greater => "Greater",
    GreaterOrEqual => "GreaterOrEqual",
    GridSample => "GridSample",
    GroupQueryAttention => "GroupQueryAttention",
    HardSigmoid => "HardSigmoid",
    HardSwish => "HardSwish",
    Hardmax => "Hardmax",
    Identity => "Identity",
    InstanceNormalization => "InstanceNormalization",
    IsInf => "IsInf",
    IsNaN => "IsNaN",
    LRN => "LRN",
    LayerNormalization => "LayerNormalization",
    LeakyRelu => "LeakyRelu",
    Less => "Less",
    LessOrEqual => "LessOrEqual",
    Log => "Log",
    LpNormalization => "LpNormalization",
    MatMul => "MatMul",
    MatMulInteger => "MatMulInteger",
    Max => "Max",
    MaxPool => "MaxPool",
    Mean => "Mean",
    Min => "Min",
    Mish => "Mish",
    Mod => "Mod",
    Mul => "Mul",
    Neg => "Neg",
    NonMaxSuppression => "NonMaxSuppression",
    NonZero => "NonZero",
    Not => "Not",
    OneHot => "OneHot",
    Or => "Or",
    Pad => "Pad",
    Pow => "Pow",
    QLinearConv => "QLinearConv",
    QLinearMatMul => "QLinearMatMul",
    QuantizeLinear => "QuantizeLinear",
    Range => "Range",
    Reciprocal => "Reciprocal",
    ReduceL1 => "ReduceL1",
    ReduceL2 => "ReduceL2",
    ReduceMax => "ReduceMax",
    ReduceMean => "ReduceMean",
    ReduceMin => "ReduceMin",
    ReduceProd => "ReduceProd",
    ReduceSum => "ReduceSum",
    Relu => "Relu",
    Reshape => "Reshape",
    Resize => "Resize",
    RoiAlign => "RoiAlign",
    Round => "Round",
    ScatterElements => "ScatterElements",
    ScatterND => "ScatterND",
    Selu => "Selu",
    Shape => "Shape",
    Sigmoid => "Sigmoid",
    Sign => "Sign",
    Sin => "Sin",
    Sinh => "Sinh",
    Slice => "Slice",
    Softmax => "Softmax",
    Softplus => "Softplus",
    Softsign => "Softsign",
    SpaceToDepth => "SpaceToDepth",
    Split => "Split",
    Sqrt => "Sqrt",
    Squeeze => "Squeeze",
    Sub => "Sub",
    Sum => "Sum",
    Tan => "Tan",
    Tanh => "Tanh",
    ThresholdedRelu => "ThresholdedRelu",
    Tile => "Tile",
    TopK => "TopK",
    Transpose => "Transpose",
    Trilu => "Trilu",
    Unsqueeze => "Unsqueeze",
    Upsample => "Upsample",
    Where => "Where",
    Xor => "Xor",
}

impl std::fmt::Display for OnnxOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl AsRef<str> for OnnxOp {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl From<&str> for OnnxOp {
    fn from(op_type: &str) -> Self {
        Self::from_op_type(op_type)
    }
}

impl From<String> for OnnxOp {
    fn from(op_type: String) -> Self {
        Self::from_op_type(&op_type)
    }
}

impl PartialEq<&str> for OnnxOp {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

impl PartialEq<OnnxOp> for &str {
    fn eq(&self, other: &OnnxOp) -> bool {
        *self == other.as_str()
    }
}

impl PartialEq<String> for OnnxOp {
    fn eq(&self, other: &String) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<OnnxOp> for String {
    fn eq(&self, other: &OnnxOp) -> bool {
        self == other.as_str()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interned_ops_round_trip() {
        for (op, name) in ALL_INTERNED {
            assert_eq!(op.as_str(), *name, "as_str drifted for {op:?}");
            assert_eq!(
                OnnxOp::from_op_type(name),
                *op,
                "from_op_type drifted for {name}"
            );
        }
    }

    #[test]
    fn interned_names_are_unique() {
        let mut names: Vec<&str> = ALL_INTERNED.iter().map(|(_, name)| *name).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(before, names.len(), "duplicate operator name in the table");
    }

    #[test]
    fn unknown_ops_round_trip_through_other() {
        let op = OnnxOp::from_op_type("CustomOp");
        assert_eq!(op, OnnxOp::Other("CustomOp".into()));
        assert_eq!(op.as_str(), "CustomOp");
    }

    #[test]
    fn empty_op_type_round_trips() {
        assert_eq!(OnnxOp::from_op_type("").as_str(), "");
    }
}
