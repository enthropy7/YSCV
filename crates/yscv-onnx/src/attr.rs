//! Operator attribute names.
//!
//! ONNX attribute names are a closed, per-operator vocabulary — `strides`,
//! `pads`, `axis` and so on — but they were spelled as string literals at every
//! one of the ~200 sites that read or wrote one. A typo produced a silently
//! missing attribute and a default value rather than a compile error, and there
//! was no single place to see which names the runtime understands.
//!
//! [`Attr`] interns them. Names outside the list survive in [`Attr::Other`], so
//! decoding and re-exporting a model is lossless even for operators this
//! runtime does not interpret.

/// The name of an operator attribute.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Attr {
    AlignCorners,
    Alpha,
    Axes,
    Axis,
    BatchDims,
    Beta,
    Bias,
    BlockSize,
    Dilations,
    Direction,
    Epsilon,
    Equation,
    Fmod,
    Gamma,
    Group,
    KeepDims,
    KernelShape,
    KvNumHeads,
    Largest,
    Mode,
    NumHeads,
    OutputHeight,
    OutputPadding,
    OutputWidth,
    P,
    PaddingMode,
    Pads,
    Perm,
    SamplingRatio,
    Size,
    SpatialScale,
    Split,
    Strides,
    To,
    TransA,
    TransB,
    Upper,
    Value,
    ValueFloat,
    ValueFloats,
    ValueInt,
    ValueInts,
    /// An attribute this runtime does not interpret, kept verbatim so the model
    /// round-trips.
    Other(Box<str>),
}

/// Pairs each variant with its exact ONNX spelling, so `from_name` and `as_str`
/// cannot drift apart. Note that ONNX is not consistently snake_case — `transA`
/// and `transB` are camelCase — which is precisely the kind of detail that made
/// the raw string literals error-prone.
macro_rules! attr_table {
    ($($variant:ident => $name:literal),+ $(,)?) => {
        impl Attr {
            /// Interns an attribute name. Unrecognized names become
            /// [`Attr::Other`] rather than an error.
            pub fn from_name(name: &str) -> Self {
                match name {
                    $($name => Attr::$variant,)+
                    other => Attr::Other(other.into()),
                }
            }

            /// The ONNX spelling of this attribute.
            pub fn as_str(&self) -> &str {
                match self {
                    $(Attr::$variant => $name,)+
                    Attr::Other(name) => name,
                }
            }
        }

        #[cfg(test)]
        const ALL_INTERNED: &[(Attr, &str)] = &[$((Attr::$variant, $name)),+];
    };
}

attr_table! {
    AlignCorners => "align_corners",
    Alpha => "alpha",
    Axes => "axes",
    Axis => "axis",
    BatchDims => "batch_dims",
    Beta => "beta",
    Bias => "bias",
    BlockSize => "blocksize",
    Dilations => "dilations",
    Direction => "direction",
    Epsilon => "epsilon",
    Equation => "equation",
    Fmod => "fmod",
    Gamma => "gamma",
    Group => "group",
    KeepDims => "keepdims",
    KernelShape => "kernel_shape",
    KvNumHeads => "kv_num_heads",
    Largest => "largest",
    Mode => "mode",
    NumHeads => "num_heads",
    OutputHeight => "output_height",
    OutputPadding => "output_padding",
    OutputWidth => "output_width",
    P => "p",
    PaddingMode => "padding_mode",
    Pads => "pads",
    Perm => "perm",
    SamplingRatio => "sampling_ratio",
    Size => "size",
    SpatialScale => "spatial_scale",
    Split => "split",
    Strides => "strides",
    To => "to",
    TransA => "transA",
    TransB => "transB",
    Upper => "upper",
    Value => "value",
    ValueFloat => "value_float",
    ValueFloats => "value_floats",
    ValueInt => "value_int",
    ValueInts => "value_ints",
}

impl std::fmt::Display for Attr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every interned name must survive a round-trip exactly, or re-exporting a
    /// model would rename its attributes.
    #[test]
    fn interned_attrs_round_trip() {
        for (attr, name) in ALL_INTERNED {
            assert_eq!(attr.as_str(), *name, "as_str drifted for {attr:?}");
            assert_eq!(Attr::from_name(name), *attr, "from_name drifted for {name}");
        }
    }

    /// Two variants mapping to one name would make `from_name` unable to
    /// recover one of them.
    #[test]
    fn interned_names_are_unique() {
        let mut names: Vec<&str> = ALL_INTERNED.iter().map(|(_, n)| *n).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(before, names.len(), "duplicate attribute name in the table");
    }

    #[test]
    fn unknown_attrs_round_trip_through_other() {
        let attr = Attr::from_name("coordinate_transformation_mode");
        assert_eq!(attr, Attr::Other("coordinate_transformation_mode".into()));
        assert_eq!(attr.as_str(), "coordinate_transformation_mode");
    }

    /// ONNX mixes snake_case and camelCase; the table must preserve both
    /// exactly rather than normalising.
    #[test]
    fn camel_case_names_are_preserved() {
        assert_eq!(Attr::TransB.as_str(), "transB");
        assert_eq!(Attr::from_name("transB"), Attr::TransB);
        assert_eq!(Attr::from_name("transb"), Attr::Other("transb".into()));
    }
}
