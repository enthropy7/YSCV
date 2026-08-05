use crate::error::OnnxError;
use crate::ir::{Changed, Graph, Op, Pass};

/// Folds a trailing `Relu` into the operator that produces its input, by
/// retagging that operator.
///
/// This is annotation-only: no arithmetic moves, and no edges are rewired
/// beyond the fused node taking over the Relu's output. The kernel picks the
/// activation up from the operator type — `NodeAction::Conv { activation }` is
/// derived from it in `build_runtime_index`, which has no `Conv -> Relu`
/// pattern matcher of its own, so this pass is what makes the fused kernel
/// reachable at all.
///
/// `fuse_conv_relu` and `fuse_bn_relu` were separate files with the same body
/// and their own copies of the O(N) consumer rescan. They differ only in which
/// operator absorbs the activation, so they are one pass parameterized twice.
pub(crate) struct FuseActivation {
    name: &'static str,
    /// The producer eligible to absorb the activation.
    base: Op,
    /// The activation being absorbed.
    activation: Op,
    /// What the producer becomes.
    fused: Op,
}

impl FuseActivation {
    pub(crate) fn conv_relu() -> Self {
        Self {
            name: "fuse_conv_relu",
            base: Op::Conv,
            activation: Op::Relu,
            fused: Op::ConvRelu,
        }
    }

    pub(crate) fn bn_relu() -> Self {
        Self {
            name: "fuse_bn_relu",
            base: Op::BatchNormalization,
            activation: Op::Relu,
            fused: Op::BatchNormRelu,
        }
    }
}

impl Pass for FuseActivation {
    fn name(&self) -> &'static str {
        self.name
    }

    fn run(&self, graph: &mut Graph) -> Result<Changed, OnnxError> {
        let mut changed = false;

        for base_id in graph.node_ids().collect::<Vec<_>>() {
            let Some(base) = graph.node(base_id) else {
                continue;
            };
            if base.op != self.base {
                continue;
            }
            let Some(&produced) = base.outputs.first() else {
                continue;
            };

            // Fusing rewrites the producer's output in place, so the
            // activation has to be the only thing observing it — including via
            // the graph's outputs, which `sole_consumer` already accounts for.
            let Some(use_site) = graph.sole_consumer(produced) else {
                continue;
            };
            let Some(act) = graph.node(use_site.node) else {
                continue;
            };
            if act.op != self.activation || act.inputs.len() != 1 || use_site.port != 0 {
                continue;
            }
            if act.outputs.len() != 1 {
                continue;
            }

            // The fused node takes over the activation's output value, so
            // everything downstream keeps reading the same edge under the same
            // name.
            graph.absorb_consumer(base_id, use_site.node);
            graph.set_op(base_id, self.fused.clone());
            changed = true;
        }

        Ok(changed)
    }
}
