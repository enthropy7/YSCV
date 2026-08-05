//! Pass infrastructure: a trait, a driver, and a way to turn passes off.
//!
//! The pre-IR optimizer was a hard-coded sequence of twelve function calls with
//! no way to disable one, no record of what each did, and a comment explaining
//! that the order was load-bearing because the passes matched on positional
//! adjacency. With def-use matching that ordering constraint is gone, so the
//! driver can iterate to a fixpoint instead of relying on a hand-tuned sequence.

use crate::error::OnnxError;
use crate::ir::Graph;

/// Whether a pass modified the graph. Drives the driver's fixpoint loop, so
/// reporting `true` when nothing changed costs an extra iteration and reporting
/// `false` when something did can leave a rewrite unfinished.
pub(crate) type Changed = bool;

/// A graph-to-graph transformation.
pub(crate) trait Pass {
    /// Stable identifier, used for `YSCV_ONNX_PASSES` and logging.
    fn name(&self) -> &'static str;

    /// Applies the transformation. Must leave the graph's invariants intact —
    /// [`Graph::validate`](crate::ir::Graph::validate) runs behind debug
    /// assertions on every mutation, so a violation surfaces in tests.
    fn run(&self, graph: &mut Graph) -> Result<Changed, OnnxError>;
}

/// How many times the driver may sweep the pipeline before giving up on
/// reaching a fixpoint.
///
/// Three is enough for the passes here — each sweep only re-fires when an
/// earlier pass exposed a new opportunity, which in practice bottoms out after
/// one extra round. The cap exists so a pair of passes that undo each other
/// cannot hang the loader.
const DEFAULT_MAX_SWEEPS: usize = 3;

/// Runs a pipeline of passes to a fixpoint.
pub(crate) struct PassManager {
    passes: Vec<Box<dyn Pass>>,
    max_sweeps: usize,
    disabled: Vec<String>,
    log: bool,
}

impl PassManager {
    /// Builds a manager with the environment-derived configuration.
    ///
    /// `YSCV_ONNX_PASSES` takes a comma-separated list of `-name` entries to
    /// disable individual passes, for bisecting a miscompiled model without a
    /// rebuild. `YSCV_ONNX_PASS_LOG=1` reports what each sweep did.
    pub(crate) fn new(passes: Vec<Box<dyn Pass>>) -> Self {
        let disabled = std::env::var("YSCV_ONNX_PASSES")
            .ok()
            .map(|spec| {
                spec.split(',')
                    .filter_map(|entry| entry.trim().strip_prefix('-'))
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        PassManager {
            passes,
            max_sweeps: DEFAULT_MAX_SWEEPS,
            disabled,
            log: std::env::var("YSCV_ONNX_PASS_LOG").as_deref() == Ok("1"),
        }
    }

    fn is_enabled(&self, name: &str) -> bool {
        !self.disabled.iter().any(|d| d == name)
    }

    /// Sweeps the pipeline until no pass reports a change, or the sweep cap is
    /// hit.
    ///
    /// A pass returning an error aborts the run; the graph is left as that pass
    /// left it, which is why passes are expected to validate before mutating
    /// rather than bailing out halfway.
    pub(crate) fn run(&self, graph: &mut Graph) -> Result<(), OnnxError> {
        for sweep in 0..self.max_sweeps {
            let mut any_changed = false;
            for pass in &self.passes {
                if !self.is_enabled(pass.name()) {
                    continue;
                }
                let before = graph.node_count();
                let changed = pass.run(graph)?;
                any_changed |= changed;
                if self.log && changed {
                    eprintln!(
                        "[yscv-onnx] sweep {sweep}: {} changed the graph ({} -> {} nodes)",
                        pass.name(),
                        before,
                        graph.node_count(),
                    );
                }
            }
            if !any_changed {
                graph.compact();
                return Ok(());
            }
        }

        if self.log {
            eprintln!(
                "[yscv-onnx] pipeline did not reach a fixpoint in {} sweeps",
                self.max_sweeps
            );
        }
        graph.compact();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    /// Shared run counter. `Rc` rather than a bare `Cell`, so the test observes
    /// the same cell the pass increments.
    type Counter = Rc<Cell<usize>>;

    /// Reports a change a fixed number of times, then stops — a stand-in for a
    /// pass that keeps finding work as earlier passes expose it.
    struct ChangesNTimes {
        remaining: Cell<usize>,
        runs: Counter,
    }

    impl Pass for ChangesNTimes {
        fn name(&self) -> &'static str {
            "changes_n_times"
        }
        fn run(&self, _graph: &mut Graph) -> Result<Changed, OnnxError> {
            self.runs.set(self.runs.get() + 1);
            let left = self.remaining.get();
            if left == 0 {
                return Ok(false);
            }
            self.remaining.set(left - 1);
            Ok(true)
        }
    }

    /// Always reports a change, so the driver can never reach a fixpoint.
    struct NeverSettles(Counter);

    impl Pass for NeverSettles {
        fn name(&self) -> &'static str {
            "never_settles"
        }
        fn run(&self, _graph: &mut Graph) -> Result<Changed, OnnxError> {
            self.0.set(self.0.get() + 1);
            Ok(true)
        }
    }

    fn empty_graph() -> Graph {
        Graph::new()
    }

    fn manager_of(
        passes: Vec<Box<dyn Pass>>,
        max_sweeps: usize,
        disabled: Vec<String>,
    ) -> PassManager {
        PassManager {
            passes,
            max_sweeps,
            disabled,
            log: false,
        }
    }

    #[test]
    fn stops_once_no_pass_reports_a_change() {
        let runs: Counter = Rc::new(Cell::new(0));
        let manager = manager_of(
            vec![Box::new(ChangesNTimes {
                remaining: Cell::new(1),
                runs: Rc::clone(&runs),
            })],
            DEFAULT_MAX_SWEEPS,
            Vec::new(),
        );

        manager.run(&mut empty_graph()).expect("pipeline runs");
        assert_eq!(runs.get(), 2, "one sweep that changed, one that did not");
    }

    /// A non-converging pipeline must terminate rather than hang model loading.
    #[test]
    fn sweep_cap_bounds_a_non_converging_pipeline() {
        let runs: Counter = Rc::new(Cell::new(0));
        let manager = manager_of(
            vec![Box::new(NeverSettles(Rc::clone(&runs)))],
            3,
            Vec::new(),
        );

        manager.run(&mut empty_graph()).expect("pipeline runs");
        assert_eq!(runs.get(), 3, "capped at max_sweeps");
    }

    #[test]
    fn disabled_passes_do_not_run() {
        let runs: Counter = Rc::new(Cell::new(0));
        let manager = manager_of(
            vec![Box::new(NeverSettles(Rc::clone(&runs)))],
            DEFAULT_MAX_SWEEPS,
            vec!["never_settles".to_string()],
        );

        manager.run(&mut empty_graph()).expect("pipeline runs");
        assert_eq!(runs.get(), 0);
    }

    /// An error aborts the run instead of being swallowed into a fixpoint.
    #[test]
    fn pass_error_propagates() {
        struct Fails;
        impl Pass for Fails {
            fn name(&self) -> &'static str {
                "fails"
            }
            fn run(&self, _graph: &mut Graph) -> Result<Changed, OnnxError> {
                Err(OnnxError::DecodeFailed {
                    message: "boom".to_string(),
                })
            }
        }
        let manager = manager_of(vec![Box::new(Fails)], DEFAULT_MAX_SWEEPS, Vec::new());
        assert!(manager.run(&mut empty_graph()).is_err());
    }
}
