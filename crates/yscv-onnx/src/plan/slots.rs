//! Phase 1: dense slot ids and the per-node tables the hot loop reads.
//!
//! Every tensor name in the graph gets an integer id, so the per-inference
//! environment indexes a `Vec` instead of hashing strings. Node inputs and
//! outputs are pre-resolved to those ids for the same reason, operator types
//! are classified once into [`NodeKind`], and use counts are recorded so the
//! runner can free a tensor as soon as its last reader has run.
//!
//! Also assigns the tower-parallel branch tags, which is scheduling rather than
//! indexing, but depends only on the graph and so belongs in the same sweep.

use rustc_hash::{FxHashMap, FxHashSet};
use yscv_tensor::Tensor;

use crate::loader::OnnxNode;

use super::*;

/// Everything downstream phases address values and nodes by.
pub(super) struct SlotIndex {
    pub(super) name_to_id: FxHashMap<String, usize>,
    pub(super) use_counts: FxHashMap<String, usize>,
    pub(super) use_counts_by_id: Vec<usize>,
    pub(super) node_kinds: Vec<NodeKind>,
    pub(super) node_branches: Vec<u8>,
    pub(super) node_input_ids: Vec<Vec<Option<usize>>>,
    pub(super) node_output_ids: Vec<Vec<Option<usize>>>,
}

pub(super) fn assign_slots(
    inputs: &[String],
    outputs: &[String],
    initializers: &FxHashMap<String, Tensor>,
    nodes: &[OnnxNode],
) -> SlotIndex {
    let mut names: FxHashSet<&str> = FxHashSet::default();
    for name in inputs {
        names.insert(name.as_str());
    }
    for name in outputs {
        names.insert(name.as_str());
    }
    for name in initializers.keys() {
        names.insert(name.as_str());
    }
    for node in nodes {
        for name in &node.inputs {
            names.insert(name.as_str());
        }
        for name in &node.outputs {
            names.insert(name.as_str());
        }
    }
    let name_to_id: FxHashMap<String, usize> = names
        .into_iter()
        .enumerate()
        .map(|(id, name)| (name.to_string(), id))
        .collect();

    let mut use_counts: FxHashMap<String, usize> = FxHashMap::default();
    for node in nodes {
        for inp in &node.inputs {
            if !inp.is_empty() {
                *use_counts.entry(inp.clone()).or_insert(0) += 1;
            }
        }
    }
    let mut use_counts_by_id = vec![0usize; name_to_id.len()];
    for (name, count) in &use_counts {
        if let Some(&id) = name_to_id.get(name) {
            use_counts_by_id[id] = *count;
        }
    }
    let node_kinds: Vec<NodeKind> = nodes
        .iter()
        .map(|node| NodeKind::from_op_type(&node.op_type))
        .collect();

    // Tower-parallel branch classification. For a siamese graph we want two
    // input-rooted subgraphs to run concurrently, then a merge tail. Nodes
    // are tagged 0 = reachable from first dynamic input only, 1 = second only,
    // 2 = shared/merge. If either branch ends up too small, we clear the
    // vector to signal "no parallel split".
    let node_branches: Vec<u8> = {
        let dyn_inputs: Vec<&str> = inputs
            .iter()
            .map(|s| s.as_str())
            .filter(|s| !initializers.contains_key(*s))
            .collect();
        if dyn_inputs.len() >= 2 {
            let mut tensor_branch: FxHashMap<&str, u8> = FxHashMap::default();
            tensor_branch.insert(dyn_inputs[0], 0);
            tensor_branch.insert(dyn_inputs[1], 1);
            let mut branches = Vec::with_capacity(nodes.len());
            for node in nodes {
                let mut seen = 0u8; // bitmask: bit 0 = branch 0, bit 1 = branch 1
                for inp in &node.inputs {
                    if inp.is_empty() || initializers.contains_key(inp.as_str()) {
                        continue;
                    }
                    match tensor_branch.get(inp.as_str()) {
                        Some(&0) => seen |= 1,
                        Some(&1) => seen |= 2,
                        Some(&2) => seen |= 3,
                        _ => {}
                    }
                }
                let branch = match seen {
                    0 => 2, // constant-fed node treated as merge-safe
                    1 => 0,
                    2 => 1,
                    _ => 2,
                };
                for out in &node.outputs {
                    tensor_branch.insert(out.as_str(), branch);
                }
                branches.push(branch);
            }
            let b0 = branches.iter().filter(|&&b| b == 0).count();
            let b1 = branches.iter().filter(|&&b| b == 1).count();
            // Require both branches to carry meaningful work, otherwise the
            // parallel split's overhead (env fork, rayon::join) dominates.
            if b0 >= 10 && b1 >= 10 {
                branches
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        }
    };

    // Pre-resolve input names to slot IDs for O(1) hot-path lookups.
    let node_input_ids: Vec<Vec<Option<usize>>> = nodes
        .iter()
        .map(|node| {
            node.inputs
                .iter()
                .map(|name| {
                    if name.is_empty() {
                        None
                    } else {
                        name_to_id.get(name).copied()
                    }
                })
                .collect()
        })
        .collect();

    // pre-resolve output names to slot IDs. Used by
    // `env.insert_by_id` on the hot path to skip the FxHashMap lookup
    // inside `resolve_id`. `node_input_ids` was already cached; this
    // extends the same optimisation to output slots.
    let node_output_ids: Vec<Vec<Option<usize>>> = nodes
        .iter()
        .map(|node| {
            node.outputs
                .iter()
                .map(|name| {
                    if name.is_empty() {
                        None
                    } else {
                        name_to_id.get(name).copied()
                    }
                })
                .collect()
        })
        .collect();

    SlotIndex {
        name_to_id,
        use_counts,
        use_counts_by_id,
        node_kinds,
        node_branches,
        node_input_ids,
        node_output_ids,
    }
}
