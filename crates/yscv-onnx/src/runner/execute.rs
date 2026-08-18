//! Top-level inference drivers: the JIT (timed-loop) and sequential
//! single-pass executors that walk the optimized plan via execute_plan_branch.

use rustc_hash::{FxBuildHasher, FxHashMap};

use super::*;

/// JIT execution path: pre-compiled dispatch table, no per-node matching.
/// Skips NodeKind matching, layout checks are minimal, Conv params pre-resolved.
pub(crate) fn run_onnx_model_jit(
    model: &OnnxModel,
    mut env: TensorEnv<'_, '_>,
    specialization: Option<&ShapeSpecialization>,
) -> Result<FxHashMap<String, Tensor>, OnnxError> {
    let reshape_shapes = specialization.map(|plan| &plan.reshape_shapes);
    let branches = &model.runtime_index.node_branches;
    let use_counts_by_id = &model.runtime_index.use_counts_by_id;
    let output_id_mask = build_output_id_mask(model, &env, use_counts_by_id.len());

    if runner_profile_active() {
        runner_profile_note_inference();
    }
    let do_profile = std::env::var("YSCV_PROFILE").is_ok();
    let mut conv_ns: u64 = 0;
    let mut other_ns: u64 = 0;
    let mut conv_count: u32 = 0;
    let mut other_count: u32 = 0;

    // Tower-parallel: if the graph splits into two input-rooted subgraphs,
    // run them concurrently, then merge back for the shared tail. Each branch
    // gets its own env fork so concurrent inserts don't race.
    //
    // Keep explicit env control for A/B:
    //   - `YSCV_NO_TOWER_PARALLEL=1`   force OFF
    //   - `YSCV_FORCE_TOWER_PARALLEL=1` force ON
    //   - `YSCV_TOWER_MIN_THREADS=<N>`  default gate override
    let thread_count = rayon::current_num_threads();
    let no_tower_parallel = std::env::var_os("YSCV_NO_TOWER_PARALLEL").is_some();
    let force_tower_parallel = matches!(
        std::env::var_os("YSCV_FORCE_TOWER_PARALLEL").as_deref(),
        Some(v) if v == "1"
    );
    let default_tower_min_threads = 2usize;
    let tower_min_threads = std::env::var("YSCV_TOWER_MIN_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(default_tower_min_threads);
    let use_tower_parallel = !branches.is_empty()
        && (force_tower_parallel || (!no_tower_parallel && thread_count >= tower_min_threads));
    if use_tower_parallel {
        let branches_ref = branches.as_slice();
        let mut env0 = env.fork();
        let mut env1 = env.fork();
        let mut remaining0 = use_counts_by_id.clone();
        let mut remaining1 = use_counts_by_id.clone();

        // A″.5: route the tower-parallel fork through the installed
        // `ParallelScope` instead of calling `rayon::join` directly.
        // Under `YSCV_POOL=yscv` this goes through `YscvPool::join_dyn`
        // (no rayon touched). Under `YSCV_POOL=rayon` (default) the
        // rayon-backed ParallelScope ends up calling `rayon::join`
        // internally — same runtime behaviour.
        let mut r0: Result<(), OnnxError> = Ok(());
        let mut r1: Result<(), OnnxError> = Ok(());
        yscv_kernels::with_scope(|scope| {
            let mut a = || {
                let mut c_ns = 0u64;
                let mut o_ns = 0u64;
                let mut c_n = 0u32;
                let mut o_n = 0u32;
                r0 = execute_plan_branch(
                    model,
                    &mut env0,
                    &mut remaining0,
                    &output_id_mask,
                    reshape_shapes,
                    |nidx| branches_ref.get(nidx).copied() == Some(0),
                    &mut c_ns,
                    &mut o_ns,
                    &mut c_n,
                    &mut o_n,
                    do_profile,
                );
            };
            let mut b = || {
                let mut c_ns = 0u64;
                let mut o_ns = 0u64;
                let mut c_n = 0u32;
                let mut o_n = 0u32;
                r1 = execute_plan_branch(
                    model,
                    &mut env1,
                    &mut remaining1,
                    &output_id_mask,
                    reshape_shapes,
                    |nidx| branches_ref.get(nidx).copied() == Some(1),
                    &mut c_ns,
                    &mut o_ns,
                    &mut c_n,
                    &mut o_n,
                    do_profile,
                );
            };
            if let Some(scope) = scope {
                scope.join_dyn(&mut a, &mut b);
            } else {
                // Fallback: no scope installed (test harness / benches).
                rayon::join(a, b);
            }
        });
        r0?;
        r1?;

        env.merge_from(env0);
        env.merge_from(env1);

        // Merge-branch (id 2) runs on the reunited env.
        let mut remaining: Vec<usize> = use_counts_by_id.clone();
        execute_plan_branch(
            model,
            &mut env,
            &mut remaining,
            &output_id_mask,
            reshape_shapes,
            |nidx| {
                branches_ref.get(nidx).copied() != Some(0)
                    && branches_ref.get(nidx).copied() != Some(1)
            },
            &mut conv_ns,
            &mut other_ns,
            &mut conv_count,
            &mut other_count,
            do_profile,
        )?;
    } else {
        let mut remaining_uses: Vec<usize> = use_counts_by_id.clone();
        execute_plan_branch(
            model,
            &mut env,
            &mut remaining_uses,
            &output_id_mask,
            reshape_shapes,
            |_| true,
            &mut conv_ns,
            &mut other_ns,
            &mut conv_count,
            &mut other_count,
            do_profile,
        )?;
    }

    if do_profile {
        eprintln!(
            "\n[JIT profile] Conv: {:.1}ms ({} ops, {:.0}µs/op) | Other: {:.1}ms ({} ops, {:.0}µs/op) | Total: {:.1}ms",
            conv_ns as f64 / 1e6,
            conv_count,
            if conv_count > 0 {
                conv_ns as f64 / conv_count as f64 / 1e3
            } else {
                0.0
            },
            other_ns as f64 / 1e6,
            other_count,
            if other_count > 0 {
                other_ns as f64 / other_count as f64 / 1e3
            } else {
                0.0
            },
            (conv_ns + other_ns) as f64 / 1e6,
        );
    }

    // Ensure outputs in NCHW
    for name in &model.outputs {
        env.materialize_quant_i8_raw(name)?;
        ensure_nchw(&mut env, name)?;
    }
    let mut result = FxHashMap::with_capacity_and_hasher(model.outputs.len(), FxBuildHasher);
    for name in &model.outputs {
        if let Some(t) = env.remove(name) {
            result.insert(name.clone(), t);
        } else if let Some(t) = env.get(name) {
            result.insert(name.clone(), t.clone());
        }
    }
    Ok(result)
}
