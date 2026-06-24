pub(crate) use std::collections::{HashMap, HashSet};

pub(crate) use yscv_kernels::{
    BatchNorm2dParams, add as kernel_add, avg_pool2d_nhwc, batch_norm2d_nhwc, matmul_2d,
    max_pool2d_nhwc, mul as kernel_mul, relu, relu_inplace, sigmoid, softmax_last_dim,
    sub as kernel_sub,
};
pub(crate) use yscv_tensor::{DType, Tensor};

pub(crate) use crate::error::OnnxError;
pub(crate) use crate::loader::{NodeKind, OnnxAttribute, OnnxModel, OnnxNode};

mod compare;
mod conv;
mod elementwise;
mod gather_scatter;
#[cfg(feature = "gpu")]
pub(crate) mod gpu;
pub mod kv_cache;
mod linear;
#[cfg(all(target_os = "macos", feature = "metal-backend"))]
#[allow(unsafe_code)]
#[path = "metal/mod.rs"]
pub mod metal_runner;
mod misc;
mod normalization;
mod pooling;
mod reduce;
mod reshape;

use compare::*;
use conv::*;
use elementwise::*;
use gather_scatter::*;
use linear::*;
use misc::*;
use normalization::*;
use pooling::*;
use reduce::*;
use reshape::*;

mod conv_kernel;
mod dispatch;
mod execute;
mod layout;
mod plan_branch;
mod profile;
mod quant_stats;
mod tensor_env;
pub(crate) use conv_kernel::*;
pub(crate) use dispatch::*;
pub(crate) use execute::*;
pub(crate) use layout::*;
pub(crate) use plan_branch::*;
pub use profile::*;
pub use quant_stats::*;
pub(crate) use tensor_env::*;

// ---------------------------------------------------------------------------
// OnnxRunner — reusable inference session with configurable threading
// ---------------------------------------------------------------------------

/// Reusable inference session with configurable threading.
///
/// By default uses all CPU cores (like ONNX Runtime's `intra_op_num_threads`).
/// Use `with_threads(1)` for single-threaded execution.
///
/// ```rust,ignore
/// use yscv_onnx::*;
///
/// let model = load_onnx_model("model.onnx")?;
/// let runner = OnnxRunner::new(&model)?;           // all cores (default)
/// let runner_1t = OnnxRunner::with_threads(&model, 1)?; // single-thread
///
/// let input = Tensor::from_vec(vec![1, 3, 640, 640], data)?;
/// let outputs = runner.run(&[("images", &input)])?;
/// ```
pub struct OnnxRunner<'m> {
    model: &'m OnnxModel,
    pool: Option<rayon::ThreadPool>,
    /// Pool-agnostic scope for kernels migrated to `&dyn ParallelScope`.
    /// Built in lock-step with `pool`: when `pool == Some(p)`, this is a
    /// rayon-backed scope wrapping `p`; when `pool == None` it wraps
    /// rayon's global pool. `YSCV_POOL=yscv` swaps in a `YscvPool`
    /// instead — once enough sites route through this scope, yscv's
    /// cold-idle spin translates to the wall-clock win the microbench
    /// showed in isolation (1.66× faster install-per-task vs rayon).
    parallel_scope: std::sync::Arc<dyn yscv_threadpool::ParallelScope>,
    single_thread: bool,
}

/// Builds the `ParallelScope` for an `OnnxRunner` given a rayon pool (or
/// `None` meaning rayon's global) and the requested thread count. When
/// `YSCV_POOL=yscv` is set at process startup, returns a `YscvPool`-backed
/// scope instead. The returned scope is what newly-refactored kernels see;
/// legacy `pool: Option<&rayon::ThreadPool>` sites still route through the
/// rayon path until 2A.3 migrates them.
fn build_parallel_scope(
    rayon_pool: Option<&rayon::ThreadPool>,
    threads: usize,
) -> std::sync::Arc<dyn yscv_threadpool::ParallelScope> {
    static YSCV_POOL_SINGLETON: std::sync::OnceLock<
        Option<std::sync::Arc<yscv_threadpool::YscvPool>>,
    > = std::sync::OnceLock::new();

    let want_yscv = std::env::var("YSCV_POOL").as_deref() == Ok("yscv");

    if want_yscv {
        // One global YscvPool — construction is expensive (spawns threads +
        // pins affinity), reusing it across runners matches rayon's
        // "one global pool" mental model.
        let cached = YSCV_POOL_SINGLETON.get_or_init(|| {
            let n = threads.max(1);
            yscv_threadpool::YscvPool::new(n)
                .ok()
                .map(std::sync::Arc::new)
        });
        if let Some(p) = cached {
            return p.clone() as std::sync::Arc<dyn yscv_threadpool::ParallelScope>;
        }
    }

    // Rayon default path. The runner already installs pool_A via
    // `pool.install(|| run_onnx_model_inner(...))`, making pool_A the
    // ambient rayon pool for the entire inference. All `par_iter` dispatch
    // inside inference automatically uses pool_A's workers — no second pool
    // is needed. A static second rayon pool (RAYON_SCOPES approach) would
    // add N scope-pool threads that spin-idle alongside N pool-A threads =
    // 2N threads on N cores: pure oversubscription measured as 6T slowdown
    // vs 4T. Use AmbientRayonScope instead — zero extra threads, all
    // dispatch goes through the ambient (pool_A) rayon context.
    let _ = rayon_pool;
    let n = if threads == 0 {
        rayon::current_num_threads()
    } else {
        threads
    };
    std::sync::Arc::new(yscv_threadpool::AmbientRayonScope::new(n))
        as std::sync::Arc<dyn yscv_threadpool::ParallelScope>
}

impl<'m> OnnxRunner<'m> {
    /// Create a runner using all CPU cores (default, matches ORT behavior).
    pub fn new(model: &'m OnnxModel) -> Result<Self, OnnxError> {
        let scope = build_parallel_scope(None, rayon::current_num_threads());
        Ok(Self {
            model,
            pool: None,
            parallel_scope: scope,
            single_thread: false,
        })
    }

    /// Create a runner with explicit thread count.
    ///
    /// - `threads = 0`: use all **big / performance** cores (caps at
    ///   `cpu_topology::big_cores_count()`; on symmetric CPUs this is the full
    ///   core count — same as earlier behavior). On ARM big.LITTLE and Intel
    ///   hybrid this excludes LITTLE / E-cores, avoiding straggler-driven
    ///   join costs.
    /// - `threads = 1`: single-threaded (no rayon overhead, sequential execution)
    /// - `threads >= 2`: multi-threaded with explicit thread count. To avoid
    ///   latency regressions from SMT oversubscription on symmetric CPUs,
    ///   explicit values above physical core count are capped unless
    ///   `YSCV_ALLOW_SMT=1` is set.
    pub fn with_threads(model: &'m OnnxModel, threads: usize) -> Result<Self, OnnxError> {
        if threads == 1 {
            let scope = build_parallel_scope(None, 1);
            return Ok(Self {
                model,
                pool: None,
                parallel_scope: scope,
                single_thread: true,
            });
        }
        let (pool, effective_threads) = if threads == 0 {
            // Auto: use big-core count. Fallback to rayon's default if
            // topology detection returns the full core list anyway (symmetric).
            let big_count = crate::cpu_topology::big_cores_count();
            let all_count = rayon::current_num_threads();
            if big_count == 0 || big_count >= all_count {
                // Symmetric CPU or topology unknown — skip pool override,
                // let rayon use its global pool (same as pre-topology code).
                (None, all_count)
            } else {
                // Heterogeneous CPU detected — cap the pool to big cores.
                (
                    Some(
                        rayon::ThreadPoolBuilder::new()
                            .num_threads(big_count)
                            .build()
                            .map_err(|e| OnnxError::DecodeFailed {
                                message: format!("failed to create thread pool: {e}"),
                            })?,
                    ),
                    big_count,
                )
            }
        } else {
            let requested_threads = threads;
            let capped_threads = if std::env::var_os("YSCV_ALLOW_SMT").is_some() {
                requested_threads
            } else {
                requested_threads.min(crate::cpu_topology::physical_cores_count())
            };
            (
                Some(
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(capped_threads)
                        .build()
                        .map_err(|e| OnnxError::DecodeFailed {
                            message: format!("failed to create thread pool: {e}"),
                        })?,
                ),
                capped_threads,
            )
        };
        let scope = build_parallel_scope(pool.as_ref(), effective_threads);
        Ok(Self {
            model,
            pool,
            parallel_scope: scope,
            single_thread: false,
        })
    }

    /// Accessor for the pool-agnostic scope. Returned reference is valid
    /// for the runner's lifetime. Kernels that have been migrated to
    /// `&dyn ParallelScope` call this to route all parallel work through
    /// the configured backend.
    pub fn parallel_scope(&self) -> &dyn yscv_threadpool::ParallelScope {
        &*self.parallel_scope
    }

    /// Run inference with borrowed input pairs (zero-copy).
    pub fn run(&self, inputs: &[(&str, &Tensor)]) -> Result<HashMap<String, Tensor>, OnnxError> {
        let env = TensorEnv::from_model_with_inputs(self.model, Some(RuntimeInputs::Slice(inputs)));
        self.run_with_env(env)
    }

    /// Run inference with owned inputs HashMap.
    pub fn run_owned(
        &self,
        inputs: HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>, OnnxError> {
        let mut env = TensorEnv::from_model(self.model);
        for (name, tensor) in inputs {
            env.insert(name, tensor);
        }
        self.run_with_env(env)
    }

    fn run_with_env(&self, env: TensorEnv<'_, '_>) -> Result<HashMap<String, Tensor>, OnnxError> {
        // Install the active ParallelScope on the inference thread so
        // migrated kernel sites (see `yscv_kernels::with_scope`) pick it
        // up without a signature-threading refactor. Dropped
        // automatically at the end of this function.
        let _scope_guard = yscv_kernels::install_scope(self.parallel_scope.clone());

        // When YSCV_POOL=yscv we skip rayon's `install` wrapping: wrapping
        // inference inside a rayon pool while dispatch goes through the yscv
        // pool leaves both pools live, so workers contend for CPU and cache
        // lines bounce between them.
        let using_yscv_pool = std::env::var("YSCV_POOL").as_deref() == Ok("yscv");

        if self.single_thread {
            // Deterministic-latency 1T mode (`YSCV_ST_INLINE`, opt-in).
            //
            // The default 1T path below hands each inference to a 1-thread
            // `ST_POOL` worker via `pool.install` while the caller blocks on a
            // futex. The OS can migrate that worker across cores between calls,
            // varying cache warmth and adding p50−min jitter (ORT intra_op=1
            // runs inline on the caller and stays deterministic).
            //
            // Running inline on the caller removes the handoff. Full recipe for
            // ORT-level determinism:
            //   1. YSCV_ST_INLINE=1        — run inference on the caller thread
            //   2. RAYON_NUM_THREADS=1     — so kernel ambient-par sites (which
            //      read `rayon::current_num_threads()` directly, not the
            //      installed scope) see 1 thread and stay sequential
            //   3. pin the caller thread to a core (e.g. taskset / sched_setaffinity)
            // Inline WITHOUT pinning does not help (the lone caller still
            // migrates), so this stays opt-in rather than the default.
            if std::env::var_os("YSCV_ST_INLINE").is_some() {
                return run_onnx_model_inner(self.model, env);
            }
            thread_local! {
                static ST_POOL: rayon::ThreadPool = rayon::ThreadPoolBuilder::new()
                    .num_threads(1)
                    .build()
                    .expect("1-thread pool");
            }
            return ST_POOL.with(|pool| pool.install(|| run_onnx_model_inner(self.model, env)));
        }

        // Optionally enter a session-scoped parallel region: inside
        // `install_session`, `yscv-kernels::par_chunks_mut_dispatch` routes
        // through [`PersistentSection::parallel_for`] instead of rayon
        // fork-join. Opt-in via `YSCV_SESSION_POOL=1`, default OFF — the
        // section's `dispatch_busy` spin-lock (serialising tower-parallel
        // branches that post to one section) and per-idle peer-deque scans
        // cost more than rayon's work-stealing on this workload. Kept in-tree
        // for A/B on other microarchs.
        let session_enabled = std::env::var("YSCV_SESSION_POOL").as_deref() == Ok("1");

        let model = self.model;
        let scope = self.parallel_scope.clone();
        let run_inner = move |env: TensorEnv<'_, '_>| -> Result<_, OnnxError> {
            if using_yscv_pool {
                // Run directly on the caller thread; yscv workers pick
                // up parallel work via `scope_ctx` when kernel sites
                // call through.
                run_onnx_model_inner(model, env)
            } else if let Some(pool) = &self.pool {
                pool.install(|| run_onnx_model_inner(model, env))
            } else {
                run_onnx_model_inner(model, env)
            }
        };

        if session_enabled {
            yscv_kernels::with_installed_session(&*scope, || run_inner(env))
        } else {
            run_inner(env)
        }
    }

    /// Number of threads this runner uses.
    pub fn num_threads(&self) -> usize {
        if self.single_thread {
            1
        } else {
            self.pool
                .as_ref()
                .map(|p| p.current_num_threads())
                .unwrap_or_else(rayon::current_num_threads)
        }
    }
}

/// Runs inference on a loaded ONNX model with the given named inputs.
///
/// Returns a map of output-name -> tensor for the graph's declared outputs.
pub fn run_onnx_model(
    model: &OnnxModel,
    inputs: HashMap<String, Tensor>,
) -> Result<HashMap<String, Tensor>, OnnxError> {
    let mut env = TensorEnv::from_model(model);

    // Initializers (weights) are accessed via zero-copy fallback reference
    // in TensorEnv::get(). Only user inputs need to be inserted.
    for (name, tensor) in inputs {
        env.insert(name, tensor);
    }
    run_onnx_model_inner(model, env)
}

/// Runs inference with borrowed inputs (zero-copy at API boundary).
///
/// This avoids cloning user input tensors before execution. Inputs are treated
/// as read-only sources; if an op needs to mutate such tensor internally, the
/// runtime performs clone-on-write into environment slots.
pub fn run_onnx_model_borrowed(
    model: &OnnxModel,
    inputs: &HashMap<String, Tensor>,
) -> Result<HashMap<String, Tensor>, OnnxError> {
    let env = TensorEnv::from_model_with_inputs(model, Some(RuntimeInputs::Map(inputs)));
    run_onnx_model_inner(model, env)
}

/// Runs inference with borrowed input pairs (`name`, `tensor_ref`) without
/// requiring a `HashMap` allocation at the call site.
///
/// This is zero-copy at the API boundary. Runtime treats inputs as read-only and
/// clones on write only when mutation is required internally.
pub fn run_onnx_model_borrowed_slice(
    model: &OnnxModel,
    inputs: &[(&str, &Tensor)],
) -> Result<HashMap<String, Tensor>, OnnxError> {
    let env = TensorEnv::from_model_with_inputs(model, Some(RuntimeInputs::Slice(inputs)));
    run_onnx_model_inner(model, env)
}

#[inline]
fn tensor_use_count(
    env: &TensorEnv<'_, '_>,
    use_counts_by_id: &[usize],
    fallback_use_counts: &HashMap<String, usize>,
    name: &str,
) -> usize {
    env.resolve_id(name)
        .and_then(|id| use_counts_by_id.get(id).copied())
        .unwrap_or_else(|| fallback_use_counts.get(name).copied().unwrap_or(0))
}

#[inline]
fn find_relu_after_identity_chain(
    nodes: &[OnnxNode],
    node_kinds: &[NodeKind],
    start_idx: usize,
    expected_input: &str,
) -> Option<(usize, Vec<usize>)> {
    let mut idx = start_idx;
    let mut current_input = expected_input.to_string();
    let mut identity_idxs = Vec::new();
    while let Some(node) = nodes.get(idx) {
        let kind = node_kind(node_kinds, nodes, idx);
        if node.op_type == "Identity"
            && node.inputs.len() == 1
            && !node.outputs.is_empty()
            && node.inputs[0] == current_input
        {
            identity_idxs.push(idx);
            current_input = node.outputs[0].clone();
            idx += 1;
            continue;
        }
        if kind == NodeKind::Relu && node.inputs.len() == 1 && node.inputs[0] == current_input {
            return Some((idx, identity_idxs));
        }
        break;
    }
    None
}

#[inline]
fn mark_skip_indices(skip: &mut [bool], indices: &[usize]) {
    for &idx in indices {
        if idx < skip.len() {
            skip[idx] = true;
        }
    }
}

fn run_onnx_model_inner(
    model: &OnnxModel,
    env: TensorEnv<'_, '_>,
) -> Result<HashMap<String, Tensor>, OnnxError> {
    // Use JIT execution plan if available (pre-compiled dispatch, no per-node matching)
    if !model.runtime_index.execution_plan.is_empty() {
        return run_onnx_model_jit(model, env);
    }

    run_onnx_model_sequential(model, env)
}

#[inline]
fn build_output_id_mask(model: &OnnxModel, env: &TensorEnv<'_, '_>, min_len: usize) -> Vec<bool> {
    let mut mask = vec![false; min_len.max(1)];
    for name in &model.outputs {
        if let Some(id) = env.resolve_id(name) {
            if id >= mask.len() {
                mask.resize(id + 1, false);
            }
            mask[id] = true;
        }
    }
    mask
}

#[inline]
fn prepacked_for_conv_node(model: &OnnxModel, node_idx: usize) -> Option<&yscv_kernels::PackedB> {
    use std::sync::OnceLock;
    static DISABLED: OnceLock<bool> = OnceLock::new();
    if *DISABLED.get_or_init(|| std::env::var_os("YSCV_NO_PREPACKED_BY_ID").is_some()) {
        return None;
    }
    let weight_id = model
        .runtime_index
        .node_input_ids
        .get(node_idx)
        .and_then(|ids| ids.get(1))
        .and_then(|opt| *opt)?;
    model
        .runtime_index
        .prepacked_weights_by_id
        .get(weight_id)
        .and_then(|opt| opt.as_deref())
}
