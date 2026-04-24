pub(crate) use std::collections::{HashMap, HashSet};

pub(crate) use yscv_kernels::{
    BatchNorm2dParams, add as kernel_add, avg_pool2d_nhwc, batch_norm2d_nhwc, matmul_2d,
    matmul_2d_slices, max_pool2d_nhwc, mul as kernel_mul, relu, relu_inplace, sigmoid,
    softmax_last_dim, sub as kernel_sub,
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
#[cfg(feature = "metal-backend")]
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

    // Rayon default path. We need a `'static + ParallelScope` — a global
    // rayon pool wrapped in Arc, reconstructed lazily on first use.
    // `rayon_pool`'s lifetime is tied to the OnnxRunner, so we can't
    // reuse it directly through Arc. Instead we build a shared global
    // rayon pool with the requested thread count.
    static RAYON_SCOPES: std::sync::OnceLock<
        std::sync::Mutex<HashMap<usize, std::sync::Arc<rayon::ThreadPool>>>,
    > = std::sync::OnceLock::new();
    let _ = rayon_pool; // not used directly — see note above
    let map = RAYON_SCOPES.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let mut guard = map.lock().expect("RAYON_SCOPES mutex");
    let pool = guard
        .entry(threads)
        .or_insert_with(|| {
            std::sync::Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(if threads == 0 {
                        rayon::current_num_threads()
                    } else {
                        threads
                    })
                    .build()
                    .expect("rayon pool"),
            )
        })
        .clone();
    pool as std::sync::Arc<dyn yscv_threadpool::ParallelScope>
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

        // When YSCV_POOL=yscv we skip rayon's `install` wrapping — wrapping
        // inference inside a rayon pool while dispatch goes through yscv
        // means both pools are simultaneously live, workers on both ends
        // contend for CPU, and cache lines bounce between them. See the
        // backus-plan decision report (2026-04-19) for the full diagnosis.
        let using_yscv_pool = std::env::var("YSCV_POOL").as_deref() == Ok("yscv");

        if self.single_thread {
            thread_local! {
                static ST_POOL: rayon::ThreadPool = rayon::ThreadPoolBuilder::new()
                    .num_threads(1)
                    .build()
                    .expect("1-thread pool");
            }
            return ST_POOL.with(|pool| pool.install(|| run_onnx_model_inner(self.model, env)));
        }

        // Step 3: enter a session-scoped parallel region. Inside
        // `install_session`, `yscv-kernels::par_chunks_mut_dispatch`
        // routes through [`PersistentSection::parallel_for`] instead
        // of rayon fork-join.
        //
        // Session C measurement (2026-04-19 Zen 4 tracker 6T p50):
        //   rayon default:           ~3 430 µs (baseline)
        //   yscv-pool + session:     ~5 636 µs (+64% regression vs rayon)
        //   yscv-pool no session:    ~4 128 µs
        //
        // The `dispatch_busy` spin-lock (required to serialise tower-
        // parallel branches posting to the same section) kills the
        // very parallelism tower-parallel is meant to deliver, and the
        // `try_run_regular_task` peer-deque scans on every idle
        // iteration burn cycles that rayon's work-stealing avoids. Net:
        // PersistentSection architecture doesn't beat rayon on this
        // workload. Infrastructure stays in-tree (opt-in via
        // `YSCV_SESSION_POOL=1`) for future A/B on different
        // microarchs / workloads, but default is OFF.
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

// ---------------------------------------------------------------------------
// TensorEnv
// ---------------------------------------------------------------------------

/// A tensor environment backed by a `Vec<Option<Tensor>>` for O(1) lookups
/// by integer index. Tensor names are mapped to dense integer IDs during
/// construction, eliminating string hashing in the hot execution loop.
///
/// Model initializers (weights) are referenced without cloning. Only when
/// mutation is needed (get_mut/remove) is a clone-on-write performed.
pub(crate) struct TensorEnv<'m, 'i> {
    static_name_to_id: &'m HashMap<String, usize>,
    dynamic_name_to_id: HashMap<String, usize>,
    slots: Vec<Option<Tensor>>,
    /// Per-slot flag: true if the tensor is stored in NHWC layout.
    nhwc_flags: Vec<bool>,
    /// Slot IDs whose tensors have been pre-permuted from OIHW to KHWC.
    khwc_weights: &'m HashSet<usize>,
    /// Slot IDs whose depthwise weights were pre-permuted [O,1,KH,KW] → [KH,KW,C,dm].
    dw_khwc_weights: &'m HashSet<usize>,
    /// Slot IDs whose grouped-conv weights were pre-permuted [O,I/G,KH,KW] → [O,KH,KW,I/G].
    group_khwc_weights: &'m HashSet<usize>,
    /// Pre-packed blocked-GEMM B-matrices keyed by weight tensor name. Built
    /// once at model load (`build_runtime_index`). Hot path looks up by the
    /// Conv/MatMul's weight input name and hands the shared `Arc<PackedB>`
    /// straight into the GEMM layer, skipping fingerprint cache + repack.
    prepacked_weights: &'m HashMap<String, std::sync::Arc<yscv_kernels::PackedB>>,
    /// Counter for dynamically allocated temporary names that were not in
    /// the pre-built mapping (e.g., "__qa", "__qb_mat").
    next_dynamic: usize,
    /// Reference to model initializers for zero-copy weight access.
    initializers: &'m HashMap<String, Tensor>,
    /// Optional borrowed runtime inputs for zero-copy inference entry.
    runtime_inputs: Option<RuntimeInputs<'i>>,
}

#[derive(Clone, Copy)]
enum RuntimeInputs<'i> {
    Map(&'i HashMap<String, Tensor>),
    Slice(&'i [(&'i str, &'i Tensor)]),
}

impl<'m, 'i> TensorEnv<'m, 'i> {
    /// Build from the model, pre-allocating a slot for every known tensor name.
    /// Holds a reference to model initializers for zero-copy weight access.
    fn from_model(model: &'m OnnxModel) -> Self {
        Self::from_model_with_inputs(model, None)
    }

    /// Build from the model with optional borrowed runtime inputs.
    fn from_model_with_inputs(
        model: &'m OnnxModel,
        runtime_inputs: Option<RuntimeInputs<'i>>,
    ) -> Self {
        let num_slots = model.runtime_index.name_to_id.len();
        TensorEnv {
            next_dynamic: num_slots,
            static_name_to_id: &model.runtime_index.name_to_id,
            dynamic_name_to_id: HashMap::new(),
            slots: vec![None; num_slots],
            nhwc_flags: vec![false; num_slots],
            khwc_weights: &model.runtime_index.khwc_weight_ids,
            dw_khwc_weights: &model.runtime_index.dw_khwc_weight_ids,
            group_khwc_weights: &model.runtime_index.group_khwc_weight_ids,
            prepacked_weights: &model.runtime_index.prepacked_weights,
            initializers: &model.initializers,
            runtime_inputs,
        }
    }

    /// Returns the pre-packed B-matrix for the given weight tensor name, if
    /// one was built at model-load time. Callers can hand the resulting
    /// `&PackedB` to the blocked-GEMM fast path.
    #[inline]
    pub(crate) fn prepacked_b(&self, weight_name: &str) -> Option<&yscv_kernels::PackedB> {
        self.prepacked_weights
            .get(weight_name)
            .map(|arc| arc.as_ref())
    }

    #[inline]
    fn runtime_input(&self, name: &str) -> Option<&Tensor> {
        match self.runtime_inputs {
            Some(RuntimeInputs::Map(inputs)) => inputs.get(name),
            Some(RuntimeInputs::Slice(inputs)) => inputs
                .iter()
                .find_map(|(n, t)| if *n == name { Some(*t) } else { None }),
            None => None,
        }
    }

    #[inline]
    fn resolve_id(&self, name: &str) -> Option<usize> {
        self.dynamic_name_to_id
            .get(name)
            .copied()
            .or_else(|| self.static_name_to_id.get(name).copied())
    }

    /// Look up a tensor by name. Falls back to model initializers if the
    /// slot is empty (zero-copy access to weights).
    #[inline]
    pub(crate) fn get(&self, name: &str) -> Option<&Tensor> {
        let id = self.resolve_id(name)?;
        self.slots[id]
            .as_ref()
            .or_else(|| self.initializers.get(name))
            .or_else(|| self.runtime_input(name))
    }

    /// Direct slot removal by pre-resolved ID — O(1), no HashMap lookup.
    #[inline]
    pub(crate) fn remove_by_id(&mut self, id: usize) {
        if id < self.slots.len() {
            self.slots[id] = None;
            if id < self.nhwc_flags.len() {
                self.nhwc_flags[id] = false;
            }
        }
    }

    /// Insert a tensor by name. If the name is unknown, a new slot is
    /// allocated dynamically (this handles temporary names created by
    /// quantization ops, etc.).
    #[inline]
    pub(crate) fn insert(&mut self, name: String, tensor: Tensor) {
        if let Some(id) = self.resolve_id(&name) {
            self.slots[id] = Some(tensor);
            if id < self.nhwc_flags.len() {
                self.nhwc_flags[id] = false;
            }
        } else {
            let id = self.next_dynamic;
            self.next_dynamic += 1;
            self.dynamic_name_to_id.insert(name, id);
            self.slots.push(Some(tensor));
            self.nhwc_flags.push(false);
        }
    }

    /// Session 13 R3: direct slot insertion by pre-resolved ID, skipping
    /// the HashMap lookup inside `resolve_id`. Caller (runner hot path)
    /// passes the slot ID from `node_output_ids` table. Mirrors
    /// `remove_by_id` for output path.
    #[inline]
    pub(crate) fn insert_by_id(&mut self, id: usize, tensor: Tensor) {
        if id < self.slots.len() {
            self.slots[id] = Some(tensor);
            if id < self.nhwc_flags.len() {
                self.nhwc_flags[id] = false;
            }
        } else {
            // Grow the slots vec if needed. This should not happen in
            // practice — node_output_ids is built at load time from
            // name_to_id which sized slots at construction.
            while self.slots.len() <= id {
                self.slots.push(None);
                self.nhwc_flags.push(false);
            }
            self.slots[id] = Some(tensor);
        }
    }

    /// Session 13 R3: direct slot NHWC mark by ID.
    #[inline]
    pub(crate) fn mark_nhwc_by_id(&mut self, id: usize) {
        if id < self.nhwc_flags.len() {
            self.nhwc_flags[id] = true;
        }
    }

    /// Get a mutable reference to a tensor by name.
    /// Clone-on-write: if the tensor is only in initializers, clone it into
    /// the slot first.
    #[inline]
    pub(crate) fn get_mut(&mut self, name: &str) -> Option<&mut Tensor> {
        let id = self.resolve_id(name)?;
        if self.slots[id].is_none()
            && let Some(t) = self.initializers.get(name)
        {
            self.slots[id] = Some(t.clone());
        } else if self.slots[id].is_none()
            && let Some(t) = self.runtime_input(name)
        {
            self.slots[id] = Some(t.clone());
        }
        self.slots[id].as_mut()
    }

    /// Fork env for tower-parallel execution. Shares all read-only metadata
    /// (initializers, name tables, pre-permute flags) and clones the slot
    /// state so each branch has its own mutable view. Only live tensors are
    /// copied — Tensor is `Arc<AlignedVec>`, so the clone is a refcount bump,
    /// not a data copy.
    pub(crate) fn fork(&self) -> Self {
        TensorEnv {
            static_name_to_id: self.static_name_to_id,
            dynamic_name_to_id: self.dynamic_name_to_id.clone(),
            slots: self.slots.clone(),
            nhwc_flags: self.nhwc_flags.clone(),
            khwc_weights: self.khwc_weights,
            dw_khwc_weights: self.dw_khwc_weights,
            group_khwc_weights: self.group_khwc_weights,
            prepacked_weights: self.prepacked_weights,
            next_dynamic: self.next_dynamic,
            initializers: self.initializers,
            runtime_inputs: self.runtime_inputs,
        }
    }

    /// Transfer new tensors from `other` back into `self`. A slot that is
    /// Some in `other` but None (or stale) in `self` gets moved over; other
    /// slots in `self` stay intact. Used after tower-parallel execution to
    /// reunite branches into one env before running the merge tail.
    pub(crate) fn merge_from(&mut self, mut other: TensorEnv<'m, 'i>) {
        let n = self.slots.len().min(other.slots.len());
        for id in 0..n {
            if let Some(t) = other.slots[id].take() {
                self.slots[id] = Some(t);
                if id < self.nhwc_flags.len() && id < other.nhwc_flags.len() {
                    self.nhwc_flags[id] = other.nhwc_flags[id];
                }
            }
        }
        // Any dynamic names added in the forked env need to be propagated so
        // the merge-branch nodes can resolve them.
        for (name, id) in other.dynamic_name_to_id.drain() {
            self.dynamic_name_to_id.insert(name, id);
        }
        self.next_dynamic = self.next_dynamic.max(other.next_dynamic);
    }

    /// Remove a tensor by name (sets the slot to `None`).
    /// If the tensor is only in initializers, clone and return it.
    #[inline]
    pub(crate) fn remove(&mut self, name: &str) -> Option<Tensor> {
        let id = self.resolve_id(name)?;
        if id < self.nhwc_flags.len() {
            self.nhwc_flags[id] = false;
        }
        self.slots[id]
            .take()
            .or_else(|| self.initializers.get(name).cloned())
            .or_else(|| self.runtime_input(name).cloned())
    }

    /// Returns true if the tensor at `name` is stored in NHWC layout.
    #[inline]
    pub(crate) fn is_nhwc(&self, name: &str) -> bool {
        self.resolve_id(name)
            .map(|id| self.nhwc_flags.get(id).copied().unwrap_or(false))
            .unwrap_or(false)
    }

    /// Mark the tensor at `name` as being in NHWC layout.
    #[inline]
    pub(crate) fn mark_nhwc(&mut self, name: &str) {
        if let Some(id) = self.resolve_id(name)
            && id < self.nhwc_flags.len()
        {
            self.nhwc_flags[id] = true;
        }
    }

    /// Returns true if the tensor has been pre-permuted to KHWC format.
    #[inline]
    pub(crate) fn is_khwc_weight(&self, name: &str) -> bool {
        self.resolve_id(name)
            .is_some_and(|id| self.khwc_weights.contains(&id))
    }

    /// Returns true if the depthwise conv weight has been pre-permuted to
    /// `[KH, KW, C, depth_multiplier]` format at load time.
    #[inline]
    pub(crate) fn is_dw_khwc_weight(&self, name: &str) -> bool {
        self.resolve_id(name)
            .is_some_and(|id| self.dw_khwc_weights.contains(&id))
    }

    /// Returns true if grouped conv weight has been pre-permuted to
    /// `[O, KH, KW, I/G]` format at load time.
    #[inline]
    pub(crate) fn is_group_khwc_weight(&self, name: &str) -> bool {
        self.resolve_id(name)
            .is_some_and(|id| self.group_khwc_weights.contains(&id))
    }

    /// Create a zero-copy alias: remap `alias_name` to the same slot as
    /// `target_name`. No tensor data is cloned — both names point to the
    /// identical storage. Safe because ONNX outputs are write-once.
    #[inline]
    pub(crate) fn alias(&mut self, alias_name: &str, target_name: &str) {
        let target_id = match self.resolve_id(target_name) {
            Some(id) => id,
            None => return,
        };
        // If the target lives only in `initializers`, materialize it into the
        // slot so the alias name can resolve via `get()` — which otherwise
        // would fall back to `initializers.get(alias_name)` and miss.
        if self.slots[target_id].is_none()
            && let Some(t) = self.initializers.get(target_name)
        {
            self.slots[target_id] = Some(t.clone());
        } else if self.slots[target_id].is_none()
            && let Some(t) = self.runtime_input(target_name)
        {
            self.slots[target_id] = Some(t.clone());
        }
        // Point alias_name to the same slot ID as target_name.
        self.dynamic_name_to_id
            .insert(alias_name.to_string(), target_id);
    }
}

/// Convert NHWC tensor to NCHW in-place in the environment.
pub(crate) fn ensure_nchw(env: &mut TensorEnv, name: &str) -> Result<(), OnnxError> {
    if env.is_nhwc(name)
        && let Some(t) = env.remove(name)
    {
        let nchw = t
            .permute(&[0, 3, 1, 2])
            .map_err(|e| OnnxError::DecodeFailed {
                message: e.to_string(),
            })?;
        env.insert(name.to_string(), nchw);
    }
    Ok(())
}

/// Map an axis from NCHW to NHWC for 4D tensors.
pub(crate) fn nchw_axis_to_nhwc(axis: usize) -> usize {
    const MAP: [usize; 4] = [0, 3, 1, 2];
    if axis < 4 { MAP[axis] } else { axis }
}

#[inline]
fn node_kind(node_kinds: &[NodeKind], nodes: &[OnnxNode], idx: usize) -> NodeKind {
    if let Some(kind) = node_kinds.get(idx).copied() {
        kind
    } else {
        NodeKind::from_op_type(&nodes[idx].op_type)
    }
}

/// CPU-side fallback entry point for the GPU runner.
///
/// Why: when the GPU backend hits an unsupported op or a degenerate tensor
/// (size < 4 for vec4 shaders, scalar shape metadata), it rematerialises
/// inputs on the CPU and delegates execution back here. This wrapper hides
/// `NodeKind` classification from the GPU module so it doesn't need to know
/// the dispatch taxonomy.
#[cfg(feature = "gpu")]
#[inline]
pub(super) fn execute_node_cpu_fallback(
    node: &OnnxNode,
    env: &mut TensorEnv,
) -> Result<(), OnnxError> {
    let kind = NodeKind::from_op_type(&node.op_type);
    execute_node_kind(node, env, kind)
}

#[inline]
fn is_nhwc_producer_with_kind(kind: NodeKind, op_type: &str) -> bool {
    matches!(
        kind,
        NodeKind::Conv | NodeKind::ConvRelu | NodeKind::ConvSilu | NodeKind::BatchNormalization
    ) || matches!(
        op_type,
        "MaxPool"
            | "AveragePool"
            | "GlobalAveragePool"
            | "BatchNormalization_Relu"
            | "Resize"
            | "Upsample"
            | "DeformConv"
    )
}

#[inline]
fn is_passthrough_op_with_kind(kind: NodeKind, op_type: &str) -> bool {
    if matches!(
        kind,
        NodeKind::Relu | NodeKind::Sigmoid | NodeKind::Add | NodeKind::Mul
    ) {
        return true;
    }
    matches!(
        op_type,
        "Tanh"
            | "Exp"
            | "Log"
            | "Neg"
            | "Abs"
            | "Sqrt"
            | "Pow"
            | "Clip"
            | "LeakyRelu"
            | "Elu"
            | "Selu"
            | "Gelu"
            | "Erf"
            | "HardSigmoid"
            | "Softplus"
            | "Softsign"
            | "HardSwish"
            | "Mish"
            | "ThresholdedRelu"
            | "Celu"
            | "Sub"
            | "Div"
            | "Min"
            | "Max"
            | "Dropout"
            | "Identity"
    )
}

/// Execute a node with automatic NHWC layout management.
#[inline]
fn execute_node_with_layout_kind(
    node: &OnnxNode,
    env: &mut TensorEnv,
    kind: NodeKind,
) -> Result<(), OnnxError> {
    let op = node.op_type.as_str();

    // NHWC producers and adjusted ops handle layout internally
    if is_nhwc_producer_with_kind(kind, op)
        || op == "Concat"
        || op == "Split"
        || op == "Transpose"
        || op == "Shape"
    {
        return execute_node_kind(node, env, kind);
    }

    let propagate_nhwc = if is_passthrough_op_with_kind(kind, op) {
        // Check for mixed 4D layouts
        let has_nhwc = node.inputs.iter().any(|n| !n.is_empty() && env.is_nhwc(n));
        let has_nchw_4d = node
            .inputs
            .iter()
            .any(|n| !n.is_empty() && !env.is_nhwc(n) && env.get(n).is_some_and(|t| t.rank() == 4));
        if has_nhwc && has_nchw_4d {
            // Mixed 4D layouts: convert all to NCHW
            for name in &node.inputs {
                if !name.is_empty() {
                    ensure_nchw(env, name)?;
                }
            }
            false
        } else {
            has_nhwc
        }
    } else {
        // NCHW-required op: ensure all inputs are NCHW
        for name in &node.inputs {
            if !name.is_empty() {
                ensure_nchw(env, name)?;
            }
        }
        false
    };

    execute_node_kind(node, env, kind)?;

    if propagate_nhwc {
        for out in &node.outputs {
            if !out.is_empty() {
                env.mark_nhwc(out);
            }
        }
    }

    Ok(())
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

/// JIT execution path: pre-compiled dispatch table, no per-node matching.
/// Skips NodeKind matching, layout checks are minimal, Conv params pre-resolved.
/// Run a subset of the JIT execution plan, filtered by a predicate on the
/// node's branch assignment. Shared body for both the single-branch path and
/// the tower-parallel wrapper.
/// Aggregated per-node timing collector for the fused runner path.
///
/// Activated when `YSCV_RUNNER_PROFILE=path` is set on process startup.
/// Accumulates timing over every inference in `run_onnx_model_jit` and
/// dumps JSON via [`dump_runner_profile`]. Unlike `profile_onnx_model_cpu`
/// (which runs a **single unfused** inference), this captures the actual
/// fused-path timings with Conv+Relu, Conv+Add, DW+PW fusions applied —
/// the numbers that `scripts/gap_diff.py` should compare against ORT's
/// Chrome-trace output.
///
/// Storage is per-node-name so repeated ops across inference iterations
/// aggregate cleanly. Shapes are captured on first sighting. Each thread
/// owns its own store via `thread_local!`; the global registry holds an
/// `Arc<Mutex<_>>` per thread so `dump_runner_profile` can walk them all
/// and aggregate without contending on the hot path. Previously this was
/// a single `Mutex<RunnerProfileStore>` — under tower-parallel two branch
/// threads recorded concurrently, and even uncontended lock+unlock costs
/// showed up in per-op timings. Thread-local stores pay zero synchronisation
/// on record (each thread is the sole writer of its own `RefCell`).
#[derive(Default)]
struct RunnerProfileStore {
    per_node: HashMap<String, RunnerNodeStat>,
}

#[derive(Clone, Default)]
struct RunnerNodeStat {
    op: String,
    total_ns: u64,
    count: u64,
    in_shape: Vec<usize>,
    out_shape: Vec<usize>,
}

/// Global registry of per-thread stores. Each thread registers its store
/// on first access; destruction is deferred via `Arc` so data survives
/// thread exit until `dump_runner_profile` runs.
fn profile_registry()
-> &'static std::sync::Mutex<Vec<std::sync::Arc<std::sync::Mutex<RunnerProfileStore>>>> {
    use std::sync::OnceLock;
    static CELL: OnceLock<
        std::sync::Mutex<Vec<std::sync::Arc<std::sync::Mutex<RunnerProfileStore>>>>,
    > = OnceLock::new();
    CELL.get_or_init(|| std::sync::Mutex::new(Vec::new()))
}

thread_local! {
    static LOCAL_PROFILE: std::sync::Arc<std::sync::Mutex<RunnerProfileStore>> = {
        let store = std::sync::Arc::new(std::sync::Mutex::new(RunnerProfileStore::default()));
        if let Ok(mut reg) = profile_registry().lock() {
            reg.push(store.clone());
        }
        store
    };
}

/// True when `YSCV_RUNNER_PROFILE` env is non-empty on startup. Cached so
/// we only pay the env read once.
fn runner_profile_active() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("YSCV_RUNNER_PROFILE")
            .ok()
            .filter(|s| !s.is_empty())
            .is_some()
    })
}

fn runner_profile_record(
    name: &str,
    op: &str,
    elapsed_ns: u64,
    in_shape: Vec<usize>,
    out_shape: Vec<usize>,
) {
    LOCAL_PROFILE.with(|store| {
        // Same-thread lock — uncontended, near-free; kept as `Mutex` rather
        // than `RefCell` so the `Arc` shared with the registry can be
        // locked from the dump thread safely.
        if let Ok(mut s) = store.lock() {
            let entry = s
                .per_node
                .entry(name.to_string())
                .or_insert_with(|| RunnerNodeStat {
                    op: op.to_string(),
                    in_shape: in_shape.clone(),
                    out_shape: out_shape.clone(),
                    ..Default::default()
                });
            entry.total_ns += elapsed_ns;
            entry.count += 1;
            if entry.in_shape.is_empty() {
                entry.in_shape = in_shape;
            }
            if entry.out_shape.is_empty() {
                entry.out_shape = out_shape;
            }
        }
    });
}

/// Dumps the accumulated runner-path profile to `path` as JSON in the
/// same schema as `profile_onnx_model_cpu` / `bench_ort.py --emit-profile`.
/// Call after a benchmark loop to get aggregated per-node timings
/// normalised by iteration count. No-op when the profiler was never
/// activated (env unset) or the store is empty.
pub fn dump_runner_profile(path: &str) -> Result<(), OnnxError> {
    use std::fmt::Write as _;
    // Aggregate per-thread stores. Each thread recorded into its own
    // registry entry; we sum counts and total_ns per node name and keep
    // the first-seen shapes.
    let merged = {
        let registry = profile_registry()
            .lock()
            .map_err(|e| OnnxError::DecodeFailed {
                message: format!("runner profile registry poisoned: {e}"),
            })?;
        let mut merged: HashMap<String, RunnerNodeStat> = HashMap::new();
        for thread_store in registry.iter() {
            let store = thread_store.lock().map_err(|e| OnnxError::DecodeFailed {
                message: format!("runner profile per-thread mutex poisoned: {e}"),
            })?;
            for (name, stat) in &store.per_node {
                merged
                    .entry(name.clone())
                    .and_modify(|e| {
                        e.total_ns += stat.total_ns;
                        e.count += stat.count;
                        if e.in_shape.is_empty() {
                            e.in_shape = stat.in_shape.clone();
                        }
                        if e.out_shape.is_empty() {
                            e.out_shape = stat.out_shape.clone();
                        }
                    })
                    .or_insert_with(|| stat.clone());
            }
        }
        merged
    };
    if merged.is_empty() {
        return Ok(());
    }
    // Normalise total_ns by `count` so each node's `ms` is the average
    // per-instance time — matches what ORT's `end_profiling()` emits for
    // a single inference.
    let mut entries: Vec<(&String, &RunnerNodeStat)> = merged.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    let iters: u64 = entries.iter().map(|(_, s)| s.count).max().unwrap_or(1);

    let mut out = String::with_capacity(256 * entries.len());
    out.push_str("{\"engine\":\"yscv\",\"total_ms\":");
    let total_ns: u64 = entries.iter().map(|(_, s)| s.total_ns).sum();
    let total_ms = (total_ns as f64) / (iters as f64) / 1_000_000.0;
    let _ = write!(out, "{:.6}", total_ms);
    out.push_str(",\"iterations\":");
    let _ = write!(out, "{}", iters);
    out.push_str(",\"nodes\":[");
    for (i, (name, stat)) in entries.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str("{\"name\":\"");
        json_escape_into_local(&mut out, name);
        out.push_str("\",\"op\":\"");
        json_escape_into_local(&mut out, &stat.op);
        out.push_str("\",\"ms\":");
        // ms per inference (total_ns / iterations / 1e6).
        let ms = (stat.total_ns as f64) / (iters as f64) / 1_000_000.0;
        let _ = write!(out, "{:.6}", ms);
        out.push_str(",\"count\":");
        let _ = write!(out, "{}", stat.count);
        out.push_str(",\"in_shape\":");
        shape_to_json_local(&mut out, &stat.in_shape);
        out.push_str(",\"out_shape\":");
        shape_to_json_local(&mut out, &stat.out_shape);
        out.push('}');
    }
    out.push_str("]}\n");
    std::fs::write(path, out).map_err(|e| OnnxError::DecodeFailed {
        message: format!("write runner profile to {path}: {e}"),
    })?;
    Ok(())
}

fn json_escape_into_local(out: &mut String, s: &str) {
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                use std::fmt::Write as _;
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
}

fn shape_to_json_local(out: &mut String, shape: &[usize]) {
    use std::fmt::Write as _;
    out.push('[');
    for (i, d) in shape.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let _ = write!(out, "{d}");
    }
    out.push(']');
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

fn execute_plan_branch(
    model: &OnnxModel,
    env: &mut TensorEnv<'_, '_>,
    remaining_uses: &mut [usize],
    output_id_mask: &[bool],
    mut accept: impl FnMut(usize) -> bool,
    conv_ns: &mut u64,
    other_ns: &mut u64,
    conv_count: &mut u32,
    other_count: &mut u32,
    do_profile: bool,
) -> Result<(), OnnxError> {
    use crate::loader::NodeAction;

    let nodes = &model.nodes;
    let plan = &model.runtime_index.execution_plan;
    let runner_profile_enabled = runner_profile_active();
    let timing_enabled = do_profile || runner_profile_enabled;

    for action in plan {
        // Lookup the representative node index to check branch filter.
        let rep_idx = match action {
            NodeAction::Skip => continue,
            NodeAction::Conv { node_idx, .. } | NodeAction::Generic { node_idx, .. } => *node_idx,
            NodeAction::FusedDwPw { dw_idx, .. } => *dw_idx,
            NodeAction::FusedPwDw { pw_idx, .. } => *pw_idx,
            NodeAction::FusedTransposeMatMul { matmul_idx, .. } => *matmul_idx,
            NodeAction::ConvAdd { conv_idx, .. } => *conv_idx,
            NodeAction::NchwcChain { members, .. } => match members.first() {
                Some(crate::loader::NchwcChainMember::Conv { node_idx, .. }) => *node_idx,
                None => continue,
            },
        };
        if !accept(rep_idx) {
            continue;
        }

        let t0 = timing_enabled.then(std::time::Instant::now);

        match action {
            NodeAction::Skip => continue,

            NodeAction::NchwcChain { members, .. } => {
                // Step A.1: Fallback dispatch — runs each inner Conv via
                // the regular NHWC path. Functionally equivalent to
                // running the original actions; enables the chain
                // infrastructure to be exercised end-to-end (detection,
                // tracker bitwise) without yet committing to NCHWc
                // layout transforms. Step A.2 replaces this with
                // `nhwc_to_nchwc` at entry, `*_nchwc` kernels per
                // member, `nchwc_to_nhwc` at exit.
                for member in members {
                    match member {
                        crate::loader::NchwcChainMember::Conv {
                            node_idx,
                            activation,
                        } => {
                            let node = &nodes[*node_idx];
                            let act = match activation {
                                1 => yscv_kernels::Activation::Relu,
                                2 => yscv_kernels::Activation::Silu,
                                _ => yscv_kernels::Activation::None,
                            };
                            let cp = model
                                .runtime_index
                                .conv_params
                                .get(*node_idx)
                                .and_then(|o| o.as_ref());
                            let prepacked = prepacked_for_conv_node(model, *node_idx);
                            exec_conv_with_params(node, env, act, cp, prepacked)?;
                            // Session 13 R3: cached output slot ID avoids a
                            // HashMap lookup inside mark_nhwc.
                            if let Some(oid) = model
                                .runtime_index
                                .node_output_ids
                                .get(*node_idx)
                                .and_then(|v| v.first())
                                .and_then(|o| *o)
                            {
                                env.mark_nhwc_by_id(oid);
                            } else {
                                env.mark_nhwc(&node.outputs[0]);
                            }
                        }
                    }
                }
            }

            NodeAction::Conv {
                node_idx,
                activation,
            } => {
                let node = &nodes[*node_idx];
                let act = match activation {
                    1 => yscv_kernels::Activation::Relu,
                    2 => yscv_kernels::Activation::Silu,
                    _ => yscv_kernels::Activation::None,
                };
                let cp = model
                    .runtime_index
                    .conv_params
                    .get(*node_idx)
                    .and_then(|o| o.as_ref());
                let prepacked = prepacked_for_conv_node(model, *node_idx);
                exec_conv_with_params(node, env, act, cp, prepacked)?;
                if let Some(oid) = model
                    .runtime_index
                    .node_output_ids
                    .get(*node_idx)
                    .and_then(|v| v.first())
                    .and_then(|o| *o)
                {
                    env.mark_nhwc_by_id(oid);
                } else {
                    env.mark_nhwc(&node.outputs[0]);
                }
            }

            NodeAction::FusedDwPw {
                dw_idx,
                pw_idx,
                dw_activation,
                pw_activation,
            } => {
                let dw_node = &nodes[*dw_idx];
                let pw_node = &nodes[*pw_idx];
                let dw_act = match dw_activation {
                    1 => yscv_kernels::Activation::Relu,
                    2 => yscv_kernels::Activation::Silu,
                    _ => yscv_kernels::Activation::None,
                };
                let pw_act = match pw_activation {
                    1 => yscv_kernels::Activation::Relu,
                    2 => yscv_kernels::Activation::Silu,
                    _ => yscv_kernels::Activation::None,
                };
                let dw_cp = model
                    .runtime_index
                    .conv_params
                    .get(*dw_idx)
                    .and_then(|o| o.as_ref());
                let pw_cp = model
                    .runtime_index
                    .conv_params
                    .get(*pw_idx)
                    .and_then(|o| o.as_ref());
                let dw_input_ids_slice: &[Option<usize>] = model
                    .runtime_index
                    .node_input_ids
                    .get(*dw_idx)
                    .map(|v| v.as_slice())
                    .unwrap_or(&[]);
                exec_fused_dw_pw(
                    dw_node,
                    pw_node,
                    env,
                    dw_act,
                    pw_act,
                    dw_cp,
                    pw_cp,
                    dw_input_ids_slice,
                    remaining_uses,
                    output_id_mask,
                )?;
            }

            NodeAction::FusedPwDw {
                pw_idx,
                dw_idx,
                pw_activation,
                dw_activation,
            } => {
                let pw_node = &nodes[*pw_idx];
                let dw_node = &nodes[*dw_idx];
                let pw_act = match pw_activation {
                    1 => yscv_kernels::Activation::Relu,
                    2 => yscv_kernels::Activation::Silu,
                    _ => yscv_kernels::Activation::None,
                };
                let dw_act = match dw_activation {
                    1 => yscv_kernels::Activation::Relu,
                    2 => yscv_kernels::Activation::Silu,
                    _ => yscv_kernels::Activation::None,
                };
                let pw_cp = model
                    .runtime_index
                    .conv_params
                    .get(*pw_idx)
                    .and_then(|o| o.as_ref());
                let dw_cp = model
                    .runtime_index
                    .conv_params
                    .get(*dw_idx)
                    .and_then(|o| o.as_ref());
                let pw_input_ids_slice: &[Option<usize>] = model
                    .runtime_index
                    .node_input_ids
                    .get(*pw_idx)
                    .map(|v| v.as_slice())
                    .unwrap_or(&[]);
                exec_fused_pw_dw(
                    pw_node,
                    dw_node,
                    env,
                    pw_act,
                    dw_act,
                    pw_cp,
                    dw_cp,
                    pw_input_ids_slice,
                    remaining_uses,
                    output_id_mask,
                )?;
            }

            NodeAction::ConvAdd {
                conv_idx,
                add_idx,
                skip_input_idx,
                post_activation,
                relu_idx,
            } => {
                let conv_node = &nodes[*conv_idx];
                let add_node = &nodes[*add_idx];
                let cp = model
                    .runtime_index
                    .conv_params
                    .get(*conv_idx)
                    .and_then(|o| o.as_ref());
                let conv_out = &conv_node.outputs[0];
                let skip_name = &add_node.inputs[*skip_input_idx as usize];

                // Fast path — pointwise Conv + residual Add + optional Relu
                // fused in one GEMM pass. Writes `out = conv_acc + bias +
                // residual + activation` inline, avoiding the 2-pass
                // `add_relu_inplace` which doubles output-side memory
                // traffic (tracker Conv_Add is ~1.2ms @ 6T = 18% of total,
                // mostly on high-k shapes). Step S.2: removed the former
                // `k_small` gate — Phase 1.2 added residual support to
                // blocked GEMM 4×24/4×16 microkernels, so ALL pointwise
                // Conv+Add now fuses (matmul dispatches row_gemm for k<32,
                // blocked for k≥32, both residual-aware).
                //
                // `blocked_residual_has_unsupported_tail(n)` at matmul
                // dispatch auto-routes to row_gemm for shapes whose jr
                // tail would hit 4×8 / scalar (no residual there yet).
                let fused_pointwise = cp
                    .map(|p| {
                        p.is_pointwise
                            && !p.has_padding
                            && p.stride_h == 1
                            && p.stride_w == 1
                            && p.group == 1
                    })
                    .unwrap_or(false);
                let activation_for_fused = if *post_activation == 1 {
                    yscv_kernels::Activation::Relu
                } else {
                    yscv_kernels::Activation::None
                };
                let fused_result: Option<Tensor> = if fused_pointwise {
                    // Scoped block so all `env.get` immutable borrows drop
                    // before the mutable `env.insert` below.
                    let input_ok = env.is_nhwc(&conv_node.inputs[0]);
                    if input_ok {
                        let input_tensor = env.get(&conv_node.inputs[0]);
                        let w_tensor = env.get(&conv_node.inputs[1]);
                        let skip_tensor = env.get(skip_name);
                        let bias_tensor = conv_node
                            .inputs
                            .get(2)
                            .and_then(|n| if n.is_empty() { None } else { env.get(n) });
                        match (input_tensor, w_tensor, skip_tensor) {
                            (Some(i), Some(w), Some(s))
                                if i.rank() == 4
                                    && w.rank() == 4
                                    && w.shape()[0] == 1
                                    && w.shape()[1] == 1 =>
                            {
                                let prepacked = prepacked_for_conv_node(model, *conv_idx);
                                yscv_kernels::conv2d_nhwc_pointwise_with_residual_relu(
                                    i,
                                    w,
                                    bias_tensor,
                                    s,
                                    activation_for_fused,
                                    None,
                                    prepacked,
                                )
                                .ok()
                            }
                            _ => None,
                        }
                    } else {
                        None
                    }
                } else {
                    None
                };

                if let Some(fused_out) = fused_result {
                    let add_out = &add_node.outputs[0];
                    // Session 13 R3: cached add_out slot ID avoids a HashMap
                    // lookup inside insert + mark_nhwc. On tracker this fires
                    // 24× per inference (one per residual block).
                    let add_out_id = model
                        .runtime_index
                        .node_output_ids
                        .get(*add_idx)
                        .and_then(|v| v.first())
                        .and_then(|o| *o);
                    if let Some(oid) = add_out_id {
                        env.insert_by_id(oid, fused_out);
                        env.mark_nhwc_by_id(oid);
                    } else {
                        env.insert(add_out.clone(), fused_out);
                        env.mark_nhwc(add_out);
                    }
                    if *post_activation == 1 {
                        let relu_out = &nodes[*relu_idx as usize].outputs[0];
                        env.alias(relu_out, add_out);
                    }
                    if do_profile {
                        let elapsed = t0
                            .as_ref()
                            .map(|start| start.elapsed().as_nanos() as u64)
                            .unwrap_or(0);
                        *conv_ns += elapsed;
                        *conv_count += 1;
                    }
                    if runner_profile_enabled {
                        let elapsed = t0
                            .as_ref()
                            .map(|start| start.elapsed().as_nanos() as u64)
                            .unwrap_or(0);
                        let in_sh = env
                            .get(&conv_node.inputs[0])
                            .map(|t| t.shape().to_vec())
                            .unwrap_or_default();
                        let out_sh = env
                            .get(&add_node.outputs[0])
                            .map(|t| t.shape().to_vec())
                            .unwrap_or_default();
                        let op_label = if *post_activation == 1 {
                            "Conv_Add_Relu_fused"
                        } else {
                            "Conv_Add_fused"
                        };
                        runner_profile_record(&conv_node.name, op_label, elapsed, in_sh, out_sh);
                    }
                    // Early-dealloc input refs the same way the generic
                    // branch does at function scope.
                    let covered = &[*conv_idx, *add_idx][..];
                    let input_ids = &model.runtime_index.node_input_ids;
                    for &nidx in covered {
                        let n = &nodes[nidx];
                        let pre_ids = if nidx < input_ids.len() {
                            &input_ids[nidx]
                        } else {
                            &[][..]
                        };
                        for (inp_idx, inp) in n.inputs.iter().enumerate() {
                            if inp.is_empty() {
                                continue;
                            }
                            let id = pre_ids
                                .get(inp_idx)
                                .and_then(|opt| *opt)
                                .or_else(|| env.resolve_id(inp));
                            if let Some(id) = id
                                && id < remaining_uses.len()
                            {
                                remaining_uses[id] = remaining_uses[id].saturating_sub(1);
                                if remaining_uses[id] == 0 && !output_id_mask[id] {
                                    env.remove_by_id(id);
                                }
                            }
                        }
                    }
                    continue;
                }

                let prepacked = prepacked_for_conv_node(model, *conv_idx);
                exec_conv_with_params(
                    conv_node,
                    env,
                    yscv_kernels::Activation::None,
                    cp,
                    prepacked,
                )?;
                // Session 13 R3: cached slot IDs for the fallback path too.
                let conv_out_id = model
                    .runtime_index
                    .node_output_ids
                    .get(*conv_idx)
                    .and_then(|v| v.first())
                    .and_then(|o| *o);
                let add_out_id = model
                    .runtime_index
                    .node_output_ids
                    .get(*add_idx)
                    .and_then(|v| v.first())
                    .and_then(|o| *o);
                if let Some(oid) = conv_out_id {
                    env.mark_nhwc_by_id(oid);
                } else {
                    env.mark_nhwc(conv_out);
                }
                if let Some(mut conv_tensor) = env.remove(conv_out) {
                    if let Some(skip_tensor) = env.get(skip_name) {
                        if *post_activation == 1 {
                            yscv_kernels::add_relu_inplace(&mut conv_tensor, skip_tensor);
                        } else {
                            yscv_kernels::add_inplace(&mut conv_tensor, skip_tensor);
                        }
                        let add_out = &add_node.outputs[0];
                        if let Some(oid) = add_out_id {
                            env.insert_by_id(oid, conv_tensor);
                            env.mark_nhwc_by_id(oid);
                        } else {
                            env.insert(add_out.clone(), conv_tensor);
                            env.mark_nhwc(add_out);
                        }
                        if *post_activation == 1 {
                            let relu_out = &nodes[*relu_idx as usize].outputs[0];
                            env.alias(relu_out, add_out);
                        }
                    } else {
                        env.insert(conv_out.clone(), conv_tensor);
                        execute_node_with_layout_kind(
                            add_node,
                            env,
                            node_kind(&model.runtime_index.node_kinds, nodes, *add_idx),
                        )?;
                    }
                }
            }

            NodeAction::FusedTransposeMatMul {
                transpose_idx,
                matmul_idx,
                ..
            } => {
                // Transpose node is elided — read the pre-transpose
                // source (its input[0]) and feed it to the MatMul via
                // `matmul_2d_slices_trans_a` (BLAS `CblasTrans` under
                // the hood, else scratch-buffer fallback in the kernel).
                let transpose_node = &nodes[*transpose_idx];
                let matmul_node = &nodes[*matmul_idx];
                exec_fused_transpose_matmul(transpose_node, matmul_node, env)?;
            }

            NodeAction::Generic { node_idx, kind } => {
                let node = &nodes[*node_idx];
                execute_node_with_layout_kind(node, env, *kind)?;
            }
        }

        if do_profile {
            let elapsed = t0
                .as_ref()
                .map(|start| start.elapsed().as_nanos() as u64)
                .unwrap_or(0);
            match action {
                NodeAction::Conv { .. }
                | NodeAction::FusedDwPw { .. }
                | NodeAction::FusedPwDw { .. }
                | NodeAction::ConvAdd { .. } => {
                    *conv_ns += elapsed;
                    *conv_count += 1;
                }
                NodeAction::Generic { .. } => {
                    *other_ns += elapsed;
                    *other_count += 1;
                }
                _ => {}
            }
        }

        // YSCV_RUNNER_PROFILE=path — per-node aggregated timing for the
        // fused path. Skips the measurement entirely when env was unset.
        if runner_profile_enabled {
            let elapsed = t0
                .as_ref()
                .map(|start| start.elapsed().as_nanos() as u64)
                .unwrap_or(0);
            let (name, op, in_shape, out_shape) = match action {
                NodeAction::Skip => continue,
                NodeAction::Conv {
                    node_idx,
                    activation,
                } => {
                    let n = &nodes[*node_idx];
                    let op_label = match activation {
                        1 => "Conv_Relu",
                        2 => "Conv_Silu",
                        _ => "Conv",
                    };
                    let in_sh = n
                        .inputs
                        .first()
                        .and_then(|nm| env.get(nm))
                        .map(|t| t.shape().to_vec())
                        .unwrap_or_default();
                    let out_sh = n
                        .outputs
                        .first()
                        .and_then(|nm| env.get(nm))
                        .map(|t| t.shape().to_vec())
                        .unwrap_or_default();
                    (n.name.clone(), op_label.to_string(), in_sh, out_sh)
                }
                NodeAction::FusedDwPw { dw_idx, pw_idx, .. } => {
                    let dw = &nodes[*dw_idx];
                    let pw = &nodes[*pw_idx];
                    let in_sh = dw
                        .inputs
                        .first()
                        .and_then(|nm| env.get(nm))
                        .map(|t| t.shape().to_vec())
                        .unwrap_or_default();
                    let out_sh = pw
                        .outputs
                        .first()
                        .and_then(|nm| env.get(nm))
                        .map(|t| t.shape().to_vec())
                        .unwrap_or_default();
                    (
                        format!("{}+{}", dw.name, pw.name),
                        "FusedDwPw".to_string(),
                        in_sh,
                        out_sh,
                    )
                }
                NodeAction::FusedPwDw { pw_idx, dw_idx, .. } => {
                    let pw = &nodes[*pw_idx];
                    let dw = &nodes[*dw_idx];
                    let in_sh = pw
                        .inputs
                        .first()
                        .and_then(|nm| env.get(nm))
                        .map(|t| t.shape().to_vec())
                        .unwrap_or_default();
                    let out_sh = dw
                        .outputs
                        .first()
                        .and_then(|nm| env.get(nm))
                        .map(|t| t.shape().to_vec())
                        .unwrap_or_default();
                    (
                        format!("{}+{}", pw.name, dw.name),
                        "FusedPwDw".to_string(),
                        in_sh,
                        out_sh,
                    )
                }
                NodeAction::ConvAdd {
                    conv_idx,
                    add_idx,
                    post_activation,
                    ..
                } => {
                    let c = &nodes[*conv_idx];
                    let a = &nodes[*add_idx];
                    let op_label = if *post_activation == 1 {
                        "Conv_Add_Relu"
                    } else {
                        "Conv_Add"
                    };
                    let in_sh = c
                        .inputs
                        .first()
                        .and_then(|nm| env.get(nm))
                        .map(|t| t.shape().to_vec())
                        .unwrap_or_default();
                    let out_sh = a
                        .outputs
                        .first()
                        .and_then(|nm| env.get(nm))
                        .map(|t| t.shape().to_vec())
                        .unwrap_or_default();
                    (c.name.clone(), op_label.to_string(), in_sh, out_sh)
                }
                NodeAction::FusedTransposeMatMul {
                    transpose_idx,
                    matmul_idx,
                    ..
                } => {
                    let t = &nodes[*transpose_idx];
                    let m = &nodes[*matmul_idx];
                    let in_sh = t
                        .inputs
                        .first()
                        .and_then(|nm| env.get(nm))
                        .map(|v| v.shape().to_vec())
                        .unwrap_or_default();
                    let out_sh = m
                        .outputs
                        .first()
                        .and_then(|nm| env.get(nm))
                        .map(|v| v.shape().to_vec())
                        .unwrap_or_default();
                    (
                        format!("{}+{}", t.name, m.name),
                        "FusedTransposeMatMul".to_string(),
                        in_sh,
                        out_sh,
                    )
                }
                NodeAction::Generic { node_idx, .. } => {
                    let n = &nodes[*node_idx];
                    let in_sh = n
                        .inputs
                        .first()
                        .and_then(|nm| env.get(nm))
                        .map(|t| t.shape().to_vec())
                        .unwrap_or_default();
                    let out_sh = n
                        .outputs
                        .first()
                        .and_then(|nm| env.get(nm))
                        .map(|t| t.shape().to_vec())
                        .unwrap_or_default();
                    (n.name.clone(), n.op_type.clone(), in_sh, out_sh)
                }
                NodeAction::NchwcChain {
                    members,
                    entry_input,
                    exit_output,
                } => {
                    let first_idx = match members.first() {
                        Some(crate::loader::NchwcChainMember::Conv { node_idx, .. }) => *node_idx,
                        None => continue,
                    };
                    let in_sh = env
                        .get(entry_input)
                        .map(|t| t.shape().to_vec())
                        .unwrap_or_default();
                    let out_sh = env
                        .get(exit_output)
                        .map(|t| t.shape().to_vec())
                        .unwrap_or_default();
                    (
                        format!("{}_chain_{}", nodes[first_idx].name, members.len()),
                        format!("NchwcChain_{}", members.len()),
                        in_sh,
                        out_sh,
                    )
                }
            };
            runner_profile_record(&name, &op, elapsed, in_shape, out_shape);
        }

        // Early deallocation.
        // NchwcChain is handled via a dynamic Vec (covered_nodes_dyn)
        // since its member count is runtime-known; the static slice
        // match below covers the other fixed-arity variants.
        let covered_dyn: Vec<usize>;
        let covered_nodes: &[usize] = match action {
            NodeAction::Conv { node_idx, .. } | NodeAction::Generic { node_idx, .. } => {
                std::slice::from_ref(node_idx)
            }
            // Session 15 R5: `exec_fused_dw_pw` does its own early
            // cleanup of DW's inputs between DW and PW exec calls — the
            // outer loop must only handle PW's inputs here, otherwise
            // DW's inputs get double-decremented and the `saturating_sub`
            // hides the resulting off-by-one.
            NodeAction::FusedDwPw { pw_idx, .. } => std::slice::from_ref(pw_idx),
            // Mirror of FusedDwPw: `exec_fused_pw_dw` cleans up PW's
            // inputs between PW and DW, so the outer loop here only
            // touches DW's inputs (which includes the locally-owned PW
            // output that was never inserted into env — harmless since
            // `resolve_id` returns None for that unresolved name, skipping
            // the decrement).
            NodeAction::FusedPwDw { dw_idx, .. } => std::slice::from_ref(dw_idx),
            // `FusedTransposeMatMul` cleanup: the Transpose node was
            // elided from the plan, but its input tensor still lives
            // in `env`. Only the variant flagged with `cleanup_transpose`
            // covers the transpose's inputs — otherwise a transpose
            // feeding N MatMuls would get its input decremented N
            // times against an original use-count of 1, evicting the
            // pre-transpose tensor before every consumer has read it.
            NodeAction::FusedTransposeMatMul {
                transpose_idx,
                matmul_idx,
                cleanup_transpose,
            } => {
                if *cleanup_transpose {
                    &[*transpose_idx, *matmul_idx][..]
                } else {
                    std::slice::from_ref(matmul_idx)
                }
            }
            NodeAction::ConvAdd {
                conv_idx, add_idx, ..
            } => &[*conv_idx, *add_idx][..],
            NodeAction::NchwcChain { members, .. } => {
                covered_dyn = members
                    .iter()
                    .map(|m| match m {
                        crate::loader::NchwcChainMember::Conv { node_idx, .. } => *node_idx,
                    })
                    .collect();
                &covered_dyn[..]
            }
            NodeAction::Skip => continue,
        };
        let input_ids = &model.runtime_index.node_input_ids;
        for &nidx in covered_nodes {
            let node = &nodes[nidx];
            let pre_ids = if nidx < input_ids.len() {
                &input_ids[nidx]
            } else {
                &[][..]
            };
            for (inp_idx, inp) in node.inputs.iter().enumerate() {
                if inp.is_empty() {
                    continue;
                }
                let id = pre_ids
                    .get(inp_idx)
                    .and_then(|opt| *opt)
                    .or_else(|| env.resolve_id(inp));
                if let Some(id) = id
                    && id < remaining_uses.len()
                {
                    remaining_uses[id] = remaining_uses[id].saturating_sub(1);
                    if remaining_uses[id] == 0 && !output_id_mask[id] {
                        env.remove_by_id(id);
                    }
                }
            }
        }
    }
    Ok(())
}

fn run_onnx_model_jit(
    model: &OnnxModel,
    mut env: TensorEnv<'_, '_>,
) -> Result<HashMap<String, Tensor>, OnnxError> {
    let branches = &model.runtime_index.node_branches;
    let use_counts_by_id = &model.runtime_index.use_counts_by_id;
    let output_id_mask = build_output_id_mask(model, &env, use_counts_by_id.len());

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
        ensure_nchw(&mut env, name)?;
    }
    let mut result = HashMap::with_capacity(model.outputs.len());
    for name in &model.outputs {
        if let Some(t) = env.remove(name) {
            result.insert(name.clone(), t);
        } else if let Some(t) = env.get(name) {
            result.insert(name.clone(), t.clone());
        }
    }
    Ok(result)
}

fn run_onnx_model_sequential(
    model: &OnnxModel,
    mut env: TensorEnv<'_, '_>,
) -> Result<HashMap<String, Tensor>, OnnxError> {
    // --- Operator fusion: scan for fusible patterns ---
    // Build a set of node indices that should be skipped because they were
    // fused into the preceding node.  We also create synthetic "fused" nodes
    // that carry a combined op_type (e.g. "Conv_Relu").
    let nodes = &model.nodes;
    let node_kinds = &model.runtime_index.node_kinds;
    let mut skip = vec![false; nodes.len()];

    // Build reference counts: how many nodes consume each tensor as input.
    // Used by SiLU fusions to decide in-place vs allocating path.
    let use_counts = &model.runtime_index.use_counts;
    let use_counts_by_id = &model.runtime_index.use_counts_by_id;

    // Mutable remaining-use counters for early tensor deallocation.
    // When a tensor's remaining uses reach zero, free it to reduce working set.
    let mut remaining_uses: Vec<usize> = use_counts_by_id.clone();
    let output_id_mask = build_output_id_mask(model, &env, use_counts_by_id.len());

    for (i, node) in nodes.iter().enumerate() {
        if skip[i] {
            continue;
        }
        let kind = node_kind(node_kinds, nodes, i);

        // --- Conv → BatchNorm → Relu 3-node fusion ---
        if kind == NodeKind::Conv
            && let Some(next) = nodes.get(i + 1)
            && node_kind(node_kinds, nodes, i + 1) == NodeKind::BatchNormalization
            && !next.inputs.is_empty()
            && next.inputs[0] == node.outputs[0]
            && let Some((relu_idx, identity_idxs)) =
                find_relu_after_identity_chain(nodes, node_kinds, i + 2, &next.outputs[0])
        {
            execute_node_with_layout_kind(node, &mut env, kind)?;
            execute_node_with_layout_kind(next, &mut env, node_kind(node_kinds, nodes, i + 1))?;
            if let Some(tensor) = env.get_mut(&next.outputs[0]) {
                relu_inplace(tensor);
            }
            let source = &next.outputs[0];
            for &id_idx in &identity_idxs {
                env.alias(&nodes[id_idx].outputs[0], source);
            }
            env.alias(&nodes[relu_idx].outputs[0], source);
            if i + 1 < skip.len() {
                skip[i + 1] = true;
            }
            mark_skip_indices(&mut skip, &identity_idxs);
            mark_skip_indices(&mut skip, &[relu_idx]);
            continue;
        }

        // --- Conv + Relu fusion ---
        if kind == NodeKind::Conv
            && let Some((relu_idx, identity_idxs)) =
                find_relu_after_identity_chain(nodes, node_kinds, i + 1, &node.outputs[0])
        {
            exec_conv(node, &mut env, yscv_kernels::Activation::Relu)?;
            let source = &node.outputs[0];
            for &id_idx in &identity_idxs {
                env.alias(&nodes[id_idx].outputs[0], source);
            }
            env.alias(&nodes[relu_idx].outputs[0], source);
            mark_skip_indices(&mut skip, &identity_idxs);
            mark_skip_indices(&mut skip, &[relu_idx]);
            continue;
        }

        // --- Conv + SiLU fusion (Conv → Sigmoid → Mul) ---
        // Detect Sigmoid at i+1 and Mul at i+2 that form SiLU on Conv output.
        if kind == NodeKind::Conv {
            let conv_out = &node.outputs[0];
            // Look for Sigmoid(conv_out) → Mul(conv_out, sigmoid_out) pattern
            let mut silu_mul_idx = None;
            for sig_offset in 1..=2 {
                if let Some(sig) = nodes.get(i + sig_offset)
                    && node_kind(node_kinds, nodes, i + sig_offset) == NodeKind::Sigmoid
                    && sig.inputs.len() == 1
                    && sig.inputs[0] == *conv_out
                {
                    let sig_out = &sig.outputs[0];
                    for mul_offset in (sig_offset + 1)..=(sig_offset + 2) {
                        if let Some(mul) = nodes.get(i + mul_offset)
                            && node_kind(node_kinds, nodes, i + mul_offset) == NodeKind::Mul
                            && mul.inputs.len() == 2
                            && ((mul.inputs[0] == *sig_out && mul.inputs[1] == *conv_out)
                                || (mul.inputs[1] == *sig_out && mul.inputs[0] == *conv_out))
                        {
                            silu_mul_idx = Some((sig_offset, mul_offset, mul.outputs[0].clone()));
                            break;
                        }
                    }
                    if silu_mul_idx.is_some() {
                        break;
                    }
                }
            }
            if let Some((sig_off, mul_off, mul_out)) = silu_mul_idx {
                let conv_out_uses = tensor_use_count(&env, use_counts_by_id, use_counts, conv_out);
                if conv_out_uses <= 2 {
                    // Fuse SiLU into Conv GEMM tiles (applied cache-hot after bias).
                    exec_conv(node, &mut env, yscv_kernels::Activation::Silu)?;
                    env.alias(&mul_out, conv_out);
                } else {
                    // Other consumers need raw conv_out — can't fuse.
                    execute_node_with_layout_kind(node, &mut env, kind)?;
                    if let Some(tensor) = env.get(conv_out) {
                        let result = yscv_kernels::silu(tensor);
                        env.insert(mul_out.clone(), result);
                    }
                }
                let is_nhwc = env.is_nhwc(conv_out);
                if is_nhwc {
                    env.mark_nhwc(&mul_out);
                }
                // Execute any intermediate nodes between Conv and Sigmoid,
                // then mark them as done so the main loop doesn't re-execute them.
                for mid in 1..sig_off {
                    if i + mid < skip.len() && !skip[i + mid] {
                        execute_node_with_layout_kind(
                            &nodes[i + mid],
                            &mut env,
                            node_kind(node_kinds, nodes, i + mid),
                        )?;
                        skip[i + mid] = true;
                    }
                }
                if i + sig_off < skip.len() {
                    skip[i + sig_off] = true;
                }
                if i + mul_off < skip.len() {
                    skip[i + mul_off] = true;
                }
                continue;
            }
        }

        // --- Conv + Add (residual connection) fusion ---
        // Pattern: Conv → Add(conv_out, skip_connection), optionally → Relu
        // Reuses conv_out buffer for the result, avoiding allocation.
        if kind == NodeKind::Conv
            && let Some(next) = nodes.get(i + 1)
            && node_kind(node_kinds, nodes, i + 1) == NodeKind::Add
            && next.inputs.len() == 2
            && (next.inputs[0] == node.outputs[0] || next.inputs[1] == node.outputs[0])
        {
            let conv_out = &node.outputs[0];
            let skip_idx = if &next.inputs[0] == conv_out { 1 } else { 0 };
            let skip_name = &next.inputs[skip_idx];
            let conv_out_uses = tensor_use_count(&env, use_counts_by_id, use_counts, conv_out);

            // Only fuse if conv_out has 2 uses (Add is its only other consumer besides the
            // initializer lookup that may happen). If it has more uses, we need to keep it
            // for other consumers.
            if conv_out_uses <= 2 {
                execute_node_with_layout_kind(node, &mut env, kind)?;

                // Capture NHWC flag before remove (remove clears it).
                let is_nhwc = env.is_nhwc(conv_out);

                // Add skip_connection in-place to conv_out
                if let Some(mut conv_tensor) = env.remove(conv_out) {
                    if let Some(skip_tensor) = env.get(skip_name) {
                        yscv_kernels::add_inplace(&mut conv_tensor, skip_tensor);
                        let add_out = &next.outputs[0];

                        // Check if Relu follows Add
                        if let Some((relu_idx, identity_idxs)) =
                            find_relu_after_identity_chain(nodes, node_kinds, i + 2, add_out)
                        {
                            relu_inplace(&mut conv_tensor);
                            env.insert(add_out.clone(), conv_tensor);
                            if is_nhwc {
                                env.mark_nhwc(add_out);
                            }
                            let source = add_out;
                            for &id_idx in &identity_idxs {
                                env.alias(&nodes[id_idx].outputs[0], source);
                            }
                            env.alias(&nodes[relu_idx].outputs[0], source);
                            mark_skip_indices(&mut skip, &identity_idxs);
                            mark_skip_indices(&mut skip, &[relu_idx]);
                        } else {
                            env.insert(add_out.clone(), conv_tensor);
                            if is_nhwc {
                                env.mark_nhwc(add_out);
                            }
                        }

                        if i + 1 < skip.len() {
                            skip[i + 1] = true;
                        }
                        continue;
                    } else {
                        env.insert(conv_out.clone(), conv_tensor);
                    }
                }
            }
        }

        // --- BatchNormalization + Relu fusion ---
        if kind == NodeKind::BatchNormalization
            && let Some((relu_idx, identity_idxs)) =
                find_relu_after_identity_chain(nodes, node_kinds, i + 1, &node.outputs[0])
        {
            execute_node_with_layout_kind(node, &mut env, kind)?;
            if let Some(tensor) = env.get_mut(&node.outputs[0]) {
                relu_inplace(tensor);
            }
            let source = &node.outputs[0];
            for &id_idx in &identity_idxs {
                env.alias(&nodes[id_idx].outputs[0], source);
            }
            env.alias(&nodes[relu_idx].outputs[0], source);
            mark_skip_indices(&mut skip, &identity_idxs);
            mark_skip_indices(&mut skip, &[relu_idx]);
            continue;
        }

        // --- Gemm + Relu fusion ---
        if kind == NodeKind::Gemm
            && let Some((relu_idx, identity_idxs)) =
                find_relu_after_identity_chain(nodes, node_kinds, i + 1, &node.outputs[0])
        {
            execute_node_with_layout_kind(node, &mut env, kind)?;
            if let Some(tensor) = env.get_mut(&node.outputs[0]) {
                relu_inplace(tensor);
            }
            let source = &node.outputs[0];
            for &id_idx in &identity_idxs {
                env.alias(&nodes[id_idx].outputs[0], source);
            }
            env.alias(&nodes[relu_idx].outputs[0], source);
            mark_skip_indices(&mut skip, &identity_idxs);
            mark_skip_indices(&mut skip, &[relu_idx]);
            continue;
        }

        // --- Add + Relu fusion (with in-place Add when possible) ---
        if kind == NodeKind::Add
            && node.inputs.len() == 2
            && let Some((relu_idx, identity_idxs)) =
                find_relu_after_identity_chain(nodes, node_kinds, i + 1, &node.outputs[0])
        {
            let a_nhwc = env.is_nhwc(&node.inputs[0]);
            let b_nhwc = env.is_nhwc(&node.inputs[1]);
            let same_shape_nhwc = a_nhwc == b_nhwc
                && match (env.get(&node.inputs[0]), env.get(&node.inputs[1])) {
                    (Some(a), Some(b)) => a.shape() == b.shape(),
                    _ => false,
                };
            let mut did_inplace = false;
            if same_shape_nhwc {
                let a_uses = tensor_use_count(&env, use_counts_by_id, use_counts, &node.inputs[0]);
                let b_uses = tensor_use_count(&env, use_counts_by_id, use_counts, &node.inputs[1]);
                if a_uses <= 1 || b_uses <= 1 {
                    let (consume_idx, other_idx) = if a_uses <= 1 { (0, 1) } else { (1, 0) };
                    if let Some(mut target) = env.remove(&node.inputs[consume_idx]) {
                        if let Some(other) = env.get(&node.inputs[other_idx]) {
                            yscv_kernels::add_relu_inplace(&mut target, other);
                            env.insert(node.outputs[0].clone(), target);
                            if a_nhwc {
                                env.mark_nhwc(&node.outputs[0]);
                            }
                            did_inplace = true;
                        } else {
                            env.insert(node.inputs[consume_idx].clone(), target);
                        }
                    }
                }
            }
            if !did_inplace {
                execute_node_with_layout_kind(node, &mut env, kind)?;
                if let Some(tensor) = env.get_mut(&node.outputs[0]) {
                    relu_inplace(tensor);
                }
            }
            let source = &node.outputs[0];
            for &id_idx in &identity_idxs {
                env.alias(&nodes[id_idx].outputs[0], source);
            }
            env.alias(&nodes[relu_idx].outputs[0], source);
            mark_skip_indices(&mut skip, &identity_idxs);
            mark_skip_indices(&mut skip, &[relu_idx]);
            continue;
        }

        // --- MatMul + Add fusion (Gemm-like) ---
        if kind == NodeKind::MatMul
            && let Some(next) = nodes.get(i + 1)
            && node_kind(node_kinds, nodes, i + 1) == NodeKind::Add
            && next.inputs.len() == 2
            && (next.inputs[0] == node.outputs[0] || next.inputs[1] == node.outputs[0])
        {
            execute_node_with_layout_kind(node, &mut env, kind)?;
            execute_node_with_layout_kind(next, &mut env, node_kind(node_kinds, nodes, i + 1))?;
            if i + 1 < skip.len() {
                skip[i + 1] = true;
            }
            continue;
        }

        // --- Sigmoid + Mul → SiLU fusion ---
        // SiLU(x) = x * sigmoid(x).  Pattern: Sigmoid(x)->y, Mul(x,y)->z
        // Single-pass SIMD kernel avoids separate sigmoid allocation + Mul dispatch.
        if kind == NodeKind::Sigmoid && node.inputs.len() == 1 {
            let sig_in = &node.inputs[0];
            let sig_out = &node.outputs[0];
            // Look ahead up to 3 positions for a matching Mul (SiLU pattern).
            let mut found_silu = false;
            for look in 1..=3 {
                if let Some(next) = nodes.get(i + look)
                    && node_kind(node_kinds, nodes, i + look) == NodeKind::Mul
                    && next.inputs.len() == 2
                {
                    let is_silu = (next.inputs[0] == *sig_out && next.inputs[1] == *sig_in)
                        || (next.inputs[1] == *sig_out && next.inputs[0] == *sig_in);
                    if is_silu {
                        let is_nhwc = env.is_nhwc(sig_in);
                        let mul_out = &next.outputs[0];
                        // sig_in is used by Sigmoid + Mul = 2 fused consumers.
                        // Only remove if no other node needs it.
                        let sig_in_uses =
                            tensor_use_count(&env, use_counts_by_id, use_counts, sig_in);
                        if sig_in_uses <= 2 {
                            if let Some(mut tensor) = env.remove(sig_in) {
                                yscv_kernels::silu_inplace(&mut tensor);
                                env.insert(mul_out.clone(), tensor);
                            }
                        } else if let Some(x_tensor) = env.get(sig_in) {
                            let result_tensor = yscv_kernels::silu(x_tensor);
                            env.insert(mul_out.clone(), result_tensor);
                        }
                        if is_nhwc {
                            env.mark_nhwc(mul_out);
                        }
                        // Execute any intermediate nodes, then mark them done
                        // so the main loop doesn't re-execute them.
                        for mid in 1..look {
                            if i + mid < skip.len() && !skip[i + mid] {
                                execute_node_with_layout_kind(
                                    &nodes[i + mid],
                                    &mut env,
                                    node_kind(node_kinds, nodes, i + mid),
                                )?;
                                skip[i + mid] = true;
                            }
                        }
                        if i + look < skip.len() {
                            skip[i + look] = true;
                        }
                        found_silu = true;
                        break;
                    }
                }
            }
            if found_silu {
                continue;
            }
        }

        // --- In-place Add: reuse buffer when one input is last-use ---
        if kind == NodeKind::Add && node.inputs.len() == 2 {
            let a_nhwc = env.is_nhwc(&node.inputs[0]);
            let b_nhwc = env.is_nhwc(&node.inputs[1]);
            if a_nhwc == b_nhwc {
                let same_shape = match (env.get(&node.inputs[0]), env.get(&node.inputs[1])) {
                    (Some(a), Some(b)) => a.shape() == b.shape(),
                    _ => false,
                };
                if same_shape {
                    let a_uses =
                        tensor_use_count(&env, use_counts_by_id, use_counts, &node.inputs[0]);
                    let b_uses =
                        tensor_use_count(&env, use_counts_by_id, use_counts, &node.inputs[1]);
                    if a_uses <= 1 || b_uses <= 1 {
                        let (consume_idx, other_idx) = if a_uses <= 1 { (0, 1) } else { (1, 0) };
                        if let Some(mut target) = env.remove(&node.inputs[consume_idx]) {
                            if let Some(other) = env.get(&node.inputs[other_idx]) {
                                yscv_kernels::add_inplace(&mut target, other);
                                env.insert(node.outputs[0].clone(), target);
                                if a_nhwc {
                                    env.mark_nhwc(&node.outputs[0]);
                                }
                                continue;
                            }
                            env.insert(node.inputs[consume_idx].clone(), target);
                        }
                    }
                }
            }
        }

        // Zero-copy Reshape: avoid data clone when the data input has only
        // one consumer (this Reshape node).
        if kind == NodeKind::Reshape {
            for name in &node.inputs {
                if !name.is_empty() {
                    ensure_nchw(&mut env, name)?;
                }
            }
            exec_reshape_zerocopy(node, &mut env, use_counts)?;
            continue;
        }

        // Fast path for Conv: use pre-computed params to skip attr HashMap lookups
        if matches!(
            kind,
            NodeKind::Conv | NodeKind::ConvRelu | NodeKind::ConvSilu
        ) {
            let cp = model
                .runtime_index
                .conv_params
                .get(i)
                .and_then(|o| o.as_ref());
            let activation = match kind {
                NodeKind::ConvRelu => yscv_kernels::Activation::Relu,
                NodeKind::ConvSilu => yscv_kernels::Activation::Silu,
                _ => yscv_kernels::Activation::None,
            };

            // --- DW+PW fusion: detect depthwise Conv followed by pointwise 1x1 ---
            if cp.is_some_and(|p| p.is_depthwise) {
                // Look ahead for pointwise 1x1 Conv consuming our output exclusively
                let dw_out = &node.outputs[0];
                let dw_uses = use_counts.get(dw_out).copied().unwrap_or(0);
                if dw_uses == 1
                    && let Some(next_idx) = (i + 1..nodes.len()).find(|&j| !skip[j])
                {
                    let next_cp = model
                        .runtime_index
                        .conv_params
                        .get(next_idx)
                        .and_then(|o| o.as_ref());
                    let next = &nodes[next_idx];
                    let next_kind = node_kind(node_kinds, nodes, next_idx);
                    if next_cp.is_some_and(|p| p.is_pointwise && !p.has_padding)
                        && next.inputs.first().map(|s| s.as_str()) == Some(dw_out.as_str())
                        && matches!(
                            next_kind,
                            NodeKind::Conv | NodeKind::ConvRelu | NodeKind::ConvSilu
                        )
                    {
                        let pw_activation = match next_kind {
                            NodeKind::ConvRelu => yscv_kernels::Activation::Relu,
                            NodeKind::ConvSilu => yscv_kernels::Activation::Silu,
                            _ => yscv_kernels::Activation::None,
                        };
                        let dw_input_ids_slice: &[Option<usize>] = model
                            .runtime_index
                            .node_input_ids
                            .get(i)
                            .map(|v| v.as_slice())
                            .unwrap_or(&[]);
                        exec_fused_dw_pw(
                            node,
                            next,
                            &mut env,
                            activation,
                            pw_activation,
                            cp,
                            next_cp,
                            dw_input_ids_slice,
                            &mut remaining_uses,
                            &output_id_mask,
                        )?;
                        // Decrement PW inputs so DW output also gets
                        // freed. Mirrors the post-action cleanup below,
                        // but tailored to the fused pair since the
                        // outer `continue` below would otherwise skip
                        // it.
                        let pw_pre_ids = model
                            .runtime_index
                            .node_input_ids
                            .get(next_idx)
                            .map(|v| v.as_slice())
                            .unwrap_or(&[]);
                        for (inp_idx, inp) in next.inputs.iter().enumerate() {
                            if inp.is_empty() {
                                continue;
                            }
                            let id = pw_pre_ids
                                .get(inp_idx)
                                .and_then(|opt| *opt)
                                .or_else(|| env.resolve_id(inp));
                            if let Some(id) = id
                                && id < remaining_uses.len()
                            {
                                remaining_uses[id] = remaining_uses[id].saturating_sub(1);
                                if remaining_uses[id] == 0 && !output_id_mask[id] {
                                    env.remove_by_id(id);
                                }
                            }
                        }
                        skip[next_idx] = true;
                        continue;
                    }
                }
            }

            let prepacked = prepacked_for_conv_node(model, i);
            exec_conv_with_params(node, &mut env, activation, cp, prepacked)?;
            env.mark_nhwc(&node.outputs[0]);
        } else {
            execute_node_with_layout_kind(node, &mut env, kind)?;
        }

        // --- Early deallocation: free tensors whose last consumer was this node ---
        let input_ids = &model.runtime_index.node_input_ids;
        let pre_ids = if i < input_ids.len() {
            &input_ids[i]
        } else {
            &[][..]
        };
        for (inp_idx, inp) in node.inputs.iter().enumerate() {
            if inp.is_empty() {
                continue;
            }
            // Use pre-resolved ID (O(1)) when available, fallback to HashMap.
            let id = pre_ids
                .get(inp_idx)
                .and_then(|opt| *opt)
                .or_else(|| env.resolve_id(inp));
            if let Some(id) = id
                && id < remaining_uses.len()
            {
                remaining_uses[id] = remaining_uses[id].saturating_sub(1);
                if remaining_uses[id] == 0 && !output_id_mask[id] {
                    env.remove_by_id(id);
                }
            }
        }
    }

    // Optional per-op trace for debugging inference divergence.
    if std::env::var("CPU_TRACE").is_ok() {
        for node in nodes {
            for out_name in &node.outputs {
                if let Some(t) = env.get(out_name) {
                    let d = t.data();
                    if d.is_empty() {
                        continue;
                    }
                    let max = d.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let min = d.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                    let mean = d.iter().sum::<f32>() / d.len() as f32;
                    let nhwc = if env.is_nhwc(out_name) { " [NHWC]" } else { "" };
                    eprintln!(
                        "[{:>20}] {:60} shape={:?} min={:>10.4} max={:>10.4} mean={:>10.4}{}",
                        node.op_type,
                        out_name,
                        t.shape(),
                        min,
                        max,
                        mean,
                        nhwc,
                    );
                }
            }
        }
    }

    // Ensure all outputs are in NCHW (ONNX standard layout)
    for name in &model.outputs {
        ensure_nchw(&mut env, name)?;
    }

    let mut result = HashMap::with_capacity(model.outputs.len());
    for name in &model.outputs {
        if let Some(t) = env.remove(name) {
            result.insert(name.clone(), t);
        } else if let Some(t) = env.get(name) {
            result.insert(name.clone(), t.clone());
        } else {
            eprintln!("warning: ONNX output '{}' not found in environment", name);
        }
    }
    Ok(result)
}

/// One node's timing sample, captured during `profile_onnx_model_cpu`.
///
/// Lives alongside the per-op-type text summary so the JSON output path
/// can emit a per-instance breakdown. Shapes are captured before the
/// node executes (in) and immediately after (out); for fused ops like
/// SiLU(Sigmoid+Mul) we record the fused name and the final Mul's output
/// shape.
#[derive(Debug, Clone)]
struct NodeTiming {
    name: String,
    op: String,
    ms: f64,
    in_shape: Vec<usize>,
    out_shape: Vec<usize>,
}

/// Profile CPU inference: measure per-op-type timing.
///
/// When the env var `YSCV_PROFILE_JSON` is set to a file path, a
/// machine-readable JSON with per-node instance timings is written to
/// that path in addition to the standard text summary. Format:
/// ```json
/// {"engine":"yscv","total_ms":11.47,"nodes":[
///   {"name":"Conv_0","op":"Conv","ms":0.12,
///    "in_shape":[1,3,128,128],"out_shape":[1,16,64,64]}, ...
/// ]}
/// ```
/// The file is consumed by `scripts/gap_diff.py` for apples-to-apples
/// comparison against ORT's profiling JSON.
pub fn profile_onnx_model_cpu(
    model: &OnnxModel,
    inputs: HashMap<String, Tensor>,
) -> Result<(), OnnxError> {
    use std::time::Instant;

    let mut env = TensorEnv::from_model(model);
    for (name, tensor) in inputs {
        env.insert(name, tensor);
    }

    let nodes = &model.nodes;
    let node_kinds = &model.runtime_index.node_kinds;
    let mut skip = vec![false; nodes.len()];
    let mut timings: HashMap<String, (f64, usize)> = HashMap::new();
    let mut conv_details: Vec<(String, f64, Vec<usize>, Vec<usize>)> = Vec::new();
    // Per-instance timings — fuel for `YSCV_PROFILE_JSON` output.
    let mut instance_timings: Vec<NodeTiming> = Vec::with_capacity(nodes.len());

    let prof_use_counts = &model.runtime_index.use_counts;
    let prof_use_counts_by_id = &model.runtime_index.use_counts_by_id;

    for (i, node) in nodes.iter().enumerate() {
        if skip[i] {
            continue;
        }

        let kind = node_kind(node_kinds, nodes, i);

        // SiLU fusion in profiler too (with look-ahead)
        if kind == NodeKind::Sigmoid && node.inputs.len() == 1 {
            let sig_in = &node.inputs[0];
            let sig_out = &node.outputs[0];
            let mut found_silu = false;
            for look in 1..=3 {
                if let Some(next) = nodes.get(i + look)
                    && node_kind(node_kinds, nodes, i + look) == NodeKind::Mul
                    && next.inputs.len() == 2
                    && ((next.inputs[0] == *sig_out && next.inputs[1] == *sig_in)
                        || (next.inputs[1] == *sig_out && next.inputs[0] == *sig_in))
                {
                    let is_nhwc = env.is_nhwc(sig_in);
                    let mul_out = &next.outputs[0];
                    let start = Instant::now();
                    let sig_in_uses =
                        tensor_use_count(&env, prof_use_counts_by_id, prof_use_counts, sig_in);
                    if sig_in_uses <= 2 {
                        if let Some(mut tensor) = env.remove(sig_in) {
                            yscv_kernels::silu_inplace(&mut tensor);
                            env.insert(mul_out.clone(), tensor);
                        }
                    } else if let Some(x_tensor) = env.get(sig_in) {
                        let result_tensor = yscv_kernels::silu(x_tensor);
                        env.insert(mul_out.clone(), result_tensor);
                    }
                    if is_nhwc {
                        env.mark_nhwc(mul_out);
                    }
                    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                    let entry = timings.entry("SiLU(fused)".to_string()).or_insert((0.0, 0));
                    entry.0 += elapsed;
                    entry.1 += 1;
                    let fused_out_shape = env
                        .get(mul_out)
                        .map(|t| t.shape().to_vec())
                        .unwrap_or_default();
                    let fused_in_shape = env
                        .get(sig_in)
                        .map(|t| t.shape().to_vec())
                        .unwrap_or_else(|| fused_out_shape.clone());
                    instance_timings.push(NodeTiming {
                        name: node.name.clone(),
                        op: "SiLU(fused)".to_string(),
                        ms: elapsed,
                        in_shape: fused_in_shape,
                        out_shape: fused_out_shape,
                    });
                    // Execute intermediate nodes, mark done to prevent re-execution.
                    for mid in 1..look {
                        if i + mid < skip.len() && !skip[i + mid] {
                            let mid_node = &nodes[i + mid];
                            let mid_in = mid_node
                                .inputs
                                .first()
                                .and_then(|n| env.get(n))
                                .map(|t| t.shape().to_vec())
                                .unwrap_or_default();
                            let mid_start = Instant::now();
                            execute_node_with_layout_kind(
                                mid_node,
                                &mut env,
                                node_kind(node_kinds, nodes, i + mid),
                            )?;
                            let mid_elapsed = mid_start.elapsed().as_secs_f64() * 1000.0;
                            let mid_entry =
                                timings.entry(mid_node.op_type.clone()).or_insert((0.0, 0));
                            mid_entry.0 += mid_elapsed;
                            mid_entry.1 += 1;
                            let mid_out = mid_node
                                .outputs
                                .first()
                                .and_then(|n| env.get(n))
                                .map(|t| t.shape().to_vec())
                                .unwrap_or_default();
                            instance_timings.push(NodeTiming {
                                name: mid_node.name.clone(),
                                op: mid_node.op_type.clone(),
                                ms: mid_elapsed,
                                in_shape: mid_in,
                                out_shape: mid_out,
                            });
                            skip[i + mid] = true;
                        }
                    }
                    if i + look < skip.len() {
                        skip[i + look] = true;
                    }
                    found_silu = true;
                    break;
                }
            }
            if found_silu {
                continue;
            }
        }

        let op_type = node.op_type.clone();
        let in_shape = node
            .inputs
            .first()
            .and_then(|name| env.get(name))
            .map(|t| t.shape().to_vec())
            .unwrap_or_default();

        let start = Instant::now();
        execute_node_with_layout_kind(node, &mut env, kind)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        let out_shape = node
            .outputs
            .first()
            .and_then(|n| env.get(n))
            .map(|t| t.shape().to_vec())
            .unwrap_or_default();

        if kind == NodeKind::Conv {
            conv_details.push((
                node.name.clone(),
                elapsed,
                in_shape.clone(),
                out_shape.clone(),
            ));
        }

        instance_timings.push(NodeTiming {
            name: node.name.clone(),
            op: op_type.clone(),
            ms: elapsed,
            in_shape,
            out_shape,
        });

        let entry = timings.entry(op_type).or_insert((0.0, 0));
        entry.0 += elapsed;
        entry.1 += 1;
    }

    for name in &model.outputs {
        ensure_nchw(&mut env, name)?;
    }

    println!("\n  ── CPU Profile (per-op timing) ──");
    let mut sorted: Vec<_> = timings.into_iter().collect();
    sorted.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());
    let total: f64 = sorted.iter().map(|(_, (t, _))| t).sum();
    for (op, (time_ms, count)) in &sorted {
        println!("    {:>8.2}ms {:>5}x  {}", time_ms, count, op);
    }
    println!("    {:>8.2}ms  total", total);

    // Per-Conv detail: top 10 slowest
    if !conv_details.is_empty() {
        conv_details.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        println!("\n  ── Top Conv layers ──");
        for (name, ms, in_s, out_s) in conv_details.iter().take(10) {
            println!("    {:>6.2}ms  {:?} → {:?}  {}", ms, in_s, out_s, name);
        }
    }

    // JSON dump for `scripts/gap_diff.py`. Written only when the env var
    // names a target path — avoids side-effects in the default UX.
    if let Ok(path) = std::env::var("YSCV_PROFILE_JSON")
        && !path.is_empty()
    {
        write_profile_json(&path, &instance_timings, total)?;
        eprintln!("  ── wrote profile JSON to {path} ──");
    }

    Ok(())
}

/// Writes a per-instance timing profile as JSON. Format is stable — it's
/// consumed by `scripts/gap_diff.py`; if you change fields, update the
/// parser in lockstep. We hand-roll the JSON instead of pulling in serde
/// because the schema is tiny (strings + numbers + shape arrays) and the
/// crate already compiles cleanly without a JSON dep.
fn write_profile_json(path: &str, nodes: &[NodeTiming], total_ms: f64) -> Result<(), OnnxError> {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(64 * nodes.len() + 128);
    out.push_str("{\"engine\":\"yscv\",\"total_ms\":");
    let _ = write!(out, "{:.6}", total_ms);
    out.push_str(",\"nodes\":[");
    for (i, n) in nodes.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str("{\"name\":\"");
        json_escape_into(&mut out, &n.name);
        out.push_str("\",\"op\":\"");
        json_escape_into(&mut out, &n.op);
        out.push_str("\",\"ms\":");
        let _ = write!(out, "{:.6}", n.ms);
        out.push_str(",\"in_shape\":");
        shape_to_json(&mut out, &n.in_shape);
        out.push_str(",\"out_shape\":");
        shape_to_json(&mut out, &n.out_shape);
        out.push('}');
    }
    out.push_str("]}\n");
    std::fs::write(path, out).map_err(|e| OnnxError::DecodeFailed {
        message: format!("write profile JSON to {path}: {e}"),
    })
}

/// Escapes a string into an existing JSON output buffer. Covers only the
/// characters that might appear in ONNX node names / op types: backslash,
/// quote, and control characters are escaped; everything else goes as-is.
fn json_escape_into(out: &mut String, s: &str) {
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                use std::fmt::Write as _;
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
}

/// Formats a `[usize]` shape as a JSON number array.
fn shape_to_json(out: &mut String, shape: &[usize]) {
    use std::fmt::Write as _;
    out.push('[');
    for (i, d) in shape.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let _ = write!(out, "{d}");
    }
    out.push(']');
}

#[inline]
pub(crate) fn get_tensor<'a>(
    env: &'a TensorEnv,
    node_name: &str,
    input_name: &str,
) -> Result<&'a Tensor, OnnxError> {
    env.get(input_name).ok_or_else(|| OnnxError::MissingInput {
        node: node_name.to_string(),
        input: input_name.to_string(),
    })
}

#[inline]
pub(crate) fn get_attr_ints(node: &OnnxNode, name: &str) -> Option<Vec<i64>> {
    match node.attributes.get(name) {
        Some(OnnxAttribute::Ints(v)) => Some(v.clone()),
        _ => None,
    }
}

#[inline]
pub(crate) fn get_attr_int(node: &OnnxNode, name: &str) -> Option<i64> {
    match node.attributes.get(name) {
        Some(OnnxAttribute::Int(v)) => Some(*v),
        _ => None,
    }
}

#[inline]
pub(crate) fn get_attr_float(node: &OnnxNode, name: &str) -> Option<f32> {
    match node.attributes.get(name) {
        Some(OnnxAttribute::Float(v)) => Some(*v),
        _ => None,
    }
}

#[inline]
pub(crate) fn get_attr_string(node: &OnnxNode, name: &str) -> Option<String> {
    match node.attributes.get(name) {
        Some(OnnxAttribute::String(v)) => Some(v.clone()),
        _ => None,
    }
}

/// Converts inputs in the environment to f32 before executing a node, then converts
/// outputs back to the original dtype. Ops that handle dtypes themselves (Cast, Shape,
/// Identity, quantization ops) are exempt.
fn execute_node_kind(
    node: &OnnxNode,
    env: &mut TensorEnv,
    kind: NodeKind,
) -> Result<(), OnnxError> {
    // Ops that should NOT have automatic dtype conversion
    let dtype_exempt = matches!(
        node.op_type.as_str(),
        "Cast"
            | "Shape"
            | "Identity"
            | "Constant"
            | "ConstantOfShape"
            | "QuantizeLinear"
            | "DequantizeLinear"
            | "DynamicQuantizeLinear"
            | "QLinearConv"
            | "QLinearMatMul"
            | "MatMulInteger"
            | "ConvInteger"
    );

    // Detect original dtype from first input (if any) and convert inputs to f32
    let orig_dtype = if !dtype_exempt && !node.inputs.is_empty() {
        let first_dt = node
            .inputs
            .iter()
            .filter_map(|name| env.get(name))
            .map(|t| t.dtype())
            .find(|&dt| dt != DType::F32);

        if let Some(dt) = first_dt {
            // Convert all non-f32 inputs to f32 in-place
            for input_name in &node.inputs {
                if let Some(tensor) = env.get(input_name)
                    && tensor.dtype() != DType::F32
                {
                    let converted = tensor.to_dtype(DType::F32);
                    env.insert(input_name.clone(), converted);
                }
            }
            Some(dt)
        } else {
            None
        }
    } else {
        None
    };

    // Execute the actual op
    execute_node_inner_kind(node, env, kind)?;

    // Convert outputs back to original dtype if needed
    if let Some(dt) = orig_dtype {
        for output_name in &node.outputs {
            if let Some(tensor) = env.get(output_name)
                && tensor.dtype() == DType::F32
            {
                let converted = tensor.to_dtype(dt);
                env.insert(output_name.clone(), converted);
            }
        }
    }

    Ok(())
}

#[inline]
fn execute_node_inner_kind(
    node: &OnnxNode,
    env: &mut TensorEnv,
    kind: NodeKind,
) -> Result<(), OnnxError> {
    execute_node_inner_kind_fast(node, env, kind, None)
}

#[inline]
fn execute_node_inner_kind_fast(
    node: &OnnxNode,
    env: &mut TensorEnv,
    kind: NodeKind,
    conv_params: Option<&crate::loader::ConvParams>,
) -> Result<(), OnnxError> {
    match kind {
        NodeKind::Conv => {
            exec_conv_with_params(node, env, yscv_kernels::Activation::None, conv_params, None)
        }
        NodeKind::ConvRelu => {
            exec_conv_with_params(node, env, yscv_kernels::Activation::Relu, conv_params, None)
        }
        NodeKind::ConvSilu => {
            exec_conv_with_params(node, env, yscv_kernels::Activation::Silu, conv_params, None)
        }
        NodeKind::Relu => exec_relu(node, env),
        NodeKind::BatchNormalization => exec_batch_norm(node, env),
        NodeKind::Gemm => exec_gemm(node, env),
        NodeKind::Add => exec_add(node, env),
        NodeKind::MatMul => exec_matmul(node, env),
        NodeKind::Mul => exec_mul(node, env),
        NodeKind::Sigmoid => exec_sigmoid(node, env),
        NodeKind::Reshape => exec_reshape(node, env),
        NodeKind::Constant => exec_constant(node, env),
        NodeKind::Concat => exec_concat(node, env),
        NodeKind::Transpose => exec_transpose(node, env),
        NodeKind::Other => execute_node_inner_slow(node, env),
    }
}

fn execute_node_inner_slow(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    match node.op_type.as_str() {
        "MaxPool" => exec_max_pool(node, env),
        "AveragePool" => exec_avg_pool(node, env),
        "GlobalAveragePool" => exec_global_avg_pool(node, env),
        "Flatten" => exec_flatten(node, env),
        "Sub" => exec_sub(node, env),
        "Softmax" => exec_softmax(node, env),
        "Transpose" => exec_transpose(node, env),
        "Concat" => exec_concat(node, env),
        "Unsqueeze" => exec_unsqueeze(node, env),
        "Squeeze" => exec_squeeze(node, env),
        "Clip" => exec_clip(node, env),
        "Shape" => exec_shape(node, env),
        "Gather" => exec_gather(node, env),
        "Constant" => exec_constant(node, env),
        "Dropout" => exec_dropout(node, env),
        "Pad" => exec_pad(node, env),
        "Pow" => exec_pow(node, env),
        "Sqrt" => exec_sqrt(node, env),
        "Exp" => exec_exp(node, env),
        "Log" => exec_log(node, env),
        "Neg" => exec_neg(node, env),
        "Abs" => exec_abs(node, env),
        "Reciprocal" => exec_reciprocal(node, env),
        "Tanh" => exec_tanh(node, env),
        "Floor" => exec_floor(node, env),
        "Ceil" => exec_ceil(node, env),
        "Equal" => exec_cmp(node, env, 0),
        "Greater" => exec_cmp(node, env, 1),
        "Less" => exec_cmp(node, env, 2),
        "Where" => exec_where(node, env),
        "ReduceMean" => exec_reduce_mean(node, env),
        "ReduceSum" => exec_reduce_sum(node, env),
        "Split" => exec_split(node, env),
        "Slice" => exec_slice(node, env),
        "Expand" => exec_expand(node, env),
        "Tile" => exec_tile(node, env),
        "Cast" => exec_cast(node, env),
        "Div" => exec_div(node, env),
        "Min" => exec_min_max(node, env, false),
        "Max" => exec_min_max(node, env, true),
        "ReduceMax" => exec_reduce_max(node, env),
        "ConvTranspose" => exec_conv_transpose(node, env),
        "DeformConv" => exec_deform_conv(node, env),
        "Resize" => exec_resize(node, env),
        "LeakyRelu" => exec_leaky_relu(node, env),
        "Elu" => exec_elu(node, env),
        "ReduceMin" => exec_reduce_min(node, env),
        "ReduceProd" => exec_reduce_prod(node, env),
        "Identity" => exec_identity(node, env),
        "QuantizeLinear" => exec_quantize_linear(node, env),
        "DequantizeLinear" => exec_dequantize_linear(node, env),
        "Gelu" => exec_gelu(node, env),
        "Erf" => exec_erf(node, env),
        "HardSigmoid" => exec_hard_sigmoid(node, env),
        "InstanceNormalization" => exec_instance_norm(node, env),
        "LpNormalization" => exec_lp_norm(node, env),
        "Upsample" => exec_resize(node, env),
        "Selu" => exec_selu(node, env),
        "Celu" => exec_celu(node, env),
        "ThresholdedRelu" => exec_thresholded_relu(node, env),
        "Hardmax" => exec_hardmax(node, env),
        "OneHot" => exec_onehot(node, env),
        "Range" => exec_range(node, env),
        "NonZero" => exec_nonzero(node, env),
        "LayerNormalization" => exec_layer_norm(node, env),
        "GatherElements" => exec_gather_elements(node, env),
        "ScatterElements" => exec_scatter_elements(node, env),
        "Einsum" => exec_einsum(node, env),
        "ReduceL2" => exec_reduce_l2(node, env),
        "ReduceL1" => exec_reduce_l1(node, env),
        "CumSum" => exec_cumsum(node, env),
        "ArgMax" => exec_argmax(node, env),
        "ArgMin" => exec_argmin(node, env),
        "TopK" => exec_topk(node, env),
        "ScatterND" => exec_scatter_nd(node, env),
        "GatherND" => exec_gather_nd(node, env),
        "DepthToSpace" => exec_depth_to_space(node, env),
        "SpaceToDepth" => exec_space_to_depth(node, env),
        "GridSample" => exec_grid_sample(node, env),
        "RoiAlign" => exec_roi_align(node, env),
        "Compress" => exec_compress(node, env),
        "QLinearConv" => exec_qlinear_conv(node, env),
        "QLinearMatMul" => exec_qlinear_matmul(node, env),
        "MatMulInteger" => exec_matmul_integer(node, env),
        "ConvInteger" => exec_conv_integer(node, env),
        "DynamicQuantizeLinear" => exec_dynamic_quantize_linear(node, env),
        "Not" => exec_not(node, env),
        "And" => exec_logical_bin(node, env, 0),
        "Or" => exec_logical_bin(node, env, 1),
        "Xor" => exec_logical_bin(node, env, 2),
        "Sin" => exec_tensor_op(node, env, |t| t.sin()),
        "Cos" => exec_tensor_op(node, env, |t| t.cos()),
        "Tan" => exec_unary(node, env, |v| v.tan()),
        "Asin" => exec_unary(node, env, |v| v.asin()),
        "Acos" => exec_unary(node, env, |v| v.acos()),
        "Atan" => exec_unary(node, env, |v| v.atan()),
        "Sinh" => exec_unary(node, env, |v| v.sinh()),
        "Cosh" => exec_unary(node, env, |v| v.cosh()),
        "Asinh" => exec_unary(node, env, |v| v.asinh()),
        "Acosh" => exec_unary(node, env, |v| v.acosh()),
        "Atanh" => exec_unary(node, env, |v| v.atanh()),
        "Round" => exec_unary(node, env, |v| v.round()),
        "Sign" => exec_unary(node, env, |v| v.signum()),
        "IsNaN" => exec_unary(node, env, |v| if v.is_nan() { 1.0 } else { 0.0 }),
        "IsInf" => exec_unary(node, env, |v| if v.is_infinite() { 1.0 } else { 0.0 }),
        "Mod" => exec_mod(node, env),
        "GreaterOrEqual" => exec_cmp(node, env, 3),
        "LessOrEqual" => exec_cmp(node, env, 4),
        "BitShift" => exec_bitshift(node, env),
        "Mean" => exec_variadic_mean(node, env),
        "Sum" => exec_variadic_sum(node, env),
        "ConstantOfShape" => exec_constant_of_shape(node, env),
        "LRN" => exec_lrn(node, env),
        "Softplus" => exec_unary(node, env, |v| (1.0 + v.exp()).ln()),
        "Softsign" => exec_unary(node, env, |v| v / (1.0 + v.abs())),
        "HardSwish" => exec_unary(node, env, |v| v * ((v + 3.0).clamp(0.0, 6.0) / 6.0)),
        "Mish" => exec_unary(node, env, |v| v * (1.0 + v.exp()).ln().tanh()),
        "NonMaxSuppression" => exec_nms(node, env),
        "GroupQueryAttention" => exec_grouped_query_attention(node, env),
        "Conv_Relu" => exec_conv(node, env, yscv_kernels::Activation::Relu),
        "BatchNormalization_Relu" => {
            exec_batch_norm(node, env)?;
            exec_relu_inplace(node, env)
        }
        other => Err(OnnxError::UnsupportedOpType {
            op_type: other.to_string(),
        }),
    }
}
