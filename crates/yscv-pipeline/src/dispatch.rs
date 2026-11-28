//! Runtime accelerator dispatch — turns a validated [`InferenceTask`]
//! into something that can actually run frames.
//!
//! The [`AcceleratorDispatcher`] trait gives every backend the same
//! shape: `&[(name, bytes)]` in, `Vec<(name, bytes)>` out, plus a
//! `recover` hook for transient faults. The factory
//! [`dispatcher_for`] matches on [`Accelerator`] and returns a boxed
//! trait object; unavailable features produce a dispatcher whose
//! `dispatch` returns an "accelerator not compiled" error, so the
//! workspace builds under any feature combo.
//!
//! # Byte-level I/O contract
//!
//! We deliberately use raw `Vec<u8>` on the trait boundary rather than
//! `yscv_tensor::Tensor`. Two reasons:
//!
//! 1. **Camera-fed inputs are bytes, not tensors.** DMA-BUF NV12 / raw
//!    RGB8 / JPEG arrive with no dtype; they're whatever the camera
//!    driver handed us. Forcing a `Tensor` wrapper would require
//!    pointless zero-copy gymnastics at the ingress point.
//!
//! 2. **Task-to-task chaining stays uniform.** Dispatcher outputs are
//!    always f32 little-endian bytes in row-major order (the same
//!    layout `yscv_tensor::Tensor::data()` exposes). Downstream tasks
//!    consume those bytes directly.
//!
//! For performance-critical consumers the low-level `RknnPipelinedPool`
//! / `compile_mpsgraph_plan` APIs remain available — this trait is the
//! **config-driven** path, optimised for correctness and uniformity,
//! not for sub-millisecond hot loops.

use rustc_hash::{FxBuildHasher, FxHashMap};

use crate::accelerator::Accelerator;
use crate::config::InferenceTask;
use crate::error::Error;

/// Trait implemented by accelerator-specific dispatchers (CPU, RKNN,
/// Metal, Gpu). One impl per accelerator family. Loaded once per task
/// at pipeline start; called per-frame on the hot path.
pub trait AcceleratorDispatcher: Send + Sync {
    /// Human label for logs (e.g. `"cpu"`, `"rknn (Core0)"`).
    fn label(&self) -> &str;

    /// Run inference on one frame's worth of inputs.
    ///
    /// `inputs` is a slice of `(tensor_name, byte_slice)` pairs, one
    /// per graph input the task declares. Lookup is by name so the
    /// caller doesn't have to know the model's internal input order.
    /// Returns output tensors as `(name, f32_le_bytes)` in the model's
    /// declared output order.
    fn dispatch(&self, inputs: &[(&str, &[u8])]) -> Result<Vec<(String, Vec<u8>)>, Error>;

    /// Attempt to recover this dispatcher after a transient fault
    /// (NPU hang, lost GPU context, etc.). Default: no-op. Backends
    /// that can reset (e.g. RKNN's `recover_failed`) override.
    fn recover(&self) -> Result<(), Error> {
        Ok(())
    }
}

/// Build the right [`AcceleratorDispatcher`] for a task's accelerator
/// assignment. Errors propagate both missing feature flags (dispatcher
/// type not compiled) and runtime failures (model file refused by the
/// SDK, librknnrt.so not loadable, etc.).
pub fn dispatcher_for(task: &InferenceTask) -> Result<Box<dyn AcceleratorDispatcher>, Error> {
    match &task.accelerator {
        Accelerator::Cpu => Ok(Box::new(CpuDispatcher::new(task)?)),
        Accelerator::Rknn { core } => {
            #[cfg(feature = "rknn")]
            {
                Ok(Box::new(RknnDispatcher::new(task, *core)?))
            }
            #[cfg(not(feature = "rknn"))]
            {
                let _ = core;
                Err(Error::AcceleratorNotCompiled {
                    task: task.name.clone(),
                    accelerator: "Rknn",
                    requires: "--features rknn",
                })
            }
        }
        Accelerator::RknnMatmul { m, k, n, dtype } => {
            #[cfg(feature = "rknn")]
            {
                Ok(Box::new(RknnMatmulDispatcher::new(
                    task, *m, *k, *n, *dtype,
                )?))
            }
            #[cfg(not(feature = "rknn"))]
            {
                let _ = (m, k, n, dtype);
                Err(Error::AcceleratorNotCompiled {
                    task: task.name.clone(),
                    accelerator: "RknnMatmul",
                    requires: "--features rknn",
                })
            }
        }
        Accelerator::MetalMps => {
            #[cfg(all(feature = "metal-backend", target_os = "macos"))]
            {
                Ok(Box::new(MetalDispatcher::new(task)?))
            }
            #[cfg(not(all(feature = "metal-backend", target_os = "macos")))]
            {
                Err(Error::AcceleratorNotCompiled {
                    task: task.name.clone(),
                    accelerator: "MetalMps",
                    requires: "--features metal-backend on macOS",
                })
            }
        }
        Accelerator::Gpu => {
            #[cfg(feature = "gpu")]
            {
                Ok(Box::new(GpuDispatcher::new(task)?))
            }
            #[cfg(not(feature = "gpu"))]
            {
                Err(Error::AcceleratorNotCompiled {
                    task: task.name.clone(),
                    accelerator: "Gpu",
                    requires: "--features gpu",
                })
            }
        }
    }
}

/// Build a task's ONNX executable, tagging a failure with the task and path.
fn load_onnx(task: &InferenceTask) -> Result<yscv_onnx::OnnxExecutable, Error> {
    yscv_onnx::OnnxLoader::load(&task.model_path)
        .build()
        .map_err(|source| Error::LoadModel {
            task: task.name.clone(),
            path: task.model_path.clone(),
            source: Box::new(source),
        })
}

// ---- CPU dispatcher -------------------------------------------------

/// CPU dispatcher — runs ONNX via `yscv-onnx`'s pure-Rust CPU runner.
/// Input bytes are interpreted as f32 little-endian in the model's
/// declared input shape (row-major, matching `Tensor::data()` layout).
struct CpuDispatcher {
    label: String,
    model: yscv_onnx::OnnxExecutable,
    /// Per-input shape snapshot (from the model's `input_info`), so
    /// `dispatch` can reshape raw bytes into proper tensors.
    input_shapes: FxHashMap<String, Vec<usize>>,
}

impl CpuDispatcher {
    fn new(task: &InferenceTask) -> Result<Self, Error> {
        let model = load_onnx(task)?;
        // The executable's declared inputs are a `Vec<String>` of input
        // names. Shapes live per-initializer attributes; for a first
        // cut we leave `input_shapes` empty and let `dispatch` require
        // exact input-size match against whatever the caller passes.
        Ok(Self {
            label: format!("cpu ({})", task.name),
            model,
            input_shapes: FxHashMap::default(),
        })
    }
}

impl AcceleratorDispatcher for CpuDispatcher {
    fn label(&self) -> &str {
        &self.label
    }

    fn dispatch(&self, inputs: &[(&str, &[u8])]) -> Result<Vec<(String, Vec<u8>)>, Error> {
        let mut feed: FxHashMap<String, yscv_tensor::Tensor> =
            FxHashMap::with_capacity_and_hasher(inputs.len(), FxBuildHasher);
        for &(name, bytes) in inputs {
            if bytes.len() % 4 != 0 {
                return Err(Error::InputNotF32 {
                    input: name.to_string(),
                    len: bytes.len(),
                });
            }
            // Reinterpret as f32 LE. For the first-cut we treat the
            // input as a 1-D tensor sized `len / 4`; downstream ops
            // reshape via the graph. If the caller wants an explicit
            // shape, they must supply it via `input_shapes`.
            let n = bytes.len() / 4;
            let mut data: Vec<f32> = Vec::with_capacity(n);
            for chunk in bytes.as_chunks::<4>().0 {
                data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            let shape = self
                .input_shapes
                .get(name)
                .cloned()
                .unwrap_or_else(|| vec![n]);
            let t =
                yscv_tensor::Tensor::from_vec(shape, data).map_err(|source| Error::BuildInput {
                    input: name.to_string(),
                    source,
                })?;
            feed.insert(name.to_string(), t);
        }
        let outs = yscv_onnx::run_onnx_model(&self.model, feed)?;
        Ok(outs
            .into_iter()
            .map(|(name, t)| {
                let mut bytes = Vec::with_capacity(t.data().len() * 4);
                for &v in t.data() {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                (name, bytes)
            })
            .collect())
    }
}

// ---- RKNN dispatcher ------------------------------------------------

#[cfg(feature = "rknn")]
struct RknnDispatcher {
    label: String,
    pool: yscv_kernels::RknnPipelinedPool,
}

#[cfg(feature = "rknn")]
impl RknnDispatcher {
    fn new(task: &InferenceTask, core: crate::accelerator::NpuCoreSpec) -> Result<Self, Error> {
        let bytes = load_or_compile_rknn(task)?;
        let pool = yscv_kernels::RknnPipelinedPool::new(&bytes, &[core.to_mask()])?;
        Ok(Self {
            label: format!("rknn {:?} ({})", core, task.name),
            pool,
        })
    }
}

/// Resolve a task's `model_path` to RKNN bytes.
///
/// If the path points at a `.rknn` file, it's read directly.
///
/// If it points at an `.onnx` file, on-device compilation kicks in:
/// the ONNX is compiled to RKNN bytes via
/// `yscv_kernels::compile_onnx_to_rknn`, cached to disk alongside the
/// source (`model.onnx` → `model.rknn`), and the result is returned.
/// Subsequent runs skip the compile and load the cached file.
///
/// Config options for the compile (fp16 vs int8 calibration) aren't
/// plumbed from TOML yet — a follow-up plan can add a
/// `[tasks.rknn_compile]` subsection. For now the default
/// (`RknnCompileConfig::default()`) kicks in: fp16 export, no
/// quantization.
#[cfg(feature = "rknn")]
fn load_or_compile_rknn(task: &InferenceTask) -> Result<Vec<u8>, Error> {
    let ext = task
        .model_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    if ext.eq_ignore_ascii_case("rknn") {
        return read_model(task);
    }

    if ext.eq_ignore_ascii_case("onnx") {
        let cache_path = task.model_path.with_extension("rknn");
        let onnx_bytes = read_model(task)?;
        let cfg = yscv_kernels::RknnCompileConfig::default();
        // `load_onnx_as_rknn` does: check cache → compile → write cache →
        // return an RknnBackend. We want the bytes (to feed the pool),
        // so replicate the cache-first logic here and bypass the backend
        // construction.
        if let Ok(cached) = std::fs::read(&cache_path) {
            return Ok(cached);
        }
        let rknn_bytes =
            yscv_kernels::compile_onnx_to_rknn(&onnx_bytes, &cfg).map_err(|source| {
                Error::RknnCompile {
                    task: task.name.clone(),
                    path: task.model_path.clone(),
                    source: Box::new(source),
                }
            })?;
        // Best-effort cache write — not fatal if it fails (e.g. read-only fs).
        if let Err(e) = std::fs::write(&cache_path, &rknn_bytes) {
            eprintln!(
                "[yscv-pipeline] task '{}': couldn't cache compiled rknn at {}: {e}",
                task.name,
                cache_path.display()
            );
        } else {
            eprintln!(
                "[yscv-pipeline] task '{}': cached compiled rknn at {} \
                 ({} bytes)",
                task.name,
                cache_path.display(),
                rknn_bytes.len()
            );
        }
        return Ok(rknn_bytes);
    }

    Err(Error::UnsupportedModelExtension {
        task: task.name.clone(),
        path: task.model_path.clone(),
    })
}

#[cfg(feature = "rknn")]
fn read_model(task: &InferenceTask) -> Result<Vec<u8>, Error> {
    std::fs::read(&task.model_path).map_err(|source| Error::ModelRead {
        task: task.name.clone(),
        path: task.model_path.clone(),
        source,
    })
}

#[cfg(feature = "rknn")]
impl AcceleratorDispatcher for RknnDispatcher {
    fn label(&self) -> &str {
        &self.label
    }

    fn dispatch(&self, inputs: &[(&str, &[u8])]) -> Result<Vec<(String, Vec<u8>)>, Error> {
        let outs = self.pool.run(inputs)?;
        // RKNN always returns dequantized `Vec<Tensor>` in output-index
        // order. The model's output_attrs gives us names.
        // Since the pool doesn't expose the names directly, synthesise
        // from `out_{i}`. Caller's graph wiring refers to tensor names
        // via `output.<task>.<name>` so this is only user-visible
        // through that path — documenting as out_{i} in the cookbook.
        Ok(outs
            .into_iter()
            .enumerate()
            .map(|(i, t)| {
                let mut bytes = Vec::with_capacity(t.data().len() * 4);
                for &v in t.data() {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                (format!("output{i}"), bytes)
            })
            .collect())
    }

    fn recover(&self) -> Result<(), Error> {
        // Single-slot pool → recover slot 0.
        Ok(self.pool.recover_failed(0)?)
    }
}

// ---- RKNN matmul dispatcher --------------------------------------

#[cfg(feature = "rknn")]
struct RknnMatmulDispatcher {
    label: String,
    matmul: yscv_kernels::RknnMatmul,
    /// Pre-allocated A/B/C buffers, kept alive for the dispatcher's
    /// lifetime. RKNN's bind functions store FFI pointers into these,
    /// so they must outlive every `dispatch` call. `Mutex` because
    /// `as_mut_slice` needs unique access; `dispatch` is single-call,
    /// not pipelined.
    a_mem: std::sync::Mutex<yscv_kernels::RknnMem>,
    b_mem: std::sync::Mutex<yscv_kernels::RknnMem>,
    c_mem: std::sync::Mutex<yscv_kernels::RknnMem>,
    /// Cached byte sizes from `io_attr` (avoid an FFI dive per call).
    a_size: usize,
    b_size: usize,
    c_size: usize,
}

#[cfg(feature = "rknn")]
impl RknnMatmulDispatcher {
    fn new(
        task: &InferenceTask,
        m: u32,
        k: u32,
        n: u32,
        dtype: crate::accelerator::MatmulDtype,
    ) -> Result<Self, Error> {
        if m == 0 || k == 0 || n == 0 {
            return Err(Error::ZeroMatmulDim {
                task: task.name.clone(),
                m,
                k,
                n,
            });
        }
        let matmul = yscv_kernels::RknnMatmul::new(m as i32, k as i32, n as i32, dtype.to_rknn())?;

        let a_size = matmul.a_attr().size as usize;
        let b_size = matmul.b_attr().size as usize;
        let c_size = matmul.c_attr().size as usize;

        // Pre-allocate + bind. `bind_*` is sticky on the matmul ctx,
        // so future `run()` calls reuse this binding.
        let a_mem = matmul.alloc_mem(a_size)?;
        matmul.bind_a(&a_mem)?;
        let b_mem = matmul.alloc_mem(b_size)?;
        matmul.bind_b(&b_mem)?;
        let c_mem = matmul.alloc_mem(c_size)?;
        matmul.bind_c(&c_mem)?;

        Ok(Self {
            label: format!("rknn-matmul {m}×{k}×{n} {dtype:?} ({})", task.name),
            matmul,
            a_mem: std::sync::Mutex::new(a_mem),
            b_mem: std::sync::Mutex::new(b_mem),
            c_mem: std::sync::Mutex::new(c_mem),
            a_size,
            b_size,
            c_size,
        })
    }
}

#[cfg(feature = "rknn")]
impl AcceleratorDispatcher for RknnMatmulDispatcher {
    fn label(&self) -> &str {
        &self.label
    }

    #[allow(unsafe_code)]
    fn dispatch(&self, inputs: &[(&str, &[u8])]) -> Result<Vec<(String, Vec<u8>)>, Error> {
        // Contract: inputs must contain exactly two named "a" and "b"
        // tensors with the SDK-declared byte counts. Output is a
        // single named "c" tensor.
        let mut a_bytes: Option<&[u8]> = None;
        let mut b_bytes: Option<&[u8]> = None;
        for &(name, bytes) in inputs {
            match name {
                "a" => a_bytes = Some(bytes),
                "b" => b_bytes = Some(bytes),
                other => {
                    return Err(Error::UnknownMatmulInput {
                        name: other.to_string(),
                    });
                }
            }
        }
        let a_bytes = a_bytes.ok_or(Error::MissingMatmulInput { name: "a" })?;
        let b_bytes = b_bytes.ok_or(Error::MissingMatmulInput { name: "b" })?;

        if a_bytes.len() != self.a_size {
            return Err(Error::MatmulInputSize {
                name: "a",
                expected: self.a_size,
                got: a_bytes.len(),
            });
        }
        if b_bytes.len() != self.b_size {
            return Err(Error::MatmulInputSize {
                name: "b",
                expected: self.b_size,
                got: b_bytes.len(),
            });
        }

        // Memcpy → flush → run → invalidate → readback.
        {
            let mut a = self.a_mem.lock().map_err(|_| Error::LockPoisoned {
                lock: "matmul a_mem",
            })?;
            // SAFETY: the matmul call is serialized by the mutex and no NPU
            // operation is in flight while inputs are being populated.
            unsafe { a.as_mut_slice() }.copy_from_slice(a_bytes);
            a.sync_to_device()?;
        }
        {
            let mut b = self.b_mem.lock().map_err(|_| Error::LockPoisoned {
                lock: "matmul b_mem",
            })?;
            // SAFETY: the matmul call is serialized by the mutex and no NPU
            // operation is in flight while inputs are being populated.
            unsafe { b.as_mut_slice() }.copy_from_slice(b_bytes);
            b.sync_to_device()?;
        }

        self.matmul.run()?;

        let c_bytes = {
            let c = self.c_mem.lock().map_err(|_| Error::LockPoisoned {
                lock: "matmul c_mem",
            })?;
            c.sync_from_device()?;
            // SAFETY: the synchronous matmul run completed before reading the
            // output buffer, and the mutex excludes concurrent CPU access.
            (unsafe { c.as_slice() })[..self.c_size].to_vec()
        };

        Ok(vec![("c".to_string(), c_bytes)])
    }
}

// ---- Metal MPSGraph dispatcher -----------------------------------

#[cfg(all(feature = "metal-backend", target_os = "macos"))]
struct MetalDispatcher {
    label: String,
    model: yscv_onnx::OnnxExecutable,
    /// Plan is lazily compiled on first `dispatch` (we need a concrete
    /// input shape for the MPSGraph placeholder, and the config doesn't
    /// carry shape hints yet). Subsequent calls reuse the plan.
    /// `Mutex` serialises all access — Apple documents MPSGraph as
    /// thread-safe for dispatch but not for concurrent mutation.
    plan: std::sync::Mutex<Option<yscv_onnx::MpsGraphPlan>>,
}

// SAFETY: `MpsGraphPlan` holds Metal / ObjC pointers (`MTLDevice`,
// `MTLCommandQueue`, `MTLBuffer`, `MpsGraph`) that the `metal-rs` and
// our bindings crates don't auto-derive `Send` for, since raw pointers
// are `!Send` by default. Sending a `MetalDispatcher` across threads
// is safe because:
//   1. Every access to the inner `MpsGraphPlan` goes through
//      `self.plan.lock()` — concurrent mutation is impossible.
//   2. Apple's ObjC retain/release is atomic (documented), so moving a
//      retained pointer across threads doesn't corrupt refcounts.
//   3. `MPSGraphExecutable`'s `runWithMTLCommandQueue:` is documented
//      as safe to invoke from any thread that wraps the call in an
//      autoreleasepool (which our dispatcher does via
//      `run_mpsgraph_plan`'s internal `autoreleasepool(|| ...)`).
// `Sync` falls out for free once `Send` is claimed on `Mutex<T>` for
// `T: Send`.
#[cfg(all(feature = "metal-backend", target_os = "macos"))]
#[allow(unsafe_code)]
unsafe impl Send for MetalDispatcher {}
#[cfg(all(feature = "metal-backend", target_os = "macos"))]
#[allow(unsafe_code)]
unsafe impl Sync for MetalDispatcher {}

#[cfg(all(feature = "metal-backend", target_os = "macos"))]
impl MetalDispatcher {
    fn new(task: &InferenceTask) -> Result<Self, Error> {
        let model = load_onnx(task)?;
        Ok(Self {
            label: format!("metal-mps ({})", task.name),
            model,
            plan: std::sync::Mutex::new(None),
        })
    }
}

#[cfg(all(feature = "metal-backend", target_os = "macos"))]
impl AcceleratorDispatcher for MetalDispatcher {
    fn label(&self) -> &str {
        &self.label
    }

    fn dispatch(&self, inputs: &[(&str, &[u8])]) -> Result<Vec<(String, Vec<u8>)>, Error> {
        // Interpret input bytes as f32 LE — same contract as
        // `CpuDispatcher`. For each input we also materialise a Tensor
        // so we can pass its view to `compile_mpsgraph_plan` on the
        // first call.
        let input_tensors: Vec<(String, yscv_tensor::Tensor)> = inputs
            .iter()
            .map(|(name, bytes)| {
                if bytes.len() % 4 != 0 {
                    return Err(Error::InputNotF32 {
                        input: (*name).to_string(),
                        len: bytes.len(),
                    });
                }
                let n = bytes.len() / 4;
                let mut data: Vec<f32> = Vec::with_capacity(n);
                for chunk in bytes.as_chunks::<4>().0 {
                    data.push(f32::from_le_bytes(*chunk));
                }
                let t = yscv_tensor::Tensor::from_vec(vec![n], data).map_err(|source| {
                    Error::BuildInput {
                        input: (*name).to_string(),
                        source,
                    }
                })?;
                Ok(((*name).to_string(), t))
            })
            .collect::<Result<_, _>>()?;

        let mut guard = self.plan.lock().map_err(|_| Error::LockPoisoned {
            lock: "MetalDispatcher plan",
        })?;
        if guard.is_none() {
            let pairs: Vec<(&str, &yscv_tensor::Tensor)> =
                input_tensors.iter().map(|(n, t)| (n.as_str(), t)).collect();
            let plan = yscv_onnx::compile_mpsgraph_plan(&self.model, &pairs)?;
            *guard = Some(plan);
        }
        let plan = guard.as_ref().expect("plan populated above");

        let feeds: Vec<(&str, &[f32])> = input_tensors
            .iter()
            .map(|(n, t)| (n.as_str(), t.data()))
            .collect();
        let outs = yscv_onnx::run_mpsgraph_plan(plan, &feeds)?;

        Ok(outs
            .into_iter()
            .map(|(name, t)| {
                let mut bytes = Vec::with_capacity(t.data().len() * 4);
                for &v in t.data() {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                (name, bytes)
            })
            .collect())
    }

    fn recover(&self) -> Result<(), Error> {
        // Drop the plan; next `dispatch` will recompile. Useful if the
        // GPU device was reset or the plan hit a stuck state — rare
        // outside driver bugs, but the hook is free.
        if let Ok(mut g) = self.plan.lock() {
            *g = None;
        }
        Ok(())
    }
}

// ---- wgpu (cross-platform GPU) dispatcher ------------------------

#[cfg(feature = "gpu")]
struct GpuDispatcher {
    label: String,
    model: yscv_onnx::OnnxExecutable,
}

#[cfg(feature = "gpu")]
impl GpuDispatcher {
    fn new(task: &InferenceTask) -> Result<Self, Error> {
        let model = load_onnx(task)?;
        Ok(Self {
            label: format!("gpu-wgpu ({})", task.name),
            model,
        })
    }
}

#[cfg(feature = "gpu")]
impl AcceleratorDispatcher for GpuDispatcher {
    fn label(&self) -> &str {
        &self.label
    }

    fn dispatch(&self, inputs: &[(&str, &[u8])]) -> Result<Vec<(String, Vec<u8>)>, Error> {
        // Same f32-LE byte contract as the other dispatchers.
        let mut feed: FxHashMap<String, yscv_tensor::Tensor> =
            FxHashMap::with_capacity_and_hasher(inputs.len(), FxBuildHasher);
        for &(name, bytes) in inputs {
            if bytes.len() % 4 != 0 {
                return Err(Error::InputNotF32 {
                    input: name.to_string(),
                    len: bytes.len(),
                });
            }
            let n = bytes.len() / 4;
            let mut data: Vec<f32> = Vec::with_capacity(n);
            for chunk in bytes.as_chunks::<4>().0 {
                data.push(f32::from_le_bytes(*chunk));
            }
            let t = yscv_tensor::Tensor::from_vec(vec![n], data).map_err(|source| {
                Error::BuildInput {
                    input: name.to_string(),
                    source,
                }
            })?;
            feed.insert(name.to_string(), t);
        }
        let outs = yscv_onnx::run_onnx_model_gpu(&self.model, feed)?;
        Ok(outs
            .into_iter()
            .map(|(name, t)| {
                let mut bytes = Vec::with_capacity(t.data().len() * 4);
                for &v in t.data() {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                (name, bytes)
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    // The two tests below are mutually exclusive on features — when
    // both `rknn` and `metal-backend` are on, both tests `cfg`-out and
    // these imports become unused. Allow that.
    #[allow(unused_imports)]
    use super::*;
    #[allow(unused_imports)]
    use crate::accelerator::Accelerator;
    #[allow(unused_imports)]
    use crate::config::{InferenceTask, TensorBinding};
    #[allow(unused_imports)]
    use std::path::PathBuf;

    /// Dispatcher factory rejects unsupported accelerator at build time
    /// when the matching cargo feature is off.
    #[test]
    #[cfg(not(feature = "rknn"))]
    fn rknn_dispatcher_errors_without_feature() {
        let task = InferenceTask {
            name: "det".into(),
            model_path: PathBuf::from("/tmp/x.rknn"),
            accelerator: Accelerator::Rknn {
                core: crate::accelerator::NpuCoreSpec::Core0,
            },
            inputs: vec![TensorBinding {
                name: "images".into(),
                source: "camera".into(),
            }],
            outputs: vec![],
        };
        let Err(err) = dispatcher_for(&task) else {
            panic!("expected feature-gate error, got Ok");
        };
        assert!(
            matches!(&err, Error::AcceleratorNotCompiled { task, accelerator: "Rknn", .. } if task == "det"),
            "{err:?}"
        );
        assert!(err.to_string().contains("--features rknn"), "{err}");
    }

    #[test]
    #[cfg(not(all(feature = "metal-backend", target_os = "macos")))]
    fn metal_dispatcher_errors_without_feature() {
        let task = InferenceTask {
            name: "det".into(),
            model_path: PathBuf::from("/tmp/x.onnx"),
            accelerator: Accelerator::MetalMps,
            inputs: vec![TensorBinding {
                name: "images".into(),
                source: "camera".into(),
            }],
            outputs: vec![],
        };
        let Err(err) = dispatcher_for(&task) else {
            panic!("MetalMps without feature should fail");
        };
        assert!(
            matches!(&err, Error::AcceleratorNotCompiled { task, accelerator: "MetalMps", .. } if task == "det"),
            "{err:?}"
        );
        assert!(
            err.to_string().contains("metal-backend"),
            "expected feature-gate error, got: {err}"
        );
    }
}
