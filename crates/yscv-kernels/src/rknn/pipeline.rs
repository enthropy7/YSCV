//! Pipelined async NPU pool: one slot per physical core, `submit`/`wait`
//! API that mirrors the MPSGraph pipelined path we use on Apple Silicon.
//!
//! Includes automatic recovery from transient NPU faults (`TIMEOUT`,
//! `CTX_INVALID`, `DEVICE_UNAVAILABLE`, `DEVICE_UNMATCH`): the pool
//! tracks per-slot consecutive failures and transparently resets the
//! offending context + rebuilds its input/output memory bindings once a
//! threshold is crossed. Caller-side bugs (`PARAM_INVALID`,
//! `INPUT_INVALID`, `OUTPUT_INVALID`, `MODEL_INVALID`) are **not**
//! recovered — they bubble up unchanged so the caller notices.
//!
//! Usage:
//!
//! ```no_run
//! use yscv_kernels::{
//!     NpuCoreMask, RknnPipelinedPool,
//! };
//! # fn run(model: &[u8], frame_bytes_0: &[u8], frame_bytes_1: &[u8]) -> Result<(), yscv_kernels::KernelError> {
//! // Triple-buffered across RK3588's 3 NPU cores.
//! let pool = RknnPipelinedPool::new(
//!     model,
//!     &[NpuCoreMask::Core0, NpuCoreMask::Core1, NpuCoreMask::Core2],
//! )?;
//!
//! // Submit frame 0 on core 0, frame 1 on core 1 (both running concurrently),
//! // then pick up frame 0's result once it's done.
//! let h0 = pool.submit(&[("input0", frame_bytes_0)])?;
//! let h1 = pool.submit(&[("input0", frame_bytes_1)])?;
//! let out0 = pool.wait(h0)?;
//! let out1 = pool.wait(h1)?;
//! # let _ = (out0, out1);
//! # Ok(())
//! # }
//! ```
//!
//! ## Back-pressure
//!
//! Each slot has a pre-allocated `RknnMem` per input / output tensor and
//! a `Mutex<Option<AsyncFrame>>` tracking the most recent submission.
//! When `submit` rotates to a slot that already has an in-flight frame
//! (caller has more outstanding handles than slots), it transparently
//! waits on the previous frame before reusing the slot's memory. No
//! CPU↔NPU buffer aliasing is possible.
//!
//! ## Scaling
//!
//! `submit`/`wait` are `&self`; the pool is `Sync` and can be shared
//! across capture / post-process threads. Per-slot `Mutex`es / `RwLock`s
//! serialise accesses to the same slot; different slots run concurrently.

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use yscv_tensor::Tensor;

use crate::KernelError;

use super::backend::{AsyncFrame, RknnBackend, RknnMem};
use super::consts::{NpuCoreMask, RKNN_FLAG_ASYNC_MASK};
use super::ffi::is_recoverable_rknn_error;

/// After this many consecutive recoverable errors on a slot, `submit` /
/// `wait` transparently reset that slot's NPU context before returning.
/// Chosen conservatively (3 strikes): a single spurious timeout should
/// not destroy warm context state, but a stuck NPU recovers within
/// ~50 ms on RK3588.
const RECOVERY_THRESHOLD: u32 = 3;

/// Handle to an inference submitted via [`RknnPipelinedPool::submit`].
/// Must be passed to [`RknnPipelinedPool::wait`] to retrieve outputs.
///
/// Dropping without calling `wait` is safe but wasteful: the slot's
/// next `submit` will block on the in-flight frame before reusing its
/// memory, so the NPU work completes either way — the caller just
/// loses visibility of its outputs.
#[must_use = "an RknnInferenceHandle represents in-flight NPU work; \
              call `wait` to collect outputs, or the next `submit` on \
              the same slot will back-pressure-wait on it"]
pub struct RknnInferenceHandle {
    slot_idx: usize,
}

impl RknnInferenceHandle {
    /// Which pipeline slot this handle refers to.
    pub fn slot_idx(&self) -> usize {
        self.slot_idx
    }
}

/// Per-slot mutable state: the pre-allocated RknnMem for every graph
/// input and output. Held behind a single `RwLock` so `submit` can take
/// a write lock for the input memcpy and `recover_failed` can take the
/// same lock to rebuild everything after an `rknn_init` re-run.
struct SlotMem {
    /// One entry per graph input, keyed by ONNX tensor name.
    inputs: Vec<(String, RknnMem)>,
    /// Output `RknnMem` are held only to keep them alive; RKNN writes
    /// into them via the binding established at slot-construction time.
    /// Reads go through `RknnBackend::wait` → `rknn_outputs_get` which
    /// honours the pre-bound output memory.
    _outputs: Vec<RknnMem>,
}

/// One pipeline slot: a single NPU core with its own `RknnBackend`,
/// pre-allocated + pre-bound `RknnMem` for every graph input and
/// output, a slot-local mutex tracking the in-flight frame, and a
/// fail-streak counter for auto-recovery.
struct PipelineSlot {
    ctx: RwLock<RknnBackend>,
    core: NpuCoreMask,
    mem: RwLock<SlotMem>,
    in_flight: Mutex<Option<AsyncFrame>>,
    /// Consecutive recoverable errors observed on this slot. Reset on
    /// any successful `submit` / `wait`. Crossing `RECOVERY_THRESHOLD`
    /// triggers an automatic `reset` inside the operation that
    /// observed the last failure.
    fail_streak: AtomicU32,
}

/// Pool of per-core pipelined RKNN contexts. Slot count equals
/// `cores.len()`. On RK3588 pass `[Core0, Core1, Core2]` for
/// triple-buffered throughput; on RV1106 or other single-core Rockchip
/// SoCs pass `&[Core0]` and the pool degenerates to a single-slot
/// async path (still useful: cleaner than managing `AsyncFrame`
/// tracking by hand, and recovery is automatic either way).
pub struct RknnPipelinedPool {
    slots: Vec<PipelineSlot>,
    next: AtomicUsize,
    /// Retained so `recover_failed` can `rknn_init` a fresh context
    /// without the caller having to hold the bytes. Behind a
    /// `RwLock<Arc<...>>` so [`reload`] can hot-swap the model
    /// in-flight: writers grab the lock briefly to install fresh
    /// bytes, readers (every recovery / reload pass) clone the `Arc`
    /// cheaply.
    model_data: RwLock<Arc<Vec<u8>>>,
}

impl RknnPipelinedPool {
    /// Construct a pool with one slot per core mask. Each slot loads a
    /// fresh `RknnBackend` with `RKNN_FLAG_ASYNC_MASK`, pins it to the
    /// given core, and allocates + binds an `RknnMem` for every graph
    /// input and output. All allocation and binding happens up front;
    /// `submit` is hot-path memcpy + `rknn_run` (async) only.
    pub fn new(model_data: &[u8], cores: &[NpuCoreMask]) -> Result<Self, KernelError> {
        if cores.is_empty() {
            return Err(KernelError::Rknn {
                message: "RknnPipelinedPool: cores list must be non-empty".into(),
            });
        }

        let retained_model = Arc::new(model_data.to_vec());
        let mut slots = Vec::with_capacity(cores.len());
        for &core in cores {
            let (ctx, mem) = Self::build_slot_state(&retained_model, core)?;
            slots.push(PipelineSlot {
                ctx: RwLock::new(ctx),
                core,
                mem: RwLock::new(mem),
                in_flight: Mutex::new(None),
                fail_streak: AtomicU32::new(0),
            });
        }

        Ok(Self {
            slots,
            next: AtomicUsize::new(0),
            model_data: RwLock::new(retained_model),
        })
    }

    /// Build a fresh `(RknnBackend, SlotMem)` pair: load the model with
    /// the async flag, pin the core, allocate + bind input/output
    /// `RknnMem`. Used by `new` (per-slot construction) and by
    /// `recover_failed` (per-slot rebuild after a fault).
    fn build_slot_state(
        model_data: &[u8],
        core: NpuCoreMask,
    ) -> Result<(RknnBackend, SlotMem), KernelError> {
        let ctx = RknnBackend::load_with_flags(model_data, RKNN_FLAG_ASYNC_MASK)?;
        ctx.set_core_mask(core)?;

        // Snapshot names + sizes so the borrow of `input_attrs()`
        // doesn't alias `alloc_mem`/`bind_*`.
        let input_specs: Vec<(String, usize)> = ctx
            .input_attrs()
            .iter()
            .map(|a| (a.name_str().to_string(), a.size as usize))
            .collect();
        let output_specs: Vec<(String, usize)> = ctx
            .output_attrs()
            .iter()
            .map(|a| (a.name_str().to_string(), a.size as usize))
            .collect();

        let mut inputs = Vec::with_capacity(input_specs.len());
        for (name, size) in &input_specs {
            let m = ctx.alloc_mem(*size)?;
            ctx.bind_input_by_name(&m, name)?;
            inputs.push((name.clone(), m));
        }

        let mut outputs = Vec::with_capacity(output_specs.len());
        for (name, size) in &output_specs {
            let m = ctx.alloc_mem(*size)?;
            ctx.bind_output_by_name(&m, name)?;
            outputs.push(m);
        }

        Ok((
            ctx,
            SlotMem {
                inputs,
                _outputs: outputs,
            },
        ))
    }

    /// Number of pipeline slots (= NPU cores the pool was built for).
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Alias for `slot_count` — matches the `ContextPool::size` name.
    pub fn size(&self) -> usize {
        self.slots.len()
    }

    /// Core masks of each slot, in slot-index order.
    pub fn cores(&self) -> Vec<NpuCoreMask> {
        self.slots.iter().map(|s| s.core).collect()
    }

    /// Current consecutive-failure counter for a slot. Useful for
    /// supervisor logic that wants to pre-emptively recover before
    /// `RECOVERY_THRESHOLD` is crossed.
    pub fn fail_streak(&self, slot_idx: usize) -> u32 {
        self.slots
            .get(slot_idx)
            .map(|s| s.fail_streak.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    /// Manually rebuild a slot from scratch: destroy its NPU context,
    /// re-`rknn_init`, re-pin the core, re-allocate + re-bind all
    /// input/output `RknnMem`. Any in-flight frame for this slot is
    /// dropped (its handle will fail on `wait`).
    ///
    /// Acquires write locks on the slot's `ctx` + `mem`; concurrent
    /// `submit` / `wait` on this slot block briefly.
    pub fn recover_failed(&self, slot_idx: usize) -> Result<(), KernelError> {
        let slot = self.slots.get(slot_idx).ok_or_else(|| KernelError::Rknn {
            message: format!(
                "recover_failed: slot {slot_idx} out of range (pool size {})",
                self.slots.len()
            ),
        })?;
        let mut ctx = slot.ctx.write().map_err(|_| KernelError::Rknn {
            message: format!("slot {slot_idx} ctx lock poisoned"),
        })?;
        let mut mem = slot.mem.write().map_err(|_| KernelError::Rknn {
            message: format!("slot {slot_idx} mem lock poisoned"),
        })?;

        // Destroy old memory first — it's tied to the to-be-destroyed
        // context. Replacing the Vecs before `reset_with_flags` avoids
        // an aliased-FFI state where `reset` destroys the ctx but the
        // RknnMem Drops still reference it.
        mem.inputs.clear();
        mem._outputs.clear();

        let model_bytes = self
            .model_data
            .read()
            .map_err(|_| KernelError::Rknn {
                message: "model_data lock poisoned".into(),
            })?
            .clone();
        ctx.reset_with_flags(&model_bytes, RKNN_FLAG_ASYNC_MASK)?;
        ctx.set_core_mask(slot.core)?;

        // Rebuild input/output memory + bindings via the shared helper.
        // We can't call `build_slot_state` directly (it constructs a
        // fresh ctx); repeat its bind loop against the existing `ctx`.
        let input_specs: Vec<(String, usize)> = ctx
            .input_attrs()
            .iter()
            .map(|a| (a.name_str().to_string(), a.size as usize))
            .collect();
        let output_specs: Vec<(String, usize)> = ctx
            .output_attrs()
            .iter()
            .map(|a| (a.name_str().to_string(), a.size as usize))
            .collect();
        for (name, size) in &input_specs {
            let m = ctx.alloc_mem(*size)?;
            ctx.bind_input_by_name(&m, name)?;
            mem.inputs.push((name.clone(), m));
        }
        for (name, size) in &output_specs {
            let m = ctx.alloc_mem(*size)?;
            ctx.bind_output_by_name(&m, name)?;
            mem._outputs.push(m);
        }

        // Any pending handle is now invalid — drop its frame.
        if let Ok(mut in_flight) = slot.in_flight.lock() {
            *in_flight = None;
        }

        slot.fail_streak.store(0, Ordering::Relaxed);
        Ok(())
    }

    /// Hot-swap the underlying model. Recompiles every slot against
    /// `new_model_data`, preserving each slot's NPU-core pinning + the
    /// pool's slot count. In-flight handles are invalidated (their
    /// `wait` will error).
    ///
    /// Acquires write locks on every slot's `ctx` + `mem` plus the
    /// pool's `model_data` lock — concurrent `submit` / `wait` will
    /// block until reload finishes. Reload is **not** atomic across
    /// slots: if it fails partway through (e.g. SDK refuses the new
    /// model on slot 2 of 3), slots 0–1 are already running the new
    /// model and slots 2+ keep the old one. The caller should treat
    /// reload errors as a "retry or fail-loud" signal — not graceful.
    ///
    /// Use case: A/B-testing model variants in flight, dropping in a
    /// freshly-quantized `.rknn` after a calibration pass without
    /// stopping the FPV loop.
    pub fn reload(&self, new_model_data: &[u8]) -> Result<(), KernelError> {
        // Install the new bytes first so any concurrent `recover_failed`
        // mid-reload uses the fresh model.
        {
            let mut slot = self.model_data.write().map_err(|_| KernelError::Rknn {
                message: "model_data lock poisoned".into(),
            })?;
            *slot = Arc::new(new_model_data.to_vec());
        }
        // Walk every slot and recover it. Failures are best-effort:
        // we collect the first error but keep going so as many slots
        // as possible end up on the new model.
        let mut first_err: Option<KernelError> = None;
        for idx in 0..self.slots.len() {
            if let Err(e) = self.recover_failed(idx) {
                eprintln!("[yscv-rknn] reload: slot {idx} failed to swap to new model: {e}");
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
        match first_err {
            None => Ok(()),
            Some(e) => Err(e),
        }
    }

    /// Internal: classify an error, optionally auto-recover, and bubble
    /// the error unchanged. Called from `submit` and `wait` on any
    /// failure path. Non-recoverable errors are returned immediately
    /// with fail_streak reset (they're caller bugs, not NPU faults).
    fn handle_error(&self, slot_idx: usize, err: &KernelError) {
        let recoverable = match err {
            KernelError::Rknn { message } => is_recoverable_rknn_error(message),
            _ => false,
        };
        let slot = &self.slots[slot_idx];
        if !recoverable {
            slot.fail_streak.store(0, Ordering::Relaxed);
            return;
        }
        let streak = slot.fail_streak.fetch_add(1, Ordering::Relaxed) + 1;
        if streak >= RECOVERY_THRESHOLD {
            // Best-effort recovery. We swallow errors from
            // `recover_failed` here: the caller is about to see the
            // original error anyway, and logging the recovery failure
            // in `eprintln!` preserves visibility without hiding the
            // primary fault.
            if let Err(recovery_err) = self.recover_failed(slot_idx) {
                eprintln!(
                    "[yscv-rknn] slot {slot_idx} recovery after {streak}-streak failed: \
                     {recovery_err} (original: {err})"
                );
            } else {
                eprintln!(
                    "[yscv-rknn] slot {slot_idx} auto-recovered after {streak}-streak \
                     (triggering error: {err})"
                );
            }
        }
    }

    /// Submit a frame to the next slot round-robin. Non-blocking return:
    /// NPU work runs in the background until `wait` is called.
    ///
    /// `inputs` is a slice of `(name, bytes)` pairs — one entry per
    /// graph input. Lookup is by name; the byte count must exactly
    /// match the model's declared input size (`input_attrs[i].size`).
    ///
    /// If the chosen slot still has a pending `AsyncFrame` from a prior
    /// submission (caller's outstanding handle count exceeds the slot
    /// count), the previous frame is waited on and its outputs
    /// discarded before the slot's buffers are reused.
    pub fn submit(&self, inputs: &[(&str, &[u8])]) -> Result<RknnInferenceHandle, KernelError> {
        let slot_idx = self.next.fetch_add(1, Ordering::Relaxed) % self.slots.len();
        match self.submit_inner(slot_idx, inputs) {
            Ok(h) => {
                self.slots[slot_idx].fail_streak.store(0, Ordering::Relaxed);
                Ok(h)
            }
            Err(e) => {
                self.handle_error(slot_idx, &e);
                Err(e)
            }
        }
    }

    fn submit_inner(
        &self,
        slot_idx: usize,
        inputs: &[(&str, &[u8])],
    ) -> Result<RknnInferenceHandle, KernelError> {
        let slot = &self.slots[slot_idx];

        // Drain any prior in-flight frame on this slot before reusing
        // its input buffers.
        let prev = slot
            .in_flight
            .lock()
            .map_err(|_| KernelError::Rknn {
                message: format!("slot {slot_idx} in-flight lock poisoned"),
            })?
            .take();
        if let Some(frame) = prev {
            let ctx = slot.ctx.read().map_err(|_| KernelError::Rknn {
                message: format!("slot {slot_idx} ctx lock poisoned"),
            })?;
            let _ = ctx.wait(frame)?;
        }

        // Copy fresh inputs into the slot's pre-bound memories. A write
        // lock on `mem` serialises concurrent submits to the same slot
        // (should be rare — the `next` ring normally rotates first).
        {
            let mut mem = slot.mem.write().map_err(|_| KernelError::Rknn {
                message: format!("slot {slot_idx} mem lock poisoned"),
            })?;
            for &(name, data) in inputs {
                let (_, mem_buf) =
                    mem.inputs
                        .iter_mut()
                        .find(|(n, _)| n == name)
                        .ok_or_else(|| KernelError::Rknn {
                            message: format!("input '{name}' not declared by slot"),
                        })?;
                let cap = mem_buf.size() as usize;
                // Require EXACT match between caller-supplied bytes and
                // the NPU input buffer size. Short writes would leave
                // the previous frame's tail bytes in place — silent
                // garbage. Over-size is obviously illegal.
                if data.len() != cap {
                    return Err(KernelError::Rknn {
                        message: format!(
                            "input '{name}': {} bytes but the model expects exactly {} \
                             (check your pre-processing matches the .rknn's input shape/dtype)",
                            data.len(),
                            cap
                        ),
                    });
                }
                mem_buf.as_mut_slice().copy_from_slice(data);
                mem_buf.sync_to_device()?;
            }
        }

        // Fire the NPU.
        let frame = {
            let ctx = slot.ctx.read().map_err(|_| KernelError::Rknn {
                message: format!("slot {slot_idx} ctx lock poisoned"),
            })?;
            ctx.run_async_bound(slot_idx as u64, 0)?
        };

        *slot.in_flight.lock().map_err(|_| KernelError::Rknn {
            message: format!("slot {slot_idx} in-flight lock poisoned"),
        })? = Some(frame);

        Ok(RknnInferenceHandle { slot_idx })
    }

    /// Block until `handle`'s NPU work completes, then collect outputs.
    /// Returns dequantized `f32` tensors — one per graph output, in the
    /// model's declared output order.
    ///
    /// Errors if the slot has no pending frame (handle was already
    /// waited on, or slot was reused by a later `submit`).
    pub fn wait(&self, handle: RknnInferenceHandle) -> Result<Vec<Tensor>, KernelError> {
        let slot_idx = handle.slot_idx;
        match self.wait_inner(slot_idx) {
            Ok(v) => {
                self.slots[slot_idx].fail_streak.store(0, Ordering::Relaxed);
                Ok(v)
            }
            Err(e) => {
                self.handle_error(slot_idx, &e);
                Err(e)
            }
        }
    }

    fn wait_inner(&self, slot_idx: usize) -> Result<Vec<Tensor>, KernelError> {
        let slot = &self.slots[slot_idx];
        let frame = slot
            .in_flight
            .lock()
            .map_err(|_| KernelError::Rknn {
                message: format!("slot {slot_idx} in-flight lock poisoned"),
            })?
            .take()
            .ok_or_else(|| KernelError::Rknn {
                message: format!(
                    "slot {slot_idx} has no pending frame — handle already waited on \
                     or slot was reused by a later submit"
                ),
            })?;
        let ctx = slot.ctx.read().map_err(|_| KernelError::Rknn {
            message: format!("slot {slot_idx} ctx lock poisoned"),
        })?;
        ctx.wait(frame)
    }

    /// Convenience wrapper: `submit` + `wait` back-to-back. Useful for
    /// sync benchmarks or caller code that doesn't want to manage
    /// pipelining explicitly. Does not exploit multi-core parallelism
    /// across consecutive calls — use explicit `submit` / `wait` for
    /// that.
    pub fn run(&self, inputs: &[(&str, &[u8])]) -> Result<Vec<Tensor>, KernelError> {
        let h = self.submit(inputs)?;
        self.wait(h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_fails_on_empty_cores() {
        match RknnPipelinedPool::new(b"fake model bytes", &[]) {
            Ok(_) => panic!("pool should reject empty core list"),
            Err(KernelError::Rknn { message }) => {
                assert!(
                    message.contains("cores list must be non-empty"),
                    "unexpected error message: {message}"
                );
            }
            Err(other) => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn pool_fails_on_non_rockchip_host() {
        // On a non-Rockchip dev host `librknnrt.so` isn't loadable;
        // `RknnBackend::load_with_flags` propagates the dlopen failure.
        match RknnPipelinedPool::new(b"fake", &[NpuCoreMask::Core0]) {
            Ok(_) => panic!("pool should fail to load on non-Rockchip hosts"),
            Err(KernelError::Rknn { .. }) => {}
            Err(other) => panic!("expected Rknn error, got {other:?}"),
        }
    }

    #[test]
    fn handle_is_must_use() {
        let h = RknnInferenceHandle { slot_idx: 0 };
        assert_eq!(h.slot_idx(), 0);
    }

    #[test]
    fn is_recoverable_rknn_error_classifies_known_codes() {
        // Sanity check on the string contract used by `handle_error`.
        assert!(is_recoverable_rknn_error(
            "rknn_wait failed: TIMEOUT (-14) — frame 0 after 2000ms"
        ));
        assert!(is_recoverable_rknn_error(
            "rknn_run failed: CTX_INVALID (-5)"
        ));
        assert!(is_recoverable_rknn_error(
            "rknn_run failed: DEVICE_UNAVAILABLE (-3)"
        ));
        assert!(!is_recoverable_rknn_error(
            "rknn_inputs_set failed: INPUT_INVALID (-8)"
        ));
        assert!(!is_recoverable_rknn_error(
            "rknn_init failed: MODEL_INVALID (-6)"
        ));
    }
}
