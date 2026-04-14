//! Lock-free SPSC ring buffer for real-time frame processing.
//!
//! Pre-allocates all memory at startup — zero allocations per frame.
//! Three-stage pipeline:
//! - Stage 1 (capture thread): writes raw pixels into next free slot
//! - Stage 2 (inference thread): converts/processes, writes detections
//! - Stage 3 (output thread): reads results, overlays, encodes

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};

/// Axis-aligned bounding box in image coordinates.
///
/// Layout-compatible with `yscv_detect::BoundingBox` so the two can be
/// transmuted at zero cost at crate boundaries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PipelineBBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

/// Lightweight detection result stored in the pipeline.
///
/// Layout-compatible with `yscv_detect::Detection`. Defined locally to
/// avoid a circular dependency (`yscv-detect` already depends on
/// `yscv-video`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PipelineDetection {
    pub bbox: PipelineBBox,
    pub score: f32,
    pub class_id: usize,
}

/// Frame slot states.
const SLOT_FREE: u8 = 0;
const SLOT_WRITING: u8 = 1;
const SLOT_CAPTURED: u8 = 2;
const SLOT_PROCESSING: u8 = 3;
const SLOT_READY: u8 = 4;

/// Mutable payload of a frame slot, accessed through `UnsafeCell` interior
/// mutability. Only one thread writes at a time, enforced by the state machine.
struct SlotPayload {
    data: Vec<u8>,
    width: u32,
    height: u32,
    pixel_format: u8,
    timestamp_us: u64,
    detections: Vec<PipelineDetection>,
}

/// A pre-allocated frame slot in the ring buffer.
///
/// The payload is wrapped in `UnsafeCell` because different pipeline stages
/// need mutable access, but only one stage owns a slot at any given time
/// (enforced by the atomic state machine).
pub struct FrameSlot {
    payload: UnsafeCell<SlotPayload>,
    state: AtomicU8,
}

// SAFETY: Access to the inner `UnsafeCell<SlotPayload>` is serialised by the
// atomic state machine — only one thread can hold a reference to the payload at
// a time (the state transitions guarantee exclusive ownership at each stage).
unsafe impl Sync for FrameSlot {}

/// Shared accessor for reading fields of an acquired slot.
pub struct SlotRef<'a> {
    payload: &'a SlotPayload,
}

impl SlotRef<'_> {
    /// Raw pixel data.
    pub fn data(&self) -> &[u8] {
        &self.payload.data
    }

    /// Frame width in pixels.
    pub fn width(&self) -> u32 {
        self.payload.width
    }

    /// Frame height in pixels.
    pub fn height(&self) -> u32 {
        self.payload.height
    }

    /// Pixel format tag (YUV420 = 0, NV12 = 1, RGB8 = 2).
    pub fn pixel_format(&self) -> u8 {
        self.payload.pixel_format
    }

    /// Capture timestamp in microseconds.
    pub fn timestamp_us(&self) -> u64 {
        self.payload.timestamp_us
    }

    /// Detection results written by the inference stage.
    pub fn detections(&self) -> &[PipelineDetection] {
        &self.payload.detections
    }
}

/// Mutable accessor for writing into an acquired slot.
pub struct SlotMut<'a> {
    payload: &'a mut SlotPayload,
}

impl SlotMut<'_> {
    /// Mutable reference to the raw pixel buffer.
    ///
    /// The buffer is pre-allocated to `max_frame_bytes`; use
    /// `data_mut()[..len].copy_from_slice(...)` to fill it.
    pub fn data_mut(&mut self) -> &mut Vec<u8> {
        &mut self.payload.data
    }

    /// Read-only access to the pixel data.
    pub fn data(&self) -> &[u8] {
        &self.payload.data
    }

    /// Set the frame width.
    pub fn set_width(&mut self, w: u32) {
        self.payload.width = w;
    }

    /// Set the frame height.
    pub fn set_height(&mut self, h: u32) {
        self.payload.height = h;
    }

    /// Set the pixel format tag.
    pub fn set_pixel_format(&mut self, fmt: u8) {
        self.payload.pixel_format = fmt;
    }

    /// Set the capture timestamp (microseconds).
    pub fn set_timestamp_us(&mut self, ts: u64) {
        self.payload.timestamp_us = ts;
    }

    /// Mutable reference to the detection results vector.
    pub fn detections_mut(&mut self) -> &mut Vec<PipelineDetection> {
        &mut self.payload.detections
    }

    /// Read-only access to detection results.
    pub fn detections(&self) -> &[PipelineDetection] {
        &self.payload.detections
    }

    /// Frame width.
    pub fn width(&self) -> u32 {
        self.payload.width
    }

    /// Frame height.
    pub fn height(&self) -> u32 {
        self.payload.height
    }

    /// Pixel format tag.
    pub fn pixel_format(&self) -> u8 {
        self.payload.pixel_format
    }

    /// Capture timestamp in microseconds.
    pub fn timestamp_us(&self) -> u64 {
        self.payload.timestamp_us
    }
}

/// Lock-free ring buffer for pipelined frame processing.
///
/// Supports three concurrent stages with non-blocking acquire/commit
/// operations. All memory is pre-allocated; the steady-state path performs
/// zero heap allocations.
pub struct FramePipeline {
    slots: Vec<FrameSlot>,
    capacity: usize,
    /// Next slot index for the capture (write) stage.
    write_pos: AtomicUsize,
    /// Next slot index for the inference (process) stage.
    read_pos: AtomicUsize,
    /// Next slot index for the output stage.
    output_pos: AtomicUsize,
}

// SAFETY: The interior `UnsafeCell` payloads inside each `FrameSlot` are
// protected by the per-slot atomic state machine. At most one thread accesses
// a given slot's payload at any time, so sharing `&FramePipeline` across
// threads is safe.
unsafe impl Sync for FramePipeline {}

impl FramePipeline {
    /// Create a pipeline with `capacity` pre-allocated slots, each holding
    /// `max_frame_bytes` of pixel data.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is zero.
    pub fn new(capacity: usize, max_frame_bytes: usize) -> Self {
        assert!(capacity > 0, "FramePipeline capacity must be > 0");

        let slots: Vec<FrameSlot> = (0..capacity)
            .map(|_| FrameSlot {
                payload: UnsafeCell::new(SlotPayload {
                    data: vec![0u8; max_frame_bytes],
                    width: 0,
                    height: 0,
                    pixel_format: 0,
                    timestamp_us: 0,
                    detections: Vec::new(),
                }),
                state: AtomicU8::new(SLOT_FREE),
            })
            .collect();

        Self {
            slots,
            capacity,
            write_pos: AtomicUsize::new(0),
            read_pos: AtomicUsize::new(0),
            output_pos: AtomicUsize::new(0),
        }
    }

    /// Try to acquire the next free slot for writing (non-blocking).
    ///
    /// Returns `Some(SlotMut)` if the slot at `write_pos` is `FREE`,
    /// transitions it to `WRITING`, and hands out an exclusive mutable handle.
    /// Returns `None` if the slot is still occupied by a downstream stage.
    pub fn try_acquire_write(&self) -> Option<SlotMut<'_>> {
        let idx = self.write_pos.load(Ordering::Relaxed) % self.capacity;
        let slot = &self.slots[idx];

        if slot
            .state
            .compare_exchange(
                SLOT_FREE,
                SLOT_WRITING,
                Ordering::Acquire,
                Ordering::Relaxed,
            )
            .is_ok()
        {
            // SAFETY: We just transitioned state from FREE -> WRITING, so no
            // other thread can access this slot's payload until we commit.
            let payload = unsafe { &mut *slot.payload.get() };
            Some(SlotMut { payload })
        } else {
            None
        }
    }

    /// Mark the current write slot as `CAPTURED` and advance `write_pos`.
    ///
    /// Must be called after a successful `try_acquire_write` + filling the slot.
    pub fn commit_write(&self) {
        let idx = self.write_pos.load(Ordering::Relaxed) % self.capacity;
        self.slots[idx]
            .state
            .store(SLOT_CAPTURED, Ordering::Release);
        self.write_pos.fetch_add(1, Ordering::Relaxed);
    }

    /// Roll back a write: return the current write slot to `FREE` without
    /// advancing `write_pos`. Used when the capture callback signals end of
    /// stream.
    pub fn rollback_write(&self) {
        let idx = self.write_pos.load(Ordering::Relaxed) % self.capacity;
        self.slots[idx].state.store(SLOT_FREE, Ordering::Release);
    }

    /// Try to acquire the next captured slot for processing (non-blocking).
    ///
    /// Returns `Some(SlotMut)` if the slot at `read_pos` is `CAPTURED`,
    /// transitions it to `PROCESSING`. Returns `None` if no captured frame
    /// is available.
    pub fn try_acquire_read(&self) -> Option<SlotMut<'_>> {
        let idx = self.read_pos.load(Ordering::Relaxed) % self.capacity;
        let slot = &self.slots[idx];

        if slot
            .state
            .compare_exchange(
                SLOT_CAPTURED,
                SLOT_PROCESSING,
                Ordering::Acquire,
                Ordering::Relaxed,
            )
            .is_ok()
        {
            // SAFETY: We just transitioned state from CAPTURED -> PROCESSING;
            // the write stage has already released the slot.
            let payload = unsafe { &mut *slot.payload.get() };
            Some(SlotMut { payload })
        } else {
            None
        }
    }

    /// Mark the current read slot as `READY` and advance `read_pos`.
    pub fn commit_read(&self) {
        let idx = self.read_pos.load(Ordering::Relaxed) % self.capacity;
        self.slots[idx].state.store(SLOT_READY, Ordering::Release);
        self.read_pos.fetch_add(1, Ordering::Relaxed);
    }

    /// Try to acquire the next ready slot for output (non-blocking).
    ///
    /// Returns a shared `SlotRef` -- the output stage only reads.
    pub fn try_acquire_output(&self) -> Option<SlotRef<'_>> {
        let idx = self.output_pos.load(Ordering::Relaxed) % self.capacity;
        let slot = &self.slots[idx];

        if slot.state.load(Ordering::Acquire) == SLOT_READY {
            // SAFETY: The slot is in READY state -- the process stage has
            // finished writing. The output stage only reads. We do not
            // transition to a separate "outputting" state because the output
            // stage is single-consumer; it will call `commit_output` before
            // advancing.
            let payload = unsafe { &*slot.payload.get() };
            Some(SlotRef { payload })
        } else {
            None
        }
    }

    /// Mark the current output slot as `FREE` (recycle) and advance
    /// `output_pos`.
    pub fn commit_output(&self) {
        let idx = self.output_pos.load(Ordering::Relaxed) % self.capacity;
        self.slots[idx].state.store(SLOT_FREE, Ordering::Release);
        self.output_pos.fetch_add(1, Ordering::Relaxed);
    }

    /// Number of slots in the ring.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

/// Run a three-stage pipeline: capture -> process -> output.
///
/// Each stage runs on its own OS thread inside a `std::thread::scope` block.
/// The pipeline stops when `max_frames` frames have been output **or** the
/// capture callback returns `false` (no more input).
///
/// Stages spin with `std::hint::spin_loop()` when no slot is available,
/// trading CPU for latency (no mutex, no condvar).
pub fn run_pipeline<C, P, O>(
    pipeline: &FramePipeline,
    capture: C,
    process: P,
    output: O,
    max_frames: usize,
) -> PipelineStats
where
    C: FnMut(&mut SlotMut<'_>) -> bool + Send,
    P: FnMut(&mut SlotMut<'_>) + Send,
    O: FnMut(&SlotRef<'_>) + Send,
{
    // `capture_done` is set when the capture stage exits (either after
    // producing `max_frames` or when the callback returns false).
    let capture_done = AtomicBool::new(false);
    // `captured_count` is the actual number of frames committed by capture.
    let captured_count = AtomicUsize::new(0);
    let processed_count = AtomicUsize::new(0);
    let outputted_count = AtomicUsize::new(0);
    // Panics caught per stage — counters exposed via PipelineStats.
    let panic_capture = AtomicUsize::new(0);
    let panic_process = AtomicUsize::new(0);
    let panic_output = AtomicUsize::new(0);

    // Wrap caller closures in Mutexes so we can call them from spawned
    // scoped threads while keeping each in `AssertUnwindSafe`. The Mutex
    // is uncontested (each stage runs on its own thread).
    let capture = std::sync::Mutex::new(capture);
    let process = std::sync::Mutex::new(process);
    let output = std::sync::Mutex::new(output);

    std::thread::scope(|s| {
        // -- Stage 1: capture ---------------------------------------------------
        s.spawn(|| {
            let mut produced = 0usize;
            while produced < max_frames {
                if let Some(mut slot) = pipeline.try_acquire_write() {
                    // Catch panic inside the user closure: a bad driver
                    // response shouldn't kill the whole pipeline.
                    let keep_going = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let mut cap = capture.lock().unwrap_or_else(|e| e.into_inner());
                        (*cap)(&mut slot)
                    }));
                    match keep_going {
                        Ok(true) => {
                            pipeline.commit_write();
                            produced += 1;
                            captured_count.store(produced, Ordering::Release);
                        }
                        Ok(false) => {
                            pipeline.rollback_write();
                            break;
                        }
                        Err(payload) => {
                            let msg = panic_message(&payload);
                            eprintln!("[yscv-video] capture stage panicked: {msg} — stopping");
                            panic_capture.fetch_add(1, Ordering::Relaxed);
                            pipeline.rollback_write();
                            break; // capture panic is terminal for this source
                        }
                    }
                } else {
                    std::hint::spin_loop();
                }
            }
            capture_done.store(true, Ordering::Release);
        });

        // -- Stage 2: process ---------------------------------------------------
        s.spawn(|| {
            let mut done = 0usize;
            loop {
                if let Some(mut slot) = pipeline.try_acquire_read() {
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let mut proc = process.lock().unwrap_or_else(|e| e.into_inner());
                        (*proc)(&mut slot);
                    }));
                    if let Err(payload) = result {
                        let msg = panic_message(&payload);
                        eprintln!(
                            "[yscv-video] process stage panicked on frame {done}: {msg} — continuing with empty detections"
                        );
                        panic_process.fetch_add(1, Ordering::Relaxed);
                        // Clear partial state so downstream sees a clean slot.
                        slot.detections_mut().clear();
                    }
                    // Always commit so the slot flows downstream — dropping
                    // would deadlock the ring.
                    pipeline.commit_read();
                    done += 1;
                    processed_count.store(done, Ordering::Release);
                } else if capture_done.load(Ordering::Acquire)
                    && done >= captured_count.load(Ordering::Acquire)
                {
                    // Capture has finished and we've processed everything.
                    return;
                } else {
                    std::hint::spin_loop();
                }
            }
        });

        // -- Stage 3: output ----------------------------------------------------
        s.spawn(|| {
            let mut done = 0usize;
            loop {
                if let Some(slot_ref) = pipeline.try_acquire_output() {
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let mut out = output.lock().unwrap_or_else(|e| e.into_inner());
                        (*out)(&slot_ref);
                    }));
                    if let Err(payload) = result {
                        let msg = panic_message(&payload);
                        eprintln!(
                            "[yscv-video] output stage panicked on frame {done}: {msg} — frame dropped"
                        );
                        panic_output.fetch_add(1, Ordering::Relaxed);
                    }
                    // Always commit so the slot is recycled back to FREE.
                    pipeline.commit_output();
                    done += 1;
                    outputted_count.store(done, Ordering::Release);
                } else if capture_done.load(Ordering::Acquire)
                    && done >= processed_count.load(Ordering::Acquire)
                    && processed_count.load(Ordering::Acquire)
                        >= captured_count.load(Ordering::Acquire)
                {
                    // All stages are done.
                    return;
                } else {
                    std::hint::spin_loop();
                }
            }
        });
    });

    PipelineStats {
        captured: captured_count.load(Ordering::Relaxed),
        processed: processed_count.load(Ordering::Relaxed),
        outputted: outputted_count.load(Ordering::Relaxed),
        panics_capture: panic_capture.load(Ordering::Relaxed),
        panics_process: panic_process.load(Ordering::Relaxed),
        panics_output: panic_output.load(Ordering::Relaxed),
        capture_latency: crate::latency_histogram::LatencyHistogram::new(),
        process_latency: crate::latency_histogram::LatencyHistogram::new(),
        output_latency: crate::latency_histogram::LatencyHistogram::new(),
    }
}

/// Counters + latency histograms returned from [`run_pipeline`] for
/// observability and health-check integration.
///
/// `LatencyHistogram` fields are populated only when
/// [`run_pipeline_with_latency`] is used; the plain [`run_pipeline`]
/// leaves them empty. Quantile snapshots are available via
/// [`PipelineStats::latency_snapshot`].
#[derive(Debug, Default)]
pub struct PipelineStats {
    /// Frames successfully produced by the capture stage.
    pub captured: usize,
    /// Frames processed (including those where the closure panicked —
    /// those are counted but yield empty detections).
    pub processed: usize,
    /// Frames emitted by the output stage.
    pub outputted: usize,
    /// Number of times the capture closure panicked (terminal).
    pub panics_capture: usize,
    /// Number of times the process closure panicked (recovered).
    pub panics_process: usize,
    /// Number of times the output closure panicked (frame dropped).
    pub panics_output: usize,
    /// Per-stage latency distributions. Empty unless
    /// [`run_pipeline_with_latency`] was used.
    pub capture_latency: crate::latency_histogram::LatencyHistogram,
    pub process_latency: crate::latency_histogram::LatencyHistogram,
    pub output_latency: crate::latency_histogram::LatencyHistogram,
}

/// Snapshot of all three per-stage quantile distributions — plain
/// `Copy` data for JSON / logging.
#[derive(Debug, Clone, Copy, Default)]
pub struct PipelineLatencySnapshot {
    pub capture: crate::latency_histogram::LatencyQuantiles,
    pub process: crate::latency_histogram::LatencyQuantiles,
    pub output: crate::latency_histogram::LatencyQuantiles,
}

impl PipelineStats {
    /// Copy-constructible view of per-stage quantiles. Cheap
    /// (~3 sorted 2048-sample arrays under the hood) but not free —
    /// call once per report interval, not per frame.
    pub fn latency_snapshot(&self) -> PipelineLatencySnapshot {
        PipelineLatencySnapshot {
            capture: self.capture_latency.snapshot(),
            process: self.process_latency.snapshot(),
            output: self.output_latency.snapshot(),
        }
    }
}

/// Best-effort extraction of a panic message.
fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> &str {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        s
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.as_str()
    } else {
        "<non-string panic payload>"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Push 10 frames through a 4-slot pipeline and verify all are received.
    #[test]
    fn pipeline_basic() {
        let pipeline = FramePipeline::new(4, 16);
        let total = 10usize;
        let capture_idx = AtomicUsize::new(0);
        let output_count = AtomicUsize::new(0);

        run_pipeline(
            &pipeline,
            |slot: &mut SlotMut<'_>| {
                let i = capture_idx.fetch_add(1, Ordering::Relaxed);
                if i >= total {
                    return false;
                }
                slot.set_timestamp_us(i as u64);
                slot.data_mut()[0] = i as u8;
                true
            },
            |slot: &mut SlotMut<'_>| {
                // Processing: double the first data byte.
                let v = slot.data()[0];
                slot.data_mut()[0] = v.wrapping_mul(2);
            },
            |slot: &SlotRef<'_>| {
                let ts = slot.timestamp_us() as u8;
                let processed_val = slot.data()[0];
                assert_eq!(processed_val, ts.wrapping_mul(2));
                output_count.fetch_add(1, Ordering::Relaxed);
            },
            total,
        );

        assert_eq!(output_count.load(Ordering::Relaxed), total);
    }

    /// Verify that no heap allocations occur during steady-state operation.
    ///
    /// We check that the `Vec` capacities inside slots remain unchanged after
    /// the pipeline runs -- no resizes or new allocations.
    #[test]
    fn pipeline_zero_alloc() {
        let max_bytes = 128;
        let pipeline = FramePipeline::new(4, max_bytes);

        // Record initial capacities.
        let initial_data_caps: Vec<usize> = pipeline
            .slots
            .iter()
            .map(|slot| {
                // SAFETY: No other thread is running; single-threaded test
                // setup before launching the pipeline.
                let p = unsafe { &*slot.payload.get() };
                p.data.capacity()
            })
            .collect();

        let capture_idx = AtomicUsize::new(0);
        let total = 20usize;

        run_pipeline(
            &pipeline,
            |slot: &mut SlotMut<'_>| {
                let i = capture_idx.fetch_add(1, Ordering::Relaxed);
                if i >= total {
                    return false;
                }
                // Write within the pre-allocated buffer -- no realloc.
                let len = slot.data().len().min(max_bytes);
                slot.data_mut()[..len]
                    .iter_mut()
                    .enumerate()
                    .for_each(|(j, b)| *b = (i + j) as u8);
                slot.set_timestamp_us(i as u64);
                true
            },
            |slot: &mut SlotMut<'_>| {
                // Touch detections vec without growing -- just clear.
                slot.detections_mut().clear();
            },
            |_slot: &SlotRef<'_>| {},
            total,
        );

        // Verify capacities are unchanged -- no reallocation happened.
        let final_data_caps: Vec<usize> = pipeline
            .slots
            .iter()
            .map(|slot| {
                // SAFETY: Pipeline has finished; single-threaded access.
                let p = unsafe { &*slot.payload.get() };
                p.data.capacity()
            })
            .collect();

        assert_eq!(
            initial_data_caps, final_data_caps,
            "data buffers must not reallocate during steady-state"
        );
    }

    /// Three real OS threads exercising the full state machine concurrently
    /// with 100 frames through an 8-slot ring.
    #[test]
    fn pipeline_concurrent() {
        let pipeline = FramePipeline::new(8, 256);
        let total = 100usize;
        let capture_idx = AtomicUsize::new(0);
        let output_sum = AtomicUsize::new(0);

        run_pipeline(
            &pipeline,
            |slot: &mut SlotMut<'_>| {
                let i = capture_idx.fetch_add(1, Ordering::Relaxed);
                if i >= total {
                    return false;
                }
                slot.set_timestamp_us(i as u64);
                slot.data_mut()[0] = (i & 0xFF) as u8;
                true
            },
            |slot: &mut SlotMut<'_>| {
                // Add 1 to the first byte as a processing marker.
                let v = slot.data()[0];
                slot.data_mut()[0] = v.wrapping_add(1);
            },
            |slot: &SlotRef<'_>| {
                output_sum.fetch_add(slot.data()[0] as usize, Ordering::Relaxed);
            },
            total,
        );

        // Each frame i contributes (i & 0xFF) + 1 to the sum.
        let expected: usize = (0..total).map(|i| (i & 0xFF) + 1).sum();
        assert_eq!(output_sum.load(Ordering::Relaxed), expected);
    }

    /// Early termination: capture returns false before max_frames.
    #[test]
    fn pipeline_early_stop() {
        let pipeline = FramePipeline::new(4, 16);
        let capture_idx = AtomicUsize::new(0);
        let output_count = AtomicUsize::new(0);
        let early_stop_at = 3usize;

        run_pipeline(
            &pipeline,
            |slot: &mut SlotMut<'_>| {
                let i = capture_idx.fetch_add(1, Ordering::Relaxed);
                if i >= early_stop_at {
                    return false;
                }
                slot.set_timestamp_us(i as u64);
                true
            },
            |_slot: &mut SlotMut<'_>| {},
            |_slot: &SlotRef<'_>| {
                output_count.fetch_add(1, Ordering::Relaxed);
            },
            1000, // large max, but capture stops at 3
        );

        assert_eq!(output_count.load(Ordering::Relaxed), early_stop_at);
    }

    /// Single slot pipeline -- extreme back-pressure.
    #[test]
    fn pipeline_single_slot() {
        let pipeline = FramePipeline::new(1, 8);
        let total = 5usize;
        let capture_idx = AtomicUsize::new(0);
        let output_count = AtomicUsize::new(0);

        run_pipeline(
            &pipeline,
            |slot: &mut SlotMut<'_>| {
                let i = capture_idx.fetch_add(1, Ordering::Relaxed);
                if i >= total {
                    return false;
                }
                slot.set_timestamp_us(i as u64);
                true
            },
            |_slot: &mut SlotMut<'_>| {},
            |_slot: &SlotRef<'_>| {
                output_count.fetch_add(1, Ordering::Relaxed);
            },
            total,
        );

        assert_eq!(output_count.load(Ordering::Relaxed), total);
    }

    /// Process stage panics on every 5th frame — pipeline must keep running,
    /// panic counter reflects the crashes, output count equals captured count.
    #[test]
    fn pipeline_survives_process_panic() {
        let pipeline = FramePipeline::new(4, 16);
        let total = 20usize;
        let capture_idx = AtomicUsize::new(0);
        let output_count = AtomicUsize::new(0);

        let stats = run_pipeline(
            &pipeline,
            |slot: &mut SlotMut<'_>| {
                let i = capture_idx.fetch_add(1, Ordering::Relaxed);
                if i >= total {
                    return false;
                }
                slot.set_timestamp_us(i as u64);
                true
            },
            |slot: &mut SlotMut<'_>| {
                let idx = slot.timestamp_us() as usize;
                if idx % 5 == 4 {
                    panic!("intentional test panic at frame {idx}");
                }
            },
            |_slot: &SlotRef<'_>| {
                output_count.fetch_add(1, Ordering::Relaxed);
            },
            total,
        );

        assert_eq!(stats.captured, total);
        assert_eq!(
            stats.processed, total,
            "all frames flow through despite panics"
        );
        assert_eq!(stats.outputted, total);
        assert_eq!(stats.panics_process, 4, "frames 4, 9, 14, 19 panic");
        assert_eq!(stats.panics_capture, 0);
        assert_eq!(stats.panics_output, 0);
        assert_eq!(output_count.load(Ordering::Relaxed), total);
    }

    /// Output stage panics occasionally — frame dropped, slot recycled, no deadlock.
    #[test]
    fn pipeline_survives_output_panic() {
        let pipeline = FramePipeline::new(4, 16);
        let total = 10usize;
        let capture_idx = AtomicUsize::new(0);

        let stats = run_pipeline(
            &pipeline,
            |slot: &mut SlotMut<'_>| {
                let i = capture_idx.fetch_add(1, Ordering::Relaxed);
                if i >= total {
                    return false;
                }
                slot.set_timestamp_us(i as u64);
                true
            },
            |_slot: &mut SlotMut<'_>| {},
            |slot: &SlotRef<'_>| {
                if slot.timestamp_us() == 3 {
                    panic!("output panic at frame 3");
                }
            },
            total,
        );

        assert_eq!(stats.captured, total);
        assert_eq!(stats.processed, total);
        assert_eq!(stats.outputted, total);
        assert_eq!(stats.panics_output, 1);
    }
}
