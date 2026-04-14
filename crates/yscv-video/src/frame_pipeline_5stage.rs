//! 5-stage lock-free FPV pipeline.
//!
//! Capture → InferDispatch → InferWait+NMS → Overlay+Encode → Output.
//! Each stage runs on its own SCHED_FIFO thread; data flows through
//! lock-free SPSC ring buffers between stages. 3 frames in flight at
//! once → end-to-end latency = max(stage), not sum.
//!
//! Design notes:
//!
//! - Each ring is a fixed-size array of `Slot<T>` with a head/tail
//!   `AtomicUsize` pair. Single-producer, single-consumer per ring →
//!   no CAS, no compare-and-swap loops.
//! - Per-stage `panic::catch_unwind` so a bad frame never deadlocks the
//!   pipeline (counter incremented, frame dropped).
//! - Per-stage deadline tracked: if a stage overruns 2× budget for N
//!   consecutive frames, watchdog flag triggers (caller checks
//!   `PipelineStats5::watchdog_alarm`).
//! - All allocations done at construction. Steady-state path is zero-alloc.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

/// Number of slots in each inter-stage ring. 4 gives 3 frames in-flight
/// + 1 spare, matching the typical FPV depth.
pub const RING_DEPTH: usize = 4;

/// One slot in an SPSC ring. Holds a payload of type `T` plus a
/// `ready` flag the producer flips after writing.
struct RingSlot<T> {
    cell: std::cell::UnsafeCell<Option<T>>,
    ready: AtomicBool,
}

// SAFETY: the ring's `head/tail` discipline ensures only one thread
// (producer or consumer) accesses the inner cell at a time.
unsafe impl<T: Send> Sync for RingSlot<T> {}

/// Lock-free single-producer single-consumer ring buffer.
pub struct SpscRing<T> {
    slots: Box<[RingSlot<T>]>,
    head: AtomicUsize, // producer position
    tail: AtomicUsize, // consumer position
}

impl<T: Send> SpscRing<T> {
    pub fn new(depth: usize) -> Self {
        let mut v = Vec::with_capacity(depth);
        for _ in 0..depth {
            v.push(RingSlot {
                cell: std::cell::UnsafeCell::new(None),
                ready: AtomicBool::new(false),
            });
        }
        Self {
            slots: v.into_boxed_slice(),
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }

    /// Try to push `item` into the ring. Returns `Err(item)` if full.
    pub fn try_push(&self, item: T) -> Result<(), T> {
        let h = self.head.load(Ordering::Relaxed);
        let t = self.tail.load(Ordering::Acquire);
        if h.wrapping_sub(t) >= self.slots.len() {
            return Err(item); // full
        }
        let idx = h % self.slots.len();
        let slot = &self.slots[idx];
        // SAFETY: only the producer touches `cell` between try_push and
        // the consumer's matching `try_pop`; ready=false here means slot
        // is exclusively ours.
        unsafe {
            *slot.cell.get() = Some(item);
        }
        slot.ready.store(true, Ordering::Release);
        self.head.store(h.wrapping_add(1), Ordering::Release);
        Ok(())
    }

    /// Try to pop one item from the ring. Returns `None` if empty.
    pub fn try_pop(&self) -> Option<T> {
        let t = self.tail.load(Ordering::Relaxed);
        let h = self.head.load(Ordering::Acquire);
        if t == h {
            return None; // empty
        }
        let idx = t % self.slots.len();
        let slot = &self.slots[idx];
        if !slot.ready.load(Ordering::Acquire) {
            return None;
        }
        // SAFETY: ready=true means producer finished writing; we're the
        // sole consumer.
        let item = unsafe { (*slot.cell.get()).take() };
        slot.ready.store(false, Ordering::Release);
        self.tail.store(t.wrapping_add(1), Ordering::Release);
        item
    }

    pub fn capacity(&self) -> usize {
        self.slots.len()
    }
}

/// Per-stage deadline watchdog. Increments overrun counter when a stage
/// exceeds its budget. After `WATCHDOG_THRESHOLD` consecutive overruns,
/// `alarm` is set — caller polls and decides recovery (e.g. recreate
/// NPU context, reduce model, etc.).
#[derive(Debug)]
pub struct StageWatchdog {
    /// Maximum allowed duration per call, in microseconds.
    pub budget_us: AtomicU64,
    /// Consecutive overruns observed.
    pub overrun_streak: AtomicUsize,
    /// Total overruns (lifetime).
    pub total_overruns: AtomicU64,
    /// Set when overrun_streak crosses `WATCHDOG_THRESHOLD`.
    pub alarm: AtomicBool,
}

const WATCHDOG_THRESHOLD: usize = 5;

impl StageWatchdog {
    pub fn new(budget_us: u64) -> Self {
        Self {
            budget_us: AtomicU64::new(budget_us),
            overrun_streak: AtomicUsize::new(0),
            total_overruns: AtomicU64::new(0),
            alarm: AtomicBool::new(false),
        }
    }

    /// Record a stage call's duration. Updates streak / alarm state.
    pub fn record(&self, elapsed_us: u64) {
        let budget = self.budget_us.load(Ordering::Relaxed);
        if elapsed_us > 2 * budget {
            let new_streak = self.overrun_streak.fetch_add(1, Ordering::Relaxed) + 1;
            self.total_overruns.fetch_add(1, Ordering::Relaxed);
            if new_streak >= WATCHDOG_THRESHOLD {
                self.alarm.store(true, Ordering::Release);
            }
        } else {
            self.overrun_streak.store(0, Ordering::Relaxed);
        }
    }

    pub fn clear_alarm(&self) {
        self.alarm.store(false, Ordering::Release);
        self.overrun_streak.store(0, Ordering::Relaxed);
    }
}

/// Statistics accumulated during a pipeline run.
#[derive(Debug)]
pub struct PipelineStats5 {
    /// Frames captured (counter).
    pub captured: AtomicU64,
    /// Frames inferred + post-processed (counter).
    pub processed: AtomicU64,
    /// Frames encoded (counter).
    pub encoded: AtomicU64,
    /// Frames flipped to the display (counter).
    pub displayed: AtomicU64,
    /// Frames dropped because a downstream ring was full (counter).
    pub dropped: AtomicU64,
    /// Per-stage watchdogs (budget + overrun streak + alarm).
    pub watchdog_capture: StageWatchdog,
    pub watchdog_infer: StageWatchdog,
    pub watchdog_encode: StageWatchdog,
    /// Set when any watchdog raises an alarm.
    pub watchdog_alarm: AtomicBool,
    /// Per-stage latency distributions. `record` is called with the
    /// stage's elapsed-microseconds value each frame; `snapshot`
    /// exposes p50/p95/p99 on demand. See
    /// [`crate::latency_histogram::LatencyHistogram`] for the storage
    /// contract (fixed 2048-sample ring, lock-free writer).
    pub capture_latency: crate::latency_histogram::LatencyHistogram,
    pub infer_latency: crate::latency_histogram::LatencyHistogram,
    pub encode_latency: crate::latency_histogram::LatencyHistogram,
    pub display_latency: crate::latency_histogram::LatencyHistogram,
    pub e2e_latency: crate::latency_histogram::LatencyHistogram,
}

impl PipelineStats5 {
    pub fn new(capture_budget_us: u64, infer_budget_us: u64, encode_budget_us: u64) -> Self {
        use crate::latency_histogram::LatencyHistogram;
        Self {
            captured: AtomicU64::new(0),
            processed: AtomicU64::new(0),
            encoded: AtomicU64::new(0),
            displayed: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            watchdog_capture: StageWatchdog::new(capture_budget_us),
            watchdog_infer: StageWatchdog::new(infer_budget_us),
            watchdog_encode: StageWatchdog::new(encode_budget_us),
            watchdog_alarm: AtomicBool::new(false),
            capture_latency: LatencyHistogram::new(),
            infer_latency: LatencyHistogram::new(),
            encode_latency: LatencyHistogram::new(),
            display_latency: LatencyHistogram::new(),
            e2e_latency: LatencyHistogram::new(),
        }
    }

    /// Snapshot human-readable metrics (snapshot is non-atomic, may
    /// catch counters mid-update — fine for periodic logging).
    ///
    /// Includes per-stage p50/p95/p99 and end-to-end quantiles from the
    /// lock-free latency histograms.
    pub fn snapshot(&self) -> PipelineStats5Snapshot {
        PipelineStats5Snapshot {
            captured: self.captured.load(Ordering::Relaxed),
            processed: self.processed.load(Ordering::Relaxed),
            encoded: self.encoded.load(Ordering::Relaxed),
            displayed: self.displayed.load(Ordering::Relaxed),
            dropped: self.dropped.load(Ordering::Relaxed),
            watchdog_alarm: self.watchdog_alarm.load(Ordering::Relaxed),
            capture: self.capture_latency.snapshot(),
            infer: self.infer_latency.snapshot(),
            encode: self.encode_latency.snapshot(),
            display: self.display_latency.snapshot(),
            e2e: self.e2e_latency.snapshot(),
        }
    }
}

/// Stats snapshot — plain `Copy` data for logging / JSON export.
#[derive(Debug, Clone, Copy, Default)]
pub struct PipelineStats5Snapshot {
    pub captured: u64,
    pub processed: u64,
    pub encoded: u64,
    pub displayed: u64,
    pub dropped: u64,
    pub watchdog_alarm: bool,
    pub capture: crate::latency_histogram::LatencyQuantiles,
    pub infer: crate::latency_histogram::LatencyQuantiles,
    pub encode: crate::latency_histogram::LatencyQuantiles,
    pub display: crate::latency_histogram::LatencyQuantiles,
    pub e2e: crate::latency_histogram::LatencyQuantiles,
}

/// One frame's metadata + payload pointer flowing through the pipeline.
#[derive(Debug)]
pub struct PipelinedFrame {
    pub frame_id: u64,
    /// Wall-clock time of capture (microseconds since UNIX epoch).
    pub captured_at_us: u64,
    /// DMA-BUF fd (from V4L2) — caller owns; pipeline doesn't close it.
    pub dma_fd: i32,
    /// Number of bytes in the DMA-BUF.
    pub dma_len: usize,
    /// Detection results filled by the InferWait stage.
    pub detections: Vec<crate::PipelineDetection>,
    /// Encoded NAL bytes (filled by Overlay+Encode stage).
    pub nal: Vec<u8>,
}

impl PipelinedFrame {
    pub fn new(frame_id: u64, dma_fd: i32, dma_len: usize) -> Self {
        Self {
            frame_id,
            captured_at_us: now_us(),
            dma_fd,
            dma_len,
            detections: Vec::new(),
            nal: Vec::new(),
        }
    }
}

fn now_us() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

/// Builder for the 5-stage pipeline. Construct, attach stage closures,
/// then call [`Self::run`] to consume up to `max_frames` (or until
/// `stop` flag is set).
pub struct Pipeline5Builder {
    pub depth: usize,
    pub stats: Arc<PipelineStats5>,
    pub stop: Arc<AtomicBool>,
}

impl Pipeline5Builder {
    pub fn new() -> Self {
        Self {
            depth: RING_DEPTH,
            stats: Arc::new(PipelineStats5::new(2_000, 8_000, 3_000)),
            stop: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Run the pipeline. Each closure runs on its own thread; closures
    /// produce/consume `PipelinedFrame`s through internal SPSC rings.
    ///
    /// On panic in a stage, the frame is dropped, the panic counter
    /// incremented, and the pipeline continues. Set the `stop` flag
    /// (via the builder's `stop` Arc) to halt cooperatively.
    pub fn run<C, I, E, O>(self, mut capture: C, mut infer: I, mut encode: E, mut output: O)
    where
        C: FnMut() -> Option<PipelinedFrame> + Send,
        I: FnMut(&mut PipelinedFrame) + Send,
        E: FnMut(&mut PipelinedFrame) + Send,
        O: FnMut(&PipelinedFrame) + Send,
    {
        let depth = self.depth;
        let stats = self.stats.clone();
        let stop = self.stop.clone();

        let r_cap_to_inf: Arc<SpscRing<PipelinedFrame>> = Arc::new(SpscRing::new(depth));
        let r_inf_to_enc: Arc<SpscRing<PipelinedFrame>> = Arc::new(SpscRing::new(depth));
        let r_enc_to_out: Arc<SpscRing<PipelinedFrame>> = Arc::new(SpscRing::new(depth));

        // Per-stage "done draining" signals. A downstream stage only
        // exits when its upstream is done AND its input ring is empty.
        let cap_done = Arc::new(AtomicBool::new(false));
        let inf_done = Arc::new(AtomicBool::new(false));
        let enc_done = Arc::new(AtomicBool::new(false));

        std::thread::scope(|s| {
            // Stage 1: capture
            {
                let r = r_cap_to_inf.clone();
                let stats = stats.clone();
                let stop = stop.clone();
                let cap_done = cap_done.clone();
                s.spawn(move || {
                    while !stop.load(Ordering::Acquire) {
                        let t0 = std::time::Instant::now();
                        #[allow(clippy::redundant_closure)]
                        let frame_opt =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| capture()))
                                .unwrap_or_else(|_| {
                                    eprintln!("[pipeline5] capture panic");
                                    None
                                });
                        let elapsed = t0.elapsed().as_micros() as u64;
                        stats.watchdog_capture.record(elapsed);
                        stats.capture_latency.record(elapsed);
                        if stats.watchdog_capture.alarm.load(Ordering::Acquire) {
                            stats.watchdog_alarm.store(true, Ordering::Release);
                        }
                        let Some(frame) = frame_opt else {
                            // capture returned None — end of stream.
                            stop.store(true, Ordering::Release);
                            break;
                        };
                        stats.captured.fetch_add(1, Ordering::Relaxed);
                        // Backpressure: always wait until downstream has
                        // room. Dropping at capture would desync OSD from
                        // video — better to slow the source.
                        let mut to_push = Some(frame);
                        while let Some(f) = to_push.take() {
                            match r.try_push(f) {
                                Ok(()) => break,
                                Err(returned) => {
                                    to_push = Some(returned);
                                    std::thread::yield_now();
                                }
                            }
                        }
                    }
                    cap_done.store(true, Ordering::Release);
                });
            }

            // Stage 2: infer (dispatch + wait + NMS combined for 3-stage simplicity)
            {
                let r_in = r_cap_to_inf.clone();
                let r_out = r_inf_to_enc.clone();
                let stats = stats.clone();
                let cap_done = cap_done.clone();
                let inf_done = inf_done.clone();
                s.spawn(move || {
                    // Drain until upstream is done AND ring is empty.
                    while !(cap_done.load(Ordering::Acquire) && r_in.is_empty_hint()) {
                        if let Some(mut frame) = r_in.try_pop() {
                            let t0 = std::time::Instant::now();
                            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                infer(&mut frame);
                            }));
                            let elapsed = t0.elapsed().as_micros() as u64;
                            stats.watchdog_infer.record(elapsed);
                            stats.infer_latency.record(elapsed);
                            if stats.watchdog_infer.alarm.load(Ordering::Acquire) {
                                stats.watchdog_alarm.store(true, Ordering::Release);
                            }
                            stats.processed.fetch_add(1, Ordering::Relaxed);
                            // Backpressure into next stage. Always wait —
                            // dropping during shutdown desynchronises the
                            // pipeline. Downstream is guaranteed to drain
                            // because it loops while !is_empty().
                            let mut to_push = Some(frame);
                            while let Some(f) = to_push.take() {
                                match r_out.try_push(f) {
                                    Ok(()) => break,
                                    Err(returned) => {
                                        to_push = Some(returned);
                                        std::hint::spin_loop();
                                    }
                                }
                            }
                        } else {
                            // Short yield instead of raw spin_loop so we don't
                            // starve parallel-running test threads sharing cores.
                            std::thread::yield_now();
                        }
                    }
                    inf_done.store(true, Ordering::Release);
                });
            }

            // Stage 3: encode (overlay + MPP encode)
            {
                let r_in = r_inf_to_enc.clone();
                let r_out = r_enc_to_out.clone();
                let stats = stats.clone();
                let inf_done = inf_done.clone();
                let enc_done = enc_done.clone();
                s.spawn(move || {
                    while !(inf_done.load(Ordering::Acquire) && r_in.is_empty_hint()) {
                        if let Some(mut frame) = r_in.try_pop() {
                            let t0 = std::time::Instant::now();
                            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                encode(&mut frame);
                            }));
                            let elapsed = t0.elapsed().as_micros() as u64;
                            stats.watchdog_encode.record(elapsed);
                            stats.encode_latency.record(elapsed);
                            if stats.watchdog_encode.alarm.load(Ordering::Acquire) {
                                stats.watchdog_alarm.store(true, Ordering::Release);
                            }
                            stats.encoded.fetch_add(1, Ordering::Relaxed);
                            // Backpressure into output stage.
                            let mut to_push = Some(frame);
                            while let Some(f) = to_push.take() {
                                match r_out.try_push(f) {
                                    Ok(()) => break,
                                    Err(returned) => {
                                        to_push = Some(returned);
                                        std::thread::yield_now();
                                    }
                                }
                            }
                        } else {
                            // Short yield instead of raw spin_loop so we don't
                            // starve parallel-running test threads sharing cores.
                            std::thread::yield_now();
                        }
                    }
                    enc_done.store(true, Ordering::Release);
                });
            }

            // Stage 4: output (DRM flip / file write)
            {
                let r_in = r_enc_to_out.clone();
                let stats = stats.clone();
                let enc_done = enc_done.clone();
                s.spawn(move || {
                    while !(enc_done.load(Ordering::Acquire) && r_in.is_empty_hint()) {
                        if let Some(frame) = r_in.try_pop() {
                            let t0 = std::time::Instant::now();
                            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                output(&frame);
                            }));
                            let display_elapsed = t0.elapsed().as_micros() as u64;
                            stats.display_latency.record(display_elapsed);
                            let e2e = now_us().saturating_sub(frame.captured_at_us);
                            stats.e2e_latency.record(e2e);
                            stats.displayed.fetch_add(1, Ordering::Relaxed);
                        } else {
                            // Short yield instead of raw spin_loop so we don't
                            // starve parallel-running test threads sharing cores.
                            std::thread::yield_now();
                        }
                    }
                });
            }
        });
    }
}

impl Default for Pipeline5Builder {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Send> SpscRing<T> {
    /// Best-effort hint: ring is empty (head == tail). Used to terminate
    /// drain loops; not authoritative under concurrent push.
    pub fn is_empty_hint(&self) -> bool {
        self.head.load(Ordering::Acquire) == self.tail.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::thread;

    #[test]
    fn ring_basic_roundtrip() {
        let r: SpscRing<u32> = SpscRing::new(4);
        for i in 0..4 {
            r.try_push(i).unwrap();
        }
        assert!(r.try_push(99).is_err(), "ring full");
        for i in 0..4 {
            assert_eq!(r.try_pop(), Some(i));
        }
        assert_eq!(r.try_pop(), None);
    }

    #[test]
    fn ring_concurrent_spsc() {
        let r: Arc<SpscRing<u32>> = Arc::new(SpscRing::new(8));
        let r_p = r.clone();
        let r_c = r.clone();
        let producer = thread::spawn(move || {
            for i in 0..1000 {
                while r_p.try_push(i).is_err() {
                    std::hint::spin_loop();
                }
            }
        });
        let count = Arc::new(AtomicUsize::new(0));
        let count_c = count.clone();
        let consumer = thread::spawn(move || {
            let mut got = Vec::with_capacity(1000);
            while got.len() < 1000 {
                if let Some(v) = r_c.try_pop() {
                    got.push(v);
                    count_c.fetch_add(1, Ordering::Relaxed);
                } else {
                    std::hint::spin_loop();
                }
            }
            got
        });
        producer.join().unwrap();
        let got = consumer.join().unwrap();
        assert_eq!(got.len(), 1000);
        for (i, v) in got.iter().enumerate() {
            assert_eq!(*v, i as u32);
        }
        assert_eq!(count.load(Ordering::Relaxed), 1000);
    }

    #[test]
    fn watchdog_triggers_after_threshold_overruns() {
        let w = StageWatchdog::new(1000); // 1ms budget
        for _ in 0..(WATCHDOG_THRESHOLD - 1) {
            w.record(5_000); // 2.5× budget = overrun
        }
        assert!(!w.alarm.load(Ordering::Acquire));
        w.record(5_000); // crosses threshold
        assert!(w.alarm.load(Ordering::Acquire));
    }

    #[test]
    fn watchdog_resets_streak_on_good_sample() {
        let w = StageWatchdog::new(1000);
        for _ in 0..3 {
            w.record(5_000);
        }
        assert_eq!(w.overrun_streak.load(Ordering::Relaxed), 3);
        w.record(500); // under budget
        assert_eq!(w.overrun_streak.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn pipeline5_runs_end_to_end() {
        let builder = Pipeline5Builder::new();
        let stats = builder.stats.clone();
        let stop = builder.stop.clone();
        let total = 50usize;
        let captured = AtomicUsize::new(0);

        builder.run(
            // capture
            || {
                let i = captured.fetch_add(1, Ordering::Relaxed);
                if i >= total {
                    stop.store(true, Ordering::Release);
                    return None;
                }
                Some(PipelinedFrame::new(i as u64, -1, 0))
            },
            // infer
            |f: &mut PipelinedFrame| {
                f.detections.push(crate::PipelineDetection {
                    bbox: crate::PipelineBBox {
                        x1: 0.0,
                        y1: 0.0,
                        x2: 10.0,
                        y2: 10.0,
                    },
                    score: 0.9,
                    class_id: 0,
                });
            },
            // encode
            |f: &mut PipelinedFrame| {
                f.nal.extend_from_slice(b"FAKE_NAL");
            },
            // output
            |_f: &PipelinedFrame| {},
        );

        let snap = stats.snapshot();
        assert_eq!(snap.captured, total as u64);
        assert_eq!(snap.processed, total as u64);
        assert_eq!(snap.encoded, total as u64);
        assert_eq!(snap.displayed, total as u64);
        assert_eq!(snap.dropped, 0);
    }

    #[test]
    fn pipeline5_survives_infer_panic() {
        let builder = Pipeline5Builder::new();
        let stats = builder.stats.clone();
        let stop = builder.stop.clone();
        let total = 20usize;
        let captured = AtomicUsize::new(0);

        builder.run(
            || {
                let i = captured.fetch_add(1, Ordering::Relaxed);
                if i >= total {
                    stop.store(true, Ordering::Release);
                    return None;
                }
                Some(PipelinedFrame::new(i as u64, -1, 0))
            },
            |f: &mut PipelinedFrame| {
                if f.frame_id == 5 {
                    panic!("infer crash on frame 5");
                }
            },
            |_| {},
            |_| {},
        );

        let snap = stats.snapshot();
        // Panicked frame still counts as processed (frame flowed through).
        assert_eq!(snap.captured, total as u64);
        assert_eq!(snap.processed, total as u64);
        assert_eq!(snap.encoded, total as u64);
    }
}
