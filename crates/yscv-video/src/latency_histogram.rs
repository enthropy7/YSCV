//! Fixed-capacity lock-free latency recorder.
//!
//! Designed for the hot path of a real-time pipeline: `record` is a
//! single atomic `fetch_add` on a ring-buffer head plus an atomic store
//! of the sample; `snapshot` copies at most `CAP` samples into a local
//! Vec, sorts them, and returns `min / p50 / p90 / p95 / p99 / max` +
//! `count`. `CAP` is a compile-time power of two so the ring index can
//! be reduced with a mask instead of a modulo.
//!
//! The recorder does not pre-compute distribution; that would need
//! either a lock or a hot-path-atomic histogram with bucket contention.
//! A ring-buffer + lazy sort costs O(N log N) only at snapshot time,
//! which is fine for once-per-second logging.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Capacity of the ring buffer: 2048 samples. At 60 FPS this is ~34
/// seconds of history — plenty for once-per-second quantile dumps.
/// Power of two so `& (CAP - 1)` is a fast index.
pub const LATENCY_HISTOGRAM_CAP: usize = 2048;

/// Lock-free latency recorder. `record` is called from hot-path
/// threads; `snapshot` is called from a supervisor / logging thread.
pub struct LatencyHistogram {
    samples: Box<[AtomicU64; LATENCY_HISTOGRAM_CAP]>,
    /// Running count of total records. Used both as the ring's write
    /// index (modulo CAP) and as the quantile-snapshot size indicator
    /// (clamped to CAP).
    total: AtomicUsize,
}

impl LatencyHistogram {
    pub fn new() -> Self {
        // Build an uninitialised-feeling array of AtomicU64 without
        // going through `vec![0; CAP].try_into()` which allocates a
        // Vec first. AtomicU64 is `Copy`-compatible for initialisation
        // purposes (zero is a valid bit pattern).
        let samples: Vec<AtomicU64> = (0..LATENCY_HISTOGRAM_CAP)
            .map(|_| AtomicU64::new(0))
            .collect();
        let samples: Box<[AtomicU64; LATENCY_HISTOGRAM_CAP]> = samples
            .into_boxed_slice()
            .try_into()
            .expect("exactly LATENCY_HISTOGRAM_CAP AtomicU64 elements");
        Self {
            samples,
            total: AtomicUsize::new(0),
        }
    }

    /// Record a latency sample in microseconds. O(1), one atomic add +
    /// one atomic store. Safe to call from any thread.
    #[inline]
    pub fn record(&self, elapsed_us: u64) {
        // `fetch_add` wraps; the `& mask` reduces to the ring index.
        let idx = self.total.fetch_add(1, Ordering::Relaxed) & (LATENCY_HISTOGRAM_CAP - 1);
        self.samples[idx].store(elapsed_us, Ordering::Relaxed);
    }

    /// Return a snapshot of current quantiles. Copies the live samples
    /// (at most `LATENCY_HISTOGRAM_CAP`), sorts, computes indices. Not
    /// hot-path — intended for once-per-second or on-demand logging.
    pub fn snapshot(&self) -> LatencyQuantiles {
        let count = self.total.load(Ordering::Relaxed);
        let n = count.min(LATENCY_HISTOGRAM_CAP);
        if n == 0 {
            return LatencyQuantiles::default();
        }
        let mut buf: Vec<u64> = (0..n)
            .map(|i| self.samples[i].load(Ordering::Relaxed))
            .collect();
        buf.sort_unstable();
        let pick = |p: f64| -> u64 {
            let idx = ((n.saturating_sub(1)) as f64 * p).round() as usize;
            buf[idx.min(n - 1)]
        };
        LatencyQuantiles {
            count,
            min_us: buf[0],
            p50_us: pick(0.50),
            p90_us: pick(0.90),
            p95_us: pick(0.95),
            p99_us: pick(0.99),
            max_us: buf[n - 1],
        }
    }

    /// Drop all recorded samples. Useful between warmup + measurement
    /// phases. Not atomic with concurrent `record` — if a writer is
    /// active during reset, its sample may survive or be lost.
    pub fn reset(&self) {
        self.total.store(0, Ordering::Relaxed);
        for s in self.samples.iter() {
            s.store(0, Ordering::Relaxed);
        }
    }
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for LatencyHistogram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let snap = self.snapshot();
        write!(
            f,
            "LatencyHistogram {{ count: {}, min: {} µs, p50: {} µs, p95: {} µs, p99: {} µs, max: {} µs }}",
            snap.count, snap.min_us, snap.p50_us, snap.p95_us, snap.p99_us, snap.max_us
        )
    }
}

/// Snapshot of a `LatencyHistogram`. Plain `Copy` so it round-trips
/// through logging / JSON / atomic writes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LatencyQuantiles {
    /// Total records observed (may exceed `LATENCY_HISTOGRAM_CAP`;
    /// quantiles were computed from at most `CAP` most-recent samples).
    pub count: usize,
    pub min_us: u64,
    pub p50_us: u64,
    pub p90_us: u64,
    pub p95_us: u64,
    pub p99_us: u64,
    pub max_us: u64,
}

impl LatencyQuantiles {
    /// Convert p50 to approximate FPS. 0 if no samples.
    pub fn fps_p50(&self) -> f64 {
        if self.p50_us == 0 {
            0.0
        } else {
            1_000_000.0 / self.p50_us as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_snapshot_is_zero() {
        let h = LatencyHistogram::new();
        let snap = h.snapshot();
        assert_eq!(snap.count, 0);
        assert_eq!(snap.min_us, 0);
        assert_eq!(snap.p50_us, 0);
    }

    #[test]
    fn exact_quantiles_for_uniform_samples() {
        let h = LatencyHistogram::new();
        // Record 0..100 µs exactly once each.
        for i in 0..100u64 {
            h.record(i);
        }
        let snap = h.snapshot();
        assert_eq!(snap.count, 100);
        assert_eq!(snap.min_us, 0);
        assert_eq!(snap.max_us, 99);
        // p50 of [0..99] is round((99)*.5) = 50 → value at index 50 = 50.
        assert_eq!(snap.p50_us, 50);
        // p95 index: round(99*.95) = 94 → value 94.
        assert_eq!(snap.p95_us, 94);
        // p99 index: round(99*.99) = 98 → value 98.
        assert_eq!(snap.p99_us, 98);
    }

    #[test]
    fn ring_wraps_beyond_capacity() {
        let h = LatencyHistogram::new();
        // Fill + overrun by CAP / 2. After 1.5×CAP writes, only the
        // last CAP values remain visible.
        for i in 0..(LATENCY_HISTOGRAM_CAP + LATENCY_HISTOGRAM_CAP / 2) {
            h.record(i as u64);
        }
        let snap = h.snapshot();
        // `count` reflects total records; snapshot reads at most CAP.
        assert_eq!(
            snap.count,
            LATENCY_HISTOGRAM_CAP + LATENCY_HISTOGRAM_CAP / 2
        );
        // Ring overwrote the first CAP/2 samples; min is the smallest
        // of whatever's still in the ring.
        assert!(snap.min_us >= (LATENCY_HISTOGRAM_CAP / 2) as u64 - 1);
    }

    #[test]
    fn reset_clears_samples() {
        let h = LatencyHistogram::new();
        for i in 0..10 {
            h.record(i);
        }
        h.reset();
        assert_eq!(h.snapshot().count, 0);
    }

    #[test]
    fn fps_p50_derivation() {
        let q = LatencyQuantiles {
            p50_us: 16_666, // ~60 FPS
            ..Default::default()
        };
        let fps = q.fps_p50();
        assert!((59.0..=61.0).contains(&fps), "got {fps}");
    }
}
