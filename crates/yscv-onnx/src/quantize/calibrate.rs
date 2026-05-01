//! Activation-statistics collection for post-training quantization (PTQ).
//!
//! `CalibrationCollector` records per-tensor min / max / count over a
//! sequence of inference runs. Install it via [`CalibrationCollector::scope`]
//! before calling `OnnxRunner::run` (or any equivalent entry point); the
//! returned [`CalibrationScope`] uninstalls the hook on drop. Read the
//! aggregated statistics with [`CalibrationCollector::snapshot`] once the
//! scope ends.
//!
//! ```ignore
//! use yscv_onnx::{CalibrationCollector, OnnxRunner, load_onnx_model_from_file};
//!
//! let model = load_onnx_model_from_file("model.onnx")?;
//! let runner = OnnxRunner::new(&model)?;
//! let collector = CalibrationCollector::new();
//! {
//!     let _scope = collector.scope();
//!     for sample in calibration_set {
//!         runner.run(&[("input", &sample)])?;
//!     }
//! } // scope ends -> hook uninstalled
//! let stats = collector.snapshot();
//! ```
//!
//! When no scope is active the hook is gated by a single relaxed atomic
//! load, so non-calibration inference pays only that one branch in the
//! tensor-insertion path.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use yscv_tensor::Tensor;

/// Per-tensor min / max / element-count aggregate. NaN values are skipped;
/// non-fp32 tensors are silently ignored at record time. `count` is the
/// total number of fp32 values seen across all `record_activation` calls
/// for the tensor name.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MinMax {
    pub min: f32,
    pub max: f32,
    pub count: u64,
}

impl Default for MinMax {
    fn default() -> Self {
        Self {
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
            count: 0,
        }
    }
}

impl MinMax {
    /// Update min / max from a slice. NaN values are skipped (not propagated
    /// into the bound) but still counted.
    pub(crate) fn update(&mut self, slice: &[f32]) {
        for &v in slice {
            if v.is_nan() {
                continue;
            }
            if v < self.min {
                self.min = v;
            }
            if v > self.max {
                self.max = v;
            }
        }
        self.count = self.count.saturating_add(slice.len() as u64);
    }

    /// True if no values have been recorded yet.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Symmetric absolute bound `max(|min|, |max|)`. Useful for symmetric
    /// quantization where `scale = abs_max / 127.0`. Returns 0.0 if empty.
    pub fn abs_max(&self) -> f32 {
        if self.is_empty() {
            return 0.0;
        }
        self.min.abs().max(self.max.abs())
    }
}

/// Reservoir-sampled distribution of one tensor's observed fp32 values.
///
/// Records `min` / `max` / `count` exactly and keeps up to
/// [`Histogram::CAPACITY`] sampled values. Used by
/// [`super::derive::derive_percentile`] and
/// [`super::derive::derive_mse_optimal`] to fit a quantization scale
/// that clips outliers more aggressively than min-max.
///
/// Sampling strategy is deterministic Vitter's R: the first `CAPACITY`
/// values fill the reservoir, then each subsequent value index `i`
/// replaces a uniformly-chosen reservoir slot with probability
/// `CAPACITY / i`. For typical PTQ runs (10-1000 calibration samples,
/// each with thousands of activation values) `CAPACITY = 16384`
/// gives sub-percent-level percentile estimates.
#[derive(Clone, Debug)]
pub struct Histogram {
    pub min: f32,
    pub max: f32,
    pub count: u64,
    /// Bounded-size reservoir. Owned `Vec<f32>` so callers can sort,
    /// fold, etc. without going through helper methods.
    pub samples: Vec<f32>,
    /// Per-tensor LCG state — keeps replacement positions reproducible
    /// and lock-free (state lives inside the per-tensor entry guarded
    /// by the same outer mutex as the rest of the entry).
    rng_state: u64,
}

impl Histogram {
    pub const CAPACITY: usize = 16384;

    /// Test-only constructor exposing the internal default state. Not
    /// part of the public API; production code receives histograms via
    /// [`CalibrationCollector::histograms`].
    #[cfg(test)]
    pub(crate) fn for_test() -> Self {
        Self::new()
    }

    fn new() -> Self {
        Self {
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
            count: 0,
            samples: Vec::with_capacity(Self::CAPACITY),
            rng_state: 0x9E37_79B9_7F4A_7C15,
        }
    }

    pub(crate) fn update(&mut self, slice: &[f32]) {
        for &v in slice {
            if v.is_nan() {
                continue;
            }
            if v < self.min {
                self.min = v;
            }
            if v > self.max {
                self.max = v;
            }
            self.count += 1;
            if self.samples.len() < Self::CAPACITY {
                self.samples.push(v);
            } else {
                // Vitter's R reservoir: replace at uniform random index in
                // [0, count). Cheap LCG keeps this lock-free.
                self.rng_state = self
                    .rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let idx = ((self.rng_state >> 33) % self.count) as usize;
                if idx < Self::CAPACITY {
                    self.samples[idx] = v;
                }
            }
        }
    }

    /// True if no values have been recorded yet.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

struct Inner {
    per_tensor: Mutex<HashMap<String, MinMax>>,
    histograms: Mutex<HashMap<String, Histogram>>,
    histogram_enabled: AtomicBool,
}

/// Collector for per-tensor activation statistics during a calibration
/// run. Cheap to clone (`Arc`-backed). See module-level docs for usage.
#[derive(Clone)]
pub struct CalibrationCollector {
    inner: Arc<Inner>,
}

impl Default for CalibrationCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl CalibrationCollector {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Inner {
                per_tensor: Mutex::new(HashMap::new()),
                histograms: Mutex::new(HashMap::new()),
                histogram_enabled: AtomicBool::new(false),
            }),
        }
    }

    /// Enable histogram (reservoir-sampling) collection alongside the
    /// default min/max. Histograms feed [`super::derive::derive_percentile`]
    /// and [`super::derive::derive_mse_optimal`]; without them only the
    /// MinMax-based [`super::derive::derive_asymmetric`] / `derive_symmetric`
    /// paths are usable. Costs one extra Mutex lock per insert when
    /// active. Default off.
    pub fn enable_histograms(&self, on: bool) {
        self.inner.histogram_enabled.store(on, Ordering::Release);
    }

    /// Snapshot of aggregated per-tensor histograms. Empty when
    /// [`Self::enable_histograms`] was never set.
    pub fn histograms(&self) -> HashMap<String, Histogram> {
        self.inner
            .histograms
            .lock()
            .expect("calibration histograms mutex poisoned")
            .clone()
    }

    /// Install this collector globally. Returned scope uninstalls on drop.
    /// Calling `scope` while another scope is alive replaces the active
    /// collector; the previous scope still uninstalls cleanly on its own
    /// drop, but no records are routed to the previous collector while
    /// this one is active. Nesting is therefore allowed but stat traffic
    /// follows the most recent scope.
    pub fn scope(&self) -> CalibrationScope<'_> {
        let cell = global_cell();
        *cell.lock().expect("calibration global mutex poisoned") = Some(self.inner.clone());
        CALIBRATION_ACTIVE.store(true, Ordering::Release);
        CalibrationScope {
            _phantom: PhantomData,
        }
    }

    /// Snapshot of aggregated per-tensor statistics. Cheap clone of the
    /// internal map; the collector continues recording into the same
    /// state if a scope is still active.
    pub fn snapshot(&self) -> HashMap<String, MinMax> {
        self.inner
            .per_tensor
            .lock()
            .expect("calibration map mutex poisoned")
            .clone()
    }

    /// Number of tensors with at least one recorded value.
    pub fn len(&self) -> usize {
        self.inner
            .per_tensor
            .lock()
            .expect("calibration map mutex poisoned")
            .len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Push a slice of f32 activation values directly into the
    /// per-tensor stats (and histogram, if enabled). Bypasses the
    /// `scope`/runner hook — useful for offline calibration with
    /// pre-recorded activations or for synthetic accuracy harnesses
    /// that want to compare derivation strategies without running
    /// inference.
    pub fn record(&self, name: &str, values: &[f32]) {
        {
            let mut map = self
                .inner
                .per_tensor
                .lock()
                .expect("calibration map mutex poisoned");
            map.entry(name.to_string()).or_default().update(values);
        }
        if self.inner.histogram_enabled.load(Ordering::Acquire) {
            let mut hmap = self
                .inner
                .histograms
                .lock()
                .expect("calibration histograms mutex poisoned");
            hmap.entry(name.to_string())
                .or_insert_with(Histogram::new)
                .update(values);
        }
    }
}

/// RAII scope handle that keeps the collector installed. Drop it (let it
/// fall out of scope) to uninstall the calibration hook.
pub struct CalibrationScope<'a> {
    _phantom: PhantomData<&'a CalibrationCollector>,
}

impl<'a> Drop for CalibrationScope<'a> {
    fn drop(&mut self) {
        CALIBRATION_ACTIVE.store(false, Ordering::Release);
        if let Some(cell) = GLOBAL_COLLECTOR.get() {
            *cell.lock().expect("calibration global mutex poisoned") = None;
        }
    }
}

static CALIBRATION_ACTIVE: AtomicBool = AtomicBool::new(false);
static GLOBAL_COLLECTOR: OnceLock<Mutex<Option<Arc<Inner>>>> = OnceLock::new();

fn global_cell() -> &'static Mutex<Option<Arc<Inner>>> {
    GLOBAL_COLLECTOR.get_or_init(|| Mutex::new(None))
}

/// Fast-path probe used by the runner's tensor-insertion sites. One relaxed
/// atomic load when no scope is active.
#[inline]
pub(crate) fn calibration_active() -> bool {
    CALIBRATION_ACTIVE.load(Ordering::Relaxed)
}

/// Record one activation tensor under the given name. No-op when calibration
/// is not active or the tensor is not f32. Safe to call concurrently from
/// rayon worker threads.
pub(crate) fn record_activation(name: &str, tensor: &Tensor) {
    if !calibration_active() {
        return;
    }
    let cell = match GLOBAL_COLLECTOR.get() {
        Some(c) => c,
        None => return,
    };
    let arc = match cell.lock().ok().and_then(|g| g.clone()) {
        Some(a) => a,
        None => return,
    };
    let slice = match tensor.try_data() {
        Ok(s) => s,
        Err(_) => return, // non-f32 tensor — calibration only meaningful for fp32
    };
    if arc.histogram_enabled.load(Ordering::Relaxed)
        && let Ok(mut hists) = arc.histograms.lock()
    {
        hists
            .entry(name.to_string())
            .or_insert_with(Histogram::new)
            .update(slice);
    }
    if let Ok(mut map) = arc.per_tensor.lock() {
        map.entry(name.to_string()).or_default().update(slice);
    }
}

/// Serialisation lock for tests that exercise the global calibration
/// hook. Multiple concurrent `scope()` calls would race on the static
/// `GLOBAL_COLLECTOR` / `CALIBRATION_ACTIVE`, so any test that creates a
/// `CalibrationScope` must hold this lock for the duration of the scope.
/// Production code creates at most one scope at a time per process and
/// does not need the lock.
#[cfg(test)]
pub(crate) fn test_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_max_default_is_empty_and_inverted() {
        let mm = MinMax::default();
        assert!(mm.is_empty());
        assert_eq!(mm.count, 0);
        assert!(mm.min.is_infinite() && mm.min.is_sign_positive());
        assert!(mm.max.is_infinite() && mm.max.is_sign_negative());
        assert_eq!(mm.abs_max(), 0.0);
    }

    #[test]
    fn min_max_update_tracks_bounds_and_count() {
        let mut mm = MinMax::default();
        mm.update(&[1.0, -2.0, 3.5, 0.0]);
        assert_eq!(mm.min, -2.0);
        assert_eq!(mm.max, 3.5);
        assert_eq!(mm.count, 4);
        assert_eq!(mm.abs_max(), 3.5);

        mm.update(&[-10.0]);
        assert_eq!(mm.min, -10.0);
        assert_eq!(mm.max, 3.5);
        assert_eq!(mm.count, 5);
        assert_eq!(mm.abs_max(), 10.0);
    }

    #[test]
    fn min_max_skips_nan_in_bounds_but_counts_them() {
        let mut mm = MinMax::default();
        mm.update(&[1.0, f32::NAN, 2.0]);
        assert_eq!(mm.min, 1.0);
        assert_eq!(mm.max, 2.0);
        assert_eq!(mm.count, 3);
    }

    #[test]
    fn collector_records_only_inside_scope() {
        let _guard = test_lock().lock().unwrap_or_else(|e| e.into_inner());
        let name = "calib_test_records_only_inside_scope_tensor";
        let tensor = Tensor::from_vec(vec![3], vec![1.0_f32, 2.0, 3.0]).unwrap();
        let collector = CalibrationCollector::new();

        // Outside scope: no record.
        record_activation(name, &tensor);
        assert!(!collector.snapshot().contains_key(name));

        {
            let _scope = collector.scope();
            record_activation(name, &tensor);
        }

        let snap = collector.snapshot();
        let stats = snap.get(name).expect("tensor should be recorded");
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 3.0);
        assert_eq!(stats.count, 3);

        // After scope ends, further records are dropped.
        record_activation(name, &tensor);
        let snap2 = collector.snapshot();
        assert_eq!(snap2.get(name).unwrap().count, 3);
    }

    #[test]
    fn collector_aggregates_across_calls() {
        let _guard = test_lock().lock().unwrap_or_else(|e| e.into_inner());
        let name = "calib_test_aggregates_tensor";
        let collector = CalibrationCollector::new();
        let _scope = collector.scope();
        record_activation(
            name,
            &Tensor::from_vec(vec![2], vec![1.0_f32, 2.0]).unwrap(),
        );
        record_activation(
            name,
            &Tensor::from_vec(vec![2], vec![-5.0_f32, 10.0]).unwrap(),
        );
        drop(_scope);

        let stats = collector.snapshot().get(name).copied().unwrap();
        assert_eq!(stats.min, -5.0);
        assert_eq!(stats.max, 10.0);
        assert_eq!(stats.count, 4);
    }
}
