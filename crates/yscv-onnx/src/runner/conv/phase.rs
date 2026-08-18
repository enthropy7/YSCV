//! Opt-in phase timers for the int8 QLinearConv fast path.
//!
//! Set `YSCV_CONV_PHASE=1` to accumulate wall-clock nanoseconds into per-phase
//! counters (im2col, GEMM, requant, output alloc, depthwise, setup). Read them
//! back with [`dump_conv_phases`]. When the env var is unset every timing site
//! costs a single relaxed atomic load and a branch, so the counters can stay
//! compiled in. Meant for board profiling, not production hot-path use.

use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::time::Instant;

pub(crate) static IM2COL_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static GEMM_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static REQUANT_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static ALLOC_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static DW_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static SETUP_NS: AtomicU64 = AtomicU64::new(0);

// 2 = not yet resolved from the environment; 0/1 = disabled/enabled.
static ENABLED: AtomicU8 = AtomicU8::new(2);

#[inline]
pub(crate) fn enabled() -> bool {
    let v = ENABLED.load(Ordering::Relaxed);
    if v != 2 {
        return v == 1;
    }
    let on = std::env::var("YSCV_CONV_PHASE")
        .map(|s| s == "1")
        .unwrap_or(false);
    ENABLED.store(on as u8, Ordering::Relaxed);
    on
}

/// Start a phase timer if profiling is enabled; `None` otherwise (zero cost).
#[inline]
pub(crate) fn start() -> Option<Instant> {
    if enabled() {
        Some(Instant::now())
    } else {
        None
    }
}

/// Add the elapsed nanoseconds since `t` (if `Some`) to `counter`.
#[inline]
pub(crate) fn stop(counter: &AtomicU64, t: Option<Instant>) {
    if let Some(t) = t {
        counter.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

/// Render the accumulated per-phase totals as `ms` divided by `iters`
/// (inference count), one line per phase. No-op-safe when never enabled.
pub fn dump_conv_phases(iters: u64) -> String {
    let iters = iters.max(1) as f64;
    let ms = |c: &AtomicU64| (c.load(Ordering::Relaxed) as f64) / 1e6 / iters;
    format!(
        "conv-phase ms/inf (iters={}): im2col={:.3} gemm={:.3} requant={:.3} out_alloc={:.3} depthwise={:.3} setup={:.3} | sum={:.3}",
        iters as u64,
        ms(&IM2COL_NS),
        ms(&GEMM_NS),
        ms(&REQUANT_NS),
        ms(&ALLOC_NS),
        ms(&DW_NS),
        ms(&SETUP_NS),
        ms(&IM2COL_NS)
            + ms(&GEMM_NS)
            + ms(&REQUANT_NS)
            + ms(&ALLOC_NS)
            + ms(&DW_NS)
            + ms(&SETUP_NS),
    )
}
