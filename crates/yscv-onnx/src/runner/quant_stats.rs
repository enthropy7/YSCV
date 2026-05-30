//! Process-wide INT8 runtime counters and the QuantTensor payload
//! used to carry quantized activations between fused QDQ ops.

use std::sync::atomic::{AtomicU64, Ordering};

static QUANT_QDQ_BOUNDARY_COUNT: AtomicU64 = AtomicU64::new(0);
static QUANT_LINEAR_CONV_FAST_COUNT: AtomicU64 = AtomicU64::new(0);
static QUANT_LINEAR_CONV_FALLBACK_COUNT: AtomicU64 = AtomicU64::new(0);
static QUANT_LINEAR_MATMUL_FAST_COUNT: AtomicU64 = AtomicU64::new(0);
static QUANT_LINEAR_MATMUL_FALLBACK_COUNT: AtomicU64 = AtomicU64::new(0);
static QUANT_I8_STORE_COUNT: AtomicU64 = AtomicU64::new(0);
static QUANT_I8_MATERIALIZE_COUNT: AtomicU64 = AtomicU64::new(0);
static QUANT_CHAIN_EXECUTED_COUNT: AtomicU64 = AtomicU64::new(0);

/// Process-wide counters for quantized ONNX runtime execution.
///
/// These are intentionally coarse and atomic: they are cheap enough to leave
/// in production builds and make benchmark logs honest about whether a run
/// actually executed INT8 kernels or silently fell back to fp32.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct QuantRuntimeStats {
    pub qdq_boundaries: u64,
    pub qlinear_conv_fast: u64,
    pub qlinear_conv_fallback: u64,
    pub qlinear_matmul_fast: u64,
    pub qlinear_matmul_fallback: u64,
    pub quant_i8_stores: u64,
    pub quant_i8_materializations: u64,
    /// Number of times a fused INT8 quant-domain chain (currently
    /// `QuantizedPwDw`) executed in the runner this run. Each chain
    /// replaces 2 `QLinearConv` fast-path executions and one QDQ
    /// boundary fold; tracking it here lets the bench tracker confirm
    /// the new action actually fires instead of silently falling
    /// through to the per-op path.
    pub quant_chain_executed: u64,
}

/// Reset all process-wide quantized-runtime counters to zero.
pub fn reset_quant_runtime_stats() {
    QUANT_QDQ_BOUNDARY_COUNT.store(0, Ordering::Relaxed);
    QUANT_LINEAR_CONV_FAST_COUNT.store(0, Ordering::Relaxed);
    QUANT_LINEAR_CONV_FALLBACK_COUNT.store(0, Ordering::Relaxed);
    QUANT_LINEAR_MATMUL_FAST_COUNT.store(0, Ordering::Relaxed);
    QUANT_LINEAR_MATMUL_FALLBACK_COUNT.store(0, Ordering::Relaxed);
    QUANT_I8_STORE_COUNT.store(0, Ordering::Relaxed);
    QUANT_I8_MATERIALIZE_COUNT.store(0, Ordering::Relaxed);
    QUANT_CHAIN_EXECUTED_COUNT.store(0, Ordering::Relaxed);
}

/// Snapshot the current process-wide quantized-runtime counters.
pub fn quant_runtime_stats() -> QuantRuntimeStats {
    QuantRuntimeStats {
        qdq_boundaries: QUANT_QDQ_BOUNDARY_COUNT.load(Ordering::Relaxed),
        qlinear_conv_fast: QUANT_LINEAR_CONV_FAST_COUNT.load(Ordering::Relaxed),
        qlinear_conv_fallback: QUANT_LINEAR_CONV_FALLBACK_COUNT.load(Ordering::Relaxed),
        qlinear_matmul_fast: QUANT_LINEAR_MATMUL_FAST_COUNT.load(Ordering::Relaxed),
        qlinear_matmul_fallback: QUANT_LINEAR_MATMUL_FALLBACK_COUNT.load(Ordering::Relaxed),
        quant_i8_stores: QUANT_I8_STORE_COUNT.load(Ordering::Relaxed),
        quant_i8_materializations: QUANT_I8_MATERIALIZE_COUNT.load(Ordering::Relaxed),
        quant_chain_executed: QUANT_CHAIN_EXECUTED_COUNT.load(Ordering::Relaxed),
    }
}

#[inline]
pub(crate) fn quant_int8_fast_enabled() -> bool {
    std::env::var("YSCV_QUANT_INT8_FAST").as_deref() != Ok("0")
}

#[inline]
pub(crate) fn note_quant_qdq_boundary() {
    QUANT_QDQ_BOUNDARY_COUNT.fetch_add(1, Ordering::Relaxed);
}

#[inline]
pub(crate) fn note_qlinear_conv_fast() {
    QUANT_LINEAR_CONV_FAST_COUNT.fetch_add(1, Ordering::Relaxed);
}

#[inline]
pub(crate) fn note_qlinear_conv_fallback() {
    QUANT_LINEAR_CONV_FALLBACK_COUNT.fetch_add(1, Ordering::Relaxed);
}

#[inline]
pub(crate) fn note_qlinear_matmul_fast() {
    QUANT_LINEAR_MATMUL_FAST_COUNT.fetch_add(1, Ordering::Relaxed);
}

#[inline]
pub(crate) fn note_qlinear_matmul_fallback() {
    QUANT_LINEAR_MATMUL_FALLBACK_COUNT.fetch_add(1, Ordering::Relaxed);
}

#[inline]
pub(crate) fn note_quant_i8_store() {
    QUANT_I8_STORE_COUNT.fetch_add(1, Ordering::Relaxed);
}

#[inline]
pub(crate) fn note_quant_i8_materialize() {
    QUANT_I8_MATERIALIZE_COUNT.fetch_add(1, Ordering::Relaxed);
}

#[inline]
pub(crate) fn note_quant_chain_executed() {
    QUANT_CHAIN_EXECUTED_COUNT.fetch_add(1, Ordering::Relaxed);
}

#[derive(Clone, Debug)]
pub(crate) struct QuantTensor {
    pub(crate) data: Vec<i8>,
    pub(crate) shape: Vec<usize>,
    pub(crate) scale: f32,
    pub(crate) zero_point: f32,
    pub(crate) nhwc: bool,
}
