//! Scale / zero-point derivation from `MinMax` activation statistics.
//!
//! The standard practice in QNN / ONNX Runtime / PyTorch quantization:
//!
//! - Activations: per-tensor asymmetric int8 in `[0, 255]` (`u8` storage,
//!   represented as i32 zero-point so signed int8 schemes share the type).
//! - Weights: per-channel symmetric int8 in `[-128, 127]` along the
//!   output-channel axis (axis 0 of OIHW or KHWC[O]).
//! - INT4 weights: per-channel symmetric in `[-8, 7]` (matches the format
//!   produced by `quantize_weights_int4`).
//!
//! NaN values are skipped at collection time (see `MinMax::update`); a
//! degenerate `MinMax` (count == 0 or `min == max`) yields a sentinel
//! `QuantParams { scale: 1.0, zero_point: 0 }` so callers don't have to
//! special-case downstream — the resulting requantization is still a
//! safe no-op for a constant tensor.
//!
//! Percentile-99.99 and MSE-optimal calibration require a histogram of
//! observed activations (not just min/max), so they are deferred to a
//! follow-on patch that extends `MinMax` with a histogram aggregate.

use super::calibrate::MinMax;

/// Quantization parameters derived for a single tensor (or a single
/// channel slice). `scale` is fp32; `zero_point` is stored as i32 so
/// the same struct can carry signed schemes (`[-128, 127]`, `[-8, 7]`)
/// and unsigned ones (`[0, 255]`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuantParams {
    pub scale: f32,
    pub zero_point: i32,
}

impl QuantParams {
    /// Sentinel parameters for a degenerate (empty / constant) tensor.
    /// `scale = 1.0`, `zero_point = 0` — quant/dequant pair through these
    /// is identity for any value that fits in the target range.
    pub const IDENTITY: QuantParams = QuantParams {
        scale: 1.0,
        zero_point: 0,
    };
}

/// Common quantization target — int8 in `[0, 255]` or `[-128, 127]`,
/// int4 in `[-8, 7]`. Encapsulates the qmin/qmax span so the same
/// derivation routines work for any signed/unsigned/narrow scheme.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantTarget {
    /// `[0, 255]` — unsigned int8 storage. Used for asymmetric
    /// activations.
    Uint8,
    /// `[-128, 127]` — signed int8 storage. Used for symmetric weights
    /// and signed-int8 activations (rarer).
    Int8,
    /// `[-8, 7]` — signed int4 storage. Matches `quantize_weights_int4`
    /// existing output format.
    Int4,
}

impl QuantTarget {
    fn qmin(self) -> i32 {
        match self {
            QuantTarget::Uint8 => 0,
            QuantTarget::Int8 => -128,
            QuantTarget::Int4 => -8,
        }
    }
    fn qmax(self) -> i32 {
        match self {
            QuantTarget::Uint8 => 255,
            QuantTarget::Int8 => 127,
            QuantTarget::Int4 => 7,
        }
    }
    /// Symmetric range half-span (used as the denominator when deriving
    /// a symmetric scale). For `[-128, 127]` we use 127 to avoid
    /// wasting representational range on the asymmetric -128 corner.
    fn sym_denom(self) -> f32 {
        match self {
            QuantTarget::Uint8 => 127.5, // unusual but defined
            QuantTarget::Int8 => 127.0,
            QuantTarget::Int4 => 7.0,
        }
    }
}

/// Derive **asymmetric** parameters for a single tensor (or single
/// channel) given its `[min, max]` aggregate.
///
/// Maps `min → qmin`, `max → qmax` linearly; rounds the resulting
/// `zero_point` and clamps it to the target range.
///
/// Empty / NaN-only / constant inputs return `QuantParams::IDENTITY`.
pub fn derive_asymmetric(stat: MinMax, target: QuantTarget) -> QuantParams {
    if stat.is_empty() || stat.min.is_nan() || stat.max.is_nan() {
        return QuantParams::IDENTITY;
    }
    let mut min = stat.min.min(0.0); // ensure 0 lies in the range so zero is exactly representable
    let mut max = stat.max.max(0.0);
    if !min.is_finite() || !max.is_finite() || min >= max {
        return QuantParams::IDENTITY;
    }
    if (max - min) <= f32::EPSILON {
        // constant tensor (after the 0-inclusion above) — identity is safe
        min = 0.0;
        max = max.max(f32::EPSILON);
    }
    let qmin = target.qmin();
    let qmax = target.qmax();
    let scale = (max - min) / (qmax - qmin) as f32;
    let zp = (qmin as f32 - min / scale).round() as i32;
    let zero_point = zp.clamp(qmin, qmax);
    QuantParams { scale, zero_point }
}

/// Derive **symmetric** parameters for a single tensor (or single
/// channel). Zero-point is always 0 (or 128 for unsigned targets where
/// "symmetric" still needs the midpoint).
///
/// Empty / NaN-only / all-zero inputs return `QuantParams::IDENTITY`.
pub fn derive_symmetric(stat: MinMax, target: QuantTarget) -> QuantParams {
    if stat.is_empty() || stat.min.is_nan() || stat.max.is_nan() {
        return QuantParams::IDENTITY;
    }
    let abs_max = stat.abs_max();
    if !abs_max.is_finite() || abs_max <= f32::EPSILON {
        return QuantParams::IDENTITY;
    }
    let scale = abs_max / target.sym_denom();
    let zero_point = match target {
        QuantTarget::Uint8 => 128, // symmetric mapping into [0,255] centres at 128
        QuantTarget::Int8 | QuantTarget::Int4 => 0,
    };
    QuantParams { scale, zero_point }
}

/// Derive symmetric per-channel parameters from a slice of per-channel
/// `MinMax` aggregates. Returns one `QuantParams` per input. Convention
/// matches ORT / PyTorch: weights are quantized along axis 0 (output
/// channel), so `stats[i]` should aggregate over all weight elements
/// belonging to output channel `i`.
pub fn derive_per_channel_symmetric(stats: &[MinMax], target: QuantTarget) -> Vec<QuantParams> {
    stats.iter().map(|s| derive_symmetric(*s, target)).collect()
}

/// Convenience: per-tensor asymmetric int8 in `[0, 255]`. Standard for
/// activations.
#[inline]
pub fn int8_asymmetric_per_tensor(stat: MinMax) -> QuantParams {
    derive_asymmetric(stat, QuantTarget::Uint8)
}

/// Convenience: per-tensor symmetric int8 in `[-128, 127]`. Used by
/// some PTQ pipelines for activations or for tiny weight tensors that
/// don't benefit from per-channel.
#[inline]
pub fn int8_symmetric_per_tensor(stat: MinMax) -> QuantParams {
    derive_symmetric(stat, QuantTarget::Int8)
}

/// Convenience: per-channel symmetric int8 in `[-128, 127]` for weights.
#[inline]
pub fn int8_symmetric_per_channel(stats: &[MinMax]) -> Vec<QuantParams> {
    derive_per_channel_symmetric(stats, QuantTarget::Int8)
}

/// Convenience: per-channel symmetric int4 in `[-8, 7]`. Pairs with the
/// existing `quantize_weights_int4` packing format.
#[inline]
pub fn int4_symmetric_per_channel(stats: &[MinMax]) -> Vec<QuantParams> {
    derive_per_channel_symmetric(stats, QuantTarget::Int4)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mm(min: f32, max: f32, count: u64) -> MinMax {
        MinMax { min, max, count }
    }

    #[test]
    fn empty_minmax_returns_identity_for_both_schemes() {
        let empty = MinMax::default();
        assert_eq!(
            derive_asymmetric(empty, QuantTarget::Uint8),
            QuantParams::IDENTITY
        );
        assert_eq!(
            derive_symmetric(empty, QuantTarget::Int8),
            QuantParams::IDENTITY
        );
    }

    #[test]
    fn asymmetric_uint8_maps_full_range() {
        // [-1, 1] -> [0, 255], zp ≈ 128, scale ≈ 2/255
        let p = derive_asymmetric(mm(-1.0, 1.0, 100), QuantTarget::Uint8);
        assert!((p.scale - 2.0 / 255.0).abs() < 1e-6);
        assert!((p.zero_point - 128).abs() <= 1);
    }

    #[test]
    fn asymmetric_uint8_one_sided_positive() {
        // [0, 4] -> [0, 255], zp = 0, scale = 4/255
        let p = derive_asymmetric(mm(0.0, 4.0, 16), QuantTarget::Uint8);
        assert!((p.scale - 4.0 / 255.0).abs() < 1e-6);
        assert_eq!(p.zero_point, 0);
    }

    #[test]
    fn asymmetric_uint8_one_sided_negative_pulls_zero_into_range() {
        // [-4, 0] -> [0, 255], zp = 255, scale = 4/255 (zero is still
        // representable as q=255, since we pin 0 inside [min,max]).
        let p = derive_asymmetric(mm(-4.0, 0.0, 16), QuantTarget::Uint8);
        assert!((p.scale - 4.0 / 255.0).abs() < 1e-6);
        assert!(p.zero_point >= 254);
    }

    #[test]
    fn symmetric_int8_uses_abs_max() {
        // |min|=3, |max|=2 -> abs_max=3, scale=3/127
        let p = derive_symmetric(mm(-3.0, 2.0, 10), QuantTarget::Int8);
        assert!((p.scale - 3.0 / 127.0).abs() < 1e-6);
        assert_eq!(p.zero_point, 0);
    }

    #[test]
    fn symmetric_int4_uses_abs_max_with_denom_seven() {
        let p = derive_symmetric(mm(-2.0, 0.5, 4), QuantTarget::Int4);
        assert!((p.scale - 2.0 / 7.0).abs() < 1e-6);
        assert_eq!(p.zero_point, 0);
    }

    #[test]
    fn all_zero_tensor_returns_identity() {
        let p = derive_symmetric(mm(0.0, 0.0, 1000), QuantTarget::Int8);
        assert_eq!(p, QuantParams::IDENTITY);
        let p = derive_asymmetric(mm(0.0, 0.0, 1000), QuantTarget::Uint8);
        assert_eq!(p, QuantParams::IDENTITY);
    }

    #[test]
    fn quantize_dequantize_round_trip_within_one_lsb() {
        // Verify the derived (scale, zp) gives a sane round-trip on
        // the bounds: quant(min) ≈ qmin, quant(max) ≈ qmax.
        let stat = mm(-3.5, 7.5, 1000);
        let p = derive_asymmetric(stat, QuantTarget::Uint8);
        let q_at_min = (stat.min / p.scale).round() as i32 + p.zero_point;
        let q_at_max = (stat.max / p.scale).round() as i32 + p.zero_point;
        assert!(q_at_min.abs() <= 1, "q(min)={q_at_min} should be ≈0");
        assert!(
            (q_at_max - 255).abs() <= 1,
            "q(max)={q_at_max} should be ≈255"
        );
    }

    #[test]
    fn per_channel_symmetric_int8_shapes_to_input_len() {
        let stats = vec![mm(-1.0, 1.0, 4), mm(-10.0, 5.0, 4), mm(0.0, 0.0, 4)];
        let qp = int8_symmetric_per_channel(&stats);
        assert_eq!(qp.len(), 3);
        assert!((qp[0].scale - 1.0 / 127.0).abs() < 1e-6);
        assert!((qp[1].scale - 10.0 / 127.0).abs() < 1e-6);
        // all-zero channel falls back to IDENTITY
        assert_eq!(qp[2], QuantParams::IDENTITY);
    }

    #[test]
    fn per_channel_int4_uses_seven_in_denominator() {
        let stats = vec![mm(-7.0, 7.0, 4), mm(-14.0, 14.0, 4)];
        let qp = int4_symmetric_per_channel(&stats);
        assert!((qp[0].scale - 1.0).abs() < 1e-6); // 7/7 = 1
        assert!((qp[1].scale - 2.0).abs() < 1e-6); // 14/7 = 2
        assert_eq!(qp[0].zero_point, 0);
        assert_eq!(qp[1].zero_point, 0);
    }

    #[test]
    fn nan_min_or_max_falls_back_to_identity() {
        let nan_lo = mm(f32::NAN, 1.0, 5);
        let nan_hi = mm(0.0, f32::NAN, 5);
        assert_eq!(
            derive_asymmetric(nan_lo, QuantTarget::Uint8),
            QuantParams::IDENTITY
        );
        assert_eq!(
            derive_asymmetric(nan_hi, QuantTarget::Uint8),
            QuantParams::IDENTITY
        );
        assert_eq!(
            derive_symmetric(nan_lo, QuantTarget::Int8),
            QuantParams::IDENTITY
        );
        assert_eq!(
            derive_symmetric(nan_hi, QuantTarget::Int8),
            QuantParams::IDENTITY
        );
    }
}
