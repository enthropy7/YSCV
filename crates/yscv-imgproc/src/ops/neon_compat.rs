//! One spelling for the NEON intrinsics whose ARMv7 form differs from ARMv8's,
//! so a kernel can be gated on both arches and still read as one kernel.
//!
//! Each shim returns exactly what the aarch64 intrinsic returns, so widening a
//! kernel to 32-bit ARM does not move its output.

#[cfg(target_arch = "aarch64")]
pub(crate) use std::arch::aarch64::float32x4_t;
#[cfg(all(target_arch = "arm", feature = "neon-v7"))]
pub(crate) use std::arch::arm::float32x4_t;

/// `[a0, b0, a1, b1]`.
///
/// # Safety
/// Caller must be on a NEON target.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn zip1q_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    std::arch::aarch64::vzip1q_f32(a, b)
}

/// See the aarch64 form above. ARMv7's `VZIP.32` writes both halves of the
/// interleave, so the single-result aarch64 form is its first output.
///
/// # Safety
/// Caller must be on a NEON target.
#[cfg(all(target_arch = "arm", feature = "neon-v7"))]
#[inline(always)]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn zip1q_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    std::arch::arm::vzipq_f32(a, b).0
}

/// `a * b[L]`.
///
/// # Safety
/// Caller must be on a NEON target.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn mulq_lane_f32<const L: i32>(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    std::arch::aarch64::vmulq_laneq_f32::<L>(a, b)
}

/// See the aarch64 form above. The 32-bit lane form takes its scalar from a
/// d-register, so it addresses two lanes; pick the half holding `L` first.
///
/// # Safety
/// Caller must be on a NEON target.
#[cfg(all(target_arch = "arm", feature = "neon-v7"))]
#[inline(always)]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn mulq_lane_f32<const L: i32>(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    use std::arch::arm::*;
    match L {
        0 => vmulq_lane_f32::<0>(a, vget_low_f32(b)),
        1 => vmulq_lane_f32::<1>(a, vget_low_f32(b)),
        2 => vmulq_lane_f32::<0>(a, vget_high_f32(b)),
        _ => vmulq_lane_f32::<1>(a, vget_high_f32(b)),
    }
}

/// Round each lane toward minus infinity.
///
/// # Safety
/// Caller must be on a NEON target.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn rndmq_f32(x: float32x4_t) -> float32x4_t {
    std::arch::aarch64::vrndmq_f32(x)
}

/// See the aarch64 form above. ARMv7 NEON has no rounding instruction at all —
/// `VRINTM` is ARMv8 — so truncate toward zero and step back the negatives that
/// moved up. Exact for every value the resize coordinate math produces.
///
/// # Safety
/// Caller must be on a NEON target.
#[cfg(all(target_arch = "arm", feature = "neon-v7"))]
#[inline(always)]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn rndmq_f32(x: float32x4_t) -> float32x4_t {
    use std::arch::arm::*;
    let trunc = vcvtq_f32_s32(vcvtq_s32_f32(x));
    let overshot = vcgtq_f32(trunc, x);
    let one = vreinterpretq_u32_f32(vdupq_n_f32(1.0));
    vsubq_f32(trunc, vreinterpretq_f32_u32(vandq_u32(overshot, one)))
}
