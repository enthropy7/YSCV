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

/// Lane-wise square root.
///
/// # Safety
/// Caller must be on a NEON target.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn sqrtq_f32(x: float32x4_t) -> float32x4_t {
    std::arch::aarch64::vsqrtq_f32(x)
}

/// See the aarch64 form above. ARMv7 NEON has no square root, but its VFP does —
/// and the two share one register file, where `q0` is `s0..s3`. So the scalar
/// instruction runs on the vector's own lanes: correctly rounded, and with no
/// trip through memory. Pinning `q0` is what makes the lane names line up.
///
/// # Safety
/// Caller must be on a NEON target.
#[cfg(all(target_arch = "arm", feature = "neon-v7"))]
#[inline(always)]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn sqrtq_f32(x: float32x4_t) -> float32x4_t {
    let mut v = x;
    core::arch::asm!(
        "vsqrt.f32 s0, s0",
        "vsqrt.f32 s1, s1",
        "vsqrt.f32 s2, s2",
        "vsqrt.f32 s3, s3",
        inout("q0") v,
        options(pure, nomem, nostack),
    );
    v
}

/// Lane-wise division.
///
/// # Safety
/// Caller must be on a NEON target.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn divq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    std::arch::aarch64::vdivq_f32(a, b)
}

/// See `sqrtq_f32` for why the scalar VFP instruction can read the vector's
/// lanes directly. `q1` is `s4..s7`.
///
/// # Safety
/// Caller must be on a NEON target.
#[cfg(all(target_arch = "arm", feature = "neon-v7"))]
#[inline(always)]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn divq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    let mut q = a;
    core::arch::asm!(
        "vdiv.f32 s0, s0, s4",
        "vdiv.f32 s1, s1, s5",
        "vdiv.f32 s2, s2, s6",
        "vdiv.f32 s3, s3, s7",
        inout("q0") q,
        in("q1") b,
        options(pure, nomem, nostack),
    );
    q
}

/// Sum of the four lanes.
///
/// # Safety
/// Caller must be on a NEON target.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn addvq_f32(v: float32x4_t) -> f32 {
    std::arch::aarch64::vaddvq_f32(v)
}

/// See the aarch64 form above. 32-bit ARM has no across-vector reduction, so
/// fold pairwise — `vpadd` twice pairs the lanes as `(v0+v1)+(v2+v3)`, the order
/// aarch64's `FADDP` pair produces, so the rounding matches.
///
/// # Safety
/// Caller must be on a NEON target.
#[cfg(all(target_arch = "arm", feature = "neon-v7"))]
#[inline(always)]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn addvq_f32(v: float32x4_t) -> f32 {
    use std::arch::arm::*;
    let pairs = vpadd_f32(vget_low_f32(v), vget_high_f32(v));
    vget_lane_f32::<0>(vpadd_f32(pairs, pairs))
}
