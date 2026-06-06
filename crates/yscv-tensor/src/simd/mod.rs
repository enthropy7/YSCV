//! SIMD-accelerated primitives for tensor operations.
//!
//! Provides runtime-dispatched SIMD implementations for reductions, binary ops,
//! and in-place operations. Falls back to scalar on unsupported platforms and under miri.

#![allow(unsafe_code)]

// ── FFI declarations ────────────────────────────────────────────────

#[cfg(all(feature = "mkl", any(target_arch = "x86", target_arch = "x86_64")))]
#[allow(dead_code)]
unsafe extern "C" {
    pub(super) fn vsAdd(n: i32, a: *const f32, b: *const f32, y: *mut f32);
    pub(super) fn vsSub(n: i32, a: *const f32, b: *const f32, y: *mut f32);
    pub(super) fn vsMul(n: i32, a: *const f32, b: *const f32, y: *mut f32);
    pub(super) fn vsDiv(n: i32, a: *const f32, b: *const f32, y: *mut f32);
    pub(super) fn vsExp(n: i32, a: *const f32, y: *mut f32);
    pub(super) fn vsSqrt(n: i32, a: *const f32, y: *mut f32);
    pub(super) fn vsLn(n: i32, a: *const f32, y: *mut f32);
}

#[cfg(all(feature = "armpl", target_arch = "aarch64", not(target_os = "macos")))]
#[allow(dead_code)]
unsafe extern "C" {
    pub(super) fn armpl_svexp_f32(n: i32, x: *const f32, y: *mut f32);
    pub(super) fn armpl_svadd_f32(n: i32, a: *const f32, b: *const f32, y: *mut f32);
    pub(super) fn armpl_svsub_f32(n: i32, a: *const f32, b: *const f32, y: *mut f32);
    pub(super) fn armpl_svmul_f32(n: i32, a: *const f32, b: *const f32, y: *mut f32);
    pub(super) fn armpl_svdiv_f32(n: i32, a: *const f32, b: *const f32, y: *mut f32);
    pub(super) fn armpl_svlog_f32(n: i32, x: *const f32, y: *mut f32);
    pub(super) fn armpl_svsqrt_f32(n: i32, x: *const f32, y: *mut f32);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe extern "C" {
    pub(super) fn vvexpf(result: *mut f32, input: *const f32, count: *const i32);
    pub(super) fn vDSP_vadd(
        __A: *const f32,
        __IA: i32,
        __B: *const f32,
        __IB: i32,
        __C: *mut f32,
        __IC: i32,
        __N: u32,
    );
    pub(super) fn vDSP_vsub(
        __B: *const f32,
        __IB: i32,
        __A: *const f32,
        __IA: i32,
        __C: *mut f32,
        __IC: i32,
        __N: u32,
    );
    pub(super) fn vDSP_vmul(
        __A: *const f32,
        __IA: i32,
        __B: *const f32,
        __IB: i32,
        __C: *mut f32,
        __IC: i32,
        __N: u32,
    );
    pub(super) fn vDSP_vdiv(
        __B: *const f32,
        __IB: i32,
        __A: *const f32,
        __IA: i32,
        __C: *mut f32,
        __IC: i32,
        __N: u32,
    );
    pub(super) fn vDSP_vneg(__A: *const f32, __IA: i32, __C: *mut f32, __IC: i32, __N: u32);
}

// ── Arch intrinsic imports ──────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
pub(super) use std::arch::aarch64::{
    vabsq_f32, vaddq_f32, vbslq_f32, vcgtq_f32, vcltq_f32, vdupq_n_f32, vld1q_f32, vmaxq_f32,
    vminq_f32, vmulq_f32, vnegq_f32, vrecpeq_f32, vrecpsq_f32, vrndaq_f32, vrndmq_f32, vrndpq_f32,
    vsqrtq_f32, vst1q_f32, vsubq_f32,
};
#[cfg(target_arch = "x86")]
pub(super) use std::arch::x86::{
    __m128, __m256, _mm_add_ps, _mm_and_ps, _mm_andnot_ps, _mm_castsi128_ps, _mm_cmpgt_ps,
    _mm_cmplt_ps, _mm_cvtepi32_ps, _mm_cvtps_epi32, _mm_loadu_ps, _mm_max_ps, _mm_min_ps,
    _mm_mul_ps, _mm_or_ps, _mm_rcp_ps, _mm_set1_epi32, _mm_set1_ps, _mm_setzero_ps, _mm_sqrt_ps,
    _mm_storeu_ps, _mm_sub_ps, _mm256_add_ps, _mm256_and_ps, _mm256_andnot_ps, _mm256_castsi256_ps,
    _mm256_ceil_ps, _mm256_cmp_ps, _mm256_cvtepi32_ps, _mm256_cvtps_epi32, _mm256_floor_ps,
    _mm256_loadu_ps, _mm256_max_ps, _mm256_min_ps, _mm256_mul_ps, _mm256_or_ps, _mm256_rcp_ps,
    _mm256_set1_epi32, _mm256_set1_ps, _mm256_setzero_ps, _mm256_sqrt_ps, _mm256_storeu_ps,
    _mm256_sub_ps,
};
#[cfg(target_arch = "x86_64")]
pub(super) use std::arch::x86_64::{
    __m128, __m256, _mm_add_ps, _mm_and_ps, _mm_andnot_ps, _mm_castsi128_ps, _mm_cmpgt_ps,
    _mm_cmplt_ps, _mm_cvtepi32_ps, _mm_cvtps_epi32, _mm_loadu_ps, _mm_max_ps, _mm_min_ps,
    _mm_mul_ps, _mm_or_ps, _mm_rcp_ps, _mm_set1_epi32, _mm_set1_ps, _mm_setzero_ps, _mm_sqrt_ps,
    _mm_storeu_ps, _mm_sub_ps, _mm256_add_ps, _mm256_and_ps, _mm256_andnot_ps, _mm256_castsi256_ps,
    _mm256_ceil_ps, _mm256_cmp_ps, _mm256_cvtepi32_ps, _mm256_cvtps_epi32, _mm256_floor_ps,
    _mm256_loadu_ps, _mm256_max_ps, _mm256_min_ps, _mm256_mul_ps, _mm256_or_ps, _mm256_rcp_ps,
    _mm256_set1_epi32, _mm256_set1_ps, _mm256_setzero_ps, _mm256_sqrt_ps, _mm256_storeu_ps,
    _mm256_sub_ps,
};

// ── Sub-modules ─────────────────────────────────────────────────────

mod binary;
mod cmp;
mod inplace;
mod reduce;
mod transcendental;
mod unary;

#[cfg(test)]
mod tests;

// ── Re-exports ──────────────────────────────────────────────────────

pub(crate) use binary::{BinaryKind, binary_dispatch};
pub(crate) use cmp::{CmpKind, clamp_dispatch, cmp_dispatch};
pub(crate) use inplace::{
    add_inplace_dispatch, add_scalar_inplace_dispatch, max_inplace_dispatch,
    mul_scalar_inplace_dispatch, relu_inplace_dispatch,
};
pub(crate) use reduce::{max_dispatch, min_dispatch, sum_dispatch};
#[allow(unused_imports)]
pub(crate) use transcendental::{
    cos_dispatch, exp_dispatch, exp_inplace_dispatch, ln_dispatch, sin_dispatch, tan_dispatch,
};
pub(crate) use unary::{UnaryKind, unary_dispatch};
