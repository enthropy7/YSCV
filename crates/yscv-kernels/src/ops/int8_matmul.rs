//! INT8 × INT8 → INT32 GEMM.
//!
//! Pure dot-product accumulation: `out[i,j] = Σ_k a[i,k] * b[k,j]` with
//! `a`, `b` ∈ `[-128, 127]` and `out` ∈ `i32`. This is the fast path for
//! symmetric quantization (zero-point == 0 on both operands); asymmetric
//! callers must pre-subtract zero-points or use the f32 fallback.
//!
//! Variants (selected at runtime):
//! - **scalar** — i32 += i8 × i8, always available; reference for tests.
//! - **NEON SDOT** — `sdot` aarch64 instruction emitted via `core::arch::asm!`
//!   so we don't depend on the still-nightly `vdotq_s32` intrinsic.
//!   Requires `target_feature = "dotprod"`.
//! - **NEON SMMLA (i8mm)** — `smmla` 2×2 i32 += dot(2×8 i8, 8×2 i8) for
//!   ARMv8.6+ (Apple M1+, Cortex-X1+, Neoverse N2+), also via inline asm.
//! - **AVX-VNNI** — `_mm256_dpbusd_avx_epi32` with `a XOR 0x80`
//!   bias-shift so unsigned-signed VNNI gives the same result as
//!   signed-signed.
//! - **AVX-512-VNNI** — `_mm512_dpbusd_epi32` with the same bias-shift.
//! - **AVX2 widen-mul** — sign-extend i8→i16, `_mm256_madd_epi16`,
//!   accumulate to i32. Correct on every AVX2 CPU; matches scalar
//!   bitwise.
//!
//! Bitwise behaviour is identical across all variants — they differ only
//! in the order of int32 additions, which is associative for true integers
//! (no overflow guard is needed for k ≤ 2³¹/(127*127) ≈ 130k).

#![allow(unsafe_code, unsafe_op_in_unsafe_fn)]

use std::sync::OnceLock;

type Int8MatmulKernel = fn(&[i8], &[i8], usize, usize, usize, &mut [i32]);
type Int8PrepackedKernel = fn(&[i8], &PackedI8B, usize, &mut [i32]);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Int8MatmulPath {
    #[cfg(target_arch = "x86_64")]
    Avx512Vnni,
    #[cfg(target_arch = "x86_64")]
    AvxVnni,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "aarch64")]
    NeonI8mm,
    #[cfg(target_arch = "aarch64")]
    NeonDotprod,
    /// Plain ARMv8.0-A NEON widening dot (`vmull_s8` + `vpadalq_s16`).
    /// Covers A53-class cores that have neither the dot-product
    /// (ARMv8.2 `dotprod`) nor `i8mm` extensions, where the only
    /// alternative is scalar. ~8× the scalar throughput.
    #[cfg(target_arch = "aarch64")]
    NeonWiden,
    Scalar,
}

impl Int8MatmulPath {
    #[inline]
    fn name(self) -> &'static str {
        match self {
            #[cfg(target_arch = "x86_64")]
            Int8MatmulPath::Avx512Vnni => "avx512-vnni",
            #[cfg(target_arch = "x86_64")]
            Int8MatmulPath::AvxVnni => "avx-vnni",
            #[cfg(target_arch = "x86_64")]
            Int8MatmulPath::Avx2 => "avx2",
            #[cfg(target_arch = "aarch64")]
            Int8MatmulPath::NeonI8mm => "neon-i8mm",
            #[cfg(target_arch = "aarch64")]
            Int8MatmulPath::NeonDotprod => "neon-dotprod",
            #[cfg(target_arch = "aarch64")]
            Int8MatmulPath::NeonWiden => "neon-widen",
            Int8MatmulPath::Scalar => "scalar",
        }
    }
}

pub(crate) fn int8_matmul_dispatch_path() -> &'static str {
    select_int8_matmul_path().name()
}

pub(crate) fn int8_prepacked_dispatch_path() -> &'static str {
    select_int8_prepacked_path().name()
}

/// Transpose `b` from row-major `[K × N]` into row-major `[N × K]` so
/// the inner kernel can issue a contiguous K-vector load per output
/// column instead of a strided per-element gather. The transpose itself
/// is O(K·N) and happens once per call — for typical M·K·N with M ≥ 8
/// the saved cache misses pay back the transpose by 1-2 orders of
/// magnitude.
fn transpose_b(b: &[i8], k: usize, n: usize) -> Vec<i8> {
    let mut bt = vec![0_i8; n * k];
    // 8×8 blocking improves throughput vs the naive scalar loop without
    // pulling in arch-specific intrinsics for the transpose itself.
    let bs = 8;
    let kb_full = (k / bs) * bs;
    let nb_full = (n / bs) * bs;
    for kk in (0..kb_full).step_by(bs) {
        for jj in (0..nb_full).step_by(bs) {
            for r in 0..bs {
                for c in 0..bs {
                    bt[(jj + c) * k + kk + r] = b[(kk + r) * n + jj + c];
                }
            }
        }
        for jj in nb_full..n {
            for r in 0..bs {
                bt[jj * k + kk + r] = b[(kk + r) * n + jj];
            }
        }
    }
    for kk in kb_full..k {
        for jj in 0..n {
            bt[jj * k + kk] = b[kk * n + jj];
        }
    }
    bt
}

/// Load-time packed INT8 GEMM RHS.
///
/// Public `int8_matmul_dispatch` accepts `b` as row-major `[K, N]` and
/// transposes it internally because every SIMD backend wants contiguous
/// K-lanes for each output column. That is correct for one-off calls but
/// wasteful for inference graphs where weights are constant. This type
/// stores that transposed `[N, K]` layout once at model load so hot-path
/// QLinearConv/QLinearMatMul can reuse it without heap work or weight
/// repacking per inference.
#[derive(Debug, Clone)]
pub struct PackedI8B {
    k: usize,
    n: usize,
    bt: Vec<i8>,
    #[cfg(target_arch = "x86_64")]
    vnni_4x16: Option<PackedI8BVnni4x16>,
}

#[cfg(target_arch = "x86_64")]
#[derive(Debug, Clone)]
struct PackedI8BVnni4x16 {
    k4_full: usize,
    bp: Vec<i8>,
    col_sum_b_vnni: Vec<i32>,
}

impl PackedI8B {
    #[inline]
    pub fn k(&self) -> usize {
        self.k
    }

    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    #[inline]
    pub fn transposed(&self) -> &[i8] {
        &self.bt
    }
}

/// Pack row-major `b` (`[K, N]`) for repeated INT8 GEMM calls.
pub fn pack_i8_b_for_matmul(b: &[i8], k: usize, n: usize) -> PackedI8B {
    debug_assert_eq!(b.len(), k * n);
    #[cfg(target_arch = "x86_64")]
    let vnni_4x16 = if n.is_multiple_of(16) && k >= 4 {
        let (bp, k4_full) = pack_b_vnni_4x16(b, k, n);
        let mut col_sum_b_vnni = vec![0_i32; n];
        for j in 0..n {
            let mut s: i32 = 0;
            for kk in 0..k4_full {
                s += b[kk * n + j] as i32;
            }
            col_sum_b_vnni[j] = s;
        }
        Some(PackedI8BVnni4x16 {
            k4_full,
            bp,
            col_sum_b_vnni,
        })
    } else {
        None
    };
    PackedI8B {
        k,
        n,
        bt: transpose_b(b, k, n),
        #[cfg(target_arch = "x86_64")]
        vnni_4x16,
    }
}

/// Scalar reference. Always correct; use for tests and as fallback when
/// no SIMD path is detected at runtime.
pub fn int8_matmul_scalar(a: &[i8], b: &[i8], m: usize, k: usize, n: usize, out: &mut [i32]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(out.len(), m * n);
    for i in 0..m {
        for j in 0..n {
            let mut acc: i32 = 0;
            for kk in 0..k {
                acc += (a[i * k + kk] as i32) * (b[kk * n + j] as i32);
            }
            out[i * n + j] = acc;
        }
    }
}

/// Scalar reference for prepacked RHS. Always correct; SIMD variants
/// below must match this bit-for-bit.
pub fn int8_matmul_prepacked_scalar(a: &[i8], b: &PackedI8B, m: usize, out: &mut [i32]) {
    let k = b.k;
    let n = b.n;
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(out.len(), m * n);
    let bt = b.transposed();
    for i in 0..m {
        for j in 0..n {
            let mut acc: i32 = 0;
            for kk in 0..k {
                acc += (a[i * k + kk] as i32) * (bt[j * k + kk] as i32);
            }
            out[i * n + j] = acc;
        }
    }
}

/// Issue the aarch64 `sdot Vd.4S, Vn.16B, Vm.16B` instruction. Each
/// i32 lane in `acc` accumulates the dot product of the corresponding
/// 4-byte slice of `a` with the matching slice of `b`. Inline asm
/// because the matching `vdotq_s32` intrinsic is still nightly-only.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn sdot_inline(
    acc: std::arch::aarch64::int32x4_t,
    a: std::arch::aarch64::int8x16_t,
    b: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    use std::arch::aarch64::int32x4_t;
    let mut out: int32x4_t = acc;
    std::arch::asm!(
        "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
        acc = inout(vreg) out,
        a = in(vreg) a,
        b = in(vreg) b,
        options(pure, nomem, nostack)
    );
    out
}

/// Issue the aarch64 `smmla Vd.4S, Vn.16B, Vm.16B` (i8mm) instruction.
/// Computes `acc += [A0·B0, A0·B1, A1·B0, A1·B1]` where Ai/Bj are the
/// two 8-byte halves of `a` and `b`.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon,i8mm")]
unsafe fn smmla_inline(
    acc: std::arch::aarch64::int32x4_t,
    a: std::arch::aarch64::int8x16_t,
    b: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    use std::arch::aarch64::int32x4_t;
    let mut out: int32x4_t = acc;
    std::arch::asm!(
        "smmla {acc:v}.4s, {a:v}.16b, {b:v}.16b",
        acc = inout(vreg) out,
        a = in(vreg) a,
        b = in(vreg) b,
        options(pure, nomem, nostack)
    );
    out
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,i8mm")]
unsafe fn int8_matmul_neon_i8mm(a: &[i8], b: &[i8], m: usize, k: usize, n: usize, out: &mut [i32]) {
    use std::arch::aarch64::*;
    // SMMLA: acc[0..4] += [A0·B0, A0·B1, A1·B0, A1·B1] where Ai is an
    // 8-byte row slice and Bj is an 8-byte column slice. Tile 2×2 in
    // (M, N); odd-edge rows/cols stay scalar so this kernel only requires
    // NEON+i8mm, not dotprod.
    let m2 = m & !1;
    let n2 = n & !1;
    let k8 = k & !7;
    for i in (0..m2).step_by(2) {
        for j in (0..n2).step_by(2) {
            let mut acc = vdupq_n_s32(0);
            let mut kk = 0;
            while kk + 8 <= k8 {
                let mut abuf = [0_i8; 16];
                for r in 0..8 {
                    abuf[r] = a[i * k + kk + r];
                    abuf[8 + r] = a[(i + 1) * k + kk + r];
                }
                let av = vld1q_s8(abuf.as_ptr());
                let mut bbuf = [0_i8; 16];
                for r in 0..8 {
                    bbuf[r] = b[(kk + r) * n + j];
                    bbuf[8 + r] = b[(kk + r) * n + j + 1];
                }
                let bv = vld1q_s8(bbuf.as_ptr());
                acc = smmla_inline(acc, av, bv);
                kk += 8;
            }
            let mut buf = [0_i32; 4];
            vst1q_s32(buf.as_mut_ptr(), acc);
            for di in 0..2 {
                for dj in 0..2 {
                    let mut tail = buf[di * 2 + dj];
                    let mut kt = kk;
                    while kt < k {
                        tail += (a[(i + di) * k + kt] as i32) * (b[kt * n + j + dj] as i32);
                        kt += 1;
                    }
                    out[(i + di) * n + j + dj] = tail;
                }
            }
        }
    }
    if m2 < m || n2 < n {
        for i in 0..m {
            for j in 0..n {
                if i < m2 && j < n2 {
                    continue;
                }
                let mut acc = 0_i32;
                for kk in 0..k {
                    acc += (a[i * k + kk] as i32) * (b[kk * n + j] as i32);
                }
                out[i * n + j] = acc;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,i8mm")]
unsafe fn int8_matmul_prepacked_neon_i8mm(a: &[i8], b: &PackedI8B, m: usize, out: &mut [i32]) {
    use std::arch::aarch64::*;
    let k = b.k;
    let n = b.n;
    let bt = b.transposed();
    let m2 = m & !1;
    let n2 = n & !1;
    let k8 = k & !7;
    for i in (0..m2).step_by(2) {
        for j in (0..n2).step_by(2) {
            let mut acc = vdupq_n_s32(0);
            let mut kk = 0;
            while kk + 8 <= k8 {
                let mut abuf = [0_i8; 16];
                for r in 0..8 {
                    abuf[r] = a[i * k + kk + r];
                    abuf[8 + r] = a[(i + 1) * k + kk + r];
                }
                let av = vld1q_s8(abuf.as_ptr());
                let mut bbuf = [0_i8; 16];
                bbuf[..8].copy_from_slice(&bt[j * k + kk..j * k + kk + 8]);
                bbuf[8..].copy_from_slice(&bt[(j + 1) * k + kk..(j + 1) * k + kk + 8]);
                let bv = vld1q_s8(bbuf.as_ptr());
                acc = smmla_inline(acc, av, bv);
                kk += 8;
            }
            let mut buf = [0_i32; 4];
            vst1q_s32(buf.as_mut_ptr(), acc);
            for di in 0..2 {
                for dj in 0..2 {
                    let mut tail = buf[di * 2 + dj];
                    let mut kt = kk;
                    while kt < k {
                        tail += (a[(i + di) * k + kt] as i32) * (bt[(j + dj) * k + kt] as i32);
                        kt += 1;
                    }
                    out[(i + di) * n + j + dj] = tail;
                }
            }
        }
    }
    if m2 < m || n2 < n {
        for i in 0..m {
            for j in 0..n {
                if i < m2 && j < n2 {
                    continue;
                }
                let mut acc = 0_i32;
                for kk in 0..k {
                    acc += (a[i * k + kk] as i32) * (bt[j * k + kk] as i32);
                }
                out[i * n + j] = acc;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn int8_matmul_prepacked_neon_sdot(a: &[i8], b: &PackedI8B, m: usize, out: &mut [i32]) {
    use std::arch::aarch64::*;
    let k = b.k;
    let n = b.n;
    let bt = b.transposed();
    for i in 0..m {
        for j in 0..n {
            let mut acc = vdupq_n_s32(0);
            let mut kk = 0;
            while kk + 16 <= k {
                let av = vld1q_s8(a.as_ptr().add(i * k + kk));
                let bv = vld1q_s8(bt.as_ptr().add(j * k + kk));
                acc = sdot_inline(acc, av, bv);
                kk += 16;
            }
            let mut tail = vaddvq_s32(acc);
            while kk < k {
                tail += (a[i * k + kk] as i32) * (bt[j * k + kk] as i32);
                kk += 1;
            }
            out[i * n + j] = tail;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn int8_matmul_neon_sdot(a: &[i8], b: &[i8], m: usize, k: usize, n: usize, out: &mut [i32]) {
    use std::arch::aarch64::*;
    let k4 = k & !3;
    for i in 0..m {
        for j in 0..n {
            let mut acc = vdupq_n_s32(0);
            let mut kk = 0;
            while kk + 16 <= k4 {
                let av = vld1q_s8(a.as_ptr().add(i * k + kk));
                let mut bbuf = [0_i8; 16];
                for r in 0..16 {
                    bbuf[r] = b[(kk + r) * n + j];
                }
                let bv = vld1q_s8(bbuf.as_ptr());
                acc = sdot_inline(acc, av, bv);
                kk += 16;
            }
            let mut tail = vaddvq_s32(acc);
            while kk < k {
                tail += (a[i * k + kk] as i32) * (b[kk * n + j] as i32);
                kk += 1;
            }
            out[i * n + j] = tail;
        }
    }
}

/// One column of the widening dot: `a_row · bt_col` over K, via `vmull_s8`
/// (8×8→16) + `vpadalq_s16` (pairwise-accumulate into i32). `bt_col` is
/// contiguous over K.
///
/// SAFETY: `ap` and `bp` must each be valid for `k` i8 reads.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn widen_dot_1col(ap: *const i8, bp: *const i8, k: usize) -> i32 {
    use std::arch::aarch64::*;
    unsafe {
        let mut acc = vdupq_n_s32(0);
        let mut kk = 0;
        while kk + 16 <= k {
            let av = vld1q_s8(ap.add(kk));
            let bv = vld1q_s8(bp.add(kk));
            acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(av), vget_low_s8(bv)));
            acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(av), vget_high_s8(bv)));
            kk += 16;
        }
        if kk + 8 <= k {
            acc = vpadalq_s16(acc, vmull_s8(vld1_s8(ap.add(kk)), vld1_s8(bp.add(kk))));
            kk += 8;
        }
        let mut s = vaddvq_s32(acc);
        while kk < k {
            s += (*ap.add(kk) as i32) * (*bp.add(kk) as i32);
            kk += 1;
        }
        s
    }
}

// Hand-scheduled aarch64 int8 GEMM microkernel: a 4×8 tile using SMLAL/SMLAL2
// lane-broadcast, accumulating straight into per-column int32 lanes (no
// horizontal reduction — the small-K killer of the widening kernel). Reimplements
// the XNNPACK qs8 mlal-lane algorithm as our own assembly (not a copy of their
// source): B is pre-widened to i16 k-major so its 8 rows for a k-chunk load with
// plain `ld1`, and all 8 B rows load up front so the 64 SMLAL flow across 8
// independent accumulator chains. Beats the widening kernel by up to ~1.7× on the
// small-K (c_in ≤ 32) MobileNet pointwise convs; ties it on large K.
//
// Branch targets in all three blocks below are numeric locals: Mach-O does not
// treat a `.L` name as assembler-local, so a conditional branch to one is
// rejected outright when the same source is assembled for Darwin.
#[cfg(target_arch = "aarch64")]
core::arch::global_asm!(
    r#"
    .text
    .align 4
    .global yscv_mlal4x8_kernel
yscv_mlal4x8_kernel:
    movi v16.4s, #0
    movi v17.4s, #0
    movi v18.4s, #0
    movi v19.4s, #0
    movi v20.4s, #0
    movi v21.4s, #0
    movi v22.4s, #0
    movi v23.4s, #0
    add  x7, x0, x1
    add  x8, x7, x1
    add  x9, x8, x1
    lsr  x10, x6, #3
2:
    ld1  {{v0.8b}}, [x0], #8
    ld1  {{v1.8b}}, [x7], #8
    ld1  {{v2.8b}}, [x8], #8
    ld1  {{v3.8b}}, [x9], #8
    sxtl v0.8h, v0.8b
    sxtl v1.8h, v1.8b
    sxtl v2.8h, v2.8b
    sxtl v3.8h, v3.8b
    mov  x11, x2
    ld1  {{v24.8h}}, [x11], x3
    ld1  {{v25.8h}}, [x11], x3
    ld1  {{v26.8h}}, [x11], x3
    ld1  {{v27.8h}}, [x11], x3
    ld1  {{v28.8h}}, [x11], x3
    ld1  {{v29.8h}}, [x11], x3
    ld1  {{v30.8h}}, [x11], x3
    ld1  {{v31.8h}}, [x11], x3
    smlal  v16.4s, v24.4h, v0.h[0]
    smlal2 v17.4s, v24.8h, v0.h[0]
    smlal  v18.4s, v24.4h, v1.h[0]
    smlal2 v19.4s, v24.8h, v1.h[0]
    smlal  v20.4s, v24.4h, v2.h[0]
    smlal2 v21.4s, v24.8h, v2.h[0]
    smlal  v22.4s, v24.4h, v3.h[0]
    smlal2 v23.4s, v24.8h, v3.h[0]
    smlal  v16.4s, v25.4h, v0.h[1]
    smlal2 v17.4s, v25.8h, v0.h[1]
    smlal  v18.4s, v25.4h, v1.h[1]
    smlal2 v19.4s, v25.8h, v1.h[1]
    smlal  v20.4s, v25.4h, v2.h[1]
    smlal2 v21.4s, v25.8h, v2.h[1]
    smlal  v22.4s, v25.4h, v3.h[1]
    smlal2 v23.4s, v25.8h, v3.h[1]
    smlal  v16.4s, v26.4h, v0.h[2]
    smlal2 v17.4s, v26.8h, v0.h[2]
    smlal  v18.4s, v26.4h, v1.h[2]
    smlal2 v19.4s, v26.8h, v1.h[2]
    smlal  v20.4s, v26.4h, v2.h[2]
    smlal2 v21.4s, v26.8h, v2.h[2]
    smlal  v22.4s, v26.4h, v3.h[2]
    smlal2 v23.4s, v26.8h, v3.h[2]
    smlal  v16.4s, v27.4h, v0.h[3]
    smlal2 v17.4s, v27.8h, v0.h[3]
    smlal  v18.4s, v27.4h, v1.h[3]
    smlal2 v19.4s, v27.8h, v1.h[3]
    smlal  v20.4s, v27.4h, v2.h[3]
    smlal2 v21.4s, v27.8h, v2.h[3]
    smlal  v22.4s, v27.4h, v3.h[3]
    smlal2 v23.4s, v27.8h, v3.h[3]
    smlal  v16.4s, v28.4h, v0.h[4]
    smlal2 v17.4s, v28.8h, v0.h[4]
    smlal  v18.4s, v28.4h, v1.h[4]
    smlal2 v19.4s, v28.8h, v1.h[4]
    smlal  v20.4s, v28.4h, v2.h[4]
    smlal2 v21.4s, v28.8h, v2.h[4]
    smlal  v22.4s, v28.4h, v3.h[4]
    smlal2 v23.4s, v28.8h, v3.h[4]
    smlal  v16.4s, v29.4h, v0.h[5]
    smlal2 v17.4s, v29.8h, v0.h[5]
    smlal  v18.4s, v29.4h, v1.h[5]
    smlal2 v19.4s, v29.8h, v1.h[5]
    smlal  v20.4s, v29.4h, v2.h[5]
    smlal2 v21.4s, v29.8h, v2.h[5]
    smlal  v22.4s, v29.4h, v3.h[5]
    smlal2 v23.4s, v29.8h, v3.h[5]
    smlal  v16.4s, v30.4h, v0.h[6]
    smlal2 v17.4s, v30.8h, v0.h[6]
    smlal  v18.4s, v30.4h, v1.h[6]
    smlal2 v19.4s, v30.8h, v1.h[6]
    smlal  v20.4s, v30.4h, v2.h[6]
    smlal2 v21.4s, v30.8h, v2.h[6]
    smlal  v22.4s, v30.4h, v3.h[6]
    smlal2 v23.4s, v30.8h, v3.h[6]
    smlal  v16.4s, v31.4h, v0.h[7]
    smlal2 v17.4s, v31.8h, v0.h[7]
    smlal  v18.4s, v31.4h, v1.h[7]
    smlal2 v19.4s, v31.8h, v1.h[7]
    smlal  v20.4s, v31.4h, v2.h[7]
    smlal2 v21.4s, v31.8h, v2.h[7]
    smlal  v22.4s, v31.4h, v3.h[7]
    smlal2 v23.4s, v31.8h, v3.h[7]
    add  x2, x2, x3, lsl #3
    subs x10, x10, #1
    b.ne 2b
    add  x12, x4, x5
    add  x13, x12, x5
    add  x14, x13, x5
    stp  q16, q17, [x4]
    stp  q18, q19, [x12]
    stp  q20, q21, [x13]
    stp  q22, q23, [x14]
    ret
"#
);

#[cfg(target_arch = "aarch64")]
unsafe extern "C" {
    fn yscv_mlal4x8_kernel(
        a: *const i8,
        a_stride: usize,
        b16: *const i16,
        b_stride: usize,
        out: *mut i32,
        out_stride: usize,
        k: usize,
    );
}

// Hand-scheduled 4×4 i8 GEMM tile for the widening (no-dotprod) path. `a` rows
// and `bt` "rows" (= output columns) are both K-contiguous i8. Each 16-K step
// pairs the low/high 8-lane halves (`smull` + `smlal2` into i16, `sadalp` into
// i32) — the c2 accumulation that gives INT8 its only edge over f32 FMLA on
// ARMv8.0. 16 i32 accumulators live in v16..v31 with zero spills (LLVM spills
// them, which is why this beats the compiled `neon_widen_gemm` core by ~1.3×);
// operands in v0..v7, products in v8..v11 (d8..d11 saved per the ABI). Requires
// k % 8 == 0 and k >= 8, and is only overflow-safe when no weight is i8::MIN
// (then each product is |a·b| ≤ 128·127 = 16256, two per i16 lane = 32512 <
// 32767). This reimplements the XNNPACK qs8 mlal-c2 technique as our own
// scheduling, not a copy of their assembly.
#[cfg(target_arch = "aarch64")]
core::arch::global_asm!(
    r#"
    .text
    .align 4
    .global yscv_gemm4x4_i8
yscv_gemm4x4_i8:
    stp  d8, d9, [sp, #-32]!
    stp  d10, d11, [sp, #16]
    add  x7,  x0, x1
    add  x8,  x7, x1
    add  x9,  x8, x1
    add  x10, x2, x3
    add  x11, x10, x3
    add  x12, x11, x3
    movi v16.4s, #0
    movi v17.4s, #0
    movi v18.4s, #0
    movi v19.4s, #0
    movi v20.4s, #0
    movi v21.4s, #0
    movi v22.4s, #0
    movi v23.4s, #0
    movi v24.4s, #0
    movi v25.4s, #0
    movi v26.4s, #0
    movi v27.4s, #0
    movi v28.4s, #0
    movi v29.4s, #0
    movi v30.4s, #0
    movi v31.4s, #0
    lsr  x14, x4, #4
    cbz  x14, 3f
2:
    ld1  {{v0.16b}}, [x0], #16
    ld1  {{v1.16b}}, [x7], #16
    ld1  {{v2.16b}}, [x8], #16
    ld1  {{v3.16b}}, [x9], #16
    ld1  {{v4.16b}}, [x2], #16
    ld1  {{v5.16b}}, [x10], #16
    ld1  {{v6.16b}}, [x11], #16
    ld1  {{v7.16b}}, [x12], #16
    smull  v8.8h,  v0.8b, v4.8b
    smull  v9.8h,  v0.8b, v5.8b
    smull  v10.8h, v0.8b, v6.8b
    smull  v11.8h, v0.8b, v7.8b
    smlal2 v8.8h,  v0.16b, v4.16b
    smlal2 v9.8h,  v0.16b, v5.16b
    smlal2 v10.8h, v0.16b, v6.16b
    smlal2 v11.8h, v0.16b, v7.16b
    sadalp v16.4s, v8.8h
    sadalp v17.4s, v9.8h
    sadalp v18.4s, v10.8h
    sadalp v19.4s, v11.8h
    smull  v8.8h,  v1.8b, v4.8b
    smull  v9.8h,  v1.8b, v5.8b
    smull  v10.8h, v1.8b, v6.8b
    smull  v11.8h, v1.8b, v7.8b
    smlal2 v8.8h,  v1.16b, v4.16b
    smlal2 v9.8h,  v1.16b, v5.16b
    smlal2 v10.8h, v1.16b, v6.16b
    smlal2 v11.8h, v1.16b, v7.16b
    sadalp v20.4s, v8.8h
    sadalp v21.4s, v9.8h
    sadalp v22.4s, v10.8h
    sadalp v23.4s, v11.8h
    smull  v8.8h,  v2.8b, v4.8b
    smull  v9.8h,  v2.8b, v5.8b
    smull  v10.8h, v2.8b, v6.8b
    smull  v11.8h, v2.8b, v7.8b
    smlal2 v8.8h,  v2.16b, v4.16b
    smlal2 v9.8h,  v2.16b, v5.16b
    smlal2 v10.8h, v2.16b, v6.16b
    smlal2 v11.8h, v2.16b, v7.16b
    sadalp v24.4s, v8.8h
    sadalp v25.4s, v9.8h
    sadalp v26.4s, v10.8h
    sadalp v27.4s, v11.8h
    smull  v8.8h,  v3.8b, v4.8b
    smull  v9.8h,  v3.8b, v5.8b
    smull  v10.8h, v3.8b, v6.8b
    smull  v11.8h, v3.8b, v7.8b
    smlal2 v8.8h,  v3.16b, v4.16b
    smlal2 v9.8h,  v3.16b, v5.16b
    smlal2 v10.8h, v3.16b, v6.16b
    smlal2 v11.8h, v3.16b, v7.16b
    sadalp v28.4s, v8.8h
    sadalp v29.4s, v9.8h
    sadalp v30.4s, v10.8h
    sadalp v31.4s, v11.8h
    subs x14, x14, #1
    b.ne 2b
3:
    tst  x4, #8
    b.eq 4f
    ld1  {{v0.8b}}, [x0]
    ld1  {{v1.8b}}, [x7]
    ld1  {{v2.8b}}, [x8]
    ld1  {{v3.8b}}, [x9]
    ld1  {{v4.8b}}, [x2]
    ld1  {{v5.8b}}, [x10]
    ld1  {{v6.8b}}, [x11]
    ld1  {{v7.8b}}, [x12]
    smull v8.8h,  v0.8b, v4.8b
    smull v9.8h,  v0.8b, v5.8b
    smull v10.8h, v0.8b, v6.8b
    smull v11.8h, v0.8b, v7.8b
    sadalp v16.4s, v8.8h
    sadalp v17.4s, v9.8h
    sadalp v18.4s, v10.8h
    sadalp v19.4s, v11.8h
    smull v8.8h,  v1.8b, v4.8b
    smull v9.8h,  v1.8b, v5.8b
    smull v10.8h, v1.8b, v6.8b
    smull v11.8h, v1.8b, v7.8b
    sadalp v20.4s, v8.8h
    sadalp v21.4s, v9.8h
    sadalp v22.4s, v10.8h
    sadalp v23.4s, v11.8h
    smull v8.8h,  v2.8b, v4.8b
    smull v9.8h,  v2.8b, v5.8b
    smull v10.8h, v2.8b, v6.8b
    smull v11.8h, v2.8b, v7.8b
    sadalp v24.4s, v8.8h
    sadalp v25.4s, v9.8h
    sadalp v26.4s, v10.8h
    sadalp v27.4s, v11.8h
    smull v8.8h,  v3.8b, v4.8b
    smull v9.8h,  v3.8b, v5.8b
    smull v10.8h, v3.8b, v6.8b
    smull v11.8h, v3.8b, v7.8b
    sadalp v28.4s, v8.8h
    sadalp v29.4s, v9.8h
    sadalp v30.4s, v10.8h
    sadalp v31.4s, v11.8h
4:
    add  x14, x5, x6, lsl #2
    add  x15, x14, x6, lsl #2
    add  x16, x15, x6, lsl #2
    addv s0, v16.4s
    addv s1, v17.4s
    addv s2, v18.4s
    addv s3, v19.4s
    str  s0, [x5]
    str  s1, [x5, #4]
    str  s2, [x5, #8]
    str  s3, [x5, #12]
    addv s0, v20.4s
    addv s1, v21.4s
    addv s2, v22.4s
    addv s3, v23.4s
    str  s0, [x14]
    str  s1, [x14, #4]
    str  s2, [x14, #8]
    str  s3, [x14, #12]
    addv s0, v24.4s
    addv s1, v25.4s
    addv s2, v26.4s
    addv s3, v27.4s
    str  s0, [x15]
    str  s1, [x15, #4]
    str  s2, [x15, #8]
    str  s3, [x15, #12]
    addv s0, v28.4s
    addv s1, v29.4s
    addv s2, v30.4s
    addv s3, v31.4s
    str  s0, [x16]
    str  s1, [x16, #4]
    str  s2, [x16, #8]
    str  s3, [x16, #12]
    ldp  d10, d11, [sp, #16]
    ldp  d8, d9, [sp], #32
    ret
"#
);

#[cfg(target_arch = "aarch64")]
unsafe extern "C" {
    fn yscv_gemm4x4_i8(
        a0: *const i8,
        a_stride: usize,
        b0: *const i8,
        b_stride: usize,
        k: usize,
        out: *mut i32,
        n: usize,
    );
}

/// Driver for [`yscv_gemm4x4_i8`]: `a` is `[M,K]` row-major, `bt` is `[N,K]`
/// (K-contiguous per output column). Tiles the 4×4 core; N%4 columns fall to
/// [`widen_dot_1col`] and M%4 rows to a scalar dot — all bit-identical to the
/// widening kernel. Caller guarantees `k % 8 == 0`, `k >= 8`, and that `bt`
/// contains no `i8::MIN` (overflow safety of the paired i16 accumulation).
///
/// SAFETY: `a.len() >= m*k`, `bt.len() >= n*k`, `out.len() >= m*n`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_gemm4x4_asm(a: &[i8], bt: &[i8], m: usize, k: usize, n: usize, out: &mut [i32]) {
    unsafe {
        let ab = a.as_ptr();
        let bb = bt.as_ptr();
        let n4 = n & !3;
        let mut i = 0;
        while i + 4 <= m {
            let mut j = 0;
            while j < n4 {
                yscv_gemm4x4_i8(
                    ab.add(i * k),
                    k,
                    bb.add(j * k),
                    k,
                    k,
                    out.as_mut_ptr().add(i * n + j),
                    n,
                );
                j += 4;
            }
            // N remainder: one output column at a time, four rows.
            while j < n {
                let bpj = bb.add(j * k);
                for r in 0..4 {
                    out[(i + r) * n + j] = widen_dot_1col(ab.add((i + r) * k), bpj, k);
                }
                j += 1;
            }
            i += 4;
        }
        // M remainder rows.
        while i < m {
            let ap = ab.add(i * k);
            for j in 0..n {
                out[i * n + j] = widen_dot_1col(ap, bb.add(j * k), k);
            }
            i += 1;
        }
    }
}

/// Small-K int8 GEMM via the hand-scheduled 4×8 `yscv_mlal4x8_kernel`. `a` is
/// `[M,K]` row-major, `b` is `[K,N]` k-major (the layout the non-prepacked
/// caller already builds). Pads K to a multiple of 8 with zeros and pre-widens
/// B to i16 k-major, then tiles 4×8 with scalar M/N remainders. Bit-identical
/// to the scalar reference (integer, unchanged k-order; zero pads contribute 0).
///
/// SAFETY: `a.len() >= m*k`, `b.len() >= k*n`, `out.len() >= m*n`.
#[cfg(target_arch = "aarch64")]
unsafe fn neon_mlal_lane_gemm(a: &[i8], b: &[i8], m: usize, k: usize, n: usize, out: &mut [i32]) {
    let kp = k.div_ceil(8) * 8;
    let mut ap = vec![0i8; m * kp];
    for i in 0..m {
        ap[i * kp..i * kp + k].copy_from_slice(&a[i * k..i * k + k]);
    }
    let mut b16 = vec![0i16; kp * n];
    for (dst, &v) in b16[..k * n].iter_mut().zip(&b[..k * n]) {
        *dst = v as i16;
    }
    let mt = m & !3;
    let nt = n & !7;
    for i in (0..mt).step_by(4) {
        for j in (0..nt).step_by(8) {
            unsafe {
                yscv_mlal4x8_kernel(
                    ap.as_ptr().add(i * kp),
                    kp,
                    b16.as_ptr().add(j),
                    n * 2,
                    out.as_mut_ptr().add(i * n + j),
                    n * 4,
                    kp,
                );
            }
        }
        for j in nt..n {
            for r in 0..4 {
                let mut s = 0i32;
                for kk in 0..k {
                    s += a[(i + r) * k + kk] as i32 * b[kk * n + j] as i32;
                }
                out[(i + r) * n + j] = s;
            }
        }
    }
    for i in mt..m {
        for j in 0..n {
            let mut s = 0i32;
            for kk in 0..k {
                s += a[i * k + kk] as i32 * b[kk * n + j] as i32;
            }
            out[i * n + j] = s;
        }
    }
}

// Hand-asm 4×16 mlal-LANE int8 GEMM tile (XNNPACK method, our scheduling). 16
// i32 accumulators (4 rows × 4 col-blocks) in v16..v31; A widened to i16 in
// v0..v3; per k-lane load 16 B cols (v4..v7 as .4h) and issue 16 `SMLAL
// Vacc.4S, Vb.4H, Va.H[lane]` — accumulate straight into output columns, NO
// horizontal reduction (the small-K killer in the c2 `sadalp` core). 8 k-lanes
// per chunk, B prefetched one chunk ahead. Beats the c2 4×4 core by 1.2–1.6× on
// K≤48 (stem/early pointwise) on the A53: stem 2.11→3.28, K=24 2.61→3.63 GMAC/s.
// Requires k % 8 == 0 (caller zero-pads); bit-identical to the scalar reference.
// Args (AAPCS64): x0=a0 (i8, padded kp), x1=lda (=kp bytes), x2=b16 (i16 at col
// j), x3=n (elems, B k-stride), x4=kp, x5=out (i32 row0), x6=ldo (bytes).
#[cfg(target_arch = "aarch64")]
core::arch::global_asm!(
    r#"
    .text
    .align 4
    .global yscv_mlal4x16
yscv_mlal4x16:
    add  x7,  x0, x1
    add  x8,  x7, x1
    add  x9,  x8, x1
    add  x10, x5, x6
    add  x11, x10, x6
    add  x12, x11, x6
    lsl  x13, x3, #1
    movi v16.4s, #0
    movi v17.4s, #0
    movi v18.4s, #0
    movi v19.4s, #0
    movi v20.4s, #0
    movi v21.4s, #0
    movi v22.4s, #0
    movi v23.4s, #0
    movi v24.4s, #0
    movi v25.4s, #0
    movi v26.4s, #0
    movi v27.4s, #0
    movi v28.4s, #0
    movi v29.4s, #0
    movi v30.4s, #0
    movi v31.4s, #0
    mov  x14, x4
    mov  x15, x2
2:
    ldr  d0, [x0], #8
    ldr  d1, [x7], #8
    ldr  d2, [x8], #8
    ldr  d3, [x9], #8
    sshll v0.8h, v0.8b, #0
    sshll v1.8h, v1.8b, #0
    sshll v2.8h, v2.8b, #0
    sshll v3.8h, v3.8b, #0
    mov  x16, x15
    prfm pldl1keep, [x15, x13, lsl #3]
    ld1  {{v4.4h, v5.4h, v6.4h, v7.4h}}, [x16], x13
    smlal v16.4s, v4.4h, v0.h[0]
    smlal v17.4s, v5.4h, v0.h[0]
    smlal v18.4s, v6.4h, v0.h[0]
    smlal v19.4s, v7.4h, v0.h[0]
    smlal v20.4s, v4.4h, v1.h[0]
    smlal v21.4s, v5.4h, v1.h[0]
    smlal v22.4s, v6.4h, v1.h[0]
    smlal v23.4s, v7.4h, v1.h[0]
    smlal v24.4s, v4.4h, v2.h[0]
    smlal v25.4s, v5.4h, v2.h[0]
    smlal v26.4s, v6.4h, v2.h[0]
    smlal v27.4s, v7.4h, v2.h[0]
    smlal v28.4s, v4.4h, v3.h[0]
    smlal v29.4s, v5.4h, v3.h[0]
    smlal v30.4s, v6.4h, v3.h[0]
    smlal v31.4s, v7.4h, v3.h[0]
    ld1  {{v4.4h, v5.4h, v6.4h, v7.4h}}, [x16], x13
    smlal v16.4s, v4.4h, v0.h[1]
    smlal v17.4s, v5.4h, v0.h[1]
    smlal v18.4s, v6.4h, v0.h[1]
    smlal v19.4s, v7.4h, v0.h[1]
    smlal v20.4s, v4.4h, v1.h[1]
    smlal v21.4s, v5.4h, v1.h[1]
    smlal v22.4s, v6.4h, v1.h[1]
    smlal v23.4s, v7.4h, v1.h[1]
    smlal v24.4s, v4.4h, v2.h[1]
    smlal v25.4s, v5.4h, v2.h[1]
    smlal v26.4s, v6.4h, v2.h[1]
    smlal v27.4s, v7.4h, v2.h[1]
    smlal v28.4s, v4.4h, v3.h[1]
    smlal v29.4s, v5.4h, v3.h[1]
    smlal v30.4s, v6.4h, v3.h[1]
    smlal v31.4s, v7.4h, v3.h[1]
    ld1  {{v4.4h, v5.4h, v6.4h, v7.4h}}, [x16], x13
    smlal v16.4s, v4.4h, v0.h[2]
    smlal v17.4s, v5.4h, v0.h[2]
    smlal v18.4s, v6.4h, v0.h[2]
    smlal v19.4s, v7.4h, v0.h[2]
    smlal v20.4s, v4.4h, v1.h[2]
    smlal v21.4s, v5.4h, v1.h[2]
    smlal v22.4s, v6.4h, v1.h[2]
    smlal v23.4s, v7.4h, v1.h[2]
    smlal v24.4s, v4.4h, v2.h[2]
    smlal v25.4s, v5.4h, v2.h[2]
    smlal v26.4s, v6.4h, v2.h[2]
    smlal v27.4s, v7.4h, v2.h[2]
    smlal v28.4s, v4.4h, v3.h[2]
    smlal v29.4s, v5.4h, v3.h[2]
    smlal v30.4s, v6.4h, v3.h[2]
    smlal v31.4s, v7.4h, v3.h[2]
    ld1  {{v4.4h, v5.4h, v6.4h, v7.4h}}, [x16], x13
    smlal v16.4s, v4.4h, v0.h[3]
    smlal v17.4s, v5.4h, v0.h[3]
    smlal v18.4s, v6.4h, v0.h[3]
    smlal v19.4s, v7.4h, v0.h[3]
    smlal v20.4s, v4.4h, v1.h[3]
    smlal v21.4s, v5.4h, v1.h[3]
    smlal v22.4s, v6.4h, v1.h[3]
    smlal v23.4s, v7.4h, v1.h[3]
    smlal v24.4s, v4.4h, v2.h[3]
    smlal v25.4s, v5.4h, v2.h[3]
    smlal v26.4s, v6.4h, v2.h[3]
    smlal v27.4s, v7.4h, v2.h[3]
    smlal v28.4s, v4.4h, v3.h[3]
    smlal v29.4s, v5.4h, v3.h[3]
    smlal v30.4s, v6.4h, v3.h[3]
    smlal v31.4s, v7.4h, v3.h[3]
    ld1  {{v4.4h, v5.4h, v6.4h, v7.4h}}, [x16], x13
    smlal v16.4s, v4.4h, v0.h[4]
    smlal v17.4s, v5.4h, v0.h[4]
    smlal v18.4s, v6.4h, v0.h[4]
    smlal v19.4s, v7.4h, v0.h[4]
    smlal v20.4s, v4.4h, v1.h[4]
    smlal v21.4s, v5.4h, v1.h[4]
    smlal v22.4s, v6.4h, v1.h[4]
    smlal v23.4s, v7.4h, v1.h[4]
    smlal v24.4s, v4.4h, v2.h[4]
    smlal v25.4s, v5.4h, v2.h[4]
    smlal v26.4s, v6.4h, v2.h[4]
    smlal v27.4s, v7.4h, v2.h[4]
    smlal v28.4s, v4.4h, v3.h[4]
    smlal v29.4s, v5.4h, v3.h[4]
    smlal v30.4s, v6.4h, v3.h[4]
    smlal v31.4s, v7.4h, v3.h[4]
    ld1  {{v4.4h, v5.4h, v6.4h, v7.4h}}, [x16], x13
    smlal v16.4s, v4.4h, v0.h[5]
    smlal v17.4s, v5.4h, v0.h[5]
    smlal v18.4s, v6.4h, v0.h[5]
    smlal v19.4s, v7.4h, v0.h[5]
    smlal v20.4s, v4.4h, v1.h[5]
    smlal v21.4s, v5.4h, v1.h[5]
    smlal v22.4s, v6.4h, v1.h[5]
    smlal v23.4s, v7.4h, v1.h[5]
    smlal v24.4s, v4.4h, v2.h[5]
    smlal v25.4s, v5.4h, v2.h[5]
    smlal v26.4s, v6.4h, v2.h[5]
    smlal v27.4s, v7.4h, v2.h[5]
    smlal v28.4s, v4.4h, v3.h[5]
    smlal v29.4s, v5.4h, v3.h[5]
    smlal v30.4s, v6.4h, v3.h[5]
    smlal v31.4s, v7.4h, v3.h[5]
    ld1  {{v4.4h, v5.4h, v6.4h, v7.4h}}, [x16], x13
    smlal v16.4s, v4.4h, v0.h[6]
    smlal v17.4s, v5.4h, v0.h[6]
    smlal v18.4s, v6.4h, v0.h[6]
    smlal v19.4s, v7.4h, v0.h[6]
    smlal v20.4s, v4.4h, v1.h[6]
    smlal v21.4s, v5.4h, v1.h[6]
    smlal v22.4s, v6.4h, v1.h[6]
    smlal v23.4s, v7.4h, v1.h[6]
    smlal v24.4s, v4.4h, v2.h[6]
    smlal v25.4s, v5.4h, v2.h[6]
    smlal v26.4s, v6.4h, v2.h[6]
    smlal v27.4s, v7.4h, v2.h[6]
    smlal v28.4s, v4.4h, v3.h[6]
    smlal v29.4s, v5.4h, v3.h[6]
    smlal v30.4s, v6.4h, v3.h[6]
    smlal v31.4s, v7.4h, v3.h[6]
    ld1  {{v4.4h, v5.4h, v6.4h, v7.4h}}, [x16], x13
    smlal v16.4s, v4.4h, v0.h[7]
    smlal v17.4s, v5.4h, v0.h[7]
    smlal v18.4s, v6.4h, v0.h[7]
    smlal v19.4s, v7.4h, v0.h[7]
    smlal v20.4s, v4.4h, v1.h[7]
    smlal v21.4s, v5.4h, v1.h[7]
    smlal v22.4s, v6.4h, v1.h[7]
    smlal v23.4s, v7.4h, v1.h[7]
    smlal v24.4s, v4.4h, v2.h[7]
    smlal v25.4s, v5.4h, v2.h[7]
    smlal v26.4s, v6.4h, v2.h[7]
    smlal v27.4s, v7.4h, v2.h[7]
    smlal v28.4s, v4.4h, v3.h[7]
    smlal v29.4s, v5.4h, v3.h[7]
    smlal v30.4s, v6.4h, v3.h[7]
    smlal v31.4s, v7.4h, v3.h[7]
    add  x15, x15, x13, lsl #3
    subs x14, x14, #8
    b.ne 2b
    st1  {{v16.4s, v17.4s, v18.4s, v19.4s}}, [x5]
    st1  {{v20.4s, v21.4s, v22.4s, v23.4s}}, [x10]
    st1  {{v24.4s, v25.4s, v26.4s, v27.4s}}, [x11]
    st1  {{v28.4s, v29.4s, v30.4s, v31.4s}}, [x12]
    ret
"#
);

#[cfg(target_arch = "aarch64")]
unsafe extern "C" {
    fn yscv_mlal4x16(
        a: *const i8,
        lda: usize,
        b16: *const i16,
        n: usize,
        kp: usize,
        out: *mut i32,
        ldo: usize,
    );
}

/// Intrinsics 4×NR (NR=8 or 4) mlal-lane tail for the columns the 4×16 asm
/// kernel can't cover. `ap` = 4 rows kp apart (padded i8); `b16` = `[kp,N]` i16
/// at the tail column. `nb` = NR/4 ∈ {1,2}. Same lane accumulation as the asm.
///
/// SAFETY: `ap` rows valid for `kp`; `b16` valid for `kp*n`; `out` rows for `nb*4`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn mlal_lane_tail_4xnr(
    ap: *const i8,
    kp: usize,
    b16: *const i16,
    n: usize,
    out: *mut i32,
    ldo: usize,
    nb: usize,
) {
    use std::arch::aarch64::*;
    unsafe {
        let ar = [ap, ap.add(kp), ap.add(2 * kp), ap.add(3 * kp)];
        let mut acc = [[vdupq_n_s32(0); 2]; 4];
        let mut kk = 0;
        while kk < kp {
            let va = [
                vmovl_s8(vld1_s8(ar[0].add(kk))),
                vmovl_s8(vld1_s8(ar[1].add(kk))),
                vmovl_s8(vld1_s8(ar[2].add(kk))),
                vmovl_s8(vld1_s8(ar[3].add(kk))),
            ];
            let vlo = [
                vget_low_s16(va[0]),
                vget_low_s16(va[1]),
                vget_low_s16(va[2]),
                vget_low_s16(va[3]),
            ];
            let vhi = [
                vget_high_s16(va[0]),
                vget_high_s16(va[1]),
                vget_high_s16(va[2]),
                vget_high_s16(va[3]),
            ];
            macro_rules! klane {
                ($kl:expr, $vsrc:ident, $lane:expr) => {{
                    let bp = b16.add((kk + $kl) * n);
                    for cb in 0..nb {
                        let bv = vld1_s16(bp.add(cb * 4));
                        for r in 0..4 {
                            acc[r][cb] = vmlal_lane_s16::<$lane>(acc[r][cb], bv, $vsrc[r]);
                        }
                    }
                }};
            }
            klane!(0, vlo, 0);
            klane!(1, vlo, 1);
            klane!(2, vlo, 2);
            klane!(3, vlo, 3);
            klane!(4, vhi, 0);
            klane!(5, vhi, 1);
            klane!(6, vhi, 2);
            klane!(7, vhi, 3);
            kk += 8;
        }
        for r in 0..4 {
            for cb in 0..nb {
                vst1q_s32(out.add(r * ldo + cb * 4), acc[r][cb]);
            }
        }
    }
}

/// Small-K int8 GEMM via the hand-asm 4×16 mlal-lane kernel. `a` is `[M,K]`
/// row-major, `b` is `[K,N]` k-major. Pads K to a multiple of 8 (only copying
/// A when `k % 8 != 0`; otherwise `a` is used in place) and pre-widens B to i16,
/// then tiles 4×16 asm → 4×8 → 4×4 intrinsics → scalar N%4 / M%4 remainders.
/// Bit-identical to the scalar reference (integer, unchanged k-order; zero pads
/// contribute 0). Replaces the older 4×8 `neon_mlal_lane_gemm` on K≤48.
///
/// SAFETY: `a.len() >= m*k`, `b.len() >= k*n`, `out.len() >= m*n`.
#[cfg(target_arch = "aarch64")]
unsafe fn neon_mlal4x16_gemm(a: &[i8], b: &[i8], m: usize, k: usize, n: usize, out: &mut [i32]) {
    let b16 = widen_kmajor_b_i16(b, k, n);
    let kp = k.div_ceil(8) * 8;
    // SAFETY: b16 is [kp,n]; a=[m,k]; out=[m,n].
    unsafe { neon_mlal4x16_i16b(a, &b16, m, k, kp, n, out) };
}

/// Widen a k-major `[K,N]` i8 RHS to i16 `[kp,N]` (kp = K padded to a multiple
/// of 8, pad rows zero). Hoist this out of a per-block GEMM loop so the constant
/// weight is widened once per conv, not once per block.
pub fn widen_kmajor_b_i16(b: &[i8], k: usize, n: usize) -> Vec<i16> {
    let kp = k.div_ceil(8) * 8;
    let mut b16 = vec![0i16; kp * n];
    for (dst, &v) in b16[..k * n].iter_mut().zip(&b[..k * n]) {
        *dst = v as i16;
    }
    b16
}

/// Runtime-dispatched 4×16 mlal-lane GEMM consuming a PRE-WIDENED i16 k-major B
/// `[kp,N]` (kp = k padded to a multiple of 8). aarch64 uses the hand-asm
/// kernel; other targets fall back to a scalar loop over the i16 B. Bit-exact
/// integer. `a` is `[m,k]` i8, `out` is `[m,n]` i32.
pub fn int8_matmul_mlal4x16_i16b_dispatch(
    a: &[i8],
    b16: &[i16],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [i32],
) {
    let kp = k.div_ceil(8) * 8;
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b16.len(), kp * n);
    debug_assert_eq!(out.len(), m * n);
    #[cfg(target_arch = "aarch64")]
    // SAFETY: shapes checked above.
    unsafe {
        neon_mlal4x16_i16b(a, b16, m, k, kp, n, out);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..m {
            for j in 0..n {
                let mut s = 0i32;
                for kk in 0..k {
                    s += a[i * k + kk] as i32 * b16[kk * n + j] as i32;
                }
                out[i * n + j] = s;
            }
        }
    }
}

/// Compute half of the 4×16 mlal-lane GEMM: pad A to `kp` (only when `kp != k`),
/// tile 4×16 asm → 4×8 → 4×4 intrinsics → scalar N%4 / M%4. `b16` is the
/// pre-widened i16 `[kp,N]` RHS. Bit-identical to the scalar reference.
///
/// SAFETY: `a.len() >= m*k`, `b16.len() >= kp*n`, `out.len() >= m*n`, `kp` is
/// `k` rounded up to a multiple of 8.
#[cfg(target_arch = "aarch64")]
unsafe fn neon_mlal4x16_i16b(
    a: &[i8],
    b16: &[i16],
    m: usize,
    k: usize,
    kp: usize,
    n: usize,
    out: &mut [i32],
) {
    // A: pad K to kp only when needed; else use `a` directly (kp == k).
    let ap_owned: Vec<i8>;
    let (ap, lda): (&[i8], usize) = if kp == k {
        (a, k)
    } else {
        let mut buf = vec![0i8; m * kp];
        for i in 0..m {
            buf[i * kp..i * kp + k].copy_from_slice(&a[i * k..i * k + k]);
        }
        ap_owned = buf;
        (&ap_owned, kp)
    };
    let mt = m & !3;
    unsafe {
        let mut i = 0;
        while i < mt {
            let arow = ap.as_ptr().add(i * lda);
            let orow = out.as_mut_ptr().add(i * n);
            let mut j = 0;
            while j + 16 <= n {
                yscv_mlal4x16(arow, lda, b16.as_ptr().add(j), n, kp, orow.add(j), n * 4);
                j += 16;
            }
            while j + 8 <= n {
                mlal_lane_tail_4xnr(arow, lda, b16.as_ptr().add(j), n, orow.add(j), n, 2);
                j += 8;
            }
            while j + 4 <= n {
                mlal_lane_tail_4xnr(arow, lda, b16.as_ptr().add(j), n, orow.add(j), n, 1);
                j += 4;
            }
            // N%4 scalar tail (b16 rows kk<k equal the i8 B, widened).
            while j < n {
                for r in 0..4 {
                    let mut s = 0i32;
                    for kk in 0..k {
                        s += a[(i + r) * k + kk] as i32 * b16[kk * n + j] as i32;
                    }
                    out[(i + r) * n + j] = s;
                }
                j += 1;
            }
            i += 4;
        }
        // M%4 remainder rows.
        for ii in mt..m {
            for j in 0..n {
                let mut s = 0i32;
                for kk in 0..k {
                    s += a[ii * k + kk] as i32 * b16[kk * n + j] as i32;
                }
                out[ii * n + j] = s;
            }
        }
    }
}

/// Plain ARMv8.0-A NEON widening int8 GEMM: `a` is `[M, K]` row-major, `bt`
/// is the transposed RHS `[N, K]` (contiguous K per column). No
/// `dotprod`/`i8mm` needed — this is the A53-class path.
///
/// A 4×4 register tile is the hot core (XNNPACK-style MR×NR blocking adapted
/// to plain NEON): 4 rows of `a` × 4 columns of `bt`, each of the 16 `[r,c]`
/// products accumulated into its own `vpadalq` chain. The four `a` and four
/// `bt` half-vectors are loaded once per k-step and reused across the tile, so
/// each `bt` panel load is shared by 4 rows (a 4× cut in RHS traffic) and each
/// `a` load by 4 columns. On the in-order A53 the 16 independent chains fully
/// hide the ~4-cycle `vmull`/`vpadalq` latency that the single-chain dot
/// stalls on. Row/column remainders fall to the original 1×4 blocking.
/// Bit-identical to the scalar reference — integer accumulation only, and the
/// k-order per `[r,c]` is unchanged.
///
/// SAFETY: `a.len() >= m*k`, `bt.len() >= n*k`, `out.len() >= m*n`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_widen_gemm(a: &[i8], bt: &[i8], m: usize, k: usize, n: usize, out: &mut [i32]) {
    use std::arch::aarch64::*;
    unsafe {
        let ab = a.as_ptr();
        let bb = bt.as_ptr();
        // Two i8×i8 products fit in i16 (|a·b| ≤ 128·127 = 16256, ×2 = 32512 <
        // 32767) as long as no weight is i8::MIN — then only ONE operand can be
        // -128, capping each product at 16256. When safe, pair the low/high
        // 8-lane halves of each 16-K step with `vmull`+`vmlal_s8` and reduce
        // with a SINGLE `vpadalq` instead of two: 3 Q-ops per 16 MACs (0.1875
        // Q-ops/MAC) vs the widen-every-step 4 (0.25). Integer add is
        // associative so the accumulated column sums are bit-identical. `bt` is
        // the (constant) weights; the scan is O(N·K), amortized over the O(M·N·K)
        // GEMM. This is the only thing that gives INT8 a real edge over f32 FMLA
        // on ARMv8.0 without dotprod (both are otherwise 4 MACs/instruction).
        let pairs_safe = !bt.iter().any(|&w| w == i8::MIN);
        // Hand-scheduled asm 4×4 core beats this compiled loop by ~1.3× when it
        // applies: pairing is overflow-safe (no i8::MIN weight) and K is a
        // multiple of 8 (the kernel consumes 16- then 8-wide K blocks).
        if pairs_safe && k >= 8 && k % 8 == 0 {
            neon_gemm4x4_asm(a, bt, m, k, n, out);
            return;
        }
        let mut i = 0;
        // 4-row register-tiled core.
        while i + 4 <= m {
            let ap = [
                ab.add(i * k),
                ab.add((i + 1) * k),
                ab.add((i + 2) * k),
                ab.add((i + 3) * k),
            ];
            let mut j = 0;
            while j + 4 <= n {
                let bp = [
                    bb.add(j * k),
                    bb.add((j + 1) * k),
                    bb.add((j + 2) * k),
                    bb.add((j + 3) * k),
                ];
                let mut acc = [[vdupq_n_s32(0); 4]; 4];
                let mut kk = 0;
                while kk + 16 <= k {
                    let al = [
                        vget_low_s8(vld1q_s8(ap[0].add(kk))),
                        vget_low_s8(vld1q_s8(ap[1].add(kk))),
                        vget_low_s8(vld1q_s8(ap[2].add(kk))),
                        vget_low_s8(vld1q_s8(ap[3].add(kk))),
                    ];
                    let ah = [
                        vget_high_s8(vld1q_s8(ap[0].add(kk))),
                        vget_high_s8(vld1q_s8(ap[1].add(kk))),
                        vget_high_s8(vld1q_s8(ap[2].add(kk))),
                        vget_high_s8(vld1q_s8(ap[3].add(kk))),
                    ];
                    if pairs_safe {
                        for (c, &bpc) in bp.iter().enumerate() {
                            let bv = vld1q_s8(bpc.add(kk));
                            let bl = vget_low_s8(bv);
                            let bh = vget_high_s8(bv);
                            for r in 0..4 {
                                // al·bl + ah·bh accumulated in i16, reduced once.
                                let p = vmlal_s8(vmull_s8(al[r], bl), ah[r], bh);
                                acc[r][c] = vpadalq_s16(acc[r][c], p);
                            }
                        }
                    } else {
                        for (c, &bpc) in bp.iter().enumerate() {
                            let bv = vld1q_s8(bpc.add(kk));
                            let bl = vget_low_s8(bv);
                            let bh = vget_high_s8(bv);
                            for r in 0..4 {
                                acc[r][c] = vpadalq_s16(acc[r][c], vmull_s8(al[r], bl));
                                acc[r][c] = vpadalq_s16(acc[r][c], vmull_s8(ah[r], bh));
                            }
                        }
                    }
                    kk += 16;
                }
                if kk + 8 <= k {
                    let av = [
                        vld1_s8(ap[0].add(kk)),
                        vld1_s8(ap[1].add(kk)),
                        vld1_s8(ap[2].add(kk)),
                        vld1_s8(ap[3].add(kk)),
                    ];
                    for (c, &bpc) in bp.iter().enumerate() {
                        let bv = vld1_s8(bpc.add(kk));
                        for r in 0..4 {
                            acc[r][c] = vpadalq_s16(acc[r][c], vmull_s8(av[r], bv));
                        }
                    }
                    kk += 8;
                }
                let mut s = [[0_i32; 4]; 4];
                for r in 0..4 {
                    for c in 0..4 {
                        s[r][c] = vaddvq_s32(acc[r][c]);
                    }
                }
                while kk < k {
                    for (r, &apr) in ap.iter().enumerate() {
                        let av = *apr.add(kk) as i32;
                        for (c, &bpc) in bp.iter().enumerate() {
                            s[r][c] += av * (*bpc.add(kk) as i32);
                        }
                    }
                    kk += 1;
                }
                for (r, sr) in s.iter().enumerate() {
                    let o = (i + r) * n + j;
                    out[o] = sr[0];
                    out[o + 1] = sr[1];
                    out[o + 2] = sr[2];
                    out[o + 3] = sr[3];
                }
                j += 4;
            }
            // Column remainder for these 4 rows.
            while j < n {
                let bpj = bb.add(j * k);
                for (r, &apr) in ap.iter().enumerate() {
                    out[(i + r) * n + j] = widen_dot_1col(apr, bpj, k);
                }
                j += 1;
            }
            i += 4;
        }
        // Row remainder: original 1×4 blocking.
        while i < m {
            let ap = a.as_ptr().add(i * k);
            let mut j = 0;
            while j + 4 <= n {
                let bp0 = bt.as_ptr().add(j * k);
                let bp1 = bt.as_ptr().add((j + 1) * k);
                let bp2 = bt.as_ptr().add((j + 2) * k);
                let bp3 = bt.as_ptr().add((j + 3) * k);
                let mut a0 = vdupq_n_s32(0);
                let mut a1 = vdupq_n_s32(0);
                let mut a2 = vdupq_n_s32(0);
                let mut a3 = vdupq_n_s32(0);
                let mut kk = 0;
                while kk + 16 <= k {
                    let av = vld1q_s8(ap.add(kk));
                    let avl = vget_low_s8(av);
                    let avh = vget_high_s8(av);
                    let b0 = vld1q_s8(bp0.add(kk));
                    let b1 = vld1q_s8(bp1.add(kk));
                    let b2 = vld1q_s8(bp2.add(kk));
                    let b3 = vld1q_s8(bp3.add(kk));
                    if pairs_safe {
                        a0 = vpadalq_s16(
                            a0,
                            vmlal_s8(vmull_s8(avl, vget_low_s8(b0)), avh, vget_high_s8(b0)),
                        );
                        a1 = vpadalq_s16(
                            a1,
                            vmlal_s8(vmull_s8(avl, vget_low_s8(b1)), avh, vget_high_s8(b1)),
                        );
                        a2 = vpadalq_s16(
                            a2,
                            vmlal_s8(vmull_s8(avl, vget_low_s8(b2)), avh, vget_high_s8(b2)),
                        );
                        a3 = vpadalq_s16(
                            a3,
                            vmlal_s8(vmull_s8(avl, vget_low_s8(b3)), avh, vget_high_s8(b3)),
                        );
                    } else {
                        a0 = vpadalq_s16(a0, vmull_s8(avl, vget_low_s8(b0)));
                        a0 = vpadalq_s16(a0, vmull_s8(avh, vget_high_s8(b0)));
                        a1 = vpadalq_s16(a1, vmull_s8(avl, vget_low_s8(b1)));
                        a1 = vpadalq_s16(a1, vmull_s8(avh, vget_high_s8(b1)));
                        a2 = vpadalq_s16(a2, vmull_s8(avl, vget_low_s8(b2)));
                        a2 = vpadalq_s16(a2, vmull_s8(avh, vget_high_s8(b2)));
                        a3 = vpadalq_s16(a3, vmull_s8(avl, vget_low_s8(b3)));
                        a3 = vpadalq_s16(a3, vmull_s8(avh, vget_high_s8(b3)));
                    }
                    kk += 16;
                }
                if kk + 8 <= k {
                    let av = vld1_s8(ap.add(kk));
                    a0 = vpadalq_s16(a0, vmull_s8(av, vld1_s8(bp0.add(kk))));
                    a1 = vpadalq_s16(a1, vmull_s8(av, vld1_s8(bp1.add(kk))));
                    a2 = vpadalq_s16(a2, vmull_s8(av, vld1_s8(bp2.add(kk))));
                    a3 = vpadalq_s16(a3, vmull_s8(av, vld1_s8(bp3.add(kk))));
                    kk += 8;
                }
                let mut s0 = vaddvq_s32(a0);
                let mut s1 = vaddvq_s32(a1);
                let mut s2 = vaddvq_s32(a2);
                let mut s3 = vaddvq_s32(a3);
                while kk < k {
                    let av = *ap.add(kk) as i32;
                    s0 += av * (*bp0.add(kk) as i32);
                    s1 += av * (*bp1.add(kk) as i32);
                    s2 += av * (*bp2.add(kk) as i32);
                    s3 += av * (*bp3.add(kk) as i32);
                    kk += 1;
                }
                let o = i * n + j;
                out[o] = s0;
                out[o + 1] = s1;
                out[o + 2] = s2;
                out[o + 3] = s3;
                j += 4;
            }
            while j < n {
                out[i * n + j] = widen_dot_1col(ap, bt.as_ptr().add(j * k), k);
                j += 1;
            }
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn int8_matmul_prepacked_neon_widen(a: &[i8], b: &PackedI8B, m: usize, out: &mut [i32]) {
    // SAFETY: caller guarantees a=[m,k], out=[m,n]; bt is [n,k].
    unsafe { neon_widen_gemm(a, b.transposed(), m, b.k, b.n, out) };
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn int8_matmul_neon_widen(
    a: &[i8],
    b: &[i8],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [i32],
) {
    // Small-K (MobileNet 1×1 c_in ≤ 48, plus the im2col'd 3×3 stem K=27): the
    // hand-asm 4×16 SMLAL-lane kernel wins big (no per-output horizontal
    // reduction, wide NR, B prefetch), consuming `b` in its native k-major
    // `[K,N]` layout — no transpose. Measured 1.2–1.6× over the c2 4×4 core on
    // the A53 (stem 2.11→3.28, K=24 2.61→3.63 GMAC/s). K∈(48,64] and above stay
    // on the 8-wide `vmull_s8` widening kernel, where the c2 pairing wins.
    if k <= 48 {
        // SAFETY: a=[m,k], b=[k,n] k-major, out=[m,n].
        unsafe { neon_mlal4x16_gemm(a, b, m, k, n, out) };
        return;
    }
    if k <= 64 {
        // SAFETY: a=[m,k], b=[k,n] k-major, out=[m,n].
        unsafe { neon_mlal_lane_gemm(a, b, m, k, n, out) };
        return;
    }
    let bt = transpose_b(b, k, n);
    // SAFETY: bt is [n,k], a=[m,k], out=[m,n].
    unsafe { neon_widen_gemm(a, &bt, m, k, n, out) };
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn int8_matmul_avx2_widen(
    a: &[i8],
    b: &[i8],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [i32],
) {
    use std::arch::x86_64::*;
    let bt = transpose_b(b, k, n);
    for i in 0..m {
        for j in 0..n {
            let mut acc = _mm256_setzero_si256();
            let mut kk = 0;
            while kk + 16 <= k {
                let av = _mm_loadu_si128(a.as_ptr().add(i * k + kk) as *const __m128i);
                let bv = _mm_loadu_si128(bt.as_ptr().add(j * k + kk) as *const __m128i);
                let av16 = _mm256_cvtepi8_epi16(av);
                let bv16 = _mm256_cvtepi8_epi16(bv);
                let prod = _mm256_madd_epi16(av16, bv16);
                acc = _mm256_add_epi32(acc, prod);
                kk += 16;
            }
            let mut buf = [0_i32; 8];
            _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, acc);
            let mut tail = buf.iter().sum::<i32>();
            while kk < k {
                tail += (a[i * k + kk] as i32) * (bt[j * k + kk] as i32);
                kk += 1;
            }
            out[i * n + j] = tail;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn int8_matmul_prepacked_avx2_widen(a: &[i8], b: &PackedI8B, m: usize, out: &mut [i32]) {
    use std::arch::x86_64::*;
    let k = b.k;
    let n = b.n;
    let bt = b.transposed();
    for i in 0..m {
        for j in 0..n {
            let mut acc = _mm256_setzero_si256();
            let mut kk = 0;
            while kk + 16 <= k {
                let av = _mm_loadu_si128(a.as_ptr().add(i * k + kk) as *const __m128i);
                let bv = _mm_loadu_si128(bt.as_ptr().add(j * k + kk) as *const __m128i);
                let av16 = _mm256_cvtepi8_epi16(av);
                let bv16 = _mm256_cvtepi8_epi16(bv);
                let prod = _mm256_madd_epi16(av16, bv16);
                acc = _mm256_add_epi32(acc, prod);
                kk += 16;
            }
            let mut buf = [0_i32; 8];
            _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, acc);
            let mut tail = buf.iter().sum::<i32>();
            while kk < k {
                tail += (a[i * k + kk] as i32) * (bt[j * k + kk] as i32);
                kk += 1;
            }
            out[i * n + j] = tail;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avxvnni")]
unsafe fn int8_matmul_avx_vnni(a: &[i8], b: &[i8], m: usize, k: usize, n: usize, out: &mut [i32]) {
    use std::arch::x86_64::*;
    let bt = transpose_b(b, k, n);
    let bias: i32 = 128;
    let bias128 = _mm256_set1_epi8(-128_i8);
    // Per-column sum_b is shape-invariant; precompute once over bt.
    let mut col_sum_b = vec![0_i32; n];
    for j in 0..n {
        let mut s: i32 = 0;
        for kk in 0..k {
            s += bt[j * k + kk] as i32;
        }
        col_sum_b[j] = s;
    }
    let k_vnni = (k / 32) * 32;
    // Per-column sum over the VNNI-covered K prefix. col_sum_b - tail.
    let mut col_sum_b_vnni = vec![0_i32; n];
    for j in 0..n {
        let mut tail = 0_i32;
        for kk in k_vnni..k {
            tail += bt[j * k + kk] as i32;
        }
        col_sum_b_vnni[j] = col_sum_b[j] - tail;
    }
    for i in 0..m {
        for j in 0..n {
            let mut acc = _mm256_setzero_si256();
            let mut kk = 0;
            while kk + 32 <= k {
                let av = _mm256_loadu_si256(a.as_ptr().add(i * k + kk) as *const __m256i);
                let bv = _mm256_loadu_si256(bt.as_ptr().add(j * k + kk) as *const __m256i);
                // a → a + 128 (mod 256, reinterpret i8→u8) so unsigned-signed
                // VNNI gives `dot(a, b) + 128 * sum(b)`. Subtract `128 * sum_b`
                // at the end to get the signed-signed result.
                let av_u = _mm256_xor_si256(av, bias128);
                acc = _mm256_dpbusd_avx_epi32(acc, av_u, bv);
                kk += 32;
            }
            let mut buf = [0_i32; 8];
            _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, acc);
            let mut tail = buf.iter().sum::<i32>() - bias * col_sum_b_vnni[j];
            while kk < k {
                tail += (a[i * k + kk] as i32) * (bt[j * k + kk] as i32);
                kk += 1;
            }
            out[i * n + j] = tail;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avxvnni")]
unsafe fn int8_matmul_prepacked_avx_vnni(a: &[i8], b: &PackedI8B, m: usize, out: &mut [i32]) {
    use std::arch::x86_64::*;
    let k = b.k;
    let n = b.n;
    let bt = b.transposed();
    let bias: i32 = 128;
    let bias128 = _mm256_set1_epi8(-128_i8);
    let k_vnni = (k / 32) * 32;
    let mut col_sum_b_vnni = vec![0_i32; n];
    for j in 0..n {
        let mut s: i32 = 0;
        for kk in 0..k_vnni {
            s += bt[j * k + kk] as i32;
        }
        col_sum_b_vnni[j] = s;
    }
    for i in 0..m {
        for j in 0..n {
            let mut acc = _mm256_setzero_si256();
            let mut kk = 0;
            while kk + 32 <= k {
                let av = _mm256_loadu_si256(a.as_ptr().add(i * k + kk) as *const __m256i);
                let bv = _mm256_loadu_si256(bt.as_ptr().add(j * k + kk) as *const __m256i);
                let av_u = _mm256_xor_si256(av, bias128);
                acc = _mm256_dpbusd_avx_epi32(acc, av_u, bv);
                kk += 32;
            }
            let mut buf = [0_i32; 8];
            _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, acc);
            let mut tail = buf.iter().sum::<i32>() - bias * col_sum_b_vnni[j];
            while kk < k {
                tail += (a[i * k + kk] as i32) * (bt[j * k + kk] as i32);
                kk += 1;
            }
            out[i * n + j] = tail;
        }
    }
}

/// Pack `b` (shape `[K, N]` row-major) into the layout the
/// register-blocked VNNI kernel consumes:
///
/// ```text
/// bp[jp][g][c][r] = b[(g*4 + r) * n + jp*16 + c]
/// ```
///
/// i.e. for each N panel of 16 columns and each K group of 4 rows, the
/// 64 bytes are interleaved so a single ZMM load gives one VNNI input.
/// Caller must guarantee `n % 16 == 0`; K is rounded down to a multiple
/// of 4 (`k4_full`), the K tail handled scalar-side.
#[cfg(target_arch = "x86_64")]
fn pack_b_vnni_4x16(b: &[i8], k: usize, n: usize) -> (Vec<i8>, usize) {
    debug_assert_eq!(n % 16, 0);
    let k4_full = (k / 4) * 4;
    let groups = k4_full / 4;
    let n_panels = n / 16;
    let mut bp = vec![0_i8; n_panels * groups * 16 * 4];
    for jp in 0..n_panels {
        for g in 0..groups {
            let dst_panel = jp * groups * 16 * 4;
            for c in 0..16 {
                let n_idx = jp * 16 + c;
                let dst_col = dst_panel + g * 16 * 4 + c * 4;
                for r in 0..4 {
                    let k_idx = g * 4 + r;
                    bp[dst_col + r] = b[k_idx * n + n_idx];
                }
            }
        }
    }
    (bp, k4_full)
}

/// Register-blocked AVX-512-VNNI kernel, MR=4 × NR=16.
///
/// Each MR×NR output tile keeps 4 ZMM accumulators alive across the
/// full K loop. Per K group of 4: one ZMM B load is reused across all
/// 4 A rows (broadcast-and-vpdpbusd), so we issue 4 vpdpbusd per
/// 64-byte B traffic. On Zen 4 vpdpbusd has 1/cycle throughput, B
/// load is L1-bound — kernel is FMA-issue-limited at ~95% of peak.
///
/// Strict shape gate: `m % 4 == 0`, `n % 16 == 0`, `k >= 4`. Caller
/// (`int8_matmul_avx512_vnni`) routes other shapes to the simple
/// transposed-B path.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn int8_matmul_avx512_vnni_blocked(
    a: &[i8],
    b: &[i8],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [i32],
) {
    use std::arch::x86_64::*;
    const MR: usize = 4;
    const NR: usize = 16;

    let (bp, k4_full) = pack_b_vnni_4x16(b, k, n);
    let groups = k4_full / 4;
    let n_panels = n / NR;

    // Per-column sum_b over the K range covered by the VNNI groups.
    // Bias-shift: dpbusd(a XOR 0x80, b) = dot(a, b) + 128 · sum(b),
    // so we subtract `128 · col_sum_b_vnni[j]` after the K loop.
    let mut col_sum_b_vnni = vec![0_i32; n];
    for j in 0..n {
        let mut s: i32 = 0;
        for kk in 0..k4_full {
            s += b[kk * n + j] as i32;
        }
        col_sum_b_vnni[j] = s;
    }
    let bias = 128_i32;
    let bias_xor = _mm512_set1_epi8(-128_i8);

    for ib in (0..m).step_by(MR) {
        for jp in 0..n_panels {
            let mut a0 = _mm512_setzero_si512();
            let mut a1 = _mm512_setzero_si512();
            let mut a2 = _mm512_setzero_si512();
            let mut a3 = _mm512_setzero_si512();
            let bp_panel = jp * groups * NR * 4;
            for g in 0..groups {
                let bv =
                    _mm512_loadu_si512(bp.as_ptr().add(bp_panel + g * NR * 4) as *const __m512i);
                let k_idx = g * 4;
                // Read 4 bytes of A row r at K position k_idx as a single
                // i32; broadcast across 16 lanes; XOR with 0x80808080 to
                // shift signed → unsigned for vpdpbusd.
                let a_p0 = (a.as_ptr().add(ib * k + k_idx) as *const i32).read_unaligned();
                let a_p1 = (a.as_ptr().add((ib + 1) * k + k_idx) as *const i32).read_unaligned();
                let a_p2 = (a.as_ptr().add((ib + 2) * k + k_idx) as *const i32).read_unaligned();
                let a_p3 = (a.as_ptr().add((ib + 3) * k + k_idx) as *const i32).read_unaligned();
                let av0 = _mm512_xor_si512(_mm512_set1_epi32(a_p0), bias_xor);
                let av1 = _mm512_xor_si512(_mm512_set1_epi32(a_p1), bias_xor);
                let av2 = _mm512_xor_si512(_mm512_set1_epi32(a_p2), bias_xor);
                let av3 = _mm512_xor_si512(_mm512_set1_epi32(a_p3), bias_xor);
                a0 = _mm512_dpbusd_epi32(a0, av0, bv);
                a1 = _mm512_dpbusd_epi32(a1, av1, bv);
                a2 = _mm512_dpbusd_epi32(a2, av2, bv);
                a3 = _mm512_dpbusd_epi32(a3, av3, bv);
            }
            let mut buf = [[0_i32; NR]; MR];
            _mm512_storeu_si512(buf[0].as_mut_ptr() as *mut __m512i, a0);
            _mm512_storeu_si512(buf[1].as_mut_ptr() as *mut __m512i, a1);
            _mm512_storeu_si512(buf[2].as_mut_ptr() as *mut __m512i, a2);
            _mm512_storeu_si512(buf[3].as_mut_ptr() as *mut __m512i, a3);
            for r_off in 0..MR {
                for c in 0..NR {
                    let n_idx = jp * NR + c;
                    let mut v = buf[r_off][c] - bias * col_sum_b_vnni[n_idx];
                    // K tail (≤ 3 elements).
                    for kk in k4_full..k {
                        v += (a[(ib + r_off) * k + kk] as i32) * (b[kk * n + n_idx] as i32);
                    }
                    out[(ib + r_off) * n + n_idx] = v;
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn int8_matmul_prepacked_avx512_vnni_blocked(
    a: &[i8],
    b: &PackedI8B,
    m: usize,
    out: &mut [i32],
) {
    use std::arch::x86_64::*;
    const MR: usize = 4;
    const NR: usize = 16;

    let k = b.k;
    let n = b.n;
    let packed = b.vnni_4x16.as_ref().expect("missing VNNI-packed RHS");
    let k4_full = packed.k4_full;
    let groups = k4_full / 4;
    let n_panels = n / NR;
    let bt = b.transposed();
    let bias = 128_i32;
    let bias_xor = _mm512_set1_epi8(-128_i8);

    for ib in (0..m).step_by(MR) {
        for jp in 0..n_panels {
            let mut a0 = _mm512_setzero_si512();
            let mut a1 = _mm512_setzero_si512();
            let mut a2 = _mm512_setzero_si512();
            let mut a3 = _mm512_setzero_si512();
            let bp_panel = jp * groups * NR * 4;
            for g in 0..groups {
                let bv = _mm512_loadu_si512(
                    packed.bp.as_ptr().add(bp_panel + g * NR * 4) as *const __m512i
                );
                let k_idx = g * 4;
                let a_p0 = (a.as_ptr().add(ib * k + k_idx) as *const i32).read_unaligned();
                let a_p1 = (a.as_ptr().add((ib + 1) * k + k_idx) as *const i32).read_unaligned();
                let a_p2 = (a.as_ptr().add((ib + 2) * k + k_idx) as *const i32).read_unaligned();
                let a_p3 = (a.as_ptr().add((ib + 3) * k + k_idx) as *const i32).read_unaligned();
                let av0 = _mm512_xor_si512(_mm512_set1_epi32(a_p0), bias_xor);
                let av1 = _mm512_xor_si512(_mm512_set1_epi32(a_p1), bias_xor);
                let av2 = _mm512_xor_si512(_mm512_set1_epi32(a_p2), bias_xor);
                let av3 = _mm512_xor_si512(_mm512_set1_epi32(a_p3), bias_xor);
                a0 = _mm512_dpbusd_epi32(a0, av0, bv);
                a1 = _mm512_dpbusd_epi32(a1, av1, bv);
                a2 = _mm512_dpbusd_epi32(a2, av2, bv);
                a3 = _mm512_dpbusd_epi32(a3, av3, bv);
            }
            let mut buf = [[0_i32; NR]; MR];
            _mm512_storeu_si512(buf[0].as_mut_ptr() as *mut __m512i, a0);
            _mm512_storeu_si512(buf[1].as_mut_ptr() as *mut __m512i, a1);
            _mm512_storeu_si512(buf[2].as_mut_ptr() as *mut __m512i, a2);
            _mm512_storeu_si512(buf[3].as_mut_ptr() as *mut __m512i, a3);
            for r_off in 0..MR {
                for c in 0..NR {
                    let n_idx = jp * NR + c;
                    let mut v = buf[r_off][c] - bias * packed.col_sum_b_vnni[n_idx];
                    for kk in k4_full..k {
                        v += (a[(ib + r_off) * k + kk] as i32) * (bt[n_idx * k + kk] as i32);
                    }
                    out[(ib + r_off) * n + n_idx] = v;
                }
            }
        }
    }
}

/// Register-blocked AVX-512-VNNI kernel, MR=8 × NR=16. 8 ZMM
/// accumulators + 1 ZMM B + 8 ZMM A broadcasts (held in 8 GP-sourced
/// `set1_epi32` results) fit comfortably in the 32-register file.
///
/// vs MR=4: B-vector reuse doubles (8 vpdpbusd per B load instead of 4),
/// halving the L1 traffic on B for fixed M. Wins on hidden×hidden LLM
/// linears where MR=4 lost to the simple kernel; loses on tiny-K shapes
/// where the extra register pressure exceeds the B-reuse benefit. The
/// dispatch gate picks between MR=8 and MR=4 per shape.
///
/// Strict gate: `m % 8 == 0 && n % 16 == 0 && k >= 4`. Caller routes
/// other shapes to MR=4 or simple paths.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn int8_matmul_avx512_vnni_blocked_mr8(
    a: &[i8],
    b: &[i8],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [i32],
) {
    use std::arch::x86_64::*;
    const MR: usize = 8;
    const NR: usize = 16;

    let (bp, k4_full) = pack_b_vnni_4x16(b, k, n);
    let groups = k4_full / 4;
    let n_panels = n / NR;

    let mut col_sum_b_vnni = vec![0_i32; n];
    for j in 0..n {
        let mut s: i32 = 0;
        for kk in 0..k4_full {
            s += b[kk * n + j] as i32;
        }
        col_sum_b_vnni[j] = s;
    }
    let bias = 128_i32;
    let bias_xor = _mm512_set1_epi8(-128_i8);

    for ib in (0..m).step_by(MR) {
        for jp in 0..n_panels {
            let mut acc0 = _mm512_setzero_si512();
            let mut acc1 = _mm512_setzero_si512();
            let mut acc2 = _mm512_setzero_si512();
            let mut acc3 = _mm512_setzero_si512();
            let mut acc4 = _mm512_setzero_si512();
            let mut acc5 = _mm512_setzero_si512();
            let mut acc6 = _mm512_setzero_si512();
            let mut acc7 = _mm512_setzero_si512();
            let bp_panel = jp * groups * NR * 4;
            for g in 0..groups {
                let bv =
                    _mm512_loadu_si512(bp.as_ptr().add(bp_panel + g * NR * 4) as *const __m512i);
                let k_idx = g * 4;
                let row_base = ib * k + k_idx;
                let p0 = (a.as_ptr().add(row_base) as *const i32).read_unaligned();
                let p1 = (a.as_ptr().add(row_base + k) as *const i32).read_unaligned();
                let p2 = (a.as_ptr().add(row_base + 2 * k) as *const i32).read_unaligned();
                let p3 = (a.as_ptr().add(row_base + 3 * k) as *const i32).read_unaligned();
                let p4 = (a.as_ptr().add(row_base + 4 * k) as *const i32).read_unaligned();
                let p5 = (a.as_ptr().add(row_base + 5 * k) as *const i32).read_unaligned();
                let p6 = (a.as_ptr().add(row_base + 6 * k) as *const i32).read_unaligned();
                let p7 = (a.as_ptr().add(row_base + 7 * k) as *const i32).read_unaligned();
                let av0 = _mm512_xor_si512(_mm512_set1_epi32(p0), bias_xor);
                let av1 = _mm512_xor_si512(_mm512_set1_epi32(p1), bias_xor);
                let av2 = _mm512_xor_si512(_mm512_set1_epi32(p2), bias_xor);
                let av3 = _mm512_xor_si512(_mm512_set1_epi32(p3), bias_xor);
                let av4 = _mm512_xor_si512(_mm512_set1_epi32(p4), bias_xor);
                let av5 = _mm512_xor_si512(_mm512_set1_epi32(p5), bias_xor);
                let av6 = _mm512_xor_si512(_mm512_set1_epi32(p6), bias_xor);
                let av7 = _mm512_xor_si512(_mm512_set1_epi32(p7), bias_xor);
                acc0 = _mm512_dpbusd_epi32(acc0, av0, bv);
                acc1 = _mm512_dpbusd_epi32(acc1, av1, bv);
                acc2 = _mm512_dpbusd_epi32(acc2, av2, bv);
                acc3 = _mm512_dpbusd_epi32(acc3, av3, bv);
                acc4 = _mm512_dpbusd_epi32(acc4, av4, bv);
                acc5 = _mm512_dpbusd_epi32(acc5, av5, bv);
                acc6 = _mm512_dpbusd_epi32(acc6, av6, bv);
                acc7 = _mm512_dpbusd_epi32(acc7, av7, bv);
            }
            let mut buf = [[0_i32; NR]; MR];
            _mm512_storeu_si512(buf[0].as_mut_ptr() as *mut __m512i, acc0);
            _mm512_storeu_si512(buf[1].as_mut_ptr() as *mut __m512i, acc1);
            _mm512_storeu_si512(buf[2].as_mut_ptr() as *mut __m512i, acc2);
            _mm512_storeu_si512(buf[3].as_mut_ptr() as *mut __m512i, acc3);
            _mm512_storeu_si512(buf[4].as_mut_ptr() as *mut __m512i, acc4);
            _mm512_storeu_si512(buf[5].as_mut_ptr() as *mut __m512i, acc5);
            _mm512_storeu_si512(buf[6].as_mut_ptr() as *mut __m512i, acc6);
            _mm512_storeu_si512(buf[7].as_mut_ptr() as *mut __m512i, acc7);
            for r_off in 0..MR {
                for c in 0..NR {
                    let n_idx = jp * NR + c;
                    let mut v = buf[r_off][c] - bias * col_sum_b_vnni[n_idx];
                    for kk in k4_full..k {
                        v += (a[(ib + r_off) * k + kk] as i32) * (b[kk * n + n_idx] as i32);
                    }
                    out[(ib + r_off) * n + n_idx] = v;
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn int8_matmul_avx512_vnni(
    a: &[i8],
    b: &[i8],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [i32],
) {
    use std::arch::x86_64::*;
    // Register-blocked dispatch: pick MR=8 only on the empirically-good
    // regime — large K with small N (e.g. Llama down-proj 64×8192×2048)
    // — where the doubled B reuse beats the simple Bᵀ kernel. On
    // hidden×hidden (K=N=2048) and on gate/up with N≫K, the strided
    // 8-row A pattern thrashes L1 and the simple kernel wins.
    if m >= 8 && m.is_multiple_of(8) && n.is_multiple_of(16) && k >= 8000 && n <= 2048 {
        return int8_matmul_avx512_vnni_blocked_mr8(a, b, m, k, n, out);
    }
    if m >= 4
        && m.is_multiple_of(4)
        && n.is_multiple_of(16)
        && k >= 4
        && (k * n <= 1_000_000 || n >= 16384)
    {
        return int8_matmul_avx512_vnni_blocked(a, b, m, k, n, out);
    }
    let bt = transpose_b(b, k, n);
    let bias: i32 = 128;
    let bias128 = _mm512_set1_epi8(-128_i8);
    let k_vnni = (k / 64) * 64;
    let mut col_sum_b_vnni = vec![0_i32; n];
    for j in 0..n {
        let mut s: i32 = 0;
        for kk in 0..k_vnni {
            s += bt[j * k + kk] as i32;
        }
        col_sum_b_vnni[j] = s;
    }
    for i in 0..m {
        for j in 0..n {
            let mut acc = _mm512_setzero_si512();
            let mut kk = 0;
            while kk + 64 <= k {
                let av = _mm512_loadu_si512(a.as_ptr().add(i * k + kk) as *const __m512i);
                let bv = _mm512_loadu_si512(bt.as_ptr().add(j * k + kk) as *const __m512i);
                // Bias-shift: a' = a XOR 0x80 makes dot_us(a',b) = dot(a,b)
                // + 128·sum(b). Subtract once at the end via col_sum_b_vnni.
                let av_u = _mm512_xor_si512(av, bias128);
                acc = _mm512_dpbusd_epi32(acc, av_u, bv);
                kk += 64;
            }
            let lane_sum = _mm512_reduce_add_epi32(acc);
            let mut tail = lane_sum - bias * col_sum_b_vnni[j];
            while kk < k {
                tail += (a[i * k + kk] as i32) * (bt[j * k + kk] as i32);
                kk += 1;
            }
            out[i * n + j] = tail;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn int8_matmul_prepacked_avx512_vnni(a: &[i8], b: &PackedI8B, m: usize, out: &mut [i32]) {
    use std::arch::x86_64::*;
    let k = b.k;
    let n = b.n;
    if b.vnni_4x16.is_some()
        && m >= 4
        && m.is_multiple_of(4)
        && n.is_multiple_of(16)
        && k >= 4
        && (k * n <= 1_000_000 || n >= 16384)
    {
        return int8_matmul_prepacked_avx512_vnni_blocked(a, b, m, out);
    }
    let bt = b.transposed();
    let bias: i32 = 128;
    let bias128 = _mm512_set1_epi8(-128_i8);
    let k_vnni = (k / 64) * 64;
    let mut col_sum_b_vnni = vec![0_i32; n];
    for j in 0..n {
        let mut s: i32 = 0;
        for kk in 0..k_vnni {
            s += bt[j * k + kk] as i32;
        }
        col_sum_b_vnni[j] = s;
    }
    for i in 0..m {
        for j in 0..n {
            let mut acc = _mm512_setzero_si512();
            let mut kk = 0;
            while kk + 64 <= k {
                let av = _mm512_loadu_si512(a.as_ptr().add(i * k + kk) as *const __m512i);
                let bv = _mm512_loadu_si512(bt.as_ptr().add(j * k + kk) as *const __m512i);
                let av_u = _mm512_xor_si512(av, bias128);
                acc = _mm512_dpbusd_epi32(acc, av_u, bv);
                kk += 64;
            }
            let lane_sum = _mm512_reduce_add_epi32(acc);
            let mut tail = lane_sum - bias * col_sum_b_vnni[j];
            while kk < k {
                tail += (a[i * k + kk] as i32) * (bt[j * k + kk] as i32);
                kk += 1;
            }
            out[i * n + j] = tail;
        }
    }
}

fn select_int8_matmul_path() -> Int8MatmulPath {
    if cfg!(miri) {
        return Int8MatmulPath::Scalar;
    }
    let features = crate::host_cpu().features;
    #[cfg(target_arch = "x86_64")]
    {
        if features.x86_avx512_vnni() {
            return Int8MatmulPath::Avx512Vnni;
        }
        if features.x86_avx_vnni() {
            return Int8MatmulPath::AvxVnni;
        }
        if features.avx2 {
            return Int8MatmulPath::Avx2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if features.aarch64_neon_i8mm() {
            return Int8MatmulPath::NeonI8mm;
        }
        if features.aarch64_neon_dotprod() {
            return Int8MatmulPath::NeonDotprod;
        }
        if features.neon {
            return Int8MatmulPath::NeonWiden;
        }
    }
    Int8MatmulPath::Scalar
}

fn select_int8_prepacked_path() -> Int8MatmulPath {
    if cfg!(miri) {
        return Int8MatmulPath::Scalar;
    }
    let features = crate::host_cpu().features;
    #[cfg(target_arch = "x86_64")]
    {
        if features.x86_avx512_vnni() {
            return Int8MatmulPath::Avx512Vnni;
        }
        if features.x86_avx_vnni() {
            return Int8MatmulPath::AvxVnni;
        }
        if features.avx2 {
            return Int8MatmulPath::Avx2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if features.aarch64_neon_i8mm() {
            return Int8MatmulPath::NeonI8mm;
        }
        if features.aarch64_neon_dotprod() {
            return Int8MatmulPath::NeonDotprod;
        }
        if features.neon {
            return Int8MatmulPath::NeonWiden;
        }
    }
    Int8MatmulPath::Scalar
}

fn selected_int8_matmul_kernel() -> Int8MatmulKernel {
    static KERNEL: OnceLock<Int8MatmulKernel> = OnceLock::new();
    *KERNEL.get_or_init(|| match select_int8_matmul_path() {
        #[cfg(target_arch = "x86_64")]
        Int8MatmulPath::Avx512Vnni => int8_matmul_avx512_vnni_kernel,
        #[cfg(target_arch = "x86_64")]
        Int8MatmulPath::AvxVnni => int8_matmul_avx_vnni_kernel,
        #[cfg(target_arch = "x86_64")]
        Int8MatmulPath::Avx2 => int8_matmul_avx2_widen_kernel,
        #[cfg(target_arch = "aarch64")]
        Int8MatmulPath::NeonI8mm => int8_matmul_neon_i8mm_kernel,
        #[cfg(target_arch = "aarch64")]
        Int8MatmulPath::NeonDotprod => int8_matmul_neon_sdot_kernel,
        #[cfg(target_arch = "aarch64")]
        Int8MatmulPath::NeonWiden => int8_matmul_neon_widen_kernel,
        _ => int8_matmul_scalar,
    })
}

fn selected_int8_prepacked_kernel() -> Int8PrepackedKernel {
    static KERNEL: OnceLock<Int8PrepackedKernel> = OnceLock::new();
    *KERNEL.get_or_init(|| match select_int8_prepacked_path() {
        #[cfg(target_arch = "x86_64")]
        Int8MatmulPath::Avx512Vnni => int8_matmul_prepacked_avx512_vnni_kernel,
        #[cfg(target_arch = "x86_64")]
        Int8MatmulPath::AvxVnni => int8_matmul_prepacked_avx_vnni_kernel,
        #[cfg(target_arch = "x86_64")]
        Int8MatmulPath::Avx2 => int8_matmul_prepacked_avx2_widen_kernel,
        #[cfg(target_arch = "aarch64")]
        Int8MatmulPath::NeonI8mm => int8_matmul_prepacked_neon_i8mm_kernel,
        #[cfg(target_arch = "aarch64")]
        Int8MatmulPath::NeonDotprod => int8_matmul_prepacked_neon_sdot_kernel,
        #[cfg(target_arch = "aarch64")]
        Int8MatmulPath::NeonWiden => int8_matmul_prepacked_neon_widen_kernel,
        _ => int8_matmul_prepacked_scalar,
    })
}

#[cfg(target_arch = "x86_64")]
fn int8_matmul_avx512_vnni_kernel(
    a: &[i8],
    b: &[i8],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [i32],
) {
    // SAFETY: selected only after `host_cpu().features.x86_avx512_vnni()`.
    unsafe { int8_matmul_avx512_vnni(a, b, m, k, n, out) };
}

#[cfg(target_arch = "x86_64")]
fn int8_matmul_avx_vnni_kernel(a: &[i8], b: &[i8], m: usize, k: usize, n: usize, out: &mut [i32]) {
    // SAFETY: selected only after `host_cpu().features.x86_avx_vnni()`.
    unsafe { int8_matmul_avx_vnni(a, b, m, k, n, out) };
}

#[cfg(target_arch = "x86_64")]
fn int8_matmul_avx2_widen_kernel(
    a: &[i8],
    b: &[i8],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [i32],
) {
    // SAFETY: selected only after `host_cpu().features.avx2`.
    unsafe { int8_matmul_avx2_widen(a, b, m, k, n, out) };
}

#[cfg(target_arch = "aarch64")]
fn int8_matmul_neon_i8mm_kernel(a: &[i8], b: &[i8], m: usize, k: usize, n: usize, out: &mut [i32]) {
    // SAFETY: selected only after `host_cpu().features.aarch64_neon_i8mm()`.
    unsafe { int8_matmul_neon_i8mm(a, b, m, k, n, out) };
}

#[cfg(target_arch = "aarch64")]
fn int8_matmul_neon_sdot_kernel(a: &[i8], b: &[i8], m: usize, k: usize, n: usize, out: &mut [i32]) {
    // SAFETY: selected only after `host_cpu().features.aarch64_neon_dotprod()`.
    unsafe { int8_matmul_neon_sdot(a, b, m, k, n, out) };
}

#[cfg(target_arch = "aarch64")]
fn int8_matmul_neon_widen_kernel(
    a: &[i8],
    b: &[i8],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [i32],
) {
    // SAFETY: NEON is mandatory on aarch64; selected after `features.neon`.
    unsafe { int8_matmul_neon_widen(a, b, m, k, n, out) };
}

#[cfg(target_arch = "x86_64")]
fn int8_matmul_prepacked_avx512_vnni_kernel(a: &[i8], b: &PackedI8B, m: usize, out: &mut [i32]) {
    // SAFETY: selected only after `host_cpu().features.x86_avx512_vnni()`.
    unsafe { int8_matmul_prepacked_avx512_vnni(a, b, m, out) };
}

#[cfg(target_arch = "x86_64")]
fn int8_matmul_prepacked_avx_vnni_kernel(a: &[i8], b: &PackedI8B, m: usize, out: &mut [i32]) {
    // SAFETY: selected only after `host_cpu().features.x86_avx_vnni()`.
    unsafe { int8_matmul_prepacked_avx_vnni(a, b, m, out) };
}

#[cfg(target_arch = "x86_64")]
fn int8_matmul_prepacked_avx2_widen_kernel(a: &[i8], b: &PackedI8B, m: usize, out: &mut [i32]) {
    // SAFETY: selected only after `host_cpu().features.avx2`.
    unsafe { int8_matmul_prepacked_avx2_widen(a, b, m, out) };
}

#[cfg(target_arch = "aarch64")]
fn int8_matmul_prepacked_neon_i8mm_kernel(a: &[i8], b: &PackedI8B, m: usize, out: &mut [i32]) {
    // SAFETY: selected only after `host_cpu().features.aarch64_neon_i8mm()`.
    unsafe { int8_matmul_prepacked_neon_i8mm(a, b, m, out) };
}

#[cfg(target_arch = "aarch64")]
fn int8_matmul_prepacked_neon_sdot_kernel(a: &[i8], b: &PackedI8B, m: usize, out: &mut [i32]) {
    // SAFETY: selected only after `host_cpu().features.aarch64_neon_dotprod()`.
    unsafe { int8_matmul_prepacked_neon_sdot(a, b, m, out) };
}

#[cfg(target_arch = "aarch64")]
fn int8_matmul_prepacked_neon_widen_kernel(a: &[i8], b: &PackedI8B, m: usize, out: &mut [i32]) {
    // SAFETY: NEON is mandatory on aarch64; selected after `features.neon`.
    unsafe { int8_matmul_prepacked_neon_widen(a, b, m, out) };
}

/// Runtime-dispatched int8 GEMM. Picks the best variant available on the
/// host CPU; falls back to scalar when no SIMD path matches.
pub fn int8_matmul_dispatch(a: &[i8], b: &[i8], m: usize, k: usize, n: usize, out: &mut [i32]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(out.len(), m * n);

    selected_int8_matmul_kernel()(a, b, m, k, n, out);
}

/// Runtime-dispatched INT8 GEMM for a load-time packed RHS.
///
/// This has the same numerical contract as [`int8_matmul_dispatch`] but
/// avoids the per-call B transpose and gives every backend contiguous
/// `[K]` slices for each output column.
pub fn int8_matmul_prepacked_dispatch(a: &[i8], b: &PackedI8B, m: usize, out: &mut [i32]) {
    debug_assert_eq!(a.len(), m * b.k);
    debug_assert_eq!(out.len(), m * b.n);

    selected_int8_prepacked_kernel()(a, b, m, out);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ref_matmul(a: &[i8], b: &[i8], m: usize, k: usize, n: usize) -> Vec<i32> {
        let mut out = vec![0_i32; m * n];
        int8_matmul_scalar(a, b, m, k, n, &mut out);
        out
    }

    fn pseudo_random(seed: u64, n: usize) -> Vec<i8> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as i64 % 256 - 128) as i8
            })
            .collect()
    }

    #[test]
    fn scalar_zero_inputs_yield_zero_output() {
        let a = vec![0_i8; 12];
        let b = vec![0_i8; 12];
        let mut out = vec![1_i32; 9];
        int8_matmul_scalar(&a, &b, 3, 4, 3, &mut out);
        assert!(out.iter().all(|&v| v == 0));
    }

    #[test]
    fn scalar_known_small_case() {
        let a: Vec<i8> = vec![1, 2, 3, 4];
        let b: Vec<i8> = vec![5, 6, 7, 8];
        let mut out = vec![0_i32; 4];
        int8_matmul_scalar(&a, &b, 2, 2, 2, &mut out);
        assert_eq!(out, vec![19, 22, 43, 50]);
    }

    #[test]
    fn dispatch_matches_scalar_on_random_shapes() {
        for &(m, k, n) in &[(1, 1, 1), (2, 3, 4), (4, 16, 8), (5, 17, 6), (8, 64, 8)] {
            let a = pseudo_random(0xDEAD ^ (m * k) as u64, m * k);
            let b = pseudo_random(0xBEEF ^ (k * n) as u64, k * n);
            let expected = ref_matmul(&a, &b, m, k, n);
            let mut got = vec![i32::MIN; m * n];
            int8_matmul_dispatch(&a, &b, m, k, n, &mut got);
            assert_eq!(got, expected, "shape m={m} k={k} n={n}");
        }
    }

    #[test]
    fn prepacked_dispatch_matches_scalar_on_random_shapes() {
        for &(m, k, n) in &[
            (1, 1, 1),
            (1, 17, 5),
            (3, 31, 7),
            (4, 64, 16),
            (8, 127, 33),
            (16, 128, 64),
        ] {
            let a = pseudo_random(0xA11CE ^ (m * k) as u64, m * k);
            let b = pseudo_random(0xB0B ^ (k * n) as u64, k * n);
            let packed = pack_i8_b_for_matmul(&b, k, n);
            assert_eq!(packed.k(), k);
            assert_eq!(packed.n(), n);
            assert_eq!(packed.transposed().len(), k * n);
            let expected = ref_matmul(&a, &b, m, k, n);
            let mut got = vec![i32::MIN; m * n];
            int8_matmul_prepacked_dispatch(&a, &packed, m, &mut got);
            assert_eq!(got, expected, "shape m={m} k={k} n={n}");
        }
    }

    #[test]
    fn dispatch_handles_large_k_with_extreme_values() {
        let a = vec![-128_i8; 128];
        let b = vec![-128_i8; 128];
        let mut out = vec![0_i32; 1];
        int8_matmul_dispatch(&a, &b, 1, 128, 1, &mut out);
        assert_eq!(out[0], 128 * 128 * 128);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_sdot_matches_scalar_when_available() {
        if !crate::host_cpu().features.aarch64_neon_dotprod() {
            return;
        }
        let a = pseudo_random(0x1234, 32 * 32);
        let b = pseudo_random(0x5678, 32 * 16);
        let expected = ref_matmul(&a, &b, 32, 32, 16);
        let mut got = vec![0_i32; 32 * 16];
        unsafe { int8_matmul_neon_sdot(&a, &b, 32, 32, 16, &mut got) };
        assert_eq!(got, expected);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_widen_matches_scalar() {
        // NEON is mandatory on aarch64, so the widening path is always live.
        // Shapes cover K exactly 16, 16+8, 16+tail, 16+8+tail, and small K to
        // exercise the 16-wide body, the 8-wide step, and the scalar tail.
        for &(m, k, n) in &[
            (1, 1, 1),
            (2, 3, 4),
            (4, 16, 8),
            (5, 17, 6),
            (3, 24, 7),
            (8, 31, 5),
            (16, 128, 64),
        ] {
            let a = pseudo_random(0x9E37 ^ (m * k) as u64, m * k);
            let b = pseudo_random(0x7F4A ^ (k * n) as u64, k * n);
            let expected = ref_matmul(&a, &b, m, k, n);
            let mut got = vec![i32::MIN; m * n];
            unsafe { int8_matmul_neon_widen(&a, &b, m, k, n, &mut got) };
            assert_eq!(got, expected, "unpacked m={m} k={k} n={n}");
            let packed = pack_i8_b_for_matmul(&b, k, n);
            let mut got2 = vec![i32::MIN; m * n];
            unsafe { int8_matmul_prepacked_neon_widen(&a, &packed, m, &mut got2) };
            assert_eq!(got2, expected, "prepacked m={m} k={k} n={n}");
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_i8mm_matches_scalar_when_available() {
        let features = crate::host_cpu().features;
        if !features.aarch64_neon_i8mm() {
            return;
        }
        for &(m, k, n) in &[(8, 64, 16), (5, 32, 7), (4, 24, 4)] {
            let a = pseudo_random(0xAA ^ (m * k) as u64, m * k);
            let b = pseudo_random(0xBB ^ (k * n) as u64, k * n);
            let expected = ref_matmul(&a, &b, m, k, n);
            let mut got = vec![0_i32; m * n];
            unsafe { int8_matmul_neon_i8mm(&a, &b, m, k, n, &mut got) };
            assert_eq!(got, expected, "shape m={m} k={k} n={n}");

            let packed = pack_i8_b_for_matmul(&b, k, n);
            let mut got_prepacked = vec![0_i32; m * n];
            unsafe { int8_matmul_prepacked_neon_i8mm(&a, &packed, m, &mut got_prepacked) };
            assert_eq!(got_prepacked, expected, "prepacked shape m={m} k={k} n={n}");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_widen_matches_scalar_when_available() {
        if !crate::host_cpu().features.avx2 {
            return;
        }
        let a = pseudo_random(0x1234, 32 * 32);
        let b = pseudo_random(0x5678, 32 * 16);
        let expected = ref_matmul(&a, &b, 32, 32, 16);
        let mut got = vec![0_i32; 32 * 16];
        unsafe { int8_matmul_avx2_widen(&a, &b, 32, 32, 16, &mut got) };
        assert_eq!(got, expected);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx_vnni_matches_scalar_when_available() {
        let features = crate::host_cpu().features;
        if !features.x86_avx_vnni() {
            return;
        }
        let a = pseudo_random(0x1234, 8 * 96);
        let b = pseudo_random(0x5678, 96 * 8);
        let expected = ref_matmul(&a, &b, 8, 96, 8);
        let mut got = vec![0_i32; 8 * 8];
        unsafe { int8_matmul_avx_vnni(&a, &b, 8, 96, 8, &mut got) };
        assert_eq!(got, expected);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_vnni_matches_scalar_when_available() {
        let features = crate::host_cpu().features;
        if !features.x86_avx512_vnni() {
            return;
        }
        let a = pseudo_random(0xC0FFEE, 8 * 128);
        let b = pseudo_random(0xCAFE, 128 * 8);
        let expected = ref_matmul(&a, &b, 8, 128, 8);
        let mut got = vec![0_i32; 8 * 8];
        unsafe { int8_matmul_avx512_vnni(&a, &b, 8, 128, 8, &mut got) };
        assert_eq!(got, expected);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_vnni_blocked_mr8_path_matches_scalar() {
        let features = crate::host_cpu().features;
        if !features.x86_avx512_vnni() {
            return;
        }
        for &(m, k, n) in &[
            (8, 8, 16),
            (16, 16, 32),
            (8, 17, 16),  // K = 4*4 + 1 tail
            (24, 19, 48), // K = 4*4 + 3 tail, larger M
            (64, 256, 64),
        ] {
            let a = pseudo_random(0xB108 ^ (m * k) as u64, m * k);
            let b = pseudo_random(0xCAFE ^ (k * n) as u64, k * n);
            let expected = ref_matmul(&a, &b, m, k, n);
            let mut got = vec![0_i32; m * n];
            unsafe { int8_matmul_avx512_vnni_blocked_mr8(&a, &b, m, k, n, &mut got) };
            assert_eq!(got, expected, "MR=8 m={m} k={k} n={n}");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_vnni_blocked_path_matches_scalar() {
        let features = crate::host_cpu().features;
        if !features.x86_avx512_vnni() {
            return;
        }
        // Shapes that hit the MR=4 NR=16 register-blocked path. Cover
        // (a) clean alignment, (b) K not divisible by 4 (tail 1-3),
        // (c) larger K to ensure many group iterations.
        for &(m, k, n) in &[
            (4, 8, 16),
            (8, 16, 32),
            (4, 17, 16),  // K = 4*4 + 1, tail of 1
            (12, 19, 48), // K = 4*4 + 3, tail of 3
            (16, 256, 64),
        ] {
            let a = pseudo_random(0xB10C ^ (m * k) as u64, m * k);
            let b = pseudo_random(0xCAFE ^ (k * n) as u64, k * n);
            let expected = ref_matmul(&a, &b, m, k, n);
            let mut got = vec![0_i32; m * n];
            unsafe { int8_matmul_avx512_vnni_blocked(&a, &b, m, k, n, &mut got) };
            assert_eq!(got, expected, "blocked m={m} k={k} n={n}");
        }
    }
}
