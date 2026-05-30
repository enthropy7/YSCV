//! Activation (A) and weight (B) panel packing for the blocked GEMM.
//! Holds the packed-B session cache and the thread-local packed-A scratch.

use super::*;

/// Pack A[ic..ic+mc, pc..pc+kc] into panel format: (mc/MR) panels × kc × MR.
///
/// Each panel stores MR rows for all kc columns contiguously:
/// `packed[(ir/MR)*kc*MR + p*MR + i] = A[(ic+ir+i), (pc+p)]`
#[allow(unsafe_code)]
pub(super) fn pack_a_panel(
    a: &[f32],
    lda: usize,
    ic: usize,
    mc: usize,
    pc: usize,
    kc: usize,
    packed: &mut [f32],
) {
    let a_ptr = a.as_ptr();
    let p_ptr = packed.as_mut_ptr();
    let mut idx = 0usize;
    for ir in (0..mc).step_by(MR) {
        let mr = MR.min(mc - ir);
        if mr == MR {
            for p in 0..kc {
                unsafe {
                    for i in 0..MR {
                        *p_ptr.add(idx + i) = *a_ptr.add((ic + ir + i) * lda + pc + p);
                    }
                }
                idx += MR;
            }
        } else {
            for p in 0..kc {
                unsafe {
                    for i in 0..mr {
                        *p_ptr.add(idx + i) = *a_ptr.add((ic + ir + i) * lda + pc + p);
                    }
                    std::ptr::write_bytes(p_ptr.add(idx + mr), 0, MR - mr);
                }
                idx += MR;
            }
        }
    }
}

/// MR=8 variant of `pack_a_panel` for the aarch64 NEON 8×12 microkernel.
/// Groups rows in 8-row panels, tail rows zero-padded. aarch64 only —
/// x86/scalar stay on MR=4.
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
pub(super) fn pack_a_panel_mr8(
    a: &[f32],
    lda: usize,
    ic: usize,
    mc: usize,
    pc: usize,
    kc: usize,
    packed: &mut [f32],
) {
    const MR8: usize = 8;
    let a_ptr = a.as_ptr();
    let p_ptr = packed.as_mut_ptr();
    let mut idx = 0usize;
    for ir in (0..mc).step_by(MR8) {
        let mr = MR8.min(mc - ir);
        if mr == MR8 {
            for p in 0..kc {
                unsafe {
                    for i in 0..MR8 {
                        *p_ptr.add(idx + i) = *a_ptr.add((ic + ir + i) * lda + pc + p);
                    }
                }
                idx += MR8;
            }
        } else {
            for p in 0..kc {
                unsafe {
                    for i in 0..mr {
                        *p_ptr.add(idx + i) = *a_ptr.add((ic + ir + i) * lda + pc + p);
                    }
                    std::ptr::write_bytes(p_ptr.add(idx + mr), 0, MR8 - mr);
                }
                idx += MR8;
            }
        }
    }
}

/// MR=6 variant of `pack_a_panel` for the x86 MR=6×16 AVX2
/// microkernel. Groups rows in 6-row panels; tail rows zero-padded.
/// Pack layout matches `pack_a_panel` style: (mc/MR6) panels × kc × MR6.
///
/// Used only by the MR=6 fast path when `use_mr6_blocked()` returns true
/// (x86 with FMA+AVX, m≥192, k≥64). MR=4 path continues for tail and
/// small shapes.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[allow(unsafe_code)]
pub(super) fn pack_a_panel_mr6(
    a: &[f32],
    lda: usize,
    ic: usize,
    mc: usize,
    pc: usize,
    kc: usize,
    packed: &mut [f32],
) {
    let a_ptr = a.as_ptr();
    let p_ptr = packed.as_mut_ptr();
    let mut idx = 0usize;
    for ir in (0..mc).step_by(MR6) {
        let mr = MR6.min(mc - ir);
        if mr == MR6 {
            for p in 0..kc {
                // SAFETY: `idx + MR6` stays within `packed` (sized
                // `div_ceil(mc, MR6) * kc * MR6` by caller);
                // `(ic+ir+i)*lda + pc + p` is inside A's full panel.
                unsafe {
                    for i in 0..MR6 {
                        *p_ptr.add(idx + i) = *a_ptr.add((ic + ir + i) * lda + pc + p);
                    }
                }
                idx += MR6;
            }
        } else {
            for p in 0..kc {
                // SAFETY: partial row — zero-pad the tail to MR6 so the
                // microkernel can always load MR6 floats per k.
                unsafe {
                    for i in 0..mr {
                        *p_ptr.add(idx + i) = *a_ptr.add((ic + ir + i) * lda + pc + p);
                    }
                    std::ptr::write_bytes(p_ptr.add(idx + mr), 0, MR6 - mr);
                }
                idx += MR6;
            }
        }
    }
}

/// Pack A from NCHWc input layout into the standard panel format.
///
/// `input` is one batch item from a `[Cb, H*W, block]` NCHWc buffer.
/// M-dimension = spatial positions (H*W), K-dimension = IC channels.
/// For spatial position `m` and channel `k`:
///   `input[cb * cb_stride + m * block + lane]`
/// where `cb = k / block`, `lane = k % block`, `cb_stride = hw * block`.
///
/// Output panel format is identical to `pack_a_panel`: `(mc/MR) × kc × MR`.
/// Allows the existing `gebp_kernel_raw` microkernels to operate on NCHWc
/// activations without a prior layout conversion.
#[allow(unsafe_code)]
pub(super) fn pack_a_panel_nchwc(
    input: &[f32],
    hw: usize,
    block: usize,
    ic: usize,
    mc: usize,
    pc: usize,
    kc: usize,
    packed: &mut [f32],
) {
    let cb_stride = hw * block;
    let i_ptr = input.as_ptr();
    let p_ptr = packed.as_mut_ptr();
    let mut idx = 0usize;
    for ir in (0..mc).step_by(MR) {
        let mr = MR.min(mc - ir);
        if mr == MR {
            for p in 0..kc {
                let k = pc + p;
                let cb = k / block;
                let lane = k % block;
                let base = cb * cb_stride + lane;
                unsafe {
                    for i in 0..MR {
                        *p_ptr.add(idx + i) = *i_ptr.add(base + (ic + ir + i) * block);
                    }
                }
                idx += MR;
            }
        } else {
            for p in 0..kc {
                let k = pc + p;
                let cb = k / block;
                let lane = k % block;
                let base = cb * cb_stride + lane;
                unsafe {
                    for i in 0..mr {
                        *p_ptr.add(idx + i) = *i_ptr.add(base + (ic + ir + i) * block);
                    }
                    std::ptr::write_bytes(p_ptr.add(idx + mr), 0, MR - mr);
                }
                idx += MR;
            }
        }
    }
}

/// Pack B[pc..pc+kc, jc..jc+nc] into panel format: (nc/NR) panels × kc × NR.
///
/// Each panel stores NR columns for all kc rows contiguously:
/// `packed[(jr/NR)*kc*NR + p*NR + j] = B[(pc+p), (jc+jr+j)]`
#[allow(unsafe_code)]
pub(super) fn pack_b_panel(
    b: &[f32],
    ldb: usize,
    pc: usize,
    kc: usize,
    jc: usize,
    nc: usize,
    packed: &mut [f32],
) {
    let b_ptr = b.as_ptr();
    let p_ptr = packed.as_mut_ptr();
    let mut idx = 0usize;
    for jr in (0..nc).step_by(NR) {
        let nr = NR.min(nc - jr);
        if nr == NR {
            for p in 0..kc {
                let src = (pc + p) * ldb + jc + jr;
                unsafe {
                    std::ptr::copy_nonoverlapping(b_ptr.add(src), p_ptr.add(idx), NR);
                }
                idx += NR;
            }
        } else {
            for p in 0..kc {
                let src = (pc + p) * ldb + jc + jr;
                unsafe {
                    std::ptr::copy_nonoverlapping(b_ptr.add(src), p_ptr.add(idx), nr);
                    std::ptr::write_bytes(p_ptr.add(idx + nr), 0, NR - nr);
                }
                idx += NR;
            }
        }
    }
}

/// aarch64-only: NR=12-aligned B packer for the MR=8×12 NEON kernel.
///
/// The generic `pack_b_panel` uses NR=8; the 8×12 kernel needs 12 cols
/// per k-step in one contiguous load (`ld1 {v24,v25,v26}, [x1]`) which
/// forces a different packed stride. Layout mirrors `pack_b_panel` but
/// with `NR_MR8=12` panels:
///   `packed[jr_panel * kc * 12 + p * 12 + j]` = `B[pc+p][jc+jr_panel*12+j]`.
/// Tail cols (nc not a multiple of 12) zero-padded so every panel is
/// exactly 12 floats per k-step.
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
pub(super) fn pack_b_panel_nr12(
    b: &[f32],
    ldb: usize,
    pc: usize,
    kc: usize,
    jc: usize,
    nc: usize,
    packed: &mut [f32],
) {
    const NR12: usize = 12;
    let b_ptr = b.as_ptr();
    let p_ptr = packed.as_mut_ptr();
    let mut idx = 0usize;
    for jr in (0..nc).step_by(NR12) {
        let nr = NR12.min(nc - jr);
        if nr == NR12 {
            for p in 0..kc {
                let src = (pc + p) * ldb + jc + jr;
                unsafe {
                    std::ptr::copy_nonoverlapping(b_ptr.add(src), p_ptr.add(idx), NR12);
                }
                idx += NR12;
            }
        } else {
            for p in 0..kc {
                let src = (pc + p) * ldb + jc + jr;
                unsafe {
                    std::ptr::copy_nonoverlapping(b_ptr.add(src), p_ptr.add(idx), nr);
                    std::ptr::write_bytes(p_ptr.add(idx + nr), 0, NR12 - nr);
                }
                idx += NR12;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Packed-B cache: kernel weights are constant across inferences, so packing
// them every call is pure waste. Cache the fully-packed form keyed by the B
// buffer's pointer + (K, N) so repeat calls with the same kernel are free.
// ---------------------------------------------------------------------------

/// One packed B contains every (pc_idx, jc_idx) block of a (K × N) matrix,
/// stored contiguously. Block layout per block matches `pack_b_panel` output.
///
/// Publicly exposed so the ONNX loader can pre-pack constant Conv/MatMul
/// weights at session-create and hand them out as `Arc<PackedB>` — the
/// per-inference hot path then skips both the fingerprint lookup and the
/// pack itself. Pre-packed data is immutable after construction, so `PackedB`
/// is `Send + Sync` (the internal `Vec<f32>` is never mutated post-build).
pub struct PackedB {
    /// Contiguous storage: blocks laid out in (pc_idx, jc_idx) row-major order.
    pub(crate) data: Vec<f32>,
    /// `block_slots` floats per block — equals `div_ceil(NC, NR) * KC * NR`.
    /// Same value for NR=32: `(NC_MR12 / NR32) * KC * NR32 == div_ceil(NC, NR) * KC * NR`.
    pub(crate) block_slots: usize,
    /// Number of column-blocks (`div_ceil(N, NC)`).
    pub(crate) num_jc: usize,
    /// Dimensions of the source B matrix. Used to reject stale packs if shape
    /// changes (should never happen for initializer kernels, but cheap check).
    pub(crate) k: usize,
    pub(crate) n: usize,
    /// NR=32 packed data for AVX-512 MR=12×NR=32 path. Shares the same block
    /// layout as `data` (same `block_slots`, `num_jc`; different intra-block
    /// stride). Non-empty only on x86_64 AVX-512F hosts when `n % 32 == 0
    /// && k >= 16`. Empty vec = unavailable; fall back to inline packing.
    #[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
    pub(crate) data_nr32: Vec<f32>,
}

impl PackedB {
    #[inline]
    pub(crate) fn block(&self, pc_idx: usize, jc_idx: usize) -> &[f32] {
        let off = (pc_idx * self.num_jc + jc_idx) * self.block_slots;
        &self.data[off..off + self.block_slots]
    }

    /// Returns the NR=32 block for `(pc_idx, jc_idx)`, or `None` if NR=32 data
    /// was not pre-packed (non-AVX-512 host or shape doesn't qualify).
    #[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
    #[inline]
    pub(crate) fn block_nr32(&self, pc_idx: usize, jc_idx: usize) -> Option<&[f32]> {
        if self.data_nr32.is_empty() {
            return None;
        }
        let off = (pc_idx * self.num_jc + jc_idx) * self.block_slots;
        Some(&self.data_nr32[off..off + self.block_slots])
    }

    #[inline]
    pub(crate) fn matches(&self, k: usize, n: usize) -> bool {
        self.k == k && self.n == n
    }

    /// Dimensions of the source B matrix (for dispatch shape checks at the
    /// callsite before handing the pre-pack to the GEMM layer).
    #[inline]
    pub fn dims(&self) -> (usize, usize) {
        (self.k, self.n)
    }
}

// Auto-derived Send+Sync via `Vec<f32>` + `usize` fields. Explicit note for
// readers: nothing in PackedB is mutated after `full_pack_b` returns, so
// sharing the `Arc<PackedB>` across threads at inference time is safe.

impl std::fmt::Debug for PackedB {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        #[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
        let has_nr32 = !self.data_nr32.is_empty();
        #[cfg(not(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos"))))]
        let has_nr32 = false;
        write!(
            f,
            "PackedB {{ k: {}, n: {}, data_len: {}, block_slots: {}, num_jc: {}, has_nr32: {} }}",
            self.k,
            self.n,
            self.data.len(),
            self.block_slots,
            self.num_jc,
            has_nr32
        )
    }
}

/// Pre-packs a constant B matrix (e.g. a Conv kernel weight) into the
/// blocked-GEMM layout for later zero-overhead reuse. Intended for use at
/// model load time — returned `Arc<PackedB>` can be handed to every
/// subsequent `matmul_2d_slices_fused_prepacked` call without hashing,
/// locking, or re-packing.
pub fn pack_b_for_session(b: &[f32], k: usize, n: usize) -> std::sync::Arc<PackedB> {
    std::sync::Arc::new(full_pack_b(b, k, n))
}

/// FEAT_FP16 half-precision 6×16 GEMM for a single tile (6 rows × 16 cols).
/// `a_panel`, `b_panel_{0,1}` are packed fp16 bit-patterns (u16); `c` is
/// fp16 output. Requires aarch64 + FEAT_FP16 — caller verifies via
/// `std::arch::is_aarch64_feature_detected!("fp16")` and routes scalar
/// otherwise. Each matmul invocation tiles at the caller layer; this
/// function is the microkernel.
///
/// Opt-in via the yscv-onnx loader env `YSCV_FP16=1`. Not wired into the
/// default Conv dispatch yet because load-time weight casting to fp16 +
/// accuracy validation on real ARM hw are prerequisites (see Phase 3.J
/// in the roadmap).
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
pub fn hgemm_6x16_neon(
    accumulate: bool,
    a_panel: &[u16],
    b_panel_0: &[u16],
    b_panel_1: &[u16],
    c: &mut [u16],
    ldc: usize,
    kc: usize,
) {
    // Safety: caller must size panels correctly (6*kc, kc*8, kc*8) and
    // ensure `c` has at least 6*ldc valid u16 entries. We debug_assert
    // shape bounds to catch misuse in tests.
    debug_assert!(a_panel.len() >= 6 * kc);
    debug_assert!(b_panel_0.len() >= kc * 8);
    debug_assert!(b_panel_1.len() >= kc * 8);
    debug_assert!(c.len() >= 5 * ldc + 16);
    unsafe {
        if accumulate {
            sgemm_asm_aarch64::yscv_hgemm_6x16_neon_acc(
                a_panel.as_ptr(),
                b_panel_0.as_ptr(),
                b_panel_1.as_ptr(),
                c.as_mut_ptr(),
                ldc,
                kc,
            );
        } else {
            sgemm_asm_aarch64::yscv_hgemm_6x16_neon_set(
                a_panel.as_ptr(),
                b_panel_0.as_ptr(),
                b_panel_1.as_ptr(),
                c.as_mut_ptr(),
                ldc,
                kc,
            );
        }
    }
}

pub(super) fn full_pack_b(b: &[f32], k: usize, n: usize) -> PackedB {
    let num_pc = div_ceil(k, KC);
    let num_jc = div_ceil(n, NC);
    let block_slots = div_ceil(NC, NR) * KC * NR;
    let total = num_pc * num_jc * block_slots;
    // Zero-filled allocation — `pack_b_panel` below overwrites every slot,
    // the pre-zero is just a safety pre-condition for `clippy::uninit_vec`.
    // Called once per (weight, K, N) tuple then cached, so the memset
    // cost is amortized to near-zero per inference.
    let mut data: Vec<f32> = vec![0.0; total];

    for pc_idx in 0..num_pc {
        let pc = pc_idx * KC;
        let kc = KC.min(k - pc);
        for jc_idx in 0..num_jc {
            let jc = jc_idx * NC;
            let nc = NC.min(n - jc);
            let block_off = (pc_idx * num_jc + jc_idx) * block_slots;
            // pack_b_panel writes `div_ceil(nc, NR) * kc * NR` floats — we give
            // it the full block_slots slice which is the max possible.
            pack_b_panel(
                b,
                n,
                pc,
                kc,
                jc,
                nc,
                &mut data[block_off..block_off + block_slots],
            );
        }
    }

    // On x86_64 AVX-512F hosts, also pre-pack in NR=32 format for the
    // MR=12×NR=32 path when the shape qualifies. `block_slots` is identical
    // for NR=32 (`(NC_MR12/NR32)*KC*NR32 = 256*KC`) so we reuse the same
    // block indexing. Pre-packing here eliminates per-inference B packing in
    // `blocked_gemm_*_mr12`, which was the cause of the P1.5 regression.
    #[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
    let data_nr32 = {
        if n.is_multiple_of(NR32)
            && k >= 16
            && avx512_mr12_enabled()
            && std::is_x86_feature_detected!("avx512f")
        {
            let mut d = vec![0.0f32; total];
            for pc_idx in 0..num_pc {
                let pc = pc_idx * KC;
                let kc = KC.min(k - pc);
                for jc_idx in 0..num_jc {
                    let jc = jc_idx * NC_MR12; // NC_MR12 == NC == 256
                    let nc = NC_MR12.min(n - jc);
                    let nc_aligned = (nc / NR32) * NR32;
                    if nc_aligned > 0 {
                        let block_off = (pc_idx * num_jc + jc_idx) * block_slots;
                        pack_b_panel_nr32(
                            b,
                            n,
                            pc,
                            kc,
                            jc,
                            nc_aligned,
                            &mut d[block_off..block_off + block_slots],
                        );
                    }
                }
            }
            d
        } else {
            Vec::new()
        }
    };

    PackedB {
        data,
        block_slots,
        num_jc,
        k,
        n,
        #[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
        data_nr32,
    }
}

thread_local! {
    /// Runtime cache for callers that don't pre-pack. Key includes a few
    /// fingerprint samples of the B buffer so we don't return a stale pack
    /// when a temporary buffer is reallocated at the same address.
    static PACKED_B_CACHE: std::cell::RefCell<
        std::collections::HashMap<(usize, usize, usize, u32, u32, u32), std::rc::Rc<PackedB>>,
    > = std::cell::RefCell::new(std::collections::HashMap::new());
}

/// aarch64-only NR=12 packed B companion to `PackedB`. Shares the
/// `(pc_idx, jc_idx)` block-layout idea but uses a 12-col stride so the
/// 8×12 NEON kernel can load three quads per k-step from one contiguous
/// panel (the generic NR=8 layout can't satisfy that).
#[cfg(target_arch = "aarch64")]
pub(crate) struct PackedBNr12 {
    pub(crate) data: Vec<f32>,
    pub(crate) block_slots: usize,
    pub(crate) num_jc: usize,
    pub(crate) k: usize,
    pub(crate) n: usize,
}

#[cfg(target_arch = "aarch64")]
impl PackedBNr12 {
    #[inline]
    pub(crate) fn block(&self, pc_idx: usize, jc_idx: usize) -> &[f32] {
        let off = (pc_idx * self.num_jc + jc_idx) * self.block_slots;
        &self.data[off..off + self.block_slots]
    }
}

#[cfg(target_arch = "aarch64")]
pub(super) fn full_pack_b_nr12(b: &[f32], k: usize, n: usize) -> PackedBNr12 {
    const NR12: usize = 12;
    let num_pc = div_ceil(k, KC);
    let num_jc = div_ceil(n, NC);
    let block_slots = div_ceil(NC, NR12) * KC * NR12;
    let total = num_pc * num_jc * block_slots;
    // Zero-filled; `pack_b_panel_nr12` writes the active slots, tail
    // padding stays zero-pre-init. See `full_pack_b` for the same pattern.
    let mut data: Vec<f32> = vec![0.0; total];

    for pc_idx in 0..num_pc {
        let pc = pc_idx * KC;
        let kc = KC.min(k - pc);
        for jc_idx in 0..num_jc {
            let jc = jc_idx * NC;
            let nc = NC.min(n - jc);
            let block_off = (pc_idx * num_jc + jc_idx) * block_slots;
            pack_b_panel_nr12(
                b,
                n,
                pc,
                kc,
                jc,
                nc,
                &mut data[block_off..block_off + block_slots],
            );
        }
    }
    PackedBNr12 {
        data,
        block_slots,
        num_jc,
        k,
        n,
    }
}

#[cfg(target_arch = "aarch64")]
thread_local! {
    static PACKED_B_NR12_CACHE: std::cell::RefCell<
        std::collections::HashMap<
            (usize, usize, usize, u32, u32, u32),
            std::rc::Rc<PackedBNr12>,
        >,
    > = std::cell::RefCell::new(std::collections::HashMap::new());
}

#[cfg(target_arch = "aarch64")]
pub(super) fn get_or_pack_b_nr12(b: &[f32], k: usize, n: usize) -> std::rc::Rc<PackedBNr12> {
    let ptr_key = b.as_ptr() as usize;
    // Fingerprint 3 evenly-spaced elements so same-address re-allocations
    // of different tensors don't silently alias. Matches the NR=8 cache.
    let len = b.len().max(1);
    let fp0 = if !b.is_empty() { b[0].to_bits() } else { 0 };
    let fp1 = b[len / 2].to_bits();
    let fp2 = b[len - 1].to_bits();
    let key = (ptr_key, k, n, fp0, fp1, fp2);

    PACKED_B_NR12_CACHE.with(|cache| {
        if let Some(hit) = cache.borrow().get(&key)
            && hit.k == k
            && hit.n == n
        {
            return std::rc::Rc::clone(hit);
        }
        let packed = std::rc::Rc::new(full_pack_b_nr12(b, k, n));
        cache.borrow_mut().insert(key, std::rc::Rc::clone(&packed));
        packed
    })
}

thread_local! {
    /// Per-thread scratch for `pack_a_panel` output. Sized on-demand
    /// via `ensure_capacity`; reused across every blocked-GEMM call on this
    /// thread. Previously each `par_for_each_ic` closure did a fresh
    /// `Vec::<f32>::with_capacity(pa_size)` per `(jc, pc)` block per worker —
    /// for tracker's 132 Conv ops × 6 threads × 2-4 blocks = thousands of
    /// heap allocs per inference, amortized ~100-200 ns each. TLS reuse
    /// eliminates the allocator pressure entirely (one warmup alloc per
    /// thread, then constant thereafter).
    ///
    /// Initial capacity 64 KB (16K f32) covers typical MC×KC×MR = 128×256×4
    /// / MR = 32 768 floats / 128 KB for m<128 shapes. Auto-grows to the
    /// max observed size across the process lifetime.
    static PACKED_A_SCRATCH: std::cell::RefCell<Vec<f32>> =
        std::cell::RefCell::new(Vec::with_capacity(16 * 1024));
}

/// Run `f` with a thread-local `pa_size`-sized packed-A scratch buffer.
/// The buffer persists across calls — zero allocator traffic in the hot
/// path once warm.
#[inline]
pub(super) fn with_packed_a_tls<R>(pa_size: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
    PACKED_A_SCRATCH.with(|cell| {
        let mut buf = cell.borrow_mut();
        let have = buf.len();
        if buf.capacity() < pa_size {
            buf.reserve(pa_size.saturating_sub(have));
        }
        // SAFETY: we just ensured `capacity >= pa_size`. `pack_a_panel`
        // writes all `pa_size` elements before the first read, so the
        // uninitialized data is never observed.
        #[allow(unsafe_code)]
        unsafe {
            buf.set_len(pa_size);
        }
        f(&mut buf[..pa_size])
    })
}

pub(super) fn get_or_pack_b(b: &[f32], k: usize, n: usize) -> std::rc::Rc<PackedB> {
    // Cache key = (pointer, K, N) ALONE is unsound: if the caller hands us a
    // temporary buffer (e.g. a per-inference layout-converted kernel), the
    // allocator can reuse the same address across calls with different data.
    // Fingerprint a few sample f32 values so we detect aliasing and re-pack.
    let key_ptr = b.as_ptr() as usize;
    let sample0 = b.first().copied().unwrap_or(0.0).to_bits();
    let sample1 = b.get(b.len() / 2).copied().unwrap_or(0.0).to_bits();
    let sample2 = b.last().copied().unwrap_or(0.0).to_bits();
    let key = (key_ptr, k, n, sample0, sample1, sample2);
    PACKED_B_CACHE.with(|cache| {
        if let Some(rc) = cache.borrow().get(&key) {
            return rc.clone();
        }
        let packed = std::rc::Rc::new(full_pack_b(b, k, n));
        cache.borrow_mut().insert(key, packed.clone());
        packed
    })
}
