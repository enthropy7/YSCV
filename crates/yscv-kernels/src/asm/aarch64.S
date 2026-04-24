// yscv SGEMM microkernels — aarch64 AAPCS64 (Linux, macOS, Android, iOS).
//
// AAPCS64 calling convention:
//   Args:      x0, x1, x2, x3, x4, x5, x6, x7 (first 8 in GPRs).
//   Callee-saved GPR: x19–x29 (frame pointer = x29, LR = x30).
//   Callee-saved FP:  v8–v15, **low 64 bits only** (we treat as d8–d15).
//
// Our 4×24 kernel uses v0–v23 as 24 accumulators, so v8–v15 need to be
// saved on entry and restored before return. `stp d{n}, d{n+1}, [sp, …]`
// saves the low 64 bits; the high 64 bits of v8–v15 are caller-saved
// and we're free to clobber them.

#include "asm_common.h"

// ============================================================================
// yscv_sgemm_4x24_neon_set / yscv_sgemm_4x24_neon_acc  (4 rows × 24 cols)
// ============================================================================
// Pack layout matches the Rust-side `pack_a_panel` / `pack_b_panel`:
//   A — MR=4 contiguous per k (4 floats, 16 bytes) then advance panel.
//   B — three NR=8 panels (b0, b1, b2), each 8 floats per k (32 bytes).
//
// Register schedule:
//   v0–v23  = 24 accumulators (4 rows × 6 quads, row-major).
//             row r, tile t  →  v[r*6 + t].
//   v24–v29 = B[k] loaded: v24..v25 from b0, v26..v27 from b1, v28..v29 from b2.
//   v30     = A[k] broadcast: all 4 MR values in one v-reg lane-wise.
//             v30.s[0..3] = A[k][0..3], consumed via `fmla v.4s, v.4s, v30.s[i]`.
//   v31     = spare (used as transient for aligned store pairs).
//
// Args:
//   x0  a_panel
//   x1  b_panel_0
//   x2  b_panel_1
//   x3  b_panel_2
//   x4  c
//   x5  ldc (in FLOATS; converted to bytes in prologue)
//   x6  kc
//
// k-loop: 24 `fmla v.4s, v.4s, v30.s[i]` per step — all 24 accumulator
// chains are independent, so the two NEON FMA pipes (A76, N1, Apple M)
// issue back-to-back without RAW stalls. Expected throughput: 12 cycles
// per k-step on a 2-FMA-pipe core.

.globl CNAME(yscv_sgemm_4x24_neon_set)
#if !defined(__APPLE__)
.type yscv_sgemm_4x24_neon_set, %function
#endif
.p2align 2
CNAME(yscv_sgemm_4x24_neon_set):
    // Save callee-saved v8–v15 (low 64 bits).
    stp d8,  d9,  [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]

    // ldc floats → bytes.
    lsl x5, x5, #2

    // Row pointers for the 4 output rows.
    mov x7,  x4
    add x8,  x4, x5
    add x9,  x8, x5
    add x10, x9, x5

    // Zero all 24 accumulators.
    movi v0.4s,  #0
    movi v1.4s,  #0
    movi v2.4s,  #0
    movi v3.4s,  #0
    movi v4.4s,  #0
    movi v5.4s,  #0
    movi v6.4s,  #0
    movi v7.4s,  #0
    movi v8.4s,  #0
    movi v9.4s,  #0
    movi v10.4s, #0
    movi v11.4s, #0
    movi v12.4s, #0
    movi v13.4s, #0
    movi v14.4s, #0
    movi v15.4s, #0
    movi v16.4s, #0
    movi v17.4s, #0
    movi v18.4s, #0
    movi v19.4s, #0
    movi v20.4s, #0
    movi v21.4s, #0
    movi v22.4s, #0
    movi v23.4s, #0
    b .Lsgemm_4x24_kloop
#if !defined(__APPLE__)
.size yscv_sgemm_4x24_neon_set, . - yscv_sgemm_4x24_neon_set
#endif

.globl CNAME(yscv_sgemm_4x24_neon_acc)
#if !defined(__APPLE__)
.type yscv_sgemm_4x24_neon_acc, %function
#endif
.p2align 2
CNAME(yscv_sgemm_4x24_neon_acc):
    stp d8,  d9,  [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]

    lsl x5, x5, #2
    mov x7,  x4
    add x8,  x4, x5
    add x9,  x8, x5
    add x10, x9, x5

    // Load existing c into accumulators (6 × 16 bytes per row).
    ldp q0,  q1,  [x7]
    ldp q2,  q3,  [x7, #32]
    ldp q4,  q5,  [x7, #64]
    ldp q6,  q7,  [x8]
    ldp q8,  q9,  [x8, #32]
    ldp q10, q11, [x8, #64]
    ldp q12, q13, [x9]
    ldp q14, q15, [x9, #32]
    ldp q16, q17, [x9, #64]
    ldp q18, q19, [x10]
    ldp q20, q21, [x10, #32]
    ldp q22, q23, [x10, #64]
    b .Lsgemm_4x24_kloop
#if !defined(__APPLE__)
.size yscv_sgemm_4x24_neon_acc, . - yscv_sgemm_4x24_neon_acc
#endif

// Shared k-loop and store tail.
.p2align 4
.Lsgemm_4x24_kloop:
    cbz x6, .Lsgemm_4x24_store

.p2align 4
.Lsgemm_4x24_kbody:
    // Load B[k] — 6 quadwords total, three panels × two NR/4 slices.
    ld1 {v24.4s, v25.4s}, [x1]
    ld1 {v26.4s, v27.4s}, [x2]
    ld1 {v28.4s, v29.4s}, [x3]
    // Load A[k] — 4 floats → one quadword with MR values lane-wise.
    ld1 {v30.4s}, [x0]

    // Row 0 (a0) — 6 FMAs
    fmla v0.4s,  v24.4s, v30.s[0]
    fmla v1.4s,  v25.4s, v30.s[0]
    fmla v2.4s,  v26.4s, v30.s[0]
    fmla v3.4s,  v27.4s, v30.s[0]
    fmla v4.4s,  v28.4s, v30.s[0]
    fmla v5.4s,  v29.4s, v30.s[0]
    // Row 1 (a1)
    fmla v6.4s,  v24.4s, v30.s[1]
    fmla v7.4s,  v25.4s, v30.s[1]
    fmla v8.4s,  v26.4s, v30.s[1]
    fmla v9.4s,  v27.4s, v30.s[1]
    fmla v10.4s, v28.4s, v30.s[1]
    fmla v11.4s, v29.4s, v30.s[1]
    // Row 2 (a2)
    fmla v12.4s, v24.4s, v30.s[2]
    fmla v13.4s, v25.4s, v30.s[2]
    fmla v14.4s, v26.4s, v30.s[2]
    fmla v15.4s, v27.4s, v30.s[2]
    fmla v16.4s, v28.4s, v30.s[2]
    fmla v17.4s, v29.4s, v30.s[2]
    // Row 3 (a3)
    fmla v18.4s, v24.4s, v30.s[3]
    fmla v19.4s, v25.4s, v30.s[3]
    fmla v20.4s, v26.4s, v30.s[3]
    fmla v21.4s, v27.4s, v30.s[3]
    fmla v22.4s, v28.4s, v30.s[3]
    fmla v23.4s, v29.4s, v30.s[3]

    add x0, x0, #16                 // a += MR floats
    add x1, x1, #32                 // b0 += NR floats
    add x2, x2, #32
    add x3, x3, #32
    subs x6, x6, #1
    b.ne .Lsgemm_4x24_kbody

.Lsgemm_4x24_store:
    // Store 6 quadwords per row.
    stp q0,  q1,  [x7]
    stp q2,  q3,  [x7, #32]
    stp q4,  q5,  [x7, #64]
    stp q6,  q7,  [x8]
    stp q8,  q9,  [x8, #32]
    stp q10, q11, [x8, #64]
    stp q12, q13, [x9]
    stp q14, q15, [x9, #32]
    stp q16, q17, [x9, #64]
    stp q18, q19, [x10]
    stp q20, q21, [x10, #32]
    stp q22, q23, [x10, #64]

    // Restore callee-saved d8–d15.
    ldp d14, d15, [sp, #48]
    ldp d12, d13, [sp, #32]
    ldp d10, d11, [sp, #16]
    ldp d8,  d9,  [sp], #64
    ret

// ============================================================================
// yscv_sgemm_8x12_neon_set / _acc  (NEON, 8 rows × 12 cols, inline epilogue)
// ============================================================================
// MR=8, NR=12 tile. 24 accumulators (v0..v23) hold 8 rows × 3 v.4s quads
// (12 cols). 2× A-broadcast amortization vs MR=4: each pair of A values
// drives 6 FMAs before the next broadcast arrives. Per k-step: 24 FMAs
// on Cortex-A76's 2 FMA pipes = 12 cycles backend.
//
// **B-PACK CONTRACT:** NR=12 packed layout (one panel = kc rows × 12 f32
// contiguous per k-step). NOT the generic NR=8 packer — MR=8 has its own
// `pack_b_panel_nr12` (see Phase 0.3 in matmul.rs). Phase 0.3 rewrite
// kept the 8×12 kernel shape intact; the ONLY change vs the previous
// broken version is that the kernel now assumes NR=12-aligned panels and
// the caller advances `jr += 12` on panels that match this stride. The
// earlier implementation called this kernel over NR=8-packed memory with
// `jr += 12`, reading the wrong cols after the first iteration.
//
// Register plan (29 of 32 v-regs):
//   v0..v23  = 24 accumulators (row r, tile t → v[r*3 + t]; t=0 cols 0..3,
//              t=1 cols 4..7, t=2 cols 8..11).
//   v24..v26 = B[k] cols 0..11 (3 quads from the single NR=12 packed panel)
//   v27..v28 = A[k] 8 floats packed in 2 × v.4s (lanes 0..3 each)
//
// Callee-saved per AAPCS64: v8-v15 low 64 bits. Save/restore at entry/exit.
//
// Args (AAPCS64):
//   x0 = a_panel (packed MR=8 × kc f32, 32 bytes per k)
//   x1 = b_panel (packed kc × NR=12, 48 bytes per k — single panel!)
//   x2 = c (output)
//   x3 = ldc (FLOATS, converted to bytes in prologue)
//   x4 = kc
//   x5 = bias ptr (0 = no bias, 12 floats when set)
//   x6 = activation id (0=None, 1=Relu); SiLU routed to MR=4 by caller
//   x7 = is_last_k (0 or 1)

    .globl CNAME(yscv_sgemm_8x12_neon_set)
#if !defined(__APPLE__)
    .type yscv_sgemm_8x12_neon_set, %function
    .p2align 2
yscv_sgemm_8x12_neon_set:
#else
    .p2align 2
_yscv_sgemm_8x12_neon_set:
#endif
    stp d8,  d9,  [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]

    lsl x3, x3, #2                      // ldc floats → bytes

    // Row pointers c0..c7 via ldc stride.
    mov x8,  x2                         // row 0
    add x9,  x2, x3                     // row 1
    add x10, x9, x3                     // row 2
    add x11, x10, x3                    // row 3
    add x12, x11, x3                    // row 4
    add x13, x12, x3                    // row 5
    add x14, x13, x3                    // row 6
    add x15, x14, x3                    // row 7

    // Zero 24 accumulators.
    movi v0.4s,  #0
    movi v1.4s,  #0
    movi v2.4s,  #0
    movi v3.4s,  #0
    movi v4.4s,  #0
    movi v5.4s,  #0
    movi v6.4s,  #0
    movi v7.4s,  #0
    movi v8.4s,  #0
    movi v9.4s,  #0
    movi v10.4s, #0
    movi v11.4s, #0
    movi v12.4s, #0
    movi v13.4s, #0
    movi v14.4s, #0
    movi v15.4s, #0
    movi v16.4s, #0
    movi v17.4s, #0
    movi v18.4s, #0
    movi v19.4s, #0
    movi v20.4s, #0
    movi v21.4s, #0
    movi v22.4s, #0
    movi v23.4s, #0
    b .Lsgemm_8x12_kloop
#if !defined(__APPLE__)
    .size yscv_sgemm_8x12_neon_set, . - yscv_sgemm_8x12_neon_set
#endif

    .globl CNAME(yscv_sgemm_8x12_neon_acc)
#if !defined(__APPLE__)
    .type yscv_sgemm_8x12_neon_acc, %function
    .p2align 2
yscv_sgemm_8x12_neon_acc:
#else
    .p2align 2
_yscv_sgemm_8x12_neon_acc:
#endif
    stp d8,  d9,  [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]

    lsl x3, x3, #2                      // ldc → bytes

    mov x8,  x2
    add x9,  x2, x3
    add x10, x9, x3
    add x11, x10, x3
    add x12, x11, x3
    add x13, x12, x3
    add x14, x13, x3
    add x15, x14, x3

    // Load existing C: 12 f32 per row = 3 quads.
    ld1 {v0.4s,  v1.4s,  v2.4s},  [x8]
    ld1 {v3.4s,  v4.4s,  v5.4s},  [x9]
    ld1 {v6.4s,  v7.4s,  v8.4s},  [x10]
    ld1 {v9.4s,  v10.4s, v11.4s}, [x11]
    ld1 {v12.4s, v13.4s, v14.4s}, [x12]
    ld1 {v15.4s, v16.4s, v17.4s}, [x13]
    ld1 {v18.4s, v19.4s, v20.4s}, [x14]
    ld1 {v21.4s, v22.4s, v23.4s}, [x15]
    b .Lsgemm_8x12_kloop

.p2align 4
.Lsgemm_8x12_kloop:
    cbz x4, .Lsgemm_8x12_epilogue

.p2align 4
.Lsgemm_8x12_body:
    // Load B[k]: 3 quads (12 cols) from the single NR=12 packed panel.
    ld1 {v24.4s, v25.4s, v26.4s}, [x1]  // b[k][0..11]
    // Load A[k]: 8 floats packed as 2 × v.4s.
    ld1 {v27.4s, v28.4s}, [x0]          // A[k][0..7]

    // 24 FMAs: each row r in 0..8 gets 3 FMAs (one per B quad).
    fmla v0.4s,  v24.4s, v27.s[0]       // row 0, cols 0..3
    fmla v1.4s,  v25.4s, v27.s[0]       // row 0, cols 4..7
    fmla v2.4s,  v26.4s, v27.s[0]       // row 0, cols 8..11
    fmla v3.4s,  v24.4s, v27.s[1]
    fmla v4.4s,  v25.4s, v27.s[1]
    fmla v5.4s,  v26.4s, v27.s[1]
    fmla v6.4s,  v24.4s, v27.s[2]
    fmla v7.4s,  v25.4s, v27.s[2]
    fmla v8.4s,  v26.4s, v27.s[2]
    fmla v9.4s,  v24.4s, v27.s[3]
    fmla v10.4s, v25.4s, v27.s[3]
    fmla v11.4s, v26.4s, v27.s[3]
    fmla v12.4s, v24.4s, v28.s[0]
    fmla v13.4s, v25.4s, v28.s[0]
    fmla v14.4s, v26.4s, v28.s[0]
    fmla v15.4s, v24.4s, v28.s[1]
    fmla v16.4s, v25.4s, v28.s[1]
    fmla v17.4s, v26.4s, v28.s[1]
    fmla v18.4s, v24.4s, v28.s[2]
    fmla v19.4s, v25.4s, v28.s[2]
    fmla v20.4s, v26.4s, v28.s[2]
    fmla v21.4s, v24.4s, v28.s[3]
    fmla v22.4s, v25.4s, v28.s[3]
    fmla v23.4s, v26.4s, v28.s[3]

    add x0, x0, #32                     // a += MR=8 floats (32 bytes)
    add x1, x1, #48                     // b += NR=12 floats (48 bytes)
    subs x4, x4, #1
    b.ne .Lsgemm_8x12_body

.Lsgemm_8x12_epilogue:
    // Apply inline epilogue only when is_last_k == 1.
    cbz x7, .Lsgemm_8x12_store

    // Bias add (if x5 != null). Caller pre-advances bias by col_offset
    // so bias[0..11] maps this tile's 12 output columns.
    cbz x5, .Lsgemm_8x12_skip_bias
    ld1 {v24.4s, v25.4s, v26.4s}, [x5]  // bias[0..11]
    fadd v0.4s,  v0.4s,  v24.4s
    fadd v1.4s,  v1.4s,  v25.4s
    fadd v2.4s,  v2.4s,  v26.4s
    fadd v3.4s,  v3.4s,  v24.4s
    fadd v4.4s,  v4.4s,  v25.4s
    fadd v5.4s,  v5.4s,  v26.4s
    fadd v6.4s,  v6.4s,  v24.4s
    fadd v7.4s,  v7.4s,  v25.4s
    fadd v8.4s,  v8.4s,  v26.4s
    fadd v9.4s,  v9.4s,  v24.4s
    fadd v10.4s, v10.4s, v25.4s
    fadd v11.4s, v11.4s, v26.4s
    fadd v12.4s, v12.4s, v24.4s
    fadd v13.4s, v13.4s, v25.4s
    fadd v14.4s, v14.4s, v26.4s
    fadd v15.4s, v15.4s, v24.4s
    fadd v16.4s, v16.4s, v25.4s
    fadd v17.4s, v17.4s, v26.4s
    fadd v18.4s, v18.4s, v24.4s
    fadd v19.4s, v19.4s, v25.4s
    fadd v20.4s, v20.4s, v26.4s
    fadd v21.4s, v21.4s, v24.4s
    fadd v22.4s, v22.4s, v25.4s
    fadd v23.4s, v23.4s, v26.4s

.Lsgemm_8x12_skip_bias:
    // Activation: id 1 = Relu. Id 0 = skip. SiLU routed to MR=4 by caller.
    cmp x6, #1
    b.ne .Lsgemm_8x12_store
    movi v24.4s, #0
    fmax v0.4s,  v0.4s,  v24.4s
    fmax v1.4s,  v1.4s,  v24.4s
    fmax v2.4s,  v2.4s,  v24.4s
    fmax v3.4s,  v3.4s,  v24.4s
    fmax v4.4s,  v4.4s,  v24.4s
    fmax v5.4s,  v5.4s,  v24.4s
    fmax v6.4s,  v6.4s,  v24.4s
    fmax v7.4s,  v7.4s,  v24.4s
    fmax v8.4s,  v8.4s,  v24.4s
    fmax v9.4s,  v9.4s,  v24.4s
    fmax v10.4s, v10.4s, v24.4s
    fmax v11.4s, v11.4s, v24.4s
    fmax v12.4s, v12.4s, v24.4s
    fmax v13.4s, v13.4s, v24.4s
    fmax v14.4s, v14.4s, v24.4s
    fmax v15.4s, v15.4s, v24.4s
    fmax v16.4s, v16.4s, v24.4s
    fmax v17.4s, v17.4s, v24.4s
    fmax v18.4s, v18.4s, v24.4s
    fmax v19.4s, v19.4s, v24.4s
    fmax v20.4s, v20.4s, v24.4s
    fmax v21.4s, v21.4s, v24.4s
    fmax v22.4s, v22.4s, v24.4s
    fmax v23.4s, v23.4s, v24.4s

.Lsgemm_8x12_store:
    st1 {v0.4s,  v1.4s,  v2.4s},  [x8]
    st1 {v3.4s,  v4.4s,  v5.4s},  [x9]
    st1 {v6.4s,  v7.4s,  v8.4s},  [x10]
    st1 {v9.4s,  v10.4s, v11.4s}, [x11]
    st1 {v12.4s, v13.4s, v14.4s}, [x12]
    st1 {v15.4s, v16.4s, v17.4s}, [x13]
    st1 {v18.4s, v19.4s, v20.4s}, [x14]
    st1 {v21.4s, v22.4s, v23.4s}, [x15]

    ldp d14, d15, [sp, #48]
    ldp d12, d13, [sp, #32]
    ldp d10, d11, [sp, #16]
    ldp d8,  d9,  [sp], #64
    ret
#if !defined(__APPLE__)
    .size yscv_sgemm_8x12_neon_acc, . - yscv_sgemm_8x12_neon_acc
#endif

// Enable FEAT_FP16 (fullfp16) for the hgemm kernel below so the assembler
// accepts `fmla v.8h, v.8h, v.h[i]` (fp16 SIMD FMA).
#ifdef __APPLE__
    .arch_extension fp16
#else
    .arch armv8.2-a+fp16
#endif

// ============================================================================
// yscv_hgemm_6x16_neon_set / _acc  (FEAT_FP16, 6 rows × 16 cols, fp16)
// ============================================================================
// Half-precision SGEMM using FEAT_FP16 (ARMv8.2-A): `fmla v.8h, v.8h, v.h[i]`
// runs 8-wide fp16 FMA — 2× throughput vs fp32 `fmla v.4s` on any core that
// supports FEAT_FP16 (Cortex-A76+, Apple M1+, Neoverse-V1+).
//
// Layout matches MLAS HalfGemmKernelNeon.S (MR=6, NR=16):
//   A pack: 6 fp16 per k-step (packed into v.8h lanes 0..5, 6..7 padded zero).
//   B pack: two adjacent NR=8 fp16 panels = 16 fp16 per k-step.
//   C:      6 rows × 16 cols = 12 v.8h regs (6 rows × 2 regs per row).
//
// Register plan (32 v-regs total):
//   v0..v11  = 12 accumulators (row r, cols [0..7]=v[2r], cols [8..15]=v[2r+1])
//   v12, v13 = B[k] cols 0..7 and 8..15
//   v14      = A[k] scattered (lanes 0..5 hold A[0..5], 6..7 unused/zeros)
//   v15..v31 = spare / callee-saved region for bias/epilogue
//
// Callee-saved per AAPCS64: v8-v15 low 64 bits. v8-v11 are accumulators
// (we use the full v-reg, but the low half must still survive the call —
// save d8..d11 at entry).
//
// Args:
//   x0 = a_panel  (*const u16 — fp16 bit-packed)
//   x1 = b_panel_0 (*const u16)
//   x2 = b_panel_1 (*const u16)
//   x3 = c (*mut u16)
//   x4 = ldc (elements, multiplied by 2 internally for bytes)
//   x5 = kc

    .globl CNAME(yscv_hgemm_6x16_neon_set)
#if !defined(__APPLE__)
    .type yscv_hgemm_6x16_neon_set, %function
    .p2align 2
yscv_hgemm_6x16_neon_set:
#else
    .p2align 2
_yscv_hgemm_6x16_neon_set:
#endif
    stp d8,  d9,  [sp, #-32]!
    stp d10, d11, [sp, #16]

    lsl x4, x4, #1                      // ldc elems → bytes (fp16 = 2B each)

    mov x6,  x3
    add x7,  x3, x4
    add x8,  x7, x4
    add x9,  x8, x4
    add x10, x9, x4
    add x11, x10, x4

    movi v0.8h,  #0
    movi v1.8h,  #0
    movi v2.8h,  #0
    movi v3.8h,  #0
    movi v4.8h,  #0
    movi v5.8h,  #0
    movi v6.8h,  #0
    movi v7.8h,  #0
    movi v8.8h,  #0
    movi v9.8h,  #0
    movi v10.8h, #0
    movi v11.8h, #0
    b .Lhgemm_6x16_kloop
#if !defined(__APPLE__)
    .size yscv_hgemm_6x16_neon_set, . - yscv_hgemm_6x16_neon_set
#endif

    .globl CNAME(yscv_hgemm_6x16_neon_acc)
#if !defined(__APPLE__)
    .type yscv_hgemm_6x16_neon_acc, %function
    .p2align 2
yscv_hgemm_6x16_neon_acc:
#else
    .p2align 2
_yscv_hgemm_6x16_neon_acc:
#endif
    stp d8,  d9,  [sp, #-32]!
    stp d10, d11, [sp, #16]

    lsl x4, x4, #1                      // ldc bytes

    mov x6,  x3
    add x7,  x3, x4
    add x8,  x7, x4
    add x9,  x8, x4
    add x10, x9, x4
    add x11, x10, x4

    // Load existing C into accumulators: 16 fp16 per row = 2 × v.8h.
    ld1 {v0.8h,  v1.8h},  [x6]
    ld1 {v2.8h,  v3.8h},  [x7]
    ld1 {v4.8h,  v5.8h},  [x8]
    ld1 {v6.8h,  v7.8h},  [x9]
    ld1 {v8.8h,  v9.8h},  [x10]
    ld1 {v10.8h, v11.8h}, [x11]
    b .Lhgemm_6x16_kloop

.p2align 4
.Lhgemm_6x16_kloop:
    cbz x5, .Lhgemm_6x16_store

.p2align 4
.Lhgemm_6x16_body:
    // Load B[k]: 16 fp16 = 2 v.8h.
    ld1 {v12.8h, v13.8h}, [x1]          // cols 0..15 — pack interleaves two NR=8 panels
    // Load A[k]: 8 fp16 (6 used + 2 pad).
    ld1 {v14.8h}, [x0]

    // 12 FMAs per k-step.
    fmla v0.8h,  v12.8h, v14.h[0]
    fmla v1.8h,  v13.8h, v14.h[0]
    fmla v2.8h,  v12.8h, v14.h[1]
    fmla v3.8h,  v13.8h, v14.h[1]
    fmla v4.8h,  v12.8h, v14.h[2]
    fmla v5.8h,  v13.8h, v14.h[2]
    fmla v6.8h,  v12.8h, v14.h[3]
    fmla v7.8h,  v13.8h, v14.h[3]
    fmla v8.8h,  v12.8h, v14.h[4]
    fmla v9.8h,  v13.8h, v14.h[4]
    fmla v10.8h, v12.8h, v14.h[5]
    fmla v11.8h, v13.8h, v14.h[5]

    add x0, x0, #16                     // a += 8 fp16 (but only 6 used; stride 8 keeps alignment)
    add x1, x1, #32                     // b += 16 fp16
    subs x5, x5, #1
    b.ne .Lhgemm_6x16_body

.Lhgemm_6x16_store:
    st1 {v0.8h,  v1.8h},  [x6]
    st1 {v2.8h,  v3.8h},  [x7]
    st1 {v4.8h,  v5.8h},  [x8]
    st1 {v6.8h,  v7.8h},  [x9]
    st1 {v8.8h,  v9.8h},  [x10]
    st1 {v10.8h, v11.8h}, [x11]

    ldp d10, d11, [sp, #16]
    ldp d8,  d9,  [sp], #32
    ret
#if !defined(__APPLE__)
    .size yscv_hgemm_6x16_neon_acc, . - yscv_hgemm_6x16_neon_acc
#endif

YSCV_NOTE_GNU_STACK
