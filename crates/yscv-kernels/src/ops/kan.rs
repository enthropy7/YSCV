// # Safety contract
//
// All `unsafe` blocks use SIMD intrinsics gated on target-feature
// detection via `crate::host_cpu().features`.

#![allow(unsafe_code)]

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    float32x4x4_t, vaddq_f32, vandq_u32, vcgeq_f32, vcltq_f32, vcvtq_s32_f32, vdupq_n_f32,
    vld1q_f32, vminvq_u32, vmulq_f32, vrndmq_f32, vst1q_f32, vst1q_s32, vst4q_f32, vsubq_f32,
};
#[cfg(target_arch = "x86")]
use std::arch::x86::{
    __m128i, _mm_add_ps, _mm_and_ps, _mm_cmpge_ps, _mm_cmplt_ps, _mm_cvttps_epi32, _mm_floor_ps,
    _mm_loadu_ps, _mm_movehl_ps, _mm_movelh_ps, _mm_movemask_ps, _mm_mul_ps, _mm_set1_ps,
    _mm_storeu_ps, _mm_storeu_si128, _mm_sub_ps, _mm_unpackhi_ps, _mm_unpacklo_ps, _mm256_add_ps,
    _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_storeu_ps,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m128i, _mm_add_ps, _mm_and_ps, _mm_cmpge_ps, _mm_cmplt_ps, _mm_cvttps_epi32, _mm_floor_ps,
    _mm_loadu_ps, _mm_movehl_ps, _mm_movelh_ps, _mm_movemask_ps, _mm_mul_ps, _mm_set1_ps,
    _mm_storeu_ps, _mm_storeu_si128, _mm_sub_ps, _mm_unpackhi_ps, _mm_unpacklo_ps, _mm256_add_ps,
    _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_storeu_ps,
};

use super::simd::silu_slice_dispatch;
use crate::KernelError;

/// Outputs are padded to a multiple of this so every SIMD path works on whole vectors.
const LANES: usize = 8;
/// Spline order: cubic B-splines, four of which are non-zero at any point.
const ACTIVE: usize = 4;

/// A B-spline Kolmogorov–Arnold (KAN) linear layer prepared for inference.
///
/// Computes, for every output `o` (Liu et al. 2024, in the `efficient-kan` formulation):
///
/// ```text
/// y[o] = Σ_i  base[o,i] · silu(x[i])  +  scaler[o,i] · Σ_j spline[o,i,j] · B_j(x[i])
/// ```
///
/// where `B_j` are the `grid_size + 3` cubic B-splines on the uniform grid over `[lo, hi]` with
/// `grid_size` intervals, extended by three knots on each side. Outside
/// `[lo - 3h, hi + 3h)` every `B_j` is zero.
///
/// At most four `B_j` are non-zero at any `x`. The kernel evaluates exactly those, in closed form
/// and without divisions (`1/h` is kept, the `1/6` of the cubic bases is folded into the
/// coefficients), and reads only their coefficients: `scaler · spline / 6` is folded once and
/// stored input-major (`[in, grid_size + 3, out]`, outputs padded to a multiple of 8), so each
/// active basis adds one contiguous row to all outputs. The NEON / AVX / SSE paths add in the
/// same order as the scalar path and give bitwise-identical results; `silu` is
/// [`silu_slice_dispatch`].
#[derive(Debug, Clone)]
pub struct KanLinear {
    in_features: usize,
    out_features: usize,
    out_padded: usize,
    grid_size: usize,
    lo: f32,
    inv_h: f32,
    base: Vec<f32>,
    coef: Vec<f32>,
}

impl KanLinear {
    /// Builds the layer from `efficient-kan` / PyTorch parameters: `base_weight` `[out, in]`,
    /// `spline_weight` `[out, in, grid_size + 3]`, `spline_scaler` `[out, in]`, all row-major,
    /// and the grid range `(lo, hi)`.
    pub fn new(
        in_features: usize,
        out_features: usize,
        grid_size: usize,
        grid_range: (f32, f32),
        base_weight: &[f32],
        spline_weight: &[f32],
        spline_scaler: &[f32],
    ) -> Result<Self, KernelError> {
        let (lo, hi) = grid_range;
        if grid_size == 0 || lo.partial_cmp(&hi) != Some(std::cmp::Ordering::Less) {
            return Err(KernelError::InvalidKanGrid {
                grid_size,
                range: format!("({lo}, {hi})"),
            });
        }
        let edges = in_features * out_features;
        let nb = grid_size + ACTIVE - 1;
        for (name, expected, got) in [
            ("base_weight", edges, base_weight.len()),
            ("spline_weight", edges * nb, spline_weight.len()),
            ("spline_scaler", edges, spline_scaler.len()),
        ] {
            if expected != got {
                return Err(KernelError::KanShapeMismatch {
                    name,
                    expected,
                    got,
                });
            }
        }
        let out_padded = out_features.div_ceil(LANES) * LANES;
        let mut base = vec![0.0; in_features * out_padded];
        let mut coef = vec![0.0; in_features * nb * out_padded];
        for o in 0..out_features {
            for i in 0..in_features {
                let e = o * in_features + i;
                base[i * out_padded + o] = base_weight[e];
                for j in 0..nb {
                    coef[(i * nb + j) * out_padded + o] =
                        spline_scaler[e] * spline_weight[e * nb + j] / 6.0;
                }
            }
        }
        Ok(Self {
            in_features,
            out_features,
            out_padded,
            grid_size,
            lo,
            inv_h: grid_size as f32 / (hi - lo),
            base,
            coef,
        })
    }

    /// Number of inputs of the layer.
    pub const fn in_features(&self) -> usize {
        self.in_features
    }

    /// Number of outputs of the layer.
    pub const fn out_features(&self) -> usize {
        self.out_features
    }

    /// `input` is `[rows, in_features]`, `output` is `[rows, out_features]`, both row-major.
    pub fn forward(
        &self,
        input: &[f32],
        rows: usize,
        output: &mut [f32],
    ) -> Result<(), KernelError> {
        self.forward_with(input, rows, output, Path::detect())
    }

    fn forward_with(
        &self,
        input: &[f32],
        rows: usize,
        output: &mut [f32],
        path: Path,
    ) -> Result<(), KernelError> {
        let (n_in, n_out) = (self.in_features, self.out_features);
        if input.len() != rows * n_in {
            return Err(KernelError::KanShapeMismatch {
                name: "input",
                expected: rows * n_in,
                got: input.len(),
            });
        }
        if output.len() != rows * n_out {
            return Err(KernelError::KanShapeMismatch {
                name: "output",
                expected: rows * n_out,
                got: output.len(),
            });
        }
        // one f32 scratch (silu, basis values, accumulators) and one for block offsets per call
        let mut scratch = vec![0.0; n_in * (1 + ACTIVE) + self.out_padded];
        let (silu, rest) = scratch.split_at_mut(n_in);
        let (values, acc) = rest.split_at_mut(n_in * ACTIVE);
        let mut block = vec![0usize; n_in];
        for (x, y) in input.chunks_exact(n_in).zip(output.chunks_exact_mut(n_out)) {
            silu_slice_dispatch(x, silu);
            self.bases_all(x, &mut block, values, path);
            acc.fill(0.0);
            path.accumulate(self, silu, &block, values, acc);
            y.copy_from_slice(&acc[..n_out]);
        }
        Ok(())
    }

    /// `bases` for every input: four inputs at a time with SIMD when all four lie inside the grid
    /// (the same arithmetic in the same order, so the results are bitwise equal), `bases` itself
    /// for the rest.
    fn bases_all(&self, x: &[f32], block: &mut [usize], values: &mut [f32], path: Path) {
        let groups = if path == Path::Scalar || cfg!(miri) {
            0
        } else {
            x.len() / ACTIVE
        };
        for g in 0..groups {
            let i = g * ACTIVE;
            let (xg, bg, vg) = (
                &x[i..i + 4],
                &mut block[i..i + 4],
                &mut values[4 * i..4 * i + 16],
            );
            let done = match path {
                // SAFETY: NEON presence checked in `Path::detect`; slices hold 4 / 4 / 16 elements.
                #[cfg(target_arch = "aarch64")]
                Path::Neon => unsafe { self.bases4_neon(i, xg, bg, vg) },
                // SAFETY: SSE4.1 checked here; slices hold 4 / 4 / 16 elements.
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                Path::Avx | Path::Sse if crate::host_cpu().features.sse41 => unsafe {
                    self.bases4_sse41(i, xg, bg, vg)
                },
                _ => false,
            };
            if !done {
                for k in i..i + 4 {
                    block[k] = self.bases(k, x[k], &mut values[k * ACTIVE..(k + 1) * ACTIVE]);
                }
            }
        }
        for k in groups * ACTIVE..x.len() {
            block[k] = self.bases(k, x[k], &mut values[k * ACTIVE..(k + 1) * ACTIVE]);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn bases4_neon(
        &self,
        i: usize,
        x: &[f32],
        block: &mut [usize],
        values: &mut [f32],
    ) -> bool {
        let nb = self.grid_size + ACTIVE - 1;
        // SAFETY: `x` holds 4 floats, `values` 16.
        unsafe {
            let (one, three) = (vdupq_n_f32(1.0), vdupq_n_f32(3.0));
            let u_all = vaddq_f32(
                vmulq_f32(
                    vsubq_f32(vld1q_f32(x.as_ptr()), vdupq_n_f32(self.lo)),
                    vdupq_n_f32(self.inv_h),
                ),
                three,
            );
            let m = vrndmq_f32(u_all);
            if vminvq_u32(vandq_u32(
                vcgeq_f32(m, three),
                vcltq_f32(m, vdupq_n_f32(nb as f32)),
            )) == 0
            {
                return false;
            }
            let u = vsubq_f32(u_all, m);
            let v = vsubq_f32(one, u);
            let (u2, u3) = (vmulq_f32(u, u), vmulq_f32(vmulq_f32(u, u), u));
            let b0 = vmulq_f32(vmulq_f32(v, v), v);
            let b1 = vaddq_f32(
                vsubq_f32(vmulq_f32(three, u3), vmulq_f32(vdupq_n_f32(6.0), u2)),
                vdupq_n_f32(4.0),
            );
            let b2 = vaddq_f32(
                vaddq_f32(
                    vaddq_f32(vmulq_f32(vdupq_n_f32(-3.0), u3), vmulq_f32(three, u2)),
                    vmulq_f32(three, u),
                ),
                one,
            );
            vst4q_f32(values.as_mut_ptr(), float32x4x4_t(b0, b1, b2, u3));
            let mut span = [0i32; 4];
            vst1q_s32(span.as_mut_ptr(), vcvtq_s32_f32(m));
            for (k, (b, &sp)) in block.iter_mut().zip(&span).enumerate() {
                *b = ((i + k) * nb + sp as usize - 3) * self.out_padded;
            }
        }
        true
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "sse4.1")]
    unsafe fn bases4_sse41(
        &self,
        i: usize,
        x: &[f32],
        block: &mut [usize],
        values: &mut [f32],
    ) -> bool {
        let nb = self.grid_size + ACTIVE - 1;
        // SAFETY: `x` holds 4 floats, `values` 16.
        unsafe {
            let (one, three) = (_mm_set1_ps(1.0), _mm_set1_ps(3.0));
            let u_all = _mm_add_ps(
                _mm_mul_ps(
                    _mm_sub_ps(_mm_loadu_ps(x.as_ptr()), _mm_set1_ps(self.lo)),
                    _mm_set1_ps(self.inv_h),
                ),
                three,
            );
            let m = _mm_floor_ps(u_all);
            if _mm_movemask_ps(_mm_and_ps(
                _mm_cmpge_ps(m, three),
                _mm_cmplt_ps(m, _mm_set1_ps(nb as f32)),
            )) != 0b1111
            {
                return false;
            }
            let u = _mm_sub_ps(u_all, m);
            let v = _mm_sub_ps(one, u);
            let (u2, u3) = (_mm_mul_ps(u, u), _mm_mul_ps(_mm_mul_ps(u, u), u));
            let b0 = _mm_mul_ps(_mm_mul_ps(v, v), v);
            let b1 = _mm_add_ps(
                _mm_sub_ps(_mm_mul_ps(three, u3), _mm_mul_ps(_mm_set1_ps(6.0), u2)),
                _mm_set1_ps(4.0),
            );
            let b2 = _mm_add_ps(
                _mm_add_ps(
                    _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-3.0), u3), _mm_mul_ps(three, u2)),
                    _mm_mul_ps(three, u),
                ),
                one,
            );
            // 4 × 4 transpose: row k = bases of input i + k
            let (t0, t1) = (_mm_unpacklo_ps(b0, b1), _mm_unpacklo_ps(b2, u3));
            let (t2, t3) = (_mm_unpackhi_ps(b0, b1), _mm_unpackhi_ps(b2, u3));
            let rows = [
                _mm_movelh_ps(t0, t1),
                _mm_movehl_ps(t1, t0),
                _mm_movelh_ps(t2, t3),
                _mm_movehl_ps(t3, t2),
            ];
            for (k, row) in rows.iter().enumerate() {
                _mm_storeu_ps(values.as_mut_ptr().add(4 * k), *row);
            }
            let mut span = [0i32; 4];
            _mm_storeu_si128(span.as_mut_ptr().cast::<__m128i>(), _mm_cvttps_epi32(m));
            for (k, (b, &sp)) in block.iter_mut().zip(&span).enumerate() {
                *b = ((i + k) * nb + sp as usize - 3) * self.out_padded;
            }
        }
        true
    }

    /// Writes the four cubic B-splines that can be non-zero at `x` (times 6) against the four
    /// consecutive coefficient rows starting at the returned offset into `coef`. Near the ends of
    /// the grid the block is moved inside the table and the values shifted with it; bases that do
    /// not exist, and all four outside the support, get value 0.
    fn bases(&self, i: usize, x: f32, out: &mut [f32]) -> usize {
        let nb = self.grid_size + ACTIVE - 1;
        let block = |row: usize| (i * nb + row) * self.out_padded;
        let u_all = (x - self.lo) * self.inv_h + 3.0;
        let last = (self.grid_size + 6) as f32;
        if !(u_all >= 0.0 && u_all < last) {
            out.fill(0.0);
            return block(0);
        }
        let m = u_all.floor();
        let (u, span) = (u_all - m, m as usize);
        let v = 1.0 - u;
        let (u2, u3) = (u * u, u * u * u);
        let b = [
            v * v * v,
            3.0 * u3 - 6.0 * u2 + 4.0,
            -3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0,
            u3,
        ];
        // b[r] belongs to basis span + r - 3. Inside the grid all four exist and the block starts
        // at span - 3; near its ends the block is moved into the table.
        if (3..nb).contains(&span) {
            out.copy_from_slice(&b);
            return block(span - 3);
        }
        let start = span.saturating_sub(3).min(nb - ACTIVE);
        out.fill(0.0);
        for (r, &value) in b.iter().enumerate() {
            if let Some(j) = (span + r).checked_sub(3).filter(|&j| j < nb) {
                out[j - start] = value;
            }
        }
        block(start)
    }

    /// Coefficient row `r` of the block at offset `block`.
    #[inline]
    fn coef_row(&self, block: usize, r: usize) -> &[f32] {
        let start = block + r * self.out_padded;
        &self.coef[start..start + self.out_padded]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Path {
    Scalar,
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Sse,
}

impl Path {
    fn detect() -> Self {
        if cfg!(miri) {
            return Self::Scalar;
        }
        let features = crate::host_cpu().features;
        #[cfg(target_arch = "aarch64")]
        if features.neon {
            return Self::Neon;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if features.avx {
                return Self::Avx;
            }
            if features.sse {
                return Self::Sse;
            }
        }
        let _ = features;
        Self::Scalar
    }

    /// `acc[o] += silu[i] · base[i, o]`, then `acc[o] += B_r(x_i) · coef[block_i + r, o]` for the
    /// four rows of the input's block, input by input, in this order on every path.
    fn accumulate(
        self,
        l: &KanLinear,
        silu: &[f32],
        block: &[usize],
        values: &[f32],
        acc: &mut [f32],
    ) {
        match self {
            Self::Scalar => accumulate_scalar(l, silu, block, values, acc),
            // SAFETY: NEON presence checked in `Path::detect`.
            #[cfg(target_arch = "aarch64")]
            Self::Neon => unsafe { accumulate_neon(l, silu, block, values, acc) },
            // SAFETY: AVX presence checked in `Path::detect`.
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Avx => unsafe { accumulate_avx(l, silu, block, values, acc) },
            // SAFETY: SSE presence checked in `Path::detect`.
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Sse => unsafe { accumulate_sse(l, silu, block, values, acc) },
        }
    }
}

/// The four paths below add `v · row` to `acc` for every input and active basis, in the same
/// order; they differ only in how many lanes one instruction covers.
fn accumulate_scalar(
    l: &KanLinear,
    silu: &[f32],
    block: &[usize],
    values: &[f32],
    acc: &mut [f32],
) {
    let op = l.out_padded;
    for i in 0..l.in_features {
        add_row_scalar(acc, &l.base[i * op..(i + 1) * op], silu[i]);
        for r in 0..ACTIVE {
            add_row_scalar(acc, l.coef_row(block[i], r), values[i * ACTIVE + r]);
        }
    }
}

fn add_row_scalar(acc: &mut [f32], row: &[f32], v: f32) {
    for (a, &c) in acc.iter_mut().zip(row) {
        *a += v * c;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn accumulate_neon(
    l: &KanLinear,
    silu: &[f32],
    block: &[usize],
    values: &[f32],
    acc: &mut [f32],
) {
    // Up to 32 outputs stay in registers for the whole pass; wider layers accumulate in memory.
    // SAFETY (all arms): NEON enabled on this function.
    match l.out_padded {
        8 => return unsafe { accumulate_neon_regs::<2>(l, silu, block, values, acc) },
        16 => return unsafe { accumulate_neon_regs::<4>(l, silu, block, values, acc) },
        24 => return unsafe { accumulate_neon_regs::<6>(l, silu, block, values, acc) },
        32 => return unsafe { accumulate_neon_regs::<8>(l, silu, block, values, acc) },
        _ => {}
    }
    let op = l.out_padded;
    for i in 0..l.in_features {
        // SAFETY: rows and `acc` have length `op`, a multiple of 8.
        unsafe { add_row_neon(acc, &l.base[i * op..(i + 1) * op], silu[i]) };
        for r in 0..ACTIVE {
            // SAFETY: as above.
            unsafe { add_row_neon(acc, l.coef_row(block[i], r), values[i * ACTIVE + r]) };
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn accumulate_neon_regs<const Q: usize>(
    l: &KanLinear,
    silu: &[f32],
    block: &[usize],
    values: &[f32],
    acc: &mut [f32],
) {
    let op = l.out_padded;
    let mut a = [vdupq_n_f32(0.0); Q];
    // SAFETY: every row and `acc` hold `op == 4 · Q` floats; `block[i]` starts ACTIVE rows
    // inside the coefficient table.
    unsafe {
        let add = |a: &mut [_; Q], row: *const f32, v: f32| {
            let vv = vdupq_n_f32(v);
            for (q, aq) in a.iter_mut().enumerate() {
                *aq = vaddq_f32(*aq, vmulq_f32(vv, vld1q_f32(row.add(4 * q))));
            }
        };
        for i in 0..l.in_features {
            add(&mut a, l.base.as_ptr().add(i * op), silu[i]);
            let rows = l.coef.as_ptr().add(block[i]);
            for r in 0..ACTIVE {
                add(&mut a, rows.add(r * op), values[i * ACTIVE + r]);
            }
        }
        for (q, aq) in a.iter().enumerate() {
            vst1q_f32(acc.as_mut_ptr().add(4 * q), *aq);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn add_row_neon(acc: &mut [f32], row: &[f32], v: f32) {
    let vv = vdupq_n_f32(v);
    for (a, c) in acc.chunks_exact_mut(4).zip(row.chunks_exact(4)) {
        // SAFETY: both chunks hold exactly 4 floats.
        unsafe {
            vst1q_f32(
                a.as_mut_ptr(),
                vaddq_f32(vld1q_f32(a.as_ptr()), vmulq_f32(vv, vld1q_f32(c.as_ptr()))),
            )
        };
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn accumulate_avx(
    l: &KanLinear,
    silu: &[f32],
    block: &[usize],
    values: &[f32],
    acc: &mut [f32],
) {
    let op = l.out_padded;
    for i in 0..l.in_features {
        // SAFETY: rows and `acc` have length `op`, a multiple of 8.
        unsafe { add_row_avx(acc, &l.base[i * op..(i + 1) * op], silu[i]) };
        for r in 0..ACTIVE {
            // SAFETY: as above.
            unsafe { add_row_avx(acc, l.coef_row(block[i], r), values[i * ACTIVE + r]) };
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
#[inline]
unsafe fn add_row_avx(acc: &mut [f32], row: &[f32], v: f32) {
    let vv = _mm256_set1_ps(v);
    for (a, c) in acc.chunks_exact_mut(8).zip(row.chunks_exact(8)) {
        // SAFETY: both chunks hold exactly 8 floats.
        unsafe {
            _mm256_storeu_ps(
                a.as_mut_ptr(),
                _mm256_add_ps(
                    _mm256_loadu_ps(a.as_ptr()),
                    _mm256_mul_ps(vv, _mm256_loadu_ps(c.as_ptr())),
                ),
            );
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse")]
unsafe fn accumulate_sse(
    l: &KanLinear,
    silu: &[f32],
    block: &[usize],
    values: &[f32],
    acc: &mut [f32],
) {
    let op = l.out_padded;
    for i in 0..l.in_features {
        // SAFETY: rows and `acc` have length `op`, a multiple of 8.
        unsafe { add_row_sse(acc, &l.base[i * op..(i + 1) * op], silu[i]) };
        for r in 0..ACTIVE {
            // SAFETY: as above.
            unsafe { add_row_sse(acc, l.coef_row(block[i], r), values[i * ACTIVE + r]) };
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse")]
#[inline]
unsafe fn add_row_sse(acc: &mut [f32], row: &[f32], v: f32) {
    let vv = _mm_set1_ps(v);
    for (a, c) in acc.chunks_exact_mut(4).zip(row.chunks_exact(4)) {
        // SAFETY: both chunks hold exactly 4 floats.
        unsafe {
            _mm_storeu_ps(
                a.as_mut_ptr(),
                _mm_add_ps(
                    _mm_loadu_ps(a.as_ptr()),
                    _mm_mul_ps(vv, _mm_loadu_ps(c.as_ptr())),
                ),
            )
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cox–de Boor recursion over the extended uniform grid, in f64: the documented definition of
    /// the B-spline bases (Liu et al. 2024, efficient-kan `b_splines`).
    fn cox_de_boor(x: f64, lo: f64, hi: f64, grid: usize) -> Vec<f64> {
        let h = (hi - lo) / grid as f64;
        let knots: Vec<f64> = (0..grid + 7).map(|q| lo + (q as f64 - 3.0) * h).collect();
        let mut b: Vec<f64> = (0..knots.len() - 1)
            .map(|q| f64::from(u8::from(x >= knots[q] && x < knots[q + 1])))
            .collect();
        for p in 1..=3 {
            b = (0..b.len() - 1)
                .map(|q| {
                    (x - knots[q]) / (knots[q + p] - knots[q]) * b[q]
                        + (knots[q + p + 1] - x) / (knots[q + p + 1] - knots[q + 1]) * b[q + 1]
                })
                .collect();
        }
        b
    }

    fn layer(
        n_in: usize,
        n_out: usize,
        grid: usize,
        seed: u32,
    ) -> (KanLinear, Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut s = seed;
        let mut next = move || {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
        };
        let nb = grid + 3;
        let base: Vec<f32> = (0..n_in * n_out).map(|_| next()).collect();
        let spline: Vec<f32> = (0..n_in * n_out * nb).map(|_| next()).collect();
        let scaler: Vec<f32> = (0..n_in * n_out).map(|_| next()).collect();
        let l = KanLinear::new(n_in, n_out, grid, (-1.0, 1.0), &base, &spline, &scaler)
            .expect("valid layer");
        (l, base, spline, scaler)
    }

    #[test]
    fn matches_the_documented_formula() {
        for &(n_in, n_out, grid) in &[(13, 7, 5), (5, 19, 3), (9, 8, 10)] {
            let (l, base, spline, scaler) = layer(n_in, n_out, grid, 7);
            let nb = grid + 3;
            let h = 2.0 / grid as f32;
            // the whole extended grid and beyond, off the knots
            let x: Vec<f32> = (0..n_in)
                .map(|i| -1.0 - 4.0 * h + (2.0 + 8.0 * h) * i as f32 / (n_in - 1) as f32 + 0.013)
                .collect();
            let mut y = vec![0.0; n_out];
            l.forward(&x, 1, &mut y).expect("forward");
            for o in 0..n_out {
                let mut want = 0.0f64;
                for i in 0..n_in {
                    let xi = f64::from(x[i]);
                    let e = o * n_in + i;
                    let silu = xi / (1.0 + (-xi).exp());
                    let spl: f64 = cox_de_boor(xi, -1.0, 1.0, grid)
                        .iter()
                        .enumerate()
                        .map(|(j, b)| b * f64::from(spline[e * nb + j]))
                        .sum();
                    want += f64::from(base[e]) * silu + f64::from(scaler[e]) * spl;
                }
                assert!(
                    (f64::from(y[o]) - want).abs() < 1e-4 * want.abs().max(1.0),
                    "o={o}: {} vs {want}",
                    y[o]
                );
            }
        }
    }

    #[test]
    fn simd_paths_match_scalar_bitwise() {
        let (l, ..) = layer(37, 13, 5, 11);
        let x: Vec<f32> = (0..3 * 37)
            .map(|i| ((i * 29 % 97) as f32 / 97.0) * 3.0 - 1.5)
            .collect();
        let (mut want, mut got) = (vec![0.0; 3 * 13], vec![0.0; 3 * 13]);
        l.forward_with(&x, 3, &mut want, Path::Scalar)
            .expect("scalar");
        l.forward_with(&x, 3, &mut got, Path::detect())
            .expect("dispatch");
        assert_eq!(
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn rejects_bad_shapes_and_grids() {
        assert!(KanLinear::new(2, 3, 0, (-1.0, 1.0), &[0.0; 6], &[], &[0.0; 6]).is_err());
        assert!(KanLinear::new(2, 3, 5, (1.0, 1.0), &[0.0; 6], &[0.0; 48], &[0.0; 6]).is_err());
        assert!(KanLinear::new(2, 3, 5, (-1.0, 1.0), &[0.0; 6], &[0.0; 47], &[0.0; 6]).is_err());
        let (l, ..) = layer(4, 2, 5, 3);
        assert!(l.forward(&[0.0; 5], 1, &mut [0.0; 2]).is_err());
    }
}
