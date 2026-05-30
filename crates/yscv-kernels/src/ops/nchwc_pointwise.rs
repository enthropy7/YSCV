//! K.2: NCHWc 1×1 pointwise conv — all-shapes blocked GEMM, vectorized NCHWc output.
//!
//! Improvements over the legacy per-pixel path in `conv.rs`:
//! - Every shape uses blocked GEMM — no K/N threshold that excluded K=16 shapes
//! - Vectorized flat→NCHWc scatter fused with bias + residual + activation
//! - When `out_channels == block` (e.g. K=16,N=16): GEMM writes directly to the
//!   NCHWc output buffer (flat and NCHWc are the same layout), eliminating the
//!   scratch allocation and the scatter pass entirely.
//!
//! Kill switch: set `YSCV_NCHWC_PW_LEGACY=1` to fall back to the old path.

use std::sync::OnceLock;

use yscv_tensor::AlignedVec;

use super::conv::Activation;
use super::matmul::{GemmEpilogue, PackedB, blocked_gemm_nchwc_a_parallel};

/// NCHWc 1×1 pointwise convolution with pre-packed weight.
///
/// Input/output in `[N, C_blocks, H, W, block]` NCHWc layout.
/// All shapes use blocked GEMM — no threshold exclusion for small K or N.
///
/// The caller supplies pre-allocated `output` (zeroed or uninitialized; the
/// GEMM always writes every output element before the epilogue reads it).
pub fn nchwc_pw_compute(
    input_nchwc: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    residual: Option<&[f32]>,
    output: &mut [f32],
    n_batch: usize,
    ci_blocks: usize,
    h: usize,
    w: usize,
    actual_ic: usize,
    out_channels: usize,
    co_blocks: usize,
    block: usize,
    activation: Activation,
    prepacked_b: Option<&PackedB>,
) {
    let spatial = h * w;
    let in_n_stride = ci_blocks * spatial * block;
    let out_n_stride = co_blocks * spatial * block;

    // When out_channels == block and co_blocks == 1 the flat GEMM output layout
    // [spatial, block] is identical to NCHWc [1, spatial, block].  The GEMM
    // can write directly to the output buffer and we skip the scatter copy.
    let direct_out = co_blocks == 1 && out_channels == block;

    for n_idx in 0..n_batch {
        let in_slice = &input_nchwc[n_idx * in_n_stride..(n_idx + 1) * in_n_stride];
        let out_slice = &mut output[n_idx * out_n_stride..(n_idx + 1) * out_n_stride];
        let res_slice = residual.map(|r| &r[n_idx * out_n_stride..(n_idx + 1) * out_n_stride]);

        if direct_out {
            blocked_gemm_nchwc_a_parallel(
                in_slice,
                spatial,
                block,
                actual_ic,
                weight,
                out_slice,
                out_channels,
                GemmEpilogue::IDENTITY,
                None,
                prepacked_b,
            );
            apply_nchwc_epilogue(
                out_slice,
                res_slice,
                bias,
                spatial,
                out_channels,
                co_blocks,
                block,
                activation,
            );
        } else {
            let mut flat = AlignedVec::<f32>::uninitialized(spatial * out_channels);
            blocked_gemm_nchwc_a_parallel(
                in_slice,
                spatial,
                block,
                actual_ic,
                weight,
                flat.as_mut_slice(),
                out_channels,
                GemmEpilogue::IDENTITY,
                None,
                prepacked_b,
            );
            flat_to_nchwc_fused(
                flat.as_slice(),
                out_slice,
                res_slice,
                bias,
                spatial,
                out_channels,
                co_blocks,
                block,
                activation,
            );
        }
    }
}

/// Returns true when the K.2 native NCHWc PW path is active (default).
/// Kill switch: `YSCV_NCHWC_PW_LEGACY=1`.
pub(crate) fn nchwc_pw_native_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("YSCV_NCHWC_PW_LEGACY").is_none())
}

/// Scatter flat `[spatial, out_channels]` → NCHWc `[co_blocks, spatial, block]`
/// with fused bias + residual + activation.
///
/// Inner `bi` loop runs over `c_take` elements (= block in the common full-block
/// case).  LLVM auto-vectorises the copy + bias + residual passes into a single
/// ZMM/YMM sequence because `c_take` is a loop-invariant constant (block) and
/// all slices are non-overlapping with known strides.
fn flat_to_nchwc_fused(
    flat: &[f32],
    out: &mut [f32],
    residual: Option<&[f32]>,
    bias: Option<&[f32]>,
    spatial: usize,
    out_channels: usize,
    co_blocks: usize,
    block: usize,
    activation: Activation,
) {
    for ocb in 0..co_blocks {
        let c_start = ocb * block;
        let c_take = block.min(out_channels - c_start);
        // Hoist bias slice outside hw loop — LLVM LICM folds the is_some check.
        let bias_blk: Option<&[f32]> = bias.map(|b| &b[c_start..c_start + c_take]);
        let cb_out_off = ocb * spatial * block;

        for hw in 0..spatial {
            let src_base = hw * out_channels + c_start;
            let dst_base = cb_out_off + hw * block;
            let dst = &mut out[dst_base..dst_base + c_take];
            let src = &flat[src_base..src_base + c_take];

            // Contiguous copy; LLVM emits a single ZMM store for block=16.
            dst.copy_from_slice(src);

            if let Some(b) = bias_blk {
                for bi in 0..c_take {
                    dst[bi] += b[bi];
                }
            }

            if let Some(r) = residual {
                let r_cell = &r[dst_base..dst_base + c_take];
                for bi in 0..c_take {
                    dst[bi] += r_cell[bi];
                }
            }

            match activation {
                Activation::None => {}
                Activation::Relu => {
                    for bi in 0..c_take {
                        dst[bi] = dst[bi].max(0.0);
                    }
                }
                Activation::Silu => {
                    for bi in 0..c_take {
                        let v = dst[bi];
                        dst[bi] = v / (1.0 + (-v).exp());
                    }
                }
            }
        }
    }
}

/// Apply bias + residual + activation in-place on an existing NCHWc buffer.
///
/// Called when the GEMM has written flat output directly to the NCHWc output
/// buffer (co_blocks == 1, out_channels == block case).
fn apply_nchwc_epilogue(
    out: &mut [f32],
    residual: Option<&[f32]>,
    bias: Option<&[f32]>,
    spatial: usize,
    out_channels: usize,
    co_blocks: usize,
    block: usize,
    activation: Activation,
) {
    for ocb in 0..co_blocks {
        let c_start = ocb * block;
        let c_take = block.min(out_channels - c_start);
        let bias_blk: Option<&[f32]> = bias.map(|b| &b[c_start..c_start + c_take]);
        let cb_out_off = ocb * spatial * block;

        for hw in 0..spatial {
            let off = cb_out_off + hw * block;
            let dst = &mut out[off..off + c_take];

            if let Some(b) = bias_blk {
                for bi in 0..c_take {
                    dst[bi] += b[bi];
                }
            }

            if let Some(r) = residual {
                let r_cell = &r[off..off + c_take];
                for bi in 0..c_take {
                    dst[bi] += r_cell[bi];
                }
            }

            match activation {
                Activation::None => {}
                Activation::Relu => {
                    for bi in 0..c_take {
                        dst[bi] = dst[bi].max(0.0);
                    }
                }
                Activation::Silu => {
                    for bi in 0..c_take {
                        let v = dst[bi];
                        dst[bi] = v / (1.0 + (-v).exp());
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_nchwc_input(n: usize, ci_blocks: usize, h: usize, w: usize, block: usize) -> Vec<f32> {
        let len = n * ci_blocks * h * w * block;
        (0..len).map(|i| (i as f32) * 0.01).collect()
    }

    fn make_weight(ic: usize, oc: usize) -> Vec<f32> {
        let len = ic * oc;
        (0..len).map(|i| (i as f32) * 0.001 - 0.5).collect()
    }

    /// Reference: blocked_gemm_nchwc_a_parallel + scalar flat→NCHWc scatter.
    fn reference_nchwc_pw(
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        residual: Option<&[f32]>,
        n_batch: usize,
        ci_blocks: usize,
        h: usize,
        w: usize,
        actual_ic: usize,
        out_channels: usize,
        block: usize,
        activation: Activation,
    ) -> Vec<f32> {
        let co_blocks = out_channels.div_ceil(block);
        let spatial = h * w;
        let in_n_stride = ci_blocks * spatial * block;
        let out_n_stride = co_blocks * spatial * block;
        let mut out = vec![0.0f32; n_batch * out_n_stride];

        for n_idx in 0..n_batch {
            let in_slice = &input[n_idx * in_n_stride..(n_idx + 1) * in_n_stride];
            let out_slice = &mut out[n_idx * out_n_stride..(n_idx + 1) * out_n_stride];

            let mut flat = vec![0.0f32; spatial * out_channels];
            blocked_gemm_nchwc_a_parallel(
                in_slice,
                spatial,
                block,
                actual_ic,
                weight,
                &mut flat,
                out_channels,
                GemmEpilogue::IDENTITY,
                None,
                None,
            );
            // Scalar scatter + epilogue
            for ocb in 0..co_blocks {
                let c_start = ocb * block;
                let c_take = block.min(out_channels - c_start);
                for hw in 0..spatial {
                    let src_base = hw * out_channels + c_start;
                    let dst_base = ocb * spatial * block + hw * block;
                    for bi in 0..c_take {
                        let mut v = flat[src_base + bi];
                        if let Some(b) = bias {
                            v += b[c_start + bi];
                        }
                        if let Some(r) = residual {
                            v += r[n_idx * out_n_stride + dst_base + bi];
                        }
                        out_slice[dst_base + bi] = match activation {
                            Activation::None => v,
                            Activation::Relu => v.max(0.0),
                            Activation::Silu => v / (1.0 + (-v).exp()),
                        };
                    }
                }
            }
        }
        out
    }

    fn run_native(
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        residual: Option<&[f32]>,
        n_batch: usize,
        ci_blocks: usize,
        h: usize,
        w: usize,
        actual_ic: usize,
        out_channels: usize,
        block: usize,
        activation: Activation,
    ) -> Vec<f32> {
        let co_blocks = out_channels.div_ceil(block);
        let spatial = h * w;
        let out_len = n_batch * co_blocks * spatial * block;
        // Tail lanes of partial co_blocks (c_take < block) are never written by
        // flat_to_nchwc_fused — zero-init so they match the production calloc path.
        let mut output = AlignedVec::<f32>::calloc(out_len);
        nchwc_pw_compute(
            input,
            weight,
            bias,
            residual,
            output.as_mut_slice(),
            n_batch,
            ci_blocks,
            h,
            w,
            actual_ic,
            out_channels,
            co_blocks,
            block,
            activation,
            None,
        );
        output.as_slice().to_vec()
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn native_matches_ref_large_shape() {
        // Typical tracker: spatial=256, IC=96, OC=96, block=16
        let block = 16;
        let (n, ci_blocks, h, w, ic, oc) = (1, 6, 16, 16, 96, 96);
        let input = make_nchwc_input(n, ci_blocks, h, w, block);
        let weight = make_weight(ic, oc);
        let bias: Vec<f32> = (0..oc).map(|i| i as f32 * 0.01).collect();

        let ref_out = reference_nchwc_pw(
            &input,
            &weight,
            Some(&bias),
            None,
            n,
            ci_blocks,
            h,
            w,
            ic,
            oc,
            block,
            Activation::Relu,
        );
        let native_out = run_native(
            &input,
            &weight,
            Some(&bias),
            None,
            n,
            ci_blocks,
            h,
            w,
            ic,
            oc,
            block,
            Activation::Relu,
        );
        assert_eq!(ref_out.len(), native_out.len());
        let diff = max_abs_diff(&ref_out, &native_out);
        assert!(diff < 1e-5, "large shape diff={diff}");
    }

    #[test]
    fn native_matches_ref_small_k16() {
        // K=16 pathological case — previously excluded from GEMM path
        let block = 16;
        let (n, ci_blocks, h, w, ic, oc) = (1, 1, 128, 128, 16, 16);
        let input = make_nchwc_input(n, ci_blocks, h, w, block);
        let weight = make_weight(ic, oc);

        let ref_out = reference_nchwc_pw(
            &input,
            &weight,
            None,
            None,
            n,
            ci_blocks,
            h,
            w,
            ic,
            oc,
            block,
            Activation::None,
        );
        let native_out = run_native(
            &input,
            &weight,
            None,
            None,
            n,
            ci_blocks,
            h,
            w,
            ic,
            oc,
            block,
            Activation::None,
        );
        let diff = max_abs_diff(&ref_out, &native_out);
        assert!(diff == 0.0, "K=16 should be bitwise identical, diff={diff}");
    }

    #[test]
    fn native_matches_ref_with_residual() {
        let block = 16;
        let (n, ci_blocks, h, w, ic, oc) = (1, 4, 32, 32, 64, 64);
        let input = make_nchwc_input(n, ci_blocks, h, w, block);
        let weight = make_weight(ic, oc);
        let co_blocks = oc.div_ceil(block);
        let residual: Vec<f32> = vec![0.1f32; n * co_blocks * h * w * block];
        let bias: Vec<f32> = vec![0.5f32; oc];

        let ref_out = reference_nchwc_pw(
            &input,
            &weight,
            Some(&bias),
            Some(&residual),
            n,
            ci_blocks,
            h,
            w,
            ic,
            oc,
            block,
            Activation::Relu,
        );
        let native_out = run_native(
            &input,
            &weight,
            Some(&bias),
            Some(&residual),
            n,
            ci_blocks,
            h,
            w,
            ic,
            oc,
            block,
            Activation::Relu,
        );
        let diff = max_abs_diff(&ref_out, &native_out);
        assert!(diff < 1e-5, "residual diff={diff}");
    }

    #[test]
    fn native_matches_ref_silu() {
        let block = 16;
        let (n, ci_blocks, h, w, ic, oc) = (1, 3, 8, 8, 48, 48);
        let input = make_nchwc_input(n, ci_blocks, h, w, block);
        let weight = make_weight(ic, oc);
        let bias: Vec<f32> = (0..oc).map(|i| (i as f32) * 0.005).collect();

        let ref_out = reference_nchwc_pw(
            &input,
            &weight,
            Some(&bias),
            None,
            n,
            ci_blocks,
            h,
            w,
            ic,
            oc,
            block,
            Activation::Silu,
        );
        let native_out = run_native(
            &input,
            &weight,
            Some(&bias),
            None,
            n,
            ci_blocks,
            h,
            w,
            ic,
            oc,
            block,
            Activation::Silu,
        );
        let diff = max_abs_diff(&ref_out, &native_out);
        assert!(diff < 1e-5, "silu diff={diff}");
    }

    #[test]
    fn native_tail_block_oc_not_multiple_of_block() {
        // OC=20, block=16: 2 co_blocks, second has c_take=4
        let block = 16;
        let (n, ci_blocks, h, w, ic, oc) = (1, 2, 8, 8, 32, 20);
        let input = make_nchwc_input(n, ci_blocks, h, w, block);
        let weight = make_weight(ic, oc);
        let bias: Vec<f32> = (0..oc).map(|i| i as f32 * 0.01).collect();

        let ref_out = reference_nchwc_pw(
            &input,
            &weight,
            Some(&bias),
            None,
            n,
            ci_blocks,
            h,
            w,
            ic,
            oc,
            block,
            Activation::Relu,
        );
        let native_out = run_native(
            &input,
            &weight,
            Some(&bias),
            None,
            n,
            ci_blocks,
            h,
            w,
            ic,
            oc,
            block,
            Activation::Relu,
        );
        let diff = max_abs_diff(&ref_out, &native_out);
        assert!(diff < 1e-5, "tail block diff={diff}");
    }

    #[test]
    fn native_block8_aarch64_compatible() {
        // block=8 path (AVX2 / NEON)
        let block = 8;
        let (n, ci_blocks, h, w, ic, oc) = (1, 4, 16, 16, 32, 32);
        let input = make_nchwc_input(n, ci_blocks, h, w, block);
        let weight = make_weight(ic, oc);

        let ref_out = reference_nchwc_pw(
            &input,
            &weight,
            None,
            None,
            n,
            ci_blocks,
            h,
            w,
            ic,
            oc,
            block,
            Activation::None,
        );
        let native_out = run_native(
            &input,
            &weight,
            None,
            None,
            n,
            ci_blocks,
            h,
            w,
            ic,
            oc,
            block,
            Activation::None,
        );
        let diff = max_abs_diff(&ref_out, &native_out);
        assert!(diff < 1e-5, "block=8 diff={diff}");
    }
}
