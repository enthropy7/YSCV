use ::metal::*;
use ::objc::rc::autoreleasepool;
use std::collections::HashMap;

use super::dispatch::{dispatch_compute_op, dispatch_with_mps};
use super::types::*;

use yscv_kernels::metal_backend::metal_conv::{MetalEncoder, mps_gemm_f16_encode};
use yscv_tensor::Tensor;

use crate::error::OnnxError;

/// Execute a compiled Metal plan.
pub fn run_metal_plan(
    plan: &MetalPlan,
    input_data: &[f32],
) -> Result<HashMap<String, Tensor>, OnnxError> {
    let t0 = std::time::Instant::now();
    // Upload input — CPU-side or GPU-side conversion depending on plan
    match &plan.input_upload {
        InputUploadMode::CpuCastNchwToNhwc { batch, c, h, w } => {
            plan.inf.write_buffer_f32_nchw_as_f16_nhwc(
                plan.bufs.get(&plan.input_buf_name).unwrap(),
                input_data,
                *batch,
                *c,
                *h,
                *w,
            );
        }
        InputUploadMode::F32GpuCast => {
            plan.inf
                .write_buffer_f32(plan.bufs.get(&plan.input_buf_name).unwrap(), input_data);
        }
    }
    let t_upload = t0.elapsed();

    // Helper to get a buffer by name — internal invariant
    let get_buf = |name: &str| -> &Buffer {
        plan.bufs.get(name).unwrap_or_else(|| {
            unreachable!(
                "Metal plan: buffer '{name}' not found (have {} buffers). Bug in MetalPlan construction.",
                plan.bufs.len()
            );
        })
    };

    // Check if plan has any MPS ops (requires segmented encoding)
    let has_mps = plan
        .ops
        .iter()
        .any(|op| matches!(op, MetalOp::MpsConv { .. }));

    // Encode all ops in a single command buffer
    let result = autoreleasepool(|| {
        let cmd = plan.inf.queue.new_command_buffer();

        if has_mps {
            // Segmented dispatch: split into compute segments and MPS ops.
            // Each compute segment gets its own encoder; MPS encodes directly to cmd buf.
            dispatch_with_mps(plan, cmd);
        } else {
            let t_enc_start = std::time::Instant::now();
            let enc_raw = cmd.new_compute_command_encoder();
            let enc = MetalEncoder::new(enc_raw, &plan.inf);

            for op in &plan.ops {
                match op {
                    MetalOp::ConvGemm {
                        input,
                        weight,
                        bias,
                        output,
                        params,
                        f16io,
                        residual,
                    } => {
                        if *f16io {
                            let is_1x1 = params.kh == 1 && params.kw == 1;
                            let has_res = residual.is_some();
                            if params.n_out <= 32 && !has_res {
                                enc.conv_gemm_small_f16io(
                                    get_buf(input),
                                    get_buf(weight),
                                    get_buf(bias),
                                    get_buf(output),
                                    params,
                                );
                            } else if is_1x1 && has_res {
                                enc.conv1x1_simd_res_f16io(
                                    get_buf(input),
                                    get_buf(weight),
                                    get_buf(bias),
                                    get_buf(output),
                                    get_buf(residual.as_ref().unwrap()),
                                    params,
                                );
                            } else if is_1x1 {
                                enc.conv1x1_simd_f16io(
                                    get_buf(input),
                                    get_buf(weight),
                                    get_buf(bias),
                                    get_buf(output),
                                    params,
                                );
                            } else if has_res {
                                enc.conv3x3_simd_res_f16io(
                                    get_buf(input),
                                    get_buf(weight),
                                    get_buf(bias),
                                    get_buf(output),
                                    get_buf(residual.as_ref().unwrap()),
                                    params,
                                );
                            } else {
                                enc.conv3x3_simd_f16io(
                                    get_buf(input),
                                    get_buf(weight),
                                    get_buf(bias),
                                    get_buf(output),
                                    params,
                                );
                            }
                        } else {
                            enc.conv_gemm(
                                get_buf(input),
                                get_buf(weight),
                                get_buf(bias),
                                get_buf(output),
                                params,
                            );
                        }
                    }
                    MetalOp::ConvDirect {
                        input,
                        weight,
                        bias,
                        output,
                        params,
                        f16io,
                    } => {
                        if *f16io {
                            enc.conv_direct_f16io(
                                get_buf(input),
                                get_buf(weight),
                                get_buf(bias),
                                get_buf(output),
                                params,
                            );
                        } else {
                            enc.conv_direct(
                                get_buf(input),
                                get_buf(weight),
                                get_buf(bias),
                                get_buf(output),
                                params,
                            );
                        }
                    }
                    MetalOp::DepthwiseConv {
                        input,
                        weight,
                        bias,
                        output,
                        params,
                    } => {
                        enc.depthwise_conv_f16io(
                            get_buf(input),
                            get_buf(weight),
                            get_buf(bias),
                            get_buf(output),
                            params,
                        );
                    }
                    MetalOp::Binary {
                        a,
                        b,
                        out,
                        n,
                        op,
                        f16,
                    } => {
                        if *f16 {
                            enc.binary_f16(get_buf(a), get_buf(b), get_buf(out), *n, *op);
                        } else {
                            enc.binary(get_buf(a), get_buf(b), get_buf(out), *n, *op);
                        }
                    }
                    MetalOp::BroadcastBinary {
                        a,
                        b,
                        out,
                        n,
                        broadcast_dim,
                        op,
                        f16,
                    } => {
                        if *f16 {
                            enc.broadcast_binary_f16(
                                get_buf(a),
                                get_buf(b),
                                get_buf(out),
                                *n,
                                *broadcast_dim,
                                *op,
                            );
                        } else {
                            enc.broadcast_binary(
                                get_buf(a),
                                get_buf(b),
                                get_buf(out),
                                *n,
                                *broadcast_dim,
                                *op,
                            );
                        }
                    }
                    MetalOp::Unary {
                        input,
                        out,
                        n,
                        op,
                        f16,
                    } => {
                        if *f16 {
                            enc.unary_f16(get_buf(input), get_buf(out), *n, *op);
                        } else {
                            enc.unary(get_buf(input), get_buf(out), *n, *op);
                        }
                    }
                    MetalOp::SiLU { input, out, n, f16 } => {
                        if *f16 {
                            enc.silu_f16(get_buf(input), get_buf(out), *n);
                        } else {
                            enc.silu(get_buf(input), get_buf(out), *n);
                        }
                    }
                    MetalOp::Concat {
                        inputs,
                        channels,
                        out,
                        total_elements,
                        out_c,
                        f16,
                    } => {
                        let input_bufs: Vec<&Buffer> =
                            inputs.iter().map(|name| get_buf(name)).collect();
                        #[repr(C)]
                        struct ConcatP {
                            total_elements: u32,
                            out_c: u32,
                            n_inputs: u32,
                            channels: [u32; 16],
                        }
                        let mut ch = [0u32; 16];
                        for (i, &c) in channels.iter().enumerate().take(16) {
                            ch[i] = c;
                        }
                        let p = ConcatP {
                            total_elements: *total_elements,
                            out_c: *out_c,
                            n_inputs: inputs.len() as u32,
                            channels: ch,
                        };
                        if *f16 {
                            enc.enc
                                .set_compute_pipeline_state(&plan.inf.concat_channels_f16v4);
                        } else {
                            enc.enc
                                .set_compute_pipeline_state(&plan.inf.concat_channels);
                        }
                        for (idx, buf) in input_bufs.iter().enumerate().take(4) {
                            enc.enc.set_buffer(idx as u64, Some(buf), 0);
                        }
                        for idx in input_bufs.len()..4 {
                            enc.enc.set_buffer(idx as u64, Some(input_bufs[0]), 0);
                        }
                        enc.enc.set_buffer(4, Some(get_buf(out)), 0);
                        enc.enc.set_bytes(
                            5,
                            std::mem::size_of::<ConcatP>() as u64,
                            &p as *const ConcatP as *const _,
                        );
                        if *f16 {
                            let n_spatial = *total_elements / *out_c;
                            let out_c4 = (*out_c).div_ceil(4);
                            let tg_x = 16u64.min(out_c4 as u64);
                            let tg_y = (256u64 / tg_x).max(1);
                            enc.enc.dispatch_thread_groups(
                                MTLSize {
                                    width: (out_c4 as u64).div_ceil(tg_x),
                                    height: (n_spatial as u64).div_ceil(tg_y),
                                    depth: 1,
                                },
                                MTLSize {
                                    width: tg_x,
                                    height: tg_y,
                                    depth: 1,
                                },
                            );
                        } else {
                            let wg = 256u64;
                            let groups = (*total_elements as u64).div_ceil(wg);
                            enc.enc.dispatch_thread_groups(
                                MTLSize {
                                    width: groups,
                                    height: 1,
                                    depth: 1,
                                },
                                MTLSize {
                                    width: wg,
                                    height: 1,
                                    depth: 1,
                                },
                            );
                        }
                    }
                    MetalOp::Split {
                        input,
                        out,
                        spatial,
                        in_c,
                        out_c,
                        offset_c,
                        f16,
                    } => {
                        if *f16 {
                            enc.split_channels_f16(
                                get_buf(input),
                                get_buf(out),
                                *spatial,
                                *in_c,
                                *out_c,
                                *offset_c,
                            );
                        } else {
                            enc.split_channels(
                                get_buf(input),
                                get_buf(out),
                                *spatial,
                                *in_c,
                                *out_c,
                                *offset_c,
                            );
                        }
                    }
                    MetalOp::SplitFused {
                        input,
                        outputs,
                        split_sizes,
                        spatial,
                        in_c,
                    } => {
                        #[repr(C)]
                        struct FusedSplitP {
                            spatial: u32,
                            in_c: u32,
                            n_outputs: u32,
                            split_c0: u32,
                            split_c1: u32,
                            split_c2: u32,
                        }
                        let p = FusedSplitP {
                            spatial: *spatial,
                            in_c: *in_c,
                            n_outputs: outputs.len() as u32,
                            split_c0: split_sizes.first().copied().unwrap_or(0),
                            split_c1: split_sizes.get(1).copied().unwrap_or(0),
                            split_c2: split_sizes.get(2).copied().unwrap_or(0),
                        };
                        enc.enc
                            .set_compute_pipeline_state(&plan.inf.split_fused_f16v4);
                        enc.enc.set_buffer(0, Some(get_buf(input)), 0);
                        enc.enc.set_buffer(1, Some(get_buf(&outputs[0])), 0);
                        enc.enc.set_buffer(
                            2,
                            Some(get_buf(outputs.get(1).unwrap_or(&outputs[0]))),
                            0,
                        );
                        enc.enc.set_buffer(
                            3,
                            Some(get_buf(outputs.get(2).unwrap_or(&outputs[0]))),
                            0,
                        );
                        enc.enc.set_bytes(
                            4,
                            std::mem::size_of::<FusedSplitP>() as u64,
                            &p as *const FusedSplitP as *const _,
                        );
                        let c4 = (*in_c as u64).div_ceil(4);
                        let tg_x = c4.min(16);
                        let tg_y = (256u64 / tg_x).min(*spatial as u64).max(1);
                        enc.enc.dispatch_thread_groups(
                            MTLSize {
                                width: c4.div_ceil(tg_x),
                                height: (*spatial as u64).div_ceil(tg_y),
                                depth: 1,
                            },
                            MTLSize {
                                width: tg_x,
                                height: tg_y,
                                depth: 1,
                            },
                        );
                    }
                    MetalOp::MaxPool {
                        input,
                        out,
                        batch,
                        ih,
                        iw,
                        ic,
                        oh,
                        ow,
                        kh,
                        kw,
                        sh,
                        sw,
                        pad_h,
                        pad_w,
                        f16,
                    } => {
                        if *f16 {
                            enc.maxpool2d_f16(
                                get_buf(input),
                                get_buf(out),
                                *batch,
                                *ih,
                                *iw,
                                *ic,
                                *oh,
                                *ow,
                                *kh,
                                *kw,
                                *sh,
                                *sw,
                                *pad_h,
                                *pad_w,
                            );
                        } else {
                            enc.maxpool2d(
                                get_buf(input),
                                get_buf(out),
                                *batch,
                                *ih,
                                *iw,
                                *ic,
                                *oh,
                                *ow,
                                *kh,
                                *kw,
                                *sh,
                                *sw,
                                *pad_h,
                                *pad_w,
                            );
                        }
                    }
                    MetalOp::Resize {
                        input,
                        out,
                        batch,
                        ih,
                        iw,
                        ic,
                        oh,
                        ow,
                        scale_h,
                        scale_w,
                        f16,
                    } => {
                        if *f16 {
                            enc.resize_nearest_f16v4(
                                get_buf(input),
                                get_buf(out),
                                *batch,
                                *ih,
                                *iw,
                                *ic,
                                *oh,
                                *ow,
                                *scale_h,
                                *scale_w,
                            );
                        } else {
                            enc.resize_nearest(
                                get_buf(input),
                                get_buf(out),
                                *batch,
                                *ih,
                                *iw,
                                *ic,
                                *oh,
                                *ow,
                                *scale_h,
                                *scale_w,
                            );
                        }
                    }
                    MetalOp::Softmax {
                        input,
                        out,
                        outer,
                        dim,
                        f16,
                    } => {
                        if *f16 {
                            enc.softmax_f16(get_buf(input), get_buf(out), *outer, *dim);
                        } else {
                            enc.softmax(get_buf(input), get_buf(out), *outer, *dim);
                        }
                    }
                    MetalOp::Transpose2D {
                        input,
                        out,
                        rows,
                        cols,
                        f16,
                    } => {
                        if *f16 {
                            enc.transpose_2d_f16(get_buf(input), get_buf(out), *rows, *cols);
                        } else {
                            enc.transpose_2d(get_buf(input), get_buf(out), *rows, *cols);
                        }
                    }
                    MetalOp::Permute0213 {
                        input,
                        out,
                        d0,
                        d1,
                        d2,
                        d3,
                        f16,
                    } => {
                        if *f16 {
                            enc.permute_0213_f16(get_buf(input), get_buf(out), *d0, *d1, *d2, *d3);
                        } else {
                            enc.permute_0213(get_buf(input), get_buf(out), *d0, *d1, *d2, *d3);
                        }
                    }
                    MetalOp::SliceCopy {
                        input,
                        out,
                        n,
                        src_offset,
                        f16,
                    } => {
                        #[repr(C)]
                        struct SliceP0 {
                            n: u32,
                            src_offset: u32,
                        }
                        let p = SliceP0 {
                            n: *n,
                            src_offset: *src_offset,
                        };
                        if *f16 {
                            enc.enc.set_compute_pipeline_state(&plan.inf.slice_copy_f16);
                        } else {
                            enc.enc.set_compute_pipeline_state(&plan.inf.slice_copy);
                        }
                        enc.enc.set_buffer(0, Some(get_buf(input)), 0);
                        enc.enc.set_buffer(1, Some(get_buf(out)), 0);
                        enc.enc.set_bytes(
                            2,
                            std::mem::size_of::<SliceP0>() as u64,
                            &p as *const SliceP0 as *const _,
                        );
                        let wg = 256u64;
                        let groups = (*n as u64).div_ceil(wg);
                        enc.enc.dispatch_thread_groups(
                            MTLSize {
                                width: groups,
                                height: 1,
                                depth: 1,
                            },
                            MTLSize {
                                width: wg,
                                height: 1,
                                depth: 1,
                            },
                        );
                    }
                    MetalOp::CpuReshape {
                        input,
                        out,
                        n,
                        nhwc_to_nchw,
                        nchw_to_nhwc,
                        f16,
                    } => {
                        if let Some((nn, h, w, c)) = nhwc_to_nchw {
                            // NHWC→NCHW permutation entirely on GPU
                            if *f16 {
                                enc.permute_nhwc_to_nchw_f16(
                                    get_buf(input),
                                    get_buf(out),
                                    *nn,
                                    *h,
                                    *w,
                                    *c,
                                );
                            } else {
                                enc.permute_nhwc_to_nchw(
                                    get_buf(input),
                                    get_buf(out),
                                    *nn,
                                    *h,
                                    *w,
                                    *c,
                                );
                            }
                        } else if let Some((nn, cc, hh, ww)) = nchw_to_nhwc {
                            // Tuple is (N, C, H, W) but encoder expects (n, h, w, c)
                            if *f16 {
                                enc.permute_nchw_to_nhwc_f16(
                                    get_buf(input),
                                    get_buf(out),
                                    *nn,
                                    *hh,
                                    *ww,
                                    *cc,
                                );
                            } else {
                                enc.permute_nchw_to_nhwc(
                                    get_buf(input),
                                    get_buf(out),
                                    *nn,
                                    *hh,
                                    *ww,
                                    *cc,
                                );
                            }
                        } else {
                            // Simple flat copy (Reshape without layout change)
                            #[repr(C)]
                            struct SliceP {
                                n: u32,
                                src_offset: u32,
                            }
                            let p = SliceP {
                                n: *n,
                                src_offset: 0,
                            };
                            if *f16 {
                                enc.enc.set_compute_pipeline_state(&plan.inf.slice_copy_f16);
                            } else {
                                enc.enc.set_compute_pipeline_state(&plan.inf.slice_copy);
                            }
                            enc.enc.set_buffer(0, Some(get_buf(input)), 0);
                            enc.enc.set_buffer(1, Some(get_buf(out)), 0);
                            enc.enc.set_bytes(
                                2,
                                std::mem::size_of::<SliceP>() as u64,
                                &p as *const SliceP as *const _,
                            );
                            let wg = 256u64;
                            let groups = (*n as u64).div_ceil(wg);
                            enc.enc.dispatch_thread_groups(
                                MTLSize {
                                    width: groups,
                                    height: 1,
                                    depth: 1,
                                },
                                MTLSize {
                                    width: wg,
                                    height: 1,
                                    depth: 1,
                                },
                            );
                        }
                    }
                    MetalOp::FlatConcat {
                        inputs,
                        sizes,
                        out,
                        f16,
                    } => {
                        // Copy each input contiguously into output at sequential offsets
                        let out_buf = get_buf(out);
                        let mut dst_offset_bytes: u64 = 0;
                        let bytes_per_elem: u64 = if *f16 { 2 } else { 4 };
                        for (input_name, &size) in inputs.iter().zip(sizes.iter()) {
                            if size == 0 {
                                continue;
                            }
                            #[repr(C)]
                            struct SliceP2 {
                                n: u32,
                                src_offset: u32,
                            }
                            let p = SliceP2 {
                                n: size,
                                src_offset: 0,
                            };
                            if *f16 {
                                enc.enc.set_compute_pipeline_state(&plan.inf.slice_copy_f16);
                            } else {
                                enc.enc.set_compute_pipeline_state(&plan.inf.slice_copy);
                            }
                            enc.enc.set_buffer(0, Some(get_buf(input_name)), 0);
                            enc.enc.set_buffer(1, Some(out_buf), dst_offset_bytes);
                            enc.enc.set_bytes(
                                2,
                                std::mem::size_of::<SliceP2>() as u64,
                                &p as *const SliceP2 as *const _,
                            );
                            let wg = 256u64;
                            let groups = (size as u64).div_ceil(wg);
                            enc.enc.dispatch_thread_groups(
                                MTLSize {
                                    width: groups,
                                    height: 1,
                                    depth: 1,
                                },
                                MTLSize {
                                    width: wg,
                                    height: 1,
                                    depth: 1,
                                },
                            );
                            dst_offset_bytes += (size as u64) * bytes_per_elem;
                        }
                    }
                    MetalOp::MatMul {
                        a,
                        b,
                        out,
                        m,
                        n,
                        k,
                        f16,
                    } => {
                        if *f16 {
                            enc.matmul_f16io(get_buf(a), get_buf(b), get_buf(out), *m, *n, *k);
                        } else {
                            enc.matmul(get_buf(a), get_buf(b), get_buf(out), *m, *n, *k);
                        }
                    }
                    MetalOp::CastF32ToF16 { input, out, n } => {
                        enc.cast_f32_to_f16(get_buf(input), get_buf(out), *n);
                    }
                    MetalOp::CastF16ToF32 { input, out, n } => {
                        enc.cast_f16_to_f32(get_buf(input), get_buf(out), *n);
                    }
                    MetalOp::ConvWinograd {
                        input,
                        weight,
                        bias,
                        output,
                        transformed_input,
                        gemm_output,
                        wino_params,
                        ic,
                        oc,
                        residual,
                    } => {
                        enc.winograd4x4_input_transform_f16(
                            get_buf(input),
                            get_buf(transformed_input),
                            wino_params,
                        );
                        enc.winograd4x4_batched_gemm_simd_f16io(
                            get_buf(transformed_input),
                            get_buf(weight),
                            get_buf(gemm_output),
                            wino_params.n_tiles,
                            *oc,
                            *ic,
                        );
                        if let Some(res) = residual {
                            enc.winograd4x4_output_transform_residual_f16(
                                get_buf(gemm_output),
                                get_buf(bias),
                                get_buf(output),
                                wino_params,
                                get_buf(res),
                            );
                        } else {
                            enc.winograd4x4_output_transform_f16(
                                get_buf(gemm_output),
                                get_buf(bias),
                                get_buf(output),
                                wino_params,
                            );
                        }
                    }
                    MetalOp::NhwcToFlatConcat { .. } | MetalOp::ChannelScatter { .. } => {
                        dispatch_compute_op(plan, &enc, op, &plan.bufs);
                    }
                    MetalOp::MpsConv { .. } => {
                        // MPS ops handled in separate pass below — skip here.
                        // This branch should not be reached in the non-MPS path.
                        unreachable!("MpsConv should be handled by the MPS dispatch path");
                    }
                }
            }

            enc_raw.end_encoding();
            let t_enc_elapsed = t_enc_start.elapsed();
            cmd.commit();
            cmd.wait_until_completed();

            // Debug: compare all Metal buffers against CPU reference
            if !plan.cpu_ref.is_empty() {
                #[inline]
                fn f16_bits_to_f32(bits: u16) -> f32 {
                    let sign = ((bits >> 15) & 1) as u32;
                    let exp = ((bits >> 10) & 0x1F) as u32;
                    let mant = (bits & 0x3FF) as u32;
                    if exp == 0 {
                        if mant == 0 {
                            return if sign == 1 { -0.0 } else { 0.0 };
                        }
                        let mut m = mant;
                        let mut e = 1u32;
                        while m & 0x400 == 0 {
                            m <<= 1;
                            e += 1;
                        }
                        let f_exp = (127 - 15 + 1 - e) << 23;
                        let f_mant = (m & 0x3FF) << 13;
                        f32::from_bits((sign << 31) | f_exp | f_mant)
                    } else if exp == 31 {
                        if mant == 0 {
                            return if sign == 1 {
                                f32::NEG_INFINITY
                            } else {
                                f32::INFINITY
                            };
                        }
                        f32::NAN
                    } else {
                        let f_exp = (exp + 127 - 15) << 23;
                        let f_mant = mant << 13;
                        f32::from_bits((sign << 31) | f_exp | f_mant)
                    }
                }
                let mut errors: Vec<(String, f32, f32, usize)> = Vec::new();
                for (name, cpu_vals) in &plan.cpu_ref {
                    if let Some(buf) = plan.bufs.get(name) {
                        let n = cpu_vals.len();
                        if buf.length() as usize >= n * 2 {
                            let ptr = buf.contents() as *const u16;
                            let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
                            let mut max_diff = 0.0f32;
                            let mut sum_diff = 0.0f32;
                            for i in 0..n {
                                let mtl_val = f16_bits_to_f32(slice[i]);
                                let diff = (mtl_val - cpu_vals[i]).abs();
                                if !diff.is_nan() {
                                    max_diff = max_diff.max(diff);
                                    sum_diff += diff;
                                }
                            }
                            let mean_diff = sum_diff / n as f32;
                            if max_diff > 1.0 {
                                errors.push((name.clone(), max_diff, mean_diff, n));
                            }
                        }
                    }
                }
                errors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                eprintln!("  [COMPARE] {} buffers with max_diff > 1.0:", errors.len());
                for (name, max_diff, mean_diff, n) in errors.iter().take(30) {
                    let shape = plan
                        .buf_shapes
                        .get(name)
                        .map(|s| format!("{:?}", s))
                        .unwrap_or_default();
                    let nhwc = plan.buf_nhwc.get(name).copied().unwrap_or(false);
                    let buf = plan.bufs.get(name).unwrap();
                    let ptr = buf.contents() as *const u16;
                    let slice = unsafe { std::slice::from_raw_parts(ptr, *n) };
                    let cpu_ref_vals = plan.cpu_ref.get(name).unwrap();
                    let show = 4.min(*n);
                    let mtl_vals: Vec<f32> = (0..show).map(|i| f16_bits_to_f32(slice[i])).collect();
                    let cpu_vals: Vec<f32> = cpu_ref_vals[..show].to_vec();
                    eprintln!(
                        "    '{}' {} nhwc={} max_diff={:.2} mean={:.4} mtl={:.3?} cpu={:.3?}",
                        name, shape, nhwc, max_diff, mean_diff, mtl_vals, cpu_vals
                    );
                }
                // Dump DFL softmax I/O for first few positions
                let dfl_names = [
                    "/model.22/dfl/Transpose_output_0",
                    "/model.22/dfl/Softmax_output_0",
                    "/model.22/dfl/Transpose_1_output_0",
                    "/model.22/dfl/conv/Conv_output_0",
                ];
                for dfl_name in &dfl_names {
                    if let Some(buf) = plan.bufs.get(*dfl_name) {
                        let shape = plan.buf_shapes.get(*dfl_name).cloned().unwrap_or_default();
                        let n: usize = shape.iter().product();
                        if buf.length() as usize >= n * 2 && n > 0 {
                            let ptr = buf.contents() as *const u16;
                            let slice = unsafe { std::slice::from_raw_parts(ptr, n.min(1000)) };
                            let vals: Vec<f32> =
                                slice.iter().map(|&b| f16_bits_to_f32(b)).collect();
                            let show = 32.min(vals.len());
                            eprintln!("  [DFL] '{}' shape={:?} first {}:", dfl_name, shape, show);
                            eprintln!("    mtl: {:?}", &vals[..show]);
                            if let Some(cpu) = plan.cpu_ref.get(*dfl_name) {
                                let cpu_show = show.min(cpu.len());
                                eprintln!("    cpu: {:?}", &cpu[..cpu_show]);
                            }
                            // For softmax output: check sum along last dim (should be ~1.0)
                            if dfl_name.contains("Softmax") && shape.len() == 4 && shape[3] > 0 {
                                let dim = shape[3];
                                let n_positions = 4.min(n / dim);
                                for p in 0..n_positions {
                                    let start = p * dim;
                                    let end = start + dim;
                                    if end <= vals.len() {
                                        let sum: f32 = vals[start..end].iter().sum();
                                        eprintln!(
                                            "    pos {} sum={:.6} vals[0..4]={:.4?}",
                                            p,
                                            sum,
                                            &vals[start..start + 4.min(dim)]
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }

            let t_gpu_done = t0.elapsed();
            if std::env::var("METAL_TIMING").is_ok() {
                eprintln!(
                    "  [TIMING] upload={:.2}ms  encode={:.2}ms  gpu={:.2}ms  total={:.2}ms",
                    t_upload.as_secs_f64() * 1000.0,
                    t_enc_elapsed.as_secs_f64() * 1000.0,
                    (t_gpu_done - t_upload - t_enc_elapsed).as_secs_f64() * 1000.0,
                    t0.elapsed().as_secs_f64() * 1000.0
                );
            }
        } // end else (non-MPS path)
        let t_encode = t0.elapsed();
        let t_gpu = t0.elapsed();

        // Per-op-type profiling: encode each op type in isolation to measure GPU time
        if std::env::var("METAL_PROFILE").is_ok() {
            // Group ops by type name
            let op_name = |op: &MetalOp| -> &'static str {
                match op {
                    MetalOp::ConvGemm { params, .. } => {
                        if params.kh == 1 && params.kw == 1 {
                            "ConvGemm1x1"
                        } else {
                            "ConvGemm3x3"
                        }
                    }
                    MetalOp::ConvDirect { .. } => "ConvDirect",
                    MetalOp::DepthwiseConv { .. } => "DepthwiseConv",
                    MetalOp::Binary { .. } => "Binary",
                    MetalOp::BroadcastBinary { .. } => "BroadcastBinary",
                    MetalOp::Unary { .. } => "Unary",
                    MetalOp::SiLU { .. } => "SiLU",
                    MetalOp::Concat { .. } => "Concat",
                    MetalOp::Split { .. } => "Split",
                    MetalOp::SplitFused { .. } => "SplitFused",
                    MetalOp::MaxPool { .. } => "MaxPool",
                    MetalOp::Resize { .. } => "Resize",
                    MetalOp::Softmax { .. } => "Softmax",
                    MetalOp::Transpose2D { .. } => "Transpose2D",
                    MetalOp::CpuReshape { .. } => "CpuReshape",
                    MetalOp::Permute0213 { .. } => "Permute0213",
                    MetalOp::SliceCopy { .. } => "SliceCopy",
                    MetalOp::FlatConcat { .. } => "FlatConcat",
                    MetalOp::MatMul { .. } => "MatMul",
                    MetalOp::CastF32ToF16 { .. } => "CastF32ToF16",
                    MetalOp::CastF16ToF32 { .. } => "CastF16ToF32",
                    MetalOp::MpsConv { kh, kw, .. } => {
                        if *kh == 1 && *kw == 1 {
                            "MpsConv1x1"
                        } else {
                            "MpsConv3x3"
                        }
                    }
                    MetalOp::ConvWinograd { .. } => "ConvWinograd",
                    MetalOp::NhwcToFlatConcat { .. } => "NhwcToFlatConcat",
                    MetalOp::ChannelScatter { .. } => "ChannelScatter",
                }
            };
            // Measure per-op by encoding each in its own command buffer
            let mut type_times: std::collections::HashMap<&str, (f64, usize)> =
                std::collections::HashMap::new();
            let mut individual_ops: Vec<(String, f64)> = Vec::new();
            for (idx, op) in plan.ops.iter().enumerate() {
                let pcmd = plan.inf.queue.new_command_buffer();
                let penc_raw = pcmd.new_compute_command_encoder();
                let penc = MetalEncoder::new(penc_raw, &plan.inf);
                // Re-dispatch single op
                match op {
                    MetalOp::ConvGemm {
                        input,
                        weight,
                        bias,
                        output,
                        params,
                        f16io,
                        residual,
                    } => {
                        if *f16io {
                            let is_1x1 = params.kh == 1 && params.kw == 1;
                            let has_res = residual.is_some();
                            if params.n_out <= 32 && !has_res {
                                penc.conv_gemm_small_f16io(
                                    get_buf(input),
                                    get_buf(weight),
                                    get_buf(bias),
                                    get_buf(output),
                                    params,
                                );
                            } else if is_1x1 && has_res {
                                penc.conv1x1_simd_res_f16io(
                                    get_buf(input),
                                    get_buf(weight),
                                    get_buf(bias),
                                    get_buf(output),
                                    get_buf(residual.as_ref().unwrap()),
                                    params,
                                );
                            } else if is_1x1 {
                                penc.conv1x1_simd_f16io(
                                    get_buf(input),
                                    get_buf(weight),
                                    get_buf(bias),
                                    get_buf(output),
                                    params,
                                );
                            } else if has_res {
                                penc.conv3x3_simd_res_f16io(
                                    get_buf(input),
                                    get_buf(weight),
                                    get_buf(bias),
                                    get_buf(output),
                                    get_buf(residual.as_ref().unwrap()),
                                    params,
                                );
                            } else {
                                penc.conv3x3_simd_f16io(
                                    get_buf(input),
                                    get_buf(weight),
                                    get_buf(bias),
                                    get_buf(output),
                                    params,
                                );
                            }
                        } else {
                            penc.conv_gemm(
                                get_buf(input),
                                get_buf(weight),
                                get_buf(bias),
                                get_buf(output),
                                params,
                            );
                        }
                    }
                    MetalOp::ConvDirect {
                        input,
                        weight,
                        bias,
                        output,
                        params,
                        f16io,
                    } => {
                        if *f16io {
                            penc.conv_direct_f16io(
                                get_buf(input),
                                get_buf(weight),
                                get_buf(bias),
                                get_buf(output),
                                params,
                            );
                        } else {
                            penc.conv_direct(
                                get_buf(input),
                                get_buf(weight),
                                get_buf(bias),
                                get_buf(output),
                                params,
                            );
                        }
                    }
                    MetalOp::DepthwiseConv {
                        input,
                        weight,
                        bias,
                        output,
                        params,
                    } => {
                        penc.depthwise_conv_f16io(
                            get_buf(input),
                            get_buf(weight),
                            get_buf(bias),
                            get_buf(output),
                            params,
                        );
                    }
                    MetalOp::MpsConv {
                        input,
                        weight,
                        bias,
                        output,
                        im2col_buf,
                        m,
                        n,
                        k,
                        act,
                        batch,
                        ih,
                        iw,
                        ic,
                        oh,
                        ow,
                        kh,
                        kw,
                        sh,
                        sw,
                        pad_h,
                        pad_w,
                    } => {
                        // For profiling: im2col + MPS GEMM + bias_act in one measurement
                        if let Some(col_name) = im2col_buf {
                            penc.im2col_f16(
                                get_buf(input),
                                get_buf(col_name),
                                *batch,
                                *ih,
                                *iw,
                                *ic,
                                *oh,
                                *ow,
                                *kh,
                                *kw,
                                *sh,
                                *sw,
                                *pad_h,
                                *pad_w,
                            );
                        }
                        penc_raw.end_encoding();
                        let gemm_a = im2col_buf.as_deref().unwrap_or(input.as_str());
                        mps_gemm_f16_encode(
                            &plan.inf.device,
                            pcmd,
                            get_buf(gemm_a),
                            get_buf(weight),
                            get_buf(output),
                            *m,
                            *n,
                            *k,
                            1.0,
                            0.0,
                            false,
                            false,
                        );
                        let penc2_raw = pcmd.new_compute_command_encoder();
                        let penc2 = MetalEncoder::new(penc2_raw, &plan.inf);
                        penc2.bias_act_f16(get_buf(output), get_buf(bias), *m * *n, *n, *act);
                        penc2_raw.end_encoding();
                        let pt = std::time::Instant::now();
                        pcmd.commit();
                        pcmd.wait_until_completed();
                        let dur = pt.elapsed().as_secs_f64() * 1000.0;
                        let name = op_name(op);
                        let entry = type_times.entry(name).or_insert((0.0, 0));
                        entry.0 += dur;
                        entry.1 += 1;
                        continue; // skip the normal end_encoding path below
                    }
                    _ => {
                        dispatch_compute_op(plan, &penc, op, &plan.bufs);
                    }
                }
                penc_raw.end_encoding();
                let pt = std::time::Instant::now();
                pcmd.commit();
                pcmd.wait_until_completed();
                let dur = pt.elapsed().as_secs_f64() * 1000.0;
                let name = op_name(op);
                let entry = type_times.entry(name).or_insert((0.0, 0));
                entry.0 += dur;
                entry.1 += 1;
                // Collect individual op details
                let detail = match op {
                    MetalOp::ConvGemm { params, .. } => {
                        format!(
                            "{} m={} n={} k={} kh={}",
                            name, params.m, params.n_out, params.k, params.kh
                        )
                    }
                    MetalOp::ConvWinograd {
                        wino_params,
                        ic,
                        oc,
                        ..
                    } => {
                        format!("{} tiles={} ic={} oc={}", name, wino_params.n_tiles, ic, oc)
                    }
                    MetalOp::ConvDirect { params, .. } => {
                        format!("{} m={} n={} k={}", name, params.m, params.n_out, params.k)
                    }
                    _ => name.to_string(),
                };
                individual_ops.push((detail, dur));
            }
            let mut sorted: Vec<_> = type_times.into_iter().collect();
            sorted.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());
            let total_prof: f64 = sorted.iter().map(|x| x.1.0).sum();
            eprintln!(
                "  [PROFILE] Per-op-type GPU time (total={:.2}ms):",
                total_prof
            );
            for (name, (ms, count)) in &sorted {
                eprintln!(
                    "    {:20} {:3}x  {:.2}ms  ({:.1}%)",
                    name,
                    count,
                    ms,
                    ms / total_prof * 100.0
                );
            }
            // Top 20 individual ops
            individual_ops.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            eprintln!("  [PROFILE] Top 20 individual ops:");
            for (detail, ms) in individual_ops.iter().take(20) {
                eprintln!("    {:.3}ms  {}", ms, detail);
            }
        }

        // Debug: find first op that produces NaN in its output buffer
        if std::env::var("METAL_NAN").is_ok() {
            for (idx, op) in plan.ops.iter().enumerate() {
                let (out_name, n_elems) = match op {
                    MetalOp::ConvGemm { output, params, .. } => {
                        (output.as_str(), (params.m * params.n_out) as usize)
                    }
                    MetalOp::ConvDirect { output, params, .. } => {
                        (output.as_str(), (params.m * params.n_out) as usize)
                    }
                    MetalOp::DepthwiseConv { output, params, .. } => {
                        (output.as_str(), (params.m * params.n_out) as usize)
                    }
                    MetalOp::Binary { out, n, .. } => (out.as_str(), *n as usize),
                    MetalOp::BroadcastBinary { out, n, .. } => (out.as_str(), *n as usize),
                    MetalOp::Unary { out, n, .. } => (out.as_str(), *n as usize),
                    MetalOp::SiLU { out, n, .. } => (out.as_str(), *n as usize),
                    MetalOp::MaxPool {
                        out,
                        batch,
                        oh,
                        ow,
                        ic,
                        ..
                    } => (out.as_str(), (*batch * *oh * *ow * *ic) as usize),
                    MetalOp::Resize {
                        out,
                        batch,
                        oh,
                        ow,
                        ic,
                        ..
                    } => (out.as_str(), (*batch * *oh * *ow * *ic) as usize),
                    MetalOp::Softmax {
                        out, outer, dim, ..
                    } => (out.as_str(), (*outer * *dim) as usize),
                    MetalOp::Transpose2D {
                        out, rows, cols, ..
                    } => (out.as_str(), (*rows * *cols) as usize),
                    MetalOp::MatMul { out, m, n, .. } => (out.as_str(), (*m * *n) as usize),
                    MetalOp::CastF32ToF16 { out, n, .. } => (out.as_str(), *n as usize),
                    MetalOp::CastF16ToF32 { out, n, .. } => (out.as_str(), *n as usize),
                    MetalOp::SliceCopy { out, n, .. } => (out.as_str(), *n as usize),
                    MetalOp::Split {
                        out,
                        spatial,
                        out_c,
                        ..
                    } => (out.as_str(), (*spatial * *out_c) as usize),
                    MetalOp::SplitFused {
                        outputs,
                        spatial,
                        in_c,
                        ..
                    } => (outputs[0].as_str(), (*spatial * *in_c) as usize),
                    MetalOp::Concat {
                        out,
                        total_elements,
                        ..
                    } => (out.as_str(), *total_elements as usize),
                    MetalOp::FlatConcat { out, sizes, .. } => {
                        (out.as_str(), sizes.iter().sum::<u32>() as usize)
                    }
                    MetalOp::CpuReshape { out, n, .. } => (out.as_str(), *n as usize),
                    MetalOp::Permute0213 {
                        out,
                        d0,
                        d1,
                        d2,
                        d3,
                        ..
                    } => (out.as_str(), (*d0 * *d1 * *d2 * *d3) as usize),
                    MetalOp::MpsConv { output, m, n, .. } => (output.as_str(), (*m * *n) as usize),
                    MetalOp::ConvWinograd {
                        output,
                        wino_params,
                        ..
                    } => {
                        let n =
                            (wino_params.batch * wino_params.oh * wino_params.ow * wino_params.oc)
                                as usize;
                        (output.as_str(), n)
                    }
                    MetalOp::NhwcToFlatConcat {
                        out,
                        c,
                        total_spatial,
                        ..
                    } => (out.as_str(), (*c * *total_spatial) as usize),
                    MetalOp::ChannelScatter {
                        out,
                        spatial,
                        dst_c,
                        ..
                    } => (out.as_str(), (*spatial * *dst_c) as usize),
                };
                if n_elems == 0 {
                    continue;
                }
                if let Some(buf) = plan.bufs.get(out_name) {
                    // Check if output is f16 or f32
                    let is_f32 = matches!(op, MetalOp::CastF16ToF32 { .. });
                    let check_n = n_elems.min(65536);
                    let has_nan = if is_f32 {
                        let data = plan.inf.read_buffer_f32(buf, check_n);
                        data.iter().any(|v| v.is_nan() || v.is_infinite())
                    } else {
                        let data = plan.inf.read_buffer_f16_as_f32(buf, check_n);
                        data.iter().any(|v| v.is_nan() || v.is_infinite())
                    };
                    if has_nan {
                        // Also check inputs for this op
                        let input_names: Vec<&str> = match op {
                            MetalOp::MatMul { a, b, .. } => vec![a.as_str(), b.as_str()],
                            MetalOp::ConvGemm {
                                input,
                                weight,
                                bias,
                                ..
                            } => vec![input.as_str(), weight.as_str(), bias.as_str()],
                            MetalOp::DepthwiseConv {
                                input,
                                weight,
                                bias,
                                ..
                            } => vec![input.as_str(), weight.as_str(), bias.as_str()],
                            MetalOp::Binary { a, b, .. } => vec![a.as_str(), b.as_str()],
                            MetalOp::BroadcastBinary { a, b, .. } => vec![a.as_str(), b.as_str()],
                            MetalOp::Unary { input, .. } | MetalOp::SiLU { input, .. } => {
                                vec![input.as_str()]
                            }
                            MetalOp::Softmax { input, .. } => vec![input.as_str()],
                            _ => vec![],
                        };
                        for inp_name in &input_names {
                            if let (Some(inp_buf), Some(inp_shape)) =
                                (plan.bufs.get(*inp_name), plan.buf_shapes.get(*inp_name))
                            {
                                let inp_n: usize = inp_shape.iter().product();
                                if inp_n > 0 {
                                    // Bias buffers (__mtl_b_) and _f32 buffers are f32
                                    let is_f32_buf = inp_name.starts_with("__mtl_b_")
                                        || inp_name.ends_with("_f32");
                                    let inp_data = if is_f32_buf {
                                        plan.inf.read_buffer_f32(inp_buf, inp_n.min(2048))
                                    } else {
                                        plan.inf.read_buffer_f16_as_f32(inp_buf, inp_n.min(2048))
                                    };
                                    let n_nan = inp_data.iter().filter(|v| v.is_nan()).count();
                                    let n_inf = inp_data.iter().filter(|v| v.is_infinite()).count();
                                    if n_nan > 0 || n_inf > 0 {
                                        eprintln!(
                                            "    Input '{}' shape={:?} has {} NaN, {} Inf in first {} elems",
                                            inp_name,
                                            inp_shape,
                                            n_nan,
                                            n_inf,
                                            inp_data.len()
                                        );
                                        let first_bad = inp_data
                                            .iter()
                                            .position(|v| v.is_nan() || v.is_infinite())
                                            .unwrap_or(0);
                                        eprintln!(
                                            "    First bad at idx {}: {:?}",
                                            first_bad,
                                            &inp_data[first_bad..inp_data.len().min(first_bad + 8)]
                                        );
                                    } else {
                                        let max_abs =
                                            inp_data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                                        eprintln!(
                                            "    Input '{}' shape={:?} OK (max_abs={:.1})",
                                            inp_name, inp_shape, max_abs
                                        );
                                    }
                                }
                            }
                        }
                        let op_name = match op {
                            MetalOp::ConvGemm { .. } => "ConvGemm",
                            MetalOp::ConvDirect { .. } => "ConvDirect",
                            MetalOp::DepthwiseConv { .. } => "DepthwiseConv",
                            MetalOp::Binary { .. } => "Binary",
                            MetalOp::BroadcastBinary { .. } => "BroadcastBinary",
                            MetalOp::Unary { .. } => "Unary",
                            MetalOp::SiLU { .. } => "SiLU",
                            MetalOp::MatMul { .. } => "MatMul",
                            MetalOp::Softmax { .. } => "Softmax",
                            MetalOp::Concat { .. } => "Concat",
                            MetalOp::Split { .. } => "Split",
                            MetalOp::CastF32ToF16 { .. } => "CastF32ToF16",
                            MetalOp::CastF16ToF32 { .. } => "CastF16ToF32",
                            _ => "Other",
                        };
                        let sample = if is_f32 {
                            plan.inf.read_buffer_f32(buf, n_elems.min(8))
                        } else {
                            plan.inf.read_buffer_f16_as_f32(buf, n_elems.min(8))
                        };
                        eprintln!(
                            "  [NaN] Op {} {} '{}' n={}: {:?}",
                            idx, op_name, out_name, n_elems, sample
                        );
                        break;
                    }
                }
            }
        }

        if std::env::var("METAL_TIME").is_ok() {
            eprintln!(
                "  [metal] upload={:.2}ms encode={:.2}ms gpu={:.2}ms",
                t_upload.as_secs_f64() * 1000.0,
                (t_encode - t_upload).as_secs_f64() * 1000.0,
                (t_gpu - t_encode).as_secs_f64() * 1000.0
            );
        }

        // Debug: compare all Metal buffers against CPU reference (works for both MPS and non-MPS paths)
        if !plan.cpu_ref.is_empty() && std::env::var("METAL_COMPARE").is_ok() {
            #[inline]
            fn f16_to_f32(bits: u16) -> f32 {
                let sign = ((bits >> 15) & 1) as u32;
                let exp = ((bits >> 10) & 0x1F) as u32;
                let mant = (bits & 0x3FF) as u32;
                if exp == 0 {
                    if mant == 0 {
                        return if sign == 1 { -0.0 } else { 0.0 };
                    }
                    let mut m = mant;
                    let mut e = 1u32;
                    while m & 0x400 == 0 {
                        m <<= 1;
                        e += 1;
                    }
                    f32::from_bits((sign << 31) | ((127 - 15 + 1 - e) << 23) | ((m & 0x3FF) << 13))
                } else if exp == 31 {
                    if mant == 0 {
                        return if sign == 1 {
                            f32::NEG_INFINITY
                        } else {
                            f32::INFINITY
                        };
                    }
                    f32::NAN
                } else {
                    f32::from_bits((sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13))
                }
            }
            let mut errors: Vec<(String, f32, f32, usize)> = Vec::new();
            for (name, cpu_vals) in &plan.cpu_ref {
                if let Some(buf) = plan.bufs.get(name) {
                    let n = cpu_vals.len();
                    let is_f32 = plan.buf_f32.contains(name);
                    let min_bytes = if is_f32 { n * 4 } else { n * 2 };
                    if buf.length() as usize >= min_bytes {
                        let mut max_diff = 0.0f32;
                        if is_f32 {
                            let ptr = buf.contents() as *const f32;
                            let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
                            for i in 0..n {
                                let d = (slice[i] - cpu_vals[i]).abs();
                                if !d.is_nan() && d > max_diff {
                                    max_diff = d;
                                }
                            }
                        } else {
                            let ptr = buf.contents() as *const u16;
                            let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
                            for i in 0..n {
                                let d = (f16_to_f32(slice[i]) - cpu_vals[i]).abs();
                                if !d.is_nan() && d > max_diff {
                                    max_diff = d;
                                }
                            }
                        }
                        if max_diff > 0.5 {
                            let nans = if is_f32 {
                                let ptr = buf.contents() as *const f32;
                                let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
                                (0..n).filter(|&i| slice[i].is_nan()).count()
                            } else {
                                let ptr = buf.contents() as *const u16;
                                let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
                                (0..n).filter(|&i| f16_to_f32(slice[i]).is_nan()).count()
                            };
                            errors.push((name.clone(), max_diff, nans as f32, n));
                        }
                    }
                }
            }
            errors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            eprintln!("  [COMPARE] {} buffers with max_diff > 0.5:", errors.len());
            for (name, max_diff, nans, n) in errors.iter().take(40) {
                let shape = plan
                    .buf_shapes
                    .get(name)
                    .map(|s| format!("{:?}", s))
                    .unwrap_or_default();
                let nhwc_flag = plan.buf_nhwc.get(name).copied().unwrap_or(false);
                let is_f32 = plan.buf_f32.contains(name);
                let buf = plan.bufs.get(name).unwrap();
                let cpu_v = &plan.cpu_ref[name];
                let show = 6.min(*n);
                let mtl: Vec<f32> = if is_f32 {
                    let ptr = buf.contents() as *const f32;
                    let slice = unsafe { std::slice::from_raw_parts(ptr, *n) };
                    (0..show).map(|i| slice[i]).collect()
                } else {
                    let ptr = buf.contents() as *const u16;
                    let slice = unsafe { std::slice::from_raw_parts(ptr, *n) };
                    (0..show).map(|i| f16_to_f32(slice[i])).collect()
                };
                let cpu: Vec<f32> = cpu_v[..show].to_vec();
                eprintln!(
                    "    '{}' {} nhwc={} f32={} max_diff={:.2} nans={} mtl={:.3?} cpu={:.3?}",
                    name, shape, nhwc_flag, is_f32, max_diff, *nans as usize, mtl, cpu
                );
            }
        }

        // Download outputs inside autoreleasepool (read from _f32out buffers cast from f16)
        let mut result = HashMap::new();
        let debug_dl = std::env::var("METAL_DEBUG_DL").is_ok();
        for name in &plan.output_names {
            let f32out_name = format!("{}_f32out", name);
            let read_name = if plan.bufs.contains_key(&f32out_name) {
                &f32out_name
            } else {
                name
            };
            if let (Some(buf), Some(shape)) = (plan.bufs.get(read_name), plan.buf_shapes.get(name))
            {
                let n: usize = shape.iter().product();
                if debug_dl {
                    eprintln!(
                        "  [DL] output '{}' read_from='{}' buf_len={} n_elem={} shape={:?}",
                        name,
                        read_name,
                        buf.length(),
                        n,
                        shape
                    );
                    // Also check the raw f16 buffer
                    if let Some(f16_buf) = plan.bufs.get(name) {
                        let f16_ptr = f16_buf.contents() as *const u16;
                        let show = 8.min(n);
                        let raw_f16: Vec<u16> =
                            unsafe { std::slice::from_raw_parts(f16_ptr, show).to_vec() };
                        let f16_as_f32: Vec<f32> = raw_f16
                            .iter()
                            .map(|&bits| {
                                let sign = ((bits >> 15) & 1) as u32;
                                let exp = ((bits >> 10) & 0x1F) as u32;
                                let mant = (bits & 0x3FF) as u32;
                                if exp == 0 {
                                    0.0
                                } else if exp == 31 {
                                    if mant == 0 { f32::INFINITY } else { f32::NAN }
                                } else {
                                    let e = exp as i32 - 15 + 127;
                                    let f = (sign << 31) | ((e as u32) << 23) | (mant << 13);
                                    f32::from_bits(f)
                                }
                            })
                            .collect();
                        eprintln!(
                            "  [DL] f16 buf '{}' len={} raw_bits={:04X?} as_f32={:.4?}",
                            name,
                            f16_buf.length(),
                            &raw_f16[..show],
                            &f16_as_f32[..show]
                        );
                    }
                }
                let data = plan.inf.read_buffer_f32(buf, n);
                let is_nhwc = *plan.buf_nhwc.get(name).unwrap_or(&false);

                if is_nhwc && shape.len() == 4 {
                    // Convert NHWC → NCHW
                    let (nn, h, w, c) = (shape[0], shape[1], shape[2], shape[3]);
                    let mut nchw = vec![0.0f32; n];
                    for ni in 0..nn {
                        for hi in 0..h {
                            for wi in 0..w {
                                for ci in 0..c {
                                    nchw[((ni * c + ci) * h + hi) * w + wi] =
                                        data[((ni * h + hi) * w + wi) * c + ci];
                                }
                            }
                        }
                    }
                    let out_shape = vec![nn, c, h, w];
                    result.insert(name.clone(), Tensor::from_vec(out_shape, nchw).unwrap());
                } else {
                    result.insert(name.clone(), Tensor::from_vec(shape.clone(), data).unwrap());
                }
            }
        }

        if std::env::var("METAL_TIMING2").is_ok() {
            eprintln!(
                "  [TIMING2] total_with_readback={:.2}ms",
                t0.elapsed().as_secs_f64() * 1000.0,
            );
        }
        result
    });

    Ok(result)
}
