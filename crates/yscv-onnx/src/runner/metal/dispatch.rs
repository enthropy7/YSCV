use ::metal::*;
use std::collections::HashMap;

use super::types::*;

use yscv_kernels::metal_backend::metal_conv::{MetalEncoder, mps_gemm_f16_encode};

/// Dispatch ops with MPS GEMM for conv layers.
/// Segments ops into compute-encoder blocks separated by MPS calls.
pub(crate) fn dispatch_with_mps(plan: &MetalPlan, cmd: &CommandBufferRef) {
    let bufs = &plan.bufs;
    let gb = |name: &str| -> &Buffer {
        bufs.get(name).unwrap_or_else(|| {
            // Internal invariant: all buffers should be allocated before dispatch.
            // If this fails, it's a bug in plan construction, not user input.
            unreachable!("Metal plan: buffer '{name}' not found (have {} buffers). This is a bug in MetalPlan construction.", bufs.len())
        })
    };

    let mut i = 0;
    while i < plan.ops.len() {
        if let MetalOp::MpsConv {
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
        } = &plan.ops[i]
        {
            // Step 1: im2col for 3×3+ (needs its own encoder)
            let gemm_a: &str = if let Some(col_name) = im2col_buf {
                let enc_raw = cmd.new_compute_command_encoder();
                let enc = MetalEncoder::new(enc_raw, &plan.inf);
                enc.im2col_f16(
                    gb(input),
                    gb(col_name),
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
                enc_raw.end_encoding();
                col_name.as_str()
            } else {
                input.as_str()
            };

            // Step 2: MPS GEMM (creates its own internal encoder)
            mps_gemm_f16_encode(
                &plan.inf.device,
                cmd,
                gb(gemm_a),
                gb(weight),
                gb(output),
                *m,
                *n,
                *k,
                1.0,
                0.0,
                false,
                false,
            );

            // Step 3: bias + activation
            let enc_raw = cmd.new_compute_command_encoder();
            let enc = MetalEncoder::new(enc_raw, &plan.inf);
            enc.bias_act_f16(gb(output), gb(bias), *m * *n, *n, *act);
            enc_raw.end_encoding();

            i += 1;
        } else {
            // Collect consecutive non-MPS ops into one encoder
            let enc_raw = cmd.new_compute_command_encoder();
            let enc = MetalEncoder::new(enc_raw, &plan.inf);
            while i < plan.ops.len() && !matches!(&plan.ops[i], MetalOp::MpsConv { .. }) {
                dispatch_compute_op(plan, &enc, &plan.ops[i], &plan.bufs);
                i += 1;
            }
            enc_raw.end_encoding();
        }
    }

    cmd.commit();
    cmd.wait_until_completed();
}

/// Dispatch a single non-MPS compute op through the encoder.
pub(crate) fn dispatch_compute_op(
    plan: &MetalPlan,
    enc: &MetalEncoder,
    op: &MetalOp,
    bufs: &HashMap<String, Buffer>,
) {
    let gb = |name: &str| -> &Buffer {
        bufs.get(name).unwrap_or_else(|| {
            unreachable!(
                "Metal plan: buffer '{name}' not found. This is a bug in MetalPlan construction."
            );
        })
    };
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
                let use_large_1x1 = std::env::var("METAL_LARGE_1X1").is_ok();
                let has_res = residual.is_some();
                if params.n_out <= 32 && !has_res {
                    enc.conv_gemm_small_f16io(gb(input), gb(weight), gb(bias), gb(output), params);
                // NOTE: conv1x1_simd_large_f16io disabled — regresses on current workloads.
                // Uncomment when occupancy tuning is done for large 1×1 convs.
                // } else if is_1x1 && use_large_1x1 && !has_res && params.m >= 256 && params.n_out >= 64 {
                //     enc.conv1x1_simd_large_f16io(gb(input), gb(weight), gb(bias), gb(output), params);
                } else if is_1x1 && has_res {
                    enc.conv1x1_simd_res_f16io(
                        gb(input),
                        gb(weight),
                        gb(bias),
                        gb(output),
                        gb(residual.as_ref().unwrap()),
                        params,
                    );
                } else if is_1x1 {
                    enc.conv1x1_simd_f16io(gb(input), gb(weight), gb(bias), gb(output), params);
                } else if has_res {
                    enc.conv3x3_simd_res_f16io(
                        gb(input),
                        gb(weight),
                        gb(bias),
                        gb(output),
                        gb(residual.as_ref().unwrap()),
                        params,
                    );
                } else {
                    enc.conv3x3_simd_f16io(gb(input), gb(weight), gb(bias), gb(output), params);
                }
            } else {
                enc.conv_gemm(gb(input), gb(weight), gb(bias), gb(output), params);
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
                enc.conv_direct_f16io(gb(input), gb(weight), gb(bias), gb(output), params);
            } else {
                enc.conv_direct(gb(input), gb(weight), gb(bias), gb(output), params);
            }
        }
        MetalOp::DepthwiseConv {
            input,
            weight,
            bias,
            output,
            params,
        } => {
            enc.depthwise_conv_f16io(gb(input), gb(weight), gb(bias), gb(output), params);
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
                enc.binary_f16(gb(a), gb(b), gb(out), *n, *op);
            } else {
                enc.binary(gb(a), gb(b), gb(out), *n, *op);
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
                enc.broadcast_binary_f16(gb(a), gb(b), gb(out), *n, *broadcast_dim, *op);
            } else {
                enc.broadcast_binary(gb(a), gb(b), gb(out), *n, *broadcast_dim, *op);
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
                enc.unary_f16(gb(input), gb(out), *n, *op);
            } else {
                enc.unary(gb(input), gb(out), *n, *op);
            }
        }
        MetalOp::SiLU { input, out, n, f16 } => {
            if *f16 {
                enc.silu_f16(gb(input), gb(out), *n);
            } else {
                enc.silu(gb(input), gb(out), *n);
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
            let input_bufs: Vec<&Buffer> = inputs.iter().map(|name| gb(name)).collect();
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
                // Use vectorized v4 kernel: grid is (out_c/4, spatial)
                enc.enc
                    .set_compute_pipeline_state(&plan.inf.concat_channels_f16v4);
            } else {
                enc.enc
                    .set_compute_pipeline_state(&plan.inf.concat_channels);
            }
            for (i, buf) in input_bufs.iter().enumerate() {
                enc.enc.set_buffer(i as u64, Some(buf), 0);
            }
            enc.enc.set_buffer(4, Some(gb(out)), 0);
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
                enc.split_channels_f16(gb(input), gb(out), *spatial, *in_c, *out_c, *offset_c);
            } else {
                enc.split_channels(gb(input), gb(out), *spatial, *in_c, *out_c, *offset_c);
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
            enc.enc.set_buffer(0, Some(gb(input)), 0);
            enc.enc.set_buffer(1, Some(gb(&outputs[0])), 0);
            enc.enc
                .set_buffer(2, Some(gb(outputs.get(1).unwrap_or(&outputs[0]))), 0);
            enc.enc
                .set_buffer(3, Some(gb(outputs.get(2).unwrap_or(&outputs[0]))), 0);
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
                    gb(input),
                    gb(out),
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
                    gb(input),
                    gb(out),
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
                    gb(input),
                    gb(out),
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
                    gb(input),
                    gb(out),
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
                enc.softmax_f16(gb(input), gb(out), *outer, *dim);
            } else {
                enc.softmax(gb(input), gb(out), *outer, *dim);
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
                enc.transpose_2d_f16(gb(input), gb(out), *rows, *cols);
            } else {
                enc.transpose_2d(gb(input), gb(out), *rows, *cols);
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
                enc.permute_0213_f16(gb(input), gb(out), *d0, *d1, *d2, *d3);
            } else {
                enc.permute_0213(gb(input), gb(out), *d0, *d1, *d2, *d3);
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
            struct SliceP {
                n: u32,
                src_offset: u32,
            }
            let p = SliceP {
                n: *n,
                src_offset: *src_offset,
            };
            if *f16 {
                enc.enc.set_compute_pipeline_state(&plan.inf.slice_copy_f16);
            } else {
                enc.enc.set_compute_pipeline_state(&plan.inf.slice_copy);
            }
            enc.enc.set_buffer(0, Some(gb(input)), 0);
            enc.enc.set_buffer(1, Some(gb(out)), 0);
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
        MetalOp::CpuReshape {
            input,
            out,
            n,
            nhwc_to_nchw,
            nchw_to_nhwc,
            f16,
        } => {
            if std::env::var("METAL_PERM_DBG").is_ok() {
                let kind = if nhwc_to_nchw.is_some() {
                    "NHWC→NCHW"
                } else if nchw_to_nhwc.is_some() {
                    "NCHW→NHWC"
                } else {
                    "copy"
                };
                let dims = nhwc_to_nchw
                    .map(|(a, b, c, d)| format!("[{},{},{},{}]", a, b, c, d))
                    .or_else(|| {
                        nchw_to_nhwc.map(|(a, b, c, d)| format!("[{},{},{},{}]", a, b, c, d))
                    })
                    .unwrap_or_else(|| format!("n={}", n));
                eprintln!("  CpuReshape: {} {} '{}' → '{}'", kind, dims, input, out);
            }
            if let Some((nn, h, w, c)) = nhwc_to_nchw {
                if *f16 {
                    enc.permute_nhwc_to_nchw_f16(gb(input), gb(out), *nn, *h, *w, *c);
                } else {
                    enc.permute_nhwc_to_nchw(gb(input), gb(out), *nn, *h, *w, *c);
                }
            } else if let Some((nn, cc, hh, ww)) = nchw_to_nhwc {
                // Tuple is (N, C, H, W) but encoder expects (n, h, w, c)
                if *f16 {
                    enc.permute_nchw_to_nhwc_f16(gb(input), gb(out), *nn, *hh, *ww, *cc);
                } else {
                    enc.permute_nchw_to_nhwc(gb(input), gb(out), *nn, *hh, *ww, *cc);
                }
            } else {
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
                enc.enc.set_buffer(0, Some(gb(input)), 0);
                enc.enc.set_buffer(1, Some(gb(out)), 0);
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
            let out_buf = gb(out);
            let mut dst_offset_bytes: u64 = 0;
            let bytes_per_elem: u64 = if *f16 { 2 } else { 4 };
            for (input_name, &size) in inputs.iter().zip(sizes.iter()) {
                if size == 0 {
                    continue;
                }
                #[repr(C)]
                struct SliceP {
                    n: u32,
                    src_offset: u32,
                }
                let p = SliceP {
                    n: size,
                    src_offset: 0,
                };
                if *f16 {
                    enc.enc.set_compute_pipeline_state(&plan.inf.slice_copy_f16);
                } else {
                    enc.enc.set_compute_pipeline_state(&plan.inf.slice_copy);
                }
                enc.enc.set_buffer(0, Some(gb(input_name)), 0);
                enc.enc.set_buffer(1, Some(out_buf), dst_offset_bytes);
                enc.enc.set_bytes(
                    2,
                    std::mem::size_of::<SliceP>() as u64,
                    &p as *const SliceP as *const _,
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
                enc.matmul_f16io(gb(a), gb(b), gb(out), *m, *n, *k);
            } else {
                enc.matmul(gb(a), gb(b), gb(out), *m, *n, *k);
            }
        }
        MetalOp::CastF32ToF16 { input, out, n } => {
            enc.cast_f32_to_f16(gb(input), gb(out), *n);
        }
        MetalOp::CastF16ToF32 { input, out, n } => {
            enc.cast_f16_to_f32(gb(input), gb(out), *n);
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
            enc.winograd4x4_input_transform_f16(gb(input), gb(transformed_input), wino_params);
            enc.winograd4x4_batched_gemm_simd_f16io(
                gb(transformed_input),
                gb(weight),
                gb(gemm_output),
                wino_params.n_tiles,
                *oc,
                *ic,
            );
            if let Some(res) = residual {
                enc.winograd4x4_output_transform_residual_f16(
                    gb(gemm_output),
                    gb(bias),
                    gb(output),
                    wino_params,
                    gb(res),
                );
            } else {
                enc.winograd4x4_output_transform_f16(
                    gb(gemm_output),
                    gb(bias),
                    gb(output),
                    wino_params,
                );
            }
        }
        MetalOp::NhwcToFlatConcat {
            inputs,
            out,
            c,
            hw,
            total_spatial,
        } => {
            #[repr(C)]
            struct NhwcToFlatP {
                c: u32,
                n_inputs: u32,
                h0: u32,
                w0: u32,
                h1: u32,
                w1: u32,
                h2: u32,
                w2: u32,
                total_spatial: u32,
            }
            let p = NhwcToFlatP {
                c: *c,
                n_inputs: inputs.len() as u32,
                h0: hw.first().map_or(0, |x| x.0),
                w0: hw.first().map_or(0, |x| x.1),
                h1: hw.get(1).map_or(0, |x| x.0),
                w1: hw.get(1).map_or(0, |x| x.1),
                h2: hw.get(2).map_or(0, |x| x.0),
                w2: hw.get(2).map_or(0, |x| x.1),
                total_spatial: *total_spatial,
            };
            enc.enc
                .set_compute_pipeline_state(&plan.inf.nhwc_to_flat_concat_f16);
            for (i, name) in inputs.iter().enumerate() {
                enc.enc.set_buffer(i as u64, Some(gb(name)), 0);
            }
            // Fill unused input slots with first input to avoid null
            for i in inputs.len()..3 {
                enc.enc.set_buffer(i as u64, Some(gb(&inputs[0])), 0);
            }
            enc.enc.set_buffer(3, Some(gb(out)), 0);
            enc.enc.set_bytes(
                4,
                std::mem::size_of::<NhwcToFlatP>() as u64,
                &p as *const NhwcToFlatP as *const _,
            );
            let tg_x = 16u64.min(*total_spatial as u64);
            let tg_y = (256u64 / tg_x).min(*c as u64).max(1);
            enc.enc.dispatch_thread_groups(
                MTLSize {
                    width: (*total_spatial as u64).div_ceil(tg_x),
                    height: (*c as u64).div_ceil(tg_y),
                    depth: 1,
                },
                MTLSize {
                    width: tg_x,
                    height: tg_y,
                    depth: 1,
                },
            );
        }
        MetalOp::ChannelScatter {
            input,
            out,
            spatial,
            src_c,
            dst_c,
            dst_offset,
        } => {
            let p = [*spatial, *src_c, *dst_c, *dst_offset];
            enc.enc
                .set_compute_pipeline_state(&plan.inf.channel_scatter_f16);
            enc.enc.set_buffer(0, Some(gb(input)), 0);
            enc.enc.set_buffer(1, Some(gb(out)), 0);
            enc.enc.set_bytes(2, 16, p.as_ptr() as *const _);
            let tg_x = 16u64.min(*src_c as u64);
            let tg_y = (256u64 / tg_x).min(*spatial as u64).max(1);
            enc.enc.dispatch_thread_groups(
                MTLSize {
                    width: (*src_c as u64).div_ceil(tg_x),
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
        MetalOp::MpsConv { .. } => unreachable!(),
    }
}
