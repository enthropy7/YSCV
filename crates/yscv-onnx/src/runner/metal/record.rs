use ::metal::*;
use rustc_hash::FxHashMap;
use rustc_hash::FxHashSet;

use super::compile::{cpu_fallback, ensure_nhwc_metal, ensure_on_metal};
use super::types::*;

use crate::runner::conv::oihw_to_khwc_cout;
use crate::runner::{TensorEnv, get_attr_int, get_attr_ints, get_tensor};

use yscv_kernels::metal_backend::metal_conv::{
    ConvParams, MetalInference, WinogradParams, winograd4x4_transform_weights_f16,
};

use crate::error::OnnxError;
use crate::loader::OnnxNode;

// ── f32 attention chain helpers ──

/// Resolve an input buffer for the attention f32 chain.
/// - If `is_attn` and the buffer is f16, insert CastF16ToF32 and return the f32 name.
/// - If `!is_attn` and the buffer is f32, insert CastF32ToF16 and return the f16 name.
/// - Otherwise, return the name unchanged.
fn resolve_attn_input(
    inf: &MetalInference,
    name: &str,
    is_attn: bool,
    bufs: &mut FxHashMap<String, Buffer>,
    shapes: &mut FxHashMap<String, Vec<usize>>,
    nhwc_map: &mut FxHashMap<String, bool>,
    ops: &mut Vec<MetalOp>,
    f32_bufs: &mut FxHashSet<String>,
) -> String {
    if name.is_empty() || !bufs.contains_key(name) {
        return name.to_string();
    }
    let is_f32 = f32_bufs.contains(name);

    if is_attn && !is_f32 {
        // Need f32 version of this f16 buffer
        let f32_name = format!("{}_attnf32", name);
        if bufs.contains_key(&f32_name) {
            return f32_name;
        }
        let n: usize = shapes.get(name).map(|s| s.iter().product()).unwrap_or(0);
        if n > 0 {
            bufs.insert(f32_name.clone(), inf.output_buffer(n));
            shapes.insert(
                f32_name.clone(),
                shapes.get(name).cloned().unwrap_or_default(),
            );
            nhwc_map.insert(f32_name.clone(), *nhwc_map.get(name).unwrap_or(&false));
            f32_bufs.insert(f32_name.clone());
            ops.push(MetalOp::CastF16ToF32 {
                input: name.to_string(),
                out: f32_name.clone(),
                n: n as u32,
            });
        }
        f32_name
    } else if !is_attn && is_f32 {
        // Need f16 version of this f32 buffer
        let f16_name = format!("{}_attnf16", name);
        if bufs.contains_key(&f16_name) {
            return f16_name;
        }
        let n: usize = shapes.get(name).map(|s| s.iter().product()).unwrap_or(0);
        if n > 0 {
            bufs.insert(f16_name.clone(), inf.output_buffer_f16(n));
            shapes.insert(
                f16_name.clone(),
                shapes.get(name).cloned().unwrap_or_default(),
            );
            nhwc_map.insert(f16_name.clone(), *nhwc_map.get(name).unwrap_or(&false));
            ops.push(MetalOp::CastF32ToF16 {
                input: name.to_string(),
                out: f16_name.clone(),
                n: n as u32,
            });
        }
        f16_name
    } else {
        name.to_string()
    }
}

// ── Record conv dispatch ──

pub(crate) fn record_conv(
    inf: &MetalInference,
    node: &OnnxNode,
    env: &TensorEnv,
    bufs: &mut FxHashMap<String, Buffer>,
    shapes: &mut FxHashMap<String, Vec<usize>>,
    nhwc: &mut FxHashMap<String, bool>,
    ops: &mut Vec<MetalOp>,
    act: u32,
) -> Result<(), OnnxError> {
    let weight = get_tensor(env, &node.name, &node.inputs[1])?;
    let bias = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
        Some(get_tensor(env, &node.name, &node.inputs[2])?)
    } else {
        None
    };

    let strides = get_attr_ints(node, Attr::Strides).unwrap_or_else(|| vec![1, 1]);
    let pads = get_attr_ints(node, Attr::Pads).unwrap_or_else(|| vec![0, 0, 0, 0]);
    let group = get_attr_int(node, Attr::Group).unwrap_or(1) as usize;

    let w_shape = weight.shape();
    let is_khwc = env.is_khwc_weight(&node.inputs[1]);
    let (o_ch, _i_per_g, kh, kw) = if is_khwc {
        (w_shape[3], w_shape[2], w_shape[0], w_shape[1])
    } else {
        (w_shape[0], w_shape[1], w_shape[2], w_shape[3])
    };

    let sh = strides[0] as usize;
    let sw = strides[1] as usize;

    // Ensure input is NHWC
    let input_name = &node.inputs[0];
    ensure_on_metal(inf, input_name, env, bufs, shapes, nhwc);
    let nhwc_input = ensure_nhwc_metal(inf, input_name, bufs, shapes, nhwc, ops);

    let in_shape = shapes.get(&nhwc_input).cloned().unwrap_or_default();
    if in_shape.len() != 4 {
        return Ok(());
    }
    let (n, ih, iw, ic) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
    let oh = (ih + pads[0] as usize + pads[2] as usize - kh) / sh + 1;
    let ow = (iw + pads[1] as usize + pads[3] as usize - kw) / sw + 1;
    let m = n * oh * ow;
    let col_k = kh * kw * ic;

    // Upload weight in KHWC format as f16 (half bandwidth on GPU)
    let w_key = format!("__mtl_w_{}", node.inputs[1]);
    if !bufs.contains_key(&w_key) {
        let w_khwc = if is_khwc {
            weight.clone()
        } else {
            oihw_to_khwc_cout(weight)?
        };
        bufs.insert(w_key.clone(), inf.buffer_from_f32_as_f16(w_khwc.data()));
        shapes.insert(w_key.clone(), w_khwc.shape().to_vec());
    }

    let b_key = format!("__mtl_b_{}", node.inputs[1]);
    if !bufs.contains_key(&b_key) {
        let bias_data = bias
            .map(|b| b.data().to_vec())
            .unwrap_or_else(|| vec![0.0f32; o_ch]);
        bufs.insert(b_key.clone(), inf.buffer_from_f32(&bias_data));
        shapes.insert(b_key.clone(), vec![o_ch]);
    }

    // Allocate output
    let out_n = m * o_ch;
    let out_name = &node.outputs[0];
    bufs.insert(out_name.clone(), inf.output_buffer_f16(out_n));
    shapes.insert(out_name.clone(), vec![n, oh, ow, o_ch]);
    nhwc.insert(out_name.clone(), true);

    let params = ConvParams {
        m: m as u32,
        n_out: o_ch as u32,
        k: col_k as u32,
        act,
        ih: ih as u32,
        iw: iw as u32,
        ic: ic as u32,
        oh: oh as u32,
        ow: ow as u32,
        kh: kh as u32,
        kw: kw as u32,
        sh: sh as u32,
        sw: sw as u32,
        pad_h: pads[0] as u32,
        pad_w: pads[1] as u32,
        batch: n as u32,
        out_stride: o_ch as u32,
        out_offset: 0,
        in_stride: ic as u32,
        in_offset: 0,
        has_residual: 0,
        _pad: 0,
    };

    // Depthwise conv: group == ic == o_ch, each output channel uses 1 input channel.
    // The GEMM kernel can't handle this (weight buffer layout mismatch), so use
    // dedicated depthwise kernel.
    if group > 1 && group == ic && group == o_ch {
        ops.push(MetalOp::DepthwiseConv {
            input: nhwc_input.clone(),
            weight: w_key,
            bias: b_key,
            output: out_name.clone(),
            params,
        });
    } else if sh == 1
        && sw == 1
        && kh == 3
        && kw == 3
        && group == 1
        && std::env::var("METAL_NO_WINO").is_err()
    {
        // Winograd FIRST for 3×3 stride=1: 4× FLOP reduction + stays in unified
        // single-encoder path (no encoder transitions). Always preferred over MPS
        // for 3×3 because Winograd's FLOP reduction outweighs MPS GEMM speedup,
        // and avoiding segmented dispatch eliminates ~0.6ms overhead per MPS op.
        let tile_h = oh.div_ceil(4);
        let tile_w = ow.div_ceil(4);
        let n_tiles = n * tile_h * tile_w;

        // Pre-transform weights: [9*ic, oc] → [36, ic, oc] as f16
        let wino_w_key = format!("__mtl_wino4_w_{}", node.inputs[1]);
        if !bufs.contains_key(&wino_w_key) {
            let w_khwc = if is_khwc {
                weight.clone()
            } else {
                oihw_to_khwc_cout(weight)?
            };
            let wdata = w_khwc.data();
            let expected = 9 * ic * o_ch;
            if wdata.len() != expected {
                eprintln!(
                    "  [winograd4] weight mismatch: {} '{}' w_shape={:?} is_khwc={} ic={} oc={} group={} data.len={} expected={}",
                    node.op_type,
                    node.name,
                    w_khwc.shape(),
                    is_khwc,
                    ic,
                    o_ch,
                    group,
                    wdata.len(),
                    expected
                );
                ops.push(MetalOp::ConvGemm {
                    input: nhwc_input.clone(),
                    weight: w_key,
                    bias: b_key,
                    output: out_name.clone(),
                    params,
                    f16io: true,
                    residual: None,
                });
                return Ok(());
            }
            let transformed = winograd4x4_transform_weights_f16(wdata, ic, o_ch);
            let bytes = unsafe {
                std::slice::from_raw_parts(transformed.as_ptr() as *const u8, transformed.len() * 2)
            };
            bufs.insert(
                wino_w_key.clone(),
                inf.device.new_buffer_with_data(
                    bytes.as_ptr() as *const _,
                    bytes.len() as u64,
                    MTLResourceOptions::StorageModeShared,
                ),
            );
        }

        // Double-buffered Winograd temp buffers: alternate between set 0 and set 1
        // to enable GPU pipelining between consecutive Winograd convolutions.
        let ti_size = 36 * n_tiles * ic;
        let go_size = 36 * n_tiles * o_ch;
        let wino_count = ops
            .iter()
            .filter(|op| matches!(op, MetalOp::ConvWinograd { .. }))
            .count();
        let wino_set = wino_count % 2;
        let ti_key = format!("__mtl_wino4_shared_ti_{}", wino_set);
        let go_key = format!("__mtl_wino4_shared_go_{}", wino_set);
        // Grow shared buffers to max needed size
        if let Some(existing) = bufs.get(&ti_key) {
            let existing_len = existing.length() as usize / 2; // f16 = 2 bytes
            if ti_size > existing_len {
                bufs.insert(ti_key.clone(), inf.output_buffer_f16(ti_size));
            }
        } else {
            bufs.insert(ti_key.clone(), inf.output_buffer_f16(ti_size));
        }
        if let Some(existing) = bufs.get(&go_key) {
            let existing_len = existing.length() as usize / 2;
            if go_size > existing_len {
                bufs.insert(go_key.clone(), inf.output_buffer_f16(go_size));
            }
        } else {
            bufs.insert(go_key.clone(), inf.output_buffer_f16(go_size));
        }

        let wino_params = WinogradParams {
            batch: n as u32,
            ih: ih as u32,
            iw: iw as u32,
            ic: ic as u32,
            oh: oh as u32,
            ow: ow as u32,
            oc: o_ch as u32,
            pad_h: pads[0] as u32,
            pad_w: pads[1] as u32,
            tile_h: tile_h as u32,
            tile_w: tile_w as u32,
            n_tiles: n_tiles as u32,
            act,
            out_stride: o_ch as u32,
            out_offset: 0,
            in_stride: ic as u32,
            in_offset: 0,
        };
        ops.push(MetalOp::ConvWinograd {
            input: nhwc_input.clone(),
            weight: wino_w_key,
            bias: b_key,
            output: out_name.clone(),
            transformed_input: ti_key,
            gemm_output: go_key,
            wino_params,
            ic: ic as u32,
            oc: o_ch as u32,
            residual: None,
        });
    } else {
        // Non-depthwise, non-Winograd conv: ConvGemm compute shaders by default.
        // MPS GEMM is opt-in via METAL_MPS=1 (has known precision issues with
        // non-aligned column counts, e.g. K=27 for first 3×3 stride=2 conv).
        let use_mps = std::env::var("METAL_MPS").is_ok();
        let is_1x1 = kh == 1 && kw == 1 && sh == 1 && sw == 1 && pads[0] == 0 && pads[1] == 0;
        let use_direct = (col_k * o_ch) < 1024;

        if use_mps && !use_direct {
            // MPS-accelerated conv: im2col (for non-1×1) → MPS GEMM → bias+act
            let im2col_buf = if is_1x1 {
                None
            } else {
                let im2col_name = format!("__mtl_im2col_{}", out_name);
                bufs.insert(im2col_name.clone(), inf.output_buffer_f16(m * col_k));
                Some(im2col_name)
            };
            ops.push(MetalOp::MpsConv {
                input: nhwc_input.clone(),
                weight: w_key,
                bias: b_key,
                output: out_name.clone(),
                im2col_buf,
                m: m as u32,
                n: o_ch as u32,
                k: col_k as u32,
                act,
                batch: n as u32,
                ih: ih as u32,
                iw: iw as u32,
                ic: ic as u32,
                oh: oh as u32,
                ow: ow as u32,
                kh: kh as u32,
                kw: kw as u32,
                sh: sh as u32,
                sw: sw as u32,
                pad_h: pads[0] as u32,
                pad_w: pads[1] as u32,
            });
        } else if use_direct {
            ops.push(MetalOp::ConvDirect {
                input: nhwc_input.clone(),
                weight: w_key,
                bias: b_key,
                output: out_name.clone(),
                params,
                f16io: true,
            });
        } else {
            ops.push(MetalOp::ConvGemm {
                input: nhwc_input.clone(),
                weight: w_key,
                bias: b_key,
                output: out_name.clone(),
                params,
                f16io: true,
                residual: None,
            });
        }
    }

    Ok(())
}

// ── Record a single node dispatch ──

pub(crate) fn record_node(
    inf: &MetalInference,
    node: &OnnxNode,
    env: &TensorEnv,
    cpu_data: &FxHashMap<String, Vec<f32>>,
    cpu_shapes: &FxHashMap<String, Vec<usize>>,
    bufs: &mut FxHashMap<String, Buffer>,
    shapes: &mut FxHashMap<String, Vec<usize>>,
    nhwc: &mut FxHashMap<String, bool>,
    ops: &mut Vec<MetalOp>,
    f32_bufs: &mut FxHashSet<String>,
) -> Result<(), OnnxError> {
    // Ensure all inputs are available
    for input_name in &node.inputs {
        if input_name.is_empty() {
            continue;
        }
        ensure_on_metal(inf, input_name, env, bufs, shapes, nhwc);
    }

    // f32 precision chain: only force f32 for MatMul and Softmax in attention blocks.
    // Other ops propagate f32 from inputs (Transpose/Reshape) or cast back to f16 (Binary/Unary).
    let is_attn_matmul = node.name.contains("/attn/") && node.op_type == "MatMul";
    let is_attn_softmax = node.name.contains("/attn/") && node.op_type == "Softmax";

    match node.op_type.as_str() {
        "Conv" => {
            record_conv(inf, node, env, bufs, shapes, nhwc, ops, 0)?;
        }

        "Add" | "Sub" | "Mul" | "Div" => {
            let op = match node.op_type.as_str() {
                "Add" => 0u32,
                "Sub" => 1,
                "Mul" => 2,
                "Div" => 3,
                _ => 0,
            };

            // Check f32 status of raw inputs BEFORE resolving
            let a_is_f32 = f32_bufs.contains(node.inputs[0].as_str());
            let b_is_f32 = f32_bufs.contains(node.inputs[1].as_str());

            // Peek at sizes to decide broadcast vs same-size
            let a_shape_raw = shapes.get(&node.inputs[0]).cloned().unwrap_or_default();
            let b_shape_raw = shapes.get(&node.inputs[1]).cloned().unwrap_or_default();
            let a_n_raw: usize = a_shape_raw.iter().product();
            let b_n_raw: usize = b_shape_raw.iter().product();

            // For broadcast ops where the big tensor is f32 (attention scale multiply),
            // keep f32. For same-size ops, cast f32→f16.
            let use_f32 = if a_n_raw != b_n_raw {
                // Broadcast: keep f32 if the big tensor is f32
                if a_n_raw > b_n_raw {
                    a_is_f32
                } else {
                    b_is_f32
                }
            } else {
                // Same-size: keep f32 only if BOTH are f32
                a_is_f32 && b_is_f32
            };

            let mut actual_a = resolve_attn_input(
                inf,
                &node.inputs[0],
                use_f32,
                bufs,
                shapes,
                nhwc,
                ops,
                f32_bufs,
            );
            let mut actual_b = resolve_attn_input(
                inf,
                &node.inputs[1],
                use_f32,
                bufs,
                shapes,
                nhwc,
                ops,
                f32_bufs,
            );
            let mut a_shape = shapes.get(&actual_a).cloned().unwrap_or_default();
            let mut b_shape = shapes.get(&actual_b).cloned().unwrap_or_default();
            let mut a_n: usize = a_shape.iter().product();
            let mut b_n: usize = b_shape.iter().product();

            if a_n == 0 || b_n == 0 {
                // CPU fallback for empty tensors
                return Ok(());
            }

            // Fix NHWC/NCHW layout mismatch: if one input is NHWC and the other is NCHW
            // with the same element count, convert the NCHW one to NHWC.
            let a_nhwc_flag = *nhwc.get(actual_a.as_str()).unwrap_or(&false);
            let b_nhwc_flag = *nhwc.get(actual_b.as_str()).unwrap_or(&false);
            if a_n == b_n && a_nhwc_flag != b_nhwc_flag && a_shape.len() == 4 && b_shape.len() == 4
            {
                // Convert the NCHW input to NHWC
                let (nchw_name, nchw_shape, is_nchw_a) = if !a_nhwc_flag {
                    (actual_a.clone(), a_shape.clone(), true)
                } else {
                    (actual_b.clone(), b_shape.clone(), false)
                };
                let (nn, cc, hh, ww) = (nchw_shape[0], nchw_shape[1], nchw_shape[2], nchw_shape[3]);
                let permuted_name = format!("{}_nhwc", nchw_name);
                let is_src_f32 = f32_bufs.contains(&nchw_name);
                if is_src_f32 {
                    bufs.insert(permuted_name.clone(), inf.output_buffer(a_n));
                    f32_bufs.insert(permuted_name.clone());
                } else {
                    bufs.insert(permuted_name.clone(), inf.output_buffer_f16(a_n));
                }
                shapes.insert(permuted_name.clone(), vec![nn, hh, ww, cc]);
                nhwc.insert(permuted_name.clone(), true);
                ops.push(MetalOp::CpuReshape {
                    input: nchw_name,
                    out: permuted_name.clone(),
                    n: a_n as u32,
                    nhwc_to_nchw: None,
                    nchw_to_nhwc: Some((nn as u32, cc as u32, hh as u32, ww as u32)),
                    f16: !is_src_f32,
                });
                if is_nchw_a {
                    actual_a = permuted_name;
                    a_shape = shapes.get(&actual_a).cloned().unwrap_or_default();
                    a_n = a_shape.iter().product();
                } else {
                    actual_b = permuted_name;
                    b_shape = shapes.get(&actual_b).cloned().unwrap_or_default();
                    b_n = b_shape.iter().product();
                }
            }

            let out_name = &node.outputs[0];
            let out_n = a_n.max(b_n);
            if use_f32 {
                bufs.insert(out_name.clone(), inf.output_buffer(out_n));
                f32_bufs.insert(out_name.clone());
            } else {
                bufs.insert(out_name.clone(), inf.output_buffer_f16(out_n));
            }

            if a_n == b_n {
                let out_shape = if a_shape.len() >= b_shape.len() {
                    a_shape.clone()
                } else {
                    b_shape.clone()
                };
                shapes.insert(out_name.clone(), out_shape);
                let a_nhwc_out = *nhwc.get(actual_a.as_str()).unwrap_or(&false);
                nhwc.insert(out_name.clone(), a_nhwc_out);
                ops.push(MetalOp::Binary {
                    a: actual_a,
                    b: actual_b,
                    out: out_name.clone(),
                    n: out_n as u32,
                    op,
                    f16: !use_f32,
                });
            } else {
                // Broadcast: smaller tensor broadcasts over larger
                let (big, _small, big_name, _small_name) = if a_n > b_n {
                    (&a_shape, &b_shape, &actual_a, &actual_b)
                } else {
                    (&b_shape, &a_shape, &actual_b, &actual_a)
                };
                let broadcast_dim = if a_n > b_n { b_n } else { a_n };
                shapes.insert(out_name.clone(), big.clone());
                let big_nhwc = *nhwc.get(big_name.as_str()).unwrap_or(&false);
                nhwc.insert(out_name.clone(), big_nhwc);

                // For broadcast, a is the big tensor, b is the small
                if a_n > b_n {
                    ops.push(MetalOp::BroadcastBinary {
                        a: actual_a,
                        b: actual_b,
                        out: out_name.clone(),
                        n: out_n as u32,
                        broadcast_dim: broadcast_dim as u32,
                        op,
                        f16: !use_f32,
                    });
                } else {
                    // Swap operands: big tensor must be 'a' for broadcast kernel.
                    let adjusted_op = match op {
                        1 => 6, // sub → rsub
                        3 => 7, // div → rdiv
                        _ => op,
                    };
                    ops.push(MetalOp::BroadcastBinary {
                        a: actual_b,
                        b: actual_a,
                        out: out_name.clone(),
                        n: out_n as u32,
                        broadcast_dim: broadcast_dim as u32,
                        op: adjusted_op,
                        f16: !use_f32,
                    });
                }
            }
        }

        "Sigmoid" => {
            // Cast any f32 inputs back to f16
            let actual_in = resolve_attn_input(
                inf,
                &node.inputs[0],
                false,
                bufs,
                shapes,
                nhwc,
                ops,
                f32_bufs,
            );
            let shape = shapes.get(&actual_in).cloned().unwrap_or_default();
            let n: usize = shape.iter().product();
            let out_name = &node.outputs[0];
            bufs.insert(out_name.clone(), inf.output_buffer_f16(n));
            shapes.insert(out_name.clone(), shape);
            let in_nhwc = *nhwc.get(actual_in.as_str()).unwrap_or(&false);
            nhwc.insert(out_name.clone(), in_nhwc);
            ops.push(MetalOp::Unary {
                input: actual_in,
                out: out_name.clone(),
                n: n as u32,
                op: 1, // sigmoid
                f16: true,
            });
        }

        "Relu" => {
            let actual_in = resolve_attn_input(
                inf,
                &node.inputs[0],
                false,
                bufs,
                shapes,
                nhwc,
                ops,
                f32_bufs,
            );
            let shape = shapes.get(&actual_in).cloned().unwrap_or_default();
            let n: usize = shape.iter().product();
            let out_name = &node.outputs[0];
            bufs.insert(out_name.clone(), inf.output_buffer_f16(n));
            shapes.insert(out_name.clone(), shape);
            let in_nhwc = *nhwc.get(actual_in.as_str()).unwrap_or(&false);
            nhwc.insert(out_name.clone(), in_nhwc);
            ops.push(MetalOp::Unary {
                input: actual_in,
                out: out_name.clone(),
                n: n as u32,
                op: 0, // relu
                f16: true,
            });
        }

        "Concat" => {
            let axis = get_attr_int(node, Attr::Axis).unwrap_or(1) as i32;
            let first_shape = shapes.get(&node.inputs[0]).cloned().unwrap_or_default();
            let is_nhwc_input = *nhwc.get(&node.inputs[0]).unwrap_or(&false);

            let ndim = first_shape.len() as i32;
            let actual_axis = if axis < 0 { ndim + axis } else { axis } as usize;
            // For NHWC 4D tensors, ONNX axis=1 (channels) maps to NHWC last dim
            let nhwc_axis = if is_nhwc_input && actual_axis == 1 && first_shape.len() == 4 {
                3
            } else {
                actual_axis
            };

            let is_last_dim = nhwc_axis == first_shape.len() - 1;
            let outer: usize = first_shape.iter().take(nhwc_axis).product();

            if is_last_dim {
                // Last-dim concat → use concat_channels kernel (interleaved)
                let mut input_names = Vec::new();
                let mut channels = Vec::new();
                let mut total_c = 0usize;
                let spatial: usize = first_shape.iter().take(nhwc_axis).product();

                for in_name in &node.inputs {
                    if in_name.is_empty() {
                        continue;
                    }
                    let s = shapes.get(in_name).cloned().unwrap_or_default();
                    let c = if s.len() > nhwc_axis { s[nhwc_axis] } else { 0 };
                    input_names.push(in_name.clone());
                    channels.push(c as u32);
                    total_c += c;
                }

                let out_name = &node.outputs[0];
                let total = spatial * total_c;
                bufs.insert(out_name.clone(), inf.output_buffer_f16(total));

                let mut out_shape = first_shape.clone();
                if out_shape.len() > nhwc_axis {
                    out_shape[nhwc_axis] = total_c;
                }
                shapes.insert(out_name.clone(), out_shape);
                nhwc.insert(out_name.clone(), is_nhwc_input);

                ops.push(MetalOp::Concat {
                    inputs: input_names,
                    channels,
                    out: out_name.clone(),
                    total_elements: total as u32,
                    out_c: total_c as u32,
                    f16: true,
                });
            } else if outer <= 1 {
                // Non-last-dim concat with outer=1: flat copy (contiguous inputs)
                let out_name = &node.outputs[0];
                let mut input_names = Vec::new();
                let mut input_sizes = Vec::new();
                let mut total_elements = 0usize;
                let mut total_concat_dim = 0usize;

                for in_name in &node.inputs {
                    if in_name.is_empty() {
                        continue;
                    }
                    let s = shapes.get(in_name).cloned().unwrap_or_default();
                    let n: usize = s.iter().product();
                    let c = s.get(nhwc_axis).copied().unwrap_or(0);
                    input_names.push(in_name.clone());
                    input_sizes.push(n as u32);
                    total_elements += n;
                    total_concat_dim += c;
                }

                bufs.insert(out_name.clone(), inf.output_buffer_f16(total_elements));
                let mut out_shape = first_shape.clone();
                if out_shape.len() > nhwc_axis {
                    out_shape[nhwc_axis] = total_concat_dim;
                }
                shapes.insert(out_name.clone(), out_shape);
                nhwc.insert(out_name.clone(), is_nhwc_input);

                ops.push(MetalOp::FlatConcat {
                    inputs: input_names,
                    sizes: input_sizes,
                    out: out_name.clone(),
                    f16: true,
                });
            } else {
                // General strided concat — CPU fallback
                cpu_fallback(node, env, cpu_data, cpu_shapes, bufs, shapes, nhwc, inf);
            }
        }

        "Split" => {
            let axis = get_attr_int(node, Attr::Axis).unwrap_or(1) as i32;
            let in_name = &node.inputs[0];
            let in_shape = shapes.get(in_name).cloned().unwrap_or_default();
            let is_nhwc_input = *nhwc.get(in_name).unwrap_or(&false);

            let ndim = in_shape.len() as i32;
            let actual_axis = if axis < 0 { ndim + axis } else { axis } as usize;
            let nhwc_axis = if is_nhwc_input && actual_axis == 1 && in_shape.len() == 4 {
                3
            } else {
                actual_axis
            };

            let in_c = in_shape.get(nhwc_axis).copied().unwrap_or(0);
            let spatial: usize = in_shape.iter().take(nhwc_axis).product();
            let inner: usize = in_shape
                .iter()
                .skip(nhwc_axis + 1)
                .product::<usize>()
                .max(1);
            let is_last_dim = nhwc_axis == in_shape.len() - 1;

            // Get split sizes
            let split_sizes: Vec<usize> = if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
                if let Some(t) = env.get(&node.inputs[1]) {
                    t.data().iter().map(|&v| v as usize).collect()
                } else {
                    let n_out = node.outputs.len();
                    vec![in_c / n_out; n_out]
                }
            } else {
                let n_out = node.outputs.len();
                vec![in_c / n_out; n_out]
            };

            // Allocate output buffers and shapes for all split outputs
            for (idx, &size) in split_sizes.iter().enumerate() {
                if idx >= node.outputs.len() {
                    break;
                }
                let out_name = &node.outputs[idx];
                let mut out_shape = in_shape.clone();
                if out_shape.len() > nhwc_axis {
                    out_shape[nhwc_axis] = size;
                }
                let out_n: usize = out_shape.iter().product();
                bufs.insert(out_name.clone(), inf.output_buffer_f16(out_n));
                shapes.insert(out_name.clone(), out_shape);
                nhwc.insert(out_name.clone(), is_nhwc_input);
            }

            // For last-dim splits with ≤3 outputs, use fused split (reads input once)
            let n_valid_outputs = split_sizes.len().min(node.outputs.len());
            if is_last_dim && (2..=3).contains(&n_valid_outputs) {
                let outputs: Vec<String> =
                    node.outputs.iter().take(n_valid_outputs).cloned().collect();
                let sizes: Vec<u32> = split_sizes
                    .iter()
                    .take(n_valid_outputs)
                    .map(|&s| s as u32)
                    .collect();
                ops.push(MetalOp::SplitFused {
                    input: in_name.clone(),
                    outputs,
                    split_sizes: sizes,
                    spatial: spatial as u32,
                    in_c: in_c as u32,
                });
            } else {
                // Fallback: individual split ops
                let mut offset = 0usize;
                for (idx, &size) in split_sizes.iter().enumerate() {
                    if idx >= node.outputs.len() {
                        break;
                    }
                    let out_name = &node.outputs[idx];
                    let out_n: usize = {
                        let mut s = in_shape.clone();
                        if s.len() > nhwc_axis {
                            s[nhwc_axis] = size;
                        }
                        s.iter().product()
                    };

                    if is_last_dim {
                        ops.push(MetalOp::Split {
                            input: in_name.clone(),
                            out: out_name.clone(),
                            spatial: spatial as u32,
                            in_c: in_c as u32,
                            out_c: size as u32,
                            offset_c: offset as u32,
                            f16: true,
                        });
                    } else if spatial <= 1 {
                        let src_offset = offset * inner;
                        ops.push(MetalOp::SliceCopy {
                            input: in_name.clone(),
                            out: out_name.clone(),
                            n: out_n as u32,
                            src_offset: src_offset as u32,
                            f16: true,
                        });
                    } else {
                        ops.push(MetalOp::Split {
                            input: in_name.clone(),
                            out: out_name.clone(),
                            spatial: spatial as u32,
                            in_c: (in_c * inner) as u32,
                            out_c: (size * inner) as u32,
                            offset_c: (offset * inner) as u32,
                            f16: true,
                        });
                    }
                    offset += size;
                }
            }
        }

        "MaxPool" => {
            let in_name = &node.inputs[0];
            let nhwc_in = ensure_nhwc_metal(inf, in_name, bufs, shapes, nhwc, ops);
            let in_shape = shapes.get(&nhwc_in).cloned().unwrap_or_default();
            if in_shape.len() != 4 {
                return Ok(());
            }
            let (n, ih, iw, ic) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);

            let kernel_shape = get_attr_ints(node, Attr::KernelShape).unwrap_or_else(|| vec![2, 2]);
            let strides = get_attr_ints(node, Attr::Strides).unwrap_or_else(|| vec![2, 2]);
            let pads = get_attr_ints(node, Attr::Pads).unwrap_or_else(|| vec![0, 0, 0, 0]);

            let kh = kernel_shape[0] as usize;
            let kw = kernel_shape[1] as usize;
            let sh = strides[0] as usize;
            let sw = strides[1] as usize;
            let oh = (ih + pads[0] as usize + pads[2] as usize - kh) / sh + 1;
            let ow = (iw + pads[1] as usize + pads[3] as usize - kw) / sw + 1;

            let out_name = &node.outputs[0];
            let out_n = n * oh * ow * ic;
            bufs.insert(out_name.clone(), inf.output_buffer_f16(out_n));
            shapes.insert(out_name.clone(), vec![n, oh, ow, ic]);
            nhwc.insert(out_name.clone(), true);

            ops.push(MetalOp::MaxPool {
                input: nhwc_in,
                out: out_name.clone(),
                batch: n as u32,
                ih: ih as u32,
                iw: iw as u32,
                ic: ic as u32,
                oh: oh as u32,
                ow: ow as u32,
                kh: kh as u32,
                kw: kw as u32,
                sh: sh as u32,
                sw: sw as u32,
                pad_h: pads[0] as u32,
                pad_w: pads[1] as u32,
                f16: true,
            });
        }

        "Resize" => {
            let in_name = &node.inputs[0];
            let nhwc_in = ensure_nhwc_metal(inf, in_name, bufs, shapes, nhwc, ops);
            let in_shape = shapes.get(&nhwc_in).cloned().unwrap_or_default();
            if in_shape.len() != 4 {
                return Ok(());
            }
            let (n, ih, iw, ic) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);

            // Get output shape from CPU result.
            // CPU shapes may be NHWC [N,H,W,C] or NCHW [N,C,H,W] depending on layout.
            // Detect by checking whether last dim matches input channels (NHWC) or dim-1 does (NCHW).
            let out_name = &node.outputs[0];
            let (oh, ow) = if let Some(s) = cpu_shapes.get(out_name) {
                if s.len() == 4 {
                    if s[3] == ic {
                        // NHWC: [N, H, W, C]
                        (s[1], s[2])
                    } else {
                        // NCHW: [N, C, H, W]
                        (s[2], s[3])
                    }
                } else {
                    (ih * 2, iw * 2)
                }
            } else if let Some(t) = env.get(out_name) {
                let s = t.shape();
                if s.len() == 4 {
                    if s[3] == ic {
                        (s[1], s[2])
                    } else {
                        (s[2], s[3])
                    }
                } else {
                    (ih * 2, iw * 2)
                }
            } else {
                (ih * 2, iw * 2)
            };

            let scale_h = oh as f32 / ih as f32;
            let scale_w = ow as f32 / iw as f32;

            let out_n = n * oh * ow * ic;
            bufs.insert(out_name.clone(), inf.output_buffer_f16(out_n));
            shapes.insert(out_name.clone(), vec![n, oh, ow, ic]);
            nhwc.insert(out_name.clone(), true);

            ops.push(MetalOp::Resize {
                input: nhwc_in,
                out: out_name.clone(),
                batch: n as u32,
                ih: ih as u32,
                iw: iw as u32,
                ic: ic as u32,
                oh: oh as u32,
                ow: ow as u32,
                scale_h,
                scale_w,
                f16: true,
            });
        }

        "Softmax" => {
            let actual_in = resolve_attn_input(
                inf,
                &node.inputs[0],
                is_attn_softmax,
                bufs,
                shapes,
                nhwc,
                ops,
                f32_bufs,
            );
            let in_shape = shapes.get(&actual_in).cloned().unwrap_or_default();
            let axis = get_attr_int(node, Attr::Axis).unwrap_or(-1);
            let ndim = in_shape.len() as i64;
            let actual_axis = if axis < 0 { ndim + axis } else { axis } as usize;

            let outer: usize = in_shape.iter().take(actual_axis).product();
            let softmax_dim: usize = if actual_axis < in_shape.len() {
                in_shape[actual_axis]
            } else {
                1
            };
            let inner: usize = in_shape.iter().skip(actual_axis + 1).product();
            let effective_outer = outer * inner;
            let dim = softmax_dim;
            let use_f16 = !is_attn_softmax;
            if std::env::var("METAL_DEBUG").is_ok() {
                eprintln!(
                    "  [metal] Softmax '{}' shape={:?} axis={} actual_axis={} outer={} dim={} inner={} effective_outer={} f32={}",
                    node.name,
                    in_shape,
                    axis,
                    actual_axis,
                    outer,
                    softmax_dim,
                    inner,
                    effective_outer,
                    is_attn_softmax,
                );
            }

            let out_name = &node.outputs[0];
            let n: usize = in_shape.iter().product();
            let in_nhwc = *nhwc.get(actual_in.as_str()).unwrap_or(&false);

            if inner == 1 {
                // Contiguous case: axis is last dim, kernel handles directly.
                if is_attn_softmax {
                    bufs.insert(out_name.clone(), inf.output_buffer(n));
                    f32_bufs.insert(out_name.clone());
                } else {
                    bufs.insert(out_name.clone(), inf.output_buffer_f16(n));
                }
                shapes.insert(out_name.clone(), in_shape);
                nhwc.insert(out_name.clone(), in_nhwc);

                ops.push(MetalOp::Softmax {
                    input: actual_in,
                    out: out_name.clone(),
                    outer: outer as u32,
                    dim: dim as u32,
                    f16: use_f16,
                });
            } else if in_shape.len() == 4 && actual_axis == 1 {
                // Strided softmax on axis=1 of 4D tensor [d0, d1, d2, d3]
                let (d0, d1, d2, d3) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);

                let tmp_pre = format!("{}_softmax_pre", out_name);
                if is_attn_softmax {
                    bufs.insert(tmp_pre.clone(), inf.output_buffer(n));
                } else {
                    bufs.insert(tmp_pre.clone(), inf.output_buffer_f16(n));
                }
                ops.push(MetalOp::CpuReshape {
                    input: actual_in,
                    out: tmp_pre.clone(),
                    n: n as u32,
                    nhwc_to_nchw: None,
                    nchw_to_nhwc: Some((d0 as u32, d1 as u32, d2 as u32, d3 as u32)),
                    f16: use_f16,
                });

                let tmp_post = format!("{}_softmax_post", out_name);
                if is_attn_softmax {
                    bufs.insert(tmp_post.clone(), inf.output_buffer(n));
                } else {
                    bufs.insert(tmp_post.clone(), inf.output_buffer_f16(n));
                }
                ops.push(MetalOp::Softmax {
                    input: tmp_pre.clone(),
                    out: tmp_post.clone(),
                    outer: effective_outer as u32,
                    dim: dim as u32,
                    f16: use_f16,
                });

                if is_attn_softmax {
                    bufs.insert(out_name.clone(), inf.output_buffer(n));
                    f32_bufs.insert(out_name.clone());
                } else {
                    bufs.insert(out_name.clone(), inf.output_buffer_f16(n));
                }
                shapes.insert(out_name.clone(), in_shape);
                nhwc.insert(out_name.clone(), in_nhwc);
                ops.push(MetalOp::CpuReshape {
                    input: tmp_post.clone(),
                    out: out_name.clone(),
                    n: n as u32,
                    nhwc_to_nchw: Some((d0 as u32, d2 as u32, d3 as u32, d1 as u32)),
                    nchw_to_nhwc: None,
                    f16: use_f16,
                });
            } else {
                // General strided softmax — fall back to CPU
                if std::env::var("METAL_DEBUG").is_ok() {
                    eprintln!(
                        "  [metal] Softmax fallback (strided): shape={:?} axis={} inner={}",
                        in_shape, axis, inner
                    );
                }
                cpu_fallback(node, env, cpu_data, cpu_shapes, bufs, shapes, nhwc, inf);
            }
        }

        "Transpose" => {
            // Transpose is data movement — propagate f32 from input, don't force
            let in_name = &node.inputs[0];
            let in_is_f32 = f32_bufs.contains(in_name.as_str());
            let in_shape = shapes.get(in_name.as_str()).cloned().unwrap_or_default();
            let perm = get_attr_ints(node, Attr::Perm).unwrap_or_default();
            let use_f16 = !in_is_f32;

            // Helper: allocate output buffer (f32 if input was f32, f16 otherwise)
            let alloc_out = |inf: &MetalInference,
                             bufs: &mut FxHashMap<String, Buffer>,
                             f32_bufs: &mut FxHashSet<String>,
                             name: &str,
                             n: usize| {
                if in_is_f32 {
                    bufs.insert(name.to_string(), inf.output_buffer(n));
                    f32_bufs.insert(name.to_string());
                } else {
                    bufs.insert(name.to_string(), inf.output_buffer_f16(n));
                }
            };

            if perm == [1, 0] && in_shape.len() == 2 {
                let rows = in_shape[0];
                let cols = in_shape[1];
                let out_name = &node.outputs[0];
                let n = rows * cols;
                alloc_out(inf, bufs, f32_bufs, out_name, n);
                shapes.insert(out_name.clone(), vec![cols, rows]);
                nhwc.insert(out_name.clone(), false);
                ops.push(MetalOp::Transpose2D {
                    input: in_name.clone(),
                    out: out_name.clone(),
                    rows: rows as u32,
                    cols: cols as u32,
                    f16: use_f16,
                });
            } else if perm == [0, 2, 1] && in_shape.len() == 3 {
                let (d0, d1, d2) = (in_shape[0], in_shape[1], in_shape[2]);
                let out_name = &node.outputs[0];
                let n = d0 * d1 * d2;
                alloc_out(inf, bufs, f32_bufs, out_name, n);
                shapes.insert(out_name.clone(), vec![d0, d2, d1]);
                nhwc.insert(out_name.clone(), false);
                ops.push(MetalOp::Permute0213 {
                    input: in_name.clone(),
                    out: out_name.clone(),
                    d0: d0 as u32,
                    d1: d1 as u32,
                    d2: d2 as u32,
                    d3: 1u32,
                    f16: use_f16,
                });
            } else if perm == [0, 1, 3, 2] && in_shape.len() == 4 {
                let (d0, d1, d2, d3) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
                let out_name = &node.outputs[0];
                let n = d0 * d1 * d2 * d3;
                alloc_out(inf, bufs, f32_bufs, out_name, n);
                shapes.insert(out_name.clone(), vec![d0, d1, d3, d2]);
                nhwc.insert(out_name.clone(), false);
                ops.push(MetalOp::Permute0213 {
                    input: in_name.clone(),
                    out: out_name.clone(),
                    d0: (d0 * d1) as u32,
                    d1: d2 as u32,
                    d2: d3 as u32,
                    d3: 1u32,
                    f16: use_f16,
                });
            } else if perm == [0, 2, 1, 3] && in_shape.len() == 4 {
                let (d0, d1, d2, d3) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
                let out_name = &node.outputs[0];
                let n = d0 * d1 * d2 * d3;
                alloc_out(inf, bufs, f32_bufs, out_name, n);
                shapes.insert(out_name.clone(), vec![d0, d2, d1, d3]);
                nhwc.insert(out_name.clone(), false);
                ops.push(MetalOp::Permute0213 {
                    input: in_name.clone(),
                    out: out_name.clone(),
                    d0: d0 as u32,
                    d1: d1 as u32,
                    d2: d2 as u32,
                    d3: d3 as u32,
                    f16: use_f16,
                });
            } else if perm == [0, 3, 1, 2] && in_shape.len() == 4 {
                let (n_dim, h, w, c) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
                let out_name = &node.outputs[0];
                let n = n_dim * h * w * c;
                alloc_out(inf, bufs, f32_bufs, out_name, n);
                shapes.insert(out_name.clone(), vec![n_dim, c, h, w]);
                nhwc.insert(out_name.clone(), false);
                ops.push(MetalOp::CpuReshape {
                    input: in_name.clone(),
                    out: out_name.clone(),
                    n: n as u32,
                    nhwc_to_nchw: Some((n_dim as u32, h as u32, w as u32, c as u32)),
                    nchw_to_nhwc: None,
                    f16: use_f16,
                });
            } else if perm == [0, 2, 3, 1] && in_shape.len() == 4 {
                let (n_dim, c, h, w) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
                let out_name = &node.outputs[0];
                let n = n_dim * c * h * w;
                alloc_out(inf, bufs, f32_bufs, out_name, n);
                shapes.insert(out_name.clone(), vec![n_dim, h, w, c]);
                nhwc.insert(out_name.clone(), true);
                ops.push(MetalOp::CpuReshape {
                    input: in_name.clone(),
                    out: out_name.clone(),
                    n: n as u32,
                    nhwc_to_nchw: None,
                    nchw_to_nhwc: Some((n_dim as u32, c as u32, h as u32, w as u32)),
                    f16: use_f16,
                });
            } else if perm == [0, 3, 2, 1] && in_shape.len() == 4 {
                let (d0, d1, d2, d3) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
                let out_name = &node.outputs[0];
                let n = d0 * d1 * d2 * d3;

                let tmp_name = format!("{}_perm_tmp", out_name);
                alloc_out(inf, bufs, f32_bufs, &tmp_name, n);
                ops.push(MetalOp::Permute0213 {
                    input: in_name.clone(),
                    out: tmp_name.clone(),
                    d0: d0 as u32,
                    d1: d1 as u32,
                    d2: d2 as u32,
                    d3: d3 as u32,
                    f16: use_f16,
                });

                alloc_out(inf, bufs, f32_bufs, out_name, n);
                shapes.insert(out_name.clone(), vec![d0, d3, d2, d1]);
                nhwc.insert(out_name.clone(), false);
                ops.push(MetalOp::CpuReshape {
                    input: tmp_name,
                    out: out_name.clone(),
                    n: n as u32,
                    nhwc_to_nchw: Some((d0 as u32, d2 as u32, d1 as u32, d3 as u32)),
                    nchw_to_nhwc: None,
                    f16: use_f16,
                });
            } else {
                if std::env::var("METAL_DEBUG").is_ok() {
                    eprintln!(
                        "  [metal] Transpose fallback: perm={:?} shape={:?}",
                        perm, in_shape
                    );
                }
                cpu_fallback(node, env, cpu_data, cpu_shapes, bufs, shapes, nhwc, inf);
            }
        }

        "MatMul" => {
            let actual_a = resolve_attn_input(
                inf,
                &node.inputs[0],
                is_attn_matmul,
                bufs,
                shapes,
                nhwc,
                ops,
                f32_bufs,
            );
            let actual_b = resolve_attn_input(
                inf,
                &node.inputs[1],
                is_attn_matmul,
                bufs,
                shapes,
                nhwc,
                ops,
                f32_bufs,
            );
            let a_shape = shapes.get(&actual_a).cloned().unwrap_or_default();
            let b_shape = shapes.get(&actual_b).cloned().unwrap_or_default();

            if a_shape.len() >= 2 && b_shape.len() >= 2 {
                let m = a_shape.iter().rev().skip(1).product::<usize>();
                let k = *a_shape.last().unwrap_or(&1);
                let n = *b_shape.last().unwrap_or(&1);

                let out_name = &node.outputs[0];
                let out_n = m * n;
                if is_attn_matmul {
                    bufs.insert(out_name.clone(), inf.output_buffer(out_n));
                    f32_bufs.insert(out_name.clone());
                } else {
                    bufs.insert(out_name.clone(), inf.output_buffer_f16(out_n));
                }

                let mut out_shape = a_shape.clone();
                if let Some(last) = out_shape.last_mut() {
                    *last = n;
                }
                shapes.insert(out_name.clone(), out_shape);
                nhwc.insert(out_name.clone(), false);

                ops.push(MetalOp::MatMul {
                    a: actual_a,
                    b: actual_b,
                    out: out_name.clone(),
                    m: m as u32,
                    n: n as u32,
                    k: k as u32,
                    f16: !is_attn_matmul,
                });
            }
        }

        "Reshape" | "Flatten" | "Unsqueeze" | "Squeeze" | "Expand" => {
            // Shape-changing ops: data is the same, only shape changes.
            let in_name = &node.inputs[0];
            let out_name = &node.outputs[0];
            if in_name.is_empty() || out_name.is_empty() {
                cpu_fallback(node, env, cpu_data, cpu_shapes, bufs, shapes, nhwc, inf);
                return Ok(());
            }

            // Get output shape from CPU snapshots
            let out_shape = if let Some(s) = cpu_shapes.get(out_name) {
                s.clone()
            } else if let Some(t) = env.get(out_name) {
                t.shape().to_vec()
            } else {
                cpu_fallback(node, env, cpu_data, cpu_shapes, bufs, shapes, nhwc, inf);
                return Ok(());
            };

            ensure_on_metal(inf, in_name, env, bufs, shapes, nhwc);

            let in_is_nhwc = *nhwc.get(in_name).unwrap_or(&false);
            let in_shape = shapes.get(in_name).cloned().unwrap_or_default();
            let in_is_f32 = f32_bufs.contains(in_name.as_str());

            if bufs.contains_key(in_name) {
                let n_in = in_shape.iter().product::<usize>().max(1);
                let n_out = out_shape.iter().product::<usize>().max(1);

                if std::env::var("METAL_DEBUG").is_ok() {
                    eprintln!(
                        "  [Reshape] '{}' in_shape={:?} nhwc={} f32={} → '{}' out_shape={:?} n_in={} n_out={}",
                        in_name, in_shape, in_is_nhwc, in_is_f32, out_name, out_shape, n_in, n_out
                    );
                }

                if in_is_nhwc && in_shape.len() == 4 {
                    // Need NHWC→NCHW permutation — must copy
                    if in_is_f32 {
                        bufs.insert(out_name.clone(), inf.output_buffer(n_out));
                        f32_bufs.insert(out_name.clone());
                    } else {
                        bufs.insert(out_name.clone(), inf.output_buffer_f16(n_out));
                    }
                    let (n_dim, h, w, c) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
                    ops.push(MetalOp::CpuReshape {
                        input: in_name.clone(),
                        out: out_name.clone(),
                        n: n_out as u32,
                        nhwc_to_nchw: Some((n_dim as u32, h as u32, w as u32, c as u32)),
                        nchw_to_nhwc: None,
                        f16: !in_is_f32,
                    });
                } else if n_in == n_out {
                    // Same element count, no layout change — zero-cost buffer alias
                    let existing_buf = bufs.get(in_name).unwrap().clone();
                    bufs.insert(out_name.clone(), existing_buf);
                    // Propagate f32 flag through reshape alias
                    if in_is_f32 {
                        f32_bufs.insert(out_name.clone());
                    }
                } else {
                    cpu_fallback(node, env, cpu_data, cpu_shapes, bufs, shapes, nhwc, inf);
                    return Ok(());
                }
                shapes.insert(out_name.clone(), out_shape);
                nhwc.insert(out_name.clone(), false);
            } else {
                cpu_fallback(node, env, cpu_data, cpu_shapes, bufs, shapes, nhwc, inf);
            }
        }

        "Slice" => {
            let in_name = &node.inputs[0];
            let in_shape = shapes.get(in_name).cloned().unwrap_or_default();
            let out_name = &node.outputs[0];

            let read_const = |idx: usize| -> Option<Vec<i64>> {
                if idx >= node.inputs.len() || node.inputs[idx].is_empty() {
                    return None;
                }
                let name = &node.inputs[idx];
                if let Some(data) = cpu_data.get(name) {
                    Some(data.iter().map(|&v| v as i64).collect())
                } else {
                    env.get(name)
                        .map(|t| t.data().iter().map(|&v| v as i64).collect())
                }
            };

            let starts = read_const(1);
            let ends = read_const(2);
            let axes_opt = read_const(3);
            let steps = read_const(4);
            let can_metal = starts.is_some()
                && ends.is_some()
                && steps.as_ref().is_none_or(|s| s.iter().all(|&v| v == 1));

            if can_metal && !in_shape.is_empty() {
                let starts = starts.unwrap();
                let ends = ends.unwrap();
                let ndim = in_shape.len();
                let axes: Vec<usize> = axes_opt.map_or_else(
                    || (0..starts.len()).collect(),
                    |a| {
                        a.iter()
                            .map(|&v| {
                                if v < 0 {
                                    (ndim as i64 + v) as usize
                                } else {
                                    v as usize
                                }
                            })
                            .collect()
                    },
                );

                let mut out_shape = in_shape.clone();
                let mut n_sliced = 0;
                let mut slice_axis = 0usize;
                let mut slice_start = 0usize;

                for (i, &ax) in axes.iter().enumerate() {
                    let dim = in_shape[ax] as i64;
                    let s = if starts[i] < 0 {
                        dim + starts[i]
                    } else {
                        starts[i]
                    };
                    let e = if ends[i] < 0 {
                        dim + ends[i]
                    } else {
                        ends[i].min(dim)
                    };
                    let s = s.max(0) as usize;
                    let e = e.max(0) as usize;
                    out_shape[ax] = e - s;
                    if out_shape[ax] != in_shape[ax] {
                        n_sliced += 1;
                        slice_axis = ax;
                        slice_start = s;
                    }
                }

                let inner: usize = in_shape.iter().skip(slice_axis + 1).product();
                let outer: usize = in_shape.iter().take(slice_axis).product();

                if n_sliced <= 1 && outer <= 1 {
                    let src_offset = slice_start * inner;
                    let out_n: usize = out_shape.iter().product();

                    ensure_on_metal(inf, in_name, env, bufs, shapes, nhwc);
                    bufs.insert(out_name.clone(), inf.output_buffer_f16(out_n));
                    shapes.insert(out_name.clone(), out_shape);
                    let in_nhwc = *nhwc.get(in_name).unwrap_or(&false);
                    nhwc.insert(out_name.clone(), in_nhwc);

                    ops.push(MetalOp::SliceCopy {
                        input: in_name.clone(),
                        out: out_name.clone(),
                        n: out_n as u32,
                        src_offset: src_offset as u32,
                        f16: true,
                    });
                } else {
                    cpu_fallback(node, env, cpu_data, cpu_shapes, bufs, shapes, nhwc, inf);
                }
            } else {
                cpu_fallback(node, env, cpu_data, cpu_shapes, bufs, shapes, nhwc, inf);
            }
        }

        "Gather" | "Shape" | "Cast" | "ConstantOfShape" | "Range" => {
            cpu_fallback(node, env, cpu_data, cpu_shapes, bufs, shapes, nhwc, inf);
        }

        _ => {
            // Unknown op — CPU fallback
            cpu_fallback(node, env, cpu_data, cpu_shapes, bufs, shapes, nhwc, inf);
        }
    }

    // Shape debug for DFL path
    if std::env::var("METAL_SHAPE_DBG").is_ok() && node.name.contains("model.22") {
        for out in &node.outputs {
            let s = shapes.get(out).cloned().unwrap_or_default();
            let is_nhwc = *nhwc.get(out).unwrap_or(&false);
            eprintln!(
                "  [shape] {} '{}' out='{}': shape={:?} nhwc={}",
                node.op_type, node.name, out, s, is_nhwc
            );
        }
    }

    Ok(())
}
