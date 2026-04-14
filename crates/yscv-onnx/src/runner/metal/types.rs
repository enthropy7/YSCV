use ::metal::*;
use std::collections::{HashMap, HashSet};

use yscv_kernels::metal_backend::metal_conv::{ConvParams, MetalInference, WinogradParams};

/// How the input is uploaded at runtime.
pub(crate) enum InputUploadMode {
    /// Upload f32 data to an f32 buffer; a GPU cast op converts to f16.
    F32GpuCast,
    /// CPU-side f32→f16 + NCHW→NHWC, write directly to f16 NHWC buffer.
    CpuCastNchwToNhwc {
        batch: usize,
        c: usize,
        h: usize,
        w: usize,
    },
}

/// Compiled Metal execution plan.
pub struct MetalPlan {
    pub(crate) inf: MetalInference,
    pub(crate) ops: Vec<MetalOp>,
    /// All named buffers (weights + intermediates).
    pub(crate) bufs: HashMap<String, Buffer>,
    pub(crate) buf_shapes: HashMap<String, Vec<usize>>,
    pub(crate) buf_nhwc: HashMap<String, bool>,
    pub(crate) input_buf_name: String,
    pub(crate) input_upload: InputUploadMode,
    pub(crate) output_names: Vec<String>,
    /// Debug: CPU reference data for per-buffer comparison (only when METAL_COMPARE is set)
    #[allow(dead_code)]
    pub(crate) cpu_ref: HashMap<String, Vec<f32>>,
    /// Buffers that hold f32 data (for attention chain)
    #[cfg_attr(not(feature = "profile"), allow(dead_code))]
    pub(crate) buf_f32: HashSet<String>,
}

#[allow(dead_code)]
pub(crate) enum MetalOp {
    ConvGemm {
        input: String,
        weight: String,
        bias: String,
        output: String,
        params: ConvParams,
        f16io: bool, // use f16 input/output pipeline
        /// Optional residual buffer for fused Conv+Add (residual connection).
        /// When Some, the output adds residual[i] to each element before activation.
        residual: Option<String>,
    },
    ConvDirect {
        input: String,
        weight: String,
        bias: String,
        output: String,
        params: ConvParams,
        f16io: bool,
    },
    DepthwiseConv {
        input: String,
        weight: String,
        bias: String,
        output: String,
        params: ConvParams,
    },
    Binary {
        a: String,
        b: String,
        out: String,
        n: u32,
        op: u32, // 0=add, 1=sub, 2=mul, 3=div
        f16: bool,
    },
    BroadcastBinary {
        a: String,
        b: String,
        out: String,
        n: u32,
        broadcast_dim: u32,
        op: u32,
        f16: bool,
    },
    Unary {
        input: String,
        out: String,
        n: u32,
        op: u32, // 0=relu, 1=sigmoid, 2=silu
        f16: bool,
    },
    SiLU {
        input: String,
        out: String,
        n: u32,
        f16: bool,
    },
    Concat {
        inputs: Vec<String>,
        channels: Vec<u32>,
        out: String,
        total_elements: u32,
        out_c: u32,
        f16: bool,
    },
    Split {
        input: String,
        out: String,
        spatial: u32,
        in_c: u32,
        out_c: u32,
        offset_c: u32,
        f16: bool,
    },
    /// Fused N-way split: reads input once, writes to up to 3 outputs
    SplitFused {
        input: String,
        outputs: Vec<String>,
        split_sizes: Vec<u32>,
        spatial: u32,
        in_c: u32,
    },
    MaxPool {
        input: String,
        out: String,
        batch: u32,
        ih: u32,
        iw: u32,
        ic: u32,
        oh: u32,
        ow: u32,
        kh: u32,
        kw: u32,
        sh: u32,
        sw: u32,
        pad_h: u32,
        pad_w: u32,
        f16: bool,
    },
    Resize {
        input: String,
        out: String,
        batch: u32,
        ih: u32,
        iw: u32,
        ic: u32,
        oh: u32,
        ow: u32,
        scale_h: f32,
        scale_w: f32,
        f16: bool,
    },
    Softmax {
        input: String,
        out: String,
        outer: u32,
        dim: u32,
        f16: bool,
    },
    Transpose2D {
        input: String,
        out: String,
        rows: u32,
        cols: u32,
        f16: bool,
    },
    /// Reshape: copy data from input to output, optionally with layout permutation.
    CpuReshape {
        input: String,
        out: String,
        n: u32, // total elements
        /// If Some((n,h,w,c)), apply NHWC→NCHW permutation during copy
        nhwc_to_nchw: Option<(u32, u32, u32, u32)>,
        /// If Some((n,c,h,w)), apply NCHW→NHWC permutation during copy
        nchw_to_nhwc: Option<(u32, u32, u32, u32)>,
        f16: bool,
    },
    /// 4D transpose: swap dim1 and dim2 [D0,D1,D2,D3] → [D0,D2,D1,D3]
    Permute0213 {
        input: String,
        out: String,
        d0: u32,
        d1: u32,
        d2: u32,
        d3: u32,
        f16: bool,
    },
    /// Slice copy: copy n elements from input[src_offset..] to output[0..n]
    SliceCopy {
        input: String,
        out: String,
        n: u32,
        src_offset: u32,
        f16: bool,
    },
    /// Flat concat: copy multiple contiguous input buffers into one output buffer
    /// at sequential offsets. Used for non-last-axis concat where outer=1.
    FlatConcat {
        inputs: Vec<String>,
        sizes: Vec<u32>, // elements per input
        out: String,
        f16: bool,
    },
    MatMul {
        a: String,
        b: String,
        out: String,
        m: u32,
        n: u32,
        k: u32,
        f16: bool,
    },
    /// Fused NHWC→flat-NCHW concat: reads up to 3 NHWC inputs, writes [C, total_spatial].
    /// Replaces CpuReshape(NHWC→NCHW) + FlatConcat in detection heads.
    NhwcToFlatConcat {
        inputs: Vec<String>, // up to 3 NHWC input buffer names
        out: String,
        c: u32,              // channel count (same for all inputs)
        hw: Vec<(u32, u32)>, // (h, w) for each input
        total_spatial: u32,  // sum of h*w for all inputs
    },
    /// Channel scatter: copy dense [spatial, src_c] → strided [spatial, dst_c] at offset.
    ChannelScatter {
        input: String,
        out: String,
        spatial: u32,
        src_c: u32,
        dst_c: u32,
        dst_offset: u32,
    },
    /// Cast f32 buffer → f16 buffer (graph input boundary)
    CastF32ToF16 { input: String, out: String, n: u32 },
    /// Cast f16 buffer → f32 buffer (graph output boundary)
    CastF16ToF32 { input: String, out: String, n: u32 },
    /// MPS-accelerated conv: im2col (optional) → MPS GEMM → bias+act.
    /// For 1×1 convs, im2col_buf is None (input IS the A matrix).
    /// For 3×3 convs, im2col_buf holds the [M, K] column matrix.
    MpsConv {
        input: String,
        weight: String,             // f16 weight buffer [K, N] (K=kh*kw*ic, N=o_ch)
        bias: String,               // f32 bias buffer [N]
        output: String,             // f16 output [M, N]
        im2col_buf: Option<String>, // temp im2col buffer for 3×3+
        m: u32,
        n: u32,   // o_ch
        k: u32,   // kh*kw*ic
        act: u32, // 0=none, 1=relu, 2=silu
        // Im2col params (only used when im2col_buf is Some)
        batch: u32,
        ih: u32,
        iw: u32,
        ic: u32,
        oh: u32,
        ow: u32,
        kh: u32,
        kw: u32,
        sh: u32,
        sw: u32,
        pad_h: u32,
        pad_w: u32,
    },
    /// Winograd F(2×2,3×3) conv: input_transform → batched GEMM × 16 → output_transform.
    ConvWinograd {
        input: String,
        weight: String, // pre-transformed weights (16, ic, oc) as f16
        bias: String,
        output: String,
        transformed_input: String, // temp buffer (16, n_tiles, ic) f16
        gemm_output: String,       // temp buffer (16, n_tiles, oc) f16
        wino_params: WinogradParams,
        ic: u32,
        oc: u32,
        /// Optional residual buffer for fused Conv+Add (residual connection).
        /// When Some, the output transform adds residual[i] to each output element.
        residual: Option<String>,
    },
}

impl MetalPlan {
    pub fn ops_count(&self) -> usize {
        self.ops.len()
    }

    /// Print op distribution and key dimensions for diagnostics.
    #[cfg(feature = "profile")]
    pub fn dump_op_stats(&self) {
        let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        for op in &self.ops {
            let name = match op {
                MetalOp::ConvGemm { .. } => "ConvGemm",
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
                MetalOp::MpsConv { .. } => "MpsConv",
                MetalOp::ConvWinograd { .. } => "ConvWinograd",
                MetalOp::NhwcToFlatConcat { .. } => "NhwcToFlatConcat",
                MetalOp::ChannelScatter { .. } => "ChannelScatter",
            };
            *counts.entry(name).or_insert(0) += 1;
        }
        let mut sorted: Vec<_> = counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        eprintln!("  Op distribution:");
        for (name, count) in &sorted {
            eprintln!("    {}: {}", name, count);
        }
        // Print Conv dimensions grouped by kernel size
        let mut conv_1x1_count = 0u32;
        let mut conv_1x1_flops = 0f64;
        let mut conv_3x3_count = 0u32;
        let mut conv_3x3_flops = 0f64;
        let mut conv_other_count = 0u32;
        let mut conv_other_flops = 0f64;
        let mut dw_count = 0u32;
        for op in &self.ops {
            match op {
                MetalOp::ConvGemm { params, .. } | MetalOp::ConvDirect { params, .. } => {
                    let flops =
                        (params.m as f64) * (params.k as f64) * (params.n_out as f64) * 2.0 / 1e6;
                    if params.kh == 1 && params.kw == 1 {
                        conv_1x1_count += 1;
                        conv_1x1_flops += flops;
                        if std::env::var("METAL_CONV_SIZES").is_ok() {
                            eprintln!(
                                "    Conv1x1: M={} K={} N={} ({:.1}MFLOP)",
                                params.m, params.k, params.n_out, flops
                            );
                        }
                    } else if params.kh == 3 && params.kw == 3 {
                        conv_3x3_count += 1;
                        conv_3x3_flops += flops;
                        if std::env::var("METAL_CONV_SIZES").is_ok() {
                            eprintln!(
                                "    Conv3x3: M={} K={} N={} ({:.1}MFLOP)",
                                params.m, params.k, params.n_out, flops
                            );
                        }
                    } else {
                        conv_other_count += 1;
                        conv_other_flops += flops;
                    }
                }
                MetalOp::MpsConv {
                    m, n, k, kh, kw, ..
                } => {
                    let flops = (*m as f64) * (*k as f64) * (*n as f64) * 2.0 / 1e6;
                    if *kh == 1 && *kw == 1 {
                        conv_1x1_count += 1;
                        conv_1x1_flops += flops;
                    } else if *kh == 3 && *kw == 3 {
                        conv_3x3_count += 1;
                        conv_3x3_flops += flops;
                    } else {
                        conv_other_count += 1;
                        conv_other_flops += flops;
                    }
                }
                MetalOp::ConvWinograd {
                    wino_params,
                    ic,
                    oc,
                    ..
                } => {
                    // Original FLOP equivalent (for comparison)
                    let orig_flops = (wino_params.batch as f64)
                        * (wino_params.oh as f64)
                        * (wino_params.ow as f64)
                        * 9.0
                        * (*ic as f64)
                        * (*oc as f64)
                        * 2.0
                        / 1e6;
                    conv_3x3_count += 1;
                    conv_3x3_flops += orig_flops;
                    if std::env::var("METAL_CONV_SIZES").is_ok() {
                        eprintln!(
                            "    ConvWino: tiles={} ic={} oc={} ({:.1}MFLOP orig)",
                            wino_params.n_tiles, ic, oc, orig_flops
                        );
                    }
                }
                MetalOp::DepthwiseConv { .. } => {
                    dw_count += 1;
                }
                MetalOp::MatMul { m, n, k, .. } => {
                    eprintln!(
                        "    MatMul: m={} n={} k={} ({:.1}M FMAs)",
                        m,
                        n,
                        k,
                        (*m as f64) * (*n as f64) * (*k as f64) / 1e6
                    );
                }
                MetalOp::Softmax { outer, dim, .. } => {
                    eprintln!("    Softmax: outer={} dim={}", outer, dim);
                }
                _ => {}
            }
        }
        eprintln!(
            "    Conv 1x1: {}x  {:.1} MFLOP",
            conv_1x1_count, conv_1x1_flops
        );
        eprintln!(
            "    Conv 3x3: {}x  {:.1} MFLOP",
            conv_3x3_count, conv_3x3_flops
        );
        if conv_other_count > 0 {
            eprintln!(
                "    Conv other: {}x  {:.1} MFLOP",
                conv_other_count, conv_other_flops
            );
        }
        if dw_count > 0 {
            eprintln!("    DepthwiseConv: {}x", dw_count);
        }
        eprintln!(
            "    Total conv MFLOP: {:.1}",
            conv_1x1_flops + conv_3x3_flops + conv_other_flops
        );
    }

    /// Stub: compiled only when `profile` feature is off.
    #[cfg(not(feature = "profile"))]
    pub fn dump_op_stats(&self) {}
}
