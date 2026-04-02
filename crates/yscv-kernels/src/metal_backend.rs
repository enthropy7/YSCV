//! Direct Metal backend for GPU inference on Apple Silicon.
//! Bypasses wgpu/naga for ~2x faster compute shader execution.

#[cfg(all(target_os = "macos", feature = "metal-backend"))]
#[allow(unexpected_cfgs)] // objc crate's msg_send! macro triggers false cfg warnings
pub mod metal_conv {
    use crate::KernelError;
    use foreign_types::ForeignTypeRef as _;
    use metal::*;
    pub use objc::rc::autoreleasepool;
    #[allow(unused_imports)]
    use objc::{msg_send, sel, sel_impl};
    use std::mem;

    const CONV_GEMM_SIMD_MSL: &str = include_str!("shaders/conv_gemm_simd.metal");
    const CONV_GEMM_BASIC_MSL: &str = include_str!("shaders/conv_gemm_metal_basic.metal");
    const METAL_OPS_MSL: &str = include_str!("shaders/metal_ops.metal");
    const CONV_WINOGRAD_MSL: &str = include_str!("shaders/conv_winograd.metal");

    /// IEEE 754 f32 → f16 conversion with round-to-nearest-even.
    fn f32_to_f16(val: f32) -> u16 {
        let bits = val.to_bits();
        let sign = (bits >> 16) & 0x8000;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let man = bits & 0x7FFFFF;
        if exp == 255 {
            // Inf/NaN
            return (sign | 0x7C00 | if man != 0 { 0x200 } else { 0 }) as u16;
        }
        let new_exp = exp - 127 + 15;
        if new_exp >= 31 {
            return (sign | 0x7C00) as u16; // overflow → Inf
        }
        if new_exp <= 0 {
            if new_exp < -10 {
                return sign as u16; // underflow → ±0
            }
            let man_with_hidden = man | 0x800000;
            let shift = (1 - new_exp) as u32;
            let half_man = man_with_hidden >> (13 + shift);
            // Round-to-nearest-even for denormals
            let round_bit = (man_with_hidden >> (12 + shift)) & 1;
            let sticky = man_with_hidden & ((1 << (12 + shift)) - 1);
            let round_up = round_bit != 0 && (sticky != 0 || (half_man & 1) != 0);
            return (sign | (half_man + round_up as u32)) as u16;
        }
        // Normal: round-to-nearest-even on the 13 dropped mantissa bits
        let truncated = man >> 13;
        let round_bit = (man >> 12) & 1; // highest dropped bit
        let sticky = man & 0xFFF; // remaining 12 bits
        let round_up = round_bit != 0 && (sticky != 0 || (truncated & 1) != 0);
        let result = sign | ((new_exp as u32) << 10) | truncated;
        (result + round_up as u32) as u16 // carry propagates into exponent if man overflows
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    pub struct ConvParams {
        pub m: u32,
        pub n_out: u32,
        pub k: u32,
        pub act: u32,
        pub ih: u32,
        pub iw: u32,
        pub ic: u32,
        pub oh: u32,
        pub ow: u32,
        pub kh: u32,
        pub kw: u32,
        pub sh: u32,
        pub sw: u32,
        pub pad_h: u32,
        pub pad_w: u32,
        pub batch: u32,
        pub out_stride: u32,
        pub out_offset: u32,
        pub in_stride: u32,
        pub in_offset: u32,
        pub has_residual: u32,
        pub _pad: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    pub struct WinogradParams {
        pub batch: u32,
        pub ih: u32,
        pub iw: u32,
        pub ic: u32,
        pub oh: u32,
        pub ow: u32,
        pub oc: u32,
        pub pad_h: u32,
        pub pad_w: u32,
        pub tile_h: u32,
        pub tile_w: u32,
        pub n_tiles: u32,
        pub act: u32,
        pub out_stride: u32, // output channel stride (default = oc, > oc for concat fusion)
        pub out_offset: u32, // output channel offset (default = 0)
        pub in_stride: u32,  // input channel stride (default = ic, > ic for split fusion)
        pub in_offset: u32,  // input channel offset (default = 0)
    }

    /// Winograd F(2×2,3×3) weight transform on CPU.
    /// Input weights: [K, N] where K = kh*kw*ic = 9*ic, N = oc (KHWC layout).
    /// Output: [16, ic, oc] as f16 bytes ready for GPU upload.
    /// G = [[1,0,0],[1/2,1/2,1/2],[1/2,-1/2,1/2],[0,0,1]]
    pub fn winograd_transform_weights_f16(weights_f32: &[f32], ic: usize, oc: usize) -> Vec<u16> {
        // weights_f32 layout: [kh*kw*ic, oc] where kh=kw=3
        // For each (ic_idx, oc_idx): g[ky][kx] = weights[(ky*3+kx)*ic + ic_idx][oc_idx]
        //                                       = weights_f32[((ky*3+kx)*ic + ic_idx) * oc + oc_idx]
        let k = 9 * ic; // total K dim
        assert_eq!(weights_f32.len(), k * oc);

        let mut out = vec![0u16; 16 * ic * oc];

        for ic_idx in 0..ic {
            for oc_idx in 0..oc {
                // Extract 3×3 filter
                let mut g = [[0.0f32; 3]; 3];
                for ky in 0..3 {
                    for kx in 0..3 {
                        g[ky][kx] = weights_f32[((ky * 3 + kx) * ic + ic_idx) * oc + oc_idx];
                    }
                }

                // u = G * g (4×3)
                let mut u = [[0.0f32; 3]; 4];
                for j in 0..3 {
                    u[0][j] = g[0][j];
                    u[1][j] = (g[0][j] + g[1][j] + g[2][j]) * 0.5;
                    u[2][j] = (g[0][j] - g[1][j] + g[2][j]) * 0.5;
                    u[3][j] = g[2][j];
                }

                // v = u * G^T (4×4)
                let mut v = [[0.0f32; 4]; 4];
                for i in 0..4 {
                    v[i][0] = u[i][0];
                    v[i][1] = (u[i][0] + u[i][1] + u[i][2]) * 0.5;
                    v[i][2] = (u[i][0] - u[i][1] + u[i][2]) * 0.5;
                    v[i][3] = u[i][2];
                }

                // Store as (16, ic, oc): out[alpha * ic * oc + ic_idx * oc + oc_idx]
                for ai in 0..4 {
                    for aj in 0..4 {
                        let alpha = ai * 4 + aj;
                        out[alpha * ic * oc + ic_idx * oc + oc_idx] = f32_to_f16(v[ai][aj]);
                    }
                }
            }
        }
        out
    }

    /// Winograd F(4×4,3×3) weight transform on CPU.
    /// Input: [K, N] where K = 9*ic, N = oc (KHWC layout).
    /// Output: [36, ic, oc] as f16 bytes.
    /// G (6×3) = [[1/4,0,0],[-1/6,-1/6,-1/6],[-1/6,1/6,-1/6],[1/24,1/12,1/6],[1/24,-1/12,1/6],[0,0,1]]
    pub fn winograd4x4_transform_weights_f16(
        weights_f32: &[f32],
        ic: usize,
        oc: usize,
    ) -> Vec<u16> {
        let k = 9 * ic;
        assert_eq!(weights_f32.len(), k * oc);

        let mut out = vec![0u16; 36 * ic * oc];

        for ic_idx in 0..ic {
            for oc_idx in 0..oc {
                let mut g = [[0.0f32; 3]; 3];
                for ky in 0..3 {
                    for kx in 0..3 {
                        g[ky][kx] = weights_f32[((ky * 3 + kx) * ic + ic_idx) * oc + oc_idx];
                    }
                }

                // u = G * g (6×3)  where G is the F(4,3) weight transform matrix
                let mut u = [[0.0f32; 3]; 6];
                for j in 0..3 {
                    u[0][j] = g[0][j] / 4.0;
                    u[1][j] = (-g[0][j] - g[1][j] - g[2][j]) / 6.0;
                    u[2][j] = (-g[0][j] + g[1][j] - g[2][j]) / 6.0;
                    u[3][j] = g[0][j] / 24.0 + g[1][j] / 12.0 + g[2][j] / 6.0;
                    u[4][j] = g[0][j] / 24.0 - g[1][j] / 12.0 + g[2][j] / 6.0;
                    u[5][j] = g[2][j];
                }

                // v = u * G^T (6×6)
                let mut v = [[0.0f32; 6]; 6];
                for i in 0..6 {
                    v[i][0] = u[i][0] / 4.0;
                    v[i][1] = (-u[i][0] - u[i][1] - u[i][2]) / 6.0;
                    v[i][2] = (-u[i][0] + u[i][1] - u[i][2]) / 6.0;
                    v[i][3] = u[i][0] / 24.0 + u[i][1] / 12.0 + u[i][2] / 6.0;
                    v[i][4] = u[i][0] / 24.0 - u[i][1] / 12.0 + u[i][2] / 6.0;
                    v[i][5] = u[i][2];
                }

                // Store as (36, ic, oc)
                for ai in 0..6 {
                    for aj in 0..6 {
                        let alpha = ai * 6 + aj;
                        out[alpha * ic * oc + ic_idx * oc + oc_idx] = f32_to_f16(v[ai][aj]);
                    }
                }
            }
        }
        out
    }

    pub struct MetalConv {
        device: Device,
        queue: CommandQueue,
        pipeline_f32w: ComputePipelineState,
        pipeline_basic: ComputePipelineState,
    }

    /// Full Metal inference backend with all op pipelines.
    pub struct MetalInference {
        pub device: Device,
        pub queue: CommandQueue,
        // Conv (f32 weight and f16 weight variants)
        pub conv_gemm: ComputePipelineState,
        pub conv_gemm_f16w: ComputePipelineState,
        pub conv_direct: ComputePipelineState,
        pub conv_direct_f16w: ComputePipelineState,
        // Conv f16 full I/O (f16 input + f16 weight + f16 output)
        pub conv_gemm_f16io: ComputePipelineState,
        pub conv_gemm_small_f16io: ComputePipelineState,
        pub conv_gemm_large_f16io: ComputePipelineState,
        pub conv_direct_f16io: ComputePipelineState,
        pub depthwise_conv_f16io: ComputePipelineState,
        // Simdgroup matrix conv (AMX accelerated)
        pub conv_gemm_simd_f16io: ComputePipelineState,
        // Elementwise / unary (f32)
        pub binary_elementwise: ComputePipelineState,
        pub broadcast_binary: ComputePipelineState,
        pub unary_op: ComputePipelineState,
        pub silu: ComputePipelineState,
        // Elementwise / unary (f16)
        pub binary_elementwise_f16: ComputePipelineState,
        pub broadcast_binary_f16: ComputePipelineState,
        pub unary_op_f16: ComputePipelineState,
        pub silu_f16: ComputePipelineState,
        // Structural (f32)
        pub concat_channels: ComputePipelineState,
        pub split_channels: ComputePipelineState,
        pub resize_nearest: ComputePipelineState,
        pub maxpool2d: ComputePipelineState,
        pub softmax: ComputePipelineState,
        pub transpose_2d: ComputePipelineState,
        pub slice_copy: ComputePipelineState,
        pub matmul: ComputePipelineState,
        pub permute_nhwc_to_nchw: ComputePipelineState,
        pub permute_nchw_to_nhwc: ComputePipelineState,
        pub permute_0213: ComputePipelineState,
        // Structural (f16)
        pub concat_channels_f16: ComputePipelineState,
        pub concat_channels_f16v4: ComputePipelineState,
        pub split_channels_f16: ComputePipelineState,
        pub split_fused_f16: ComputePipelineState,
        pub split_fused_f16v4: ComputePipelineState,
        pub resize_nearest_f16: ComputePipelineState,
        pub resize_nearest_f16v4: ComputePipelineState,
        pub maxpool2d_f16: ComputePipelineState,
        pub softmax_f16: ComputePipelineState,
        pub transpose_2d_f16: ComputePipelineState,
        pub slice_copy_f16: ComputePipelineState,
        pub matmul_f16io: ComputePipelineState,
        pub permute_nhwc_to_nchw_f16: ComputePipelineState,
        pub permute_nhwc_to_nchw_f16v4: ComputePipelineState,
        pub permute_nchw_to_nhwc_f16: ComputePipelineState,
        pub permute_nchw_to_nhwc_f16v4: ComputePipelineState,
        pub permute_0213_f16: ComputePipelineState,
        // Fused detection head kernel
        pub nhwc_to_flat_concat_f16: ComputePipelineState,
        pub channel_scatter_f16: ComputePipelineState,
        // Cast kernels (f32 ↔ f16 boundary)
        pub cast_f32_to_f16: ComputePipelineState,
        pub cast_f32_to_f16_nchw_to_nhwc: ComputePipelineState,
        pub cast_f16_to_f32: ComputePipelineState,
        // MPS support kernels
        pub bias_act_f16: ComputePipelineState,
        pub im2col_f16: ComputePipelineState,
        // Optimized conv v2 (BK=32, vectorized stores)
        pub conv_gemm_v2_f16io: ComputePipelineState,
        pub conv1x1_simd_f16io: ComputePipelineState,
        pub conv1x1_simd_bk32_f16io: ComputePipelineState,
        pub conv1x1_simd_4sg_f16io: ComputePipelineState,
        pub conv1x1_simd_large_f16io: ComputePipelineState,
        pub conv3x3_simd_f16io: ComputePipelineState,
        // Scalar softmax (one thread per row, no reduction)
        pub softmax_scalar_f16: ComputePipelineState,
        // Winograd F(2×2, 3×3) conv kernels
        pub winograd_input_transform_f16: ComputePipelineState,
        pub winograd_batched_gemm_f16io: ComputePipelineState,
        pub winograd_batched_gemm_simd_f16io: ComputePipelineState,
        pub winograd_output_transform_f16: ComputePipelineState,
        // Winograd F(4×4, 3×3) conv kernels
        pub winograd4x4_input_transform_f16: ComputePipelineState,
        pub winograd4x4_batched_gemm_f16io: ComputePipelineState,
        pub winograd4x4_output_transform_f16: ComputePipelineState,
        pub winograd4x4_output_transform_residual_f16: ComputePipelineState,
    }

    impl MetalInference {
        pub fn new() -> Option<Self> {
            let device = Device::system_default()?;
            let queue = device.new_command_queue();
            let options = CompileOptions::new();
            options.set_fast_math_enabled(true);

            let conv_lib = device
                .new_library_with_source(CONV_GEMM_BASIC_MSL, &options)
                .map_err(|e| eprintln!("Metal conv compile: {}", e))
                .ok()?;
            let simd_lib = device
                .new_library_with_source(CONV_GEMM_SIMD_MSL, &options)
                .map_err(|e| eprintln!("Metal simd compile: {}", e))
                .ok()?;
            let ops_lib = device
                .new_library_with_source(METAL_OPS_MSL, &options)
                .map_err(|e| eprintln!("Metal ops compile: {}", e))
                .ok()?;
            let wino_lib = device
                .new_library_with_source(CONV_WINOGRAD_MSL, &options)
                .map_err(|e| eprintln!("Metal winograd compile: {}", e))
                .ok()?;

            let pipe = |lib: &Library, name: &str| -> Option<ComputePipelineState> {
                let f = lib.get_function(name, None).ok()?;
                device
                    .new_compute_pipeline_state_with_function(&f)
                    .map_err(|e| eprintln!("Metal pipeline '{}': {}", name, e))
                    .ok()
            };

            Some(MetalInference {
                conv_gemm: pipe(&conv_lib, "conv_gemm_basic")?,
                conv_gemm_f16w: pipe(&conv_lib, "conv_gemm_basic_f16w")?,
                conv_direct: pipe(&conv_lib, "conv_direct")?,
                conv_direct_f16w: pipe(&conv_lib, "conv_direct_f16w")?,
                binary_elementwise: pipe(&ops_lib, "binary_elementwise")?,
                broadcast_binary: pipe(&ops_lib, "broadcast_binary")?,
                unary_op: pipe(&ops_lib, "unary_op")?,
                silu: pipe(&ops_lib, "silu")?,
                concat_channels: pipe(&ops_lib, "concat_channels")?,
                split_channels: pipe(&ops_lib, "split_channels")?,
                resize_nearest: pipe(&ops_lib, "resize_nearest")?,
                maxpool2d: pipe(&ops_lib, "maxpool2d")?,
                softmax: pipe(&ops_lib, "softmax")?,
                transpose_2d: pipe(&ops_lib, "transpose_2d")?,
                slice_copy: pipe(&ops_lib, "slice_copy")?,
                matmul: pipe(&ops_lib, "matmul")?,
                permute_nhwc_to_nchw: pipe(&ops_lib, "permute_nhwc_to_nchw")?,
                permute_nchw_to_nhwc: pipe(&ops_lib, "permute_nchw_to_nhwc")?,
                permute_0213: pipe(&ops_lib, "permute_0213")?,
                // F16 full I/O conv
                conv_gemm_f16io: pipe(&conv_lib, "conv_gemm_basic_f16io")?,
                conv_gemm_small_f16io: pipe(&conv_lib, "conv_gemm_small_f16io")?,
                conv_gemm_large_f16io: pipe(&conv_lib, "conv_gemm_large_f16io")?,
                conv_direct_f16io: pipe(&conv_lib, "conv_direct_f16io")?,
                depthwise_conv_f16io: pipe(&conv_lib, "depthwise_conv_f16io")?,
                conv_gemm_simd_f16io: pipe(&simd_lib, "conv_gemm_simd_f16io")?,
                // F16 elementwise / unary
                binary_elementwise_f16: pipe(&ops_lib, "binary_elementwise_f16")?,
                broadcast_binary_f16: pipe(&ops_lib, "broadcast_binary_f16")?,
                unary_op_f16: pipe(&ops_lib, "unary_op_f16")?,
                silu_f16: pipe(&ops_lib, "silu_f16")?,
                // F16 structural
                concat_channels_f16: pipe(&ops_lib, "concat_channels_f16")?,
                concat_channels_f16v4: pipe(&ops_lib, "concat_channels_f16v4")?,
                split_channels_f16: pipe(&ops_lib, "split_channels_f16")?,
                split_fused_f16: pipe(&ops_lib, "split_fused_f16")?,
                split_fused_f16v4: pipe(&ops_lib, "split_fused_f16v4")?,
                resize_nearest_f16: pipe(&ops_lib, "resize_nearest_f16")?,
                resize_nearest_f16v4: pipe(&ops_lib, "resize_nearest_f16v4")?,
                maxpool2d_f16: pipe(&ops_lib, "maxpool2d_f16")?,
                softmax_f16: pipe(&ops_lib, "softmax_f16")?,
                transpose_2d_f16: pipe(&ops_lib, "transpose_2d_f16")?,
                slice_copy_f16: pipe(&ops_lib, "slice_copy_f16")?,
                matmul_f16io: pipe(&ops_lib, "matmul_f16io")?,
                permute_nhwc_to_nchw_f16: pipe(&ops_lib, "permute_nhwc_to_nchw_f16")?,
                permute_nhwc_to_nchw_f16v4: pipe(&ops_lib, "permute_nhwc_to_nchw_f16v4")?,
                permute_nchw_to_nhwc_f16: pipe(&ops_lib, "permute_nchw_to_nhwc_f16")?,
                permute_nchw_to_nhwc_f16v4: pipe(&ops_lib, "permute_nchw_to_nhwc_f16v4")?,
                permute_0213_f16: pipe(&ops_lib, "permute_0213_f16")?,
                // Fused detection head
                nhwc_to_flat_concat_f16: pipe(&ops_lib, "nhwc_to_flat_concat_f16")?,
                channel_scatter_f16: pipe(&ops_lib, "channel_scatter_f16")?,
                // Cast kernels
                cast_f32_to_f16: pipe(&ops_lib, "cast_f32_to_f16")?,
                cast_f32_to_f16_nchw_to_nhwc: pipe(&ops_lib, "cast_f32_to_f16_nchw_to_nhwc")?,
                cast_f16_to_f32: pipe(&ops_lib, "cast_f16_to_f32")?,
                // MPS support kernels
                bias_act_f16: pipe(&ops_lib, "bias_act_f16")?,
                im2col_f16: pipe(&ops_lib, "im2col_f16")?,
                // Optimized conv v2
                conv_gemm_v2_f16io: pipe(&conv_lib, "conv_gemm_v2_f16io")?,
                conv1x1_simd_f16io: pipe(&conv_lib, "conv1x1_simd_f16io")?,
                conv1x1_simd_bk32_f16io: pipe(&conv_lib, "conv1x1_simd_bk32_f16io")?,
                conv1x1_simd_4sg_f16io: pipe(&conv_lib, "conv1x1_simd_4sg_f16io")?,
                conv1x1_simd_large_f16io: pipe(&conv_lib, "conv1x1_simd_large_f16io")?,
                conv3x3_simd_f16io: pipe(&conv_lib, "conv3x3_simd_f16io")?,
                // Scalar softmax
                softmax_scalar_f16: pipe(&ops_lib, "softmax_scalar_f16")?,
                // Winograd F(2×2, 3×3) conv
                winograd_input_transform_f16: pipe(&wino_lib, "winograd_input_transform_f16")?,
                winograd_batched_gemm_f16io: pipe(&wino_lib, "winograd_batched_gemm_f16io")?,
                winograd_batched_gemm_simd_f16io: pipe(
                    &wino_lib,
                    "winograd_batched_gemm_simd_f16io",
                )?,
                winograd_output_transform_f16: pipe(&wino_lib, "winograd_output_transform_f16")?,
                // Winograd F(4×4, 3×3) conv
                winograd4x4_input_transform_f16: pipe(
                    &wino_lib,
                    "winograd4x4_input_transform_f16",
                )?,
                winograd4x4_batched_gemm_f16io: pipe(&wino_lib, "winograd4x4_batched_gemm_f16io")?,
                winograd4x4_output_transform_f16: pipe(
                    &wino_lib,
                    "winograd4x4_output_transform_f16",
                )?,
                winograd4x4_output_transform_residual_f16: pipe(
                    &wino_lib,
                    "winograd4x4_output_transform_residual_f16",
                )?,
                device,
                queue,
            })
        }

        pub fn device_name(&self) -> String {
            self.device.name().to_string()
        }

        pub fn buffer_from_f32(&self, data: &[f32]) -> Buffer {
            let bytes =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
            self.device.new_buffer_with_data(
                bytes.as_ptr() as *const _,
                bytes.len() as u64,
                MTLResourceOptions::StorageModeShared,
            )
        }

        /// Pre-convert f32 data to f16 and upload. Halves weight bandwidth on GPU.
        pub fn buffer_from_f32_as_f16(&self, data: &[f32]) -> Buffer {
            let f16_data: Vec<u16> = data.iter().map(|&v| f32_to_f16(v)).collect();
            let bytes = unsafe {
                std::slice::from_raw_parts(f16_data.as_ptr() as *const u8, f16_data.len() * 2)
            };
            self.device.new_buffer_with_data(
                bytes.as_ptr() as *const _,
                bytes.len() as u64,
                MTLResourceOptions::StorageModeShared,
            )
        }

        pub fn output_buffer(&self, num_f32: usize) -> Buffer {
            let len = if num_f32 == 0 { 4 } else { num_f32 * 4 };
            self.device
                .new_buffer(len as u64, MTLResourceOptions::StorageModeShared)
        }

        /// Allocate an f16 buffer for intermediates. Uses StorageModeShared
        /// (unified memory on Apple Silicon — no copy overhead, CPU-accessible
        /// for debugging).
        pub fn output_buffer_f16(&self, num_f16: usize) -> Buffer {
            let len = if num_f16 == 0 { 4 } else { num_f16 * 2 };
            self.device
                .new_buffer(len as u64, MTLResourceOptions::StorageModeShared)
        }

        /// Allocate a CPU-accessible f16 buffer. Use for input upload and
        /// output readback (anywhere CPU needs to read/write the buffer).
        pub fn output_buffer_f16_shared(&self, num_f16: usize) -> Buffer {
            let len = if num_f16 == 0 { 4 } else { num_f16 * 2 };
            self.device
                .new_buffer(len as u64, MTLResourceOptions::StorageModeShared)
        }

        /// Allocate a GPU-private f16 buffer. Alias for output_buffer_f16.
        pub fn output_buffer_f16_private(&self, num_f16: usize) -> Buffer {
            self.output_buffer_f16(num_f16)
        }

        pub fn read_buffer_f32(&self, buf: &Buffer, count: usize) -> Vec<f32> {
            debug_assert!(
                count * 4 <= buf.length() as usize,
                "read_buffer_f32: count={} exceeds buffer length={}",
                count,
                buf.length()
            );
            let ptr = buf.contents() as *const f32;
            unsafe { std::slice::from_raw_parts(ptr, count).to_vec() }
        }

        pub fn write_buffer_f32(&self, buf: &Buffer, data: &[f32]) {
            debug_assert!(
                data.len() * 4 <= buf.length() as usize,
                "write_buffer_f32: data.len()={} exceeds buffer length={}",
                data.len(),
                buf.length()
            );
            let dst = buf.contents() as *mut f32;
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
            }
        }

        /// Write f32 NCHW data as f16 NHWC directly into an f16 buffer.
        /// Uses NEON FCVTN for fast f32→f16 conversion + NCHW→NHWC permutation.
        /// `data` layout: [batch, channels, height, width].
        pub fn write_buffer_f32_nchw_as_f16_nhwc(
            &self,
            buf: &Buffer,
            data: &[f32],
            batch: usize,
            c: usize,
            h: usize,
            w: usize,
        ) {
            let spatial = h * w;
            let n = batch * c * spatial;
            debug_assert_eq!(data.len(), n);
            debug_assert!(
                n * 2 <= buf.length() as usize,
                "write_buffer_f32_nchw_as_f16_nhwc: n={} exceeds buffer",
                n
            );
            let dst = buf.contents() as *mut u16;

            // Fast path for common image case: C=3
            // NEON: fcvtn (f32→f16) + st3 (interleave 3 channels into NHWC)
            if c == 3 {
                for b_idx in 0..batch {
                    let src_base = b_idx * 3 * spatial;
                    let dst_base = b_idx * spatial * 3;
                    let src0 = &data[src_base..src_base + spatial];
                    let src1 = &data[src_base + spatial..src_base + 2 * spatial];
                    let src2 = &data[src_base + 2 * spatial..src_base + 3 * spatial];

                    // Process 4 spatial positions at a time using NEON fcvtn + st3
                    let chunks = spatial / 4;
                    for chunk in 0..chunks {
                        let s = chunk * 4;
                        unsafe {
                            #[cfg(target_arch = "aarch64")]
                            {
                                use std::arch::asm;
                                // fcvtn: 4×f32 → 4×f16 for each channel
                                // st3: interleave-store {v0.4h, v1.4h, v2.4h} as R0G0B0 R1G1B1 ...
                                asm!(
                                    "ldr q0, [{s0}]",
                                    "ldr q1, [{s1}]",
                                    "ldr q2, [{s2}]",
                                    "fcvtn v0.4h, v0.4s",
                                    "fcvtn v1.4h, v1.4s",
                                    "fcvtn v2.4h, v2.4s",
                                    "st3 {{v0.4h, v1.4h, v2.4h}}, [{d}]",
                                    s0 = in(reg) src0.as_ptr().add(s),
                                    s1 = in(reg) src1.as_ptr().add(s),
                                    s2 = in(reg) src2.as_ptr().add(s),
                                    d = in(reg) dst.add(dst_base + s * 3),
                                    out("v0") _, out("v1") _, out("v2") _,
                                );
                            }
                            #[cfg(not(target_arch = "aarch64"))]
                            {
                                let d = dst.add(dst_base + s * 3);
                                for i in 0..4usize {
                                    *d.add(i * 3) = f32_to_f16(src0[s + i]);
                                    *d.add(i * 3 + 1) = f32_to_f16(src1[s + i]);
                                    *d.add(i * 3 + 2) = f32_to_f16(src2[s + i]);
                                }
                            }
                        }
                    }
                    // Handle remaining spatial positions
                    for s in (chunks * 4)..spatial {
                        unsafe {
                            let d = dst.add(dst_base + s * 3);
                            *d.add(0) = f32_to_f16(src0[s]);
                            *d.add(1) = f32_to_f16(src1[s]);
                            *d.add(2) = f32_to_f16(src2[s]);
                        }
                    }
                }
                return;
            }

            // General path: any number of channels
            for b_idx in 0..batch {
                let src_batch = b_idx * c * spatial;
                let dst_batch = b_idx * spatial * c;
                for ch in 0..c {
                    let src_ch = src_batch + ch * spatial;
                    let dst_ch_off = dst_batch + ch;
                    // Process 4 at a time with NEON
                    let chunks = spatial / 4;
                    for chunk in 0..chunks {
                        let s = chunk * 4;
                        unsafe {
                            let mut h4 = [0u16; 4];
                            #[cfg(target_arch = "aarch64")]
                            {
                                use std::arch::asm;
                                asm!(
                                    "ldr q0, [{src}]",
                                    "fcvtn v0.4h, v0.4s",
                                    "str d0, [{dst}]",
                                    src = in(reg) data.as_ptr().add(src_ch + s),
                                    dst = in(reg) h4.as_mut_ptr(),
                                    out("v0") _,
                                );
                            }
                            #[cfg(not(target_arch = "aarch64"))]
                            {
                                for i in 0..4 {
                                    h4[i] = f32_to_f16(data[src_ch + s + i]);
                                }
                            }
                            for i in 0..4 {
                                *dst.add(dst_ch_off + (s + i) * c) = h4[i];
                            }
                        }
                    }
                    for s in (chunks * 4)..spatial {
                        unsafe {
                            *dst.add(dst_ch_off + s * c) = f32_to_f16(data[src_ch + s]);
                        }
                    }
                }
            }
        }

        /// Read f16 buffer as f32 values (converts on the fly).
        pub fn read_buffer_f16_as_f32(&self, buf: &Buffer, count: usize) -> Vec<f32> {
            debug_assert!(
                count * 2 <= buf.length() as usize,
                "read_buffer_f16_as_f32: count={} exceeds buffer length={}",
                count,
                buf.length()
            );
            let ptr = buf.contents() as *const u16;
            let raw = unsafe { std::slice::from_raw_parts(ptr, count) };
            raw.iter().map(|&bits| f16_to_f32(bits)).collect()
        }
    }

    fn f16_to_f32(bits: u16) -> f32 {
        let sign = ((bits >> 15) & 1) as u32;
        let exp = ((bits >> 10) & 0x1F) as u32;
        let man = (bits & 0x3FF) as u32;
        if exp == 31 {
            // Inf/NaN
            let f_bits = (sign << 31) | 0x7F800000 | (man << 13);
            return f32::from_bits(f_bits);
        }
        if exp == 0 {
            if man == 0 {
                return f32::from_bits(sign << 31); // ±0
            }
            // Denormal
            let mut m = man;
            let mut e = 0i32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e += 1;
            }
            m &= 0x3FF;
            let f_exp = (127 - 15 - e) as u32;
            let f_bits = (sign << 31) | (f_exp << 23) | (m << 13);
            return f32::from_bits(f_bits);
        }
        let f_exp = (exp as i32 - 15 + 127) as u32;
        let f_bits = (sign << 31) | (f_exp << 23) | (man << 13);
        f32::from_bits(f_bits)
    }

    // ── MPS GEMM: Apple's optimized matrix multiply ──
    // Uses MPSMatrixMultiplication for much higher GPU utilization than custom kernels.
    // Note: `msg_send!` macro from `objc` crate triggers `unexpected_cfgs` warnings
    // for `cargo-clippy` — this is an upstream issue, harmless.
    // C = alpha * A * B + beta * C, where A=[M,K], B=[K,N], C=[M,N], all f16.

    /// Perform MPS GEMM on f16 buffers: out[M,N] = A[M,K] * B[K,N] (f16 I/O)
    /// Encodes into a NEW command buffer (MPS requires its own encoding).
    /// Returns immediately; caller must commit/wait.
    /// Perform MPS GEMM on f16 buffers: out[M,N] = alpha * A[M,K] * B[K,N] + beta * C
    /// Creates+commits its own command buffer (MPS needs its own encoding).
    #[allow(unexpected_cfgs)]
    pub fn mps_gemm_f16(
        device: &Device,
        queue: &CommandQueue,
        a: &Buffer,
        b: &Buffer,
        out: &Buffer,
        m: u32,
        n: u32,
        k: u32,
        alpha: f64,
        beta: f64,
        transpose_a: bool,
        transpose_b: bool,
    ) -> Result<(), KernelError> {
        use objc::runtime::{Class, NO, Object, YES};

        // autoreleasepool drains MPSMatrixDescriptor factory objects (autoreleased)
        autoreleasepool(|| unsafe {
            // MPSDataType.float16 = 0x10000000 | 16
            let mps_float16: u32 = 0x10000000 | 16;

            let mps_matrix_desc_cls =
                Class::get("MPSMatrixDescriptor").ok_or_else(|| KernelError::Gpu {
                    message: "MPSMatrixDescriptor class not available".into(),
                })?;

            let row_bytes_a = if transpose_a {
                m as u64 * 2
            } else {
                k as u64 * 2
            };
            let rows_a = if transpose_a { k as u64 } else { m as u64 };
            let cols_a = if transpose_a { m as u64 } else { k as u64 };
            let desc_a: *mut Object = msg_send![mps_matrix_desc_cls,
                matrixDescriptorWithRows: rows_a
                columns: cols_a
                rowBytes: row_bytes_a
                dataType: mps_float16];

            let row_bytes_b = if transpose_b {
                k as u64 * 2
            } else {
                n as u64 * 2
            };
            let rows_b = if transpose_b { n as u64 } else { k as u64 };
            let cols_b = if transpose_b { k as u64 } else { n as u64 };
            let desc_b: *mut Object = msg_send![mps_matrix_desc_cls,
                matrixDescriptorWithRows: rows_b
                columns: cols_b
                rowBytes: row_bytes_b
                dataType: mps_float16];

            let desc_c: *mut Object = msg_send![mps_matrix_desc_cls,
                matrixDescriptorWithRows: m as u64
                columns: n as u64
                rowBytes: (n as u64) * 2
                dataType: mps_float16];

            // MPSMatrix from buffer + descriptor
            let mps_matrix_cls = Class::get("MPSMatrix").ok_or_else(|| KernelError::Gpu {
                message: "MPSMatrix class not available".into(),
            })?;
            let alloc_a: *mut Object = msg_send![mps_matrix_cls, alloc];
            let mat_a: *mut Object = msg_send![alloc_a,
                initWithBuffer: a.as_ptr() descriptor: desc_a];
            let alloc_b: *mut Object = msg_send![mps_matrix_cls, alloc];
            let mat_b: *mut Object = msg_send![alloc_b,
                initWithBuffer: b.as_ptr() descriptor: desc_b];
            let alloc_c: *mut Object = msg_send![mps_matrix_cls, alloc];
            let mat_c: *mut Object = msg_send![alloc_c,
                initWithBuffer: out.as_ptr() descriptor: desc_c];

            // MPSMatrixMultiplication
            let mps_mm_cls =
                Class::get("MPSMatrixMultiplication").ok_or_else(|| KernelError::Gpu {
                    message: "MPSMatrixMultiplication class not available".into(),
                })?;
            let ta = if transpose_a { YES } else { NO };
            let tb = if transpose_b { YES } else { NO };
            let alloc_mm: *mut Object = msg_send![mps_mm_cls, alloc];
            let mm: *mut Object = msg_send![alloc_mm,
                initWithDevice: device.as_ptr()
                transposeLeft: ta
                transposeRight: tb
                resultRows: m as u64
                resultColumns: n as u64
                interiorColumns: k as u64
                alpha: alpha
                beta: beta];

            // Encode into command buffer
            let cmd = queue.new_command_buffer();
            let _: () = msg_send![mm, encodeToCommandBuffer: cmd.as_ptr()
                leftMatrix: mat_a
                rightMatrix: mat_b
                resultMatrix: mat_c];

            cmd.commit();
            cmd.wait_until_completed();

            let _: () = msg_send![mm, release];
            let _: () = msg_send![mat_a, release];
            let _: () = msg_send![mat_b, release];
            let _: () = msg_send![mat_c, release];
            Ok(())
        })
    }

    /// Encode MPS GEMM into an existing command buffer (does NOT commit).
    /// For batching multiple MPS ops into one submission.
    #[allow(unexpected_cfgs)]
    pub fn mps_gemm_f16_encode(
        device: &Device,
        cmd: &CommandBufferRef,
        a: &Buffer,
        b: &Buffer,
        out: &Buffer,
        m: u32,
        n: u32,
        k: u32,
        alpha: f64,
        beta: f64,
        transpose_a: bool,
        transpose_b: bool,
    ) -> Result<(), KernelError> {
        use objc::runtime::{Class, NO, Object, YES};

        // autoreleasepool drains MPSMatrixDescriptor factory objects (autoreleased)
        autoreleasepool(|| unsafe {
            let mps_float16: u32 = 0x10000000 | 16;
            let mps_matrix_desc_cls =
                Class::get("MPSMatrixDescriptor").ok_or_else(|| KernelError::Gpu {
                    message: "MPSMatrixDescriptor class not available".into(),
                })?;

            let row_bytes_a = if transpose_a {
                m as u64 * 2
            } else {
                k as u64 * 2
            };
            let rows_a = if transpose_a { k as u64 } else { m as u64 };
            let cols_a = if transpose_a { m as u64 } else { k as u64 };
            let desc_a: *mut Object = msg_send![mps_matrix_desc_cls,
                matrixDescriptorWithRows: rows_a
                columns: cols_a
                rowBytes: row_bytes_a
                dataType: mps_float16];

            let row_bytes_b = if transpose_b {
                k as u64 * 2
            } else {
                n as u64 * 2
            };
            let rows_b = if transpose_b { n as u64 } else { k as u64 };
            let cols_b = if transpose_b { k as u64 } else { n as u64 };
            let desc_b: *mut Object = msg_send![mps_matrix_desc_cls,
                matrixDescriptorWithRows: rows_b
                columns: cols_b
                rowBytes: row_bytes_b
                dataType: mps_float16];

            let desc_c: *mut Object = msg_send![mps_matrix_desc_cls,
                matrixDescriptorWithRows: m as u64
                columns: n as u64
                rowBytes: (n as u64) * 2
                dataType: mps_float16];

            let mps_matrix_cls = Class::get("MPSMatrix").ok_or_else(|| KernelError::Gpu {
                message: "MPSMatrix class not available".into(),
            })?;
            let alloc_a: *mut Object = msg_send![mps_matrix_cls, alloc];
            let mat_a: *mut Object = msg_send![alloc_a,
                initWithBuffer: a.as_ptr() descriptor: desc_a];
            let alloc_b: *mut Object = msg_send![mps_matrix_cls, alloc];
            let mat_b: *mut Object = msg_send![alloc_b,
                initWithBuffer: b.as_ptr() descriptor: desc_b];
            let alloc_c: *mut Object = msg_send![mps_matrix_cls, alloc];
            let mat_c: *mut Object = msg_send![alloc_c,
                initWithBuffer: out.as_ptr() descriptor: desc_c];

            let mps_mm_cls =
                Class::get("MPSMatrixMultiplication").ok_or_else(|| KernelError::Gpu {
                    message: "MPSMatrixMultiplication class not available".into(),
                })?;
            let ta = if transpose_a { YES } else { NO };
            let tb = if transpose_b { YES } else { NO };
            let alloc_mm: *mut Object = msg_send![mps_mm_cls, alloc];
            let mm: *mut Object = msg_send![alloc_mm,
                initWithDevice: device.as_ptr()
                transposeLeft: ta
                transposeRight: tb
                resultRows: m as u64
                resultColumns: n as u64
                interiorColumns: k as u64
                alpha: alpha
                beta: beta];

            let _: () = msg_send![mm, encodeToCommandBuffer: cmd.as_ptr()
                leftMatrix: mat_a
                rightMatrix: mat_b
                resultMatrix: mat_c];

            let _: () = msg_send![mm, release];
            let _: () = msg_send![mat_a, release];
            let _: () = msg_send![mat_b, release];
            let _: () = msg_send![mat_c, release];
            Ok(())
        })
    }

    // ── Encoder helper: records ops into a compute command encoder ──
    pub struct MetalEncoder<'a> {
        pub enc: &'a ComputeCommandEncoderRef,
        pub inf: &'a MetalInference,
    }

    impl<'a> MetalEncoder<'a> {
        pub fn new(enc: &'a ComputeCommandEncoderRef, inf: &'a MetalInference) -> Self {
            MetalEncoder { enc, inf }
        }

        fn dispatch_1d(&self, pipeline: &ComputePipelineState, n: u32) {
            self.enc.set_compute_pipeline_state(pipeline);
            let wg = 256u64;
            let groups = (n as u64).div_ceil(wg);
            self.enc.dispatch_thread_groups(
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

        pub fn conv_gemm(
            &self,
            input: &Buffer,
            weight: &Buffer,
            bias: &Buffer,
            output: &Buffer,
            params: &ConvParams,
        ) {
            self.enc
                .set_compute_pipeline_state(&self.inf.conv_gemm_f16w);
            self.enc.set_buffer(0, Some(input), 0);
            self.enc.set_buffer(1, Some(weight), 0);
            self.enc.set_buffer(2, Some(bias), 0);
            self.enc.set_buffer(3, Some(output), 0);
            self.enc.set_bytes(
                4,
                mem::size_of::<ConvParams>() as u64,
                params as *const ConvParams as *const _,
            );

            let bm = 64u32;
            let bn = 64u32;
            let wg_x = params.n_out.div_ceil(bn);
            let wg_y = params.m.div_ceil(bm);

            self.enc.dispatch_thread_groups(
                MTLSize {
                    width: wg_x as u64,
                    height: wg_y as u64,
                    depth: 1,
                },
                MTLSize {
                    width: 16,
                    height: 16,
                    depth: 1,
                },
            );
        }

        /// Direct per-pixel conv — 2D dispatch: (M, ceil(n_out/4)).
        /// Best for small n_out (≤ 64) or small K.
        pub fn conv_direct(
            &self,
            input: &Buffer,
            weight: &Buffer,
            bias: &Buffer,
            output: &Buffer,
            params: &ConvParams,
        ) {
            self.enc
                .set_compute_pipeline_state(&self.inf.conv_direct_f16w);
            self.enc.set_buffer(0, Some(input), 0);
            self.enc.set_buffer(1, Some(weight), 0);
            self.enc.set_buffer(2, Some(bias), 0);
            self.enc.set_buffer(3, Some(output), 0);
            self.enc.set_bytes(
                4,
                mem::size_of::<ConvParams>() as u64,
                params as *const ConvParams as *const _,
            );

            let threads_x = params.m as u64;
            let threads_y = params.n_out.div_ceil(4) as u64;
            let wg_x = 16u64.min(threads_x);
            let wg_y = 16u64.min(threads_y);
            let groups_x = threads_x.div_ceil(wg_x);
            let groups_y = threads_y.div_ceil(wg_y);

            self.enc.dispatch_thread_groups(
                MTLSize {
                    width: groups_x,
                    height: groups_y,
                    depth: 1,
                },
                MTLSize {
                    width: wg_x,
                    height: wg_y,
                    depth: 1,
                },
            );
        }

        /// op: 0=add, 1=sub, 2=mul, 3=div
        pub fn binary(&self, a: &Buffer, b: &Buffer, out: &Buffer, n: u32, op: u32) {
            #[repr(C)]
            struct P {
                n: u32,
                op: u32,
            }
            let p = P { n, op };
            self.enc
                .set_compute_pipeline_state(&self.inf.binary_elementwise);
            self.enc.set_buffer(0, Some(a), 0);
            self.enc.set_buffer(1, Some(b), 0);
            self.enc.set_buffer(2, Some(out), 0);
            self.enc
                .set_bytes(3, mem::size_of::<P>() as u64, &p as *const P as *const _);
            self.dispatch_1d(&self.inf.binary_elementwise, n);
        }

        pub fn broadcast_binary(
            &self,
            a: &Buffer,
            b: &Buffer,
            out: &Buffer,
            n: u32,
            broadcast_dim: u32,
            op: u32,
        ) {
            #[repr(C)]
            struct P {
                n: u32,
                broadcast_dim: u32,
                op: u32,
            }
            let p = P {
                n,
                broadcast_dim,
                op,
            };
            self.enc
                .set_compute_pipeline_state(&self.inf.broadcast_binary);
            self.enc.set_buffer(0, Some(a), 0);
            self.enc.set_buffer(1, Some(b), 0);
            self.enc.set_buffer(2, Some(out), 0);
            self.enc
                .set_bytes(3, mem::size_of::<P>() as u64, &p as *const P as *const _);
            self.dispatch_1d(&self.inf.broadcast_binary, n);
        }

        /// op: 0=relu, 1=sigmoid, 2=silu, 3=neg, 4=exp, 5=sqrt, 6=tanh
        pub fn unary(&self, input: &Buffer, out: &Buffer, n: u32, op: u32) {
            #[repr(C)]
            struct P {
                n: u32,
                op: u32,
            }
            let p = P { n, op };
            self.enc.set_compute_pipeline_state(&self.inf.unary_op);
            self.enc.set_buffer(0, Some(input), 0);
            self.enc.set_buffer(1, Some(out), 0);
            self.enc
                .set_bytes(2, mem::size_of::<P>() as u64, &p as *const P as *const _);
            self.dispatch_1d(&self.inf.unary_op, n);
        }

        pub fn silu(&self, input: &Buffer, out: &Buffer, n: u32) {
            self.enc.set_compute_pipeline_state(&self.inf.silu);
            self.enc.set_buffer(0, Some(input), 0);
            self.enc.set_buffer(1, Some(out), 0);
            self.enc.set_bytes(2, 4, &n as *const u32 as *const _);
            self.dispatch_1d(&self.inf.silu, n);
        }

        pub fn maxpool2d(
            &self,
            input: &Buffer,
            out: &Buffer,
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
        ) {
            #[repr(C)]
            struct P {
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
            }
            let p = P {
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
            };
            self.enc.set_compute_pipeline_state(&self.inf.maxpool2d);
            self.enc.set_buffer(0, Some(input), 0);
            self.enc.set_buffer(1, Some(out), 0);
            self.enc
                .set_bytes(2, mem::size_of::<P>() as u64, &p as *const P as *const _);
            let n = batch * oh * ow * ic;
            self.dispatch_1d(&self.inf.maxpool2d, n);
        }

        pub fn resize_nearest(
            &self,
            input: &Buffer,
            out: &Buffer,
            batch: u32,
            ih: u32,
            iw: u32,
            ic: u32,
            oh: u32,
            ow: u32,
            scale_h: f32,
            scale_w: f32,
        ) {
            #[repr(C)]
            struct P {
                batch: u32,
                ih: u32,
                iw: u32,
                ic: u32,
                oh: u32,
                ow: u32,
                scale_h: f32,
                scale_w: f32,
            }
            let p = P {
                batch,
                ih,
                iw,
                ic,
                oh,
                ow,
                scale_h,
                scale_w,
            };
            self.enc
                .set_compute_pipeline_state(&self.inf.resize_nearest);
            self.enc.set_buffer(0, Some(input), 0);
            self.enc.set_buffer(1, Some(out), 0);
            self.enc
                .set_bytes(2, mem::size_of::<P>() as u64, &p as *const P as *const _);
            let n = batch * oh * ow * ic;
            self.dispatch_1d(&self.inf.resize_nearest, n);
        }

        pub fn softmax(&self, input: &Buffer, out: &Buffer, outer: u32, dim: u32) {
            #[repr(C)]
            struct P {
                outer: u32,
                dim: u32,
            }
            let p = P { outer, dim };
            self.enc.set_compute_pipeline_state(&self.inf.softmax);
            self.enc.set_buffer(0, Some(input), 0);
            self.enc.set_buffer(1, Some(out), 0);
            self.enc
                .set_bytes(2, mem::size_of::<P>() as u64, &p as *const P as *const _);
            // One threadgroup per row. Use fewer threads for small dims to reduce overhead.
            let tg_size = if dim <= 32 {
                32u64
            } else if dim <= 128 {
                128u64
            } else {
                256u64
            };
            self.enc.dispatch_thread_groups(
                MTLSize {
                    width: outer as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: tg_size,
                    height: 1,
                    depth: 1,
                },
            );
        }

        pub fn transpose_2d(&self, input: &Buffer, out: &Buffer, rows: u32, cols: u32) {
            #[repr(C)]
            struct P {
                rows: u32,
                cols: u32,
            }
            let p = P { rows, cols };
            self.enc.set_compute_pipeline_state(&self.inf.transpose_2d);
            self.enc.set_buffer(0, Some(input), 0);
            self.enc.set_buffer(1, Some(out), 0);
            self.enc
                .set_bytes(2, mem::size_of::<P>() as u64, &p as *const P as *const _);
            let n = rows * cols;
            self.dispatch_1d(&self.inf.transpose_2d, n);
        }

        pub fn split_channels(
            &self,
            input: &Buffer,
            out: &Buffer,
            spatial: u32,
            in_c: u32,
            out_c: u32,
            offset_c: u32,
        ) {
            #[repr(C)]
            struct P {
                spatial: u32,
                in_c: u32,
                out_c: u32,
                offset_c: u32,
            }
            let p = P {
                spatial,
                in_c,
                out_c,
                offset_c,
            };
            self.enc
                .set_compute_pipeline_state(&self.inf.split_channels);
            self.enc.set_buffer(0, Some(input), 0);
            self.enc.set_buffer(1, Some(out), 0);
            self.enc
                .set_bytes(2, mem::size_of::<P>() as u64, &p as *const P as *const _);
            let n = spatial * out_c;
            self.dispatch_1d(&self.inf.split_channels, n);
        }

        /// Permute [0,2,1,3]: swap dim1 and dim2 of 4D tensor [d0,d1,d2,d3] → [d0,d2,d1,d3]
        pub fn permute_0213(
            &self,
            input: &Buffer,
            out: &Buffer,
            d0: u32,
            d1: u32,
            d2: u32,
            d3: u32,
        ) {
            #[repr(C)]
            struct P {
                d0: u32,
                d1: u32,
                d2: u32,
                d3: u32,
            }
            let p = P { d0, d1, d2, d3 };
            self.enc.set_compute_pipeline_state(&self.inf.permute_0213);
            self.enc.set_buffer(0, Some(input), 0);
            self.enc.set_buffer(1, Some(out), 0);
            self.enc
                .set_bytes(2, mem::size_of::<P>() as u64, &p as *const P as *const _);
            let total = d0 * d1 * d2 * d3;
            self.dispatch_1d(&self.inf.permute_0213, total);
        }

        pub fn permute_nhwc_to_nchw(
            &self,
            input: &Buffer,
            out: &Buffer,
            n: u32,
            h: u32,
            w: u32,
            c: u32,
        ) {
            #[repr(C)]
            struct P {
                n: u32,
                h: u32,
                w: u32,
                c: u32,
            }
            let p = P { n, h, w, c };
            self.enc
                .set_compute_pipeline_state(&self.inf.permute_nhwc_to_nchw);
            self.enc.set_buffer(0, Some(input), 0);
            self.enc.set_buffer(1, Some(out), 0);
            self.enc
                .set_bytes(2, mem::size_of::<P>() as u64, &p as *const P as *const _);
            let total = n * h * w * c;
            self.dispatch_1d(&self.inf.permute_nhwc_to_nchw, total);
        }

        pub fn permute_nchw_to_nhwc(
            &self,
            input: &Buffer,
            out: &Buffer,
            n: u32,
            h: u32,
            w: u32,
            c: u32,
        ) {
            #[repr(C)]
            struct P {
                n: u32,
                h: u32,
                w: u32,
                c: u32,
            }
            let p = P { n, h, w, c };
            self.enc
                .set_compute_pipeline_state(&self.inf.permute_nchw_to_nhwc);
            self.enc.set_buffer(0, Some(input), 0);
            self.enc.set_buffer(1, Some(out), 0);
            self.enc
                .set_bytes(2, mem::size_of::<P>() as u64, &p as *const P as *const _);
            let total = n * h * w * c;
            self.dispatch_1d(&self.inf.permute_nchw_to_nhwc, total);
        }

        pub fn matmul(&self, a: &Buffer, b: &Buffer, out: &Buffer, m: u32, n: u32, k: u32) {
            #[repr(C)]
            struct P {
                m: u32,
                n: u32,
                k: u32,
            }
            let p = P { m, n, k };
            self.enc.set_compute_pipeline_state(&self.inf.matmul);
            self.enc.set_buffer(0, Some(a), 0);
            self.enc.set_buffer(1, Some(b), 0);
            self.enc.set_buffer(2, Some(out), 0);
            self.enc
                .set_bytes(3, mem::size_of::<P>() as u64, &p as *const P as *const _);
            let bm = 64u32;
            let bn = 64u32;
            let wg_x = n.div_ceil(bn);
            let wg_y = m.div_ceil(bm);
            self.enc.dispatch_thread_groups(
                MTLSize {
                    width: wg_x as u64,
                    height: wg_y as u64,
                    depth: 1,
                },
                MTLSize {
                    width: 16,
                    height: 16,
                    depth: 1,
                },
            );
        }

        // ══════════════════════════════════════════════════════════
        // F16 I/O encoder methods — same signatures, f16 pipelines
        // ══════════════════════════════════════════════════════════

        pub fn cast_f32_to_f16(&self, input: &Buffer, out: &Buffer, n: u32) {
            self.enc
                .set_compute_pipeline_state(&self.inf.cast_f32_to_f16);
            self.enc.set_buffer(0, Some(input), 0);
            self.enc.set_buffer(1, Some(out), 0);
            self.enc.set_bytes(2, 4, &n as *const u32 as *const _);
            self.dispatch_1d(&self.inf.cast_f32_to_f16, n);
        }

        pub fn cast_f32_to_f16_nchw_to_nhwc(
            &self,
            input: &Buffer,
            out: &Buffer,
            n: u32,
            c: u32,
            h: u32,
            w: u32,
        ) {
            #[repr(C)]
            struct P {
                n: u32,
                c: u32,
                h: u32,
                w: u32,
            }
            let p = P { n, c, h, w };
            self.enc
                .set_compute_pipeline_state(&self.inf.cast_f32_to_f16_nchw_to_nhwc);
            self.enc.set_buffer(0, Some(input), 0);
            self.enc.set_buffer(1, Some(out), 0);
            self.enc.set_bytes(
                2,
                std::mem::size_of::<P>() as u64,
                &p as *const P as *const _,
            );
            let total = n * c * h * w;
            self.dispatch_1d(&self.inf.cast_f32_to_f16_nchw_to_nhwc, total);
        }

        pub fn cast_f16_to_f32(&self, input: &Buffer, out: &Buffer, n: u32) {
            self.enc
                .set_compute_pipeline_state(&self.inf.cast_f16_to_f32);
            self.enc.set_buffer(0, Some(input), 0);
            self.enc.set_buffer(1, Some(out), 0);
            self.enc.set_bytes(2, 4, &n as *const u32 as *const _);
            self.dispatch_1d(&self.inf.cast_f16_to_f32, n);
        }

        pub fn conv_gemm_f16io(
            &self,
            input: &Buffer,
            weight: &Buffer,
            bias: &Buffer,
            output: &Buffer,
            params: &ConvParams,
        ) {
            self.enc
                .set_compute_pipeline_state(&self.inf.conv_gemm_f16io);
            self.enc.set_buffer(0, Some(input), 0);
            self.enc.set_buffer(1, Some(weight), 0);
            self.enc.set_buffer(2, Some(bias), 0);
            self.enc.set_buffer(3, Some(output), 0);
            self.enc.set_bytes(
                4,
                mem::size_of::<ConvParams>() as u64,
                params as *const ConvParams as *const _,
            );
            let bm = 64u32;
            let bn = 64u32;
            let wg_x = params.n_out.div_ceil(bn);
            let wg_y = params.m.div_ceil(bm);
            self.enc.dispatch_thread_groups(
                MTLSize {
                    width: wg_x as u64,
                    height: wg_y as u64,
                    depth: 1,
                },
                MTLSize {
                    width: 16,
                    height: 16,
                    depth: 1,
                },
            );
        }

        /// SIMD 1×1 conv GEMM: uses simdgroup_multiply_accumulate (half×half→float).
        /// 256 threads = 8 simdgroups in 2×4 layout over 64×64 tile.
        pub fn conv1x1_simd_f16io(
            &self,
            input: &Buffer,
            weight: &Buffer,
            bias: &Buffer,
            output: &Buffer,
            params: &ConvParams,
        ) {
            self.enc
                .set_compute_pipeline_state(&self.inf.conv1x1_simd_f16io);
            self.enc.set_buffer(0, Some(input), 0);
            self.enc.set_buffer(1, Some(weight), 0);
            self.enc.set_buffer(2, Some(bias), 0);
            self.enc.set_buffer(3, Some(output), 0);
            self.enc.set_bytes(
                4,
                mem::size_of::<ConvParams>() as u64,
                params as *const ConvParams as *const _,
            );
            // Buffer 5: residual (dummy when has_residual=0)
            self.enc.set_buffer(5, Some(output), 0);
            let bm = 64u32;
            let bn = 64u32;
            let wg_x = params.n_out.div_ceil(bn);
            let wg_y = params.m.div_ceil(bm);
            self.enc.dispatch_thread_groups(
                MTLSize {
                    width: wg_x as u64,
                    height: wg_y as u64,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
        }

        /// SIMD 1×1 conv with fused residual add before activation.
        pub fn conv1x1_simd_res_f16io(
            &self,
            input: &Buffer,
            weight: &Buffer,
            bias: &Buffer,
            output: &Buffer,
            residual: &Buffer,
            params: &ConvParams,
        ) {
            self.enc
                .set_compute_pipeline_state(&self.inf.conv1x1_simd_f16io);
            self.enc.set_buffer(0, Some(input), 0);
            self.enc.set_buffer(1, Some(weight), 0);
            self.enc.set_buffer(2, Some(bias), 0);
            self.enc.set_buffer(3, Some(output), 0);
            self.enc.set_bytes(
                4,
                mem::size_of::<ConvParams>() as u64,
                params as *const ConvParams as *const _,
            );
            self.enc.set_buffer(5, Some(residual), 0);
            let bm = 64u32;
            let bn = 64u32;
            let wg_x = params.n_out.div_ceil(bn);
            let wg_y = params.m.div_ceil(bm);
            self.enc.dispatch_thread_groups(
                MTLSize {
                    width: wg_x as u64,
                    height: wg_y as u64,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
        }

        /// SIMD 1×1 conv GEMM BK=32: same 64×64 tile as conv1x1_simd but BK=32.
        /// Halves barrier count for large-K (≥128) convolutions.
        pub fn conv1x1_simd_bk32_f16io(
            &self,
            input: &Buffer,
            weight: &Buffer,
            bias: &Buffer,
            output: &Buffer,
            params: &ConvParams,
        ) {
            self.enc
                .set_compute_pipeline_state(&self.inf.conv1x1_simd_bk32_f16io);
            self.enc.set_buffer(0, Some(input), 0);
            self.enc.set_buffer(1, Some(weight), 0);
            self.enc.set_buffer(2, Some(bias), 0);
            self.enc.set_buffer(3, Some(output), 0);
            self.enc.set_bytes(
                4,
                mem::size_of::<ConvParams>() as u64,
                params as *const ConvParams as *const _,
            );
            let bm = 64u32;
            let bn = 64u32;
            let wg_x = params.n_out.div_ceil(bn);
            let wg_y = params.m.div_ceil(bm);
            self.enc.dispatch_thread_groups(
                MTLSize {
                    width: wg_x as u64,
                    height: wg_y as u64,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
        }

        /// SIMD 1×1 conv GEMM with 4 simdgroups (128 threads): 2×2 sg layout,
        /// acc[4][4] = 16 accumulators per simdgroup → 67% MAC utilization.
        /// Lower shared memory (5.3KB) → higher occupancy (6 TGs/core).
        pub fn conv1x1_simd_4sg_f16io(
            &self,
            input: &Buffer,
            weight: &Buffer,
            bias: &Buffer,
            output: &Buffer,
            params: &ConvParams,
        ) {
            self.enc
                .set_compute_pipeline_state(&self.inf.conv1x1_simd_4sg_f16io);
            self.enc.set_buffer(0, Some(input), 0);
            self.enc.set_buffer(1, Some(weight), 0);
            self.enc.set_buffer(2, Some(bias), 0);
            self.enc.set_buffer(3, Some(output), 0);
            self.enc.set_bytes(
                4,
                mem::size_of::<ConvParams>() as u64,
                params as *const ConvParams as *const _,
            );
            let bm = 64u32;
            let bn = 64u32;
            let wg_x = params.n_out.div_ceil(bn);
            let wg_y = params.m.div_ceil(bm);
            self.enc.dispatch_thread_groups(
                MTLSize {
                    width: wg_x as u64,
                    height: wg_y as u64,
                    depth: 1,
                },
                MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                },
            );
        }

        /// SIMD 1×1 conv GEMM LARGE: BM=128, BN=64 for higher arithmetic intensity.
        /// 256 threads = 8 simdgroups in 4×2 layout over 128×64 tile.
        pub fn conv1x1_simd_large_f16io(
            &self,
            input: &Buffer,
            weight: &Buffer,
            bias: &Buffer,
            output: &Buffer,
            params: &ConvParams,
        ) {
            self.enc
                .set_compute_pipeline_state(&self.inf.conv1x1_simd_large_f16io);
            self.enc.set_buffer(0, Some(input), 0);
            self.enc.set_buffer(1, Some(weight), 0);
            self.enc.set_buffer(2, Some(bias), 0);
            self.enc.set_buffer(3, Some(output), 0);
            self.enc.set_bytes(
                4,
                mem::size_of::<ConvParams>() as u64,
                params as *const ConvParams as *const _,
            );
            let bm = 128u32;
            let bn = 64u32;
            let wg_x = params.n_out.div_ceil(bn);
            let wg_y = params.m.div_ceil(bm);
            self.enc.dispatch_thread_groups(
                MTLSize {
                    width: wg_x as u64,
                    height: wg_y as u64,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
        }

        /// SIMD 3×3+ conv GEMM with im2col: simdgroup_multiply_accumulate (half×half→float).
        pub fn conv3x3_simd_f16io(
            &self,
            input: &Buffer,
            weight: &Buffer,
            bias: &Buffer,
            output: &Buffer,
            params: &ConvParams,
        ) {
            self.enc
                .set_compute_pipeline_state(&self.inf.conv3x3_simd_f16io);
            self.enc.set_buffer(0, Some(input), 0);
            self.enc.set_buffer(1, Some(weight), 0);
            self.enc.set_buffer(2, Some(bias), 0);
            self.enc.set_buffer(3, Some(output), 0);
            self.enc.set_bytes(
                4,
                mem::size_of::<ConvParams>() as u64,
                params as *const ConvParams as *const _,
            );
            // Buffer 5: residual (dummy when has_residual=0)
            self.enc.set_buffer(5, Some(output), 0);
            let bm = 64u32;
            let bn = 64u32;
            let wg_x = params.n_out.div_ceil(bn);
            let wg_y = params.m.div_ceil(bm);
            self.enc.dispatch_thread_groups(
                MTLSize {
                    width: wg_x as u64,
                    height: wg_y as u64,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
        }

        /// SIMD 3×3 conv with fused residual add before activation.
        pub fn conv3x3_simd_res_f16io(
            &self,
            input: &Buffer,
            weight: &Buffer,
            bias: &Buffer,
            output: &Buffer,
            residual: &Buffer,
            params: &ConvParams,
        ) {
            self.enc
                .set_compute_pipeline_state(&self.inf.conv3x3_simd_f16io);
            self.enc.set_buffer(0, Some(input), 0);
            self.enc.set_buffer(1, Some(weight), 0);
            self.enc.set_buffer(2, Some(bias), 0);
            self.enc.set_buffer(3, Some(output), 0);
            self.enc.set_bytes(
                4,
                mem::size_of::<ConvParams>() as u64,
                params as *const ConvParams as *const _,
            );
            self.enc.set_buffer(5, Some(residual), 0);
            let bm = 64u32;
            let bn = 64u32;
            let wg_x = params.n_out.div_ceil(bn);
            let wg_y = params.m.div_ceil(bm);
            self.enc.dispatch_thread_groups(
                MTLSize {
                    width: wg_x as u64,
                    height: wg_y as u64,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
        }

        /// Large-M conv GEMM: BM=128, BN=64 — for layers with large M.
        pub fn conv_gemm_large_f16io(
            &self,
            input: &Buffer,
            weight: &Buffer,
            bias: &Buffer,
            output: &Buffer,
            params: &ConvParams,
        ) {
            self.enc
                .set_compute_pipeline_state(&self.inf.conv_gemm_large_f16io);
            self.enc.set_buffer(0, Some(input), 0);
            self.enc.set_buffer(1, Some(weight), 0);
            self.enc.set_buffer(2, Some(bias), 0);
            self.enc.set_buffer(3, Some(output), 0);
            self.enc.set_bytes(
                4,
                mem::size_of::<ConvParams>() as u64,
                params as *const ConvParams as *const _,
            );
            let bm = 128u32;
            let bn = 64u32;
            let wg_x = params.n_out.div_ceil(bn);
            let wg_y = params.m.div_ceil(bm);
            self.enc.dispatch_thread_groups(
                MTLSize {
                    width: wg_x as u64,
                    height: wg_y as u64,
                    depth: 1,
                },
                MTLSize {
                    width: 16,
                    height: 16,
                    depth: 1,
                },
            );
        }

        /// Small-N conv GEMM: BM=64, BN=32 — for layers with N≤32.
        pub fn conv_gemm_small_f16io(
            &self,
            input: &Buffer,
            weight: &Buffer,
            bias: &Buffer,
            output: &Buffer,
            params: &ConvParams,
        ) {
            self.enc
                .set_compute_pipeline_state(&self.inf.conv_gemm_small_f16io);
            self.enc.set_buffer(0, Some(input), 0);
            self.enc.set_buffer(1, Some(weight), 0);
            self.enc.set_buffer(2, Some(bias), 0);
            self.enc.set_buffer(3, Some(output), 0);
            self.enc.set_bytes(
                4,
                mem::size_of::<ConvParams>() as u64,
                params as *const ConvParams as *const _,
            );
            // Buffer 5: residual (dummy when has_residual=0)
            self.enc.set_buffer(5, Some(output), 0);
            let bm = 64u32;
            let bn = 32u32;
            let wg_x = params.n_out.div_ceil(bn);
            let wg_y = params.m.div_ceil(bm);
            self.enc.dispatch_thread_groups(
                MTLSize {
                    width: wg_x as u64,
                    height: wg_y as u64,
                    depth: 1,
                },
                MTLSize {
                    width: 16,
                    height: 16,
                    depth: 1,
                },
            );
        }

        /// Optimized conv GEMM v2: BK=32, vectorized half4 stores/reads.
        /// Same threadgroup size (16×16=256) and tile (64×64), just faster inner loop.
        pub fn conv_gemm_v2_f16io(
            &self,
            input: &Buffer,
            weight: &Buffer,
            bias: &Buffer,
            output: &Buffer,
            params: &ConvParams,
        ) {
            self.enc
                .set_compute_pipeline_state(&self.inf.conv_gemm_v2_f16io);
            self.enc.set_buffer(0, Some(input), 0);
            self.enc.set_buffer(1, Some(weight), 0);
            self.enc.set_buffer(2, Some(bias), 0);
            self.enc.set_buffer(3, Some(output), 0);
            self.enc.set_bytes(
                4,
                mem::size_of::<ConvParams>() as u64,
                params as *const ConvParams as *const _,
            );
            let bm = 64u32;
            let bn = 64u32;
            let wg_x = params.n_out.div_ceil(bn);
            let wg_y = params.m.div_ceil(bm);
            self.enc.dispatch_thread_groups(
                MTLSize {
                    width: wg_x as u64,
                    height: wg_y as u64,
                    depth: 1,
                },
                MTLSize {
                    width: 16,
                    height: 16,
                    depth: 1,
                },
            );
        }

        pub fn conv_direct_f16io(
            &self,
            input: &Buffer,
            weight: &Buffer,
            bias: &Buffer,
            output: &Buffer,
            params: &ConvParams,
        ) {
            self.enc
                .set_compute_pipeline_state(&self.inf.conv_direct_f16io);
            self.enc.set_buffer(0, Some(input), 0);
            self.enc.set_buffer(1, Some(weight), 0);
            self.enc.set_buffer(2, Some(bias), 0);
            self.enc.set_buffer(3, Some(output), 0);
            self.enc.set_bytes(
                4,
                mem::size_of::<ConvParams>() as u64,
                params as *const ConvParams as *const _,
            );
            let threads_x = params.m as u64;
            let threads_y = params.n_out.div_ceil(4) as u64;
            let wg_x = 16u64.min(threads_x);
            let wg_y = 16u64.min(threads_y);
            let groups_x = threads_x.div_ceil(wg_x);
            let groups_y = threads_y.div_ceil(wg_y);
            self.enc.dispatch_thread_groups(
                MTLSize {
                    width: groups_x,
                    height: groups_y,
                    depth: 1,
                },
                MTLSize {
                    width: wg_x,
                    height: wg_y,
                    depth: 1,
                },
            );
        }

        pub fn conv_gemm_simd_f16io(
            &self,
            input: &Buffer,
            weight: &Buffer,
            bias: &Buffer,
            output: &Buffer,
            params: &ConvParams,
        ) {
            self.enc
                .set_compute_pipeline_state(&self.inf.conv_gemm_simd_f16io);
            self.enc.set_buffer(0, Some(input), 0);
            self.enc.set_buffer(1, Some(weight), 0);
            self.enc.set_buffer(2, Some(bias), 0);
            self.enc.set_buffer(3, Some(output), 0);
            self.enc.set_bytes(
                4,
                mem::size_of::<ConvParams>() as u64,
                params as *const ConvParams as *const _,
            );
            let bm = 64u32;
            let bn = 64u32;
            let wg_x = params.n_out.div_ceil(bn);
            let wg_y = params.m.div_ceil(bm);
            // 256 threads = 8 simdgroups × 32 threads
            self.enc.dispatch_thread_groups(
                MTLSize {
                    width: wg_x as u64,
                    height: wg_y as u64,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
        }

        pub fn depthwise_conv_f16io(
            &self,
            input: &Buffer,
            weight: &Buffer,
            bias: &Buffer,
            output: &Buffer,
            params: &ConvParams,
        ) {
            self.enc
                .set_compute_pipeline_state(&self.inf.depthwise_conv_f16io);
            self.enc.set_buffer(0, Some(input), 0);
            self.enc.set_buffer(1, Some(weight), 0);
            self.enc.set_buffer(2, Some(bias), 0);
            self.enc.set_buffer(3, Some(output), 0);
            self.enc.set_bytes(
                4,
                mem::size_of::<ConvParams>() as u64,
                params as *const ConvParams as *const _,
            );
            let threads_x = params.m as u64;
            let threads_y = params.n_out.div_ceil(4) as u64;
            let wg_x = 16u64.min(threads_x);
            let wg_y = 16u64.min(threads_y);
            let groups_x = threads_x.div_ceil(wg_x);
            let groups_y = threads_y.div_ceil(wg_y);
            self.enc.dispatch_thread_groups(
                MTLSize {
                    width: groups_x,
                    height: groups_y,
                    depth: 1,
                },
                MTLSize {
                    width: wg_x,
                    height: wg_y,
                    depth: 1,
                },
            );
        }

        pub fn binary_f16(&self, a: &Buffer, b: &Buffer, out: &Buffer, n: u32, op: u32) {
            #[repr(C)]
            struct P {
                n: u32,
                op: u32,
            }
            let p = P { n, op };
            self.enc
                .set_compute_pipeline_state(&self.inf.binary_elementwise_f16);
            self.enc.set_buffer(0, Some(a), 0);
            self.enc.set_buffer(1, Some(b), 0);
            self.enc.set_buffer(2, Some(out), 0);
            self.enc
                .set_bytes(3, mem::size_of::<P>() as u64, &p as *const P as *const _);
            self.dispatch_1d(&self.inf.binary_elementwise_f16, n);
        }

        pub fn broadcast_binary_f16(
            &self,
            a: &Buffer,
            b: &Buffer,
            out: &Buffer,
            n: u32,
            broadcast_dim: u32,
            op: u32,
        ) {
            #[repr(C)]
            struct P {
                n: u32,
                broadcast_dim: u32,
                op: u32,
            }
            let p = P {
                n,
                broadcast_dim,
                op,
            };
            self.enc
                .set_compute_pipeline_state(&self.inf.broadcast_binary_f16);
            self.enc.set_buffer(0, Some(a), 0);
            self.enc.set_buffer(1, Some(b), 0);
            self.enc.set_buffer(2, Some(out), 0);
            self.enc
                .set_bytes(3, mem::size_of::<P>() as u64, &p as *const P as *const _);
            self.dispatch_1d(&self.inf.broadcast_binary_f16, n);
        }

        pub fn unary_f16(&self, input: &Buffer, out: &Buffer, n: u32, op: u32) {
            #[repr(C)]
            struct P {
                n: u32,
                op: u32,
            }
            let p = P { n, op };
            self.enc.set_compute_pipeline_state(&self.inf.unary_op_f16);
            self.enc.set_buffer(0, Some(input), 0);
            self.enc.set_buffer(1, Some(out), 0);
            self.enc
                .set_bytes(2, mem::size_of::<P>() as u64, &p as *const P as *const _);
            self.dispatch_1d(&self.inf.unary_op_f16, n);
        }

        pub fn silu_f16(&self, input: &Buffer, out: &Buffer, n: u32) {
            self.enc.set_compute_pipeline_state(&self.inf.silu_f16);
            self.enc.set_buffer(0, Some(input), 0);
            self.enc.set_buffer(1, Some(out), 0);
            self.enc.set_bytes(2, 4, &n as *const u32 as *const _);
            self.dispatch_1d(&self.inf.silu_f16, n);
        }

        pub fn maxpool2d_f16(
            &self,
            input: &Buffer,
            out: &Buffer,
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
        ) {
            #[repr(C)]
            struct P {
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
            }
            let p = P {
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
            };
            self.enc.set_compute_pipeline_state(&self.inf.maxpool2d_f16);
            self.enc.set_buffer(0, Some(input), 0);
            self.enc.set_buffer(1, Some(out), 0);
            self.enc
                .set_bytes(2, mem::size_of::<P>() as u64, &p as *const P as *const _);
            let n = batch * oh * ow * ic;
            self.dispatch_1d(&self.inf.maxpool2d_f16, n);
        }

        pub fn resize_nearest_f16(
            &self,
            input: &Buffer,
            out: &Buffer,
            batch: u32,
            ih: u32,
            iw: u32,
            ic: u32,
            oh: u32,
            ow: u32,
            scale_h: f32,
            scale_w: f32,
        ) {
            #[repr(C)]
            struct P {
                batch: u32,
                ih: u32,
                iw: u32,
                ic: u32,
                oh: u32,
                ow: u32,
                scale_h: f32,
                scale_w: f32,
            }
            let p = P {
                batch,
                ih,
                iw,
                ic,
                oh,
                ow,
                scale_h,
                scale_w,
            };
            self.enc
                .set_compute_pipeline_state(&self.inf.resize_nearest_f16);
            self.enc.set_buffer(0, Some(input), 0);
            self.enc.set_buffer(1, Some(out), 0);
            self.enc
                .set_bytes(2, mem::size_of::<P>() as u64, &p as *const P as *const _);
            let n = batch * oh * ow * ic;
            self.dispatch_1d(&self.inf.resize_nearest_f16, n);
        }

        /// Vectorized resize nearest (f16): 3D grid (ic/4, ow, oh*batch), half4 copies.
        pub fn resize_nearest_f16v4(
            &self,
            input: &Buffer,
            out: &Buffer,
            batch: u32,
            ih: u32,
            iw: u32,
            ic: u32,
            oh: u32,
            ow: u32,
            scale_h: f32,
            scale_w: f32,
        ) {
            #[repr(C)]
            struct P {
                batch: u32,
                ih: u32,
                iw: u32,
                ic: u32,
                oh: u32,
                ow: u32,
                scale_h: f32,
                scale_w: f32,
            }
            let p = P {
                batch,
                ih,
                iw,
                ic,
                oh,
                ow,
                scale_h,
                scale_w,
            };
            self.enc
                .set_compute_pipeline_state(&self.inf.resize_nearest_f16v4);
            self.enc.set_buffer(0, Some(input), 0);
            self.enc.set_buffer(1, Some(out), 0);
            self.enc
                .set_bytes(2, mem::size_of::<P>() as u64, &p as *const P as *const _);
            let ic4 = ic.div_ceil(4);
            let tg_x = ic4.min(32);
            self.enc.dispatch_thread_groups(
                MTLSize {
                    width: ic4.div_ceil(tg_x) as u64,
                    height: ow as u64,
                    depth: (oh * batch) as u64,
                },
                MTLSize {
                    width: tg_x as u64,
                    height: 1,
                    depth: 1,
                },
            );
        }

        pub fn softmax_f16(&self, input: &Buffer, out: &Buffer, outer: u32, dim: u32) {
            #[repr(C)]
            struct P {
                outer: u32,
                dim: u32,
            }
            let p = P { outer, dim };
            if dim <= 128 {
                // Scalar path: one thread per row, no threadgroup reduction
                self.enc
                    .set_compute_pipeline_state(&self.inf.softmax_scalar_f16);
                self.enc.set_buffer(0, Some(input), 0);
                self.enc.set_buffer(1, Some(out), 0);
                self.enc
                    .set_bytes(2, mem::size_of::<P>() as u64, &p as *const P as *const _);
                self.dispatch_1d(&self.inf.softmax_scalar_f16, outer);
            } else {
                // Threadgroup reduction path for large dim
                self.enc.set_compute_pipeline_state(&self.inf.softmax_f16);
                self.enc.set_buffer(0, Some(input), 0);
                self.enc.set_buffer(1, Some(out), 0);
                self.enc
                    .set_bytes(2, mem::size_of::<P>() as u64, &p as *const P as *const _);
                self.enc.dispatch_thread_groups(
                    MTLSize {
                        width: outer as u64,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 256,
                        height: 1,
                        depth: 1,
                    },
                );
            }
        }

        pub fn transpose_2d_f16(&self, input: &Buffer, out: &Buffer, rows: u32, cols: u32) {
            #[repr(C)]
            struct P {
                rows: u32,
                cols: u32,
            }
            let p = P { rows, cols };
            self.enc
                .set_compute_pipeline_state(&self.inf.transpose_2d_f16);
            self.enc.set_buffer(0, Some(input), 0);
            self.enc.set_buffer(1, Some(out), 0);
            self.enc
                .set_bytes(2, mem::size_of::<P>() as u64, &p as *const P as *const _);
            let n = rows * cols;
            self.dispatch_1d(&self.inf.transpose_2d_f16, n);
        }

        pub fn split_channels_f16(
            &self,
            input: &Buffer,
            out: &Buffer,
            spatial: u32,
            in_c: u32,
            out_c: u32,
            offset_c: u32,
        ) {
            #[repr(C)]
            struct P {
                spatial: u32,
                in_c: u32,
                out_c: u32,
                offset_c: u32,
            }
            let p = P {
                spatial,
                in_c,
                out_c,
                offset_c,
            };
            self.enc
                .set_compute_pipeline_state(&self.inf.split_channels_f16);
            self.enc.set_buffer(0, Some(input), 0);
            self.enc.set_buffer(1, Some(out), 0);
            self.enc
                .set_bytes(2, mem::size_of::<P>() as u64, &p as *const P as *const _);
            let n = spatial * out_c;
            self.dispatch_1d(&self.inf.split_channels_f16, n);
        }

        pub fn permute_0213_f16(
            &self,
            input: &Buffer,
            out: &Buffer,
            d0: u32,
            d1: u32,
            d2: u32,
            d3: u32,
        ) {
            #[repr(C)]
            struct P {
                d0: u32,
                d1: u32,
                d2: u32,
                d3: u32,
            }
            let p = P { d0, d1, d2, d3 };
            self.enc
                .set_compute_pipeline_state(&self.inf.permute_0213_f16);
            self.enc.set_buffer(0, Some(input), 0);
            self.enc.set_buffer(1, Some(out), 0);
            self.enc
                .set_bytes(2, mem::size_of::<P>() as u64, &p as *const P as *const _);
            let total = d0 * d1 * d2 * d3;
            self.dispatch_1d(&self.inf.permute_0213_f16, total);
        }

        pub fn permute_nhwc_to_nchw_f16(
            &self,
            input: &Buffer,
            out: &Buffer,
            n: u32,
            h: u32,
            w: u32,
            c: u32,
        ) {
            #[repr(C)]
            struct P {
                n: u32,
                h: u32,
                w: u32,
                c: u32,
            }
            let p = P { n, h, w, c };
            self.enc
                .set_compute_pipeline_state(&self.inf.permute_nhwc_to_nchw_f16v4);
            self.enc.set_buffer(0, Some(input), 0);
            self.enc.set_buffer(1, Some(out), 0);
            self.enc
                .set_bytes(2, mem::size_of::<P>() as u64, &p as *const P as *const _);
            let c4 = c.div_ceil(4);
            let tg_w = w.min(16);
            let tg_hb = (256 / tg_w).min(h * n).max(1);
            self.enc.dispatch_thread_groups(
                MTLSize {
                    width: w.div_ceil(tg_w) as u64,
                    height: (h * n).div_ceil(tg_hb) as u64,
                    depth: c4 as u64,
                },
                MTLSize {
                    width: tg_w as u64,
                    height: tg_hb as u64,
                    depth: 1,
                },
            );
        }

        pub fn permute_nchw_to_nhwc_f16(
            &self,
            input: &Buffer,
            out: &Buffer,
            n: u32,
            h: u32,
            w: u32,
            c: u32,
        ) {
            #[repr(C)]
            struct P {
                n: u32,
                h: u32,
                w: u32,
                c: u32,
            }
            let p = P { n, h, w, c };
            self.enc
                .set_compute_pipeline_state(&self.inf.permute_nchw_to_nhwc_f16v4);
            self.enc.set_buffer(0, Some(input), 0);
            self.enc.set_buffer(1, Some(out), 0);
            self.enc
                .set_bytes(2, mem::size_of::<P>() as u64, &p as *const P as *const _);
            let c4 = c.div_ceil(4);
            let tg_w = w.min(16);
            let tg_hb = (256 / tg_w).min(h * n).max(1);
            self.enc.dispatch_thread_groups(
                MTLSize {
                    width: w.div_ceil(tg_w) as u64,
                    height: (h * n).div_ceil(tg_hb) as u64,
                    depth: c4 as u64,
                },
                MTLSize {
                    width: tg_w as u64,
                    height: tg_hb as u64,
                    depth: 1,
                },
            );
        }

        pub fn matmul_f16io(&self, a: &Buffer, b: &Buffer, out: &Buffer, m: u32, n: u32, k: u32) {
            #[repr(C)]
            struct P {
                m: u32,
                n: u32,
                k: u32,
            }
            let p = P { m, n, k };
            self.enc.set_compute_pipeline_state(&self.inf.matmul_f16io);
            self.enc.set_buffer(0, Some(a), 0);
            self.enc.set_buffer(1, Some(b), 0);
            self.enc.set_buffer(2, Some(out), 0);
            self.enc
                .set_bytes(3, mem::size_of::<P>() as u64, &p as *const P as *const _);
            let bm = 64u32;
            let bn = 64u32;
            let wg_x = n.div_ceil(bn);
            let wg_y = m.div_ceil(bm);
            self.enc.dispatch_thread_groups(
                MTLSize {
                    width: wg_x as u64,
                    height: wg_y as u64,
                    depth: 1,
                },
                MTLSize {
                    width: 16,
                    height: 16,
                    depth: 1,
                },
            );
        }

        /// In-place bias addition + optional activation on f16 buffer.
        /// data: [M, N] f16 buffer, bias: [N] f32 buffer, act: 0=none, 1=relu, 2=silu.
        pub fn bias_act_f16(
            &self,
            data: &Buffer,
            bias: &Buffer,
            total: u32,
            n_channels: u32,
            act: u32,
        ) {
            #[repr(C)]
            struct P {
                total: u32,
                n_channels: u32,
                act: u32,
            }
            let p = P {
                total,
                n_channels,
                act,
            };
            self.enc.set_compute_pipeline_state(&self.inf.bias_act_f16);
            self.enc.set_buffer(0, Some(data), 0);
            self.enc.set_buffer(1, Some(bias), 0);
            self.enc
                .set_bytes(2, mem::size_of::<P>() as u64, &p as *const P as *const _);
            self.dispatch_1d(&self.inf.bias_act_f16, total);
        }

        /// Im2col: convert NHWC input to [M, K] column matrix for GEMM conv.
        /// M = batch*oh*ow, K = kh*kw*ic.
        pub fn im2col_f16(
            &self,
            input: &Buffer,
            col: &Buffer,
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
        ) {
            #[repr(C)]
            struct P {
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
                m: u32,
                k: u32,
            }
            let m = batch * oh * ow;
            let k = kh * kw * ic;
            let p = P {
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
                m,
                k,
            };
            self.enc.set_compute_pipeline_state(&self.inf.im2col_f16);
            self.enc.set_buffer(0, Some(input), 0);
            self.enc.set_buffer(1, Some(col), 0);
            self.enc
                .set_bytes(2, mem::size_of::<P>() as u64, &p as *const P as *const _);
            self.dispatch_1d(&self.inf.im2col_f16, m * k);
        }

        // ── Winograd F(2×2, 3×3) dispatch ──

        pub fn winograd_input_transform_f16(
            &self,
            input: &Buffer,
            transformed: &Buffer,
            params: &WinogradParams,
        ) {
            self.enc
                .set_compute_pipeline_state(&self.inf.winograd_input_transform_f16);
            self.enc.set_buffer(0, Some(input), 0);
            self.enc.set_buffer(1, Some(transformed), 0);
            self.enc.set_bytes(
                2,
                mem::size_of::<WinogradParams>() as u64,
                params as *const WinogradParams as *const _,
            );
            // Grid: (ic, n_tiles) — each thread does one (channel, tile) pair
            let groups_x = (params.ic as u64).div_ceil(16);
            let groups_y = (params.n_tiles as u64).div_ceil(16);
            self.enc.dispatch_thread_groups(
                MTLSize {
                    width: groups_x,
                    height: groups_y,
                    depth: 1,
                },
                MTLSize {
                    width: 16,
                    height: 16,
                    depth: 1,
                },
            );
        }

        pub fn winograd_batched_gemm_f16io(
            &self,
            a: &Buffer,
            b: &Buffer,
            c: &Buffer,
            n_tiles: u32,
            oc: u32,
            ic: u32,
        ) {
            #[repr(C)]
            struct P {
                m: u32,
                n: u32,
                k: u32,
            }
            let p = P {
                m: n_tiles,
                n: oc,
                k: ic,
            };
            self.enc
                .set_compute_pipeline_state(&self.inf.winograd_batched_gemm_f16io);
            self.enc.set_buffer(0, Some(a), 0);
            self.enc.set_buffer(1, Some(b), 0);
            self.enc.set_buffer(2, Some(c), 0);
            self.enc
                .set_bytes(3, mem::size_of::<P>() as u64, &p as *const P as *const _);
            let bm = 64u32;
            let bn = 64u32;
            let wg_x = oc.div_ceil(bn);
            let wg_y = n_tiles.div_ceil(bm);
            self.enc.dispatch_thread_groups(
                MTLSize {
                    width: wg_x as u64,
                    height: wg_y as u64,
                    depth: 16,
                },
                MTLSize {
                    width: 16,
                    height: 16,
                    depth: 1,
                },
            );
        }

        pub fn winograd_output_transform_f16(
            &self,
            gemm_out: &Buffer,
            bias: &Buffer,
            output: &Buffer,
            params: &WinogradParams,
        ) {
            self.enc
                .set_compute_pipeline_state(&self.inf.winograd_output_transform_f16);
            self.enc.set_buffer(0, Some(gemm_out), 0);
            self.enc.set_buffer(1, Some(bias), 0);
            self.enc.set_buffer(2, Some(output), 0);
            self.enc.set_bytes(
                3,
                mem::size_of::<WinogradParams>() as u64,
                params as *const WinogradParams as *const _,
            );
            let groups_x = (params.oc as u64).div_ceil(16);
            let groups_y = (params.n_tiles as u64).div_ceil(16);
            self.enc.dispatch_thread_groups(
                MTLSize {
                    width: groups_x,
                    height: groups_y,
                    depth: 1,
                },
                MTLSize {
                    width: 16,
                    height: 16,
                    depth: 1,
                },
            );
        }

        // ── Winograd F(4×4, 3×3) dispatch ──

        pub fn winograd4x4_input_transform_f16(
            &self,
            input: &Buffer,
            transformed: &Buffer,
            params: &WinogradParams,
        ) {
            self.enc
                .set_compute_pipeline_state(&self.inf.winograd4x4_input_transform_f16);
            self.enc.set_buffer(0, Some(input), 0);
            self.enc.set_buffer(1, Some(transformed), 0);
            self.enc.set_bytes(
                2,
                mem::size_of::<WinogradParams>() as u64,
                params as *const WinogradParams as *const _,
            );
            let groups_x = (params.ic as u64).div_ceil(16);
            let groups_y = (params.n_tiles as u64).div_ceil(16);
            self.enc.dispatch_thread_groups(
                MTLSize {
                    width: groups_x,
                    height: groups_y,
                    depth: 1,
                },
                MTLSize {
                    width: 16,
                    height: 16,
                    depth: 1,
                },
            );
        }

        pub fn winograd4x4_batched_gemm_f16io(
            &self,
            a: &Buffer,
            b: &Buffer,
            c: &Buffer,
            n_tiles: u32,
            oc: u32,
            ic: u32,
        ) {
            #[repr(C)]
            struct P {
                m: u32,
                n: u32,
                k: u32,
            }
            let p = P {
                m: n_tiles,
                n: oc,
                k: ic,
            };
            self.enc
                .set_compute_pipeline_state(&self.inf.winograd4x4_batched_gemm_f16io);
            self.enc.set_buffer(0, Some(a), 0);
            self.enc.set_buffer(1, Some(b), 0);
            self.enc.set_buffer(2, Some(c), 0);
            self.enc
                .set_bytes(3, mem::size_of::<P>() as u64, &p as *const P as *const _);
            let bm = 64u32;
            let bn = 64u32;
            let wg_x = oc.div_ceil(bn);
            let wg_y = n_tiles.div_ceil(bm);
            self.enc.dispatch_thread_groups(
                MTLSize {
                    width: wg_x as u64,
                    height: wg_y as u64,
                    depth: 36,
                },
                MTLSize {
                    width: 16,
                    height: 16,
                    depth: 1,
                },
            );
        }

        /// F(4,3) Winograd GEMM using simdgroup_multiply_accumulate (half×half→float).
        /// Same throughput as half but float accumulation prevents overflow.
        pub fn winograd4x4_batched_gemm_simd_f16io(
            &self,
            a: &Buffer,
            b: &Buffer,
            c: &Buffer,
            n_tiles: u32,
            oc: u32,
            ic: u32,
        ) {
            #[repr(C)]
            struct P {
                m: u32,
                n: u32,
                k: u32,
            }
            let p = P {
                m: n_tiles,
                n: oc,
                k: ic,
            };
            self.enc
                .set_compute_pipeline_state(&self.inf.winograd_batched_gemm_simd_f16io);
            self.enc.set_buffer(0, Some(a), 0);
            self.enc.set_buffer(1, Some(b), 0);
            self.enc.set_buffer(2, Some(c), 0);
            self.enc
                .set_bytes(3, mem::size_of::<P>() as u64, &p as *const P as *const _);
            let bm = 64u32;
            let bn = 64u32;
            let wg_x = oc.div_ceil(bn);
            let wg_y = n_tiles.div_ceil(bm);
            self.enc.dispatch_thread_groups(
                MTLSize {
                    width: wg_x as u64,
                    height: wg_y as u64,
                    depth: 36,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
        }

        pub fn winograd4x4_output_transform_f16(
            &self,
            gemm_out: &Buffer,
            bias: &Buffer,
            output: &Buffer,
            params: &WinogradParams,
        ) {
            self.enc
                .set_compute_pipeline_state(&self.inf.winograd4x4_output_transform_f16);
            self.enc.set_buffer(0, Some(gemm_out), 0);
            self.enc.set_buffer(1, Some(bias), 0);
            self.enc.set_buffer(2, Some(output), 0);
            self.enc.set_bytes(
                3,
                mem::size_of::<WinogradParams>() as u64,
                params as *const WinogradParams as *const _,
            );
            let groups_x = (params.oc as u64).div_ceil(16);
            let groups_y = (params.n_tiles as u64).div_ceil(16);
            self.enc.dispatch_thread_groups(
                MTLSize {
                    width: groups_x,
                    height: groups_y,
                    depth: 1,
                },
                MTLSize {
                    width: 16,
                    height: 16,
                    depth: 1,
                },
            );
        }

        /// Winograd F(4,3) output transform with fused residual add.
        pub fn winograd4x4_output_transform_residual_f16(
            &self,
            gemm_out: &Buffer,
            bias: &Buffer,
            output: &Buffer,
            params: &WinogradParams,
            residual: &Buffer,
        ) {
            self.enc
                .set_compute_pipeline_state(&self.inf.winograd4x4_output_transform_residual_f16);
            self.enc.set_buffer(0, Some(gemm_out), 0);
            self.enc.set_buffer(1, Some(bias), 0);
            self.enc.set_buffer(2, Some(output), 0);
            self.enc.set_bytes(
                3,
                mem::size_of::<WinogradParams>() as u64,
                params as *const WinogradParams as *const _,
            );
            self.enc.set_buffer(4, Some(residual), 0);
            let groups_x = (params.oc as u64).div_ceil(16);
            let groups_y = (params.n_tiles as u64).div_ceil(16);
            self.enc.dispatch_thread_groups(
                MTLSize {
                    width: groups_x,
                    height: groups_y,
                    depth: 1,
                },
                MTLSize {
                    width: 16,
                    height: 16,
                    depth: 1,
                },
            );
        }
    }

    // ── Original MetalConv (for benchmark comparisons) ──

    impl MetalConv {
        pub fn new() -> Option<Self> {
            let device = Device::system_default()?;
            let queue = device.new_command_queue();
            let options = CompileOptions::new();
            options.set_fast_math_enabled(true);

            let library = device
                .new_library_with_source(CONV_GEMM_SIMD_MSL, &options)
                .map_err(|e| eprintln!("Metal shader compile error: {}", e))
                .ok()?;

            let fn_f32w = library.get_function("conv_gemm_simd", None).ok()?;
            let pipeline_f32w = device
                .new_compute_pipeline_state_with_function(&fn_f32w)
                .map_err(|e| eprintln!("Metal pipeline error: {}", e))
                .ok()?;

            let basic_lib = device
                .new_library_with_source(CONV_GEMM_BASIC_MSL, &options)
                .map_err(|e| eprintln!("Metal basic compile: {}", e))
                .ok()?;
            let fn_basic = basic_lib.get_function("conv_gemm_basic", None).ok()?;
            let pipeline_basic = device
                .new_compute_pipeline_state_with_function(&fn_basic)
                .map_err(|e| eprintln!("Metal basic pipeline: {}", e))
                .ok()?;

            Some(MetalConv {
                device,
                queue,
                pipeline_f32w,
                pipeline_basic,
            })
        }

        pub fn device_name(&self) -> String {
            self.device.name().to_string()
        }

        pub fn buffer_from_f32(&self, data: &[f32]) -> Buffer {
            let bytes =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
            self.device.new_buffer_with_data(
                bytes.as_ptr() as *const _,
                bytes.len() as u64,
                MTLResourceOptions::StorageModeShared,
            )
        }

        pub fn output_buffer(&self, num_f32: usize) -> Buffer {
            self.device
                .new_buffer((num_f32 * 4) as u64, MTLResourceOptions::StorageModeShared)
        }

        pub fn dispatch_conv_gemm(
            &self,
            input: &Buffer,
            weight: &Buffer,
            bias: &Buffer,
            output: &Buffer,
            params: &ConvParams,
        ) {
            autoreleasepool(|| {
                let cmd = self.queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&self.pipeline_f32w);
                enc.set_buffer(0, Some(input), 0);
                enc.set_buffer(1, Some(weight), 0);
                enc.set_buffer(2, Some(bias), 0);
                enc.set_buffer(3, Some(output), 0);
                enc.set_bytes(
                    4,
                    mem::size_of::<ConvParams>() as u64,
                    params as *const ConvParams as *const _,
                );
                let wg_x = params.n_out.div_ceil(64);
                let wg_y = params.m.div_ceil(64);
                enc.dispatch_thread_groups(
                    MTLSize {
                        width: wg_x as u64,
                        height: wg_y as u64,
                        depth: 1,
                    },
                    MTLSize {
                        width: 128,
                        height: 1,
                        depth: 1,
                    },
                );
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
            });
        }

        pub fn dispatch_conv_gemm_basic(
            &self,
            input: &Buffer,
            weight: &Buffer,
            bias: &Buffer,
            output: &Buffer,
            params: &ConvParams,
        ) {
            autoreleasepool(|| {
                let cmd = self.queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&self.pipeline_basic);
                enc.set_buffer(0, Some(input), 0);
                enc.set_buffer(1, Some(weight), 0);
                enc.set_buffer(2, Some(bias), 0);
                enc.set_buffer(3, Some(output), 0);
                enc.set_bytes(
                    4,
                    mem::size_of::<ConvParams>() as u64,
                    params as *const ConvParams as *const _,
                );
                let wg_x = params.n_out.div_ceil(64);
                let wg_y = params.m.div_ceil(64);
                enc.dispatch_thread_groups(
                    MTLSize {
                        width: wg_x as u64,
                        height: wg_y as u64,
                        depth: 1,
                    },
                    MTLSize {
                        width: 16,
                        height: 16,
                        depth: 1,
                    },
                );
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
            });
        }

        pub fn bench_conv_gemm(
            &self,
            input: &Buffer,
            weight: &Buffer,
            bias: &Buffer,
            output: &Buffer,
            params: &ConvParams,
            warmup: usize,
            runs: usize,
        ) -> Vec<f64> {
            for _ in 0..warmup {
                self.dispatch_conv_gemm(input, weight, bias, output, params);
            }
            (0..runs)
                .map(|_| {
                    let s = std::time::Instant::now();
                    self.dispatch_conv_gemm(input, weight, bias, output, params);
                    s.elapsed().as_secs_f64() * 1000.0
                })
                .collect()
        }

        pub fn bench_conv_gemm_basic(
            &self,
            input: &Buffer,
            weight: &Buffer,
            bias: &Buffer,
            output: &Buffer,
            params: &ConvParams,
            warmup: usize,
            runs: usize,
        ) -> Vec<f64> {
            for _ in 0..warmup {
                self.dispatch_conv_gemm_basic(input, weight, bias, output, params);
            }
            (0..runs)
                .map(|_| {
                    let s = std::time::Instant::now();
                    self.dispatch_conv_gemm_basic(input, weight, bias, output, params);
                    s.elapsed().as_secs_f64() * 1000.0
                })
                .collect()
        }

        pub fn dispatch_batch_conv_gemm(
            &self,
            dispatches: &[(Buffer, Buffer, Buffer, Buffer, ConvParams)],
        ) {
            autoreleasepool(|| {
                let cmd = self.queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&self.pipeline_f32w);
                for (input, weight, bias, output, params) in dispatches {
                    enc.set_buffer(0, Some(input), 0);
                    enc.set_buffer(1, Some(weight), 0);
                    enc.set_buffer(2, Some(bias), 0);
                    enc.set_buffer(3, Some(output), 0);
                    enc.set_bytes(
                        4,
                        mem::size_of::<ConvParams>() as u64,
                        params as *const ConvParams as *const _,
                    );
                    let wg_x = params.n_out.div_ceil(64);
                    let wg_y = params.m.div_ceil(64);
                    enc.dispatch_thread_groups(
                        MTLSize {
                            width: wg_x as u64,
                            height: wg_y as u64,
                            depth: 1,
                        },
                        MTLSize {
                            width: 128,
                            height: 1,
                            depth: 1,
                        },
                    );
                }
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
            });
        }

        pub fn dispatch_batch_conv_gemm_basic(
            &self,
            dispatches: &[(Buffer, Buffer, Buffer, Buffer, ConvParams)],
        ) {
            autoreleasepool(|| {
                let cmd = self.queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&self.pipeline_basic);
                for (input, weight, bias, output, params) in dispatches {
                    enc.set_buffer(0, Some(input), 0);
                    enc.set_buffer(1, Some(weight), 0);
                    enc.set_buffer(2, Some(bias), 0);
                    enc.set_buffer(3, Some(output), 0);
                    enc.set_bytes(
                        4,
                        mem::size_of::<ConvParams>() as u64,
                        params as *const ConvParams as *const _,
                    );
                    let wg_x = params.n_out.div_ceil(64);
                    let wg_y = params.m.div_ceil(64);
                    enc.dispatch_thread_groups(
                        MTLSize {
                            width: wg_x as u64,
                            height: wg_y as u64,
                            depth: 1,
                        },
                        MTLSize {
                            width: 16,
                            height: 16,
                            depth: 1,
                        },
                    );
                }
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
            });
        }

        pub fn read_buffer_f32(&self, buf: &Buffer, count: usize) -> Vec<f32> {
            let ptr = buf.contents() as *const f32;
            unsafe { std::slice::from_raw_parts(ptr, count).to_vec() }
        }
    }
}

// ── MPSGraph-based whole-model inference ──
// Builds a MetalPerformanceShadersGraph for the entire model, then executes
// the graph as a single GPU dispatch — eliminating per-op encoder transitions.

#[cfg(all(target_os = "macos", feature = "metal-backend"))]
#[allow(unexpected_cfgs)]
pub mod mpsgraph {
    use crate::KernelError;
    use foreign_types::ForeignTypeRef as _;
    use metal::*;
    use objc::rc::autoreleasepool;
    use objc::runtime::{Class, NO, Object, YES};
    use objc::{msg_send, sel, sel_impl};

    // MPSDataType constants
    pub const MPS_DATA_TYPE_FLOAT16: u32 = 0x10000000 | 16;
    pub const MPS_DATA_TYPE_FLOAT32: u32 = 0x10000000 | 32;

    /// Wraps an MPSGraph pointer with a safe Rust interface.
    pub struct MpsGraph {
        pub(crate) graph: *mut Object,
    }

    impl Drop for MpsGraph {
        fn drop(&mut self) {
            unsafe {
                let _: () = msg_send![self.graph, release];
            }
        }
    }

    /// Wraps an MPSGraphTensor pointer (graph node output).
    #[derive(Clone, Copy)]
    pub struct MpsGraphTensorRef {
        pub(crate) ptr: *mut Object,
    }

    /// Wraps a compiled MPSGraphExecutable for repeated inference.
    pub struct MpsGraphExecutable {
        pub(crate) exe: *mut Object,
    }

    impl Drop for MpsGraphExecutable {
        fn drop(&mut self) {
            unsafe {
                let _: () = msg_send![self.exe, release];
            }
        }
    }

    /// Descriptor for Conv2d parameters.
    pub struct Conv2dDesc {
        pub stride_h: usize,
        pub stride_w: usize,
        pub dilation_h: usize,
        pub dilation_w: usize,
        pub pad_top: usize,
        pub pad_bottom: usize,
        pub pad_left: usize,
        pub pad_right: usize,
        pub groups: usize,
    }

    /// Descriptor for Pool2d parameters.
    pub struct Pool2dDesc {
        pub kernel_h: usize,
        pub kernel_w: usize,
        pub stride_h: usize,
        pub stride_w: usize,
        pub pad_top: usize,
        pub pad_bottom: usize,
        pub pad_left: usize,
        pub pad_right: usize,
    }

    // Helper: create NSArray from a &[i64] shape
    unsafe fn ns_array_from_shape(shape: &[i64]) -> Result<*mut Object, KernelError> {
        let ns_number_cls = Class::get("NSNumber").ok_or_else(|| KernelError::Gpu {
            message: "NSNumber class not available".into(),
        })?;
        let ns_array_cls = Class::get("NSArray").ok_or_else(|| KernelError::Gpu {
            message: "NSArray class not available".into(),
        })?;

        // Build NSNumber objects
        let mut numbers: Vec<*mut Object> = Vec::with_capacity(shape.len());
        for &dim in shape {
            let n: *mut Object = msg_send![ns_number_cls, numberWithLongLong: dim];
            numbers.push(n);
        }
        let arr: *mut Object = msg_send![ns_array_cls,
            arrayWithObjects: numbers.as_ptr()
            count: numbers.len()];
        Ok(arr)
    }

    // Helper: create NSArray from a &[usize] shape (converts to i64)
    unsafe fn ns_array_from_usize(shape: &[usize]) -> Result<*mut Object, KernelError> {
        let dims: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
        unsafe { ns_array_from_shape(&dims) }
    }

    impl MpsGraph {
        /// Create a new empty MPSGraph.
        pub fn new() -> Option<Self> {
            unsafe {
                let cls = Class::get("MPSGraph")?;
                let graph: *mut Object = msg_send![cls, new];
                if graph.is_null() {
                    return None;
                }
                Some(MpsGraph { graph })
            }
        }

        /// Create a placeholder input tensor (NCHW layout, f16).
        pub fn placeholder_f16(
            &self,
            shape: &[usize],
            name: &str,
        ) -> Result<MpsGraphTensorRef, KernelError> {
            unsafe {
                let shape_arr = ns_array_from_usize(shape)?;
                let ns_name = ns_string(name)?;
                let tensor: *mut Object = msg_send![self.graph,
                    placeholderWithShape: shape_arr
                    dataType: MPS_DATA_TYPE_FLOAT16
                    name: ns_name];
                Ok(MpsGraphTensorRef { ptr: tensor })
            }
        }

        /// Create a placeholder input tensor (NCHW layout, f32).
        pub fn placeholder_f32(
            &self,
            shape: &[usize],
            name: &str,
        ) -> Result<MpsGraphTensorRef, KernelError> {
            unsafe {
                let shape_arr = ns_array_from_usize(shape)?;
                let ns_name = ns_string(name)?;
                let tensor: *mut Object = msg_send![self.graph,
                    placeholderWithShape: shape_arr
                    dataType: MPS_DATA_TYPE_FLOAT32
                    name: ns_name];
                Ok(MpsGraphTensorRef { ptr: tensor })
            }
        }

        /// Create a constant tensor from f16 data (NCHW layout).
        pub fn constant_f16(
            &self,
            data: &[u16],
            shape: &[usize],
        ) -> Result<MpsGraphTensorRef, KernelError> {
            unsafe {
                let shape_arr = ns_array_from_usize(shape)?;
                let ns_data = ns_data_from_bytes(data.as_ptr() as *const u8, data.len() * 2)?;
                let tensor: *mut Object = msg_send![self.graph,
                    constantWithData: ns_data
                    shape: shape_arr
                    dataType: MPS_DATA_TYPE_FLOAT16];
                Ok(MpsGraphTensorRef { ptr: tensor })
            }
        }

        /// Create a constant tensor from f32 data.
        pub fn constant_f32(
            &self,
            data: &[f32],
            shape: &[usize],
        ) -> Result<MpsGraphTensorRef, KernelError> {
            unsafe {
                let shape_arr = ns_array_from_usize(shape)?;
                let ns_data = ns_data_from_bytes(data.as_ptr() as *const u8, data.len() * 4)?;
                let tensor: *mut Object = msg_send![self.graph,
                    constantWithData: ns_data
                    shape: shape_arr
                    dataType: MPS_DATA_TYPE_FLOAT32];
                Ok(MpsGraphTensorRef { ptr: tensor })
            }
        }

        /// Cast tensor to f16.
        pub fn cast_to_f16(
            &self,
            input: MpsGraphTensorRef,
        ) -> Result<MpsGraphTensorRef, KernelError> {
            unsafe {
                let ns_name = ns_string("cast_f16")?;
                let tensor: *mut Object = msg_send![self.graph,
                    castTensor: input.ptr
                    toType: MPS_DATA_TYPE_FLOAT16
                    name: ns_name];
                Ok(MpsGraphTensorRef { ptr: tensor })
            }
        }

        /// Cast tensor to f32.
        pub fn cast_to_f32(
            &self,
            input: MpsGraphTensorRef,
        ) -> Result<MpsGraphTensorRef, KernelError> {
            unsafe {
                let ns_name = ns_string("cast_f32")?;
                let tensor: *mut Object = msg_send![self.graph,
                    castTensor: input.ptr
                    toType: MPS_DATA_TYPE_FLOAT32
                    name: ns_name];
                Ok(MpsGraphTensorRef { ptr: tensor })
            }
        }

        /// Conv2d: input [N,C,H,W] f16, weights [O,I/g,kH,kW] f16.
        /// MPSGraph expects NCHW layout with OIHW weights.
        pub fn conv2d(
            &self,
            input: MpsGraphTensorRef,
            weights: MpsGraphTensorRef,
            desc: &Conv2dDesc,
        ) -> Result<MpsGraphTensorRef, KernelError> {
            unsafe {
                let conv_desc_cls =
                    Class::get("MPSGraphConvolution2DOpDescriptor").ok_or_else(|| {
                        KernelError::Gpu {
                            message: "MPSGraphConvolution2DOpDescriptor class not available".into(),
                        }
                    })?;
                let d: *mut Object = msg_send![conv_desc_cls,
                    descriptorWithStrideInX: desc.stride_w as u64
                    strideInY: desc.stride_h as u64
                    dilationRateInX: desc.dilation_w as u64
                    dilationRateInY: desc.dilation_h as u64
                    groups: desc.groups as u64
                    paddingLeft: desc.pad_left as u64
                    paddingRight: desc.pad_right as u64
                    paddingTop: desc.pad_top as u64
                    paddingBottom: desc.pad_bottom as u64
                    paddingStyle: 0u64  // explicit padding
                    dataLayout: 0u64    // NCHW
                    weightsLayout: 2u64]; // OIHW

                let tensor: *mut Object = msg_send![self.graph,
                    convolution2DWithSourceTensor: input.ptr
                    weightsTensor: weights.ptr
                    descriptor: d
                    name: std::ptr::null::<Object>()];
                Ok(MpsGraphTensorRef { ptr: tensor })
            }
        }

        /// Depthwise conv2d (groups == in_channels == out_channels).
        /// Weights shape for MPSGraph depthwise: [out_ch, 1, kH, kW] with groups=out_ch.
        pub fn depthwise_conv2d(
            &self,
            input: MpsGraphTensorRef,
            weights: MpsGraphTensorRef,
            desc: &Conv2dDesc,
        ) -> Result<MpsGraphTensorRef, KernelError> {
            // Same as conv2d — MPSGraph handles depthwise when groups == in_channels
            self.conv2d(input, weights, desc)
        }

        /// Add bias to conv output. Input [N,C,H,W], bias [C] → broadcast add.
        pub fn add_bias(
            &self,
            input: MpsGraphTensorRef,
            bias: MpsGraphTensorRef,
        ) -> MpsGraphTensorRef {
            unsafe {
                let tensor: *mut Object = msg_send![self.graph,
                    additionWithPrimaryTensor: input.ptr
                    secondaryTensor: bias.ptr
                    name: std::ptr::null::<Object>()];
                MpsGraphTensorRef { ptr: tensor }
            }
        }

        /// Element-wise addition.
        pub fn add(&self, a: MpsGraphTensorRef, b: MpsGraphTensorRef) -> MpsGraphTensorRef {
            unsafe {
                let tensor: *mut Object = msg_send![self.graph,
                    additionWithPrimaryTensor: a.ptr
                    secondaryTensor: b.ptr
                    name: std::ptr::null::<Object>()];
                MpsGraphTensorRef { ptr: tensor }
            }
        }

        /// Element-wise subtraction.
        pub fn sub(&self, a: MpsGraphTensorRef, b: MpsGraphTensorRef) -> MpsGraphTensorRef {
            unsafe {
                let tensor: *mut Object = msg_send![self.graph,
                    subtractionWithPrimaryTensor: a.ptr
                    secondaryTensor: b.ptr
                    name: std::ptr::null::<Object>()];
                MpsGraphTensorRef { ptr: tensor }
            }
        }

        /// Element-wise multiplication.
        pub fn mul(&self, a: MpsGraphTensorRef, b: MpsGraphTensorRef) -> MpsGraphTensorRef {
            unsafe {
                let tensor: *mut Object = msg_send![self.graph,
                    multiplicationWithPrimaryTensor: a.ptr
                    secondaryTensor: b.ptr
                    name: std::ptr::null::<Object>()];
                MpsGraphTensorRef { ptr: tensor }
            }
        }

        /// Element-wise division.
        pub fn div(&self, a: MpsGraphTensorRef, b: MpsGraphTensorRef) -> MpsGraphTensorRef {
            unsafe {
                let tensor: *mut Object = msg_send![self.graph,
                    divisionWithPrimaryTensor: a.ptr
                    secondaryTensor: b.ptr
                    name: std::ptr::null::<Object>()];
                MpsGraphTensorRef { ptr: tensor }
            }
        }

        /// ReLU activation.
        pub fn relu(&self, input: MpsGraphTensorRef) -> MpsGraphTensorRef {
            unsafe {
                let tensor: *mut Object = msg_send![self.graph,
                    reLUWithTensor: input.ptr
                    name: std::ptr::null::<Object>()];
                MpsGraphTensorRef { ptr: tensor }
            }
        }

        /// Sigmoid activation.
        pub fn sigmoid(&self, input: MpsGraphTensorRef) -> MpsGraphTensorRef {
            unsafe {
                let tensor: *mut Object = msg_send![self.graph,
                    sigmoidWithTensor: input.ptr
                    name: std::ptr::null::<Object>()];
                MpsGraphTensorRef { ptr: tensor }
            }
        }

        /// SiLU = x * sigmoid(x).
        pub fn silu(&self, input: MpsGraphTensorRef) -> MpsGraphTensorRef {
            let sig = self.sigmoid(input);
            self.mul(input, sig)
        }

        /// MaxPool2d: NCHW layout.
        pub fn max_pool2d(
            &self,
            input: MpsGraphTensorRef,
            desc: &Pool2dDesc,
        ) -> Result<MpsGraphTensorRef, KernelError> {
            unsafe {
                let pool_desc_cls =
                    Class::get("MPSGraphPooling2DOpDescriptor").ok_or_else(|| {
                        KernelError::Gpu {
                            message: "MPSGraphPooling2DOpDescriptor class not available".into(),
                        }
                    })?;
                let d: *mut Object = msg_send![pool_desc_cls,
                    descriptorWithKernelWidth: desc.kernel_w as u64
                    kernelHeight: desc.kernel_h as u64
                    strideInX: desc.stride_w as u64
                    strideInY: desc.stride_h as u64
                    dilationRateInX: 1u64
                    dilationRateInY: 1u64
                    paddingLeft: desc.pad_left as u64
                    paddingRight: desc.pad_right as u64
                    paddingTop: desc.pad_top as u64
                    paddingBottom: desc.pad_bottom as u64
                    paddingStyle: 0u64  // explicit
                    dataLayout: 0u64]; // NCHW

                let tensor: *mut Object = msg_send![self.graph,
                    maxPooling2DWithSourceTensor: input.ptr
                    descriptor: d
                    name: std::ptr::null::<Object>()];
                Ok(MpsGraphTensorRef { ptr: tensor })
            }
        }

        /// AvgPool2d: NCHW layout.
        pub fn avg_pool2d(
            &self,
            input: MpsGraphTensorRef,
            desc: &Pool2dDesc,
        ) -> Result<MpsGraphTensorRef, KernelError> {
            unsafe {
                let pool_desc_cls =
                    Class::get("MPSGraphPooling2DOpDescriptor").ok_or_else(|| {
                        KernelError::Gpu {
                            message: "MPSGraphPooling2DOpDescriptor class not available".into(),
                        }
                    })?;
                let d: *mut Object = msg_send![pool_desc_cls,
                    descriptorWithKernelWidth: desc.kernel_w as u64
                    kernelHeight: desc.kernel_h as u64
                    strideInX: desc.stride_w as u64
                    strideInY: desc.stride_h as u64
                    dilationRateInX: 1u64
                    dilationRateInY: 1u64
                    paddingLeft: desc.pad_left as u64
                    paddingRight: desc.pad_right as u64
                    paddingTop: desc.pad_top as u64
                    paddingBottom: desc.pad_bottom as u64
                    paddingStyle: 0u64
                    dataLayout: 0u64];

                // Set countIncludesPadding to NO for standard avg pool
                let _: () = msg_send![d, setCountIncludesZeroPadding: NO];

                let tensor: *mut Object = msg_send![self.graph,
                    avgPooling2DWithSourceTensor: input.ptr
                    descriptor: d
                    name: std::ptr::null::<Object>()];
                Ok(MpsGraphTensorRef { ptr: tensor })
            }
        }

        /// Global average pooling: reduce H and W dims. Input [N,C,H,W] → [N,C,1,1].
        pub fn global_avg_pool(
            &self,
            input: MpsGraphTensorRef,
        ) -> Result<MpsGraphTensorRef, KernelError> {
            unsafe {
                // Reduce mean over axes [2, 3] (H, W in NCHW)
                let axes = ns_array_from_shape(&[2i64, 3i64])?;
                let tensor: *mut Object = msg_send![self.graph,
                    meanOfTensor: input.ptr
                    axes: axes
                    name: std::ptr::null::<Object>()];
                Ok(MpsGraphTensorRef { ptr: tensor })
            }
        }

        /// Batch normalization: input [N,C,H,W], mean/var/gamma/beta [C].
        pub fn batch_norm(
            &self,
            input: MpsGraphTensorRef,
            mean: MpsGraphTensorRef,
            variance: MpsGraphTensorRef,
            gamma: MpsGraphTensorRef,
            beta: MpsGraphTensorRef,
            epsilon: f32,
        ) -> MpsGraphTensorRef {
            unsafe {
                // (x - mean) / sqrt(var + eps) * gamma + beta
                let eps_tensor = self.constant_scalar_f32(epsilon);
                let var_eps: *mut Object = msg_send![self.graph,
                    additionWithPrimaryTensor: variance.ptr
                    secondaryTensor: eps_tensor.ptr
                    name: std::ptr::null::<Object>()];
                let std_dev: *mut Object = msg_send![self.graph,
                    squareRootWithTensor: var_eps
                    name: std::ptr::null::<Object>()];
                let centered: *mut Object = msg_send![self.graph,
                    subtractionWithPrimaryTensor: input.ptr
                    secondaryTensor: mean.ptr
                    name: std::ptr::null::<Object>()];
                let normed: *mut Object = msg_send![self.graph,
                    divisionWithPrimaryTensor: centered
                    secondaryTensor: std_dev
                    name: std::ptr::null::<Object>()];
                let scaled: *mut Object = msg_send![self.graph,
                    multiplicationWithPrimaryTensor: normed
                    secondaryTensor: gamma.ptr
                    name: std::ptr::null::<Object>()];
                let result: *mut Object = msg_send![self.graph,
                    additionWithPrimaryTensor: scaled
                    secondaryTensor: beta.ptr
                    name: std::ptr::null::<Object>()];
                MpsGraphTensorRef { ptr: result }
            }
        }

        /// Softmax along a given axis.
        pub fn softmax(&self, input: MpsGraphTensorRef, axis: i64) -> MpsGraphTensorRef {
            unsafe {
                let tensor: *mut Object = msg_send![self.graph,
                    softMaxWithTensor: input.ptr
                    axis: axis
                    name: std::ptr::null::<Object>()];
                MpsGraphTensorRef { ptr: tensor }
            }
        }

        /// Reshape tensor to new shape.
        pub fn reshape(
            &self,
            input: MpsGraphTensorRef,
            shape: &[i64],
        ) -> Result<MpsGraphTensorRef, KernelError> {
            unsafe {
                let shape_arr = ns_array_from_shape(shape)?;
                let tensor: *mut Object = msg_send![self.graph,
                    reshapeTensor: input.ptr
                    withShape: shape_arr
                    name: std::ptr::null::<Object>()];
                Ok(MpsGraphTensorRef { ptr: tensor })
            }
        }

        /// Transpose (permute) dimensions.
        pub fn transpose(
            &self,
            input: MpsGraphTensorRef,
            dim0: usize,
            dim1: usize,
        ) -> MpsGraphTensorRef {
            unsafe {
                let tensor: *mut Object = msg_send![self.graph,
                    transposeTensor: input.ptr
                    dimension: dim0 as u64
                    withDimension: dim1 as u64
                    name: std::ptr::null::<Object>()];
                MpsGraphTensorRef { ptr: tensor }
            }
        }

        /// Concat tensors along a given axis.
        pub fn concat(
            &self,
            tensors: &[MpsGraphTensorRef],
            axis: i64,
        ) -> Result<MpsGraphTensorRef, KernelError> {
            unsafe {
                let ns_array_cls = Class::get("NSArray").ok_or_else(|| KernelError::Gpu {
                    message: "NSArray class not available".into(),
                })?;
                let ptrs: Vec<*mut Object> = tensors.iter().map(|t| t.ptr).collect();
                let arr: *mut Object = msg_send![ns_array_cls,
                    arrayWithObjects: ptrs.as_ptr()
                    count: ptrs.len()];
                let tensor: *mut Object = msg_send![self.graph,
                    concatTensors: arr
                    dimension: axis
                    name: std::ptr::null::<Object>()];
                Ok(MpsGraphTensorRef { ptr: tensor })
            }
        }

        /// Slice (StridedSlice) for Split-like operations.
        pub fn slice(
            &self,
            input: MpsGraphTensorRef,
            starts: &[i64],
            ends: &[i64],
            strides: &[i64],
        ) -> Result<MpsGraphTensorRef, KernelError> {
            unsafe {
                let starts_arr = ns_array_from_shape(starts)?;
                let ends_arr = ns_array_from_shape(ends)?;
                let strides_arr = ns_array_from_shape(strides)?;
                let tensor: *mut Object = msg_send![self.graph,
                    sliceTensor: input.ptr
                    starts: starts_arr
                    ends: ends_arr
                    strides: strides_arr
                    name: std::ptr::null::<Object>()];
                Ok(MpsGraphTensorRef { ptr: tensor })
            }
        }

        /// MatMul: [M, K] x [K, N] → [M, N].
        pub fn matmul(&self, a: MpsGraphTensorRef, b: MpsGraphTensorRef) -> MpsGraphTensorRef {
            unsafe {
                let tensor: *mut Object = msg_send![self.graph,
                    matrixMultiplicationWithPrimaryTensor: a.ptr
                    secondaryTensor: b.ptr
                    name: std::ptr::null::<Object>()];
                MpsGraphTensorRef { ptr: tensor }
            }
        }

        /// Resize nearest-neighbor (upsampling).
        pub fn resize_nearest(
            &self,
            input: MpsGraphTensorRef,
            out_h: usize,
            out_w: usize,
        ) -> Result<MpsGraphTensorRef, KernelError> {
            unsafe {
                let size_arr = ns_array_from_shape(&[out_h as i64, out_w as i64])?;
                // MPSGraphResizeMode: nearest=0, bilinear=1
                let tensor: *mut Object = msg_send![self.graph,
                    resizeTensor: input.ptr
                    size: size_arr
                    mode: 0u64  // nearest
                    centerResult: YES
                    alignCorners: NO
                    layout: 0u64  // NCHW
                    name: std::ptr::null::<Object>()];
                Ok(MpsGraphTensorRef { ptr: tensor })
            }
        }

        /// Create a scalar f32 constant.
        fn constant_scalar_f32(&self, val: f32) -> MpsGraphTensorRef {
            unsafe {
                let tensor: *mut Object = msg_send![self.graph,
                    constantWithScalar: val as f64
                    dataType: MPS_DATA_TYPE_FLOAT32];
                MpsGraphTensorRef { ptr: tensor }
            }
        }

        /// Compile the graph into an executable for repeated inference.
        /// `feeds` maps placeholder tensor → shape + datatype for each input.
        /// `target_tensors` is the list of output tensors.
        pub fn compile(
            &self,
            device: &Device,
            feeds: &[(MpsGraphTensorRef, &[usize], u32)], // (placeholder, shape, datatype)
            target_tensors: &[MpsGraphTensorRef],
        ) -> Result<Option<MpsGraphExecutable>, KernelError> {
            unsafe {
                // Build feeds dictionary: MPSGraphTensor → MPSGraphShapedType
                let ns_dict_cls =
                    Class::get("NSMutableDictionary").ok_or_else(|| KernelError::Gpu {
                        message: "NSMutableDictionary class not available".into(),
                    })?;
                let feeds_dict: *mut Object = msg_send![ns_dict_cls, new];

                let shaped_type_cls =
                    Class::get("MPSGraphShapedType").ok_or_else(|| KernelError::Gpu {
                        message: "MPSGraphShapedType class not available".into(),
                    })?;
                for &(ref tensor, shape, dtype) in feeds {
                    let shape_arr = ns_array_from_usize(shape)?;
                    let shaped: *mut Object = msg_send![shaped_type_cls, alloc];
                    let shaped: *mut Object = msg_send![shaped,
                        initWithShape: shape_arr
                        dataType: dtype];
                    let _: () = msg_send![feeds_dict,
                        setObject: shaped
                        forKey: tensor.ptr];
                    let _: () = msg_send![shaped, release];
                }

                // Build targets NSArray
                let ns_array_cls = Class::get("NSArray").ok_or_else(|| KernelError::Gpu {
                    message: "NSArray class not available".into(),
                })?;
                let target_ptrs: Vec<*mut Object> = target_tensors.iter().map(|t| t.ptr).collect();
                let targets: *mut Object = msg_send![ns_array_cls,
                    arrayWithObjects: target_ptrs.as_ptr()
                    count: target_ptrs.len()];

                // Compilation descriptor (optional optimizations)
                let comp_desc_cls =
                    Class::get("MPSGraphCompilationDescriptor").ok_or_else(|| {
                        KernelError::Gpu {
                            message: "MPSGraphCompilationDescriptor class not available".into(),
                        }
                    })?;
                let comp_desc: *mut Object = msg_send![comp_desc_cls, new];
                // optimization level 1 = default optimization
                let _: () = msg_send![comp_desc,
                    setOptimizationLevel: 1u64];

                // Create MPSGraphDevice from MTLDevice
                let mpsg_device_cls =
                    Class::get("MPSGraphDevice").ok_or_else(|| KernelError::Gpu {
                        message: "MPSGraphDevice class not available".into(),
                    })?;
                let mpsg_device: *mut Object = msg_send![mpsg_device_cls,
                    deviceWithMTLDevice: device.as_ptr()];

                // Compile
                let exe: *mut Object = msg_send![self.graph,
                    compileWithDevice: mpsg_device
                    feeds: feeds_dict
                    targetTensors: targets
                    targetOperations: std::ptr::null::<Object>()
                    compilationDescriptor: comp_desc];

                let _: () = msg_send![comp_desc, release];
                let _: () = msg_send![feeds_dict, release];

                if exe.is_null() {
                    return Ok(None);
                }
                // exe is autoreleased; retain it
                let _: () = msg_send![exe, retain];
                Ok(Some(MpsGraphExecutable { exe }))
            }
        }

        /// Run the graph with Metal buffers as input.
        /// Returns output data as raw bytes in the order of `target_tensors`.
        pub fn run_with_buffers(
            &self,
            executable: &MpsGraphExecutable,
            queue: &CommandQueue,
            inputs: &[(MpsGraphTensorRef, &Buffer, &[usize], u32)], // (placeholder, buffer, shape, dtype)
        ) -> Result<Vec<Buffer>, KernelError> {
            unsafe {
                autoreleasepool(|| {
                    // Build inputsArray: [MPSGraphTensorData]
                    let tensor_data_cls =
                        Class::get("MPSGraphTensorData").ok_or_else(|| KernelError::Gpu {
                            message: "MPSGraphTensorData class not available".into(),
                        })?;
                    let ns_array_cls = Class::get("NSArray").ok_or_else(|| KernelError::Gpu {
                        message: "NSArray class not available".into(),
                    })?;

                    let mut input_datas: Vec<*mut Object> = Vec::new();
                    for &(_, buf, shape, dtype) in inputs {
                        let shape_arr = ns_array_from_usize(shape)?;
                        let alloc: *mut Object = msg_send![tensor_data_cls, alloc];
                        let td: *mut Object = msg_send![alloc,
                            initWithMTLBuffer: buf.as_ptr()
                            shape: shape_arr
                            dataType: dtype];
                        input_datas.push(td);
                    }
                    let inputs_arr: *mut Object = msg_send![ns_array_cls,
                        arrayWithObjects: input_datas.as_ptr()
                        count: input_datas.len()];

                    // Execute
                    let results: *mut Object = msg_send![executable.exe,
                        runWithMTLCommandQueue: queue.as_ptr()
                        inputsArray: inputs_arr
                        resultsArray: std::ptr::null::<Object>()
                        executionDescriptor: std::ptr::null::<Object>()];

                    // Extract output buffers
                    let count: u64 = msg_send![results, count];
                    let mut out_bufs = Vec::new();
                    for i in 0..count {
                        let td: *mut Object = msg_send![results, objectAtIndex: i];
                        // MPSGraphTensorData.mpsndarray returns an MPSNDArray.
                        // We read data from it into a shared buffer.
                        let nd_array: *mut Object = msg_send![td, mpsndarray];
                        // Get total bytes
                        let n_dims: u64 = msg_send![nd_array, numberOfDimensions];
                        let mut total_elements: u64 = 1;
                        for d in 0..n_dims {
                            let dim_size: u64 = msg_send![nd_array, lengthOfDimension: d];
                            total_elements *= dim_size;
                        }
                        // dataType from nd_array
                        let dtype: u32 = msg_send![nd_array, dataType];
                        let bytes_per_elem = if dtype == MPS_DATA_TYPE_FLOAT16 {
                            2u64
                        } else {
                            4u64
                        };
                        let total_bytes = total_elements * bytes_per_elem;

                        // Read into a new shared buffer
                        let mtl_device: &DeviceRef = queue.device();
                        let out_buf = mtl_device
                            .new_buffer(total_bytes, MTLResourceOptions::StorageModeShared);
                        let _: () = msg_send![nd_array,
                            readBytes: out_buf.contents()
                            strideBytes: std::ptr::null::<Object>()];

                        out_bufs.push(out_buf);
                    }

                    // Release input tensor datas
                    for td in &input_datas {
                        let _: () = msg_send![*td, release];
                    }

                    Ok(out_bufs)
                })
            }
        }
    }

    // Helper: create NSString from &str
    unsafe fn ns_string(s: &str) -> Result<*mut Object, KernelError> {
        let ns_string_cls = Class::get("NSString").ok_or_else(|| KernelError::Gpu {
            message: "NSString class not available".into(),
        })?;
        let ns_str: *mut Object = msg_send![ns_string_cls,
            stringWithUTF8String: s.as_ptr() as *const i8];
        Ok(ns_str)
    }

    // Helper: create NSData from raw bytes
    unsafe fn ns_data_from_bytes(ptr: *const u8, len: usize) -> Result<*mut Object, KernelError> {
        let ns_data_cls = Class::get("NSData").ok_or_else(|| KernelError::Gpu {
            message: "NSData class not available".into(),
        })?;
        let data: *mut Object = msg_send![ns_data_cls,
            dataWithBytes: ptr
            length: len];
        Ok(data)
    }
}
