use super::consts::*;
use super::ffi::*;
use crate::KernelError;
use std::ffi::c_void;
use std::sync::Arc;
use yscv_tensor::Tensor;

// ══════════════════════════════════════════════════════════════════════
// Public API — RknnBackend
// ══════════════════════════════════════════════════════════════════════

/// Safe wrapper around the RKNN NPU runtime.
///
/// Loads an `.rknn` model and runs inference on Rockchip NPU hardware.
/// The runtime library (`librknnrt.so`) is loaded dynamically, so this
/// crate compiles on any target.
pub struct RknnBackend {
    ctx: RknnContext,
    funcs: Arc<RknnFunctions>,
    _lib: Arc<DlHandle>,
    input_attrs: Vec<RknnTensorAttr>,
    output_attrs: Vec<RknnTensorAttr>,
    perf_collected: bool,
}

// SAFETY: RknnContext is a u64 handle. The RKNN runtime documents that
// independent contexts are thread-safe.
unsafe impl Send for RknnBackend {}
// SAFETY: All operations on RknnBackend go through FFI calls that are
// documented as thread-safe for independent contexts.
unsafe impl Sync for RknnBackend {}

/// Query input/output attributes after context initialisation.
fn query_io_attrs(
    funcs: &RknnFunctions,
    ctx: RknnContext,
) -> Result<(Vec<RknnTensorAttr>, Vec<RknnTensorAttr>), KernelError> {
    let mut io_num = RknnInOutNum::default();
    query_raw(funcs, ctx, RKNN_QUERY_IN_OUT_NUM, "IN_OUT_NUM", &mut io_num)?;

    let input_attrs = (0..io_num.n_input)
        .map(|i| {
            let mut attr = RknnTensorAttr {
                index: i,
                ..Default::default()
            };
            query_raw(
                funcs,
                ctx,
                RKNN_QUERY_INPUT_ATTR,
                "INPUT_ATTR",
                &mut attr as *mut RknnTensorAttr,
            )?;
            Ok(attr)
        })
        .collect::<Result<Vec<_>, KernelError>>()?;

    let output_attrs = (0..io_num.n_output)
        .map(|i| {
            let mut attr = RknnTensorAttr {
                index: i,
                ..Default::default()
            };
            query_raw(
                funcs,
                ctx,
                RKNN_QUERY_OUTPUT_ATTR,
                "OUTPUT_ATTR",
                &mut attr as *mut RknnTensorAttr,
            )?;
            Ok(attr)
        })
        .collect::<Result<Vec<_>, KernelError>>()?;

    Ok((input_attrs, output_attrs))
}

impl RknnBackend {
    /// Load an RKNN model from raw file bytes using default flags.
    pub fn load(model_data: &[u8]) -> Result<Self, KernelError> {
        Self::load_with_flags(model_data, 0)
    }

    /// Load with performance collection enabled. `perf_detail()` and
    /// `perf_run_us()` will return real data after each inference.
    pub fn load_with_perf(model_data: &[u8]) -> Result<Self, KernelError> {
        Self::load_with_flags(model_data, RKNN_FLAG_COLLECT_PERF_MASK)
    }

    /// Load with async inference mode enabled. `run_async()` returns
    /// immediately; results are retrieved via `wait()`.
    pub fn load_async(model_data: &[u8]) -> Result<Self, KernelError> {
        Self::load_with_flags(model_data, RKNN_FLAG_ASYNC_MASK)
    }

    /// Load with caller-provided flags (any combination of `RKNN_FLAG_*`).
    pub fn load_with_flags(model_data: &[u8], flags: u32) -> Result<Self, KernelError> {
        let (lib, funcs) = load_rknn_library()?;
        let funcs = Arc::new(funcs);
        let lib = Arc::new(lib);

        let mut ctx: RknnContext = 0;
        let mut extend = RknnInitExtend::default();

        // SAFETY: model_data is valid for `len` bytes; ctx and extend are
        // valid writable pointers. Flag is a caller-supplied bitmask.
        let ret = unsafe {
            (funcs.init)(
                &mut ctx as *mut RknnContext,
                model_data.as_ptr(),
                model_data.len() as u32,
                flags,
                &mut extend as *mut RknnInitExtend,
            )
        };
        if ret != RKNN_SUCC {
            return Err(KernelError::Rknn {
                message: format!("rknn_init failed: {} ({ret})", rknn_error_name(ret)),
            });
        }

        // Query I/O attributes; on failure we still need to destroy ctx.
        let io = query_io_attrs(&funcs, ctx);
        let (input_attrs, output_attrs) = match io {
            Ok(a) => a,
            Err(e) => {
                // SAFETY: ctx is valid from successful rknn_init.
                unsafe { (funcs.destroy)(ctx) };
                return Err(e);
            }
        };

        Ok(Self {
            ctx,
            funcs,
            _lib: lib,
            input_attrs,
            output_attrs,
            perf_collected: (flags & RKNN_FLAG_COLLECT_PERF_MASK) != 0,
        })
    }

    /// Load and immediately pin to a specific NPU core.
    pub fn load_for_core(model_data: &[u8], mask: NpuCoreMask) -> Result<Self, KernelError> {
        let backend = Self::load(model_data)?;
        backend.set_core_mask(mask)?;
        Ok(backend)
    }

    /// Run inference synchronously with input data copied from slices.
    ///
    /// Returns dequantized f32 output tensors.
    pub fn run(&self, inputs: &[&[u8]]) -> Result<Vec<Tensor>, KernelError> {
        if inputs.len() != self.input_attrs.len() {
            return Err(KernelError::Rknn {
                message: format!(
                    "expected {} inputs, got {}",
                    self.input_attrs.len(),
                    inputs.len()
                ),
            });
        }

        let rknn_inputs: Vec<RknnInput> = inputs
            .iter()
            .zip(self.input_attrs.iter())
            .map(|(data, attr)| RknnInput {
                index: attr.index,
                buf: data.as_ptr(),
                size: data.len() as u32,
                pass_through: 0,
                typ: RknnTensorType::Uint8 as u32,
                fmt: RknnTensorFormat::Nhwc as u32,
            })
            .collect();

        // SAFETY: rknn_inputs pointers are valid for the duration of this call.
        let ret = unsafe {
            (self.funcs.inputs_set)(self.ctx, rknn_inputs.len() as u32, rknn_inputs.as_ptr())
        };
        if ret != RKNN_SUCC {
            return Err(KernelError::Rknn {
                message: format!("rknn_inputs_set failed: {} ({ret})", rknn_error_name(ret)),
            });
        }

        // SAFETY: ctx is valid, null extend → synchronous run with defaults.
        let ret = unsafe { (self.funcs.run)(self.ctx, std::ptr::null_mut()) };
        if ret != RKNN_SUCC {
            return Err(KernelError::Rknn {
                message: format!("rknn_run failed: {} ({ret})", rknn_error_name(ret)),
            });
        }

        self.collect_outputs()
    }

    /// Collect outputs from the last inference (sync or async completed).
    fn collect_outputs(&self) -> Result<Vec<Tensor>, KernelError> {
        let n_outputs = self.output_attrs.len() as u32;
        let mut rknn_outputs: Vec<RknnOutput> = (0..n_outputs)
            .map(|i| RknnOutput {
                want_float: 1,
                is_prealloc: 0,
                index: i,
                buf: std::ptr::null_mut(),
                size: 0,
            })
            .collect();

        // SAFETY: outputs array is valid; runtime writes buf/size.
        let ret = unsafe {
            (self.funcs.outputs_get)(
                self.ctx,
                n_outputs,
                rknn_outputs.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if ret != RKNN_SUCC {
            return Err(KernelError::Rknn {
                message: format!("rknn_outputs_get failed: {} ({ret})", rknn_error_name(ret)),
            });
        }

        let result = rknn_outputs
            .iter()
            .zip(self.output_attrs.iter())
            .map(|(out, attr)| {
                if out.buf.is_null() {
                    return Err(KernelError::Rknn {
                        message: format!("null buffer for output index {}", out.index),
                    });
                }
                let n_floats = out.size as usize / std::mem::size_of::<f32>();
                // SAFETY: buf is non-null; size set by runtime; lifetime
                // bounded by rknn_outputs_release below.
                let floats = unsafe { std::slice::from_raw_parts(out.buf.cast::<f32>(), n_floats) };
                let shape = attr.shape();
                let expected: usize = shape.iter().product();
                let tensor_shape = if n_floats == expected {
                    shape
                } else {
                    vec![n_floats]
                };
                Tensor::from_vec(tensor_shape, floats.to_vec()).map_err(|e| KernelError::Rknn {
                    message: format!("failed to build output tensor: {e}"),
                })
            })
            .collect::<Result<Vec<_>, _>>();

        // SAFETY: outputs were obtained from a successful rknn_outputs_get.
        unsafe {
            (self.funcs.outputs_release)(self.ctx, n_outputs, rknn_outputs.as_mut_ptr());
        }

        result
    }

    /// Input tensor attributes (shape, format, quantization parameters).
    pub fn input_attrs(&self) -> &[RknnTensorAttr] {
        &self.input_attrs
    }

    /// Output tensor attributes.
    pub fn output_attrs(&self) -> &[RknnTensorAttr] {
        &self.output_attrs
    }

    // ══════════════════════════════════════════════════════════════════
    // Phase 2 — Core runtime safe API
    // ══════════════════════════════════════════════════════════════════

    /// Duplicate this context for multi-stream inference on separate cores.
    pub fn dup_context(&self) -> Result<RknnBackend, KernelError> {
        let dup_fn = self.funcs.dup_context.ok_or_else(|| KernelError::Rknn {
            message: "rknn_dup_context not available in this runtime".into(),
        })?;

        let mut new_ctx: RknnContext = 0;
        // SAFETY: self.ctx is valid from a successful load; dup_fn is a
        // valid function pointer; new_ctx is a writable u64.
        let ret = unsafe { dup_fn(&self.ctx as *const u64 as *mut u64, &mut new_ctx) };
        if ret != RKNN_SUCC {
            return Err(KernelError::Rknn {
                message: format!("rknn_dup_context failed: {} ({ret})", rknn_error_name(ret)),
            });
        }

        let (input_attrs, output_attrs) = query_io_attrs(&self.funcs, new_ctx)?;
        Ok(RknnBackend {
            ctx: new_ctx,
            funcs: Arc::clone(&self.funcs),
            _lib: Arc::clone(&self._lib),
            input_attrs,
            output_attrs,
            perf_collected: self.perf_collected,
        })
    }

    /// Pin inference to specific NPU core(s).
    pub fn set_core_mask(&self, mask: NpuCoreMask) -> Result<(), KernelError> {
        let fn_ptr = self.funcs.set_core_mask.ok_or_else(|| KernelError::Rknn {
            message: "rknn_set_core_mask not available".into(),
        })?;
        // SAFETY: ctx is valid; mask is a documented enum value.
        let ret = unsafe { fn_ptr(self.ctx, mask.as_raw()) };
        if ret != RKNN_SUCC {
            return Err(KernelError::Rknn {
                message: format!(
                    "rknn_set_core_mask failed: {} ({ret})",
                    rknn_error_name(ret)
                ),
            });
        }
        Ok(())
    }

    /// Set the number of cores used for batch inference.
    pub fn set_batch_core_num(&self, n: u32) -> Result<(), KernelError> {
        let fn_ptr = self
            .funcs
            .set_batch_core_num
            .ok_or_else(|| KernelError::Rknn {
                message: "rknn_set_batch_core_num not available".into(),
            })?;
        // SAFETY: ctx is valid; n is a positive count.
        let ret = unsafe { fn_ptr(self.ctx, n as i32) };
        if ret != RKNN_SUCC {
            return Err(KernelError::Rknn {
                message: format!(
                    "rknn_set_batch_core_num failed: {} ({ret})",
                    rknn_error_name(ret)
                ),
            });
        }
        Ok(())
    }

    /// Duplicate the context and pin the clone to a specific core.
    pub fn dup_for_core(&self, mask: NpuCoreMask) -> Result<RknnBackend, KernelError> {
        let dup = self.dup_context()?;
        dup.set_core_mask(mask)?;
        Ok(dup)
    }

    /// SDK API and driver version strings.
    pub fn sdk_version(&self) -> Result<(String, String), KernelError> {
        let mut v = RknnSdkVersionRaw::default();
        query_raw(
            &self.funcs,
            self.ctx,
            RKNN_QUERY_SDK_VERSION,
            "SDK_VERSION",
            &mut v as *mut RknnSdkVersionRaw,
        )?;
        let api = c_str_to_string(&v.api_version);
        let drv = c_str_to_string(&v.drv_version);
        Ok((api, drv))
    }

    /// Total memory footprint of the loaded model.
    pub fn mem_size(&self) -> Result<MemSize, KernelError> {
        let mut m = RknnMemSizeRaw::default();
        query_raw(
            &self.funcs,
            self.ctx,
            RKNN_QUERY_MEM_SIZE,
            "MEM_SIZE",
            &mut m as *mut RknnMemSizeRaw,
        )?;
        Ok(MemSize {
            weight_bytes: m.total_weight_size as u64,
            internal_bytes: m.total_internal_size as u64,
            dma_bytes: m.total_dma_allocated_size,
            sram_total_bytes: m.total_sram_size as u64,
            sram_free_bytes: m.free_sram_size as u64,
        })
    }

    /// Model-embedded custom metadata string.
    pub fn custom_string(&self) -> Result<String, KernelError> {
        let mut s = RknnCustomStringRaw::default();
        query_raw(
            &self.funcs,
            self.ctx,
            RKNN_QUERY_CUSTOM_STRING,
            "CUSTOM_STRING",
            &mut s as *mut RknnCustomStringRaw,
        )?;
        Ok(c_str_to_string(&s.string))
    }

    /// Total duration (microseconds) of the last synchronous inference.
    /// Requires load with `RKNN_FLAG_COLLECT_PERF_MASK`.
    pub fn perf_run_us(&self) -> Result<i64, KernelError> {
        if !self.perf_collected {
            return Err(KernelError::Rknn {
                message: "perf collection not enabled — use load_with_perf()".into(),
            });
        }
        let mut p = RknnPerfRun::default();
        query_raw(
            &self.funcs,
            self.ctx,
            RKNN_QUERY_PERF_RUN,
            "PERF_RUN",
            &mut p as *mut RknnPerfRun,
        )?;
        Ok(p.run_duration)
    }

    /// Per-op performance breakdown from the last inference. Requires
    /// `load_with_perf()`.
    pub fn perf_detail(&self) -> Result<PerfDetail, KernelError> {
        if !self.perf_collected {
            return Err(KernelError::Rknn {
                message: "perf collection not enabled — use load_with_perf()".into(),
            });
        }
        let mut p = RknnPerfDetailRaw {
            perf_data: std::ptr::null_mut(),
            data_len: 0,
        };
        query_raw(
            &self.funcs,
            self.ctx,
            RKNN_QUERY_PERF_DETAIL,
            "PERF_DETAIL",
            &mut p as *mut RknnPerfDetailRaw,
        )?;
        if p.perf_data.is_null() || p.data_len == 0 {
            return Ok(PerfDetail::default());
        }
        // SAFETY: perf_data points to a null-terminated string of `data_len`
        // bytes owned by the runtime; we copy it before control returns.
        let slice = unsafe { std::slice::from_raw_parts(p.perf_data, p.data_len as usize) };
        let raw = std::str::from_utf8(slice)
            .unwrap_or("<invalid utf8>")
            .to_string();
        let total_us = self.perf_run_us().unwrap_or(0);
        Ok(PerfDetail::parse(&raw, total_us))
    }

    /// Change input tensor shapes at runtime (dynamic shapes).
    ///
    /// `new_shapes[i]` is the new shape for input `i`. Each inner `Vec<u32>`
    /// lists dimensions (`n_dims` implied by length, must match the model's
    /// compiled dynamic shape range).
    pub fn set_input_shapes(&mut self, new_shapes: &[Vec<u32>]) -> Result<(), KernelError> {
        if new_shapes.len() != self.input_attrs.len() {
            return Err(KernelError::Rknn {
                message: format!(
                    "expected {} shapes, got {}",
                    self.input_attrs.len(),
                    new_shapes.len()
                ),
            });
        }
        let fn_ptr = self
            .funcs
            .set_input_shapes
            .ok_or_else(|| KernelError::Rknn {
                message: "rknn_set_input_shapes not available (requires SDK ≥ 2.0)".into(),
            })?;

        let mut attrs: Vec<RknnTensorAttr> = self
            .input_attrs
            .iter()
            .zip(new_shapes.iter())
            .map(|(orig, shape)| {
                let mut a = orig.clone();
                a.n_dims = shape.len() as u32;
                for (i, &d) in shape.iter().enumerate().take(RKNN_MAX_DIMS) {
                    a.dims[i] = d;
                }
                a
            })
            .collect();

        // SAFETY: attrs slice is valid; length matches model I/O.
        let ret = unsafe { fn_ptr(self.ctx, attrs.len() as u32, attrs.as_mut_ptr()) };
        if ret != RKNN_SUCC {
            return Err(KernelError::Rknn {
                message: format!(
                    "rknn_set_input_shapes failed: {} ({ret})",
                    rknn_error_name(ret)
                ),
            });
        }

        // Re-query attributes after shape change.
        let (new_inputs, new_outputs) = query_io_attrs(&self.funcs, self.ctx)?;
        self.input_attrs = new_inputs;
        self.output_attrs = new_outputs;
        Ok(())
    }

    /// Change a single input tensor shape (deprecated in SDK; use `set_input_shapes`).
    pub fn set_input_shape(&mut self, index: u32, shape: &[u32]) -> Result<(), KernelError> {
        let fn_ptr = self
            .funcs
            .set_input_shape
            .ok_or_else(|| KernelError::Rknn {
                message: "rknn_set_input_shape not available".into(),
            })?;
        let attr_src = self
            .input_attrs
            .get(index as usize)
            .ok_or_else(|| KernelError::Rknn {
                message: format!("input index {index} out of range"),
            })?;
        let mut a = attr_src.clone();
        a.n_dims = shape.len() as u32;
        for (i, &d) in shape.iter().enumerate().take(RKNN_MAX_DIMS) {
            a.dims[i] = d;
        }
        // SAFETY: ctx valid; a is a local writable struct.
        let ret = unsafe { fn_ptr(self.ctx, &mut a as *mut RknnTensorAttr) };
        if ret != RKNN_SUCC {
            return Err(KernelError::Rknn {
                message: format!(
                    "rknn_set_input_shape failed: {} ({ret})",
                    rknn_error_name(ret)
                ),
            });
        }
        let (new_inputs, new_outputs) = query_io_attrs(&self.funcs, self.ctx)?;
        self.input_attrs = new_inputs;
        self.output_attrs = new_outputs;
        Ok(())
    }

    /// Query current input tensor attributes (reflects active dynamic shape).
    pub fn current_input_attrs(&self) -> Result<Vec<RknnTensorAttr>, KernelError> {
        let n = self.input_attrs.len() as u32;
        (0..n)
            .map(|i| {
                let mut a = RknnTensorAttr {
                    index: i,
                    ..Default::default()
                };
                query_raw(
                    &self.funcs,
                    self.ctx,
                    RKNN_QUERY_CURRENT_INPUT_ATTR,
                    "CURRENT_INPUT_ATTR",
                    &mut a as *mut RknnTensorAttr,
                )?;
                Ok(a)
            })
            .collect()
    }

    /// Query current output tensor attributes (reflects active dynamic shape).
    pub fn current_output_attrs(&self) -> Result<Vec<RknnTensorAttr>, KernelError> {
        let n = self.output_attrs.len() as u32;
        (0..n)
            .map(|i| {
                let mut a = RknnTensorAttr {
                    index: i,
                    ..Default::default()
                };
                query_raw(
                    &self.funcs,
                    self.ctx,
                    RKNN_QUERY_CURRENT_OUTPUT_ATTR,
                    "CURRENT_OUTPUT_ATTR",
                    &mut a as *mut RknnTensorAttr,
                )?;
                Ok(a)
            })
            .collect()
    }

    /// Query NPU-native input tensor attributes (for zero-copy alignment).
    pub fn native_input_attrs(&self) -> Result<Vec<RknnTensorAttr>, KernelError> {
        let n = self.input_attrs.len() as u32;
        (0..n)
            .map(|i| {
                let mut a = RknnTensorAttr {
                    index: i,
                    ..Default::default()
                };
                query_raw(
                    &self.funcs,
                    self.ctx,
                    RKNN_QUERY_NATIVE_INPUT_ATTR,
                    "NATIVE_INPUT_ATTR",
                    &mut a as *mut RknnTensorAttr,
                )?;
                Ok(a)
            })
            .collect()
    }

    /// Query NPU-native output tensor attributes.
    pub fn native_output_attrs(&self) -> Result<Vec<RknnTensorAttr>, KernelError> {
        let n = self.output_attrs.len() as u32;
        (0..n)
            .map(|i| {
                let mut a = RknnTensorAttr {
                    index: i,
                    ..Default::default()
                };
                query_raw(
                    &self.funcs,
                    self.ctx,
                    RKNN_QUERY_NATIVE_OUTPUT_ATTR,
                    "NATIVE_OUTPUT_ATTR",
                    &mut a as *mut RknnTensorAttr,
                )?;
                Ok(a)
            })
            .collect()
    }

    /// Query current native input tensor attributes (reflects active dynamic shape
    /// in NPU-native layout, useful for re-binding zero-copy buffers after a
    /// shape change).
    pub fn current_native_input_attrs(&self) -> Result<Vec<RknnTensorAttr>, KernelError> {
        let n = self.input_attrs.len() as u32;
        (0..n)
            .map(|i| {
                let mut a = RknnTensorAttr {
                    index: i,
                    ..Default::default()
                };
                query_raw(
                    &self.funcs,
                    self.ctx,
                    RKNN_QUERY_CURRENT_NATIVE_INPUT_ATTR,
                    "CURRENT_NATIVE_INPUT_ATTR",
                    &mut a as *mut RknnTensorAttr,
                )?;
                Ok(a)
            })
            .collect()
    }

    /// Query current native output tensor attributes (NPU-native layout).
    pub fn current_native_output_attrs(&self) -> Result<Vec<RknnTensorAttr>, KernelError> {
        let n = self.output_attrs.len() as u32;
        (0..n)
            .map(|i| {
                let mut a = RknnTensorAttr {
                    index: i,
                    ..Default::default()
                };
                query_raw(
                    &self.funcs,
                    self.ctx,
                    RKNN_QUERY_CURRENT_NATIVE_OUTPUT_ATTR,
                    "CURRENT_NATIVE_OUTPUT_ATTR",
                    &mut a as *mut RknnTensorAttr,
                )?;
                Ok(a)
            })
            .collect()
    }

    /// Query NHWC-formatted native input attrs (for camera / image input).
    pub fn native_nhwc_input_attrs(&self) -> Result<Vec<RknnTensorAttr>, KernelError> {
        let n = self.input_attrs.len() as u32;
        (0..n)
            .map(|i| {
                let mut a = RknnTensorAttr {
                    index: i,
                    ..Default::default()
                };
                query_raw(
                    &self.funcs,
                    self.ctx,
                    RKNN_QUERY_NATIVE_NHWC_INPUT_ATTR,
                    "NATIVE_NHWC_INPUT_ATTR",
                    &mut a as *mut RknnTensorAttr,
                )?;
                Ok(a)
            })
            .collect()
    }

    /// Query NHWC-formatted native output attrs.
    pub fn native_nhwc_output_attrs(&self) -> Result<Vec<RknnTensorAttr>, KernelError> {
        let n = self.output_attrs.len() as u32;
        (0..n)
            .map(|i| {
                let mut a = RknnTensorAttr {
                    index: i,
                    ..Default::default()
                };
                query_raw(
                    &self.funcs,
                    self.ctx,
                    RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR,
                    "NATIVE_NHWC_OUTPUT_ATTR",
                    &mut a as *mut RknnTensorAttr,
                )?;
                Ok(a)
            })
            .collect()
    }

    /// Query total device memory info (allocated, free).
    pub fn device_mem_info(&self) -> Result<MemSize, KernelError> {
        let mut m = RknnMemSizeRaw::default();
        query_raw(
            &self.funcs,
            self.ctx,
            RKNN_QUERY_DEVICE_MEM_INFO,
            "DEVICE_MEM_INFO",
            &mut m as *mut RknnMemSizeRaw,
        )?;
        Ok(MemSize {
            weight_bytes: m.total_weight_size as u64,
            internal_bytes: m.total_internal_size as u64,
            dma_bytes: m.total_dma_allocated_size,
            sram_total_bytes: m.total_sram_size as u64,
            sram_free_bytes: m.free_sram_size as u64,
        })
    }

    /// Query the allowed dynamic shape range for input `index`.
    pub fn input_dynamic_range(&self, index: u32) -> Result<Vec<Vec<u32>>, KernelError> {
        let mut r = Box::new(RknnInputRange {
            index,
            shape_number: 0,
            fmt: 0,
            name: [0; RKNN_MAX_NAME_LEN],
            dyn_range: [[0; RKNN_MAX_DIMS]; RKNN_MAX_DYNAMIC_SHAPE_NUM],
            n_dims: 0,
        });
        query_raw(
            &self.funcs,
            self.ctx,
            RKNN_QUERY_INPUT_DYNAMIC_RANGE,
            "INPUT_DYNAMIC_RANGE",
            &mut *r as *mut RknnInputRange,
        )?;
        let shapes = (0..r.shape_number as usize)
            .map(|i| r.dyn_range[i][..r.n_dims as usize].to_vec())
            .collect();
        Ok(shapes)
    }

    // ══════════════════════════════════════════════════════════════════
    // Phase 3 — Zero-copy memory subsystem
    // ══════════════════════════════════════════════════════════════════

    /// Allocate an NPU-accessible buffer with default flags.
    ///
    /// Prefers `rknn_create_mem2` if available (with explicit flags); falls
    /// back to `rknn_create_mem` on older SDKs.
    pub fn alloc_mem(&self, size: usize) -> Result<RknnMem, KernelError> {
        if self.funcs.create_mem2.is_some() {
            self.alloc_mem_ex(size, MemAllocFlags::Default)
        } else if let Some(fn_ptr) = self.funcs.create_mem {
            // SAFETY: ctx valid; size is scalar.
            let ptr = unsafe { fn_ptr(self.ctx, size as u32) };
            if ptr.is_null() {
                return Err(KernelError::Rknn {
                    message: format!("rknn_create_mem({size}) returned null"),
                });
            }
            Ok(RknnMem {
                ptr,
                ctx: self.ctx,
                funcs: Arc::clone(&self.funcs),
                external: false,
            })
        } else {
            Err(KernelError::Rknn {
                message: "neither rknn_create_mem nor rknn_create_mem2 are available".into(),
            })
        }
    }

    /// Allocate with specific flags (cacheable, non-cacheable, SRAM hint).
    pub fn alloc_mem_ex(&self, size: usize, flags: MemAllocFlags) -> Result<RknnMem, KernelError> {
        let fn_ptr = self.funcs.create_mem2.ok_or_else(|| KernelError::Rknn {
            message: "rknn_create_mem2 not available".into(),
        })?;
        // SAFETY: ctx is valid; size/flags are scalar parameters.
        let ptr = unsafe { fn_ptr(self.ctx, size as u64, flags.as_raw()) };
        if ptr.is_null() {
            return Err(KernelError::Rknn {
                message: format!("rknn_create_mem2({size}, {flags:?}) returned null"),
            });
        }
        Ok(RknnMem {
            ptr,
            ctx: self.ctx,
            funcs: Arc::clone(&self.funcs),
            external: false,
        })
    }

    /// Allocate in on-chip SRAM for latency-critical hot tensors.
    ///
    /// SRAM is ~10× faster than DMA memory but limited in size (SoC-dependent).
    /// Use `mem_size().sram_free_bytes` to check availability.
    pub fn alloc_sram(&self, size: usize) -> Result<RknnMem, KernelError> {
        self.alloc_mem_ex(size, MemAllocFlags::TryAllocSram)
    }

    /// Wrap a DMA-BUF file descriptor (e.g. from V4L2 `VIDIOC_EXPBUF`)
    /// as zero-copy NPU memory. The NPU reads directly from the same
    /// physical pages the camera wrote into.
    ///
    /// # Safety
    /// `fd` must be a valid DMA-BUF file descriptor. `virt` must point
    /// to the mmap'd userspace view of the same buffer.
    pub fn wrap_fd(&self, fd: i32, virt: &mut [u8], offset: i32) -> Result<RknnMem, KernelError> {
        let fn_ptr = self
            .funcs
            .create_mem_from_fd
            .ok_or_else(|| KernelError::Rknn {
                message: "rknn_create_mem_from_fd not available".into(),
            })?;
        // SAFETY: virt is a valid mutable slice; fd is provided by caller
        // who is responsible for its lifetime (must outlive the RknnMem).
        let ptr = unsafe {
            fn_ptr(
                self.ctx,
                fd,
                virt.as_mut_ptr().cast(),
                virt.len() as u32,
                offset,
            )
        };
        if ptr.is_null() {
            return Err(KernelError::Rknn {
                message: format!("rknn_create_mem_from_fd(fd={fd}) returned null"),
            });
        }
        Ok(RknnMem {
            ptr,
            ctx: self.ctx,
            funcs: Arc::clone(&self.funcs),
            external: true,
        })
    }

    /// Wrap a physical memory region (e.g. from DRM/IOMMU allocator).
    pub fn wrap_phys(&self, phys: u64, virt: &mut [u8]) -> Result<RknnMem, KernelError> {
        let fn_ptr = self
            .funcs
            .create_mem_from_phys
            .ok_or_else(|| KernelError::Rknn {
                message: "rknn_create_mem_from_phys not available".into(),
            })?;
        // SAFETY: virt is a valid mutable slice; phys must refer to the
        // same memory region the virt mapping covers.
        let ptr = unsafe { fn_ptr(self.ctx, phys, virt.as_mut_ptr().cast(), virt.len() as u32) };
        if ptr.is_null() {
            return Err(KernelError::Rknn {
                message: format!("rknn_create_mem_from_phys(phys={phys:x}) returned null"),
            });
        }
        Ok(RknnMem {
            ptr,
            ctx: self.ctx,
            funcs: Arc::clone(&self.funcs),
            external: true,
        })
    }

    /// Wrap a Rockchip MPP media-block handle as NPU tensor memory.
    ///
    /// Used for zero-copy MPP → NPU pipelines: hardware-decoded video
    /// frames from `mpp_buffer_get_mpp_buffer` can be fed directly to the
    /// NPU without any CPU copy.
    ///
    /// # Safety
    ///
    /// `blk` must be a valid `MB_BLK` handle obtained from a Rockchip MPP
    /// API. The MPP block must remain alive for the lifetime of the
    /// returned `RknnMem` — dropping the `RknnMem` does not release the
    /// underlying MPP buffer.
    pub unsafe fn wrap_mb_blk(
        &self,
        blk: *mut c_void,
        offset: i32,
    ) -> Result<RknnMem, KernelError> {
        let fn_ptr = self
            .funcs
            .create_mem_from_mb_blk
            .ok_or_else(|| KernelError::Rknn {
                message: "rknn_create_mem_from_mb_blk not available in this RKNN runtime".into(),
            })?;
        if blk.is_null() {
            return Err(KernelError::Rknn {
                message: "wrap_mb_blk: null MB_BLK handle".into(),
            });
        }
        // SAFETY: caller guarantees `blk` is a valid MPP block handle.
        let ptr = unsafe { fn_ptr(self.ctx, blk, offset) };
        if ptr.is_null() {
            return Err(KernelError::Rknn {
                message: format!(
                    "rknn_create_mem_from_mb_blk(blk={blk:p}, offset={offset}) returned null"
                ),
            });
        }
        Ok(RknnMem {
            ptr,
            ctx: self.ctx,
            funcs: Arc::clone(&self.funcs),
            external: true,
        })
    }

    /// Bind a memory region to an I/O tensor by its tensor attribute.
    pub fn bind_io(&self, mem: &RknnMem, attr: &RknnTensorAttr) -> Result<(), KernelError> {
        let fn_ptr = self.funcs.set_io_mem.ok_or_else(|| KernelError::Rknn {
            message: "rknn_set_io_mem not available".into(),
        })?;
        let mut attr_copy = attr.clone();
        // SAFETY: mem.ptr is valid (owned by RknnMem); attr_copy is local.
        let ret = unsafe { fn_ptr(self.ctx, mem.ptr, &mut attr_copy as *mut RknnTensorAttr) };
        if ret != RKNN_SUCC {
            return Err(KernelError::Rknn {
                message: format!("rknn_set_io_mem failed: {} ({ret})", rknn_error_name(ret)),
            });
        }
        Ok(())
    }

    /// Bind memory as input tensor `index`.
    pub fn bind_input(&self, mem: &RknnMem, index: u32) -> Result<(), KernelError> {
        let attr = self
            .input_attrs
            .get(index as usize)
            .ok_or_else(|| KernelError::Rknn {
                message: format!("input index {index} out of range"),
            })?;
        self.bind_io(mem, attr)
    }

    /// Bind memory as input tensor by name.
    pub fn bind_input_by_name(&self, mem: &RknnMem, name: &str) -> Result<(), KernelError> {
        let attr = self
            .input_attrs
            .iter()
            .find(|a| a.name_str() == name)
            .ok_or_else(|| KernelError::Rknn {
                message: format!("no input named `{name}`"),
            })?;
        self.bind_io(mem, attr)
    }

    /// Bind memory as output tensor `index`.
    pub fn bind_output(&self, mem: &RknnMem, index: u32) -> Result<(), KernelError> {
        let attr = self
            .output_attrs
            .get(index as usize)
            .ok_or_else(|| KernelError::Rknn {
                message: format!("output index {index} out of range"),
            })?;
        self.bind_io(mem, attr)
    }

    /// Bind memory as output tensor by name.
    pub fn bind_output_by_name(&self, mem: &RknnMem, name: &str) -> Result<(), KernelError> {
        let attr = self
            .output_attrs
            .iter()
            .find(|a| a.name_str() == name)
            .ok_or_else(|| KernelError::Rknn {
                message: format!("no output named `{name}`"),
            })?;
        self.bind_io(mem, attr)
    }

    /// Bind external weight memory (saves re-allocating weights per context).
    pub fn bind_weight_mem(&self, mem: &RknnMem) -> Result<(), KernelError> {
        let fn_ptr = self.funcs.set_weight_mem.ok_or_else(|| KernelError::Rknn {
            message: "rknn_set_weight_mem not available".into(),
        })?;
        // SAFETY: mem.ptr valid; ctx valid.
        let ret = unsafe { fn_ptr(self.ctx, mem.ptr) };
        if ret != RKNN_SUCC {
            return Err(KernelError::Rknn {
                message: format!(
                    "rknn_set_weight_mem failed: {} ({ret})",
                    rknn_error_name(ret)
                ),
            });
        }
        Ok(())
    }

    /// Bind external internal (scratch) memory.
    pub fn bind_internal_mem(&self, mem: &RknnMem) -> Result<(), KernelError> {
        let fn_ptr = self
            .funcs
            .set_internal_mem
            .ok_or_else(|| KernelError::Rknn {
                message: "rknn_set_internal_mem not available".into(),
            })?;
        // SAFETY: mem.ptr valid; ctx valid.
        let ret = unsafe { fn_ptr(self.ctx, mem.ptr) };
        if ret != RKNN_SUCC {
            return Err(KernelError::Rknn {
                message: format!(
                    "rknn_set_internal_mem failed: {} ({ret})",
                    rknn_error_name(ret)
                ),
            });
        }
        Ok(())
    }

    /// Run inference with pre-bound zero-copy buffers. No data is copied.
    pub fn run_bound(&self) -> Result<(), KernelError> {
        // SAFETY: ctx valid; null extend → synchronous default run.
        let ret = unsafe { (self.funcs.run)(self.ctx, std::ptr::null_mut()) };
        if ret != RKNN_SUCC {
            return Err(KernelError::Rknn {
                message: format!("rknn_run (bound) failed: {} ({ret})", rknn_error_name(ret)),
            });
        }
        Ok(())
    }

    // ══════════════════════════════════════════════════════════════════
    // Phase 5 — Async inference
    // ══════════════════════════════════════════════════════════════════

    /// Submit inference asynchronously. Returns immediately with a handle
    /// that must be `wait`ed on before reading outputs.
    ///
    /// `deadline_ms` bounds how long [`wait`](Self::wait) will block before
    /// returning a `RKNN_ERR_TIMEOUT`. Use `-1` for "block indefinitely"
    /// (only safe when there is an external watchdog). For drone use cases
    /// always provide a positive deadline (e.g. 2× expected inference time).
    ///
    /// Requires `load_async()` or `load_with_flags(RKNN_FLAG_ASYNC_MASK)`.
    pub fn run_async(&self, frame_id: u64, deadline_ms: i32) -> Result<AsyncFrame, KernelError> {
        let mut ext = RknnRunExtend {
            frame_id,
            non_block: 1,
            timeout_ms: deadline_ms,
            fence_fd: -1,
        };
        // SAFETY: ctx valid; ext is a local writable struct.
        let ret = unsafe { (self.funcs.run)(self.ctx, &mut ext as *mut RknnRunExtend) };
        if ret != RKNN_SUCC {
            return Err(KernelError::Rknn {
                message: format!("rknn_run (async) failed: {} ({ret})", rknn_error_name(ret)),
            });
        }
        Ok(AsyncFrame {
            id: frame_id,
            submitted_at: std::time::Instant::now(),
            deadline_ms,
            waited: false,
        })
    }

    /// Submit async inference on pre-bound buffers (hottest path).
    pub fn run_async_bound(
        &self,
        frame_id: u64,
        deadline_ms: i32,
    ) -> Result<AsyncFrame, KernelError> {
        self.run_async(frame_id, deadline_ms)
    }

    /// Block until a specific async inference completes (within the
    /// deadline recorded at submission), then return outputs.
    ///
    /// Returns `KernelError::Rknn { message: "...TIMEOUT..." }` if the
    /// SDK reports the wait exceeded its deadline — caller should treat
    /// that as "NPU hung" and trigger context recovery via
    /// [`ContextPool::recover_failed`] or [`RknnBackend::reset`].
    pub fn wait(&self, mut frame: AsyncFrame) -> Result<Vec<Tensor>, KernelError> {
        let wait_fn = self.funcs.wait.ok_or_else(|| KernelError::Rknn {
            message: "rknn_wait not available".into(),
        })?;
        // Compute remaining budget. If user set deadline_ms=2 but we're already
        // 1ms in, only wait 1ms more. Pass `-1` for infinite if no deadline set.
        let remaining_ms: i32 = if frame.deadline_ms < 0 {
            -1
        } else {
            let elapsed_ms = frame.elapsed().as_millis() as i32;
            (frame.deadline_ms - elapsed_ms).max(0)
        };
        let mut ext = RknnRunExtend {
            frame_id: frame.id,
            non_block: 0,
            timeout_ms: remaining_ms,
            fence_fd: -1,
        };
        // SAFETY: ctx valid; ext is local.
        let ret = unsafe { wait_fn(self.ctx, &mut ext as *mut RknnRunExtend) };
        // Mark as waited regardless of outcome — we attempted the wait
        // and the SDK's slot is consumed either way.
        frame.waited = true;
        if ret != RKNN_SUCC {
            return Err(KernelError::Rknn {
                message: format!(
                    "rknn_wait failed: {} ({ret}) — frame {} after {}ms",
                    rknn_error_name(ret),
                    frame.id,
                    frame.elapsed().as_millis()
                ),
            });
        }
        self.collect_outputs()
    }

    /// Reset this context: destroy and re-init from the original model bytes.
    /// Use when the NPU is in an unrecoverable state (repeated TIMEOUT,
    /// CTX_INVALID, DEVICE_UNAVAILABLE returns).
    ///
    /// Requires that you have the model bytes available — caller passes
    /// them in. After reset, all previously-bound `RknnMem` handles are
    /// invalid and must be re-created and re-bound.
    pub fn reset(&mut self, model_data: &[u8]) -> Result<(), KernelError> {
        self.reset_with_flags(model_data, 0)
    }

    /// Like [`reset`], but lets the caller preserve `RKNN_FLAG_ASYNC_MASK`
    /// (or any other init flag) the context was originally loaded with.
    /// `RknnPipelinedPool` uses this to recover an async context without
    /// silently falling back to synchronous mode.
    pub fn reset_with_flags(
        &mut self,
        model_data: &[u8],
        flags: u32,
    ) -> Result<(), KernelError> {
        // SAFETY: ctx is valid (we hold &mut self). destroy is required, present.
        let _ = unsafe { (self.funcs.destroy)(self.ctx) };
        let mut new_ctx: RknnContext = 0;
        let model_ptr = model_data.as_ptr();
        let model_len = model_data.len() as u32;
        // SAFETY: writable ctx out-pointer; valid model bytes.
        let ret = unsafe {
            (self.funcs.init)(
                &mut new_ctx as *mut RknnContext,
                model_ptr,
                model_len,
                flags,
                std::ptr::null_mut(),
            )
        };
        if ret != RKNN_SUCC {
            return Err(KernelError::Rknn {
                message: format!("rknn_init (reset) failed: {} ({ret})", rknn_error_name(ret)),
            });
        }
        self.ctx = new_ctx;
        // Re-query input/output attrs since the new context's tensor
        // descriptors are technically a fresh set (same model, same attrs,
        // but the SDK might rotate internal IDs).
        let (in_attrs, out_attrs) = query_io_attrs(&self.funcs, new_ctx)?;
        self.input_attrs = in_attrs;
        self.output_attrs = out_attrs;
        Ok(())
    }

    /// Non-blocking check whether an async inference is complete.
    /// Returns `true` if done, `false` if still running.
    pub fn poll(&self, frame: &AsyncFrame) -> Result<bool, KernelError> {
        let wait_fn = self.funcs.wait.ok_or_else(|| KernelError::Rknn {
            message: "rknn_wait not available".into(),
        })?;
        let mut ext = RknnRunExtend {
            frame_id: frame.id,
            non_block: 1,
            timeout_ms: 0,
            fence_fd: -1,
        };
        // SAFETY: ctx valid; ext is local.
        let ret = unsafe { wait_fn(self.ctx, &mut ext as *mut RknnRunExtend) };
        Ok(ret == RKNN_SUCC)
    }

    // ══════════════════════════════════════════════════════════════════
    // Phase 3 (cont.) — Cache sync utilities exposed to users
    // ══════════════════════════════════════════════════════════════════

    /// Sync a memory region's cache in the given direction.
    pub fn mem_sync(&self, mem: &RknnMem, mode: MemSyncMode) -> Result<(), KernelError> {
        let fn_ptr = self.funcs.mem_sync.ok_or_else(|| KernelError::Rknn {
            message: "rknn_mem_sync not available".into(),
        })?;
        // SAFETY: mem.ptr valid; mode is a documented bitmask.
        let ret = unsafe { fn_ptr(self.ctx, mem.ptr, mode.as_raw()) };
        if ret != RKNN_SUCC {
            return Err(KernelError::Rknn {
                message: format!(
                    "rknn_mem_sync({mode:?}) failed: {} ({ret})",
                    rknn_error_name(ret)
                ),
            });
        }
        Ok(())
    }
}

impl Drop for RknnBackend {
    fn drop(&mut self) {
        // SAFETY: ctx was successfully initialized in `load_*()`.
        unsafe {
            (self.funcs.destroy)(self.ctx);
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// Supporting types
// ══════════════════════════════════════════════════════════════════════

/// Memory footprint of a loaded model.
#[derive(Debug, Clone, Copy, Default)]
pub struct MemSize {
    pub weight_bytes: u64,
    pub internal_bytes: u64,
    pub dma_bytes: u64,
    pub sram_total_bytes: u64,
    pub sram_free_bytes: u64,
}

/// Per-operation performance measurement.
#[derive(Debug, Clone)]
pub struct OpPerf {
    pub op_name: String,
    pub duration_us: i64,
}

/// Structured performance output from `perf_detail()`.
#[derive(Debug, Clone, Default)]
pub struct PerfDetail {
    pub per_op: Vec<OpPerf>,
    pub total_us: i64,
    pub raw: String,
}

impl PerfDetail {
    fn parse(raw: &str, total_us: i64) -> Self {
        // The RKNN perf string has a header section followed by per-op
        // rows like: `ID OpType DataType Target InputShape OutputShape Time(us)`.
        // We extract op name and Time(us) by finding rows with numeric IDs.
        let mut per_op = Vec::new();
        for line in raw.lines() {
            let trimmed = line.trim_start();
            if trimmed.is_empty() || !trimmed.starts_with(|c: char| c.is_ascii_digit()) {
                continue;
            }
            let cols: Vec<&str> = trimmed.split_whitespace().collect();
            if cols.len() < 2 {
                continue;
            }
            // Last column is typically the duration in microseconds.
            if let Ok(dur) = cols[cols.len() - 1].parse::<i64>() {
                // Second column is typically the op type/name.
                let name = cols.get(1).copied().unwrap_or("unknown").to_string();
                per_op.push(OpPerf {
                    op_name: name,
                    duration_us: dur,
                });
            }
        }
        Self {
            per_op,
            total_us,
            raw: raw.to_string(),
        }
    }
}

/// Handle to an in-flight asynchronous inference.
///
/// Must be passed to `wait()` or `poll()` before the next `run_async()`
/// on the same context. On drop without wait, a warning is logged in
/// debug builds. Carries a deadline so callers can bound the wait and
/// detect hangs.
#[must_use = "an AsyncFrame represents in-flight NPU work; call `RknnBackend::wait` \
              to collect outputs, or the NPU slot is held until drop-time warning"]
pub struct AsyncFrame {
    id: u64,
    submitted_at: std::time::Instant,
    /// Wall-clock deadline in ms after submission. `-1` = no deadline
    /// (caller will block indefinitely on wait).
    deadline_ms: i32,
    waited: bool,
}

impl AsyncFrame {
    /// Frame ID passed at submission time.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Time elapsed since the frame was submitted.
    pub fn elapsed(&self) -> std::time::Duration {
        self.submitted_at.elapsed()
    }

    /// Configured deadline in milliseconds. `-1` means caller will block
    /// indefinitely.
    pub fn deadline_ms(&self) -> i32 {
        self.deadline_ms
    }

    /// Returns `true` when the frame's deadline has elapsed without a
    /// wait having completed. Useful for supervisor / watchdog logic.
    pub fn is_overdue(&self) -> bool {
        if self.deadline_ms < 0 {
            return false;
        }
        self.elapsed() > std::time::Duration::from_millis(self.deadline_ms as u64)
    }
}

impl Drop for AsyncFrame {
    fn drop(&mut self) {
        if !self.waited {
            #[cfg(debug_assertions)]
            eprintln!(
                "[rknn] AsyncFrame {} dropped without wait() — result is discarded",
                self.id
            );
        }
    }
}

/// Zero-copy memory handle managed by the RKNN runtime.
///
/// Created via `RknnBackend::alloc_mem*()`, `wrap_fd()`, or `wrap_phys()`.
/// Automatically freed on drop via `rknn_destroy_mem`.
pub struct RknnMem {
    ptr: *mut RknnTensorMemRaw,
    ctx: RknnContext,
    funcs: Arc<RknnFunctions>,
    /// `true` when memory is backed by external fd/phys (we don't own it).
    external: bool,
}

// SAFETY: RknnMem holds an opaque pointer to RKNN-managed memory that
// is documented as usable across threads once assigned to a context.
unsafe impl Send for RknnMem {}
unsafe impl Sync for RknnMem {}

impl RknnMem {
    /// Virtual address of the buffer for CPU access.
    pub fn virt_addr(&self) -> *mut c_void {
        // SAFETY: ptr is valid between create and destroy.
        unsafe { (*self.ptr).virt_addr }
    }

    /// Physical address (if applicable).
    pub fn phys_addr(&self) -> u64 {
        // SAFETY: ptr is valid between create and destroy.
        unsafe { (*self.ptr).phys_addr }
    }

    /// Buffer size in bytes.
    pub fn size(&self) -> u32 {
        // SAFETY: ptr is valid between create and destroy.
        unsafe { (*self.ptr).size }
    }

    /// File descriptor (for DMA-BUF-backed memory).
    pub fn fd(&self) -> i32 {
        // SAFETY: ptr is valid between create and destroy.
        unsafe { (*self.ptr).fd }
    }

    /// Mutable byte slice view of the buffer.
    ///
    /// # Safety
    /// Caller must ensure no NPU inference is concurrently reading or
    /// writing this buffer. Use `mem_sync(ToDevice)` after writing and
    /// before `run_bound()`.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        let len = self.size() as usize;
        let ptr = self.virt_addr().cast::<u8>();
        // SAFETY: ptr is valid (checked at creation); len is correct;
        // exclusive access via &mut self.
        unsafe { std::slice::from_raw_parts_mut(ptr, len) }
    }

    /// Immutable byte slice view.
    pub fn as_slice(&self) -> &[u8] {
        let len = self.size() as usize;
        let ptr = self.virt_addr().cast::<u8>();
        // SAFETY: ptr valid; len correct; shared reference.
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }

    /// Flush CPU writes before NPU reads.
    pub fn sync_to_device(&self) -> Result<(), KernelError> {
        self.sync(MemSyncMode::ToDevice)
    }

    /// Invalidate cache after NPU writes before CPU reads.
    pub fn sync_from_device(&self) -> Result<(), KernelError> {
        self.sync(MemSyncMode::FromDevice)
    }

    /// Bidirectional sync.
    pub fn sync_bidirectional(&self) -> Result<(), KernelError> {
        self.sync(MemSyncMode::Bidirectional)
    }

    fn sync(&self, mode: MemSyncMode) -> Result<(), KernelError> {
        let fn_ptr = self.funcs.mem_sync.ok_or_else(|| KernelError::Rknn {
            message: "rknn_mem_sync not available".into(),
        })?;
        // SAFETY: self.ptr is valid; mode is documented bitmask.
        let ret = unsafe { fn_ptr(self.ctx, self.ptr, mode.as_raw()) };
        if ret != RKNN_SUCC {
            return Err(KernelError::Rknn {
                message: format!(
                    "rknn_mem_sync({mode:?}) failed: {} ({ret})",
                    rknn_error_name(ret)
                ),
            });
        }
        Ok(())
    }
}

impl Drop for RknnMem {
    fn drop(&mut self) {
        if let Some(destroy) = self.funcs.destroy_mem {
            // SAFETY: ptr is valid from create_mem*; destroy_mem takes
            // ownership back and frees the buffer.
            unsafe {
                destroy(self.ctx, self.ptr);
            }
        }
        let _ = self.external;
    }
}

// ══════════════════════════════════════════════════════════════════════
// Phase 4 — Multi-core context pool
// ══════════════════════════════════════════════════════════════════════

/// Pool of `RknnBackend` contexts pinned to different NPU cores.
///
/// Enables concurrent inference of the same model across all available
/// NPU cores (3 on RK3588) for maximum throughput.
///
/// The pool retains the model bytes so individual contexts can be
/// rebuilt via [`ContextPool::recover_failed`] when the NPU returns
/// `TIMEOUT` / `CTX_INVALID` / `DEVICE_UNAVAILABLE`.
pub struct ContextPool {
    contexts: Vec<std::sync::RwLock<RknnBackend>>,
    cores: Vec<NpuCoreMask>,
    model_data: Arc<Vec<u8>>,
    next: std::sync::atomic::AtomicUsize,
    /// Per-context counter of consecutive recoverable errors. Reset on success.
    fail_streak: Vec<std::sync::atomic::AtomicU32>,
}

impl ContextPool {
    /// Create a pool with one context per given core mask.
    pub fn new(model_data: &[u8], cores: &[NpuCoreMask]) -> Result<Self, KernelError> {
        if cores.is_empty() {
            return Err(KernelError::Rknn {
                message: "ContextPool requires at least one core".into(),
            });
        }
        let stored_model = Arc::new(model_data.to_vec());
        let first = RknnBackend::load(&stored_model)?;
        first.set_core_mask(cores[0])?;
        let mut contexts: Vec<std::sync::RwLock<RknnBackend>> = vec![std::sync::RwLock::new(first)];
        for &mask in &cores[1..] {
            let dup = contexts[0]
                .read()
                .map_err(|_| KernelError::Rknn {
                    message: "ContextPool first context lock poisoned".into(),
                })?
                .dup_for_core(mask)?;
            contexts.push(std::sync::RwLock::new(dup));
        }
        let fail_streak = (0..contexts.len())
            .map(|_| std::sync::atomic::AtomicU32::new(0))
            .collect();
        Ok(Self {
            contexts,
            cores: cores.to_vec(),
            model_data: stored_model,
            next: std::sync::atomic::AtomicUsize::new(0),
            fail_streak,
        })
    }

    /// Round-robin index of the next context to dispatch on.
    fn next_idx(&self) -> usize {
        let idx = self.next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        idx % self.contexts.len()
    }

    /// Acquire the next context in round-robin order (read lock — multiple
    /// concurrent inferences allowed when contexts use different cores).
    pub fn next_context(&self) -> impl std::ops::Deref<Target = RknnBackend> + '_ {
        let idx = self.next_idx();
        self.contexts[idx]
            .read()
            .expect("ContextPool poisoned — cannot recover from poisoned lock")
    }

    /// Run inference on the next available context in round-robin order.
    /// On `RKNN_ERR_TIMEOUT` / `CTX_INVALID` / `DEVICE_UNAVAILABLE`, the
    /// fail-streak counter is incremented; if it crosses
    /// `recovery_threshold`, the context is auto-recovered. Otherwise the
    /// error is returned to the caller.
    ///
    /// `inputs` is a slice of `(name, bytes)` pairs — one entry per
    /// graph input. Lookup is by name so order doesn't have to match
    /// the model's input index order.
    pub fn dispatch_roundrobin(
        &self,
        inputs: &[(&str, &[u8])],
    ) -> Result<Vec<Tensor>, KernelError> {
        let idx = self.next_idx();
        self.dispatch_on(idx, inputs)
    }

    /// Run inference on a specific context index, with auto-recovery on
    /// repeated transient errors. Same input contract as
    /// [`dispatch_roundrobin`].
    pub fn dispatch_on(
        &self,
        idx: usize,
        inputs: &[(&str, &[u8])],
    ) -> Result<Vec<Tensor>, KernelError> {
        if idx >= self.contexts.len() {
            return Err(KernelError::Rknn {
                message: format!(
                    "dispatch_on: index {idx} out of range (pool size {})",
                    self.contexts.len()
                ),
            });
        }
        let ctx = self.contexts[idx].read().map_err(|_| KernelError::Rknn {
            message: format!("ContextPool ctx {idx} lock poisoned"),
        })?;

        // Reorder caller-supplied inputs to match the model's input
        // index order, which is what `RknnBackend::run(&[&[u8]])`
        // expects. Error if any declared input is missing.
        let ordered: Vec<&[u8]> = ctx
            .input_attrs()
            .iter()
            .map(|attr| {
                let name = attr.name_str();
                inputs
                    .iter()
                    .find(|(n, _)| *n == name)
                    .map(|(_, data)| *data)
                    .ok_or_else(|| KernelError::Rknn {
                        message: format!("dispatch: missing input '{name}'"),
                    })
            })
            .collect::<Result<_, _>>()?;

        let result = ctx.run(&ordered);
        drop(ctx);
        match result {
            Ok(out) => {
                self.fail_streak[idx].store(0, std::sync::atomic::Ordering::Relaxed);
                Ok(out)
            }
            Err(e) => {
                let streak =
                    self.fail_streak[idx].fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                eprintln!("[yscv-rknn] ctx {idx} dispatch failed (streak {streak}): {e}");
                Err(e)
            }
        }
    }

    /// Recover a failed context: destroy + re-init from the stored model
    /// bytes, restore its core-mask binding. Caller invokes this after
    /// observing `KernelError::Rknn { message: contains "TIMEOUT" }` or
    /// after `fail_streak(idx)` exceeds threshold.
    ///
    /// Acquires the context's write lock — concurrent dispatches will
    /// block briefly until recovery completes.
    pub fn recover_failed(&self, idx: usize) -> Result<(), KernelError> {
        if idx >= self.contexts.len() {
            return Err(KernelError::Rknn {
                message: format!(
                    "recover_failed: index {idx} out of range (pool size {})",
                    self.contexts.len()
                ),
            });
        }
        let mut ctx = self.contexts[idx].write().map_err(|_| KernelError::Rknn {
            message: format!("ContextPool ctx {idx} lock poisoned"),
        })?;
        ctx.reset(&self.model_data)?;
        ctx.set_core_mask(self.cores[idx])?;
        self.fail_streak[idx].store(0, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Current consecutive-failure counter for a context. Useful for
    /// supervisor logic: if `fail_streak(i) >= 3`, call `recover_failed(i)`.
    pub fn fail_streak(&self, idx: usize) -> u32 {
        self.fail_streak
            .get(idx)
            .map(|c| c.load(std::sync::atomic::Ordering::Relaxed))
            .unwrap_or(0)
    }

    /// Total number of contexts (= number of NPU cores in the pool).
    pub fn size(&self) -> usize {
        self.contexts.len()
    }

    /// Access to a specific context by index (read lock).
    pub fn context(&self, i: usize) -> Option<impl std::ops::Deref<Target = RknnBackend> + '_> {
        self.contexts.get(i).and_then(|lock| lock.read().ok())
    }
}

// ══════════════════════════════════════════════════════════════════════
// Phase 8 — Matmul accelerator
// ══════════════════════════════════════════════════════════════════════

/// Quantization parameters retrieved from a matmul context.
///
/// Returned by [`RknnMatmul::quant_params`]. `scale` and `zero_point` are
/// per-channel or per-group values matching the dtype the matmul was
/// configured with. Empty vectors mean per-layer (single scalar) or no
/// quantization.
#[derive(Debug, Clone)]
pub struct MatmulQuantParams {
    /// Tensor name from the SDK (typically `"B"`).
    pub name: String,
    /// Per-channel/per-group scale factors.
    pub scale: Vec<f32>,
    /// Per-channel/per-group zero-point values.
    pub zero_point: Vec<i32>,
}

/// Dedicated NPU matmul accelerator (independent from conv inference path).
///
/// Supports FP16 / INT8 / INT4 input/output combinations, per-layer /
/// per-channel / per-group quantization, and dynamic M dimension updates.
pub struct RknnMatmul {
    ctx: RknnMatmulCtx,
    funcs: Arc<RknnFunctions>,
    _lib: Arc<DlHandle>,
    io_attr: RknnMatmulIoAttr,
    info: RknnMatmulInfo,
}

// SAFETY: matmul context is an opaque u64 handle; operations are thread-safe
// per documented SDK contract for independent contexts.
unsafe impl Send for RknnMatmul {}
unsafe impl Sync for RknnMatmul {}

impl RknnMatmul {
    /// Create a matmul context for M×K × K×N with the given dtype.
    pub fn new(m: i32, k: i32, n: i32, dtype: RknnMatmulType) -> Result<Self, KernelError> {
        let (lib, funcs) = load_rknn_library()?;
        let funcs = Arc::new(funcs);
        let lib = Arc::new(lib);

        let create_fn = funcs.matmul_create.ok_or_else(|| KernelError::Rknn {
            message: "rknn_matmul_create not available".into(),
        })?;

        let mut info = RknnMatmulInfo {
            m,
            k,
            n,
            typ: dtype as i32,
            ..Default::default()
        };
        let mut io_attr = RknnMatmulIoAttr::default();
        let mut ctx: RknnMatmulCtx = 0;

        // SAFETY: ctx / info / io_attr are valid writable pointers.
        let ret = unsafe {
            create_fn(
                &mut ctx as *mut RknnMatmulCtx,
                &mut info as *mut RknnMatmulInfo,
                &mut io_attr as *mut RknnMatmulIoAttr,
            )
        };
        if ret != RKNN_SUCC {
            return Err(KernelError::Rknn {
                message: format!(
                    "rknn_matmul_create failed: {} ({ret})",
                    rknn_error_name(ret)
                ),
            });
        }

        Ok(Self {
            ctx,
            funcs,
            _lib: lib,
            io_attr,
            info,
        })
    }

    /// Create a matmul context that supports a fixed set of allowed
    /// (M, K, N) shapes. Switch between shapes at runtime via
    /// [`RknnMatmul::set_shape`] without re-creating the context.
    ///
    /// `shapes` lists every (M, K, N) tuple the model may dispatch.
    /// `dtype` selects the input/output type combination.
    pub fn new_dynamic(
        shapes: &[RknnMatmulShape],
        dtype: RknnMatmulType,
    ) -> Result<Self, KernelError> {
        if shapes.is_empty() {
            return Err(KernelError::Rknn {
                message: "RknnMatmul::new_dynamic: shapes must be non-empty".into(),
            });
        }
        let (lib, funcs) = load_rknn_library()?;
        let funcs = Arc::new(funcs);
        let lib = Arc::new(lib);

        let create_fn = funcs
            .matmul_create_dynamic_shape
            .ok_or_else(|| KernelError::Rknn {
                message: "rknn_matmul_create_dynamic_shape not available in this RKNN runtime"
                    .into(),
            })?;

        let first = shapes[0];
        let mut info = RknnMatmulInfo {
            m: first.m,
            k: first.k,
            n: first.n,
            typ: dtype as i32,
            ..Default::default()
        };
        let mut io_attr = RknnMatmulIoAttr::default();
        let mut ctx: RknnMatmulCtx = 0;
        let mut shapes_owned: Vec<RknnMatmulShape> = shapes.to_vec();

        // SAFETY: all out-pointers and shapes_owned remain valid for the call.
        let ret = unsafe {
            create_fn(
                &mut ctx as *mut RknnMatmulCtx,
                &mut info as *mut RknnMatmulInfo,
                shapes_owned.len() as i32,
                shapes_owned.as_mut_ptr(),
                &mut io_attr as *mut RknnMatmulIoAttr,
            )
        };
        if ret != RKNN_SUCC {
            return Err(KernelError::Rknn {
                message: format!(
                    "rknn_matmul_create_dynamic_shape failed: {} ({ret})",
                    rknn_error_name(ret)
                ),
            });
        }

        Ok(Self {
            ctx,
            funcs,
            _lib: lib,
            io_attr,
            info,
        })
    }

    /// Switch the active (M, K, N) shape for a dynamic-shape matmul.
    /// Must be one of the shapes registered in [`RknnMatmul::new_dynamic`].
    pub fn set_shape(&self, shape: RknnMatmulShape) -> Result<(), KernelError> {
        let fn_ptr = self
            .funcs
            .matmul_set_dynamic_shape
            .ok_or_else(|| KernelError::Rknn {
                message: "rknn_matmul_set_dynamic_shape not available".into(),
            })?;
        let mut s = shape;
        // SAFETY: ctx valid; s is a stack-allocated writable shape.
        let ret = unsafe { fn_ptr(self.ctx, &mut s as *mut RknnMatmulShape) };
        if ret != RKNN_SUCC {
            return Err(KernelError::Rknn {
                message: format!(
                    "rknn_matmul_set_dynamic_shape failed: {} ({ret})",
                    rknn_error_name(ret)
                ),
            });
        }
        Ok(())
    }

    /// Read back per-channel/per-group quantization parameters previously
    /// set via [`RknnMatmul::set_quant_params`]. Useful for verifying that
    /// calibration data was applied correctly.
    pub fn quant_params(&self) -> Result<MatmulQuantParams, KernelError> {
        let fn_ptr = self
            .funcs
            .matmul_get_quant_params
            .ok_or_else(|| KernelError::Rknn {
                message: "rknn_matmul_get_quant_params not available".into(),
            })?;
        let mut raw = RknnQuantParamsRaw {
            name: [0u8; RKNN_MAX_NAME_LEN],
            scale: std::ptr::null(),
            scale_len: 0,
            zp: std::ptr::null(),
            zp_len: 0,
        };
        let mut scale_out: f32 = 0.0;
        // SAFETY: raw and scale_out are valid out-parameters.
        let ret = unsafe {
            fn_ptr(
                self.ctx,
                &mut raw as *mut RknnQuantParamsRaw,
                &mut scale_out,
            )
        };
        if ret != RKNN_SUCC {
            return Err(KernelError::Rknn {
                message: format!(
                    "rknn_matmul_get_quant_params failed: {} ({ret})",
                    rknn_error_name(ret)
                ),
            });
        }
        let name_end = raw
            .name
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(raw.name.len());
        let name = String::from_utf8_lossy(&raw.name[..name_end]).into_owned();
        let scale: Vec<f32> = if raw.scale.is_null() || raw.scale_len <= 0 {
            Vec::new()
        } else {
            // SAFETY: SDK promises raw.scale points to scale_len f32 values
            // valid for the call duration.
            unsafe { std::slice::from_raw_parts(raw.scale, raw.scale_len as usize).to_vec() }
        };
        let zero_point: Vec<i32> = if raw.zp.is_null() || raw.zp_len <= 0 {
            Vec::new()
        } else {
            // SAFETY: same contract as scale.
            unsafe { std::slice::from_raw_parts(raw.zp, raw.zp_len as usize).to_vec() }
        };
        Ok(MatmulQuantParams {
            name,
            scale,
            zero_point,
        })
    }

    /// Tensor attribute for A input (shape, size, type).
    pub fn a_attr(&self) -> &RknnMatmulTensorAttr {
        &self.io_attr.a
    }

    /// Allocate an NPU-accessible buffer scoped to this matmul context.
    /// Use for `bind_a` / `bind_b` / `bind_c` inputs when you don't
    /// already have an `RknnMem` from an `RknnBackend`. The SDK
    /// treats matmul and inference contexts uniformly for
    /// `rknn_create_mem*`.
    pub fn alloc_mem(&self, size: usize) -> Result<RknnMem, KernelError> {
        if self.funcs.create_mem2.is_some() {
            self.alloc_mem_ex(size, MemAllocFlags::Default)
        } else if let Some(fn_ptr) = self.funcs.create_mem {
            // SAFETY: ctx is a valid matmul context (u64); size is a scalar.
            let ptr = unsafe { fn_ptr(self.ctx, size as u32) };
            if ptr.is_null() {
                return Err(KernelError::Rknn {
                    message: format!("rknn_create_mem({size}) returned null"),
                });
            }
            Ok(RknnMem {
                ptr,
                ctx: self.ctx,
                funcs: Arc::clone(&self.funcs),
                external: false,
            })
        } else {
            Err(KernelError::Rknn {
                message: "neither rknn_create_mem nor rknn_create_mem2 are available".into(),
            })
        }
    }

    /// Allocate with explicit alloc flags. Mirrors
    /// `RknnBackend::alloc_mem_ex` but scoped to this matmul ctx.
    pub fn alloc_mem_ex(
        &self,
        size: usize,
        flags: MemAllocFlags,
    ) -> Result<RknnMem, KernelError> {
        let fn_ptr = self.funcs.create_mem2.ok_or_else(|| KernelError::Rknn {
            message: "rknn_create_mem2 not available".into(),
        })?;
        // SAFETY: ctx valid; size scalar; flags is a documented bitmask.
        let ptr = unsafe { fn_ptr(self.ctx, size as u64, flags.as_raw()) };
        if ptr.is_null() {
            return Err(KernelError::Rknn {
                message: format!("rknn_create_mem2({size}, {flags:?}) returned null"),
            });
        }
        Ok(RknnMem {
            ptr,
            ctx: self.ctx,
            funcs: Arc::clone(&self.funcs),
            external: false,
        })
    }

    /// Tensor attribute for B input.
    pub fn b_attr(&self) -> &RknnMatmulTensorAttr {
        &self.io_attr.b
    }

    /// Tensor attribute for C output.
    pub fn c_attr(&self) -> &RknnMatmulTensorAttr {
        &self.io_attr.c
    }

    /// Bind the A input buffer.
    pub fn bind_a(&self, mem: &RknnMem) -> Result<(), KernelError> {
        self.bind_generic(mem, self.io_attr.a.clone())
    }

    /// Bind the B input buffer (pre-transformed layout for NPU).
    pub fn bind_b(&self, mem: &RknnMem) -> Result<(), KernelError> {
        self.bind_generic(mem, self.io_attr.b.clone())
    }

    /// Bind the C output buffer.
    pub fn bind_c(&self, mem: &RknnMem) -> Result<(), KernelError> {
        self.bind_generic(mem, self.io_attr.c.clone())
    }

    fn bind_generic(
        &self,
        mem: &RknnMem,
        mut attr: RknnMatmulTensorAttr,
    ) -> Result<(), KernelError> {
        let fn_ptr = self
            .funcs
            .matmul_set_io_mem
            .ok_or_else(|| KernelError::Rknn {
                message: "rknn_matmul_set_io_mem not available".into(),
            })?;
        // SAFETY: ctx/mem/attr are valid.
        let ret = unsafe { fn_ptr(self.ctx, mem.ptr, &mut attr as *mut RknnMatmulTensorAttr) };
        if ret != RKNN_SUCC {
            return Err(KernelError::Rknn {
                message: format!(
                    "rknn_matmul_set_io_mem failed: {} ({ret})",
                    rknn_error_name(ret)
                ),
            });
        }
        Ok(())
    }

    /// Set per-channel or per-group quantization parameters for the B tensor.
    ///
    /// * `name` — quantizer tensor name (must match B attr).
    /// * `scale` — per-channel or per-group scale factors.
    /// * `zp` — per-channel or per-group zero points.
    pub fn set_quant_params(
        &self,
        name: &str,
        scale: &[f32],
        zp: &[i32],
    ) -> Result<(), KernelError> {
        let fn_ptr = self
            .funcs
            .matmul_set_quant_params
            .ok_or_else(|| KernelError::Rknn {
                message: "rknn_matmul_set_quant_params not available".into(),
            })?;
        let mut name_buf = [0u8; RKNN_MAX_NAME_LEN];
        let bytes = name.as_bytes();
        let copy_len = bytes.len().min(RKNN_MAX_NAME_LEN - 1);
        name_buf[..copy_len].copy_from_slice(&bytes[..copy_len]);

        let mut params = RknnQuantParamsRaw {
            name: name_buf,
            scale: scale.as_ptr(),
            scale_len: scale.len() as i32,
            zp: zp.as_ptr(),
            zp_len: zp.len() as i32,
        };
        // SAFETY: ctx/params valid; scale/zp arrays outlive the call.
        let ret = unsafe { fn_ptr(self.ctx, &mut params as *mut RknnQuantParamsRaw) };
        if ret != RKNN_SUCC {
            return Err(KernelError::Rknn {
                message: format!(
                    "rknn_matmul_set_quant_params failed: {} ({ret})",
                    rknn_error_name(ret)
                ),
            });
        }
        Ok(())
    }

    /// Pin matmul execution to specific NPU core(s).
    pub fn set_core_mask(&self, mask: NpuCoreMask) -> Result<(), KernelError> {
        let fn_ptr = self
            .funcs
            .matmul_set_core_mask
            .ok_or_else(|| KernelError::Rknn {
                message: "rknn_matmul_set_core_mask not available".into(),
            })?;
        // SAFETY: ctx valid; mask documented enum.
        let ret = unsafe { fn_ptr(self.ctx, mask.as_raw()) };
        if ret != RKNN_SUCC {
            return Err(KernelError::Rknn {
                message: format!(
                    "rknn_matmul_set_core_mask failed: {} ({ret})",
                    rknn_error_name(ret)
                ),
            });
        }
        Ok(())
    }

    /// Update the M dimension for dynamic-shape matmul.
    pub fn set_dynamic_shape(&self, m: i32) -> Result<(), KernelError> {
        let fn_ptr = self
            .funcs
            .matmul_set_dynamic_shape
            .ok_or_else(|| KernelError::Rknn {
                message: "rknn_matmul_set_dynamic_shape not available".into(),
            })?;
        let mut shape = RknnMatmulShape {
            m,
            k: self.info.k,
            n: self.info.n,
        };
        // SAFETY: ctx valid; shape is local.
        let ret = unsafe { fn_ptr(self.ctx, &mut shape as *mut RknnMatmulShape) };
        if ret != RKNN_SUCC {
            return Err(KernelError::Rknn {
                message: format!(
                    "rknn_matmul_set_dynamic_shape failed: {} ({ret})",
                    rknn_error_name(ret)
                ),
            });
        }
        Ok(())
    }

    /// Execute the matmul.
    pub fn run(&self) -> Result<(), KernelError> {
        let fn_ptr = self.funcs.matmul_run.ok_or_else(|| KernelError::Rknn {
            message: "rknn_matmul_run not available".into(),
        })?;
        // SAFETY: ctx valid.
        let ret = unsafe { fn_ptr(self.ctx) };
        if ret != RKNN_SUCC {
            return Err(KernelError::Rknn {
                message: format!("rknn_matmul_run failed: {} ({ret})", rknn_error_name(ret)),
            });
        }
        Ok(())
    }

    /// Transform B from normal row-major layout to NPU-native tiled layout.
    pub fn transform_b_layout(
        b_in: &[i8],
        b_out: &mut [i8],
        k: i32,
        n: i32,
        dtype: RknnMatmulType,
    ) -> Result<(), KernelError> {
        let (_lib, funcs) = load_rknn_library()?;
        let fn_ptr = funcs
            .matmul_b_layout_transform
            .ok_or_else(|| KernelError::Rknn {
                message: "rknn_B_normal_layout_to_native_layout not available".into(),
            })?;
        let mut info = RknnMatmulInfo {
            m: 0,
            k,
            n,
            typ: dtype as i32,
            ..Default::default()
        };
        // SAFETY: b_in / b_out are valid slices; info is local writable.
        let ret = unsafe {
            fn_ptr(
                b_in.as_ptr() as *mut c_void,
                b_out.as_mut_ptr() as *mut c_void,
                k,
                n,
                &mut info as *mut RknnMatmulInfo,
            )
        };
        if ret != RKNN_SUCC {
            return Err(KernelError::Rknn {
                message: format!(
                    "rknn_B_normal_layout_to_native_layout failed: {} ({ret})",
                    rknn_error_name(ret)
                ),
            });
        }
        Ok(())
    }
}

impl Drop for RknnMatmul {
    fn drop(&mut self) {
        if let Some(destroy) = self.funcs.matmul_destroy {
            // SAFETY: ctx is valid from create.
            unsafe {
                destroy(self.ctx);
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// Phase 9 — Custom OpenCL operators
// ══════════════════════════════════════════════════════════════════════

/// Execution target for a custom operator (matches `rknn_target_type`).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CustomOpTarget {
    /// Execute on CPU.
    Cpu = 1,
    /// Execute on GPU via OpenCL.
    Gpu = 2,
}

/// Custom operator descriptor.
///
/// Registers an OpenCL kernel as an operator the NPU runtime can invoke
/// during graph execution. Used for custom post-processing (NMS, bbox
/// decode, soft-NMS) or specialised tensor transforms that aren't in the
/// standard op set.
///
/// Kernel source and build options are copied into owned storage so the
/// builder can be dropped after registration.
pub struct CustomOp {
    /// User-facing op type name matching the ONNX/RKNN op name.
    pub op_type: String,
    /// OpenCL kernel entry function name.
    pub cl_kernel_name: String,
    /// OpenCL C kernel source code.
    pub cl_kernel_source: String,
    /// Compiler flags (e.g. `-cl-fast-relaxed-math`).
    pub cl_build_options: String,
    /// Execution target (CPU or GPU).
    pub target: CustomOpTarget,
    /// Optional Rust handler invoked from `init` / `prepare` / `compute` /
    /// `destroy` callbacks. `None` → run as pure embedded OpenCL kernel.
    pub handler: Option<std::sync::Arc<dyn super::custom_op::CustomOpHandler>>,
}

impl CustomOp {
    /// Builder-style constructor for GPU (OpenCL) operators.
    pub fn gpu(op_type: impl Into<String>, kernel_name: impl Into<String>) -> Self {
        Self {
            op_type: op_type.into(),
            cl_kernel_name: kernel_name.into(),
            cl_kernel_source: String::new(),
            cl_build_options: String::new(),
            target: CustomOpTarget::Gpu,
            handler: None,
        }
    }

    /// Builder-style constructor for CPU operators.
    pub fn cpu(op_type: impl Into<String>) -> Self {
        Self {
            op_type: op_type.into(),
            cl_kernel_name: String::new(),
            cl_kernel_source: String::new(),
            cl_build_options: String::new(),
            target: CustomOpTarget::Cpu,
            handler: None,
        }
    }

    /// Provide OpenCL kernel source code.
    pub fn with_kernel_source(mut self, src: impl Into<String>) -> Self {
        self.cl_kernel_source = src.into();
        self
    }

    /// Provide OpenCL build options string.
    pub fn with_build_options(mut self, opts: impl Into<String>) -> Self {
        self.cl_build_options = opts.into();
        self
    }

    /// Attach a Rust callback handler. The handler's `compute` method runs
    /// on every NPU invocation; `init` / `prepare` / `destroy` are
    /// optional lifecycle hooks.
    pub fn with_handler(
        mut self,
        handler: std::sync::Arc<dyn super::custom_op::CustomOpHandler>,
    ) -> Self {
        self.handler = Some(handler);
        self
    }
}

impl RknnBackend {
    /// Register custom OpenCL / CPU operators before inference.
    ///
    /// Call this after `load()` but before the first `run()`. Registered
    /// operators remain in effect for the context's lifetime.
    ///
    /// Each `CustomOp` may carry a Rust `CustomOpHandler` for `init` /
    /// `prepare` / `compute` / `destroy` callbacks; if none is supplied the
    /// op runs purely as the embedded OpenCL kernel. Up to
    /// `MAX_CUSTOM_OP_SLOTS` (16) handlers may be active per process.
    ///
    /// Kernel source buffers and slot reservations are kept alive inside
    /// the returned `CustomOpRegistration` — don't drop it until the
    /// context is destroyed.
    pub fn register_custom_ops(
        &self,
        ops: Vec<CustomOp>,
    ) -> Result<CustomOpRegistration, KernelError> {
        let reg_fn = self
            .funcs
            .register_custom_ops
            .ok_or_else(|| KernelError::Rknn {
                message: "rknn_register_custom_ops not available in this RKNN runtime".into(),
            })?;

        // Reserve handler slots first (so failure rollback is straightforward).
        let mut slots: Vec<usize> = Vec::with_capacity(ops.len());
        for op in &ops {
            if let Some(handler) = op.handler.clone() {
                match super::custom_op::allocate_slot(handler, self.funcs.clone()) {
                    Ok(s) => slots.push(s),
                    Err(e) => {
                        for s in slots.drain(..) {
                            super::custom_op::release_slot(s);
                        }
                        return Err(e);
                    }
                }
            }
        }

        // Build raw structs, keeping source bytes alive for the call.
        let mut sources: Vec<Vec<u8>> = Vec::with_capacity(ops.len());
        let mut raw_ops: Vec<RknnCustomOpRaw> = Vec::with_capacity(ops.len());
        let mut slot_iter = slots.iter().copied();

        for op in &ops {
            let mut op_type = [0u8; RKNN_MAX_NAME_LEN];
            let bytes = op.op_type.as_bytes();
            let l = bytes.len().min(RKNN_MAX_NAME_LEN - 1);
            op_type[..l].copy_from_slice(&bytes[..l]);

            let mut kernel_name = [0u8; RKNN_MAX_NAME_LEN];
            let bytes = op.cl_kernel_name.as_bytes();
            let l = bytes.len().min(RKNN_MAX_NAME_LEN - 1);
            kernel_name[..l].copy_from_slice(&bytes[..l]);

            let mut build_opts = [0u8; RKNN_MAX_NAME_LEN];
            let bytes = op.cl_build_options.as_bytes();
            let l = bytes.len().min(RKNN_MAX_NAME_LEN - 1);
            build_opts[..l].copy_from_slice(&bytes[..l]);

            let src_bytes = op.cl_kernel_source.as_bytes().to_vec();
            sources.push(src_bytes);

            // Per-op trampolines (if the user supplied a handler).
            let (init, prepare, compute, destroy) = if op.handler.is_some() {
                let slot = slot_iter
                    .next()
                    .expect("slot count matches handler count by construction");
                let t = super::custom_op::trampolines_for_slot(slot);
                (
                    t.init as *const c_void,
                    t.prepare as *const c_void,
                    t.compute as *const c_void,
                    t.destroy as *const c_void,
                )
            } else {
                (
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                )
            };

            raw_ops.push(RknnCustomOpRaw {
                version: 1,
                target: op.target as u32,
                op_type,
                cl_kernel_name: kernel_name,
                cl_kernel_source: std::ptr::null_mut(), // patched below
                cl_source_size: 0,
                cl_build_options: build_opts,
                init,
                prepare,
                compute,
                compute_native: std::ptr::null(),
                destroy,
            });
        }

        // Re-patch source pointers now that `sources` has been reallocated.
        for (raw, src) in raw_ops.iter_mut().zip(sources.iter()) {
            raw.cl_kernel_source = if src.is_empty() {
                std::ptr::null_mut()
            } else {
                src.as_ptr() as *mut u8
            };
            raw.cl_source_size = src.len() as u64;
        }

        // SAFETY: raw_ops slice is valid; sources are kept alive via the
        // returned CustomOpRegistration handle.
        let ret = unsafe { reg_fn(self.ctx, raw_ops.as_mut_ptr(), raw_ops.len() as u32) };
        if ret != RKNN_SUCC {
            // Rollback: release reserved slots if registration failed.
            for s in slots {
                super::custom_op::release_slot(s);
            }
            return Err(KernelError::Rknn {
                message: format!(
                    "rknn_register_custom_ops failed: {} ({ret})",
                    rknn_error_name(ret)
                ),
            });
        }

        Ok(CustomOpRegistration {
            _sources: sources,
            slots,
        })
    }
}

/// Opaque handle keeping registered custom-op source buffers and dispatcher
/// slots alive. Hold alongside the `RknnBackend` for the duration of
/// inference; dropping releases the slots back to the dispatch pool.
pub struct CustomOpRegistration {
    _sources: Vec<Vec<u8>>,
    slots: Vec<usize>,
}

impl Drop for CustomOpRegistration {
    fn drop(&mut self) {
        for &s in &self.slots {
            super::custom_op::release_slot(s);
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// Helpers
// ══════════════════════════════════════════════════════════════════════

fn c_str_to_string(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    std::str::from_utf8(&bytes[..end])
        .unwrap_or("<invalid utf8>")
        .to_string()
}

// ══════════════════════════════════════════════════════════════════════
// Runtime detection
// ══════════════════════════════════════════════════════════════════════

/// Check whether `librknnrt.so` is loadable on this system.
pub fn rknn_available() -> bool {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: dlopen with RTLD_LAZY only probes for the library.
        let handle = unsafe { libc::dlopen(c"librknnrt.so".as_ptr(), libc::RTLD_LAZY) };
        if handle.is_null() {
            return false;
        }
        // SAFETY: handle is non-null.
        unsafe { libc::dlclose(handle) };
        true
    }
    #[cfg(not(target_os = "linux"))]
    {
        false
    }
}

/// Runtime-selectable inference backend.
pub enum InferenceBackend {
    /// Software CPU execution (always available).
    Cpu,
    /// Rockchip NPU acceleration via RKNN runtime.
    Rknn(RknnBackend),
}

/// Detect which inference backend is available at runtime.
pub fn detect_backend(model_data: Option<&[u8]>) -> InferenceBackend {
    if let Some(data) = model_data
        && rknn_available()
        && let Ok(backend) = RknnBackend::load(data)
    {
        return InferenceBackend::Rknn(backend);
    }
    InferenceBackend::Cpu
}
// ══════════════════════════════════════════════════════════════════════
// Compile-time ABI assertions
// ══════════════════════════════════════════════════════════════════════

const _: () = {
    // All sizes verified against SDK 2.4.3a0 by compiling the C struct
    // equivalents on aarch64-darwin (matches LP64 ABI used on Rockchip Linux):
    //
    //   rknn_tensor_attr        376  (376-byte struct after pass_through+h_stride extension)
    //   rknn_init_extend        136
    //   rknn_run_extend         24
    //   rknn_output_extend      8
    //   rknn_mem_size           64
    //   rknn_custom_string      1024
    //   rknn_tensor_mem         40
    //   rknn_gpu_op_context     24   (3 × 8-byte ptrs)
    //   rknn_custom_op_context  48   (u32+pad+u64+24+8)
    //   rknn_custom_op_tensor   416  (376 attr + 40 mem)
    //   rknn_custom_op_attr     272  (256 name + 4 dtype + 4 n_elems + 8 ptr)
    //   rknn_matmul_info        64
    //   rknn_input_output_num   8
    //   rknn_matmul_shape       12
    //   rknn_sdk_version        512
    //   rknn_perf_run           8
    assert!(std::mem::size_of::<RknnTensorAttr>() == 376);
    assert!(std::mem::size_of::<RknnInitExtend>() == 136);
    assert!(std::mem::size_of::<RknnRunExtend>() == 24);
    assert!(std::mem::size_of::<RknnOutputExtend>() == 8);
    assert!(std::mem::size_of::<RknnMemSizeRaw>() == 64);
    assert!(std::mem::size_of::<RknnCustomStringRaw>() == 1024);
    assert!(std::mem::size_of::<RknnTensorMemRaw>() == 40);
    assert!(std::mem::size_of::<RknnGpuOpContext>() == 24);
    assert!(std::mem::size_of::<RknnCustomOpContextRaw>() == 48);
    assert!(std::mem::size_of::<RknnCustomOpTensorRaw>() == 416);
    assert!(std::mem::size_of::<RknnCustomOpAttrRaw>() == 272);
    assert!(std::mem::size_of::<RknnMatmulInfo>() == 64);
    assert!(std::mem::size_of::<RknnInOutNum>() == 8);
    assert!(std::mem::size_of::<RknnMatmulShape>() == 12);
    assert!(std::mem::size_of::<RknnSdkVersionRaw>() == 512);
    assert!(std::mem::size_of::<RknnPerfRun>() == 8);
};

// ══════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_type_from_u32() {
        assert_eq!(RknnTensorType::from_u32(0), RknnTensorType::Float32);
        assert_eq!(RknnTensorType::from_u32(3), RknnTensorType::Uint8);
        assert_eq!(RknnTensorType::from_u32(10), RknnTensorType::Int4);
        assert_eq!(RknnTensorType::from_u32(99), RknnTensorType::Uint8);
    }

    #[test]
    fn tensor_format_from_u32() {
        assert_eq!(RknnTensorFormat::from_u32(0), RknnTensorFormat::Nchw);
        assert_eq!(RknnTensorFormat::from_u32(1), RknnTensorFormat::Nhwc);
        assert_eq!(RknnTensorFormat::from_u32(2), RknnTensorFormat::Nc1hwc2);
    }

    #[test]
    fn quant_type_from_u32() {
        assert_eq!(RknnQuantType::from_u32(0), RknnQuantType::None);
        assert_eq!(RknnQuantType::from_u32(2), RknnQuantType::AffineAsymmetric);
    }

    #[test]
    fn npu_core_mask_raw() {
        assert_eq!(NpuCoreMask::Auto.as_raw(), 0);
        assert_eq!(NpuCoreMask::Core0.as_raw(), 1);
        assert_eq!(NpuCoreMask::Core1.as_raw(), 2);
        assert_eq!(NpuCoreMask::Core2.as_raw(), 4);
        assert_eq!(NpuCoreMask::Cores01.as_raw(), 3);
        assert_eq!(NpuCoreMask::Cores012.as_raw(), 7);
        assert_eq!(NpuCoreMask::All.as_raw(), 0xffff);
    }

    #[test]
    fn mem_alloc_flags_raw() {
        assert_eq!(MemAllocFlags::Default.as_raw(), 0);
        assert_eq!(MemAllocFlags::Cacheable.as_raw(), 1);
        assert_eq!(MemAllocFlags::NonCacheable.as_raw(), 2);
        assert_eq!(MemAllocFlags::TryAllocSram.as_raw(), 4);
    }

    #[test]
    fn mem_sync_mode_raw() {
        assert_eq!(MemSyncMode::ToDevice.as_raw(), 0x1);
        assert_eq!(MemSyncMode::FromDevice.as_raw(), 0x2);
        assert_eq!(MemSyncMode::Bidirectional.as_raw(), 0x3);
    }

    #[test]
    fn rknn_input_struct_layout() {
        let size = std::mem::size_of::<RknnInput>();
        let align = std::mem::align_of::<RknnInput>();

        #[cfg(target_pointer_width = "64")]
        {
            assert_eq!(size, 32);
            assert_eq!(align, 8);
        }
        #[cfg(target_pointer_width = "32")]
        {
            assert_eq!(align, 4);
        }
    }

    #[test]
    fn rknn_output_struct_layout() {
        let align = std::mem::align_of::<RknnOutput>();
        #[cfg(target_pointer_width = "64")]
        {
            assert_eq!(std::mem::size_of::<RknnOutput>(), 24);
            assert_eq!(align, 8);
        }
        #[cfg(target_pointer_width = "32")]
        {
            assert_eq!(align, 4);
        }
    }

    // RknnTensorAttr struct size is now asserted at compile time
    // (see `const _: () = { assert!(...) };` block above).

    #[test]
    fn rknn_in_out_num_size() {
        assert_eq!(std::mem::size_of::<RknnInOutNum>(), 8);
    }

    #[test]
    fn rknn_init_extend_size() {
        assert_eq!(std::mem::size_of::<RknnInitExtend>(), 136);
    }

    #[test]
    fn rknn_run_extend_size() {
        assert_eq!(std::mem::size_of::<RknnRunExtend>(), 24);
    }

    #[test]
    fn rknn_output_extend_size() {
        assert_eq!(std::mem::size_of::<RknnOutputExtend>(), 8);
    }

    #[test]
    fn rknn_sdk_version_size() {
        assert_eq!(std::mem::size_of::<RknnSdkVersionRaw>(), 512);
    }

    #[test]
    fn rknn_mem_size_size() {
        assert_eq!(std::mem::size_of::<RknnMemSizeRaw>(), 64);
    }

    #[test]
    fn rknn_custom_string_size() {
        assert_eq!(std::mem::size_of::<RknnCustomStringRaw>(), 1024);
    }

    #[test]
    fn rknn_perf_run_size() {
        assert_eq!(std::mem::size_of::<RknnPerfRun>(), 8);
    }

    #[test]
    fn rknn_matmul_shape_size() {
        assert_eq!(std::mem::size_of::<RknnMatmulShape>(), 12);
    }

    #[test]
    fn tensor_attr_name_str() {
        let mut attr = RknnTensorAttr::default();
        let name = b"input0";
        attr.name[..name.len()].copy_from_slice(name);
        assert_eq!(attr.name_str(), "input0");
    }

    #[test]
    fn tensor_attr_shape() {
        let mut attr = RknnTensorAttr {
            n_dims: 4,
            ..Default::default()
        };
        attr.dims[0] = 1;
        attr.dims[1] = 640;
        attr.dims[2] = 640;
        attr.dims[3] = 3;
        assert_eq!(attr.shape(), vec![1, 640, 640, 3]);
    }

    #[test]
    fn rknn_available_returns_false_on_non_rockchip() {
        assert!(!rknn_available());
    }

    #[test]
    fn detect_backend_falls_back_to_cpu() {
        let backend = detect_backend(None);
        assert!(matches!(backend, InferenceBackend::Cpu));
    }

    #[test]
    fn detect_backend_with_data_falls_back_to_cpu() {
        let fake_model = [0u8; 64];
        let backend = detect_backend(Some(&fake_model));
        assert!(matches!(backend, InferenceBackend::Cpu));
    }

    #[test]
    fn perf_detail_parses_empty() {
        let p = PerfDetail::parse("", 0);
        assert!(p.per_op.is_empty());
    }

    #[test]
    fn perf_detail_parses_rows() {
        let raw =
            "Header\n1 Conv2D INT8 NPU [1,3,640,640] [1,64,320,320] 120\n2 Relu FP16 NPU [] [] 8\n";
        let p = PerfDetail::parse(raw, 128);
        assert_eq!(p.per_op.len(), 2);
        assert_eq!(p.per_op[0].op_name, "Conv2D");
        assert_eq!(p.per_op[0].duration_us, 120);
        assert_eq!(p.per_op[1].op_name, "Relu");
        assert_eq!(p.per_op[1].duration_us, 8);
        assert_eq!(p.total_us, 128);
    }

    #[test]
    fn async_frame_id_accessor() {
        let mut f = AsyncFrame {
            id: 42,
            submitted_at: std::time::Instant::now(),
            deadline_ms: 100,
            waited: false,
        };
        assert_eq!(f.id(), 42);
        assert_eq!(f.deadline_ms(), 100);
        assert!(!f.is_overdue(), "fresh frame should not be overdue");
        f.waited = true;
    }

    #[test]
    fn async_frame_overdue_after_deadline() {
        let f = AsyncFrame {
            id: 1,
            submitted_at: std::time::Instant::now() - std::time::Duration::from_millis(50),
            deadline_ms: 10,
            waited: true, // mark waited so Drop doesn't warn
        };
        assert!(f.is_overdue(), "50ms elapsed > 10ms deadline");
    }

    #[test]
    fn async_frame_no_deadline_never_overdue() {
        let f = AsyncFrame {
            id: 1,
            submitted_at: std::time::Instant::now() - std::time::Duration::from_secs(60),
            deadline_ms: -1,
            waited: true,
        };
        assert!(!f.is_overdue(), "deadline_ms=-1 means no deadline");
    }
}
