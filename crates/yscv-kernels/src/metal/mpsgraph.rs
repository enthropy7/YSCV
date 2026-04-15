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
        // SAFETY: (category 4) graph is a valid ObjC object allocated in new().
        unsafe {
            let _: () = msg_send![self.graph, release];
        }
    }
}

// SAFETY: `MpsGraph` owns a retained ObjC pointer. Sending it across
// threads is safe as long as the receiving thread has an active
// autoreleasepool when it performs ObjC messages (every call site in
// this crate wraps its work in `autoreleasepool(|| ...)`). The graph
// does not mutate shared state on message send — it records into its
// own internal node table, which is serialised by the caller (we put
// `MpsGraph` behind a `Mutex` in config-driven dispatchers). `Sync`
// is NOT claimed — concurrent `&self` graph mutation from multiple
// threads is not defined by Apple.
unsafe impl Send for MpsGraph {}

/// Wraps an MPSGraphTensor pointer (graph node output).
#[derive(Clone, Copy)]
pub struct MpsGraphTensorRef {
    pub(crate) ptr: *mut Object,
}

// SAFETY: `MpsGraphTensorRef` is a bare retained pointer; it represents
// an SSA node in the graph, so accessing it concurrently from multiple
// threads is as safe as concurrent reads of a Rust `Arc<T>` — MPSGraph
// treats these as immutable handles once the graph is compiled.
unsafe impl Send for MpsGraphTensorRef {}
unsafe impl Sync for MpsGraphTensorRef {}

/// Wraps a compiled MPSGraphExecutable for repeated inference.
pub struct MpsGraphExecutable {
    pub(crate) exe: *mut Object,
}

impl Drop for MpsGraphExecutable {
    fn drop(&mut self) {
        // SAFETY: (category 4) exe is a valid ObjC object from compilation.
        unsafe {
            let _: () = msg_send![self.exe, release];
        }
    }
}

// SAFETY: same argument as `MpsGraph` — the executable is a
// thread-safe-to-move compiled plan. Apple's documented contract is
// that a compiled executable can be dispatched on any `MTLCommandQueue`
// from any thread as long as each dispatch is individually
// autoreleasepool-scoped.
unsafe impl Send for MpsGraphExecutable {}

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
    // SAFETY: (category 4) delegates to ns_array_from_shape with valid dims.
    unsafe { ns_array_from_shape(&dims) }
}

// All `unsafe` blocks in MpsGraph methods fall under category 4: ObjC msg_send! calls
// to MPS framework classes. Class existence is checked; self.graph is a valid ObjC object
// created in new().
impl MpsGraph {
    /// Create a new empty MPSGraph.
    pub fn new() -> Option<Self> {
        // SAFETY: (category 4) MPSGraph class checked; result null-checked.
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
        // SAFETY: (category 4) msg_send to valid MPS graph object.
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
        // SAFETY: (category 4) msg_send to valid MPS graph object.
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
        // SAFETY: (category 4) msg_send to valid MPS graph; ns_data wraps valid slice.
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
        // SAFETY: (category 4) msg_send to valid MPS graph; ns_data wraps valid slice.
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
    pub fn cast_to_f16(&self, input: MpsGraphTensorRef) -> Result<MpsGraphTensorRef, KernelError> {
        // SAFETY: (category 4) msg_send to valid MPS graph object.
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
    pub fn cast_to_f32(&self, input: MpsGraphTensorRef) -> Result<MpsGraphTensorRef, KernelError> {
        // SAFETY: (category 4) msg_send to valid MPS graph/descriptor objects; classes checked.
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
        // SAFETY: (category 4) msg_send to valid MPS graph/descriptor objects; classes checked.
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
    pub fn add_bias(&self, input: MpsGraphTensorRef, bias: MpsGraphTensorRef) -> MpsGraphTensorRef {
        // SAFETY: (category 4) msg_send to valid MPS graph object.
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
        // SAFETY: (category 4) msg_send to valid MPS graph object.
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
        // SAFETY: (category 4) msg_send to valid MPS graph object.
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
        // SAFETY: (category 4) msg_send to valid MPS graph object.
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
        // SAFETY: (category 4) msg_send to valid MPS graph object.
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
        // SAFETY: (category 4) msg_send to valid MPS graph object.
        unsafe {
            let tensor: *mut Object = msg_send![self.graph,
                reLUWithTensor: input.ptr
                name: std::ptr::null::<Object>()];
            MpsGraphTensorRef { ptr: tensor }
        }
    }

    /// Sigmoid activation.
    pub fn sigmoid(&self, input: MpsGraphTensorRef) -> MpsGraphTensorRef {
        // SAFETY: (category 4) msg_send to valid MPS graph object.
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

    /// Elementwise natural exponent.
    pub fn exp(&self, input: MpsGraphTensorRef) -> MpsGraphTensorRef {
        // SAFETY: (category 4) msg_send to valid MPS graph object.
        unsafe {
            let tensor: *mut Object = msg_send![self.graph,
                exponentWithTensor: input.ptr
                name: std::ptr::null::<Object>()];
            MpsGraphTensorRef { ptr: tensor }
        }
    }

    /// MaxPool2d: NCHW layout.
    pub fn max_pool2d(
        &self,
        input: MpsGraphTensorRef,
        desc: &Pool2dDesc,
    ) -> Result<MpsGraphTensorRef, KernelError> {
        // SAFETY: (category 4) msg_send to valid MPS graph/descriptor objects; classes checked.
        unsafe {
            let pool_desc_cls =
                Class::get("MPSGraphPooling2DOpDescriptor").ok_or_else(|| KernelError::Gpu {
                    message: "MPSGraphPooling2DOpDescriptor class not available".into(),
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
        // SAFETY: (category 4) msg_send to valid MPS graph/descriptor objects; classes checked.
        unsafe {
            let pool_desc_cls =
                Class::get("MPSGraphPooling2DOpDescriptor").ok_or_else(|| KernelError::Gpu {
                    message: "MPSGraphPooling2DOpDescriptor class not available".into(),
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
        // SAFETY: (category 4) msg_send to valid MPS graph/descriptor objects; classes checked.
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
        // SAFETY: (category 4) chained msg_send calls to valid MPS graph object.
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
        // SAFETY: (category 4) msg_send to valid MPS graph object.
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
        // SAFETY: (category 4) msg_send to valid MPS graph/descriptor objects; classes checked.
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
        // SAFETY: (category 4) msg_send to valid MPS graph object.
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
        // SAFETY: (category 4) msg_send to valid MPS graph/descriptor objects; classes checked.
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
        // SAFETY: (category 4) msg_send to valid MPS graph/descriptor objects; classes checked.
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
        // SAFETY: (category 4) msg_send to valid MPS graph object.
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
        // SAFETY: (category 4) msg_send to valid MPS graph/descriptor objects; classes checked.
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
        // SAFETY: (category 4) msg_send to valid MPS graph object.
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
        // SAFETY: (category 4) ObjC classes checked; feeds dict built from valid tensors;
        // compiled executable retained before return.
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
                Class::get("MPSGraphCompilationDescriptor").ok_or_else(|| KernelError::Gpu {
                    message: "MPSGraphCompilationDescriptor class not available".into(),
                })?;
            let comp_desc: *mut Object = msg_send![comp_desc_cls, new];
            // optimization level 1 = default optimization
            let _: () = msg_send![comp_desc,
                setOptimizationLevel: 1u64];

            // Create MPSGraphDevice from MTLDevice
            let mpsg_device_cls = Class::get("MPSGraphDevice").ok_or_else(|| KernelError::Gpu {
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
        // SAFETY: (category 4) ObjC msg_send calls with valid objects; autoreleasepool drains.
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
                    let out_buf =
                        mtl_device.new_buffer(total_bytes, MTLResourceOptions::StorageModeShared);
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

/// Pre-built, retained I/O binding for a compiled MPSGraph.
///
/// Wraps both the user's input `Buffer`s **and** the caller-supplied output
/// `Buffer`s in `MPSGraphTensorData` objects, packed into two `NSArray`s —
/// all built **once** at compile time. The hot path (`run_with_prepared`)
/// reuses these arrays on every call and tells MPSGraph to write each
/// output tensor directly into the matching pre-allocated shared buffer.
///
/// Savings per call versus a naive path:
/// - No `MPSGraphTensorData alloc/init` for inputs or outputs
/// - No `NSArray arrayWithObjects:count:` for inputs or outputs
/// - No `mpsndarray readBytes:` copy (MPS writes in-place into our bufs)
/// - No `new_buffer` allocation for outputs (we own them for the plan's life)
///
/// `Drop` releases both retained `NSArray`s; those in turn release the
/// tensor-data wrappers they contain.
pub struct PreparedInputs {
    inputs_ns_array: *mut Object,
    results_ns_array: *mut Object,
}

// SAFETY: `PreparedInputs` holds retained ObjC objects that are safe to
// access from any thread as long as no concurrent mutation happens. The
// hot path only *reads* the NSArrays (passes them to `runWith...`), so
// moving the struct across threads is sound. Sending must be paired with
// an autoreleasepool on the receiving thread (same contract as other Metal
// types in this crate).
unsafe impl Send for PreparedInputs {}
unsafe impl Sync for PreparedInputs {}

impl Drop for PreparedInputs {
    fn drop(&mut self) {
        // SAFETY: (category 4) release of retained ObjC objects. Releasing
        // an NSArray transitively releases its contained tensor-datas.
        unsafe {
            if !self.inputs_ns_array.is_null() {
                let _: () = msg_send![self.inputs_ns_array, release];
            }
            if !self.results_ns_array.is_null() {
                let _: () = msg_send![self.results_ns_array, release];
            }
        }
    }
}

/// Build one retained NSArray of MPSGraphTensorData, each wrapping the
/// caller's MTL buffer at the given shape/dtype. Returns a +1 retain that
/// the caller must balance. The per-tensor-data +1 from `init` is released
/// before return; the NSArray holds the strong references.
///
/// # Safety
/// - `buffers` must outlive the returned NSArray.
/// - Must be called from a thread with an active autoreleasepool.
unsafe fn build_retained_tensor_data_array(
    tensor_data_cls: &Class,
    ns_array_cls: &Class,
    specs: &[(&Buffer, &[usize], u32)],
) -> Result<*mut Object, KernelError> {
    let mut datas: Vec<*mut Object> = Vec::with_capacity(specs.len());
    for &(buf, shape, dtype) in specs {
        // SAFETY: `shape` is a borrowed slice; `ns_array_from_usize` documents
        // that it reads the slice and builds an autoreleased NSArray<NSNumber>.
        // Caller of `build_retained_tensor_data_array` guarantees an active
        // autoreleasepool per this fn's `# Safety` contract.
        let shape_arr = unsafe { ns_array_from_usize(shape)? };
        // SAFETY: `[Class alloc]` is always safe to call on a valid Class*;
        // `tensor_data_cls` is the MPSGraphTensorData class we looked up at
        // compile time and retained.
        let alloc: *mut Object = unsafe { msg_send![tensor_data_cls, alloc] };
        // SAFETY: `alloc` is a freshly-allocated MPSGraphTensorData instance
        // (one retain). `buf.as_ptr()` returns a valid MTLBuffer pointer whose
        // lifetime is guaranteed by this fn's contract ("buffers must outlive
        // the returned NSArray"). `shape_arr` is a valid autoreleased
        // NSArray<NSNumber>. `dtype` is a u32 enum value expected by the init.
        let td: *mut Object = unsafe {
            msg_send![alloc,
                initWithMTLBuffer: buf.as_ptr()
                shape: shape_arr
                dataType: dtype]
        };
        datas.push(td);
    }
    // SAFETY: `datas` is a contiguous slice of valid Objective-C pointers we
    // just created; `arrayWithObjects:count:` reads `datas.len()` pointers
    // and returns a new autoreleased NSArray that retains each element.
    let ns_array: *mut Object = unsafe {
        msg_send![ns_array_cls,
            arrayWithObjects: datas.as_ptr()
            count: datas.len()]
    };
    // SAFETY: `ns_array` is a valid NSArray pointer returned from
    // `arrayWithObjects:count:`. `retain` is a no-arg message on the
    // NSObject root class — always safe on a valid pointer.
    let _: () = unsafe { msg_send![ns_array, retain] };
    for td in &datas {
        // SAFETY: every `td` in `datas` is balanced with exactly one retain
        // from `alloc/init` above. The NSArray took its own retain in
        // `arrayWithObjects:count:`, so we can safely release our reference
        // here — the array keeps the element alive.
        let _: () = unsafe { msg_send![*td, release] };
    }
    Ok(ns_array)
}

impl MpsGraph {
    /// Build retained `MPSGraphTensorData` NSArrays for a fixed set of
    /// input + output buffers. Call this **once** at compile time; pass
    /// the result to `run_with_prepared` on every subsequent run.
    ///
    /// `inputs`: each tuple is `(placeholder, backing_buffer, shape, dtype)`.
    /// The placeholder is kept implicit (by feed order) — the compiled
    /// executable binds by positional feed order.
    ///
    /// `outputs`: each tuple is `(backing_buffer, shape, dtype)` in the
    /// same order as `target_tensors` passed to `compile`. MPSGraph writes
    /// results directly into these buffers.
    pub fn prepare_inputs(
        &self,
        inputs: &[(MpsGraphTensorRef, &Buffer, &[usize], u32)],
        outputs: &[(&Buffer, &[usize], u32)],
    ) -> Result<PreparedInputs, KernelError> {
        // SAFETY: (category 4) ObjC class lookups + retained NSArray construction;
        // the +1 retains on both arrays balance with Drop's releases. The
        // autoreleasepool drains transient shape arrays created inside the
        // helper.
        unsafe {
            autoreleasepool(|| {
                let tensor_data_cls =
                    Class::get("MPSGraphTensorData").ok_or_else(|| KernelError::Gpu {
                        message: "MPSGraphTensorData class not available".into(),
                    })?;
                let ns_array_cls = Class::get("NSArray").ok_or_else(|| KernelError::Gpu {
                    message: "NSArray class not available".into(),
                })?;

                let input_specs: Vec<(&Buffer, &[usize], u32)> = inputs
                    .iter()
                    .map(|&(_, buf, shape, dtype)| (buf, shape, dtype))
                    .collect();
                let inputs_ns_array =
                    build_retained_tensor_data_array(tensor_data_cls, ns_array_cls, &input_specs)?;
                let results_ns_array =
                    build_retained_tensor_data_array(tensor_data_cls, ns_array_cls, outputs)?;

                Ok(PreparedInputs {
                    inputs_ns_array,
                    results_ns_array,
                })
            })
        }
    }

    /// Synchronous run: submits + waits internally. Uses the plan's cached
    /// input/output NSArrays — no ObjC allocation in the hot path.
    ///
    /// After return, the caller reads f32 (or f16) output data directly
    /// from the output buffers registered in `prepare_inputs`.
    pub fn run_with_prepared(
        &self,
        executable: &MpsGraphExecutable,
        queue: &CommandQueue,
        prepared: &PreparedInputs,
    ) -> Result<(), KernelError> {
        // SAFETY: (category 4) msg_send with valid retained objects.
        unsafe {
            autoreleasepool(|| {
                let _results: *mut Object = msg_send![executable.exe,
                    runWithMTLCommandQueue: queue.as_ptr()
                    inputsArray: prepared.inputs_ns_array
                    resultsArray: prepared.results_ns_array
                    executionDescriptor: std::ptr::null::<Object>()];
                Ok(())
            })
        }
    }

    /// Encode the graph into a caller-owned MTL command buffer **and
    /// commit it asynchronously**. Use this to pipeline inferences:
    /// encode+commit frame N, then immediately encode+commit frame N+1
    /// using a different `PreparedInputs` (different backing buffers)
    /// while the GPU keeps chewing on frame N.
    ///
    /// The MTL command buffer is committed through MPS's
    /// `MPSCommandBuffer` wrapper so that all graph work is properly
    /// flushed to the underlying MTL queue. Caller later waits for
    /// completion via `command_buffer.wait_until_completed()`.
    ///
    /// Returns immediately after commit — CPU does not block on GPU.
    pub fn encode_and_commit_with_prepared(
        &self,
        executable: &MpsGraphExecutable,
        command_buffer: &CommandBufferRef,
        prepared: &PreparedInputs,
    ) -> Result<(), KernelError> {
        // SAFETY: (category 4) ObjC class lookup + msg_send with valid
        // retained objects. MPSCommandBuffer wraps the MTL command buffer
        // transparently; its `-commit` forwards to the underlying MTL CB
        // after draining any pending MPS-internal work. The wrapper is
        // autoreleased — its lifetime ends at the pool drain, which is
        // AFTER its commit finalizes encoding onto the MTL CB.
        unsafe {
            autoreleasepool(|| {
                let mps_cb_cls =
                    Class::get("MPSCommandBuffer").ok_or_else(|| KernelError::Gpu {
                        message: "MPSCommandBuffer class not available".into(),
                    })?;
                let mps_cb: *mut Object = msg_send![mps_cb_cls,
                    commandBufferWithCommandBuffer: command_buffer.as_ptr()];
                let _results: *mut Object = msg_send![executable.exe,
                    encodeToCommandBuffer: mps_cb
                    inputsArray: prepared.inputs_ns_array
                    resultsArray: prepared.results_ns_array
                    executionDescriptor: std::ptr::null::<Object>()];
                // Commit through the MPS wrapper so any internal staging
                // (weights uploads, auxiliary dispatches) is finalized.
                let _: () = msg_send![mps_cb, commit];
                Ok(())
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
