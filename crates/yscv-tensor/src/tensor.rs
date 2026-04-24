use std::sync::Arc;

use super::aligned::AlignedVec;
use super::error::{DType, TensorError};
use super::shape::{compute_strides, shape_element_count};

// ── Inline shape/strides storage (no heap alloc for ≤6D tensors) ─────────

// WHY 6: covers all common tensor ranks (scalar(0)..conv weight(5)) without heap allocation.
const INLINE_CAP: usize = 6;

/// Stack-allocated small vector for tensor shape/strides.
/// Stores up to 6 dimensions inline; falls back to heap for higher ranks.
#[derive(Clone)]
pub(crate) enum DimsVec {
    Inline { buf: [usize; INLINE_CAP], len: u8 },
    Heap(Vec<usize>),
}

impl DimsVec {
    #[inline]
    fn new() -> Self {
        DimsVec::Inline {
            buf: [0; INLINE_CAP],
            len: 0,
        }
    }

    #[inline]
    fn as_slice(&self) -> &[usize] {
        match self {
            DimsVec::Inline { buf, len } => &buf[..*len as usize],
            DimsVec::Heap(v) => v,
        }
    }

    #[inline]
    fn to_vec(&self) -> Vec<usize> {
        self.as_slice().to_vec()
    }
}

impl std::ops::Deref for DimsVec {
    type Target = [usize];
    #[inline]
    fn deref(&self) -> &[usize] {
        self.as_slice()
    }
}

impl From<Vec<usize>> for DimsVec {
    #[inline]
    fn from(v: Vec<usize>) -> Self {
        if v.len() <= INLINE_CAP {
            let mut buf = [0usize; INLINE_CAP];
            buf[..v.len()].copy_from_slice(&v);
            DimsVec::Inline {
                buf,
                len: v.len() as u8,
            }
        } else {
            DimsVec::Heap(v)
        }
    }
}

impl From<&[usize]> for DimsVec {
    #[inline]
    fn from(s: &[usize]) -> Self {
        if s.len() <= INLINE_CAP {
            let mut buf = [0usize; INLINE_CAP];
            buf[..s.len()].copy_from_slice(s);
            DimsVec::Inline {
                buf,
                len: s.len() as u8,
            }
        } else {
            DimsVec::Heap(s.to_vec())
        }
    }
}

impl PartialEq for DimsVec {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl std::fmt::Debug for DimsVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_slice().fmt(f)
    }
}

/// Logical device where a tensor resides.
///
/// This is currently a metadata tag only — no actual data transfer occurs.
/// GPU data movement is handled externally (e.g. via `GpuSession` in yscv-kernels).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Device {
    /// CPU (host) memory — the default.
    #[default]
    Cpu,
    /// GPU device with the given index.
    Gpu(usize),
}

/// Memory-layout tag describing how Conv-shaped tensors are laid out.
///
/// NCHW is ONNX's native layout and the default. NHWC is what yscv's CPU
/// runner currently uses internally for Conv hot paths. NCHWc packs
/// channels into SIMD-sized blocks (8 for AVX2/NEON, 16 for AVX-512),
/// which lets 3×3 non-depthwise Conv use direct SIMD loads over the
/// C-block stride rather than paying tail-handling cost.
///
/// The tag is **metadata** — it does not change how `data()` / `shape()`
/// are interpreted by Tensor itself. Consumers (conv kernels, layout
/// transformers) read the tag to pick the right kernel dispatch.
/// Shape ordering matches the layout: NCHW tensors have shape
/// `[N, C, H, W]`, NHWC `[N, H, W, C]`, NCHWc `[N, C/block, H, W, block]`
/// (5-D). Data is always contiguous in its declared ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Layout {
    /// ONNX default: `[N, C, H, W]`.
    #[default]
    NCHW,
    /// Channels-last: `[N, H, W, C]`. Used by yscv's CPU conv paths.
    NHWC,
    /// Blocked channels-first: `[N, C/block, H, W, block]`. `block` is a
    /// SIMD-lane-aligned chunk size (8 on AVX2/NEON, 16 on AVX-512).
    NCHWc {
        /// Channels-per-block. Must be a power of two.
        block: u8,
    },
}

/// Internal typed storage for tensor data.
#[derive(Debug, Clone)]
pub(crate) enum Storage {
    F32(AlignedVec<f32>),
    F16(Vec<u16>),
    BF16(Vec<u16>),
}

impl PartialEq for Storage {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Storage::F32(a), Storage::F32(b)) => a == b,
            (Storage::F16(a), Storage::F16(b)) => a == b,
            (Storage::BF16(a), Storage::BF16(b)) => a == b,
            _ => false,
        }
    }
}

impl Storage {
    fn len(&self) -> usize {
        match self {
            Storage::F32(v) => v.len(),
            Storage::F16(v) => v.len(),
            Storage::BF16(v) => v.len(),
        }
    }

    fn dtype(&self) -> DType {
        match self {
            Storage::F32(_) => DType::F32,
            Storage::F16(_) => DType::F16,
            Storage::BF16(_) => DType::BF16,
        }
    }
}

/// A compact, contiguous multi-dtype tensor representation.
///
/// Default dtype is `F32`. FP16 and BF16 dtypes are stored natively as `u16` bit patterns
/// and can be created via `from_f16` / `from_bf16` or converted via `to_dtype`.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    shape: DimsVec,
    strides: DimsVec,
    // Arc-shared storage so `Tensor::clone()` / `reshape()` are O(1) refcount
    // bumps. Writes go through `Arc::make_mut` (copy-on-write): free when
    // refcount == 1, clone-once semantics otherwise. This lets the runtime
    // pass `Arc<PackedWeight>` around at zero cost and makes Reshape a
    // pointer-swap. Storage itself is still typed (F32/F16/BF16) — the Arc
    // only guards the data buffer, not the dtype tag.
    storage: Arc<Storage>,
    device: Device,
    layout: Layout,
}

impl Tensor {
    /// Builds a scalar tensor from one value.
    pub fn scalar(value: f32) -> Self {
        Self {
            shape: DimsVec::new(),
            strides: DimsVec::new(),
            storage: Arc::new(Storage::F32(AlignedVec::filled(1, value))),
            device: Device::Cpu,
            layout: Layout::NCHW,
        }
    }

    /// Creates a tensor from pre-validated shape, strides, and data.
    /// No validation, no heap allocation (shape/strides stored inline for ≤6D).
    #[inline]
    pub fn from_raw_parts(shape: &[usize], strides: &[usize], data: AlignedVec<f32>) -> Self {
        debug_assert_eq!(
            shape.iter().copied().product::<usize>(),
            data.len(),
            "from_raw_parts: shape product != data.len()"
        );
        Self {
            shape: DimsVec::from(shape),
            strides: DimsVec::from(strides),
            storage: Arc::new(Storage::F32(data)),
            device: Device::Cpu,
            layout: Layout::NCHW,
        }
    }

    /// Builds a tensor from `shape` and a pre-filled [`AlignedVec`].
    ///
    /// This avoids the extra copy that [`from_vec`](Self::from_vec) performs when
    /// converting a `Vec<f32>` into aligned storage.  Use this when the output
    /// buffer was already allocated as an `AlignedVec` (e.g. via
    /// [`AlignedVec::uninitialized`] filled by a BLAS/SIMD kernel).
    pub fn from_aligned(shape: Vec<usize>, data: AlignedVec<f32>) -> Result<Self, TensorError> {
        let expected = shape_element_count(&shape).ok_or_else(|| TensorError::SizeOverflow {
            shape: shape.clone(),
        })?;
        if expected != data.len() {
            return Err(TensorError::SizeMismatch {
                shape,
                data_len: data.len(),
            });
        }

        let strides = compute_strides(&shape).ok_or_else(|| TensorError::SizeOverflow {
            shape: shape.clone(),
        })?;

        Ok(Self {
            shape: DimsVec::from(shape),
            strides: DimsVec::from(strides),
            storage: Arc::new(Storage::F32(data)),
            device: Device::Cpu,
            layout: Layout::NCHW,
        })
    }

    /// Builds a tensor from `shape` and raw contiguous `f32` data.
    pub fn from_vec(shape: Vec<usize>, data: Vec<f32>) -> Result<Self, TensorError> {
        let expected = shape_element_count(&shape).ok_or_else(|| TensorError::SizeOverflow {
            shape: shape.clone(),
        })?;
        if expected != data.len() {
            return Err(TensorError::SizeMismatch {
                shape,
                data_len: data.len(),
            });
        }

        let strides = compute_strides(&shape).ok_or_else(|| TensorError::SizeOverflow {
            shape: shape.clone(),
        })?;

        Ok(Self {
            shape: DimsVec::from(shape),
            strides: DimsVec::from(strides),
            storage: Arc::new(Storage::F32(AlignedVec::from_vec(data))),
            device: Device::Cpu,
            layout: Layout::NCHW,
        })
    }

    /// Builds a tensor from `shape` and raw FP16 bit patterns (`u16`).
    pub fn from_f16(shape: Vec<usize>, data: Vec<u16>) -> Result<Self, TensorError> {
        let expected = shape_element_count(&shape).ok_or_else(|| TensorError::SizeOverflow {
            shape: shape.clone(),
        })?;
        if expected != data.len() {
            return Err(TensorError::SizeMismatch {
                shape,
                data_len: data.len(),
            });
        }
        let strides = compute_strides(&shape).ok_or_else(|| TensorError::SizeOverflow {
            shape: shape.clone(),
        })?;
        Ok(Self {
            shape: DimsVec::from(shape),
            strides: DimsVec::from(strides),
            storage: Arc::new(Storage::F16(data)),
            device: Device::Cpu,
            layout: Layout::NCHW,
        })
    }

    /// Builds a tensor from `shape` and raw BF16 bit patterns (`u16`).
    pub fn from_bf16(shape: Vec<usize>, data: Vec<u16>) -> Result<Self, TensorError> {
        let expected = shape_element_count(&shape).ok_or_else(|| TensorError::SizeOverflow {
            shape: shape.clone(),
        })?;
        if expected != data.len() {
            return Err(TensorError::SizeMismatch {
                shape,
                data_len: data.len(),
            });
        }
        let strides = compute_strides(&shape).ok_or_else(|| TensorError::SizeOverflow {
            shape: shape.clone(),
        })?;
        Ok(Self {
            shape: DimsVec::from(shape),
            strides: DimsVec::from(strides),
            storage: Arc::new(Storage::BF16(data)),
            device: Device::Cpu,
            layout: Layout::NCHW,
        })
    }

    /// Builds a 1-D tensor from a slice. Equivalent to `from_vec(vec![data.len()], data.to_vec())`.
    pub fn from_slice(data: &[f32]) -> Self {
        let n = data.len();
        Self {
            shape: DimsVec::from(vec![n]),
            strides: DimsVec::from(vec![1usize]),
            storage: Arc::new(Storage::F32(AlignedVec::from_vec(data.to_vec()))),
            device: Device::Cpu,
            layout: Layout::NCHW,
        }
    }

    /// Builds a value-initialized tensor for a given shape.
    ///
    /// Alias: [`full`](Self::full) is provided as a more familiar name.
    pub fn filled(shape: Vec<usize>, value: f32) -> Result<Self, TensorError> {
        let count = shape_element_count(&shape).ok_or_else(|| TensorError::SizeOverflow {
            shape: shape.clone(),
        })?;
        let strides = compute_strides(&shape).ok_or_else(|| TensorError::SizeOverflow {
            shape: shape.clone(),
        })?;

        Ok(Self {
            shape: DimsVec::from(shape),
            strides: DimsVec::from(strides),
            storage: Arc::new(Storage::F32(AlignedVec::filled(count, value))),
            device: Device::Cpu,
            layout: Layout::NCHW,
        })
    }

    /// Builds a zero-initialized tensor for a given shape.
    ///
    /// Uses `alloc_zeroed` under the hood so the OS can provide pre-zeroed pages
    /// without writing every byte — significantly faster for large tensors.
    pub fn zeros(shape: Vec<usize>) -> Result<Self, TensorError> {
        let count = shape_element_count(&shape).ok_or_else(|| TensorError::SizeOverflow {
            shape: shape.clone(),
        })?;
        let strides = compute_strides(&shape).ok_or_else(|| TensorError::SizeOverflow {
            shape: shape.clone(),
        })?;

        Ok(Self {
            shape: DimsVec::from(shape),
            strides: DimsVec::from(strides),
            storage: Arc::new(Storage::F32(AlignedVec::calloc(count))),
            device: Device::Cpu,
            layout: Layout::NCHW,
        })
    }

    /// Builds a one-initialized tensor for a given shape.
    pub fn ones(shape: Vec<usize>) -> Result<Self, TensorError> {
        Self::filled(shape, 1.0)
    }

    /// Builds a tensor filled with `value`. Alias for [`filled`](Self::filled).
    pub fn full(shape: Vec<usize>, value: f32) -> Result<Self, TensorError> {
        Self::filled(shape, value)
    }

    /// Returns the tensor shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the tensor strides.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Returns tensor rank (number of axes).
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Returns element count.
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    /// Returns `true` if tensor contains zero elements.
    pub fn is_empty(&self) -> bool {
        self.storage.len() == 0
    }

    /// Returns the element data type.
    pub fn dtype(&self) -> DType {
        self.storage.dtype()
    }

    /// Returns the logical device this tensor is associated with.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Returns a copy of this tensor tagged with the given device.
    ///
    /// This is currently a metadata-only operation — no actual data transfer
    /// occurs. GPU data movement is handled by `GpuSession` in yscv-kernels.
    pub fn to_device(&self, device: Device) -> Self {
        Self {
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            storage: self.storage.clone(),
            device,
            layout: self.layout,
        }
    }

    /// Returns the memory-layout tag. Defaults to `Layout::NCHW`.
    pub fn layout(&self) -> Layout {
        self.layout
    }

    /// Returns a clone of this tensor retagged with `layout`. This is a
    /// **metadata** change only — the caller is responsible for having
    /// already laid out the data correctly for the new layout (use
    /// `yscv_kernels::ops::layout` kernels for actual reordering).
    pub fn with_layout(mut self, layout: Layout) -> Self {
        self.layout = layout;
        self
    }

    /// Returns `true` if the tensor stores f32 data.
    pub fn is_f32(&self) -> bool {
        matches!(&*self.storage, Storage::F32(_))
    }

    /// Fallible version of `data()` — returns an error for non-F32 tensors.
    pub fn try_data(&self) -> Result<&[f32], TensorError> {
        match &*self.storage {
            Storage::F32(v) => Ok(v),
            _ => Err(TensorError::DTypeMismatch {
                expected: DType::F32,
                got: self.dtype(),
            }),
        }
    }

    /// Fallible version of `data_mut()` — returns an error for non-F32 tensors.
    ///
    /// Triggers a copy-on-write clone of the underlying storage if it is shared
    /// (`Arc::strong_count > 1`). When this tensor uniquely owns its storage —
    /// the common case — the call is a no-op and returns a direct `&mut [f32]`.
    pub fn try_data_mut(&mut self) -> Result<&mut [f32], TensorError> {
        let dt = self.storage.dtype();
        match Arc::make_mut(&mut self.storage) {
            Storage::F32(v) => Ok(v),
            _ => Err(TensorError::DTypeMismatch {
                expected: DType::F32,
                got: dt,
            }),
        }
    }

    /// Returns an immutable view over contiguous f32 storage.
    ///
    /// # Panics
    /// Panics if the tensor dtype is not F32. Use `try_data()` for a fallible version.
    pub fn data(&self) -> &[f32] {
        self.try_data().expect("tensor is not F32")
    }

    /// Returns a mutable view over contiguous f32 storage.
    ///
    /// # Panics
    /// Panics if the tensor dtype is not F32. Use `try_data_mut()` for a fallible version.
    pub fn data_mut(&mut self) -> &mut [f32] {
        self.try_data_mut().expect("tensor is not F32")
    }

    /// Alias for `try_data()` for backward compatibility.
    pub fn try_data_f32(&self) -> Result<&[f32], TensorError> {
        self.try_data()
    }

    /// Returns raw FP16 bit-pattern data if dtype is F16.
    pub fn data_f16(&self) -> Result<&[u16], TensorError> {
        match &*self.storage {
            Storage::F16(v) => Ok(v),
            _ => Err(TensorError::DTypeMismatch {
                expected: DType::F16,
                got: self.dtype(),
            }),
        }
    }

    /// Returns raw BF16 bit-pattern data if dtype is BF16.
    pub fn data_bf16(&self) -> Result<&[u16], TensorError> {
        match &*self.storage {
            Storage::BF16(v) => Ok(v),
            _ => Err(TensorError::DTypeMismatch {
                expected: DType::BF16,
                got: self.dtype(),
            }),
        }
    }

    /// Converts this tensor to the specified dtype, returning a new tensor.
    /// Converting to the same dtype is a no-op clone.
    pub fn to_dtype(&self, target: DType) -> Self {
        if self.dtype() == target {
            return self.clone();
        }
        let f32_data = self.to_f32_vec();
        let storage = match target {
            DType::F32 => Storage::F32(AlignedVec::from_vec(f32_data)),
            DType::F16 => Storage::F16(f32_data.iter().map(|&v| f32_to_fp16_bits(v)).collect()),
            DType::BF16 => Storage::BF16(f32_data.iter().map(|&v| f32_to_bf16_bits(v)).collect()),
        };
        Self {
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            storage: Arc::new(storage),
            device: self.device,
            layout: self.layout,
        }
    }

    /// Returns f32 data regardless of internal dtype (converts if necessary).
    pub(crate) fn to_f32_vec(&self) -> Vec<f32> {
        match &*self.storage {
            Storage::F32(v) => v.as_slice().to_vec(),
            Storage::F16(v) => v.iter().map(|&bits| fp16_bits_to_f32(bits)).collect(),
            Storage::BF16(v) => v.iter().map(|&bits| bf16_bits_to_f32(bits)).collect(),
        }
    }

    /// Reads one element by multi-dimensional index (always returns f32).
    pub fn get(&self, indices: &[usize]) -> Result<f32, TensorError> {
        let offset = self.offset_from_indices(indices)?;
        Ok(match &*self.storage {
            Storage::F32(v) => v[offset],
            Storage::F16(v) => fp16_bits_to_f32(v[offset]),
            Storage::BF16(v) => bf16_bits_to_f32(v[offset]),
        })
    }

    /// Writes one element by multi-dimensional index (stores as native dtype).
    ///
    /// Triggers a copy-on-write clone of the underlying storage if it is shared
    /// with another `Tensor` (`Arc::strong_count > 1`). Single-owner writes are
    /// in-place.
    pub fn set(&mut self, indices: &[usize], value: f32) -> Result<(), TensorError> {
        let offset = self.offset_from_indices(indices)?;
        match Arc::make_mut(&mut self.storage) {
            Storage::F32(v) => v[offset] = value,
            Storage::F16(v) => v[offset] = f32_to_fp16_bits(value),
            Storage::BF16(v) => v[offset] = f32_to_bf16_bits(value),
        }
        Ok(())
    }

    /// Returns a reshaped tensor that shares the underlying data buffer
    /// with the original via `Arc`. This is an O(1) refcount bump — no
    /// allocation, no copy. If either tensor is subsequently written via
    /// `data_mut` or `set`, the storage is cloned lazily (copy-on-write).
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self, TensorError> {
        let new_count =
            shape_element_count(&new_shape).ok_or_else(|| TensorError::SizeOverflow {
                shape: new_shape.clone(),
            })?;
        if new_count != self.len() {
            return Err(TensorError::ReshapeSizeMismatch {
                from: self.shape.to_vec(),
                to: new_shape,
            });
        }

        let new_strides = compute_strides(&new_shape).ok_or_else(|| TensorError::SizeOverflow {
            shape: new_shape.clone(),
        })?;

        Ok(Self {
            shape: DimsVec::from(new_shape),
            strides: DimsVec::from(new_strides),
            storage: self.storage.clone(),
            device: self.device,
            layout: self.layout,
        })
    }

    /// Consumes the tensor and returns a reshaped version without copying data.
    pub fn into_reshape(self, new_shape: Vec<usize>) -> Result<Self, TensorError> {
        let new_count =
            shape_element_count(&new_shape).ok_or_else(|| TensorError::SizeOverflow {
                shape: new_shape.clone(),
            })?;
        if new_count != self.len() {
            return Err(TensorError::ReshapeSizeMismatch {
                from: self.shape.to_vec(),
                to: new_shape,
            });
        }

        let new_strides = compute_strides(&new_shape).ok_or_else(|| TensorError::SizeOverflow {
            shape: new_shape.clone(),
        })?;

        Ok(Self {
            shape: DimsVec::from(new_shape),
            strides: DimsVec::from(new_strides),
            storage: self.storage,
            device: self.device,
            layout: self.layout,
        })
    }

    pub(crate) fn offset_from_indices(&self, indices: &[usize]) -> Result<usize, TensorError> {
        if indices.len() != self.rank() {
            return Err(TensorError::InvalidIndexRank {
                expected: self.rank(),
                got: indices.len(),
            });
        }

        let mut offset = 0usize;
        for (axis, (index, dim)) in indices.iter().zip(self.shape.iter()).enumerate() {
            if *index >= *dim {
                return Err(TensorError::IndexOutOfBounds {
                    axis,
                    index: *index,
                    dim: *dim,
                });
            }
            offset = offset
                .checked_add(index.checked_mul(self.strides[axis]).ok_or_else(|| {
                    TensorError::SizeOverflow {
                        shape: self.shape.to_vec(),
                    }
                })?)
                .ok_or_else(|| TensorError::SizeOverflow {
                    shape: self.shape.to_vec(),
                })?;
        }
        Ok(offset)
    }
}

// ── FP16/BF16 bit conversion primitives ────────────────────────────

fn f32_to_fp16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007FFFFF;

    if exponent == 0xFF {
        return sign | 0x7C00 | if mantissa != 0 { 0x0200 } else { 0 };
    }
    let unbiased = exponent - 127;
    if unbiased < -24 {
        return sign;
    }
    if unbiased < -14 {
        let shift = -1 - unbiased;
        let subnormal = ((mantissa | 0x00800000) >> (shift + 13)) as u16;
        return sign | subnormal;
    }
    if unbiased > 15 {
        return sign | 0x7C00;
    }
    let fp16_exp = ((unbiased + 15) as u16) << 10;
    let fp16_man = (mantissa >> 13) as u16;
    sign | fp16_exp | fp16_man
}

fn fp16_bits_to_f32(half: u16) -> f32 {
    let sign = ((half & 0x8000) as u32) << 16;
    let exponent = (half >> 10) & 0x1F;
    let mantissa = (half & 0x03FF) as u32;
    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign);
        }
        let mut e = 0i32;
        let mut m = mantissa;
        while m & 0x0400 == 0 {
            m <<= 1;
            e += 1;
        }
        let f32_exp = ((127 - 15 - e) as u32) << 23;
        let f32_man = (m & 0x03FF) << 13;
        return f32::from_bits(sign | f32_exp | f32_man);
    }
    if exponent == 31 {
        let f32_bits = sign | 0x7F800000 | if mantissa != 0 { 0x00400000 } else { 0 };
        return f32::from_bits(f32_bits);
    }
    let f32_exp = ((exponent as u32) + 112) << 23;
    let f32_man = mantissa << 13;
    f32::from_bits(sign | f32_exp | f32_man)
}

fn f32_to_bf16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    // Round to nearest even
    let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

// ── Display impl ────────────────────────────────────────────────────

impl std::fmt::Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shape = self.shape();
        let dtype = self.dtype();
        let n = self.len();

        write!(f, "Tensor({shape:?}, {dtype:?}")?;

        // Show first few and last few values for a compact preview.
        const MAX_SHOW: usize = 6;
        if n == 0 {
            write!(f, ", []")?;
        } else {
            let vals = self.to_f32_vec();
            write!(f, ", [")?;
            if n <= MAX_SHOW {
                for (i, v) in vals.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{v}")?;
                }
            } else {
                let head = MAX_SHOW / 2;
                let tail = MAX_SHOW - head;
                for (i, v) in vals[..head].iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{v}")?;
                }
                write!(f, ", ...")?;
                for v in &vals[n - tail..] {
                    write!(f, ", {v}")?;
                }
            }
            write!(f, "]")?;
        }
        write!(f, ")")
    }
}

// ── Operator overloading (std::ops) ─────────────────────────────────

impl std::ops::Add for &Tensor {
    type Output = Tensor;
    /// Element-wise addition. Panics on shape mismatch.
    fn add(self, rhs: Self) -> Tensor {
        Tensor::add(self, rhs).expect("Tensor + Tensor: shape mismatch")
    }
}

impl std::ops::Add for Tensor {
    type Output = Tensor;
    fn add(self, rhs: Self) -> Tensor {
        Tensor::add(&self, &rhs).expect("Tensor + Tensor: shape mismatch")
    }
}

impl std::ops::Sub for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Self) -> Tensor {
        Tensor::sub(self, rhs).expect("Tensor - Tensor: shape mismatch")
    }
}

impl std::ops::Sub for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Self) -> Tensor {
        Tensor::sub(&self, &rhs).expect("Tensor - Tensor: shape mismatch")
    }
}

impl std::ops::Mul for &Tensor {
    type Output = Tensor;
    /// Element-wise multiplication. Panics on shape mismatch.
    fn mul(self, rhs: Self) -> Tensor {
        Tensor::mul(self, rhs).expect("Tensor * Tensor: shape mismatch")
    }
}

impl std::ops::Mul for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Self) -> Tensor {
        Tensor::mul(&self, &rhs).expect("Tensor * Tensor: shape mismatch")
    }
}

impl std::ops::Mul<f32> for &Tensor {
    type Output = Tensor;
    /// Scalar multiplication.
    fn mul(self, rhs: f32) -> Tensor {
        Tensor::scale(self, rhs)
    }
}

impl std::ops::Mul<f32> for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: f32) -> Tensor {
        Tensor::scale(&self, rhs)
    }
}

impl std::ops::Div for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: Self) -> Tensor {
        Tensor::div(self, rhs).expect("Tensor / Tensor: shape mismatch")
    }
}

impl std::ops::Div for Tensor {
    type Output = Tensor;
    fn div(self, rhs: Self) -> Tensor {
        Tensor::div(&self, &rhs).expect("Tensor / Tensor: shape mismatch")
    }
}

impl std::ops::Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        Tensor::neg(self)
    }
}

impl std::ops::Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        Tensor::neg(&self)
    }
}
