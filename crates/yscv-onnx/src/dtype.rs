/// ONNX tensor element type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnnxDtype {
    Float32,
    Float16,
    Int8,
    UInt8,
    Int32,
    Int64,
    Bool,
}

impl OnnxDtype {
    /// Convert ONNX protobuf data type code to `OnnxDtype`.
    ///
    /// See <https://onnx.ai/onnx/repo-docs/IR.html#standard-data-types> for
    /// the canonical mapping.
    pub const fn from_onnx_type(t: i32) -> Option<Self> {
        match t {
            1 => Some(Self::Float32),
            10 => Some(Self::Float16),
            3 => Some(Self::Int8),
            2 => Some(Self::UInt8),
            6 => Some(Self::Int32),
            7 => Some(Self::Int64),
            9 => Some(Self::Bool),
            _ => None,
        }
    }

    /// Number of bytes occupied by a single element of this dtype.
    pub const fn byte_size(&self) -> usize {
        match self {
            Self::Float32 | Self::Int32 => 4,
            Self::Float16 => 2,
            Self::Int8 | Self::UInt8 | Self::Bool => 1,
            Self::Int64 => 8,
        }
    }
}

/// Typed tensor data for ONNX runtime.
#[derive(Debug, Clone)]
pub enum OnnxTensorData {
    Float32(Vec<f32>),
    Int8(Vec<i8>),
    UInt8(Vec<u8>),
    Int32(Vec<i32>),
    Int64(Vec<i64>),
}

impl OnnxTensorData {
    /// Convert to `Vec<f32>` for computation.
    pub fn to_f32(&self) -> Vec<f32> {
        match self {
            Self::Float32(v) => v.clone(),
            Self::Int8(v) => v.iter().map(|&x| x as f32).collect(),
            Self::UInt8(v) => v.iter().map(|&x| x as f32).collect(),
            Self::Int32(v) => v.iter().map(|&x| x as f32).collect(),
            Self::Int64(v) => v.iter().map(|&x| x as f32).collect(),
        }
    }

    /// Quantize f32 data to int8 with given scale and zero_point.
    ///
    /// Each element is computed as:
    ///   quantized = clamp(round(value / scale) + zero_point, -128, 127)
    pub fn quantize_to_int8(data: &[f32], scale: f32, zero_point: i8) -> Self {
        let quantized: Vec<i8> = data
            .iter()
            .map(|&v| ((v / scale).round() as i32 + zero_point as i32).clamp(-128, 127) as i8)
            .collect();
        Self::Int8(quantized)
    }

    /// Dequantize int8 data back to f32.
    ///
    /// Each element is computed as:
    ///   value = (quantized - zero_point) * scale
    pub fn dequantize_int8(data: &[i8], scale: f32, zero_point: i8) -> Vec<f32> {
        data.iter()
            .map(|&v| (v as f32 - zero_point as f32) * scale)
            .collect()
    }

    /// Number of elements in this tensor data.
    pub const fn len(&self) -> usize {
        match self {
            Self::Float32(v) => v.len(),
            Self::Int8(v) => v.len(),
            Self::UInt8(v) => v.len(),
            Self::Int32(v) => v.len(),
            Self::Int64(v) => v.len(),
        }
    }

    /// Returns `true` if the tensor data is empty.
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn onnx_dtype_from_protobuf() {
        assert_eq!(OnnxDtype::from_onnx_type(1), Some(OnnxDtype::Float32));
        assert_eq!(OnnxDtype::from_onnx_type(10), Some(OnnxDtype::Float16));
        assert_eq!(OnnxDtype::from_onnx_type(3), Some(OnnxDtype::Int8));
        assert_eq!(OnnxDtype::from_onnx_type(2), Some(OnnxDtype::UInt8));
        assert_eq!(OnnxDtype::from_onnx_type(6), Some(OnnxDtype::Int32));
        assert_eq!(OnnxDtype::from_onnx_type(7), Some(OnnxDtype::Int64));
        assert_eq!(OnnxDtype::from_onnx_type(9), Some(OnnxDtype::Bool));
        // Unknown type
        assert_eq!(OnnxDtype::from_onnx_type(0), None);
        assert_eq!(OnnxDtype::from_onnx_type(99), None);
    }

    #[test]
    fn onnx_dtype_byte_size() {
        assert_eq!(OnnxDtype::Float32.byte_size(), 4);
        assert_eq!(OnnxDtype::Float16.byte_size(), 2);
        assert_eq!(OnnxDtype::Int8.byte_size(), 1);
        assert_eq!(OnnxDtype::UInt8.byte_size(), 1);
        assert_eq!(OnnxDtype::Int32.byte_size(), 4);
        assert_eq!(OnnxDtype::Int64.byte_size(), 8);
        assert_eq!(OnnxDtype::Bool.byte_size(), 1);
    }

    #[test]
    fn quantize_dequantize_roundtrip() {
        let original = vec![0.0, 0.5, 1.0, -0.5, -1.0, 0.25, -0.25];
        let scale = 0.01f32;
        let zero_point = 0i8;

        let quantized = OnnxTensorData::quantize_to_int8(&original, scale, zero_point);
        let OnnxTensorData::Int8(ref int8_data) = quantized else {
            unreachable!("quantize_to_int8 always returns Int8");
        };

        let recovered = OnnxTensorData::dequantize_int8(int8_data, scale, zero_point);

        assert_eq!(original.len(), recovered.len());
        for (orig, rec) in original.iter().zip(recovered.iter()) {
            assert!(
                (orig - rec).abs() < scale + 1e-6,
                "roundtrip error too large: original={}, recovered={}",
                orig,
                rec,
            );
        }
    }

    #[test]
    fn onnx_tensor_data_to_f32() {
        // Float32
        let f32_data = OnnxTensorData::Float32(vec![1.0, 2.0, 3.0]);
        assert_eq!(f32_data.to_f32(), vec![1.0, 2.0, 3.0]);

        // Int8
        let i8_data = OnnxTensorData::Int8(vec![-128, 0, 127]);
        let result = i8_data.to_f32();
        assert_eq!(result, vec![-128.0, 0.0, 127.0]);

        // UInt8
        let u8_data = OnnxTensorData::UInt8(vec![0, 128, 255]);
        let result = u8_data.to_f32();
        assert_eq!(result, vec![0.0, 128.0, 255.0]);

        // Int32
        let i32_data = OnnxTensorData::Int32(vec![-100_000, 0, 100_000]);
        let result = i32_data.to_f32();
        assert_eq!(result, vec![-100_000.0, 0.0, 100_000.0]);

        // Int64
        let i64_data = OnnxTensorData::Int64(vec![-1, 0, 1]);
        let result = i64_data.to_f32();
        assert_eq!(result, vec![-1.0, 0.0, 1.0]);
    }
}
