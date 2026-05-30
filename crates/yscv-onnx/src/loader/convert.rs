//! ONNX protobuf decoding: TensorProto -> Tensor and AttributeProto
//! -> OnnxAttribute, with raw-bytes reinterpretation helpers.

use super::*;

pub(super) fn convert_tensor_proto(tp: &onnx::TensorProto) -> Result<Tensor, OnnxError> {
    let shape: Vec<usize> = tp.dims.iter().map(|&d| d as usize).collect();
    let expected_len: usize = if shape.is_empty() {
        1
    } else {
        shape.iter().product()
    };
    let data_type = tp.data_type.unwrap_or(0);

    let data = match data_type {
        // FLOAT = 1
        1 => {
            if !tp.float_data.is_empty() {
                tp.float_data.clone()
            } else if let Some(ref raw) = tp.raw_data {
                raw_bytes_to_f32(raw)
            } else {
                vec![0.0f32; expected_len]
            }
        }
        // UINT8 = 2
        2 => {
            if !tp.int32_data.is_empty() {
                tp.int32_data.iter().map(|&v| v as u8 as f32).collect()
            } else if let Some(ref raw) = tp.raw_data {
                raw.iter().map(|&v| v as f32).collect()
            } else {
                vec![0.0f32; expected_len]
            }
        }
        // INT8 = 3
        3 => {
            if !tp.int32_data.is_empty() {
                tp.int32_data.iter().map(|&v| v as i8 as f32).collect()
            } else if let Some(ref raw) = tp.raw_data {
                raw.iter().map(|&v| (v as i8) as f32).collect()
            } else {
                vec![0.0f32; expected_len]
            }
        }
        // DOUBLE = 11
        11 => {
            if !tp.double_data.is_empty() {
                tp.double_data.iter().map(|&d| d as f32).collect()
            } else if let Some(ref raw) = tp.raw_data {
                raw_bytes_to_f64_as_f32(raw)
            } else {
                vec![0.0f32; expected_len]
            }
        }
        // INT64 = 7
        7 => {
            if !tp.int64_data.is_empty() {
                tp.int64_data.iter().map(|&v| v as f32).collect()
            } else if let Some(ref raw) = tp.raw_data {
                raw_bytes_to_i64_as_f32(raw)
            } else {
                vec![0.0f32; expected_len]
            }
        }
        // INT32 = 6
        6 => {
            if !tp.int32_data.is_empty() {
                tp.int32_data.iter().map(|&v| v as f32).collect()
            } else if let Some(ref raw) = tp.raw_data {
                raw_bytes_to_i32_as_f32(raw)
            } else {
                vec![0.0f32; expected_len]
            }
        }
        other => {
            return Err(OnnxError::UnsupportedDataType { data_type: other });
        }
    };

    if data.len() != expected_len {
        return Err(OnnxError::InitializerShapeMismatch {
            name: tp.name.clone().unwrap_or_default(),
            expected: expected_len,
            got: data.len(),
        });
    }

    // Preserve 0-D scalar shapes: ONNX TensorProto with dims=[] is a 0-D
    // scalar, not a 1-D tensor.  Many graph patterns (Gather with scalar
    // indices → Unsqueeze → Concat for reshape targets) depend on correct
    // rank propagation.
    Tensor::from_vec(shape, data).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })
}

fn raw_bytes_to_f32(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn raw_bytes_to_f64_as_f32(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(8)
        .map(|c| {
            let v = f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
            v as f32
        })
        .collect()
}

fn raw_bytes_to_i64_as_f32(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(8)
        .map(|c| {
            let v = i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
            v as f32
        })
        .collect()
}

fn raw_bytes_to_i32_as_f32(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4)
        .map(|c| {
            let v = i32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            v as f32
        })
        .collect()
}

pub(super) fn convert_attribute(attr: &onnx::AttributeProto) -> Option<OnnxAttribute> {
    // Some exporter/toolchain combinations omit `AttributeProto.type` for
    // Constant nodes and rely on the populated value field (`t`, `ints`, ...).
    // Infer the type from payload presence when the enum tag is missing.
    let attr_type = attr.r#type.unwrap_or_else(|| {
        if attr.t.is_some() {
            4
        } else if attr.f.is_some() {
            1
        } else if attr.i.is_some() {
            2
        } else if attr.s.is_some() {
            3
        } else if !attr.floats.is_empty() {
            6
        } else if !attr.ints.is_empty() {
            7
        } else {
            0
        }
    });
    match attr_type {
        1 => Some(OnnxAttribute::Float(attr.f.unwrap_or(0.0))),
        2 => Some(OnnxAttribute::Int(attr.i.unwrap_or(0))),
        3 => {
            let s = attr
                .s
                .as_deref()
                .map(|b| String::from_utf8_lossy(b).to_string())
                .unwrap_or_default();
            Some(OnnxAttribute::String(s))
        }
        // TENSOR — used by Constant nodes to embed full tensor values
        4 => {
            let tp = attr.t.as_ref()?;
            convert_tensor_proto(tp).ok().map(OnnxAttribute::Tensor)
        }
        6 => Some(OnnxAttribute::Floats(attr.floats.clone())),
        7 => Some(OnnxAttribute::Ints(attr.ints.clone())),
        _ => None,
    }
}
