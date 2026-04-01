use std::collections::HashMap;
use std::path::Path;

use yscv_tensor::Tensor;

use crate::ModelError;

/// Lightweight safetensors-compatible weight file format.
///
/// File format: JSON header (length-prefixed) + raw f32 data.
/// Header: `{ "tensor_name": { "shape": [d0, d1, ...], "offset": N, "length": M }, ... }`
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct TensorMeta {
    shape: Vec<usize>,
    offset: usize,
    length: usize,
}

/// Saves a named set of tensors to a binary file (safetensors-like format).
///
/// Format: [8 bytes: header_len as u64 LE] [header JSON bytes] [raw f32 data].
pub fn save_weights(path: &Path, tensors: &HashMap<String, Tensor>) -> Result<(), ModelError> {
    let mut meta_map: HashMap<String, TensorMeta> = HashMap::new();
    let mut raw_data: Vec<u8> = Vec::new();

    let mut names: Vec<&String> = tensors.keys().collect();
    names.sort();

    for name in &names {
        let t = &tensors[*name];
        let offset = raw_data.len();
        let bytes = f32_slice_to_bytes(t.data());
        let byte_len = bytes.len();
        raw_data.extend_from_slice(&bytes);
        meta_map.insert(
            (*name).clone(),
            TensorMeta {
                shape: t.shape().to_vec(),
                offset,
                length: byte_len,
            },
        );
    }

    let header_json =
        serde_json::to_string(&meta_map).map_err(|e| ModelError::CheckpointSerialization {
            message: e.to_string(),
        })?;
    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;

    let mut file_data = Vec::new();
    file_data.extend_from_slice(&header_len.to_le_bytes());
    file_data.extend_from_slice(header_bytes);
    file_data.extend_from_slice(&raw_data);

    std::fs::write(path, &file_data).map_err(|e| ModelError::DatasetLoadIo {
        path: path.display().to_string(),
        message: e.to_string(),
    })
}

/// Loads named tensors from a binary weight file.
///
/// Reads the entire file into memory. For very large models (>RAM),
/// consider using memory-mapped I/O or streaming instead.
pub fn load_weights(path: &Path) -> Result<HashMap<String, Tensor>, ModelError> {
    let file_data = std::fs::read(path).map_err(|e| ModelError::DatasetLoadIo {
        path: path.display().to_string(),
        message: e.to_string(),
    })?;

    if file_data.len() < 8 {
        return Err(ModelError::CheckpointSerialization {
            message: "weight file too small".into(),
        });
    }

    let header_len = u64::from_le_bytes(file_data[..8].try_into().expect("8-byte slice")) as usize;
    if file_data.len() < 8 + header_len {
        return Err(ModelError::CheckpointSerialization {
            message: "weight file header truncated".into(),
        });
    }

    let header_str = std::str::from_utf8(&file_data[8..8 + header_len]).map_err(|e| {
        ModelError::CheckpointSerialization {
            message: e.to_string(),
        }
    })?;
    let meta_map: HashMap<String, TensorMeta> =
        serde_json::from_str(header_str).map_err(|e| ModelError::CheckpointSerialization {
            message: e.to_string(),
        })?;

    let data_start = 8 + header_len;
    let raw = &file_data[data_start..];

    let mut tensors = HashMap::new();
    for (name, meta) in &meta_map {
        if meta.offset + meta.length > raw.len() {
            return Err(ModelError::CheckpointSerialization {
                message: format!("tensor '{name}' data out of bounds"),
            });
        }
        let bytes = &raw[meta.offset..meta.offset + meta.length];
        let f32_data = bytes_to_f32_vec(bytes);
        let t = Tensor::from_vec(meta.shape.clone(), f32_data)?;
        tensors.insert(name.clone(), t);
    }

    Ok(tensors)
}

/// Lists tensor names and shapes from a weight file without loading data.
pub fn inspect_weights(path: &Path) -> Result<HashMap<String, Vec<usize>>, ModelError> {
    let file_data = std::fs::read(path).map_err(|e| ModelError::DatasetLoadIo {
        path: path.display().to_string(),
        message: e.to_string(),
    })?;
    if file_data.len() < 8 {
        return Err(ModelError::CheckpointSerialization {
            message: "weight file too small".into(),
        });
    }
    let header_len = u64::from_le_bytes(file_data[..8].try_into().expect("8-byte slice")) as usize;
    let header_str = std::str::from_utf8(&file_data[8..8 + header_len]).map_err(|e| {
        ModelError::CheckpointSerialization {
            message: e.to_string(),
        }
    })?;
    let meta_map: HashMap<String, TensorMeta> =
        serde_json::from_str(header_str).map_err(|e| ModelError::CheckpointSerialization {
            message: e.to_string(),
        })?;
    Ok(meta_map.into_iter().map(|(k, v)| (k, v.shape)).collect())
}

fn f32_slice_to_bytes(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

fn bytes_to_f32_vec(data: &[u8]) -> Vec<f32> {
    assert!(
        data.len().is_multiple_of(4),
        "byte slice length must be multiple of 4"
    );
    data.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}
