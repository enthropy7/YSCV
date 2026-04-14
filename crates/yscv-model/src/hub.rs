//! Pretrained model hub: download and cache weights from remote sources.
//!
//! Uses `curl` via `std::process::Command` to avoid adding heavy HTTP
//! dependencies.  Downloaded `.safetensors` files are cached under
//! `$RUSTCV_CACHE_DIR` (or `~/.yscv/models/` by default) and validated
//! by expected file size.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::ModelError;
use crate::load_state_dict;
use yscv_tensor::Tensor;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Registry entry for a pretrained model.
#[derive(Debug, Clone)]
pub struct HubEntry {
    /// URL to download the `.safetensors` file.
    pub url: String,
    /// Expected file size in bytes (used for validation after download).
    pub expected_size: u64,
    /// Local filename inside the cache directory.
    pub filename: String,
}

/// Model hub for downloading and caching pretrained weights.
pub struct ModelHub {
    cache_dir: PathBuf,
    registry: HashMap<String, HubEntry>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns the default cache directory for downloaded model weights.
///
/// Uses `$RUSTCV_CACHE_DIR` if set, otherwise `~/.yscv/models/`.
pub fn default_cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("RUSTCV_CACHE_DIR") {
        return PathBuf::from(dir);
    }
    // Fall back to ~/.yscv/models/
    let home = std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));
    home.join(".yscv").join("models")
}

// ---------------------------------------------------------------------------
// Registry population
// ---------------------------------------------------------------------------

fn build_registry() -> HashMap<String, HubEntry> {
    let mut m = HashMap::new();

    m.insert(
        "resnet18".into(),
        HubEntry {
            url: "https://huggingface.co/timm/resnet18.a1_in1k/resolve/main/model.safetensors"
                .into(),
            expected_size: 46_830_408,
            filename: "resnet18.safetensors".into(),
        },
    );
    m.insert(
        "resnet34".into(),
        HubEntry {
            url: "https://huggingface.co/timm/resnet34.a1_in1k/resolve/main/model.safetensors"
                .into(),
            expected_size: 87_338_584,
            filename: "resnet34.safetensors".into(),
        },
    );
    m.insert(
        "resnet50".into(),
        HubEntry {
            url: "https://huggingface.co/timm/resnet50.a1_in1k/resolve/main/model.safetensors"
                .into(),
            expected_size: 102_170_688,
            filename: "resnet50.safetensors".into(),
        },
    );
    m.insert(
        "resnet101".into(),
        HubEntry {
            url: "https://huggingface.co/timm/resnet101.a1_in1k/resolve/main/model.safetensors"
                .into(),
            expected_size: 178_834_240,
            filename: "resnet101.safetensors".into(),
        },
    );
    m.insert(
        "vgg16".into(),
        HubEntry {
            url: "https://huggingface.co/timm/vgg16.tv_in1k/resolve/main/model.safetensors".into(),
            expected_size: 553_507_904,
            filename: "vgg16.safetensors".into(),
        },
    );
    m.insert(
        "vgg19".into(),
        HubEntry {
            url: "https://huggingface.co/timm/vgg19.tv_in1k/resolve/main/model.safetensors".into(),
            expected_size: 574_879_552,
            filename: "vgg19.safetensors".into(),
        },
    );
    m.insert(
        "mobilenet_v2".into(),
        HubEntry {
            url:
                "https://huggingface.co/timm/mobilenetv2_100.ra_in1k/resolve/main/model.safetensors"
                    .into(),
            expected_size: 14_214_848,
            filename: "mobilenet_v2.safetensors".into(),
        },
    );
    m.insert(
        "efficientnet_b0".into(),
        HubEntry {
            url:
                "https://huggingface.co/timm/efficientnet_b0.ra_in1k/resolve/main/model.safetensors"
                    .into(),
            expected_size: 21_388_928,
            filename: "efficientnet_b0.safetensors".into(),
        },
    );
    m.insert(
        "alexnet".into(),
        HubEntry {
            url: "https://huggingface.co/pytorch/alexnet/resolve/main/model.safetensors".into(),
            expected_size: 244_408_336,
            filename: "alexnet.safetensors".into(),
        },
    );
    m.insert(
        "clip_vit_b32".into(),
        HubEntry {
            url:
                "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/model.safetensors"
                    .into(),
            expected_size: 605_552_640,
            filename: "clip_vit_b32.safetensors".into(),
        },
    );
    m.insert(
        "dinov2_vit_s14".into(),
        HubEntry {
            url: "https://huggingface.co/facebook/dinov2-small/resolve/main/model.safetensors"
                .into(),
            expected_size: 88_222_464,
            filename: "dinov2_vit_s14.safetensors".into(),
        },
    );
    m.insert(
        "whisper_tiny".into(),
        HubEntry {
            url: "https://huggingface.co/openai/whisper-tiny/resolve/main/model.safetensors".into(),
            expected_size: 151_127_872,
            filename: "whisper_tiny.safetensors".into(),
        },
    );
    m.insert(
        "sam_vit_b".into(),
        HubEntry {
            url: "https://huggingface.co/facebook/sam-vit-base/resolve/main/model.safetensors"
                .into(),
            expected_size: 375_042_048,
            filename: "sam_vit_b.safetensors".into(),
        },
    );

    m
}

// ---------------------------------------------------------------------------
// ModelHub implementation
// ---------------------------------------------------------------------------

impl ModelHub {
    /// Creates a new hub with the default cache directory and built-in
    /// registry of known pretrained models.
    pub fn new() -> Self {
        Self {
            cache_dir: default_cache_dir(),
            registry: build_registry(),
        }
    }

    /// Returns the cache directory path.
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Returns a reference to the internal registry.
    pub fn registry(&self) -> &HashMap<String, HubEntry> {
        &self.registry
    }

    /// Ensures the weight file for `name` is present in the local cache,
    /// downloading it via `curl` if necessary.
    ///
    /// Returns the path to the cached file on success.
    pub fn download_if_missing(&self, name: &str) -> Result<PathBuf, ModelError> {
        let entry = self
            .registry
            .get(name)
            .ok_or_else(|| ModelError::DownloadFailed {
                url: name.to_string(),
                reason: format!("model '{name}' is not in the hub registry"),
            })?;

        let dest = self.cache_dir.join(&entry.filename);

        // Already cached — validate size and return.
        if dest.is_file() {
            validate_file_size(&dest, entry.expected_size)?;
            return Ok(dest);
        }

        // Ensure cache directory exists.
        std::fs::create_dir_all(&self.cache_dir).map_err(|e| ModelError::DownloadFailed {
            url: entry.url.clone(),
            reason: format!(
                "failed to create cache dir {}: {e}",
                self.cache_dir.display()
            ),
        })?;

        // Download with curl.
        let output = Command::new("curl")
            .args(["-fSL", "-o"])
            .arg(&dest)
            .arg(&entry.url)
            .output()
            .map_err(|e| ModelError::DownloadFailed {
                url: entry.url.clone(),
                reason: format!("failed to run curl: {e}"),
            })?;

        if !output.status.success() {
            // Clean up partial file.
            let _ = std::fs::remove_file(&dest);
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(ModelError::DownloadFailed {
                url: entry.url.clone(),
                reason: format!("curl exited with {}: {stderr}", output.status),
            });
        }

        validate_file_size(&dest, entry.expected_size)?;
        Ok(dest)
    }

    /// Downloads (if needed) and loads all tensors from the safetensors
    /// weight file for the given model name.
    pub fn load_weights(&self, name: &str) -> Result<HashMap<String, Tensor>, ModelError> {
        let path = self.download_if_missing(name)?;
        load_state_dict(&path)
    }
}

impl Default for ModelHub {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn validate_file_size(path: &Path, expected: u64) -> Result<(), ModelError> {
    let meta = std::fs::metadata(path).map_err(|e| ModelError::DownloadFailed {
        url: path.display().to_string(),
        reason: format!("cannot stat downloaded file: {e}"),
    })?;
    let actual = meta.len();
    if actual != expected {
        return Err(ModelError::DownloadFailed {
            url: path.display().to_string(),
            reason: format!("file size mismatch: expected {expected} bytes, got {actual} bytes"),
        });
    }
    Ok(())
}
