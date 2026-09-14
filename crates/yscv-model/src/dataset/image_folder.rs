use image::ImageReader;
use std::fs;
use std::path::{Path, PathBuf};
use yscv_imgproc::resize_nearest;
use yscv_tensor::Tensor;

use crate::ModelError;

use super::helpers::build_supervised_dataset_from_flat_values;
use super::types::SupervisedDataset;

/// Configuration for loading supervised image-folder datasets.
///
/// Expected directory layout:
/// `root/<class_name>/<image_file>`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImageFolderTargetMode {
    #[default]
    ClassIndex,
    OneHot,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SupervisedImageFolderConfig {
    output_height: usize,
    output_width: usize,
    target_mode: ImageFolderTargetMode,
    allowed_extensions: Vec<String>,
}

impl SupervisedImageFolderConfig {
    pub fn new(output_height: usize, output_width: usize) -> Result<Self, ModelError> {
        if output_height == 0 {
            return Err(ModelError::InvalidDatasetAdapterShape {
                field: "output_height",
                shape: vec![output_height],
                message: "output_height must be > 0".to_string(),
            });
        }
        if output_width == 0 {
            return Err(ModelError::InvalidDatasetAdapterShape {
                field: "output_width",
                shape: vec![output_width],
                message: "output_width must be > 0".to_string(),
            });
        }
        Ok(Self {
            output_height,
            output_width,
            target_mode: ImageFolderTargetMode::ClassIndex,
            allowed_extensions: default_image_folder_extensions(),
        })
    }

    pub const fn output_height(&self) -> usize {
        self.output_height
    }

    pub const fn output_width(&self) -> usize {
        self.output_width
    }

    pub const fn with_target_mode(mut self, target_mode: ImageFolderTargetMode) -> Self {
        self.target_mode = target_mode;
        self
    }

    pub const fn target_mode(&self) -> ImageFolderTargetMode {
        self.target_mode
    }

    pub fn with_allowed_extensions(
        mut self,
        allowed_extensions: Vec<String>,
    ) -> Result<Self, ModelError> {
        self.allowed_extensions = normalize_image_extensions(allowed_extensions)?;
        Ok(self)
    }

    pub fn allowed_extensions(&self) -> &[String] {
        &self.allowed_extensions
    }
}

/// Result payload for image-folder dataset loading with explicit class mapping.
#[derive(Debug, Clone, PartialEq)]
pub struct SupervisedImageFolderLoadResult {
    pub dataset: SupervisedDataset,
    pub class_names: Vec<String>,
}

/// Loads supervised training samples from an image-folder classification tree.
///
/// Directory contract:
/// - direct child directories under `root` are class buckets,
/// - class index is assigned by deterministic lexicographic directory ordering,
/// - supported image extensions are `jpg`, `jpeg`, and `png` (case-insensitive),
/// - non-image files are ignored.
///
/// Runtime mapping:
/// - inputs shape: `[N, output_height, output_width, 3]`,
/// - targets shape depends on `config.target_mode()`:
///   - `ClassIndex`: `[N, 1]` with scalar class ids,
///   - `OneHot`: `[N, class_count]` one-hot vectors.
pub fn load_supervised_image_folder_dataset<P: AsRef<Path>>(
    root: P,
    config: &SupervisedImageFolderConfig,
) -> Result<SupervisedDataset, ModelError> {
    load_supervised_image_folder_dataset_with_classes(root, config).map(|loaded| loaded.dataset)
}

/// Loads supervised training samples from an image-folder classification tree and returns class mapping.
///
/// Class names are returned in deterministic lexicographic directory order
/// and correspond to class indices used in targets.
pub fn load_supervised_image_folder_dataset_with_classes<P: AsRef<Path>>(
    root: P,
    config: &SupervisedImageFolderConfig,
) -> Result<SupervisedImageFolderLoadResult, ModelError> {
    let root_ref = root.as_ref();
    let class_dirs = read_sorted_class_directories(root_ref)?;
    let class_names = class_dirs
        .iter()
        .map(|class_dir| class_name_from_path(class_dir))
        .collect::<Result<Vec<_>, _>>()?;
    let class_count = class_dirs.len();

    let mut input_values = Vec::new();
    let mut target_values = Vec::new();
    let mut sample_count = 0usize;

    for (class_id, class_dir) in class_dirs.iter().enumerate() {
        let image_files =
            read_sorted_supported_image_files(class_dir, config.allowed_extensions())?;
        for image_path in image_files {
            let image_tensor = load_image_as_normalized_rgb_tensor(
                &image_path,
                config.output_height(),
                config.output_width(),
            )?;
            input_values.extend_from_slice(image_tensor.data());
            append_image_folder_target(
                &mut target_values,
                class_id,
                class_count,
                config.target_mode(),
            )?;
            sample_count = sample_count.checked_add(1).ok_or_else(|| {
                ModelError::InvalidDatasetAdapterShape {
                    field: "sample_count",
                    shape: vec![sample_count],
                    message: "sample count overflow".to_string(),
                }
            })?;
        }
    }

    if sample_count == 0 {
        return Err(ModelError::EmptyDataset);
    }

    let target_shape = match config.target_mode() {
        ImageFolderTargetMode::ClassIndex => vec![1],
        ImageFolderTargetMode::OneHot => vec![class_count],
    };

    let dataset = build_supervised_dataset_from_flat_values(
        &[config.output_height(), config.output_width(), 3],
        &target_shape,
        sample_count,
        input_values,
        target_values,
    )?;

    Ok(SupervisedImageFolderLoadResult {
        dataset,
        class_names,
    })
}

fn class_name_from_path(class_dir: &Path) -> Result<String, ModelError> {
    class_dir
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .ok_or_else(|| ModelError::InvalidDatasetAdapterShape {
            field: "class_name",
            shape: Vec::new(),
            message: format!(
                "failed to infer class name from directory path {}",
                class_dir.display()
            ),
        })
}

fn append_image_folder_target(
    target_values: &mut Vec<f32>,
    class_id: usize,
    class_count: usize,
    target_mode: ImageFolderTargetMode,
) -> Result<(), ModelError> {
    match target_mode {
        ImageFolderTargetMode::ClassIndex => {
            target_values.push(class_id as f32);
            Ok(())
        }
        ImageFolderTargetMode::OneHot => {
            if class_count == 0 || class_id >= class_count {
                return Err(ModelError::InvalidDatasetAdapterShape {
                    field: "target_shape",
                    shape: vec![class_count],
                    message: "invalid one-hot class configuration".to_string(),
                });
            }
            let next_len = target_values
                .len()
                .checked_add(class_count)
                .ok_or_else(|| ModelError::InvalidDatasetAdapterShape {
                    field: "target_values",
                    shape: vec![target_values.len(), class_count],
                    message: "target vector length overflow".to_string(),
                })?;
            target_values.resize(next_len, 0.0);
            let target_start = next_len - class_count;
            target_values[target_start + class_id] = 1.0;
            Ok(())
        }
    }
}

fn read_sorted_class_directories(root: &Path) -> Result<Vec<PathBuf>, ModelError> {
    let entries = fs::read_dir(root).map_err(|error| ModelError::DatasetLoadIo {
        path: root.display().to_string(),
        message: error.to_string(),
    })?;

    let mut class_dirs = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| ModelError::DatasetLoadIo {
            path: root.display().to_string(),
            message: error.to_string(),
        })?;
        let file_type = entry
            .file_type()
            .map_err(|error| ModelError::DatasetLoadIo {
                path: entry.path().display().to_string(),
                message: error.to_string(),
            })?;
        if file_type.is_dir() {
            class_dirs.push(entry.path());
        }
    }

    class_dirs.sort_by(|left, right| {
        let left_name = left
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        let right_name = right
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        left_name.cmp(&right_name).then_with(|| left.cmp(right))
    });
    Ok(class_dirs)
}

fn read_sorted_supported_image_files(
    class_dir: &Path,
    allowed_extensions: &[String],
) -> Result<Vec<PathBuf>, ModelError> {
    let entries = fs::read_dir(class_dir).map_err(|error| ModelError::DatasetLoadIo {
        path: class_dir.display().to_string(),
        message: error.to_string(),
    })?;

    let mut image_files = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| ModelError::DatasetLoadIo {
            path: class_dir.display().to_string(),
            message: error.to_string(),
        })?;
        let file_type = entry
            .file_type()
            .map_err(|error| ModelError::DatasetLoadIo {
                path: entry.path().display().to_string(),
                message: error.to_string(),
            })?;
        if file_type.is_file() && has_supported_image_extension(&entry.path(), allowed_extensions) {
            image_files.push(entry.path());
        }
    }

    image_files.sort_by(|left, right| {
        let left_name = left
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        let right_name = right
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        left_name.cmp(&right_name).then_with(|| left.cmp(right))
    });
    Ok(image_files)
}

fn has_supported_image_extension(path: &Path, allowed_extensions: &[String]) -> bool {
    let Some(extension) = path.extension().and_then(|extension| extension.to_str()) else {
        return false;
    };
    allowed_extensions
        .iter()
        .any(|allowed| extension.eq_ignore_ascii_case(allowed))
}

fn default_image_folder_extensions() -> Vec<String> {
    ["jpg", "jpeg", "png", "bmp", "webp"]
        .into_iter()
        .map(str::to_string)
        .collect()
}

fn normalize_image_extensions(extensions: Vec<String>) -> Result<Vec<String>, ModelError> {
    if extensions.is_empty() {
        return Err(ModelError::InvalidImageFolderExtension {
            extension: "<list>".to_string(),
            message: "extension list must be non-empty".to_string(),
        });
    }

    let mut normalized = Vec::with_capacity(extensions.len());
    for extension in extensions {
        let trimmed = extension.trim();
        if trimmed.is_empty() {
            return Err(ModelError::InvalidImageFolderExtension {
                extension,
                message: "extension must be non-empty".to_string(),
            });
        }
        if trimmed.starts_with('.') {
            return Err(ModelError::InvalidImageFolderExtension {
                extension: trimmed.to_string(),
                message: "extension must not start with '.'".to_string(),
            });
        }
        if !trimmed
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        {
            return Err(ModelError::InvalidImageFolderExtension {
                extension: trimmed.to_string(),
                message: "extension must contain only ASCII letters, digits, or '_'".to_string(),
            });
        }
        let lowered = trimmed.to_ascii_lowercase();
        if !normalized.iter().any(|existing| existing == &lowered) {
            normalized.push(lowered);
        }
    }

    Ok(normalized)
}

pub(super) fn load_image_as_normalized_rgb_tensor(
    path: &Path,
    output_height: usize,
    output_width: usize,
) -> Result<Tensor, ModelError> {
    let path_string = path.display().to_string();
    let decoded = ImageReader::open(path)
        .map_err(|error| ModelError::DatasetImageDecode {
            path: path_string.clone(),
            message: error.to_string(),
        })?
        .decode()
        .map_err(|error| ModelError::DatasetImageDecode {
            path: path_string.clone(),
            message: error.to_string(),
        })?;
    let rgb = decoded.to_rgb8();
    let (width, height) = rgb.dimensions();

    let image_height = height as usize;
    let image_width = width as usize;
    let normalized = rgb
        .as_raw()
        .iter()
        .map(|value| (*value as f32) * (1.0 / 255.0))
        .collect::<Vec<_>>();
    let image_tensor = Tensor::from_vec(vec![image_height, image_width, 3], normalized)?;

    if image_height == output_height && image_width == output_width {
        Ok(image_tensor)
    } else {
        resize_nearest(&image_tensor, output_height, output_width).map_err(Into::into)
    }
}
