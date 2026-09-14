use std::fmt;
use std::sync::Arc;

use yscv_imgproc::{
    box_blur_3x3, flip_horizontal, flip_vertical, normalize, resize_nearest, rotate90_cw,
};
use yscv_tensor::{Tensor, TensorError};

use super::transform::Transform;
use crate::{ModelError, SupervisedDataset};

/// Per-sample image augmentations for rank-4 NHWC training tensors.
pub enum ImageAugmentationOp {
    /// Flip image across width axis with configured probability.
    HorizontalFlip { probability: f32 },
    /// Flip image across height axis with configured probability.
    VerticalFlip { probability: f32 },
    /// Rotate sample by random multiples of 90 degrees with configured probability.
    ///
    /// For square samples, rotation is sampled from {0, 90, 180, 270} degrees.
    /// For non-square samples, rotation is sampled from {0, 180} degrees to preserve shape.
    RandomRotate90 { probability: f32 },
    /// Add random uniform brightness delta in `[-max_delta, +max_delta]` and clamp to `[0, 1]`.
    BrightnessJitter { max_delta: f32 },
    /// Scale contrast around per-sample mean by factor in `[1-max_scale_delta, 1+max_scale_delta]`.
    ContrastJitter { max_scale_delta: f32 },
    /// Apply gamma correction with gamma sampled in `[1-max_gamma_delta, 1+max_gamma_delta]`.
    GammaJitter { max_gamma_delta: f32 },
    /// Add per-value Gaussian noise with configured standard deviation and clamp to `[0, 1]`.
    GaussianNoise { probability: f32, std_dev: f32 },
    /// Apply 3x3 box blur with configured probability.
    BoxBlur3x3 { probability: f32 },
    /// Crop a random window and resize it back to original sample size.
    RandomResizedCrop {
        probability: f32,
        min_scale: f32,
        max_scale: f32,
    },
    /// Apply random rectangular erasing with configured max size fractions and fill value.
    Cutout {
        probability: f32,
        max_height_fraction: f32,
        max_width_fraction: f32,
        fill_value: f32,
    },
    /// Per-channel normalization in HWC layout: `(x - mean[c]) / std[c]`.
    ChannelNormalize { mean: Vec<f32>, std: Vec<f32> },
    /// User-provided closure for custom augmentation logic.
    Custom(Arc<dyn Fn(&Tensor) -> Result<Tensor, ModelError> + Send + Sync>),
    /// Random crop from larger image (does not resize back; changes spatial dims).
    RandomCrop { height: usize, width: usize },
    /// Apply gaussian blur with the given square kernel size (must be odd and >= 1).
    GaussianBlur { kernel_size: usize },
}

impl fmt::Debug for ImageAugmentationOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HorizontalFlip { probability } => f
                .debug_struct("HorizontalFlip")
                .field("probability", probability)
                .finish(),
            Self::VerticalFlip { probability } => f
                .debug_struct("VerticalFlip")
                .field("probability", probability)
                .finish(),
            Self::RandomRotate90 { probability } => f
                .debug_struct("RandomRotate90")
                .field("probability", probability)
                .finish(),
            Self::BrightnessJitter { max_delta } => f
                .debug_struct("BrightnessJitter")
                .field("max_delta", max_delta)
                .finish(),
            Self::ContrastJitter { max_scale_delta } => f
                .debug_struct("ContrastJitter")
                .field("max_scale_delta", max_scale_delta)
                .finish(),
            Self::GammaJitter { max_gamma_delta } => f
                .debug_struct("GammaJitter")
                .field("max_gamma_delta", max_gamma_delta)
                .finish(),
            Self::GaussianNoise {
                probability,
                std_dev,
            } => f
                .debug_struct("GaussianNoise")
                .field("probability", probability)
                .field("std_dev", std_dev)
                .finish(),
            Self::BoxBlur3x3 { probability } => f
                .debug_struct("BoxBlur3x3")
                .field("probability", probability)
                .finish(),
            Self::RandomResizedCrop {
                probability,
                min_scale,
                max_scale,
            } => f
                .debug_struct("RandomResizedCrop")
                .field("probability", probability)
                .field("min_scale", min_scale)
                .field("max_scale", max_scale)
                .finish(),
            Self::Cutout {
                probability,
                max_height_fraction,
                max_width_fraction,
                fill_value,
            } => f
                .debug_struct("Cutout")
                .field("probability", probability)
                .field("max_height_fraction", max_height_fraction)
                .field("max_width_fraction", max_width_fraction)
                .field("fill_value", fill_value)
                .finish(),
            Self::ChannelNormalize { mean, std } => f
                .debug_struct("ChannelNormalize")
                .field("mean", mean)
                .field("std", std)
                .finish(),
            Self::Custom(_) => f.debug_tuple("Custom").field(&"<closure>").finish(),
            Self::RandomCrop { height, width } => f
                .debug_struct("RandomCrop")
                .field("height", height)
                .field("width", width)
                .finish(),
            Self::GaussianBlur { kernel_size } => f
                .debug_struct("GaussianBlur")
                .field("kernel_size", kernel_size)
                .finish(),
        }
    }
}

impl Clone for ImageAugmentationOp {
    fn clone(&self) -> Self {
        match self {
            Self::HorizontalFlip { probability } => Self::HorizontalFlip {
                probability: *probability,
            },
            Self::VerticalFlip { probability } => Self::VerticalFlip {
                probability: *probability,
            },
            Self::RandomRotate90 { probability } => Self::RandomRotate90 {
                probability: *probability,
            },
            Self::BrightnessJitter { max_delta } => Self::BrightnessJitter {
                max_delta: *max_delta,
            },
            Self::ContrastJitter { max_scale_delta } => Self::ContrastJitter {
                max_scale_delta: *max_scale_delta,
            },
            Self::GammaJitter { max_gamma_delta } => Self::GammaJitter {
                max_gamma_delta: *max_gamma_delta,
            },
            Self::GaussianNoise {
                probability,
                std_dev,
            } => Self::GaussianNoise {
                probability: *probability,
                std_dev: *std_dev,
            },
            Self::BoxBlur3x3 { probability } => Self::BoxBlur3x3 {
                probability: *probability,
            },
            Self::RandomResizedCrop {
                probability,
                min_scale,
                max_scale,
            } => Self::RandomResizedCrop {
                probability: *probability,
                min_scale: *min_scale,
                max_scale: *max_scale,
            },
            Self::Cutout {
                probability,
                max_height_fraction,
                max_width_fraction,
                fill_value,
            } => Self::Cutout {
                probability: *probability,
                max_height_fraction: *max_height_fraction,
                max_width_fraction: *max_width_fraction,
                fill_value: *fill_value,
            },
            Self::ChannelNormalize { mean, std } => Self::ChannelNormalize {
                mean: mean.clone(),
                std: std.clone(),
            },
            Self::Custom(f) => Self::Custom(Arc::clone(f)),
            Self::RandomCrop { height, width } => Self::RandomCrop {
                height: *height,
                width: *width,
            },
            Self::GaussianBlur { kernel_size } => Self::GaussianBlur {
                kernel_size: *kernel_size,
            },
        }
    }
}

impl PartialEq for ImageAugmentationOp {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::HorizontalFlip { probability: a }, Self::HorizontalFlip { probability: b }) => {
                a == b
            }
            (Self::VerticalFlip { probability: a }, Self::VerticalFlip { probability: b }) => {
                a == b
            }
            (Self::RandomRotate90 { probability: a }, Self::RandomRotate90 { probability: b }) => {
                a == b
            }
            (Self::BrightnessJitter { max_delta: a }, Self::BrightnessJitter { max_delta: b }) => {
                a == b
            }
            (
                Self::ContrastJitter { max_scale_delta: a },
                Self::ContrastJitter { max_scale_delta: b },
            ) => a == b,
            (
                Self::GammaJitter { max_gamma_delta: a },
                Self::GammaJitter { max_gamma_delta: b },
            ) => a == b,
            (
                Self::GaussianNoise {
                    probability: p1,
                    std_dev: s1,
                },
                Self::GaussianNoise {
                    probability: p2,
                    std_dev: s2,
                },
            ) => p1 == p2 && s1 == s2,
            (Self::BoxBlur3x3 { probability: a }, Self::BoxBlur3x3 { probability: b }) => a == b,
            (
                Self::RandomResizedCrop {
                    probability: p1,
                    min_scale: mn1,
                    max_scale: mx1,
                },
                Self::RandomResizedCrop {
                    probability: p2,
                    min_scale: mn2,
                    max_scale: mx2,
                },
            ) => p1 == p2 && mn1 == mn2 && mx1 == mx2,
            (
                Self::Cutout {
                    probability: p1,
                    max_height_fraction: h1,
                    max_width_fraction: w1,
                    fill_value: f1,
                },
                Self::Cutout {
                    probability: p2,
                    max_height_fraction: h2,
                    max_width_fraction: w2,
                    fill_value: f2,
                },
            ) => p1 == p2 && h1 == h2 && w1 == w2 && f1 == f2,
            (
                Self::ChannelNormalize { mean: m1, std: s1 },
                Self::ChannelNormalize { mean: m2, std: s2 },
            ) => m1 == m2 && s1 == s2,
            (Self::Custom(_), Self::Custom(_)) => false, // closures cannot be compared
            (
                Self::RandomCrop {
                    height: h1,
                    width: w1,
                },
                Self::RandomCrop {
                    height: h2,
                    width: w2,
                },
            ) => h1 == h2 && w1 == w2,
            (Self::GaussianBlur { kernel_size: a }, Self::GaussianBlur { kernel_size: b }) => {
                a == b
            }
            _ => false,
        }
    }
}

impl ImageAugmentationOp {
    fn validate(&self) -> Result<(), ModelError> {
        match self {
            Self::HorizontalFlip { probability } => {
                validate_probability("horizontal_flip", *probability)
            }
            Self::VerticalFlip { probability } => {
                validate_probability("vertical_flip", *probability)
            }
            Self::RandomRotate90 { probability } => {
                validate_probability("random_rotate90", *probability)
            }
            Self::BrightnessJitter { max_delta } => {
                if !max_delta.is_finite() || *max_delta < 0.0 {
                    return Err(ModelError::InvalidAugmentationArgument {
                        operation: "brightness_jitter",
                        message: format!("max_delta must be finite and >= 0, got {max_delta}"),
                    });
                }
                Ok(())
            }
            Self::ContrastJitter { max_scale_delta } => {
                if !max_scale_delta.is_finite() || *max_scale_delta < 0.0 {
                    return Err(ModelError::InvalidAugmentationArgument {
                        operation: "contrast_jitter",
                        message: format!(
                            "max_scale_delta must be finite and >= 0, got {max_scale_delta}"
                        ),
                    });
                }
                Ok(())
            }
            Self::GammaJitter { max_gamma_delta } => {
                if !max_gamma_delta.is_finite() || *max_gamma_delta < 0.0 {
                    return Err(ModelError::InvalidAugmentationArgument {
                        operation: "gamma_jitter",
                        message: format!(
                            "max_gamma_delta must be finite and >= 0, got {max_gamma_delta}"
                        ),
                    });
                }
                Ok(())
            }
            Self::GaussianNoise {
                probability,
                std_dev,
            } => {
                validate_probability("gaussian_noise", *probability)?;
                if !std_dev.is_finite() || *std_dev < 0.0 {
                    return Err(ModelError::InvalidAugmentationArgument {
                        operation: "gaussian_noise",
                        message: format!("std_dev must be finite and >= 0, got {std_dev}"),
                    });
                }
                Ok(())
            }
            Self::BoxBlur3x3 { probability } => validate_probability("box_blur_3x3", *probability),
            Self::RandomResizedCrop {
                probability,
                min_scale,
                max_scale,
            } => {
                validate_probability("random_resized_crop", *probability)?;
                validate_fraction("random_resized_crop", "min_scale", *min_scale)?;
                validate_fraction("random_resized_crop", "max_scale", *max_scale)?;
                if min_scale > max_scale {
                    return Err(ModelError::InvalidAugmentationArgument {
                        operation: "random_resized_crop",
                        message: format!(
                            "min_scale must be <= max_scale, got min_scale={min_scale}, max_scale={max_scale}"
                        ),
                    });
                }
                Ok(())
            }
            Self::Cutout {
                probability,
                max_height_fraction,
                max_width_fraction,
                fill_value,
            } => {
                validate_probability("cutout", *probability)?;
                validate_fraction("cutout", "max_height_fraction", *max_height_fraction)?;
                validate_fraction("cutout", "max_width_fraction", *max_width_fraction)?;
                if !fill_value.is_finite() {
                    return Err(ModelError::InvalidAugmentationArgument {
                        operation: "cutout",
                        message: format!("fill_value must be finite, got {fill_value}"),
                    });
                }
                Ok(())
            }
            Self::ChannelNormalize { mean, std } => {
                if mean.is_empty() || std.is_empty() {
                    return Err(ModelError::InvalidAugmentationArgument {
                        operation: "channel_normalize",
                        message: "mean/std must be non-empty".to_string(),
                    });
                }
                if mean.len() != std.len() {
                    return Err(ModelError::InvalidAugmentationArgument {
                        operation: "channel_normalize",
                        message: format!(
                            "mean/std length mismatch: mean_len={}, std_len={}",
                            mean.len(),
                            std.len()
                        ),
                    });
                }
                for (channel, mean_value) in mean.iter().enumerate() {
                    if !mean_value.is_finite() {
                        return Err(ModelError::InvalidAugmentationArgument {
                            operation: "channel_normalize",
                            message: format!("mean[{channel}] must be finite"),
                        });
                    }
                }
                for (channel, std_value) in std.iter().enumerate() {
                    if !std_value.is_finite() || *std_value <= 0.0 {
                        return Err(ModelError::InvalidAugmentationArgument {
                            operation: "channel_normalize",
                            message: format!("std[{channel}] must be finite and > 0"),
                        });
                    }
                }
                Ok(())
            }
            Self::Custom(_) => Ok(()),
            Self::RandomCrop { height, width } => {
                if *height == 0 || *width == 0 {
                    return Err(ModelError::InvalidAugmentationArgument {
                        operation: "random_crop",
                        message: format!(
                            "height and width must be > 0, got height={height}, width={width}"
                        ),
                    });
                }
                Ok(())
            }
            Self::GaussianBlur { kernel_size } => {
                if *kernel_size == 0 || *kernel_size % 2 == 0 {
                    return Err(ModelError::InvalidAugmentationArgument {
                        operation: "gaussian_blur",
                        message: format!("kernel_size must be odd and >= 1, got {kernel_size}"),
                    });
                }
                Ok(())
            }
        }
    }

    /// Create an augmentation op from any Transform implementation.
    pub fn from_transform<T: Transform + Send + Sync + 'static>(t: T) -> Self {
        Self::Custom(Arc::new(move |input| t.apply(input)))
    }
}

/// Ordered per-sample augmentation pipeline for NHWC mini-batch data.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageAugmentationPipeline {
    ops: Vec<ImageAugmentationOp>,
}

impl ImageAugmentationPipeline {
    pub fn new(ops: Vec<ImageAugmentationOp>) -> Result<Self, ModelError> {
        for op in &ops {
            op.validate()?;
        }
        Ok(Self { ops })
    }

    pub fn ops(&self) -> &[ImageAugmentationOp] {
        &self.ops
    }

    pub fn apply_nhwc(&self, inputs: &Tensor, seed: u64) -> Result<Tensor, ModelError> {
        if inputs.rank() != 4 {
            return Err(ModelError::InvalidAugmentationInputShape {
                got: inputs.shape().to_vec(),
            });
        }

        let shape = inputs.shape();
        let sample_count = shape[0];
        let sample_len = shape[1..].iter().try_fold(1usize, |acc, dim| {
            acc.checked_mul(*dim)
                .ok_or_else(|| TensorError::SizeOverflow {
                    shape: shape.to_vec(),
                })
        })?;

        let mut out = Vec::with_capacity(inputs.data().len());
        let mut rng = LcgRng::new(seed);
        let mut out_sample_shape: Option<Vec<usize>> = None;

        for sample_idx in 0..sample_count {
            let start =
                sample_idx
                    .checked_mul(sample_len)
                    .ok_or_else(|| TensorError::SizeOverflow {
                        shape: shape.to_vec(),
                    })?;
            let end = start
                .checked_add(sample_len)
                .ok_or_else(|| TensorError::SizeOverflow {
                    shape: shape.to_vec(),
                })?;

            let mut sample =
                Tensor::from_vec(shape[1..].to_vec(), inputs.data()[start..end].to_vec())?;
            for op in &self.ops {
                sample = apply_op(sample, op, &mut rng)?;
            }
            // Record the expected sample shape from the first sample and verify
            // all subsequent samples match (they may differ from the original
            // input shape when shape-changing ops like RandomCrop are used).
            if sample_idx == 0 {
                out_sample_shape = Some(sample.shape().to_vec());
            } else if let Some(ref expected) = out_sample_shape
                && sample.shape() != expected.as_slice()
            {
                return Err(ModelError::InvalidAugmentationArgument {
                    operation: "pipeline",
                    message: format!(
                        "augmentation produced inconsistent sample shapes: first={:?}, sample[{sample_idx}]={:?}",
                        expected,
                        sample.shape()
                    ),
                });
            }
            out.extend_from_slice(sample.data());
        }

        let final_sample_shape = out_sample_shape.unwrap_or_else(|| shape[1..].to_vec());
        let mut final_shape = vec![sample_count];
        final_shape.extend_from_slice(&final_sample_shape);
        Tensor::from_vec(final_shape, out).map_err(Into::into)
    }
}

impl SupervisedDataset {
    /// Returns a new dataset with NHWC image augmentations applied to `inputs`.
    pub fn augment_nhwc(
        &self,
        pipeline: &ImageAugmentationPipeline,
        seed: u64,
    ) -> Result<Self, ModelError> {
        let augmented_inputs = pipeline.apply_nhwc(self.inputs(), seed)?;
        Self::new(augmented_inputs, self.targets().clone())
    }
}

fn apply_op(
    input: Tensor,
    op: &ImageAugmentationOp,
    rng: &mut LcgRng,
) -> Result<Tensor, ModelError> {
    match op {
        ImageAugmentationOp::HorizontalFlip { probability } => {
            if should_apply(*probability, rng) {
                flip_horizontal(&input).map_err(Into::into)
            } else {
                Ok(input)
            }
        }
        ImageAugmentationOp::VerticalFlip { probability } => {
            if should_apply(*probability, rng) {
                flip_vertical(&input).map_err(Into::into)
            } else {
                Ok(input)
            }
        }
        ImageAugmentationOp::RandomRotate90 { probability } => {
            if should_apply(*probability, rng) {
                apply_random_rotate90(input, rng)
            } else {
                Ok(input)
            }
        }
        ImageAugmentationOp::BrightnessJitter { max_delta } => {
            if *max_delta == 0.0 {
                return Ok(input);
            }
            let delta = rng.next_signed_unit() * *max_delta;
            apply_brightness_delta(&input, delta)
        }
        ImageAugmentationOp::ContrastJitter { max_scale_delta } => {
            if *max_scale_delta == 0.0 {
                return Ok(input);
            }
            let scale = (1.0 + rng.next_signed_unit() * *max_scale_delta).max(0.0);
            apply_contrast_scale(&input, scale)
        }
        ImageAugmentationOp::GammaJitter { max_gamma_delta } => {
            if *max_gamma_delta == 0.0 {
                return Ok(input);
            }
            let gamma = (1.0 + rng.next_signed_unit() * *max_gamma_delta).max(0.01);
            apply_gamma_correction(&input, gamma)
        }
        ImageAugmentationOp::GaussianNoise {
            probability,
            std_dev,
        } => {
            if !should_apply(*probability, rng) || *std_dev == 0.0 {
                return Ok(input);
            }
            apply_gaussian_noise(&input, *std_dev, rng)
        }
        ImageAugmentationOp::BoxBlur3x3 { probability } => {
            if should_apply(*probability, rng) {
                box_blur_3x3(&input).map_err(Into::into)
            } else {
                Ok(input)
            }
        }
        ImageAugmentationOp::RandomResizedCrop {
            probability,
            min_scale,
            max_scale,
        } => {
            if !should_apply(*probability, rng) {
                return Ok(input);
            }
            apply_random_resized_crop(&input, *min_scale, *max_scale, rng)
        }
        ImageAugmentationOp::Cutout {
            probability,
            max_height_fraction,
            max_width_fraction,
            fill_value,
        } => {
            if !should_apply(*probability, rng) {
                return Ok(input);
            }
            apply_cutout(
                &input,
                *max_height_fraction,
                *max_width_fraction,
                *fill_value,
                rng,
            )
        }
        ImageAugmentationOp::ChannelNormalize { mean, std } => {
            normalize(&input, mean, std).map_err(Into::into)
        }
        ImageAugmentationOp::Custom(f) => f(&input),
        ImageAugmentationOp::RandomCrop { height, width } => {
            apply_random_crop(&input, *height, *width, rng)
        }
        ImageAugmentationOp::GaussianBlur { kernel_size } => {
            apply_gaussian_blur(&input, *kernel_size)
        }
    }
}

fn apply_brightness_delta(input: &Tensor, delta: f32) -> Result<Tensor, ModelError> {
    let mut output = Vec::with_capacity(input.data().len());
    for value in input.data() {
        output.push((*value + delta).clamp(0.0, 1.0));
    }
    Tensor::from_vec(input.shape().to_vec(), output).map_err(Into::into)
}

fn apply_random_rotate90(input: Tensor, rng: &mut LcgRng) -> Result<Tensor, ModelError> {
    if input.rank() != 3 {
        return Err(ModelError::InvalidAugmentationArgument {
            operation: "random_rotate90",
            message: format!("expected rank-3 HWC sample, got shape {:?}", input.shape()),
        });
    }

    let height = input.shape()[0];
    let width = input.shape()[1];
    if height == 0 || width == 0 {
        return Ok(input);
    }

    let rotation_count = if height == width {
        rng.next_usize(4)
    } else {
        rng.next_usize(2) * 2
    };

    let mut rotated = input;
    for _ in 0..rotation_count {
        rotated = rotate90_cw(&rotated).map_err(ModelError::from)?;
    }
    Ok(rotated)
}

fn apply_contrast_scale(input: &Tensor, scale: f32) -> Result<Tensor, ModelError> {
    let mean = if input.is_empty() {
        0.0
    } else {
        input.data().iter().copied().sum::<f32>() / input.len() as f32
    };

    let mut output = Vec::with_capacity(input.data().len());
    for value in input.data() {
        let scaled = (*value - mean) * scale + mean;
        output.push(scaled.clamp(0.0, 1.0));
    }
    Tensor::from_vec(input.shape().to_vec(), output).map_err(Into::into)
}

fn apply_gamma_correction(input: &Tensor, gamma: f32) -> Result<Tensor, ModelError> {
    let mut output = Vec::with_capacity(input.data().len());
    for value in input.data() {
        output.push(value.clamp(0.0, 1.0).powf(gamma));
    }
    Tensor::from_vec(input.shape().to_vec(), output).map_err(Into::into)
}

fn apply_gaussian_noise(
    input: &Tensor,
    std_dev: f32,
    rng: &mut LcgRng,
) -> Result<Tensor, ModelError> {
    let mut output = Vec::with_capacity(input.data().len());
    for value in input.data() {
        let noise = rng.next_gaussian() * std_dev;
        output.push((*value + noise).clamp(0.0, 1.0));
    }
    Tensor::from_vec(input.shape().to_vec(), output).map_err(Into::into)
}

fn apply_cutout(
    input: &Tensor,
    max_height_fraction: f32,
    max_width_fraction: f32,
    fill_value: f32,
    rng: &mut LcgRng,
) -> Result<Tensor, ModelError> {
    if input.rank() != 3 {
        return Err(ModelError::InvalidAugmentationArgument {
            operation: "cutout",
            message: format!("expected rank-3 HWC sample, got shape {:?}", input.shape()),
        });
    }
    let height = input.shape()[0];
    let width = input.shape()[1];
    let channels = input.shape()[2];
    if height == 0 || width == 0 || channels == 0 {
        return Ok(input.clone());
    }

    let max_cut_height = ((height as f32 * max_height_fraction).floor() as usize)
        .max(1)
        .min(height);
    let max_cut_width = ((width as f32 * max_width_fraction).floor() as usize)
        .max(1)
        .min(width);

    let cut_height = rng.next_usize_inclusive(max_cut_height - 1) + 1;
    let cut_width = rng.next_usize_inclusive(max_cut_width - 1) + 1;
    let top = rng.next_usize(height - cut_height + 1);
    let left = rng.next_usize(width - cut_width + 1);

    let mut output = input.data().to_vec();
    for y in top..(top + cut_height) {
        for x in left..(left + cut_width) {
            let base = (y * width + x) * channels;
            for channel in 0..channels {
                output[base + channel] = fill_value;
            }
        }
    }
    Tensor::from_vec(input.shape().to_vec(), output).map_err(Into::into)
}

fn apply_random_resized_crop(
    input: &Tensor,
    min_scale: f32,
    max_scale: f32,
    rng: &mut LcgRng,
) -> Result<Tensor, ModelError> {
    if input.rank() != 3 {
        return Err(ModelError::InvalidAugmentationArgument {
            operation: "random_resized_crop",
            message: format!("expected rank-3 HWC sample, got shape {:?}", input.shape()),
        });
    }
    let height = input.shape()[0];
    let width = input.shape()[1];
    let channels = input.shape()[2];
    if height == 0 || width == 0 || channels == 0 {
        return Ok(input.clone());
    }

    let scale = if (max_scale - min_scale).abs() <= f32::EPSILON {
        min_scale
    } else {
        min_scale + rng.next_unit() * (max_scale - min_scale)
    };

    let crop_height = ((height as f32 * scale).floor() as usize)
        .max(1)
        .min(height);
    let crop_width = ((width as f32 * scale).floor() as usize).max(1).min(width);
    let top = rng.next_usize(height - crop_height + 1);
    let left = rng.next_usize(width - crop_width + 1);

    let mut cropped = vec![0.0f32; crop_height * crop_width * channels];
    for y in 0..crop_height {
        for x in 0..crop_width {
            let src_base = ((top + y) * width + (left + x)) * channels;
            let dst_base = (y * crop_width + x) * channels;
            cropped[dst_base..(dst_base + channels)]
                .copy_from_slice(&input.data()[src_base..(src_base + channels)]);
        }
    }

    let cropped_tensor = Tensor::from_vec(vec![crop_height, crop_width, channels], cropped)?;
    resize_nearest(&cropped_tensor, height, width).map_err(Into::into)
}

fn apply_random_crop(
    input: &Tensor,
    crop_height: usize,
    crop_width: usize,
    rng: &mut LcgRng,
) -> Result<Tensor, ModelError> {
    if input.rank() != 3 {
        return Err(ModelError::InvalidAugmentationArgument {
            operation: "random_crop",
            message: format!("expected rank-3 HWC sample, got shape {:?}", input.shape()),
        });
    }
    let height = input.shape()[0];
    let width = input.shape()[1];
    let channels = input.shape()[2];

    if crop_height > height || crop_width > width {
        return Err(ModelError::InvalidAugmentationArgument {
            operation: "random_crop",
            message: format!(
                "crop size ({crop_height}x{crop_width}) exceeds input size ({height}x{width})"
            ),
        });
    }

    let top = rng.next_usize(height - crop_height + 1);
    let left = rng.next_usize(width - crop_width + 1);

    let mut cropped = vec![0.0f32; crop_height * crop_width * channels];
    for y in 0..crop_height {
        for x in 0..crop_width {
            let src_base = ((top + y) * width + (left + x)) * channels;
            let dst_base = (y * crop_width + x) * channels;
            cropped[dst_base..(dst_base + channels)]
                .copy_from_slice(&input.data()[src_base..(src_base + channels)]);
        }
    }

    Tensor::from_vec(vec![crop_height, crop_width, channels], cropped).map_err(Into::into)
}

fn apply_gaussian_blur(input: &Tensor, kernel_size: usize) -> Result<Tensor, ModelError> {
    if input.rank() != 3 {
        return Err(ModelError::InvalidAugmentationArgument {
            operation: "gaussian_blur",
            message: format!("expected rank-3 HWC sample, got shape {:?}", input.shape()),
        });
    }
    let height = input.shape()[0];
    let width = input.shape()[1];
    let channels = input.shape()[2];
    if height == 0 || width == 0 || channels == 0 {
        return Ok(input.clone());
    }

    // Build 1-D Gaussian kernel weights.
    let sigma = (kernel_size as f32 - 1.0) / 2.0 * 0.5 + 0.8;
    let half = (kernel_size / 2) as isize;
    let mut kernel = Vec::with_capacity(kernel_size);
    let mut sum = 0.0f32;
    for i in 0..kernel_size {
        let x = (i as isize - half) as f32;
        let w = (-x * x / (2.0 * sigma * sigma)).exp();
        kernel.push(w);
        sum += w;
    }
    for w in &mut kernel {
        *w /= sum;
    }

    // Horizontal pass.
    let mut tmp = vec![0.0f32; height * width * channels];
    for y in 0..height {
        for x in 0..width {
            for c in 0..channels {
                let mut acc = 0.0f32;
                for ki in 0..kernel_size {
                    let sx =
                        (x as isize + ki as isize - half).clamp(0, width as isize - 1) as usize;
                    acc += input.data()[(y * width + sx) * channels + c] * kernel[ki];
                }
                tmp[(y * width + x) * channels + c] = acc;
            }
        }
    }

    // Vertical pass.
    let mut out = vec![0.0f32; height * width * channels];
    for y in 0..height {
        for x in 0..width {
            for c in 0..channels {
                let mut acc = 0.0f32;
                for ki in 0..kernel_size {
                    let sy =
                        (y as isize + ki as isize - half).clamp(0, height as isize - 1) as usize;
                    acc += tmp[(sy * width + x) * channels + c] * kernel[ki];
                }
                out[(y * width + x) * channels + c] = acc;
            }
        }
    }

    Tensor::from_vec(vec![height, width, channels], out).map_err(Into::into)
}

fn validate_probability(operation: &'static str, probability: f32) -> Result<(), ModelError> {
    if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
        return Err(ModelError::InvalidAugmentationProbability {
            operation,
            value: probability,
        });
    }
    Ok(())
}

fn validate_fraction(
    operation: &'static str,
    parameter: &'static str,
    value: f32,
) -> Result<(), ModelError> {
    if !value.is_finite() || value <= 0.0 || value > 1.0 {
        return Err(ModelError::InvalidAugmentationArgument {
            operation,
            message: format!("{parameter} must be finite in (0, 1], got {value}"),
        });
    }
    Ok(())
}

fn should_apply(probability: f32, rng: &mut LcgRng) -> bool {
    if probability <= 0.0 {
        return false;
    }
    if probability >= 1.0 {
        return true;
    }
    rng.next_unit() < probability
}

#[derive(Debug, Clone, Copy)]
struct LcgRng {
    state: u64,
}

impl LcgRng {
    const MULTIPLIER: u64 = 6364136223846793005;
    const INCREMENT: u64 = 1;

    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    const fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(Self::MULTIPLIER)
            .wrapping_add(Self::INCREMENT);
        (self.state >> 32) as u32
    }

    fn next_unit(&mut self) -> f32 {
        self.next_u32() as f32 / (u32::MAX as f32 + 1.0)
    }

    fn next_signed_unit(&mut self) -> f32 {
        self.next_unit() * 2.0 - 1.0
    }

    fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_unit().max(f32::MIN_POSITIVE);
        let u2 = self.next_unit();
        let magnitude = (-2.0 * u1.ln()).sqrt();
        let angle = 2.0 * std::f32::consts::PI * u2;
        magnitude * angle.cos()
    }

    const fn next_usize_inclusive(&mut self, upper_inclusive: usize) -> usize {
        self.next_usize(upper_inclusive.saturating_add(1))
    }

    const fn next_usize(&mut self, upper_exclusive: usize) -> usize {
        if upper_exclusive == 0 {
            return 0;
        }
        (self.next_u32() as usize) % upper_exclusive
    }
}
