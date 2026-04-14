//! Pretrained model zoo: architecture registry, builders, and weight management.

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use yscv_autograd::Graph;
use yscv_tensor::Tensor;

use crate::{
    ModelError, SequentialModel, add_bottleneck_block, add_residual_block,
    build_resnet_feature_extractor, load_weights, save_weights,
};

// ---------------------------------------------------------------------------
// Architecture registry
// ---------------------------------------------------------------------------

/// Known model architectures in the zoo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelArchitecture {
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet101,
    VGG16,
    VGG19,
    MobileNetV2,
    EfficientNetB0,
    AlexNet,
    ViTTiny,
    ViTBase,
    ViTLarge,
    DeiTTiny,
    ClipViTB32,
    DINOv2ViTS14,
    WhisperTiny,
    SAMViTB,
}

impl ModelArchitecture {
    /// Returns the canonical configuration for this architecture.
    pub fn config(&self) -> ArchitectureConfig {
        match self {
            Self::ResNet18 => ArchitectureConfig {
                input_channels: 3,
                num_classes: 1000,
                stage_channels: vec![64, 128, 256, 512],
                blocks_per_stage: vec![2, 2, 2, 2],
            },
            Self::ResNet34 => ArchitectureConfig {
                input_channels: 3,
                num_classes: 1000,
                stage_channels: vec![64, 128, 256, 512],
                blocks_per_stage: vec![3, 4, 6, 3],
            },
            Self::ResNet50 => ArchitectureConfig {
                input_channels: 3,
                num_classes: 1000,
                stage_channels: vec![64, 128, 256, 512],
                blocks_per_stage: vec![3, 4, 6, 3],
            },
            Self::ResNet101 => ArchitectureConfig {
                input_channels: 3,
                num_classes: 1000,
                stage_channels: vec![64, 128, 256, 512],
                blocks_per_stage: vec![3, 4, 23, 3],
            },
            Self::VGG16 => ArchitectureConfig {
                input_channels: 3,
                num_classes: 1000,
                stage_channels: vec![64, 128, 256, 512, 512],
                blocks_per_stage: vec![2, 2, 3, 3, 3],
            },
            Self::VGG19 => ArchitectureConfig {
                input_channels: 3,
                num_classes: 1000,
                stage_channels: vec![64, 128, 256, 512, 512],
                blocks_per_stage: vec![2, 2, 4, 4, 4],
            },
            Self::MobileNetV2 => ArchitectureConfig {
                input_channels: 3,
                num_classes: 1000,
                stage_channels: vec![32, 16, 24, 32, 64, 96, 160, 320],
                blocks_per_stage: vec![1, 1, 2, 3, 4, 3, 3, 1],
            },
            Self::EfficientNetB0 => ArchitectureConfig {
                input_channels: 3,
                num_classes: 1000,
                stage_channels: vec![32, 16, 24, 40, 80, 112, 192, 320],
                blocks_per_stage: vec![1, 1, 2, 2, 3, 3, 4, 1],
            },
            Self::AlexNet => ArchitectureConfig {
                input_channels: 3,
                num_classes: 1000,
                stage_channels: vec![64, 192, 384, 256, 256],
                blocks_per_stage: vec![1, 1, 1, 1, 1],
            },
            Self::ViTTiny => ArchitectureConfig {
                input_channels: 3,
                num_classes: 1000,
                stage_channels: vec![192],  // embed_dim
                blocks_per_stage: vec![12], // num_layers
            },
            Self::ViTBase => ArchitectureConfig {
                input_channels: 3,
                num_classes: 1000,
                stage_channels: vec![768],
                blocks_per_stage: vec![12],
            },
            Self::ViTLarge => ArchitectureConfig {
                input_channels: 3,
                num_classes: 1000,
                stage_channels: vec![1024],
                blocks_per_stage: vec![24],
            },
            Self::DeiTTiny => ArchitectureConfig {
                input_channels: 3,
                num_classes: 1000,
                stage_channels: vec![192],
                blocks_per_stage: vec![12],
            },
            Self::ClipViTB32 => ArchitectureConfig {
                input_channels: 3,
                num_classes: 512,
                stage_channels: vec![768],
                blocks_per_stage: vec![12],
            },
            Self::DINOv2ViTS14 => ArchitectureConfig {
                input_channels: 3,
                num_classes: 384,
                stage_channels: vec![384],
                blocks_per_stage: vec![12],
            },
            Self::WhisperTiny => ArchitectureConfig {
                input_channels: 1,
                num_classes: 51865,
                stage_channels: vec![384],
                blocks_per_stage: vec![4],
            },
            Self::SAMViTB => ArchitectureConfig {
                input_channels: 3,
                num_classes: 256,
                stage_channels: vec![768],
                blocks_per_stage: vec![12],
            },
        }
    }

    /// Returns a filesystem-safe name for this architecture (used for weight files).
    pub fn name(&self) -> &'static str {
        match self {
            Self::ResNet18 => "resnet18",
            Self::ResNet34 => "resnet34",
            Self::ResNet50 => "resnet50",
            Self::ResNet101 => "resnet101",
            Self::VGG16 => "vgg16",
            Self::VGG19 => "vgg19",
            Self::MobileNetV2 => "mobilenet_v2",
            Self::EfficientNetB0 => "efficientnet_b0",
            Self::AlexNet => "alexnet",
            Self::ViTTiny => "vit_tiny",
            Self::ViTBase => "vit_base",
            Self::ViTLarge => "vit_large",
            Self::DeiTTiny => "deit_tiny",
            Self::ClipViTB32 => "clip_vit_b32",
            Self::DINOv2ViTS14 => "dinov2_vit_s14",
            Self::WhisperTiny => "whisper_tiny",
            Self::SAMViTB => "sam_vit_b",
        }
    }

    /// All known architectures.
    pub fn all() -> &'static [ModelArchitecture] {
        &[
            Self::ResNet18,
            Self::ResNet34,
            Self::ResNet50,
            Self::ResNet101,
            Self::VGG16,
            Self::VGG19,
            Self::MobileNetV2,
            Self::EfficientNetB0,
            Self::AlexNet,
            Self::ViTTiny,
            Self::ViTBase,
            Self::ViTLarge,
            Self::DeiTTiny,
            Self::ClipViTB32,
            Self::DINOv2ViTS14,
            Self::WhisperTiny,
            Self::SAMViTB,
        ]
    }
}

impl std::fmt::Display for ModelArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

// ---------------------------------------------------------------------------
// Architecture config
// ---------------------------------------------------------------------------

/// Describes the shape of a model architecture.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArchitectureConfig {
    /// Number of input channels (typically 3 for RGB).
    pub input_channels: usize,
    /// Number of output classes (default 1000 for ImageNet).
    pub num_classes: usize,
    /// Channel widths per stage.
    pub stage_channels: Vec<usize>,
    /// Block counts per stage.
    pub blocks_per_stage: Vec<usize>,
}

impl ArchitectureConfig {
    /// Returns a copy with a different number of output classes.
    pub fn with_num_classes(&self, num_classes: usize) -> Self {
        let mut cfg = self.clone();
        cfg.num_classes = num_classes;
        cfg
    }
}

// ---------------------------------------------------------------------------
// Architecture builders
// ---------------------------------------------------------------------------

const BN_EPSILON: f32 = 1e-5;

/// Builds a ResNet-family model: stem + residual stages + global-avg-pool + linear head.
pub fn build_resnet(
    graph: &mut Graph,
    config: &ArchitectureConfig,
) -> Result<SequentialModel, ModelError> {
    let mut model = SequentialModel::new(graph);
    let max_blocks = config.blocks_per_stage.iter().copied().max().unwrap_or(2);
    build_resnet_feature_extractor(
        &mut model,
        config.input_channels,
        &config.stage_channels,
        max_blocks,
        BN_EPSILON,
    )?;
    let final_ch = config.stage_channels.last().copied().unwrap_or(512);
    model.add_linear_zero(graph, final_ch, config.num_classes)?;
    Ok(model)
}

/// Builds a ResNet with per-stage block counts (bypasses the single-count helper).
pub fn build_resnet_custom(
    graph: &mut Graph,
    config: &ArchitectureConfig,
) -> Result<SequentialModel, ModelError> {
    let mut model = SequentialModel::new(graph);
    let initial_ch = config.stage_channels.first().copied().unwrap_or(64);

    model.add_conv2d_zero(config.input_channels, initial_ch, 7, 7, 2, 2, true)?;
    model.add_batch_norm2d_identity(initial_ch, BN_EPSILON)?;
    model.add_relu();
    model.add_max_pool2d(3, 3, 2, 2)?;

    let mut ch = initial_ch;
    for (stage_idx, &stage_ch) in config.stage_channels.iter().enumerate() {
        if stage_ch != ch {
            model.add_conv2d_zero(ch, stage_ch, 1, 1, 1, 1, false)?;
            model.add_batch_norm2d_identity(stage_ch, BN_EPSILON)?;
            model.add_relu();
        }
        let blocks = config.blocks_per_stage.get(stage_idx).copied().unwrap_or(2);
        for _ in 0..blocks {
            add_residual_block(&mut model, stage_ch, BN_EPSILON)?;
        }
        ch = stage_ch;
    }

    model.add_global_avg_pool2d();
    model.add_flatten();
    model.add_linear_zero(graph, ch, config.num_classes)?;
    Ok(model)
}

/// Builds a VGG-style sequential conv network.
///
/// Each stage: `blocks_per_stage[i]` x (Conv3x3 -> BN -> ReLU), then MaxPool2x2.
pub fn build_vgg(
    graph: &mut Graph,
    config: &ArchitectureConfig,
) -> Result<SequentialModel, ModelError> {
    let mut model = SequentialModel::new(graph);
    let mut ch = config.input_channels;

    for (stage_idx, &out_ch) in config.stage_channels.iter().enumerate() {
        let blocks = config.blocks_per_stage.get(stage_idx).copied().unwrap_or(2);
        for b in 0..blocks {
            let in_ch = if b == 0 { ch } else { out_ch };
            model.add_conv2d_zero(in_ch, out_ch, 3, 3, 1, 1, true)?;
            model.add_batch_norm2d_identity(out_ch, BN_EPSILON)?;
            model.add_relu();
        }
        model.add_max_pool2d(2, 2, 2, 2)?;
        ch = out_ch;
    }

    model.add_global_avg_pool2d();
    model.add_flatten();
    model.add_linear_zero(graph, ch, config.num_classes)?;
    Ok(model)
}

/// Builds a MobileNetV2-style model using inverted bottleneck blocks.
pub fn build_mobilenet_v2(
    graph: &mut Graph,
    config: &ArchitectureConfig,
) -> Result<SequentialModel, ModelError> {
    let mut model = SequentialModel::new(graph);
    let stem_ch = config.stage_channels.first().copied().unwrap_or(32);
    model.add_conv2d_zero(config.input_channels, stem_ch, 3, 3, 2, 2, false)?;
    model.add_batch_norm2d_identity(stem_ch, BN_EPSILON)?;
    model.add_relu();

    let mut ch = stem_ch;
    for (stage_idx, &out_ch) in config.stage_channels.iter().enumerate().skip(1) {
        let blocks = config.blocks_per_stage.get(stage_idx).copied().unwrap_or(1);
        let expand_ratio = 6;
        for b in 0..blocks {
            let stride = if b == 0 && stage_idx > 1 { 2 } else { 1 };
            let expand_ch = ch * expand_ratio;
            add_bottleneck_block(&mut model, ch, expand_ch, out_ch, stride, BN_EPSILON)?;
            ch = out_ch;
        }
    }

    let last_ch = 1280;
    model.add_conv2d_zero(ch, last_ch, 1, 1, 1, 1, false)?;
    model.add_batch_norm2d_identity(last_ch, BN_EPSILON)?;
    model.add_relu();
    model.add_global_avg_pool2d();
    model.add_flatten();
    model.add_linear_zero(graph, last_ch, config.num_classes)?;
    Ok(model)
}

/// Builds a simple AlexNet-style conv stack.
pub fn build_alexnet(
    graph: &mut Graph,
    config: &ArchitectureConfig,
) -> Result<SequentialModel, ModelError> {
    let mut model = SequentialModel::new(graph);
    let channels = &config.stage_channels;

    let ch0 = channels.first().copied().unwrap_or(64);
    model.add_conv2d_zero(config.input_channels, ch0, 11, 11, 4, 4, true)?;
    model.add_relu();
    model.add_max_pool2d(3, 3, 2, 2)?;

    let ch1 = channels.get(1).copied().unwrap_or(192);
    model.add_conv2d_zero(ch0, ch1, 5, 5, 1, 1, true)?;
    model.add_relu();
    model.add_max_pool2d(3, 3, 2, 2)?;

    let ch2 = channels.get(2).copied().unwrap_or(384);
    model.add_conv2d_zero(ch1, ch2, 3, 3, 1, 1, true)?;
    model.add_relu();

    let ch3 = channels.get(3).copied().unwrap_or(256);
    model.add_conv2d_zero(ch2, ch3, 3, 3, 1, 1, true)?;
    model.add_relu();

    let ch4 = channels.get(4).copied().unwrap_or(256);
    model.add_conv2d_zero(ch3, ch4, 3, 3, 1, 1, true)?;
    model.add_relu();
    model.add_max_pool2d(3, 3, 2, 2)?;

    model.add_global_avg_pool2d();
    model.add_flatten();
    model.add_linear_zero(graph, ch4, config.num_classes)?;
    Ok(model)
}

// ---------------------------------------------------------------------------
// Model Zoo (weight loading / saving)
// ---------------------------------------------------------------------------

/// File-based pretrained model registry.
///
/// Points to a local directory that stores `{arch_name}.bin` weight files
/// in the same format as `save_weights` / `load_weights`.
pub struct ModelZoo {
    registry_dir: PathBuf,
}

impl ModelZoo {
    /// Creates a new zoo pointing at `registry_dir`.
    pub fn new(registry_dir: impl Into<PathBuf>) -> Self {
        Self {
            registry_dir: registry_dir.into(),
        }
    }

    fn weight_path(&self, arch: ModelArchitecture) -> PathBuf {
        self.registry_dir.join(format!("{}.bin", arch.name()))
    }

    /// Builds the architecture and loads pretrained weights from
    /// `{registry_dir}/{arch_name}.bin`.
    pub fn load_pretrained(
        &self,
        arch: ModelArchitecture,
        graph: &mut Graph,
    ) -> Result<SequentialModel, ModelError> {
        let path = self.weight_path(arch);
        let weights = load_weights(&path)?;
        let config = arch.config();
        let mut model = build_architecture(arch, graph, &config)?;
        apply_weights(&mut model, graph, &weights)?;
        Ok(model)
    }

    /// Lists architectures for which a `.bin` weight file exists in the registry.
    pub fn list_available(&self) -> Vec<ModelArchitecture> {
        ModelArchitecture::all()
            .iter()
            .copied()
            .filter(|a| self.weight_path(*a).is_file())
            .collect()
    }

    /// Saves a model's layer weights to `{registry_dir}/{arch_name}.bin`.
    pub fn save_pretrained(
        &self,
        arch: ModelArchitecture,
        model: &SequentialModel,
        graph: &Graph,
    ) -> Result<(), ModelError> {
        let path = self.weight_path(arch);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| ModelError::DatasetLoadIo {
                path: parent.display().to_string(),
                message: e.to_string(),
            })?;
        }
        let tensors = collect_model_tensors(model, graph)?;
        save_weights(&path, &tensors)
    }
}

/// Collects all named tensors from a `SequentialModel` for serialization.
fn collect_model_tensors(
    model: &SequentialModel,
    graph: &Graph,
) -> Result<HashMap<String, yscv_tensor::Tensor>, ModelError> {
    let mut tensors = HashMap::new();
    for (idx, layer) in model.layers().iter().enumerate() {
        match layer {
            crate::ModelLayer::Conv2d(l) => {
                tensors.insert(format!("layer.{idx}.conv2d.weight"), l.weight().clone());
                if let Some(b) = l.bias() {
                    tensors.insert(format!("layer.{idx}.conv2d.bias"), b.clone());
                }
            }
            crate::ModelLayer::BatchNorm2d(l) => {
                tensors.insert(format!("layer.{idx}.bn.gamma"), l.gamma().clone());
                tensors.insert(format!("layer.{idx}.bn.beta"), l.beta().clone());
                tensors.insert(
                    format!("layer.{idx}.bn.running_mean"),
                    l.running_mean().clone(),
                );
                tensors.insert(
                    format!("layer.{idx}.bn.running_var"),
                    l.running_var().clone(),
                );
            }
            crate::ModelLayer::Linear(l) => {
                let w = graph
                    .value(l.weight_node().expect("linear layer has weight node"))?
                    .clone();
                let b = graph
                    .value(l.bias_node().expect("linear layer has bias node"))?
                    .clone();
                tensors.insert(format!("layer.{idx}.linear.weight"), w);
                tensors.insert(format!("layer.{idx}.linear.bias"), b);
            }
            _ => {}
        }
    }
    Ok(tensors)
}

/// Apply named weight tensors to a SequentialModel's layers.
/// Uses the same naming convention as `collect_model_tensors`:
/// - `layer.{idx}.conv2d.weight`, `layer.{idx}.conv2d.bias`
/// - `layer.{idx}.bn.gamma`, `layer.{idx}.bn.beta`, `layer.{idx}.bn.running_mean`, `layer.{idx}.bn.running_var`
/// - `layer.{idx}.linear.weight`, `layer.{idx}.linear.bias`
fn apply_weights(
    model: &mut SequentialModel,
    graph: &mut Graph,
    weights: &HashMap<String, Tensor>,
) -> Result<(), ModelError> {
    for (idx, layer) in model.layers_mut().iter_mut().enumerate() {
        match layer {
            crate::ModelLayer::Conv2d(l) => {
                if let Some(w) = weights.get(&format!("layer.{idx}.conv2d.weight")) {
                    *l.weight_mut() = w.clone();
                }
                if let Some(b) = weights.get(&format!("layer.{idx}.conv2d.bias"))
                    && let Some(bias) = l.bias_mut()
                {
                    *bias = b.clone();
                }
            }
            crate::ModelLayer::BatchNorm2d(l) => {
                if let Some(g) = weights.get(&format!("layer.{idx}.bn.gamma")) {
                    *l.gamma_mut() = g.clone();
                }
                if let Some(b) = weights.get(&format!("layer.{idx}.bn.beta")) {
                    *l.beta_mut() = b.clone();
                }
                if let Some(m) = weights.get(&format!("layer.{idx}.bn.running_mean")) {
                    *l.running_mean_mut() = m.clone();
                }
                if let Some(v) = weights.get(&format!("layer.{idx}.bn.running_var")) {
                    *l.running_var_mut() = v.clone();
                }
            }
            crate::ModelLayer::Linear(l) => {
                if let Some(w) = weights.get(&format!("layer.{idx}.linear.weight")) {
                    *graph.value_mut(l.weight_node().expect("linear layer has weight node"))? =
                        w.clone();
                }
                if let Some(b) = weights.get(&format!("layer.{idx}.linear.bias")) {
                    *graph.value_mut(l.bias_node().expect("linear layer has bias node"))? =
                        b.clone();
                }
            }
            _ => {}
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Feature extraction API
// ---------------------------------------------------------------------------

/// Builds a backbone (feature extractor) without the final classifier head.
pub fn build_feature_extractor(
    arch: ModelArchitecture,
    graph: &mut Graph,
    config: &ArchitectureConfig,
) -> Result<SequentialModel, ModelError> {
    match arch {
        ModelArchitecture::ResNet18
        | ModelArchitecture::ResNet34
        | ModelArchitecture::ResNet50
        | ModelArchitecture::ResNet101 => {
            let mut model = SequentialModel::new(graph);
            let max_blocks = config.blocks_per_stage.iter().copied().max().unwrap_or(2);
            build_resnet_feature_extractor(
                &mut model,
                config.input_channels,
                &config.stage_channels,
                max_blocks,
                BN_EPSILON,
            )?;
            Ok(model)
        }
        ModelArchitecture::VGG16 | ModelArchitecture::VGG19 => {
            let mut model = SequentialModel::new(graph);
            let mut ch = config.input_channels;
            for (stage_idx, &out_ch) in config.stage_channels.iter().enumerate() {
                let blocks = config.blocks_per_stage.get(stage_idx).copied().unwrap_or(2);
                for b in 0..blocks {
                    let in_ch = if b == 0 { ch } else { out_ch };
                    model.add_conv2d_zero(in_ch, out_ch, 3, 3, 1, 1, true)?;
                    model.add_batch_norm2d_identity(out_ch, BN_EPSILON)?;
                    model.add_relu();
                }
                model.add_max_pool2d(2, 2, 2, 2)?;
                ch = out_ch;
            }
            model.add_global_avg_pool2d();
            model.add_flatten();
            Ok(model)
        }
        ModelArchitecture::MobileNetV2 | ModelArchitecture::EfficientNetB0 => {
            let mut model = SequentialModel::new(graph);
            let stem_ch = config.stage_channels.first().copied().unwrap_or(32);
            model.add_conv2d_zero(config.input_channels, stem_ch, 3, 3, 2, 2, false)?;
            model.add_batch_norm2d_identity(stem_ch, BN_EPSILON)?;
            model.add_relu();

            let mut ch = stem_ch;
            for (stage_idx, &out_ch) in config.stage_channels.iter().enumerate().skip(1) {
                let blocks = config.blocks_per_stage.get(stage_idx).copied().unwrap_or(1);
                for b in 0..blocks {
                    let stride = if b == 0 && stage_idx > 1 { 2 } else { 1 };
                    let expand_ch = ch * 6;
                    add_bottleneck_block(&mut model, ch, expand_ch, out_ch, stride, BN_EPSILON)?;
                    ch = out_ch;
                }
            }
            let last_ch = 1280;
            model.add_conv2d_zero(ch, last_ch, 1, 1, 1, 1, false)?;
            model.add_batch_norm2d_identity(last_ch, BN_EPSILON)?;
            model.add_relu();
            model.add_global_avg_pool2d();
            model.add_flatten();
            Ok(model)
        }
        ModelArchitecture::AlexNet => {
            let mut model = SequentialModel::new(graph);
            let channels = &config.stage_channels;
            let ch0 = channels.first().copied().unwrap_or(64);
            model.add_conv2d_zero(config.input_channels, ch0, 11, 11, 4, 4, true)?;
            model.add_relu();
            model.add_max_pool2d(3, 3, 2, 2)?;

            let ch1 = channels.get(1).copied().unwrap_or(192);
            model.add_conv2d_zero(ch0, ch1, 5, 5, 1, 1, true)?;
            model.add_relu();
            model.add_max_pool2d(3, 3, 2, 2)?;

            let ch2 = channels.get(2).copied().unwrap_or(384);
            model.add_conv2d_zero(ch1, ch2, 3, 3, 1, 1, true)?;
            model.add_relu();

            let ch3 = channels.get(3).copied().unwrap_or(256);
            model.add_conv2d_zero(ch2, ch3, 3, 3, 1, 1, true)?;
            model.add_relu();

            let ch4 = channels.get(4).copied().unwrap_or(256);
            model.add_conv2d_zero(ch3, ch4, 3, 3, 1, 1, true)?;
            model.add_relu();
            model.add_max_pool2d(3, 3, 2, 2)?;
            model.add_global_avg_pool2d();
            model.add_flatten();
            Ok(model)
        }
        ModelArchitecture::ViTTiny
        | ModelArchitecture::ViTBase
        | ModelArchitecture::ViTLarge
        | ModelArchitecture::DeiTTiny
        | ModelArchitecture::ClipViTB32
        | ModelArchitecture::DINOv2ViTS14
        | ModelArchitecture::WhisperTiny
        | ModelArchitecture::SAMViTB => {
            let embed_dim = config.stage_channels.first().copied().unwrap_or(192);
            let mut model = SequentialModel::new(graph);
            model.add_conv2d_zero(config.input_channels, embed_dim, 16, 16, 16, 16, false)?;
            model.add_flatten();
            Ok(model)
        }
    }
}

/// Builds a full classifier with a custom number of output classes.
pub fn build_classifier(
    arch: ModelArchitecture,
    graph: &mut Graph,
    num_classes: usize,
) -> Result<SequentialModel, ModelError> {
    let config = arch.config().with_num_classes(num_classes);
    build_architecture(arch, graph, &config)
}

/// Internal dispatcher: builds a complete model for any architecture.
fn build_architecture(
    arch: ModelArchitecture,
    graph: &mut Graph,
    config: &ArchitectureConfig,
) -> Result<SequentialModel, ModelError> {
    match arch {
        ModelArchitecture::ResNet18
        | ModelArchitecture::ResNet34
        | ModelArchitecture::ResNet50
        | ModelArchitecture::ResNet101 => build_resnet_custom(graph, config),
        ModelArchitecture::VGG16 | ModelArchitecture::VGG19 => build_vgg(graph, config),
        ModelArchitecture::MobileNetV2 | ModelArchitecture::EfficientNetB0 => {
            build_mobilenet_v2(graph, config)
        }
        ModelArchitecture::AlexNet => build_alexnet(graph, config),
        ModelArchitecture::ViTTiny
        | ModelArchitecture::ViTBase
        | ModelArchitecture::ViTLarge
        | ModelArchitecture::DeiTTiny
        | ModelArchitecture::ClipViTB32
        | ModelArchitecture::DINOv2ViTS14
        | ModelArchitecture::WhisperTiny
        | ModelArchitecture::SAMViTB => {
            let embed_dim = config.stage_channels.first().copied().unwrap_or(192);
            let mut model = SequentialModel::new(graph);
            model.add_conv2d_zero(config.input_channels, embed_dim, 16, 16, 16, 16, false)?;
            model.add_flatten();
            model.add_linear_zero(graph, embed_dim, config.num_classes)?;
            Ok(model)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_pretrained_applies_weights() -> Result<(), Box<dyn std::error::Error>> {
        let mut graph = Graph::new();
        let arch = ModelArchitecture::AlexNet;
        let config = arch.config();
        let model = build_architecture(arch, &mut graph, &config)?;

        // Collect initial (zero) tensors and set some non-zero values.
        let mut tensors = collect_model_tensors(&model, &graph)?;

        // Verify we actually have tensors to work with.
        assert!(!tensors.is_empty(), "model should have named tensors");

        // Fill every tensor with 0.42 so we can detect whether they get applied.
        for t in tensors.values_mut() {
            let len = t.data().len();
            *t = yscv_tensor::Tensor::from_vec(t.shape().to_vec(), vec![0.42_f32; len])?;
        }

        // Save to a temp file, then load into a fresh model.
        let tmp_dir = std::env::temp_dir().join("yscv_test_zoo");
        let zoo = ModelZoo::new(&tmp_dir);
        let path = zoo.weight_path(arch);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        save_weights(&path, &tensors)?;

        // Build a fresh model (zero-init) and apply the saved weights.
        let mut graph2 = Graph::new();
        let loaded_model = zoo.load_pretrained(arch, &mut graph2)?;

        // Collect tensors from the loaded model and verify they are non-zero (0.42).
        let loaded_tensors = collect_model_tensors(&loaded_model, &graph2)?;

        for (name, original) in &tensors {
            let loaded = loaded_tensors
                .get(name)
                .ok_or_else(|| ModelError::WeightNotFound { name: name.clone() })?;
            assert_eq!(
                original.shape(),
                loaded.shape(),
                "shape mismatch for {name}"
            );
            // Every value should be 0.42, not zero.
            for (i, (&orig, &load)) in original.data().iter().zip(loaded.data().iter()).enumerate()
            {
                assert!(
                    (orig - load).abs() < 1e-6,
                    "value mismatch for {name}[{i}]: expected {orig}, got {load}"
                );
            }
        }

        // Cleanup.
        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&tmp_dir).ok();
        Ok(())
    }
}
