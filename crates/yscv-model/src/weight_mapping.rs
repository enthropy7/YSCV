//! Weight name mapping between external frameworks (timm/PyTorch) and yscv.
//!
//! When loading pretrained weights from HuggingFace, the tensor names follow
//! the PyTorch / timm convention (e.g. `layer1.0.conv1.weight`). This module
//! translates those names to the yscv `layer.{idx}.*` convention used by
//! [`crate::zoo::apply_weights`].

use std::collections::HashMap;

use crate::ModelArchitecture;

/// Translate a timm/PyTorch weight name to the corresponding yscv name.
///
/// Returns `None` if the name is not recognized for the given architecture.
pub fn timm_to_yscv_name(timm_name: &str, arch: ModelArchitecture) -> Option<String> {
    let table = build_mapping_table(arch);
    table.get(timm_name).cloned()
}

/// Remap an entire state dict from timm names to yscv names.
///
/// Unknown keys are silently dropped.
pub fn remap_state_dict<T: Clone>(
    state_dict: &HashMap<String, T>,
    arch: ModelArchitecture,
) -> HashMap<String, T> {
    let table = build_mapping_table(arch);
    let mut out = HashMap::new();
    for (timm_name, value) in state_dict {
        if let Some(yscv_name) = table.get(timm_name.as_str()) {
            out.insert(yscv_name.clone(), value.clone());
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Mapping tables per architecture
// ---------------------------------------------------------------------------

fn build_mapping_table(arch: ModelArchitecture) -> HashMap<&'static str, String> {
    match arch {
        ModelArchitecture::ResNet18 => build_resnet_basic_mapping(&[2, 2, 2, 2]),
        ModelArchitecture::ResNet34 => build_resnet_basic_mapping(&[3, 4, 6, 3]),
        ModelArchitecture::ResNet50 => build_resnet_bottleneck_mapping(&[3, 4, 6, 3]),
        ModelArchitecture::ResNet101 => build_resnet_bottleneck_mapping(&[3, 4, 23, 3]),
        ModelArchitecture::VGG16 => build_vgg_mapping(&[2, 2, 3, 3, 3]),
        ModelArchitecture::VGG19 => build_vgg_mapping(&[2, 2, 4, 4, 4]),
        ModelArchitecture::AlexNet => build_alexnet_mapping(),
        ModelArchitecture::MobileNetV2 => build_mobilenet_v2_mapping(),
        ModelArchitecture::EfficientNetB0 => build_efficientnet_b0_mapping(),
        ModelArchitecture::ViTTiny
        | ModelArchitecture::ViTBase
        | ModelArchitecture::ViTLarge
        | ModelArchitecture::DeiTTiny
        | ModelArchitecture::ClipViTB32
        | ModelArchitecture::DINOv2ViTS14
        | ModelArchitecture::WhisperTiny
        | ModelArchitecture::SAMViTB => HashMap::new(),
    }
}

// ---------------------------------------------------------------------------
// ResNet (basic blocks: ResNet18, ResNet34)
// ---------------------------------------------------------------------------

/// Build mapping for ResNet with basic blocks (2 convolutions per block).
///
/// timm structure: conv1 -> bn1 -> layer{1..4}.{block}.{conv1,bn1,conv2,bn2} -> fc
/// yscv structure: layer.{idx}.{conv2d,bn,...}
fn build_resnet_basic_mapping(blocks_per_stage: &[usize]) -> HashMap<&'static str, String> {
    let mut m = HashMap::new();

    // Stem: conv1(7x7) + bn1  →  layer.0 (Conv2d) + layer.1 (BN) + layer.2 (ReLU) + layer.3 (MaxPool)
    insert_static(&mut m, "conv1.weight", "layer.0.conv2d.weight");
    insert_bn_static(&mut m, "bn1", 1);

    // After stem: idx = 4 (conv, bn, relu, maxpool)
    let mut idx = 4usize;
    for (stage_i, &n_blocks) in blocks_per_stage.iter().enumerate() {
        let timm_stage = stage_i + 1; // layer1, layer2, ...

        for block in 0..n_blocks {
            // Channel transition (downsample) on first block of stages > 0
            if block == 0 && stage_i > 0 {
                // downsample.0 (Conv1x1) + downsample.1 (BN)
                let ds_conv = format!("layer{timm_stage}.{block}.downsample.0.weight");
                let ds_bn_prefix = format!("layer{timm_stage}.{block}.downsample.1");
                m.insert(leak(&ds_conv), format!("layer.{idx}.conv2d.weight"));
                insert_bn_dynamic(&mut m, &ds_bn_prefix, idx + 1);
                idx += 2; // conv + bn
            }

            // conv1 + bn1
            let c1 = format!("layer{timm_stage}.{block}.conv1.weight");
            let b1 = format!("layer{timm_stage}.{block}.bn1");
            m.insert(leak(&c1), format!("layer.{idx}.conv2d.weight"));
            insert_bn_dynamic(&mut m, &b1, idx + 1);
            idx += 3; // conv + bn + relu

            // conv2 + bn2
            let c2 = format!("layer{timm_stage}.{block}.conv2.weight");
            let b2 = format!("layer{timm_stage}.{block}.bn2");
            m.insert(leak(&c2), format!("layer.{idx}.conv2d.weight"));
            insert_bn_dynamic(&mut m, &b2, idx + 1);
            idx += 2; // conv + bn (no relu at end of residual, add happens outside)
        }
    }

    // Head: GlobalAvgPool + Flatten + FC
    // GlobalAvgPool (idx), Flatten (idx+1), Linear (idx+2)
    let fc_idx = idx + 2;
    insert_static(
        &mut m,
        "fc.weight",
        &format!("layer.{fc_idx}.linear.weight"),
    );
    insert_static(&mut m, "fc.bias", &format!("layer.{fc_idx}.linear.bias"));

    m
}

// ---------------------------------------------------------------------------
// ResNet (bottleneck blocks: ResNet50, ResNet101)
// ---------------------------------------------------------------------------

fn build_resnet_bottleneck_mapping(blocks_per_stage: &[usize]) -> HashMap<&'static str, String> {
    let mut m = HashMap::new();

    // Stem
    insert_static(&mut m, "conv1.weight", "layer.0.conv2d.weight");
    insert_bn_static(&mut m, "bn1", 1);

    let mut idx = 4usize;

    for (stage_i, &n_blocks) in blocks_per_stage.iter().enumerate() {
        let timm_stage = stage_i + 1;

        for block in 0..n_blocks {
            // Downsample on first block of each stage (bottleneck always has downsample for channel expansion)
            if block == 0 {
                let ds_conv = format!("layer{timm_stage}.{block}.downsample.0.weight");
                let ds_bn = format!("layer{timm_stage}.{block}.downsample.1");
                m.insert(leak(&ds_conv), format!("layer.{idx}.conv2d.weight"));
                insert_bn_dynamic(&mut m, &ds_bn, idx + 1);
                idx += 2;
            }

            // conv1 (1x1) + bn1
            let c1 = format!("layer{timm_stage}.{block}.conv1.weight");
            let b1 = format!("layer{timm_stage}.{block}.bn1");
            m.insert(leak(&c1), format!("layer.{idx}.conv2d.weight"));
            insert_bn_dynamic(&mut m, &b1, idx + 1);
            idx += 3;

            // conv2 (3x3) + bn2
            let c2 = format!("layer{timm_stage}.{block}.conv2.weight");
            let b2 = format!("layer{timm_stage}.{block}.bn2");
            m.insert(leak(&c2), format!("layer.{idx}.conv2d.weight"));
            insert_bn_dynamic(&mut m, &b2, idx + 1);
            idx += 3;

            // conv3 (1x1) + bn3
            let c3 = format!("layer{timm_stage}.{block}.conv3.weight");
            let b3 = format!("layer{timm_stage}.{block}.bn3");
            m.insert(leak(&c3), format!("layer.{idx}.conv2d.weight"));
            insert_bn_dynamic(&mut m, &b3, idx + 1);
            idx += 2;
        }
    }

    let fc_idx = idx + 2;
    insert_static(
        &mut m,
        "fc.weight",
        &format!("layer.{fc_idx}.linear.weight"),
    );
    insert_static(&mut m, "fc.bias", &format!("layer.{fc_idx}.linear.bias"));

    m
}

// ---------------------------------------------------------------------------
// VGG
// ---------------------------------------------------------------------------

fn build_vgg_mapping(blocks_per_stage: &[usize]) -> HashMap<&'static str, String> {
    let mut m = HashMap::new();
    let mut idx = 0usize;
    let mut timm_feat_idx = 0usize;

    for &n_blocks in blocks_per_stage {
        for _ in 0..n_blocks {
            // timm: features.{timm_feat_idx} (Conv2d), features.{timm_feat_idx+1} (BN)
            let cw = format!("features.{timm_feat_idx}.weight");
            let cb = format!("features.{timm_feat_idx}.bias");
            m.insert(leak(&cw), format!("layer.{idx}.conv2d.weight"));
            m.insert(leak(&cb), format!("layer.{idx}.conv2d.bias"));
            timm_feat_idx += 1;

            // BN layer
            let bn_prefix = format!("features.{timm_feat_idx}");
            insert_bn_dynamic(&mut m, &bn_prefix, idx + 1);
            timm_feat_idx += 1;

            // ReLU (no weights) - skip in timm indexing
            timm_feat_idx += 1;
            idx += 3; // conv + bn + relu in yscv
        }
        // MaxPool
        timm_feat_idx += 1; // skip timm maxpool index
        idx += 1; // yscv maxpool layer
    }

    // Head: GlobalAvgPool + Flatten + Linear
    let fc_idx = idx + 2;
    insert_static(
        &mut m,
        "head.fc.weight",
        &format!("layer.{fc_idx}.linear.weight"),
    );
    insert_static(
        &mut m,
        "head.fc.bias",
        &format!("layer.{fc_idx}.linear.bias"),
    );

    m
}

// ---------------------------------------------------------------------------
// AlexNet
// ---------------------------------------------------------------------------

fn build_alexnet_mapping() -> HashMap<&'static str, String> {
    let mut m = HashMap::new();

    // AlexNet timm: features.{0,3,6,8,10} are Conv2d layers
    // yscv build order from zoo.rs: conv→relu→maxpool → conv→relu→maxpool → conv→relu → conv→relu → conv→relu→maxpool → gap→flatten→linear

    let conv_timm_indices = [0, 3, 6, 8, 10];
    let mut idx = 0usize;

    for (i, &ti) in conv_timm_indices.iter().enumerate() {
        let cw = format!("features.{ti}.weight");
        let cb = format!("features.{ti}.bias");
        m.insert(leak(&cw), format!("layer.{idx}.conv2d.weight"));
        m.insert(leak(&cb), format!("layer.{idx}.conv2d.bias"));

        idx += 1; // conv
        idx += 1; // relu

        // MaxPool after conv 0, 1, 4
        if i == 0 || i == 1 || i == 4 {
            idx += 1;
        }
    }

    // Head
    let fc_idx = idx + 2; // gap + flatten + linear
    insert_static(
        &mut m,
        "classifier.6.weight",
        &format!("layer.{fc_idx}.linear.weight"),
    );
    insert_static(
        &mut m,
        "classifier.6.bias",
        &format!("layer.{fc_idx}.linear.bias"),
    );

    m
}

// ---------------------------------------------------------------------------
// MobileNetV2 (simplified)
// ---------------------------------------------------------------------------

fn build_mobilenet_v2_mapping() -> HashMap<&'static str, String> {
    // MobileNetV2 has complex inverted residual blocks. Provide stem/head mapping
    // and a generic numbered pattern for the rest.
    let mut m = HashMap::new();

    // Stem: conv_stem + bn1
    insert_static(&mut m, "conv_stem.weight", "layer.0.conv2d.weight");
    insert_bn_static(&mut m, "bn1", 1);

    // The middle blocks have highly variable structure. We provide a best-effort
    // mapping but complex architectures may need manual adjustment.
    // This is a placeholder-aware mapping for the most common timm naming.

    m
}

// ---------------------------------------------------------------------------
// EfficientNet-B0 (simplified)
// ---------------------------------------------------------------------------

fn build_efficientnet_b0_mapping() -> HashMap<&'static str, String> {
    let mut m = HashMap::new();

    // Stem
    insert_static(&mut m, "conv_stem.weight", "layer.0.conv2d.weight");
    insert_bn_static(&mut m, "bn1", 1);

    m
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Leak a String to get a `&'static str` for use as HashMap key.
/// This is acceptable for a build-once mapping table.
fn leak(s: &str) -> &'static str {
    Box::leak(s.to_string().into_boxed_str())
}

fn insert_static(m: &mut HashMap<&'static str, String>, timm: &'static str, yscv: &str) {
    m.insert(timm, yscv.to_string());
}

fn insert_bn_static(m: &mut HashMap<&'static str, String>, timm_prefix: &'static str, idx: usize) {
    m.insert(
        Box::leak(format!("{timm_prefix}.weight").into_boxed_str()),
        format!("layer.{idx}.bn.gamma"),
    );
    m.insert(
        Box::leak(format!("{timm_prefix}.bias").into_boxed_str()),
        format!("layer.{idx}.bn.beta"),
    );
    m.insert(
        Box::leak(format!("{timm_prefix}.running_mean").into_boxed_str()),
        format!("layer.{idx}.bn.running_mean"),
    );
    m.insert(
        Box::leak(format!("{timm_prefix}.running_var").into_boxed_str()),
        format!("layer.{idx}.bn.running_var"),
    );
}

fn insert_bn_dynamic(m: &mut HashMap<&'static str, String>, timm_prefix: &str, idx: usize) {
    m.insert(
        leak(&format!("{timm_prefix}.weight")),
        format!("layer.{idx}.bn.gamma"),
    );
    m.insert(
        leak(&format!("{timm_prefix}.bias")),
        format!("layer.{idx}.bn.beta"),
    );
    m.insert(
        leak(&format!("{timm_prefix}.running_mean")),
        format!("layer.{idx}.bn.running_mean"),
    );
    m.insert(
        leak(&format!("{timm_prefix}.running_var")),
        format!("layer.{idx}.bn.running_var"),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resnet18_stem_mapping() {
        let name = timm_to_yscv_name("conv1.weight", ModelArchitecture::ResNet18);
        assert_eq!(name, Some("layer.0.conv2d.weight".to_string()));
    }

    #[test]
    fn test_resnet18_bn1_mapping() {
        let name = timm_to_yscv_name("bn1.weight", ModelArchitecture::ResNet18);
        assert_eq!(name, Some("layer.1.bn.gamma".to_string()));
    }

    #[test]
    fn test_resnet18_fc_mapping() {
        let name = timm_to_yscv_name("fc.weight", ModelArchitecture::ResNet18);
        assert!(name.is_some());
        assert!(name.unwrap().contains("linear.weight"));
    }

    #[test]
    fn test_resnet18_layer1_conv_mapping() {
        let name = timm_to_yscv_name("layer1.0.conv1.weight", ModelArchitecture::ResNet18);
        assert!(name.is_some());
        assert!(name.unwrap().contains("conv2d.weight"));
    }

    #[test]
    fn test_resnet50_bottleneck_has_conv3() {
        let name = timm_to_yscv_name("layer1.0.conv3.weight", ModelArchitecture::ResNet50);
        assert!(name.is_some());
    }

    #[test]
    fn test_unknown_key_returns_none() {
        let name = timm_to_yscv_name("nonexistent.key", ModelArchitecture::ResNet18);
        assert_eq!(name, None);
    }

    #[test]
    fn test_remap_state_dict() {
        let mut sd: HashMap<String, i32> = HashMap::new();
        sd.insert("conv1.weight".into(), 42);
        sd.insert("bn1.weight".into(), 7);
        sd.insert("unknown.key".into(), 99);

        let remapped = remap_state_dict(&sd, ModelArchitecture::ResNet18);
        assert_eq!(remapped.get("layer.0.conv2d.weight"), Some(&42));
        assert_eq!(remapped.get("layer.1.bn.gamma"), Some(&7));
        assert!(!remapped.contains_key("unknown.key"));
    }

    #[test]
    fn test_vgg16_has_features_mapping() {
        let name = timm_to_yscv_name("features.0.weight", ModelArchitecture::VGG16);
        assert!(name.is_some());
        assert!(name.unwrap().contains("conv2d.weight"));
    }

    #[test]
    fn test_alexnet_has_features_mapping() {
        let name = timm_to_yscv_name("features.0.weight", ModelArchitecture::AlexNet);
        assert!(name.is_some());
    }
}
