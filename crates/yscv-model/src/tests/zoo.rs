use yscv_autograd::Graph;

#[test]
fn zoo_resnet18_config_is_correct() {
    let cfg = crate::ModelArchitecture::ResNet18.config();
    assert_eq!(cfg.input_channels, 3);
    assert_eq!(cfg.num_classes, 1000);
    assert_eq!(cfg.stage_channels, vec![64, 128, 256, 512]);
    assert_eq!(cfg.blocks_per_stage, vec![2, 2, 2, 2]);
}

#[test]
fn zoo_architecture_names() {
    assert_eq!(crate::ModelArchitecture::ResNet50.name(), "resnet50");
    assert_eq!(crate::ModelArchitecture::VGG16.name(), "vgg16");
    assert_eq!(crate::ModelArchitecture::MobileNetV2.name(), "mobilenet_v2");
    assert_eq!(crate::ModelArchitecture::AlexNet.name(), "alexnet");
}

#[test]
fn zoo_all_architectures_listed() {
    let all = crate::ModelArchitecture::all();
    assert_eq!(all.len(), 17);
}

#[test]
fn zoo_build_classifier_creates_model() {
    let mut graph = Graph::new();
    let model = crate::build_classifier(crate::ModelArchitecture::AlexNet, &mut graph, 10).unwrap();
    assert!(!model.layers().is_empty());
}

#[test]
fn zoo_config_with_num_classes() {
    let cfg = crate::ModelArchitecture::ResNet18
        .config()
        .with_num_classes(10);
    assert_eq!(cfg.num_classes, 10);
    assert_eq!(cfg.input_channels, 3);
}

#[test]
fn zoo_save_load_roundtrip() {
    let mut graph = Graph::new();
    let model = crate::build_classifier(crate::ModelArchitecture::AlexNet, &mut graph, 10).unwrap();

    let dir = std::env::temp_dir().join(format!("yscv-zoo-test-{}", std::process::id()));
    let zoo = crate::ModelZoo::new(&dir);
    zoo.save_pretrained(crate::ModelArchitecture::AlexNet, &model, &graph)
        .unwrap();

    let available = zoo.list_available();
    assert!(available.contains(&crate::ModelArchitecture::AlexNet));

    // Cleanup
    let _ = std::fs::remove_dir_all(&dir);
}
