use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

use serde_json::Value;
use yscv_onnx::{OnnxModel, OnnxRunner, load_onnx_model_from_file, optimize_onnx_graph};
use yscv_tensor::Tensor;

struct CaseSpec {
    name: String,
    model: String,
    inputs: Vec<(String, Vec<usize>)>,
    fill_random: bool,
    image: Option<String>,
    image_input: Option<String>,
    image_size: usize,
}

struct PreparedCase {
    name: String,
    model: OnnxModel,
    inputs: Vec<(String, Tensor)>,
}

struct XorShift(u32);

impl XorShift {
    fn new(seed: u32) -> Self {
        Self(seed.max(1))
    }

    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0 as f32 / u32::MAX as f32
    }
}

fn asset_dir() -> PathBuf {
    env::var_os("YSCV_ONNX_PR_BENCH_ASSET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/criterion-assets")
        })
}

fn run_command(command: &mut Command, description: &str) {
    let status = command
        .status()
        .unwrap_or_else(|error| panic!("{description}: {error}"));
    assert!(status.success(), "{description} exited with {status}");
}

fn prepare_assets(asset_dir: &Path) {
    fs::create_dir_all(asset_dir).expect("create Criterion asset directory");
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    run_command(
        Command::new("bash")
            .arg(manifest_dir.join("download-assets.sh"))
            .arg(asset_dir),
        "download Criterion benchmark assets",
    );
}

fn required_str<'a>(value: &'a Value, field: &str, case: &str) -> &'a str {
    value[field]
        .as_str()
        .unwrap_or_else(|| panic!("case '{case}' has no string '{field}'"))
}

fn suite_cases() -> Vec<CaseSpec> {
    let suite: Value =
        serde_json::from_str(include_str!("../suite.json")).expect("parse suite.json");
    let model_assets: Vec<_> = suite["assets"]
        .as_array()
        .expect("suite.json assets array")
        .iter()
        .filter_map(|asset| asset["path"].as_str())
        .filter(|path| path.ends_with(".onnx"))
        .collect();
    suite["cases"]
        .as_array()
        .expect("suite.json cases array")
        .iter()
        .filter(|case| {
            let case_name = case["name"].as_str().unwrap_or("unknown");
            let model = required_str(case, "model", case_name);
            model_assets.contains(&model)
        })
        .map(|case| {
            let name = required_str(case, "name", "unknown").to_string();
            let inputs = case["inputs"]
                .as_array()
                .unwrap_or_else(|| panic!("case '{name}' has no inputs"))
                .iter()
                .map(|input| {
                    let input_name = required_str(input, "name", &name).to_string();
                    let shape = input["shape"]
                        .as_array()
                        .unwrap_or_else(|| {
                            panic!("case '{name}' input '{input_name}' has no shape")
                        })
                        .iter()
                        .map(|dimension| {
                            dimension.as_u64().unwrap_or_else(|| {
                                panic!("case '{name}' input '{input_name}' has a non-integer shape")
                            }) as usize
                        })
                        .collect();
                    (input_name, shape)
                })
                .collect();
            let has_random_input = case["inputs"]
                .as_array()
                .is_some_and(|inputs| inputs.iter().any(|input| input["source"] == "random"));
            CaseSpec {
                name: name.clone(),
                model: required_str(case, "model", &name).to_string(),
                inputs,
                fill_random: case["fill"] == "random" || has_random_input,
                image: case["image"].as_str().map(ToOwned::to_owned),
                image_input: case["image_input"].as_str().map(ToOwned::to_owned),
                image_size: case["image_size"].as_u64().unwrap_or(640) as usize,
            }
        })
        .collect()
}

fn random_tensor(shape: &[usize], seed: u32) -> Tensor {
    let mut rng = XorShift::new(seed);
    let data = (0..shape.iter().product())
        .map(|_| rng.next_f32())
        .collect();
    Tensor::from_vec(shape.to_vec(), data).expect("build random benchmark input")
}

fn image_tensor(path: &Path, target_size: usize) -> Tensor {
    let image = image::ImageReader::open(path)
        .expect("open benchmark image")
        .decode()
        .expect("decode benchmark image")
        .to_rgb8();
    let (width, height) = image.dimensions();
    let rgb = image
        .as_raw()
        .iter()
        .map(|&value| value as f32 / 255.0)
        .collect();
    let hwc = Tensor::from_vec(vec![height as usize, width as usize, 3], rgb)
        .expect("build benchmark image tensor");
    let (letterboxed, _, _, _) = yscv_detect::letterbox_preprocess(&hwc, target_size);
    let src = letterboxed.data();
    let hw = target_size * target_size;
    let mut nchw = vec![0.0; 3 * hw];
    for y in 0..target_size {
        for x in 0..target_size {
            let source = (y * target_size + x) * 3;
            let destination = y * target_size + x;
            nchw[destination] = src[source];
            nchw[hw + destination] = src[source + 1];
            nchw[2 * hw + destination] = src[source + 2];
        }
    }
    Tensor::from_vec(vec![1, 3, target_size, target_size], nchw)
        .expect("build NCHW benchmark image tensor")
}

fn prepare_case(spec: CaseSpec, asset_dir: &Path) -> PreparedCase {
    let mut model = load_onnx_model_from_file(asset_dir.join(&spec.model))
        .unwrap_or_else(|error| panic!("load {}: {error}", spec.model));
    optimize_onnx_graph(&mut model);
    let image_name = spec.image.as_ref().map(|_| {
        spec.image_input.clone().unwrap_or_else(|| {
            if spec.inputs.len() == 1 {
                spec.inputs[0].0.clone()
            } else {
                "images".to_string()
            }
        })
    });
    let image = spec
        .image
        .as_ref()
        .map(|path| image_tensor(&asset_dir.join(path), spec.image_size));
    let inputs = spec
        .inputs
        .iter()
        .enumerate()
        .map(|(index, (name, shape))| {
            let tensor = if image_name.as_deref() == Some(name) {
                let tensor = image.as_ref().expect("image input requires an image asset");
                assert_eq!(
                    tensor.shape(),
                    shape,
                    "image shape for case '{}'",
                    spec.name
                );
                tensor.clone()
            } else if spec.fill_random {
                random_tensor(shape, 0xC0FFEE ^ index as u32)
            } else {
                Tensor::zeros(shape.clone()).expect("build zero benchmark input")
            };
            (name.clone(), tensor)
        })
        .collect();

    PreparedCase {
        name: spec.name,
        model,
        inputs,
    }
}

fn bench_onnx_models(criterion: &mut Criterion) {
    let asset_dir = asset_dir();
    prepare_assets(&asset_dir);
    let cases: Vec<_> = suite_cases()
        .into_iter()
        .map(|case| prepare_case(case, &asset_dir))
        .collect();
    let mut group = criterion.benchmark_group("onnx_models");

    for case in &cases {
        let runner = OnnxRunner::new(&case.model).expect("initialize ONNX runner");
        let feed: Vec<_> = case
            .inputs
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect();
        runner.run(&feed).expect("warm up ONNX model");
        group.bench_function(BenchmarkId::from_parameter(&case.name), |bencher| {
            bencher.iter(|| {
                black_box(runner.run(black_box(&feed)).expect("run ONNX model"));
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_onnx_models);
criterion_main!(benches);
