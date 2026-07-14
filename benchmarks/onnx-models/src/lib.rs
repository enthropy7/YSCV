use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;
use yscv_tensor::Tensor;

#[derive(Clone, Copy)]
pub enum FillMode {
    Zero,
    Random,
}

pub struct ModelCase {
    pub name: String,
    pub model: String,
    pub inputs: Vec<(String, Vec<usize>)>,
    pub fill: FillMode,
    pub image: Option<String>,
    pub image_input: Option<String>,
    pub image_size: usize,
}

struct XorShift(u32);

impl XorShift {
    fn new(seed: u32) -> Self {
        Self(seed.max(1))
    }

    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x as f32 / u32::MAX as f32
    }
}

pub fn asset_dir() -> PathBuf {
    env::var_os("YSCV_ONNX_MODEL_BENCH_ASSET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/assets"))
}

pub fn download_assets(asset_dir: &Path) -> Result<(), String> {
    fs::create_dir_all(asset_dir).map_err(|error| format!("create asset directory: {error}"))?;
    let status = Command::new("bash")
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("download-assets.sh"))
        .arg(asset_dir)
        .status()
        .map_err(|error| format!("start asset download: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("asset download exited with {status}"))
    }
}

fn required_str<'a>(value: &'a Value, field: &str, case: &str) -> Result<&'a str, String> {
    value[field]
        .as_str()
        .ok_or_else(|| format!("case '{case}' has no string '{field}'"))
}

pub fn model_cases() -> Result<Vec<ModelCase>, String> {
    let suite: Value = serde_json::from_str(include_str!("../suite.json"))
        .map_err(|error| format!("parse suite.json: {error}"))?;
    let assets = suite["assets"]
        .as_array()
        .ok_or("suite.json has no assets array")?;
    let model_assets: Vec<_> = assets
        .iter()
        .filter_map(|asset| asset["path"].as_str())
        .filter(|path| path.ends_with(".onnx"))
        .collect();
    suite["cases"]
        .as_array()
        .ok_or("suite.json has no cases array")?
        .iter()
        .filter(|case| {
            let case_name = case["name"].as_str().unwrap_or("unknown");
            required_str(case, "model", case_name).is_ok_and(|model| model_assets.contains(&model))
        })
        .map(|case| {
            let name = required_str(case, "name", "unknown")?.to_string();
            let inputs = case["inputs"]
                .as_array()
                .ok_or_else(|| format!("case '{name}' has no inputs"))?
                .iter()
                .map(|input| {
                    let input_name = required_str(input, "name", &name)?.to_string();
                    let shape = input["shape"]
                        .as_array()
                        .ok_or_else(|| format!("case '{name}' input '{input_name}' has no shape"))?
                        .iter()
                        .map(|dimension| {
                            dimension
                                .as_u64()
                                .map(|dimension| dimension as usize)
                                .ok_or_else(|| {
                                    format!(
                                        "case '{name}' input '{input_name}' has a non-integer shape"
                                    )
                                })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok((input_name, shape))
                })
                .collect::<Result<Vec<_>, String>>()?;
            let has_random_input = case["inputs"]
                .as_array()
                .is_some_and(|inputs| inputs.iter().any(|input| input["source"] == "random"));
            Ok(ModelCase {
                model: required_str(case, "model", &name)?.to_string(),
                inputs,
                fill: if case["fill"] == "random" || has_random_input {
                    FillMode::Random
                } else {
                    FillMode::Zero
                },
                image: case["image"].as_str().map(ToOwned::to_owned),
                image_input: case["image_input"].as_str().map(ToOwned::to_owned),
                image_size: case["image_size"].as_u64().unwrap_or(640) as usize,
                name,
            })
        })
        .collect()
}

pub fn make_tensor(shape: &[usize], fill: FillMode, seed: u32) -> Result<Tensor, String> {
    let len = shape.iter().product();
    let data = match fill {
        FillMode::Zero => vec![0.0; len],
        FillMode::Random => {
            let mut rng = XorShift::new(seed);
            (0..len).map(|_| rng.next_f32()).collect()
        }
    };
    Tensor::from_vec(shape.to_vec(), data)
        .map_err(|error| format!("input tensor build failed: {error}"))
}

pub fn load_image_tensor(path: &Path, target_size: usize) -> Result<Tensor, String> {
    let image = image::ImageReader::open(path)
        .map_err(|error| format!("open image {}: {error}", path.display()))?
        .decode()
        .map_err(|error| format!("decode image {}: {error}", path.display()))?
        .to_rgb8();
    let (width, height) = image.dimensions();
    let rgb = image
        .as_raw()
        .iter()
        .map(|&value| value as f32 / 255.0)
        .collect();
    let hwc = Tensor::from_vec(vec![height as usize, width as usize, 3], rgb)
        .map_err(|error| format!("image tensor build failed: {error}"))?;
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
        .map_err(|error| format!("image NCHW tensor build failed: {error}"))
}

pub fn make_inputs(case: &ModelCase, asset_dir: &Path) -> Result<Vec<(String, Tensor)>, String> {
    let image_name = case.image.as_ref().map(|_| {
        case.image_input.clone().unwrap_or_else(|| {
            if case.inputs.len() == 1 {
                case.inputs[0].0.clone()
            } else {
                "images".to_string()
            }
        })
    });
    let image = case
        .image
        .as_ref()
        .map(|path| load_image_tensor(&asset_dir.join(path), case.image_size))
        .transpose()?;
    case.inputs
        .iter()
        .enumerate()
        .map(|(index, (name, shape))| {
            if image_name.as_deref() == Some(name) {
                let tensor = image
                    .as_ref()
                    .ok_or("image input requires an image asset")?;
                if tensor.shape() != shape {
                    return Err(format!(
                        "image shape for case '{}' does not match input",
                        case.name
                    ));
                }
                Ok((name.clone(), tensor.clone()))
            } else {
                Ok((
                    name.clone(),
                    make_tensor(shape, case.fill, 0xC0FFEE ^ index as u32)?,
                ))
            }
        })
        .collect()
}
