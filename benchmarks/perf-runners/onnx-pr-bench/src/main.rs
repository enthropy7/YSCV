#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use serde_json::json;
use yscv_onnx::{
    OnnxExportAttr, OnnxExportGraph, OnnxExportNode, OnnxExportValueInfo, OnnxRunner,
    export_onnx_model_to_file, load_onnx_model_from_file, optimize_onnx_graph,
};
use yscv_tensor::Tensor;

#[derive(Clone, Copy)]
enum FillMode {
    Zero,
    Random,
}

enum Command {
    PrepareSmall {
        asset_dir: PathBuf,
    },
    RunCase {
        case: String,
        model: PathBuf,
        inputs: Vec<(String, Vec<usize>)>,
        image: Option<PathBuf>,
        image_input: Option<String>,
        image_size: usize,
        iters: usize,
        runs: usize,
        threads: usize,
        fill: FillMode,
        output: PathBuf,
    },
}

fn usage() -> String {
    "usage:
  yscv-onnx-pr-bench prepare-small --asset-dir DIR
  yscv-onnx-pr-bench run-case --case NAME --model M.onnx --input NAME:DxDxD \\
      [--image IMG --image-input NAME --image-size 640] [--iters N] [--runs N] \\
      [--threads N] [--fill zero|random] --output OUT.json"
        .to_string()
}

fn take_value<I>(it: &mut I, flag: &str) -> Result<String, String>
where
    I: Iterator<Item = String>,
{
    it.next().ok_or_else(|| format!("{flag} needs a value"))
}

fn parse_shape(spec: &str) -> Result<(String, Vec<usize>), String> {
    let (name, dims) = spec
        .split_once(':')
        .ok_or_else(|| "--input must be NAME:DxDxD".to_string())?;
    let shape = dims
        .split('x')
        .map(|d| {
            d.parse::<usize>()
                .map_err(|e| format!("bad dimension '{d}': {e}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if shape.is_empty() {
        return Err(format!("input '{name}' has empty shape"));
    }
    Ok((name.to_string(), shape))
}

fn parse_args() -> Result<Command, String> {
    let mut it = std::env::args().skip(1);
    match it.next().as_deref() {
        Some("prepare-small") => {
            let mut asset_dir = None;
            while let Some(flag) = it.next() {
                match flag.as_str() {
                    "--asset-dir" => asset_dir = Some(PathBuf::from(take_value(&mut it, &flag)?)),
                    "-h" | "--help" => return Err(usage()),
                    other => return Err(format!("unknown flag: {other}")),
                }
            }
            Ok(Command::PrepareSmall {
                asset_dir: asset_dir.ok_or("--asset-dir is required")?,
            })
        }
        Some("run-case") => {
            let mut case = None;
            let mut model = None;
            let mut inputs = Vec::new();
            let mut image = None;
            let mut image_input = None;
            let mut image_size = 640usize;
            let mut iters = 100usize;
            let mut runs = 3usize;
            let mut threads = 0usize;
            let mut fill = FillMode::Zero;
            let mut output = None;
            while let Some(flag) = it.next() {
                match flag.as_str() {
                    "--case" => case = Some(take_value(&mut it, &flag)?),
                    "--model" => model = Some(PathBuf::from(take_value(&mut it, &flag)?)),
                    "--input" => inputs.push(parse_shape(&take_value(&mut it, &flag)?)?),
                    "--image" => image = Some(PathBuf::from(take_value(&mut it, &flag)?)),
                    "--image-input" => image_input = Some(take_value(&mut it, &flag)?),
                    "--image-size" => {
                        image_size = take_value(&mut it, &flag)?
                            .parse()
                            .map_err(|e| format!("bad --image-size: {e}"))?;
                    }
                    "--iters" => {
                        iters = take_value(&mut it, &flag)?
                            .parse()
                            .map_err(|e| format!("bad --iters: {e}"))?;
                    }
                    "--runs" => {
                        runs = take_value(&mut it, &flag)?
                            .parse()
                            .map_err(|e| format!("bad --runs: {e}"))?;
                    }
                    "--threads" => {
                        threads = take_value(&mut it, &flag)?
                            .parse()
                            .map_err(|e| format!("bad --threads: {e}"))?;
                    }
                    "--fill" => {
                        fill = match take_value(&mut it, &flag)?.as_str() {
                            "zero" => FillMode::Zero,
                            "random" => FillMode::Random,
                            other => return Err(format!("bad --fill: {other}")),
                        };
                    }
                    "--output" => output = Some(PathBuf::from(take_value(&mut it, &flag)?)),
                    "-h" | "--help" => return Err(usage()),
                    other => return Err(format!("unknown flag: {other}")),
                }
            }
            if inputs.is_empty() {
                return Err("--input is required".to_string());
            }
            if image_size == 0 || iters == 0 || runs == 0 {
                return Err("--image-size, --iters, and --runs must be > 0".to_string());
            }
            Ok(Command::RunCase {
                case: case.ok_or("--case is required")?,
                model: model.ok_or("--model is required")?,
                inputs,
                image,
                image_input,
                image_size,
                iters,
                runs,
                threads,
                fill,
                output: output.ok_or("--output is required")?,
            })
        }
        _ => Err(usage()),
    }
}

fn patterned(n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let v = ((i.wrapping_mul(37).wrapping_add(11)) % 29) as f32 - 14.0;
            v * scale
        })
        .collect()
}

fn tensor(shape: Vec<usize>, data: Vec<f32>) -> Result<Tensor, String> {
    Tensor::from_vec(shape, data).map_err(|e| format!("tensor build failed: {e}"))
}

fn prepare_small(asset_dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(asset_dir)
        .map_err(|e| format!("create {}: {e}", asset_dir.display()))?;
    let conv = OnnxExportGraph {
        nodes: vec![
            OnnxExportNode {
                op_type: "Conv".to_string(),
                name: "conv".to_string(),
                inputs: vec![
                    "input".to_string(),
                    "weight".to_string(),
                    "bias".to_string(),
                ],
                outputs: vec!["conv_out".to_string()],
                attributes: vec![
                    OnnxExportAttr::Ints("kernel_shape".to_string(), vec![3, 3]),
                    OnnxExportAttr::Ints("pads".to_string(), vec![1, 1, 1, 1]),
                    OnnxExportAttr::Ints("strides".to_string(), vec![1, 1]),
                ],
            },
            OnnxExportNode {
                op_type: "Relu".to_string(),
                name: "relu".to_string(),
                inputs: vec!["conv_out".to_string()],
                outputs: vec!["output".to_string()],
                attributes: Vec::new(),
            },
        ],
        initializers: vec![
            (
                "weight".to_string(),
                tensor(vec![8, 3, 3, 3], patterned(8 * 3 * 3 * 3, 0.01))?,
            ),
            ("bias".to_string(), tensor(vec![8], patterned(8, 0.005))?),
        ],
        inputs: vec![OnnxExportValueInfo {
            name: "input".to_string(),
            shape: vec![1, 3, 64, 64],
        }],
        outputs: vec![OnnxExportValueInfo {
            name: "output".to_string(),
            shape: vec![1, 8, 64, 64],
        }],
        opset_version: 13,
        int64_initializers: Vec::new(),
    };
    export_onnx_model_to_file(
        &conv,
        "yscv-pr-bench",
        "small_conv_relu_64",
        &asset_dir.join("small-conv-relu-64.onnx"),
    )
    .map_err(|e| format!("export small conv: {e}"))?;

    let dw_pw = OnnxExportGraph {
        nodes: vec![
            OnnxExportNode {
                op_type: "Conv".to_string(),
                name: "depthwise".to_string(),
                inputs: vec![
                    "input".to_string(),
                    "dw_weight".to_string(),
                    "dw_bias".to_string(),
                ],
                outputs: vec!["dw_out".to_string()],
                attributes: vec![
                    OnnxExportAttr::Ints("kernel_shape".to_string(), vec![3, 3]),
                    OnnxExportAttr::Ints("pads".to_string(), vec![1, 1, 1, 1]),
                    OnnxExportAttr::Ints("strides".to_string(), vec![1, 1]),
                    OnnxExportAttr::Int("group".to_string(), 16),
                ],
            },
            OnnxExportNode {
                op_type: "Conv".to_string(),
                name: "pointwise".to_string(),
                inputs: vec![
                    "dw_out".to_string(),
                    "pw_weight".to_string(),
                    "pw_bias".to_string(),
                ],
                outputs: vec!["pw_out".to_string()],
                attributes: vec![
                    OnnxExportAttr::Ints("kernel_shape".to_string(), vec![1, 1]),
                    OnnxExportAttr::Ints("pads".to_string(), vec![0, 0, 0, 0]),
                    OnnxExportAttr::Ints("strides".to_string(), vec![1, 1]),
                ],
            },
            OnnxExportNode {
                op_type: "Relu".to_string(),
                name: "relu".to_string(),
                inputs: vec!["pw_out".to_string()],
                outputs: vec!["output".to_string()],
                attributes: Vec::new(),
            },
        ],
        initializers: vec![
            (
                "dw_weight".to_string(),
                tensor(vec![16, 1, 3, 3], patterned(16 * 3 * 3, 0.01))?,
            ),
            (
                "dw_bias".to_string(),
                tensor(vec![16], patterned(16, 0.002))?,
            ),
            (
                "pw_weight".to_string(),
                tensor(vec![24, 16, 1, 1], patterned(24 * 16, 0.006))?,
            ),
            (
                "pw_bias".to_string(),
                tensor(vec![24], patterned(24, 0.002))?,
            ),
        ],
        inputs: vec![OnnxExportValueInfo {
            name: "input".to_string(),
            shape: vec![1, 16, 64, 64],
        }],
        outputs: vec![OnnxExportValueInfo {
            name: "output".to_string(),
            shape: vec![1, 24, 64, 64],
        }],
        opset_version: 13,
        int64_initializers: Vec::new(),
    };
    export_onnx_model_to_file(
        &dw_pw,
        "yscv-pr-bench",
        "small_dw_pw_64",
        &asset_dir.join("small-dw-pw-64.onnx"),
    )
    .map_err(|e| format!("export small dw-pw: {e}"))?;

    let residual = OnnxExportGraph {
        nodes: vec![
            OnnxExportNode {
                op_type: "Conv".to_string(),
                name: "conv".to_string(),
                inputs: vec![
                    "input".to_string(),
                    "weight".to_string(),
                    "bias".to_string(),
                ],
                outputs: vec!["conv_out".to_string()],
                attributes: vec![
                    OnnxExportAttr::Ints("kernel_shape".to_string(), vec![3, 3]),
                    OnnxExportAttr::Ints("pads".to_string(), vec![1, 1, 1, 1]),
                    OnnxExportAttr::Ints("strides".to_string(), vec![1, 1]),
                ],
            },
            OnnxExportNode {
                op_type: "Add".to_string(),
                name: "residual_add".to_string(),
                inputs: vec!["conv_out".to_string(), "input".to_string()],
                outputs: vec!["add_out".to_string()],
                attributes: Vec::new(),
            },
            OnnxExportNode {
                op_type: "Relu".to_string(),
                name: "relu".to_string(),
                inputs: vec!["add_out".to_string()],
                outputs: vec!["output".to_string()],
                attributes: Vec::new(),
            },
        ],
        initializers: vec![
            (
                "weight".to_string(),
                tensor(vec![8, 8, 3, 3], patterned(8 * 8 * 3 * 3, 0.006))?,
            ),
            ("bias".to_string(), tensor(vec![8], patterned(8, 0.002))?),
        ],
        inputs: vec![OnnxExportValueInfo {
            name: "input".to_string(),
            shape: vec![1, 8, 64, 64],
        }],
        outputs: vec![OnnxExportValueInfo {
            name: "output".to_string(),
            shape: vec![1, 8, 64, 64],
        }],
        opset_version: 13,
        int64_initializers: Vec::new(),
    };
    export_onnx_model_to_file(
        &residual,
        "yscv-pr-bench",
        "small_residual_conv_64",
        &asset_dir.join("small-residual-conv-64.onnx"),
    )
    .map_err(|e| format!("export small residual: {e}"))?;

    let gemm = OnnxExportGraph {
        nodes: vec![
            OnnxExportNode {
                op_type: "Gemm".to_string(),
                name: "gemm".to_string(),
                inputs: vec![
                    "input".to_string(),
                    "weight".to_string(),
                    "bias".to_string(),
                ],
                outputs: vec!["gemm_out".to_string()],
                attributes: vec![
                    OnnxExportAttr::Float("alpha".to_string(), 1.0),
                    OnnxExportAttr::Float("beta".to_string(), 1.0),
                    OnnxExportAttr::Int("transB".to_string(), 1),
                ],
            },
            OnnxExportNode {
                op_type: "Relu".to_string(),
                name: "relu".to_string(),
                inputs: vec!["gemm_out".to_string()],
                outputs: vec!["output".to_string()],
                attributes: Vec::new(),
            },
        ],
        initializers: vec![
            (
                "weight".to_string(),
                tensor(vec![64, 128], patterned(64 * 128, 0.002))?,
            ),
            ("bias".to_string(), tensor(vec![64], patterned(64, 0.001))?),
        ],
        inputs: vec![OnnxExportValueInfo {
            name: "input".to_string(),
            shape: vec![1, 128],
        }],
        outputs: vec![OnnxExportValueInfo {
            name: "output".to_string(),
            shape: vec![1, 64],
        }],
        opset_version: 13,
        int64_initializers: Vec::new(),
    };
    export_onnx_model_to_file(
        &gemm,
        "yscv-pr-bench",
        "small_gemm_relu_1x128",
        &asset_dir.join("small-gemm-relu-1x128.onnx"),
    )
    .map_err(|e| format!("export small gemm: {e}"))
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

fn make_tensor(shape: &[usize], fill: FillMode, seed: u32) -> Result<Tensor, String> {
    let len = shape.iter().product();
    let data = match fill {
        FillMode::Zero => vec![0.0; len],
        FillMode::Random => {
            let mut rng = XorShift::new(seed);
            (0..len).map(|_| rng.next_f32()).collect()
        }
    };
    Tensor::from_vec(shape.to_vec(), data).map_err(|e| format!("input tensor build failed: {e}"))
}

fn load_image_tensor(path: &Path, target_size: usize) -> Result<Tensor, String> {
    let img = image::ImageReader::open(path)
        .map_err(|e| format!("open image {}: {e}", path.display()))?
        .decode()
        .map_err(|e| format!("decode image {}: {e}", path.display()))?
        .to_rgb8();
    let (width, height) = img.dimensions();
    let rgb: Vec<f32> = img.as_raw().iter().map(|&v| v as f32 / 255.0).collect();
    let hwc = Tensor::from_vec(vec![height as usize, width as usize, 3], rgb)
        .map_err(|e| format!("image tensor build failed: {e}"))?;
    let (letterboxed, _, _, _) = yscv_detect::letterbox_preprocess(&hwc, target_size);
    let src = letterboxed.data();
    let hw = target_size * target_size;
    let mut nchw = vec![0.0; 3 * hw];
    for y in 0..target_size {
        for x in 0..target_size {
            let s = (y * target_size + x) * 3;
            let d = y * target_size + x;
            nchw[d] = src[s];
            nchw[hw + d] = src[s + 1];
            nchw[2 * hw + d] = src[s + 2];
        }
    }
    Tensor::from_vec(vec![1, 3, target_size, target_size], nchw)
        .map_err(|e| format!("image NCHW tensor build failed: {e}"))
}

fn percentile(sorted: &[u64], p: f64) -> u64 {
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn summarize(mut samples: Vec<u64>) -> serde_json::Value {
    samples.sort_unstable();
    let avg = samples.iter().sum::<u64>() / samples.len() as u64;
    json!({
        "min_us": samples[0],
        "p50_us": percentile(&samples, 0.50),
        "avg_us": avg,
        "p95_us": percentile(&samples, 0.95),
        "p99_us": percentile(&samples, 0.99),
        "max_us": samples[samples.len() - 1],
    })
}

fn summarize_profile(path: &Path) -> Result<serde_json::Value, String> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Ok(json!({
                "profiled": false,
                "reason": "runner profile was empty"
            }));
        }
        Err(e) => return Err(format!("read profile {}: {e}", path.display())),
    };
    let value: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("parse profile JSON: {e}"))?;
    let nodes = value["nodes"].as_array().cloned().unwrap_or_default();
    let mut op_counts: BTreeMap<String, u64> = BTreeMap::new();
    let mut op_invocations: BTreeMap<String, u64> = BTreeMap::new();
    let mut kernel_counts: BTreeMap<String, u64> = BTreeMap::new();
    let mut op_kernel_counts: BTreeMap<String, u64> = BTreeMap::new();
    let mut top_nodes = Vec::new();

    for node in &nodes {
        let op = node["op"].as_str().unwrap_or("unknown").to_string();
        let count = node["count"].as_u64().unwrap_or(0);
        *op_counts.entry(op.clone()).or_default() += 1;
        *op_invocations.entry(op.clone()).or_default() += count;
        if let Some(kernel) = node["kernel"].as_str() {
            *kernel_counts.entry(kernel.to_string()).or_default() += 1;
            *op_kernel_counts
                .entry(format!("{op} via {kernel}"))
                .or_default() += 1;
        }
        top_nodes.push(node.clone());
    }

    top_nodes.sort_by(|a, b| {
        b["ms"]
            .as_f64()
            .unwrap_or(0.0)
            .partial_cmp(&a["ms"].as_f64().unwrap_or(0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    top_nodes.truncate(8);

    Ok(json!({
        "profiled": true,
        "total_ms": value["total_ms"].as_f64().unwrap_or(0.0),
        "iterations": value["iterations"].as_u64().unwrap_or(0),
        "node_count": nodes.len(),
        "op_counts": op_counts,
        "op_invocations": op_invocations,
        "kernel_counts": kernel_counts,
        "op_kernel_counts": op_kernel_counts,
        "top_nodes": top_nodes,
    }))
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    case: &str,
    model_path: &Path,
    input_specs: &[(String, Vec<usize>)],
    image: Option<&Path>,
    image_input: Option<&str>,
    image_size: usize,
    iters: usize,
    runs: usize,
    threads: usize,
    fill: FillMode,
    output: &Path,
) -> Result<(), String> {
    let profile_path = output.with_extension("profile.json");
    // SAFETY: set before the runner is constructed and before any benchmark
    // worker threads are spawned. The process handles one case and then exits.
    unsafe {
        std::env::set_var("YSCV_RUNNER_PROFILE", &profile_path);
    }
    let mut model = load_onnx_model_from_file(model_path)
        .map_err(|e| format!("load {}: {e}", model_path.display()))?;
    let nodes_before = model.node_count();
    optimize_onnx_graph(&mut model);
    let nodes_after = model.node_count();
    let runner = if threads == 0 {
        OnnxRunner::new(&model).map_err(|e| format!("runner init: {e}"))?
    } else {
        OnnxRunner::with_threads(&model, threads).map_err(|e| format!("runner init: {e}"))?
    };

    let image_name = if image.is_some() {
        Some(image_input.unwrap_or_else(|| {
            if input_specs.len() == 1 {
                input_specs[0].0.as_str()
            } else {
                "images"
            }
        }))
    } else {
        None
    };
    let image_tensor = match image {
        Some(path) => Some(load_image_tensor(path, image_size)?),
        None => None,
    };
    if let (Some(name), Some(tensor)) = (image_name, image_tensor.as_ref()) {
        let Some((_, shape)) = input_specs.iter().find(|(n, _)| n == name) else {
            return Err(format!("image input '{name}' does not match any --input"));
        };
        if shape.as_slice() != tensor.shape() {
            return Err(format!(
                "image input '{name}' has shape {:?}, expected {:?}",
                tensor.shape(),
                shape
            ));
        }
    }

    let inputs: Vec<(String, Tensor)> = input_specs
        .iter()
        .enumerate()
        .map(|(i, (name, shape))| {
            if Some(name.as_str()) == image_name {
                Ok((
                    name.clone(),
                    image_tensor.as_ref().expect("checked").clone(),
                ))
            } else {
                make_tensor(shape, fill, 0xC0FFEE ^ i as u32).map(|t| (name.clone(), t))
            }
        })
        .collect::<Result<_, _>>()?;
    let feed: Vec<(&str, &Tensor)> = inputs.iter().map(|(n, t)| (n.as_str(), t)).collect();

    let mut run_summaries = Vec::with_capacity(runs);
    for run_idx in 0..runs {
        runner
            .run(&feed)
            .map_err(|e| format!("warmup failed: {e}"))?;
        let mut samples = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let _ = runner.run(&feed).map_err(|e| format!("run failed: {e}"))?;
            samples.push(t0.elapsed().as_micros() as u64);
        }
        let mut summary = summarize(samples);
        summary["run"] = json!(run_idx);
        run_summaries.push(summary);
    }

    let mut p50s: Vec<u64> = run_summaries
        .iter()
        .filter_map(|r| r["p50_us"].as_u64())
        .collect();
    p50s.sort_unstable();
    let median_p50_us = p50s[p50s.len() / 2];
    yscv_onnx::dump_runner_profile(&profile_path.to_string_lossy())
        .map_err(|e| format!("dump runner profile: {e}"))?;
    let profile_summary = summarize_profile(&profile_path)?;
    let report = json!({
        "case": case,
        "model": model_path.file_name().and_then(|s| s.to_str()).unwrap_or("model.onnx"),
        "arch": std::env::consts::ARCH,
        "os": std::env::consts::OS,
        "target": option_env!("TARGET").unwrap_or("unknown"),
        "threads": runner.num_threads(),
        "iters": iters,
        "runs": run_summaries,
        "median_p50_us": median_p50_us,
        "nodes_before_opt": nodes_before,
        "nodes_after_opt": nodes_after,
        "dispatch": yscv_kernels::runtime_dispatch_report().to_string(),
        "profile_summary": profile_summary,
    });
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    std::fs::write(
        output,
        serde_json::to_vec_pretty(&report).map_err(|e| format!("encode json: {e}"))?,
    )
    .map_err(|e| format!("write {}: {e}", output.display()))
}

fn run() -> Result<(), String> {
    match parse_args()? {
        Command::PrepareSmall { asset_dir } => prepare_small(&asset_dir),
        Command::RunCase {
            case,
            model,
            inputs,
            image,
            image_input,
            image_size,
            iters,
            runs,
            threads,
            fill,
            output,
        } => run_case(
            &case,
            &model,
            &inputs,
            image.as_deref(),
            image_input.as_deref(),
            image_size,
            iters,
            runs,
            threads,
            fill,
            &output,
        ),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
