//! `yscv-quantize` — post-training quantization CLI.
//!
//! Loads an fp32 ONNX model, runs it on a calibration dataset to collect
//! per-tensor activation statistics, rewrites Conv / MatMul / Gemm nodes
//! into QDQ format with per-channel int8 weights, and saves the result
//! as a new ONNX file.
//!
//! ```text
//! yscv-quantize <input.onnx> --output <output.onnx>
//!                            [--calibration <samples.jsonl>]
//!                            [--calibration name1=stream1.jsonl,name2=stream2.jsonl]
//!                            [--format qdq|qlinear]
//!                            [--weights-only]
//! ```
//!
//! Without `--calibration` the tool runs in weight-only mode: only Conv /
//! MatMul / Gemm initializers are quantized, activations stay fp32. With
//! `--calibration`, each line of the jsonl file is one sample feeding
//! every named graph input.
//!
//! ## Calibration JSONL format
//!
//! One JSON object per line. Top-level keys are graph input names; values
//! are objects with `shape` (list of `usize`) and `values` (flat row-major
//! list of `f32`).
//!
//! ```json
//! {"input": {"shape": [1, 3, 224, 224], "values": [0.1, 0.2, ...]}}
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::ExitCode;

use serde::Deserialize;
use yscv_onnx::{
    CalibrationCollector, OnnxRunner, load_onnx_model_from_file, optimize_onnx_graph,
    prune_unused_initializers, rewrite_to_qdq, rewrite_to_qlinear, save_onnx_model_to_file,
    strip_qdq_within_fusion_chains,
};
use yscv_tensor::Tensor;

#[derive(Debug, thiserror::Error)]
enum CliError {
    #[error(
        "usage: yscv-quantize <input.onnx> --output <output.onnx> [--calibration <samples.jsonl>] [--format qdq|qlinear] [--weights-only] [--strip-inner-qdq]"
    )]
    Usage,
    #[error("missing required argument: {0}")]
    MissingArg(&'static str),
    #[error("invalid argument: {0}")]
    InvalidArg(String),
    #[error("io: {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("calibration sample {line}: {message}")]
    BadSample { line: usize, message: String },
    #[error("yscv-onnx: {0}")]
    Onnx(#[from] yscv_onnx::OnnxError),
}

#[derive(Debug)]
struct Args {
    input: PathBuf,
    output: PathBuf,
    calibration: Option<String>,
    format: QuantFormat,
    weights_only: bool,
    strip_inner_qdq: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QuantFormat {
    Qdq,
    QLinear,
}

fn parse_args() -> Result<Args, CliError> {
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut calibration: Option<String> = None;
    let mut format = QuantFormat::Qdq;
    let mut weights_only = false;
    let mut strip_inner_qdq = false;

    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "-h" | "--help" => return Err(CliError::Usage),
            "--output" | "-o" => {
                output = Some(PathBuf::from(
                    iter.next().ok_or(CliError::MissingArg("--output value"))?,
                ));
            }
            "--calibration" | "-c" => {
                calibration = Some(
                    iter.next()
                        .ok_or(CliError::MissingArg("--calibration value"))?,
                );
            }
            "--format" => {
                let value = iter.next().ok_or(CliError::MissingArg("--format value"))?;
                format = match value.as_str() {
                    "qdq" => QuantFormat::Qdq,
                    "qlinear" => QuantFormat::QLinear,
                    _ => {
                        return Err(CliError::InvalidArg(format!(
                            "--format must be `qdq` or `qlinear`, got `{value}`"
                        )));
                    }
                };
            }
            "--weights-only" => weights_only = true,
            "--strip-inner-qdq" => strip_inner_qdq = true,
            other if other.starts_with('-') => {
                return Err(CliError::InvalidArg(other.to_string()));
            }
            other => {
                if input.is_some() {
                    return Err(CliError::InvalidArg(format!(
                        "unexpected positional `{other}` (input.onnx already set)"
                    )));
                }
                input = Some(PathBuf::from(other));
            }
        }
    }

    Ok(Args {
        input: input.ok_or(CliError::MissingArg("input.onnx"))?,
        output: output.ok_or(CliError::MissingArg("--output"))?,
        calibration,
        format,
        weights_only,
        strip_inner_qdq,
    })
}

#[derive(Debug, Deserialize)]
struct SampleTensor {
    shape: Vec<usize>,
    values: Vec<f32>,
}

/// Read calibration samples from a JSONL file. Each line is a
/// `HashMap<String, SampleTensor>` mapping graph-input names to tensors.
fn read_calibration(path: &PathBuf) -> Result<Vec<HashMap<String, Tensor>>, CliError> {
    let text = std::fs::read_to_string(path).map_err(|e| CliError::Io {
        path: path.clone(),
        source: e,
    })?;
    let mut samples = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parsed: HashMap<String, SampleTensor> =
            serde_json::from_str(line).map_err(|e| CliError::BadSample {
                line: idx + 1,
                message: e.to_string(),
            })?;
        let mut sample = HashMap::new();
        for (name, st) in parsed {
            let expected: usize = st.shape.iter().product();
            if expected != st.values.len() {
                return Err(CliError::BadSample {
                    line: idx + 1,
                    message: format!(
                        "tensor `{name}`: shape {:?} expects {expected} values, got {}",
                        st.shape,
                        st.values.len()
                    ),
                });
            }
            let tensor =
                Tensor::from_vec(st.shape, st.values).map_err(|e| CliError::BadSample {
                    line: idx + 1,
                    message: format!("tensor `{name}`: {e}"),
                })?;
            sample.insert(name, tensor);
        }
        samples.push(sample);
    }
    Ok(samples)
}

fn read_calibration_spec(spec: &str) -> Result<Vec<HashMap<String, Tensor>>, CliError> {
    if spec.contains('=') {
        read_paired_calibration(spec)
    } else {
        read_calibration(&PathBuf::from(spec))
    }
}

fn read_paired_calibration(spec: &str) -> Result<Vec<HashMap<String, Tensor>>, CliError> {
    let streams: Vec<(String, PathBuf)> = spec
        .split(',')
        .map(|pair| {
            let (name, path) = pair.split_once('=').ok_or_else(|| {
                CliError::InvalidArg(format!(
                    "paired calibration entry `{pair}` must be NAME=PATH"
                ))
            })?;
            if name.is_empty() || path.is_empty() {
                return Err(CliError::InvalidArg(format!(
                    "paired calibration entry `{pair}` must be NAME=PATH"
                )));
            }
            Ok((name.to_string(), PathBuf::from(path)))
        })
        .collect::<Result<_, _>>()?;
    if streams.is_empty() {
        return Err(CliError::InvalidArg(
            "paired calibration spec must contain at least one stream".to_string(),
        ));
    }

    let mut parsed: Vec<(String, Vec<Tensor>)> = Vec::with_capacity(streams.len());
    for (name, path) in streams {
        parsed.push((name.clone(), read_single_tensor_stream(&name, &path)?));
    }
    let expected_len = parsed[0].1.len();
    for (name, tensors) in &parsed {
        if tensors.len() != expected_len {
            return Err(CliError::InvalidArg(format!(
                "paired calibration stream `{name}` has {} samples, expected {expected_len}",
                tensors.len()
            )));
        }
    }

    let mut samples = Vec::with_capacity(expected_len);
    for idx in 0..expected_len {
        let mut sample = HashMap::with_capacity(parsed.len());
        for (name, tensors) in &parsed {
            sample.insert(name.clone(), tensors[idx].clone());
        }
        samples.push(sample);
    }
    Ok(samples)
}

fn read_single_tensor_stream(name: &str, path: &PathBuf) -> Result<Vec<Tensor>, CliError> {
    let text = std::fs::read_to_string(path).map_err(|e| CliError::Io {
        path: path.clone(),
        source: e,
    })?;
    let mut tensors = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let st: SampleTensor = match serde_json::from_str(line) {
            Ok(st) => st,
            Err(_) => {
                let wrapped: HashMap<String, SampleTensor> =
                    serde_json::from_str(line).map_err(|e| CliError::BadSample {
                        line: idx + 1,
                        message: e.to_string(),
                    })?;
                wrapped
                    .into_iter()
                    .find(|(k, _)| k == name)
                    .map(|(_, v)| v)
                    .ok_or_else(|| CliError::BadSample {
                        line: idx + 1,
                        message: format!("wrapped sample does not contain `{name}`"),
                    })?
            }
        };
        let expected: usize = st.shape.iter().product();
        if expected != st.values.len() {
            return Err(CliError::BadSample {
                line: idx + 1,
                message: format!(
                    "tensor `{name}`: shape {:?} expects {expected} values, got {}",
                    st.shape,
                    st.values.len()
                ),
            });
        }
        tensors.push(
            Tensor::from_vec(st.shape, st.values).map_err(|e| CliError::BadSample {
                line: idx + 1,
                message: format!("tensor `{name}`: {e}"),
            })?,
        );
    }
    Ok(tensors)
}

fn run(args: Args) -> Result<(), CliError> {
    eprintln!("loading {}…", args.input.display());
    let mut model = load_onnx_model_from_file(&args.input)?;
    optimize_onnx_graph(&mut model);

    let cal_path = if args.weights_only {
        None
    } else {
        args.calibration.as_ref()
    };
    let stats = if let Some(cal_path) = cal_path {
        eprintln!("loading calibration samples from {cal_path}…");
        let samples = read_calibration_spec(cal_path)?;
        if samples.is_empty() {
            return Err(CliError::BadSample {
                line: 0,
                message: "calibration file produced 0 samples".to_string(),
            });
        }
        eprintln!(
            "running {} sample(s) through the model to collect activation stats…",
            samples.len()
        );
        let collector = CalibrationCollector::new();
        let runner = OnnxRunner::new(&model)?;
        {
            let _scope = collector.scope();
            for (idx, sample) in samples.iter().enumerate() {
                let feed: Vec<(&str, &Tensor)> =
                    sample.iter().map(|(k, v)| (k.as_str(), v)).collect();
                runner.run(&feed).map_err(|e| {
                    eprintln!("sample {idx}: inference failed: {e}");
                    CliError::Onnx(e)
                })?;
            }
        }
        let snap = collector.snapshot();
        eprintln!("collected stats for {} tensor(s)", snap.len());
        snap
    } else {
        eprintln!("calibration: weights-only mode, no activation stats collected");
        HashMap::new()
    };

    eprintln!(
        "rewriting model to {} format…",
        match args.format {
            QuantFormat::Qdq => "QDQ",
            QuantFormat::QLinear => "QLinear",
        }
    );
    match args.format {
        QuantFormat::Qdq => rewrite_to_qdq(&mut model, &stats)?,
        QuantFormat::QLinear => rewrite_to_qlinear(&mut model, &stats)?,
    }

    if args.strip_inner_qdq && args.format == QuantFormat::Qdq {
        let removed = strip_qdq_within_fusion_chains(&mut model);
        eprintln!("strip-inner-qdq: removed {removed} Q+DQ pair(s) between Conv-like ops");
    } else if args.strip_inner_qdq {
        eprintln!("strip-inner-qdq: ignored for --format qlinear");
    }
    let pruned = prune_unused_initializers(&mut model);
    if pruned != 0 {
        eprintln!("quant cleanup: pruned {pruned} unused initializer(s)");
    }

    eprintln!("saving to {}…", args.output.display());
    save_onnx_model_to_file(&model, &args.output)?;

    eprintln!("done.");
    Ok(())
}

fn main() -> ExitCode {
    match parse_args().and_then(run) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("yscv-quantize: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp(content: &str, name: &str) -> PathBuf {
        let dir = std::env::temp_dir();
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        path
    }

    #[test]
    fn read_calibration_parses_two_samples() {
        let jsonl = r#"{"x": {"shape": [2, 3], "values": [1, 2, 3, 4, 5, 6]}}
{"x": {"shape": [2, 3], "values": [-1, -2, -3, -4, -5, -6]}}
"#;
        let path = write_temp(jsonl, "yscv_quantize_test_two_samples.jsonl");
        let samples = read_calibration(&path).unwrap();
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0]["x"].shape(), &[2, 3]);
        assert_eq!(samples[0]["x"].data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(samples[1]["x"].data()[0], -1.0);
    }

    #[test]
    fn read_calibration_rejects_shape_mismatch() {
        let bad = r#"{"x": {"shape": [2, 3], "values": [1, 2, 3]}}"#;
        let path = write_temp(bad, "yscv_quantize_test_shape_mismatch.jsonl");
        let err = read_calibration(&path).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("expects 6 values"), "unexpected msg: {msg}");
    }

    #[test]
    fn read_calibration_skips_blank_and_comment_lines() {
        let jsonl = "
# first comment
{\"x\": {\"shape\": [1], \"values\": [42]}}

# trailing
";
        let path = write_temp(jsonl, "yscv_quantize_test_blank_lines.jsonl");
        let samples = read_calibration(&path).unwrap();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0]["x"].data(), &[42.0]);
    }

    #[test]
    fn read_calibration_handles_multi_input_samples() {
        let jsonl = r#"{"a": {"shape": [2], "values": [1, 2]}, "b": {"shape": [3], "values": [10, 20, 30]}}"#;
        let path = write_temp(jsonl, "yscv_quantize_test_multi_input.jsonl");
        let samples = read_calibration(&path).unwrap();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].len(), 2);
        assert_eq!(samples[0]["a"].data(), &[1.0, 2.0]);
        assert_eq!(samples[0]["b"].data(), &[10.0, 20.0, 30.0]);
    }

    #[test]
    fn read_paired_calibration_zips_single_input_streams() {
        let a = r#"{"shape": [2], "values": [1, 2]}
{"shape": [2], "values": [3, 4]}
"#;
        let b = r#"{"shape": [1], "values": [10]}
{"shape": [1], "values": [20]}
"#;
        let a_path = write_temp(a, "yscv_quantize_test_pair_a.jsonl");
        let b_path = write_temp(b, "yscv_quantize_test_pair_b.jsonl");
        let spec = format!(
            "input.1={},input.249={}",
            a_path.display(),
            b_path.display()
        );
        let samples = read_calibration_spec(&spec).unwrap();
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0]["input.1"].data(), &[1.0, 2.0]);
        assert_eq!(samples[0]["input.249"].data(), &[10.0]);
        assert_eq!(samples[1]["input.1"].data(), &[3.0, 4.0]);
        assert_eq!(samples[1]["input.249"].data(), &[20.0]);
    }

    #[test]
    fn read_paired_calibration_rejects_length_mismatch() {
        let a_path = write_temp(
            r#"{"shape": [1], "values": [1]}
{"shape": [1], "values": [2]}
"#,
            "yscv_quantize_test_pair_len_a.jsonl",
        );
        let b_path = write_temp(
            r#"{"shape": [1], "values": [10]}
"#,
            "yscv_quantize_test_pair_len_b.jsonl",
        );
        let spec = format!("a={},b={}", a_path.display(), b_path.display());
        let err = read_calibration_spec(&spec).unwrap_err();
        assert!(
            format!("{err}").contains("has 1 samples, expected 2"),
            "unexpected err: {err}"
        );
    }
}
