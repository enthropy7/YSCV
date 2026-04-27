//! `yscv-quantize` — post-training quantization CLI.
//!
//! Loads an fp32 ONNX model, runs it on a calibration dataset to collect
//! per-tensor activation statistics, rewrites Conv / MatMul / Gemm nodes
//! into QDQ format with per-channel int8 weights, and saves the result
//! as a new ONNX file.
//!
//! ```text
//! yscv-quantize <input.onnx> --output <output.onnx> [--calibration <samples.jsonl>]
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
    CalibrationCollector, OnnxRunner, load_onnx_model_from_file, rewrite_to_qdq,
    save_onnx_model_to_file, strip_qdq_within_fusion_chains,
};
use yscv_tensor::Tensor;

#[derive(Debug, thiserror::Error)]
enum CliError {
    #[error(
        "usage: yscv-quantize <input.onnx> --output <output.onnx> [--calibration <samples.jsonl>] [--weights-only] [--strip-inner-qdq]"
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
    calibration: Option<PathBuf>,
    weights_only: bool,
    strip_inner_qdq: bool,
}

fn parse_args() -> Result<Args, CliError> {
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut calibration: Option<PathBuf> = None;
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
                calibration = Some(PathBuf::from(
                    iter.next()
                        .ok_or(CliError::MissingArg("--calibration value"))?,
                ));
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

fn run(args: Args) -> Result<(), CliError> {
    eprintln!("loading {}…", args.input.display());
    let model = load_onnx_model_from_file(&args.input)?;

    let cal_path = if args.weights_only {
        None
    } else {
        args.calibration.as_ref()
    };
    let stats = if let Some(cal_path) = cal_path {
        eprintln!("loading calibration samples from {}…", cal_path.display());
        let samples = read_calibration(cal_path)?;
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

    eprintln!("rewriting model to QDQ format…");
    let mut model = model;
    rewrite_to_qdq(&mut model, &stats)?;

    if args.strip_inner_qdq {
        let removed = strip_qdq_within_fusion_chains(&mut model);
        eprintln!("strip-inner-qdq: removed {removed} Q+DQ pair(s) between Conv-like ops",);
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
}
