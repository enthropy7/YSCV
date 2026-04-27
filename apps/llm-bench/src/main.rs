//! `yscv-llm-bench` — decode-throughput harness for Llama-style ONNX
//! LLMs running through `yscv-onnx`.
//!
//! Two phases are timed independently:
//! - **prefill** — one forward pass on the full prompt (`P` tokens), the
//!   compute-heavy phase that benefits from `packed_int4_gemm` (M = P).
//! - **decode** — `N` autoregressive forward passes appending one token
//!   each, the memory-bound phase that benefits from `packed_int4_gemv`
//!   (M = 1) and from packed-INT4 weight bandwidth savings.
//!
//! ## Inputs
//!
//! - `--model <path.onnx>` — decoder-only ONNX (TinyLlama, Phi-2, Llama-style).
//! - `--input-ids <path.json>` — JSON array of `u32` token IDs (the prompt).
//!   Pre-tokenize externally; we don't bundle a tokenizer to keep the
//!   harness binary small (yscv stays no-Python at runtime).
//! - `--max-tokens N` — number of new tokens to generate during the
//!   decode phase (default 64).
//! - `--warmup K` — discarded warm-up forward passes before timed runs
//!   (default 2). Catches first-touch allocations / page faults.
//!
//! ## Output
//!
//! A single line of summary statistics — prompt length, prefill latency,
//! decode tokens-per-second, total wall — plus an explicit JSON line so
//! external scripts can ingest. Compatible with the comparison harness
//! (run llama.cpp / onnxruntime separately and diff numbers).
//!
//! ## Workflow
//!
//! See `apps/llm-bench/README.md` for end-to-end download + quantize +
//! bench instructions for TinyLlama-1.1B.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use yscv_onnx::{GenerateConfig, OnnxRunner, generate, load_onnx_model_from_file};
use yscv_tensor::Tensor;

#[derive(Debug, thiserror::Error)]
enum CliError {
    #[error(
        "usage: yscv-llm-bench --model <path.onnx> --input-ids <ids.json> [--max-tokens N] [--warmup K] [--input-name NAME]"
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
    #[error("input-ids parse: {0}")]
    BadIds(String),
    #[error("yscv-onnx: {0}")]
    Onnx(#[from] yscv_onnx::OnnxError),
}

#[derive(Debug)]
struct Args {
    model: PathBuf,
    input_ids: PathBuf,
    max_tokens: usize,
    warmup: usize,
    input_name: String,
    output_name: String,
}

fn parse_args() -> Result<Args, CliError> {
    let mut model: Option<PathBuf> = None;
    let mut input_ids: Option<PathBuf> = None;
    let mut max_tokens = 64_usize;
    let mut warmup = 2_usize;
    let mut input_name = "input_ids".to_string();
    let mut output_name = "logits".to_string();

    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "-h" | "--help" => return Err(CliError::Usage),
            "--model" => {
                model = Some(PathBuf::from(
                    iter.next().ok_or(CliError::MissingArg("--model"))?,
                ))
            }
            "--input-ids" => {
                input_ids = Some(PathBuf::from(
                    iter.next().ok_or(CliError::MissingArg("--input-ids"))?,
                ))
            }
            "--max-tokens" => {
                let v = iter.next().ok_or(CliError::MissingArg("--max-tokens"))?;
                max_tokens = v
                    .parse()
                    .map_err(|e| CliError::InvalidArg(format!("--max-tokens: {e}")))?;
            }
            "--warmup" => {
                let v = iter.next().ok_or(CliError::MissingArg("--warmup"))?;
                warmup = v
                    .parse()
                    .map_err(|e| CliError::InvalidArg(format!("--warmup: {e}")))?;
            }
            "--input-name" => {
                input_name = iter.next().ok_or(CliError::MissingArg("--input-name"))?;
            }
            "--output-name" => {
                output_name = iter.next().ok_or(CliError::MissingArg("--output-name"))?;
            }
            other => return Err(CliError::InvalidArg(other.to_string())),
        }
    }

    Ok(Args {
        model: model.ok_or(CliError::MissingArg("--model"))?,
        input_ids: input_ids.ok_or(CliError::MissingArg("--input-ids"))?,
        max_tokens,
        warmup,
        input_name,
        output_name,
    })
}

fn read_input_ids(path: &PathBuf) -> Result<Vec<u32>, CliError> {
    let text = std::fs::read_to_string(path).map_err(|e| CliError::Io {
        path: path.clone(),
        source: e,
    })?;
    serde_json::from_str::<Vec<u32>>(&text).map_err(|e| CliError::BadIds(e.to_string()))
}

#[derive(serde::Serialize)]
struct BenchSummary {
    model: String,
    prompt_tokens: usize,
    decode_tokens: usize,
    prefill_ms: f64,
    decode_total_ms: f64,
    decode_tokens_per_sec: f64,
    total_wall_ms: f64,
}

fn run(args: Args) -> Result<(), CliError> {
    eprintln!("loading model {}…", args.model.display());
    let model = load_onnx_model_from_file(&args.model)?;
    let runner = OnnxRunner::new(&model)?;

    let prompt_ids = read_input_ids(&args.input_ids)?;
    if prompt_ids.is_empty() {
        return Err(CliError::BadIds("empty token list".to_string()));
    }
    eprintln!("prompt: {} tokens", prompt_ids.len());

    // run_fn: feeds the current token sequence to the model, returns
    // flat fp32 logits `[1, seq_len, vocab]`. The model is expected to
    // take its tokens as `i64` under `input_name`; we cast on the fly.
    let run_once = |tokens: &[u32]| -> Result<Vec<f32>, yscv_onnx::OnnxError> {
        let ids_f32: Vec<f32> = tokens.iter().map(|&t| t as f32).collect();
        let input = Tensor::from_vec(vec![1, tokens.len()], ids_f32).map_err(|e| {
            yscv_onnx::OnnxError::DecodeFailed {
                message: format!("input tensor: {e}"),
            }
        })?;
        let feed: Vec<(&str, &Tensor)> = vec![(args.input_name.as_str(), &input)];
        let out = runner.run(&feed)?;
        let logits =
            out.get(&args.output_name)
                .ok_or_else(|| yscv_onnx::OnnxError::DecodeFailed {
                    message: format!("output `{}` missing from model run", args.output_name),
                })?;
        Ok(logits.data().to_vec())
    };

    // Warm-up: discard timings, catches first-touch allocs.
    eprintln!("warm-up: {} run(s)…", args.warmup);
    for _ in 0..args.warmup {
        let _ = run_once(&prompt_ids)?;
    }

    // Prefill: one full forward pass on the prompt.
    let total_start = Instant::now();
    let prefill_start = Instant::now();
    let _logits = run_once(&prompt_ids)?;
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

    // Decode: N autoregressive passes via `generate`.
    let decode_start = Instant::now();
    let cfg = GenerateConfig {
        max_tokens: args.max_tokens,
        temperature: 1.0,
        top_k: 1, // greedy — bench numbers comparable across runs
        top_p: 1.0,
        eos_token_id: None,
        repetition_penalty: 1.0,
    };
    let generated = generate(&prompt_ids, &cfg, |toks| run_once(toks))?;
    let decode_total_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
    let total_wall_ms = total_start.elapsed().as_secs_f64() * 1000.0;
    let decode_tps = if decode_total_ms > 0.0 {
        (generated.len() as f64 * 1000.0) / decode_total_ms
    } else {
        0.0
    };

    let summary = BenchSummary {
        model: args.model.display().to_string(),
        prompt_tokens: prompt_ids.len(),
        decode_tokens: generated.len(),
        prefill_ms,
        decode_total_ms,
        decode_tokens_per_sec: decode_tps,
        total_wall_ms,
    };

    eprintln!(
        "prompt={}t  prefill={:.1}ms  decode={}t / {:.1}ms = {:.2} tok/s  total={:.1}ms",
        summary.prompt_tokens,
        summary.prefill_ms,
        summary.decode_tokens,
        summary.decode_total_ms,
        summary.decode_tokens_per_sec,
        summary.total_wall_ms,
    );
    println!(
        "{}",
        serde_json::to_string(&summary).expect("BenchSummary serialises")
    );
    Ok(())
}

fn main() -> ExitCode {
    match parse_args().and_then(run) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("yscv-llm-bench: {e}");
            ExitCode::FAILURE
        }
    }
}
