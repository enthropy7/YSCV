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

use yscv_onnx::quantize::quantize_matmul_weights_int4_packed;
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
    /// Some HuggingFace ONNX exports require an `attention_mask` input
    /// alongside `input_ids` — a 0/1 tensor with the same shape. When
    /// set, the harness synthesises an all-ones mask each forward pass.
    with_attention_mask: bool,
    /// Some exports also require `position_ids` (Llama / Qwen / Phi).
    /// When set, the harness synthesises `[0, 1, ..., past_len + cur_len-1]`.
    with_position_ids: bool,
    /// Cached-decoder shape, comma-separated: `"n_layers,n_heads,head_dim"`.
    /// When provided, the harness:
    ///   - auto-detects `past_key_values.{i}.{key,value}` inputs by name,
    ///   - allocates empty past tensors `[1, n_heads, past_len, head_dim]`
    ///     (past_len starts 0),
    ///   - cycles `present.{i}.{key,value}` outputs back into past for the
    ///     next forward pass.
    ///
    /// Without this flag past inputs (if any) are not fed and the model
    /// runs in stateless mode (re-encodes the whole sequence per step).
    kv_shape: Option<(usize, usize, usize)>,
    /// In-process pack of every eligible MatMul/Gemm weight to symmetric
    /// INT4 with the given group size (32 / 64 / 128 typical) before
    /// `OnnxRunner::new`. Routes the runtime through the
    /// `packed_int4_gemv_dispatch` / `packed_int4_gemm_dispatch` paths.
    /// Skipped (left fp32) for any weight whose K is not a multiple of
    /// the group size or whose K * M_w < 32.
    int4_group_size: Option<usize>,
    /// KV-cache storage dtype. `f32` (default) keeps the runner's
    /// present.* outputs verbatim and feeds them back. `i8` rounds-trips
    /// each (head, token) row through symmetric int8 with one fp32
    /// scale per row before re-feeding. Halves KV memory footprint
    /// (1 byte + 4 byte/row scale vs 4 byte/element); introduces a
    /// small accuracy drift logged at decode end.
    kv_int8: bool,
}

fn parse_args() -> Result<Args, CliError> {
    let mut model: Option<PathBuf> = None;
    let mut input_ids: Option<PathBuf> = None;
    let mut max_tokens = 64_usize;
    let mut warmup = 2_usize;
    let mut input_name = "input_ids".to_string();
    let mut output_name = "logits".to_string();
    let mut with_attention_mask = false;
    let mut with_position_ids = false;
    let mut kv_shape: Option<(usize, usize, usize)> = None;
    let mut int4_group_size: Option<usize> = None;
    let mut kv_int8 = false;

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
            "--with-attention-mask" => with_attention_mask = true,
            "--with-position-ids" => with_position_ids = true,
            "--kv-shape" => {
                let v = iter.next().ok_or(CliError::MissingArg("--kv-shape"))?;
                let parts: Vec<&str> = v.split(',').collect();
                if parts.len() != 3 {
                    return Err(CliError::InvalidArg(format!(
                        "--kv-shape expects 'n_layers,n_heads,head_dim', got {v}"
                    )));
                }
                let p: Result<Vec<usize>, _> = parts.iter().map(|s| s.trim().parse()).collect();
                let p = p.map_err(|e| CliError::InvalidArg(format!("--kv-shape: {e}")))?;
                kv_shape = Some((p[0], p[1], p[2]));
            }
            "--int4-weights" => {
                let v = iter.next().ok_or(CliError::MissingArg("--int4-weights"))?;
                let g: usize = v
                    .parse()
                    .map_err(|e| CliError::InvalidArg(format!("--int4-weights: {e}")))?;
                if g == 0 || !g.is_multiple_of(2) {
                    return Err(CliError::InvalidArg(format!(
                        "--int4-weights group size must be even and > 0, got {g}"
                    )));
                }
                int4_group_size = Some(g);
            }
            "--kv-dtype" => {
                let v = iter.next().ok_or(CliError::MissingArg("--kv-dtype"))?;
                match v.as_str() {
                    "f32" => kv_int8 = false,
                    "i8" | "int8" => kv_int8 = true,
                    other => {
                        return Err(CliError::InvalidArg(format!(
                            "--kv-dtype expects f32|i8, got {other}"
                        )));
                    }
                }
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
        with_attention_mask,
        with_position_ids,
        kv_shape,
        int4_group_size,
        kv_int8,
    })
}

/// Round-trip a KV tensor through per-row symmetric int8 quantisation
/// and dequantisation. Each (head × token) row gets one fp32 scale.
/// Mirrors the storage scheme in `crates/yscv-onnx/src/runner/kv_cache.rs::LayerStorage::I8`
/// but inlined here so the harness doesn't need to refactor the
/// 4-D `[1, n_heads, past_len, head_dim]` past tensors into the 2-D
/// `[seq, kv_dim]` shape that `KvCache` expects.
///
/// Returns the dequantised tensor (same shape as input) plus the
/// total bytes that the int8 store would occupy — for accounting
/// the memory savings vs fp32.
fn kv_int8_round_trip(t: &Tensor) -> (Tensor, usize) {
    let shape = t.shape().to_vec();
    // Treat the trailing dim as the per-row width. Shape is
    // [1, n_heads, S, head_dim]; row width = head_dim, num_rows =
    // n_heads * S.
    let head_dim = *shape.last().unwrap_or(&1);
    let total = t.len();
    if head_dim == 0 || total == 0 {
        return (t.clone(), 0);
    }
    let num_rows = total / head_dim;
    let src = t.data();
    let mut out = vec![0.0_f32; total];
    let mut bytes = 0_usize;
    for r in 0..num_rows {
        let base = r * head_dim;
        let row = &src[base..base + head_dim];
        let abs_max = row.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
        let scale = if abs_max <= f32::EPSILON {
            1.0
        } else {
            abs_max / 127.0
        };
        let inv = 1.0 / scale;
        for (i, &v) in row.iter().enumerate() {
            let q = (v * inv).round().clamp(-127.0, 127.0) as i8;
            // dequantise immediately — simulates the read-side dequant
            out[base + i] = (q as f32) * scale;
        }
        bytes += head_dim + std::mem::size_of::<f32>();
    }
    let result = Tensor::from_vec(shape, out).expect("shape preserved");
    (result, bytes)
}

/// Detect the cached-decoder past_key_values input naming convention
/// from the model's input list: `"past_key_values.{i}.{key|value}"`
/// (HuggingFace optimum-onnx export style). Returns `(name, layer, kind)`
/// triples sorted by layer-then-kind; kind is "key" or "value".
fn detect_kv_inputs(inputs: &[String]) -> Vec<(String, usize, &'static str)> {
    let mut out: Vec<(String, usize, &'static str)> = Vec::new();
    for name in inputs {
        let Some(rest) = name.strip_prefix("past_key_values.") else {
            continue;
        };
        let mut parts = rest.split('.');
        let (Some(idx_str), Some(kind)) = (parts.next(), parts.next()) else {
            continue;
        };
        let Ok(layer) = idx_str.parse::<usize>() else {
            continue;
        };
        let kind_static = match kind {
            "key" => "key",
            "value" => "value",
            _ => continue,
        };
        out.push((name.clone(), layer, kind_static));
    }
    out.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.2.cmp(b.2)));
    out
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
    let mut model = load_onnx_model_from_file(&args.model)?;
    if let Some(group_size) = args.int4_group_size {
        eprintln!("packing MatMul/Gemm weights to symmetric INT4 (group_size={group_size})…");
        let packed = quantize_matmul_weights_int4_packed(&mut model, group_size)?;
        eprintln!(
            "INT4 pack: {} weight(s) packed, {} fp32 initializer(s) remaining",
            packed,
            model.initializers.len()
        );
    }
    let runner = OnnxRunner::new(&model)?;

    let prompt_ids = read_input_ids(&args.input_ids)?;
    if prompt_ids.is_empty() {
        return Err(CliError::BadIds("empty token list".to_string()));
    }
    eprintln!("prompt: {} tokens", prompt_ids.len());

    let kv_inputs = detect_kv_inputs(&model.inputs);
    let use_kv_cache = args.kv_shape.is_some() && !kv_inputs.is_empty();
    if use_kv_cache {
        let (n_layers, n_heads, head_dim) = args.kv_shape.unwrap();
        eprintln!(
            "kv-cache: {} past inputs detected, shape per K/V = [1, {}, past_len, {}], n_layers={}",
            kv_inputs.len(),
            n_heads,
            head_dim,
            n_layers
        );
    } else if args.kv_shape.is_some() && kv_inputs.is_empty() {
        eprintln!("kv-cache: --kv-shape supplied but no past_key_values.* inputs in model");
    } else if !kv_inputs.is_empty() && args.kv_shape.is_none() {
        eprintln!(
            "kv-cache: model has {} past_key_values.* inputs but --kv-shape not supplied; \
             feeding empty past won't work — pass --kv-shape n_layers,n_heads,head_dim",
            kv_inputs.len()
        );
    }

    // KV cache state: one (key, value) Tensor pair per layer, growing
    // along the past-length dim each forward pass. Empty at start.
    let n_layers_for_kv = if use_kv_cache {
        args.kv_shape.unwrap().0
    } else {
        0
    };
    let mut past_kv: Vec<(Tensor, Tensor)> = Vec::with_capacity(n_layers_for_kv);
    if use_kv_cache {
        let (_, n_heads, head_dim) = args.kv_shape.unwrap();
        for _ in 0..n_layers_for_kv {
            let k = Tensor::from_vec(vec![1, n_heads, 0, head_dim], Vec::new()).map_err(|e| {
                yscv_onnx::OnnxError::DecodeFailed {
                    message: format!("empty past key tensor: {e}"),
                }
            })?;
            let v = Tensor::from_vec(vec![1, n_heads, 0, head_dim], Vec::new()).map_err(|e| {
                yscv_onnx::OnnxError::DecodeFailed {
                    message: format!("empty past value tensor: {e}"),
                }
            })?;
            past_kv.push((k, v));
        }
    }

    // For cached decoders we feed `tokens_new` only (last token at decode,
    // full prompt at prefill); for stateless we always feed the full
    // sequence. `position_offset` tracks past length for position_ids.
    let mut position_offset: usize = 0;

    let kv_int8_active = args.kv_int8 && use_kv_cache;
    if kv_int8_active {
        eprintln!(
            "kv-cache: int8 storage active — present.* outputs will be \
             round-tripped through symmetric int8 (per-row scale) before \
             re-feeding."
        );
    }
    let mut kv_int8_bytes: usize = 0;

    // run_fn: returns flat fp32 logits `[1, seq_len, vocab]`. Honours
    // attention_mask, position_ids, past_key_values inputs as configured.
    let run_once = |tokens_new: &[u32],
                    past: &mut [(Tensor, Tensor)],
                    pos_off: usize,
                    kv_bytes_total: &mut usize|
     -> Result<Vec<f32>, yscv_onnx::OnnxError> {
        let cur_len = tokens_new.len();
        let ids_f32: Vec<f32> = tokens_new.iter().map(|&t| t as f32).collect();
        let input = Tensor::from_vec(vec![1, cur_len], ids_f32).map_err(|e| {
            yscv_onnx::OnnxError::DecodeFailed {
                message: format!("input tensor: {e}"),
            }
        })?;
        let total_len = pos_off + cur_len;
        let mask = if args.with_attention_mask {
            let m = vec![1.0_f32; total_len];
            Some(Tensor::from_vec(vec![1, total_len], m).map_err(|e| {
                yscv_onnx::OnnxError::DecodeFailed {
                    message: format!("attention_mask tensor: {e}"),
                }
            })?)
        } else {
            None
        };
        let pos_ids = if args.with_position_ids {
            let p: Vec<f32> = (pos_off..pos_off + cur_len).map(|i| i as f32).collect();
            Some(Tensor::from_vec(vec![1, cur_len], p).map_err(|e| {
                yscv_onnx::OnnxError::DecodeFailed {
                    message: format!("position_ids tensor: {e}"),
                }
            })?)
        } else {
            None
        };
        let mut feed: Vec<(&str, &Tensor)> = Vec::with_capacity(3 + 2 * past.len());
        feed.push((args.input_name.as_str(), &input));
        if let Some(m) = mask.as_ref() {
            feed.push(("attention_mask", m));
        }
        if let Some(p) = pos_ids.as_ref() {
            feed.push(("position_ids", p));
        }
        // Borrow each (k, v) separately so feed entries match name conventions.
        let mut kv_names: Vec<String> = Vec::with_capacity(past.len() * 2);
        for layer in 0..past.len() {
            kv_names.push(format!("past_key_values.{layer}.key"));
            kv_names.push(format!("past_key_values.{layer}.value"));
        }
        if !past.is_empty() {
            for (i, (k, v)) in past.iter().enumerate() {
                feed.push((kv_names[i * 2].as_str(), k));
                feed.push((kv_names[i * 2 + 1].as_str(), v));
            }
        }
        let mut out = runner.run(&feed)?;
        let logits =
            out.get(&args.output_name)
                .ok_or_else(|| yscv_onnx::OnnxError::DecodeFailed {
                    message: format!("output `{}` missing from model run", args.output_name),
                })?;
        let logits_vec = logits.data().to_vec();
        // Cycle present → past for next call. `HashMap::remove` moves the
        // owned Tensor out of the runner's output map — no copy of the
        // present-K/V buffers (each layer's K and V can be MB-class on
        // long contexts; a `clone()` here would dominate decode time).
        // With --kv-dtype i8 we round-trip each KV tensor through a
        // per-row symmetric int8 quant/dequant before re-feeding so
        // the harness models a true int8-storage cache.
        let mut step_bytes = 0_usize;
        for layer in 0..past.len() {
            let kk = format!("present.{layer}.key");
            let vk = format!("present.{layer}.value");
            if let (Some(k), Some(v)) = (out.remove(&kk), out.remove(&vk)) {
                if kv_int8_active {
                    let (kq, kb) = kv_int8_round_trip(&k);
                    let (vq, vb) = kv_int8_round_trip(&v);
                    step_bytes += kb + vb;
                    past[layer] = (kq, vq);
                } else {
                    past[layer] = (k, v);
                }
            }
        }
        if kv_int8_active {
            *kv_bytes_total = step_bytes;
        }
        Ok(logits_vec)
    };

    // Warm-up: discard timings, catches first-touch allocs.
    eprintln!("warm-up: {} run(s)…", args.warmup);
    for _ in 0..args.warmup {
        // Stateless warm-up — pass the full prompt with empty pasts;
        // restored after warmup so timed runs see a fresh cache.
        let mut throwaway = past_kv.clone();
        let _ = run_once(&prompt_ids, &mut throwaway, 0, &mut kv_int8_bytes)?;
    }

    // Prefill: one full forward pass on the prompt. Past starts empty.
    let total_start = Instant::now();
    let prefill_start = Instant::now();
    let _logits = run_once(
        &prompt_ids,
        &mut past_kv,
        position_offset,
        &mut kv_int8_bytes,
    )?;
    if use_kv_cache {
        position_offset += prompt_ids.len();
    }
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

    // Decode: N autoregressive passes. With KV-cache feed only the
    // last token; without, re-feed the whole sequence (stateless).
    let decode_start = Instant::now();
    let cfg = GenerateConfig {
        max_tokens: args.max_tokens,
        temperature: 1.0,
        top_k: 1, // greedy — bench numbers comparable across runs
        top_p: 1.0,
        eos_token_id: None,
        repetition_penalty: 1.0,
    };
    let generated = if use_kv_cache {
        let mut acc = Vec::with_capacity(args.max_tokens);
        let mut last_token = *prompt_ids.last().unwrap();
        for _ in 0..args.max_tokens {
            let logits = run_once(
                &[last_token],
                &mut past_kv,
                position_offset,
                &mut kv_int8_bytes,
            )?;
            position_offset += 1;
            // logits shape [1, 1, vocab]; argmax over vocab.
            let vocab = logits.len();
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
            let _ = vocab;
            acc.push(next);
            last_token = next;
        }
        acc
    } else {
        // Stateless path uses the existing yscv-onnx generate helper.
        generate(&prompt_ids, &cfg, |toks| {
            run_once(toks, &mut past_kv, 0, &mut kv_int8_bytes)
        })?
    };
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

#[cfg(test)]
mod tests {
    use super::detect_kv_inputs;

    #[test]
    fn detects_hf_past_key_values_pattern() {
        let inputs = vec![
            "input_ids".to_string(),
            "attention_mask".to_string(),
            "past_key_values.0.key".to_string(),
            "past_key_values.0.value".to_string(),
            "past_key_values.1.key".to_string(),
            "past_key_values.1.value".to_string(),
        ];
        let kv = detect_kv_inputs(&inputs);
        assert_eq!(kv.len(), 4);
        assert_eq!(kv[0], ("past_key_values.0.key".to_string(), 0, "key"));
        assert_eq!(kv[1], ("past_key_values.0.value".to_string(), 0, "value"));
        assert_eq!(kv[2], ("past_key_values.1.key".to_string(), 1, "key"));
        assert_eq!(kv[3], ("past_key_values.1.value".to_string(), 1, "value"));
    }

    #[test]
    fn ignores_non_kv_inputs() {
        let inputs = vec![
            "input_ids".to_string(),
            "past_key_values.foo.key".to_string(), // bad layer index
            "past_key_values.0.queries".to_string(), // wrong kind
        ];
        assert!(detect_kv_inputs(&inputs).is_empty());
    }

    #[test]
    fn sorts_layers_numerically_via_struct_order() {
        // Both layer-13 entries should follow layer-2 entries since the
        // sort is on (layer: usize, kind: &str), not lexicographic on name.
        let inputs = vec![
            "past_key_values.13.value".to_string(),
            "past_key_values.2.key".to_string(),
            "past_key_values.13.key".to_string(),
            "past_key_values.2.value".to_string(),
        ];
        let kv = detect_kv_inputs(&inputs);
        assert_eq!(kv[0].1, 2);
        assert_eq!(kv[1].1, 2);
        assert_eq!(kv[2].1, 13);
        assert_eq!(kv[3].1, 13);
    }
}
