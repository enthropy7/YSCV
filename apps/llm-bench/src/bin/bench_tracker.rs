//! Minimal yscv tracker benchmark that uses the current workspace crates.

use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use yscv_onnx::{
    OnnxAttribute, OnnxModel, OnnxNode, OnnxRunner, dump_runner_profile, load_onnx_model_from_file,
    optimize_onnx_graph, quant_runtime_stats, reset_quant_runtime_stats,
};
use yscv_tensor::Tensor;

#[derive(Debug)]
struct InputSpec {
    name: String,
    shape: Vec<usize>,
}

#[derive(Debug)]
struct Args {
    model: String,
    inputs: Vec<InputSpec>,
    iters: usize,
    threads: usize,
    json: bool,
}

fn parse_shape(s: &str) -> Result<Vec<usize>, String> {
    s.split('x')
        .map(|p| {
            p.parse::<usize>()
                .map_err(|e| format!("bad dim `{p}`: {e}"))
        })
        .collect()
}

fn parse_input(s: &str) -> Result<InputSpec, String> {
    let (name, shape) = s
        .split_once(':')
        .ok_or_else(|| format!("--input expects NAME:DxDxD, got `{s}`"))?;
    Ok(InputSpec {
        name: name.to_string(),
        shape: parse_shape(shape)?,
    })
}

fn parse_args() -> Result<Args, String> {
    let mut model = None;
    let mut inputs = Vec::new();
    let mut iters = 200usize;
    let mut threads = 1usize;
    let mut json = false;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => model = it.next(),
            "--input" => inputs.push(parse_input(&it.next().ok_or("missing --input value")?)?),
            "--iters" => {
                iters = it
                    .next()
                    .ok_or("missing --iters value")?
                    .parse()
                    .map_err(|e| format!("--iters: {e}"))?
            }
            "--threads" => {
                threads = it
                    .next()
                    .ok_or("missing --threads value")?
                    .parse()
                    .map_err(|e| format!("--threads: {e}"))?
            }
            "--json" => json = true,
            "-h" | "--help" => return Err("help".into()),
            other => return Err(format!("unknown arg: {other}")),
        }
    }
    if inputs.is_empty() {
        return Err("at least one --input is required".into());
    }
    Ok(Args {
        model: model.ok_or("missing --model")?,
        inputs,
        iters,
        threads: threads.max(1),
        json,
    })
}

fn tensor_len(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn attr_int(node: &OnnxNode, name: &str, default: i64) -> i64 {
    match node.attributes.get(name) {
        Some(OnnxAttribute::Int(v)) => *v,
        _ => default,
    }
}

fn qlinear_conv_kind(model: &OnnxModel, node: &OnnxNode) -> Option<&'static str> {
    if node.op_type != "QLinearConv" {
        return None;
    }
    let w = node
        .inputs
        .get(3)
        .and_then(|name| model.initializers.get(name))?;
    let shape = w.shape();
    if shape.len() != 4 {
        return None;
    }
    let group = attr_int(node, "group", 1) as usize;
    if group == 1 && shape[2] == 1 && shape[3] == 1 {
        return Some("pw");
    }
    if group > 1 && group == shape[0] && shape[1] == 1 && (shape[2] == 3 || shape[2] == 5) {
        return Some("dw");
    }
    None
}

fn qlinear_boundary_after(nodes: &[OnnxNode], start: usize, input: &str) -> Option<(usize, usize)> {
    let dq = nodes.get(start)?;
    if dq.op_type != "DequantizeLinear" || dq.inputs.first()? != input {
        return None;
    }
    let mut quant_idx = start + 1;
    let mut quant_input = dq.outputs.first()?.as_str();
    if let Some(relu) = nodes.get(quant_idx)
        && relu.op_type == "Relu"
        && relu.inputs.first().map(String::as_str) == Some(quant_input)
    {
        quant_input = relu.outputs.first()?.as_str();
        quant_idx += 1;
    }
    let q = nodes.get(quant_idx)?;
    if q.op_type == "QuantizeLinear" && q.inputs.first().map(String::as_str) == Some(quant_input) {
        return Some((start, quant_idx));
    }
    None
}

fn quant_chain_candidates(model: &OnnxModel) -> usize {
    let nodes = &model.nodes;
    let mut count = 0usize;
    for i in 0..nodes.len() {
        let Some(kind) = qlinear_conv_kind(model, &nodes[i]) else {
            continue;
        };
        let Some(qconv_out) = nodes[i].outputs.first() else {
            continue;
        };
        if let Some((_, q_idx)) = qlinear_boundary_after(nodes, i + 1, qconv_out)
            && let Some(next) = nodes.get(q_idx + 1)
            && next.inputs.first() == nodes[q_idx].outputs.first()
            && let Some(next_kind) = qlinear_conv_kind(model, next)
            && ((kind == "pw" && next_kind == "dw") || (kind == "dw" && next_kind == "pw"))
        {
            count += 1;
            continue;
        }
        if let Some((dq_idx, q_idx)) = qlinear_boundary_after(nodes, i + 1, qconv_out) {
            let relu_offset = if nodes.get(dq_idx + 1).is_some_and(|n| n.op_type == "Relu") {
                1
            } else {
                0
            };
            let add_idx = dq_idx + 1 + relu_offset;
            if nodes.get(add_idx).is_some_and(|n| n.op_type == "Add")
                && nodes
                    .get(q_idx)
                    .is_some_and(|n| n.op_type == "QuantizeLinear")
            {
                count += 1;
            }
        }
    }
    count
}

fn run(args: Args) -> Result<(), String> {
    let mut model =
        load_onnx_model_from_file(Path::new(&args.model)).map_err(|e| format!("load: {e}"))?;
    optimize_onnx_graph(&mut model);
    let chain_candidates = quant_chain_candidates(&model);
    let runner =
        OnnxRunner::with_threads(&model, args.threads).map_err(|e| format!("runner: {e}"))?;
    let tensors: Vec<Tensor> = args
        .inputs
        .iter()
        .map(|spec| Tensor::from_vec(spec.shape.clone(), vec![0.0; tensor_len(&spec.shape)]))
        .collect::<Result<_, _>>()
        .map_err(|e| format!("input tensor: {e}"))?;
    let feed: Vec<(&str, &Tensor)> = args
        .inputs
        .iter()
        .zip(&tensors)
        .map(|(spec, tensor)| (spec.name.as_str(), tensor))
        .collect();

    runner.run(&feed).map_err(|e| format!("warmup: {e}"))?;
    reset_quant_runtime_stats();
    let mut samples = Vec::with_capacity(args.iters);
    for _ in 0..args.iters {
        let t0 = Instant::now();
        let _ = runner.run(&feed).map_err(|e| format!("run: {e}"))?;
        samples.push(t0.elapsed().as_secs_f64() * 1_000.0);
    }
    samples.sort_by(|a, b| a.total_cmp(b));
    let min = samples[0];
    let p50 = samples[samples.len() / 2];
    let p95 = samples[(samples.len() * 95 / 100).min(samples.len() - 1)];
    let avg = samples.iter().sum::<f64>() / samples.len() as f64;
    let qs = quant_runtime_stats();
    if let Ok(path) = std::env::var("YSCV_RUNNER_PROFILE")
        && !path.is_empty()
    {
        dump_runner_profile(&path).map_err(|e| format!("profile dump: {e}"))?;
    }

    // `executed` is the cumulative chain-dispatch count over all
    // benchmark iterations — divide by `iters` for per-inference. The
    // counter increments once per fused chain action, so on a clean
    // model with N candidates the per-iter rate equals N once landing
    // is complete.
    let executed_total = qs.quant_chain_executed;
    let executed_per_iter = if args.iters > 0 {
        executed_total / args.iters as u64
    } else {
        0
    };
    let chain_candidates_per_iter = chain_candidates as u64;
    let fallback_per_iter = chain_candidates_per_iter.saturating_sub(executed_per_iter);

    if args.json {
        println!(
            "{{\"model\":\"{}\",\"threads\":{},\"iters\":{},\"min_ms\":{:.6},\"p50_ms\":{:.6},\"avg_ms\":{:.6},\"p95_ms\":{:.6},\"quant_chain_candidates\":{},\"quant_chain_executed\":{},\"quant_chain_executed_total\":{},\"quant_chain_fallback\":{},\"qdq_boundaries\":{},\"qlinear_conv_fast\":{},\"qlinear_conv_fallback\":{},\"qlinear_matmul_fast\":{},\"qlinear_matmul_fallback\":{},\"quant_i8_stores\":{},\"quant_i8_materializations\":{}}}",
            args.model,
            args.threads,
            args.iters,
            min,
            p50,
            avg,
            p95,
            chain_candidates,
            executed_per_iter,
            executed_total,
            fallback_per_iter,
            qs.qdq_boundaries,
            qs.qlinear_conv_fast,
            qs.qlinear_conv_fallback,
            qs.qlinear_matmul_fast,
            qs.qlinear_matmul_fallback,
            qs.quant_i8_stores,
            qs.quant_i8_materializations
        );
    } else {
        println!("model: {}", args.model);
        println!("threads: {}", args.threads);
        println!("iters: {}", args.iters);
        println!("min={min:.3}ms p50={p50:.3}ms avg={avg:.3}ms p95={p95:.3}ms");
        println!(
            "quant chains: candidates={} executed={} (total={}) fallback={}",
            chain_candidates, executed_per_iter, executed_total, fallback_per_iter
        );
        println!(
            "quant runtime: qdq_boundaries={} qlinear_conv_fast={} qlinear_conv_fallback={} qlinear_matmul_fast={} qlinear_matmul_fallback={} quant_i8_stores={} quant_i8_materializations={}",
            qs.qdq_boundaries,
            qs.qlinear_conv_fast,
            qs.qlinear_conv_fallback,
            qs.qlinear_matmul_fast,
            qs.qlinear_matmul_fallback,
            qs.quant_i8_stores,
            qs.quant_i8_materializations
        );
    }
    Ok(())
}

fn main() -> ExitCode {
    match parse_args().and_then(run) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) if e == "help" => {
            eprintln!(
                "usage: bench_tracker --model MODEL.onnx --input NAME:DxDxD [--input NAME:DxDxD] [--iters N] [--threads N] [--json]"
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("bench_tracker: {e}");
            ExitCode::FAILURE
        }
    }
}
