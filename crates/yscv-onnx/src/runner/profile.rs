//! Two profilers: the fused-path per-node timing collector
//! (YSCV_RUNNER_PROFILE) and the single-pass unfused CPU profiler.

use super::*;

/// Aggregated per-node timing collector for the fused runner path.
///
/// Activated when `YSCV_RUNNER_PROFILE=path` is set on process startup.
/// Accumulates timing over every inference in `run_onnx_model_jit` and
/// dumps JSON via [`dump_runner_profile`]. Unlike `profile_onnx_model_cpu`
/// (which runs a **single unfused** inference), this captures the actual
/// fused-path timings with Conv+Relu, Conv+Add, DW+PW fusions applied —
/// the numbers that `scripts/gap_diff.py` should compare against ORT's
/// Chrome-trace output.
///
/// Storage is per-node-name so repeated ops across inference iterations
/// aggregate cleanly. Shapes are captured on first sighting. Each thread
/// owns its own store via `thread_local!`; the global registry holds an
/// `Arc<Mutex<_>>` per thread so `dump_runner_profile` can walk them all
/// and aggregate without contending on the hot path. Previously this was
/// a single `Mutex<RunnerProfileStore>` — under tower-parallel two branch
/// threads recorded concurrently, and even uncontended lock+unlock costs
/// showed up in per-op timings. Thread-local stores pay zero synchronisation
/// on record (each thread is the sole writer of its own `RefCell`).
#[derive(Default)]
struct RunnerProfileStore {
    per_node: HashMap<String, RunnerNodeStat>,
}

#[derive(Clone, Default)]
struct RunnerNodeStat {
    op: String,
    total_ns: u64,
    count: u64,
    in_shape: Vec<usize>,
    out_shape: Vec<usize>,
}

/// Global registry of per-thread stores. Each thread registers its store
/// on first access; destruction is deferred via `Arc` so data survives
/// thread exit until `dump_runner_profile` runs.
fn profile_registry()
-> &'static std::sync::Mutex<Vec<std::sync::Arc<std::sync::Mutex<RunnerProfileStore>>>> {
    use std::sync::OnceLock;
    static CELL: OnceLock<
        std::sync::Mutex<Vec<std::sync::Arc<std::sync::Mutex<RunnerProfileStore>>>>,
    > = OnceLock::new();
    CELL.get_or_init(|| std::sync::Mutex::new(Vec::new()))
}

thread_local! {
    static LOCAL_PROFILE: std::sync::Arc<std::sync::Mutex<RunnerProfileStore>> = {
        let store = std::sync::Arc::new(std::sync::Mutex::new(RunnerProfileStore::default()));
        if let Ok(mut reg) = profile_registry().lock() {
            reg.push(store.clone());
        }
        store
    };
}

/// True when `YSCV_RUNNER_PROFILE` env is non-empty on startup. Cached so
/// we only pay the env read once.
pub(crate) fn runner_profile_active() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("YSCV_RUNNER_PROFILE")
            .ok()
            .filter(|s| !s.is_empty())
            .is_some()
    })
}

pub(crate) fn runner_profile_record(
    name: &str,
    op: &str,
    elapsed_ns: u64,
    in_shape: Vec<usize>,
    out_shape: Vec<usize>,
) {
    LOCAL_PROFILE.with(|store| {
        // Same-thread lock — uncontended, near-free; kept as `Mutex` rather
        // than `RefCell` so the `Arc` shared with the registry can be
        // locked from the dump thread safely.
        if let Ok(mut s) = store.lock() {
            let entry = s
                .per_node
                .entry(name.to_string())
                .or_insert_with(|| RunnerNodeStat {
                    op: op.to_string(),
                    in_shape: in_shape.clone(),
                    out_shape: out_shape.clone(),
                    ..Default::default()
                });
            entry.total_ns += elapsed_ns;
            entry.count += 1;
            if entry.in_shape.is_empty() {
                entry.in_shape = in_shape;
            }
            if entry.out_shape.is_empty() {
                entry.out_shape = out_shape;
            }
        }
    });
}

/// Dumps the accumulated runner-path profile to `path` as JSON in the
/// same schema as `profile_onnx_model_cpu` / `bench_ort.py --emit-profile`.
/// Call after a benchmark loop to get aggregated per-node timings
/// normalised by iteration count. No-op when the profiler was never
/// activated (env unset) or the store is empty.
pub fn dump_runner_profile(path: &str) -> Result<(), OnnxError> {
    use std::fmt::Write as _;
    // Aggregate per-thread stores. Each thread recorded into its own
    // registry entry; we sum counts and total_ns per node name and keep
    // the first-seen shapes.
    let merged = {
        let registry = profile_registry()
            .lock()
            .map_err(|e| OnnxError::DecodeFailed {
                message: format!("runner profile registry poisoned: {e}"),
            })?;
        let mut merged: HashMap<String, RunnerNodeStat> = HashMap::new();
        for thread_store in registry.iter() {
            let store = thread_store.lock().map_err(|e| OnnxError::DecodeFailed {
                message: format!("runner profile per-thread mutex poisoned: {e}"),
            })?;
            for (name, stat) in &store.per_node {
                merged
                    .entry(name.clone())
                    .and_modify(|e| {
                        e.total_ns += stat.total_ns;
                        e.count += stat.count;
                        if e.in_shape.is_empty() {
                            e.in_shape = stat.in_shape.clone();
                        }
                        if e.out_shape.is_empty() {
                            e.out_shape = stat.out_shape.clone();
                        }
                    })
                    .or_insert_with(|| stat.clone());
            }
        }
        merged
    };
    if merged.is_empty() {
        return Ok(());
    }
    // Normalise total_ns by `count` so each node's `ms` is the average
    // per-instance time — matches what ORT's `end_profiling()` emits for
    // a single inference.
    let mut entries: Vec<(&String, &RunnerNodeStat)> = merged.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    let iters: u64 = entries.iter().map(|(_, s)| s.count).max().unwrap_or(1);

    let mut out = String::with_capacity(256 * entries.len());
    out.push_str("{\"engine\":\"yscv\",\"total_ms\":");
    let total_ns: u64 = entries.iter().map(|(_, s)| s.total_ns).sum();
    let total_ms = (total_ns as f64) / (iters as f64) / 1_000_000.0;
    let _ = write!(out, "{:.6}", total_ms);
    out.push_str(",\"iterations\":");
    let _ = write!(out, "{}", iters);
    out.push_str(",\"nodes\":[");
    for (i, (name, stat)) in entries.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str("{\"name\":\"");
        json_escape_into_local(&mut out, name);
        out.push_str("\",\"op\":\"");
        json_escape_into_local(&mut out, &stat.op);
        out.push_str("\",\"ms\":");
        // ms per inference (total_ns / iterations / 1e6).
        let ms = (stat.total_ns as f64) / (iters as f64) / 1_000_000.0;
        let _ = write!(out, "{:.6}", ms);
        out.push_str(",\"count\":");
        let _ = write!(out, "{}", stat.count);
        out.push_str(",\"in_shape\":");
        shape_to_json_local(&mut out, &stat.in_shape);
        out.push_str(",\"out_shape\":");
        shape_to_json_local(&mut out, &stat.out_shape);
        out.push('}');
    }
    out.push_str("]}\n");
    std::fs::write(path, out).map_err(|e| OnnxError::DecodeFailed {
        message: format!("write runner profile to {path}: {e}"),
    })?;
    Ok(())
}

fn json_escape_into_local(out: &mut String, s: &str) {
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                use std::fmt::Write as _;
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
}

fn shape_to_json_local(out: &mut String, shape: &[usize]) {
    use std::fmt::Write as _;
    out.push('[');
    for (i, d) in shape.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let _ = write!(out, "{d}");
    }
    out.push(']');
}

/// One node's timing sample, captured during `profile_onnx_model_cpu`.
///
/// Lives alongside the per-op-type text summary so the JSON output path
/// can emit a per-instance breakdown. Shapes are captured before the
/// node executes (in) and immediately after (out); for fused ops like
/// SiLU(Sigmoid+Mul) we record the fused name and the final Mul's output
/// shape.
#[derive(Debug, Clone)]
struct NodeTiming {
    name: String,
    op: String,
    ms: f64,
    in_shape: Vec<usize>,
    out_shape: Vec<usize>,
}

#[derive(Debug, Clone)]
struct ConvDetail {
    name: String,
    ms: f64,
    in_shape: Vec<usize>,
    out_shape: Vec<usize>,
    kernel_shape: Vec<i64>,
    strides: Vec<i64>,
}

/// Profile CPU inference: measure per-op-type timing.
///
/// When the env var `YSCV_PROFILE_JSON` is set to a file path, a
/// machine-readable JSON with per-node instance timings is written to
/// that path in addition to the standard text summary. Format:
/// ```json
/// {"engine":"yscv","total_ms":11.47,"nodes":[
///   {"name":"Conv_0","op":"Conv","ms":0.12,
///    "in_shape":[1,3,128,128],"out_shape":[1,16,64,64]}, ...
/// ]}
/// ```
/// The file is consumed by `scripts/gap_diff.py` for apples-to-apples
/// comparison against ORT's profiling JSON.
pub fn profile_onnx_model_cpu(
    model: &OnnxModel,
    inputs: HashMap<String, Tensor>,
) -> Result<(), OnnxError> {
    use std::time::Instant;

    let mut env = TensorEnv::from_model(model);
    for (name, tensor) in inputs {
        env.insert(name, tensor);
    }

    let nodes = &model.nodes;
    let node_kinds = &model.runtime_index.node_kinds;
    let mut skip = vec![false; nodes.len()];
    let mut timings: HashMap<String, (f64, usize)> = HashMap::new();
    let mut conv_details: Vec<ConvDetail> = Vec::new();
    // Per-instance timings — fuel for `YSCV_PROFILE_JSON` output.
    let mut instance_timings: Vec<NodeTiming> = Vec::with_capacity(nodes.len());

    let prof_use_counts = &model.runtime_index.use_counts;
    let prof_use_counts_by_id = &model.runtime_index.use_counts_by_id;

    for (i, node) in nodes.iter().enumerate() {
        if skip[i] {
            continue;
        }

        let kind = node_kind(node_kinds, nodes, i);

        // SiLU fusion in profiler too (with look-ahead)
        if kind == NodeKind::Sigmoid && node.inputs.len() == 1 {
            let sig_in = &node.inputs[0];
            let sig_out = &node.outputs[0];
            let mut found_silu = false;
            for look in 1..=3 {
                if let Some(next) = nodes.get(i + look)
                    && node_kind(node_kinds, nodes, i + look) == NodeKind::Mul
                    && next.inputs.len() == 2
                    && ((next.inputs[0] == *sig_out && next.inputs[1] == *sig_in)
                        || (next.inputs[1] == *sig_out && next.inputs[0] == *sig_in))
                {
                    let is_nhwc = env.is_nhwc(sig_in);
                    let mul_out = &next.outputs[0];
                    let start = Instant::now();
                    let sig_in_uses =
                        tensor_use_count(&env, prof_use_counts_by_id, prof_use_counts, sig_in);
                    if sig_in_uses <= 2 {
                        if let Some(mut tensor) = env.remove(sig_in) {
                            yscv_kernels::silu_inplace(&mut tensor);
                            env.insert(mul_out.clone(), tensor);
                        }
                    } else if let Some(x_tensor) = env.get(sig_in) {
                        let result_tensor = yscv_kernels::silu(x_tensor);
                        env.insert(mul_out.clone(), result_tensor);
                    }
                    if is_nhwc {
                        env.mark_nhwc(mul_out);
                    }
                    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                    let entry = timings.entry("SiLU(fused)".to_string()).or_insert((0.0, 0));
                    entry.0 += elapsed;
                    entry.1 += 1;
                    let fused_out_shape = env
                        .get(mul_out)
                        .map(|t| t.shape().to_vec())
                        .unwrap_or_default();
                    let fused_in_shape = env
                        .get(sig_in)
                        .map(|t| t.shape().to_vec())
                        .unwrap_or_else(|| fused_out_shape.clone());
                    instance_timings.push(NodeTiming {
                        name: node.name.clone(),
                        op: "SiLU(fused)".to_string(),
                        ms: elapsed,
                        in_shape: fused_in_shape,
                        out_shape: fused_out_shape,
                    });
                    // Execute intermediate nodes, mark done to prevent re-execution.
                    for mid in 1..look {
                        if i + mid < skip.len() && !skip[i + mid] {
                            let mid_node = &nodes[i + mid];
                            let mid_in = mid_node
                                .inputs
                                .first()
                                .and_then(|n| env.get(n))
                                .map(|t| t.shape().to_vec())
                                .unwrap_or_default();
                            let mid_start = Instant::now();
                            execute_node_with_layout_kind(
                                mid_node,
                                &mut env,
                                node_kind(node_kinds, nodes, i + mid),
                            )?;
                            let mid_elapsed = mid_start.elapsed().as_secs_f64() * 1000.0;
                            let mid_entry =
                                timings.entry(mid_node.op_type.clone()).or_insert((0.0, 0));
                            mid_entry.0 += mid_elapsed;
                            mid_entry.1 += 1;
                            let mid_out = mid_node
                                .outputs
                                .first()
                                .and_then(|n| env.get(n))
                                .map(|t| t.shape().to_vec())
                                .unwrap_or_default();
                            instance_timings.push(NodeTiming {
                                name: mid_node.name.clone(),
                                op: mid_node.op_type.clone(),
                                ms: mid_elapsed,
                                in_shape: mid_in,
                                out_shape: mid_out,
                            });
                            skip[i + mid] = true;
                        }
                    }
                    if i + look < skip.len() {
                        skip[i + look] = true;
                    }
                    found_silu = true;
                    break;
                }
            }
            if found_silu {
                continue;
            }
        }

        let op_type = node.op_type.clone();
        let in_shape = node
            .inputs
            .first()
            .and_then(|name| env.get(name))
            .map(|t| t.shape().to_vec())
            .unwrap_or_default();

        let start = Instant::now();
        execute_node_with_layout_kind(node, &mut env, kind)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        let out_shape = node
            .outputs
            .first()
            .and_then(|n| env.get(n))
            .map(|t| t.shape().to_vec())
            .unwrap_or_default();

        if kind == NodeKind::Conv {
            let kernel_shape = match node.attributes.get("kernel_shape") {
                Some(OnnxAttribute::Ints(v)) => v.clone(),
                _ => Vec::new(),
            };
            let strides = match node.attributes.get("strides") {
                Some(OnnxAttribute::Ints(v)) => v.clone(),
                _ => vec![1, 1],
            };
            conv_details.push(ConvDetail {
                name: node.name.clone(),
                ms: elapsed,
                in_shape: in_shape.clone(),
                out_shape: out_shape.clone(),
                kernel_shape,
                strides,
            });
        }

        instance_timings.push(NodeTiming {
            name: node.name.clone(),
            op: op_type.clone(),
            ms: elapsed,
            in_shape,
            out_shape,
        });

        let entry = timings.entry(op_type).or_insert((0.0, 0));
        entry.0 += elapsed;
        entry.1 += 1;
    }

    for name in &model.outputs {
        env.materialize_quant_i8_raw(name)?;
        ensure_nchw(&mut env, name)?;
    }

    println!("\n  ── CPU Profile (per-op timing) ──");
    let mut sorted: Vec<_> = timings.into_iter().collect();
    sorted.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());
    let total: f64 = sorted.iter().map(|(_, (t, _))| t).sum();
    for (op, (time_ms, count)) in &sorted {
        println!("    {:>8.2}ms {:>5}x  {}", time_ms, count, op);
    }
    println!("    {:>8.2}ms  total", total);

    // Per-Conv detail: top 10 slowest
    if !conv_details.is_empty() {
        conv_details.sort_by(|a, b| b.ms.partial_cmp(&a.ms).unwrap());
        println!("\n  ── Top Conv layers ──");
        for detail in conv_details.iter().take(10) {
            println!(
                "    {:>6.2}ms  {:?} → {:?}  k={:?} s={:?} at {}",
                detail.ms,
                detail.in_shape,
                detail.out_shape,
                detail.kernel_shape,
                detail.strides,
                detail.name
            );
        }
    }

    // JSON dump for `scripts/gap_diff.py`. Written only when the env var
    // names a target path — avoids side-effects in the default UX.
    if let Ok(path) = std::env::var("YSCV_PROFILE_JSON")
        && !path.is_empty()
    {
        write_profile_json(&path, &instance_timings, total)?;
        eprintln!("  ── wrote profile JSON to {path} ──");
    }

    Ok(())
}

/// Writes a per-instance timing profile as JSON. Format is stable — it's
/// consumed by `scripts/gap_diff.py`; if you change fields, update the
/// parser in lockstep. We hand-roll the JSON instead of pulling in serde
/// because the schema is tiny (strings + numbers + shape arrays) and the
/// crate already compiles cleanly without a JSON dep.
fn write_profile_json(path: &str, nodes: &[NodeTiming], total_ms: f64) -> Result<(), OnnxError> {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(64 * nodes.len() + 128);
    out.push_str("{\"engine\":\"yscv\",\"total_ms\":");
    let _ = write!(out, "{:.6}", total_ms);
    out.push_str(",\"nodes\":[");
    for (i, n) in nodes.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str("{\"name\":\"");
        json_escape_into(&mut out, &n.name);
        out.push_str("\",\"op\":\"");
        json_escape_into(&mut out, &n.op);
        out.push_str("\",\"ms\":");
        let _ = write!(out, "{:.6}", n.ms);
        out.push_str(",\"in_shape\":");
        shape_to_json(&mut out, &n.in_shape);
        out.push_str(",\"out_shape\":");
        shape_to_json(&mut out, &n.out_shape);
        out.push('}');
    }
    out.push_str("]}\n");
    std::fs::write(path, out).map_err(|e| OnnxError::DecodeFailed {
        message: format!("write profile JSON to {path}: {e}"),
    })
}

/// Escapes a string into an existing JSON output buffer. Covers only the
/// characters that might appear in ONNX node names / op types: backslash,
/// quote, and control characters are escaped; everything else goes as-is.
fn json_escape_into(out: &mut String, s: &str) {
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                use std::fmt::Write as _;
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
}

/// Formats a `[usize]` shape as a JSON number array.
fn shape_to_json(out: &mut String, shape: &[usize]) {
    use std::fmt::Write as _;
    out.push('[');
    for (i, d) in shape.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let _ = write!(out, "{d}");
    }
    out.push(']');
}
