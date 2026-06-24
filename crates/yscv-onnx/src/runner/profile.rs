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
/// without contending on the hot path. A per-thread `RefCell` pays zero
/// synchronisation on record — each thread is the sole writer of its own
/// store, which matters under tower-parallel runs where two branch threads
/// would otherwise contend on a shared lock inside per-op timing.
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
    /// Dispatched kernel label (Conv sub-path / MatMul-Gemm family), captured
    /// from the per-thread recorder on first sighting. `None` for ops with no
    /// recorder, or kernels (e.g. fused DW+PW streaming) that bypass the
    /// instrumented dispatch.
    kernel: Option<String>,
}

/// Reads the dispatched-kernel label for a just-executed node off the
/// per-thread recorders, consuming it. The fused runner records a node right
/// after it runs on the same thread, so the slot holds this node's dispatch.
fn runner_profile_kernel_label(op: &str) -> Option<String> {
    if op.starts_with("Conv") {
        take_conv_kernel().map(|d| d.label())
    } else if op == "MatMul" || op == "Gemm" {
        yscv_kernels::take_matmul_kernel().map(|k| k.label().to_string())
    } else {
        None
    }
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
    let kernel = runner_profile_kernel_label(op);
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
                    kernel: kernel.clone(),
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
            if entry.kernel.is_none() {
                entry.kernel = kernel;
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
                        if e.kernel.is_none() {
                            e.kernel = stat.kernel.clone();
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
        if let Some(k) = &stat.kernel {
            out.push_str(",\"kernel\":\"");
            json_escape_into_local(&mut out, k);
            out.push('"');
        }
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
    /// Conv-only kernel spatial dims / strides from the ONNX attributes; empty
    /// for non-Conv ops.
    kernel_shape: Vec<i64>,
    strides: Vec<i64>,
    /// Rendered dispatch label: for Conv the runner dispatch + kernel-internal
    /// sub-path (`nhwc-padded/first-layer-rgb`), for MatMul the GEMM family
    /// (`blocked-mr6`). `None` for ops with no recorder.
    kernel: Option<String>,
}

/// Node filter for the profiler's detail table and JSON dump, parsed from
/// `YSCV_PROFILE_FILTER`. The spec is a comma-separated list of tokens: a bare
/// token matches an op type exactly (`Conv`, `MatMul`), `name:<substr>` matches
/// a node-name substring. A node matches if it matches any token. When the env
/// var is unset the profiler keeps its default view — the Conv detail table and
/// an unfiltered JSON.
struct ProfileFilter {
    active: bool,
    ops: Vec<String>,
    name_subs: Vec<String>,
    label: String,
}

impl ProfileFilter {
    fn from_env() -> Self {
        Self::parse(&std::env::var("YSCV_PROFILE_FILTER").unwrap_or_default())
    }

    fn parse(raw: &str) -> Self {
        let raw = raw.trim();
        if raw.is_empty() {
            return Self {
                active: false,
                ops: Vec::new(),
                name_subs: Vec::new(),
                label: "Conv".to_string(),
            };
        }
        let mut ops = Vec::new();
        let mut name_subs = Vec::new();
        for tok in raw.split(',').map(str::trim).filter(|t| !t.is_empty()) {
            match tok.strip_prefix("name:") {
                Some(sub) => name_subs.push(sub.to_string()),
                None => ops.push(tok.to_string()),
            }
        }
        Self {
            active: true,
            ops,
            name_subs,
            label: raw.to_string(),
        }
    }

    fn token_match(&self, op: &str, name: &str) -> bool {
        self.ops.iter().any(|o| o == op) || self.name_subs.iter().any(|s| name.contains(s.as_str()))
    }

    /// Whether a node belongs in the detail table. Unset filter → Conv only.
    fn detail_match(&self, op: &str, name: &str) -> bool {
        if self.active {
            self.token_match(op, name)
        } else {
            op == "Conv"
        }
    }

    /// Whether a node belongs in the JSON dump. Unset filter → everything.
    fn json_match(&self, op: &str, name: &str) -> bool {
        !self.active || self.token_match(op, name)
    }
}

/// Profile CPU inference: measure per-op-type timing.
///
/// When the env var `YSCV_PROFILE_JSON` is set to a file path, a
/// machine-readable JSON with per-node instance timings is written to
/// that path in addition to the standard text summary. Conv nodes also
/// carry `kernel_shape`, `strides` and the dispatched `kernel`. Format:
/// ```json
/// {"engine":"yscv","total_ms":11.47,"nodes":[
///   {"name":"Conv_0","op":"Conv","ms":0.12,
///    "in_shape":[1,3,128,128],"out_shape":[1,16,64,64],
///    "kernel_shape":[3,3],"strides":[2,2],"kernel":"nhwc-padded"}, ...
/// ]}
/// ```
/// The file is consumed by `scripts/gap_diff.py` for apples-to-apples
/// comparison against ORT's profiling JSON.
///
/// `YSCV_PROFILE_FILTER` narrows both the detail table and the JSON to a
/// chosen op/name set (e.g. `Conv`, `Conv,MatMul`, `name:head`); see
/// [`ProfileFilter`]. Unset, the profiler shows the Conv detail table and an
/// unfiltered JSON.
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
    let filter = ProfileFilter::from_env();
    // Per-instance timings — feed the detail table and `YSCV_PROFILE_JSON`.
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
                        kernel_shape: Vec::new(),
                        strides: Vec::new(),
                        kernel: None,
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
                                kernel_shape: Vec::new(),
                                strides: Vec::new(),
                                kernel: None,
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

        let (kernel_shape, strides, kernel) = if kind == NodeKind::Conv {
            let kernel_shape = match node.attributes.get("kernel_shape") {
                Some(OnnxAttribute::Ints(v)) => v.clone(),
                _ => Vec::new(),
            };
            let strides = match node.attributes.get("strides") {
                Some(OnnxAttribute::Ints(v)) => v.clone(),
                _ => vec![1, 1],
            };
            (kernel_shape, strides, take_conv_kernel().map(|d| d.label()))
        } else if kind == NodeKind::MatMul || kind == NodeKind::Gemm {
            let kernel = yscv_kernels::take_matmul_kernel().map(|k| k.label().to_string());
            (Vec::new(), Vec::new(), kernel)
        } else {
            (Vec::new(), Vec::new(), None)
        };

        instance_timings.push(NodeTiming {
            name: node.name.clone(),
            op: op_type.clone(),
            ms: elapsed,
            in_shape,
            out_shape,
            kernel_shape,
            strides,
            kernel,
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

    // Per-node detail for the filtered op set (default: Conv): top 10 slowest.
    let mut detail: Vec<&NodeTiming> = instance_timings
        .iter()
        .filter(|n| filter.detail_match(&n.op, &n.name))
        .collect();
    if !detail.is_empty() {
        detail.sort_by(|a, b| b.ms.partial_cmp(&a.ms).unwrap());
        println!("\n  ── Top {} layers ──", filter.label);
        for n in detail.iter().take(10) {
            if n.op == "Conv" {
                println!(
                    "    {:>6.2}ms  {:?} → {:?}  k={:?} s={:?} via {} at {}",
                    n.ms,
                    n.in_shape,
                    n.out_shape,
                    n.kernel_shape,
                    n.strides,
                    n.kernel.as_deref().unwrap_or("?"),
                    n.name,
                );
            } else if let Some(via) = &n.kernel {
                println!(
                    "    {:>6.2}ms  {:?} → {:?}  via {} at {}",
                    n.ms, n.in_shape, n.out_shape, via, n.name,
                );
            } else {
                println!(
                    "    {:>6.2}ms  {:?} → {:?}  {} at {}",
                    n.ms, n.in_shape, n.out_shape, n.op, n.name,
                );
            }
        }
    }

    // JSON dump for `scripts/gap_diff.py`. Written only when the env var
    // names a target path — avoids side-effects in the default UX.
    if let Ok(path) = std::env::var("YSCV_PROFILE_JSON")
        && !path.is_empty()
    {
        let json_nodes: Vec<&NodeTiming> = instance_timings
            .iter()
            .filter(|n| filter.json_match(&n.op, &n.name))
            .collect();
        write_profile_json(&path, &json_nodes, total)?;
        eprintln!("  ── wrote profile JSON to {path} ──");
    }

    Ok(())
}

/// Writes a per-instance timing profile as JSON. Format is stable — it's
/// consumed by `scripts/gap_diff.py`; if you change fields, update the
/// parser in lockstep. We hand-roll the JSON instead of pulling in serde
/// because the schema is tiny (strings + numbers + shape arrays) and the
/// crate already compiles cleanly without a JSON dep.
fn write_profile_json(path: &str, nodes: &[&NodeTiming], total_ms: f64) -> Result<(), OnnxError> {
    std::fs::write(path, profile_json_string(nodes, total_ms)).map_err(|e| {
        OnnxError::DecodeFailed {
            message: format!("write profile JSON to {path}: {e}"),
        }
    })
}

/// Renders the per-node profile as the JSON string consumed by
/// `scripts/gap_diff.py`. Split out from the file write so the schema can be
/// asserted directly in tests.
fn profile_json_string(nodes: &[&NodeTiming], total_ms: f64) -> String {
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
        // Conv extras — emitted only for Conv nodes, so non-Conv entries keep
        // the original schema and `gap_diff.py`'s op-level mode is unaffected.
        if !n.kernel_shape.is_empty() {
            out.push_str(",\"kernel_shape\":");
            ints_to_json(&mut out, &n.kernel_shape);
        }
        if !n.strides.is_empty() {
            out.push_str(",\"strides\":");
            ints_to_json(&mut out, &n.strides);
        }
        if let Some(k) = &n.kernel {
            out.push_str(",\"kernel\":\"");
            json_escape_into(&mut out, k);
            out.push('"');
        }
        out.push('}');
    }
    out.push_str("]}\n");
    out
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

/// Formats an `[i64]` attribute (kernel_shape / strides) as a JSON array.
fn ints_to_json(out: &mut String, values: &[i64]) {
    use std::fmt::Write as _;
    out.push('[');
    for (i, d) in values.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let _ = write!(out, "{d}");
    }
    out.push(']');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_filter_default_is_conv_only() {
        let f = ProfileFilter::parse("   ");
        assert!(!f.active);
        assert_eq!(f.label, "Conv");
        // Detail table defaults to Conv; everything else is excluded.
        assert!(f.detail_match("Conv", "backbone.0"));
        assert!(!f.detail_match("MatMul", "x"));
        // JSON is unfiltered when the env var is unset.
        assert!(f.json_match("MatMul", "x"));
        assert!(f.json_match("Conv", "x"));
    }

    #[test]
    fn profile_filter_matches_ops_and_name_substrings() {
        let f = ProfileFilter::parse("Conv, MatMul, name:head");
        assert!(f.active);
        assert!(f.detail_match("Conv", "backbone.0"));
        assert!(f.detail_match("MatMul", "x"));
        // A name substring matches regardless of op type.
        assert!(f.detail_match("Relu", "head_box"));
        assert!(!f.detail_match("Relu", "tail"));
        // When active, the JSON filter mirrors the token match.
        assert!(f.json_match("Conv", "x"));
        assert!(!f.json_match("Sigmoid", "tail"));
    }

    #[test]
    fn profile_json_emits_conv_extras_only_for_conv() {
        let conv = NodeTiming {
            name: "c0".into(),
            op: "Conv".into(),
            ms: 0.5,
            in_shape: vec![1, 3, 8, 8],
            out_shape: vec![1, 16, 8, 8],
            kernel_shape: vec![3, 3],
            strides: vec![1, 1],
            kernel: Some("nhwc-gemm/pw-gemm".to_string()),
        };
        let relu = NodeTiming {
            name: "r0".into(),
            op: "Relu".into(),
            ms: 0.1,
            in_shape: vec![1, 16, 8, 8],
            out_shape: vec![1, 16, 8, 8],
            kernel_shape: Vec::new(),
            strides: Vec::new(),
            kernel: None,
        };
        let json = profile_json_string(&[&conv, &relu], 0.6);
        // Conv carries the dispatched kernel (runner/sub) + shape attrs.
        assert!(json.contains("\"kernel\":\"nhwc-gemm/pw-gemm\""), "{json}");
        assert!(json.contains("\"kernel_shape\":[3,3]"), "{json}");
        assert!(json.contains("\"strides\":[1,1]"), "{json}");
        // The Relu entry keeps the original schema — no conv fields.
        let relu_obj = json.split("{\"name\":\"r0\"").nth(1).unwrap();
        assert!(
            !relu_obj.contains("kernel"),
            "relu must stay clean: {relu_obj}"
        );
    }

    #[test]
    fn runner_profile_label_skips_unrecorded_ops() {
        // Only Conv* / MatMul / Gemm have a kernel recorder; everything else
        // (incl. fused kernels that bypass the instrumented dispatch, like
        // FusedDwPw) reports no label.
        assert_eq!(runner_profile_kernel_label("Add"), None);
        assert_eq!(runner_profile_kernel_label("Relu"), None);
        assert_eq!(runner_profile_kernel_label("Softmax"), None);
        assert_eq!(runner_profile_kernel_label("FusedDwPw"), None);
    }
}
