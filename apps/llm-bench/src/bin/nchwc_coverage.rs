/// NCHWc layout coverage probe — Step 0 of NCHWc-everywhere plan.
///
/// Loads a tracker ONNX model and classifies every Conv op as NCHWc-eligible
/// or rejected (with reason). Simulates BFS through passthrough ops to count
/// how many required layout-convert boundaries would be needed.
///
/// Usage:
///   cargo run --release --no-default-features -p yscv-llm-bench --bin nchwc_coverage \
///     -- private/private/model.onnx
use std::collections::{HashMap, HashSet};

use yscv_onnx::{OnnxAttribute, OnnxModel, OnnxNode, load_onnx_model_from_file};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("Usage: nchwc_coverage <model.onnx>");
    let model = load_onnx_model_from_file(path).expect("failed to load model");
    run_coverage_probe(&model);
}

fn get_int_attr(node: &OnnxNode, key: &str, default: i64) -> i64 {
    match node.attributes.get(key) {
        Some(OnnxAttribute::Int(v)) => *v,
        Some(OnnxAttribute::Ints(v)) => v.first().copied().unwrap_or(default),
        _ => default,
    }
}

fn get_ints_attr(node: &OnnxNode, key: &str) -> Vec<i64> {
    match node.attributes.get(key) {
        Some(OnnxAttribute::Ints(v)) => v.clone(),
        _ => vec![],
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ConvKind {
    Pointwise,      // group=1, kh=kw=1
    Depthwise3x3S1, // group=Cin, kh=kw=3, stride=1
    Depthwise3x3S2, // group=Cin, kh=kw=3, stride=2
    Depthwise5x5,   // group=Cin, kh=kw=5
    General3x3S2,   // group=1, kh=kw=3, stride=2 (first-layer style)
}

#[derive(Debug, Clone)]
enum Eligibility {
    Eligible(ConvKind),
    Rejected(String),
}

/// Ops that can pass NCHWc tensors through without layout conversion.
fn is_passthrough(op_type: &str) -> bool {
    matches!(
        op_type,
        "Identity"  // weight sharing aliases — fully transparent
            | "Relu"
            | "LeakyRelu"
            | "Sigmoid"
            | "Silu"
            | "HardSigmoid"
            | "HardSwish"
            | "Mul"
            | "Add"
            | "Sub"
            | "MaxPool"
            | "AveragePool"
            | "GlobalAveragePool"
            | "Clip"
            | "Pad"
            | "BatchNormalization"
            | "InstanceNormalization"
            | "Dropout"
    )
}

/// Build a map: tensor_name → shape, tracing through Identity nodes to
/// resolve weight-sharing aliases. In Siamese networks, branch1 weights
/// are Identity(branch0_initializer) — treat them as static weights.
fn build_effective_initializers(model: &OnnxModel) -> HashMap<String, Vec<usize>> {
    let mut effective: HashMap<String, Vec<usize>> = model
        .initializers
        .iter()
        .map(|(k, v)| (k.clone(), v.shape().to_vec()))
        .collect();

    // Trace Identity nodes: if Identity's input is a static weight (direct
    // or already resolved), its output inherits the same shape.
    // Iterate repeatedly until no new entries are added.
    loop {
        let mut added = 0;
        for node in &model.nodes {
            if node.op_type != "Identity" {
                continue;
            }
            let inp = node.inputs.first().map(|s| s.as_str()).unwrap_or("");
            let out = node.outputs.first().map(|s| s.as_str()).unwrap_or("");
            if out.is_empty() || effective.contains_key(out) {
                continue;
            }
            if let Some(shape) = effective.get(inp).cloned() {
                effective.insert(out.to_string(), shape);
                added += 1;
            }
        }
        if added == 0 {
            break;
        }
    }
    effective
}

fn run_coverage_probe(model: &OnnxModel) {
    // Resolve weight shapes, including Identity-aliased weights (Siamese branches).
    let effective_weights = build_effective_initializers(model);
    println!(
        "Effective static weights (incl. Identity aliases): {}",
        effective_weights.len()
    );
    println!("Raw initializers: {}", model.initializers.len());
    println!();

    // Count all conv ops
    let conv_ops: Vec<(usize, &yscv_onnx::OnnxNode)> = model
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| {
            matches!(
                n.op_type.as_str(),
                "Conv" | "Conv_Relu" | "Conv_SiLU" | "ConvRelu" | "ConvSilu"
            )
        })
        .collect();

    println!("Total Conv ops: {}", conv_ops.len());
    println!("Total nodes: {}", model.nodes.len());
    println!();

    // Classify each conv op
    let mut eligible = 0usize;
    let mut rejected = 0usize;
    let mut kind_counts: HashMap<String, usize> = HashMap::new();
    let mut reject_reasons: HashMap<String, usize> = HashMap::new();

    for (_, node) in &conv_ops {
        let weight_name = node.inputs.get(1).map(|s| s.as_str()).unwrap_or("");
        // Use effective_weights to trace through Identity nodes (Siamese weight sharing)
        let weight_shape = effective_weights
            .get(weight_name)
            .cloned()
            .unwrap_or_default();

        let group = get_int_attr(node, "group", 1);
        let strides = get_ints_attr(node, "strides");
        let sh = strides.first().copied().unwrap_or(1);
        let sw = strides.get(1).copied().unwrap_or(1);
        let dilations = get_ints_attr(node, "dilations");
        let dh = dilations.first().copied().unwrap_or(1);
        let dw_d = dilations.get(1).copied().unwrap_or(1);

        let elig = if dh != 1 || dw_d != 1 {
            Eligibility::Rejected(format!("dilation={dh}×{dw_d}"))
        } else if weight_shape.is_empty() {
            Eligibility::Rejected("no static weight".to_string())
        } else if weight_shape.len() != 4 {
            Eligibility::Rejected(format!("weight rank={}", weight_shape.len()))
        } else {
            // Determine kernel dims
            let (kh, kw) = if group > 1 {
                let a = weight_shape[0];
                let b = weight_shape[1];
                if a <= 7 && b <= 7 {
                    (a as i64, b as i64)
                } else {
                    (weight_shape[2] as i64, weight_shape[3] as i64)
                }
            } else {
                // KHWC [kh,kw,cin,cout]
                (weight_shape[0] as i64, weight_shape[1] as i64)
            };

            if kh > 5 || kw > 5 {
                Eligibility::Rejected(format!("kernel {kh}×{kw} > 5×5"))
            } else if sh > 2 || sw > 2 {
                Eligibility::Rejected(format!("stride {sh}×{sw} > 2"))
            } else {
                if group == 1 && kh == 1 && kw == 1 {
                    Eligibility::Eligible(ConvKind::Pointwise)
                } else if group > 1 && kh == 3 && kw == 3 && sh == 1 {
                    Eligibility::Eligible(ConvKind::Depthwise3x3S1)
                } else if group > 1 && kh == 3 && kw == 3 && sh == 2 {
                    Eligibility::Eligible(ConvKind::Depthwise3x3S2)
                } else if group > 1 && kh == 5 && kw == 5 {
                    Eligibility::Eligible(ConvKind::Depthwise5x5)
                } else if group == 1 && kh == 3 && kw == 3 && sh == 2 {
                    Eligibility::Eligible(ConvKind::General3x3S2)
                } else {
                    Eligibility::Rejected(format!(
                        "group={group} kernel={kh}×{kw} stride={sh}×{sw}"
                    ))
                }
            }
        };

        match &elig {
            Eligibility::Eligible(kind) => {
                eligible += 1;
                *kind_counts.entry(format!("{kind:?}")).or_default() += 1;
            }
            Eligibility::Rejected(reason) => {
                rejected += 1;
                *reject_reasons.entry(reason.clone()).or_default() += 1;
            }
        }
    }

    println!(
        "NCHWc-eligible: {eligible}/{} ({:.1}%)",
        conv_ops.len(),
        eligible as f64 / conv_ops.len() as f64 * 100.0
    );
    println!("Rejected: {rejected}");
    println!();

    println!("Eligible by kind:");
    let mut kinds: Vec<_> = kind_counts.iter().collect();
    kinds.sort_by_key(|(k, _)| (*k).clone());
    for (kind, count) in &kinds {
        println!("  {:30} {count}", kind);
    }

    if !reject_reasons.is_empty() {
        println!();
        println!("Reject reasons:");
        let mut reasons: Vec<_> = reject_reasons.iter().collect();
        reasons.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
        for (reason, count) in reasons {
            println!("  {count:3}× {reason}");
        }
    }

    // Show sample rejected ops to diagnose "no static weight" cause
    println!();
    println!("Sample rejected Conv ops (first 10):");
    let mut shown = 0;
    for (_, node) in &conv_ops {
        let weight_name = node.inputs.get(1).map(|s| s.as_str()).unwrap_or("");
        let in_init = model.initializers.contains_key(weight_name);
        if !in_init {
            let input0 = node.inputs.first().map(|s| s.as_str()).unwrap_or("");
            println!(
                "  {} | op={} | weight_input='{}' | data_input='{}...'",
                node.name
                    .get(..40.min(node.name.len()))
                    .unwrap_or(&node.name),
                node.op_type,
                weight_name
                    .get(..30.min(weight_name.len()))
                    .unwrap_or(weight_name),
                input0.get(..30.min(input0.len())).unwrap_or(input0),
            );
            shown += 1;
            if shown >= 10 {
                break;
            }
        }
    }

    // Investigate Identity weight-sharing: trace onnx::Conv_1037 producer
    {
        let target = "onnx::Conv_1037";
        for node in &model.nodes {
            if node.outputs.iter().any(|o| o == target) {
                println!();
                println!(
                    "Producer of '{target}': op={} name={}",
                    node.op_type, node.name
                );
                println!("  inputs: {:?}", node.inputs);
                // Check if its input is an initializer
                for inp in &node.inputs {
                    if model.initializers.contains_key(inp.as_str()) {
                        println!("  → input '{}' IS in initializers!", inp);
                    }
                }
            }
        }
    }

    // Investigate: what are the branch0 equivalent weight names?
    println!();
    println!("Branch0 vs branch1 weight comparison (first 5 matching ops):");
    let b0_ops: Vec<_> = conv_ops
        .iter()
        .filter(|(_, n)| n.name.contains("/conv/"))
        .take(5)
        .collect();
    let b1_ops: Vec<_> = conv_ops
        .iter()
        .filter(|(_, n)| n.name.contains("/conv_1/"))
        .take(5)
        .collect();
    for (b0, b1) in b0_ops.iter().zip(b1_ops.iter()) {
        let w0 = b0.1.inputs.get(1).map(|s| s.as_str()).unwrap_or("");
        let w1 = b1.1.inputs.get(1).map(|s| s.as_str()).unwrap_or("");
        let in_init0 = model.initializers.contains_key(w0);
        let in_init1 = model.initializers.contains_key(w1);
        println!(
            "  b0: {} weight='{}' (in_init={})",
            &b0.1.name[..b0.1.name.len().min(30)],
            w0,
            in_init0
        );
        println!(
            "  b1: {} weight='{}' (in_init={})",
            &b1.1.name[..b1.1.name.len().min(30)],
            w1,
            in_init1
        );
        println!("  same weight? {}", w0 == w1);
    }
    println!();
    println!("Total initializers: {}", model.initializers.len());

    // Count QLinearConv ops
    let qlinear_conv: Vec<_> = model
        .nodes
        .iter()
        .filter(|n| n.op_type == "QLinearConv")
        .collect();
    println!();
    println!("QLinearConv ops in model: {}", qlinear_conv.len());
    // Show all non-Conv node op types and counts
    let mut op_counts: HashMap<&str, usize> = HashMap::new();
    for node in &model.nodes {
        *op_counts.entry(node.op_type.as_str()).or_default() += 1;
    }
    println!("All op types:");
    let mut ops: Vec<_> = op_counts.iter().collect();
    ops.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
    for (op, count) in ops {
        println!("  {count:4}× {op}");
    }

    // --- BFS to count layout convert boundaries ---
    // Tensor name → whether it's in NCHWc domain
    // Start: graph inputs are NHWC → nchwc converter needed before first NCHWc op
    // Passthrough ops propagate the layout of their primary input
    // Non-passthrough, non-Conv ops require NHWC → insert converter at boundary

    // Build tensor → producing node index
    let mut tensor_producer: HashMap<&str, usize> = HashMap::new();
    for (i, node) in model.nodes.iter().enumerate() {
        for out in &node.outputs {
            if !out.is_empty() {
                tensor_producer.insert(out.as_str(), i);
            }
        }
    }

    // Build node → consumers (node indices)
    let mut node_consumers: Vec<Vec<usize>> = vec![vec![]; model.nodes.len()];
    for (i, node) in model.nodes.iter().enumerate() {
        for inp in &node.inputs {
            if let Some(&prod) = tensor_producer.get(inp.as_str()) {
                node_consumers[prod].push(i);
            }
        }
    }

    // Classify each node eligible for NCHWc
    let node_eligible: Vec<bool> = model
        .nodes
        .iter()
        .map(|node| {
            if matches!(
                node.op_type.as_str(),
                "Conv" | "Conv_Relu" | "Conv_SiLU" | "ConvRelu" | "ConvSilu"
            ) {
                let weight_name = node.inputs.get(1).map(|s| s.as_str()).unwrap_or("");
                // Use effective_weights (traces through Identity nodes)
                let weight_shape = effective_weights
                    .get(weight_name)
                    .cloned()
                    .unwrap_or_default();

                let group = get_int_attr(node, "group", 1);
                let strides = get_ints_attr(node, "strides");
                let sh = strides.first().copied().unwrap_or(1);
                let sw = strides.get(1).copied().unwrap_or(1);
                let dilations = get_ints_attr(node, "dilations");
                let dh = dilations.first().copied().unwrap_or(1);
                let dw_d = dilations.get(1).copied().unwrap_or(1);

                if dh != 1 || dw_d != 1 || weight_shape.len() != 4 {
                    return false;
                }
                let (kh, kw) = if group > 1 {
                    let a = weight_shape[0];
                    let b = weight_shape[1];
                    if a <= 7 && b <= 7 {
                        (a as i64, b as i64)
                    } else {
                        (weight_shape[2] as i64, weight_shape[3] as i64)
                    }
                } else {
                    (weight_shape[0] as i64, weight_shape[1] as i64)
                };
                kh <= 5 && kw <= 5 && sh <= 2 && sw <= 2
            } else {
                is_passthrough(&node.op_type)
            }
        })
        .collect();

    // BFS: propagate "nchwc domain" from each graph input
    // A node enters nchwc domain if:
    //   - ALL its non-initializer inputs come from nchwc tensors OR graph inputs (after converter)
    //   - AND node itself is eligible (Conv kind) or passthrough
    // We count: number of entry converters (one per dynamic graph input that feeds ≥1 eligible node)
    // And number of exit converters (nchwc tensors consumed by non-nchwc non-Conv nodes)

    // Use effective_weights keys as the "static" set (includes Identity-aliased weights)
    let static_names: HashSet<&str> = effective_weights.keys().map(|s| s.as_str()).collect();
    let initializer_names: HashSet<&str> = static_names.clone();
    let dynamic_inputs: Vec<&str> = model
        .inputs
        .iter()
        .map(|s| s.as_str())
        .filter(|s| !initializer_names.contains(s))
        .collect();

    println!();
    println!("Dynamic graph inputs: {}", dynamic_inputs.len());
    for inp in &dynamic_inputs {
        println!("  {inp}");
    }

    // Mark which tensors are in nchwc domain (after potential converter)
    let mut nchwc_tensors: HashSet<String> = HashSet::new();
    // Graph inputs start as NHWC; we will count converters needed for them
    // if any eligible node consumes them directly.

    // BFS over topological order (nodes are already in topo order in ONNX)
    let mut entry_converts = 0usize;
    let mut internal_boundaries = 0usize;

    // Track which dynamic inputs get a converter
    let mut converted_inputs: HashSet<String> = HashSet::new();

    for (i, node) in model.nodes.iter().enumerate() {
        if !node_eligible[i] && !matches!(node.op_type.as_str(), "Conv" | "Conv_Relu" | "Conv_SiLU")
        {
            // Non-NCHWc node: outputs are NHWC.
            continue;
        }

        if !node_eligible[i] {
            // Rejected Conv — treat as non-NCHWc.
            continue;
        }

        // Eligible node: check all non-initializer inputs
        for inp in &node.inputs {
            if inp.is_empty() || initializer_names.contains(inp.as_str()) {
                continue;
            }
            if dynamic_inputs.contains(&inp.as_str()) {
                // Graph input needs converter
                if !converted_inputs.contains(inp.as_str()) {
                    converted_inputs.insert(inp.to_string());
                    entry_converts += 1;
                }
                nchwc_tensors.insert(inp.to_string());
            } else if !nchwc_tensors.contains(inp) {
                // Intermediate tensor coming from non-nchwc node
                internal_boundaries += 1;
                nchwc_tensors.insert(inp.to_string());
            }
        }

        // Outputs are NCHWc
        for out in &node.outputs {
            if !out.is_empty() {
                nchwc_tensors.insert(out.clone());
            }
        }
    }

    // Count exit converters for graph outputs
    let mut output_converts = 0usize;
    for out in &model.outputs {
        if nchwc_tensors.contains(out) {
            output_converts += 1;
        }
    }

    println!();
    println!("Layout converter analysis:");
    println!("  Entry converters (graph input → NCHWc): {entry_converts}");
    println!("  Exit converters  (NCHWc → graph output): {output_converts}");
    println!("  Internal boundaries (mid-graph NHWC→NCHWc): {internal_boundaries}");
    println!(
        "  Total LayoutConvert ops per inference: {}",
        entry_converts + output_converts + internal_boundaries
    );

    println!();
    let coverage = eligible as f64 / conv_ops.len() as f64 * 100.0;
    if coverage >= 85.0 && entry_converts + output_converts + internal_boundaries <= 6 {
        println!(
            "✓ PASS: {coverage:.1}% coverage, {} total converters — proceed to layout_planner.rs",
            entry_converts + output_converts + internal_boundaries
        );
    } else if coverage >= 80.0 {
        println!("⚠ MARGINAL: {coverage:.1}% coverage — review rejected ops before proceeding");
    } else {
        println!(
            "✗ FAIL: {coverage:.1}% coverage < 80% — need to expand eligibility rules or switch to scoped NCHWc strategy"
        );
    }
}
