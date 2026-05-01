//! Quick model-input/output inspector — prints names + initializer shapes
//! of an ONNX model. Used to detect cached-decoder protocols
//! (past_key_values.{i}.{key,value}) before wiring up the bench harness.

use yscv_onnx::load_onnx_model_from_file;

fn main() {
    let path = std::env::args().nth(1).expect("usage: inspect <path.onnx>");
    let m = load_onnx_model_from_file(&path).expect("load");
    println!("inputs ({}):", m.inputs.len());
    for n in &m.inputs {
        println!("  {n}");
    }
    println!("\noutputs ({}):", m.outputs.len());
    for n in &m.outputs {
        println!("  {n}");
    }
    println!(
        "\ninitializers: {}, nodes: {}",
        m.initializers.len(),
        m.nodes.len()
    );

    // Optional 2nd arg: focus on a specific node name and walk one
    // level back through its input producers.
    if let Some(target) = std::env::args().nth(2) {
        println!("\n=== focus: {target}");
        let target_node = m
            .nodes
            .iter()
            .find(|n| n.name == target || n.outputs.iter().any(|o| o == &target));
        let Some(node) = target_node else {
            println!("(not found)");
            return;
        };
        println!("op: {}  inputs:", node.op_type);
        for inp in &node.inputs {
            print!("  {inp}");
            if let Some(t) = m.initializers.get(inp) {
                println!("  [init shape={:?}]", t.shape());
            } else if let Some(producer) =
                m.nodes.iter().find(|n| n.outputs.iter().any(|o| o == inp))
            {
                println!("  [from {} op={}]", producer.name, producer.op_type);
            } else {
                println!("  [graph input or external]");
            }
        }
    }
}
