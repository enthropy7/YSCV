// Softmax backward: dL/dx_i = s_i * (dL/dy_i - sum_j(dL/dy_j * s_j))
// where s = softmax output (forward_output).
// Each workgroup handles one row (batch × ... × last_dim).
struct Params { rows: u32, cols: u32 }
@group(0) @binding(0) var<storage, read> upstream: array<f32>;
@group(0) @binding(1) var<storage, read> forward_output: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_input: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= p.rows) { return; }
    let base = row * p.cols;
    // Compute dot product: sum_j(dL/dy_j * s_j)
    var dot: f32 = 0.0;
    for (var j: u32 = 0u; j < p.cols; j++) {
        dot += upstream[base + j] * forward_output[base + j];
    }
    // grad_input[i] = s[i] * (upstream[i] - dot)
    for (var j: u32 = 0u; j < p.cols; j++) {
        grad_input[base + j] = forward_output[base + j] * (upstream[base + j] - dot);
    }
}
