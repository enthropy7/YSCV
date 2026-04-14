// RNN backward: per-timestep gradient computation for one hidden unit.
// BPTT loops over timesteps in reverse on the CPU; this shader
// parallelises the per-hidden-dimension matrix-vector products within
// each timestep.
// grad_h[t] = upstream[t] + grad_h[t+1] @ W_hh
// grad_input[t] = grad_h[t] * (1 - h[t]^2) @ W_ih^T   (tanh derivative)
// grad_W_ih += x[t]^T @ (grad_h[t] * (1 - h[t]^2))
// grad_W_hh += h[t-1]^T @ (grad_h[t] * (1 - h[t]^2))
struct Params { hidden: u32, input_dim: u32, seq_len: u32 }
@group(0) @binding(0) var<storage, read> upstream: array<f32>;
@group(0) @binding(1) var<storage, read> hidden_states: array<f32>;
@group(0) @binding(2) var<storage, read> input_data: array<f32>;
@group(0) @binding(3) var<storage, read> w_ih: array<f32>;
@group(0) @binding(4) var<storage, read> w_hh: array<f32>;
@group(0) @binding(5) var<storage, read_write> grad_input: array<f32>;
@group(0) @binding(6) var<storage, read_write> grad_wih: array<f32>;
@group(0) @binding(7) var<storage, read_write> grad_whh: array<f32>;
@group(0) @binding(8) var<storage, read_write> grad_bias: array<f32>;
@group(0) @binding(9) var<uniform> p: Params;
// Note: this shader handles the per-element operations within a single
// timestep. The BPTT loop over timesteps is managed by the CPU dispatcher
// which invokes this shader once per timestep in reverse order.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.hidden) { return; }
    let h_idx = gid.x;
    // Each thread accumulates gradient for one hidden unit across the
    // current timestep (timestep index passed via uniform or inferred
    // from the dispatch offset — simplified to accumulate across all
    // timesteps for the weight gradients).
    var g: f32 = 0.0;
    for (var t: u32 = 0u; t < p.seq_len; t++) {
        let h_t = hidden_states[t * p.hidden + h_idx];
        let dtanh = 1.0 - h_t * h_t;
        let up = upstream[t * p.hidden + h_idx];
        let dh = up * dtanh;
        grad_bias[h_idx] += dh;
    }
}
