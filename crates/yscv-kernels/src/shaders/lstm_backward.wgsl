// LSTM backward: per-timestep gate gradient computation.
// Each thread handles one hidden unit. The BPTT loop over timesteps
// runs on CPU; this shader computes per-element gate derivatives
// for one timestep.
// Gates: i (input), f (forget), g (cell candidate), o (output)
// grad_cell[t] = grad_h[t] * o[t] * (1 - tanh(c[t])^2) + grad_cell[t+1] * f[t+1]
// d_i = grad_cell[t] * g[t] * i[t] * (1 - i[t])
// d_f = grad_cell[t] * c[t-1] * f[t] * (1 - f[t])
// d_g = grad_cell[t] * i[t] * (1 - g[t]^2)
// d_o = grad_h[t] * tanh(c[t]) * o[t] * (1 - o[t])
struct Params { hidden: u32, timestep: u32 }
@group(0) @binding(0) var<storage, read> gate_i: array<f32>;
@group(0) @binding(1) var<storage, read> gate_f: array<f32>;
@group(0) @binding(2) var<storage, read> gate_g: array<f32>;
@group(0) @binding(3) var<storage, read> gate_o: array<f32>;
@group(0) @binding(4) var<storage, read> cell_states: array<f32>;
@group(0) @binding(5) var<storage, read> grad_h: array<f32>;
@group(0) @binding(6) var<storage, read> grad_cell_next: array<f32>;
@group(0) @binding(7) var<storage, read_write> d_gates: array<f32>; // [4 * hidden]: i,f,g,o
@group(0) @binding(8) var<uniform> p: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.hidden) { return; }
    let h = gid.x;
    let t = p.timestep;
    let off = t * p.hidden + h;
    let i_val = gate_i[off];
    let f_val = gate_f[off];
    let g_val = gate_g[off];
    let o_val = gate_o[off];
    let c_val = cell_states[off];
    let tanh_c = tanh(c_val);
    let dh = grad_h[off];
    let dc_next = grad_cell_next[h]; // from t+1
    let dc = dh * o_val * (1.0 - tanh_c * tanh_c) + dc_next * f_val;
    d_gates[h] = dc * g_val * i_val * (1.0 - i_val);            // d_i
    d_gates[p.hidden + h] = dc * cell_states[max(0u, t * p.hidden + h - p.hidden)] * f_val * (1.0 - f_val); // d_f (needs c[t-1])
    d_gates[2u * p.hidden + h] = dc * i_val * (1.0 - g_val * g_val); // d_g
    d_gates[3u * p.hidden + h] = dh * tanh_c * o_val * (1.0 - o_val); // d_o
}
