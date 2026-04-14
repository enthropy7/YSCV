// GRU backward: per-timestep gate gradient computation.
// Gates: r (reset), z (update), n (candidate)
// Each thread handles one hidden unit.
struct Params { hidden: u32, timestep: u32 }
@group(0) @binding(0) var<storage, read> gate_r: array<f32>;
@group(0) @binding(1) var<storage, read> gate_z: array<f32>;
@group(0) @binding(2) var<storage, read> gate_n: array<f32>;
@group(0) @binding(3) var<storage, read> h_prev: array<f32>;
@group(0) @binding(4) var<storage, read> grad_h: array<f32>;
@group(0) @binding(5) var<storage, read_write> d_gates: array<f32>; // [3 * hidden]: r,z,n
@group(0) @binding(6) var<uniform> p: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.hidden) { return; }
    let h = gid.x;
    let t = p.timestep;
    let off = t * p.hidden + h;
    let r_val = gate_r[off];
    let z_val = gate_z[off];
    let n_val = gate_n[off];
    let h_p = h_prev[off];
    let dh = grad_h[off];
    // d_h_tilde = dh * (1 - z) * (1 - n^2)
    let dn = dh * (1.0 - z_val) * (1.0 - n_val * n_val);
    // d_z = dh * (h_prev - n) * z * (1 - z)
    let dz = dh * (h_p - n_val) * z_val * (1.0 - z_val);
    // d_r = dn * (W_hh @ h_prev component) * r * (1 - r) — simplified
    let dr = dn * h_p * r_val * (1.0 - r_val);
    d_gates[h] = dr;
    d_gates[p.hidden + h] = dz;
    d_gates[2u * p.hidden + h] = dn;
}
