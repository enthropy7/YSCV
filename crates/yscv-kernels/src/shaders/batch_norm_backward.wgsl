// BatchNorm2d input backward: grad_input[i] = upstream[i] * gamma[ch] / sqrt(var[ch] + eps)
// NHWC layout: channel index = i % C.
struct Params { len: u32, c: u32, epsilon: f32 }
@group(0) @binding(0) var<storage, read> upstream: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> running_var: array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_input: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.len) { return; }
    let ch = i % p.c;
    let inv_std = 1.0 / sqrt(running_var[ch] + p.epsilon);
    grad_input[i] = upstream[i] * gamma[ch] * inv_std;
}
