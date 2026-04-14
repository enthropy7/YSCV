// Conv2d bias gradient: reduce upstream over N×H×W, keeping C_out.
// Each thread handles one output channel.
struct Params { total_elements: u32, c_out: u32 }
@group(0) @binding(0) var<storage, read> upstream: array<f32>;
@group(0) @binding(1) var<storage, read_write> grad_bias: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let co = gid.x;
    if (co >= p.c_out) { return; }
    var g: f32 = 0.0;
    for (var i: u32 = co; i < p.total_elements; i += p.c_out) {
        g += upstream[i];
    }
    grad_bias[co] = g;
}
