// Conv2d weight gradient: dL/dW[ky,kx,ci,co] = sum over N,OH,OW of
// input[b,oy*sh+ky,ox*sw+kx,ci] * upstream[b,oy,ox,co]. NHWC layout.
struct Params {
    n: u32, ih: u32, iw: u32, c_in: u32,
    oh: u32, ow: u32, c_out: u32,
    kh: u32, kw: u32,
    sh: u32, sw: u32,
}
@group(0) @binding(0) var<storage, read> upstream: array<f32>;
@group(0) @binding(1) var<storage, read> forward_input: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_weight: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = p.kh * p.kw * p.c_in * p.c_out;
    if (gid.x >= total) { return; }
    let idx = gid.x;
    let co = idx % p.c_out;
    let ci = (idx / p.c_out) % p.c_in;
    let kx = (idx / (p.c_out * p.c_in)) % p.kw;
    let ky = idx / (p.c_out * p.c_in * p.kw);
    var g: f32 = 0.0;
    for (var b: u32 = 0u; b < p.n; b++) {
        for (var oy: u32 = 0u; oy < p.oh; oy++) {
            for (var ox: u32 = 0u; ox < p.ow; ox++) {
                let iy = oy * p.sh + ky;
                let ix = ox * p.sw + kx;
                let in_val = forward_input[((b * p.ih + iy) * p.iw + ix) * p.c_in + ci];
                let up_val = upstream[((b * p.oh + oy) * p.ow + ox) * p.c_out + co];
                g += in_val * up_val;
            }
        }
    }
    grad_weight[idx] = g;
}
