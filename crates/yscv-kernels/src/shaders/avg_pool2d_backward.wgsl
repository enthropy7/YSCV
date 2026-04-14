// AvgPool2d backward: uniformly scatter upstream gradient to all positions
// in each pooling window. NHWC layout.
struct Params {
    n: u32, ih: u32, iw: u32, c: u32,
    oh: u32, ow: u32,
    kh: u32, kw: u32,
    sh: u32, sw: u32,
}
@group(0) @binding(0) var<storage, read> upstream: array<f32>;
@group(0) @binding(1) var<storage, read_write> grad_input: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = p.n * p.ih * p.iw * p.c;
    if (gid.x >= total) { return; }
    let idx = gid.x;
    let ch = idx % p.c;
    let ix = (idx / p.c) % p.iw;
    let iy = (idx / (p.c * p.iw)) % p.ih;
    let b = idx / (p.c * p.iw * p.ih);
    let pool_size = f32(p.kh * p.kw);
    var grad: f32 = 0.0;
    let oy_start = max(0i, i32(iy) - i32(p.kh) + 1i) / i32(p.sh);
    let oy_end = min(i32(p.oh), i32(iy) / i32(p.sh) + 1i);
    let ox_start = max(0i, i32(ix) - i32(p.kw) + 1i) / i32(p.sw);
    let ox_end = min(i32(p.ow), i32(ix) / i32(p.sw) + 1i);
    for (var oy = max(0i, oy_start); oy < oy_end; oy++) {
        for (var ox = max(0i, ox_start); ox < ox_end; ox++) {
            grad += upstream[((b * p.oh + u32(oy)) * p.ow + u32(ox)) * p.c + ch] / pool_size;
        }
    }
    grad_input[idx] = grad;
}
