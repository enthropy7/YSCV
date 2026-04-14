// MaxPool2d backward: scatter upstream gradient to the position of the
// maximum value in each pooling window. NHWC layout.
// Each thread handles one element of the INPUT gradient.
struct Params {
    n: u32, ih: u32, iw: u32, c: u32,
    oh: u32, ow: u32,
    kh: u32, kw: u32,
    sh: u32, sw: u32,
}
@group(0) @binding(0) var<storage, read> upstream: array<f32>;
@group(0) @binding(1) var<storage, read> forward_input: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_input: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = p.n * p.ih * p.iw * p.c;
    if (gid.x >= total) { return; }
    let idx = gid.x;
    let ch = idx % p.c;
    let ix = (idx / p.c) % p.iw;
    let iy = (idx / (p.c * p.iw)) % p.ih;
    let b = idx / (p.c * p.iw * p.ih);
    // For each output position that this input position contributes to,
    // check if this input was the max — if so, accumulate the upstream grad.
    var grad: f32 = 0.0;
    let in_val = forward_input[idx];
    // Output positions that could have pooled from (iy, ix):
    let oy_start = max(0i, i32(iy) - i32(p.kh) + 1i) / i32(p.sh);
    let oy_end = min(i32(p.oh), i32(iy) / i32(p.sh) + 1i);
    let ox_start = max(0i, i32(ix) - i32(p.kw) + 1i) / i32(p.sw);
    let ox_end = min(i32(p.ow), i32(ix) / i32(p.sw) + 1i);
    for (var oy = max(0i, oy_start); oy < oy_end; oy++) {
        for (var ox = max(0i, ox_start); ox < ox_end; ox++) {
            // Find the max in this pooling window
            var max_val: f32 = -1e38;
            let base_iy = u32(oy) * p.sh;
            let base_ix = u32(ox) * p.sw;
            for (var ky: u32 = 0u; ky < p.kh; ky++) {
                for (var kx: u32 = 0u; kx < p.kw; kx++) {
                    let sy = base_iy + ky;
                    let sx = base_ix + kx;
                    if (sy < p.ih && sx < p.iw) {
                        let v = forward_input[((b * p.ih + sy) * p.iw + sx) * p.c + ch];
                        max_val = max(max_val, v);
                    }
                }
            }
            if (abs(in_val - max_val) < 1e-6) {
                grad += upstream[((b * p.oh + u32(oy)) * p.ow + u32(ox)) * p.c + ch];
            }
        }
    }
    grad_input[idx] = grad;
}
