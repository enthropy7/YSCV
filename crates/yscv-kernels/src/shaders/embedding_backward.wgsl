// Embedding lookup backward: scatter upstream gradients to weight rows.
// Each thread handles one (index, embedding_dim) pair.
struct Params { num_indices: u32, embed_dim: u32 }
@group(0) @binding(0) var<storage, read> upstream: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_weight: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = p.num_indices * p.embed_dim;
    if (gid.x >= total) { return; }
    let idx = gid.x / p.embed_dim;
    let dim = gid.x % p.embed_dim;
    let row = u32(indices[idx]);
    // Atomic add would be ideal but WGSL doesn't support atomicAdd on f32.
    // For non-GPU path, CPU scatter is used.
    grad_weight[row * p.embed_dim + dim] += upstream[idx * p.embed_dim + dim];
}
