# yscv-autograd

Dynamic computation graph with tape-based reverse-mode automatic differentiation.

```rust
use yscv_autograd::*;

let mut graph = Graph::new();
let x = graph.variable(Tensor::from_vec(vec![2], vec![3.0, 4.0])?);
let w = graph.variable(Tensor::from_vec(vec![2], vec![1.0, 2.0])?);

let y = graph.matmul(x, w);
let loss = graph.sum(y);

let grads = graph.backward(loss);
let dw = &grads[&w]; // gradient of loss w.r.t. w
```

## Features

- **Reverse-mode autodiff**: compute gradients via backward pass
- **Dynamic graphs**: build computation graph at runtime (like PyTorch)
- **Tape-based**: records operations, replays for gradient computation
- **61 `Op` variants** in `src/node.rs`: matmul, conv1d/2d/3d, conv_transpose, depthwise_conv, add/sub/mul/div, neg/pow/abs/clamp, relu/leaky_relu/sigmoid/tanh/exp/log/sqrt/gelu/silu/mish, softmax/log_softmax, sum/mean/sum_axis/mean_axis, max_pool2d/avg_pool2d/global_avg_pool/adaptive_pool, batch_norm/layer_norm/group_norm/instance_norm, multi_head_attention/embedding, rnn/lstm/gru, dropout, upsample/pixel_shuffle, residual, einsum, grid_sample, pad/flip/repeat, reshape/transpose/unsqueeze/squeeze/flatten/expand/cat/select/narrow/slice, gather/gather_elements/scatter_add
- **Gradient accumulation**: for multi-step updates
- **Checkpointing**: trade compute for memory in deep networks

## Tests

106 tests covering gradient correctness, graph operations, edge cases.
