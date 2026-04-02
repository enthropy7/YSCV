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
- **Operations**: matmul, conv2d, add, mul, relu, sigmoid, softmax, cross_entropy, batch_norm, layer_norm, RNN, LSTM, GRU
- **Gradient accumulation**: for multi-step updates
- **Checkpointing**: trade compute for memory in deep networks

## Tests

106 tests covering gradient correctness, graph operations, edge cases.
