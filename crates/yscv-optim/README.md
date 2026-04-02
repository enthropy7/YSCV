# yscv-optim

Optimizers, learning rate schedulers, and gradient clipping for neural network training.

```rust
use yscv_optim::*;

let mut optimizer = Adam::new(parameters, 1e-3);
let scheduler = CosineScheduler::new(100, 1e-3, 1e-6);

for epoch in 0..100 {
    optimizer.set_lr(scheduler.get_lr(epoch));
    optimizer.step();
    optimizer.zero_grad();
}
```

## Optimizers

SGD, Adam, AdamW, RMSprop, RAdam, LARS, LAMB, AdaGrad, Lookahead (wrapper)

## LR Schedulers

Step, MultiStep, Exponential, Cosine, CosineWarmRestart, Linear, Polynomial, OneCycle, ReduceOnPlateau, Warmup

## Gradient Clipping

- `clip_grad_norm` — L2 norm clipping
- `clip_grad_value` — element-wise value clipping

## Tests

76 tests covering optimizer convergence, scheduler curves, clipping behavior.
