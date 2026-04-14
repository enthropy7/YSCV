# yscv-optim

Optimizers, learning rate schedulers, and gradient clipping for neural network training.

```rust,ignore
use yscv_optim::*;

let mut optimizer = Adam::new(parameters, 1e-3);
let scheduler = CosineScheduler::new(100, 1e-3, 1e-6);

for epoch in 0..100 {
    optimizer.set_lr(scheduler.get_lr(epoch));
    optimizer.step();
    optimizer.zero_grad();
}
```

## Optimizers (8 + Lookahead meta-optimizer)

`Sgd`, `Adam`, `AdamW`, `RmsProp`, `RAdam`, `Lars`, `Lamb`, `Adagrad`, plus `Lookahead<O>` which wraps any of them.

## LR Schedulers (11)

`StepLr`, `MultiStepLr`, `ExponentialLr`, `CosineAnnealingLr`, `CosineAnnealingWarmRestarts`, `LinearWarmupLr`, `PolynomialDecayLr`, `OneCycleLr`, `ReduceLrOnPlateau`, `CyclicLr`, `LambdaLr`.

## Gradient Clipping

- `clip_grad_norm` — L2 norm clipping
- `clip_grad_value` — element-wise value clipping

## Tests

76 tests covering optimizer convergence, scheduler curves, clipping behavior.
