# yscv Optimizers And Training APIs

This page documents the optimizer surface provided by `yscv-optim` and how it integrates with `yscv-model` training helpers.

## Optimizers

### Core Optimizers
- `Sgd`:
  - learning-rate required (`Sgd::new(lr)`),
  - momentum/dampening/nesterov controls,
  - optional L2 `weight_decay`,
  - state reset via `clear_state()`.
- `Adam`:
  - learning-rate required (`Adam::new(lr)`),
  - configurable `beta1`, `beta2`, `epsilon`,
  - optional L2 `weight_decay`,
  - state reset via `clear_state()`.
- `AdamW`:
  - learning-rate required (`AdamW::new(lr)`),
  - configurable `beta1`, `beta2`, `epsilon`,
  - decoupled `weight_decay`,
  - state reset via `clear_state()`.
- `RAdam`:
  - learning-rate required (`RAdam::new(lr)`),
  - configurable `beta1`, `beta2`, `epsilon`,
  - rectified adaptive learning rate with variance term,
  - state reset via `clear_state()`.
- `RmsProp`:
  - learning-rate required (`RmsProp::new(lr)`),
  - configurable `alpha`, `epsilon`, `momentum`,
  - optional `centered` variance mode and optional L2 `weight_decay`,
  - state reset via `clear_state()`.
- `Adagrad`:
  - learning-rate required (`Adagrad::new(lr)`),
  - configurable `epsilon` and `initial_accumulator_value`,
  - optional L2 `weight_decay`,
  - state reset via `clear_state()`.

### Large-Batch Optimizers
- `Lamb` (Layer-wise Adaptive Moments):
  - learning-rate required (`Lamb::new(lr)`),
  - configurable `beta1`, `beta2`, `epsilon`,
  - trust ratio scaling per layer (ratio of weight norm to update norm),
  - optional `weight_decay`,
  - state reset via `clear_state()`.
- `Lars` (Layer-wise Adaptive Rate Scaling):
  - learning-rate required (`Lars::new(lr)`),
  - configurable `momentum`, `trust_coefficient`,
  - layer-wise adaptive rate scaling with trust coefficient,
  - optional `weight_decay`,
  - state reset via `clear_state()`.

### Meta-Optimizer
- `Lookahead`:
  - wraps any optimizer implementing `StepOptimizer` trait,
  - configurable `alpha` (interpolation weight) and `k` (slow-weight update interval),
  - maintains slow weights that are periodically interpolated with fast weights,
  - `Lookahead::new(inner, k, alpha)`.

## Learning Rate Schedulers (11)

- `StepLr`: multiplies LR by `gamma` every `step_size` steps.
- `CosineAnnealingLr`: cosine decay to `min_lr` over `t_max` steps, optional fixed base LR.
- `LinearWarmupLr`: linear warmup from `start_lr` to `base_lr` over `warmup_steps`.
- `OneCycleLr`: linear warmup to `max_lr`, then linear cooldown, with configurable phase split.
- `ExponentialLr`: multiplies LR by `gamma` every step (`lr * gamma^step`).
- `MultiStepLr`: multiplies LR by `gamma` at each milestone step.
- `ReduceOnPlateauLr`: reduces LR by `factor` when a monitored metric stops improving for `patience` steps.
- `CyclicLr`: cycles LR between `base_lr` and `max_lr` with configurable cycle length.
- `CosineWarmRestartLr`: cosine annealing with warm restarts (SGDR), configurable `T_0` and `T_mult`.
- `PolynomialLr`: polynomial decay from `base_lr` to `end_lr` over `total_steps`.
- `SequentialLr`: chains multiple schedulers, each active for a configured number of steps.

## Gradient Clipping

- `clip_grad_norm`: clips gradient tensor by L2 norm with configurable `max_norm`.
- `clip_grad_value`: clips gradient tensor element-wise to `[-clip_value, +clip_value]`.

## Validation Contracts
- `lr` must be finite and `>= 0`.
- `Sgd::momentum` must be finite and in `[0, 1)`.
- `Sgd::dampening` must be finite and in `[0, 1]`.
- `Adam/AdamW/RAdam::beta1` and `beta2` must be finite and in `[0, 1)`.
- `Adam/AdamW/RAdam::epsilon` must be finite and `> 0`.
- `RmsProp::alpha` must be finite and in `[0, 1)`.
- `RmsProp::epsilon` must be finite and `> 0`.
- `RmsProp::momentum` must be finite and in `[0, 1)`.
- `Lamb::beta1/beta2` must be finite and in `[0, 1)`, `epsilon > 0`.
- `Lars::trust_coefficient` must be finite and `> 0`.
- `weight_decay` must be finite and `>= 0`.
- Optimizer `step` requires matching weight/gradient shapes.
- All scheduler parameters must be finite with domain-specific constraints (e.g., `step_size > 0`, `gamma in (0, 1]`).

## High-Level Trainer API

`yscv-model` provides `Trainer` for end-to-end supervised training:

```rust
use yscv::prelude::*;

let config = TrainerConfig {
    optimizer: OptimizerKind::Adam { lr: 0.001 },
    loss: LossKind::CrossEntropy,
    epochs: 50,
    batch_size: 32,
    validation_split: Some(0.2), // 20% validation
};

let mut trainer = Trainer::new(config);

// Optional callbacks
trainer.add_callback(EarlyStopping::new(10)); // patience=10
trainer.add_callback(BestModelCheckpoint::new("best.bin"));

let result = trainer.fit(&model, &mut graph, &inputs, &targets)?;
println!("Final loss: {:.6}", result.final_loss);
```

### OptimizerKind
- `Sgd { lr, momentum }`, `Adam { lr }`, `AdamW { lr }`, `RmsProp { lr }`, `RAdam { lr }`, `Adagrad { lr }`

### LossKind
- `Mse`, `Mae`, `Huber`, `Hinge`, `Bce`, `Nll`, `CrossEntropy`, `Focal`, `Dice`

## Model Training Helpers (Low-Level)

`yscv-model` also exposes per-optimizer train helpers:
- `train_step_{sgd|adam|adamw|rmsprop|radam|adagrad}(...)` (default `SupervisedLoss::Mse`),
- `train_step_{sgd|adam|adamw|rmsprop|radam|adagrad}_with_loss(...)`,
- `train_epoch_{...}(...)` and `train_epoch_{...}_with_loss(...)`,
- `train_epochs_{...}_with_scheduler(...)` and `train_epochs_{...}_with_scheduler_and_loss(...)`.

## Loss Functions (17)

All 17 loss functions live in `crates/yscv-model/src/loss.rs` as `pub fn` items:

| Loss | Function | Notes |
|------|----------|-------|
| MSE | `mse_loss(...)` | Mean squared error |
| MAE | `mae_loss(...)` | Mean absolute error |
| Huber | `huber_loss(...)` | Smooth L1 (`delta > 0`) |
| Smooth L1 | `smooth_l1_loss(...)` | Smooth L1 (β-parameterised) |
| Hinge | `hinge_loss(...)` | SVM margin loss |
| BCE | `bce_loss(...)` | Binary cross-entropy |
| NLL | `nll_loss(...)` | Negative log-likelihood |
| CrossEntropy | `cross_entropy_loss(...)` | Softmax + NLL |
| Label Smoothing CE | `label_smoothing_cross_entropy(...)` | Regularised cross-entropy |
| Focal | `focal_loss(...)` | Class-imbalance focal loss |
| Dice | `dice_loss(...)` | Segmentation overlap loss |
| Triplet | `triplet_loss(...)` | Metric learning (anchor/pos/neg) |
| Contrastive | `contrastive_loss(...)` | Pair-based metric learning |
| Cosine Embedding | `cosine_embedding_loss(...)` | Cosine similarity margin |
| KL Divergence | `kl_div_loss(...)` | Distribution matching |
| CTC | `ctc_loss(...)` | Sequence-to-sequence alignment |
| Distillation | `distillation_loss(...)` | Knowledge distillation (soft + hard targets) |

## Example (Low-Level API)

```rust
use yscv_autograd::Graph;
use yscv_model::{SequentialModel, SupervisedLoss, train_step_rmsprop_with_loss};
use yscv_optim::{OneCycleLr, RmsProp};
use yscv_tensor::Tensor;

let mut graph = Graph::new();
let mut model = SequentialModel::new(&graph);
model.add_linear_zero(&mut graph, 2, 1)?;

let input = graph.constant(Tensor::from_vec(vec![1, 2], vec![0.2, 0.8])?);
let target = graph.constant(Tensor::from_vec(vec![1, 1], vec![1.0])?);
let prediction = model.forward(&mut graph, input)?;

let mut optimizer = RmsProp::new(0.01)?
    .with_alpha(0.95)?
    .with_momentum(0.9)?
    .with_centered(true);
let loss = train_step_rmsprop_with_loss(
    &mut graph,
    &mut optimizer,
    prediction,
    target,
    &model.trainable_nodes(),
    SupervisedLoss::Huber { delta: 1.0 },
)?;
assert!(loss.is_finite());

let mut scheduler = OneCycleLr::new(10, 0.05)?
    .with_pct_start(0.3)?
    .with_final_div_factor(100.0)?;
let _current_lr = scheduler.step(&mut optimizer)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```
