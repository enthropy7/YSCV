use super::validate::{
    validate_cosine_t_max, validate_lr, validate_one_cycle_final_div_factor,
    validate_one_cycle_pct_start, validate_one_cycle_total_steps, validate_step_gamma,
    validate_step_size, validate_warmup_steps,
};
use super::{LearningRate, OptimError};

/// Scheduler abstraction for stateful learning-rate policies.
pub trait LrScheduler {
    /// Advances scheduler by one epoch and returns resulting optimizer LR.
    fn step<O: LearningRate>(&mut self, optimizer: &mut O) -> Result<f32, OptimError>;

    /// Returns number of already-processed step calls.
    fn epoch(&self) -> usize;

    /// Resets scheduler internal state.
    fn reset(&mut self);
}

/// Piecewise constant learning-rate scheduler.
///
/// Every `step_size` calls to [`StepLr::step`], the optimizer learning rate is
/// multiplied by `gamma`.
#[derive(Debug, Clone, PartialEq)]
pub struct StepLr {
    step_size: usize,
    gamma: f32,
    epoch: usize,
}

impl StepLr {
    /// Creates step scheduler with required `step_size > 0` and `gamma in (0, 1]`.
    pub fn new(step_size: usize, gamma: f32) -> Result<Self, OptimError> {
        validate_step_size(step_size)?;
        validate_step_gamma(gamma)?;
        Ok(Self {
            step_size,
            gamma,
            epoch: 0,
        })
    }

    /// Returns configured step size.
    pub const fn step_size(&self) -> usize {
        self.step_size
    }

    /// Returns configured decay factor.
    pub const fn gamma(&self) -> f32 {
        self.gamma
    }

    /// Returns number of already-processed step calls.
    pub const fn epoch(&self) -> usize {
        self.epoch
    }

    /// Resets internal epoch counter.
    pub fn reset(&mut self) {
        <Self as LrScheduler>::reset(self);
    }

    /// Advances scheduler by one epoch and returns resulting optimizer LR.
    pub fn step<O: LearningRate>(&mut self, optimizer: &mut O) -> Result<f32, OptimError> {
        <Self as LrScheduler>::step(self, optimizer)
    }
}

impl LrScheduler for StepLr {
    fn step<O: LearningRate>(&mut self, optimizer: &mut O) -> Result<f32, OptimError> {
        self.epoch = self.epoch.saturating_add(1);
        if self.epoch.is_multiple_of(self.step_size) {
            let next_lr = optimizer.learning_rate() * self.gamma;
            optimizer.set_learning_rate(next_lr)?;
        }
        Ok(optimizer.learning_rate())
    }

    fn epoch(&self) -> usize {
        self.epoch
    }

    fn reset(&mut self) {
        self.epoch = 0;
    }
}

/// Cosine annealing learning-rate scheduler.
///
/// Computes:
/// `lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * t_cur / t_max))`,
/// where `t_cur` is clamped to `t_max`.
#[derive(Debug, Clone, PartialEq)]
pub struct CosineAnnealingLr {
    t_max: usize,
    min_lr: f32,
    epoch: usize,
    base_lr: Option<f32>,
}

impl CosineAnnealingLr {
    /// Creates cosine scheduler with `t_max > 0` and finite `min_lr >= 0`.
    pub fn new(t_max: usize, min_lr: f32) -> Result<Self, OptimError> {
        validate_cosine_t_max(t_max)?;
        validate_lr(min_lr)?;
        Ok(Self {
            t_max,
            min_lr,
            epoch: 0,
            base_lr: None,
        })
    }

    /// Pins explicit base LR used by cosine schedule.
    pub fn with_base_lr(mut self, base_lr: f32) -> Result<Self, OptimError> {
        validate_lr(base_lr)?;
        if self.min_lr > base_lr {
            return Err(OptimError::SchedulerMinLrExceedsBase {
                min_lr: self.min_lr,
                base_lr,
            });
        }
        self.base_lr = Some(base_lr);
        Ok(self)
    }

    pub const fn t_max(&self) -> usize {
        self.t_max
    }

    pub const fn min_lr(&self) -> f32 {
        self.min_lr
    }

    pub const fn base_lr(&self) -> Option<f32> {
        self.base_lr
    }

    pub const fn epoch(&self) -> usize {
        self.epoch
    }

    pub fn reset(&mut self) {
        <Self as LrScheduler>::reset(self);
    }

    pub fn step<O: LearningRate>(&mut self, optimizer: &mut O) -> Result<f32, OptimError> {
        <Self as LrScheduler>::step(self, optimizer)
    }
}

impl LrScheduler for CosineAnnealingLr {
    fn step<O: LearningRate>(&mut self, optimizer: &mut O) -> Result<f32, OptimError> {
        self.epoch = self.epoch.saturating_add(1);

        let base_lr = match self.base_lr {
            Some(base) => base,
            None => {
                let current = optimizer.learning_rate();
                self.base_lr = Some(current);
                current
            }
        };
        if self.min_lr > base_lr {
            return Err(OptimError::SchedulerMinLrExceedsBase {
                min_lr: self.min_lr,
                base_lr,
            });
        }

        let t_cur = self.epoch.min(self.t_max) as f32;
        let t_max = self.t_max as f32;
        let cos_term = (std::f32::consts::PI * t_cur / t_max).cos();
        let next_lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1.0 + cos_term);
        optimizer.set_learning_rate(next_lr)?;
        Ok(next_lr)
    }

    fn epoch(&self) -> usize {
        self.epoch
    }

    fn reset(&mut self) {
        self.epoch = 0;
    }
}

/// Linear warmup learning-rate scheduler.
///
/// Computes:
/// `lr = start_lr + (base_lr - start_lr) * min(epoch, warmup_steps)/warmup_steps`.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearWarmupLr {
    warmup_steps: usize,
    start_lr: Option<f32>,
    base_lr: Option<f32>,
    epoch: usize,
}

impl LinearWarmupLr {
    /// Creates warmup scheduler with `warmup_steps > 0`.
    pub fn new(warmup_steps: usize) -> Result<Self, OptimError> {
        validate_warmup_steps(warmup_steps)?;
        Ok(Self {
            warmup_steps,
            start_lr: None,
            base_lr: None,
            epoch: 0,
        })
    }

    /// Sets explicit warmup start learning rate.
    pub fn with_start_lr(mut self, start_lr: f32) -> Result<Self, OptimError> {
        validate_lr(start_lr)?;
        if let Some(base_lr) = self.base_lr
            && start_lr > base_lr
        {
            return Err(OptimError::SchedulerStartLrExceedsBase { start_lr, base_lr });
        }
        self.start_lr = Some(start_lr);
        Ok(self)
    }

    /// Sets explicit warmup end/base learning rate.
    pub fn with_base_lr(mut self, base_lr: f32) -> Result<Self, OptimError> {
        validate_lr(base_lr)?;
        if let Some(start_lr) = self.start_lr
            && start_lr > base_lr
        {
            return Err(OptimError::SchedulerStartLrExceedsBase { start_lr, base_lr });
        }
        self.base_lr = Some(base_lr);
        Ok(self)
    }

    pub const fn warmup_steps(&self) -> usize {
        self.warmup_steps
    }

    pub const fn start_lr(&self) -> Option<f32> {
        self.start_lr
    }

    pub const fn base_lr(&self) -> Option<f32> {
        self.base_lr
    }

    pub const fn epoch(&self) -> usize {
        self.epoch
    }

    pub fn reset(&mut self) {
        <Self as LrScheduler>::reset(self);
    }

    pub fn step<O: LearningRate>(&mut self, optimizer: &mut O) -> Result<f32, OptimError> {
        <Self as LrScheduler>::step(self, optimizer)
    }
}

impl LrScheduler for LinearWarmupLr {
    fn step<O: LearningRate>(&mut self, optimizer: &mut O) -> Result<f32, OptimError> {
        self.epoch = self.epoch.saturating_add(1);

        let base_lr = match self.base_lr {
            Some(base_lr) => base_lr,
            None => {
                let current = optimizer.learning_rate();
                self.base_lr = Some(current);
                current
            }
        };
        let start_lr = self.start_lr.unwrap_or(0.0);
        if start_lr > base_lr {
            return Err(OptimError::SchedulerStartLrExceedsBase { start_lr, base_lr });
        }

        let warmup_ratio = self.epoch.min(self.warmup_steps) as f32 / self.warmup_steps as f32;
        let next_lr = start_lr + (base_lr - start_lr) * warmup_ratio;
        optimizer.set_learning_rate(next_lr)?;
        Ok(next_lr)
    }

    fn epoch(&self) -> usize {
        self.epoch
    }

    fn reset(&mut self) {
        self.epoch = 0;
    }
}

/// One-cycle learning-rate scheduler with linear warmup and linear cooldown.
#[derive(Debug, Clone, PartialEq)]
pub struct OneCycleLr {
    total_steps: usize,
    max_lr: f32,
    pct_start: f32,
    final_div_factor: f32,
    initial_lr: Option<f32>,
    epoch: usize,
}

impl OneCycleLr {
    /// Creates one-cycle scheduler.
    ///
    /// - `total_steps > 0`
    /// - `max_lr >= 0`
    pub fn new(total_steps: usize, max_lr: f32) -> Result<Self, OptimError> {
        validate_one_cycle_total_steps(total_steps)?;
        validate_lr(max_lr)?;
        Ok(Self {
            total_steps,
            max_lr,
            pct_start: 0.3,
            final_div_factor: 1_000.0,
            initial_lr: None,
            epoch: 0,
        })
    }

    /// Sets fraction of cycle spent in the up phase.
    pub fn with_pct_start(mut self, pct_start: f32) -> Result<Self, OptimError> {
        validate_one_cycle_pct_start(pct_start)?;
        self.pct_start = pct_start;
        Ok(self)
    }

    /// Sets divisor used for final LR (`final_lr = initial_lr / final_div_factor`).
    pub fn with_final_div_factor(mut self, final_div_factor: f32) -> Result<Self, OptimError> {
        validate_one_cycle_final_div_factor(final_div_factor)?;
        self.final_div_factor = final_div_factor;
        Ok(self)
    }

    /// Pins explicit initial LR used by the schedule.
    pub fn with_initial_lr(mut self, initial_lr: f32) -> Result<Self, OptimError> {
        validate_lr(initial_lr)?;
        if self.max_lr < initial_lr {
            return Err(OptimError::SchedulerMaxLrBelowInitial {
                max_lr: self.max_lr,
                initial_lr,
            });
        }
        self.initial_lr = Some(initial_lr);
        Ok(self)
    }

    pub const fn total_steps(&self) -> usize {
        self.total_steps
    }

    pub const fn max_lr(&self) -> f32 {
        self.max_lr
    }

    pub const fn pct_start(&self) -> f32 {
        self.pct_start
    }

    pub const fn final_div_factor(&self) -> f32 {
        self.final_div_factor
    }

    pub const fn initial_lr(&self) -> Option<f32> {
        self.initial_lr
    }

    pub const fn epoch(&self) -> usize {
        self.epoch
    }

    pub fn reset(&mut self) {
        <Self as LrScheduler>::reset(self);
    }

    pub fn step<O: LearningRate>(&mut self, optimizer: &mut O) -> Result<f32, OptimError> {
        <Self as LrScheduler>::step(self, optimizer)
    }
}

impl LrScheduler for OneCycleLr {
    fn step<O: LearningRate>(&mut self, optimizer: &mut O) -> Result<f32, OptimError> {
        self.epoch = self.epoch.saturating_add(1);

        let initial_lr = match self.initial_lr {
            Some(initial_lr) => initial_lr,
            None => {
                let current = optimizer.learning_rate();
                self.initial_lr = Some(current);
                current
            }
        };
        if self.max_lr < initial_lr {
            return Err(OptimError::SchedulerMaxLrBelowInitial {
                max_lr: self.max_lr,
                initial_lr,
            });
        }

        let final_lr = initial_lr / self.final_div_factor;
        let up_steps = one_cycle_up_steps(self.total_steps, self.pct_start);
        let clamped_epoch = self.epoch.min(self.total_steps);
        let next_lr = if clamped_epoch <= up_steps {
            let progress = clamped_epoch as f32 / up_steps as f32;
            initial_lr + (self.max_lr - initial_lr) * progress
        } else {
            let down_steps = self.total_steps.saturating_sub(up_steps).max(1);
            let down_epoch = clamped_epoch - up_steps;
            let progress = down_epoch as f32 / down_steps as f32;
            self.max_lr - (self.max_lr - final_lr) * progress
        };
        optimizer.set_learning_rate(next_lr)?;
        Ok(next_lr)
    }

    fn epoch(&self) -> usize {
        self.epoch
    }

    fn reset(&mut self) {
        self.epoch = 0;
    }
}

fn one_cycle_up_steps(total_steps: usize, pct_start: f32) -> usize {
    ((total_steps as f32 * pct_start).ceil() as usize).clamp(1, total_steps)
}

/// Exponential learning-rate scheduler.
///
/// Every step, the optimizer learning rate is multiplied by `gamma`:
/// `lr = lr * gamma`.
#[derive(Debug, Clone, PartialEq)]
pub struct ExponentialLr {
    gamma: f32,
    epoch: usize,
}

impl ExponentialLr {
    /// Creates exponential scheduler with `gamma in (0, 1]`.
    pub fn new(gamma: f32) -> Result<Self, OptimError> {
        validate_step_gamma(gamma)?;
        Ok(Self { gamma, epoch: 0 })
    }

    /// Returns configured decay factor.
    pub const fn gamma(&self) -> f32 {
        self.gamma
    }

    /// Returns number of already-processed step calls.
    pub const fn epoch(&self) -> usize {
        self.epoch
    }

    /// Resets internal epoch counter.
    pub fn reset(&mut self) {
        <Self as LrScheduler>::reset(self);
    }

    /// Advances scheduler by one epoch and returns resulting optimizer LR.
    pub fn step<O: LearningRate>(&mut self, optimizer: &mut O) -> Result<f32, OptimError> {
        <Self as LrScheduler>::step(self, optimizer)
    }
}

impl LrScheduler for ExponentialLr {
    fn step<O: LearningRate>(&mut self, optimizer: &mut O) -> Result<f32, OptimError> {
        self.epoch = self.epoch.saturating_add(1);
        let next_lr = optimizer.learning_rate() * self.gamma;
        optimizer.set_learning_rate(next_lr)?;
        Ok(next_lr)
    }

    fn epoch(&self) -> usize {
        self.epoch
    }

    fn reset(&mut self) {
        self.epoch = 0;
    }
}

/// Polynomial decay learning-rate scheduler.
///
/// Decays the learning rate from its initial value to `end_lr` over `total_steps`
/// using a polynomial of the given `power`:
/// `lr = (base_lr - end_lr) * (1 - epoch/total_steps)^power + end_lr`.
#[derive(Debug, Clone, PartialEq)]
pub struct PolynomialDecayLr {
    total_steps: usize,
    power: f32,
    end_lr: f32,
    base_lr: Option<f32>,
    epoch: usize,
}

impl PolynomialDecayLr {
    /// Creates polynomial decay scheduler.
    ///
    /// - `total_steps > 0`
    /// - `power > 0` and finite
    /// - `end_lr >= 0` and finite
    pub fn new(total_steps: usize, power: f32, end_lr: f32) -> Result<Self, OptimError> {
        if total_steps == 0 {
            return Err(OptimError::InvalidStepSize {
                step_size: total_steps,
            });
        }
        if !power.is_finite() || power <= 0.0 {
            return Err(OptimError::InvalidStepGamma { gamma: power });
        }
        validate_lr(end_lr)?;
        Ok(Self {
            total_steps,
            power,
            end_lr,
            base_lr: None,
            epoch: 0,
        })
    }

    /// Pins explicit base LR used by the schedule.
    pub fn with_base_lr(mut self, base_lr: f32) -> Result<Self, OptimError> {
        validate_lr(base_lr)?;
        self.base_lr = Some(base_lr);
        Ok(self)
    }

    pub const fn total_steps(&self) -> usize {
        self.total_steps
    }

    pub const fn power(&self) -> f32 {
        self.power
    }

    pub const fn end_lr(&self) -> f32 {
        self.end_lr
    }

    pub const fn base_lr(&self) -> Option<f32> {
        self.base_lr
    }

    pub const fn epoch(&self) -> usize {
        self.epoch
    }

    pub fn reset(&mut self) {
        <Self as LrScheduler>::reset(self);
    }

    pub fn step<O: LearningRate>(&mut self, optimizer: &mut O) -> Result<f32, OptimError> {
        <Self as LrScheduler>::step(self, optimizer)
    }
}

impl LrScheduler for PolynomialDecayLr {
    fn step<O: LearningRate>(&mut self, optimizer: &mut O) -> Result<f32, OptimError> {
        self.epoch = self.epoch.saturating_add(1);

        let base_lr = match self.base_lr {
            Some(base) => base,
            None => {
                let current = optimizer.learning_rate();
                self.base_lr = Some(current);
                current
            }
        };

        let t = (self.epoch.min(self.total_steps) as f32) / (self.total_steps as f32);
        let next_lr = (base_lr - self.end_lr) * (1.0 - t).powf(self.power) + self.end_lr;
        optimizer.set_learning_rate(next_lr)?;
        Ok(next_lr)
    }

    fn epoch(&self) -> usize {
        self.epoch
    }

    fn reset(&mut self) {
        self.epoch = 0;
    }
}

/// Reduce learning rate when a metric has stopped improving.
///
/// When the metric has not improved for `patience` consecutive calls to
/// [`ReduceLrOnPlateau::step_with_metric`], the learning rate is multiplied
/// by `factor` (clamped to `min_lr`).
#[derive(Debug, Clone, PartialEq)]
pub struct ReduceLrOnPlateau {
    factor: f32,
    patience: usize,
    min_lr: f32,
    best_metric: f32,
    wait: usize,
    epoch: usize,
}

impl ReduceLrOnPlateau {
    /// Creates a plateau scheduler.
    ///
    /// - `factor in (0, 1]`
    /// - `patience >= 1`
    /// - `min_lr >= 0` and finite
    pub fn new(factor: f32, patience: usize, min_lr: f32) -> Result<Self, OptimError> {
        validate_step_gamma(factor)?;
        if patience == 0 {
            return Err(OptimError::InvalidStepSize {
                step_size: patience,
            });
        }
        validate_lr(min_lr)?;
        Ok(Self {
            factor,
            patience,
            min_lr,
            best_metric: f32::INFINITY,
            wait: 0,
            epoch: 0,
        })
    }

    pub const fn factor(&self) -> f32 {
        self.factor
    }

    pub const fn patience(&self) -> usize {
        self.patience
    }

    pub const fn min_lr(&self) -> f32 {
        self.min_lr
    }

    pub const fn best_metric(&self) -> f32 {
        self.best_metric
    }

    pub const fn wait(&self) -> usize {
        self.wait
    }

    pub const fn epoch(&self) -> usize {
        self.epoch
    }

    pub fn reset(&mut self) {
        <Self as LrScheduler>::reset(self);
    }

    /// Steps the scheduler with a metric value. If the metric has not improved
    /// for `patience` consecutive steps, the LR is reduced by `factor`.
    /// Lower metric is considered better.
    pub fn step_with_metric<O: LearningRate>(
        &mut self,
        metric: f32,
        optimizer: &mut O,
    ) -> Result<f32, OptimError> {
        self.epoch = self.epoch.saturating_add(1);

        if metric < self.best_metric {
            self.best_metric = metric;
            self.wait = 0;
        } else {
            self.wait += 1;
            if self.wait >= self.patience {
                let next_lr = (optimizer.learning_rate() * self.factor).max(self.min_lr);
                optimizer.set_learning_rate(next_lr)?;
                self.wait = 0;
            }
        }

        Ok(optimizer.learning_rate())
    }
}

impl LrScheduler for ReduceLrOnPlateau {
    /// Standard step without a metric does nothing to the LR
    /// (use `step_with_metric` instead).
    fn step<O: LearningRate>(&mut self, optimizer: &mut O) -> Result<f32, OptimError> {
        self.epoch = self.epoch.saturating_add(1);
        Ok(optimizer.learning_rate())
    }

    fn epoch(&self) -> usize {
        self.epoch
    }

    fn reset(&mut self) {
        self.epoch = 0;
        self.best_metric = f32::INFINITY;
        self.wait = 0;
    }
}

/// Cyclic learning-rate scheduler with triangular policy.
///
/// Cycles the learning rate between `base_lr` and `max_lr` with a triangular
/// waveform defined by `step_size_up` (ascending half) and `step_size_down`
/// (descending half).
#[derive(Debug, Clone, PartialEq)]
pub struct CyclicLr {
    base_lr: f32,
    max_lr: f32,
    step_size_up: usize,
    step_size_down: usize,
    epoch: usize,
}

impl CyclicLr {
    /// Creates cyclic LR scheduler.
    ///
    /// - `base_lr >= 0`, `max_lr >= base_lr`
    /// - `step_size_up > 0`, `step_size_down > 0`
    pub fn new(
        base_lr: f32,
        max_lr: f32,
        step_size_up: usize,
        step_size_down: usize,
    ) -> Result<Self, OptimError> {
        validate_lr(base_lr)?;
        validate_lr(max_lr)?;
        if max_lr < base_lr {
            return Err(OptimError::SchedulerMaxLrBelowInitial {
                max_lr,
                initial_lr: base_lr,
            });
        }
        if step_size_up == 0 {
            return Err(OptimError::InvalidStepSize {
                step_size: step_size_up,
            });
        }
        if step_size_down == 0 {
            return Err(OptimError::InvalidStepSize {
                step_size: step_size_down,
            });
        }
        Ok(Self {
            base_lr,
            max_lr,
            step_size_up,
            step_size_down,
            epoch: 0,
        })
    }

    pub const fn base_lr(&self) -> f32 {
        self.base_lr
    }

    pub const fn max_lr(&self) -> f32 {
        self.max_lr
    }

    pub const fn step_size_up(&self) -> usize {
        self.step_size_up
    }

    pub const fn step_size_down(&self) -> usize {
        self.step_size_down
    }

    pub const fn epoch(&self) -> usize {
        self.epoch
    }

    pub fn reset(&mut self) {
        <Self as LrScheduler>::reset(self);
    }

    pub fn step<O: LearningRate>(&mut self, optimizer: &mut O) -> Result<f32, OptimError> {
        <Self as LrScheduler>::step(self, optimizer)
    }
}

impl LrScheduler for CyclicLr {
    fn step<O: LearningRate>(&mut self, optimizer: &mut O) -> Result<f32, OptimError> {
        self.epoch = self.epoch.saturating_add(1);

        let cycle_len = self.step_size_up + self.step_size_down;
        let pos = (self.epoch - 1) % cycle_len; // 0-indexed position in current cycle

        let next_lr = if pos < self.step_size_up {
            // ascending
            let progress = pos as f32 / self.step_size_up as f32;
            self.base_lr + (self.max_lr - self.base_lr) * progress
        } else {
            // descending
            let down_pos = pos - self.step_size_up;
            let progress = down_pos as f32 / self.step_size_down as f32;
            self.max_lr - (self.max_lr - self.base_lr) * progress
        };

        optimizer.set_learning_rate(next_lr)?;
        Ok(next_lr)
    }

    fn epoch(&self) -> usize {
        self.epoch
    }

    fn reset(&mut self) {
        self.epoch = 0;
    }
}

/// Lambda learning-rate scheduler.
///
/// Computes `lr = base_lr * lr_lambda(step_count)` at each step, where
/// `lr_lambda` is a user-provided closure mapping epoch to a multiplicative
/// factor.
pub struct LambdaLr {
    base_lr: f32,
    current_lr: f32,
    lr_lambda: Box<dyn Fn(usize) -> f32>,
    step_count: usize,
}

impl LambdaLr {
    /// Creates a lambda scheduler with the given base learning rate and lambda
    /// function. The lambda receives the current epoch (after increment) and
    /// returns a multiplicative factor applied to `base_lr`.
    pub fn new(base_lr: f32, lr_lambda: Box<dyn Fn(usize) -> f32>) -> Self {
        Self {
            base_lr,
            current_lr: base_lr,
            lr_lambda,
            step_count: 0,
        }
    }

    /// Returns the base learning rate.
    pub const fn base_lr(&self) -> f32 {
        self.base_lr
    }

    /// Returns the current learning rate.
    pub const fn current_lr(&self) -> f32 {
        self.current_lr
    }

    /// Returns the current step count.
    pub const fn step_count(&self) -> usize {
        self.step_count
    }

    /// Returns number of already-processed step calls.
    pub const fn epoch(&self) -> usize {
        self.step_count
    }

    /// Resets internal state.
    pub fn reset(&mut self) {
        <Self as LrScheduler>::reset(self);
    }

    /// Advances scheduler by one epoch and returns resulting optimizer LR.
    pub fn step<O: LearningRate>(&mut self, optimizer: &mut O) -> Result<f32, OptimError> {
        <Self as LrScheduler>::step(self, optimizer)
    }
}

impl LrScheduler for LambdaLr {
    fn step<O: LearningRate>(&mut self, optimizer: &mut O) -> Result<f32, OptimError> {
        self.step_count = self.step_count.saturating_add(1);
        self.current_lr = self.base_lr * (self.lr_lambda)(self.step_count);
        optimizer.set_learning_rate(self.current_lr)?;
        Ok(self.current_lr)
    }

    fn epoch(&self) -> usize {
        self.step_count
    }

    fn reset(&mut self) {
        self.step_count = 0;
        self.current_lr = self.base_lr;
    }
}

/// Multi-step learning-rate scheduler.
///
/// Drops the learning rate by `gamma` at each milestone epoch in `milestones`.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiStepLr {
    milestones: Vec<usize>,
    gamma: f32,
    epoch: usize,
    base_lr: Option<f32>,
}

impl MultiStepLr {
    /// Creates multi-step scheduler with sorted `milestones` and `gamma in (0, 1]`.
    pub fn new(mut milestones: Vec<usize>, gamma: f32) -> Result<Self, OptimError> {
        validate_step_gamma(gamma)?;
        milestones.sort();
        milestones.dedup();
        Ok(Self {
            milestones,
            gamma,
            epoch: 0,
            base_lr: None,
        })
    }

    pub fn milestones(&self) -> &[usize] {
        &self.milestones
    }

    pub const fn gamma(&self) -> f32 {
        self.gamma
    }

    pub const fn epoch(&self) -> usize {
        self.epoch
    }

    pub const fn reset(&mut self) {
        self.epoch = 0;
        self.base_lr = None;
    }

    pub fn step<O: LearningRate>(&mut self, optimizer: &mut O) -> Result<f32, OptimError> {
        <Self as LrScheduler>::step(self, optimizer)
    }
}

impl LrScheduler for MultiStepLr {
    fn step<O: LearningRate>(&mut self, optimizer: &mut O) -> Result<f32, OptimError> {
        self.epoch = self.epoch.saturating_add(1);

        let base_lr = match self.base_lr {
            Some(base) => base,
            None => {
                let current = optimizer.learning_rate();
                self.base_lr = Some(current);
                current
            }
        };

        let num_decays = self.milestones.iter().filter(|&&m| self.epoch >= m).count();
        let next_lr = base_lr * self.gamma.powi(num_decays as i32);
        optimizer.set_learning_rate(next_lr)?;
        Ok(next_lr)
    }

    fn epoch(&self) -> usize {
        self.epoch
    }

    fn reset(&mut self) {
        self.epoch = 0;
        self.base_lr = None;
    }
}

/// Cosine annealing with warm restarts learning-rate scheduler.
///
/// Within each period, LR follows cosine decay from `base_lr` to `eta_min`.
/// After `t_0` epochs, the schedule restarts and the next period length is
/// `t_0 * t_mult`.
///
/// Formula within each period:
/// `lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(pi * t_cur / t_i))`
#[derive(Debug, Clone, PartialEq)]
pub struct CosineAnnealingWarmRestarts {
    t_0: usize,
    t_mult: usize,
    eta_min: f32,
    base_lr: Option<f32>,
    epoch: usize,
}

impl CosineAnnealingWarmRestarts {
    /// Creates a cosine warm restarts scheduler.
    ///
    /// - `t_0 > 0`: initial period length
    /// - `t_mult >= 1`: period multiplier after each restart
    /// - `eta_min >= 0`: minimum learning rate
    pub fn new(t_0: usize, t_mult: usize, eta_min: f32) -> Result<Self, OptimError> {
        validate_cosine_t_max(t_0)?;
        if t_mult == 0 {
            return Err(OptimError::InvalidStepSize { step_size: 0 });
        }
        validate_lr(eta_min)?;
        Ok(Self {
            t_0,
            t_mult,
            eta_min,
            base_lr: None,
            epoch: 0,
        })
    }

    /// Pins explicit base LR used by the schedule.
    pub fn with_base_lr(mut self, base_lr: f32) -> Result<Self, OptimError> {
        validate_lr(base_lr)?;
        if self.eta_min > base_lr {
            return Err(OptimError::SchedulerMinLrExceedsBase {
                min_lr: self.eta_min,
                base_lr,
            });
        }
        self.base_lr = Some(base_lr);
        Ok(self)
    }

    pub const fn t_0(&self) -> usize {
        self.t_0
    }

    pub const fn t_mult(&self) -> usize {
        self.t_mult
    }

    pub const fn eta_min(&self) -> f32 {
        self.eta_min
    }

    pub const fn base_lr(&self) -> Option<f32> {
        self.base_lr
    }

    pub const fn epoch(&self) -> usize {
        self.epoch
    }

    pub fn reset(&mut self) {
        <Self as LrScheduler>::reset(self);
    }

    pub fn step<O: LearningRate>(&mut self, optimizer: &mut O) -> Result<f32, OptimError> {
        <Self as LrScheduler>::step(self, optimizer)
    }
}

impl LrScheduler for CosineAnnealingWarmRestarts {
    fn step<O: LearningRate>(&mut self, optimizer: &mut O) -> Result<f32, OptimError> {
        self.epoch = self.epoch.saturating_add(1);

        let base_lr = match self.base_lr {
            Some(base) => base,
            None => {
                let current = optimizer.learning_rate();
                self.base_lr = Some(current);
                current
            }
        };
        if self.eta_min > base_lr {
            return Err(OptimError::SchedulerMinLrExceedsBase {
                min_lr: self.eta_min,
                base_lr,
            });
        }

        // Determine current period and position within it
        let (t_cur, t_i) = cosine_warm_restarts_position(self.epoch, self.t_0, self.t_mult);

        let cos_term = (std::f32::consts::PI * t_cur as f32 / t_i as f32).cos();
        let next_lr = self.eta_min + 0.5 * (base_lr - self.eta_min) * (1.0 + cos_term);
        optimizer.set_learning_rate(next_lr)?;
        Ok(next_lr)
    }

    fn epoch(&self) -> usize {
        self.epoch
    }

    fn reset(&mut self) {
        self.epoch = 0;
    }
}

/// Returns `(t_cur, t_i)` where `t_cur` is the position within the current
/// period and `t_i` is the current period length.
const fn cosine_warm_restarts_position(epoch: usize, t_0: usize, t_mult: usize) -> (usize, usize) {
    if t_mult == 1 {
        // All periods have the same length t_0
        let t_cur = ((epoch - 1) % t_0) + 1;
        (t_cur, t_0)
    } else {
        // Periods grow: t_0, t_0*t_mult, t_0*t_mult^2, ...
        let mut t_i = t_0;
        let mut cumulative = 0usize;
        loop {
            if epoch <= cumulative + t_i {
                let t_cur = epoch - cumulative;
                return (t_cur, t_i);
            }
            cumulative += t_i;
            t_i *= t_mult;
        }
    }
}
