use std::collections::HashMap;

use yscv_autograd::Graph;
use yscv_optim::{Adam, AdamW, Sgd};
use yscv_tensor::Tensor;

use crate::{
    EpochTrainOptions, ModelError, SequentialModel, SupervisedDataset, SupervisedLoss,
    TrainingCallback, TrainingLog, train_epoch_adam_with_options_and_loss,
    train_epoch_adamw_with_options_and_loss, train_epoch_sgd_with_options_and_loss,
};

/// Compute a scalar loss value from prediction and target tensors (no autograd).
fn compute_raw_loss(predictions: &Tensor, targets: &Tensor, loss_kind: LossKind) -> f32 {
    match loss_kind {
        LossKind::Mse => {
            let diff = predictions
                .sub(targets)
                .expect("shape mismatch in val loss");
            let sq = diff.mul(&diff).expect("shape mismatch in val loss");
            sq.mean()
        }
        LossKind::CrossEntropy => {
            // -mean(target * ln(clamp(pred)))
            let clamped = predictions.clamp(1e-7, 1.0);
            let log_pred = clamped.ln();
            let product = targets.mul(&log_pred).expect("shape mismatch in val loss");
            -product.mean()
        }
        LossKind::Bce => {
            // -mean(target*ln(pred) + (1-target)*ln(1-pred))
            let eps = 1e-7_f32;
            let clamped = predictions.clamp(eps, 1.0 - eps);
            let log_p = clamped.ln();
            let one_minus_p = clamped.neg().add(&Tensor::scalar(1.0)).expect("bce add");
            let log_1mp = one_minus_p.clamp(eps, 1.0).ln();
            let one_minus_t = targets.neg().add(&Tensor::scalar(1.0)).expect("bce add");
            let term1 = targets.mul(&log_p).expect("bce mul");
            let term2 = one_minus_t.mul(&log_1mp).expect("bce mul");
            let sum = term1.add(&term2).expect("bce add");
            -sum.mean()
        }
    }
}

/// Which optimizer to use.
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizerKind {
    Sgd { lr: f32, momentum: f32 },
    Adam { lr: f32 },
    AdamW { lr: f32, weight_decay: f32 },
}

/// Which loss function to use.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LossKind {
    Mse,
    CrossEntropy,
    Bce,
}

impl LossKind {
    fn to_supervised_loss(self) -> SupervisedLoss {
        match self {
            LossKind::Mse => SupervisedLoss::Mse,
            LossKind::CrossEntropy => SupervisedLoss::CrossEntropy,
            LossKind::Bce => SupervisedLoss::Bce,
        }
    }
}

/// High-level training configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainerConfig {
    pub optimizer: OptimizerKind,
    pub loss: LossKind,
    pub epochs: usize,
    pub batch_size: usize,
    /// Optional fraction of data to hold out for validation (e.g. 0.2 = 20%).
    /// When `None`, no validation is performed.
    pub validation_split: Option<f32>,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            optimizer: OptimizerKind::Sgd {
                lr: 0.01,
                momentum: 0.0,
            },
            loss: LossKind::Mse,
            epochs: 10,
            batch_size: 32,
            validation_split: None,
        }
    }
}

/// Training result returned after fitting.
#[derive(Debug, Clone)]
pub struct TrainResult {
    pub epochs_trained: usize,
    pub final_loss: f32,
    pub history: Vec<HashMap<String, f32>>,
    /// Structured training log with CSV export and per-metric history queries.
    pub log: TrainingLog,
}

/// High-level trainer that wraps optimizer + loss + callbacks configuration.
pub struct Trainer {
    config: TrainerConfig,
    callbacks: Vec<Box<dyn TrainingCallback>>,
}

impl Trainer {
    /// Create a new trainer with the given configuration.
    pub fn new(config: TrainerConfig) -> Self {
        Self {
            config,
            callbacks: Vec::new(),
        }
    }

    /// Add a callback (EarlyStopping, BestModelCheckpoint, etc.).
    pub fn add_callback(&mut self, cb: Box<dyn TrainingCallback>) -> &mut Self {
        self.callbacks.push(cb);
        self
    }

    /// Train the model on the given data.
    ///
    /// `inputs` and `targets` are combined into a `SupervisedDataset`. For each
    /// epoch the method runs a full pass over mini-batches, computes the loss,
    /// back-propagates, and steps the optimizer. Callbacks are invoked after
    /// every epoch and may request early stopping.
    pub fn fit(
        &mut self,
        model: &mut SequentialModel,
        graph: &mut Graph,
        inputs: &Tensor,
        targets: &Tensor,
    ) -> Result<TrainResult, ModelError> {
        // Auto-register CNN/attention/recurrent layer parameters as graph
        // variables so that layers created without `new_in_graph` work in
        // training mode.
        model.register_cnn_params(graph);

        if self.config.epochs == 0 {
            return Err(ModelError::InvalidEpochCount { epochs: 0 });
        }
        if self.config.batch_size == 0 {
            return Err(ModelError::InvalidBatchSize { batch_size: 0 });
        }

        // Split data into train / validation if requested.
        let n_samples = inputs.shape()[0];
        let (train_inputs, train_targets, val_data) = match self.config.validation_split {
            Some(frac) if frac > 0.0 && frac < 1.0 => {
                let val_count = ((n_samples as f32) * frac).round() as usize;
                let val_count = val_count.max(1).min(n_samples - 1);
                let train_count = n_samples - val_count;
                let ti = inputs.narrow(0, 0, train_count)?;
                let tt = targets.narrow(0, 0, train_count)?;
                let vi = inputs.narrow(0, train_count, val_count)?;
                let vt = targets.narrow(0, train_count, val_count)?;
                (ti, tt, Some((vi, vt)))
            }
            _ => (inputs.clone(), targets.clone(), None),
        };

        let dataset = SupervisedDataset::new(train_inputs, train_targets)?;
        let supervised_loss = self.config.loss.to_supervised_loss();
        let loss_kind = self.config.loss;
        let epoch_options = EpochTrainOptions {
            batch_size: self.config.batch_size,
            ..EpochTrainOptions::default()
        };

        let mut history: Vec<HashMap<String, f32>> = Vec::with_capacity(self.config.epochs);
        let mut log = TrainingLog::new();
        let mut epochs_trained = 0usize;
        let mut final_loss = f32::NAN;

        macro_rules! epoch_body {
            ($epoch:expr, $metrics:expr) => {{
                final_loss = $metrics.mean_loss;
                epochs_trained = $epoch + 1;
                let mut epoch_metrics = HashMap::default();
                epoch_metrics.insert("loss".to_string(), $metrics.mean_loss);
                if let Some((ref val_inputs, ref val_targets)) = val_data {
                    // Use graph-based forward pass (works for all layer types).
                    let vi_node = graph.variable(val_inputs.clone());
                    let vo_node = model.forward(graph, vi_node)?;
                    let val_preds = graph.value(vo_node)?.clone();
                    let val_loss = compute_raw_loss(&val_preds, val_targets, loss_kind);
                    epoch_metrics.insert("val_loss".to_string(), val_loss);
                }
                let should_stop = self.callbacks.iter_mut().fold(false, |stop, cb| {
                    cb.on_epoch_end($epoch, &epoch_metrics) || stop
                });
                log.log_epoch(epoch_metrics.clone());
                history.push(epoch_metrics);
                should_stop
            }};
        }

        match &self.config.optimizer {
            OptimizerKind::Sgd { lr, momentum } => {
                let mut opt = Sgd::new(*lr)?;
                if *momentum != 0.0 {
                    opt = opt.with_momentum(*momentum)?;
                }
                for epoch in 0..self.config.epochs {
                    let metrics = train_epoch_sgd_with_options_and_loss(
                        graph,
                        model,
                        &mut opt,
                        &dataset,
                        epoch_options.clone(),
                        supervised_loss,
                    )?;
                    if epoch_body!(epoch, metrics) {
                        break;
                    }
                }
            }
            OptimizerKind::Adam { lr } => {
                let mut opt = Adam::new(*lr)?;
                for epoch in 0..self.config.epochs {
                    let metrics = train_epoch_adam_with_options_and_loss(
                        graph,
                        model,
                        &mut opt,
                        &dataset,
                        epoch_options.clone(),
                        supervised_loss,
                    )?;
                    if epoch_body!(epoch, metrics) {
                        break;
                    }
                }
            }
            OptimizerKind::AdamW { lr, weight_decay } => {
                let mut opt = AdamW::new(*lr)?.with_weight_decay(*weight_decay)?;
                for epoch in 0..self.config.epochs {
                    let metrics = train_epoch_adamw_with_options_and_loss(
                        graph,
                        model,
                        &mut opt,
                        &dataset,
                        epoch_options.clone(),
                        supervised_loss,
                    )?;
                    if epoch_body!(epoch, metrics) {
                        break;
                    }
                }
            }
        }

        Ok(TrainResult {
            epochs_trained,
            final_loss,
            history,
            log,
        })
    }
}
