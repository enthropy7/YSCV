use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Mode for metric monitoring.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MonitorMode {
    /// Lower is better (e.g., loss).
    Min,
    /// Higher is better (e.g., accuracy).
    Max,
}

/// Early stopping to halt training when a metric stops improving.
#[derive(Debug, Clone)]
pub struct EarlyStopping {
    patience: usize,
    min_delta: f32,
    mode: MonitorMode,
    best_value: f32,
    counter: usize,
    stopped: bool,
    monitor: String,
}

impl EarlyStopping {
    /// Create a new early stopping monitor.
    ///
    /// - `patience`: number of epochs without improvement before stopping.
    /// - `min_delta`: minimum change to qualify as an improvement.
    /// - `mode`: whether lower or higher metric values are better.
    pub fn new(patience: usize, min_delta: f32, mode: MonitorMode) -> Self {
        let best_value = match mode {
            MonitorMode::Min => f32::INFINITY,
            MonitorMode::Max => f32::NEG_INFINITY,
        };
        Self {
            patience,
            min_delta,
            mode,
            best_value,
            counter: 0,
            stopped: false,
            monitor: "loss".to_string(),
        }
    }

    /// Check if training should stop. Call once per epoch with the monitored metric.
    ///
    /// Returns `true` when the patience has been exhausted (training should stop).
    pub fn check(&mut self, value: f32) -> bool {
        let improved = match self.mode {
            MonitorMode::Min => value < self.best_value - self.min_delta,
            MonitorMode::Max => value > self.best_value + self.min_delta,
        };

        if improved {
            self.best_value = value;
            self.counter = 0;
        } else {
            self.counter += 1;
        }

        if self.counter >= self.patience {
            self.stopped = true;
        }

        self.stopped
    }

    /// Whether stop was triggered.
    pub fn stopped(&self) -> bool {
        self.stopped
    }

    /// Best value seen so far.
    pub fn best_value(&self) -> f32 {
        self.best_value
    }

    /// Number of epochs without improvement.
    pub fn counter(&self) -> usize {
        self.counter
    }

    /// Reset state for a new training run.
    pub fn reset(&mut self) {
        self.best_value = match self.mode {
            MonitorMode::Min => f32::INFINITY,
            MonitorMode::Max => f32::NEG_INFINITY,
        };
        self.counter = 0;
        self.stopped = false;
    }
}

/// Saves model weights when monitored metric improves.
///
/// This struct tracks the best metric value and signals when a new best is found.
/// The caller is responsible for performing the actual model serialisation.
#[derive(Debug, Clone)]
pub struct BestModelCheckpoint {
    save_path: PathBuf,
    mode: MonitorMode,
    best_value: f32,
    monitor: String,
}

impl BestModelCheckpoint {
    /// Create a new checkpoint tracker.
    pub fn new(save_path: PathBuf, mode: MonitorMode) -> Self {
        let best_value = match mode {
            MonitorMode::Min => f32::INFINITY,
            MonitorMode::Max => f32::NEG_INFINITY,
        };
        Self {
            save_path,
            mode,
            best_value,
            monitor: "loss".to_string(),
        }
    }

    /// Check if metric improved. Returns `true` if a new best was found.
    pub fn check(&mut self, value: f32) -> bool {
        let improved = match self.mode {
            MonitorMode::Min => value < self.best_value,
            MonitorMode::Max => value > self.best_value,
        };

        if improved {
            self.best_value = value;
        }

        improved
    }

    /// Get the save path.
    pub fn save_path(&self) -> &Path {
        &self.save_path
    }

    /// Best value seen.
    pub fn best_value(&self) -> f32 {
        self.best_value
    }
}

/// Trait for training callbacks invoked after each epoch.
pub trait TrainingCallback {
    /// Called after each epoch. Returns true if training should stop.
    fn on_epoch_end(&mut self, epoch: usize, metrics: &HashMap<String, f32>) -> bool;

    /// Called after each batch within an epoch. Default implementation does nothing.
    fn on_batch_end(&mut self, _epoch: usize, _batch: usize, _loss: f32) {}
}

impl EarlyStopping {
    /// Set the metric key to monitor (default: `"loss"`).
    pub fn with_monitor(mut self, key: impl Into<String>) -> Self {
        self.monitor = key.into();
        self
    }
}

impl TrainingCallback for EarlyStopping {
    fn on_epoch_end(&mut self, _epoch: usize, metrics: &HashMap<String, f32>) -> bool {
        if let Some(&value) = metrics.get(&self.monitor) {
            self.check(value)
        } else {
            false
        }
    }
}

impl BestModelCheckpoint {
    /// Set the metric key to monitor (default: `"loss"`).
    pub fn with_monitor(mut self, key: impl Into<String>) -> Self {
        self.monitor = key.into();
        self
    }
}

impl TrainingCallback for BestModelCheckpoint {
    fn on_epoch_end(&mut self, _epoch: usize, metrics: &HashMap<String, f32>) -> bool {
        if let Some(&value) = metrics.get(&self.monitor) {
            self.check(value);
        }
        // BestModelCheckpoint never requests training to stop.
        false
    }
}

/// Logs training metrics to a CSV file and prints a summary line to stdout.
///
/// Creates a CSV file at the specified path with columns:
/// `epoch,train_loss,val_loss,learning_rate,duration_ms`
///
/// On each `on_epoch_end`, appends a row with the current metrics and prints a
/// formatted summary to stdout.
pub struct MetricsLogger {
    path: PathBuf,
    file: Option<std::fs::File>,
    start_time: std::time::Instant,
}

impl MetricsLogger {
    /// Create a new metrics logger that writes CSV rows to `path`.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            file: None,
            start_time: std::time::Instant::now(),
        }
    }

    /// Returns the path of the CSV file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Ensure the CSV file is open and the header has been written.
    fn ensure_file(&mut self) -> Option<&mut std::fs::File> {
        if self.file.is_none()
            && let Ok(mut f) = std::fs::File::create(&self.path)
        {
            let _ = writeln!(f, "epoch,train_loss,val_loss,learning_rate,duration_ms");
            self.file = Some(f);
            self.start_time = std::time::Instant::now();
        }
        self.file.as_mut()
    }
}

impl TrainingCallback for MetricsLogger {
    fn on_epoch_end(&mut self, epoch: usize, metrics: &HashMap<String, f32>) -> bool {
        let train_loss = metrics
            .get("train_loss")
            .or_else(|| metrics.get("loss"))
            .copied()
            .unwrap_or(f32::NAN);
        let val_loss = metrics.get("val_loss").copied().unwrap_or(f32::NAN);
        let lr = metrics
            .get("learning_rate")
            .or_else(|| metrics.get("lr"))
            .copied()
            .unwrap_or(f32::NAN);
        let duration_ms = self.start_time.elapsed().as_millis() as u64;

        if let Some(f) = self.ensure_file() {
            let _ = writeln!(f, "{epoch},{train_loss},{val_loss},{lr},{duration_ms}");
            let _ = f.flush();
        }

        println!(
            "[Epoch {epoch}] train_loss={train_loss:.4} val_loss={val_loss:.4} lr={lr:.6} elapsed={duration_ms}ms"
        );

        // Reset timer for next epoch measurement.
        self.start_time = std::time::Instant::now();

        // MetricsLogger never requests training to stop.
        false
    }
}

/// Train for multiple epochs with callbacks.
///
/// Training stops early if any callback returns `true` from `on_epoch_end`.
/// Returns the number of epochs actually trained.
pub fn train_epochs_with_callbacks<F>(
    mut train_fn: F,
    epochs: usize,
    callbacks: &mut [&mut dyn TrainingCallback],
) -> usize
where
    F: FnMut(usize) -> HashMap<String, f32>,
{
    for epoch in 0..epochs {
        let metrics = train_fn(epoch);
        let mut should_stop = false;
        for cb in callbacks.iter_mut() {
            should_stop = cb.on_epoch_end(epoch, &metrics) || should_stop;
        }
        if should_stop {
            return epoch + 1;
        }
    }
    epochs
}
