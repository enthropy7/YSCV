use crate::{
    BestModelCheckpoint, EarlyStopping, MonitorMode, TrainingCallback, train_epochs_with_callbacks,
};
use std::collections::HashMap;
use std::path::PathBuf;

#[test]
fn test_early_stopping_triggers_after_patience() {
    let mut es = EarlyStopping::new(3, 0.0, MonitorMode::Min);
    // Initial improvement
    assert!(!es.check(1.0));
    // No improvement for 3 epochs
    assert!(!es.check(1.0));
    assert!(!es.check(1.0));
    assert!(es.check(1.0)); // patience exhausted
    assert!(es.stopped());
}

#[test]
fn test_early_stopping_resets_on_improvement() {
    let mut es = EarlyStopping::new(3, 0.0, MonitorMode::Min);
    assert!(!es.check(1.0)); // best=1.0
    assert!(!es.check(1.1)); // counter=1
    assert!(!es.check(1.2)); // counter=2
    assert!(!es.check(0.5)); // improvement! counter=0
    assert_eq!(es.counter(), 0);
    assert!(!es.check(0.6)); // counter=1
    assert!(!es.check(0.7)); // counter=2
    assert!(es.check(0.8)); // counter=3, stop
}

#[test]
fn test_early_stopping_min_delta() {
    let mut es = EarlyStopping::new(2, 0.1, MonitorMode::Min);
    assert!(!es.check(1.0)); // best=1.0
    // Improvement of 0.05 < min_delta=0.1, doesn't count
    assert!(!es.check(0.95)); // counter=1
    assert!(es.check(0.95)); // counter=2, stop

    // But a real improvement resets
    let mut es2 = EarlyStopping::new(2, 0.1, MonitorMode::Min);
    assert!(!es2.check(1.0));
    assert!(!es2.check(0.8)); // improvement of 0.2 > 0.1
    assert_eq!(es2.counter(), 0);
    assert!((es2.best_value() - 0.8).abs() < 1e-6);
}

#[test]
fn test_early_stopping_max_mode() {
    let mut es = EarlyStopping::new(2, 0.0, MonitorMode::Max);
    assert!(!es.check(0.7)); // best=0.7
    assert!(!es.check(0.8)); // improvement, best=0.8
    assert!(!es.check(0.8)); // counter=1
    assert!(es.check(0.7)); // counter=2, stop
}

#[test]
fn test_early_stopping_reset() {
    let mut es = EarlyStopping::new(2, 0.0, MonitorMode::Min);
    es.check(1.0);
    es.check(2.0);
    es.check(3.0); // stopped
    assert!(es.stopped());

    es.reset();
    assert!(!es.stopped());
    assert_eq!(es.counter(), 0);
    assert!(es.best_value().is_infinite());
}

#[test]
fn test_best_model_checkpoint_tracks_improvement() {
    let mut ckpt = BestModelCheckpoint::new(PathBuf::from("/tmp/best.pt"), MonitorMode::Min);
    assert!(ckpt.check(1.0)); // first value is always best
    assert!(!ckpt.check(1.5)); // worse
    assert!(ckpt.check(0.5)); // better
    assert!((ckpt.best_value() - 0.5).abs() < 1e-6);
}

#[test]
fn test_best_model_checkpoint_min_mode() {
    let mut ckpt = BestModelCheckpoint::new(PathBuf::from("/tmp/best.pt"), MonitorMode::Min);
    assert!(ckpt.check(5.0));
    assert!(ckpt.check(3.0));
    assert!(!ckpt.check(4.0));
    assert!(ckpt.check(2.0));
    assert!((ckpt.best_value() - 2.0).abs() < 1e-6);
    assert_eq!(ckpt.save_path(), PathBuf::from("/tmp/best.pt").as_path());
}

#[test]
fn test_best_model_checkpoint_max_mode() {
    let mut ckpt = BestModelCheckpoint::new(PathBuf::from("/tmp/best.pt"), MonitorMode::Max);
    assert!(ckpt.check(0.5));
    assert!(ckpt.check(0.7));
    assert!(!ckpt.check(0.6));
    assert!(ckpt.check(0.9));
    assert!((ckpt.best_value() - 0.9).abs() < 1e-6);
}

#[test]
fn test_early_stopping_callback_stops_training() {
    let mut es = EarlyStopping::new(2, 0.0, MonitorMode::Min);
    // Losses: 1.0, 0.9, 0.8, 0.85, 0.86, 0.87 ...
    // Loss stops improving at epoch 3 (index 3). Patience=2 means stop after epoch 4.
    let losses = [1.0_f32, 0.9, 0.8, 0.85, 0.86, 0.87, 0.88, 0.89];
    let trained = train_epochs_with_callbacks(
        |epoch| {
            let mut m = HashMap::default();
            m.insert("loss".to_string(), losses[epoch]);
            m
        },
        losses.len(),
        &mut [&mut es],
    );
    // Epochs 0,1,2 improve. Epoch 3: counter=1. Epoch 4: counter=2 -> stop.
    assert_eq!(trained, 5);
    assert!(es.stopped());
}

#[test]
fn test_callbacks_full_training() {
    let mut es = EarlyStopping::new(2, 0.0, MonitorMode::Min);
    // Loss keeps decreasing every epoch, so early stopping never triggers.
    let trained = train_epochs_with_callbacks(
        |epoch| {
            let mut m = HashMap::default();
            m.insert("loss".to_string(), 1.0 - epoch as f32 * 0.1);
            m
        },
        5,
        &mut [&mut es],
    );
    assert_eq!(trained, 5);
    assert!(!es.stopped());
}

/// A custom callback that stops training after a fixed number of epochs.
struct StopAfterCallback {
    max_epochs: usize,
}

impl TrainingCallback for StopAfterCallback {
    fn on_epoch_end(&mut self, epoch: usize, _metrics: &HashMap<String, f32>) -> bool {
        epoch + 1 >= self.max_epochs
    }
}

#[test]
fn test_multiple_callbacks() {
    // EarlyStopping won't trigger (loss keeps improving), but the custom
    // callback forces a stop after 3 epochs.
    let mut es = EarlyStopping::new(10, 0.0, MonitorMode::Min);
    let mut stop_after = StopAfterCallback { max_epochs: 3 };
    let trained = train_epochs_with_callbacks(
        |epoch| {
            let mut m = HashMap::default();
            m.insert("loss".to_string(), 1.0 - epoch as f32 * 0.1);
            m
        },
        10,
        &mut [&mut es, &mut stop_after],
    );
    assert_eq!(trained, 3);
    assert!(!es.stopped());
}
