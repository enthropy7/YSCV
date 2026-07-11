use std::collections::HashMap;

use crate::TrainingLog;

#[test]
fn test_training_log_records_epochs() {
    let mut log = TrainingLog::new();
    for i in 0..3 {
        let mut m = HashMap::default();
        m.insert("loss".to_string(), 1.0 - i as f32 * 0.1);
        log.log_epoch(m);
    }
    assert_eq!(log.num_epochs(), 3);
    assert_eq!(log.entries().len(), 3);
}

#[test]
fn test_training_log_get_metric_history() {
    let mut log = TrainingLog::new();
    let losses = [0.9, 0.7, 0.5];
    for &l in &losses {
        let mut m = HashMap::default();
        m.insert("loss".to_string(), l);
        m.insert("acc".to_string(), 1.0 - l);
        log.log_epoch(m);
    }
    let history = log.get_metric_history("loss");
    assert_eq!(history.len(), 3);
    assert!((history[0] - 0.9).abs() < 1e-6);
    assert!((history[1] - 0.7).abs() < 1e-6);
    assert!((history[2] - 0.5).abs() < 1e-6);

    let acc_history = log.get_metric_history("acc");
    assert_eq!(acc_history.len(), 3);

    // Non-existent metric returns empty
    assert!(log.get_metric_history("lr").is_empty());
}

#[test]
fn test_training_log_to_csv() {
    let mut log = TrainingLog::new();

    let mut m1 = HashMap::default();
    m1.insert("acc".to_string(), 0.8);
    m1.insert("loss".to_string(), 0.5);
    log.log_epoch(m1);

    let mut m2 = HashMap::default();
    m2.insert("acc".to_string(), 0.9);
    m2.insert("loss".to_string(), 0.3);
    log.log_epoch(m2);

    let csv = log.to_csv();
    let lines: Vec<&str> = csv.lines().collect();
    assert_eq!(lines.len(), 3); // header + 2 rows
    // Keys are sorted alphabetically
    assert_eq!(lines[0], "acc,loss");
    assert!(lines[1].starts_with("0.8,"));
    assert!(lines[2].starts_with("0.9,"));
}
