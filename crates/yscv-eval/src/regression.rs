use crate::error::EvalError;

const fn check_lengths(predictions: &[f32], targets: &[f32]) -> Result<(), EvalError> {
    if predictions.len() != targets.len() {
        return Err(EvalError::CountLengthMismatch {
            ground_truth: targets.len(),
            predictions: predictions.len(),
        });
    }
    Ok(())
}

/// Coefficient of determination: 1 - SS_res / SS_tot.
///
/// Returns `0.0` for empty inputs (no data to explain).
pub fn r2_score(predictions: &[f32], targets: &[f32]) -> Result<f32, EvalError> {
    check_lengths(predictions, targets)?;
    if targets.is_empty() {
        return Ok(0.0);
    }
    let mean = targets.iter().sum::<f32>() / targets.len() as f32;
    let ss_tot: f32 = targets.iter().map(|t| (t - mean).powi(2)).sum();
    if ss_tot == 0.0 {
        // All targets identical. If predictions match, perfect; otherwise undefined — return 0.
        let ss_res: f32 = predictions
            .iter()
            .zip(targets)
            .map(|(p, t)| (p - t).powi(2))
            .sum();
        return Ok(if ss_res == 0.0 { 1.0 } else { 0.0 });
    }
    let ss_res: f32 = predictions
        .iter()
        .zip(targets)
        .map(|(p, t)| (p - t).powi(2))
        .sum();
    Ok(1.0 - ss_res / ss_tot)
}

/// Mean absolute error.
///
/// Returns `0.0` for empty inputs.
pub fn mae(predictions: &[f32], targets: &[f32]) -> Result<f32, EvalError> {
    check_lengths(predictions, targets)?;
    if targets.is_empty() {
        return Ok(0.0);
    }
    let sum: f32 = predictions
        .iter()
        .zip(targets)
        .map(|(p, t)| (p - t).abs())
        .sum();
    Ok(sum / targets.len() as f32)
}

/// Root mean squared error.
///
/// Returns `0.0` for empty inputs.
pub fn rmse(predictions: &[f32], targets: &[f32]) -> Result<f32, EvalError> {
    check_lengths(predictions, targets)?;
    if targets.is_empty() {
        return Ok(0.0);
    }
    let sum: f32 = predictions
        .iter()
        .zip(targets)
        .map(|(p, t)| (p - t).powi(2))
        .sum();
    Ok((sum / targets.len() as f32).sqrt())
}

/// Mean absolute percentage error, skipping pairs where the target is zero.
///
/// Returns `0.0` for empty inputs or when all targets are zero.
pub fn mape(predictions: &[f32], targets: &[f32]) -> Result<f32, EvalError> {
    check_lengths(predictions, targets)?;
    if targets.is_empty() {
        return Ok(0.0);
    }
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for (p, t) in predictions.iter().zip(targets) {
        if *t == 0.0 {
            continue;
        }
        sum += ((p - t) / t).abs();
        count += 1;
    }
    if count == 0 {
        return Ok(0.0);
    }
    Ok(sum / count as f32)
}
