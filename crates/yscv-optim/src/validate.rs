use super::OptimError;

pub(crate) fn validate_lr(lr: f32) -> Result<(), OptimError> {
    if !lr.is_finite() || lr < 0.0 {
        return Err(OptimError::InvalidLearningRate { lr });
    }
    Ok(())
}

pub(crate) fn validate_momentum(momentum: f32) -> Result<(), OptimError> {
    if !momentum.is_finite() || !(0.0..1.0).contains(&momentum) {
        return Err(OptimError::InvalidMomentum { momentum });
    }
    Ok(())
}

pub(crate) fn validate_beta1(beta1: f32) -> Result<(), OptimError> {
    if !beta1.is_finite() || !(0.0..1.0).contains(&beta1) {
        return Err(OptimError::InvalidBeta1 { beta1 });
    }
    Ok(())
}

pub(crate) fn validate_beta2(beta2: f32) -> Result<(), OptimError> {
    if !beta2.is_finite() || !(0.0..1.0).contains(&beta2) {
        return Err(OptimError::InvalidBeta2 { beta2 });
    }
    Ok(())
}

pub(crate) fn validate_epsilon(epsilon: f32) -> Result<(), OptimError> {
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(OptimError::InvalidEpsilon { epsilon });
    }
    Ok(())
}

pub(crate) fn validate_rmsprop_alpha(alpha: f32) -> Result<(), OptimError> {
    if !alpha.is_finite() || !(0.0..1.0).contains(&alpha) {
        return Err(OptimError::InvalidRmsPropAlpha { alpha });
    }
    Ok(())
}

pub(crate) fn validate_step_gamma(gamma: f32) -> Result<(), OptimError> {
    if !gamma.is_finite() || !(0.0..=1.0).contains(&gamma) || gamma == 0.0 {
        return Err(OptimError::InvalidStepGamma { gamma });
    }
    Ok(())
}

pub(crate) const fn validate_step_size(step_size: usize) -> Result<(), OptimError> {
    if step_size == 0 {
        return Err(OptimError::InvalidStepSize { step_size });
    }
    Ok(())
}

pub(crate) const fn validate_cosine_t_max(t_max: usize) -> Result<(), OptimError> {
    if t_max == 0 {
        return Err(OptimError::InvalidCosineTMax { t_max });
    }
    Ok(())
}

pub(crate) const fn validate_warmup_steps(warmup_steps: usize) -> Result<(), OptimError> {
    if warmup_steps == 0 {
        return Err(OptimError::InvalidWarmupSteps { warmup_steps });
    }
    Ok(())
}

pub(crate) const fn validate_one_cycle_total_steps(total_steps: usize) -> Result<(), OptimError> {
    if total_steps == 0 {
        return Err(OptimError::InvalidOneCycleTotalSteps { total_steps });
    }
    Ok(())
}

pub(crate) fn validate_one_cycle_pct_start(pct_start: f32) -> Result<(), OptimError> {
    if !pct_start.is_finite() || pct_start <= 0.0 || pct_start > 1.0 {
        return Err(OptimError::InvalidOneCyclePctStart { pct_start });
    }
    Ok(())
}

pub(crate) fn validate_one_cycle_final_div_factor(final_div_factor: f32) -> Result<(), OptimError> {
    if !final_div_factor.is_finite() || final_div_factor <= 1.0 {
        return Err(OptimError::InvalidOneCycleFinalDivFactor { final_div_factor });
    }
    Ok(())
}

pub(crate) fn validate_dampening(dampening: f32) -> Result<(), OptimError> {
    if !dampening.is_finite() || !(0.0..=1.0).contains(&dampening) {
        return Err(OptimError::InvalidDampening { dampening });
    }
    Ok(())
}
