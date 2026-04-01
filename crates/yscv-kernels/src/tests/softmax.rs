use yscv_tensor::Tensor;

use crate::{
    KernelError, ParallelElementwiseConfig, log_softmax_last_dim, log_softmax_last_dim_with_config,
    logsumexp_last_dim, logsumexp_last_dim_with_config, softmax_last_dim,
    softmax_last_dim_with_config,
};

use super::{assert_slice_close, build_tensor};

#[test]
fn softmax_last_dim_computes_expected_result() {
    let input = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let out = softmax_last_dim(&input).unwrap();
    assert_eq!(out.shape(), &[3]);
    assert_slice_close(out.data(), &[0.09003057, 0.24472848, 0.66524094], 1e-6);
}

#[test]
fn softmax_last_dim_normalizes_each_row() {
    let input = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 0.1, 0.2, 0.3]).unwrap();
    let out = softmax_last_dim(&input).unwrap();
    assert_eq!(out.shape(), &[2, 3]);
    let row0_sum = out.data()[0] + out.data()[1] + out.data()[2];
    let row1_sum = out.data()[3] + out.data()[4] + out.data()[5];
    assert!((row0_sum - 1.0).abs() <= 1e-6);
    assert!((row1_sum - 1.0).abs() <= 1e-6);
}

#[test]
fn softmax_last_dim_rejects_scalar_rank() {
    let input = Tensor::scalar(1.0);
    let err = softmax_last_dim(&input).unwrap_err();
    assert_eq!(err, KernelError::InvalidSoftmaxRank { got_rank: 0 });
}

#[test]
// Miri's software FP emulation is non-deterministic: two calls to f32::exp()
// with the same input can return different bit patterns (known Miri limitation).
#[cfg_attr(miri, ignore)]
fn softmax_last_dim_with_config_disabled_matches_default() {
    let input = build_tensor(&[64, 32], 0.52);
    let baseline = softmax_last_dim(&input).unwrap();
    let disabled =
        softmax_last_dim_with_config(&input, ParallelElementwiseConfig::disabled()).unwrap();
    assert_eq!(baseline, disabled);
}

#[test]
fn log_softmax_last_dim_computes_expected_result() {
    let input = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let out = log_softmax_last_dim(&input).unwrap();
    assert_eq!(out.shape(), &[3]);
    let log_denom = (1.0f32.exp() + 2.0f32.exp() + 3.0f32.exp()).ln();
    assert_slice_close(
        out.data(),
        &[1.0 - log_denom, 2.0 - log_denom, 3.0 - log_denom],
        1e-6,
    );
}

#[test]
fn log_softmax_last_dim_matches_ln_softmax() {
    let input = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 0.1, 0.2, 0.3]).unwrap();
    let softmax = softmax_last_dim(&input).unwrap();
    let log_softmax = log_softmax_last_dim(&input).unwrap();
    assert_eq!(softmax.shape(), log_softmax.shape());
    for index in 0..softmax.len() {
        assert!(
            (softmax.data()[index].ln() - log_softmax.data()[index]).abs() <= 1e-5,
            "mismatch at {index}: ln(softmax)={} log_softmax={}",
            softmax.data()[index].ln(),
            log_softmax.data()[index]
        );
    }
}

#[test]
fn log_softmax_last_dim_rejects_scalar_rank() {
    let input = Tensor::scalar(1.0);
    let err = log_softmax_last_dim(&input).unwrap_err();
    assert_eq!(err, KernelError::InvalidLogSoftmaxRank { got_rank: 0 });
}

#[test]
// Miri's software FP emulation is non-deterministic: two calls to f32::exp()
// with the same input can return different bit patterns (known Miri limitation).
#[cfg_attr(miri, ignore)]
fn log_softmax_last_dim_with_config_disabled_matches_default() {
    let input = build_tensor(&[64, 32], 0.66);
    let baseline = log_softmax_last_dim(&input).unwrap();
    let disabled =
        log_softmax_last_dim_with_config(&input, ParallelElementwiseConfig::disabled()).unwrap();
    assert_eq!(baseline, disabled);
}

#[test]
fn logsumexp_last_dim_computes_expected_result() {
    let input = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let out = logsumexp_last_dim(&input).unwrap();
    assert_eq!(out.shape(), &[1]);
    let expected = (1.0f32.exp() + 2.0f32.exp() + 3.0f32.exp()).ln();
    assert_slice_close(out.data(), &[expected], 1e-5);
}

#[test]
fn logsumexp_last_dim_reduces_each_row() {
    let input = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 0.1, 0.2, 0.3]).unwrap();
    let out = logsumexp_last_dim(&input).unwrap();
    assert_eq!(out.shape(), &[2, 1]);
    let row0 = (1.0f32.exp() + 2.0f32.exp() + 3.0f32.exp()).ln();
    let row1 = (0.1f32.exp() + 0.2f32.exp() + 0.3f32.exp()).ln();
    assert_slice_close(out.data(), &[row0, row1], 1e-5);
}

#[test]
fn logsumexp_last_dim_rejects_scalar_rank() {
    let input = Tensor::scalar(1.0);
    let err = logsumexp_last_dim(&input).unwrap_err();
    assert_eq!(err, KernelError::InvalidLogSumExpRank { got_rank: 0 });
}

#[test]
// Miri's software FP emulation is non-deterministic: two calls to f32::exp()
// with the same input can return different bit patterns (known Miri limitation).
#[cfg_attr(miri, ignore)]
fn logsumexp_last_dim_with_config_disabled_matches_default() {
    let input = build_tensor(&[64, 32], 0.77);
    let baseline = logsumexp_last_dim(&input).unwrap();
    let disabled =
        logsumexp_last_dim_with_config(&input, ParallelElementwiseConfig::disabled()).unwrap();
    assert_eq!(baseline, disabled);
}
