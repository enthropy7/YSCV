//! Tests for the `Layout` metadata tag on `Tensor`.
//!
//! The tag is **metadata-only**: it does not reorder data. These tests
//! verify that (1) the default is NCHW, (2) `with_layout` round-trips,
//! (3) layout is preserved through reshape / to_dtype / to_device,
//! and (4) `Layout::NCHWc { block }` compares by block size.

use crate::{DType, Device, Layout, Tensor};

#[test]
fn default_layout_is_nchw() {
    let t = Tensor::zeros(vec![1, 3, 4, 4]).unwrap();
    assert_eq!(t.layout(), Layout::NCHW);
}

#[test]
fn scalar_default_layout_is_nchw() {
    let t = Tensor::scalar(1.0);
    assert_eq!(t.layout(), Layout::NCHW);
}

#[test]
fn with_layout_nhwc_sets_tag_only() {
    let t = Tensor::zeros(vec![1, 4, 4, 3]).unwrap();
    let nhwc = t.clone().with_layout(Layout::NHWC);
    assert_eq!(nhwc.layout(), Layout::NHWC);
    // Metadata-only change: data and shape untouched.
    assert_eq!(nhwc.data(), t.data());
    assert_eq!(nhwc.shape(), t.shape());
}

#[test]
fn with_layout_nchwc_carries_block_size() {
    let t = Tensor::zeros(vec![1, 2, 4, 4, 8]).unwrap();
    let blk = t.with_layout(Layout::NCHWc { block: 8 });
    assert_eq!(blk.layout(), Layout::NCHWc { block: 8 });
}

#[test]
fn layout_preserved_by_reshape() {
    let t = Tensor::ones(vec![1, 3, 4, 4])
        .unwrap()
        .with_layout(Layout::NHWC);
    let r = t.reshape(vec![1, 48]).unwrap();
    assert_eq!(r.layout(), Layout::NHWC);
}

#[test]
fn layout_preserved_by_into_reshape() {
    let t = Tensor::ones(vec![1, 3, 4, 4])
        .unwrap()
        .with_layout(Layout::NHWC);
    let r = t.into_reshape(vec![1, 48]).unwrap();
    assert_eq!(r.layout(), Layout::NHWC);
}

#[test]
fn layout_preserved_by_to_dtype() {
    let t = Tensor::ones(vec![1, 3, 4, 4])
        .unwrap()
        .with_layout(Layout::NCHWc { block: 8 });
    let h = t.to_dtype(DType::F16);
    assert_eq!(h.layout(), Layout::NCHWc { block: 8 });
}

#[test]
fn layout_preserved_by_to_device() {
    let t = Tensor::ones(vec![1, 3, 4, 4])
        .unwrap()
        .with_layout(Layout::NHWC);
    let g = t.to_device(Device::Gpu(0));
    assert_eq!(g.layout(), Layout::NHWC);
    assert_eq!(g.device(), Device::Gpu(0));
}

#[test]
fn layout_enum_equality() {
    assert_eq!(Layout::NCHW, Layout::NCHW);
    assert_eq!(Layout::NHWC, Layout::NHWC);
    assert_eq!(Layout::NCHWc { block: 8 }, Layout::NCHWc { block: 8 });
    assert_ne!(Layout::NCHW, Layout::NHWC);
    assert_ne!(Layout::NCHWc { block: 8 }, Layout::NCHWc { block: 16 });
}

#[test]
fn layout_default_is_nchw() {
    assert_eq!(Layout::default(), Layout::NCHW);
}
