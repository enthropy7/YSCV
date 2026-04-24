mod attention;
mod conv2d;
mod conv3d;
mod deformable_conv;
mod elementwise;
mod embedding;
mod gpu_ops;
mod matmul;
mod nchwc;
mod normalization;
mod pooling;
mod softmax;
mod threaded;

use yscv_tensor::Tensor;

fn build_tensor(shape: &[usize], seed: f32) -> Tensor {
    let len = shape.iter().copied().product::<usize>();
    let mut data = Vec::with_capacity(len);
    for idx in 0..len {
        data.push(((idx % 97) as f32 * 0.017 + seed).fract());
    }
    Tensor::from_vec(shape.to_vec(), data).unwrap()
}

fn assert_slice_close(lhs: &[f32], rhs: &[f32], tolerance: f32) {
    assert_eq!(lhs.len(), rhs.len());
    for index in 0..lhs.len() {
        let distance = (lhs[index] - rhs[index]).abs();
        assert!(
            distance <= tolerance,
            "index {index}: left={} right={} distance={} tolerance={}",
            lhs[index],
            rhs[index],
            distance,
            tolerance
        );
    }
}
