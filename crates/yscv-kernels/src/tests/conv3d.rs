use crate::conv3d;

use super::assert_slice_close;

/// 1x1x1 kernel with C_in == C_out acts as identity (passthrough).
#[test]
fn test_conv3d_identity_kernel() {
    // Input: [1, 2, 2, 2, 3]  (batch=1, D=2, H=2, W=2, C_in=3)
    // Kernel: [1, 1, 1, 3, 3] identity matrix per spatial position
    let input: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let input_shape = [1, 2, 2, 2, 3];

    // Build a 1x1x1 identity kernel: for each (ci, co), value = 1 if ci==co else 0
    let c = 3;
    let mut kernel = vec![0.0f32; c * c];
    for i in 0..c {
        kernel[i * c + i] = 1.0;
    }
    let kernel_shape = [1, 1, 1, 3, 3];

    let (output, output_shape) = conv3d(
        &input,
        &input_shape,
        &kernel,
        &kernel_shape,
        (1, 1, 1),
        (0, 0, 0),
    );

    assert_eq!(output_shape, vec![1, 2, 2, 2, 3]);
    // 1e-3 tolerance covers cross-platform BLAS implementation variance
    // (Accelerate on macOS vs OpenBLAS on Linux/Windows reorder inner
    // products differently — identity passthrough is exact on macOS but
    // can drift by ~1e-5 on OpenBLAS).
    assert_slice_close(&output, &input, 1e-3);
}

/// Stride > 1 reduces output spatial dimensions.
#[test]
fn test_conv3d_stride() {
    // Input: [1, 4, 4, 4, 1]
    let input = vec![1.0f32; 4 * 4 * 4];
    let input_shape = [1, 4, 4, 4, 1];

    // Kernel: [1, 1, 1, 1, 1] with value 1.0
    let kernel = vec![1.0f32; 1];
    let kernel_shape = [1, 1, 1, 1, 1];

    let (output, output_shape) = conv3d(
        &input,
        &input_shape,
        &kernel,
        &kernel_shape,
        (2, 2, 2),
        (0, 0, 0),
    );

    // out_dim = (4 - 1) / 2 + 1 = 2 for each spatial dim
    assert_eq!(output_shape, vec![1, 2, 2, 2, 1]);
    assert_eq!(output.len(), 8);
    // All values should be 1.0 since kernel is 1x1x1 with weight 1
    assert_slice_close(&output, &[1.0f32; 8], 1e-3);
}

/// Padding preserves spatial dimensions when kernel=3, stride=1, padding=1.
#[test]
fn test_conv3d_padding() {
    // Input: [1, 3, 3, 3, 1]
    let input = vec![1.0f32; 27];
    let input_shape = [1, 3, 3, 3, 1];

    // Kernel: [3, 3, 3, 1, 1] all ones
    let kernel = vec![1.0f32; 27];
    let kernel_shape = [3, 3, 3, 1, 1];

    let (output, output_shape) = conv3d(
        &input,
        &input_shape,
        &kernel,
        &kernel_shape,
        (1, 1, 1),
        (1, 1, 1),
    );

    // out_dim = (3 + 2*1 - 3) / 1 + 1 = 3 for each spatial dim
    assert_eq!(output_shape, vec![1, 3, 3, 3, 1]);

    // Center voxel (1,1,1) sees all 27 input values => sum = 27
    // index for (0, 1, 1, 1, 0) = 1*9 + 1*3 + 1 = 13
    assert!((output[13] - 27.0).abs() < 1e-3);

    // Corner voxel (0,0,0) with padding=1: kernel covers [-1..1] in each dim,
    // only [0..1] is in-bounds => 2×2×2 = 8 real values of 1.0
    assert!((output[0] - 8.0).abs() < 1e-3);
}

/// Small 2x2x2 input with a known kernel, verify exact output values.
#[test]
fn test_conv3d_known_values() {
    // Input: [1, 2, 2, 2, 1], values 1..=8
    let input: Vec<f32> = (1..=8).map(|i| i as f32).collect();
    let input_shape = [1, 2, 2, 2, 1];

    // Kernel: [2, 2, 2, 1, 1], all ones => output is sum of all input
    let kernel = vec![1.0f32; 8];
    let kernel_shape = [2, 2, 2, 1, 1];

    let (output, output_shape) = conv3d(
        &input,
        &input_shape,
        &kernel,
        &kernel_shape,
        (1, 1, 1),
        (0, 0, 0),
    );

    // out_dim = (2 - 2)/1 + 1 = 1 for each dim
    assert_eq!(output_shape, vec![1, 1, 1, 1, 1]);
    // Sum of 1+2+3+4+5+6+7+8 = 36
    assert!((output[0] - 36.0).abs() < 1e-3);

    // Now test with 2 output channels and a weighted kernel
    // Kernel: [2, 2, 2, 1, 2]
    // Channel 0: all ones, Channel 1: all twos
    let mut kernel2 = vec![0.0f32; 16];
    for i in 0..8 {
        kernel2[i * 2] = 1.0; // c_out=0
        kernel2[i * 2 + 1] = 2.0; // c_out=1
    }
    let kernel2_shape = [2, 2, 2, 1, 2];

    let (output2, output2_shape) = conv3d(
        &input,
        &input_shape,
        &kernel2,
        &kernel2_shape,
        (1, 1, 1),
        (0, 0, 0),
    );

    assert_eq!(output2_shape, vec![1, 1, 1, 1, 2]);
    assert!((output2[0] - 36.0).abs() < 1e-3); // channel 0: sum * 1
    assert!((output2[1] - 72.0).abs() < 1e-3); // channel 1: sum * 2
}
