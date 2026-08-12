use yscv_tensor::{AlignedVec, Tensor, TensorError};

use super::super::error::KernelError;

/// Deformable convolution: like standard conv2d but with learned offsets.
///
/// Input: [N, H, W, C_in] (NHWC format)
/// Weight: [kH, kW, C_in, C_out]
/// Offsets: [N, H_out, W_out, kH * kW * 2] — dx,dy offsets for each kernel position
///
/// For each output position (oh, ow) and each kernel position (kh, kw):
///   offset_idx = (kh * kW + kw) * 2
///   sampled_y = oh * stride + kh - padding + offsets[n, oh, ow, offset_idx]
///   sampled_x = ow * stride + kw - padding + offsets[n, oh, ow, offset_idx + 1]
///   Use bilinear interpolation at (sampled_y, sampled_x) in input
#[allow(unsafe_code)]
pub fn deformable_conv2d_nhwc(
    input: &Tensor,
    weight: &Tensor,
    offsets: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
) -> Result<Tensor, KernelError> {
    // Validate ranks
    if input.rank() != 4 || weight.rank() != 4 {
        return Err(KernelError::InvalidConvRank {
            input_rank: input.rank(),
            kernel_rank: weight.rank(),
        });
    }
    if stride == 0 {
        return Err(KernelError::InvalidConvParameters {
            kernel_h: weight.shape()[0],
            kernel_w: weight.shape()[1],
            stride_h: stride,
            stride_w: stride,
        });
    }

    let batch = input.shape()[0];
    let in_h = input.shape()[1];
    let in_w = input.shape()[2];
    let in_channels = input.shape()[3];

    let kernel_h = weight.shape()[0];
    let kernel_w = weight.shape()[1];
    let kernel_in_channels = weight.shape()[2];
    let out_channels = weight.shape()[3];

    if kernel_h == 0 || kernel_w == 0 {
        return Err(KernelError::InvalidConvParameters {
            kernel_h,
            kernel_w,
            stride_h: stride,
            stride_w: stride,
        });
    }
    if kernel_in_channels != in_channels {
        return Err(KernelError::ConvChannelMismatch {
            input_channels: in_channels,
            kernel_in_channels,
        });
    }

    let out_h = (in_h + 2 * padding - kernel_h) / stride + 1;
    let out_w = (in_w + 2 * padding - kernel_w) / stride + 1;

    // Validate offsets shape: [N, out_h, out_w, kH*kW*2]
    let expected_offset_last = kernel_h * kernel_w * 2;
    if offsets.rank() != 4
        || offsets.shape()[0] != batch
        || offsets.shape()[1] != out_h
        || offsets.shape()[2] != out_w
        || offsets.shape()[3] != expected_offset_last
    {
        return Err(KernelError::DeformableConvOffsetShapeMismatch {
            expected: vec![batch, out_h, out_w, expected_offset_last],
            got: offsets.shape().to_vec(),
        });
    }

    if let Some(bias_tensor) = bias
        && (bias_tensor.rank() != 1 || bias_tensor.shape()[0] != out_channels)
    {
        return Err(KernelError::ConvBiasShapeMismatch {
            bias_shape: bias_tensor.shape().to_vec(),
            out_channels,
        });
    }

    let output_len = batch
        .checked_mul(out_h)
        .and_then(|v| v.checked_mul(out_w))
        .and_then(|v| v.checked_mul(out_channels))
        .ok_or_else(|| {
            KernelError::Tensor(TensorError::SizeOverflow {
                shape: vec![batch, out_h, out_w, out_channels],
            })
        })?;

    // SAFETY: deformable convolution writes every output element before returning.
    let mut output = unsafe { AlignedVec::<f32>::uninitialized(output_len) };
    let input_data = input.data();
    let weight_data = weight.data();
    let offset_data = offsets.data();
    let bias_data = bias.map(Tensor::data);

    let in_hwc = in_h * in_w * in_channels;
    let in_wc = in_w * in_channels;

    let offset_hwk = out_h * out_w * expected_offset_last;
    let offset_wk = out_w * expected_offset_last;

    let out_hwc = out_h * out_w * out_channels;
    let out_wc = out_w * out_channels;

    for n in 0..batch {
        let batch_input_base = n * in_hwc;
        let batch_offset_base = n * offset_hwk;
        let batch_output_base = n * out_hwc;

        for oh in 0..out_h {
            for ow in 0..out_w {
                let out_base = batch_output_base + oh * out_wc + ow * out_channels;
                let off_base = batch_offset_base + oh * offset_wk + ow * expected_offset_last;

                for oc in 0..out_channels {
                    let mut acc = bias_data.map_or(0.0, |b| b[oc]);

                    for kh in 0..kernel_h {
                        for kw in 0..kernel_w {
                            let offset_idx = (kh * kernel_w + kw) * 2;
                            let dy = offset_data[off_base + offset_idx];
                            let dx = offset_data[off_base + offset_idx + 1];

                            let sampled_y = (oh * stride + kh) as f32 - padding as f32 + dy;
                            let sampled_x = (ow * stride + kw) as f32 - padding as f32 + dx;

                            // Bilinear interpolation
                            let kernel_base = (kh * kernel_w + kw) * in_channels * out_channels;

                            for ic in 0..in_channels {
                                let val = bilinear_sample(
                                    input_data,
                                    batch_input_base,
                                    in_h,
                                    in_w,
                                    in_channels,
                                    in_wc,
                                    sampled_y,
                                    sampled_x,
                                    ic,
                                );
                                let w_idx = kernel_base + ic * out_channels + oc;
                                acc += val * weight_data[w_idx];
                            }
                        }
                    }

                    output[out_base + oc] = acc;
                }
            }
        }
    }

    Tensor::from_aligned(vec![batch, out_h, out_w, out_channels], output).map_err(Into::into)
}

/// Bilinear interpolation sampling from a single image in an NHWC tensor.
///
/// Returns 0.0 for out-of-bounds samples (zero-padding).
#[inline]
fn bilinear_sample(
    data: &[f32],
    batch_base: usize,
    in_h: usize,
    in_w: usize,
    in_channels: usize,
    in_wc: usize,
    y: f32,
    x: f32,
    channel: usize,
) -> f32 {
    if y < -1.0 || y > in_h as f32 || x < -1.0 || x > in_w as f32 {
        return 0.0;
    }

    let y0 = y.floor() as i32;
    let x0 = x.floor() as i32;
    let y1 = y0 + 1;
    let x1 = x0 + 1;

    let ly = y - y0 as f32;
    let lx = x - x0 as f32;
    let hy = 1.0 - ly;
    let hx = 1.0 - lx;

    let fetch = |iy: i32, ix: i32| -> f32 {
        if iy < 0 || iy >= in_h as i32 || ix < 0 || ix >= in_w as i32 {
            return 0.0;
        }
        let idx = batch_base + (iy as usize) * in_wc + (ix as usize) * in_channels + channel;
        data[idx]
    };

    let v00 = fetch(y0, x0);
    let v01 = fetch(y0, x1);
    let v10 = fetch(y1, x0);
    let v11 = fetch(y1, x1);

    hy * hx * v00 + hy * lx * v01 + ly * hx * v10 + ly * lx * v11
}
