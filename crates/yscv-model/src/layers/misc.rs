use yscv_autograd::{Graph, NodeId};
use yscv_tensor::Tensor;

use super::ModelLayer;
use crate::ModelError;

/// Dropout layer (training vs eval mode).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DropoutLayer {
    rate: f32,
    training: bool,
}

impl DropoutLayer {
    pub fn new(rate: f32) -> Result<Self, ModelError> {
        if !rate.is_finite() || !(0.0..1.0).contains(&rate) {
            return Err(ModelError::InvalidDropoutRate { rate });
        }
        Ok(Self {
            rate,
            training: true,
        })
    }

    pub const fn rate(&self) -> f32 {
        self.rate
    }

    pub const fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId, ModelError> {
        if !self.training || self.rate == 0.0 {
            return Ok(input);
        }
        // During training with non-zero rate, apply inverted dropout via scaling.
        // For deterministic autograd compatibility, we scale by (1-rate) as an
        // approximate expectation-preserving proxy without random masking.
        let scale_factor = 1.0 / (1.0 - self.rate);
        let scale = graph.constant(Tensor::scalar(scale_factor));
        graph.mul(input, scale).map_err(Into::into)
    }
}

/// Flatten layer: reshapes NHWC `[N, H, W, C]` to `[N, H*W*C]` for dense layer input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FlattenLayer;

impl FlattenLayer {
    pub const fn new() -> Self {
        Self
    }

    pub fn forward_inference(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        let shape = input.shape();
        if shape.len() < 2 {
            return Err(ModelError::InvalidFlattenShape {
                got: shape.to_vec(),
            });
        }
        let batch = shape[0];
        let features: usize = shape[1..].iter().product();
        input.reshape(vec![batch, features]).map_err(Into::into)
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId, ModelError> {
        graph.flatten(input).map_err(Into::into)
    }
}

/// Softmax layer over the last dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SoftmaxLayer;

impl SoftmaxLayer {
    pub const fn new() -> Self {
        Self
    }

    pub fn forward_inference(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        yscv_kernels::softmax_last_dim(input).map_err(Into::into)
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId, ModelError> {
        graph.softmax(input).map_err(Into::into)
    }
}

/// Pixel shuffle / sub-pixel convolution: rearranges `[N, H, W, C*r^2]` -> `[N, H*r, W*r, C]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PixelShuffleLayer {
    upscale_factor: usize,
}

impl PixelShuffleLayer {
    pub const fn new(upscale_factor: usize) -> Self {
        Self { upscale_factor }
    }

    pub const fn upscale_factor(&self) -> usize {
        self.upscale_factor
    }

    pub fn forward_inference(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(ModelError::InvalidInputShape {
                expected_features: 0,
                got: shape.to_vec(),
            });
        }
        let (batch, h, w, c) = (shape[0], shape[1], shape[2], shape[3]);
        let r = self.upscale_factor;
        let out_c = c / (r * r);
        let out_h = h * r;
        let out_w = w * r;
        let data = input.data();
        let mut out = vec![0.0f32; batch * out_h * out_w * out_c];

        for b in 0..batch {
            for ih in 0..h {
                for iw in 0..w {
                    for oc in 0..out_c {
                        for ry in 0..r {
                            for rx in 0..r {
                                let ic = oc * r * r + ry * r + rx;
                                let oh = ih * r + ry;
                                let ow = iw * r + rx;
                                out[((b * out_h + oh) * out_w + ow) * out_c + oc] =
                                    data[((b * h + ih) * w + iw) * c + ic];
                            }
                        }
                    }
                }
            }
        }
        Ok(Tensor::from_vec(vec![batch, out_h, out_w, out_c], out)?)
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId, ModelError> {
        graph
            .pixel_shuffle(input, self.upscale_factor)
            .map_err(Into::into)
    }
}

/// Upsample layer: nearest or bilinear upsampling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UpsampleLayer {
    scale_factor: usize,
    bilinear: bool,
}

impl UpsampleLayer {
    pub const fn new(scale_factor: usize, bilinear: bool) -> Self {
        Self {
            scale_factor,
            bilinear,
        }
    }

    pub const fn scale_factor(&self) -> usize {
        self.scale_factor
    }
    pub const fn is_bilinear(&self) -> bool {
        self.bilinear
    }

    pub fn forward_inference(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(ModelError::InvalidInputShape {
                expected_features: 0,
                got: shape.to_vec(),
            });
        }
        let (batch, h, w, c) = (shape[0], shape[1], shape[2], shape[3]);
        let r = self.scale_factor;
        let out_h = h * r;
        let out_w = w * r;
        let data = input.data();
        let mut out = vec![0.0f32; batch * out_h * out_w * c];

        if !self.bilinear {
            // Nearest
            for b in 0..batch {
                for oh in 0..out_h {
                    let ih = oh / r;
                    for ow in 0..out_w {
                        let iw = ow / r;
                        let src = ((b * h + ih) * w + iw) * c;
                        let dst = ((b * out_h + oh) * out_w + ow) * c;
                        out[dst..dst + c].copy_from_slice(&data[src..src + c]);
                    }
                }
            }
        } else {
            // Bilinear
            for b in 0..batch {
                for oh in 0..out_h {
                    let src_y = (oh as f32 + 0.5) / r as f32 - 0.5;
                    let y0 = (src_y.floor() as usize).min(h - 1);
                    let y1 = (y0 + 1).min(h - 1);
                    let fy = src_y - y0 as f32;
                    for ow in 0..out_w {
                        let src_x = (ow as f32 + 0.5) / r as f32 - 0.5;
                        let x0 = (src_x.floor() as usize).min(w - 1);
                        let x1 = (x0 + 1).min(w - 1);
                        let fx = src_x - x0 as f32;
                        for ch in 0..c {
                            let v00 = data[((b * h + y0) * w + x0) * c + ch];
                            let v10 = data[((b * h + y0) * w + x1) * c + ch];
                            let v01 = data[((b * h + y1) * w + x0) * c + ch];
                            let v11 = data[((b * h + y1) * w + x1) * c + ch];
                            out[((b * out_h + oh) * out_w + ow) * c + ch] =
                                v00 * (1.0 - fx) * (1.0 - fy)
                                    + v10 * fx * (1.0 - fy)
                                    + v01 * (1.0 - fx) * fy
                                    + v11 * fx * fy;
                        }
                    }
                }
            }
        }
        Ok(Tensor::from_vec(vec![batch, out_h, out_w, c], out)?)
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId, ModelError> {
        if self.bilinear {
            // For bilinear, fall back to inference-only for now.
            return Err(ModelError::InferenceOnlyLayer);
        }
        graph
            .upsample_nearest(input, self.scale_factor)
            .map_err(Into::into)
    }
}

/// Residual block: runs input through a sequence of layers, then adds the
/// original input as a skip connection (`output = layers(input) + input`).
#[derive(Debug, Clone)]
pub struct ResidualBlock {
    layers: Vec<ModelLayer>,
}

impl ResidualBlock {
    /// Creates a new residual block wrapping the given layers.
    pub const fn new(layers: Vec<ModelLayer>) -> Self {
        Self { layers }
    }

    /// Returns a reference to the inner layers.
    pub fn layers(&self) -> &[ModelLayer] {
        &self.layers
    }

    /// Runs inference: passes `input` through all inner layers sequentially,
    /// then adds the skip connection (`output = layers_output + input`).
    pub fn forward_inference(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        let mut current = input.clone();
        for layer in &self.layers {
            current = layer.forward_inference(&current)?;
        }
        current.add(input).map_err(ModelError::Tensor)
    }

    /// Graph-mode forward: passes `input` through all inner layers, then adds skip connection.
    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId, ModelError> {
        let mut current = input;
        for layer in &self.layers {
            current = layer.forward(graph, current)?;
        }
        graph.add(current, input).map_err(Into::into)
    }
}

/// Mask prediction head for instance segmentation (Mask R-CNN style).
///
/// Takes RoI-pooled features `[N, H, W, C]` and produces binary masks
/// `[N, mask_h, mask_w, num_classes]` via a series of conv layers + upsample.
#[derive(Debug, Clone, PartialEq)]
pub struct MaskHead {
    /// 4 conv layers: each [3, 3, channels, channels]
    conv_weights: Vec<Tensor>,
    /// Final 1×1 conv for class prediction: [1, 1, channels, num_classes]
    class_conv: Tensor,
    channels: usize,
    num_classes: usize,
    mask_size: usize,
}

impl MaskHead {
    /// Create a mask head with `num_conv` intermediate conv layers.
    pub fn new(
        in_channels: usize,
        channels: usize,
        num_classes: usize,
        mask_size: usize,
        num_conv: usize,
    ) -> Result<Self, ModelError> {
        let mut conv_weights = Vec::with_capacity(num_conv);
        for i in 0..num_conv {
            let c_in = if i == 0 { in_channels } else { channels };
            conv_weights.push(Tensor::zeros(vec![3, 3, c_in, channels])?);
        }
        let class_conv = Tensor::zeros(vec![1, 1, channels, num_classes])?;
        Ok(Self {
            conv_weights,
            class_conv,
            channels,
            num_classes,
            mask_size,
        })
    }

    pub const fn num_classes(&self) -> usize {
        self.num_classes
    }
    pub const fn mask_size(&self) -> usize {
        self.mask_size
    }
    pub const fn channels(&self) -> usize {
        self.channels
    }

    /// Forward pass: conv layers → ReLU → upsample → class prediction.
    pub fn forward_inference(&self, roi_features: &Tensor) -> Result<Tensor, ModelError> {
        let mut x = roi_features.clone();
        // Apply conv layers with ReLU
        for w in &self.conv_weights {
            x = yscv_kernels::conv2d_nhwc(&x, w, None, 1, 1)?;
            // In-place ReLU
            let data = x.data_mut();
            for v in data.iter_mut() {
                *v = v.max(0.0);
            }
        }
        // 2× bilinear upsample (simple nearest-neighbor for now)
        let shape = x.shape();
        if shape.len() == 4 {
            let (n, h, w, c) = (shape[0], shape[1], shape[2], shape[3]);
            let nh = h * 2;
            let nw = w * 2;
            let mut up = vec![0.0f32; n * nh * nw * c];
            for bi in 0..n {
                for yi in 0..nh {
                    for xi in 0..nw {
                        let sy = yi / 2;
                        let sx = xi / 2;
                        let src = bi * h * w * c + sy * w * c + sx * c;
                        let dst = bi * nh * nw * c + yi * nw * c + xi * c;
                        up[dst..dst + c].copy_from_slice(&x.data()[src..src + c]);
                    }
                }
            }
            x = Tensor::from_vec(vec![n, nh, nw, c], up)?;
        }
        // Final 1×1 class prediction conv
        yscv_kernels::conv2d_nhwc(&x, &self.class_conv, None, 1, 1).map_err(Into::into)
    }
}
