use yscv_autograd::{Graph, NodeId};
use yscv_tensor::Tensor;

use crate::ModelError;

/// 2D max-pooling layer (NHWC layout).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaxPool2dLayer {
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
}

impl MaxPool2dLayer {
    pub const fn new(
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Self, ModelError> {
        if kernel_h == 0 || kernel_w == 0 {
            return Err(ModelError::InvalidPoolKernel { kernel_h, kernel_w });
        }
        if stride_h == 0 || stride_w == 0 {
            return Err(ModelError::InvalidPoolStride { stride_h, stride_w });
        }
        Ok(Self {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
        })
    }

    pub const fn kernel_h(&self) -> usize {
        self.kernel_h
    }
    pub const fn kernel_w(&self) -> usize {
        self.kernel_w
    }
    pub const fn stride_h(&self) -> usize {
        self.stride_h
    }
    pub const fn stride_w(&self) -> usize {
        self.stride_w
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId, ModelError> {
        graph
            .max_pool2d_nhwc(
                input,
                self.kernel_h,
                self.kernel_w,
                self.stride_h,
                self.stride_w,
            )
            .map_err(Into::into)
    }

    pub fn forward_inference(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        yscv_kernels::max_pool2d_nhwc(
            input,
            self.kernel_h,
            self.kernel_w,
            self.stride_h,
            self.stride_w,
        )
        .map_err(Into::into)
    }
}

/// 2D average-pooling layer (NHWC layout).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AvgPool2dLayer {
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
}

impl AvgPool2dLayer {
    pub const fn new(
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Self, ModelError> {
        if kernel_h == 0 || kernel_w == 0 {
            return Err(ModelError::InvalidPoolKernel { kernel_h, kernel_w });
        }
        if stride_h == 0 || stride_w == 0 {
            return Err(ModelError::InvalidPoolStride { stride_h, stride_w });
        }
        Ok(Self {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
        })
    }

    pub const fn kernel_h(&self) -> usize {
        self.kernel_h
    }
    pub const fn kernel_w(&self) -> usize {
        self.kernel_w
    }
    pub const fn stride_h(&self) -> usize {
        self.stride_h
    }
    pub const fn stride_w(&self) -> usize {
        self.stride_w
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId, ModelError> {
        graph
            .avg_pool2d_nhwc(
                input,
                self.kernel_h,
                self.kernel_w,
                self.stride_h,
                self.stride_w,
            )
            .map_err(Into::into)
    }

    pub fn forward_inference(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        yscv_kernels::avg_pool2d_nhwc(
            input,
            self.kernel_h,
            self.kernel_w,
            self.stride_h,
            self.stride_w,
        )
        .map_err(Into::into)
    }
}

/// Global average pooling: NHWC `[N,H,W,C]` -> `[N,1,1,C]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GlobalAvgPool2dLayer;

impl GlobalAvgPool2dLayer {
    pub const fn new() -> Self {
        Self
    }

    pub fn forward_inference(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(ModelError::InvalidFlattenShape {
                got: shape.to_vec(),
            });
        }
        let (n, h, w, c) = (shape[0], shape[1], shape[2], shape[3]);
        let hw = (h * w) as f32;
        let data = input.data();
        let mut out = vec![0.0f32; n * c];
        for batch in 0..n {
            for ch in 0..c {
                let mut sum = 0.0f32;
                for y in 0..h {
                    for x in 0..w {
                        sum += data[((batch * h + y) * w + x) * c + ch];
                    }
                }
                out[batch * c + ch] = sum / hw;
            }
        }
        Tensor::from_vec(vec![n, 1, 1, c], out).map_err(Into::into)
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId, ModelError> {
        let shape = graph.value(input)?.shape().to_vec();
        if shape.len() != 4 {
            return Err(ModelError::InvalidFlattenShape { got: shape });
        }
        let (h, w) = (shape[1], shape[2]);
        graph.avg_pool2d_nhwc(input, h, w, 1, 1).map_err(Into::into)
    }
}

/// Adaptive average pooling: output a fixed spatial size regardless of input size.
///
/// NHWC layout: `[batch, H, W, C]` -> `[batch, out_h, out_w, C]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdaptiveAvgPool2dLayer {
    out_h: usize,
    out_w: usize,
}

impl AdaptiveAvgPool2dLayer {
    pub const fn new(out_h: usize, out_w: usize) -> Self {
        Self { out_h, out_w }
    }

    pub const fn output_h(&self) -> usize {
        self.out_h
    }
    pub const fn output_w(&self) -> usize {
        self.out_w
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId, ModelError> {
        graph
            .adaptive_avg_pool2d_nhwc(input, self.out_h, self.out_w)
            .map_err(Into::into)
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
        let data = input.data();
        let mut out = vec![0.0f32; batch * self.out_h * self.out_w * c];

        for b in 0..batch {
            for oh in 0..self.out_h {
                let h_start = oh * h / self.out_h;
                let h_end = ((oh + 1) * h / self.out_h).max(h_start + 1);
                for ow in 0..self.out_w {
                    let w_start = ow * w / self.out_w;
                    let w_end = ((ow + 1) * w / self.out_w).max(w_start + 1);
                    let count = (h_end - h_start) * (w_end - w_start);
                    for ch in 0..c {
                        let mut sum = 0.0f32;
                        for ih in h_start..h_end {
                            for iw in w_start..w_end {
                                sum += data[((b * h + ih) * w + iw) * c + ch];
                            }
                        }
                        out[((b * self.out_h + oh) * self.out_w + ow) * c + ch] =
                            sum / count as f32;
                    }
                }
            }
        }
        Ok(Tensor::from_vec(
            vec![batch, self.out_h, self.out_w, c],
            out,
        )?)
    }
}

/// Adaptive max pooling: output a fixed spatial size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdaptiveMaxPool2dLayer {
    out_h: usize,
    out_w: usize,
}

impl AdaptiveMaxPool2dLayer {
    pub const fn new(out_h: usize, out_w: usize) -> Self {
        Self { out_h, out_w }
    }

    pub const fn output_h(&self) -> usize {
        self.out_h
    }
    pub const fn output_w(&self) -> usize {
        self.out_w
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId, ModelError> {
        graph
            .adaptive_max_pool2d_nhwc(input, self.out_h, self.out_w)
            .map_err(Into::into)
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
        let data = input.data();
        let mut out = vec![f32::NEG_INFINITY; batch * self.out_h * self.out_w * c];

        for b in 0..batch {
            for oh in 0..self.out_h {
                let h_start = oh * h / self.out_h;
                let h_end = ((oh + 1) * h / self.out_h).max(h_start + 1);
                for ow in 0..self.out_w {
                    let w_start = ow * w / self.out_w;
                    let w_end = ((ow + 1) * w / self.out_w).max(w_start + 1);
                    for ch in 0..c {
                        let mut max_v = f32::NEG_INFINITY;
                        for ih in h_start..h_end {
                            for iw in w_start..w_end {
                                let v = data[((b * h + ih) * w + iw) * c + ch];
                                if v > max_v {
                                    max_v = v;
                                }
                            }
                        }
                        out[((b * self.out_h + oh) * self.out_w + ow) * c + ch] = max_v;
                    }
                }
            }
        }
        Ok(Tensor::from_vec(
            vec![batch, self.out_h, self.out_w, c],
            out,
        )?)
    }
}
