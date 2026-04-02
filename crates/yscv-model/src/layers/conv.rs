use yscv_autograd::{Graph, NodeId};
use yscv_tensor::Tensor;

use crate::ModelError;

/// 2D convolution layer (NHWC layout).
///
/// Supports both inference-mode (raw tensor) and graph-mode (autograd training).
/// Kernel shape: `[KH, KW, C_in, C_out]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Conv2dLayer {
    in_channels: usize,
    out_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    weight: Tensor,
    bias: Option<Tensor>,
    weight_node: Option<NodeId>,
    bias_node: Option<NodeId>,
}

impl Conv2dLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        weight: Tensor,
        bias: Option<Tensor>,
    ) -> Result<Self, ModelError> {
        let expected_weight = vec![kernel_h, kernel_w, in_channels, out_channels];
        if weight.shape() != expected_weight {
            return Err(ModelError::InvalidParameterShape {
                parameter: "conv2d weight",
                expected: expected_weight,
                got: weight.shape().to_vec(),
            });
        }
        if let Some(ref b) = bias
            && b.shape() != [out_channels]
        {
            return Err(ModelError::InvalidParameterShape {
                parameter: "conv2d bias",
                expected: vec![out_channels],
                got: b.shape().to_vec(),
            });
        }
        if stride_h == 0 || stride_w == 0 {
            return Err(ModelError::InvalidConv2dStride { stride_h, stride_w });
        }
        Ok(Self {
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            weight,
            bias,
            weight_node: None,
            bias_node: None,
        })
    }

    /// Creates a conv2d layer and registers its parameters as graph variables.
    #[allow(clippy::too_many_arguments)]
    pub fn new_in_graph(
        graph: &mut Graph,
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        weight: Tensor,
        bias: Option<Tensor>,
    ) -> Result<Self, ModelError> {
        let mut layer = Self::new(
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            weight,
            bias,
        )?;
        layer.register_params(graph);
        Ok(layer)
    }

    pub fn zero_init(
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        use_bias: bool,
    ) -> Result<Self, ModelError> {
        let weight = Tensor::zeros(vec![kernel_h, kernel_w, in_channels, out_channels])?;
        let bias = if use_bias {
            Some(Tensor::zeros(vec![out_channels])?)
        } else {
            None
        };
        Self::new(
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            weight,
            bias,
        )
    }

    /// Registers weight/bias tensors as graph variables for autograd training.
    pub fn register_params(&mut self, graph: &mut Graph) {
        self.weight_node = Some(graph.variable(self.weight.clone()));
        self.bias_node = self.bias.as_ref().map(|b| graph.variable(b.clone()));
    }

    /// Synchronizes owned tensors from the graph (e.g. after optimizer step).
    pub fn sync_from_graph(&mut self, graph: &Graph) -> Result<(), ModelError> {
        if let Some(w_id) = self.weight_node {
            self.weight = graph.value(w_id)?.clone();
        }
        if let Some(b_id) = self.bias_node {
            self.bias = Some(graph.value(b_id)?.clone());
        }
        Ok(())
    }

    pub fn in_channels(&self) -> usize {
        self.in_channels
    }
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }
    pub fn kernel_h(&self) -> usize {
        self.kernel_h
    }
    pub fn kernel_w(&self) -> usize {
        self.kernel_w
    }
    pub fn stride_h(&self) -> usize {
        self.stride_h
    }
    pub fn stride_w(&self) -> usize {
        self.stride_w
    }
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
    pub fn weight_mut(&mut self) -> &mut Tensor {
        &mut self.weight
    }
    pub fn bias_mut(&mut self) -> Option<&mut Tensor> {
        self.bias.as_mut()
    }
    pub fn weight_node(&self) -> Option<NodeId> {
        self.weight_node
    }
    pub fn bias_node(&self) -> Option<NodeId> {
        self.bias_node
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId, ModelError> {
        let w_id = self
            .weight_node
            .ok_or(ModelError::ParamsNotRegistered { layer: "Conv2d" })?;
        graph
            .conv2d_nhwc(input, w_id, self.bias_node, self.stride_h, self.stride_w)
            .map_err(Into::into)
    }

    pub fn forward_inference(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        yscv_kernels::conv2d_nhwc(
            input,
            &self.weight,
            self.bias.as_ref(),
            self.stride_h,
            self.stride_w,
        )
        .map_err(Into::into)
    }
}

/// Depthwise 2D convolution layer (NHWC layout).
///
/// Each input channel is convolved with its own filter.
/// Kernel shape: `[KH, KW, C, 1]`.
#[derive(Debug, Clone, PartialEq)]
pub struct DepthwiseConv2dLayer {
    channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    weight: Tensor,
    bias: Option<Tensor>,
    weight_node: Option<NodeId>,
    bias_node: Option<NodeId>,
}

impl DepthwiseConv2dLayer {
    pub fn new(
        channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        weight: Tensor,
        bias: Option<Tensor>,
    ) -> Result<Self, ModelError> {
        let expected_weight = vec![kernel_h, kernel_w, channels, 1];
        if weight.shape() != expected_weight {
            return Err(ModelError::InvalidParameterShape {
                parameter: "depthwise_conv2d weight",
                expected: expected_weight,
                got: weight.shape().to_vec(),
            });
        }
        if let Some(ref b) = bias
            && b.shape() != [channels]
        {
            return Err(ModelError::InvalidParameterShape {
                parameter: "depthwise_conv2d bias",
                expected: vec![channels],
                got: b.shape().to_vec(),
            });
        }
        if stride_h == 0 || stride_w == 0 {
            return Err(ModelError::InvalidConv2dStride { stride_h, stride_w });
        }
        Ok(Self {
            channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            weight,
            bias,
            weight_node: None,
            bias_node: None,
        })
    }

    pub fn zero_init(
        channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        use_bias: bool,
    ) -> Result<Self, ModelError> {
        let weight = Tensor::zeros(vec![kernel_h, kernel_w, channels, 1])?;
        let bias = if use_bias {
            Some(Tensor::zeros(vec![channels])?)
        } else {
            None
        };
        Self::new(
            channels, kernel_h, kernel_w, stride_h, stride_w, weight, bias,
        )
    }

    pub fn register_params(&mut self, graph: &mut Graph) {
        self.weight_node = Some(graph.variable(self.weight.clone()));
        self.bias_node = self.bias.as_ref().map(|b| graph.variable(b.clone()));
    }

    pub fn sync_from_graph(&mut self, graph: &Graph) -> Result<(), ModelError> {
        if let Some(w_id) = self.weight_node {
            self.weight = graph.value(w_id)?.clone();
        }
        if let Some(b_id) = self.bias_node {
            self.bias = Some(graph.value(b_id)?.clone());
        }
        Ok(())
    }

    pub fn channels(&self) -> usize {
        self.channels
    }
    pub fn kernel_h(&self) -> usize {
        self.kernel_h
    }
    pub fn kernel_w(&self) -> usize {
        self.kernel_w
    }
    pub fn stride_h(&self) -> usize {
        self.stride_h
    }
    pub fn stride_w(&self) -> usize {
        self.stride_w
    }
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
    pub fn weight_mut(&mut self) -> &mut Tensor {
        &mut self.weight
    }
    pub fn bias_mut(&mut self) -> Option<&mut Tensor> {
        self.bias.as_mut()
    }
    pub fn weight_node(&self) -> Option<NodeId> {
        self.weight_node
    }
    pub fn bias_node(&self) -> Option<NodeId> {
        self.bias_node
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId, ModelError> {
        let w_id = self.weight_node.ok_or(ModelError::ParamsNotRegistered {
            layer: "DepthwiseConv2d",
        })?;
        graph
            .depthwise_conv2d_nhwc(input, w_id, self.bias_node, self.stride_h, self.stride_w)
            .map_err(Into::into)
    }

    pub fn forward_inference(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        yscv_kernels::depthwise_conv2d_nhwc(
            input,
            &self.weight,
            self.bias.as_ref(),
            self.stride_h,
            self.stride_w,
        )
        .map_err(Into::into)
    }
}

/// Separable 2D convolution layer (NHWC layout).
///
/// Composed of a depthwise convolution followed by a pointwise (1x1) convolution.
/// Depthwise kernel shape: `[KH, KW, C_in, 1]`, pointwise kernel: `[1, 1, C_in, C_out]`.
#[derive(Debug, Clone, PartialEq)]
pub struct SeparableConv2dLayer {
    depthwise: DepthwiseConv2dLayer,
    pointwise: Conv2dLayer,
}

impl SeparableConv2dLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        depthwise_weight: Tensor,
        pointwise_weight: Tensor,
        bias: Option<Tensor>,
    ) -> Result<Self, ModelError> {
        let depthwise = DepthwiseConv2dLayer::new(
            in_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            depthwise_weight,
            None,
        )?;
        let pointwise = Conv2dLayer::new(
            in_channels,
            out_channels,
            1,
            1,
            1,
            1,
            pointwise_weight,
            bias,
        )?;
        Ok(Self {
            depthwise,
            pointwise,
        })
    }

    pub fn zero_init(
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        use_bias: bool,
    ) -> Result<Self, ModelError> {
        let depthwise = DepthwiseConv2dLayer::zero_init(
            in_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            false,
        )?;
        let pointwise = Conv2dLayer::zero_init(in_channels, out_channels, 1, 1, 1, 1, use_bias)?;
        Ok(Self {
            depthwise,
            pointwise,
        })
    }

    pub fn register_params(&mut self, graph: &mut Graph) {
        self.depthwise.register_params(graph);
        self.pointwise.register_params(graph);
    }

    pub fn sync_from_graph(&mut self, graph: &Graph) -> Result<(), ModelError> {
        self.depthwise.sync_from_graph(graph)?;
        self.pointwise.sync_from_graph(graph)?;
        Ok(())
    }

    pub fn in_channels(&self) -> usize {
        self.depthwise.channels()
    }
    pub fn out_channels(&self) -> usize {
        self.pointwise.out_channels()
    }
    pub fn kernel_h(&self) -> usize {
        self.depthwise.kernel_h()
    }
    pub fn kernel_w(&self) -> usize {
        self.depthwise.kernel_w()
    }
    pub fn stride_h(&self) -> usize {
        self.depthwise.stride_h()
    }
    pub fn stride_w(&self) -> usize {
        self.depthwise.stride_w()
    }
    pub fn depthwise(&self) -> &DepthwiseConv2dLayer {
        &self.depthwise
    }
    pub fn pointwise(&self) -> &Conv2dLayer {
        &self.pointwise
    }
    pub fn depthwise_mut(&mut self) -> &mut DepthwiseConv2dLayer {
        &mut self.depthwise
    }
    pub fn pointwise_mut(&mut self) -> &mut Conv2dLayer {
        &mut self.pointwise
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId, ModelError> {
        let dw_out = self.depthwise.forward(graph, input)?;
        self.pointwise.forward(graph, dw_out)
    }

    pub fn forward_inference(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        let dw_out = self.depthwise.forward_inference(input)?;
        self.pointwise.forward_inference(&dw_out)
    }
}

/// Deformable 2D convolution layer (NHWC layout).
///
/// Like standard conv2d but sampling positions are offset by learned offsets.
/// An internal offset convolution produces offsets from the input, then those
/// offsets are used to sample input with bilinear interpolation.
///
/// Weight shape: `[kH, kW, C_in, C_out]`.
/// Offset weight shape: `[kH, kW, C_in, kH*kW*2]` -- conv producing offsets.
#[derive(Debug, Clone, PartialEq)]
pub struct DeformableConv2dLayer {
    pub weight: Tensor,
    pub offset_weight: Tensor,
    pub bias: Option<Tensor>,
    pub stride: usize,
    pub padding: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    in_channels: usize,
    out_channels: usize,
    weight_node: Option<NodeId>,
    offset_weight_node: Option<NodeId>,
    bias_node: Option<NodeId>,
}

impl DeformableConv2dLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride: usize,
        padding: usize,
        weight: Tensor,
        offset_weight: Tensor,
        bias: Option<Tensor>,
    ) -> Result<Self, ModelError> {
        let expected_weight = vec![kernel_h, kernel_w, in_channels, out_channels];
        if weight.shape() != expected_weight {
            return Err(ModelError::InvalidParameterShape {
                parameter: "deformable_conv2d weight",
                expected: expected_weight,
                got: weight.shape().to_vec(),
            });
        }
        let offset_out = kernel_h * kernel_w * 2;
        let expected_offset_weight = vec![kernel_h, kernel_w, in_channels, offset_out];
        if offset_weight.shape() != expected_offset_weight {
            return Err(ModelError::InvalidParameterShape {
                parameter: "deformable_conv2d offset_weight",
                expected: expected_offset_weight,
                got: offset_weight.shape().to_vec(),
            });
        }
        if let Some(ref b) = bias
            && b.shape() != [out_channels]
        {
            return Err(ModelError::InvalidParameterShape {
                parameter: "deformable_conv2d bias",
                expected: vec![out_channels],
                got: b.shape().to_vec(),
            });
        }
        if stride == 0 {
            return Err(ModelError::InvalidConv2dStride {
                stride_h: stride,
                stride_w: stride,
            });
        }
        Ok(Self {
            weight,
            offset_weight,
            bias,
            stride,
            padding,
            kernel_h,
            kernel_w,
            in_channels,
            out_channels,
            weight_node: None,
            offset_weight_node: None,
            bias_node: None,
        })
    }

    pub fn zero_init(
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride: usize,
        padding: usize,
        use_bias: bool,
    ) -> Result<Self, ModelError> {
        let offset_out = kernel_h * kernel_w * 2;
        let weight = Tensor::zeros(vec![kernel_h, kernel_w, in_channels, out_channels])?;
        let offset_weight = Tensor::zeros(vec![kernel_h, kernel_w, in_channels, offset_out])?;
        let bias = if use_bias {
            Some(Tensor::zeros(vec![out_channels])?)
        } else {
            None
        };
        Self::new(
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            stride,
            padding,
            weight,
            offset_weight,
            bias,
        )
    }

    pub fn in_channels(&self) -> usize {
        self.in_channels
    }
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }
    pub fn kernel_h(&self) -> usize {
        self.kernel_h
    }
    pub fn kernel_w(&self) -> usize {
        self.kernel_w
    }
    pub fn stride(&self) -> usize {
        self.stride
    }
    pub fn padding(&self) -> usize {
        self.padding
    }
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
    pub fn offset_weight(&self) -> &Tensor {
        &self.offset_weight
    }
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    pub fn weight_node(&self) -> Option<NodeId> {
        self.weight_node
    }

    pub fn register_params(&mut self, graph: &mut Graph) {
        self.weight_node = Some(graph.variable(self.weight.clone()));
        self.offset_weight_node = Some(graph.variable(self.offset_weight.clone()));
        self.bias_node = self.bias.as_ref().map(|b| graph.variable(b.clone()));
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId, ModelError> {
        let w = self.weight_node.ok_or(ModelError::ParamsNotRegistered {
            layer: "DeformableConv2d",
        })?;
        let ow = self
            .offset_weight_node
            .ok_or(ModelError::ParamsNotRegistered {
                layer: "DeformableConv2d",
            })?;

        // Step 1: Compute offsets via standard conv2d of input with offset_weight.
        // If padding > 0, we need to pad the input first.
        let padded = if self.padding > 0 {
            let pad_per_dim = &[
                0,
                0,
                self.padding,
                self.padding,
                self.padding,
                self.padding,
                0,
                0,
            ];
            graph.pad(input, pad_per_dim, 0.0)?
        } else {
            input
        };
        let offsets = graph.conv2d_nhwc(padded, ow, None, self.stride, self.stride)?;

        // Step 2: Deformable conv using the computed offsets.
        graph
            .deformable_conv2d_nhwc(input, w, offsets, self.bias_node, self.stride, self.padding)
            .map_err(Into::into)
    }

    pub fn forward_inference(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        // Step 1: Compute offsets by convolving input with offset_weight (standard conv2d).
        // The offset conv uses the same kernel size, stride, and padding as the main conv
        // so that the offset map has the same spatial dimensions as the output.
        // Since conv2d_nhwc does not support padding, we pad the input manually.
        let padded = if self.padding > 0 {
            Self::pad_nhwc(input, self.padding)?
        } else {
            input.clone()
        };
        let offsets = yscv_kernels::conv2d_nhwc(
            &padded,
            &self.offset_weight,
            None,
            self.stride,
            self.stride,
        )?;

        // Step 2: Apply deformable conv with those offsets.
        Ok(yscv_kernels::deformable_conv2d_nhwc(
            input,
            &self.weight,
            &offsets,
            self.bias.as_ref(),
            self.stride,
            self.padding,
        )?)
    }

    /// Zero-pads an NHWC tensor by `pad` on each spatial side.
    fn pad_nhwc(input: &Tensor, pad: usize) -> Result<Tensor, ModelError> {
        let batch = input.shape()[0];
        let h = input.shape()[1];
        let w = input.shape()[2];
        let c = input.shape()[3];
        let new_h = h + 2 * pad;
        let new_w = w + 2 * pad;
        let mut data = vec![0.0f32; batch * new_h * new_w * c];
        let src = input.data();
        for n in 0..batch {
            for y in 0..h {
                let src_row = n * h * w * c + y * w * c;
                let dst_row = n * new_h * new_w * c + (y + pad) * new_w * c + pad * c;
                data[dst_row..dst_row + w * c].copy_from_slice(&src[src_row..src_row + w * c]);
            }
        }
        Tensor::from_vec(vec![batch, new_h, new_w, c], data).map_err(Into::into)
    }
}

/// 1D convolution layer (NLC layout: `[batch, length, channels]`).
///
/// Kernel shape: `[K, C_in, C_out]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Conv1dLayer {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    weight: Tensor,
    bias: Option<Tensor>,
    weight_node: Option<NodeId>,
    bias_node: Option<NodeId>,
}

impl Conv1dLayer {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        weight: Tensor,
        bias: Option<Tensor>,
    ) -> Result<Self, ModelError> {
        let expected = vec![kernel_size, in_channels, out_channels];
        if weight.shape() != expected {
            return Err(ModelError::InvalidParameterShape {
                parameter: "conv1d weight",
                expected,
                got: weight.shape().to_vec(),
            });
        }
        Ok(Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            weight,
            bias,
            weight_node: None,
            bias_node: None,
        })
    }

    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }
    pub fn kernel(&self) -> &Tensor {
        &self.weight
    }
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }
    pub fn stride(&self) -> usize {
        self.stride
    }
    pub fn weight_node(&self) -> Option<NodeId> {
        self.weight_node
    }
    pub fn bias_node(&self) -> Option<NodeId> {
        self.bias_node
    }

    pub fn register_params(&mut self, graph: &mut Graph) {
        self.weight_node = Some(graph.variable(self.weight.clone()));
        self.bias_node = self.bias.as_ref().map(|b| graph.variable(b.clone()));
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId, ModelError> {
        let w_id = self
            .weight_node
            .ok_or(ModelError::ParamsNotRegistered { layer: "Conv1d" })?;
        graph
            .conv1d_nlc(input, w_id, self.bias_node, self.stride)
            .map_err(Into::into)
    }

    pub fn forward_inference(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        let shape = input.shape();
        if shape.len() != 3 || shape[2] != self.in_channels {
            return Err(ModelError::InvalidInputShape {
                expected_features: self.in_channels,
                got: shape.to_vec(),
            });
        }
        let (batch, length, _) = (shape[0], shape[1], shape[2]);
        let out_len = (length - self.kernel_size) / self.stride + 1;
        let data = input.data();
        let w = self.weight.data();

        let mut out = vec![0.0f32; batch * out_len * self.out_channels];
        for b in 0..batch {
            for ol in 0..out_len {
                let start = ol * self.stride;
                for oc in 0..self.out_channels {
                    let mut sum = 0.0f32;
                    for k in 0..self.kernel_size {
                        for ic in 0..self.in_channels {
                            sum += data[(b * length + start + k) * self.in_channels + ic]
                                * w[(k * self.in_channels + ic) * self.out_channels + oc];
                        }
                    }
                    if let Some(ref bias) = self.bias {
                        sum += bias.data()[oc];
                    }
                    out[(b * out_len + ol) * self.out_channels + oc] = sum;
                }
            }
        }
        Ok(Tensor::from_vec(
            vec![batch, out_len, self.out_channels],
            out,
        )?)
    }
}

/// Transposed 2D convolution layer (NHWC layout).
///
/// Kernel shape: `[KH, KW, C_out, C_in]` (note: reversed from Conv2d).
#[derive(Debug, Clone, PartialEq)]
pub struct ConvTranspose2dLayer {
    in_channels: usize,
    out_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    weight: Tensor,
    bias: Option<Tensor>,
    weight_node: Option<NodeId>,
    bias_node: Option<NodeId>,
}

impl ConvTranspose2dLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        weight: Tensor,
        bias: Option<Tensor>,
    ) -> Result<Self, ModelError> {
        let expected = vec![kernel_h, kernel_w, out_channels, in_channels];
        if weight.shape() != expected {
            return Err(ModelError::InvalidParameterShape {
                parameter: "conv_transpose2d weight",
                expected,
                got: weight.shape().to_vec(),
            });
        }
        Ok(Self {
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            weight,
            bias,
            weight_node: None,
            bias_node: None,
        })
    }

    pub fn kernel(&self) -> &Tensor {
        &self.weight
    }
    pub fn stride(&self) -> usize {
        self.stride_h
    }
    pub fn weight_node(&self) -> Option<NodeId> {
        self.weight_node
    }
    pub fn bias_node(&self) -> Option<NodeId> {
        self.bias_node
    }

    pub fn register_params(&mut self, graph: &mut Graph) {
        self.weight_node = Some(graph.variable(self.weight.clone()));
        self.bias_node = self.bias.as_ref().map(|b| graph.variable(b.clone()));
    }

    pub fn sync_from_graph(&mut self, graph: &Graph) -> Result<(), ModelError> {
        if let Some(w_id) = self.weight_node {
            self.weight = graph.value(w_id)?.clone();
        }
        if let Some(b_id) = self.bias_node {
            self.bias = Some(graph.value(b_id)?.clone());
        }
        Ok(())
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId, ModelError> {
        let w_id = self.weight_node.ok_or(ModelError::ParamsNotRegistered {
            layer: "ConvTranspose2d",
        })?;
        graph
            .conv_transpose2d_nhwc(input, w_id, self.bias_node, self.stride_h, self.stride_w)
            .map_err(Into::into)
    }

    pub fn forward_inference(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        let shape = input.shape();
        if shape.len() != 4 || shape[3] != self.in_channels {
            return Err(ModelError::InvalidInputShape {
                expected_features: self.in_channels,
                got: shape.to_vec(),
            });
        }
        let (batch, h, w, _) = (shape[0], shape[1], shape[2], shape[3]);
        let out_h = (h - 1) * self.stride_h + self.kernel_h;
        let out_w = (w - 1) * self.stride_w + self.kernel_w;
        let data = input.data();
        let wt = self.weight.data();

        let ic = self.in_channels;
        let oc = self.out_channels;
        let mut out = vec![0.0f32; batch * out_h * out_w * oc];

        // Flatten the 7-deep loop: for each scatter position (b, ih, iw, kh, kw)
        // compute out[oh,ow,o] += dot(input[ih,iw,:], weight[kh,kw,o,:]) for each o.
        // Weight layout [KH,KW,C_out,C_in] makes the ic dimension contiguous per oc,
        // so each output channel is a dot product of two contiguous ic-length slices.
        for b in 0..batch {
            let in_batch = b * h * w * ic;
            let out_batch = b * out_h * out_w * oc;
            for ih in 0..h {
                for iw_idx in 0..w {
                    let in_base = in_batch + (ih * w + iw_idx) * ic;
                    let in_slice = &data[in_base..in_base + ic];
                    for kh in 0..self.kernel_h {
                        let oh = ih * self.stride_h + kh;
                        for kw in 0..self.kernel_w {
                            let ow = iw_idx * self.stride_w + kw;
                            let out_base = out_batch + (oh * out_w + ow) * oc;
                            let k_spatial = (kh * self.kernel_w + kw) * oc * ic;
                            for o in 0..oc {
                                let k_start = k_spatial + o * ic;
                                let k_slice = &wt[k_start..k_start + ic];
                                let mut acc = 0.0f32;
                                for c in 0..ic {
                                    acc += in_slice[c] * k_slice[c];
                                }
                                out[out_base + o] += acc;
                            }
                        }
                    }
                }
            }
        }

        if let Some(ref bias) = self.bias {
            let bd = bias.data();
            for i in 0..(batch * out_h * out_w) {
                for oc in 0..self.out_channels {
                    out[i * self.out_channels + oc] += bd[oc];
                }
            }
        }

        Ok(Tensor::from_vec(
            vec![batch, out_h, out_w, self.out_channels],
            out,
        )?)
    }
}

/// 3D convolution layer (BDHWC layout).
///
/// Wraps the `conv3d` kernel for volumetric data (video, medical imaging).
/// Kernel shape: `[KD, KH, KW, C_in, C_out]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Conv3dLayer {
    in_channels: usize,
    out_channels: usize,
    kernel_d: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    weight: Tensor,
    bias: Option<Tensor>,
    weight_node: Option<NodeId>,
    bias_node: Option<NodeId>,
}

impl Conv3dLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_d: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        weight: Tensor,
        bias: Option<Tensor>,
    ) -> Result<Self, ModelError> {
        let expected_weight = vec![kernel_d, kernel_h, kernel_w, in_channels, out_channels];
        if weight.shape() != expected_weight {
            return Err(ModelError::InvalidParameterShape {
                parameter: "conv3d weight",
                expected: expected_weight,
                got: weight.shape().to_vec(),
            });
        }
        if let Some(ref b) = bias
            && b.shape() != [out_channels]
        {
            return Err(ModelError::InvalidParameterShape {
                parameter: "conv3d bias",
                expected: vec![out_channels],
                got: b.shape().to_vec(),
            });
        }
        if stride.0 == 0 || stride.1 == 0 || stride.2 == 0 {
            return Err(ModelError::InvalidConv2dStride {
                stride_h: stride.1,
                stride_w: stride.2,
            });
        }
        Ok(Self {
            in_channels,
            out_channels,
            kernel_d,
            kernel_h,
            kernel_w,
            stride,
            padding,
            weight,
            bias,
            weight_node: None,
            bias_node: None,
        })
    }

    pub fn zero_init(
        in_channels: usize,
        out_channels: usize,
        kernel_d: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        use_bias: bool,
    ) -> Result<Self, ModelError> {
        let weight = Tensor::zeros(vec![
            kernel_d,
            kernel_h,
            kernel_w,
            in_channels,
            out_channels,
        ])?;
        let bias = if use_bias {
            Some(Tensor::zeros(vec![out_channels])?)
        } else {
            None
        };
        Self::new(
            in_channels,
            out_channels,
            kernel_d,
            kernel_h,
            kernel_w,
            stride,
            padding,
            weight,
            bias,
        )
    }

    pub fn register_params(&mut self, graph: &mut Graph) {
        self.weight_node = Some(graph.variable(self.weight.clone()));
        self.bias_node = self.bias.as_ref().map(|b| graph.variable(b.clone()));
    }

    pub fn sync_from_graph(&mut self, graph: &Graph) -> Result<(), ModelError> {
        if let Some(w_id) = self.weight_node {
            self.weight = graph.value(w_id)?.clone();
        }
        if let Some(b_id) = self.bias_node {
            self.bias = Some(graph.value(b_id)?.clone());
        }
        Ok(())
    }

    pub fn in_channels(&self) -> usize {
        self.in_channels
    }
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
    pub fn weight_node(&self) -> Option<NodeId> {
        self.weight_node
    }
    pub fn bias_node(&self) -> Option<NodeId> {
        self.bias_node
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId, ModelError> {
        let w_id = self
            .weight_node
            .ok_or(ModelError::ParamsNotRegistered { layer: "Conv3d" })?;
        graph
            .conv3d_ndhwc(
                input,
                w_id,
                self.bias_node,
                self.stride.0,
                self.stride.1,
                self.stride.2,
            )
            .map_err(Into::into)
    }

    pub fn forward_inference(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        let input_shape = input.shape();
        if input_shape.len() != 5 {
            return Err(ModelError::InvalidParameterShape {
                parameter: "conv3d input",
                expected: vec![0, 0, 0, 0, self.in_channels], // [B, D, H, W, C_in]
                got: input_shape.to_vec(),
            });
        }
        let kernel_shape = self.weight.shape();
        let (out_data, out_shape) = yscv_kernels::conv3d(
            input.data(),
            input_shape,
            self.weight.data(),
            kernel_shape,
            self.stride,
            self.padding,
        );
        let mut result = Tensor::from_vec(out_shape, out_data)?;
        if let Some(ref b) = self.bias {
            let c_out = self.out_channels;
            let data = result.data_mut();
            let bias_data = b.data();
            for pixel in data.chunks_mut(c_out) {
                for (v, &bv) in pixel.iter_mut().zip(bias_data.iter()) {
                    *v += bv;
                }
            }
        }
        Ok(result)
    }
}
