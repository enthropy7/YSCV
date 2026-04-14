use std::num::NonZeroUsize;

use yscv_kernels::{
    BackwardOps, BatchNorm2dParams, ThreadedCpuBackend, add as kernel_add, avg_pool2d_nhwc,
    batch_norm2d_nhwc, conv2d_nhwc, conv3d, depthwise_conv2d_nhwc, gelu as kernel_gelu, matmul_2d,
    mish as kernel_mish, mul as kernel_mul, relu, sigmoid as kernel_sigmoid, silu as kernel_silu,
    sub as kernel_sub,
};
use yscv_tensor::Tensor;

use super::error::AutogradError;
use super::node::{AuxData, Node, NodeId, Op};

/// Eager autograd graph with explicit backward pass.
pub struct Graph {
    pub(crate) nodes: Vec<Node>,
    pub(crate) backend: Option<Box<dyn BackwardOps>>,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    /// Creates an empty graph with automatic parallel backend.
    ///
    /// Uses all available CPU threads for parallel matmul, conv2d, softmax, etc.
    pub fn new() -> Self {
        let threads = std::thread::available_parallelism()
            .unwrap_or(NonZeroUsize::new(1).expect("1 is non-zero"));
        let backend = ThreadedCpuBackend::new(threads)
            .ok()
            .map(|b| Box::new(b) as Box<dyn BackwardOps>);
        Self {
            nodes: Vec::new(),
            backend,
        }
    }

    /// Creates an empty graph without a parallel backend (single-threaded).
    pub fn new_single_threaded() -> Self {
        Self {
            nodes: Vec::new(),
            backend: None,
        }
    }

    /// Set a compute backend for GPU-accelerated operations.
    /// When set, supported ops will dispatch through this backend.
    /// When None (default), ops use direct CPU kernel calls.
    pub fn set_backend(&mut self, backend: Box<dyn BackwardOps>) {
        self.backend = Some(backend);
    }

    /// Remove the backend, reverting to CPU kernel calls.
    pub fn clear_backend(&mut self) {
        self.backend = None;
    }

    /// Adds a trainable leaf node.
    pub fn variable(&mut self, value: Tensor) -> NodeId {
        self.push_node(value, true, Op::Leaf)
    }

    /// Adds a non-trainable leaf node.
    pub fn constant(&mut self, value: Tensor) -> NodeId {
        self.push_node(value, false, Op::Leaf)
    }

    /// Returns immutable node value.
    pub fn value(&self, node: NodeId) -> Result<&Tensor, AutogradError> {
        Ok(&self.node(node)?.value)
    }

    /// Returns mutable node value.
    pub fn value_mut(&mut self, node: NodeId) -> Result<&mut Tensor, AutogradError> {
        Ok(&mut self.node_mut(node)?.value)
    }

    /// Returns whether node is trainable.
    pub fn requires_grad(&self, node: NodeId) -> Result<bool, AutogradError> {
        Ok(self.node(node)?.requires_grad)
    }

    /// Returns immutable gradient if already computed.
    pub fn grad(&self, node: NodeId) -> Result<Option<&Tensor>, AutogradError> {
        Ok(self.node(node)?.grad.as_ref())
    }

    /// Returns mutable gradient if already computed.
    pub fn grad_mut(&mut self, node: NodeId) -> Result<Option<&mut Tensor>, AutogradError> {
        Ok(self.node_mut(node)?.grad.as_mut())
    }

    /// Sets the gradient for a node, replacing any existing gradient.
    pub fn set_grad(&mut self, node: NodeId, grad: Tensor) -> Result<(), AutogradError> {
        self.node_mut(node)?.grad = Some(grad);
        Ok(())
    }

    /// Returns current node count in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Clears gradients for all nodes.
    pub fn zero_grads(&mut self) {
        for node in &mut self.nodes {
            node.grad = None;
        }
    }

    /// Truncates graph to a given node count.
    pub fn truncate(&mut self, keep_nodes: usize) -> Result<(), AutogradError> {
        if keep_nodes > self.nodes.len() {
            return Err(AutogradError::InvalidTruncate {
                requested: keep_nodes,
                available: self.nodes.len(),
            });
        }
        self.nodes.truncate(keep_nodes);
        Ok(())
    }

    /// Adds two nodes with broadcasting support.
    pub fn add(&mut self, left: NodeId, right: NodeId) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let lv = &self.nodes[left.0].value;
            let rv = &self.nodes[right.0].value;
            let result = if let Some(ref backend) = self.backend {
                backend.add(lv, rv)?
            } else {
                kernel_add(lv, rv)?
            };
            (
                result,
                self.nodes[left.0].requires_grad || self.nodes[right.0].requires_grad,
            )
        };
        Ok(self.push_node(value, requires_grad, Op::Add(left, right)))
    }

    /// Subtracts two nodes with broadcasting support.
    pub fn sub(&mut self, left: NodeId, right: NodeId) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let lv = &self.nodes[left.0].value;
            let rv = &self.nodes[right.0].value;
            let result = if let Some(ref backend) = self.backend {
                backend.sub(lv, rv)?
            } else {
                kernel_sub(lv, rv)?
            };
            (
                result,
                self.nodes[left.0].requires_grad || self.nodes[right.0].requires_grad,
            )
        };
        Ok(self.push_node(value, requires_grad, Op::Sub(left, right)))
    }

    /// Multiplies two nodes elementwise with broadcasting support.
    pub fn mul(&mut self, left: NodeId, right: NodeId) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let lv = &self.nodes[left.0].value;
            let rv = &self.nodes[right.0].value;
            let result = if let Some(ref backend) = self.backend {
                backend.mul(lv, rv)?
            } else {
                kernel_mul(lv, rv)?
            };
            (
                result,
                self.nodes[left.0].requires_grad || self.nodes[right.0].requires_grad,
            )
        };
        Ok(self.push_node(value, requires_grad, Op::Mul(left, right)))
    }

    /// Applies ReLU activation to one node.
    pub fn relu(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let v = &self.nodes[input.0].value;
            let result = if let Some(ref backend) = self.backend {
                backend.relu(v)
            } else {
                relu(v)
            };
            (result, self.nodes[input.0].requires_grad)
        };
        Ok(self.push_node(value, requires_grad, Op::Relu(input)))
    }

    /// Performs rank-2 matrix multiplication.
    pub fn matmul_2d(&mut self, left: NodeId, right: NodeId) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let lv = &self.nodes[left.0].value;
            let rv = &self.nodes[right.0].value;
            let result = if let Some(ref backend) = self.backend {
                backend.matmul_2d(lv, rv)?
            } else {
                matmul_2d(lv, rv)?
            };
            (
                result,
                self.nodes[left.0].requires_grad || self.nodes[right.0].requires_grad,
            )
        };
        Ok(self.push_node(value, requires_grad, Op::MatMul2D(left, right)))
    }

    /// Divides two nodes elementwise with broadcasting support.
    pub fn div(&mut self, left: NodeId, right: NodeId) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let lv = &self.nodes[left.0].value;
            let rv = &self.nodes[right.0].value;
            (
                lv.div(rv)?,
                self.nodes[left.0].requires_grad || self.nodes[right.0].requires_grad,
            )
        };
        Ok(self.push_node(value, requires_grad, Op::Div(left, right)))
    }

    /// Applies element-wise negation.
    pub fn neg(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let v = &self.nodes[input.0].value;
            (v.neg(), self.nodes[input.0].requires_grad)
        };
        Ok(self.push_node(value, requires_grad, Op::Neg(input)))
    }

    /// Applies element-wise natural exponential.
    pub fn exp(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let v = &self.nodes[input.0].value;
            (v.exp(), self.nodes[input.0].requires_grad)
        };
        Ok(self.push_node(value, requires_grad, Op::Exp(input)))
    }

    /// Applies element-wise natural logarithm.
    pub fn log(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let v = &self.nodes[input.0].value;
            (v.ln(), self.nodes[input.0].requires_grad)
        };
        Ok(self.push_node(value, requires_grad, Op::Log(input)))
    }

    /// Applies element-wise square root.
    pub fn sqrt(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let v = &self.nodes[input.0].value;
            (v.sqrt(), self.nodes[input.0].requires_grad)
        };
        Ok(self.push_node(value, requires_grad, Op::Sqrt(input)))
    }

    /// Applies element-wise sigmoid activation: `1 / (1 + exp(-x))`.
    pub fn sigmoid(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let v = &self.nodes[input.0].value;
            let result = if let Some(ref backend) = self.backend {
                backend.sigmoid(v)
            } else {
                kernel_sigmoid(v)
            };
            (result, self.nodes[input.0].requires_grad)
        };
        Ok(self.push_node(value, requires_grad, Op::Sigmoid(input)))
    }

    /// Applies element-wise GELU activation (fast approximation): `x * sigmoid(1.702 * x)`.
    pub fn gelu(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let v = &self.nodes[input.0].value;
            (kernel_gelu(v), self.nodes[input.0].requires_grad)
        };
        Ok(self.push_node(value, requires_grad, Op::Gelu(input)))
    }

    /// Applies element-wise SiLU (Swish) activation: `x * sigmoid(x)`.
    pub fn silu(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let v = &self.nodes[input.0].value;
            (kernel_silu(v), self.nodes[input.0].requires_grad)
        };
        Ok(self.push_node(value, requires_grad, Op::Silu(input)))
    }

    /// Applies element-wise Mish activation: `x * tanh(softplus(x))`.
    pub fn mish(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let v = &self.nodes[input.0].value;
            (kernel_mish(v), self.nodes[input.0].requires_grad)
        };
        Ok(self.push_node(value, requires_grad, Op::Mish(input)))
    }

    /// Applies element-wise hyperbolic tangent.
    pub fn tanh(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let v = &self.nodes[input.0].value;
            let data: Vec<f32> = v.data().iter().map(|&x| x.tanh()).collect();
            (
                Tensor::from_vec(v.shape().to_vec(), data)?,
                self.nodes[input.0].requires_grad,
            )
        };
        Ok(self.push_node(value, requires_grad, Op::Tanh(input)))
    }

    /// Applies element-wise absolute value.
    pub fn abs(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let v = &self.nodes[input.0].value;
            let data: Vec<f32> = v.data().iter().map(|&x| x.abs()).collect();
            (
                Tensor::from_vec(v.shape().to_vec(), data)?,
                self.nodes[input.0].requires_grad,
            )
        };
        Ok(self.push_node(value, requires_grad, Op::Abs(input)))
    }

    /// Applies element-wise power: `base ^ exponent`.
    pub fn pow(&mut self, base: NodeId, exponent: NodeId) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let bv = &self.nodes[base.0].value;
            let ev = &self.nodes[exponent.0].value;
            (
                bv.pow(ev)?,
                self.nodes[base.0].requires_grad || self.nodes[exponent.0].requires_grad,
            )
        };
        Ok(self.push_node(value, requires_grad, Op::Pow(base, exponent)))
    }

    /// Applies element-wise clamping to `[min_val, max_val]`.
    pub fn clamp(
        &mut self,
        input: NodeId,
        min_val: f32,
        max_val: f32,
    ) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let v = &self.nodes[input.0].value;
            (v.clamp(min_val, max_val), self.nodes[input.0].requires_grad)
        };
        Ok(self.push_node(
            value,
            requires_grad,
            Op::Clamp {
                input,
                min_bits: min_val.to_bits(),
                max_bits: max_val.to_bits(),
            },
        ))
    }

    /// Applies element-wise leaky ReLU: `max(0, x) + negative_slope * min(0, x)`.
    pub fn leaky_relu(
        &mut self,
        input: NodeId,
        negative_slope: f32,
    ) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let v = &self.nodes[input.0].value;
            let data: Vec<f32> = v
                .data()
                .iter()
                .map(|&x| if x >= 0.0 { x } else { negative_slope * x })
                .collect();
            (
                Tensor::from_vec(v.shape().to_vec(), data)?,
                self.nodes[input.0].requires_grad,
            )
        };
        Ok(self.push_node(
            value,
            requires_grad,
            Op::LeakyRelu {
                input,
                negative_slope: negative_slope.to_bits(),
            },
        ))
    }

    /// Applies softmax along the last dimension.
    pub fn softmax(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let v = &self.nodes[input.0].value;
            let result = if let Some(ref backend) = self.backend {
                backend.softmax_last_dim(v)?
            } else {
                softmax_last_dim(v)
            };
            (result, self.nodes[input.0].requires_grad)
        };
        Ok(self.push_node(value, requires_grad, Op::Softmax(input)))
    }

    /// Applies log-softmax along the last dimension.
    pub fn log_softmax(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let v = &self.nodes[input.0].value;
            let sm = softmax_last_dim(v);
            let data: Vec<f32> = sm.data().iter().map(|&x| x.max(1e-12).ln()).collect();
            (
                Tensor::from_vec(sm.shape().to_vec(), data)?,
                self.nodes[input.0].requires_grad,
            )
        };
        Ok(self.push_node(value, requires_grad, Op::LogSoftmax(input)))
    }

    /// 2D matrix transpose in graph (for backward).
    pub fn transpose_2d(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let v = &self.nodes[input.0].value;
            (v.transpose_2d()?, self.nodes[input.0].requires_grad)
        };
        Ok(self.push_node(value, requires_grad, Op::Transpose2D(input)))
    }

    /// Reshape in graph (preserves backward path).
    pub fn reshape(
        &mut self,
        input: NodeId,
        new_shape: Vec<usize>,
    ) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let v = &self.nodes[input.0].value;
            (v.reshape(new_shape)?, self.nodes[input.0].requires_grad)
        };
        Ok(self.push_node(value, requires_grad, Op::ReshapeView { input }))
    }

    /// Unsqueeze in graph (preserves backward path).
    pub fn unsqueeze(&mut self, input: NodeId, axis: usize) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let v = &self.nodes[input.0].value;
            (v.unsqueeze(axis)?, self.nodes[input.0].requires_grad)
        };
        Ok(self.push_node(
            value,
            requires_grad,
            Op::UnsqueezeView {
                input,
                axis: axis as u16,
            },
        ))
    }

    /// Squeeze in graph (preserves backward path).
    pub fn squeeze(&mut self, input: NodeId, axis: usize) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let v = &self.nodes[input.0].value;
            (v.squeeze(axis)?, self.nodes[input.0].requires_grad)
        };
        Ok(self.push_node(
            value,
            requires_grad,
            Op::SqueezeView {
                input,
                axis: axis as u16,
            },
        ))
    }

    /// Concatenates multiple nodes along `axis`.
    pub fn cat(&mut self, inputs: &[NodeId], axis: usize) -> Result<NodeId, AutogradError> {
        if inputs.is_empty() {
            return Err(AutogradError::InvalidRankForOperation {
                op: "cat",
                expected: 1,
                got: 0,
            });
        }
        let tensors: Vec<&Tensor> = inputs.iter().map(|&id| &self.nodes[id.0].value).collect();
        let value = Tensor::cat(&tensors, axis)?;
        let requires_grad = inputs.iter().any(|&id| self.nodes[id.0].requires_grad);
        Ok(self.push_node(
            value,
            requires_grad,
            Op::Cat {
                inputs: inputs.to_vec(),
                axis: axis as u16,
            },
        ))
    }

    /// Selects a single index along `axis`, reducing that dimension.
    pub fn select(
        &mut self,
        input: NodeId,
        axis: usize,
        index: usize,
    ) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let v = &self.nodes[input.0].value;
            (v.select(axis, index)?, self.nodes[input.0].requires_grad)
        };
        Ok(self.push_node(
            value,
            requires_grad,
            Op::Select {
                input,
                axis: axis as u16,
                index: index as u32,
            },
        ))
    }

    /// Narrows (slices) a node along `axis` from `start` for `length` elements.
    pub fn narrow(
        &mut self,
        input: NodeId,
        axis: usize,
        start: usize,
        length: usize,
    ) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let v = &self.nodes[input.0].value;
            (
                v.narrow(axis, start, length)?,
                self.nodes[input.0].requires_grad,
            )
        };
        Ok(self.push_node(
            value,
            requires_grad,
            Op::Narrow {
                input,
                axis: axis as u16,
                start: start as u32,
                len: length as u32,
            },
        ))
    }

    /// Gathers elements along `axis` using an index tensor (from another node).
    ///
    /// For each position in the index tensor, retrieves the value from `input` at the index along `axis`.
    pub fn gather(
        &mut self,
        input: NodeId,
        axis: usize,
        index: NodeId,
    ) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let iv = &self.nodes[input.0].value;
            let idx = &self.nodes[index.0].value;
            (iv.gather(axis, idx)?, self.nodes[input.0].requires_grad)
        };
        Ok(self.push_node(
            value,
            requires_grad,
            Op::Gather {
                input,
                axis: axis as u16,
                index,
            },
        ))
    }

    /// Scatter-add operation: scatters `src` values into `input` at `index` positions along `axis`.
    ///
    /// Forward: `output = input.scatter_add(axis, index, src)`
    pub fn scatter_add(
        &mut self,
        input: NodeId,
        index: NodeId,
        src: NodeId,
        axis: usize,
    ) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let iv = &self.nodes[input.0].value;
            let idx = &self.nodes[index.0].value;
            let sv = &self.nodes[src.0].value;
            (
                iv.scatter_add(axis, idx, sv)?,
                self.nodes[input.0].requires_grad || self.nodes[src.0].requires_grad,
            )
        };
        Ok(self.push_node(
            value,
            requires_grad,
            Op::ScatterAdd {
                input,
                axis: axis as u16,
                index,
                src,
            },
        ))
    }

    /// Pads the tensor with a constant value along each dimension.
    ///
    /// `padding` is a flat array of `[before_0, after_0, before_1, after_1, ...]` pairs per dim.
    pub fn pad(
        &mut self,
        input: NodeId,
        padding: &[usize],
        value: f32,
    ) -> Result<NodeId, AutogradError> {
        let (result, requires_grad, pad_before, pad_after) = {
            let iv = &self.nodes[input.0].value;
            let shape = iv.shape();
            let rank = shape.len();
            if padding.len() != rank * 2 {
                return Err(AutogradError::InvalidRankForOperation {
                    op: "pad",
                    expected: rank * 2,
                    got: padding.len(),
                });
            }
            let mut new_shape = Vec::with_capacity(rank);
            let mut pad_before = Vec::with_capacity(rank);
            let mut pad_after = Vec::with_capacity(rank);
            for d in 0..rank {
                let pb = padding[d * 2];
                let pa = padding[d * 2 + 1];
                pad_before.push(pb as u32);
                pad_after.push(pa as u32);
                new_shape.push(shape[d] + pb + pa);
            }
            let total: usize = new_shape.iter().product();
            let mut out_data = vec![value; total];
            let data = iv.data();
            copy_region_nd(data, shape, &mut out_data, &new_shape, &pad_before);

            let result = Tensor::from_vec(new_shape, out_data)?;
            (
                result,
                self.nodes[input.0].requires_grad,
                pad_before,
                pad_after,
            )
        };
        Ok(self.push_node(
            result,
            requires_grad,
            Op::Pad {
                input,
                pad_before,
                pad_after,
            },
        ))
    }

    /// Repeats the tensor along each dimension.
    pub fn repeat(&mut self, input: NodeId, repeats: &[usize]) -> Result<NodeId, AutogradError> {
        let (result, requires_grad) = {
            let v = &self.nodes[input.0].value;
            (v.repeat(repeats)?, self.nodes[input.0].requires_grad)
        };
        let reps: Vec<u32> = repeats.iter().map(|&r| r as u32).collect();
        Ok(self.push_node(
            result,
            requires_grad,
            Op::Repeat {
                input,
                repeats: reps,
            },
        ))
    }

    /// Reduces one node to scalar sum.
    pub fn sum(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let v = &self.nodes[input.0].value;
            (Tensor::scalar(v.sum()), self.nodes[input.0].requires_grad)
        };
        Ok(self.push_node(value, requires_grad, Op::Sum(input)))
    }

    /// Reduces one node to scalar mean.
    pub fn mean(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let v = &self.nodes[input.0].value;
            (Tensor::scalar(v.mean()), self.nodes[input.0].requires_grad)
        };
        Ok(self.push_node(value, requires_grad, Op::Mean(input)))
    }

    /// NHWC 2-D convolution forward.
    /// `input` shape \[N,H,W,C_in\], `weight` shape \[KH,KW,C_in,C_out\],
    /// optional `bias` shape \[C_out\].
    pub fn conv2d_nhwc(
        &mut self,
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let iv = &self.nodes[input.0].value;
            let wv = &self.nodes[weight.0].value;
            let bv: Option<&Tensor> = bias.map(|b| &self.nodes[b.0].value);
            let rg = self.nodes[input.0].requires_grad
                || self.nodes[weight.0].requires_grad
                || bias.is_some_and(|b| self.nodes[b.0].requires_grad);
            let result = if let Some(ref backend) = self.backend {
                backend.conv2d_nhwc(iv, wv, bv, stride_h, stride_w)?
            } else {
                conv2d_nhwc(iv, wv, bv, stride_h, stride_w)?
            };
            (result, rg)
        };
        Ok(self.push_node(
            value,
            requires_grad,
            Op::Conv2dNhwc {
                input,
                weight,
                bias,
                stride_h: stride_h as u16,
                stride_w: stride_w as u16,
            },
        ))
    }

    /// NHWC max-pooling forward with argmax tracking for backward.
    pub fn max_pool2d_nhwc(
        &mut self,
        input: NodeId,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<NodeId, AutogradError> {
        let (value, requires_grad, indices) = {
            let iv = &self.nodes[input.0].value;
            let shape = iv.shape();
            if shape.len() != 4 {
                return Err(AutogradError::InvalidRankForOperation {
                    op: "max_pool2d_nhwc",
                    expected: 4,
                    got: shape.len(),
                });
            }
            let (n, ih, iw, c) = (shape[0], shape[1], shape[2], shape[3]);
            let oh = (ih - kernel_h) / stride_h + 1;
            let ow = (iw - kernel_w) / stride_w + 1;

            // Always compute argmax indices for backward support.
            let mut indices = vec![0usize; n * oh * ow * c];
            let in_data = iv.data();

            for batch in 0..n {
                for row in 0..oh {
                    for col in 0..ow {
                        for ch in 0..c {
                            let out_idx = ((batch * oh + row) * ow + col) * c + ch;
                            let mut best_val = f32::NEG_INFINITY;
                            let mut best_offset = 0usize;
                            for kh in 0..kernel_h {
                                for kw in 0..kernel_w {
                                    let ih_pos = row * stride_h + kh;
                                    let iw_pos = col * stride_w + kw;
                                    let in_idx = ((batch * ih + ih_pos) * iw + iw_pos) * c + ch;
                                    let v = in_data[in_idx];
                                    if v > best_val {
                                        best_val = v;
                                        best_offset = in_idx;
                                    }
                                }
                            }
                            indices[out_idx] = best_offset;
                        }
                    }
                }
            }

            let value = if let Some(ref backend) = self.backend {
                backend.max_pool2d_nhwc(iv, kernel_h, kernel_w, stride_h, stride_w)?
            } else {
                let mut out_data = vec![f32::NEG_INFINITY; n * oh * ow * c];
                for batch in 0..n {
                    for row in 0..oh {
                        for col in 0..ow {
                            for ch in 0..c {
                                let out_idx = ((batch * oh + row) * ow + col) * c + ch;
                                out_data[out_idx] = in_data[indices[out_idx]];
                            }
                        }
                    }
                }
                Tensor::from_vec(vec![n, oh, ow, c], out_data)?
            };
            (value, self.nodes[input.0].requires_grad, indices)
        };
        Ok(self.push_node_with_aux(
            value,
            requires_grad,
            Op::MaxPool2dNhwc {
                input,
                kernel_h: kernel_h as u16,
                kernel_w: kernel_w as u16,
                stride_h: stride_h as u16,
                stride_w: stride_w as u16,
            },
            AuxData::MaxPoolIndices(indices),
        ))
    }

    /// NHWC average-pooling forward.
    pub fn avg_pool2d_nhwc(
        &mut self,
        input: NodeId,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let v = &self.nodes[input.0].value;
            let result = if let Some(ref backend) = self.backend {
                backend.avg_pool2d_nhwc(v, kernel_h, kernel_w, stride_h, stride_w)?
            } else {
                avg_pool2d_nhwc(v, kernel_h, kernel_w, stride_h, stride_w)?
            };
            (result, self.nodes[input.0].requires_grad)
        };
        Ok(self.push_node(
            value,
            requires_grad,
            Op::AvgPool2dNhwc {
                input,
                kernel_h: kernel_h as u16,
                kernel_w: kernel_w as u16,
                stride_h: stride_h as u16,
                stride_w: stride_w as u16,
            },
        ))
    }

    /// NHWC batch-normalization forward (inference mode: uses running stats).
    /// `gamma`/`beta`/`running_mean`/`running_var` must be rank-1 of size `C`.
    pub fn batch_norm2d_nhwc(
        &mut self,
        input: NodeId,
        gamma: NodeId,
        beta: NodeId,
        running_mean: NodeId,
        running_var: NodeId,
        epsilon: f32,
    ) -> Result<NodeId, AutogradError> {
        let eps_bits = epsilon.to_bits();
        let (value, requires_grad, norm_tensor) = {
            let iv = &self.nodes[input.0].value;
            let gv = &self.nodes[gamma.0].value;
            let bv = &self.nodes[beta.0].value;
            let mv = &self.nodes[running_mean.0].value;
            let vv = &self.nodes[running_var.0].value;

            let shape = iv.shape();
            if shape.len() != 4 {
                return Err(AutogradError::InvalidRankForOperation {
                    op: "batch_norm2d_nhwc",
                    expected: 4,
                    got: shape.len(),
                });
            }
            // Always compute normalized tensor for backward support.
            let c = shape[3];
            let total = iv.len();
            let in_data = iv.data();
            let mean_data = mv.data();
            let var_data = vv.data();
            let mut normalized = vec![0.0f32; total];
            for i in 0..total {
                let ch = i % c;
                let inv_std = 1.0 / (var_data[ch] + epsilon).sqrt();
                normalized[i] = (in_data[i] - mean_data[ch]) * inv_std;
            }
            let norm_tensor = Tensor::from_vec(shape.to_vec(), normalized)?;

            let params = BatchNorm2dParams {
                gamma: gv,
                beta: bv,
                mean: mv,
                variance: vv,
                epsilon,
            };
            let value = if let Some(ref backend) = self.backend {
                backend.batch_norm2d_nhwc(iv, params)?
            } else {
                batch_norm2d_nhwc(iv, params)?
            };
            let rg = self.nodes[input.0].requires_grad
                || self.nodes[gamma.0].requires_grad
                || self.nodes[beta.0].requires_grad;
            (value, rg, norm_tensor)
        };
        Ok(self.push_node_with_aux(
            value,
            requires_grad,
            Op::BatchNorm2dNhwc {
                input,
                gamma,
                beta,
                running_mean,
                running_var,
                epsilon: eps_bits,
            },
            AuxData::BatchNormNormalized(norm_tensor),
        ))
    }

    /// Layer normalization over the last dimension.
    ///
    /// Input can be any rank; normalization is applied over the last axis.
    /// `gamma` and `beta` must have shape `[last_dim]`.
    pub fn layer_norm(
        &mut self,
        input: NodeId,
        gamma: NodeId,
        beta: NodeId,
        epsilon: f32,
    ) -> Result<NodeId, AutogradError> {
        let eps_bits = epsilon.to_bits();
        let (value, requires_grad, norm_tensor) = {
            let iv = &self.nodes[input.0].value;
            let gv = &self.nodes[gamma.0].value;
            let bv = &self.nodes[beta.0].value;
            let shape = iv.shape();
            let last_dim = *shape.last().ok_or(AutogradError::InvalidRankForOperation {
                op: "layer_norm",
                expected: 1,
                got: 0,
            })?;
            let data = iv.data();
            let gamma_data = gv.data();
            let beta_data = bv.data();
            let num_groups = data.len() / last_dim;
            let mut out = vec![0.0f32; data.len()];
            let mut normalized = vec![0.0f32; data.len()];
            for g in 0..num_groups {
                let base = g * last_dim;
                let slice = &data[base..base + last_dim];
                let mean = slice.iter().sum::<f32>() / last_dim as f32;
                let var =
                    slice.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / last_dim as f32;
                let inv_std = 1.0 / (var + epsilon).sqrt();
                for i in 0..last_dim {
                    let x_hat = (slice[i] - mean) * inv_std;
                    normalized[base + i] = x_hat;
                    out[base + i] = x_hat * gamma_data[i] + beta_data[i];
                }
            }
            let value = Tensor::from_vec(shape.to_vec(), out)?;
            let norm_tensor = Tensor::from_vec(shape.to_vec(), normalized)?;
            let rg = self.nodes[input.0].requires_grad
                || self.nodes[gamma.0].requires_grad
                || self.nodes[beta.0].requires_grad;
            (value, rg, norm_tensor)
        };
        Ok(self.push_node_with_aux(
            value,
            requires_grad,
            Op::LayerNorm {
                input,
                gamma,
                beta,
                eps_bits,
            },
            AuxData::NormNormalized(norm_tensor),
        ))
    }

    /// Group normalization on NHWC input `[N, H, W, C]`.
    ///
    /// `gamma` and `beta` must have shape `[C]`.
    /// `num_groups` must divide `C`.
    pub fn group_norm(
        &mut self,
        input: NodeId,
        gamma: NodeId,
        beta: NodeId,
        num_groups: usize,
        epsilon: f32,
    ) -> Result<NodeId, AutogradError> {
        let eps_bits = epsilon.to_bits();
        let (value, requires_grad, norm_tensor) = {
            let iv = &self.nodes[input.0].value;
            let gv = &self.nodes[gamma.0].value;
            let bv = &self.nodes[beta.0].value;
            let shape = iv.shape();
            if shape.len() != 4 {
                return Err(AutogradError::InvalidRankForOperation {
                    op: "group_norm",
                    expected: 4,
                    got: shape.len(),
                });
            }
            let (n, h, w, c) = (shape[0], shape[1], shape[2], shape[3]);
            let channels_per_group = c / num_groups;
            let spatial = h * w;
            let data = iv.data();
            let gamma_data = gv.data();
            let beta_data = bv.data();
            let mut out = vec![0.0f32; data.len()];
            let mut normalized = vec![0.0f32; data.len()];

            for ni in 0..n {
                for gi in 0..num_groups {
                    let c_start = gi * channels_per_group;
                    let c_end = c_start + channels_per_group;
                    let group_size = spatial * channels_per_group;
                    let mut sum = 0.0f32;
                    for hi in 0..h {
                        for wi in 0..w {
                            let base = ((ni * h + hi) * w + wi) * c;
                            for ci in c_start..c_end {
                                sum += data[base + ci];
                            }
                        }
                    }
                    let mean = sum / group_size as f32;
                    let mut var_sum = 0.0f32;
                    for hi in 0..h {
                        for wi in 0..w {
                            let base = ((ni * h + hi) * w + wi) * c;
                            for ci in c_start..c_end {
                                let d = data[base + ci] - mean;
                                var_sum += d * d;
                            }
                        }
                    }
                    let inv_std = 1.0 / (var_sum / group_size as f32 + epsilon).sqrt();
                    for hi in 0..h {
                        for wi in 0..w {
                            let base = ((ni * h + hi) * w + wi) * c;
                            for ci in c_start..c_end {
                                let x_hat = (data[base + ci] - mean) * inv_std;
                                normalized[base + ci] = x_hat;
                                out[base + ci] = x_hat * gamma_data[ci] + beta_data[ci];
                            }
                        }
                    }
                }
            }
            let value = Tensor::from_vec(shape.to_vec(), out)?;
            let norm_tensor = Tensor::from_vec(shape.to_vec(), normalized)?;
            let rg = self.nodes[input.0].requires_grad
                || self.nodes[gamma.0].requires_grad
                || self.nodes[beta.0].requires_grad;
            (value, rg, norm_tensor)
        };
        Ok(self.push_node_with_aux(
            value,
            requires_grad,
            Op::GroupNorm {
                input,
                gamma,
                beta,
                num_groups: num_groups as u16,
                eps_bits,
            },
            AuxData::NormNormalized(norm_tensor),
        ))
    }

    /// Flatten rank-4 NHWC tensor `[N,H,W,C]` to rank-2 `[N, H*W*C]`.
    pub fn flatten(&mut self, input: NodeId) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let v = &self.nodes[input.0].value;
            let shape = v.shape();
            if shape.len() < 2 {
                return Err(AutogradError::InvalidRankForOperation {
                    op: "flatten",
                    expected: 2,
                    got: shape.len(),
                });
            }
            let n = shape[0];
            let flat = v.len() / n;
            (v.reshape(vec![n, flat])?, self.nodes[input.0].requires_grad)
        };
        Ok(self.push_node(value, requires_grad, Op::Flatten(input)))
    }

    /// Reduces one node by summing along a single axis (removing that dimension).
    pub fn sum_axis(&mut self, input: NodeId, axis: usize) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let v = &self.nodes[input.0].value;
            (v.sum_axis(axis)?, self.nodes[input.0].requires_grad)
        };
        Ok(self.push_node(
            value,
            requires_grad,
            Op::SumAxis {
                input,
                axis: axis as u16,
            },
        ))
    }

    /// Reduces one node by averaging along a single axis (removing that dimension).
    pub fn mean_axis(&mut self, input: NodeId, axis: usize) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let v = &self.nodes[input.0].value;
            (v.mean_axis(axis)?, self.nodes[input.0].requires_grad)
        };
        Ok(self.push_node(
            value,
            requires_grad,
            Op::MeanAxis {
                input,
                axis: axis as u16,
            },
        ))
    }

    /// NHWC depthwise 2-D convolution forward.
    /// `input` shape `[N,H,W,C]`, `weight` shape `[KH,KW,C,1]`,
    /// optional `bias` shape `[C]`.
    pub fn depthwise_conv2d_nhwc(
        &mut self,
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let iv = &self.nodes[input.0].value;
            let wv = &self.nodes[weight.0].value;
            let bv: Option<&Tensor> = bias.map(|b| &self.nodes[b.0].value);
            let rg = self.nodes[input.0].requires_grad
                || self.nodes[weight.0].requires_grad
                || bias.is_some_and(|b| self.nodes[b.0].requires_grad);
            let result = if let Some(ref backend) = self.backend {
                backend.depthwise_conv2d_nhwc(iv, wv, bv, stride_h, stride_w)?
            } else {
                depthwise_conv2d_nhwc(iv, wv, bv, stride_h, stride_w)?
            };
            (result, rg)
        };
        Ok(self.push_node(
            value,
            requires_grad,
            Op::DepthwiseConv2dNhwc {
                input,
                weight,
                bias,
                stride_h: stride_h as u16,
                stride_w: stride_w as u16,
            },
        ))
    }

    /// Scatter: write values from `src` into `input` at row positions given by `indices`.
    /// input shape: `[N, D]`, indices shape: `[M]`, src shape: `[M, D]`.
    /// Result: input with rows at indices replaced by src rows.
    pub fn scatter(
        &mut self,
        input: NodeId,
        indices: NodeId,
        src: NodeId,
    ) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let iv = &self.nodes[input.0].value;
            let idx = &self.nodes[indices.0].value;
            let sv = &self.nodes[src.0].value;
            let shape = iv.shape();
            let d = shape[1];
            let mut out_data = iv.data().to_vec();
            for (i, &raw_idx) in idx.data().iter().enumerate() {
                let row = raw_idx as usize;
                let src_offset = i * d;
                let dst_offset = row * d;
                out_data[dst_offset..dst_offset + d]
                    .copy_from_slice(&sv.data()[src_offset..src_offset + d]);
            }
            let rg = self.nodes[input.0].requires_grad || self.nodes[src.0].requires_grad;
            (Tensor::from_vec(shape.to_vec(), out_data)?, rg)
        };
        Ok(self.push_node(
            value,
            requires_grad,
            Op::Scatter {
                input,
                indices,
                src,
            },
        ))
    }

    /// Embedding lookup: gather rows from weight matrix at given indices.
    /// weight shape: `[vocab_size, embed_dim]`, indices shape: `[seq_len]`.
    /// Result shape: `[seq_len, embed_dim]`.
    pub fn embedding_lookup(
        &mut self,
        weight: NodeId,
        indices: NodeId,
    ) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let wv = &self.nodes[weight.0].value;
            let idx = &self.nodes[indices.0].value;
            let embed_dim = wv.shape()[1];
            let seq_len = idx.data().len();
            let w_data = wv.data();
            let mut out_data = vec![0.0f32; seq_len * embed_dim];
            for (i, &raw_idx) in idx.data().iter().enumerate() {
                let row = raw_idx as usize;
                let src_offset = row * embed_dim;
                let dst_offset = i * embed_dim;
                out_data[dst_offset..dst_offset + embed_dim]
                    .copy_from_slice(&w_data[src_offset..src_offset + embed_dim]);
            }
            (
                Tensor::from_vec(vec![seq_len, embed_dim], out_data)?,
                self.nodes[weight.0].requires_grad,
            )
        };
        Ok(self.push_node(
            value,
            requires_grad,
            Op::EmbeddingLookup { weight, indices },
        ))
    }

    /// NLC 1-D convolution forward.
    /// `input` shape \[N,L,C_in\], `weight` shape \[K,C_in,C_out\],
    /// optional `bias` shape \[C_out\].
    pub fn conv1d_nlc(
        &mut self,
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        stride: usize,
    ) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let iv = &self.nodes[input.0].value;
            let wv = &self.nodes[weight.0].value;
            let bv: Option<&Tensor> = bias.map(|b| &self.nodes[b.0].value);
            let rg = self.nodes[input.0].requires_grad
                || self.nodes[weight.0].requires_grad
                || bias.is_some_and(|b| self.nodes[b.0].requires_grad);
            let in_shape = iv.shape();
            let w_shape = wv.shape();
            let (batch, length, _c_in) = (in_shape[0], in_shape[1], in_shape[2]);
            let (kernel_size, c_in, c_out) = (w_shape[0], w_shape[1], w_shape[2]);
            let out_len = (length - kernel_size) / stride + 1;
            let in_data = iv.data();
            let w_data = wv.data();
            let mut out = vec![0.0f32; batch * out_len * c_out];
            for b in 0..batch {
                for ol in 0..out_len {
                    let start = ol * stride;
                    for oc in 0..c_out {
                        let mut sum = 0.0f32;
                        for k in 0..kernel_size {
                            for ci in 0..c_in {
                                sum += in_data[(b * length + start + k) * c_in + ci]
                                    * w_data[(k * c_in + ci) * c_out + oc];
                            }
                        }
                        if let Some(bv) = bv {
                            sum += bv.data()[oc];
                        }
                        out[(b * out_len + ol) * c_out + oc] = sum;
                    }
                }
            }
            (Tensor::from_vec(vec![batch, out_len, c_out], out)?, rg)
        };
        Ok(self.push_node(
            value,
            requires_grad,
            Op::Conv1dNlc {
                input,
                weight,
                bias,
                stride: stride as u16,
            },
        ))
    }

    /// NDHWC 3-D convolution forward (no padding).
    /// `input` shape \[N,D,H,W,C_in\], `weight` shape \[KD,KH,KW,C_in,C_out\],
    /// optional `bias` shape \[C_out\].
    pub fn conv3d_ndhwc(
        &mut self,
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        stride_d: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let iv = &self.nodes[input.0].value;
            let wv = &self.nodes[weight.0].value;
            let bv: Option<&Tensor> = bias.map(|b| &self.nodes[b.0].value);
            let rg = self.nodes[input.0].requires_grad
                || self.nodes[weight.0].requires_grad
                || bias.is_some_and(|b| self.nodes[b.0].requires_grad);
            let (out_data, out_shape) = conv3d(
                iv.data(),
                iv.shape(),
                wv.data(),
                wv.shape(),
                (stride_d, stride_h, stride_w),
                (0, 0, 0),
            );
            let mut result = Tensor::from_vec(out_shape, out_data)?;
            if let Some(bv) = bv {
                let c_out = wv.shape()[4];
                let data = result.data_mut();
                let bias_data = bv.data();
                for pixel in data.chunks_mut(c_out) {
                    for (v, &bval) in pixel.iter_mut().zip(bias_data.iter()) {
                        *v += bval;
                    }
                }
            }
            (result, rg)
        };
        Ok(self.push_node(
            value,
            requires_grad,
            Op::Conv3dNdhwc {
                input,
                weight,
                bias,
                stride_d: stride_d as u16,
                stride_h: stride_h as u16,
                stride_w: stride_w as u16,
            },
        ))
    }

    /// Scaled dot-product attention forward.
    /// `query` shape `[seq_q, d_k]`, `key` shape `[seq_k, d_k]`, `value` shape `[seq_k, d_v]`.
    /// Returns `[seq_q, d_v]`.
    pub fn scaled_dot_product_attention(
        &mut self,
        query: NodeId,
        key: NodeId,
        value: NodeId,
    ) -> Result<NodeId, AutogradError> {
        let (output, attn_weights, requires_grad) = {
            let qv = &self.nodes[query.0].value;
            let kv = &self.nodes[key.0].value;
            let vv = &self.nodes[value.0].value;
            let rg = self.nodes[query.0].requires_grad
                || self.nodes[key.0].requires_grad
                || self.nodes[value.0].requires_grad;
            let d_k = qv.shape()[1];
            let scale = (d_k as f32).sqrt().recip();

            // scores = Q @ K^T, scaled
            let kt = kv.transpose_2d()?;
            let scores = matmul_2d(qv, &kt)?;
            let scaled = scores.scale(scale);

            // softmax along last dim
            let weights = yscv_kernels::softmax_last_dim(&scaled)?;

            // output = weights @ V
            let out = matmul_2d(&weights, vv)?;
            (out, weights, rg)
        };
        Ok(self.push_node_with_aux(
            output,
            requires_grad,
            Op::ScaledDotProductAttention { query, key, value },
            AuxData::AttentionWeights(attn_weights),
        ))
    }

    /// NHWC transposed 2-D convolution forward.
    /// `input` shape \[N,H,W,C_in\], `weight` shape \[KH,KW,C_out,C_in\],
    /// optional `bias` shape \[C_out\].
    /// Output shape: `[N, (H-1)*stride_h + KH, (W-1)*stride_w + KW, C_out]`.
    pub fn conv_transpose2d_nhwc(
        &mut self,
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let iv = &self.nodes[input.0].value;
            let wv = &self.nodes[weight.0].value;
            let rg = self.nodes[input.0].requires_grad
                || self.nodes[weight.0].requires_grad
                || bias.is_some_and(|b| self.nodes[b.0].requires_grad);
            let in_shape = iv.shape();
            let w_shape = wv.shape();
            let (n, h, w_dim, c_in) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
            let (kh, kw, c_out, _) = (w_shape[0], w_shape[1], w_shape[2], w_shape[3]);
            let out_h = (h - 1) * stride_h + kh;
            let out_w = (w_dim - 1) * stride_w + kw;
            let in_data = iv.data();
            let w_data = wv.data();
            let mut out = vec![0.0f32; n * out_h * out_w * c_out];
            for batch in 0..n {
                for ih in 0..h {
                    for iw in 0..w_dim {
                        for ic in 0..c_in {
                            let val = in_data[((batch * h + ih) * w_dim + iw) * c_in + ic];
                            for ki in 0..kh {
                                for kj in 0..kw {
                                    let oh = ih * stride_h + ki;
                                    let ow = iw * stride_w + kj;
                                    for oc in 0..c_out {
                                        let w_idx = ((ki * kw + kj) * c_out + oc) * c_in + ic;
                                        out[((batch * out_h + oh) * out_w + ow) * c_out + oc] +=
                                            val * w_data[w_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if let Some(b_id) = bias {
                let bv = &self.nodes[b_id.0].value;
                let bd = bv.data();
                for i in 0..(n * out_h * out_w) {
                    for oc in 0..c_out {
                        out[i * c_out + oc] += bd[oc];
                    }
                }
            }
            (Tensor::from_vec(vec![n, out_h, out_w, c_out], out)?, rg)
        };
        Ok(self.push_node(
            value,
            requires_grad,
            Op::ConvTranspose2dNhwc {
                input,
                weight,
                bias,
                stride_h: stride_h as u16,
                stride_w: stride_w as u16,
            },
        ))
    }

    /// NHWC adaptive average pool 2d forward.
    /// `input` shape `[N,H,W,C]`, output shape `[N,out_h,out_w,C]`.
    pub fn adaptive_avg_pool2d_nhwc(
        &mut self,
        input: NodeId,
        out_h: usize,
        out_w: usize,
    ) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let iv = &self.nodes[input.0].value;
            let shape = iv.shape();
            if shape.len() != 4 {
                return Err(AutogradError::InvalidRankForOperation {
                    op: "adaptive_avg_pool2d_nhwc",
                    expected: 4,
                    got: shape.len(),
                });
            }
            let (n, h, w, c) = (shape[0], shape[1], shape[2], shape[3]);
            let data = iv.data();
            let mut out = vec![0.0f32; n * out_h * out_w * c];
            for b in 0..n {
                for oh in 0..out_h {
                    let h_start = oh * h / out_h;
                    let h_end = ((oh + 1) * h / out_h).max(h_start + 1);
                    for ow in 0..out_w {
                        let w_start = ow * w / out_w;
                        let w_end = ((ow + 1) * w / out_w).max(w_start + 1);
                        let count = (h_end - h_start) * (w_end - w_start);
                        for ch in 0..c {
                            let mut sum = 0.0f32;
                            for ih in h_start..h_end {
                                for iw in w_start..w_end {
                                    sum += data[((b * h + ih) * w + iw) * c + ch];
                                }
                            }
                            out[((b * out_h + oh) * out_w + ow) * c + ch] = sum / count as f32;
                        }
                    }
                }
            }
            (
                Tensor::from_vec(vec![n, out_h, out_w, c], out)?,
                self.nodes[input.0].requires_grad,
            )
        };
        Ok(self.push_node(
            value,
            requires_grad,
            Op::AdaptiveAvgPool2dNhwc {
                input,
                out_h: out_h as u16,
                out_w: out_w as u16,
            },
        ))
    }

    /// NHWC adaptive max pool 2d forward with argmax tracking for backward.
    pub fn adaptive_max_pool2d_nhwc(
        &mut self,
        input: NodeId,
        out_h: usize,
        out_w: usize,
    ) -> Result<NodeId, AutogradError> {
        let (value, requires_grad, indices) = {
            let iv = &self.nodes[input.0].value;
            let shape = iv.shape();
            if shape.len() != 4 {
                return Err(AutogradError::InvalidRankForOperation {
                    op: "adaptive_max_pool2d_nhwc",
                    expected: 4,
                    got: shape.len(),
                });
            }
            let (n, h, w, c) = (shape[0], shape[1], shape[2], shape[3]);
            let data = iv.data();
            let out_len = n * out_h * out_w * c;
            let mut out = vec![f32::NEG_INFINITY; out_len];
            let mut indices = vec![0usize; out_len];
            for b in 0..n {
                for oh in 0..out_h {
                    let h_start = oh * h / out_h;
                    let h_end = ((oh + 1) * h / out_h).max(h_start + 1);
                    for ow in 0..out_w {
                        let w_start = ow * w / out_w;
                        let w_end = ((ow + 1) * w / out_w).max(w_start + 1);
                        for ch in 0..c {
                            let out_idx = ((b * out_h + oh) * out_w + ow) * c + ch;
                            let mut best_val = f32::NEG_INFINITY;
                            let mut best_in = 0usize;
                            for ih in h_start..h_end {
                                for iw in w_start..w_end {
                                    let in_idx = ((b * h + ih) * w + iw) * c + ch;
                                    let v = data[in_idx];
                                    if v > best_val {
                                        best_val = v;
                                        best_in = in_idx;
                                    }
                                }
                            }
                            out[out_idx] = best_val;
                            indices[out_idx] = best_in;
                        }
                    }
                }
            }
            (
                Tensor::from_vec(vec![n, out_h, out_w, c], out)?,
                self.nodes[input.0].requires_grad,
                indices,
            )
        };
        Ok(self.push_node_with_aux(
            value,
            requires_grad,
            Op::AdaptiveMaxPool2dNhwc {
                input,
                out_h: out_h as u16,
                out_w: out_w as u16,
            },
            AuxData::MaxPoolIndices(indices),
        ))
    }

    /// Instance normalization (NHWC) forward.
    /// Normalizes per (N,C) pair across H*W spatial dimensions.
    /// `gamma` and `beta` must have shape `[C]`.
    pub fn instance_norm_nhwc(
        &mut self,
        input: NodeId,
        gamma: NodeId,
        beta: NodeId,
        epsilon: f32,
    ) -> Result<NodeId, AutogradError> {
        let eps_bits = epsilon.to_bits();
        let (value, requires_grad, norm_tensor) = {
            let iv = &self.nodes[input.0].value;
            let gv = &self.nodes[gamma.0].value;
            let bv = &self.nodes[beta.0].value;
            let shape = iv.shape();
            if shape.len() != 4 {
                return Err(AutogradError::InvalidRankForOperation {
                    op: "instance_norm_nhwc",
                    expected: 4,
                    got: shape.len(),
                });
            }
            let (n, h, w, c) = (shape[0], shape[1], shape[2], shape[3]);
            let spatial = h * w;
            let data = iv.data();
            let gamma_data = gv.data();
            let beta_data = bv.data();
            let mut out = vec![0.0f32; data.len()];
            let mut normalized = vec![0.0f32; data.len()];

            for ni in 0..n {
                for ch in 0..c {
                    let mut sum = 0.0f32;
                    for s in 0..spatial {
                        let idx = (ni * h * w + s) * c + ch;
                        sum += data[idx];
                    }
                    let mean = sum / spatial as f32;
                    let mut var_sum = 0.0f32;
                    for s in 0..spatial {
                        let idx = (ni * h * w + s) * c + ch;
                        let d = data[idx] - mean;
                        var_sum += d * d;
                    }
                    let inv_std = 1.0 / (var_sum / spatial as f32 + epsilon).sqrt();
                    for s in 0..spatial {
                        let idx = (ni * h * w + s) * c + ch;
                        let x_hat = (data[idx] - mean) * inv_std;
                        normalized[idx] = x_hat;
                        out[idx] = x_hat * gamma_data[ch] + beta_data[ch];
                    }
                }
            }
            let value = Tensor::from_vec(shape.to_vec(), out)?;
            let norm_tensor = Tensor::from_vec(shape.to_vec(), normalized)?;
            let rg = self.nodes[input.0].requires_grad
                || self.nodes[gamma.0].requires_grad
                || self.nodes[beta.0].requires_grad;
            (value, rg, norm_tensor)
        };
        Ok(self.push_node_with_aux(
            value,
            requires_grad,
            Op::InstanceNormNhwc {
                input,
                gamma,
                beta,
                eps_bits,
            },
            AuxData::NormNormalized(norm_tensor),
        ))
    }

    /// PReLU activation forward.
    /// `alpha` is a parameter node with shape `[C]` or `[1]`.
    /// For NHWC inputs, channels are the last dimension.
    pub fn prelu(&mut self, input: NodeId, alpha: NodeId) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let iv = &self.nodes[input.0].value;
            let av = &self.nodes[alpha.0].value;
            let in_data = iv.data();
            let alpha_data = av.data();
            let alpha_len = alpha_data.len();
            let out: Vec<f32> = in_data
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    if x > 0.0 {
                        x
                    } else {
                        let a = if alpha_len == 1 {
                            alpha_data[0]
                        } else {
                            alpha_data[i % alpha_len]
                        };
                        a * x
                    }
                })
                .collect();
            let rg = self.nodes[input.0].requires_grad || self.nodes[alpha.0].requires_grad;
            (Tensor::from_vec(iv.shape().to_vec(), out)?, rg)
        };
        Ok(self.push_node(value, requires_grad, Op::PRelu { input, alpha }))
    }

    /// Pixel shuffle forward: rearranges [N, H, W, C*r^2] -> [N, H*r, W*r, C].
    pub fn pixel_shuffle(
        &mut self,
        input: NodeId,
        upscale_factor: usize,
    ) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let iv = &self.nodes[input.0].value;
            let shape = iv.shape();
            if shape.len() != 4 {
                return Err(AutogradError::InvalidRankForOperation {
                    op: "pixel_shuffle",
                    expected: 4,
                    got: shape.len(),
                });
            }
            let (batch, h, w, c) = (shape[0], shape[1], shape[2], shape[3]);
            let r = upscale_factor;
            let out_c = c / (r * r);
            let out_h = h * r;
            let out_w = w * r;
            let data = iv.data();
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
            (
                Tensor::from_vec(vec![batch, out_h, out_w, out_c], out)?,
                self.nodes[input.0].requires_grad,
            )
        };
        Ok(self.push_node(
            value,
            requires_grad,
            Op::PixelShuffle {
                input,
                upscale_factor: upscale_factor as u16,
            },
        ))
    }

    /// Nearest-neighbor upsample forward: [N, H, W, C] -> [N, H*r, W*r, C].
    pub fn upsample_nearest(
        &mut self,
        input: NodeId,
        scale_factor: usize,
    ) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let iv = &self.nodes[input.0].value;
            let shape = iv.shape();
            if shape.len() != 4 {
                return Err(AutogradError::InvalidRankForOperation {
                    op: "upsample_nearest",
                    expected: 4,
                    got: shape.len(),
                });
            }
            let (batch, h, w, c) = (shape[0], shape[1], shape[2], shape[3]);
            let r = scale_factor;
            let out_h = h * r;
            let out_w = w * r;
            let data = iv.data();
            let mut out = vec![0.0f32; batch * out_h * out_w * c];
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
            (
                Tensor::from_vec(vec![batch, out_h, out_w, c], out)?,
                self.nodes[input.0].requires_grad,
            )
        };
        Ok(self.push_node(
            value,
            requires_grad,
            Op::UpsampleNearest {
                input,
                scale_factor: scale_factor as u16,
            },
        ))
    }

    /// RNN forward pass through all timesteps (for BPTT).
    /// input: `[seq_len, input_size]`, w_ih: `[input_size, hidden_size]`,
    /// w_hh: `[hidden_size, hidden_size]`, bias: `[hidden_size]`.
    /// Returns output `[seq_len, hidden_size]`.
    pub fn rnn_forward(
        &mut self,
        input: NodeId,
        w_ih: NodeId,
        w_hh: NodeId,
        bias: NodeId,
    ) -> Result<NodeId, AutogradError> {
        let (value, requires_grad, hidden_states) = {
            let iv = &self.nodes[input.0].value;
            let wih = &self.nodes[w_ih.0].value;
            let whh = &self.nodes[w_hh.0].value;
            let bv = &self.nodes[bias.0].value;
            let shape = iv.shape();
            let seq_len = shape[0];
            let hidden_size = wih.shape()[1];
            let in_data = iv.data();
            let wih_data = wih.data();
            let whh_data = whh.data();
            let b_data = bv.data();
            let input_size = shape[1];

            let mut hidden_states = Vec::with_capacity(seq_len + 1);
            // h_0 = zeros
            hidden_states.push(Tensor::zeros(vec![hidden_size])?);
            let mut output_data = vec![0.0f32; seq_len * hidden_size];

            for t in 0..seq_len {
                let h_prev = hidden_states[t].data();
                let x_base = t * input_size;
                let mut h_new = vec![0.0f32; hidden_size];
                for j in 0..hidden_size {
                    let mut sum = b_data[j];
                    for i in 0..input_size {
                        sum += in_data[x_base + i] * wih_data[i * hidden_size + j];
                    }
                    for i in 0..hidden_size {
                        sum += h_prev[i] * whh_data[i * hidden_size + j];
                    }
                    h_new[j] = sum.tanh();
                }
                output_data[t * hidden_size..(t + 1) * hidden_size].copy_from_slice(&h_new);
                hidden_states.push(Tensor::from_vec(vec![hidden_size], h_new)?);
            }

            let rg = self.nodes[input.0].requires_grad
                || self.nodes[w_ih.0].requires_grad
                || self.nodes[w_hh.0].requires_grad
                || self.nodes[bias.0].requires_grad;
            (
                Tensor::from_vec(vec![seq_len, hidden_size], output_data)?,
                rg,
                hidden_states,
            )
        };
        Ok(self.push_node_with_aux(
            value,
            requires_grad,
            Op::Rnn {
                input,
                w_ih,
                w_hh,
                bias,
            },
            AuxData::RnnHiddenStates(hidden_states),
        ))
    }

    /// LSTM forward pass through all timesteps (for BPTT).
    /// input: `[seq_len, input_size]`, w_ih: `[input_size, 4*hidden_size]`,
    /// w_hh: `[hidden_size, 4*hidden_size]`, bias: `[4*hidden_size]`.
    /// Returns output `[seq_len, hidden_size]`.
    pub fn lstm_forward(
        &mut self,
        input: NodeId,
        w_ih: NodeId,
        w_hh: NodeId,
        bias: NodeId,
    ) -> Result<NodeId, AutogradError> {
        let (value, requires_grad, hidden_states, cell_states, gates) = {
            let iv = &self.nodes[input.0].value;
            let wih = &self.nodes[w_ih.0].value;
            let whh = &self.nodes[w_hh.0].value;
            let bv = &self.nodes[bias.0].value;
            let shape = iv.shape();
            let seq_len = shape[0];
            let input_size = shape[1];
            let hidden_size = wih.shape()[1] / 4;
            let in_data = iv.data();
            let wih_data = wih.data();
            let whh_data = whh.data();
            let b_data = bv.data();

            let mut hidden_states = Vec::with_capacity(seq_len + 1);
            let mut cell_states = Vec::with_capacity(seq_len + 1);
            let mut gates_vec: Vec<(Tensor, Tensor, Tensor, Tensor)> = Vec::with_capacity(seq_len);
            hidden_states.push(Tensor::zeros(vec![hidden_size])?);
            cell_states.push(Tensor::zeros(vec![hidden_size])?);
            let mut output_data = vec![0.0f32; seq_len * hidden_size];

            for t in 0..seq_len {
                let h_prev = hidden_states[t].data();
                let c_prev = cell_states[t].data();
                let x_base = t * input_size;
                let h4 = 4 * hidden_size;

                // Compute gates: [i, f, g, o] = x @ W_ih + h @ W_hh + bias
                let mut raw_gates = vec![0.0f32; h4];
                for j in 0..h4 {
                    let mut sum = b_data[j];
                    for i in 0..input_size {
                        sum += in_data[x_base + i] * wih_data[i * h4 + j];
                    }
                    for i in 0..hidden_size {
                        sum += h_prev[i] * whh_data[i * h4 + j];
                    }
                    raw_gates[j] = sum;
                }

                let mut i_gate = vec![0.0f32; hidden_size];
                let mut f_gate = vec![0.0f32; hidden_size];
                let mut g_gate = vec![0.0f32; hidden_size];
                let mut o_gate = vec![0.0f32; hidden_size];
                let mut c_new = vec![0.0f32; hidden_size];
                let mut h_new = vec![0.0f32; hidden_size];

                for j in 0..hidden_size {
                    i_gate[j] = sigmoid_f32(raw_gates[j]);
                    f_gate[j] = sigmoid_f32(raw_gates[hidden_size + j]);
                    g_gate[j] = raw_gates[2 * hidden_size + j].tanh();
                    o_gate[j] = sigmoid_f32(raw_gates[3 * hidden_size + j]);
                    c_new[j] = f_gate[j] * c_prev[j] + i_gate[j] * g_gate[j];
                    h_new[j] = o_gate[j] * c_new[j].tanh();
                }

                output_data[t * hidden_size..(t + 1) * hidden_size].copy_from_slice(&h_new);
                hidden_states.push(Tensor::from_vec(vec![hidden_size], h_new)?);
                cell_states.push(Tensor::from_vec(vec![hidden_size], c_new)?);
                gates_vec.push((
                    Tensor::from_vec(vec![hidden_size], i_gate)?,
                    Tensor::from_vec(vec![hidden_size], f_gate)?,
                    Tensor::from_vec(vec![hidden_size], g_gate)?,
                    Tensor::from_vec(vec![hidden_size], o_gate)?,
                ));
            }

            let rg = self.nodes[input.0].requires_grad
                || self.nodes[w_ih.0].requires_grad
                || self.nodes[w_hh.0].requires_grad
                || self.nodes[bias.0].requires_grad;
            (
                Tensor::from_vec(vec![seq_len, hidden_size], output_data)?,
                rg,
                hidden_states,
                cell_states,
                gates_vec,
            )
        };
        Ok(self.push_node_with_aux(
            value,
            requires_grad,
            Op::Lstm {
                input,
                w_ih,
                w_hh,
                bias,
            },
            AuxData::LstmStates {
                hidden_states,
                cell_states,
                gates,
            },
        ))
    }

    /// GRU forward pass through all timesteps (for BPTT).
    /// input: `[seq_len, input_size]`, w_ih: `[input_size, 3*hidden_size]`,
    /// w_hh: `[hidden_size, 3*hidden_size]`, bias_ih: `[3*hidden_size]`, bias_hh: `[3*hidden_size]`.
    /// Returns output `[seq_len, hidden_size]`.
    pub fn gru_forward(
        &mut self,
        input: NodeId,
        w_ih: NodeId,
        w_hh: NodeId,
        bias_ih: NodeId,
        bias_hh: NodeId,
    ) -> Result<NodeId, AutogradError> {
        let (value, requires_grad, hidden_states, gates) = {
            let iv = &self.nodes[input.0].value;
            let wih = &self.nodes[w_ih.0].value;
            let whh = &self.nodes[w_hh.0].value;
            let bih = &self.nodes[bias_ih.0].value;
            let bhh = &self.nodes[bias_hh.0].value;
            let shape = iv.shape();
            let seq_len = shape[0];
            let input_size = shape[1];
            let hidden_size = wih.shape()[1] / 3;
            let in_data = iv.data();
            let wih_data = wih.data();
            let whh_data = whh.data();
            let bih_data = bih.data();
            let bhh_data = bhh.data();

            let mut hidden_states = Vec::with_capacity(seq_len + 1);
            let mut gates_vec: Vec<(Tensor, Tensor, Tensor)> = Vec::with_capacity(seq_len);
            hidden_states.push(Tensor::zeros(vec![hidden_size])?);
            let mut output_data = vec![0.0f32; seq_len * hidden_size];

            for t in 0..seq_len {
                let h_prev = hidden_states[t].data();
                let x_base = t * input_size;
                let h3 = 3 * hidden_size;

                // x_proj = x @ W_ih + bias_ih
                let mut x_proj = vec![0.0f32; h3];
                for j in 0..h3 {
                    let mut sum = bih_data[j];
                    for i in 0..input_size {
                        sum += in_data[x_base + i] * wih_data[i * h3 + j];
                    }
                    x_proj[j] = sum;
                }
                // h_proj = h @ W_hh + bias_hh
                let mut h_proj = vec![0.0f32; h3];
                for j in 0..h3 {
                    let mut sum = bhh_data[j];
                    for i in 0..hidden_size {
                        sum += h_prev[i] * whh_data[i * h3 + j];
                    }
                    h_proj[j] = sum;
                }

                let mut r_gate = vec![0.0f32; hidden_size];
                let mut z_gate = vec![0.0f32; hidden_size];
                let mut n_candidate = vec![0.0f32; hidden_size];
                let mut h_new = vec![0.0f32; hidden_size];

                for j in 0..hidden_size {
                    r_gate[j] = sigmoid_f32(x_proj[j] + h_proj[j]);
                    z_gate[j] = sigmoid_f32(x_proj[hidden_size + j] + h_proj[hidden_size + j]);
                    n_candidate[j] = (x_proj[2 * hidden_size + j]
                        + r_gate[j] * h_proj[2 * hidden_size + j])
                        .tanh();
                    h_new[j] = (1.0 - z_gate[j]) * n_candidate[j] + z_gate[j] * h_prev[j];
                }

                output_data[t * hidden_size..(t + 1) * hidden_size].copy_from_slice(&h_new);
                hidden_states.push(Tensor::from_vec(vec![hidden_size], h_new)?);
                gates_vec.push((
                    Tensor::from_vec(vec![hidden_size], r_gate)?,
                    Tensor::from_vec(vec![hidden_size], z_gate)?,
                    Tensor::from_vec(vec![hidden_size], n_candidate)?,
                ));
            }

            let rg = self.nodes[input.0].requires_grad
                || self.nodes[w_ih.0].requires_grad
                || self.nodes[w_hh.0].requires_grad
                || self.nodes[bias_ih.0].requires_grad
                || self.nodes[bias_hh.0].requires_grad;
            (
                Tensor::from_vec(vec![seq_len, hidden_size], output_data)?,
                rg,
                hidden_states,
                gates_vec,
            )
        };
        Ok(self.push_node_with_aux(
            value,
            requires_grad,
            Op::Gru {
                input,
                w_ih,
                w_hh,
                bias_ih,
                bias_hh,
            },
            AuxData::GruStates {
                hidden_states,
                gates,
            },
        ))
    }

    /// Deformable conv2d NHWC forward.
    /// input: \[N,H,W,C_in\], weight: \[KH,KW,C_in,C_out\], offsets: \[N,OH,OW,KH\*KW\*2\].
    pub fn deformable_conv2d_nhwc(
        &mut self,
        input: NodeId,
        weight: NodeId,
        offsets: NodeId,
        bias: Option<NodeId>,
        stride: usize,
        padding: usize,
    ) -> Result<NodeId, AutogradError> {
        let (value, requires_grad) = {
            let iv = &self.nodes[input.0].value;
            let wv = &self.nodes[weight.0].value;
            let ov = &self.nodes[offsets.0].value;
            let bv: Option<&Tensor> = bias.map(|b| &self.nodes[b.0].value);
            let rg = self.nodes[input.0].requires_grad
                || self.nodes[weight.0].requires_grad
                || self.nodes[offsets.0].requires_grad
                || bias.is_some_and(|b| self.nodes[b.0].requires_grad);
            let result = yscv_kernels::deformable_conv2d_nhwc(iv, wv, ov, bv, stride, padding)?;
            (result, rg)
        };
        Ok(self.push_node(
            value,
            requires_grad,
            Op::DeformableConv2dNhwc {
                input,
                weight,
                offsets,
                bias,
                stride: stride as u16,
                padding: padding as u16,
            },
        ))
    }

    pub(crate) fn push_node(&mut self, value: Tensor, requires_grad: bool, op: Op) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            value,
            grad: None,
            requires_grad,
            op,
            aux: None,
        });
        id
    }

    pub(crate) fn push_node_with_aux(
        &mut self,
        value: Tensor,
        requires_grad: bool,
        op: Op,
        aux: super::node::AuxData,
    ) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            value,
            grad: None,
            requires_grad,
            op,
            aux: Some(aux),
        });
        id
    }

    pub(crate) fn node(&self, id: NodeId) -> Result<&Node, AutogradError> {
        self.nodes
            .get(id.0)
            .ok_or(AutogradError::NodeNotFound { id: id.0 })
    }

    pub(crate) fn node_mut(&mut self, id: NodeId) -> Result<&mut Node, AutogradError> {
        self.nodes
            .get_mut(id.0)
            .ok_or(AutogradError::NodeNotFound { id: id.0 })
    }

    /// Clips gradients by global L2 norm. Returns the original norm before clipping.
    /// If total_norm > max_norm, scales all gradients by max_norm / total_norm.
    pub fn clip_grad_norm(&mut self, param_nodes: &[NodeId], max_norm: f32) -> f32 {
        let mut total_norm_sq = 0.0f32;
        for &node_id in param_nodes {
            if let Some(grad) = &self.nodes[node_id.0].grad {
                for &v in grad.data() {
                    total_norm_sq += v * v;
                }
            }
        }
        let total_norm = total_norm_sq.sqrt();

        if total_norm > max_norm {
            let scale = max_norm / total_norm;
            for &node_id in param_nodes {
                if let Some(grad) = &mut self.nodes[node_id.0].grad {
                    for v in grad.data_mut() {
                        *v *= scale;
                    }
                }
            }
        }
        total_norm
    }

    /// Clips each gradient element to [-max_value, max_value].
    pub fn clip_grad_value(&mut self, param_nodes: &[NodeId], max_value: f32) {
        for &node_id in param_nodes {
            if let Some(grad) = &mut self.nodes[node_id.0].grad {
                for v in grad.data_mut() {
                    *v = v.clamp(-max_value, max_value);
                }
            }
        }
    }
}

/// Softmax along last dimension (local utility to avoid tight coupling with kernel version).
fn softmax_last_dim(input: &Tensor) -> Tensor {
    let shape = input.shape();
    if shape.is_empty() {
        return input.clone();
    }
    let last = *shape.last().expect("non-empty shape");
    let outer = input.len() / last;
    let data = input.data();
    let mut out = vec![0.0f32; input.len()];

    for o in 0..outer {
        let base = o * last;
        let max_val = data[base..base + last]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for i in 0..last {
            let e = (data[base + i] - max_val).exp();
            out[base + i] = e;
            sum += e;
        }
        let inv = 1.0 / sum;
        for i in 0..last {
            out[base + i] *= inv;
        }
    }

    Tensor::from_vec(shape.to_vec(), out).expect("softmax_last_dim preserves shape")
}

#[inline]
fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Copy src data (with `src_shape`) into a region of dst data (with `dst_shape`) at the given offsets.
#[allow(clippy::needless_range_loop)]
fn copy_region_nd(
    src: &[f32],
    src_shape: &[usize],
    dst: &mut [f32],
    dst_shape: &[usize],
    offsets: &[u32],
) {
    let rank = src_shape.len();
    if rank == 0 {
        return;
    }
    let total: usize = src_shape.iter().product();
    let mut src_strides = vec![1usize; rank];
    let mut dst_strides = vec![1usize; rank];
    for d in (0..rank - 1).rev() {
        src_strides[d] = src_strides[d + 1] * src_shape[d + 1];
        dst_strides[d] = dst_strides[d + 1] * dst_shape[d + 1];
    }
    for flat in 0..total {
        let mut rem = flat;
        let mut dst_flat = 0usize;
        for d in 0..rank {
            let coord = rem / src_strides[d];
            rem %= src_strides[d];
            dst_flat += (coord + offsets[d] as usize) * dst_strides[d];
        }
        dst[dst_flat] = src[flat];
    }
}
