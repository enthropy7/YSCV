use yscv_autograd::{Graph, NodeId};
use yscv_tensor::Tensor;

use crate::ModelError;

/// Stateless ReLU layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ReLULayer;

impl ReLULayer {
    pub const fn new() -> Self {
        Self
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId, ModelError> {
        graph.relu(input).map_err(Into::into)
    }
}

/// Stateless LeakyReLU layer with configurable negative slope.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LeakyReLULayer {
    negative_slope: f32,
}

impl LeakyReLULayer {
    pub fn new(negative_slope: f32) -> Result<Self, ModelError> {
        if !negative_slope.is_finite() || negative_slope < 0.0 {
            return Err(ModelError::InvalidLeakyReluSlope { negative_slope });
        }
        Ok(Self { negative_slope })
    }

    pub const fn negative_slope(&self) -> f32 {
        self.negative_slope
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId, ModelError> {
        let positive = graph.relu(input)?;
        let zero = graph.constant(Tensor::scalar(0.0));
        let neg_input = graph.sub(zero, input)?;
        let negative_magnitude = graph.relu(neg_input)?;
        let slope = graph.constant(Tensor::scalar(self.negative_slope));
        let scaled_negative = graph.mul(negative_magnitude, slope)?;
        graph.sub(positive, scaled_negative).map_err(Into::into)
    }
}

/// Stateless sigmoid activation layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SigmoidLayer;

impl SigmoidLayer {
    pub const fn new() -> Self {
        Self
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId, ModelError> {
        graph.sigmoid(input).map_err(Into::into)
    }
}

/// Stateless tanh activation layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TanhLayer;

impl TanhLayer {
    pub const fn new() -> Self {
        Self
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId, ModelError> {
        graph.tanh(input).map_err(Into::into)
    }
}

/// GELU activation layer.
#[derive(Debug, Clone)]
pub struct GELULayer;

impl Default for GELULayer {
    fn default() -> Self {
        Self::new()
    }
}

impl GELULayer {
    pub const fn new() -> Self {
        Self
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId, ModelError> {
        graph.gelu(input).map_err(Into::into)
    }

    pub fn forward_inference(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        Ok(yscv_kernels::gelu(input))
    }
}

/// SiLU (Swish) activation layer.
#[derive(Debug, Clone)]
pub struct SiLULayer;

impl Default for SiLULayer {
    fn default() -> Self {
        Self::new()
    }
}

impl SiLULayer {
    pub const fn new() -> Self {
        Self
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId, ModelError> {
        graph.silu(input).map_err(Into::into)
    }

    pub fn forward_inference(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        Ok(yscv_kernels::silu(input))
    }
}

/// Mish activation layer.
#[derive(Debug, Clone)]
pub struct MishLayer;

impl Default for MishLayer {
    fn default() -> Self {
        Self::new()
    }
}

impl MishLayer {
    pub const fn new() -> Self {
        Self
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId, ModelError> {
        graph.mish(input).map_err(Into::into)
    }

    pub fn forward_inference(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        Ok(yscv_kernels::mish(input))
    }
}

/// PReLU activation layer.
/// Uses per-channel or single alpha for the negative slope.
#[derive(Debug, Clone)]
pub struct PReLULayer {
    alpha: Vec<f32>,
    alpha_node: Option<NodeId>,
}

impl PReLULayer {
    pub const fn new(alpha: Vec<f32>) -> Self {
        Self {
            alpha,
            alpha_node: None,
        }
    }

    pub fn alpha(&self) -> &[f32] {
        &self.alpha
    }

    pub const fn alpha_node(&self) -> Option<NodeId> {
        self.alpha_node
    }

    pub fn register_params(&mut self, graph: &mut Graph) {
        self.alpha_node = Some(
            graph.variable(
                Tensor::from_vec(vec![self.alpha.len()], self.alpha.clone())
                    .expect("shape matches data"),
            ),
        );
    }

    pub fn sync_from_graph(&mut self, graph: &Graph) -> Result<(), ModelError> {
        if let Some(a_id) = self.alpha_node {
            self.alpha = graph.value(a_id)?.data().to_vec();
        }
        Ok(())
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId, ModelError> {
        let a_id = self
            .alpha_node
            .ok_or(ModelError::ParamsNotRegistered { layer: "PReLU" })?;
        graph.prelu(input, a_id).map_err(Into::into)
    }

    pub fn forward_inference(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        let data = input.data();
        let out: Vec<f32> = if self.alpha.len() == 1 {
            let a = self.alpha[0];
            data.iter()
                .map(|&x| if x > 0.0 { x } else { a * x })
                .collect()
        } else {
            let shape = input.shape();
            let channels = if shape.len() >= 2 { shape[1] } else { 1 };
            let spatial: usize = shape[2..].iter().product();
            let mut result = data.to_vec();
            for (i, v) in result.iter_mut().enumerate() {
                let c = (i / spatial) % channels;
                let a = self.alpha.get(c).copied().unwrap_or(0.01);
                if *v < 0.0 {
                    *v *= a;
                }
            }
            result
        };
        Tensor::from_vec(input.shape().to_vec(), out).map_err(ModelError::Tensor)
    }
}
