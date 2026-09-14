use yscv_autograd::{Graph, NodeId};
use yscv_tensor::Tensor;

use crate::ModelError;

/// Dense linear layer: `y = x @ weight + bias`.
///
/// Supports both graph-mode (autograd training) and inference-mode (raw tensors).
#[derive(Debug, Clone, PartialEq)]
pub struct LinearLayer {
    in_features: usize,
    out_features: usize,
    weight_tensor: Tensor,
    bias_tensor: Tensor,
    weight_node: Option<NodeId>,
    bias_node: Option<NodeId>,
}

impl LinearLayer {
    /// Creates a layer from explicit parameter tensors.
    pub fn new(
        graph: &mut Graph,
        in_features: usize,
        out_features: usize,
        weight_init: Tensor,
        bias_init: Tensor,
    ) -> Result<Self, ModelError> {
        let expected_weight = vec![in_features, out_features];
        if weight_init.shape() != expected_weight {
            return Err(ModelError::InvalidParameterShape {
                parameter: "weight",
                expected: expected_weight,
                got: weight_init.shape().to_vec(),
            });
        }
        let expected_bias = vec![out_features];
        if bias_init.shape() != expected_bias {
            return Err(ModelError::InvalidParameterShape {
                parameter: "bias",
                expected: expected_bias,
                got: bias_init.shape().to_vec(),
            });
        }

        let weight_node = graph.variable(weight_init.clone());
        let bias_node = graph.variable(bias_init.clone());
        Ok(Self {
            in_features,
            out_features,
            weight_tensor: weight_init,
            bias_tensor: bias_init,
            weight_node: Some(weight_node),
            bias_node: Some(bias_node),
        })
    }

    /// Creates a zero-initialized layer.
    pub fn zero_init(
        graph: &mut Graph,
        in_features: usize,
        out_features: usize,
    ) -> Result<Self, ModelError> {
        let weight = Tensor::zeros(vec![in_features, out_features])?;
        let bias = Tensor::zeros(vec![out_features])?;
        Self::new(graph, in_features, out_features, weight, bias)
    }

    /// Synchronizes owned tensors from the graph (e.g. after optimizer step).
    pub fn sync_from_graph(&mut self, graph: &Graph) -> Result<(), ModelError> {
        if let Some(w_id) = self.weight_node {
            self.weight_tensor = graph.value(w_id)?.clone();
        }
        if let Some(b_id) = self.bias_node {
            self.bias_tensor = graph.value(b_id)?.clone();
        }
        Ok(())
    }

    /// Graph-mode forward pass (for training with autograd).
    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId, ModelError> {
        let w_id = self
            .weight_node
            .ok_or(ModelError::ParamsNotRegistered { layer: "Linear" })?;
        let b_id = self
            .bias_node
            .ok_or(ModelError::ParamsNotRegistered { layer: "Linear" })?;
        let input_shape = graph.value(input)?.shape().to_vec();
        if input_shape.len() != 2 || input_shape[1] != self.in_features {
            return Err(ModelError::InvalidInputShape {
                expected_features: self.in_features,
                got: input_shape,
            });
        }

        let projected = graph.matmul_2d(input, w_id)?;
        let output = graph.add(projected, b_id)?;
        Ok(output)
    }

    /// Inference-mode forward pass (no graph needed).
    pub fn forward_inference(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        if input.rank() != 2 || input.shape()[1] != self.in_features {
            return Err(ModelError::InvalidInputShape {
                expected_features: self.in_features,
                got: input.shape().to_vec(),
            });
        }
        let projected = yscv_kernels::matmul_2d(input, &self.weight_tensor)?;
        projected.add(&self.bias_tensor).map_err(ModelError::Tensor)
    }

    pub const fn in_features(&self) -> usize {
        self.in_features
    }

    pub const fn out_features(&self) -> usize {
        self.out_features
    }

    pub(crate) fn trainable_nodes(&self) -> Vec<NodeId> {
        let mut nodes = Vec::new();
        if let Some(w) = self.weight_node {
            nodes.push(w);
        }
        if let Some(b) = self.bias_node {
            nodes.push(b);
        }
        nodes
    }

    pub const fn weight_node(&self) -> Option<NodeId> {
        self.weight_node
    }

    pub const fn bias_node(&self) -> Option<NodeId> {
        self.bias_node
    }

    pub const fn weight(&self) -> &Tensor {
        &self.weight_tensor
    }

    pub const fn bias(&self) -> &Tensor {
        &self.bias_tensor
    }
}
