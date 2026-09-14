use yscv_autograd::{Graph, NodeId};
use yscv_tensor::Tensor;

use super::super::recurrent::{
    GruCell, LstmCell, RnnCell, gru_forward_sequence, lstm_forward_sequence, rnn_forward_sequence,
};
use crate::ModelError;

/// RNN layer wrapping `rnn_forward_sequence`.
///
/// Input: `[seq_len, input_size]`, output: `[seq_len, hidden_size]`.
#[derive(Debug, Clone)]
pub struct RnnLayer {
    pub input_size: usize,
    pub hidden_size: usize,
    pub w_ih: Tensor,
    pub w_hh: Tensor,
    pub b_h: Tensor,
    w_ih_node: Option<NodeId>,
    w_hh_node: Option<NodeId>,
    b_h_node: Option<NodeId>,
}

impl RnnLayer {
    /// Creates an RNN layer with small random weights seeded by `seed`.
    pub fn new(input_size: usize, hidden_size: usize, seed: u64) -> Self {
        let scale = 1.0 / (hidden_size as f32).sqrt();
        let w_ih = Self::pseudo_random_tensor(vec![input_size, hidden_size], scale, seed);
        let w_hh =
            Self::pseudo_random_tensor(vec![hidden_size, hidden_size], scale, seed.wrapping_add(1));
        let b_h = Tensor::from_vec(vec![hidden_size], vec![0.0; hidden_size]).expect("valid bias");
        Self {
            input_size,
            hidden_size,
            w_ih,
            w_hh,
            b_h,
            w_ih_node: None,
            w_hh_node: None,
            b_h_node: None,
        }
    }

    pub(crate) fn pseudo_random_tensor(shape: Vec<usize>, scale: f32, seed: u64) -> Tensor {
        let len: usize = shape.iter().product();
        let mut data = Vec::with_capacity(len);
        let mut s = seed;
        for _ in 0..len {
            // Simple LCG PRNG
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v = ((s >> 33) as f32 / u32::MAX as f32 - 0.5) * 2.0 * scale;
            data.push(v);
        }
        Tensor::from_vec(shape, data).expect("valid tensor")
    }

    pub const fn w_ih_node(&self) -> Option<NodeId> {
        self.w_ih_node
    }

    pub fn register_params(&mut self, graph: &mut Graph) {
        self.w_ih_node = Some(graph.variable(self.w_ih.clone()));
        self.w_hh_node = Some(graph.variable(self.w_hh.clone()));
        self.b_h_node = Some(graph.variable(self.b_h.clone()));
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId, ModelError> {
        let w_ih = self
            .w_ih_node
            .ok_or(ModelError::ParamsNotRegistered { layer: "Rnn" })?;
        let w_hh = self
            .w_hh_node
            .ok_or(ModelError::ParamsNotRegistered { layer: "Rnn" })?;
        let bias = self
            .b_h_node
            .ok_or(ModelError::ParamsNotRegistered { layer: "Rnn" })?;
        graph
            .rnn_forward(input, w_ih, w_hh, bias)
            .map_err(Into::into)
    }

    /// Forward inference: input `[seq_len, input_size]` -> `[seq_len, hidden_size]`.
    pub fn forward_inference(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        let shape = input.shape();
        if shape.len() != 2 {
            return Err(ModelError::InvalidInputShape {
                expected_features: self.input_size,
                got: shape.to_vec(),
            });
        }
        let seq_len = shape[0];
        // Wrap as [1, seq_len, input_size] for rnn_forward_sequence
        let batched = input.reshape(vec![1, seq_len, self.input_size])?;
        let cell = RnnCell {
            w_ih: self.w_ih.clone(),
            w_hh: self.w_hh.clone(),
            bias: self.b_h.clone(),
            hidden_size: self.hidden_size,
        };
        let (output, _) = rnn_forward_sequence(&cell, &batched, None)?;
        // output is [1, seq_len, hidden_size], reshape to [seq_len, hidden_size]
        output
            .reshape(vec![seq_len, self.hidden_size])
            .map_err(Into::into)
    }
}

/// LSTM layer wrapping `lstm_forward_sequence`.
///
/// Input: `[seq_len, input_size]`, output: `[seq_len, hidden_size]`.
#[derive(Debug, Clone)]
pub struct LstmLayer {
    pub input_size: usize,
    pub hidden_size: usize,
    pub cell: LstmCell,
    w_ih_node: Option<NodeId>,
    w_hh_node: Option<NodeId>,
    bias_node: Option<NodeId>,
}

impl LstmLayer {
    pub fn new(input_size: usize, hidden_size: usize, seed: u64) -> Self {
        let h4 = 4 * hidden_size;
        let scale = 1.0 / (hidden_size as f32).sqrt();
        let w_ih = RnnLayer::pseudo_random_tensor(vec![input_size, h4], scale, seed);
        let w_hh =
            RnnLayer::pseudo_random_tensor(vec![hidden_size, h4], scale, seed.wrapping_add(1));
        let bias = Tensor::from_vec(vec![h4], vec![0.0; h4]).expect("valid bias");
        let cell = LstmCell {
            w_ih,
            w_hh,
            bias,
            hidden_size,
        };
        Self {
            input_size,
            hidden_size,
            cell,
            w_ih_node: None,
            w_hh_node: None,
            bias_node: None,
        }
    }

    pub const fn w_ih_node(&self) -> Option<NodeId> {
        self.w_ih_node
    }

    pub fn register_params(&mut self, graph: &mut Graph) {
        self.w_ih_node = Some(graph.variable(self.cell.w_ih.clone()));
        self.w_hh_node = Some(graph.variable(self.cell.w_hh.clone()));
        self.bias_node = Some(graph.variable(self.cell.bias.clone()));
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId, ModelError> {
        let w_ih = self
            .w_ih_node
            .ok_or(ModelError::ParamsNotRegistered { layer: "Lstm" })?;
        let w_hh = self
            .w_hh_node
            .ok_or(ModelError::ParamsNotRegistered { layer: "Lstm" })?;
        let bias = self
            .bias_node
            .ok_or(ModelError::ParamsNotRegistered { layer: "Lstm" })?;
        graph
            .lstm_forward(input, w_ih, w_hh, bias)
            .map_err(Into::into)
    }

    pub fn forward_inference(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        let shape = input.shape();
        if shape.len() != 2 {
            return Err(ModelError::InvalidInputShape {
                expected_features: self.input_size,
                got: shape.to_vec(),
            });
        }
        let seq_len = shape[0];
        let batched = input.reshape(vec![1, seq_len, self.input_size])?;
        let (output, _, _) = lstm_forward_sequence(&self.cell, &batched, None, None)?;
        output
            .reshape(vec![seq_len, self.hidden_size])
            .map_err(Into::into)
    }
}

/// GRU layer wrapping `gru_forward_sequence`.
///
/// Input: `[seq_len, input_size]`, output: `[seq_len, hidden_size]`.
#[derive(Debug, Clone)]
pub struct GruLayer {
    pub input_size: usize,
    pub hidden_size: usize,
    pub cell: GruCell,
    w_ih_node: Option<NodeId>,
    w_hh_node: Option<NodeId>,
    bias_ih_node: Option<NodeId>,
    bias_hh_node: Option<NodeId>,
}

impl GruLayer {
    pub fn new(input_size: usize, hidden_size: usize, seed: u64) -> Self {
        let h3 = 3 * hidden_size;
        let scale = 1.0 / (hidden_size as f32).sqrt();
        let w_ih = RnnLayer::pseudo_random_tensor(vec![input_size, h3], scale, seed);
        let w_hh =
            RnnLayer::pseudo_random_tensor(vec![hidden_size, h3], scale, seed.wrapping_add(1));
        let bias_ih = Tensor::from_vec(vec![h3], vec![0.0; h3]).expect("valid bias");
        let bias_hh = Tensor::from_vec(vec![h3], vec![0.0; h3]).expect("valid bias");
        let cell = GruCell {
            w_ih,
            w_hh,
            bias_ih,
            bias_hh,
            hidden_size,
        };
        Self {
            input_size,
            hidden_size,
            cell,
            w_ih_node: None,
            w_hh_node: None,
            bias_ih_node: None,
            bias_hh_node: None,
        }
    }

    pub const fn w_ih_node(&self) -> Option<NodeId> {
        self.w_ih_node
    }

    pub fn register_params(&mut self, graph: &mut Graph) {
        self.w_ih_node = Some(graph.variable(self.cell.w_ih.clone()));
        self.w_hh_node = Some(graph.variable(self.cell.w_hh.clone()));
        self.bias_ih_node = Some(graph.variable(self.cell.bias_ih.clone()));
        self.bias_hh_node = Some(graph.variable(self.cell.bias_hh.clone()));
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId, ModelError> {
        let w_ih = self
            .w_ih_node
            .ok_or(ModelError::ParamsNotRegistered { layer: "Gru" })?;
        let w_hh = self
            .w_hh_node
            .ok_or(ModelError::ParamsNotRegistered { layer: "Gru" })?;
        let bias_ih = self
            .bias_ih_node
            .ok_or(ModelError::ParamsNotRegistered { layer: "Gru" })?;
        let bias_hh = self
            .bias_hh_node
            .ok_or(ModelError::ParamsNotRegistered { layer: "Gru" })?;
        graph
            .gru_forward(input, w_ih, w_hh, bias_ih, bias_hh)
            .map_err(Into::into)
    }

    pub fn forward_inference(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        let shape = input.shape();
        if shape.len() != 2 {
            return Err(ModelError::InvalidInputShape {
                expected_features: self.input_size,
                got: shape.to_vec(),
            });
        }
        let seq_len = shape[0];
        let batched = input.reshape(vec![1, seq_len, self.input_size])?;
        let (output, _) = gru_forward_sequence(&self.cell, &batched, None)?;
        output
            .reshape(vec![seq_len, self.hidden_size])
            .map_err(Into::into)
    }
}
