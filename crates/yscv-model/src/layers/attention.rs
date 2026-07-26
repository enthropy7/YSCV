use yscv_autograd::{Graph, NodeId};
use yscv_tensor::Tensor;

use super::super::attention::{
    FeedForward, MultiHeadAttention, MultiHeadAttentionConfig, TransformerEncoderBlock,
};
use crate::ModelError;

/// Embedding lookup table: maps integer indices to dense vectors.
#[derive(Debug, Clone)]
pub struct EmbeddingLayer {
    num_embeddings: usize,
    embedding_dim: usize,
    weight: NodeId,
}

impl EmbeddingLayer {
    pub fn new(
        graph: &mut Graph,
        num_embeddings: usize,
        embedding_dim: usize,
        weight_init: Tensor,
    ) -> Result<Self, ModelError> {
        let expected = vec![num_embeddings, embedding_dim];
        if weight_init.shape() != expected {
            return Err(ModelError::InvalidParameterShape {
                parameter: "embedding_weight",
                expected,
                got: weight_init.shape().to_vec(),
            });
        }
        let weight = graph.variable(weight_init);
        Ok(Self {
            num_embeddings,
            embedding_dim,
            weight,
        })
    }

    pub fn num_embeddings(&self) -> usize {
        self.num_embeddings
    }
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
    pub fn weight_node(&self) -> NodeId {
        self.weight
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId, ModelError> {
        Ok(graph.embedding_lookup(self.weight, input)?)
    }

    pub fn forward_inference(&self, graph: &Graph, indices: &Tensor) -> Result<Tensor, ModelError> {
        let weight = graph.value(self.weight)?;
        let w_data = weight.data();
        let idx_data = indices.data();
        let batch = idx_data.len();
        let dim = self.embedding_dim;
        let mut out = vec![0.0f32; batch * dim];
        for (i, &idx_f) in idx_data.iter().enumerate() {
            let idx = idx_f as usize;
            if idx >= self.num_embeddings {
                return Err(ModelError::InvalidInputShape {
                    expected_features: self.num_embeddings,
                    got: indices.shape().to_vec(),
                });
            }
            out[i * dim..(i + 1) * dim].copy_from_slice(&w_data[idx * dim..(idx + 1) * dim]);
        }
        let mut shape = indices.shape().to_vec();
        shape.push(dim);
        Ok(Tensor::from_vec(shape, out)?)
    }
}

/// Multi-head attention layer wrapping `MultiHeadAttention`.
///
/// Self-attention: Q=K=V=input. Input/output: `[seq_len, d_model]`.
pub struct MultiHeadAttentionLayer {
    pub mha: MultiHeadAttention,
    w_q_node: Option<NodeId>,
    w_k_node: Option<NodeId>,
    w_v_node: Option<NodeId>,
    w_o_node: Option<NodeId>,
}

impl std::fmt::Debug for MultiHeadAttentionLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiHeadAttentionLayer")
            .field("num_heads", &self.mha.num_heads)
            .field("d_k", &self.mha.d_k)
            .finish()
    }
}

impl Clone for MultiHeadAttentionLayer {
    fn clone(&self) -> Self {
        Self {
            mha: MultiHeadAttention {
                w_q: self.mha.w_q.clone(),
                w_k: self.mha.w_k.clone(),
                w_v: self.mha.w_v.clone(),
                w_o: self.mha.w_o.clone(),
                num_heads: self.mha.num_heads,
                d_k: self.mha.d_k,
            },
            w_q_node: self.w_q_node,
            w_k_node: self.w_k_node,
            w_v_node: self.w_v_node,
            w_o_node: self.w_o_node,
        }
    }
}

impl MultiHeadAttentionLayer {
    pub fn w_q_node(&self) -> Option<NodeId> {
        self.w_q_node
    }

    pub fn new(d_model: usize, num_heads: usize, _seed: u64) -> Self {
        let config = MultiHeadAttentionConfig { d_model, num_heads };
        let mha = MultiHeadAttention::new(&config).expect("valid MHA config");
        Self {
            mha,
            w_q_node: None,
            w_k_node: None,
            w_v_node: None,
            w_o_node: None,
        }
    }

    pub fn register_params(&mut self, graph: &mut Graph) {
        self.w_q_node = Some(graph.variable(self.mha.w_q.clone()));
        self.w_k_node = Some(graph.variable(self.mha.w_k.clone()));
        self.w_v_node = Some(graph.variable(self.mha.w_v.clone()));
        self.w_o_node = Some(graph.variable(self.mha.w_o.clone()));
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId, ModelError> {
        let w_q = self.w_q_node.ok_or(ModelError::ParamsNotRegistered {
            layer: "MultiHeadAttention",
        })?;
        let w_k = self.w_k_node.ok_or(ModelError::ParamsNotRegistered {
            layer: "MultiHeadAttention",
        })?;
        let w_v = self.w_v_node.ok_or(ModelError::ParamsNotRegistered {
            layer: "MultiHeadAttention",
        })?;
        let w_o = self.w_o_node.ok_or(ModelError::ParamsNotRegistered {
            layer: "MultiHeadAttention",
        })?;

        // Project: Q = input @ W_q, K = input @ W_k, V = input @ W_v
        let q = graph.matmul_2d(input, w_q)?;
        let k = graph.matmul_2d(input, w_k)?;
        let v = graph.matmul_2d(input, w_v)?;

        // Per-head attention with narrow + scaled_dot_product_attention
        let d_k = self.mha.d_k;
        let num_heads = self.mha.num_heads;
        let mut head_outputs = Vec::new();
        for h in 0..num_heads {
            let start = h * d_k;
            let qh = graph.narrow(q, 1, start, d_k)?;
            let kh = graph.narrow(k, 1, start, d_k)?;
            let vh = graph.narrow(v, 1, start, d_k)?;
            let attn = graph.scaled_dot_product_attention(qh, kh, vh)?;
            head_outputs.push(attn);
        }

        // Concatenate heads along last dim -> [seq_len, d_model]
        let concat = graph.cat(&head_outputs, 1)?;

        // Output projection
        let output = graph.matmul_2d(concat, w_o)?;
        Ok(output)
    }

    pub fn forward_inference(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        self.mha.forward(input)
    }
}

/// Transformer encoder layer wrapping `TransformerEncoderBlock`.
///
/// Input/output: `[seq_len, d_model]`.
pub struct TransformerEncoderLayer {
    pub block: TransformerEncoderBlock,
    mha_layer: Option<MultiHeadAttentionLayer>,
    ff_layer: Option<FeedForwardLayer>,
    ln1_gamma_node: Option<NodeId>,
    ln1_beta_node: Option<NodeId>,
    ln2_gamma_node: Option<NodeId>,
    ln2_beta_node: Option<NodeId>,
}

impl std::fmt::Debug for TransformerEncoderLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransformerEncoderLayer")
            .field("d_model", &self.block.d_model)
            .finish()
    }
}

impl Clone for TransformerEncoderLayer {
    fn clone(&self) -> Self {
        Self {
            block: TransformerEncoderBlock {
                mha: MultiHeadAttention {
                    w_q: self.block.mha.w_q.clone(),
                    w_k: self.block.mha.w_k.clone(),
                    w_v: self.block.mha.w_v.clone(),
                    w_o: self.block.mha.w_o.clone(),
                    num_heads: self.block.mha.num_heads,
                    d_k: self.block.mha.d_k,
                },
                ffn: FeedForward {
                    w1: self.block.ffn.w1.clone(),
                    b1: self.block.ffn.b1.clone(),
                    w2: self.block.ffn.w2.clone(),
                    b2: self.block.ffn.b2.clone(),
                },
                ln1_gamma: self.block.ln1_gamma.clone(),
                ln1_beta: self.block.ln1_beta.clone(),
                ln2_gamma: self.block.ln2_gamma.clone(),
                ln2_beta: self.block.ln2_beta.clone(),
                d_model: self.block.d_model,
            },
            mha_layer: self.mha_layer.clone(),
            ff_layer: self.ff_layer.clone(),
            ln1_gamma_node: self.ln1_gamma_node,
            ln1_beta_node: self.ln1_beta_node,
            ln2_gamma_node: self.ln2_gamma_node,
            ln2_beta_node: self.ln2_beta_node,
        }
    }
}

impl TransformerEncoderLayer {
    pub fn ln1_gamma_node(&self) -> Option<NodeId> {
        self.ln1_gamma_node
    }

    pub fn new(d_model: usize, num_heads: usize, d_ff: usize, _seed: u64) -> Self {
        let block = TransformerEncoderBlock::new(d_model, num_heads, d_ff)
            .expect("valid TransformerEncoderBlock config");
        Self {
            block,
            mha_layer: None,
            ff_layer: None,
            ln1_gamma_node: None,
            ln1_beta_node: None,
            ln2_gamma_node: None,
            ln2_beta_node: None,
        }
    }

    pub fn register_params(&mut self, graph: &mut Graph) {
        let mut mha = MultiHeadAttentionLayer {
            mha: MultiHeadAttention {
                w_q: self.block.mha.w_q.clone(),
                w_k: self.block.mha.w_k.clone(),
                w_v: self.block.mha.w_v.clone(),
                w_o: self.block.mha.w_o.clone(),
                num_heads: self.block.mha.num_heads,
                d_k: self.block.mha.d_k,
            },
            w_q_node: None,
            w_k_node: None,
            w_v_node: None,
            w_o_node: None,
        };
        mha.register_params(graph);
        self.mha_layer = Some(mha);

        let mut ff = FeedForwardLayer {
            ff: FeedForward {
                w1: self.block.ffn.w1.clone(),
                b1: self.block.ffn.b1.clone(),
                w2: self.block.ffn.w2.clone(),
                b2: self.block.ffn.b2.clone(),
            },
            w1_node: None,
            b1_node: None,
            w2_node: None,
            b2_node: None,
        };
        ff.register_params(graph);
        self.ff_layer = Some(ff);

        self.ln1_gamma_node = Some(graph.variable(self.block.ln1_gamma.clone()));
        self.ln1_beta_node = Some(graph.variable(self.block.ln1_beta.clone()));
        self.ln2_gamma_node = Some(graph.variable(self.block.ln2_gamma.clone()));
        self.ln2_beta_node = Some(graph.variable(self.block.ln2_beta.clone()));
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId, ModelError> {
        let mha = self
            .mha_layer
            .as_ref()
            .ok_or(ModelError::ParamsNotRegistered {
                layer: "TransformerEncoder",
            })?;
        let ff = self
            .ff_layer
            .as_ref()
            .ok_or(ModelError::ParamsNotRegistered {
                layer: "TransformerEncoder",
            })?;
        let ln1_g = self.ln1_gamma_node.ok_or(ModelError::ParamsNotRegistered {
            layer: "TransformerEncoder",
        })?;
        let ln1_b = self.ln1_beta_node.ok_or(ModelError::ParamsNotRegistered {
            layer: "TransformerEncoder",
        })?;
        let ln2_g = self.ln2_gamma_node.ok_or(ModelError::ParamsNotRegistered {
            layer: "TransformerEncoder",
        })?;
        let ln2_b = self.ln2_beta_node.ok_or(ModelError::ParamsNotRegistered {
            layer: "TransformerEncoder",
        })?;

        let attn_out = mha.forward(graph, input)?;
        // norm1 = layer_norm(input + attn_out)
        let residual1 = graph.add(input, attn_out)?;
        let norm1 = graph.layer_norm(residual1, ln1_g, ln1_b, 1e-5)?;
        let ff_out = ff.forward(graph, norm1)?;
        // output = layer_norm(norm1 + ff_out)
        let residual2 = graph.add(norm1, ff_out)?;
        let output = graph.layer_norm(residual2, ln2_g, ln2_b, 1e-5)?;
        Ok(output)
    }

    pub fn forward_inference(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        self.block.forward(input)
    }
}

/// Feed-forward layer wrapping `FeedForward`.
///
/// Input/output: `[seq_len, d_model]`.
pub struct FeedForwardLayer {
    pub ff: FeedForward,
    w1_node: Option<NodeId>,
    b1_node: Option<NodeId>,
    w2_node: Option<NodeId>,
    b2_node: Option<NodeId>,
}

impl std::fmt::Debug for FeedForwardLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FeedForwardLayer").finish()
    }
}

impl Clone for FeedForwardLayer {
    fn clone(&self) -> Self {
        Self {
            ff: FeedForward {
                w1: self.ff.w1.clone(),
                b1: self.ff.b1.clone(),
                w2: self.ff.w2.clone(),
                b2: self.ff.b2.clone(),
            },
            w1_node: self.w1_node,
            b1_node: self.b1_node,
            w2_node: self.w2_node,
            b2_node: self.b2_node,
        }
    }
}

impl FeedForwardLayer {
    pub fn w1_node(&self) -> Option<NodeId> {
        self.w1_node
    }

    pub fn new(d_model: usize, d_ff: usize, _seed: u64) -> Self {
        let ff = FeedForward::new(d_model, d_ff).expect("valid FeedForward config");
        Self {
            ff,
            w1_node: None,
            b1_node: None,
            w2_node: None,
            b2_node: None,
        }
    }

    pub fn register_params(&mut self, graph: &mut Graph) {
        self.w1_node = Some(graph.variable(self.ff.w1.clone()));
        self.b1_node = Some(graph.variable(self.ff.b1.clone()));
        self.w2_node = Some(graph.variable(self.ff.w2.clone()));
        self.b2_node = Some(graph.variable(self.ff.b2.clone()));
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId, ModelError> {
        let w1 = self.w1_node.ok_or(ModelError::ParamsNotRegistered {
            layer: "FeedForward",
        })?;
        let b1 = self.b1_node.ok_or(ModelError::ParamsNotRegistered {
            layer: "FeedForward",
        })?;
        let w2 = self.w2_node.ok_or(ModelError::ParamsNotRegistered {
            layer: "FeedForward",
        })?;
        let b2 = self.b2_node.ok_or(ModelError::ParamsNotRegistered {
            layer: "FeedForward",
        })?;

        // output = relu(input @ w1 + b1) @ w2 + b2
        let h = graph.matmul_2d(input, w1)?;
        let h = graph.add(h, b1)?;
        let h = graph.relu(h)?;
        let h = graph.matmul_2d(h, w2)?;
        let out = graph.add(h, b2)?;
        Ok(out)
    }

    pub fn forward_inference(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        self.ff.forward(input)
    }
}
