mod activation;
mod attention;
mod conv;
mod linear;
mod misc;
mod norm;
mod pool;
mod recurrent;

use yscv_autograd::{Graph, NodeId};
use yscv_tensor::Tensor;

use super::lora::LoraLinear;
use crate::ModelError;

// Re-export all public types from sub-modules.
pub use activation::{
    GELULayer, LeakyReLULayer, MishLayer, PReLULayer, ReLULayer, SiLULayer, SigmoidLayer, TanhLayer,
};
pub use attention::{
    EmbeddingLayer, FeedForwardLayer, MultiHeadAttentionLayer, TransformerEncoderLayer,
};
pub use conv::{
    Conv1dLayer, Conv2dLayer, Conv3dLayer, ConvTranspose2dLayer, DeformableConv2dLayer,
    DepthwiseConv2dLayer, SeparableConv2dLayer,
};
pub use linear::LinearLayer;
pub use misc::{
    DropoutLayer, FlattenLayer, MaskHead, PixelShuffleLayer, ResidualBlock, SoftmaxLayer,
    UpsampleLayer,
};
pub use norm::{BatchNorm2dLayer, GroupNormLayer, InstanceNormLayer, LayerNormLayer};
pub use pool::{
    AdaptiveAvgPool2dLayer, AdaptiveMaxPool2dLayer, AvgPool2dLayer, GlobalAvgPool2dLayer,
    MaxPool2dLayer,
};
pub use recurrent::{GruLayer, LstmLayer, RnnLayer};

#[derive(Debug, Clone)]
pub enum ModelLayer {
    Linear(LinearLayer),
    ReLU(ReLULayer),
    LeakyReLU(LeakyReLULayer),
    Sigmoid(SigmoidLayer),
    Tanh(TanhLayer),
    Dropout(DropoutLayer),
    Conv2d(Conv2dLayer),
    BatchNorm2d(BatchNorm2dLayer),
    MaxPool2d(MaxPool2dLayer),
    AvgPool2d(AvgPool2dLayer),
    GlobalAvgPool2d(GlobalAvgPool2dLayer),
    Flatten(FlattenLayer),
    Softmax(SoftmaxLayer),
    Embedding(EmbeddingLayer),
    LayerNorm(LayerNormLayer),
    GroupNorm(GroupNormLayer),
    DepthwiseConv2d(DepthwiseConv2dLayer),
    SeparableConv2d(SeparableConv2dLayer),
    LoraLinear(LoraLinear),
    Conv1d(Conv1dLayer),
    Conv3d(Conv3dLayer),
    ConvTranspose2d(ConvTranspose2dLayer),
    AdaptiveAvgPool2d(AdaptiveAvgPool2dLayer),
    AdaptiveMaxPool2d(AdaptiveMaxPool2dLayer),
    InstanceNorm(InstanceNormLayer),
    PixelShuffle(PixelShuffleLayer),
    Upsample(UpsampleLayer),
    GELU(GELULayer),
    SiLU(SiLULayer),
    Mish(MishLayer),
    PReLU(PReLULayer),
    ResidualBlock(ResidualBlock),
    Rnn(RnnLayer),
    Lstm(LstmLayer),
    Gru(GruLayer),
    MultiHeadAttention(MultiHeadAttentionLayer),
    TransformerEncoder(TransformerEncoderLayer),
    FeedForward(FeedForwardLayer),
    DeformableConv2d(DeformableConv2dLayer),
}

impl ModelLayer {
    pub(crate) fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId, ModelError> {
        match self {
            Self::Linear(layer) => layer.forward(graph, input),
            Self::ReLU(layer) => layer.forward(graph, input),
            Self::LeakyReLU(layer) => layer.forward(graph, input),
            Self::Sigmoid(layer) => layer.forward(graph, input),
            Self::Tanh(layer) => layer.forward(graph, input),
            Self::Dropout(layer) => layer.forward(graph, input),
            Self::Flatten(layer) => layer.forward(graph, input),
            Self::Conv2d(layer) => layer.forward(graph, input),
            Self::BatchNorm2d(layer) => layer.forward(graph, input),
            Self::MaxPool2d(layer) => layer.forward(graph, input),
            Self::AvgPool2d(layer) => layer.forward(graph, input),
            Self::GlobalAvgPool2d(layer) => layer.forward(graph, input),
            Self::Embedding(layer) => layer.forward(graph, input),
            Self::DepthwiseConv2d(layer) => layer.forward(graph, input),
            Self::SeparableConv2d(layer) => layer.forward(graph, input),
            Self::LoraLinear(layer) => layer.forward(graph, input),
            Self::GELU(layer) => layer.forward(graph, input),
            Self::SiLU(layer) => layer.forward(graph, input),
            Self::Mish(layer) => layer.forward(graph, input),
            Self::LayerNorm(layer) => layer.forward(graph, input),
            Self::GroupNorm(layer) => layer.forward(graph, input),
            Self::Conv1d(layer) => layer.forward(graph, input),
            Self::Conv3d(layer) => layer.forward(graph, input),
            Self::MultiHeadAttention(layer) => layer.forward(graph, input),
            Self::ConvTranspose2d(layer) => layer.forward(graph, input),
            Self::AdaptiveAvgPool2d(layer) => layer.forward(graph, input),
            Self::AdaptiveMaxPool2d(layer) => layer.forward(graph, input),
            Self::InstanceNorm(layer) => layer.forward(graph, input),
            Self::PReLU(layer) => layer.forward(graph, input),
            Self::Softmax(layer) => layer.forward(graph, input),
            Self::PixelShuffle(layer) => layer.forward(graph, input),
            Self::Upsample(layer) => layer.forward(graph, input),
            Self::ResidualBlock(layer) => layer.forward(graph, input),
            Self::Rnn(layer) => layer.forward(graph, input),
            Self::Lstm(layer) => layer.forward(graph, input),
            Self::Gru(layer) => layer.forward(graph, input),
            Self::TransformerEncoder(layer) => layer.forward(graph, input),
            Self::FeedForward(layer) => layer.forward(graph, input),
            Self::DeformableConv2d(layer) => layer.forward(graph, input),
        }
    }

    pub fn forward_inference(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        match self {
            Self::Conv2d(layer) => layer.forward_inference(input),
            Self::BatchNorm2d(layer) => layer.forward_inference(input),
            Self::MaxPool2d(layer) => layer.forward_inference(input),
            Self::AvgPool2d(layer) => layer.forward_inference(input),
            Self::GlobalAvgPool2d(layer) => layer.forward_inference(input),
            Self::Flatten(layer) => layer.forward_inference(input),
            Self::Softmax(layer) => layer.forward_inference(input),
            Self::DepthwiseConv2d(layer) => layer.forward_inference(input),
            Self::SeparableConv2d(layer) => layer.forward_inference(input),
            Self::Conv1d(layer) => layer.forward_inference(input),
            Self::Conv3d(layer) => layer.forward_inference(input),
            Self::ConvTranspose2d(layer) => layer.forward_inference(input),
            Self::AdaptiveAvgPool2d(layer) => layer.forward_inference(input),
            Self::AdaptiveMaxPool2d(layer) => layer.forward_inference(input),
            Self::InstanceNorm(layer) => layer.forward_inference(input),
            Self::PixelShuffle(layer) => layer.forward_inference(input),
            Self::Upsample(layer) => layer.forward_inference(input),
            Self::GELU(layer) => layer.forward_inference(input),
            Self::SiLU(layer) => layer.forward_inference(input),
            Self::Mish(layer) => layer.forward_inference(input),
            Self::PReLU(layer) => layer.forward_inference(input),
            Self::ResidualBlock(layer) => layer.forward_inference(input),
            Self::Rnn(layer) => layer.forward_inference(input),
            Self::Lstm(layer) => layer.forward_inference(input),
            Self::Gru(layer) => layer.forward_inference(input),
            Self::MultiHeadAttention(layer) => layer.forward_inference(input),
            Self::TransformerEncoder(layer) => layer.forward_inference(input),
            Self::FeedForward(layer) => layer.forward_inference(input),
            Self::DeformableConv2d(layer) => layer.forward_inference(input),
            Self::Linear(layer) => layer.forward_inference(input),
            Self::ReLU(_)
            | Self::LeakyReLU(_)
            | Self::Sigmoid(_)
            | Self::Tanh(_)
            | Self::Dropout(_)
            | Self::Embedding(_)
            | Self::LayerNorm(_)
            | Self::GroupNorm(_)
            | Self::LoraLinear(_) => Err(ModelError::GraphOnlyLayer),
        }
    }

    pub const fn supports_graph_forward(&self) -> bool {
        // Bilinear upsample still requires inference-only mode.
        if let Self::Upsample(u) = self {
            !u.is_bilinear()
        } else {
            true
        }
    }

    pub const fn supports_inference_forward(&self) -> bool {
        matches!(
            self,
            Self::Conv2d(_)
                | Self::BatchNorm2d(_)
                | Self::MaxPool2d(_)
                | Self::AvgPool2d(_)
                | Self::GlobalAvgPool2d(_)
                | Self::Flatten(_)
                | Self::Softmax(_)
                | Self::DepthwiseConv2d(_)
                | Self::SeparableConv2d(_)
                | Self::Conv1d(_)
                | Self::Conv3d(_)
                | Self::ConvTranspose2d(_)
                | Self::AdaptiveAvgPool2d(_)
                | Self::AdaptiveMaxPool2d(_)
                | Self::InstanceNorm(_)
                | Self::PixelShuffle(_)
                | Self::Upsample(_)
                | Self::GELU(_)
                | Self::SiLU(_)
                | Self::Mish(_)
                | Self::PReLU(_)
                | Self::ResidualBlock(_)
                | Self::Rnn(_)
                | Self::Lstm(_)
                | Self::Gru(_)
                | Self::MultiHeadAttention(_)
                | Self::TransformerEncoder(_)
                | Self::FeedForward(_)
                | Self::DeformableConv2d(_)
        )
    }
}
