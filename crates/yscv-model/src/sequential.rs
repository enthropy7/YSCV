use std::collections::HashMap;
use yscv_autograd::{Graph, NodeId};
use yscv_tensor::Tensor;

use crate::{
    AvgPool2dLayer, BatchNorm2dLayer, Conv2dLayer, DeformableConv2dLayer, DepthwiseConv2dLayer,
    DropoutLayer, EmbeddingLayer, FeedForwardLayer, FlattenLayer, GlobalAvgPool2dLayer,
    GroupNormLayer, GruLayer, LayerCheckpoint, LayerNormLayer, LeakyReLULayer, LinearLayer,
    LoraConfig, LoraLinear, LstmLayer, MaxPool2dLayer, ModelError, ModelLayer,
    MultiHeadAttentionLayer, ReLULayer, ResidualBlock, RnnLayer, SeparableConv2dLayer,
    SequentialCheckpoint, SigmoidLayer, SoftmaxLayer, TanhLayer, TensorSnapshot,
    TransformerEncoderLayer, optimize_sequential,
};

/// Ordered stack of layers executed one-by-one.
#[derive(Debug, Clone)]
pub struct SequentialModel {
    layers: Vec<ModelLayer>,
    frozen: Vec<bool>,
    persistent_node_count: usize,
    training: bool,
}

impl SequentialModel {
    /// Creates an empty model and records current graph prefix as persistent base.
    pub fn new(graph: &Graph) -> Self {
        Self {
            layers: Vec::new(),
            frozen: Vec::new(),
            persistent_node_count: graph.node_count(),
            training: true,
        }
    }

    pub fn add_linear(
        &mut self,
        graph: &mut Graph,
        in_features: usize,
        out_features: usize,
        weight_init: Tensor,
        bias_init: Tensor,
    ) -> Result<(), ModelError> {
        let layer = LinearLayer::new(graph, in_features, out_features, weight_init, bias_init)?;
        self.layers.push(ModelLayer::Linear(layer));
        self.frozen.push(false);
        self.persistent_node_count = graph.node_count();
        Ok(())
    }

    pub fn add_linear_zero(
        &mut self,
        graph: &mut Graph,
        in_features: usize,
        out_features: usize,
    ) -> Result<(), ModelError> {
        let layer = LinearLayer::zero_init(graph, in_features, out_features)?;
        self.layers.push(ModelLayer::Linear(layer));
        self.frozen.push(false);
        self.persistent_node_count = graph.node_count();
        Ok(())
    }

    pub fn add_relu(&mut self) {
        self.layers.push(ModelLayer::ReLU(ReLULayer::new()));
        self.frozen.push(false);
    }

    pub fn add_leaky_relu(&mut self, negative_slope: f32) -> Result<(), ModelError> {
        let layer = LeakyReLULayer::new(negative_slope)?;
        self.layers.push(ModelLayer::LeakyReLU(layer));
        self.frozen.push(false);
        Ok(())
    }

    pub fn add_sigmoid(&mut self) {
        self.layers.push(ModelLayer::Sigmoid(SigmoidLayer::new()));
        self.frozen.push(false);
    }

    pub fn add_tanh(&mut self) {
        self.layers.push(ModelLayer::Tanh(TanhLayer::new()));
        self.frozen.push(false);
    }

    pub fn add_gelu(&mut self) {
        self.layers.push(ModelLayer::GELU(crate::GELULayer::new()));
        self.frozen.push(false);
    }

    pub fn add_silu(&mut self) {
        self.layers.push(ModelLayer::SiLU(crate::SiLULayer::new()));
        self.frozen.push(false);
    }

    pub fn add_mish(&mut self) {
        self.layers.push(ModelLayer::Mish(crate::MishLayer::new()));
        self.frozen.push(false);
    }

    pub fn add_prelu(&mut self, alpha: Vec<f32>) {
        self.layers
            .push(ModelLayer::PReLU(crate::PReLULayer::new(alpha)));
        self.frozen.push(false);
    }

    pub fn add_dropout(&mut self, rate: f32) -> Result<(), ModelError> {
        let layer = DropoutLayer::new(rate)?;
        self.layers.push(ModelLayer::Dropout(layer));
        self.frozen.push(false);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_conv2d(
        &mut self,
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        weight: Tensor,
        bias: Option<Tensor>,
    ) -> Result<(), ModelError> {
        let layer = Conv2dLayer::new(
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            weight,
            bias,
        )?;
        self.layers.push(ModelLayer::Conv2d(layer));
        self.frozen.push(false);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_conv2d_zero(
        &mut self,
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        use_bias: bool,
    ) -> Result<(), ModelError> {
        let layer = Conv2dLayer::zero_init(
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            use_bias,
        )?;
        self.layers.push(ModelLayer::Conv2d(layer));
        self.frozen.push(false);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_deformable_conv2d(
        &mut self,
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride: usize,
        padding: usize,
        weight: Tensor,
        offset_weight: Tensor,
        bias: Option<Tensor>,
    ) -> Result<(), ModelError> {
        let layer = DeformableConv2dLayer::new(
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            stride,
            padding,
            weight,
            offset_weight,
            bias,
        )?;
        self.layers.push(ModelLayer::DeformableConv2d(layer));
        self.frozen.push(false);
        Ok(())
    }

    pub fn add_deformable_conv2d_zero(
        &mut self,
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride: usize,
        padding: usize,
        use_bias: bool,
    ) -> Result<(), ModelError> {
        let layer = DeformableConv2dLayer::zero_init(
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            stride,
            padding,
            use_bias,
        )?;
        self.layers.push(ModelLayer::DeformableConv2d(layer));
        self.frozen.push(false);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_depthwise_conv2d(
        &mut self,
        channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        weight: Tensor,
        bias: Option<Tensor>,
    ) -> Result<(), ModelError> {
        let layer = DepthwiseConv2dLayer::new(
            channels, kernel_h, kernel_w, stride_h, stride_w, weight, bias,
        )?;
        self.layers.push(ModelLayer::DepthwiseConv2d(layer));
        self.frozen.push(false);
        Ok(())
    }

    pub fn add_depthwise_conv2d_zero(
        &mut self,
        channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        use_bias: bool,
    ) -> Result<(), ModelError> {
        let layer = DepthwiseConv2dLayer::zero_init(
            channels, kernel_h, kernel_w, stride_h, stride_w, use_bias,
        )?;
        self.layers.push(ModelLayer::DepthwiseConv2d(layer));
        self.frozen.push(false);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_separable_conv2d(
        &mut self,
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        depthwise_weight: Tensor,
        pointwise_weight: Tensor,
        bias: Option<Tensor>,
    ) -> Result<(), ModelError> {
        let layer = SeparableConv2dLayer::new(
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            depthwise_weight,
            pointwise_weight,
            bias,
        )?;
        self.layers.push(ModelLayer::SeparableConv2d(layer));
        self.frozen.push(false);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_separable_conv2d_zero(
        &mut self,
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        use_bias: bool,
    ) -> Result<(), ModelError> {
        let layer = SeparableConv2dLayer::zero_init(
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            use_bias,
        )?;
        self.layers.push(ModelLayer::SeparableConv2d(layer));
        self.frozen.push(false);
        Ok(())
    }

    pub fn add_batch_norm2d(
        &mut self,
        num_features: usize,
        epsilon: f32,
        gamma: Tensor,
        beta: Tensor,
        running_mean: Tensor,
        running_var: Tensor,
    ) -> Result<(), ModelError> {
        let layer = BatchNorm2dLayer::new(
            num_features,
            epsilon,
            gamma,
            beta,
            running_mean,
            running_var,
        )?;
        self.layers.push(ModelLayer::BatchNorm2d(layer));
        self.frozen.push(false);
        Ok(())
    }

    pub fn add_batch_norm2d_identity(
        &mut self,
        num_features: usize,
        epsilon: f32,
    ) -> Result<(), ModelError> {
        let layer = BatchNorm2dLayer::identity_init(num_features, epsilon)?;
        self.layers.push(ModelLayer::BatchNorm2d(layer));
        self.frozen.push(false);
        Ok(())
    }

    pub fn add_max_pool2d(
        &mut self,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<(), ModelError> {
        let layer = MaxPool2dLayer::new(kernel_h, kernel_w, stride_h, stride_w)?;
        self.layers.push(ModelLayer::MaxPool2d(layer));
        self.frozen.push(false);
        Ok(())
    }

    pub fn add_avg_pool2d(
        &mut self,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<(), ModelError> {
        let layer = AvgPool2dLayer::new(kernel_h, kernel_w, stride_h, stride_w)?;
        self.layers.push(ModelLayer::AvgPool2d(layer));
        self.frozen.push(false);
        Ok(())
    }

    pub fn add_flatten(&mut self) {
        self.layers.push(ModelLayer::Flatten(FlattenLayer::new()));
        self.frozen.push(false);
    }

    pub fn add_global_avg_pool2d(&mut self) {
        self.layers
            .push(ModelLayer::GlobalAvgPool2d(GlobalAvgPool2dLayer::new()));
        self.frozen.push(false);
    }

    pub fn add_softmax(&mut self) {
        self.layers.push(ModelLayer::Softmax(SoftmaxLayer::new()));
        self.frozen.push(false);
    }

    pub fn add_embedding(
        &mut self,
        graph: &mut Graph,
        num_embeddings: usize,
        embedding_dim: usize,
        weight_init: Tensor,
    ) -> Result<(), ModelError> {
        let layer = EmbeddingLayer::new(graph, num_embeddings, embedding_dim, weight_init)?;
        self.layers.push(ModelLayer::Embedding(layer));
        self.frozen.push(false);
        Ok(())
    }

    pub fn add_layer_norm(
        &mut self,
        graph: &mut Graph,
        normalized_shape: usize,
        eps: f32,
    ) -> Result<(), ModelError> {
        let layer = LayerNormLayer::new(graph, normalized_shape, eps)?;
        self.layers.push(ModelLayer::LayerNorm(layer));
        self.frozen.push(false);
        Ok(())
    }

    pub fn add_group_norm(
        &mut self,
        graph: &mut Graph,
        num_groups: usize,
        num_channels: usize,
        eps: f32,
    ) -> Result<(), ModelError> {
        let layer = GroupNormLayer::new(graph, num_groups, num_channels, eps)?;
        self.layers.push(ModelLayer::GroupNorm(layer));
        self.frozen.push(false);
        Ok(())
    }

    /// Replace all Linear layers with LoRA-adapted versions.
    /// The original weights are frozen; only the low-rank A and B matrices are trainable.
    /// Returns the number of layers converted.
    pub fn apply_lora(
        &mut self,
        graph: &mut Graph,
        config: &LoraConfig,
    ) -> Result<usize, ModelError> {
        let mut count = 0;
        for layer in self.layers.iter_mut() {
            if let ModelLayer::Linear(linear) = layer {
                let in_features = linear.in_features();
                let out_features = linear.out_features();
                let weight_node = linear.weight_node().expect("linear layer has weight node");
                let bias_node = linear.bias_node().expect("linear layer has bias node");
                let lora = LoraLinear::from_linear(
                    graph,
                    weight_node,
                    bias_node,
                    in_features,
                    out_features,
                    config,
                )?;
                *layer = ModelLayer::LoraLinear(lora);
                count += 1;
            }
        }
        self.persistent_node_count = graph.node_count();
        Ok(count)
    }

    /// Merge all LoRA layers back into regular Linear layers.
    /// Call this after fine-tuning for inference without overhead.
    /// Returns the number of layers merged.
    pub fn merge_lora(&mut self, graph: &mut Graph) -> Result<usize, ModelError> {
        let mut count = 0;
        for layer in self.layers.iter_mut() {
            if let ModelLayer::LoraLinear(lora) = layer {
                let merged_weight = lora.merge(graph)?;
                let in_features = lora.in_features;
                let out_features = lora.out_features;
                let bias_tensor = if let Some(bias_node) = lora.bias {
                    graph.value(bias_node)?.clone()
                } else {
                    Tensor::zeros(vec![out_features])?
                };
                let linear =
                    LinearLayer::new(graph, in_features, out_features, merged_weight, bias_tensor)?;
                *layer = ModelLayer::Linear(linear);
                count += 1;
            }
        }
        self.persistent_node_count = graph.node_count();
        Ok(count)
    }

    /// Optimize the model by fusing Conv+BN layers.
    /// This reduces the number of layers and operations for faster inference.
    /// Should be called before inference, after training is complete.
    /// Returns the number of fusions performed.
    pub fn optimize(&mut self, graph: &mut Graph) -> usize {
        let before = self.layers.len();
        let optimized = optimize_sequential(self, graph);
        let after = optimized.layers.len();
        *self = optimized;
        before - after
    }

    /// Set training/eval mode for all dropout layers.
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
        for layer in &mut self.layers {
            if let ModelLayer::Dropout(d) = layer {
                d.set_training(training);
            }
        }
    }

    /// Switch to evaluation mode (disables dropout).
    pub fn eval(&mut self) {
        self.set_training(false);
    }

    /// Switch to training mode (enables dropout).
    pub fn train_mode(&mut self) {
        self.set_training(true);
    }

    /// Returns whether the model is in training mode.
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Print a human-readable summary of the model architecture.
    ///
    /// Shows each layer's type, output shape info, and parameter count.
    /// Returns a `String` so callers can print or log it.
    pub fn summary(&self) -> String {
        use std::fmt::Write;

        let mut out = String::new();
        let sep = "-".repeat(65);
        writeln!(out, "{sep}").expect("write to String");
        writeln!(
            out,
            " {:>3}  {:<25} {:>20}  {:>8}",
            "#", "Layer", "Details", "Params"
        )
        .expect("write to String");
        writeln!(out, "{sep}").expect("write to String");

        let mut total_params = 0usize;

        for (i, layer) in self.layers.iter().enumerate() {
            let (name, details, params) = match layer {
                ModelLayer::Linear(l) => {
                    let p = l.in_features() * l.out_features() + l.out_features();
                    (
                        "Linear".to_string(),
                        format!("{}→{}", l.in_features(), l.out_features()),
                        p,
                    )
                }
                ModelLayer::Conv2d(l) => {
                    let p = l.out_channels() * l.in_channels() * l.kernel_h() * l.kernel_w()
                        + if l.bias().is_some() {
                            l.out_channels()
                        } else {
                            0
                        };
                    (
                        "Conv2d".to_string(),
                        format!(
                            "{}→{} {}x{} s{}",
                            l.in_channels(),
                            l.out_channels(),
                            l.kernel_h(),
                            l.kernel_w(),
                            l.stride_h(),
                        ),
                        p,
                    )
                }
                ModelLayer::BatchNorm2d(l) => {
                    let p = l.num_features() * 2; // gamma + beta
                    (
                        "BatchNorm2d".to_string(),
                        format!("{}", l.num_features()),
                        p,
                    )
                }
                ModelLayer::MaxPool2d(l) => (
                    "MaxPool2d".to_string(),
                    format!("{}x{} s{}", l.kernel_h(), l.kernel_w(), l.stride_h()),
                    0,
                ),
                ModelLayer::AvgPool2d(l) => (
                    "AvgPool2d".to_string(),
                    format!("{}x{} s{}", l.kernel_h(), l.kernel_w(), l.stride_h()),
                    0,
                ),
                ModelLayer::GlobalAvgPool2d(_) => ("GlobalAvgPool2d".to_string(), String::new(), 0),
                ModelLayer::Flatten(_) => ("Flatten".to_string(), String::new(), 0),
                ModelLayer::ReLU(_) => ("ReLU".to_string(), String::new(), 0),
                ModelLayer::LeakyReLU(_) => ("LeakyReLU".to_string(), String::new(), 0),
                ModelLayer::Sigmoid(_) => ("Sigmoid".to_string(), String::new(), 0),
                ModelLayer::Tanh(_) => ("Tanh".to_string(), String::new(), 0),
                ModelLayer::Softmax(_) => ("Softmax".to_string(), String::new(), 0),
                ModelLayer::Dropout(d) => ("Dropout".to_string(), format!("p={:.2}", d.rate()), 0),
                ModelLayer::Embedding(e) => {
                    let p = e.num_embeddings() * e.embedding_dim();
                    (
                        "Embedding".to_string(),
                        format!("{}x{}", e.num_embeddings(), e.embedding_dim()),
                        p,
                    )
                }
                ModelLayer::LayerNorm(_) => ("LayerNorm".to_string(), String::new(), 0),
                ModelLayer::GroupNorm(_) => ("GroupNorm".to_string(), String::new(), 0),
                ModelLayer::DepthwiseConv2d(_) => ("DepthwiseConv2d".to_string(), String::new(), 0),
                ModelLayer::SeparableConv2d(_) => ("SeparableConv2d".to_string(), String::new(), 0),
                ModelLayer::DeformableConv2d(l) => {
                    let p = l.weight.data().len() + l.offset_weight.data().len();
                    (
                        "DeformableConv2d".to_string(),
                        format!(
                            "{}→{} {}x{}",
                            l.in_channels(),
                            l.out_channels(),
                            l.kernel_h,
                            l.kernel_w
                        ),
                        p,
                    )
                }
                ModelLayer::LoraLinear(l) => (
                    "LoraLinear".to_string(),
                    format!("{}→{} r={}", l.in_features, l.out_features, l.rank),
                    l.in_features * l.rank + l.rank * l.out_features,
                ),
                ModelLayer::Conv1d(l) => {
                    let p = l.kernel().data().len();
                    ("Conv1d".to_string(), format!("k={}", l.kernel_size()), p)
                }
                ModelLayer::Conv3d(l) => {
                    let p = l.weight().data().len();
                    (
                        "Conv3d".to_string(),
                        format!("{}→{}", l.in_channels(), l.out_channels()),
                        p,
                    )
                }
                ModelLayer::ConvTranspose2d(l) => {
                    let p = l.kernel().data().len();
                    (
                        "ConvTranspose2d".to_string(),
                        format!("s={}", l.stride()),
                        p,
                    )
                }
                ModelLayer::AdaptiveAvgPool2d(l) => (
                    "AdaptiveAvgPool2d".to_string(),
                    format!("{}x{}", l.output_h(), l.output_w()),
                    0,
                ),
                ModelLayer::AdaptiveMaxPool2d(l) => (
                    "AdaptiveMaxPool2d".to_string(),
                    format!("{}x{}", l.output_h(), l.output_w()),
                    0,
                ),
                ModelLayer::InstanceNorm(_) => ("InstanceNorm".to_string(), String::new(), 0),
                ModelLayer::PixelShuffle(l) => (
                    "PixelShuffle".to_string(),
                    format!("r={}", l.upscale_factor()),
                    0,
                ),
                ModelLayer::Upsample(l) => {
                    ("Upsample".to_string(), format!("{}x", l.scale_factor()), 0)
                }
                ModelLayer::GELU(_) => ("GELU".to_string(), String::new(), 0),
                ModelLayer::SiLU(_) => ("SiLU".to_string(), String::new(), 0),
                ModelLayer::Mish(_) => ("Mish".to_string(), String::new(), 0),
                ModelLayer::PReLU(l) => (
                    "PReLU".to_string(),
                    format!("channels={}", l.alpha().len()),
                    l.alpha().len(),
                ),
                ModelLayer::ResidualBlock(r) => (
                    "ResidualBlock".to_string(),
                    format!("{} layers", r.layers().len()),
                    0,
                ),
                ModelLayer::Rnn(l) => {
                    let p = l.input_size * l.hidden_size
                        + l.hidden_size * l.hidden_size
                        + l.hidden_size;
                    (
                        "Rnn".to_string(),
                        format!("{}→{}", l.input_size, l.hidden_size),
                        p,
                    )
                }
                ModelLayer::Lstm(l) => {
                    let h4 = 4 * l.hidden_size;
                    let p = l.input_size * h4 + l.hidden_size * h4 + h4;
                    (
                        "Lstm".to_string(),
                        format!("{}→{}", l.input_size, l.hidden_size),
                        p,
                    )
                }
                ModelLayer::Gru(l) => {
                    let h3 = 3 * l.hidden_size;
                    let p = l.input_size * h3 + l.hidden_size * h3 + 2 * h3;
                    (
                        "Gru".to_string(),
                        format!("{}→{}", l.input_size, l.hidden_size),
                        p,
                    )
                }
                ModelLayer::MultiHeadAttention(_) => {
                    ("MultiHeadAttention".to_string(), String::new(), 0)
                }
                ModelLayer::TransformerEncoder(_) => {
                    ("TransformerEncoder".to_string(), String::new(), 0)
                }
                ModelLayer::FeedForward(_) => ("FeedForward".to_string(), String::new(), 0),
            };

            total_params += params;
            let params_str = if params > 0 {
                format_param_count(params)
            } else {
                "-".to_string()
            };
            writeln!(
                out,
                " {:>3}  {:<25} {:>20}  {:>8}",
                i, name, details, params_str
            )
            .expect("write to String");
        }

        writeln!(out, "{sep}").expect("write to String");
        writeln!(
            out,
            " Total: {} layers, {} parameters",
            self.layers.len(),
            format_param_count(total_params)
        )
        .expect("write to String");
        writeln!(out, "{sep}").expect("write to String");
        out
    }

    /// Returns the total number of parameters (weights + biases) across all layers.
    pub fn num_parameters(&self) -> usize {
        let mut total = 0usize;
        for layer in &self.layers {
            let params = match layer {
                ModelLayer::Linear(l) => l.in_features() * l.out_features() + l.out_features(),
                ModelLayer::Conv2d(l) => {
                    l.out_channels() * l.in_channels() * l.kernel_h() * l.kernel_w()
                        + if l.bias().is_some() {
                            l.out_channels()
                        } else {
                            0
                        }
                }
                ModelLayer::BatchNorm2d(l) => l.num_features() * 2,
                ModelLayer::Embedding(e) => e.num_embeddings() * e.embedding_dim(),
                ModelLayer::LoraLinear(l) => l.in_features * l.rank + l.rank * l.out_features,
                ModelLayer::Conv1d(l) => l.kernel().data().len(),
                ModelLayer::Conv3d(l) => l.weight().data().len(),
                ModelLayer::ConvTranspose2d(l) => l.kernel().data().len(),
                ModelLayer::DepthwiseConv2d(l) => {
                    l.weight().data().len() + l.bias().map_or(0, |b| b.data().len())
                }
                ModelLayer::SeparableConv2d(l) => {
                    l.depthwise().weight().data().len()
                        + l.depthwise().bias().map_or(0, |b| b.data().len())
                        + l.pointwise().weight().data().len()
                        + l.pointwise().bias().map_or(0, |b| b.data().len())
                }
                ModelLayer::DeformableConv2d(l) => {
                    l.weight.data().len()
                        + l.offset_weight.data().len()
                        + l.bias.as_ref().map_or(0, |b| b.data().len())
                }
                ModelLayer::InstanceNorm(_) => 0,
                ModelLayer::LayerNorm(_)
                | ModelLayer::GroupNorm(_)
                | ModelLayer::MaxPool2d(_)
                | ModelLayer::AvgPool2d(_)
                | ModelLayer::GlobalAvgPool2d(_)
                | ModelLayer::Flatten(_)
                | ModelLayer::ReLU(_)
                | ModelLayer::LeakyReLU(_)
                | ModelLayer::Sigmoid(_)
                | ModelLayer::Tanh(_)
                | ModelLayer::Softmax(_)
                | ModelLayer::Dropout(_)
                | ModelLayer::AdaptiveAvgPool2d(_)
                | ModelLayer::AdaptiveMaxPool2d(_)
                | ModelLayer::PixelShuffle(_)
                | ModelLayer::Upsample(_)
                | ModelLayer::GELU(_)
                | ModelLayer::SiLU(_)
                | ModelLayer::Mish(_) => 0,
                ModelLayer::PReLU(l) => l.alpha().len(),
                ModelLayer::ResidualBlock(_) => 0,
                ModelLayer::Rnn(l) => {
                    l.input_size * l.hidden_size + l.hidden_size * l.hidden_size + l.hidden_size
                }
                ModelLayer::Lstm(l) => {
                    let h4 = 4 * l.hidden_size;
                    l.input_size * h4 + l.hidden_size * h4 + h4
                }
                ModelLayer::Gru(l) => {
                    let h3 = 3 * l.hidden_size;
                    l.input_size * h3 + l.hidden_size * h3 + 2 * h3
                }
                ModelLayer::MultiHeadAttention(_)
                | ModelLayer::TransformerEncoder(_)
                | ModelLayer::FeedForward(_) => 0,
            };
            total += params;
        }
        total
    }

    /// Freeze the layer at `index` so it is excluded from `trainable_parameters`.
    pub fn freeze_layer(&mut self, index: usize) -> Result<(), ModelError> {
        if index >= self.layers.len() {
            return Err(ModelError::InvalidLayerIndex {
                index,
                count: self.layers.len(),
            });
        }
        self.frozen[index] = true;
        Ok(())
    }

    /// Unfreeze the layer at `index` so it is included in `trainable_parameters` again.
    pub fn unfreeze_layer(&mut self, index: usize) -> Result<(), ModelError> {
        if index >= self.layers.len() {
            return Err(ModelError::InvalidLayerIndex {
                index,
                count: self.layers.len(),
            });
        }
        self.frozen[index] = false;
        Ok(())
    }

    /// Returns a slice of booleans indicating which layers are frozen.
    pub fn frozen_mask(&self) -> &[bool] {
        &self.frozen
    }

    /// Returns the total number of parameters in non-frozen layers.
    pub fn trainable_parameters(&self) -> usize {
        let mut total = 0usize;
        for (i, layer) in self.layers.iter().enumerate() {
            if self.frozen[i] {
                continue;
            }
            let params = match layer {
                ModelLayer::Linear(l) => l.in_features() * l.out_features() + l.out_features(),
                ModelLayer::Conv2d(l) => {
                    l.out_channels() * l.in_channels() * l.kernel_h() * l.kernel_w()
                        + if l.bias().is_some() {
                            l.out_channels()
                        } else {
                            0
                        }
                }
                ModelLayer::BatchNorm2d(l) => l.num_features() * 2,
                ModelLayer::Embedding(e) => e.num_embeddings() * e.embedding_dim(),
                ModelLayer::LoraLinear(l) => l.in_features * l.rank + l.rank * l.out_features,
                ModelLayer::Conv1d(l) => l.kernel().data().len(),
                ModelLayer::Conv3d(l) => l.weight().data().len(),
                ModelLayer::ConvTranspose2d(l) => l.kernel().data().len(),
                ModelLayer::DepthwiseConv2d(l) => {
                    l.weight().data().len() + l.bias().map_or(0, |b| b.data().len())
                }
                ModelLayer::SeparableConv2d(l) => {
                    l.depthwise().weight().data().len()
                        + l.depthwise().bias().map_or(0, |b| b.data().len())
                        + l.pointwise().weight().data().len()
                        + l.pointwise().bias().map_or(0, |b| b.data().len())
                }
                ModelLayer::DeformableConv2d(l) => {
                    l.weight.data().len()
                        + l.offset_weight.data().len()
                        + l.bias.as_ref().map_or(0, |b| b.data().len())
                }
                ModelLayer::InstanceNorm(_) => 0,
                ModelLayer::LayerNorm(_)
                | ModelLayer::GroupNorm(_)
                | ModelLayer::MaxPool2d(_)
                | ModelLayer::AvgPool2d(_)
                | ModelLayer::GlobalAvgPool2d(_)
                | ModelLayer::Flatten(_)
                | ModelLayer::ReLU(_)
                | ModelLayer::LeakyReLU(_)
                | ModelLayer::Sigmoid(_)
                | ModelLayer::Tanh(_)
                | ModelLayer::Softmax(_)
                | ModelLayer::Dropout(_)
                | ModelLayer::AdaptiveAvgPool2d(_)
                | ModelLayer::AdaptiveMaxPool2d(_)
                | ModelLayer::PixelShuffle(_)
                | ModelLayer::Upsample(_)
                | ModelLayer::GELU(_)
                | ModelLayer::SiLU(_)
                | ModelLayer::Mish(_) => 0,
                ModelLayer::PReLU(l) => l.alpha().len(),
                ModelLayer::ResidualBlock(_) => 0,
                ModelLayer::Rnn(l) => {
                    l.input_size * l.hidden_size + l.hidden_size * l.hidden_size + l.hidden_size
                }
                ModelLayer::Lstm(l) => {
                    let h4 = 4 * l.hidden_size;
                    l.input_size * h4 + l.hidden_size * h4 + h4
                }
                ModelLayer::Gru(l) => {
                    let h3 = 3 * l.hidden_size;
                    l.input_size * h3 + l.hidden_size * h3 + 2 * h3
                }
                ModelLayer::MultiHeadAttention(_)
                | ModelLayer::TransformerEncoder(_)
                | ModelLayer::FeedForward(_) => 0,
            };
            total += params;
        }
        total
    }

    /// Returns named parameter tensors from all layers.
    ///
    /// For layers with graph-registered weights (Linear, Embedding, LayerNorm,
    /// GroupNorm, LoraLinear), the tensors are retrieved from the graph.
    /// For layers that own their tensors directly (Conv2d, BatchNorm2d,
    /// DepthwiseConv2d, SeparableConv2d, Conv1d, ConvTranspose2d), the tensors
    /// are accessed via the layer's accessor methods.
    ///
    /// Names follow the pattern `"{type}{index}_{param}"`, e.g. `"linear0_weight"`.
    pub fn named_parameters<'a>(
        &'a self,
        graph: &'a Graph,
    ) -> Result<Vec<(String, &'a Tensor)>, ModelError> {
        let mut result = Vec::new();
        let mut type_counts = HashMap::<&str, usize>::default();

        for layer in &self.layers {
            match layer {
                ModelLayer::Linear(l) => {
                    let idx = type_counts.entry("linear").or_insert(0);
                    let i = *idx;
                    *idx += 1;
                    result.push((
                        format!("linear{i}_weight"),
                        graph.value(l.weight_node().expect("linear layer has weight node"))?,
                    ));
                    result.push((
                        format!("linear{i}_bias"),
                        graph.value(l.bias_node().expect("linear layer has bias node"))?,
                    ));
                }
                ModelLayer::Conv2d(l) => {
                    let idx = type_counts.entry("conv2d").or_insert(0);
                    let i = *idx;
                    *idx += 1;
                    result.push((format!("conv2d{i}_weight"), l.weight()));
                    if let Some(b) = l.bias() {
                        result.push((format!("conv2d{i}_bias"), b));
                    }
                }
                ModelLayer::BatchNorm2d(l) => {
                    let idx = type_counts.entry("batchnorm2d").or_insert(0);
                    let i = *idx;
                    *idx += 1;
                    result.push((format!("batchnorm2d{i}_gamma"), l.gamma()));
                    result.push((format!("batchnorm2d{i}_beta"), l.beta()));
                }
                ModelLayer::Embedding(e) => {
                    let idx = type_counts.entry("embedding").or_insert(0);
                    let i = *idx;
                    *idx += 1;
                    result.push((
                        format!("embedding{i}_weight"),
                        graph.value(e.weight_node())?,
                    ));
                }
                ModelLayer::LayerNorm(l) => {
                    let idx = type_counts.entry("layernorm").or_insert(0);
                    let i = *idx;
                    *idx += 1;
                    result.push((format!("layernorm{i}_gamma"), graph.value(l.gamma_node())?));
                    result.push((format!("layernorm{i}_beta"), graph.value(l.beta_node())?));
                }
                ModelLayer::GroupNorm(l) => {
                    let idx = type_counts.entry("groupnorm").or_insert(0);
                    let i = *idx;
                    *idx += 1;
                    result.push((format!("groupnorm{i}_gamma"), graph.value(l.gamma_node())?));
                    result.push((format!("groupnorm{i}_beta"), graph.value(l.beta_node())?));
                }
                ModelLayer::LoraLinear(l) => {
                    let idx = type_counts.entry("loralinear").or_insert(0);
                    let i = *idx;
                    *idx += 1;
                    result.push((
                        format!("loralinear{i}_frozen_weight"),
                        graph.value(l.frozen_weight)?,
                    ));
                    result.push((format!("loralinear{i}_lora_a"), graph.value(l.lora_a)?));
                    result.push((format!("loralinear{i}_lora_b"), graph.value(l.lora_b)?));
                    if let Some(bias_node) = l.bias {
                        result.push((format!("loralinear{i}_bias"), graph.value(bias_node)?));
                    }
                }
                ModelLayer::DepthwiseConv2d(l) => {
                    let idx = type_counts.entry("depthwiseconv2d").or_insert(0);
                    let i = *idx;
                    *idx += 1;
                    result.push((format!("depthwiseconv2d{i}_weight"), l.weight()));
                    if let Some(b) = l.bias() {
                        result.push((format!("depthwiseconv2d{i}_bias"), b));
                    }
                }
                ModelLayer::SeparableConv2d(l) => {
                    let idx = type_counts.entry("separableconv2d").or_insert(0);
                    let i = *idx;
                    *idx += 1;
                    result.push((
                        format!("separableconv2d{i}_depthwise_weight"),
                        l.depthwise().weight(),
                    ));
                    if let Some(b) = l.depthwise().bias() {
                        result.push((format!("separableconv2d{i}_depthwise_bias"), b));
                    }
                    result.push((
                        format!("separableconv2d{i}_pointwise_weight"),
                        l.pointwise().weight(),
                    ));
                    if let Some(b) = l.pointwise().bias() {
                        result.push((format!("separableconv2d{i}_pointwise_bias"), b));
                    }
                }
                ModelLayer::Conv1d(l) => {
                    let idx = type_counts.entry("conv1d").or_insert(0);
                    let i = *idx;
                    *idx += 1;
                    result.push((format!("conv1d{i}_weight"), l.kernel()));
                }
                ModelLayer::Conv3d(l) => {
                    let idx = type_counts.entry("conv3d").or_insert(0);
                    let i = *idx;
                    *idx += 1;
                    result.push((format!("conv3d{i}_weight"), l.weight()));
                }
                ModelLayer::ConvTranspose2d(l) => {
                    let idx = type_counts.entry("convtranspose2d").or_insert(0);
                    let i = *idx;
                    *idx += 1;
                    result.push((format!("convtranspose2d{i}_weight"), l.kernel()));
                }
                // Layers without parameters
                ModelLayer::ReLU(_)
                | ModelLayer::LeakyReLU(_)
                | ModelLayer::Sigmoid(_)
                | ModelLayer::Tanh(_)
                | ModelLayer::Softmax(_)
                | ModelLayer::Dropout(_)
                | ModelLayer::MaxPool2d(_)
                | ModelLayer::AvgPool2d(_)
                | ModelLayer::GlobalAvgPool2d(_)
                | ModelLayer::Flatten(_)
                | ModelLayer::AdaptiveAvgPool2d(_)
                | ModelLayer::AdaptiveMaxPool2d(_)
                | ModelLayer::InstanceNorm(_)
                | ModelLayer::PixelShuffle(_)
                | ModelLayer::Upsample(_)
                | ModelLayer::GELU(_)
                | ModelLayer::SiLU(_)
                | ModelLayer::Mish(_)
                | ModelLayer::PReLU(_)
                | ModelLayer::ResidualBlock(_)
                | ModelLayer::Rnn(_)
                | ModelLayer::Lstm(_)
                | ModelLayer::Gru(_)
                | ModelLayer::MultiHeadAttention(_)
                | ModelLayer::TransformerEncoder(_)
                | ModelLayer::FeedForward(_)
                | ModelLayer::DeformableConv2d(_) => {}
            }
        }
        Ok(result)
    }

    pub fn layers(&self) -> &[ModelLayer] {
        &self.layers
    }

    pub fn layers_mut(&mut self) -> &mut [ModelLayer] {
        &mut self.layers
    }

    /// Adds a residual block wrapping the given layers.
    pub fn add_residual_block(&mut self, layers: Vec<ModelLayer>) {
        self.layers
            .push(ModelLayer::ResidualBlock(ResidualBlock::new(layers)));
        self.frozen.push(false);
    }

    pub fn add_rnn(&mut self, input_size: usize, hidden_size: usize, seed: u64) {
        self.layers.push(ModelLayer::Rnn(RnnLayer::new(
            input_size,
            hidden_size,
            seed,
        )));
        self.frozen.push(false);
    }

    pub fn add_lstm(&mut self, input_size: usize, hidden_size: usize, seed: u64) {
        self.layers.push(ModelLayer::Lstm(LstmLayer::new(
            input_size,
            hidden_size,
            seed,
        )));
        self.frozen.push(false);
    }

    pub fn add_gru(&mut self, input_size: usize, hidden_size: usize, seed: u64) {
        self.layers.push(ModelLayer::Gru(GruLayer::new(
            input_size,
            hidden_size,
            seed,
        )));
        self.frozen.push(false);
    }

    pub fn add_multi_head_attention(&mut self, d_model: usize, num_heads: usize, seed: u64) {
        self.layers.push(ModelLayer::MultiHeadAttention(
            MultiHeadAttentionLayer::new(d_model, num_heads, seed),
        ));
        self.frozen.push(false);
    }

    pub fn add_transformer_encoder(
        &mut self,
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        seed: u64,
    ) {
        self.layers.push(ModelLayer::TransformerEncoder(
            TransformerEncoderLayer::new(d_model, num_heads, d_ff, seed),
        ));
        self.frozen.push(false);
    }

    pub fn add_feed_forward(&mut self, d_model: usize, d_ff: usize, seed: u64) {
        self.layers
            .push(ModelLayer::FeedForward(FeedForwardLayer::new(
                d_model, d_ff, seed,
            )));
        self.frozen.push(false);
    }

    /// Push a pre-built layer directly (used by fusion for inference-only layers).
    pub fn push_raw_layer(&mut self, layer: ModelLayer) {
        self.layers.push(layer);
        self.frozen.push(false);
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId, ModelError> {
        let mut current = input;
        for layer in &self.layers {
            current = layer.forward(graph, current)?;
        }
        Ok(current)
    }

    /// Pure-tensor inference forward pass (no autograd graph).
    ///
    /// Supports Conv2d, BatchNorm2d, MaxPool2d, AvgPool2d, Flatten, Softmax,
    /// and simple activation layers (ReLU, Sigmoid, Tanh, LeakyReLU).
    pub fn forward_inference(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        let mut current = input.clone();
        for layer in &self.layers {
            current = match layer {
                ModelLayer::Conv2d(l) => l.forward_inference(&current)?,
                ModelLayer::BatchNorm2d(l) => l.forward_inference(&current)?,
                ModelLayer::MaxPool2d(l) => l.forward_inference(&current)?,
                ModelLayer::AvgPool2d(l) => l.forward_inference(&current)?,
                ModelLayer::Flatten(l) => l.forward_inference(&current)?,
                ModelLayer::Softmax(l) => l.forward_inference(&current)?,
                ModelLayer::ReLU(_) => {
                    let out_data: Vec<f32> = current.data().iter().map(|&v| v.max(0.0)).collect();
                    Tensor::from_vec(current.shape().to_vec(), out_data)?
                }
                ModelLayer::Sigmoid(_) => {
                    let out_data: Vec<f32> = current
                        .data()
                        .iter()
                        .map(|&v| 1.0 / (1.0 + (-v).exp()))
                        .collect();
                    Tensor::from_vec(current.shape().to_vec(), out_data)?
                }
                ModelLayer::Tanh(_) => {
                    let out_data: Vec<f32> = current.data().iter().map(|&v| v.tanh()).collect();
                    Tensor::from_vec(current.shape().to_vec(), out_data)?
                }
                ModelLayer::LeakyReLU(l) => {
                    let slope = l.negative_slope();
                    let out_data: Vec<f32> = current
                        .data()
                        .iter()
                        .map(|&v| if v >= 0.0 { v } else { slope * v })
                        .collect();
                    Tensor::from_vec(current.shape().to_vec(), out_data)?
                }
                ModelLayer::GlobalAvgPool2d(l) => l.forward_inference(&current)?,
                ModelLayer::DepthwiseConv2d(l) => l.forward_inference(&current)?,
                ModelLayer::SeparableConv2d(l) => l.forward_inference(&current)?,
                ModelLayer::Dropout(d) => {
                    if self.training && d.rate() > 0.0 {
                        let scale = 1.0 / (1.0 - d.rate());
                        let scaled: Vec<f32> = current.data().iter().map(|&v| v * scale).collect();
                        Tensor::from_vec(current.shape().to_vec(), scaled)?
                    } else {
                        current
                    }
                }
                ModelLayer::Conv1d(l) => l.forward_inference(&current)?,
                ModelLayer::Conv3d(l) => l.forward_inference(&current)?,
                ModelLayer::ConvTranspose2d(l) => l.forward_inference(&current)?,
                ModelLayer::AdaptiveAvgPool2d(l) => l.forward_inference(&current)?,
                ModelLayer::AdaptiveMaxPool2d(l) => l.forward_inference(&current)?,
                ModelLayer::InstanceNorm(l) => l.forward_inference(&current)?,
                ModelLayer::PixelShuffle(l) => l.forward_inference(&current)?,
                ModelLayer::Upsample(l) => l.forward_inference(&current)?,
                ModelLayer::GELU(l) => l.forward_inference(&current)?,
                ModelLayer::SiLU(l) => l.forward_inference(&current)?,
                ModelLayer::Mish(l) => l.forward_inference(&current)?,
                ModelLayer::PReLU(l) => l.forward_inference(&current)?,
                ModelLayer::ResidualBlock(l) => l.forward_inference(&current)?,
                ModelLayer::Rnn(l) => l.forward_inference(&current)?,
                ModelLayer::Lstm(l) => l.forward_inference(&current)?,
                ModelLayer::Gru(l) => l.forward_inference(&current)?,
                ModelLayer::MultiHeadAttention(l) => l.forward_inference(&current)?,
                ModelLayer::TransformerEncoder(l) => l.forward_inference(&current)?,
                ModelLayer::FeedForward(l) => l.forward_inference(&current)?,
                ModelLayer::DeformableConv2d(l) => l.forward_inference(&current)?,
                ModelLayer::Linear(l) => l.forward_inference(&current)?,
                ModelLayer::Embedding(_)
                | ModelLayer::LayerNorm(_)
                | ModelLayer::GroupNorm(_)
                | ModelLayer::LoraLinear(_) => return Err(ModelError::GraphOnlyLayer),
            };
        }
        Ok(current)
    }

    /// Registers CNN layer parameters (Conv2d weight/bias, BatchNorm2d gamma/beta, etc.)
    /// as graph variables for autograd training.
    ///
    /// Layers whose parameters are already registered (i.e. `weight_node().is_some()`)
    /// are skipped, so this method is safe to call multiple times.
    pub fn register_cnn_params(&mut self, graph: &mut Graph) {
        for layer in &mut self.layers {
            match layer {
                ModelLayer::Conv2d(l) if l.weight_node().is_none() => l.register_params(graph),
                ModelLayer::BatchNorm2d(l) if l.gamma_node().is_none() => l.register_params(graph),
                ModelLayer::DepthwiseConv2d(l) if l.weight_node().is_none() => {
                    l.register_params(graph)
                }
                ModelLayer::SeparableConv2d(l) if l.depthwise().weight_node().is_none() => {
                    l.register_params(graph)
                }
                ModelLayer::Conv1d(l) if l.weight_node().is_none() => l.register_params(graph),
                ModelLayer::Conv3d(l) if l.weight_node().is_none() => l.register_params(graph),
                ModelLayer::MultiHeadAttention(l) if l.w_q_node().is_none() => {
                    l.register_params(graph)
                }
                ModelLayer::FeedForward(l) if l.w1_node().is_none() => l.register_params(graph),
                ModelLayer::TransformerEncoder(l) if l.ln1_gamma_node().is_none() => {
                    l.register_params(graph)
                }
                ModelLayer::Rnn(l) if l.w_ih_node().is_none() => l.register_params(graph),
                ModelLayer::Lstm(l) if l.w_ih_node().is_none() => l.register_params(graph),
                ModelLayer::Gru(l) if l.w_ih_node().is_none() => l.register_params(graph),
                ModelLayer::DeformableConv2d(l) if l.weight_node().is_none() => {
                    l.register_params(graph)
                }
                ModelLayer::ConvTranspose2d(l) if l.weight_node().is_none() => {
                    l.register_params(graph)
                }
                ModelLayer::InstanceNorm(l) if l.gamma_node().is_none() => l.register_params(graph),
                ModelLayer::PReLU(l) if l.alpha_node().is_none() => l.register_params(graph),
                _ => {}
            }
        }
        self.persistent_node_count = graph.node_count();
    }

    /// Synchronizes CNN layer owned tensors from the graph (e.g. after optimizer step).
    pub fn sync_cnn_from_graph(&mut self, graph: &Graph) -> Result<(), ModelError> {
        for layer in &mut self.layers {
            match layer {
                ModelLayer::Conv2d(l) => l.sync_from_graph(graph)?,
                ModelLayer::BatchNorm2d(l) => l.sync_from_graph(graph)?,
                ModelLayer::DepthwiseConv2d(l) => l.sync_from_graph(graph)?,
                ModelLayer::SeparableConv2d(l) => l.sync_from_graph(graph)?,
                _ => {}
            }
        }
        Ok(())
    }

    pub fn trainable_nodes(&self) -> Vec<NodeId> {
        let mut out = Vec::new();
        for layer in &self.layers {
            match layer {
                ModelLayer::Linear(linear) => {
                    out.extend(linear.trainable_nodes());
                }
                ModelLayer::Conv2d(conv) => {
                    if let Some(w) = conv.weight_node() {
                        out.push(w);
                    }
                    if let Some(b) = conv.bias_node() {
                        out.push(b);
                    }
                }
                ModelLayer::BatchNorm2d(bn) => {
                    if let Some(g) = bn.gamma_node() {
                        out.push(g);
                    }
                    if let Some(b) = bn.beta_node() {
                        out.push(b);
                    }
                }
                ModelLayer::DepthwiseConv2d(dw) => {
                    if let Some(w) = dw.weight_node() {
                        out.push(w);
                    }
                    if let Some(b) = dw.bias_node() {
                        out.push(b);
                    }
                }
                ModelLayer::SeparableConv2d(sep) => {
                    if let Some(w) = sep.depthwise().weight_node() {
                        out.push(w);
                    }
                    if let Some(b) = sep.depthwise().bias_node() {
                        out.push(b);
                    }
                    if let Some(w) = sep.pointwise().weight_node() {
                        out.push(w);
                    }
                    if let Some(b) = sep.pointwise().bias_node() {
                        out.push(b);
                    }
                }
                ModelLayer::LoraLinear(lora) => {
                    out.extend(lora.trainable_params());
                }
                _ => {}
            }
        }
        out
    }

    pub fn persistent_node_count(&self) -> usize {
        self.persistent_node_count
    }

    pub fn checkpoint(&self, graph: &Graph) -> Result<SequentialCheckpoint, ModelError> {
        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            match layer {
                ModelLayer::Linear(linear) => {
                    let weight = graph
                        .value(linear.weight_node().expect("linear layer has weight node"))?
                        .clone();
                    let bias = graph
                        .value(linear.bias_node().expect("linear layer has bias node"))?
                        .clone();
                    layers.push(LayerCheckpoint::Linear {
                        in_features: linear.in_features(),
                        out_features: linear.out_features(),
                        weight: TensorSnapshot::from_tensor(&weight),
                        bias: TensorSnapshot::from_tensor(&bias),
                    });
                }
                ModelLayer::ReLU(_) => layers.push(LayerCheckpoint::ReLU),
                ModelLayer::LeakyReLU(layer) => layers.push(LayerCheckpoint::LeakyReLU {
                    negative_slope: layer.negative_slope(),
                }),
                ModelLayer::Sigmoid(_) => layers.push(LayerCheckpoint::Sigmoid),
                ModelLayer::Tanh(_) => layers.push(LayerCheckpoint::Tanh),
                ModelLayer::Dropout(layer) => {
                    layers.push(LayerCheckpoint::Dropout { rate: layer.rate() })
                }
                ModelLayer::Conv2d(layer) => layers.push(LayerCheckpoint::Conv2d {
                    in_channels: layer.in_channels(),
                    out_channels: layer.out_channels(),
                    kernel_h: layer.kernel_h(),
                    kernel_w: layer.kernel_w(),
                    stride_h: layer.stride_h(),
                    stride_w: layer.stride_w(),
                    weight: TensorSnapshot::from_tensor(layer.weight()),
                    bias: layer.bias().map(TensorSnapshot::from_tensor),
                }),
                ModelLayer::BatchNorm2d(layer) => layers.push(LayerCheckpoint::BatchNorm2d {
                    num_features: layer.num_features(),
                    epsilon: layer.epsilon(),
                    gamma: TensorSnapshot::from_tensor(layer.gamma()),
                    beta: TensorSnapshot::from_tensor(layer.beta()),
                    running_mean: TensorSnapshot::from_tensor(layer.running_mean()),
                    running_var: TensorSnapshot::from_tensor(layer.running_var()),
                }),
                ModelLayer::MaxPool2d(layer) => layers.push(LayerCheckpoint::MaxPool2d {
                    kernel_h: layer.kernel_h(),
                    kernel_w: layer.kernel_w(),
                    stride_h: layer.stride_h(),
                    stride_w: layer.stride_w(),
                }),
                ModelLayer::AvgPool2d(layer) => layers.push(LayerCheckpoint::AvgPool2d {
                    kernel_h: layer.kernel_h(),
                    kernel_w: layer.kernel_w(),
                    stride_h: layer.stride_h(),
                    stride_w: layer.stride_w(),
                }),
                ModelLayer::Flatten(_) => layers.push(LayerCheckpoint::Flatten),
                ModelLayer::GlobalAvgPool2d(_) => layers.push(LayerCheckpoint::GlobalAvgPool2d),
                ModelLayer::Softmax(_) => layers.push(LayerCheckpoint::Softmax),
                ModelLayer::Embedding(layer) => {
                    let w = graph.value(layer.weight_node())?;
                    layers.push(LayerCheckpoint::Embedding {
                        num_embeddings: layer.num_embeddings(),
                        embedding_dim: layer.embedding_dim(),
                        weight: TensorSnapshot::from_tensor(w),
                    });
                }
                ModelLayer::LayerNorm(layer) => {
                    let g = graph.value(layer.gamma_node())?;
                    let b = graph.value(layer.beta_node())?;
                    layers.push(LayerCheckpoint::LayerNorm {
                        normalized_shape: layer.normalized_shape(),
                        eps: 1e-5,
                        gamma: TensorSnapshot::from_tensor(g),
                        beta: TensorSnapshot::from_tensor(b),
                    });
                }
                ModelLayer::GroupNorm(layer) => {
                    let g = graph.value(layer.gamma_node())?;
                    let b = graph.value(layer.beta_node())?;
                    layers.push(LayerCheckpoint::GroupNorm {
                        num_groups: layer.num_groups(),
                        num_channels: layer.num_channels(),
                        eps: 1e-5,
                        gamma: TensorSnapshot::from_tensor(g),
                        beta: TensorSnapshot::from_tensor(b),
                    });
                }
                ModelLayer::DepthwiseConv2d(layer) => {
                    layers.push(LayerCheckpoint::DepthwiseConv2d {
                        channels: layer.channels(),
                        kernel_h: layer.kernel_h(),
                        kernel_w: layer.kernel_w(),
                        stride_h: layer.stride_h(),
                        stride_w: layer.stride_w(),
                        weight: TensorSnapshot::from_tensor(layer.weight()),
                        bias: layer.bias().map(TensorSnapshot::from_tensor),
                    });
                }
                ModelLayer::SeparableConv2d(layer) => {
                    layers.push(LayerCheckpoint::SeparableConv2d {
                        in_channels: layer.in_channels(),
                        out_channels: layer.out_channels(),
                        kernel_h: layer.kernel_h(),
                        kernel_w: layer.kernel_w(),
                        stride_h: layer.stride_h(),
                        stride_w: layer.stride_w(),
                        depthwise_weight: TensorSnapshot::from_tensor(layer.depthwise().weight()),
                        pointwise_weight: TensorSnapshot::from_tensor(layer.pointwise().weight()),
                        bias: layer.pointwise().bias().map(TensorSnapshot::from_tensor),
                    });
                }
                ModelLayer::LoraLinear(lora) => {
                    // Merge LoRA weights and checkpoint as a regular Linear layer.
                    let merged_weight = lora.merge(graph)?;
                    let bias = if let Some(bias_node) = lora.bias {
                        graph.value(bias_node)?.clone()
                    } else {
                        Tensor::zeros(vec![lora.out_features])?
                    };
                    layers.push(LayerCheckpoint::Linear {
                        in_features: lora.in_features,
                        out_features: lora.out_features,
                        weight: TensorSnapshot::from_tensor(&merged_weight),
                        bias: TensorSnapshot::from_tensor(&bias),
                    });
                }
                // Inference-only layers have no graph-registered weights to checkpoint.
                ModelLayer::Conv1d(_)
                | ModelLayer::Conv3d(_)
                | ModelLayer::ConvTranspose2d(_)
                | ModelLayer::AdaptiveAvgPool2d(_)
                | ModelLayer::AdaptiveMaxPool2d(_)
                | ModelLayer::InstanceNorm(_)
                | ModelLayer::PixelShuffle(_)
                | ModelLayer::Upsample(_)
                | ModelLayer::GELU(_)
                | ModelLayer::SiLU(_)
                | ModelLayer::Mish(_)
                | ModelLayer::PReLU(_)
                | ModelLayer::ResidualBlock(_)
                | ModelLayer::Rnn(_)
                | ModelLayer::Lstm(_)
                | ModelLayer::Gru(_)
                | ModelLayer::MultiHeadAttention(_)
                | ModelLayer::TransformerEncoder(_)
                | ModelLayer::FeedForward(_)
                | ModelLayer::DeformableConv2d(_) => {
                    return Err(ModelError::InferenceOnlyLayer);
                }
            }
        }
        Ok(SequentialCheckpoint { layers })
    }

    pub fn from_checkpoint(
        graph: &mut Graph,
        checkpoint: &SequentialCheckpoint,
    ) -> Result<Self, ModelError> {
        let mut model = Self::new(graph);
        for layer in &checkpoint.layers {
            match layer {
                LayerCheckpoint::Linear {
                    in_features,
                    out_features,
                    weight,
                    bias,
                } => {
                    model.add_linear(
                        graph,
                        *in_features,
                        *out_features,
                        weight.clone().into_tensor()?,
                        bias.clone().into_tensor()?,
                    )?;
                }
                LayerCheckpoint::ReLU => model.add_relu(),
                LayerCheckpoint::LeakyReLU { negative_slope } => {
                    model.add_leaky_relu(*negative_slope)?
                }
                LayerCheckpoint::Sigmoid => model.add_sigmoid(),
                LayerCheckpoint::Tanh => model.add_tanh(),
                LayerCheckpoint::Dropout { rate } => model.add_dropout(*rate)?,
                LayerCheckpoint::Conv2d {
                    in_channels,
                    out_channels,
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    weight,
                    bias,
                } => {
                    let bias_tensor = match bias {
                        Some(b) => Some(b.clone().into_tensor()?),
                        None => None,
                    };
                    model.add_conv2d(
                        *in_channels,
                        *out_channels,
                        *kernel_h,
                        *kernel_w,
                        *stride_h,
                        *stride_w,
                        weight.clone().into_tensor()?,
                        bias_tensor,
                    )?;
                }
                LayerCheckpoint::BatchNorm2d {
                    num_features,
                    epsilon,
                    gamma,
                    beta,
                    running_mean,
                    running_var,
                } => {
                    model.add_batch_norm2d(
                        *num_features,
                        *epsilon,
                        gamma.clone().into_tensor()?,
                        beta.clone().into_tensor()?,
                        running_mean.clone().into_tensor()?,
                        running_var.clone().into_tensor()?,
                    )?;
                }
                LayerCheckpoint::MaxPool2d {
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                } => model.add_max_pool2d(*kernel_h, *kernel_w, *stride_h, *stride_w)?,
                LayerCheckpoint::AvgPool2d {
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                } => model.add_avg_pool2d(*kernel_h, *kernel_w, *stride_h, *stride_w)?,
                LayerCheckpoint::Flatten => model.add_flatten(),
                LayerCheckpoint::GlobalAvgPool2d => model.add_global_avg_pool2d(),
                LayerCheckpoint::Softmax => model.add_softmax(),
                LayerCheckpoint::Embedding {
                    num_embeddings,
                    embedding_dim,
                    weight,
                } => {
                    model.add_embedding(
                        graph,
                        *num_embeddings,
                        *embedding_dim,
                        weight.clone().into_tensor()?,
                    )?;
                }
                LayerCheckpoint::LayerNorm {
                    normalized_shape,
                    eps,
                    gamma: _,
                    beta: _,
                } => {
                    model.add_layer_norm(graph, *normalized_shape, *eps)?;
                }
                LayerCheckpoint::GroupNorm {
                    num_groups,
                    num_channels,
                    eps,
                    gamma: _,
                    beta: _,
                } => {
                    model.add_group_norm(graph, *num_groups, *num_channels, *eps)?;
                }
                LayerCheckpoint::DepthwiseConv2d {
                    channels,
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    weight,
                    bias,
                } => {
                    let bias_tensor = match bias {
                        Some(b) => Some(b.clone().into_tensor()?),
                        None => None,
                    };
                    model.add_depthwise_conv2d(
                        *channels,
                        *kernel_h,
                        *kernel_w,
                        *stride_h,
                        *stride_w,
                        weight.clone().into_tensor()?,
                        bias_tensor,
                    )?;
                }
                LayerCheckpoint::SeparableConv2d {
                    in_channels,
                    out_channels,
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    depthwise_weight,
                    pointwise_weight,
                    bias,
                } => {
                    let bias_tensor = match bias {
                        Some(b) => Some(b.clone().into_tensor()?),
                        None => None,
                    };
                    model.add_separable_conv2d(
                        *in_channels,
                        *out_channels,
                        *kernel_h,
                        *kernel_w,
                        *stride_h,
                        *stride_w,
                        depthwise_weight.clone().into_tensor()?,
                        pointwise_weight.clone().into_tensor()?,
                        bias_tensor,
                    )?;
                }
            }
        }
        Ok(model)
    }
}

fn format_param_count(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}
