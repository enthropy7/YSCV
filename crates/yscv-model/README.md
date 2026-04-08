# yscv-model

Neural network layers, model architectures, training loop, data loading, and model serialization.

```rust
use yscv_model::*;

let model = Sequential::new()
    .add(Conv2d::new(3, 64, 3, 1, 1))
    .add(BatchNorm2d::new(64))
    .add(Relu)
    .add(GlobalAvgPool2d)
    .add(Linear::new(64, 10));

let mut trainer = Trainer::new(model, Adam::new(1e-3), CrossEntropyLoss);
trainer.fit(&train_loader, &val_loader, 100);
```

## Layers (39 `ModelLayer` variants)

| Category | Layers |
|----------|--------|
| **Conv** | Conv1d, Conv2d, Conv3d, DepthwiseConv2d, SeparableConv2d, ConvTranspose2d, DeformableConv2d |
| **Linear / Embedding** | Linear, LoraLinear, Embedding, FeedForward |
| **Norm** | BatchNorm2d, LayerNorm, GroupNorm, InstanceNorm |
| **Activation** | ReLU, LeakyReLU, PReLU, Sigmoid, Tanh, GELU, SiLU, Mish, Softmax |
| **Pool** | MaxPool2d, AvgPool2d, GlobalAvgPool2d, AdaptiveAvgPool2d, AdaptiveMaxPool2d |
| **Attention** | MultiHeadAttention, TransformerEncoder |
| **Recurrent** | Rnn, Lstm, Gru |
| **Spatial / Misc** | Flatten, Dropout, PixelShuffle, Upsample, ResidualBlock |

## Loss Functions (17)

mse, mae, huber, smooth_l1, hinge, bce, nll, cross_entropy, label_smoothing_cross_entropy, focal, dice, triplet, contrastive, cosine_embedding, kl_div, ctc, distillation — all live in `src/loss.rs`.

## Architectures (13 in `zoo.rs`)

ResNet18 / ResNet34 / ResNet50 / ResNet101, VGG16 / VGG19, MobileNetV2, EfficientNetB0, AlexNet, ViTTiny / ViTBase / ViTLarge, DeiTTiny — all loadable through `ModelHub` with remote weight download.

## Training

- Mixed precision (f16/bf16)
- LoRA adaptation
- EMA (exponential moving average)
- Checkpointing (safetensors format)
- Callbacks: early stopping, best model, LR logging

## Data Loading

- `DataLoader` with multi-threaded batching
- `ImageFolder` dataset
- Built-in: CIFAR-10, ImageNet, COCO
- Augmentation: random crop, flip, color jitter, normalize

## Tests

365 tests covering layer correctness, training convergence, serialization.
