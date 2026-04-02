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

## Layers (39)

| Category | Layers |
|----------|--------|
| **Conv** | Conv1d, Conv2d, Conv3d, DepthwiseConv2d, SeparableConv2d, TransposeConv2d, DeformableConv2d |
| **Linear** | Linear, Bilinear, Embedding |
| **Norm** | BatchNorm1d/2d, LayerNorm, GroupNorm, RMSNorm, InstanceNorm |
| **Activation** | ReLU, GELU, SiLU, Mish, Softmax, LogSoftmax |
| **Pool** | MaxPool2d, AvgPool2d, GlobalAvgPool2d, AdaptiveAvgPool2d |
| **Attention** | MultiHeadAttention, TransformerEncoder/Decoder |
| **Recurrent** | RNN, LSTM, GRU |
| **Regularization** | Dropout, DropPath |

## Architectures

ResNet (18/34/50/101/152), MobileNetV2/V3, EfficientNet, Vision Transformer (ViT), UNet, YOLO backbone

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
