# yscv-onnx

Pure Rust ONNX runtime. 126 operators, graph optimization, CPU and GPU inference.

```rust
use yscv_onnx::*;

let model = load_onnx_model("yolov8n.onnx")?;
let mut runner = OnnxRunner::new(&model)?;

let input = Tensor::from_vec(vec![1, 3, 640, 640], image_data)?;
let outputs = runner.run(&[("images", &input)])?;
```

## Capabilities

- **126 ONNX operators**: Conv, MatMul, Relu, Sigmoid, Softmax, Resize, Concat, Split, Gather, Scatter, LSTM, GRU, Attention, ...
- **Graph optimizations**: constant folding, Conv+BN fusion, Conv+Relu fusion, dead node elimination
- **Quantization**: INT8 inference support
- **GPU inference**: wgpu (Vulkan/Metal/DX12) and native Metal (MPSGraph)
- **Model export**: save optimized graphs back to ONNX format

## Features

```toml
[features]
gpu = []             # wgpu GPU inference
metal-backend = []   # macOS Metal (MPSGraph)
```

## Tests

77 tests covering operator correctness, shape inference, graph optimization, model loading.
