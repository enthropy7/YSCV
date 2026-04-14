# yscv-onnx

Pure Rust ONNX runtime. 128 CPU operators, graph optimization, CPU and Metal GPU inference.

```rust,ignore
use yscv_onnx::*;

let model = load_onnx_model("yolov8n.onnx")?;
let mut runner = OnnxRunner::new(&model)?;

let input = Tensor::from_vec(vec![1, 3, 640, 640], image_data)?;
let outputs = runner.run(&[("images", &input)])?;
```

## Capabilities

- **128 ONNX CPU operators**: Conv, MatMul, Gemm, Relu/LeakyRelu/Sigmoid/Tanh/Gelu/Erf/Mish/HardSwish/Softmax/LogSoftmax, BatchNormalization/LayerNormalization/InstanceNormalization, MaxPool/AveragePool/GlobalAveragePool, Resize/Upsample, Concat/Split/Reshape/Flatten/Transpose/Gather/GatherElements/GatherND/ScatterElements/ScatterND/Slice/Tile/Expand, Cast/Pad/Clip/Where/Identity/CumSum/ArgMax/ArgMin/TopK, DepthToSpace/SpaceToDepth, GridSample/RoiAlign/NonMaxSuppression, full quantized stack (QuantizeLinear/DequantizeLinear/QLinearConv/QLinearMatMul/MatMulInteger/ConvInteger/DynamicQuantizeLinear), trig + hyperbolic, logical, fused `Conv_Relu` / `BatchNormalization_Relu`, … (verified against the dispatch in `src/runner/mod.rs`)
- **Graph optimizations**: constant folding, Conv+BN fusion, Conv+Relu fusion, dead node elimination
- **Quantization**: INT4 weight-only (`quantize_weights_int4`) + INT8 symmetric/asymmetric/per-channel inference support
- **LLM inference**: autoregressive `generate()` with KV-cache, top-k/top-p sampling, temperature, repetition penalty; RoPE and GroupQueryAttention for decoder-only transformers
- **GPU inference**: wgpu (Vulkan/Metal/DX12) and native Metal/MPSGraph plan compiler with triple-buffered pipelined `submit`/`wait` API (multi-input models, f16 end-to-end, zero-alloc hot path, ~20 op kinds, automatic CPU fallback for unsupported subgraphs)
- **Model export**: save optimized graphs back to ONNX format

## Features

```toml
[features]
gpu = []             # wgpu GPU inference
metal-backend = []   # macOS Metal (MPSGraph)
```

## Tests

166+ tests covering all 128 operator dispatch arms, shape inference, graph optimization, quantization, fusion, and model loading.
