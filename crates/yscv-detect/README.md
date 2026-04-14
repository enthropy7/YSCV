# yscv-detect

Object detection pipeline: YOLOv8/v11 decoding, NMS, heatmap detection, ROI pooling, and anchor generation.

```rust,ignore
use yscv_detect::*;

let detections = detect_people_from_rgb8(width, height, &rgb_data, 0.5, 3, 0.45, 100)?;
for det in &detections {
    println!("{}: {:.0}% at ({}, {}, {}, {})",
        det.label, det.score * 100.0, det.x, det.y, det.w, det.h);
}
```

## Features

- **YOLO decoding**: YOLOv8 and YOLOv11 output tensor parsing with letterbox preprocessing
- **NMS**: standard, soft-NMS, batched (class-aware), with early exit optimization
- **Heatmap detection**: keypoint-based detection with local maxima suppression
- **ROI ops**: ROI pooling and bilinear ROI align
- **Anchors**: multi-scale anchor generation for SSD/Faster R-CNN
- **Scratch buffers**: zero-alloc detection with reusable scratch objects

## Optional Features

```toml
[features]
onnx = []  # ONNX model inference via yscv-onnx
```

## Tests

60 tests covering decoding, NMS edge cases, ROI alignment.
