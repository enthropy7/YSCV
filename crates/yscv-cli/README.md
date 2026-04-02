# yscv-cli

Command-line tool for real-time inference, benchmarking, and evaluation.

```bash
# Run detection on a video
yscv detect --model yolov8n.onnx --source video.mp4

# Benchmark inference pipeline
yscv bench --model yolov8n.onnx --frames 1000

# Evaluate against ground truth
yscv eval --predictions pred.json --ground-truth gt.json
```

## Commands

| Command | What it does |
|---------|-------------|
| `detect` | Run object detection on image/video/camera |
| `track` | Detection + multi-object tracking |
| `bench` | Benchmark inference FPS with statistics |
| `eval` | Compute mAP/MOTA against annotations |
| `diagnose` | Camera diagnostics and quality report |

## Features

```toml
[features]
native-camera = []  # Live camera input
```

## Configuration

JSON-based config for thresholds, model paths, and pipeline options.

## Tests

42 tests covering config parsing, pipeline setup, evaluation output.
