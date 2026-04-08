# yscv-cli

Command-line tool for real-time inference (detection + tracking + recognition pipeline), camera diagnostics, and dataset evaluation. Argument-driven (no subcommands).

```bash
# Enumerate camera devices (filtered by optional name match)
cargo run -p yscv-cli --features native-camera -- --list-cameras

# Run camera diagnostics with a frame budget and JSON report output
cargo run -p yscv-cli --features native-camera -- \
    --diagnose-camera \
    --diagnose-frames 120 \
    --diagnose-report report.json

# Evaluate detection predictions against COCO ground truth
cargo run -p yscv-cli -- \
    --eval-detection-coco-gt gt.json \
    --eval-detection-coco-pred pred.json \
    --eval-iou 0.5 \
    --eval-score 0.0

# Default invocation runs the live pipeline (detect → track → recognize)
cargo run -p yscv-cli --features native-camera -- \
    --camera --device 0 --detect-target people --detect-score 0.4
```

## Modes

| Flag | What it does |
|------|-------------|
| `--list-cameras` | Enumerate camera devices, optionally filtered by name |
| `--diagnose-camera` | Run camera capture diagnostics, write JSON report |
| `--validate-diagnostics-report <path>` | Validate a previously saved diagnostics report against thresholds |
| `--eval-detection-{jsonl,coco,openimages,yolo,voc,kitti,widerface}-*` | Run detection evaluation against the named dataset format |
| `--eval-tracking-{jsonl,mot}-*` | Run tracking evaluation (MOTA / MOTP / IDF1 / HOTA) |
| _(default)_ | Live pipeline: camera capture → detection → tracking → optional recognition, with benchmarking and event logging |

## Features

```toml
[features]
native-camera = []  # Live camera input
```

## Configuration

JSON-based config for thresholds, model paths, and pipeline options.

## Tests

42 tests covering config parsing, pipeline setup, evaluation output.
