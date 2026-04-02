# yscv-eval

Evaluation metrics for classification, detection, and tracking. Dataset adapters for COCO, Pascal VOC, and CSV.

```rust
use yscv_eval::*;

let ap = average_precision(&predictions, &ground_truths, 0.5);
let report = classification_report(&predicted_labels, &true_labels);
println!("{}", report);
```

## Metrics

| Task | Metrics |
|------|---------|
| **Classification** | accuracy, precision, recall, F1, confusion matrix, Cohen's kappa, ROC AUC |
| **Detection** | mAP, AP@IoU, precision-recall curve |
| **Tracking** | MOTA, MOTP, HOTA, IDF1, ID switches |
| **Image quality** | PSNR, SSIM, MSE |

## Dataset Adapters

- **COCO**: JSON annotation loading
- **Pascal VOC**: XML annotation parsing
- **CSV**: generic label file reader
- **Camera diagnostics**: capture quality reports

## Tests

95 tests covering metric correctness, edge cases, dataset parsing.
