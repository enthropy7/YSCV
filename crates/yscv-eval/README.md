# yscv-eval

Evaluation metrics for classification, detection, and tracking. Dataset adapters for COCO, Pascal VOC, and CSV.

```rust
use yscv_eval::*;

let ap = average_precision(&predictions, &ground_truths, 0.5);
let report = classification_report(&predicted_labels, &true_labels);
println!("{}", report);
```

## Metrics

37 public metric/eval functions across the crate:

| Task | Metrics |
|------|---------|
| **Classification** | accuracy, precision, recall, F1, confusion matrix, Cohen's kappa, ROC curve + AUC, average_precision, top_k_accuracy, classification_report |
| **Detection** | mAP, COCO mAP, AP@IoU, precision-recall, `evaluate_detections{,_coco,_from_dataset}` |
| **Tracking** | MOTA, MOTP, HOTA, IDF1, `evaluate_tracking{,_from_dataset}` |
| **Regression** | MAE, RMSE, MAPE, R² |
| **Image quality / segmentation** | PSNR, SSIM, dice_score, mean_iou, per_class_iou |
| **Counting / pipeline / camera** | counting metrics, pipeline benchmark thresholds, camera diagnostics validation |

## Dataset Adapters (8)

Under `src/dataset/`:

- **COCO** — `parse_detection_dataset_coco`, `load_detection_dataset_coco_files`
- **JSONL** — detection + tracking
- **MOT** — `parse_tracking_dataset_mot`, `load_tracking_dataset_mot_txt_files`
- **OpenImages** — CSV pair (`parse_detection_dataset_openimages_csv`)
- **VOC** — XML directories
- **YOLO** — label directories
- **KITTI** — label directories
- **WIDERFACE** — TXT pair

## Tests

95 tests covering metric correctness, edge cases, dataset parsing.
