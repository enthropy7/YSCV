# yscv-track

Multi-object tracking: DeepSORT, ByteTrack, Kalman filter, Hungarian assignment, and Re-ID.

```rust
use yscv_track::*;

let mut tracker = Tracker::new(TrackerConfig::default());
loop {
    let detections = detect_objects(&frame);
    let tracked = tracker.update(&detections);
    for t in &tracked {
        println!("Track {} at ({}, {})", t.track_id, t.x, t.y);
    }
}
```

## Algorithms

| Algorithm | Use case |
|-----------|----------|
| **DeepSORT** | Appearance + motion, best for re-identification |
| **ByteTrack** | High-speed temporal association, no appearance model |
| **Kalman Filter** | Motion prediction with constant velocity model |
| **Hungarian** | Optimal assignment between tracks and detections |

## Re-ID

- Color histogram features (no model needed)
- Cosine similarity matching
- VP-Tree approximate nearest neighbor
- Feature gallery with FIFO eviction (max 100 per track)

## Tests

57 tests covering track lifecycle, assignment, Kalman prediction, Re-ID matching.
