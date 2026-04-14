//! Re-identification feature extraction for DeepSORT appearance matching.

use std::collections::HashMap;

use yscv_tensor::Tensor;

use crate::TrackError;

/// Re-identification feature extractor trait.
/// Implementations extract appearance embeddings from image crops.
pub trait ReIdExtractor: Send + Sync {
    /// Extract an embedding vector from an image crop [H, W, C].
    fn extract(&self, crop: &Tensor) -> Result<Vec<f32>, TrackError>;
    /// Embedding dimension.
    fn dim(&self) -> usize;
}

/// Simple re-id using average color histogram as embedding.
/// Useful as a baseline / fallback when no learned model is available.
pub struct ColorHistogramReId {
    bins: usize,
}

impl ColorHistogramReId {
    pub fn new(bins: usize) -> Self {
        Self { bins }
    }
}

impl ReIdExtractor for ColorHistogramReId {
    fn extract(&self, crop: &Tensor) -> Result<Vec<f32>, TrackError> {
        // Compute color histogram per channel, normalize to unit vector
        let shape = crop.shape();
        let c = *shape.last().unwrap_or(&3);
        let data = crop.data();
        let pixels = data.len() / c;
        let mut hist = vec![0.0f32; self.bins * c];
        for px in 0..pixels {
            for ch in 0..c {
                let val = data[px * c + ch].clamp(0.0, 1.0);
                let bin = ((val * self.bins as f32) as usize).min(self.bins - 1);
                hist[ch * self.bins + bin] += 1.0;
            }
        }
        // L2 normalize
        let norm: f32 = hist.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            hist.iter_mut().for_each(|v| *v /= norm);
        }
        Ok(hist)
    }

    fn dim(&self) -> usize {
        self.bins * 3
    }
}

/// Re-id feature gallery that stores per-track appearance features
/// and computes cosine distance for matching.
pub struct ReIdGallery {
    features: HashMap<u64, Vec<Vec<f32>>>,
    max_features: usize,
}

impl ReIdGallery {
    /// Create a new gallery that keeps at most `max_features` per track.
    pub fn new(max_features: usize) -> Self {
        Self {
            features: HashMap::new(),
            max_features,
        }
    }

    /// Add a feature vector for the given track, evicting the oldest if
    /// the gallery for that track is full.
    pub fn update(&mut self, track_id: u64, feature: Vec<f32>) {
        let entry = self.features.entry(track_id).or_default();
        entry.push(feature);
        if entry.len() > self.max_features {
            entry.remove(0);
        }
    }

    /// Remove all stored features for a track.
    pub fn remove(&mut self, track_id: u64) {
        self.features.remove(&track_id);
    }

    /// Minimum cosine distance between `feature` and all stored features
    /// for `track_id`. Returns 1.0 (maximum distance) if the track has no
    /// stored features.
    pub fn min_cosine_distance(&self, track_id: u64, feature: &[f32]) -> f32 {
        match self.features.get(&track_id) {
            Some(gallery) if !gallery.is_empty() => gallery
                .iter()
                .map(|g| cosine_distance(feature, g))
                .fold(f32::INFINITY, f32::min),
            _ => 1.0,
        }
    }

    /// Build a cost matrix of shape `[track_ids.len(), features.len()]`
    /// where each entry is the minimum cosine distance between the track's
    /// gallery and the candidate feature.
    pub fn cost_matrix(&self, track_ids: &[u64], features: &[Vec<f32>]) -> Vec<Vec<f32>> {
        track_ids
            .iter()
            .map(|&tid| {
                features
                    .iter()
                    .map(|f| self.min_cosine_distance(tid, f))
                    .collect()
            })
            .collect()
    }
}

/// Cosine distance between two feature vectors: `1 - cos(a, b)`.
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 {
        return 1.0;
    }
    1.0 - (dot / denom)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_histogram_reid_basic() {
        let extractor = ColorHistogramReId::new(8);
        assert_eq!(extractor.dim(), 24); // 8 bins * 3 channels

        // Create a small 2x2 RGB image with known values
        let data = vec![
            0.1, 0.2, 0.3, // pixel (0,0)
            0.4, 0.5, 0.6, // pixel (0,1)
            0.7, 0.8, 0.9, // pixel (1,0)
            0.0, 0.1, 0.2, // pixel (1,1)
        ];
        let crop = Tensor::from_vec(vec![2, 2, 3], data).expect("valid tensor shape");
        let embedding = extractor.extract(&crop).expect("extraction should succeed");

        // Verify dimension matches
        assert_eq!(embedding.len(), extractor.dim());

        // Verify L2 normalization: ||embedding|| should be ~1.0
        let norm: f32 = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "Expected unit norm, got {norm}");
    }

    #[test]
    fn reid_gallery_update_and_distance() {
        let mut gallery = ReIdGallery::new(10);

        // Track with no features should have max distance
        assert!((gallery.min_cosine_distance(1, &[1.0, 0.0, 0.0]) - 1.0).abs() < 1e-6);

        // Add a feature for track 1
        gallery.update(1, vec![1.0, 0.0, 0.0]);

        // Same direction -> distance ~0
        let dist = gallery.min_cosine_distance(1, &[1.0, 0.0, 0.0]);
        assert!(
            dist < 1e-5,
            "Expected ~0 distance for identical vectors, got {dist}"
        );

        // Orthogonal -> distance ~1
        let dist = gallery.min_cosine_distance(1, &[0.0, 1.0, 0.0]);
        assert!(
            (dist - 1.0).abs() < 1e-5,
            "Expected ~1 distance for orthogonal vectors, got {dist}"
        );

        // Add a second feature; min distance should pick the closer one
        gallery.update(1, vec![0.0, 1.0, 0.0]);
        let dist = gallery.min_cosine_distance(1, &[0.0, 1.0, 0.0]);
        assert!(
            dist < 1e-5,
            "Expected ~0 after adding matching feature, got {dist}"
        );

        // Remove track and verify distance returns to max
        gallery.remove(1);
        assert!((gallery.min_cosine_distance(1, &[1.0, 0.0, 0.0]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn reid_gallery_cost_matrix() {
        let mut gallery = ReIdGallery::new(10);

        gallery.update(10, vec![1.0, 0.0, 0.0]);
        gallery.update(20, vec![0.0, 1.0, 0.0]);

        let track_ids = vec![10, 20];
        let features = vec![
            vec![1.0, 0.0, 0.0], // should be close to track 10, far from 20
            vec![0.0, 1.0, 0.0], // should be far from track 10, close to 20
            vec![0.0, 0.0, 1.0], // orthogonal to both
        ];

        let matrix = gallery.cost_matrix(&track_ids, &features);

        // Shape: 2 tracks x 3 features
        assert_eq!(matrix.len(), 2);
        assert_eq!(matrix[0].len(), 3);
        assert_eq!(matrix[1].len(), 3);

        // track 10 vs feature 0 (identical) -> ~0
        assert!(matrix[0][0] < 1e-5);
        // track 10 vs feature 1 (orthogonal) -> ~1
        assert!((matrix[0][1] - 1.0).abs() < 1e-5);
        // track 20 vs feature 0 (orthogonal) -> ~1
        assert!((matrix[1][0] - 1.0).abs() < 1e-5);
        // track 20 vs feature 1 (identical) -> ~0
        assert!(matrix[1][1] < 1e-5);
    }
}
