use super::similarity::cosine_similarity_prevalidated;

/// Vantage-point tree for approximate nearest-neighbor search.
#[derive(Debug, Clone)]
pub struct VpTree {
    nodes: Vec<VpNode>,
    embeddings: Vec<Vec<f32>>,
    ids: Vec<String>,
}

#[derive(Debug, Clone)]
struct VpNode {
    index: usize,
    threshold: f32,
    left: Option<usize>,
    right: Option<usize>,
}

/// A k-nearest-neighbor result entry.
#[derive(Debug, Clone, PartialEq)]
pub struct KnnResult {
    pub id: String,
    pub distance: f32,
}

impl VpTree {
    /// Create an empty VP-tree.
    pub const fn new() -> Self {
        Self {
            nodes: Vec::new(),
            embeddings: Vec::new(),
            ids: Vec::new(),
        }
    }

    /// Build a VP-tree from a list of (id, embedding) pairs.
    pub fn build(entries: Vec<(String, Vec<f32>)>) -> Self {
        if entries.is_empty() {
            return Self::new();
        }

        let mut ids: Vec<String> = Vec::with_capacity(entries.len());
        let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(entries.len());
        for (id, emb) in entries {
            ids.push(id);
            embeddings.push(emb);
        }

        let mut tree = VpTree {
            nodes: Vec::with_capacity(embeddings.len()),
            embeddings,
            ids,
        };

        let indices: Vec<usize> = (0..tree.embeddings.len()).collect();
        tree.build_recursive(&indices);
        tree
    }

    fn build_recursive(&mut self, indices: &[usize]) -> Option<usize> {
        if indices.is_empty() {
            return None;
        }

        if indices.len() == 1 {
            let node_index = self.nodes.len();
            self.nodes.push(VpNode {
                index: indices[0],
                threshold: 0.0,
                left: None,
                right: None,
            });
            return Some(node_index);
        }

        // Pick the last element as the vantage point (deterministic).
        let vp_idx = indices[indices.len() - 1];
        let rest: Vec<usize> = indices[..indices.len() - 1].to_vec();

        // Compute distances from vantage point to all others.
        let mut dists: Vec<(usize, f32)> = rest
            .iter()
            .map(|&i| {
                let d = cosine_distance(&self.embeddings[vp_idx], &self.embeddings[i]);
                (i, d)
            })
            .collect();

        // Sort by distance to find the median.
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let median_pos = dists.len() / 2;
        let threshold = dists[median_pos].1;

        // Split into inside (dist < threshold) and outside (dist >= threshold).
        let inside: Vec<usize> = dists[..median_pos].iter().map(|&(i, _)| i).collect();
        let outside: Vec<usize> = dists[median_pos..].iter().map(|&(i, _)| i).collect();

        // Reserve the node index before recursing.
        let node_index = self.nodes.len();
        self.nodes.push(VpNode {
            index: vp_idx,
            threshold,
            left: None,
            right: None,
        });

        let left = self.build_recursive(&inside);
        let right = self.build_recursive(&outside);

        self.nodes[node_index].left = left;
        self.nodes[node_index].right = right;

        Some(node_index)
    }

    /// Query the tree for the k nearest neighbors to `point`.
    ///
    /// Returns a list of `(id, distance)` pairs sorted by ascending distance.
    pub fn query(&self, point: &[f32], k: usize) -> Vec<KnnResult> {
        if self.nodes.is_empty() || k == 0 {
            return Vec::new();
        }

        let mut heap = BoundedMaxHeap::new(k);
        self.search(0, point, &mut heap);

        let mut results: Vec<KnnResult> = heap
            .entries
            .into_iter()
            .map(|(dist, idx)| KnnResult {
                id: self.ids[idx].clone(),
                distance: dist,
            })
            .collect();
        results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    fn search(&self, node_idx: usize, point: &[f32], heap: &mut BoundedMaxHeap) {
        let node = &self.nodes[node_idx];
        let dist = cosine_distance(point, &self.embeddings[node.index]);

        heap.push(dist, node.index);

        let tau = heap.max_dist();

        if dist < node.threshold {
            // Point is inside; search inside first.
            if let Some(left) = node.left
                && dist - tau < node.threshold
            {
                self.search(left, point, heap);
            }
            if let Some(right) = node.right
                && dist + tau >= node.threshold
            {
                self.search(right, point, heap);
            }
        } else {
            // Point is outside; search outside first.
            if let Some(right) = node.right
                && dist + tau >= node.threshold
            {
                self.search(right, point, heap);
            }
            if let Some(left) = node.left
                && dist - tau < node.threshold
            {
                self.search(left, point, heap);
            }
        }
    }

    /// Number of embeddings in the tree.
    pub const fn len(&self) -> usize {
        self.embeddings.len()
    }

    /// Whether the tree is empty.
    pub const fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }
}

impl Default for VpTree {
    fn default() -> Self {
        Self::new()
    }
}

/// Distance metric: 1.0 - cosine_similarity.
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let sim = cosine_similarity_prevalidated(a, b).unwrap_or(0.0);
    1.0 - sim
}

/// A max-heap with bounded capacity k, keeping the k smallest distances.
struct BoundedMaxHeap {
    capacity: usize,
    entries: Vec<(f32, usize)>, // (distance, embedding_index)
}

impl BoundedMaxHeap {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: Vec::with_capacity(capacity + 1),
        }
    }

    fn push(&mut self, dist: f32, index: usize) {
        if self.entries.len() < self.capacity {
            self.entries.push((dist, index));
            // Sift up to maintain max-heap property
            let mut i = self.entries.len() - 1;
            while i > 0 {
                let parent = (i - 1) / 2;
                if self.entries[i].0 > self.entries[parent].0 {
                    self.entries.swap(i, parent);
                    i = parent;
                } else {
                    break;
                }
            }
        } else if dist < self.entries[0].0 {
            self.entries[0] = (dist, index);
            // Sift down to maintain max-heap property
            let mut i = 0;
            let n = self.entries.len();
            loop {
                let left = 2 * i + 1;
                let right = 2 * i + 2;
                let mut largest = i;
                if left < n && self.entries[left].0 > self.entries[largest].0 {
                    largest = left;
                }
                if right < n && self.entries[right].0 > self.entries[largest].0 {
                    largest = right;
                }
                if largest == i {
                    break;
                }
                self.entries.swap(i, largest);
                i = largest;
            }
        }
    }

    fn max_dist(&self) -> f32 {
        if self.entries.len() < self.capacity {
            f32::INFINITY
        } else {
            self.entries[0].0
        }
    }
}
