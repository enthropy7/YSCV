use std::fs;
use std::path::Path;

use yscv_tensor::Tensor;

use super::RecognizeError;
use super::similarity::cosine_similarity_prevalidated;
use super::snapshot::{IdentitySnapshot, RecognizerSnapshot};
use super::types::{IdentityEmbedding, Recognition};
use super::validate::{validate_embedding, validate_embedding_slice, validate_threshold};
use super::vp_tree::VpTree;

#[derive(Debug, Clone)]
pub struct Recognizer {
    threshold: f32,
    entries: Vec<IdentityEmbedding>,
    embedding_dim: Option<usize>,
    index: Option<VpTree>,
}

impl Recognizer {
    pub fn new(threshold: f32) -> Result<Self, RecognizeError> {
        validate_threshold(threshold)?;
        Ok(Self {
            threshold,
            entries: Vec::new(),
            embedding_dim: None,
            index: None,
        })
    }

    pub const fn threshold(&self) -> f32 {
        self.threshold
    }

    pub fn set_threshold(&mut self, threshold: f32) -> Result<(), RecognizeError> {
        validate_threshold(threshold)?;
        self.threshold = threshold;
        Ok(())
    }

    pub fn enroll(
        &mut self,
        id: impl Into<String>,
        embedding: Tensor,
    ) -> Result<(), RecognizeError> {
        validate_embedding(&embedding)?;
        let id = id.into();
        if self.entries.iter().any(|entry| entry.id == id) {
            return Err(RecognizeError::DuplicateIdentity { id });
        }
        self.enforce_dim(embedding.len())?;
        self.entries.push(IdentityEmbedding { id, embedding });
        Ok(())
    }

    pub fn enroll_or_replace(
        &mut self,
        id: impl Into<String>,
        embedding: Tensor,
    ) -> Result<(), RecognizeError> {
        validate_embedding(&embedding)?;
        self.enforce_dim(embedding.len())?;
        let id = id.into();
        if let Some(existing) = self.entries.iter_mut().find(|entry| entry.id == id) {
            existing.embedding = embedding;
            return Ok(());
        }
        self.entries.push(IdentityEmbedding { id, embedding });
        Ok(())
    }

    pub fn remove(&mut self, id: &str) -> bool {
        if let Some(position) = self.entries.iter().position(|entry| entry.id == id) {
            self.entries.remove(position);
            if self.entries.is_empty() {
                self.embedding_dim = None;
            }
            true
        } else {
            false
        }
    }

    pub fn identities(&self) -> &[IdentityEmbedding] {
        &self.entries
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.embedding_dim = None;
    }

    pub fn recognize(&self, embedding: &Tensor) -> Result<Recognition, RecognizeError> {
        validate_embedding(embedding)?;
        self.recognize_prevalidated(embedding.data())
    }

    pub fn recognize_slice(&self, embedding: &[f32]) -> Result<Recognition, RecognizeError> {
        validate_embedding_slice(embedding)?;
        self.recognize_prevalidated(embedding)
    }

    fn recognize_prevalidated(&self, embedding: &[f32]) -> Result<Recognition, RecognizeError> {
        if let Some(expected_dim) = self.embedding_dim {
            if expected_dim != embedding.len() {
                return Err(RecognizeError::EmbeddingDimMismatch {
                    expected: expected_dim,
                    got: embedding.len(),
                });
            }
        } else {
            return Ok(Recognition {
                identity: None,
                score: 0.0,
            });
        }

        let mut best_index = None::<usize>;
        let mut best_score = -1.0f32;
        for (index, entry) in self.entries.iter().enumerate() {
            let score = cosine_similarity_prevalidated(embedding, entry.embedding.data())?;
            if score > best_score {
                best_score = score;
                best_index = Some(index);
            }
        }

        if best_score >= self.threshold {
            Ok(Recognition {
                identity: best_index.map(|index| self.entries[index].id.clone()),
                score: best_score,
            })
        } else {
            Ok(Recognition {
                identity: None,
                score: best_score,
            })
        }
    }

    pub fn to_snapshot(&self) -> RecognizerSnapshot {
        let mut identities = Vec::with_capacity(self.entries.len());
        for entry in &self.entries {
            identities.push(IdentitySnapshot {
                id: entry.id.clone(),
                embedding: entry.embedding.data().to_vec(),
            });
        }

        RecognizerSnapshot {
            threshold: self.threshold,
            identities,
        }
    }

    pub fn from_snapshot(snapshot: RecognizerSnapshot) -> Result<Self, RecognizeError> {
        let mut recognizer = Self::new(snapshot.threshold)?;
        for entry in snapshot.identities {
            let embedding = Tensor::from_vec(vec![entry.embedding.len()], entry.embedding)
                .map_err(|err| RecognizeError::Serialization {
                    message: err.to_string(),
                })?;
            recognizer.enroll(entry.id, embedding)?;
        }
        Ok(recognizer)
    }

    pub fn to_json_pretty(&self) -> Result<String, RecognizeError> {
        serde_json::to_string_pretty(&self.to_snapshot()).map_err(|err| {
            RecognizeError::Serialization {
                message: err.to_string(),
            }
        })
    }

    pub fn from_json(json: &str) -> Result<Self, RecognizeError> {
        let snapshot: RecognizerSnapshot =
            serde_json::from_str(json).map_err(|err| RecognizeError::Serialization {
                message: err.to_string(),
            })?;
        Self::from_snapshot(snapshot)
    }

    pub fn save_json_file(&self, path: impl AsRef<Path>) -> Result<(), RecognizeError> {
        let json = self.to_json_pretty()?;
        fs::write(path, json).map_err(|err| RecognizeError::Io {
            message: err.to_string(),
        })
    }

    pub fn load_json_file(path: impl AsRef<Path>) -> Result<Self, RecognizeError> {
        let json = fs::read_to_string(path).map_err(|err| RecognizeError::Io {
            message: err.to_string(),
        })?;
        Self::from_json(&json)
    }

    /// Build a VP-tree index from the current gallery for fast nearest-neighbor search.
    pub fn build_index(&mut self) {
        let entries: Vec<(String, Vec<f32>)> = self
            .entries
            .iter()
            .map(|e| (e.id.clone(), e.embedding.data().to_vec()))
            .collect();
        self.index = Some(VpTree::build(entries));
    }

    /// Search using the VP-tree index if available, otherwise fall back to linear scan.
    ///
    /// Returns the `k` nearest identities that meet the recognition threshold.
    pub fn search_indexed(
        &self,
        embedding: &Tensor,
        k: usize,
    ) -> Result<Vec<Recognition>, RecognizeError> {
        validate_embedding(embedding)?;

        if let Some(expected_dim) = self.embedding_dim {
            if expected_dim != embedding.len() {
                return Err(RecognizeError::EmbeddingDimMismatch {
                    expected: expected_dim,
                    got: embedding.len(),
                });
            }
        } else {
            return Ok(Vec::new());
        }

        if let Some(ref index) = self.index {
            let results = index.query(embedding.data(), k);
            Ok(results
                .into_iter()
                .filter_map(|r| {
                    let score = 1.0 - r.distance;
                    if score >= self.threshold {
                        Some(Recognition {
                            identity: Some(r.id),
                            score,
                        })
                    } else {
                        None
                    }
                })
                .collect())
        } else {
            // Fall back to linear scan: collect all scores, sort, take top k.
            let mut scored: Vec<(usize, f32)> = Vec::with_capacity(self.entries.len());
            for (i, entry) in self.entries.iter().enumerate() {
                let score =
                    cosine_similarity_prevalidated(embedding.data(), entry.embedding.data())?;
                scored.push((i, score));
            }
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(k);
            Ok(scored
                .into_iter()
                .filter_map(|(i, score)| {
                    if score >= self.threshold {
                        Some(Recognition {
                            identity: Some(self.entries[i].id.clone()),
                            score,
                        })
                    } else {
                        None
                    }
                })
                .collect())
        }
    }

    const fn enforce_dim(&mut self, dim: usize) -> Result<(), RecognizeError> {
        if let Some(expected_dim) = self.embedding_dim {
            if expected_dim != dim {
                return Err(RecognizeError::EmbeddingDimMismatch {
                    expected: expected_dim,
                    got: dim,
                });
            }
        } else {
            self.embedding_dim = Some(dim);
        }
        Ok(())
    }
}
