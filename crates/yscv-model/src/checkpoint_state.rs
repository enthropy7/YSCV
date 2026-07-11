//! Save and restore full training checkpoints (model weights + optimizer state).
//!
//! A training checkpoint bundles model parameters and optimizer state into a
//! single binary file so training can be resumed exactly where it left off.

use std::collections::HashMap;
use std::path::Path;

use yscv_tensor::Tensor;

use super::weights::{load_weights, save_weights};
use crate::ModelError;

/// Prefix used to distinguish optimizer state tensors from model weights.
const OPT_PREFIX: &str = "__opt__.";

/// Save a full training checkpoint: model weights + optimizer state.
///
/// Both maps use string keys. Optimizer state keys are automatically prefixed
/// to avoid collisions with model weight names.
pub fn save_training_checkpoint(
    path: &Path,
    model_weights: &HashMap<String, Tensor>,
    optimizer_state: &HashMap<String, Tensor>,
) -> Result<(), ModelError> {
    let mut combined = model_weights.clone();
    for (key, tensor) in optimizer_state {
        combined.insert(format!("{OPT_PREFIX}{key}"), tensor.clone());
    }
    save_weights(path, &combined)
}

/// Load a full training checkpoint, splitting model weights from optimizer state.
///
/// Returns `(model_weights, optimizer_state)`.
pub fn load_training_checkpoint(
    path: &Path,
) -> Result<(HashMap<String, Tensor>, HashMap<String, Tensor>), ModelError> {
    let all = load_weights(path)?;
    let mut model_weights = HashMap::default();
    let mut optimizer_state = HashMap::default();

    for (key, tensor) in all {
        if let Some(stripped) = key.strip_prefix(OPT_PREFIX) {
            optimizer_state.insert(stripped.to_owned(), tensor);
        } else {
            model_weights.insert(key, tensor);
        }
    }

    Ok((model_weights, optimizer_state))
}

/// Flatten SGD velocity buffers into a string-keyed map for serialization.
///
/// Keys: `"sgd.{param_id}.velocity"`
pub fn sgd_state_to_map(velocity: &HashMap<u64, Tensor>) -> HashMap<String, Tensor> {
    velocity
        .iter()
        .map(|(id, t)| (format!("sgd.{id}.velocity"), t.clone()))
        .collect()
}

/// Restore SGD velocity buffers from a string-keyed map.
pub fn sgd_state_from_map(map: &HashMap<String, Tensor>) -> HashMap<u64, Tensor> {
    let mut velocity = HashMap::default();
    for (key, tensor) in map {
        if let Some(rest) = key.strip_prefix("sgd.")
            && let Some(id_str) = rest.strip_suffix(".velocity")
            && let Ok(id) = id_str.parse::<u64>()
        {
            velocity.insert(id, tensor.clone());
        }
    }
    velocity
}

/// Flatten Adam/AdamW state into a string-keyed map for serialization.
///
/// Keys: `"adam.{param_id}.m"`, `"adam.{param_id}.v"`, `"adam.{param_id}.step"`
pub fn adam_state_to_map(state: &[(u64, Tensor, Tensor, u64)]) -> HashMap<String, Tensor> {
    let mut map = HashMap::default();
    for (id, m, v, step) in state {
        map.insert(format!("adam.{id}.m"), m.clone());
        map.insert(format!("adam.{id}.v"), v.clone());
        // Store step as a scalar tensor.
        map.insert(
            format!("adam.{id}.step"),
            Tensor::from_vec(vec![1], vec![*step as f32]).expect("scalar shape matches data"),
        );
    }
    map
}

/// Restore Adam/AdamW state from a string-keyed map.
///
/// Returns `Vec<(param_id, first_moment, second_moment, step)>`.
pub fn adam_state_from_map(map: &HashMap<String, Tensor>) -> Vec<(u64, Tensor, Tensor, u64)> {
    // Collect unique param IDs from "adam.{id}.m" keys.
    let mut ids: Vec<u64> = map
        .keys()
        .filter_map(|k| {
            k.strip_prefix("adam.")
                .and_then(|rest| rest.strip_suffix(".m"))
                .and_then(|id_str| id_str.parse::<u64>().ok())
        })
        .collect();
    ids.sort();
    ids.dedup();

    ids.into_iter()
        .filter_map(|id| {
            let m = map.get(&format!("adam.{id}.m"))?.clone();
            let v = map.get(&format!("adam.{id}.v"))?.clone();
            let step = map
                .get(&format!("adam.{id}.step"))
                .map(|t| t.data()[0] as u64)
                .unwrap_or(0);
            Some((id, m, v, step))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_save_load_training_checkpoint_roundtrip() {
        let dir = std::env::temp_dir().join("yscv_checkpoint_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("checkpoint.bin");

        let mut model_weights = HashMap::default();
        model_weights.insert(
            "layer.0.weight".to_string(),
            Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        );

        let mut opt_state = HashMap::default();
        opt_state.insert(
            "sgd.0.velocity".to_string(),
            Tensor::from_vec(vec![2, 2], vec![0.1, 0.2, 0.3, 0.4]).unwrap(),
        );

        save_training_checkpoint(&path, &model_weights, &opt_state).unwrap();

        let (loaded_weights, loaded_opt) = load_training_checkpoint(&path).unwrap();

        assert!(loaded_weights.contains_key("layer.0.weight"));
        assert!(loaded_opt.contains_key("sgd.0.velocity"));
        assert_eq!(
            loaded_weights["layer.0.weight"].data(),
            &[1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(loaded_opt["sgd.0.velocity"].data(), &[0.1, 0.2, 0.3, 0.4]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_sgd_state_roundtrip() {
        let mut velocity = HashMap::default();
        velocity.insert(
            42u64,
            Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap(),
        );

        let map = sgd_state_to_map(&velocity);
        let restored = sgd_state_from_map(&map);

        assert!(restored.contains_key(&42));
        assert_eq!(restored[&42].data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_adam_state_roundtrip() {
        let state = vec![(
            7u64,
            Tensor::from_vec(vec![2], vec![0.1, 0.2]).unwrap(),
            Tensor::from_vec(vec![2], vec![0.01, 0.02]).unwrap(),
            100u64,
        )];

        let map = adam_state_to_map(&state);
        let restored = adam_state_from_map(&map);

        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0].0, 7);
        assert_eq!(restored[0].3, 100);
    }
}
