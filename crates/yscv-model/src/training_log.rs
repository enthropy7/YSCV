use std::collections::HashMap;

/// Records per-epoch training metrics.
#[derive(Debug, Clone, Default)]
pub struct TrainingLog {
    entries: Vec<HashMap<String, f32>>,
}

impl TrainingLog {
    /// Create a new empty training log.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one epoch's metrics.
    pub fn log_epoch(&mut self, metrics: HashMap<String, f32>) {
        self.entries.push(metrics);
    }

    /// All logged entries.
    pub fn entries(&self) -> &[HashMap<String, f32>] {
        &self.entries
    }

    /// Extract the history of a single metric across all epochs.
    ///
    /// Epochs that did not record the metric are skipped.
    pub fn get_metric_history(&self, name: &str) -> Vec<f32> {
        self.entries
            .iter()
            .filter_map(|e| e.get(name).copied())
            .collect()
    }

    /// Number of epochs logged so far.
    pub fn num_epochs(&self) -> usize {
        self.entries.len()
    }

    /// Export the log as a CSV string.
    ///
    /// The header row contains the union of all metric names across every epoch,
    /// sorted alphabetically for deterministic output. Missing values are empty.
    pub fn to_csv(&self) -> String {
        if self.entries.is_empty() {
            return String::new();
        }

        // Collect all unique keys, sorted.
        let mut keys: Vec<String> = self
            .entries
            .iter()
            .flat_map(|e| e.keys().cloned())
            .collect();
        keys.sort();
        keys.dedup();

        let mut out = keys.join(",");
        out.push('\n');

        for entry in &self.entries {
            let row: Vec<String> = keys
                .iter()
                .map(|k| match entry.get(k) {
                    Some(v) => v.to_string(),
                    None => String::new(),
                })
                .collect();
            out.push_str(&row.join(","));
            out.push('\n');
        }

        out
    }

    /// Export the log as JSONL (one JSON object per epoch).
    ///
    /// Compatible with TensorBoard's `--logdir` when saved as `.jsonl`.
    /// Each line: `{"epoch": N, "loss": 0.5, "accuracy": 0.9, ...}`
    pub fn to_jsonl(&self) -> String {
        let mut out = String::new();
        for (epoch, entry) in self.entries.iter().enumerate() {
            out.push_str("{\"epoch\":");
            out.push_str(&epoch.to_string());
            let mut keys: Vec<&String> = entry.keys().collect();
            keys.sort();
            for key in keys {
                let val = entry[key];
                out.push_str(",\"");
                out.push_str(key);
                out.push_str("\":");
                if val.is_finite() {
                    out.push_str(&format!("{val:.6}"));
                } else {
                    out.push_str("null");
                }
            }
            out.push_str("}\n");
        }
        out
    }

    /// Write the log to a JSONL file (append-friendly).
    pub fn save_jsonl(&self, path: &std::path::Path) -> std::io::Result<()> {
        std::fs::write(path, self.to_jsonl())
    }

    /// Append a single epoch's metrics to a JSONL file (streaming mode).
    pub fn append_epoch_jsonl(
        path: &std::path::Path,
        epoch: usize,
        metrics: &HashMap<String, f32>,
    ) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        write!(file, "{{\"epoch\":{epoch}")?;
        let mut keys: Vec<&String> = metrics.keys().collect();
        keys.sort();
        for key in keys {
            let val = metrics[key];
            if val.is_finite() {
                write!(file, ",\"{key}\":{val:.6}")?;
            } else {
                write!(file, ",\"{key}\":null")?;
            }
        }
        writeln!(file, "}}")?;
        Ok(())
    }
}
