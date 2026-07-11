use rustc_hash::FxHashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use super::callbacks::TrainingCallback;
use super::error::ModelError;

// ---------- CRC32C (Castagnoli) ----------

const CRC32C_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0x82F6_3B78;
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
};

fn crc32c(data: &[u8]) -> u32 {
    let mut crc: u32 = !0;
    for &b in data {
        crc = CRC32C_TABLE[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    !crc
}

fn masked_crc32c(data: &[u8]) -> u32 {
    let crc = crc32c(data);
    crc.rotate_right(15).wrapping_add(0xa282_ead8)
}

// ---------- Protobuf helpers ----------

fn encode_varint(mut value: u64, buf: &mut Vec<u8>) {
    while value >= 0x80 {
        buf.push((value as u8) | 0x80);
        value >>= 7;
    }
    buf.push(value as u8);
}

/// Encode a Summary.Value proto: field 1 = tag (string), field 2 = simple_value (float)
fn encode_summary_value(tag: &str, value: f32) -> Vec<u8> {
    let mut buf = Vec::new();
    // field 1, wire type 2 (length-delimited): tag string
    buf.push(0x0a);
    encode_varint(tag.len() as u64, &mut buf);
    buf.extend_from_slice(tag.as_bytes());
    // field 2, wire type 5 (32-bit): simple_value float
    buf.push(0x15);
    buf.extend_from_slice(&value.to_le_bytes());
    buf
}

/// Encode a Summary proto: repeated field 1 = Value
fn encode_summary(tag: &str, value: f32) -> Vec<u8> {
    let val = encode_summary_value(tag, value);
    let mut buf = Vec::new();
    // field 1, wire type 2
    buf.push(0x0a);
    encode_varint(val.len() as u64, &mut buf);
    buf.extend_from_slice(&val);
    buf
}

/// Encode an Event proto with wall_time, step, and summary.
fn encode_event_summary(wall_time: f64, step: i64, tag: &str, value: f32) -> Vec<u8> {
    let summary = encode_summary(tag, value);
    let mut buf = Vec::new();
    // field 1, wire type 1 (64-bit): wall_time (double)
    buf.push(0x09);
    buf.extend_from_slice(&wall_time.to_le_bytes());
    // field 2, wire type 0 (varint): step
    buf.push(0x10);
    encode_varint(step as u64, &mut buf);
    // field 5, wire type 2 (length-delimited): summary
    buf.push(0x2a);
    encode_varint(summary.len() as u64, &mut buf);
    buf.extend_from_slice(&summary);
    buf
}

/// Encode an Event proto with file_version string (field 6).
fn encode_event_file_version(wall_time: f64, step: i64, version: &str) -> Vec<u8> {
    let mut buf = Vec::new();
    // field 1, wire type 1: wall_time
    buf.push(0x09);
    buf.extend_from_slice(&wall_time.to_le_bytes());
    // field 2, wire type 0: step
    buf.push(0x10);
    encode_varint(step as u64, &mut buf);
    // field 6, wire type 2 (length-delimited): file_version string
    buf.push(0x32);
    encode_varint(version.len() as u64, &mut buf);
    buf.extend_from_slice(version.as_bytes());
    buf
}

// ---------- TensorBoardWriter ----------

/// Writes TensorBoard-compatible event files in TFRecord format.
pub struct TensorBoardWriter {
    file: BufWriter<File>,
}

impl TensorBoardWriter {
    /// Create a new writer that stores events under `log_dir`.
    ///
    /// Creates `log_dir` if it does not exist and writes the initial
    /// `file_version` event required by TensorBoard.
    pub fn new(log_dir: impl AsRef<Path>) -> Result<Self, ModelError> {
        let log_dir = log_dir.as_ref();
        fs::create_dir_all(log_dir).map_err(|e| ModelError::DatasetLoadIo {
            path: log_dir.display().to_string(),
            message: e.to_string(),
        })?;

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let hostname = "localhost";
        let filename = format!("events.out.tfevents.{timestamp}.{hostname}");
        let filepath = log_dir.join(filename);

        let f = File::create(&filepath).map_err(|e| ModelError::DatasetLoadIo {
            path: filepath.display().to_string(),
            message: e.to_string(),
        })?;
        let mut writer = Self {
            file: BufWriter::new(f),
        };

        let wall_time = timestamp as f64;
        let event = encode_event_file_version(wall_time, 0, "brain.Event:2");
        writer.write_record(&event)?;

        Ok(writer)
    }

    /// Log a scalar value with the given tag and step.
    pub fn add_scalar(&mut self, tag: &str, value: f32, step: i64) -> Result<(), ModelError> {
        let wall_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        let event = encode_event_summary(wall_time, step, tag, value);
        self.write_record(&event)
    }

    /// Flush buffered data to disk.
    pub fn flush(&mut self) -> Result<(), ModelError> {
        self.file.flush().map_err(|e| ModelError::DatasetLoadIo {
            path: "tensorboard events".to_string(),
            message: e.to_string(),
        })
    }

    /// Write a single TFRecord.
    fn write_record(&mut self, data: &[u8]) -> Result<(), ModelError> {
        let len = data.len() as u64;
        let len_bytes = len.to_le_bytes();
        let len_crc = masked_crc32c(&len_bytes);
        let data_crc = masked_crc32c(data);

        let w = &mut self.file;
        w.write_all(&len_bytes)
            .and_then(|_| w.write_all(&len_crc.to_le_bytes()))
            .and_then(|_| w.write_all(data))
            .and_then(|_| w.write_all(&data_crc.to_le_bytes()))
            .map_err(|e| ModelError::DatasetLoadIo {
                path: "tensorboard events".to_string(),
                message: e.to_string(),
            })
    }
}

// ---------- TensorBoardCallback ----------

/// Training callback that logs scalar metrics to TensorBoard event files.
pub struct TensorBoardCallback {
    writer: TensorBoardWriter,
    global_step: i64,
}

impl TensorBoardCallback {
    /// Create a new callback writing events to `log_dir`.
    pub fn new(log_dir: impl AsRef<Path>) -> Result<Self, ModelError> {
        Ok(Self {
            writer: TensorBoardWriter::new(log_dir)?,
            global_step: 0,
        })
    }

    /// Returns the current global step counter.
    pub fn global_step(&self) -> i64 {
        self.global_step
    }
}

impl TrainingCallback for TensorBoardCallback {
    fn on_epoch_end(&mut self, _epoch: usize, metrics: &FxHashMap<String, f32>) -> bool {
        self.global_step += 1;
        for (tag, &value) in metrics {
            let _ = self.writer.add_scalar(tag, value, self.global_step);
        }
        let _ = self.writer.flush();
        false
    }

    fn on_batch_end(&mut self, _epoch: usize, _batch: usize, loss: f32) {
        self.global_step += 1;
        let _ = self.writer.add_scalar("batch_loss", loss, self.global_step);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masked_crc32c_known_values() {
        // Empty data
        let crc = crc32c(b"");
        assert_eq!(crc, 0x0000_0000);

        // Verify masking transforms the CRC
        let m = masked_crc32c(b"hello");
        assert_ne!(m, 0);
    }

    #[test]
    fn encode_varint_simple() {
        let mut buf = Vec::new();
        encode_varint(150, &mut buf);
        assert_eq!(buf, &[0x96, 0x01]);
    }

    #[test]
    fn writer_creates_event_file() {
        let dir = std::env::temp_dir().join("yscv_tb_test");
        let _ = fs::remove_dir_all(&dir);
        {
            let mut w = TensorBoardWriter::new(&dir).unwrap();
            w.add_scalar("loss", 0.5, 1).unwrap();
            w.add_scalar("loss", 0.3, 2).unwrap();
            w.flush().unwrap();
        }
        // Verify a file was created
        let entries: Vec<_> = fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()).collect();
        assert_eq!(entries.len(), 1);
        let name = entries[0].file_name().to_string_lossy().to_string();
        assert!(name.starts_with("events.out.tfevents."));
        // File must be non-empty (header + 2 records)
        let meta = entries[0].metadata().unwrap();
        assert!(meta.len() > 50);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn callback_logs_metrics() {
        let dir = std::env::temp_dir().join("yscv_tb_cb_test");
        let _ = fs::remove_dir_all(&dir);
        {
            let mut cb = TensorBoardCallback::new(&dir).unwrap();
            let mut metrics = FxHashMap::default();
            metrics.insert("train_loss".to_string(), 0.42f32);
            let stop = cb.on_epoch_end(0, &metrics);
            assert!(!stop);
            assert_eq!(cb.global_step(), 1);
        }
        let _ = fs::remove_dir_all(&dir);
    }
}
