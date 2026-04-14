#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

mod byte_track;
mod config;
mod deep_sort;
mod error;
pub mod hungarian;
mod kalman;
mod motion;
pub mod reid;
#[cfg(test)]
mod tests;
mod tracker;
mod types;

pub const CRATE_ID: &str = "yscv-track";

pub use byte_track::ByteTracker;
pub use config::TrackerConfig;
pub use deep_sort::{DeepSortConfig, DeepSortTrack, DeepSortTracker, TrackState};
pub use error::TrackError;
pub use hungarian::hungarian_assignment;
pub use kalman::KalmanFilter;
pub use reid::{ColorHistogramReId, ReIdExtractor, ReIdGallery};
pub use tracker::Tracker;
pub use types::{Track, TrackedDetection};
