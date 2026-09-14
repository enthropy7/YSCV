use crate::VideoError;

#[cfg(feature = "native-camera")]
use std::fmt;
#[cfg(feature = "native-camera")]
use std::time::Instant;

#[cfg(feature = "native-camera")]
use nokhwa::Camera;
#[cfg(feature = "native-camera")]
use nokhwa::pixel_format::RgbFormat;
#[cfg(feature = "native-camera")]
use nokhwa::utils::{ApiBackend, CameraIndex, RequestedFormat, RequestedFormatType, Resolution};

#[cfg(feature = "native-camera")]
use super::convert::{micros_to_u64, rgb8_bytes_to_frame};
use super::frame::{Frame, Rgb8Frame};
use super::source::FrameSource;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CameraConfig {
    pub device_index: u32,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            device_index: 0,
            width: 640,
            height: 480,
            fps: 30,
        }
    }
}

impl CameraConfig {
    pub const fn validate(&self) -> Result<(), VideoError> {
        if self.width == 0 || self.height == 0 {
            return Err(VideoError::InvalidCameraResolution {
                width: self.width,
                height: self.height,
            });
        }
        if self.fps == 0 {
            return Err(VideoError::InvalidCameraFps { fps: self.fps });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CameraDeviceInfo {
    pub index: u32,
    pub label: String,
}

#[cfg(feature = "native-camera")]
pub fn list_camera_devices() -> Result<Vec<CameraDeviceInfo>, VideoError> {
    let backend = preferred_camera_backend();
    let cameras = match nokhwa::query(backend) {
        Ok(cameras) => cameras,
        Err(primary_err) if backend != ApiBackend::Auto => {
            nokhwa::query(ApiBackend::Auto).map_err(|fallback_err| {
                VideoError::Source(format!(
                    "failed to enumerate cameras with preferred backend {backend:?}: {primary_err}; \
                     auto fallback failed: {fallback_err}"
                ))
            })?
        }
        Err(err) => {
            return Err(VideoError::Source(format!(
                "failed to enumerate cameras with backend {backend:?}: {err}"
            )));
        }
    };
    let mut devices = Vec::with_capacity(cameras.len());
    for (idx, info) in cameras.into_iter().enumerate() {
        let fallback_index = u32::try_from(idx).unwrap_or(u32::MAX);
        let index = match info.index() {
            CameraIndex::Index(value) => *value,
            CameraIndex::String(value) => value.parse::<u32>().unwrap_or(fallback_index),
        };
        let human_name = info.human_name().trim().to_string();
        let description = info.description().trim().to_string();
        let label = if description.is_empty() {
            human_name
        } else {
            format!("{human_name} ({description})")
        };
        devices.push(CameraDeviceInfo { index, label });
    }
    devices.sort_by(|left, right| {
        left.index
            .cmp(&right.index)
            .then_with(|| left.label.cmp(&right.label))
    });
    devices.dedup_by(|left, right| left.index == right.index && left.label == right.label);
    Ok(devices)
}

#[cfg(not(feature = "native-camera"))]
pub const fn list_camera_devices() -> Result<Vec<CameraDeviceInfo>, VideoError> {
    Err(VideoError::CameraBackendDisabled)
}

pub fn resolve_camera_device(query: &str) -> Result<CameraDeviceInfo, VideoError> {
    let devices = list_camera_devices()?;
    select_camera_device(&devices, query)
}

pub fn resolve_camera_device_index(query: &str) -> Result<u32, VideoError> {
    Ok(resolve_camera_device(query)?.index)
}

pub fn query_camera_devices(query: &str) -> Result<Vec<CameraDeviceInfo>, VideoError> {
    let devices = list_camera_devices()?;
    filter_camera_devices(&devices, query)
}

pub fn filter_camera_devices(
    devices: &[CameraDeviceInfo],
    query: &str,
) -> Result<Vec<CameraDeviceInfo>, VideoError> {
    let normalized_query = query.trim();
    if normalized_query.is_empty() {
        return Err(VideoError::InvalidCameraDeviceQuery {
            query: query.to_string(),
        });
    }

    let index_query = normalized_query.parse::<u32>().ok();
    let query_lc = normalized_query.to_lowercase();
    let mut matches = devices
        .iter()
        .filter(|device| {
            if let Some(index_query) = index_query
                && device.index == index_query
            {
                return true;
            }
            device.label.to_lowercase().contains(&query_lc)
        })
        .cloned()
        .collect::<Vec<_>>();

    matches.sort_by(|left, right| {
        left.index
            .cmp(&right.index)
            .then_with(|| left.label.cmp(&right.label))
    });
    matches.dedup_by(|left, right| left.index == right.index && left.label == right.label);
    Ok(matches)
}

pub(crate) fn select_camera_device(
    devices: &[CameraDeviceInfo],
    query: &str,
) -> Result<CameraDeviceInfo, VideoError> {
    let normalized_query = query.trim();
    if normalized_query.is_empty() {
        return Err(VideoError::InvalidCameraDeviceQuery {
            query: query.to_string(),
        });
    }

    if let Ok(index_query) = normalized_query.parse::<u32>() {
        let by_index = devices
            .iter()
            .filter(|device| device.index == index_query)
            .cloned()
            .collect::<Vec<_>>();
        if let Some(device) = unique_match(&by_index) {
            return Ok(device);
        }
        if by_index.len() > 1 {
            return Err(VideoError::CameraDeviceAmbiguous {
                query: normalized_query.to_string(),
                matches: format_device_matches(&by_index),
            });
        }
    }

    let query_lc = normalized_query.to_lowercase();

    let mut exact = Vec::new();
    let mut partial = Vec::new();
    for device in devices {
        let label_lc = device.label.to_lowercase();
        if label_lc == query_lc {
            exact.push(device.clone());
            continue;
        }
        if label_lc.contains(&query_lc) {
            partial.push(device.clone());
        }
    }

    if let Some(device) = unique_match(&exact) {
        return Ok(device);
    }
    if exact.len() > 1 {
        return Err(VideoError::CameraDeviceAmbiguous {
            query: normalized_query.to_string(),
            matches: format_device_matches(&exact),
        });
    }
    if let Some(device) = unique_match(&partial) {
        return Ok(device);
    }
    if partial.len() > 1 {
        return Err(VideoError::CameraDeviceAmbiguous {
            query: normalized_query.to_string(),
            matches: format_device_matches(&partial),
        });
    }
    Err(VideoError::CameraDeviceNotFound {
        query: normalized_query.to_string(),
    })
}

fn unique_match(matches: &[CameraDeviceInfo]) -> Option<CameraDeviceInfo> {
    if matches.len() == 1 {
        Some(matches[0].clone())
    } else {
        None
    }
}

fn format_device_matches(matches: &[CameraDeviceInfo]) -> Vec<String> {
    let mut values = matches
        .iter()
        .map(|device| format!("{}: {}", device.index, device.label))
        .collect::<Vec<_>>();
    values.sort();
    values
}

#[cfg(feature = "native-camera")]
pub struct CameraFrameSource {
    camera: Camera,
    next_index: u64,
    started_at: Instant,
}

#[cfg(feature = "native-camera")]
impl fmt::Debug for CameraFrameSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CameraFrameSource")
            .field("next_index", &self.next_index)
            .finish_non_exhaustive()
    }
}

#[cfg(not(feature = "native-camera"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CameraFrameSource;

#[cfg(feature = "native-camera")]
impl CameraFrameSource {
    pub fn open(config: CameraConfig) -> Result<Self, VideoError> {
        config.validate()?;
        let backend = preferred_camera_backend();
        let mut camera =
            open_camera_with_backend(config.device_index, backend).or_else(|primary| {
                if backend == ApiBackend::Auto {
                    return Err(primary);
                }
                open_camera_with_backend(config.device_index, ApiBackend::Auto).map_err(|fallback| {
                VideoError::Source(format!(
                    "failed to create camera source with preferred backend {backend:?}: {primary}; \
                     auto fallback failed: {fallback}"
                ))
            })
            })?;
        camera
            .set_resolution(Resolution::new(config.width, config.height))
            .map_err(|err| VideoError::Source(format!("failed to set camera resolution: {err}")))?;
        camera
            .set_frame_rate(config.fps)
            .map_err(|err| VideoError::Source(format!("failed to set camera frame rate: {err}")))?;
        camera
            .open_stream()
            .map_err(|err| VideoError::Source(format!("failed to open camera stream: {err}")))?;

        Ok(Self {
            camera,
            next_index: 0,
            started_at: Instant::now(),
        })
    }

    pub fn next_rgb8_frame(&mut self) -> Result<Option<Rgb8Frame>, VideoError> {
        let captured = self
            .camera
            .frame()
            .map_err(|err| VideoError::Source(format!("failed to read camera frame: {err}")))?;
        let resolution = captured.resolution();
        let width = usize::try_from(resolution.width()).map_err(|err| {
            VideoError::Source(format!("failed to convert frame width to usize: {err}"))
        })?;
        let mut height = usize::try_from(resolution.height()).map_err(|err| {
            VideoError::Source(format!("failed to convert frame height to usize: {err}"))
        })?;
        let buf = captured.buffer_bytes();
        // Some backends (notably macOS AVFoundation via nokhwa) report a
        // resolution that does not match the actual buffer size.  When the
        // width looks correct but height doesn't, derive the real height from
        // the buffer length so we don't reject the frame.
        let expected = width.saturating_mul(height).saturating_mul(3);
        if buf.len() != expected && width > 0 {
            let actual_pixels = buf.len() / 3;
            if actual_pixels > 0 && actual_pixels % width == 0 {
                height = actual_pixels / width;
            }
        }
        let timestamp_us = micros_to_u64(self.started_at.elapsed().as_micros());
        let frame = Rgb8Frame::from_bytes(self.next_index, timestamp_us, width, height, buf)?;
        self.next_index += 1;
        Ok(Some(frame))
    }
}

#[cfg(feature = "native-camera")]
const fn preferred_camera_backend() -> ApiBackend {
    if cfg!(target_os = "linux") {
        ApiBackend::Video4Linux
    } else if cfg!(target_os = "windows") {
        ApiBackend::MediaFoundation
    } else if cfg!(target_os = "macos") {
        ApiBackend::AVFoundation
    } else {
        ApiBackend::Auto
    }
}

#[cfg(feature = "native-camera")]
fn open_camera_with_backend(device_index: u32, backend: ApiBackend) -> Result<Camera, VideoError> {
    let requested =
        RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
    Camera::with_backend(CameraIndex::Index(device_index), requested, backend).map_err(|err| {
        VideoError::Source(format!(
            "failed to create camera source with backend {backend:?}: {err}"
        ))
    })
}

#[cfg(not(feature = "native-camera"))]
impl CameraFrameSource {
    pub fn open(config: CameraConfig) -> Result<Self, VideoError> {
        config.validate()?;
        Err(VideoError::CameraBackendDisabled)
    }

    pub const fn next_rgb8_frame(&mut self) -> Result<Option<Rgb8Frame>, VideoError> {
        Err(VideoError::CameraBackendDisabled)
    }
}

#[cfg(feature = "native-camera")]
impl FrameSource for CameraFrameSource {
    fn next_frame(&mut self) -> Result<Option<Frame>, VideoError> {
        let Some(raw_frame) = self.next_rgb8_frame()? else {
            return Ok(None);
        };
        let frame = rgb8_bytes_to_frame(
            raw_frame.index(),
            raw_frame.timestamp_us(),
            raw_frame.width(),
            raw_frame.height(),
            raw_frame.data(),
        )?;
        Ok(Some(frame))
    }
}

#[cfg(not(feature = "native-camera"))]
impl FrameSource for CameraFrameSource {
    fn next_frame(&mut self) -> Result<Option<Frame>, VideoError> {
        Err(VideoError::CameraBackendDisabled)
    }
}
