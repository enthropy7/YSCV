use bytes::Bytes;
use yscv_tensor::Tensor;

use crate::VideoError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    GrayF32,
    RgbF32,
}

/// Raw RGB8 frame payload that can be consumed without f32 conversion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rgb8Frame {
    index: u64,
    timestamp_us: u64,
    width: usize,
    height: usize,
    data: Bytes,
}

impl Rgb8Frame {
    pub fn new(
        index: u64,
        timestamp_us: u64,
        width: usize,
        height: usize,
        data: Vec<u8>,
    ) -> Result<Self, VideoError> {
        Self::from_bytes(index, timestamp_us, width, height, data.into())
    }

    pub fn from_bytes(
        index: u64,
        timestamp_us: u64,
        width: usize,
        height: usize,
        data: Bytes,
    ) -> Result<Self, VideoError> {
        let expected = width
            .checked_mul(height)
            .and_then(|pixels| pixels.checked_mul(3))
            .ok_or_else(|| {
                VideoError::Source(format!(
                    "raw frame dimensions overflow for width={width}, height={height}"
                ))
            })?;
        if data.len() != expected {
            return Err(VideoError::RawFrameSizeMismatch {
                expected,
                got: data.len(),
            });
        }
        Ok(Self {
            index,
            timestamp_us,
            width,
            height,
            data,
        })
    }

    pub const fn index(&self) -> u64 {
        self.index
    }

    pub const fn timestamp_us(&self) -> u64 {
        self.timestamp_us
    }

    pub const fn width(&self) -> usize {
        self.width
    }

    pub const fn height(&self) -> usize {
        self.height
    }

    pub fn data(&self) -> &[u8] {
        &self.data
    }

    pub fn into_data(self) -> Vec<u8> {
        self.data.to_vec()
    }

    pub fn into_bytes(self) -> Bytes {
        self.data
    }
}

/// One decoded/produced frame in HWC tensor layout.
#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    index: u64,
    timestamp_us: u64,
    pixel_format: PixelFormat,
    image: Tensor,
}

impl Frame {
    pub fn new(index: u64, timestamp_us: u64, image: Tensor) -> Result<Self, VideoError> {
        if image.rank() != 3 {
            return Err(VideoError::InvalidFrameShape {
                got: image.shape().to_vec(),
            });
        }
        let channels = image.shape()[2];
        let pixel_format = match channels {
            1 => PixelFormat::GrayF32,
            3 => PixelFormat::RgbF32,
            _ => return Err(VideoError::UnsupportedChannelCount { channels }),
        };
        Ok(Self {
            index,
            timestamp_us,
            pixel_format,
            image,
        })
    }

    pub const fn index(&self) -> u64 {
        self.index
    }

    pub const fn timestamp_us(&self) -> u64 {
        self.timestamp_us
    }

    pub const fn pixel_format(&self) -> PixelFormat {
        self.pixel_format
    }

    pub const fn image(&self) -> &Tensor {
        &self.image
    }
}
