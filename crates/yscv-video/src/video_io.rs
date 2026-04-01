use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;

use super::codec::VideoDecoder as _;
use super::error::VideoError;
use super::frame::Rgb8Frame;

/// Video container metadata.
#[derive(Debug, Clone)]
pub struct VideoMeta {
    pub width: u32,
    pub height: u32,
    pub frame_count: u32,
    pub fps: f32,
    pub properties: HashMap<String, String>,
}

/// Reads raw RGB8 frames from a simple uncompressed video file.
///
/// Format: [8 bytes: magic "RCVVIDEO"] [4 bytes: width LE] [4 bytes: height LE] [4 bytes: frame_count LE] [4 bytes: fps as f32 LE bits] [frames: width*height*3 bytes each].
pub struct RawVideoReader {
    pub meta: VideoMeta,
    data: Vec<u8>,
    frame_offset: usize,
    current_frame: u32,
}

const MAGIC: &[u8; 8] = b"RCVVIDEO";

impl RawVideoReader {
    /// Opens a raw video file for reading.
    pub fn open(path: &Path) -> Result<Self, VideoError> {
        let mut file = std::fs::File::open(path)
            .map_err(|e| VideoError::Source(format!("{}: {e}", path.display())))?;
        // Prevent OOM on large files
        let file_size = file.metadata().map(|m| m.len()).unwrap_or(0);
        if file_size > 2 * 1024 * 1024 * 1024 {
            return Err(VideoError::Source("raw video file too large (>2GB)".into()));
        }
        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .map_err(|e| VideoError::Source(e.to_string()))?;

        if data.len() < 24 || &data[..8] != MAGIC {
            return Err(VideoError::Source("invalid raw video file header".into()));
        }

        let width = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
        let height = u32::from_le_bytes([data[12], data[13], data[14], data[15]]);
        let frame_count = u32::from_le_bytes([data[16], data[17], data[18], data[19]]);
        let fps = f32::from_le_bytes([data[20], data[21], data[22], data[23]]);

        Ok(Self {
            meta: VideoMeta {
                width,
                height,
                frame_count,
                fps,
                properties: HashMap::new(),
            },
            data,
            frame_offset: 24,
            current_frame: 0,
        })
    }

    /// Reads the next frame, if available.
    pub fn next_frame(&mut self) -> Option<Rgb8Frame> {
        if self.current_frame >= self.meta.frame_count {
            return None;
        }
        let frame_size = (self.meta.width as usize)
            .checked_mul(self.meta.height as usize)
            .and_then(|x| x.checked_mul(3))?;
        let start = self
            .frame_offset
            .checked_add((self.current_frame as usize).checked_mul(frame_size)?)?;
        let end = start + frame_size;
        if end > self.data.len() {
            return None;
        }
        self.current_frame += 1;
        Rgb8Frame::from_bytes(
            self.current_frame as u64 - 1,
            0,
            self.meta.width as usize,
            self.meta.height as usize,
            bytes::Bytes::copy_from_slice(&self.data[start..end]),
        )
        .ok()
    }

    /// Resets to the beginning.
    pub fn seek_start(&mut self) {
        self.current_frame = 0;
    }

    /// Returns the frame count.
    pub fn frame_count(&self) -> u32 {
        self.meta.frame_count
    }
}

/// Writes raw RGB8 frames to a simple uncompressed video file.
pub struct RawVideoWriter {
    width: u32,
    height: u32,
    fps: f32,
    frames: Vec<Vec<u8>>,
}

impl RawVideoWriter {
    pub fn new(width: u32, height: u32, fps: f32) -> Self {
        Self {
            width,
            height,
            fps,
            frames: Vec::new(),
        }
    }

    /// Appends an RGB8 frame (must be width*height*3 bytes).
    pub fn push_frame(&mut self, rgb8_data: &[u8]) -> Result<(), VideoError> {
        let expected = self.width as usize * self.height as usize * 3;
        if rgb8_data.len() != expected {
            return Err(VideoError::Source(format!(
                "frame size mismatch: expected {expected}, got {}",
                rgb8_data.len()
            )));
        }
        self.frames.push(rgb8_data.to_vec());
        Ok(())
    }

    /// Writes the video to a file.
    pub fn save(&self, path: &Path) -> Result<(), VideoError> {
        let mut file = std::fs::File::create(path)
            .map_err(|e| VideoError::Source(format!("{}: {e}", path.display())))?;

        let wr = |f: &mut std::fs::File, d: &[u8]| -> Result<(), VideoError> {
            f.write_all(d)
                .map_err(|e| VideoError::Source(e.to_string()))
        };
        wr(&mut file, MAGIC)?;
        wr(&mut file, &self.width.to_le_bytes())?;
        wr(&mut file, &self.height.to_le_bytes())?;
        wr(&mut file, &(self.frames.len() as u32).to_le_bytes())?;
        wr(&mut file, &self.fps.to_le_bytes())?;

        for frame in &self.frames {
            wr(&mut file, frame)?;
        }

        Ok(())
    }

    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }
}

/// Image sequence reader: reads numbered image files as a video stream.
///
/// Pattern example: `frames/frame_%04d.png` -> reads `frame_0000.png`, `frame_0001.png`, etc.
pub struct ImageSequenceReader {
    pub width: usize,
    pub height: usize,
    paths: Vec<std::path::PathBuf>,
    current: usize,
}

impl ImageSequenceReader {
    /// Creates a reader from a sorted list of image file paths.
    pub fn from_paths(paths: Vec<std::path::PathBuf>) -> Self {
        Self {
            width: 0,
            height: 0,
            paths,
            current: 0,
        }
    }

    /// Returns the total number of frames.
    pub fn frame_count(&self) -> usize {
        self.paths.len()
    }

    /// Resets to the beginning.
    pub fn seek_start(&mut self) {
        self.current = 0;
    }

    /// Returns the next image path without loading.
    pub fn next_path(&mut self) -> Option<&Path> {
        if self.current >= self.paths.len() {
            return None;
        }
        let path = &self.paths[self.current];
        self.current += 1;
        Some(path)
    }
}

// ---------------------------------------------------------------------------
// MP4 / H.264 Video Reader
// ---------------------------------------------------------------------------

/// Detected video codec in an MP4 container.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mp4Codec {
    H264,
    Hevc,
}

/// Reads H.264 or HEVC encoded MP4 video files and decodes frames to RGB8.
///
/// Combines the MP4 box parser, NAL extraction, and the appropriate decoder
/// (H.264 or HEVC) into a single end-to-end reader. The codec is auto-detected
/// from the MP4 sample entry type (avc1/hvc1/hev1).
///
/// ```ignore
/// let mut reader = Mp4VideoReader::open("input.mp4")?;
/// while let Some(frame) = reader.next_frame()? {
///     // frame.rgb8_data is Vec<u8>, frame.width / frame.height
/// }
/// ```
/// Internal decoder — either H.264, HEVC, or hardware-accelerated.
enum Mp4Decoder {
    H264(super::h264_decoder::H264Decoder),
    Hevc(super::hevc_decoder::HevcDecoder),
    /// Hardware decoder (VideoToolbox/VAAPI/NVDEC/MediaFoundation).
    Hw(super::hw_decode::HwVideoDecoder),
}

pub struct Mp4VideoReader {
    decoder: Mp4Decoder,
    codec: Mp4Codec,
    /// File handle for lazy sample reading.
    file: Option<std::fs::File>,
    /// Audio track info (if present in MP4).
    audio: Option<super::audio::AudioTrackInfo>,
    /// Parameter set NALs (SPS/PPS/VPS) — small, kept in memory.
    param_nals_h264: Vec<super::codec::NalUnit>,
    param_nals_hevc: Vec<Vec<u8>>,
    /// Sample table: (file_offset, size) per video sample. ~12 bytes per frame.
    sample_table: Vec<(u64, u32)>,
    /// NAL length size from avcC/hvcC (typically 4).
    nal_length_size: usize,
    /// Current sample index.
    current_sample: usize,
    /// Whether parameter sets have been fed to the decoder.
    params_fed: bool,

    /// Parameter set init AU for HW decode (prepended to first sample).
    hw_init_au: Option<Vec<u8>>,
    /// Current HW sample index for streaming decode.
    hw_sample_idx: usize,
}

impl Mp4VideoReader {
    /// Open an MP4 file containing H.264 or HEVC video.
    ///
    /// Reads the entire file, parses MP4 boxes to find the `mdat` box,
    /// auto-detects the codec from sample entry types (avc1/hvc1/hev1),
    /// and extracts NAL units from the raw data.
    pub fn open(path: &Path) -> Result<Self, VideoError> {
        use std::io::{Read as _, Seek, SeekFrom};

        // Step 1: Read only enough to find moov box (scan box headers, not mdat)
        let mut file = std::fs::File::open(path)
            .map_err(|e| VideoError::Source(format!("{}: {e}", path.display())))?;
        let file_size = file.metadata().map(|m| m.len()).unwrap_or(0);

        // Read moov box: scan for it by reading box headers
        // moov is typically at the start or end of the file, usually < 5MB
        let mut moov_data: Vec<u8> = Vec::new();
        let mut pos = 0u64;
        let mut header_buf = [0u8; 8];
        #[allow(unused_assignments)]
        let mut detected_codec = Mp4Codec::H264;

        while pos < file_size {
            file.seek(SeekFrom::Start(pos))
                .map_err(|e| VideoError::Source(format!("seek: {e}")))?;
            if file.read_exact(&mut header_buf).is_err() {
                break;
            }
            let box_size =
                u32::from_be_bytes([header_buf[0], header_buf[1], header_buf[2], header_buf[3]])
                    as u64;
            let box_type = &header_buf[4..8];
            let real_size = if box_size == 1 {
                // Extended size
                let mut ext = [0u8; 8];
                let _ = file.read_exact(&mut ext);
                u64::from_be_bytes(ext)
            } else if box_size == 0 {
                file_size - pos
            } else {
                box_size
            };

            if box_type == b"moov" {
                let content_size = (real_size - 8) as usize;
                moov_data.resize(content_size, 0);
                let _ = file.read_exact(&mut moov_data);
                break;
            }
            pos += real_size.max(8);
        }

        if moov_data.is_empty() {
            return Err(VideoError::ContainerParse("no moov box found".into()));
        }

        // Detect codec from moov
        detected_codec = if moov_data.windows(4).any(|w| w == b"hvc1" || w == b"hev1") {
            Mp4Codec::Hevc
        } else {
            Mp4Codec::H264
        };

        // Step 2: Build sample table (just offsets + sizes, no data read)
        let nal_length_size = find_nal_length_size(&moov_data, detected_codec);
        let sample_table = build_sample_table(&moov_data)?;

        if sample_table.is_empty() {
            return Err(VideoError::ContainerParse("no video samples found".into()));
        }

        // Step 3: Extract parameter sets (tiny — SPS/PPS/VPS < 1KB)
        let mut param_nals_h264 = Vec::new();
        let mut param_nals_hevc = Vec::new();
        match detected_codec {
            Mp4Codec::Hevc => {
                param_nals_hevc = extract_hvcc_nals(&moov_data);
            }
            Mp4Codec::H264 => {
                param_nals_h264 = extract_avcc_nals(&moov_data);
            }
        }

        let decoder = match detected_codec {
            Mp4Codec::H264 => Mp4Decoder::H264(super::h264_decoder::H264Decoder::new()),
            Mp4Codec::Hevc => Mp4Decoder::Hevc(super::hevc_decoder::HevcDecoder::new()),
        };

        // Parse audio track info (if present)
        let audio = find_audio_trak(&moov_data).and_then(parse_mp4_audio_info);

        Ok(Self {
            decoder,
            codec: detected_codec,
            file: Some(file),
            audio,
            param_nals_h264,
            param_nals_hevc,
            sample_table,
            nal_length_size,
            current_sample: 0,
            params_fed: false,
            hw_init_au: None,
            hw_sample_idx: 0,
        })
    }

    /// Open with hardware decode (auto-detect best HW backend, SW fallback).
    ///
    /// For HW decode, reads all samples into memory (needed for VT/NVDEC).
    /// For large files, use `open()` with software decode instead.
    pub fn open_hw(path: &std::path::Path) -> Result<Self, VideoError> {
        let mut reader = Self::open(path)?;
        let vc = match reader.codec {
            Mp4Codec::H264 => crate::VideoCodec::H264,
            Mp4Codec::Hevc => crate::VideoCodec::H265,
        };
        let hw = super::hw_decode::HwVideoDecoder::new(vc)?;
        if hw.is_hardware() {
            // Build parameter set init blob (prepended to first AU only)
            let mut init_au = Vec::new();
            match reader.codec {
                Mp4Codec::H264 => {
                    for nal in &reader.param_nals_h264 {
                        init_au.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
                        init_au.extend_from_slice(&nal.data);
                    }
                }
                Mp4Codec::Hevc => {
                    for nal in &reader.param_nals_hevc {
                        init_au.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
                        init_au.extend_from_slice(nal);
                    }
                }
            }

            // Store init AU for streaming — samples are read one-at-a-time in next_frame
            reader.hw_init_au = Some(init_au);
            reader.decoder = Mp4Decoder::Hw(hw);
        }
        Ok(reader)
    }

    /// Which hardware backend is being used (Software if no HW).
    pub fn hw_backend(&self) -> super::hw_decode::HwBackend {
        match &self.decoder {
            Mp4Decoder::Hw(hw) => hw.backend(),
            _ => super::hw_decode::HwBackend::Software,
        }
    }

    /// Read one sample from file at the given sample_table index.
    fn read_sample(&mut self, idx: usize) -> Result<Vec<u8>, VideoError> {
        use std::io::{Read as _, Seek, SeekFrom};
        let (offset, size) = self.sample_table[idx];
        let file = self
            .file
            .as_mut()
            .ok_or_else(|| VideoError::Source("file handle closed".into()))?;
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| VideoError::Source(format!("seek: {e}")))?;
        let mut buf = vec![0u8; size as usize];
        file.read_exact(&mut buf)
            .map_err(|e| VideoError::Source(format!("read sample: {e}")))?;
        Ok(buf)
    }

    /// Decode the next frame. Reads one sample at a time from disk — O(1) memory.
    pub fn next_frame(&mut self) -> Result<Option<super::codec::DecodedFrame>, VideoError> {
        // HW decode path — streaming: read one sample from disk per call
        if matches!(self.decoder, Mp4Decoder::Hw(_)) {
            while self.hw_sample_idx < self.sample_table.len() {
                let idx = self.hw_sample_idx;
                self.hw_sample_idx += 1;

                // Read sample from disk (borrows self.file only)
                let sample_data = self.read_sample(idx)?;
                let nals = parse_length_prefixed_nals(&sample_data, self.nal_length_size);

                let mut au = if idx == 0 {
                    self.hw_init_au.take().unwrap_or_default()
                } else {
                    Vec::new()
                };

                for nal_data in nals {
                    au.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
                    au.extend_from_slice(&nal_data);
                }
                if !au.is_empty()
                    && let Mp4Decoder::Hw(ref mut hw) = self.decoder
                    && let Some(frame) = hw.decode(&au, 0)?
                {
                    return Ok(Some(frame));
                }
            }
            return Ok(None);
        }

        // Feed parameter sets first (once)
        if !self.params_fed {
            self.params_fed = true;
            match self.codec {
                Mp4Codec::H264 => {
                    let Mp4Decoder::H264(ref mut dec) = self.decoder else {
                        return Ok(None);
                    };
                    for nal in &self.param_nals_h264 {
                        let _ = dec.process_nal(nal);
                    }
                }
                Mp4Codec::Hevc => {
                    let Mp4Decoder::Hevc(ref mut dec) = self.decoder else {
                        return Ok(None);
                    };
                    for nal in &self.param_nals_hevc {
                        let _ = dec.decode_nal(nal);
                    }
                }
            }
        }

        // Streaming: read one sample at a time from file
        while self.current_sample < self.sample_table.len() {
            let sample_data = self.read_sample(self.current_sample)?;
            self.current_sample += 1;

            // Parse NALs from this sample and feed to decoder
            let nals = parse_length_prefixed_nals(&sample_data, self.nal_length_size);
            for nal_data in nals {
                if nal_data.is_empty() {
                    continue;
                }
                match self.codec {
                    Mp4Codec::H264 => {
                        let Mp4Decoder::H264(ref mut dec) = self.decoder else {
                            continue;
                        };
                        let header = nal_data[0];
                        let nal = super::codec::NalUnit {
                            nal_type: super::codec::NalUnitType::from_byte(header),
                            nal_ref_idc: (header >> 5) & 3,
                            data: nal_data,
                        };
                        if let Some(frame) = dec.process_nal(&nal)? {
                            return Ok(Some(frame));
                        }
                    }
                    Mp4Codec::Hevc => {
                        let Mp4Decoder::Hevc(ref mut dec) = self.decoder else {
                            continue;
                        };
                        if let Some(frame) = dec.decode_nal(&nal_data)? {
                            return Ok(Some(frame));
                        }
                    }
                }
            }
        }
        Ok(None)
    }

    /// Decode the next frame in luma-only mode (skip RGB conversion entirely).
    /// Fair comparison with ffmpeg `-f null` which also skips color conversion.
    pub fn next_frame_luma_only(
        &mut self,
    ) -> Result<Option<super::codec::DecodedFrame>, VideoError> {
        // Enable skip_rgb on HEVC decoder
        if let Mp4Decoder::Hevc(ref mut dec) = self.decoder {
            dec.skip_rgb = true;
        }
        self.next_frame()
    }

    /// Reset to the beginning (re-decode from first sample).
    pub fn seek_start(&mut self) {
        self.current_sample = 0;
        self.hw_sample_idx = 0;
        self.params_fed = false;
        match self.codec {
            Mp4Codec::H264 => {
                self.decoder = Mp4Decoder::H264(super::h264_decoder::H264Decoder::new());
            }
            Mp4Codec::Hevc => {
                self.decoder = Mp4Decoder::Hevc(super::hevc_decoder::HevcDecoder::new());
            }
        }
    }

    /// Total number of video samples (approximately = frame count).
    pub fn nal_count(&self) -> usize {
        self.sample_table.len()
    }

    /// Returns the detected video codec.
    pub fn codec(&self) -> super::codec::VideoCodec {
        match self.codec {
            Mp4Codec::H264 => super::codec::VideoCodec::H264,
            Mp4Codec::Hevc => super::codec::VideoCodec::H265,
        }
    }

    /// Audio track info, if the MP4 contains an audio track.
    pub fn audio_info(&self) -> Option<&super::audio::AudioTrackInfo> {
        self.audio.as_ref()
    }
}

// ---------------------------------------------------------------------------
// MP4 sample table parsing (stbl/stco/stsz/stsc)
// ---------------------------------------------------------------------------

/// Find a box by tag scanning raw data (recursive search within moov).
fn find_box_data<'a>(data: &'a [u8], tag: &[u8; 4]) -> Option<&'a [u8]> {
    let mut i = 0;
    while i + 8 <= data.len() {
        let sz = u32::from_be_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]) as usize;
        if sz < 8 || i + sz > data.len() {
            break;
        }
        if &data[i + 4..i + 8] == tag {
            return Some(&data[i + 8..i + sz]);
        }
        // Recurse into container boxes
        let box_tag = &data[i + 4..i + 8];
        if (box_tag == b"trak" || box_tag == b"mdia" || box_tag == b"minf" || box_tag == b"stbl")
            && let Some(found) = find_box_data(&data[i + 8..i + sz], tag)
        {
            return Some(found);
        }
        i += sz;
    }
    None
}

/// Find the video trak in moov data (contains 'vmhd' in minf).
fn find_video_trak(moov_data: &[u8]) -> Option<&[u8]> {
    let mut i = 0;
    while i + 8 <= moov_data.len() {
        let sz = u32::from_be_bytes([
            moov_data[i],
            moov_data[i + 1],
            moov_data[i + 2],
            moov_data[i + 3],
        ]) as usize;
        if sz < 8 || i + sz > moov_data.len() {
            break;
        }
        if &moov_data[i + 4..i + 8] == b"trak" {
            let trak_data = &moov_data[i + 8..i + sz];
            // Check if this trak has a vmhd (video media header) — means it's video
            if find_box_data(trak_data, b"vmhd").is_some() {
                return Some(trak_data);
            }
        }
        i += sz;
    }
    None
}

/// Find the first audio trak in a moov box (looks for smhd = Sound Media Header).
fn find_audio_trak(moov_data: &[u8]) -> Option<&[u8]> {
    let mut i = 0;
    while i + 8 <= moov_data.len() {
        let sz = u32::from_be_bytes([
            moov_data[i],
            moov_data[i + 1],
            moov_data[i + 2],
            moov_data[i + 3],
        ]) as usize;
        if sz < 8 || i + sz > moov_data.len() {
            break;
        }
        if &moov_data[i + 4..i + 8] == b"trak" {
            let trak_data = &moov_data[i + 8..i + sz];
            if find_box_data(trak_data, b"smhd").is_some() {
                return Some(trak_data);
            }
        }
        i += sz;
    }
    None
}

/// Parse audio info from an audio trak's stsd box (mp4a → esds).
fn parse_mp4_audio_info(trak_data: &[u8]) -> Option<super::audio::AudioTrackInfo> {
    let stsd = find_box_data(trak_data, b"stsd")?;
    // stsd: version(1) + flags(3) + entry_count(4) + entries...
    if stsd.len() < 8 {
        return None;
    }
    let entry_data = &stsd[8..];

    // Scan for mp4a/alac/Opus codec box
    let mut codec = super::audio::AudioCodec::Unknown;
    let mut sample_rate = 0u32;
    let mut channels = 0u16;

    for i in 0..entry_data.len().saturating_sub(32) {
        let tag = &entry_data[i..i + 4];
        if tag == b"mp4a" || tag == b"alac" || tag == b"Opus" || tag == b"fLaC" {
            codec = super::audio::audio_codec_from_mp4(tag.try_into().unwrap_or(&[0; 4]));
            // AudioSampleEntry layout AFTER the 4-byte codec tag:
            // +0:  reserved(6)
            // +6:  data_ref_index(2)
            // +8:  reserved(8)
            // +16: channel_count(2)
            // +18: sample_size(2)
            // +20: compression_id(2)
            // +22: packet_size(2)
            // +24: sample_rate(4, 16.16 fixed-point)
            let base = i + 4; // skip tag
            if base + 28 <= entry_data.len() {
                channels = u16::from_be_bytes([entry_data[base + 16], entry_data[base + 17]]);
                let sr_fixed = u32::from_be_bytes([
                    entry_data[base + 24],
                    entry_data[base + 25],
                    entry_data[base + 26],
                    entry_data[base + 27],
                ]);
                sample_rate = sr_fixed >> 16;
            }
            break;
        }
    }

    if codec == super::audio::AudioCodec::Unknown {
        return None;
    }

    Some(super::audio::AudioTrackInfo {
        codec,
        sample_rate,
        channels,
        bits_per_sample: 0,
        duration_ms: 0,
        codec_private: Vec::new(),
    })
}

/// Read the NAL length size from avcC/hvcC config (typically 4).
fn find_nal_length_size(moov_data: &[u8], codec: Mp4Codec) -> usize {
    let tag = match codec {
        Mp4Codec::H264 => b"avcC",
        Mp4Codec::Hevc => b"hvcC",
    };
    // Scan for the config tag
    for i in 0..moov_data.len().saturating_sub(8) {
        if &moov_data[i..i + 4] == tag {
            let config = &moov_data[i + 4..];
            if config.len() >= 5 {
                return match codec {
                    Mp4Codec::H264 => (config[4] & 0x03) as usize + 1,
                    Mp4Codec::Hevc => (config[21] & 0x03) as usize + 1,
                };
            }
        }
    }
    4 // default
}

/// Parse stco (chunk offset) box → list of absolute file offsets.
fn parse_stco(box_data: &[u8]) -> Vec<u64> {
    if box_data.len() < 8 {
        return Vec::new();
    }
    // version(1) + flags(3) + entry_count(4)
    let count = u32::from_be_bytes([box_data[4], box_data[5], box_data[6], box_data[7]]) as usize;
    let mut offsets = Vec::with_capacity(count);
    for j in 0..count {
        let pos = 8 + j * 4;
        if pos + 4 > box_data.len() {
            break;
        }
        offsets.push(u32::from_be_bytes([
            box_data[pos],
            box_data[pos + 1],
            box_data[pos + 2],
            box_data[pos + 3],
        ]) as u64);
    }
    offsets
}

/// Parse co64 (64-bit chunk offset) box.
fn parse_co64(box_data: &[u8]) -> Vec<u64> {
    if box_data.len() < 8 {
        return Vec::new();
    }
    let count = u32::from_be_bytes([box_data[4], box_data[5], box_data[6], box_data[7]]) as usize;
    let mut offsets = Vec::with_capacity(count);
    for j in 0..count {
        let pos = 8 + j * 8;
        if pos + 8 > box_data.len() {
            break;
        }
        offsets.push(u64::from_be_bytes([
            box_data[pos],
            box_data[pos + 1],
            box_data[pos + 2],
            box_data[pos + 3],
            box_data[pos + 4],
            box_data[pos + 5],
            box_data[pos + 6],
            box_data[pos + 7],
        ]));
    }
    offsets
}

/// Parse stsz (sample size) box → list of per-sample sizes.
fn parse_stsz(box_data: &[u8]) -> Vec<u32> {
    if box_data.len() < 12 {
        return Vec::new();
    }
    let default_size = u32::from_be_bytes([box_data[4], box_data[5], box_data[6], box_data[7]]);
    let count = u32::from_be_bytes([box_data[8], box_data[9], box_data[10], box_data[11]]) as usize;
    if default_size > 0 {
        return vec![default_size; count];
    }
    let mut sizes = Vec::with_capacity(count);
    for j in 0..count {
        let pos = 12 + j * 4;
        if pos + 4 > box_data.len() {
            break;
        }
        sizes.push(u32::from_be_bytes([
            box_data[pos],
            box_data[pos + 1],
            box_data[pos + 2],
            box_data[pos + 3],
        ]));
    }
    sizes
}

/// Parse stsc (sample-to-chunk) box → entries of (first_chunk, samples_per_chunk, sdi).
fn parse_stsc(box_data: &[u8]) -> Vec<(u32, u32, u32)> {
    if box_data.len() < 8 {
        return Vec::new();
    }
    let count = u32::from_be_bytes([box_data[4], box_data[5], box_data[6], box_data[7]]) as usize;
    let mut entries = Vec::with_capacity(count);
    for j in 0..count {
        let pos = 8 + j * 12;
        if pos + 12 > box_data.len() {
            break;
        }
        let first_chunk = u32::from_be_bytes([
            box_data[pos],
            box_data[pos + 1],
            box_data[pos + 2],
            box_data[pos + 3],
        ]);
        let spc = u32::from_be_bytes([
            box_data[pos + 4],
            box_data[pos + 5],
            box_data[pos + 6],
            box_data[pos + 7],
        ]);
        let sdi = u32::from_be_bytes([
            box_data[pos + 8],
            box_data[pos + 9],
            box_data[pos + 10],
            box_data[pos + 11],
        ]);
        entries.push((first_chunk, spc, sdi));
    }
    entries
}

/// Extract raw sample data for each video frame using the sample table.
/// Build sample table: Vec<(file_offset, size)> without reading sample data.
/// Only reads metadata from moov — O(1) memory relative to file size.
fn build_sample_table(moov_data: &[u8]) -> Result<Vec<(u64, u32)>, VideoError> {
    let video_trak = find_video_trak(moov_data)
        .ok_or_else(|| VideoError::ContainerParse("no video trak found".into()))?;

    let chunk_offsets = if let Some(co64_data) = find_box_data(video_trak, b"co64") {
        parse_co64(co64_data)
    } else if let Some(stco_data) = find_box_data(video_trak, b"stco") {
        parse_stco(stco_data)
    } else {
        return Err(VideoError::ContainerParse("no stco/co64".into()));
    };

    let sample_sizes = if let Some(stsz_data) = find_box_data(video_trak, b"stsz") {
        parse_stsz(stsz_data)
    } else {
        return Err(VideoError::ContainerParse("no stsz".into()));
    };

    let stsc_entries = if let Some(stsc_data) = find_box_data(video_trak, b"stsc") {
        parse_stsc(stsc_data)
    } else {
        vec![(1, 1, 1)]
    };

    let num_samples = sample_sizes.len();
    let num_chunks = chunk_offsets.len();
    let mut table = Vec::with_capacity(num_samples);
    let mut sample_idx = 0usize;

    for chunk_idx in 0..num_chunks {
        let chunk_num = chunk_idx as u32 + 1;
        let mut samples_in_chunk = 1u32;
        for entry in stsc_entries.iter().rev() {
            if chunk_num >= entry.0 {
                samples_in_chunk = entry.1;
                break;
            }
        }
        let mut offset = chunk_offsets[chunk_idx];
        for _ in 0..samples_in_chunk {
            if sample_idx >= num_samples {
                break;
            }
            let sz = sample_sizes[sample_idx];
            table.push((offset, sz));
            offset += sz as u64;
            sample_idx += 1;
        }
    }
    Ok(table)
}

/// Parse length-prefixed NAL units from a single sample.
fn parse_length_prefixed_nals(sample: &[u8], nal_length_size: usize) -> Vec<Vec<u8>> {
    let mut nals = Vec::new();
    let mut pos = 0;
    while pos + nal_length_size <= sample.len() {
        let nal_len = match nal_length_size {
            1 => sample[pos] as usize,
            2 => u16::from_be_bytes([sample[pos], sample[pos + 1]]) as usize,
            4 => u32::from_be_bytes([
                sample[pos],
                sample[pos + 1],
                sample[pos + 2],
                sample[pos + 3],
            ]) as usize,
            _ => break,
        };
        pos += nal_length_size;
        if nal_len == 0 || pos + nal_len > sample.len() {
            break;
        }
        nals.push(sample[pos..pos + nal_len].to_vec());
        pos += nal_len;
    }
    nals
}

/// Detect video codec from MP4 moov box by scanning for sample entry types.
/// Extract VPS/SPS/PPS NAL units from an hvcC (HEVC Decoder Configuration Record)
/// found inside the moov box. Returns raw NAL data suitable for HevcDecoder.
fn extract_hvcc_nals(moov_data: &[u8]) -> Vec<Vec<u8>> {
    let mut nals = Vec::new();

    // Scan for "hvcC" tag in moov data
    for i in 0..moov_data.len().saturating_sub(4) {
        if &moov_data[i..i + 4] == b"hvcC" {
            // hvcC box: skip 22 bytes of config header to reach arrays
            let config_start = i + 4;
            if config_start + 23 > moov_data.len() {
                break;
            }
            let num_arrays = moov_data[config_start + 22];
            let mut pos = config_start + 23;

            for _ in 0..num_arrays {
                if pos + 3 > moov_data.len() {
                    break;
                }
                let _array_completeness_and_type = moov_data[pos];
                pos += 1;
                let num_nalus = u16::from_be_bytes([moov_data[pos], moov_data[pos + 1]]) as usize;
                pos += 2;

                for _ in 0..num_nalus {
                    if pos + 2 > moov_data.len() {
                        break;
                    }
                    let nal_len = u16::from_be_bytes([moov_data[pos], moov_data[pos + 1]]) as usize;
                    pos += 2;
                    if pos + nal_len > moov_data.len() {
                        break;
                    }
                    nals.push(moov_data[pos..pos + nal_len].to_vec());
                    pos += nal_len;
                }
            }
            break;
        }
    }

    nals
}

/// Extract SPS/PPS NAL units from an avcC (AVC Decoder Configuration Record)
/// found inside the moov box. Returns H.264 NalUnit structs ready for the decoder.
///
/// avcC format (ISO 14496-15):
/// ```text
/// +0   1 byte   configurationVersion (1)
/// +1   1 byte   AVCProfileIndication
/// +2   1 byte   profile_compatibility
/// +3   1 byte   AVCLevelIndication
/// +4   1 byte   lengthSizeMinusOne (lower 2 bits)
/// +5   1 byte   numSPS (lower 5 bits)
/// +6.. SPS array: [2 bytes BE length][SPS NAL data] × numSPS
///      1 byte   numPPS
///      PPS array: [2 bytes BE length][PPS NAL data] × numPPS
/// ```
pub(crate) fn extract_avcc_nals(moov_data: &[u8]) -> Vec<super::codec::NalUnit> {
    let mut nals = Vec::new();

    // Scan for "avcC" tag in moov data
    for i in 0..moov_data.len().saturating_sub(4) {
        if &moov_data[i..i + 4] == b"avcC" {
            let config_start = i + 4;
            // Need at least 6 bytes: version(1) + profile(1) + compat(1) + level(1) + lengthSize(1) + numSPS(1)
            if config_start + 6 > moov_data.len() {
                break;
            }

            let num_sps = (moov_data[config_start + 5] & 0x1F) as usize;
            let mut pos = config_start + 6;

            // Parse SPS NALs
            for _ in 0..num_sps {
                if pos + 2 > moov_data.len() {
                    break;
                }
                let sps_len = u16::from_be_bytes([moov_data[pos], moov_data[pos + 1]]) as usize;
                pos += 2;
                if pos + sps_len > moov_data.len() || sps_len == 0 {
                    break;
                }
                let header = moov_data[pos];
                nals.push(super::codec::NalUnit {
                    nal_type: super::codec::NalUnitType::Sps,
                    nal_ref_idc: (header >> 5) & 3,
                    data: moov_data[pos..pos + sps_len].to_vec(),
                });
                pos += sps_len;
            }

            // Parse PPS NALs
            if pos >= moov_data.len() {
                break;
            }
            let num_pps = moov_data[pos] as usize;
            pos += 1;

            for _ in 0..num_pps {
                if pos + 2 > moov_data.len() {
                    break;
                }
                let pps_len = u16::from_be_bytes([moov_data[pos], moov_data[pos + 1]]) as usize;
                pos += 2;
                if pos + pps_len > moov_data.len() || pps_len == 0 {
                    break;
                }
                let header = moov_data[pos];
                nals.push(super::codec::NalUnit {
                    nal_type: super::codec::NalUnitType::Pps,
                    nal_ref_idc: (header >> 5) & 3,
                    data: moov_data[pos..pos + pps_len].to_vec(),
                });
                pos += pps_len;
            }
            break;
        }
    }

    nals
}
