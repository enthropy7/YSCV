use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;

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
        let frame_size = self.meta.width as usize * self.meta.height as usize * 3;
        let start = self.frame_offset + self.current_frame as usize * frame_size;
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
/// Internal decoder — either H.264 or HEVC, stored concretely for direct access.
enum Mp4Decoder {
    H264(super::h264_decoder::H264Decoder),
    Hevc(super::hevc_decoder::HevcDecoder),
}

pub struct Mp4VideoReader {
    decoder: Mp4Decoder,
    codec: Mp4Codec,
    /// H.264 NAL units (used only for H.264 codec path)
    h264_nals: Vec<super::codec::NalUnit>,
    /// HEVC raw NAL unit data (used only for HEVC codec path)
    hevc_nals: Vec<Vec<u8>>,
    current_nal: usize,
}

impl Mp4VideoReader {
    /// Open an MP4 file containing H.264 or HEVC video.
    ///
    /// Reads the entire file, parses MP4 boxes to find the `mdat` box,
    /// auto-detects the codec from sample entry types (avc1/hvc1/hev1),
    /// and extracts NAL units from the raw data.
    pub fn open(path: &Path) -> Result<Self, VideoError> {
        let data = std::fs::read(path)
            .map_err(|e| VideoError::Source(format!("{}: {e}", path.display())))?;

        let boxes = super::codec::parse_mp4_boxes(&data)?;
        let detected_codec = detect_mp4_codec(&data, &boxes);

        // Find moov box for parameter sets and sample table
        let moov = boxes
            .iter()
            .find(|b| b.type_str() == "moov")
            .ok_or_else(|| VideoError::ContainerParse("no moov box found".into()))?;
        let moov_start = (moov.offset + moov.header_size as u64) as usize;
        let moov_end = ((moov.offset + moov.size) as usize).min(data.len());
        let moov_data = &data[moov_start..moov_end];

        // Extract video samples using the sample table (stbl/stco/stsz/stsc)
        let nal_length_size = find_nal_length_size(moov_data, detected_codec);
        let video_samples = extract_video_samples(&data, moov_data, nal_length_size)?;

        match detected_codec {
            Mp4Codec::Hevc => {
                let mut hevc_nals: Vec<Vec<u8>> = Vec::new();

                // Extract VPS/SPS/PPS from hvcC
                let param_nals = extract_hvcc_nals(moov_data);
                for nal in param_nals {
                    hevc_nals.push(nal);
                }

                // Add video sample NALs
                for sample in &video_samples {
                    let sample_nals = parse_length_prefixed_nals(sample, nal_length_size);
                    for nal_data in sample_nals {
                        hevc_nals.push(nal_data);
                    }
                }

                if hevc_nals.is_empty() {
                    return Err(VideoError::ContainerParse(
                        "no HEVC NAL units found in MP4".into(),
                    ));
                }

                Ok(Self {
                    decoder: Mp4Decoder::Hevc(super::hevc_decoder::HevcDecoder::new()),
                    codec: Mp4Codec::Hevc,
                    h264_nals: Vec::new(),
                    hevc_nals,
                    current_nal: 0,
                })
            }
            Mp4Codec::H264 => {
                let mut nal_units: Vec<super::codec::NalUnit> = Vec::new();

                // Extract SPS/PPS from avcC
                let param_nals = extract_avcc_nals(moov_data);
                nal_units.extend(param_nals);

                // Add video sample NALs
                for sample in &video_samples {
                    let sample_nals = parse_length_prefixed_nals(sample, nal_length_size);
                    for nal_data in sample_nals {
                        if !nal_data.is_empty() {
                            let header = nal_data[0];
                            nal_units.push(super::codec::NalUnit {
                                nal_type: super::codec::NalUnitType::from_byte(header),
                                nal_ref_idc: (header >> 5) & 3,
                                data: nal_data,
                            });
                        }
                    }
                }

                if nal_units.is_empty() {
                    return Err(VideoError::ContainerParse(
                        "no H.264 NAL units found in MP4".into(),
                    ));
                }

                Ok(Self {
                    decoder: Mp4Decoder::H264(super::h264_decoder::H264Decoder::new()),
                    codec: Mp4Codec::H264,
                    h264_nals: nal_units,
                    hevc_nals: Vec::new(),
                    current_nal: 0,
                })
            }
        }
    }

    /// Decode the next frame. Returns `None` when all NAL units are consumed.
    pub fn next_frame(&mut self) -> Result<Option<super::codec::DecodedFrame>, VideoError> {
        match self.codec {
            Mp4Codec::H264 => {
                let Mp4Decoder::H264(ref mut dec) = self.decoder else {
                    return Ok(None);
                };
                while self.current_nal < self.h264_nals.len() {
                    let nal = &self.h264_nals[self.current_nal];
                    self.current_nal += 1;
                    // Feed NAL directly to the H264 decoder (no Annex B re-wrapping)
                    if let Some(frame) = dec.process_nal(nal)? {
                        return Ok(Some(frame));
                    }
                }
                Ok(None)
            }
            Mp4Codec::Hevc => {
                let Mp4Decoder::Hevc(ref mut dec) = self.decoder else {
                    return Ok(None);
                };
                while self.current_nal < self.hevc_nals.len() {
                    let nal_data = &self.hevc_nals[self.current_nal];
                    self.current_nal += 1;
                    // Feed NAL directly to HEVC decoder (no start code wrapping)
                    if let Some(frame) = dec.decode_nal(nal_data)? {
                        return Ok(Some(frame));
                    }
                }
                Ok(None)
            }
        }
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

    /// Reset to the beginning (re-decode from first NAL).
    pub fn seek_start(&mut self) {
        self.current_nal = 0;
        match self.codec {
            Mp4Codec::H264 => {
                self.decoder = Mp4Decoder::H264(super::h264_decoder::H264Decoder::new());
            }
            Mp4Codec::Hevc => {
                self.decoder = Mp4Decoder::Hevc(super::hevc_decoder::HevcDecoder::new());
            }
        }
    }

    /// Total number of NAL units found.
    pub fn nal_count(&self) -> usize {
        match self.codec {
            Mp4Codec::H264 => self.h264_nals.len(),
            Mp4Codec::Hevc => self.hevc_nals.len(),
        }
    }

    /// Returns the detected video codec.
    pub fn codec(&self) -> super::codec::VideoCodec {
        match self.codec {
            Mp4Codec::H264 => super::codec::VideoCodec::H264,
            Mp4Codec::Hevc => super::codec::VideoCodec::H265,
        }
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
fn extract_video_samples(
    file_data: &[u8],
    moov_data: &[u8],
    _nal_length_size: usize,
) -> Result<Vec<Vec<u8>>, VideoError> {
    let video_trak = find_video_trak(moov_data)
        .ok_or_else(|| VideoError::ContainerParse("no video trak found".into()))?;

    // Parse sample table boxes from the video trak
    let chunk_offsets = if let Some(co64_data) = find_box_data(video_trak, b"co64") {
        parse_co64(co64_data)
    } else if let Some(stco_data) = find_box_data(video_trak, b"stco") {
        parse_stco(stco_data)
    } else {
        return Err(VideoError::ContainerParse(
            "no stco/co64 in video trak".into(),
        ));
    };

    let sample_sizes = if let Some(stsz_data) = find_box_data(video_trak, b"stsz") {
        parse_stsz(stsz_data)
    } else {
        return Err(VideoError::ContainerParse("no stsz in video trak".into()));
    };

    let stsc_entries = if let Some(stsc_data) = find_box_data(video_trak, b"stsc") {
        parse_stsc(stsc_data)
    } else {
        // Default: 1 sample per chunk
        vec![(1, 1, 1)]
    };

    // Build sample-to-file-offset mapping using stsc + stco + stsz
    let num_samples = sample_sizes.len();
    let num_chunks = chunk_offsets.len();
    let mut samples = Vec::with_capacity(num_samples);
    let mut sample_idx = 0usize;

    for chunk_idx in 0..num_chunks {
        // Find how many samples in this chunk (from stsc)
        let chunk_num = chunk_idx as u32 + 1; // 1-based
        let mut samples_in_chunk = 1u32;
        for entry in stsc_entries.iter().rev() {
            if chunk_num >= entry.0 {
                samples_in_chunk = entry.1;
                break;
            }
        }

        let mut offset = chunk_offsets[chunk_idx] as usize;
        for _ in 0..samples_in_chunk {
            if sample_idx >= num_samples {
                break;
            }
            let sz = sample_sizes[sample_idx] as usize;
            if offset + sz <= file_data.len() {
                samples.push(file_data[offset..offset + sz].to_vec());
            }
            offset += sz;
            sample_idx += 1;
        }
    }

    Ok(samples)
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
/// Looks for avc1/avc3 (H.264) or hvc1/hev1 (HEVC) in the raw moov data.
fn detect_mp4_codec(data: &[u8], boxes: &[super::codec::Mp4Box]) -> Mp4Codec {
    if let Some(moov) = boxes.iter().find(|b| b.type_str() == "moov") {
        let start = (moov.offset + moov.header_size as u64) as usize;
        let end = ((moov.offset + moov.size) as usize).min(data.len());
        if start < end {
            let moov_data = &data[start..end];
            // Scan for sample entry type codes in the moov box
            for i in 0..moov_data.len().saturating_sub(4) {
                let tag = &moov_data[i..i + 4];
                if tag == b"hvc1" || tag == b"hev1" {
                    return Mp4Codec::Hevc;
                }
            }
        }
    }
    Mp4Codec::H264 // default
}

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
