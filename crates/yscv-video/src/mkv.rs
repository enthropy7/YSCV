//! Minimal Matroska (MKV/WebM) demuxer.
//!
//! Parses EBML structure to extract H.264 and HEVC video tracks from MKV/WebM files.
//! Only video track extraction is supported — audio tracks are skipped.

use crate::VideoError;

// ---------------------------------------------------------------------------
// EBML element IDs (subset needed for video extraction)
// ---------------------------------------------------------------------------

const EBML_HEADER: u32 = 0x1A45_DFA3;
const SEGMENT: u32 = 0x1853_8067;
const SEGMENT_INFO: u32 = 0x1549_A966;
const TRACKS: u32 = 0x1654_AE6B;
const TRACK_ENTRY: u32 = 0xAE;
const TRACK_TYPE: u32 = 0x83;
const CODEC_ID: u32 = 0x86;
const CODEC_PRIVATE: u32 = 0x63A2;
const CLUSTER: u32 = 0x1F43_B675;
const SIMPLE_BLOCK: u32 = 0xA3;
const BLOCK_GROUP: u32 = 0xA0;
const BLOCK: u32 = 0xA1;
const TIMECODE: u32 = 0xE7;

/// Codec type detected from MKV track.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MkvCodec {
    H264,
    Hevc,
    Unknown,
}

/// A parsed video frame from an MKV file.
#[derive(Debug)]
pub struct MkvFrame {
    pub data: Vec<u8>,
    pub timestamp_ms: u64,
    pub keyframe: bool,
}

/// Minimal MKV demuxer state.
///
/// Uses streaming design: stores file path + frame index (offset, size, timestamp),
/// reads frame data lazily from disk on `next_frame()`.
pub struct MkvDemuxer {
    /// Full file data (only for files < 512MB; needed for EBML structure traversal).
    data: Vec<u8>,
    codec: MkvCodec,
    codec_private: Vec<u8>,
    /// Frame index: (data_offset, data_size, timestamp_ms, keyframe).
    frame_index: Vec<(usize, usize, u64, bool)>,
    current_frame: usize,
    /// Audio track info (if any audio track found).
    audio_info: Option<super::audio::AudioTrackInfo>,
}

impl MkvDemuxer {
    /// Open an MKV/WebM file and parse its structure.
    ///
    /// **Warning**: currently reads the entire file into memory.
    /// For files > 256MB, consider using `Mp4VideoReader` instead.
    /// TODO: streaming MKV parser.
    pub fn open(path: &std::path::Path) -> Result<Self, VideoError> {
        // Check file size to prevent OOM
        let meta = std::fs::metadata(path)
            .map_err(|e| VideoError::Codec(format!("failed to stat MKV: {e}")))?;
        if meta.len() > 512 * 1024 * 1024 {
            return Err(VideoError::Codec(format!(
                "MKV file too large for in-memory parsing ({:.0}MB > 512MB limit). \
                 Use Mp4VideoReader for large files.",
                meta.len() as f64 / 1024.0 / 1024.0,
            )));
        }

        let data = std::fs::read(path)
            .map_err(|e| VideoError::Codec(format!("failed to read MKV: {e}")))?;

        if data.len() < 4 {
            return Err(VideoError::Codec("MKV file too short".into()));
        }

        let mut demuxer = MkvDemuxer {
            data,
            audio_info: None,
            codec: MkvCodec::Unknown,
            codec_private: Vec::new(),
            frame_index: Vec::new(),
            current_frame: 0,
        };
        demuxer.parse()?;
        Ok(demuxer)
    }

    /// Detected video codec.
    pub fn codec(&self) -> MkvCodec {
        self.codec
    }

    /// Codec-specific initialization data (avcC/hvcC box equivalent).
    pub fn codec_private(&self) -> &[u8] {
        &self.codec_private
    }

    /// Total number of video frames found.
    pub fn frame_count(&self) -> usize {
        self.frame_index.len()
    }

    /// Get the next frame, reading data lazily from the in-memory buffer.
    /// Returns None when exhausted.
    pub fn next_frame(&mut self) -> Option<MkvFrame> {
        if self.current_frame < self.frame_index.len() {
            let (offset, size, ts, kf) = self.frame_index[self.current_frame];
            self.current_frame += 1;
            if offset + size <= self.data.len() {
                Some(MkvFrame {
                    data: self.data[offset..offset + size].to_vec(),
                    timestamp_ms: ts,
                    keyframe: kf,
                })
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Audio track info, if an audio track was found.
    pub fn audio_info(&self) -> Option<&super::audio::AudioTrackInfo> {
        self.audio_info.as_ref()
    }

    /// Reset to beginning.
    pub fn seek_start(&mut self) {
        self.current_frame = 0;
    }

    // -----------------------------------------------------------------------
    // EBML parsing
    // -----------------------------------------------------------------------

    fn parse(&mut self) -> Result<(), VideoError> {
        let mut pos = 0;
        let len = self.data.len();

        // Parse EBML header
        if pos + 4 > len {
            return Err(VideoError::Codec("MKV: truncated EBML header".into()));
        }

        while pos < len {
            let (id, id_len) = read_ebml_id(&self.data[pos..])?;
            pos += id_len;
            let (size, size_len) = read_ebml_size(&self.data[pos..])?;
            pos += size_len;

            match id {
                EBML_HEADER => {
                    // Skip EBML header content
                    pos += size as usize;
                }
                SEGMENT => {
                    // Parse segment children (don't skip — descend)
                    // The segment contains tracks and clusters
                }
                TRACKS => {
                    self.parse_tracks(pos, size as usize)?;
                    pos += size as usize;
                }
                CLUSTER => {
                    self.parse_cluster(pos, size as usize)?;
                    pos += size as usize;
                }
                SEGMENT_INFO => {
                    pos += size as usize;
                }
                _ => {
                    // Skip unknown elements
                    if size > 0 && size < u64::MAX {
                        pos += size as usize;
                    }
                }
            }

            if pos > len {
                break;
            }
        }

        Ok(())
    }

    fn parse_tracks(&mut self, start: usize, size: usize) -> Result<(), VideoError> {
        let end = (start + size).min(self.data.len());
        let mut pos = start;

        while pos < end {
            let (id, id_len) = read_ebml_id(&self.data[pos..])?;
            pos += id_len;
            let (el_size, size_len) = read_ebml_size(&self.data[pos..])?;
            pos += size_len;
            let el_end = pos + el_size as usize;

            if id == TRACK_ENTRY {
                self.parse_track_entry(pos, el_size as usize)?;
            }
            pos = el_end.min(end);
        }
        Ok(())
    }

    fn parse_track_entry(&mut self, start: usize, size: usize) -> Result<(), VideoError> {
        let end = (start + size).min(self.data.len());
        let mut pos = start;
        let mut track_type = 0u64;
        let mut codec_id = String::new();
        let mut codec_private = Vec::new();

        while pos < end {
            let (id, id_len) = read_ebml_id(&self.data[pos..])?;
            pos += id_len;
            let (el_size, size_len) = read_ebml_size(&self.data[pos..])?;
            pos += size_len;

            match id {
                TRACK_TYPE => {
                    track_type = read_ebml_uint(&self.data[pos..pos + el_size as usize]);
                }
                CODEC_ID => {
                    if let Ok(s) = std::str::from_utf8(&self.data[pos..pos + el_size as usize]) {
                        codec_id = s.to_string();
                    }
                }
                CODEC_PRIVATE => {
                    codec_private = self.data[pos..pos + el_size as usize].to_vec();
                }
                _ => {}
            }
            pos += el_size as usize;
        }

        // Track type 1 = video
        if track_type == 1 {
            self.codec = match codec_id.as_str() {
                "V_MPEG4/ISO/AVC" => MkvCodec::H264,
                "V_MPEGH/ISO/HEVC" => MkvCodec::Hevc,
                _ => MkvCodec::Unknown,
            };
            self.codec_private = codec_private;
        }

        // Track type 2 = audio
        if track_type == 2 && self.audio_info.is_none() {
            let audio_codec = super::audio::audio_codec_from_mkv(&codec_id);
            self.audio_info = Some(super::audio::AudioTrackInfo {
                codec: audio_codec,
                sample_rate: 0,
                channels: 0,
                bits_per_sample: 0,
                duration_ms: 0,
                codec_private: Vec::new(),
            });
        }

        Ok(())
    }

    fn parse_cluster(&mut self, start: usize, size: usize) -> Result<(), VideoError> {
        let end = (start + size).min(self.data.len());
        let mut pos = start;
        let mut cluster_timestamp = 0u64;

        while pos < end {
            let (id, id_len) = read_ebml_id(&self.data[pos..])?;
            pos += id_len;
            let (el_size, size_len) = read_ebml_size(&self.data[pos..])?;
            pos += size_len;
            let el_end = (pos + el_size as usize).min(end);

            match id {
                TIMECODE => {
                    cluster_timestamp = read_ebml_uint(&self.data[pos..el_end]);
                }
                SIMPLE_BLOCK => {
                    if pos < el_end {
                        self.parse_simple_block(pos, el_size as usize, cluster_timestamp);
                    }
                }
                BLOCK_GROUP => {
                    // Parse Block inside BlockGroup
                    self.parse_block_group(pos, el_size as usize, cluster_timestamp);
                }
                _ => {}
            }
            pos = el_end;
        }
        Ok(())
    }

    fn parse_simple_block(&mut self, start: usize, size: usize, cluster_ts: u64) {
        if size < 4 || start + size > self.data.len() {
            return;
        }
        // SimpleBlock: track_number (vint) + timecode (i16) + flags (u8) + data
        let (_, track_len) = match read_ebml_size(&self.data[start..]) {
            Ok(v) => v,
            Err(_) => return,
        };
        let header_start = start + track_len;
        if header_start + 3 > start + size {
            return;
        }
        let block_ts =
            i16::from_be_bytes([self.data[header_start], self.data[header_start + 1]]) as i64;
        let flags = self.data[header_start + 2];
        let keyframe = (flags & 0x80) != 0;
        let data_start = header_start + 3;
        let data_end = start + size;

        if data_start < data_end {
            // Store index only — data read lazily in next_frame()
            self.frame_index.push((
                data_start,
                data_end - data_start,
                (cluster_ts as i64 + block_ts) as u64,
                keyframe,
            ));
        }
    }

    fn parse_block_group(&mut self, start: usize, size: usize, cluster_ts: u64) {
        let end = (start + size).min(self.data.len());
        let mut pos = start;
        while pos < end {
            let (id, id_len) = match read_ebml_id(&self.data[pos..]) {
                Ok(v) => v,
                Err(_) => return,
            };
            pos += id_len;
            let (el_size, size_len) = match read_ebml_size(&self.data[pos..]) {
                Ok(v) => v,
                Err(_) => return,
            };
            pos += size_len;

            if id == BLOCK {
                self.parse_simple_block(pos, el_size as usize, cluster_ts);
            }
            pos += el_size as usize;
        }
    }
}

// ---------------------------------------------------------------------------
// EBML primitives
// ---------------------------------------------------------------------------

/// Read a variable-length EBML element ID. Returns (id, bytes_consumed).
fn read_ebml_id(data: &[u8]) -> Result<(u32, usize), VideoError> {
    if data.is_empty() {
        return Err(VideoError::Codec(
            "MKV: unexpected EOF reading EBML ID".into(),
        ));
    }
    let first = data[0];
    let (len, mask) = if first & 0x80 != 0 {
        (1, 0x80u32)
    } else if first & 0x40 != 0 {
        (2, 0x4000u32)
    } else if first & 0x20 != 0 {
        (3, 0x20_0000u32)
    } else if first & 0x10 != 0 {
        (4, 0x1000_0000u32)
    } else {
        return Err(VideoError::Codec(
            "MKV: invalid EBML ID leading byte".into(),
        ));
    };
    if data.len() < len {
        return Err(VideoError::Codec("MKV: truncated EBML ID".into()));
    }
    let mut id = 0u32;
    for i in 0..len {
        id = (id << 8) | data[i] as u32;
    }
    // Keep the class bits in the ID (MKV IDs include them)
    let _ = mask;
    Ok((id, len))
}

/// Read a variable-length EBML size. Returns (size, bytes_consumed).
fn read_ebml_size(data: &[u8]) -> Result<(u64, usize), VideoError> {
    if data.is_empty() {
        return Err(VideoError::Codec(
            "MKV: unexpected EOF reading EBML size".into(),
        ));
    }
    let first = data[0];
    let (len, mask) = if first & 0x80 != 0 {
        (1, 0x7Fu8)
    } else if first & 0x40 != 0 {
        (2, 0x3Fu8)
    } else if first & 0x20 != 0 {
        (3, 0x1Fu8)
    } else if first & 0x10 != 0 {
        (4, 0x0Fu8)
    } else if first & 0x08 != 0 {
        (5, 0x07u8)
    } else if first & 0x04 != 0 {
        (6, 0x03u8)
    } else if first & 0x02 != 0 {
        (7, 0x01u8)
    } else if first & 0x01 != 0 {
        (8, 0x00u8)
    } else {
        return Err(VideoError::Codec(
            "MKV: invalid EBML size leading byte".into(),
        ));
    };
    if data.len() < len {
        return Err(VideoError::Codec("MKV: truncated EBML size".into()));
    }
    let mut size = (first & mask) as u64;
    for i in 1..len {
        size = (size << 8) | data[i] as u64;
    }
    // Check for "unknown size" marker (all data bits set to 1)
    let all_ones: u64 = (1u64 << (7 * len)) - 1;
    if size == all_ones {
        // Unknown/indeterminate size — treat as "read until parent ends"
        size = 0;
    }
    Ok((size, len))
}

/// Read a big-endian unsigned integer from EBML data.
fn read_ebml_uint(data: &[u8]) -> u64 {
    let mut val = 0u64;
    for &b in data {
        val = (val << 8) | b as u64;
    }
    val
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ebml_id_parsing() {
        // 1-byte ID: 0xA3 (SimpleBlock)
        assert_eq!(read_ebml_id(&[0xA3]).unwrap(), (0xA3, 1));
        // 2-byte ID: 0x4286 (EBMLVersion)
        assert_eq!(read_ebml_id(&[0x42, 0x86]).unwrap(), (0x4286, 2));
        // 4-byte ID: 0x1A45DFA3 (EBML)
        assert_eq!(
            read_ebml_id(&[0x1A, 0x45, 0xDF, 0xA3]).unwrap(),
            (0x1A45DFA3, 4)
        );
    }

    #[test]
    fn ebml_size_parsing() {
        // 1-byte size: 0x85 = 5
        assert_eq!(read_ebml_size(&[0x85]).unwrap(), (5, 1));
        // 2-byte size: 0x40 0x05 = 5
        assert_eq!(read_ebml_size(&[0x40, 0x05]).unwrap(), (5, 2));
    }

    #[test]
    fn ebml_uint_parsing() {
        assert_eq!(read_ebml_uint(&[0x01]), 1);
        assert_eq!(read_ebml_uint(&[0x01, 0x00]), 256);
    }
}
