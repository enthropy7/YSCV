//! Audio track metadata and raw frame extraction.
//!
//! This module provides read-only access to audio track information in
//! MP4 and MKV containers. No audio decoding is performed — only metadata
//! (codec, sample rate, channels) and raw compressed frames are extracted,
//! suitable for passthrough to external decoders.

/// Audio codec identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioCodec {
    /// AAC (Advanced Audio Coding).
    Aac,
    /// Apple Lossless Audio Codec.
    Alac,
    /// Opus.
    Opus,
    /// Vorbis (typically in WebM/MKV).
    Vorbis,
    /// MP3 (MPEG-1 Audio Layer III).
    Mp3,
    /// FLAC (Free Lossless Audio Codec).
    Flac,
    /// Unknown or unsupported codec.
    Unknown,
}

/// Audio track metadata extracted from a container.
#[derive(Debug, Clone)]
pub struct AudioTrackInfo {
    /// Audio codec.
    pub codec: AudioCodec,
    /// Sample rate in Hz (e.g. 44100, 48000).
    pub sample_rate: u32,
    /// Number of channels (1 = mono, 2 = stereo, 6 = 5.1).
    pub channels: u16,
    /// Bits per sample (0 if not applicable, e.g. for AAC).
    pub bits_per_sample: u16,
    /// Duration in milliseconds (0 if unknown).
    pub duration_ms: u64,
    /// Codec-specific initialization data (e.g. AudioSpecificConfig for AAC).
    pub codec_private: Vec<u8>,
}

/// A raw compressed audio frame (not decoded).
#[derive(Debug)]
pub struct AudioFrame {
    /// Raw compressed audio data.
    pub data: Vec<u8>,
    /// Presentation timestamp in milliseconds.
    pub timestamp_ms: u64,
}

/// Detect audio codec from an MP4 codec box type.
pub fn audio_codec_from_mp4(box_type: &[u8; 4]) -> AudioCodec {
    match box_type {
        b"mp4a" => AudioCodec::Aac,
        b"alac" => AudioCodec::Alac,
        b"Opus" => AudioCodec::Opus,
        b"fLaC" => AudioCodec::Flac,
        b".mp3" => AudioCodec::Mp3,
        _ => AudioCodec::Unknown,
    }
}

/// Detect audio codec from an MKV CodecID string.
pub fn audio_codec_from_mkv(codec_id: &str) -> AudioCodec {
    match codec_id {
        "A_AAC" | "A_AAC/MPEG4/LC" | "A_AAC/MPEG2/LC" => AudioCodec::Aac,
        "A_ALAC" => AudioCodec::Alac,
        "A_OPUS" => AudioCodec::Opus,
        "A_VORBIS" => AudioCodec::Vorbis,
        "A_FLAC" => AudioCodec::Flac,
        "A_MPEG/L3" => AudioCodec::Mp3,
        _ => AudioCodec::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mp4_codec_detection() {
        assert_eq!(audio_codec_from_mp4(b"mp4a"), AudioCodec::Aac);
        assert_eq!(audio_codec_from_mp4(b"alac"), AudioCodec::Alac);
        assert_eq!(audio_codec_from_mp4(b"Opus"), AudioCodec::Opus);
        assert_eq!(audio_codec_from_mp4(b"xxxx"), AudioCodec::Unknown);
    }

    #[test]
    fn mkv_codec_detection() {
        assert_eq!(audio_codec_from_mkv("A_AAC"), AudioCodec::Aac);
        assert_eq!(audio_codec_from_mkv("A_OPUS"), AudioCodec::Opus);
        assert_eq!(audio_codec_from_mkv("A_VORBIS"), AudioCodec::Vorbis);
        assert_eq!(audio_codec_from_mkv("A_UNKNOWN"), AudioCodec::Unknown);
    }
}
