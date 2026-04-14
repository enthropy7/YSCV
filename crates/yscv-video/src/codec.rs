use super::error::VideoError;
use super::frame::Rgb8Frame;

/// Supported video codec identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VideoCodec {
    H264,
    H265,
    Av1,
    Raw,
}

/// Trait for video decoders that convert compressed NAL units to RGB8 frames.
pub trait VideoDecoder: Send {
    fn codec(&self) -> VideoCodec;

    /// Decode a single compressed access unit into an RGB8 frame.
    /// Returns `None` if the decoder needs more data (e.g., initial SPS/PPS).
    fn decode(
        &mut self,
        data: &[u8],
        timestamp_us: u64,
    ) -> Result<Option<DecodedFrame>, VideoError>;

    /// Flush any remaining buffered frames.
    fn flush(&mut self) -> Result<Vec<DecodedFrame>, VideoError>;
}

/// Trait for video encoders that compress RGB8 frames.
pub trait VideoEncoder: Send {
    fn codec(&self) -> VideoCodec;

    /// Encode one RGB8 frame, returning zero or more compressed packets.
    fn encode(&mut self, frame: &Rgb8Frame) -> Result<Vec<EncodedPacket>, VideoError>;

    /// Flush remaining buffered packets.
    fn flush(&mut self) -> Result<Vec<EncodedPacket>, VideoError>;
}

/// A decoded video frame with metadata.
#[derive(Debug, Clone)]
pub struct DecodedFrame {
    pub width: usize,
    pub height: usize,
    pub rgb8_data: Vec<u8>,
    pub timestamp_us: u64,
    pub keyframe: bool,
    pub bit_depth: u8,
    pub rgb16_data: Option<Vec<u16>>,
}

impl DecodedFrame {
    pub fn into_rgb8_frame(self, frame_index: u64) -> Result<Rgb8Frame, VideoError> {
        Rgb8Frame::from_bytes(
            frame_index,
            self.timestamp_us,
            self.width,
            self.height,
            bytes::Bytes::from(self.rgb8_data),
        )
    }
}

/// A compressed video packet.
#[derive(Debug, Clone)]
pub struct EncodedPacket {
    pub data: Vec<u8>,
    pub timestamp_us: u64,
    pub keyframe: bool,
}

// ── H.264 Annex B NAL unit parser ──────────────────────────────────

/// H.264 NAL unit types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NalUnitType {
    Slice,
    SliceA,
    SliceB,
    SliceC,
    Idr,
    Sei,
    Sps,
    Pps,
    Aud,
    Other(u8),
}

impl NalUnitType {
    pub fn from_byte(b: u8) -> Self {
        match b & 0x1F {
            1 => NalUnitType::Slice,
            2 => NalUnitType::SliceA,
            3 => NalUnitType::SliceB,
            4 => NalUnitType::SliceC,
            5 => NalUnitType::Idr,
            6 => NalUnitType::Sei,
            7 => NalUnitType::Sps,
            8 => NalUnitType::Pps,
            9 => NalUnitType::Aud,
            other => NalUnitType::Other(other),
        }
    }

    pub fn is_vcl(&self) -> bool {
        matches!(
            self,
            NalUnitType::Slice
                | NalUnitType::SliceA
                | NalUnitType::SliceB
                | NalUnitType::SliceC
                | NalUnitType::Idr
        )
    }
}

/// A parsed NAL unit from an Annex B bitstream.
#[derive(Debug, Clone)]
pub struct NalUnit {
    pub nal_type: NalUnitType,
    pub nal_ref_idc: u8,
    pub data: Vec<u8>,
}

/// Parses H.264 Annex B bitstream into NAL units.
/// Splits on `0x000001` or `0x00000001` start codes.
pub fn parse_annex_b(data: &[u8]) -> Vec<NalUnit> {
    let mut units = Vec::new();
    let mut i = 0;
    let len = data.len();

    while i < len {
        if i + 3 <= len && data[i] == 0 && data[i + 1] == 0 {
            let (start_code_len, found) = if i + 4 <= len && data[i + 2] == 0 && data[i + 3] == 1 {
                (4, true)
            } else if data[i + 2] == 1 {
                (3, true)
            } else {
                (0, false)
            };

            if found {
                let nal_start = i + start_code_len;
                let mut nal_end = nal_start;
                let mut j = nal_start;
                while j < len {
                    if j + 3 <= len
                        && data[j] == 0
                        && data[j + 1] == 0
                        && ((j + 4 <= len && data[j + 2] == 0 && data[j + 3] == 1)
                            || data[j + 2] == 1)
                    {
                        nal_end = j;
                        break;
                    }
                    j += 1;
                }
                if j >= len {
                    nal_end = len;
                }

                if nal_start < nal_end {
                    let header = data[nal_start];
                    let nal_ref_idc = (header >> 5) & 0x03;
                    let nal_type = NalUnitType::from_byte(header);
                    units.push(NalUnit {
                        nal_type,
                        nal_ref_idc,
                        data: data[nal_start..nal_end].to_vec(),
                    });
                }
                i = nal_end;
                continue;
            }
        }
        i += 1;
    }

    units
}

/// Extracts SPS and PPS NAL units from an Annex B bitstream.
pub fn extract_parameter_sets(nals: &[NalUnit]) -> (Option<&NalUnit>, Option<&NalUnit>) {
    let sps = nals.iter().find(|n| n.nal_type == NalUnitType::Sps);
    let pps = nals.iter().find(|n| n.nal_type == NalUnitType::Pps);
    (sps, pps)
}

// ── Simple MP4 box parser ──────────────────────────────────────────

/// A parsed MP4 box header.
#[derive(Debug, Clone)]
pub struct Mp4Box {
    pub box_type: [u8; 4],
    pub size: u64,
    pub header_size: u8,
    pub offset: u64,
}

impl Mp4Box {
    pub fn type_str(&self) -> &str {
        std::str::from_utf8(&self.box_type).unwrap_or("????")
    }
}

/// Parses top-level MP4 boxes from a byte buffer.
pub fn parse_mp4_boxes(data: &[u8]) -> Result<Vec<Mp4Box>, VideoError> {
    let mut boxes = Vec::new();
    let mut offset = 0u64;
    let len = data.len() as u64;

    while offset + 8 <= len {
        let o = offset as usize;
        let size_32 = u32::from_be_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
        let box_type = [data[o + 4], data[o + 5], data[o + 6], data[o + 7]];

        let (size, header_size) = if size_32 == 1 {
            if offset + 16 > len {
                return Err(VideoError::ContainerParse(
                    "truncated extended box size".into(),
                ));
            }
            let extended = u64::from_be_bytes([
                data[o + 8],
                data[o + 9],
                data[o + 10],
                data[o + 11],
                data[o + 12],
                data[o + 13],
                data[o + 14],
                data[o + 15],
            ]);
            (extended, 16u8)
        } else if size_32 == 0 {
            (len - offset, 8u8)
        } else {
            (size_32 as u64, 8u8)
        };

        if size < header_size as u64 {
            return Err(VideoError::ContainerParse(format!(
                "box size {} smaller than header at offset {offset}",
                size
            )));
        }

        boxes.push(Mp4Box {
            box_type,
            size,
            header_size,
            offset,
        });

        offset += size;
    }

    Ok(boxes)
}

/// Finds a box by 4-char type code.
pub fn find_box<'a>(boxes: &'a [Mp4Box], box_type: &[u8; 4]) -> Option<&'a Mp4Box> {
    boxes.iter().find(|b| &b.box_type == box_type)
}

/// Parses child boxes inside a parent box.
pub fn parse_child_boxes(data: &[u8], parent: &Mp4Box) -> Result<Vec<Mp4Box>, VideoError> {
    let start = (parent.offset + parent.header_size as u64) as usize;
    let end = (parent.offset + parent.size) as usize;
    if end > data.len() || start >= end {
        return Ok(Vec::new());
    }
    let child_data = &data[start..end];
    let mut children = parse_mp4_boxes(child_data)?;
    for child in &mut children {
        child.offset += start as u64;
    }
    Ok(children)
}
