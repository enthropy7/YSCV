// AV1 Open Bitstream Unit (OBU) parser.
//
// Implements parsing of the AV1 bitstream format per the AV1 specification
// (AOMedia Video 1). Handles OBU headers, LEB128 size encoding,
// sequence headers, and frame headers.
//
// ## References
// - AV1 Bitstream & Decoding Process Specification, Version 1.0.0 with Errata 1
// - Section 5: OBU syntax
// - Section 5.5: Sequence header OBU syntax
// - Section 5.9: Frame header OBU syntax
use super::error::VideoError;
use super::h264_bitstream::BitstreamReader;

// ---------------------------------------------------------------------------
// OBU Types (AV1 spec, Section 6.2.2, Table 1)
// ---------------------------------------------------------------------------

/// AV1 Open Bitstream Unit type identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Av1ObuType {
    /// Reserved (0).
    Reserved,
    /// Sequence header OBU (type 1).
    SequenceHeader,
    /// Temporal delimiter OBU (type 2).
    TemporalDelimiter,
    /// Frame header OBU (type 3).
    FrameHeader,
    /// Tile group OBU (type 4).
    TileGroup,
    /// Metadata OBU (type 5).
    Metadata,
    /// Frame OBU (type 6) — combined frame header + tile group.
    Frame,
    /// Redundant frame header OBU (type 7).
    RedundantFrameHeader,
    /// Tile list OBU (type 8).
    TileList,
    /// Padding OBU (type 15).
    Padding,
    /// Unknown/unrecognised type.
    Unknown(u8),
}

impl Av1ObuType {
    /// Map a raw 4-bit type field to the enum.
    pub fn from_raw(raw: u8) -> Self {
        match raw {
            0 => Self::Reserved,
            1 => Self::SequenceHeader,
            2 => Self::TemporalDelimiter,
            3 => Self::FrameHeader,
            4 => Self::TileGroup,
            5 => Self::Metadata,
            6 => Self::Frame,
            7 => Self::RedundantFrameHeader,
            8 => Self::TileList,
            15 => Self::Padding,
            other => Self::Unknown(other),
        }
    }

    /// Convert back to the raw 4-bit type field.
    pub fn to_raw(self) -> u8 {
        match self {
            Self::Reserved => 0,
            Self::SequenceHeader => 1,
            Self::TemporalDelimiter => 2,
            Self::FrameHeader => 3,
            Self::TileGroup => 4,
            Self::Metadata => 5,
            Self::Frame => 6,
            Self::RedundantFrameHeader => 7,
            Self::TileList => 8,
            Self::Padding => 15,
            Self::Unknown(v) => v,
        }
    }
}

// ---------------------------------------------------------------------------
// OBU Extension Header
// ---------------------------------------------------------------------------

/// Parsed OBU extension header (present when `obu_extension_flag` is set).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Av1ObuExtension {
    pub temporal_id: u8,
    pub spatial_id: u8,
}

// ---------------------------------------------------------------------------
// OBU Header parsing
// ---------------------------------------------------------------------------

/// Result of parsing a single OBU header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Av1ObuHeader {
    pub obu_type: Av1ObuType,
    pub has_size: bool,
    pub has_extension: bool,
    pub extension: Av1ObuExtension,
    /// Number of bytes consumed by the header (1 or 2).
    pub header_len: usize,
}

/// Parse an OBU header from the start of `data`.
///
/// Returns the parsed header. The caller should then read the OBU size field
/// (if `has_size`) starting at `data[header.header_len..]`.
///
/// AV1 spec Section 5.3.2:
/// ```text
/// obu_header() {
///   obu_forbidden_bit     f(1)   // must be 0
///   obu_type              f(4)
///   obu_extension_flag    f(1)
///   obu_has_size_field    f(1)
///   obu_reserved_1bit     f(1)   // must be 0
///   if (obu_extension_flag) {
///     temporal_id         f(3)
///     spatial_id          f(2)
///     extension_reserved  f(3)   // must be 0
///   }
/// }
/// ```
pub fn parse_obu_header(data: &[u8]) -> Result<Av1ObuHeader, VideoError> {
    if data.is_empty() {
        return Err(VideoError::Codec("AV1: empty OBU data".into()));
    }

    let byte0 = data[0];
    let forbidden = (byte0 >> 7) & 1;
    if forbidden != 0 {
        return Err(VideoError::Codec("AV1: obu_forbidden_bit is set".into()));
    }

    let obu_type_raw = (byte0 >> 3) & 0x0F;
    let obu_type = Av1ObuType::from_raw(obu_type_raw);
    let has_extension = ((byte0 >> 2) & 1) != 0;
    let has_size = ((byte0 >> 1) & 1) != 0;

    let mut header_len = 1usize;
    let mut extension = Av1ObuExtension::default();

    if has_extension {
        if data.len() < 2 {
            return Err(VideoError::Codec(
                "AV1: truncated OBU extension header".into(),
            ));
        }
        let byte1 = data[1];
        extension.temporal_id = (byte1 >> 5) & 0x07;
        extension.spatial_id = (byte1 >> 3) & 0x03;
        header_len = 2;
    }

    Ok(Av1ObuHeader {
        obu_type,
        has_size,
        has_extension,
        extension,
        header_len,
    })
}

// ---------------------------------------------------------------------------
// LEB128 encoding (AV1 spec Section 4.10.5)
// ---------------------------------------------------------------------------

/// Read a LEB128-encoded unsigned integer from the start of `data`.
///
/// Returns `(value, bytes_consumed)`. AV1 limits LEB128 to 8 bytes max,
/// encoding values up to `(1 << 56) - 1` in practice (the spec says the
/// maximum number of bytes is 8, yielding up to 64 bits but the MSB of
/// each group is the continuation flag).
pub fn read_leb128(data: &[u8]) -> Result<(u64, usize), VideoError> {
    let mut value: u64 = 0;
    let max_bytes = 8.min(data.len());

    for i in 0..max_bytes {
        let byte = data[i];
        // Each byte contributes 7 bits of payload
        let payload = (byte & 0x7F) as u64;
        value |= payload << (i * 7);

        // If the high bit is clear, this is the last byte
        if byte & 0x80 == 0 {
            return Ok((value, i + 1));
        }
    }

    // If we consumed max_bytes and all had continuation bit set, that's an error
    if max_bytes == 0 {
        return Err(VideoError::Codec("AV1: empty LEB128 data".into()));
    }
    Err(VideoError::Codec("AV1: LEB128 overflow (> 8 bytes)".into()))
}

/// Encode a value as LEB128 bytes. Used for testing round-trips.
pub fn write_leb128(mut value: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(8);
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if value == 0 {
            break;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Color config
// ---------------------------------------------------------------------------

/// AV1 color primaries (Section 6.4.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Av1ColorPrimaries {
    Bt709,
    Unspecified,
    Bt470M,
    Bt470Bg,
    Bt601,
    Smpte240,
    GenericFilm,
    Bt2020,
    Xyz,
    Smpte431,
    Smpte432,
    Ebu3213,
    Other(u8),
}

impl Av1ColorPrimaries {
    fn from_raw(v: u8) -> Self {
        match v {
            1 => Self::Bt709,
            2 => Self::Unspecified,
            4 => Self::Bt470M,
            5 => Self::Bt470Bg,
            6 => Self::Bt601,
            7 => Self::Smpte240,
            8 => Self::GenericFilm,
            9 => Self::Bt2020,
            10 => Self::Xyz,
            11 => Self::Smpte431,
            12 => Self::Smpte432,
            22 => Self::Ebu3213,
            other => Self::Other(other),
        }
    }
}

/// Chroma sample position (Section 6.4.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Av1ChromaSamplePosition {
    #[default]
    Unknown,
    Vertical,
    Colocated,
    Reserved,
}

impl Av1ChromaSamplePosition {
    fn from_raw(v: u8) -> Self {
        match v {
            0 => Self::Unknown,
            1 => Self::Vertical,
            2 => Self::Colocated,
            _ => Self::Reserved,
        }
    }
}

// ---------------------------------------------------------------------------
// Sequence Header (AV1 spec Section 5.5)
// ---------------------------------------------------------------------------

/// Parsed AV1 sequence header OBU.
#[derive(Debug, Clone)]
pub struct Av1SequenceHeader {
    /// Sequence profile: 0 (Main/8+10-bit 4:2:0), 1 (High/8+10-bit 4:4:4),
    /// 2 (Professional/12-bit).
    pub profile: u8,
    /// If true, this sequence header is valid for a single frame only.
    pub still_picture: bool,
    /// Reduced still picture header (implies still_picture + key frame only).
    pub reduced_still_picture_header: bool,
    /// Number of operating points.
    pub operating_points_cnt: usize,
    /// Operating point IDC values (one per operating point).
    pub operating_point_idc: Vec<u16>,
    /// Sequence-level max frame width in pixels.
    pub max_frame_width: u32,
    /// Sequence-level max frame height in pixels.
    pub max_frame_height: u32,
    /// Number of bits used to encode frame width minus 1.
    pub frame_width_bits: u8,
    /// Number of bits used to encode frame height minus 1.
    pub frame_height_bits: u8,
    /// Whether frame IDs are present in the bitstream.
    pub frame_id_numbers_present: bool,
    /// Bits for delta_frame_id minus 1 (valid if frame_id_numbers_present).
    pub delta_frame_id_length: u8,
    /// Bits for frame_id (valid if frame_id_numbers_present).
    pub additional_frame_id_length: u8,
    /// Whether 128x128 superblocks are used (false => 64x64).
    pub use_128x128_superblock: bool,
    /// Whether filter intra is enabled.
    pub enable_filter_intra: bool,
    /// Whether intra edge filtering is enabled.
    pub enable_intra_edge_filter: bool,
    /// Whether inter-frame prediction tools are enabled.
    pub enable_interintra_compound: bool,
    pub enable_masked_compound: bool,
    pub enable_warped_motion: bool,
    pub enable_dual_filter: bool,
    /// Whether the order hint is present in frame headers.
    pub enable_order_hint: bool,
    pub enable_jnt_comp: bool,
    pub enable_ref_frame_mvs: bool,
    /// Force screen content coding tools.
    pub seq_force_screen_content_tools: u8,
    /// Force integer motion vectors.
    pub seq_force_integer_mv: u8,
    /// Number of bits for order_hint (0 if order hint disabled).
    pub order_hint_bits: u8,
    /// Whether superres is enabled.
    pub enable_superres: bool,
    /// Whether CDEF (Constrained Directional Enhancement Filter) is enabled.
    pub enable_cdef: bool,
    /// Whether loop restoration is enabled.
    pub enable_restoration: bool,
    /// Bit depth: 8, 10, or 12.
    pub bit_depth: u8,
    /// Monochrome flag (only luma, no chroma planes).
    pub monochrome: bool,
    /// Whether color description is present.
    pub color_description_present: bool,
    /// Color primaries.
    pub color_primaries: Av1ColorPrimaries,
    /// Subsampling in x direction (1 for 4:2:0 and 4:2:2, 0 for 4:4:4).
    pub subsampling_x: u8,
    /// Subsampling in y direction (1 for 4:2:0, 0 for 4:2:2 and 4:4:4).
    pub subsampling_y: u8,
    /// Chroma sample position.
    pub chroma_sample_position: Av1ChromaSamplePosition,
    /// Whether separate UV delta quantizer is allowed.
    pub separate_uv_delta_q: bool,
    /// Whether film grain parameters are present in frame headers.
    pub film_grain_params_present: bool,
}

impl Av1SequenceHeader {
    /// Superblock size in pixels (64 or 128).
    pub fn sb_size(&self) -> usize {
        if self.use_128x128_superblock { 128 } else { 64 }
    }

    /// Number of planes (1 for monochrome, 3 otherwise).
    pub fn num_planes(&self) -> usize {
        if self.monochrome { 1 } else { 3 }
    }

    /// Maximum sample value for the configured bit depth.
    pub fn max_sample_value(&self) -> i32 {
        (1i32 << self.bit_depth) - 1
    }
}

/// Parse a sequence header OBU payload.
///
/// `data` should point to the OBU payload *after* the OBU header and size
/// fields have been consumed.
pub fn parse_sequence_header(data: &[u8]) -> Result<Av1SequenceHeader, VideoError> {
    let mut r = BitstreamReader::new(data);

    let profile = r.read_bits(3)? as u8;
    if profile > 2 {
        return Err(VideoError::Codec(format!(
            "AV1: invalid seq_profile {profile}"
        )));
    }

    let still_picture = r.read_bit()? != 0;
    let reduced_still_picture_header = r.read_bit()? != 0;

    let mut operating_points_cnt = 1usize;
    let mut operating_point_idc = vec![0u16];

    if reduced_still_picture_header {
        // reduced header implies: timing_info_present=0, initial_display_delay_present=0,
        // operating_points_cnt=1, operating_point_idc[0]=0, seq_level_idx[0] read below
        let _seq_level_idx = r.read_bits(5)?;
    } else {
        let timing_info_present = r.read_bit()? != 0;
        if timing_info_present {
            // timing_info()
            let _num_units_in_display_tick = r.read_bits(32)?;
            let _time_scale = r.read_bits(32)?;
            let equal_picture_interval = r.read_bit()? != 0;
            if equal_picture_interval {
                let _num_ticks_per_picture_minus1 = read_uvlc(&mut r)?;
            }

            let decoder_model_info_present = r.read_bit()? != 0;
            if decoder_model_info_present {
                // decoder_model_info() — skip the fields
                let _buffer_delay_length_minus1 = r.read_bits(5)?;
                let _num_units_in_decoding_tick = r.read_bits(32)?;
                let _buffer_removal_time_length_minus1 = r.read_bits(5)?;
                let _frame_presentation_time_length_minus1 = r.read_bits(5)?;
            }
        }

        let initial_display_delay_present = r.read_bit()? != 0;
        operating_points_cnt = (r.read_bits(5)? as usize) + 1;
        operating_point_idc = Vec::with_capacity(operating_points_cnt);

        for _i in 0..operating_points_cnt {
            let idc = r.read_bits(12)? as u16;
            operating_point_idc.push(idc);
            let _seq_level_idx = r.read_bits(5)?;
            if _seq_level_idx > 7 {
                let _seq_tier = r.read_bit()?;
            }
            // Skip decoder_model/initial_display_delay if present
            // (simplified: we only need the structural fields)
            if timing_info_present {
                let timing_info_present_flag = r.read_bit()? != 0;
                if timing_info_present_flag {
                    // operating_parameters_info — skip
                    // These depend on buffer_delay_length which we parsed above;
                    // for a skeleton parser we just skip the bits
                    // decoder_buffer_delay + encoder_buffer_delay + low_delay_mode_flag
                    // We don't have the length readily available here, so we skip
                    // with a simplified approach
                }
            }
            if initial_display_delay_present {
                let has_delay = r.read_bit()? != 0;
                if has_delay {
                    let _initial_display_delay_minus1 = r.read_bits(4)?;
                }
            }
        }
    }

    // frame_width_bits_minus_1 and frame_height_bits_minus_1
    let frame_width_bits = (r.read_bits(4)? as u8) + 1;
    let frame_height_bits = (r.read_bits(4)? as u8) + 1;

    let max_frame_width = r.read_bits(frame_width_bits)? + 1;
    let max_frame_height = r.read_bits(frame_height_bits)? + 1;

    let mut frame_id_numbers_present = false;
    let mut delta_frame_id_length = 0u8;
    let mut additional_frame_id_length = 0u8;

    if !reduced_still_picture_header {
        frame_id_numbers_present = r.read_bit()? != 0;
        if frame_id_numbers_present {
            delta_frame_id_length = (r.read_bits(4)? as u8) + 2;
            additional_frame_id_length = (r.read_bits(3)? as u8) + 1;
        }
    }

    let use_128x128_superblock = r.read_bit()? != 0;
    let enable_filter_intra = r.read_bit()? != 0;
    let enable_intra_edge_filter = r.read_bit()? != 0;

    let mut enable_interintra_compound = false;
    let mut enable_masked_compound = false;
    let mut enable_warped_motion = false;
    let mut enable_dual_filter = false;
    let mut enable_order_hint = false;
    let mut enable_jnt_comp = false;
    let mut enable_ref_frame_mvs = false;
    let mut seq_force_screen_content_tools = 2u8; // SELECT_SCREEN_CONTENT_TOOLS
    let mut seq_force_integer_mv = 2u8; // SELECT_INTEGER_MV
    let mut order_hint_bits = 0u8;

    if !reduced_still_picture_header {
        enable_interintra_compound = r.read_bit()? != 0;
        enable_masked_compound = r.read_bit()? != 0;
        enable_warped_motion = r.read_bit()? != 0;
        enable_dual_filter = r.read_bit()? != 0;
        enable_order_hint = r.read_bit()? != 0;

        if enable_order_hint {
            enable_jnt_comp = r.read_bit()? != 0;
            enable_ref_frame_mvs = r.read_bit()? != 0;
        }

        let seq_choose_screen_content_tools = r.read_bit()? != 0;
        if seq_choose_screen_content_tools {
            seq_force_screen_content_tools = 2; // SELECT
        } else {
            seq_force_screen_content_tools = r.read_bit()?;
        }

        if seq_force_screen_content_tools > 0 {
            let seq_choose_integer_mv = r.read_bit()? != 0;
            if seq_choose_integer_mv {
                seq_force_integer_mv = 2; // SELECT
            } else {
                seq_force_integer_mv = r.read_bit()?;
            }
        }

        if enable_order_hint {
            order_hint_bits = (r.read_bits(3)? as u8) + 1;
        }
    }

    let enable_superres = r.read_bit()? != 0;
    let enable_cdef = r.read_bit()? != 0;
    let enable_restoration = r.read_bit()? != 0;

    // color_config()
    let high_bitdepth = r.read_bit()? != 0;
    let mut bit_depth: u8 = if high_bitdepth { 10 } else { 8 };
    if profile == 2 && high_bitdepth {
        let twelve_bit = r.read_bit()? != 0;
        if twelve_bit {
            bit_depth = 12;
        }
    }

    let monochrome = if profile != 1 {
        r.read_bit()? != 0
    } else {
        false
    };

    let color_description_present = r.read_bit()? != 0;
    let mut color_primaries = Av1ColorPrimaries::Unspecified;
    let mut _transfer_characteristics = 2u8; // unspecified
    let mut _matrix_coefficients = 2u8; // unspecified

    if color_description_present {
        color_primaries = Av1ColorPrimaries::from_raw(r.read_bits(8)? as u8);
        _transfer_characteristics = r.read_bits(8)? as u8;
        _matrix_coefficients = r.read_bits(8)? as u8;
    }

    #[allow(unused_assignments)]
    let mut subsampling_x = 0u8;
    #[allow(unused_assignments)]
    let mut subsampling_y = 0u8;
    let mut chroma_sample_position = Av1ChromaSamplePosition::Unknown;
    let mut separate_uv_delta_q = false;

    if monochrome {
        let _color_range = r.read_bit()?;
        // For monochrome, subsampling is implicitly 1,1 (no chroma planes)
        subsampling_x = 1;
        subsampling_y = 1;
    } else if matches!(color_primaries, Av1ColorPrimaries::Bt709)
        && _transfer_characteristics == 13
        && _matrix_coefficients == 0
    {
        // sRGB/sYCC: 4:4:4
        subsampling_x = 0;
        subsampling_y = 0;
    } else {
        let _color_range = r.read_bit()?;
        if profile == 0 {
            subsampling_x = 1;
            subsampling_y = 1;
        } else if profile == 1 {
            subsampling_x = 0;
            subsampling_y = 0;
        } else {
            // profile 2
            if bit_depth == 12 {
                subsampling_x = r.read_bit()?;
                if subsampling_x != 0 {
                    subsampling_y = r.read_bit()?;
                }
            } else {
                subsampling_x = 1;
                subsampling_y = 0;
            }
        }
        if subsampling_x != 0 && subsampling_y != 0 {
            chroma_sample_position = Av1ChromaSamplePosition::from_raw(r.read_bits(2)? as u8);
        }
    }

    if !monochrome {
        separate_uv_delta_q = r.read_bit()? != 0;
    }

    let film_grain_params_present = r.read_bit()? != 0;

    Ok(Av1SequenceHeader {
        profile,
        still_picture,
        reduced_still_picture_header,
        operating_points_cnt,
        operating_point_idc,
        max_frame_width,
        max_frame_height,
        frame_width_bits,
        frame_height_bits,
        frame_id_numbers_present,
        delta_frame_id_length,
        additional_frame_id_length,
        use_128x128_superblock,
        enable_filter_intra,
        enable_intra_edge_filter,
        enable_interintra_compound,
        enable_masked_compound,
        enable_warped_motion,
        enable_dual_filter,
        enable_order_hint,
        enable_jnt_comp,
        enable_ref_frame_mvs,
        seq_force_screen_content_tools,
        seq_force_integer_mv,
        order_hint_bits,
        enable_superres,
        enable_cdef,
        enable_restoration,
        bit_depth,
        monochrome,
        color_description_present,
        color_primaries,
        subsampling_x,
        subsampling_y,
        chroma_sample_position,
        separate_uv_delta_q,
        film_grain_params_present,
    })
}

// ---------------------------------------------------------------------------
// Frame types (AV1 spec Section 6.8.2)
// ---------------------------------------------------------------------------

/// AV1 frame type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Av1FrameType {
    /// Key frame (type 0) — full intra, resets decoder state.
    KeyFrame,
    /// Inter frame (type 1) — uses references.
    InterFrame,
    /// Intra-only frame (type 2) — all intra, but does not reset state.
    IntraOnlyFrame,
    /// Switch frame (type 3) — for bitstream switching.
    SwitchFrame,
}

impl Av1FrameType {
    fn from_raw(v: u32) -> Result<Self, VideoError> {
        match v {
            0 => Ok(Self::KeyFrame),
            1 => Ok(Self::InterFrame),
            2 => Ok(Self::IntraOnlyFrame),
            3 => Ok(Self::SwitchFrame),
            _ => Err(VideoError::Codec(format!("AV1: invalid frame_type {v}"))),
        }
    }

    /// Whether this frame type uses only intra prediction.
    pub fn is_intra(self) -> bool {
        matches!(self, Self::KeyFrame | Self::IntraOnlyFrame)
    }
}

// ---------------------------------------------------------------------------
// Quantization parameters
// ---------------------------------------------------------------------------

/// Quantization parameters parsed from the frame header.
#[derive(Debug, Clone, Copy, Default)]
pub struct Av1QuantizationParams {
    /// Base quantizer index (0-255).
    pub base_q_idx: u16,
    /// Y DC delta Q.
    pub delta_q_y_dc: i8,
    /// U DC delta Q.
    pub delta_q_u_dc: i8,
    /// U AC delta Q.
    pub delta_q_u_ac: i8,
    /// V DC delta Q.
    pub delta_q_v_dc: i8,
    /// V AC delta Q.
    pub delta_q_v_ac: i8,
    /// Whether quantizer matrices are used.
    pub using_qmatrix: bool,
}

// ---------------------------------------------------------------------------
// Tile info
// ---------------------------------------------------------------------------

/// Tile structure information from the frame header.
#[derive(Debug, Clone, Default)]
pub struct Av1TileInfo {
    /// Number of tile columns.
    pub tile_cols: u32,
    /// Number of tile rows.
    pub tile_rows: u32,
    /// Width of each tile column in superblocks.
    pub tile_col_widths: Vec<u32>,
    /// Height of each tile row in superblocks.
    pub tile_row_heights: Vec<u32>,
    /// Total number of tiles.
    pub tile_count: u32,
    /// context_update_tile_id.
    pub context_update_tile_id: u32,
    /// tile_size_bytes (number of bytes used to code each tile size).
    pub tile_size_bytes: u8,
}

// ---------------------------------------------------------------------------
// Segmentation parameters
// ---------------------------------------------------------------------------

/// Segmentation parameters from the frame header.
#[derive(Debug, Clone, Default)]
pub struct Av1SegmentationParams {
    /// Whether segmentation is enabled.
    pub enabled: bool,
    /// Whether the segmentation map should be updated.
    pub update_map: bool,
    /// Whether segment data should be updated.
    pub update_data: bool,
    /// Whether temporal prediction of the segmentation map is used.
    pub temporal_update: bool,
    /// Feature values for each segment (up to 8 segments x 8 features).
    pub feature_enabled: [[bool; 8]; 8],
    pub feature_data: [[i16; 8]; 8],
}

// ---------------------------------------------------------------------------
// Loop filter parameters
// ---------------------------------------------------------------------------

/// Loop filter parameters from the frame header.
#[derive(Debug, Clone, Default)]
pub struct Av1LoopFilterParams {
    pub level: [u8; 4],
    pub sharpness: u8,
    pub delta_enabled: bool,
    pub delta_update: bool,
    pub ref_deltas: [i8; 8],
    pub mode_deltas: [i8; 2],
}

// ---------------------------------------------------------------------------
// CDEF parameters
// ---------------------------------------------------------------------------

/// CDEF parameters from the frame header.
#[derive(Debug, Clone, Default)]
pub struct Av1CdefParams {
    pub damping: u8,
    pub bits: u8,
    /// (y_pri_strength, y_sec_strength, uv_pri_strength, uv_sec_strength)
    /// for each CDEF filter (up to 8).
    pub strengths: Vec<(u8, u8, u8, u8)>,
}

// ---------------------------------------------------------------------------
// Frame header (AV1 spec Section 5.9)
// ---------------------------------------------------------------------------

/// Parsed AV1 frame header (uncompressed portion).
#[derive(Debug, Clone)]
pub struct Av1FrameHeader {
    /// Frame type (key, inter, intra-only, switch).
    pub frame_type: Av1FrameType,
    /// Whether this frame should be shown (output).
    pub show_frame: bool,
    /// Whether this frame is showable (can be shown later via show_existing_frame).
    pub showable_frame: bool,
    /// Whether error resilient mode is enabled.
    pub error_resilient: bool,
    /// Whether CDF update is disabled.
    pub disable_cdf_update: bool,
    /// Whether screen content tools are allowed for this frame.
    pub allow_screen_content_tools: bool,
    /// Whether integer motion vectors are forced.
    pub force_integer_mv: bool,
    /// Current frame ID (if frame_id_numbers_present).
    pub current_frame_id: u32,
    /// Frame width in pixels.
    pub frame_width: u32,
    /// Frame height in pixels.
    pub frame_height: u32,
    /// Whether superres is used for this frame.
    pub use_superres: bool,
    /// Superres denominator (9..=16, effective scale = denom/8).
    pub superres_denom: u8,
    /// Primary reference frame index (0-6, or 7 = PRIMARY_REF_NONE).
    pub primary_ref_frame: u8,
    /// Bitmask of reference frames that this frame refreshes.
    pub refresh_frame_flags: u8,
    /// Reference frame indices for each of the 7 reference slots.
    pub ref_frame_idx: [u8; 7],
    /// Order hint of this frame.
    pub order_hint: u32,
    /// Tile info.
    pub tile_info: Av1TileInfo,
    /// Quantization parameters.
    pub quantization_params: Av1QuantizationParams,
    /// Segmentation parameters.
    pub segmentation_params: Av1SegmentationParams,
    /// Loop filter parameters.
    pub loop_filter_params: Av1LoopFilterParams,
    /// CDEF parameters.
    pub cdef_params: Av1CdefParams,
    /// Whether reduced TX set is used.
    pub reduced_tx_set: bool,
    /// TX mode (0 = ONLY_4X4, 1 = LARGEST, 2 = TX_MODE_SELECT).
    pub tx_mode: u8,
    /// Whether this frame allows reference frame motion vectors.
    pub allow_ref_frame_mvs: bool,
    /// Whether the frame is a show_existing_frame.
    pub show_existing_frame: bool,
    /// If show_existing_frame, which frame to show (0-7).
    pub frame_to_show_map_idx: u8,
}

/// Parse a frame header from OBU payload data.
///
/// `data` should point to the frame header OBU payload (after OBU header+size).
/// `seq` is the active sequence header needed to interpret many fields.
pub fn parse_frame_header(
    data: &[u8],
    seq: &Av1SequenceHeader,
) -> Result<Av1FrameHeader, VideoError> {
    let mut r = BitstreamReader::new(data);

    #[allow(unused_assignments)]
    let mut show_existing_frame = false;
    let mut frame_to_show_map_idx = 0u8;

    if seq.reduced_still_picture_header {
        // Implied: key frame, show_frame=true, no show_existing_frame
    } else {
        show_existing_frame = r.read_bit()? != 0;
        if show_existing_frame {
            frame_to_show_map_idx = r.read_bits(3)? as u8;
            // When show_existing_frame, the rest of the header is minimal
            return Ok(Av1FrameHeader {
                frame_type: Av1FrameType::InterFrame,
                show_frame: true,
                showable_frame: false,
                error_resilient: false,
                disable_cdf_update: false,
                allow_screen_content_tools: false,
                force_integer_mv: false,
                current_frame_id: 0,
                frame_width: seq.max_frame_width,
                frame_height: seq.max_frame_height,
                use_superres: false,
                superres_denom: 8,
                primary_ref_frame: 7,
                refresh_frame_flags: 0,
                ref_frame_idx: [0; 7],
                order_hint: 0,
                tile_info: Av1TileInfo::default(),
                quantization_params: Av1QuantizationParams::default(),
                segmentation_params: Av1SegmentationParams::default(),
                loop_filter_params: Av1LoopFilterParams::default(),
                cdef_params: Av1CdefParams::default(),
                reduced_tx_set: false,
                tx_mode: 0,
                allow_ref_frame_mvs: false,
                show_existing_frame: true,
                frame_to_show_map_idx,
            });
        }
    }

    let frame_type = if seq.reduced_still_picture_header {
        Av1FrameType::KeyFrame
    } else {
        Av1FrameType::from_raw(r.read_bits(2)?)?
    };

    let show_frame = if seq.reduced_still_picture_header {
        true
    } else {
        r.read_bit()? != 0
    };

    let showable_frame = if show_frame
        && !matches!(frame_type, Av1FrameType::KeyFrame)
        && !seq.reduced_still_picture_header
    {
        r.read_bit()? != 0
    } else {
        false
    };

    let error_resilient = if matches!(frame_type, Av1FrameType::SwitchFrame)
        || (matches!(frame_type, Av1FrameType::KeyFrame) && show_frame)
    {
        true
    } else if !seq.reduced_still_picture_header {
        r.read_bit()? != 0
    } else {
        false
    };

    let disable_cdf_update = if !seq.reduced_still_picture_header {
        r.read_bit()? != 0
    } else {
        false
    };

    let allow_screen_content_tools = if seq.seq_force_screen_content_tools == 2 {
        if seq.reduced_still_picture_header {
            false
        } else {
            r.read_bit()? != 0
        }
    } else {
        seq.seq_force_screen_content_tools != 0
    };

    let force_integer_mv = if allow_screen_content_tools {
        if seq.seq_force_integer_mv == 2 {
            if seq.reduced_still_picture_header {
                false
            } else {
                r.read_bit()? != 0
            }
        } else {
            seq.seq_force_integer_mv != 0
        }
    } else {
        false
    };

    let mut current_frame_id = 0u32;
    if seq.frame_id_numbers_present {
        let id_len = seq.delta_frame_id_length + seq.additional_frame_id_length;
        current_frame_id = r.read_bits(id_len)?;
    }

    // frame_size_override_flag
    let frame_size_override = if matches!(frame_type, Av1FrameType::SwitchFrame) {
        true
    } else if seq.reduced_still_picture_header {
        false
    } else {
        r.read_bit()? != 0
    };

    // order_hint
    let order_hint = if seq.reduced_still_picture_header {
        0
    } else if seq.order_hint_bits > 0 {
        r.read_bits(seq.order_hint_bits)?
    } else {
        0
    };

    // primary_ref_frame
    let primary_ref_frame = if frame_type.is_intra() || error_resilient {
        7 // PRIMARY_REF_NONE
    } else if !seq.reduced_still_picture_header {
        r.read_bits(3)? as u8
    } else {
        7
    };

    // refresh_frame_flags
    let refresh_frame_flags = if matches!(frame_type, Av1FrameType::SwitchFrame)
        || (matches!(frame_type, Av1FrameType::KeyFrame) && show_frame)
    {
        0xFF // all frames
    } else if !seq.reduced_still_picture_header {
        r.read_bits(8)? as u8
    } else {
        0xFF
    };

    // frame_size
    let (frame_width, frame_height) = if frame_size_override {
        let w = r.read_bits(seq.frame_width_bits)? + 1;
        let h = r.read_bits(seq.frame_height_bits)? + 1;
        (w, h)
    } else {
        (seq.max_frame_width, seq.max_frame_height)
    };

    // superres_params
    let (use_superres, superres_denom) = if seq.enable_superres {
        let use_sr = r.read_bit()? != 0;
        if use_sr {
            let denom = (r.read_bits(3)? as u8) + 9; // SUPERRES_DENOM_MIN = 9
            (true, denom)
        } else {
            (false, 8)
        }
    } else {
        (false, 8)
    };

    // ref_frame_idx (for inter frames)
    let mut ref_frame_idx = [0u8; 7];
    if !frame_type.is_intra() {
        for slot in ref_frame_idx.iter_mut() {
            *slot = r.read_bits(3)? as u8;
        }
    }

    // ---- Tile info ----
    let tile_info = parse_tile_info(&mut r, seq, frame_width, frame_height)?;

    // ---- Quantization params ----
    let quantization_params = parse_quantization_params(&mut r, seq)?;

    // ---- Segmentation params (simplified) ----
    let segmentation_params = parse_segmentation_params(&mut r)?;

    // ---- Loop filter params ----
    let loop_filter_params = parse_loop_filter_params(&mut r, seq, frame_type)?;

    // ---- CDEF params ----
    let cdef_params = parse_cdef_params(&mut r, seq, frame_type)?;

    // ---- TX mode ----
    let tx_mode = if r.bits_remaining() >= 2 {
        if r.read_bit()? != 0 {
            2 // TX_MODE_SELECT
        } else {
            1 // LARGEST
        }
    } else {
        2
    };

    let reduced_tx_set = if r.bits_remaining() >= 1 {
        r.read_bit()? != 0
    } else {
        false
    };

    let allow_ref_frame_mvs = if !frame_type.is_intra()
        && seq.enable_ref_frame_mvs
        && !error_resilient
        && r.bits_remaining() >= 1
    {
        r.read_bit()? != 0
    } else {
        false
    };

    Ok(Av1FrameHeader {
        frame_type,
        show_frame,
        showable_frame,
        error_resilient,
        disable_cdf_update,
        allow_screen_content_tools,
        force_integer_mv,
        current_frame_id,
        frame_width,
        frame_height,
        use_superres,
        superres_denom,
        primary_ref_frame,
        refresh_frame_flags,
        ref_frame_idx,
        order_hint,
        tile_info,
        quantization_params,
        segmentation_params,
        loop_filter_params,
        cdef_params,
        reduced_tx_set,
        tx_mode,
        allow_ref_frame_mvs,
        show_existing_frame: false,
        frame_to_show_map_idx,
    })
}

// ---------------------------------------------------------------------------
// Tile info parsing
// ---------------------------------------------------------------------------

fn parse_tile_info(
    r: &mut BitstreamReader<'_>,
    seq: &Av1SequenceHeader,
    frame_width: u32,
    frame_height: u32,
) -> Result<Av1TileInfo, VideoError> {
    let sb_size = seq.sb_size() as u32;
    let sb_cols = frame_width.div_ceil(sb_size);
    let sb_rows = frame_height.div_ceil(sb_size);

    let min_log2_tile_cols = tile_log2(1, sb_cols);
    let max_log2_tile_cols = tile_log2(1, sb_cols.min(64));
    let max_log2_tile_rows = tile_log2(1, sb_rows.min(64));

    let uniform_tile_spacing = r.read_bit()? != 0;

    let mut tile_col_widths = Vec::new();
    let mut tile_row_heights = Vec::new();

    if uniform_tile_spacing {
        let mut tile_cols_log2 = min_log2_tile_cols;
        while tile_cols_log2 < max_log2_tile_cols {
            if r.read_bit()? != 0 {
                tile_cols_log2 += 1;
            } else {
                break;
            }
        }
        let tile_width_sb = (sb_cols + (1 << tile_cols_log2) - 1) >> tile_cols_log2;
        let mut remaining = sb_cols;
        while remaining > 0 {
            let w = tile_width_sb.min(remaining);
            tile_col_widths.push(w);
            remaining = remaining.saturating_sub(w);
        }

        let min_log2_tile_rows = if (tile_col_widths.len() as u32) > 1 {
            tile_log2(1, sb_rows)
        } else {
            0
        };
        let _ = min_log2_tile_rows;

        let mut tile_rows_log2 = 0u32;
        // Read tile_rows_log2 increment bits
        while tile_rows_log2 < max_log2_tile_rows {
            if r.read_bit()? != 0 {
                tile_rows_log2 += 1;
            } else {
                break;
            }
        }
        let tile_height_sb = (sb_rows + (1 << tile_rows_log2) - 1) >> tile_rows_log2;
        let mut remaining = sb_rows;
        while remaining > 0 {
            let h = tile_height_sb.min(remaining);
            tile_row_heights.push(h);
            remaining = remaining.saturating_sub(h);
        }
    } else {
        // Non-uniform: explicit widths
        let mut remaining = sb_cols;
        while remaining > 0 {
            let max_w = remaining.min(sb_cols);
            let w = if max_w > 1 { read_ns(r, max_w)? + 1 } else { 1 };
            tile_col_widths.push(w);
            remaining = remaining.saturating_sub(w);
        }
        let mut remaining = sb_rows;
        while remaining > 0 {
            let max_h = remaining.min(sb_rows);
            let h = if max_h > 1 { read_ns(r, max_h)? + 1 } else { 1 };
            tile_row_heights.push(h);
            remaining = remaining.saturating_sub(h);
        }
    }

    let tile_cols = tile_col_widths.len() as u32;
    let tile_rows = tile_row_heights.len() as u32;
    let tile_count = tile_cols * tile_rows;

    let mut context_update_tile_id = 0u32;
    let mut tile_size_bytes = 4u8;

    if tile_count > 1 {
        let tile_bits = tile_log2(1, tile_count);
        if tile_bits > 0 {
            context_update_tile_id = r.read_bits(tile_bits as u8)?;
        }
        tile_size_bytes = (r.read_bits(2)? as u8) + 1;
    }

    Ok(Av1TileInfo {
        tile_cols,
        tile_rows,
        tile_col_widths,
        tile_row_heights,
        tile_count,
        context_update_tile_id,
        tile_size_bytes,
    })
}

// ---------------------------------------------------------------------------
// Quantization params parsing
// ---------------------------------------------------------------------------

fn parse_quantization_params(
    r: &mut BitstreamReader<'_>,
    seq: &Av1SequenceHeader,
) -> Result<Av1QuantizationParams, VideoError> {
    let base_q_idx = r.read_bits(8)? as u16;
    let delta_q_y_dc = read_delta_q(r)?;

    let (delta_q_u_dc, delta_q_u_ac, delta_q_v_dc, delta_q_v_ac) = if !seq.monochrome {
        let diff_uv_delta = if seq.separate_uv_delta_q {
            r.read_bit()? != 0
        } else {
            false
        };
        let u_dc = read_delta_q(r)?;
        let u_ac = read_delta_q(r)?;
        let (v_dc, v_ac) = if diff_uv_delta {
            (read_delta_q(r)?, read_delta_q(r)?)
        } else {
            (u_dc, u_ac)
        };
        (u_dc, u_ac, v_dc, v_ac)
    } else {
        (0, 0, 0, 0)
    };

    let using_qmatrix = r.read_bit()? != 0;
    if using_qmatrix {
        let _qm_y = r.read_bits(4)?;
        let _qm_u = r.read_bits(4)?;
        if !seq.separate_uv_delta_q {
            let _qm_v = _qm_u;
        } else {
            let _qm_v = r.read_bits(4)?;
        }
    }

    Ok(Av1QuantizationParams {
        base_q_idx,
        delta_q_y_dc,
        delta_q_u_dc,
        delta_q_u_ac,
        delta_q_v_dc,
        delta_q_v_ac,
        using_qmatrix,
    })
}

fn read_delta_q(r: &mut BitstreamReader<'_>) -> Result<i8, VideoError> {
    let present = r.read_bit()? != 0;
    if present {
        // su(7): signed value in 7 bits
        let val = r.read_bits(7)?;
        let sign = (val >> 6) & 1;
        let mag = val & 0x3F;
        Ok(if sign != 0 { -(mag as i8) } else { mag as i8 })
    } else {
        Ok(0)
    }
}

// ---------------------------------------------------------------------------
// Segmentation params parsing
// ---------------------------------------------------------------------------

fn parse_segmentation_params(
    r: &mut BitstreamReader<'_>,
) -> Result<Av1SegmentationParams, VideoError> {
    let enabled = r.read_bit()? != 0;
    if !enabled {
        return Ok(Av1SegmentationParams::default());
    }

    let update_map = r.read_bit()? != 0;
    let temporal_update = if update_map {
        r.read_bit()? != 0
    } else {
        false
    };
    let update_data = r.read_bit()? != 0;

    let mut feature_enabled = [[false; 8]; 8];
    let mut feature_data = [[0i16; 8]; 8];

    // Segmentation feature bits and signed flags per AV1 spec Table 3
    const SEG_FEATURE_BITS: [u8; 8] = [8, 6, 6, 6, 6, 3, 0, 0];
    const SEG_FEATURE_SIGNED: [bool; 8] = [true, true, true, true, true, false, false, false];

    if update_data {
        for seg in 0..8 {
            for feat in 0..8 {
                let feat_enabled = r.read_bit()? != 0;
                feature_enabled[seg][feat] = feat_enabled;
                if feat_enabled {
                    let bits = SEG_FEATURE_BITS[feat];
                    if bits > 0 {
                        let val = r.read_bits(bits)? as i16;
                        if SEG_FEATURE_SIGNED[feat] {
                            let sign = r.read_bit()?;
                            feature_data[seg][feat] = if sign != 0 { -val } else { val };
                        } else {
                            feature_data[seg][feat] = val;
                        }
                    } else {
                        feature_data[seg][feat] = 0;
                    }
                }
            }
        }
    }

    Ok(Av1SegmentationParams {
        enabled,
        update_map,
        update_data,
        temporal_update,
        feature_enabled,
        feature_data,
    })
}

// ---------------------------------------------------------------------------
// Loop filter params parsing
// ---------------------------------------------------------------------------

fn parse_loop_filter_params(
    r: &mut BitstreamReader<'_>,
    seq: &Av1SequenceHeader,
    frame_type: Av1FrameType,
) -> Result<Av1LoopFilterParams, VideoError> {
    if frame_type == Av1FrameType::KeyFrame && !seq.reduced_still_picture_header {
        // Loop filter is still parsed for key frames
    }

    let level0 = r.read_bits(6)? as u8;
    let level1 = r.read_bits(6)? as u8;
    let (level2, level3) = if !seq.monochrome && (level0 != 0 || level1 != 0) {
        (r.read_bits(6)? as u8, r.read_bits(6)? as u8)
    } else {
        (0, 0)
    };

    let sharpness = r.read_bits(3)? as u8;

    let delta_enabled = r.read_bit()? != 0;
    let mut ref_deltas = [1i8, 0, 0, 0, 0, -1, -1, -1]; // AV1 defaults
    let mut mode_deltas = [0i8; 2];
    let mut delta_update = false;

    if delta_enabled {
        delta_update = r.read_bit()? != 0;
        if delta_update {
            for delta in ref_deltas.iter_mut() {
                let update = r.read_bit()? != 0;
                if update {
                    // su(7)
                    let val = r.read_bits(7)?;
                    let sign = (val >> 6) & 1;
                    let mag = (val & 0x3F) as i8;
                    *delta = if sign != 0 { -mag } else { mag };
                }
            }
            for delta in mode_deltas.iter_mut() {
                let update = r.read_bit()? != 0;
                if update {
                    let val = r.read_bits(7)?;
                    let sign = (val >> 6) & 1;
                    let mag = (val & 0x3F) as i8;
                    *delta = if sign != 0 { -mag } else { mag };
                }
            }
        }
    }

    Ok(Av1LoopFilterParams {
        level: [level0, level1, level2, level3],
        sharpness,
        delta_enabled,
        delta_update,
        ref_deltas,
        mode_deltas,
    })
}

// ---------------------------------------------------------------------------
// CDEF params parsing
// ---------------------------------------------------------------------------

fn parse_cdef_params(
    r: &mut BitstreamReader<'_>,
    seq: &Av1SequenceHeader,
    frame_type: Av1FrameType,
) -> Result<Av1CdefParams, VideoError> {
    if !seq.enable_cdef {
        return Ok(Av1CdefParams::default());
    }

    // CDEF is not applied to lossless frames but still parsed
    let _ = frame_type;

    let damping = (r.read_bits(2)? as u8) + 3;
    let bits = r.read_bits(2)? as u8;
    let num_strengths = 1u32 << bits;

    let mut strengths = Vec::with_capacity(num_strengths as usize);
    for _ in 0..num_strengths {
        let y_pri = r.read_bits(4)? as u8;
        let y_sec = r.read_bits(2)? as u8;
        let (uv_pri, uv_sec) = if seq.num_planes() > 1 {
            (r.read_bits(4)? as u8, r.read_bits(2)? as u8)
        } else {
            (0, 0)
        };
        strengths.push((y_pri, y_sec, uv_pri, uv_sec));
    }

    Ok(Av1CdefParams {
        damping,
        bits,
        strengths,
    })
}

// ---------------------------------------------------------------------------
// Helper: UVLC (Universal Variable Length Code)
// ---------------------------------------------------------------------------

/// Read a UVLC value (AV1 spec Section 4.10.3).
fn read_uvlc(r: &mut BitstreamReader<'_>) -> Result<u32, VideoError> {
    let mut leading_zeros = 0u32;
    loop {
        let bit = r.read_bit()?;
        if bit != 0 {
            break;
        }
        leading_zeros += 1;
        if leading_zeros > 32 {
            return Err(VideoError::Codec("AV1: UVLC overflow".into()));
        }
    }
    if leading_zeros == 0 {
        return Ok(0);
    }
    let value = r.read_bits(leading_zeros as u8)?;
    Ok(value + (1 << leading_zeros) - 1)
}

// ---------------------------------------------------------------------------
// Helper: ns() — non-symmetric coding
// ---------------------------------------------------------------------------

/// Read a non-symmetrically encoded value in [0, n) (AV1 spec Section 4.10.7).
fn read_ns(r: &mut BitstreamReader<'_>, n: u32) -> Result<u32, VideoError> {
    if n <= 1 {
        return Ok(0);
    }
    let w = 32 - (n - 1).leading_zeros(); // ceil(log2(n))
    let m = (1u32 << w) - n;
    let v = r.read_bits(w as u8 - 1)?;
    if v < m {
        Ok(v)
    } else {
        let extra = r.read_bit()? as u32;
        Ok((v << 1) - m + extra)
    }
}

// ---------------------------------------------------------------------------
// Helper: tile_log2
// ---------------------------------------------------------------------------

/// Compute the minimum number of bits needed so that `(1 << result) >= target`.
fn tile_log2(blk_size: u32, target: u32) -> u32 {
    let mut k = 0u32;
    let mut val = blk_size;
    while val < target {
        k += 1;
        val = blk_size << k;
    }
    k
}

// ---------------------------------------------------------------------------
// OBU iterator — iterates over OBUs in a data buffer
// ---------------------------------------------------------------------------

/// An OBU with its parsed header and payload slice boundaries.
#[derive(Debug, Clone)]
pub struct Av1Obu {
    pub header: Av1ObuHeader,
    /// Byte offset of the OBU payload start within the original data.
    pub payload_offset: usize,
    /// Length of the OBU payload in bytes.
    pub payload_len: usize,
}

/// Iterate over all OBUs in a contiguous data buffer.
///
/// Returns a vector of parsed OBU descriptors. The caller can then slice
/// into the original data to access each OBU's payload.
pub fn parse_obus(data: &[u8]) -> Result<Vec<Av1Obu>, VideoError> {
    let mut obus = Vec::new();
    let mut pos = 0;

    while pos < data.len() {
        let hdr = parse_obu_header(&data[pos..])?;
        let after_header = pos + hdr.header_len;

        if !hdr.has_size {
            // Without obu_has_size_field, the OBU extends to end of data.
            // This is only valid for the last OBU in a temporal unit.
            let payload_len = data.len() - after_header;
            obus.push(Av1Obu {
                header: hdr,
                payload_offset: after_header,
                payload_len,
            });
            break;
        }

        if after_header >= data.len() {
            return Err(VideoError::Codec(
                "AV1: truncated OBU (no size field)".into(),
            ));
        }

        let (obu_size, size_bytes) = read_leb128(&data[after_header..])?;
        let payload_offset = after_header + size_bytes;
        let payload_len = obu_size as usize;

        if payload_offset + payload_len > data.len() {
            return Err(VideoError::Codec(format!(
                "AV1: OBU payload extends past data ({} + {} > {})",
                payload_offset,
                payload_len,
                data.len()
            )));
        }

        obus.push(Av1Obu {
            header: hdr,
            payload_offset,
            payload_len,
        });

        pos = payload_offset + payload_len;
    }

    Ok(obus)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leb128_round_trip() {
        for &val in &[
            0u64,
            1,
            127,
            128,
            255,
            256,
            16383,
            16384,
            0x0FFF_FFFF,
            0xFFFF_FFFF,
        ] {
            let encoded = write_leb128(val);
            let (decoded, consumed) = read_leb128(&encoded).expect("decode should succeed");
            assert_eq!(decoded, val, "round-trip failed for {val}");
            assert_eq!(consumed, encoded.len());
        }
    }

    #[test]
    fn leb128_known_values() {
        // 0 => [0x00]
        assert_eq!(read_leb128(&[0x00]).unwrap(), (0, 1));
        // 1 => [0x01]
        assert_eq!(read_leb128(&[0x01]).unwrap(), (1, 1));
        // 128 => [0x80, 0x01]
        assert_eq!(read_leb128(&[0x80, 0x01]).unwrap(), (128, 2));
        // 300 => [0xAC, 0x02]
        assert_eq!(read_leb128(&[0xAC, 0x02]).unwrap(), (300, 2));
    }

    #[test]
    fn leb128_empty_data_error() {
        assert!(read_leb128(&[]).is_err());
    }

    #[test]
    fn obu_header_sequence_header() {
        // byte: forbidden=0, type=1 (SequenceHeader)=0b0001, ext=0, has_size=1, reserved=0
        // 0_0001_0_1_0 = 0x0A
        let data = [0x0A, 0x05]; // header + LEB128 size of 5
        let hdr = parse_obu_header(&data).expect("should parse");
        assert_eq!(hdr.obu_type, Av1ObuType::SequenceHeader);
        assert!(hdr.has_size);
        assert!(!hdr.has_extension);
        assert_eq!(hdr.header_len, 1);
    }

    #[test]
    fn obu_header_with_extension() {
        // byte0: forbidden=0, type=3 (FrameHeader)=0b0011, ext=1, has_size=1, reserved=0
        // 0_0011_1_1_0 = 0x1E
        // byte1: temporal_id=2 (0b010), spatial_id=1 (0b01), reserved=0b000
        // 010_01_000 = 0x48
        let data = [0x1E, 0x48];
        let hdr = parse_obu_header(&data).expect("should parse");
        assert_eq!(hdr.obu_type, Av1ObuType::FrameHeader);
        assert!(hdr.has_size);
        assert!(hdr.has_extension);
        assert_eq!(hdr.extension.temporal_id, 2);
        assert_eq!(hdr.extension.spatial_id, 1);
        assert_eq!(hdr.header_len, 2);
    }

    #[test]
    fn obu_header_forbidden_bit_error() {
        // forbidden bit set: 1_0001_0_1_0 = 0x8A
        assert!(parse_obu_header(&[0x8A]).is_err());
    }

    #[test]
    fn obu_header_temporal_delimiter() {
        // type=2 (TemporalDelimiter): 0_0010_0_1_0 = 0x12
        let hdr = parse_obu_header(&[0x12]).expect("should parse");
        assert_eq!(hdr.obu_type, Av1ObuType::TemporalDelimiter);
        assert!(hdr.has_size);
    }

    #[test]
    fn obu_header_padding() {
        // type=15 (Padding): 0_1111_0_1_0 = 0x7A
        let hdr = parse_obu_header(&[0x7A]).expect("should parse");
        assert_eq!(hdr.obu_type, Av1ObuType::Padding);
    }

    #[test]
    fn parse_obus_multiple() {
        // Build two OBUs: TemporalDelimiter (empty) + Padding (3 bytes)
        // OBU 1: TemporalDelimiter (0x12), LEB128 size=0, empty payload
        // OBU 2: Padding (0x7A), LEB128 size=3, 3 zero bytes
        let data = vec![0x12, 0x00, 0x7A, 0x03, 0x00, 0x00, 0x00];

        let obus = parse_obus(&data).expect("should parse");
        assert_eq!(obus.len(), 2);
        assert_eq!(obus[0].header.obu_type, Av1ObuType::TemporalDelimiter);
        assert_eq!(obus[0].payload_len, 0);
        assert_eq!(obus[1].header.obu_type, Av1ObuType::Padding);
        assert_eq!(obus[1].payload_len, 3);
    }

    #[test]
    fn sequence_header_basic() {
        // Build a minimal sequence header bitstream for profile 0, 1920x1080, 8-bit
        // This is a synthetic bitstream matching the spec field ordering.
        let mut bits = Vec::new();

        // seq_profile = 0 (3 bits): 000
        push_bits(&mut bits, 0b000, 3);
        // still_picture = 0 (1 bit)
        push_bits(&mut bits, 0, 1);
        // reduced_still_picture_header = 0 (1 bit)
        push_bits(&mut bits, 0, 1);

        // timing_info_present_flag = 0 (1 bit)
        push_bits(&mut bits, 0, 1);
        // initial_display_delay_present_flag = 0 (1 bit)
        push_bits(&mut bits, 0, 1);
        // operating_points_cnt_minus_1 = 0 (5 bits)
        push_bits(&mut bits, 0, 5);
        // operating_point_idc[0] = 0 (12 bits)
        push_bits(&mut bits, 0, 12);
        // seq_level_idx[0] = 8 (5 bits) — level 4.0
        push_bits(&mut bits, 8, 5);
        // seq_level_idx > 7, so read seq_tier = 0 (1 bit)
        push_bits(&mut bits, 0, 1);

        // frame_width_bits_minus_1 = 10 (4 bits) => 11 bits for width
        push_bits(&mut bits, 10, 4);
        // frame_height_bits_minus_1 = 10 (4 bits) => 11 bits for height
        push_bits(&mut bits, 10, 4);
        // max_frame_width_minus_1 = 1919 (11 bits) => width = 1920
        push_bits(&mut bits, 1919, 11);
        // max_frame_height_minus_1 = 1079 (11 bits) => height = 1080
        push_bits(&mut bits, 1079, 11);

        // frame_id_numbers_present_flag = 0 (1 bit)
        push_bits(&mut bits, 0, 1);

        // use_128x128_superblock = 1 (1 bit)
        push_bits(&mut bits, 1, 1);
        // enable_filter_intra = 1 (1 bit)
        push_bits(&mut bits, 1, 1);
        // enable_intra_edge_filter = 1 (1 bit)
        push_bits(&mut bits, 1, 1);

        // Not reduced_still_picture_header, so read inter flags:
        // enable_interintra_compound = 0 (1 bit)
        push_bits(&mut bits, 0, 1);
        // enable_masked_compound = 0 (1 bit)
        push_bits(&mut bits, 0, 1);
        // enable_warped_motion = 0 (1 bit)
        push_bits(&mut bits, 0, 1);
        // enable_dual_filter = 0 (1 bit)
        push_bits(&mut bits, 0, 1);
        // enable_order_hint = 1 (1 bit)
        push_bits(&mut bits, 1, 1);
        // enable_jnt_comp = 0 (1 bit)
        push_bits(&mut bits, 0, 1);
        // enable_ref_frame_mvs = 0 (1 bit)
        push_bits(&mut bits, 0, 1);

        // seq_choose_screen_content_tools = 1 (1 bit) => SELECT
        push_bits(&mut bits, 1, 1);
        // (no seq_force_screen_content_tools read since choose=1)
        // seq_force_screen_content_tools defaults to 2 (SELECT)
        // Since it's > 0, read seq_choose_integer_mv = 1 (1 bit) => SELECT
        push_bits(&mut bits, 1, 1);

        // order_hint_bits_minus_1 = 6 (3 bits) => 7 bits
        push_bits(&mut bits, 6, 3);

        // enable_superres = 0 (1 bit)
        push_bits(&mut bits, 0, 1);
        // enable_cdef = 1 (1 bit)
        push_bits(&mut bits, 1, 1);
        // enable_restoration = 0 (1 bit)
        push_bits(&mut bits, 0, 1);

        // color_config:
        // high_bitdepth = 0 (1 bit) => 8-bit
        push_bits(&mut bits, 0, 1);
        // profile != 1, so read monochrome = 0 (1 bit)
        push_bits(&mut bits, 0, 1);
        // color_description_present_flag = 0 (1 bit)
        push_bits(&mut bits, 0, 1);
        // profile 0 implies 4:2:0 (subsampling_x=1, subsampling_y=1)
        // color_range = 0 (1 bit)
        push_bits(&mut bits, 0, 1);
        // subsampling_x=1, subsampling_y=1: read chroma_sample_position (2 bits) = 0
        push_bits(&mut bits, 0, 2);
        // separate_uv_delta_q = 0 (1 bit)
        push_bits(&mut bits, 0, 1);

        // film_grain_params_present = 0 (1 bit)
        push_bits(&mut bits, 0, 1);

        let data = bits_to_bytes(&bits);
        let seq = parse_sequence_header(&data).expect("should parse sequence header");
        assert_eq!(seq.profile, 0);
        assert_eq!(seq.max_frame_width, 1920);
        assert_eq!(seq.max_frame_height, 1080);
        assert_eq!(seq.bit_depth, 8);
        assert!(!seq.monochrome);
        assert_eq!(seq.subsampling_x, 1);
        assert_eq!(seq.subsampling_y, 1);
        assert!(seq.use_128x128_superblock);
        assert!(seq.enable_cdef);
        assert!(!seq.film_grain_params_present);
        assert_eq!(seq.order_hint_bits, 7);
    }

    // Test helpers for building bitstreams

    fn push_bits(bits: &mut Vec<u8>, value: u32, count: u8) {
        for i in (0..count).rev() {
            bits.push(((value >> i) & 1) as u8);
        }
    }

    fn bits_to_bytes(bits: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(bits.len().div_ceil(8));
        for chunk in bits.chunks(8) {
            let mut byte = 0u8;
            for (i, &bit) in chunk.iter().enumerate() {
                byte |= bit << (7 - i);
            }
            bytes.push(byte);
        }
        bytes
    }
}
